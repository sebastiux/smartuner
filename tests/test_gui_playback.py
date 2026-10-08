"""Pruebas del reproductor de la tablatura (:mod:`gui.playback`).

No hay tarjeta de sonido en las pruebas: ``sounddevice`` se sustituye por un
módulo FALSO que registra las llamadas a ``play``/``stop``. Casi todas las
pruebas usan además un reloj y un «widget» falsos (``after`` guarda los
avisos y la prueba los ejecuta a mano), así que son deterministas y no
necesitan tkinter. La última abre una ventana real (``@pytest.mark.gui``;
sin ``DISPLAY`` se omite)::

    xvfb-run -a -s "-screen 0 1400x900x24" .venv/bin/python -m pytest tests/test_gui_playback.py -q
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import gui.playback as playback
from gui.playback import SoundDeviceOutput, TabPlayback, load_sounddevice, notes_fingerprint, open_with_system_player
from src.synth_dataset import TAIL_S
from src.tab import TabNote

SR = 8000  # baja para que la síntesis de las pruebas sea instantánea


# ---------------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------------


class FakeStream:
    """Stream de sounddevice: solo ``active`` y ``latency``."""

    def __init__(self, latency: float) -> None:
        self.active = True
        self.latency = latency


class FakeSoundDevice(types.ModuleType):
    """Módulo ``sounddevice`` falso: registra ``play``/``stop`` y simula un único stream actual."""

    def __init__(self, latency: float = 0.0, reject_rates: tuple[int, ...] = ()) -> None:
        super().__init__("sounddevice")
        self.latency = latency
        self.reject_rates = reject_rates
        self.plays: list[tuple[np.ndarray, int]] = []
        self.attempts: list[int] = []  # frecuencias de TODAS las llamadas a play (también las rechazadas)
        self.stops = 0
        self.stream: FakeStream | None = None

    def play(self, data: Any, samplerate: int, **_kwargs: Any) -> None:
        self.attempts.append(int(samplerate))
        if int(samplerate) in self.reject_rates:
            raise RuntimeError(f"Invalid sample rate {samplerate}")
        if self.stream is not None:
            self.stream.active = False
        self.plays.append((np.array(data, copy=True), int(samplerate)))
        self.stream = FakeStream(self.latency)

    def stop(self, *_args: Any, **_kwargs: Any) -> None:
        self.stops += 1
        if self.stream is not None:
            self.stream.active = False

    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        return {"name": "salida falsa", "default_samplerate": 44100.0, "max_output_channels": 2}

    def get_stream(self) -> FakeStream:
        if self.stream is None:
            raise RuntimeError("play()/rec()/playrec() was not called yet")
        return self.stream


class FakeClock:
    """Reloj monotónico controlado por la prueba (s)."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class FakeWidget:
    """Sustituto de un widget de tkinter: ``after`` guarda el callback y :meth:`run_pending` lo ejecuta."""

    def __init__(self) -> None:
        self.jobs: dict[str, Callable[[], None]] = {}
        self._counter = 0

    def after(self, _ms: int, fn: Callable[[], None]) -> str:
        self._counter += 1
        job = f"after#{self._counter}"
        self.jobs[job] = fn
        return job

    def after_cancel(self, job: str) -> None:
        self.jobs.pop(job, None)

    def run_pending(self) -> None:
        jobs, self.jobs = list(self.jobs.values()), {}
        for fn in jobs:
            fn()


class Recorder:
    """Guarda lo que el reproductor avisa por sus callbacks."""

    def __init__(self) -> None:
        self.positions: list[float] = []
        self.states: list[str] = []
        self.messages: list[str] = []
        self.finished = 0

    def kwargs(self) -> dict[str, Any]:
        return {"on_position": self.positions.append, "on_state": self.states.append,
                "on_message": self.messages.append, "on_finished": self._finish}

    def _finish(self) -> None:
        self.finished += 1


def notes_a() -> list[TabNote]:
    """Tres notas separadas por silencios; fin de la última a 1.3 s → audio de 1.8 s."""
    return [TabNote(0, 0.2, 0.5, "A", 0, 33), TabNote(1, 0.6, 0.9, "D", 2, 40), TabNote(2, 1.0, 1.3, "G", 0, 43)]


def notes_b() -> list[TabNote]:
    """Otra tablatura (otro pitch en la nota 1)."""
    notes = notes_a()
    notes[1] = TabNote(1, 0.6, 0.9, "D", 4, 42)
    return notes


def make_player(fake: FakeSoundDevice | None = None, **kwargs: Any) -> tuple[TabPlayback, FakeSoundDevice, FakeClock,
                                                                           FakeWidget, Recorder]:
    """Reproductor con sounddevice, reloj y widget falsos (síntesis Karplus-Strong síncrona)."""
    fake = fake or FakeSoundDevice()
    clock, widget, rec = FakeClock(), FakeWidget(), Recorder()
    options: dict[str, Any] = {"output": SoundDeviceOutput(fake), "clock": clock, "method": "karplus-strong"}
    options.update(rec.kwargs())
    options.update(kwargs)
    player = TabPlayback(widget, **options)
    return player, fake, clock, widget, rec


# ---------------------------------------------------------------------------
# Detección de sounddevice
# ---------------------------------------------------------------------------


def test_load_sounddevice_detects_module_and_output_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con un módulo que tiene salida, se usa; sin módulo o sin dispositivo de salida, se explica por qué no."""
    fake = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    assert load_sounddevice() == (fake, "")
    assert SoundDeviceOutput().available

    def no_device(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Error querying device -1")

    monkeypatch.setattr(fake, "query_devices", no_device)
    module, reason = load_sounddevice()
    assert module is None and "dispositivo de salida" in reason

    monkeypatch.setitem(sys.modules, "sounddevice", None)  # «import sounddevice» lanza ImportError
    output = SoundDeviceOutput()
    assert not output.available and "sounddevice" in output.error and "libportaudio2" in output.error
    with pytest.raises(RuntimeError):
        output.play(np.zeros(10, dtype=np.float32), SR)
    output.stop()  # no falla aunque no haya salida


def test_output_resamples_when_device_rejects_the_rate() -> None:
    """Si el dispositivo rechaza 8000 Hz (p. ej. WASAPI), se remuestrea a su frecuencia nativa y se reintenta."""
    fake = FakeSoundDevice(latency=0.05, reject_rates=(SR,))
    output = SoundDeviceOutput(fake)
    latency = output.play(np.zeros(SR, dtype=np.float32), SR)
    assert latency == pytest.approx(0.05)
    (data, rate), = fake.plays
    assert rate == 44100 and data.shape == (44100,)
    assert output.is_active() is True
    fake.play(np.zeros(4, dtype=np.float32), SR * 2)  # otra parte de la app lanza su propio sd.play
    assert output.is_active() is False


def test_output_remembers_the_rejected_rate_and_reuses_the_resampled_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: tras el primer rechazo no se reintenta la frecuencia rechazada, y la misma señal se remuestrea una vez."""
    calls: list[int] = []
    real_resample = playback._resample

    def counting_resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        calls.append(len(y))
        return real_resample(y, sr_from, sr_to)

    monkeypatch.setattr(playback, "_resample", counting_resample)
    fake = FakeSoundDevice(reject_rates=(SR,))
    output = SoundDeviceOutput(fake)
    track = np.zeros(SR, dtype=np.float32)
    output.play(track, SR)
    assert output.forced_sr == 44100 and fake.attempts == [SR, 44100] and len(calls) == 1
    output.play(track, SR)                      # «▶ Todo» otra vez: ni intento fallido ni remuestreo
    assert fake.attempts == [SR, 44100, 44100] and len(calls) == 1
    output.play(track[: SR // 2], SR)           # otra señal: se remuestrea, pero directo a 44 100 Hz
    assert fake.attempts[-1] == 44100 and len(calls) == 2
    assert fake.plays[-1][0].shape == (22050,)


def test_output_clips_out_of_range_samples() -> None:
    """Las muestras fuera de [-1, 1] se recortan antes de llegar a PortAudio (sin tocar el arreglo del llamador)."""
    fake = FakeSoundDevice()
    output = SoundDeviceOutput(fake)
    loud = np.array([0.5, 1.4, -1.7, 0.2], dtype=np.float32)
    output.play(loud, SR)
    data, _ = fake.plays[-1]
    np.testing.assert_array_equal(data, np.array([0.5, 1.0, -1.0, 0.2], dtype=np.float32))
    assert loud[1] == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Transporte y reloj
# ---------------------------------------------------------------------------


def test_play_pause_resume_stop() -> None:
    """play → pausa (la posición se congela) → reanudar (desde la muestra de la pausa) → stop (vuelve a 0)."""
    player, fake, clock, widget, rec = make_player()
    player.set_source(notes_a(), sr=SR)
    assert player.duration_s == pytest.approx(1.3 + TAIL_S)
    player.play()
    assert player.is_playing and player.state == "playing"
    total = int(round((1.3 + TAIL_S) * SR))
    data, rate = fake.plays[-1]
    assert rate == SR and data.shape == (total,)
    assert player.position_s == 0.0

    clock.advance(0.5)
    assert player.position_s == pytest.approx(0.5)
    player.pause()
    assert player.is_paused and not player.is_playing and fake.stream is not None and not fake.stream.active
    clock.advance(10.0)
    assert player.position_s == pytest.approx(0.5)  # en pausa el reloj no cuenta

    player.resume()
    assert player.is_playing
    data, _ = fake.plays[-1]
    assert data.shape == (total - int(round(0.5 * SR)),)  # continúa desde la muestra de la pausa
    assert np.array_equal(data, fake.plays[0][0][int(round(0.5 * SR)):])
    clock.advance(0.25)
    assert player.position_s == pytest.approx(0.75)

    player.stop()
    assert player.state == "stopped" and player.position_s == 0.0
    assert widget.jobs == {}  # sin avisos pendientes
    assert rec.states == ["preparing", "playing", "paused", "playing", "stopped"]
    player.resume()  # parado: reanudar no hace nada
    assert player.state == "stopped"


def test_only_one_chain_of_position_ticks() -> None:
    """Regresión: llamar a ``_tick`` fuera de su temporizador no crea una segunda cadena de avisos.

    Con dos cadenas, cada aviso programaba otro y su número crecía sin límite
    (miles de avisos por segundo con una tablatura de 990 notas: la GUI se congelaba).
    """
    player, _fake, clock, widget, _rec = make_player()
    player.set_source(notes_a(), sr=SR)
    player.play()
    assert len(widget.jobs) == 1
    for _ in range(5):
        clock.advance(0.01)
        player._tick()  # p. ej. una prueba o un script que fuerza un aviso
        assert len(widget.jobs) == 1
    widget.run_pending()
    assert len(widget.jobs) == 1


def test_position_is_monotonic_and_compensates_output_latency() -> None:
    """Con latencia de salida L, el cursor espera L segundos y después avanza al ritmo del reloj, sin retroceder."""
    player, _fake, clock, widget, rec = make_player(FakeSoundDevice(latency=0.1))
    player.set_source(notes_a(), sr=SR)
    player.play(0.4)
    seen = [player.position_s]
    assert seen[0] == pytest.approx(0.4)
    clock.advance(0.05)
    assert player.position_s == pytest.approx(0.4)  # aún dentro de la latencia: no suena nada nuevo
    for step in range(12):
        clock.advance(0.07)
        if step == 4:
            player.pause()
            clock.advance(3.0)
            player.resume()
        widget.run_pending()
        seen.append(player.position_s)
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))
    assert all(b >= a - 1e-12 for a, b in zip(rec.positions, rec.positions[1:]))
    assert seen[-1] <= player.duration_s


def test_on_position_ticks_and_on_finished_at_the_end() -> None:
    """Cada aviso programado informa la posición; al llegar al final se llama una vez a on_finished."""
    player, _fake, clock, widget, rec = make_player()
    player.set_source(notes_a(), sr=SR)
    player.play()
    assert rec.positions == [0.0]
    for _ in range(3):
        clock.advance(0.03)
        widget.run_pending()
    assert rec.positions == pytest.approx([0.0, 0.03, 0.06, 0.09])
    assert player.note_at(rec.positions[-1]) is None  # antes de la primera nota (0.2 s)
    clock.advance(0.2)
    widget.run_pending()
    assert player.note_at(rec.positions[-1]) == 0
    clock.advance(5.0)
    widget.run_pending()
    assert rec.finished == 1 and player.state == "stopped"
    assert rec.positions[-1] == pytest.approx(player.duration_s)
    assert widget.jobs == {}
    widget.run_pending()
    assert rec.finished == 1


def test_play_seeks_and_starts_over_past_the_end() -> None:
    """play(t) sirve para saltar mientras suena; un inicio más allá del final empieza desde 0."""
    player, fake, clock, _widget, _rec = make_player()
    player.set_source(notes_a(), sr=SR)
    player.play(0.0)
    clock.advance(0.3)
    player.play(1.0)
    assert player.position_s == pytest.approx(1.0)
    assert fake.plays[-1][0].shape[0] == int(round((1.3 + TAIL_S) * SR)) - SR
    player.play(99.0)
    assert player.position_s == pytest.approx(0.0)


def test_external_playback_interrupts_and_pauses() -> None:
    """Si otra parte de la app usa la salida (otro sd.play), el reproductor queda en pausa en ese instante."""
    player, fake, clock, widget, rec = make_player()
    player.set_source(notes_a(), sr=SR)
    player.play()
    clock.advance(0.4)
    fake.play(np.zeros(10, dtype=np.float32), SR)  # p. ej. «▶ Todo» de la pestaña Audio
    widget.run_pending()
    assert player.is_paused and player.position_s == pytest.approx(0.4)
    assert any("interrumpió" in m for m in rec.messages)


def test_note_at_finds_the_sounding_note() -> None:
    """note_at devuelve la nota con inicio ≤ t < fin (su posición), o None en los silencios."""
    player, *_ = make_player()
    player.set_source(list(reversed(notes_a())), sr=SR)  # se ordenan por inicio
    assert [player.note_at(t) for t in (0.0, 0.2, 0.49, 0.5, 0.75, 1.29, 1.3, 9.0)] == [
        None, 0, 0, None, 1, 2, None, None]


# ---------------------------------------------------------------------------
# Caché, modos y fuente
# ---------------------------------------------------------------------------


def test_render_cache_mode_switch_and_source_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """La síntesis se hace una vez por (notas audibles, sr, método); cambiar de modo solo rehace la mezcla."""
    calls: list[int] = []
    real_render = playback.render_notes

    def counting_render(notes: Any, **kwargs: Any) -> np.ndarray:
        calls.append(len(notes))
        return real_render(notes, **kwargs)

    monkeypatch.setattr(playback, "render_notes", counting_render)
    player, fake, clock, _widget, _rec = make_player()
    original = np.random.default_rng(0).uniform(-0.2, 0.2, int(2.5 * SR)).astype(np.float32)
    player.set_source(notes_a(), original=original, sr=SR)
    player.play()
    player.stop()
    player.play()
    assert len(calls) == 1
    assert fake.plays[-1][0].shape == (original.size,)  # con original, todos los modos duran lo mismo

    # Cambiar el modo mientras suena: misma posición, audio estéreo, sin volver a sintetizar.
    clock.advance(1.0)
    player.mode = "estereo"
    assert player.is_playing and player.position_s == pytest.approx(1.0)
    data, _ = fake.plays[-1]
    assert data.shape == (original.size - SR, 2) and len(calls) == 1
    # El canal izquierdo es el original (escalado), desde el segundo 1.
    assert np.corrcoef(data[:, 0], original[SR:])[0, 1] == pytest.approx(1.0, abs=1e-5)

    # Misma tablatura con otra digitación (A-0 → E-5): suena igual, no se re-sintetiza.
    same_sound = notes_a()
    same_sound[0] = TabNote(0, 0.2, 0.5, "E", 5, 33)
    assert notes_fingerprint(same_sound) == notes_fingerprint(notes_a())
    player.set_source(same_sound, original=original, sr=SR)
    assert len(calls) == 1 and player.notes[0].string == "E"

    # Otra tablatura (otro algoritmo): se sintetiza mientras sigue «sonando» desde el mismo instante.
    clock.advance(0.2)
    player.set_source(notes_b(), original=original, sr=SR)
    assert len(calls) == 2 and player.is_playing and player.position_s == pytest.approx(1.2)
    # Volver a la primera: está en la caché.
    player.set_source(notes_a(), original=original, sr=SR)
    assert len(calls) == 2
    player.method = "auto"
    player.play()
    assert len(calls) == 3  # otro método → otra síntesis
    with pytest.raises(ValueError):
        player.mode = "karaoke"
    with pytest.raises(ValueError):
        player.method = "sinusoide"


def test_without_original_only_midi_is_played() -> None:
    """Sin audio original, los modos que lo necesitan reproducen solo el MIDI (y se avisa)."""
    player, fake, _clock, _widget, rec = make_player(mode="mezcla")
    player.set_source(notes_a(), sr=SR)
    player.play()
    assert player.is_playing and fake.plays[-1][0].ndim == 1
    assert any("solo el MIDI" in m for m in rec.messages)


def test_background_preparation_and_cancelled_start() -> None:
    """Con run_in_background el estado pasa por «preparing»; stop durante la síntesis evita que arranque sola."""
    jobs: list[tuple[str, Callable[..., Any], Callable[[Any], None]]] = []

    def runner(title: str, fn: Callable[..., Any], on_done: Callable[[Any], None]) -> bool:
        jobs.append((title, fn, on_done))
        return True

    def run_job() -> None:
        _title, fn, on_done = jobs.pop(0)
        progress_calls: list[float] = []
        result = fn(lambda f, _m="": progress_calls.append(f), threading.Event())
        assert progress_calls and progress_calls[0] < progress_calls[-1] <= 1.0
        on_done(result)

    player, fake, clock, _widget, rec = make_player(run_in_background=runner)
    player.set_source(notes_a(), sr=SR)
    player.play(0.3)
    assert player.state == "preparing" and fake.plays == [] and jobs[0][0] == "Sintetizando la tablatura"
    assert player.position_s == pytest.approx(0.3)
    player.play(0.6)  # otro clic mientras se sintetiza: no lanza una segunda tarea
    assert len(jobs) == 1
    run_job()
    assert player.is_playing and player.position_s == pytest.approx(0.6)

    # Detener mientras se prepara otra fuente: la síntesis termina (y se guarda) pero no suena.
    player.set_source(notes_b(), sr=SR)
    assert player.state == "preparing" and len(jobs) == 1
    player.stop()
    run_job()
    assert player.state == "stopped" and len(fake.plays) == 1
    player.play()
    assert player.is_playing and jobs == []  # ya estaba en la caché

    # Si el lanzador no acepta la tarea (otra tarea en curso), se avisa y queda parado.
    busy, *_ = make_player(run_in_background=lambda *_a: False)
    busy.set_source(notes_a(), sr=SR)
    busy.play()
    assert busy.state == "stopped"
    assert rec.states[:2] == ["preparing", "playing"]


def test_rejected_rate_is_handled_in_the_background_after_the_first_play(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: con una tarjeta que rechaza la frecuencia del análisis, reanudar, saltar o cambiar de modo ya no
    remuestrea en el hilo de la GUI: el audio se prepara UNA vez, en segundo plano, a la frecuencia aceptada."""
    jobs: list[tuple[str, Callable[..., Any], Callable[[Any], None]]] = []

    def runner(title: str, fn: Callable[..., Any], on_done: Callable[[Any], None]) -> bool:
        jobs.append((title, fn, on_done))
        return True

    def run_job() -> None:
        _title, fn, on_done = jobs.pop(0)
        on_done(fn(lambda _f, _m="": None, threading.Event()))

    calls: list[str] = []
    real_resample = playback._resample

    def counting_resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        calls.append(threading.current_thread().name)
        return real_resample(y, sr_from, sr_to)

    monkeypatch.setattr(playback, "_resample", counting_resample)
    fake = FakeSoundDevice(reject_rates=(SR,))
    player, fake, clock, _widget, _rec = make_player(fake, run_in_background=runner)
    original = np.zeros(int(2.0 * SR), dtype=np.float32)
    player.set_source(notes_a(), original=original, sr=SR)
    player.play()
    run_job()                                   # síntesis
    assert player.is_playing and fake.plays[-1][1] == 44100
    assert len(calls) == 1                      # el primer rechazo: una vez, inevitable
    clock.advance(0.5)
    player.pause()
    player.resume()                             # el audio aún está a 8000 Hz → se prepara en segundo plano
    assert player.state == "preparing" and len(calls) == 1
    assert jobs[0][0].startswith("Adaptando el audio de la tablatura a 44100 Hz")
    run_job()
    assert player.is_playing and player.position_s == pytest.approx(0.5)
    data, rate = fake.plays[-1]
    assert rate == 44100 and data.shape[0] == pytest.approx(player.duration_s * 44100 - 0.5 * 44100, abs=2)
    n_calls, n_attempts = len(calls), len(fake.attempts)
    player.pause()
    player.resume()                             # reanudar y saltar: solo recortar, sin remuestrear ni rechazos
    player.play(1.0)
    assert len(calls) == n_calls and jobs == []
    assert fake.attempts[n_attempts:] == [44100, 44100]
    player.mode = "estereo"                     # cambiar de modo: mezcla en segundo plano, remuestreos reutilizados
    run_job()
    assert player.is_playing and fake.plays[-1][0].ndim == 2 and fake.plays[-1][1] == 44100
    assert len(calls) == n_calls


def test_synthesis_errors_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un fallo de la síntesis no deja el reproductor colgado en «preparing»: se avisa y queda parado."""
    def broken(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise RuntimeError("fluidsynth falló")

    monkeypatch.setattr(playback, "render_notes", broken)
    player, fake, _clock, _widget, rec = make_player()
    player.set_source(notes_a(), sr=SR)
    player.play()
    assert player.state == "stopped" and fake.plays == []
    assert any("fluidsynth falló" in m for m in rec.messages)


# ---------------------------------------------------------------------------
# Sin sounddevice: reproductor del sistema
# ---------------------------------------------------------------------------


def test_fallback_exports_wav_and_midi_and_opens_system_player(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin sounddevice, play exporta WAV (desde el instante pedido) y MIDI y abre el WAV; el cursor no se mueve."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    opened: list[Path] = []
    widget, rec = FakeWidget(), Recorder()
    player = TabPlayback(widget, output=SoundDeviceOutput(), opener=opened.append, export_dir=tmp_path,
                         method="karplus-strong", mode="estereo", **rec.kwargs())
    assert not player.available and "sounddevice" in player.unavailable_reason
    original = np.zeros(int(2.0 * SR), dtype=np.float32)
    player.set_source(notes_a(), original=original, sr=SR, name="Money (UCB1)")
    player.play(0.5)
    assert len(opened) == 1 and opened[0].parent == tmp_path and opened[0].suffix == ".wav"
    assert opened[0].name.startswith("Money_UCB1_estereo") and player.last_export == opened[0]
    info = sf.info(str(opened[0]))
    assert (info.samplerate, info.channels) == (SR, 2)
    assert info.frames == original.size - int(0.5 * SR)
    midi = tmp_path / "Money_UCB1.mid"
    assert [n.pitch for n in pretty_midi.PrettyMIDI(str(midi)).instruments[0].notes] == [33, 40, 43]
    assert player.state == "stopped" and widget.jobs == {}
    assert any("no se sincroniza" in m for m in rec.messages)

    # Si el reproductor del sistema falla, se dice dónde quedó el archivo.
    def failing_opener(_path: Path) -> None:
        raise OSError("xdg-open no existe")

    player._opener = failing_opener
    player.play()
    assert "xdg-open no existe" in rec.messages[-1] and str(tmp_path) in rec.messages[-1]


def test_fallback_keeps_only_the_latest_wav_and_survives_a_locked_file(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: sin sounddevice, cada ▶ desde otro instante ya no deja un WAV más (se borran los viejos y todos
    al cerrar), y un WAV bloqueado (soundfile lanza LibsndfileError, un RuntimeError) se escribe con otro nombre."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    opened: list[Path] = []
    player = TabPlayback(FakeWidget(), output=SoundDeviceOutput(), opener=opened.append, export_dir=tmp_path,
                         method="karplus-strong")
    player.set_source(notes_a(), sr=SR, name="riff")
    for start in (0.0, 0.5, 1.0):
        player.play(start)
    assert len(opened) == 3 and len(set(opened)) == 3
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [opened[-1].name]

    # El WAV de «desde 0» está «abierto en otro programa»: aquí, una carpeta con su nombre.
    blocked = tmp_path / "riff_midi.wav"
    blocked.mkdir()
    player.play(0.0)
    assert opened[-1] != blocked and opened[-1].name.startswith("riff_midi_") and opened[-1].is_file()
    player.close()
    assert list(tmp_path.glob("*.wav")) == [blocked]  # la carpeta no es un WAV exportado: se deja
    assert (tmp_path / "riff.mid").is_file()


@pytest.mark.parametrize("platform, expected", [("darwin", ["open"]), ("linux", ["xdg-open"])])
def test_open_with_system_player_uses_open_or_xdg_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                       platform: str, expected: list[str]) -> None:
    """macOS usa ``open`` y Linux ``xdg-open`` (sin esperar a que terminen)."""
    launched: list[list[str]] = []
    monkeypatch.setattr(playback.subprocess, "Popen", lambda cmd, **_kw: launched.append(cmd))
    open_with_system_player(tmp_path / "a.wav", platform=platform)
    assert launched == [expected + [str(tmp_path / "a.wav")]]


def test_open_with_system_player_uses_startfile_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Windows usa ``os.startfile`` (abre el reproductor predeterminado)."""
    started: list[str] = []
    monkeypatch.setattr(os, "startfile", started.append, raising=False)
    open_with_system_player(tmp_path / "a.wav", platform="win32")
    assert started == [str(tmp_path / "a.wav")]


# ---------------------------------------------------------------------------
# Con tkinter real
# ---------------------------------------------------------------------------


try:
    import tkinter

    HAS_TK = True
except ImportError:  # pragma: no cover - depende de la instalación de Python
    HAS_TK = False


@pytest.mark.gui
@pytest.mark.skipif(not HAS_TK, reason="tkinter no está disponible")
@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="sin servidor X (DISPLAY); usa xvfb-run")
def test_real_tk_after_loop_and_worker_thread() -> None:
    """Con una ventana real, la síntesis corre en un hilo (WorkerManager) y el cursor avanza vía ``after``."""
    from gui.workers import WorkerManager

    root = tkinter.Tk()
    root.withdraw()
    try:
        workers = WorkerManager(root)
        rec = Recorder()
        fake = FakeSoundDevice()

        def run_in_background(title: str, fn: Callable[..., Any], on_done: Callable[[Any], None]) -> bool:
            workers.start(title, fn, on_done=on_done)
            return True

        player = TabPlayback(root, run_in_background, output=SoundDeviceOutput(fake), method="karplus-strong",
                             poll_ms=10, **rec.kwargs())
        notes = [TabNote(0, 0.02, 0.12, "A", 0, 33), TabNote(1, 0.15, 0.25, "D", 2, 40)]
        player.set_source(notes, sr=SR)
        player.play()
        assert player.state == "preparing"
        end = time.perf_counter() + 10.0
        while rec.finished == 0 and time.perf_counter() < end:
            root.update()
            time.sleep(0.005)
        assert rec.finished == 1 and player.state == "stopped"
        assert len(fake.plays) == 1 and rec.states == ["preparing", "playing", "stopped"]
        assert len(rec.positions) >= 5  # ~0.75 s de audio a un aviso cada 10 ms
        assert all(b >= a for a, b in zip(rec.positions, rec.positions[1:]))
        assert rec.positions[-1] == pytest.approx(player.duration_s)
        player.close()
    finally:
        root.destroy()
