"""Pruebas de «escuchar la tablatura» en :mod:`src.tab`: MIDI, síntesis alineada y modos de mezcla."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from src.config import OPEN_STRING_MIDI
from src.pitch import midi_to_hz
from src.synth_dataset import TAIL_S, fluidsynth_available
from src.tab import (
    MIDI_PROGRAM,
    PLAYBACK_MODES,
    PLAYBACK_PEAK,
    TabNote,
    export_midi,
    export_tab,
    notes_to_midi,
    playback_mix,
    render_notes,
    resolve_synth_method,
)

SR = 22050


def _note(i: int, start_s: float, end_s: float, string: str, fret: int) -> TabNote:
    """TabNote con MIDI coherente con (cuerda, traste)."""
    return TabNote(position=i, start_s=start_s, end_s=end_s, string=string, fret=fret,
                   midi=OPEN_STRING_MIDI[string] + fret)


def _riff() -> list[TabNote]:
    """Cuatro notas separadas por silencios (Karplus-Strong apaga cada nota en su fin)."""
    return [_note(0, 0.30, 0.70, "A", 0), _note(1, 0.95, 1.35, "D", 2),
            _note(2, 1.60, 2.00, "E", 3), _note(3, 2.25, 2.70, "G", 0)]


#: Umbral (−60 dBFS) a partir del cual se considera que «empieza la energía» de una nota.
ONSET_THRESHOLD = 1e-3


def _onset_s(y: np.ndarray, sr: int, after_s: float) -> float:
    """Primer instante ≥ ``after_s`` (s) en que |y| supera :data:`ONSET_THRESHOLD`."""
    first = int(round(after_s * sr))
    window = np.abs(y[first:first + int(0.3 * sr)])
    return after_s + int(np.argmax(window > ONSET_THRESHOLD)) / sr


def _fft_peak_hz(y: np.ndarray, sr: int) -> float:
    """Frecuencia (Hz) del máximo del espectro de magnitud (ventana de Hann, FFT con relleno)."""
    n_fft = 1 << 17
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs > 20) & (freqs < 1000)
    return float(freqs[band][np.argmax(spectrum[band])])


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------


def test_notes_to_midi_round_trip_keeps_pitches_and_absolute_times(tmp_path: Path) -> None:
    """El MIDI escrito y releído conserva pitch, inicio y fin absolutos (± 1 ms), el programa GM 33 y las etiquetas."""
    notes = _riff()
    path = tmp_path / "riff.mid"
    notes_to_midi(notes).write(str(path))
    pm = pretty_midi.PrettyMIDI(str(path))
    assert len(pm.instruments) == 1
    inst = pm.instruments[0]
    assert inst.program == MIDI_PROGRAM == 33 and not inst.is_drum
    assert [n.pitch for n in inst.notes] == [n.midi for n in notes] == [33, 40, 31, 43]
    assert [n.start for n in inst.notes] == pytest.approx([n.start_s for n in notes], abs=1e-3)
    assert [n.end for n in inst.notes] == pytest.approx([n.end_s for n in notes], abs=1e-3)
    assert {n.velocity for n in inst.notes} == {100}
    assert [lyric.text for lyric in pm.lyrics] == ["A-0", "D-2", "E-3", "G-0"]


def test_notes_to_midi_sorts_validates_and_avoids_zero_length() -> None:
    """Las notas se escriben por orden de inicio, una nota de duración 0 recibe 10 ms y los rangos se validan."""
    notes = [_note(1, 1.0, 1.0, "A", 2), _note(0, 0.5, 0.8, "E", 0)]
    inst = notes_to_midi(notes, program=34, velocity=80).instruments[0]
    assert [n.pitch for n in inst.notes] == [28, 35]
    assert inst.notes[1].end - inst.notes[1].start == pytest.approx(0.01, abs=1e-3)
    assert inst.program == 34 and {n.velocity for n in inst.notes} == {80}
    with pytest.raises(ValueError):
        notes_to_midi(notes, program=128)
    with pytest.raises(ValueError):
        notes_to_midi(notes, velocity=0)
    with pytest.raises(ValueError):
        notes_to_midi([TabNote(0, 0.0, 0.5, "E", 0, 200)])


def test_export_midi_and_export_tab_mid(tmp_path: Path) -> None:
    """export_midi y export_tab(.mid/.MIDI) escriben un MIDI legible con las mismas notas."""
    notes = _riff()
    for path in (export_midi(notes, tmp_path / "sub" / "a.mid"), export_tab(notes, tmp_path / "b.MIDI"),
                 export_tab(notes, tmp_path / "c.mid", meta={"title": "ignorado"})):
        assert path.is_file()
        pm = pretty_midi.PrettyMIDI(str(path))
        assert [n.pitch for n in pm.instruments[0].notes] == [n.midi for n in notes]
    with pytest.raises(ValueError, match=r"\.mid"):
        export_tab(notes, tmp_path / "a.wav")


# ---------------------------------------------------------------------------
# Síntesis
# ---------------------------------------------------------------------------


def test_render_karplus_strong_length_pitch_and_onsets() -> None:
    """Karplus-Strong: longitud = fin + cola, silencio antes de la 1.ª nota, f0 correcta y onsets a ± 20 ms."""
    notes = _riff()
    y = render_notes(notes, sr=SR, method="karplus-strong")
    assert y.dtype == np.float32 and y.ndim == 1
    assert y.size == int(round((notes[-1].end_s + TAIL_S) * SR))
    assert float(np.abs(y).max()) == pytest.approx(PLAYBACK_PEAK, rel=1e-4)
    assert not np.any(y[:int(0.30 * SR)])  # tiempos absolutos: nada suena antes del primer onset
    for note in notes:
        assert _onset_s(y, SR, note.start_s - 0.1) == pytest.approx(note.start_s, abs=0.02)
        body = y[int((note.start_s + 0.05) * SR):int((note.end_s - 0.02) * SR)]
        assert _fft_peak_hz(body, SR) == pytest.approx(midi_to_hz(note.midi), rel=0.02)


def test_render_is_deterministic_and_handles_edge_cases() -> None:
    """Misma semilla → mismo audio; sin notas → solo la cola en silencio; método o sr inválidos → error."""
    notes = _riff()[:2]
    a = render_notes(notes, sr=8000, method="karplus-strong", seed=3)
    b = render_notes(notes, sr=8000, method="karplus-strong", seed=3)
    assert np.array_equal(a, b)
    empty = render_notes([], sr=8000, method="karplus-strong")
    assert empty.shape == (int(round(TAIL_S * 8000)),) and not empty.any()
    with pytest.raises(ValueError):
        render_notes(notes, method="sinusoide")
    with pytest.raises(ValueError):
        render_notes(notes, sr=0)
    assert resolve_synth_method("auto") in ("fluidsynth", "karplus-strong")


@pytest.mark.skipif(not fluidsynth_available(), reason="fluidsynth o el soundfont GM no están instalados")
def test_render_fluidsynth_is_aligned_and_in_tune() -> None:
    """fluidsynth (GM 33): misma longitud que Karplus-Strong, energía desde el onset (± 20 ms) y f0 correcta (YIN)."""
    import librosa

    notes = _riff()
    y = render_notes(notes, sr=SR, method="fluidsynth")
    assert y.size == int(round((notes[-1].end_s + TAIL_S) * SR))
    assert float(np.abs(y[:int(0.25 * SR)]).max()) < 0.01
    for note in notes:
        # La energía empieza ~6 ms después del onset (aunque el ataque de algunas muestras
        # grabadas, como el E grave, tarda ~50 ms en llegar a su máximo).
        assert _onset_s(y, SR, note.start_s - 0.1) == pytest.approx(note.start_s, abs=0.02)
        body = y[int((note.start_s + 0.05) * SR):int((note.end_s - 0.02) * SR)]
        f0 = librosa.yin(body, fmin=30.0, fmax=300.0, sr=SR, frame_length=4096)
        assert float(np.median(f0)) == pytest.approx(midi_to_hz(note.midi), rel=0.03)


@pytest.mark.slow
def test_render_six_minutes_of_notes_is_fast() -> None:
    """~1000 notas / 6 min 20 s («Money») se sintetizan con Karplus-Strong en pocos segundos."""
    rng = np.random.default_rng(1)
    starts = np.sort(rng.uniform(0.5, 380.0, 990))
    notes = [TabNote(i, float(s), float(s) + 0.35, "E", 0, int(rng.integers(28, 55))) for i, s in enumerate(starts)]
    t0 = time.perf_counter()
    y = render_notes(notes, sr=SR, method="karplus-strong")
    elapsed = time.perf_counter() - t0
    assert y.size == int(round((max(n.end_s for n in notes) + TAIL_S) * SR))
    assert elapsed < 10.0, f"la síntesis tardó {elapsed:.1f} s"  # ~1 s en un portátil; margen para CI cargado


# ---------------------------------------------------------------------------
# Mezcla
# ---------------------------------------------------------------------------


def _signals() -> tuple[np.ndarray, np.ndarray]:
    """Original (seno de 55 Hz, 1.0 s, nivel bajo) y síntesis (seno de 110 Hz, 1.5 s, nivel alto)."""
    t1, t2 = np.arange(SR) / SR, np.arange(int(1.5 * SR)) / SR
    original = (0.1 * np.sin(2 * np.pi * 55.0 * t1)).astype(np.float32)
    synth = (0.8 * np.sin(2 * np.pi * 110.0 * t2)).astype(np.float32)
    return original, synth


@pytest.mark.parametrize("mode", PLAYBACK_MODES)
def test_playback_mix_shapes_padding_and_peak(mode: str) -> None:
    """Todos los modos duran lo mismo (relleno con ceros hasta la señal más larga) y tienen pico 0.9."""
    original, synth = _signals()
    before = (original.copy(), synth.copy())
    out = playback_mix(original, synth, mode)
    n = synth.size
    assert out.dtype == np.float32
    assert out.shape == ((n, 2) if mode == "estereo" else (n,))
    assert float(np.abs(out).max()) == pytest.approx(PLAYBACK_PEAK, rel=1e-4)
    assert np.array_equal(original, before[0]) and np.array_equal(synth, before[1])  # entradas intactas
    if mode == "original":
        assert not out[original.size:].any()  # el original se rellenó con ceros


def test_playback_mix_stereo_puts_original_left_and_synth_right_at_equal_loudness() -> None:
    """A/B estéreo: original a la izquierda, síntesis a la derecha, con el mismo RMS (sonoridad igualada)."""
    original, synth = _signals()
    out = playback_mix(original, synth, "estereo")
    left, right = out[:, 0].astype(np.float64), out[:, 1].astype(np.float64)
    # Cada canal es proporcional a su señal (correlación 1) y en silencio donde ella lo está.
    assert np.corrcoef(left[:original.size], original)[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert np.corrcoef(right, synth)[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert not left[original.size:].any()
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    assert rms(left) / rms(right) == pytest.approx(1.0, rel=1e-3)  # el original era 8 veces más bajo
    # −6 dB de síntesis: el canal derecho baja a la mitad respecto al izquierdo.
    quieter = playback_mix(original, synth, "estereo", synth_gain_db=-6.0206)
    assert rms(quieter[:, 1].astype(np.float64)) / rms(quieter[:, 0].astype(np.float64)) == pytest.approx(0.5, rel=1e-3)


def test_playback_mix_mezcla_is_the_loudness_matched_sum() -> None:
    """Mezcla = original/RMS + síntesis/RMS, normalizada: ambas señales pesan lo mismo aunque el original sea bajo."""
    original, synth = _signals()
    out = playback_mix(original, synth, "mezcla").astype(np.float64)
    o = np.concatenate([original, np.zeros(synth.size - original.size)]).astype(np.float64)
    s = synth.astype(np.float64)
    expected = o / np.sqrt(np.mean(o ** 2)) + s / np.sqrt(np.mean(s ** 2))
    expected *= PLAYBACK_PEAK / np.abs(expected).max()
    assert np.allclose(out, expected, atol=1e-4)


def test_playback_mix_validates_mode_and_original() -> None:
    """Modo desconocido o modo que necesita el original sin él → ValueError; «midi» funciona sin original."""
    original, synth = _signals()
    with pytest.raises(ValueError, match="desconocido"):
        playback_mix(original, synth, "karaoke")
    for mode in ("original", "mezcla", "estereo"):
        with pytest.raises(ValueError, match="original"):
            playback_mix(None, synth, mode)
    assert playback_mix(None, synth, "midi").shape == synth.shape
    # Un original estéreo (n, 2) se promedia a mono antes de mezclar.
    stereo_original = np.stack([original, original], axis=1)
    assert np.allclose(playback_mix(stereo_original, synth, "estereo"), playback_mix(original, synth, "estereo"))
