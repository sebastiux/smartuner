"""Pruebas de la etapa 5 (pitch con pYIN) y de las conversiones Hz ↔ MIDI ↔ nombre.

pYIN es lento, así que las señales son cortas y la trayectoria de la pista de
tonos se calcula una sola vez por módulo (fixture ``scope="module"``).
"""

from __future__ import annotations

import logging
from typing import Any

import librosa
import numpy as np
import pytest

from src.config import PitchConfig
from src.preprocessing import lowpass
from src.pitch import (
    PitchTrack,
    annotate_segments,
    estimate_pitch_track,
    hz_to_midi,
    midi_to_hz,
    midi_to_name,
    segment_f0,
)
from src.segmentation import Segment

SR = 22050
HOP = 256
TEST_F0S = (41.2, 55.0, 73.42, 98.0, 146.8)
NOTE_S = 0.4
GAP_S = 0.1


def _lowpass(y: np.ndarray, cutoff_hz: float = 400.0) -> np.ndarray:
    """Pasa-bajas del pipeline (:func:`src.preprocessing.lowpass`, Butterworth de fase cero).

    Se usa el filtro REAL del pipeline (no una copia): si dejara de funcionar,
    estas pruebas fallarían en lugar de seguir en verde con otro filtro.
    """
    return lowpass(y, SR, cutoff_hz)


def _bass_tone(f0: float, dur_s: float, rng: np.random.Generator) -> np.ndarray:
    """Tono de bajo: 5 armónicos 1/h, ataque de 5 ms, decaimiento exponencial."""
    t = np.arange(int(round(dur_s * SR))) / SR
    env = np.minimum(t / 0.005, 1.0) * np.exp(-t / 0.5)
    phases = rng.uniform(0.0, 2.0 * np.pi, 5)
    return env * sum((1.0 / h) * np.sin(2.0 * np.pi * h * f0 * t + phases[h - 1]) for h in range(1, 6))


def _segment(index: int, start_s: float, end_s: float, **kwargs: object) -> Segment:
    """Segmento conservado de prueba entre ``start_s`` y ``end_s`` (campos extra por ``kwargs``)."""
    return Segment(
        index=index, start_s=start_s, end_s=end_s,
        start_sample=int(round(start_s * SR)), end_sample=int(round(end_s * SR)), rms_db=-3.0, **kwargs,
    )


@pytest.fixture(scope="module")
def tones() -> tuple[PitchTrack, list[Segment]]:
    """Pista con los 5 tonos de prueba separados por silencios + sus segmentos exactos."""
    rng = np.random.default_rng(0)
    parts: list[np.ndarray] = [np.zeros(int(GAP_S * SR))]
    segments: list[Segment] = []
    t = GAP_S
    for i, f0 in enumerate(TEST_F0S):
        tone = _bass_tone(f0, NOTE_S, rng)
        segments.append(_segment(i, t, t + tone.size / SR))
        parts += [tone, np.zeros(int(GAP_S * SR))]
        t += (tone.size + int(GAP_S * SR)) / SR
    y = _lowpass(np.concatenate(parts))
    y = (0.9 * y / np.max(np.abs(y))).astype(np.float32)
    return estimate_pitch_track(y, SR, PitchConfig(), hop_length=HOP), segments


# ---------------------------------------------------------------------------
# Conversiones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hz", "midi", "name"),
    [(41.20, 28, "E1"), (55.00, 33, "A1"), (73.42, 38, "D2"), (98.00, 43, "G2"),
     (196.00, 55, "G3"), (261.63, 60, "C4"), (440.0, 69, "A4")],
)
def test_hz_midi_name_reference_values(hz: float, midi: int, name: str) -> None:
    """Valores de referencia: Hz ↔ MIDI ↔ nombre de nota (E1 = 41.2 Hz = MIDI 28...)."""
    assert hz_to_midi(hz) == pytest.approx(midi, abs=0.01)
    assert midi_to_hz(midi) == pytest.approx(hz, rel=1e-3)
    assert midi_to_name(midi) == name


def test_hz_midi_round_trip_and_semitone_ratio() -> None:
    """hz_to_midi y midi_to_hz son inversas; un semitono = ×2^(1/12) y una octava = 12 semitonos."""
    for m in np.linspace(20.0, 80.0, 13):
        assert hz_to_midi(midi_to_hz(m)) == pytest.approx(m)
    # Un semitono = factor 2^(1/12); una octava = factor 2.
    assert midi_to_hz(34) / midi_to_hz(33) == pytest.approx(2 ** (1 / 12))
    assert hz_to_midi(110.0) - hz_to_midi(55.0) == pytest.approx(12.0)


def test_midi_to_name_rounds_and_uses_sharps() -> None:
    """midi_to_name redondea al semitono más cercano y usa sostenidos (C#4, D#2)."""
    assert midi_to_name(61) == "C#4"
    assert midi_to_name(32.6) == "A1"
    assert midi_to_name(33.4) == "A1"
    assert midi_to_name(29) == "F1"
    assert midi_to_name(39) == "D#2"


@pytest.mark.parametrize("bad", [0.0, -55.0, float("nan"), float("inf")])
def test_hz_to_midi_rejects_invalid_frequencies(bad: float) -> None:
    """Frecuencias ≤ 0, NaN o infinitas son un error con mensaje en español."""
    with pytest.raises(ValueError, match="positiva"):
        hz_to_midi(bad)


# ---------------------------------------------------------------------------
# pYIN sobre tonos de bajo
# ---------------------------------------------------------------------------


def test_pitch_track_structure(tones: tuple[PitchTrack, list[Segment]]) -> None:
    """La trayectoria de pYIN tiene un frame por hop, NaN donde no hay voz y probabilidades en [0, 1]."""
    track, _ = tones
    n = track.n_frames
    assert track.f0_hz.shape == track.voiced_prob.shape == track.voiced.shape == (n,)
    np.testing.assert_allclose(track.times_s, librosa.times_like(np.zeros(n), sr=SR, hop_length=HOP))
    assert track.voiced.dtype == bool
    assert np.all(np.isnan(track.f0_hz[~track.voiced]))
    assert np.all(np.isfinite(track.f0_hz[track.voiced]))
    assert np.all((track.voiced_prob >= 0.0) & (track.voiced_prob <= 1.0))


@pytest.mark.parametrize("position", range(len(TEST_F0S)))
def test_segment_f0_on_bass_tones(tones: tuple[PitchTrack, list[Segment]], position: int) -> None:
    """La f0 de cada nota de bajo sintética queda a < 0.3 semitonos de la real con > 80 % de frames con voz."""
    track, segments = tones
    f0, ratio = segment_f0(track, segments[position], PitchConfig())
    assert f0 is not None
    error_semitones = abs(hz_to_midi(f0) - hz_to_midi(TEST_F0S[position]))
    assert error_semitones < 0.3
    assert ratio > 0.8


def test_annotate_segments_fills_only_kept(
    tones: tuple[PitchTrack, list[Segment]], caplog: pytest.LogCaptureFixture
) -> None:
    """annotate_segments rellena la f0 solo de los segmentos conservados y la narra en el log por POSICIÓN."""
    track, segments = tones
    segs = [_segment(s.index, s.start_s, s.end_s) for s in segments]  # copias limpias
    segs[2].kept, segs[2].reason = False, "corto"
    with caplog.at_level(logging.INFO, logger="src.pitch"):
        annotate_segments(track, segs, PitchConfig())
    assert segs[2].f0_hz is None and segs[2].voiced_ratio == 0.0
    for i in (0, 1, 3, 4):
        assert segs[i].f0_hz is not None
        assert abs(hz_to_midi(segs[i].f0_hz) - hz_to_midi(TEST_F0S[i])) < 0.3
        assert segs[i].midi == pytest.approx(hz_to_midi(segs[i].f0_hz))
    # Mensaje por segmento en español con nota y porcentaje de frames con voz.
    assert "Segmento 1: f0=55." in caplog.text and "Hz (A1)" in caplog.text
    assert "% de frames con voz" in caplog.text
    # Regresión: el log numera por POSICIÓN entre los conservados (como el entorno y la
    # tablatura); el segmento detectado n.º 3 es el conservado n.º 2 (el 2 se descartó).
    assert "Segmento 2 [detectado n.º 3]: f0=" in caplog.text
    assert "[detectado n.º 2]" not in caplog.text


@pytest.mark.parametrize("lowpassed", [False, True])
def test_white_noise_has_unknown_f0(lowpassed: bool) -> None:
    """El ruido blanco no tiene pitch: la f0 del segmento debe ser None.

    Con algunas semillas el Viterbi de pYIN marca tramos de ruido como "con
    voz" (con probabilidad ≈ 0.01); el filtro ``MIN_VOICED_PROB`` los elimina.
    """
    cfg = PitchConfig()
    for seed in (2, 3):  # semillas en las que pYIN, sin filtro, marcaba > 40 % de frames con voz
        y = np.random.default_rng(seed).normal(0.0, 0.3, int(0.5 * SR))
        if lowpassed:
            y = _lowpass(y)
        track = estimate_pitch_track(y.astype(np.float32), SR, cfg, hop_length=HOP)
        f0, ratio = segment_f0(track, _segment(0, 0.0, 0.5), cfg)
        assert f0 is None
        assert ratio < cfg.min_voiced_ratio


# ---------------------------------------------------------------------------
# segment_f0 con trayectorias construidas a mano (sin pYIN)
# ---------------------------------------------------------------------------


def _manual_track(f0: np.ndarray, step_s: float = 0.01) -> PitchTrack:
    """PitchTrack hecho a mano con un frame cada ``step_s`` segundos (NaN = sin voz)."""
    voiced = np.isfinite(f0)
    return PitchTrack(np.arange(f0.size) * step_s, f0, np.where(voiced, 0.9, 0.01), voiced)


def test_segment_f0_skips_attack_and_uses_median() -> None:
    # 100 frames de 10 ms: ataque (3 frames) a 110 Hz, luego 55 Hz con 2
    # frames atípicos (error de octava) que la mediana ignora.
    """La f0 de un segmento salta el ataque y usa la mediana, robusta a frames con error de octava."""
    f0 = np.full(100, 55.0)
    f0[:3] = 110.0
    f0[50:52] = 27.5
    track = _manual_track(f0)
    seg = _segment(0, 0.0, 1.0)
    value, ratio = segment_f0(track, seg, PitchConfig(), attack_skip_s=0.03)
    assert value == pytest.approx(55.0)
    assert ratio == pytest.approx(1.0)
    # Sin saltar el ataque la mediana sigue siendo 55 Hz (robustez).
    assert segment_f0(track, seg, PitchConfig(), attack_skip_s=0.0)[0] == pytest.approx(55.0)


def test_segment_f0_respects_min_voiced_ratio() -> None:
    """Con menos frames con voz que min_voiced_ratio la f0 es desconocida (None)."""
    f0 = np.full(100, np.nan)
    f0[10:25] = 73.42  # 15 % de los frames tras el ataque con voz
    track = _manual_track(f0)
    seg = _segment(0, 0.0, 1.0)
    value, ratio = segment_f0(track, seg, PitchConfig(min_voiced_ratio=0.2), attack_skip_s=0.0)
    assert value is None and ratio == pytest.approx(0.15)
    value, ratio = segment_f0(track, seg, PitchConfig(min_voiced_ratio=0.1), attack_skip_s=0.0)
    assert value == pytest.approx(73.42) and ratio == pytest.approx(0.15)


def test_segment_f0_short_segment_uses_whole_segment() -> None:
    """Si el segmento es más corto que el ataque se usan todos sus frames."""
    f0 = np.full(100, 98.0)
    track = _manual_track(f0)
    value, ratio = segment_f0(track, _segment(0, 0.50, 0.52), PitchConfig(), attack_skip_s=0.03)
    assert value == pytest.approx(98.0) and ratio == pytest.approx(1.0)
    # Más corto que un frame: se usa el frame más cercano a su centro.
    value, _ = segment_f0(track, _segment(1, 0.501, 0.503), PitchConfig(), attack_skip_s=0.03)
    assert value == pytest.approx(98.0)


def test_annotate_segments_logs_unknown_f0(caplog: pytest.LogCaptureFixture) -> None:
    """Una f0 desconocida se anuncia en el log con la posición del segmento y su n.º de detección."""
    track = _manual_track(np.full(100, np.nan))
    segs = [_segment(4, 0.0, 1.0)]
    with caplog.at_level(logging.INFO, logger="src.pitch"):
        annotate_segments(track, segs, PitchConfig())
    assert segs[0].f0_hz is None and segs[0].voiced_ratio == 0.0
    assert "Segmento 0 [detectado n.º 4]: f0 desconocida" in caplog.text


def test_estimate_pitch_track_empty_signal() -> None:
    """Una señal vacía da una trayectoria vacía (no un error de librosa)."""
    track = estimate_pitch_track(np.zeros(0, dtype=np.float32), SR, PitchConfig())
    assert track.n_frames == 0


def test_pyin_transition_parameters_come_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_transition_rate`` y ``switch_prob`` de PitchConfig llegan a ``librosa.pyin``."""
    seen: dict[str, float] = {}

    def fake_pyin(y: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """pYIN simulado: registra los argumentos recibidos y devuelve 55 Hz con voz en todos los frames."""
        seen.update(kwargs)
        n = 1 + len(y) // kwargs["hop_length"]
        return np.full(n, 55.0), np.ones(n, dtype=bool), np.ones(n)

    monkeypatch.setattr(librosa, "pyin", fake_pyin)
    cfg = PitchConfig(max_transition_rate=123.0, switch_prob=0.07)
    track = estimate_pitch_track(np.zeros(SR // 2, dtype=np.float32), SR, cfg)
    assert seen["max_transition_rate"] == 123.0 and seen["switch_prob"] == 0.07
    assert track.voiced.all()
    # Valores por defecto: los que evitan los errores de octava (ver PitchConfig).
    assert (PitchConfig().max_transition_rate, PitchConfig().switch_prob) == (150.0, 0.1)


def _long_bass_line(n_notes: int, rng: np.random.Generator) -> np.ndarray:
    """Línea de bajo de ``n_notes`` notas de 0.4 s (+0.1 s de silencio) recorriendo TEST_F0S, filtrada a 400 Hz."""
    parts: list[np.ndarray] = []
    for i in range(n_notes):
        parts += [_bass_tone(TEST_F0S[i % len(TEST_F0S)], NOTE_S, rng), np.zeros(int(GAP_S * SR))]
    y = _lowpass(np.concatenate(parts))
    return (0.9 * y / np.max(np.abs(y))).astype(np.float32)


def test_blockwise_pyin_matches_a_single_call() -> None:
    """Regresión (GUI congelada ~15 s con una canción): pYIN por bloques da los MISMOS frames que una sola llamada.

    Bloques de 2 s con 1 s de contexto sobre una pista de 8 s: misma longitud y
    tiempos, misma decisión de voz y la misma f0 (±½ semitono) en cada frame.
    """
    y = _long_bass_line(16, np.random.default_rng(3))   # 16 × 0.5 s = 8 s
    whole = estimate_pitch_track(y, SR, PitchConfig(), hop_length=HOP)
    fractions: list[float] = []
    blocks = estimate_pitch_track(y, SR, PitchConfig(), hop_length=HOP, block_s=2.0, overlap_s=1.0,
                                  progress=lambda f, _m: fractions.append(f))
    block = int(2.0 * SR) // HOP * HOP                     # los bloques empiezan en múltiplos del hop
    n_blocks = -(-y.size // block)
    assert len(fractions) == n_blocks + 1 >= 4              # un aviso por bloque + el final
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    np.testing.assert_allclose(blocks.times_s, whole.times_s)
    assert np.mean(blocks.voiced != whole.voiced) < 0.01
    both = blocks.voiced & whole.voiced
    assert np.all(np.abs(12.0 * np.log2(blocks.f0_hz[both] / whole.f0_hz[both])) < 0.5)


def test_blockwise_pyin_can_be_cancelled_between_blocks() -> None:
    """Con ``cancel`` activo, pYIN por bloques se detiene con CancelledError (la GUI puede cancelar el análisis)."""
    import threading

    from src.config import CancelledError

    cancel = threading.Event()
    calls: list[float] = []

    def progress(fraction: float, _message: str) -> None:
        """Cancela al empezar el segundo bloque."""
        calls.append(fraction)
        if len(calls) == 2:
            cancel.set()

    y = _long_bass_line(8, np.random.default_rng(4))    # 4 s → bloques de 1 s
    with pytest.raises(CancelledError, match="bloque 3"):
        estimate_pitch_track(y, SR, PitchConfig(), hop_length=HOP, block_s=1.0, overlap_s=0.5,
                             progress=progress, cancel=cancel)
    assert len(calls) == 2
