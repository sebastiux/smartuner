"""Pruebas de la etapa 4 (segmentación por onsets) con señales sintéticas de bajo.

Las señales imitan notas de bajo pulsadas: 5 armónicos con amplitud 1/h,
ataque lineal de 5 ms y decaimiento exponencial, filtradas a 400 Hz como la
señal de análisis del pipeline. Cada nota termina con un "apagado" de 20 ms
(la mano que silencia la cuerda): un corte instantáneo produciría un clic de
banda ancha que no existe en una grabación real.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.config import SegmentationConfig
from src.preprocessing import lowpass
from src.segmentation import (
    Segment,
    detect_onsets,
    kept_segments,
    onset_envelope,
    segment_audio,
)

SR = 22050
BASS_F0S = (41.2, 55.0, 73.42, 98.0, 146.8, 196.0)
ONSET_TOL_S = 0.03


# ---------------------------------------------------------------------------
# Señales de prueba
# ---------------------------------------------------------------------------


def _lowpass(y: np.ndarray, cutoff_hz: float = 400.0) -> np.ndarray:
    """Pasa-bajas del pipeline (:func:`src.preprocessing.lowpass`, Butterworth de fase cero).

    Se usa el filtro REAL del pipeline (no una copia): si dejara de funcionar,
    estas pruebas fallarían en lugar de seguir en verde con otro filtro.
    """
    return lowpass(y, SR, cutoff_hz)


def _bass_note(
    f0: float,
    dur_s: float,
    rng: np.random.Generator,
    tau_s: float = 0.5,
    release_s: float = 0.02,
    phases: np.ndarray | None = None,
    t_offset_s: float = 0.0,
    start_level: float = 0.0,
) -> np.ndarray:
    """Nota de bajo sintética: Σ_h (1/h)·sin(2π·h·f0·t + φ_h) · envolvente.

    La envolvente sube linealmente de ``start_level`` a 1 en 5 ms, decae como
    exp(−t/τ) y, si ``release_s > 0``, termina con una rampa a 0.
    """
    n = int(round(dur_s * SR))
    t = np.arange(n) / SR
    attack = 0.005
    env = np.exp(-np.maximum(t - attack, 0.0) / tau_s)
    env = np.where(t < attack, start_level + (1.0 - start_level) * t / attack, env)
    if release_s > 0:
        n_rel = int(round(release_s * SR))
        env[-n_rel:] *= np.linspace(1.0, 0.0, n_rel)
    ph = rng.uniform(0.0, 2.0 * np.pi, 5) if phases is None else phases
    x = sum((1.0 / h) * np.sin(2.0 * np.pi * h * f0 * (t + t_offset_s) + ph[h - 1]) for h in range(1, 6))
    return x * env


def _finish(parts: list[np.ndarray]) -> np.ndarray:
    """Concatena, filtra a 400 Hz y normaliza a pico 0.9."""
    y = _lowpass(np.concatenate(parts))
    return (0.9 * y / np.max(np.abs(y))).astype(np.float32)


def _notes_with_gaps(seed: int, f0s: tuple[float, ...] = BASS_F0S) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notas distintas de 0.3–0.6 s separadas por silencios de 60–120 ms.

    Returns
    -------
    y, onsets_s, offsets_s
    """
    rng = np.random.default_rng(seed)
    t = 0.3
    parts = [np.zeros(int(t * SR))]
    onsets, offsets = [], []
    for f0 in f0s:
        note = _bass_note(f0, rng.uniform(0.3, 0.6), rng, tau_s=rng.uniform(0.3, 0.8))
        gap = np.zeros(int(round(rng.uniform(0.06, 0.12) * SR)))
        onsets.append(t)
        offsets.append(t + note.size / SR)
        parts += [note, gap]
        t += (note.size + gap.size) / SR
    parts.append(np.zeros(int(0.3 * SR)))
    return _finish(parts), np.array(onsets), np.array(offsets)


def _repeated_notes(
    seed: int, f0: float, n_notes: int = 5, continuous_phase: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Notas repetidas del MISMO pitch sin silencio intermedio (re-ataques).

    Cada nueva pulsación corta la anterior mientras aún suena (nivel residual
    de ~−4 dB). Con ``continuous_phase=False`` la cuerda se re-pulsa con fase
    nueva; con ``True`` la vibración continúa y solo salta la amplitud (el
    caso más difícil: no hay ningún cambio espectral salvo el nivel).
    """
    rng = np.random.default_rng(seed)
    tau = 0.5
    t = 0.3
    parts = [np.zeros(int(t * SR))]
    onsets = []
    phases = rng.uniform(0.0, 2.0 * np.pi, 5)
    level = 0.0
    for i in range(n_notes):
        dur = rng.uniform(0.3, 0.45)
        last = i == n_notes - 1
        note = _bass_note(
            f0, dur, rng, tau_s=tau,
            release_s=0.02 if last else 0.0,
            phases=phases if continuous_phase else None,
            t_offset_s=t if continuous_phase else 0.0,
            start_level=level if continuous_phase else 0.0,
        )
        level = float(np.exp(-(dur - 0.005) / tau))
        onsets.append(t)
        parts.append(note)
        t += note.size / SR
    parts.append(np.zeros(int(0.3 * SR)))
    return _finish(parts), np.array(onsets)


@pytest.fixture
def cfg() -> SegmentationConfig:
    """Configuración de segmentación por defecto."""
    return SegmentationConfig()


# ---------------------------------------------------------------------------
# Función de novedad y onsets
# ---------------------------------------------------------------------------


def test_onset_envelope_shape_and_peaks(cfg: SegmentationConfig) -> None:
    """La función de novedad tiene un valor por frame y picos claros (> 3×) en los onsets frente al interior de las notas."""
    y, onsets, _ = _notes_with_gaps(seed=0)
    env = onset_envelope(y, SR, cfg)
    assert env.shape == (1 + len(y) // cfg.hop_length,)
    assert np.all(env >= 0.0)
    # La novedad alrededor de cada onset (±30 ms) supera con creces a la del
    # interior de las notas (entre 100 ms después del onset y el final).
    frame_times = np.arange(env.size) * cfg.hop_length / SR
    for t0 in onsets:
        near = env[np.abs(frame_times - t0) <= 0.03].max()
        inside = env[(frame_times > t0 + 0.1) & (frame_times < t0 + 0.25)].max()
        assert near > 3.0 * inside


def test_onset_envelope_of_silence_is_zero(cfg: SegmentationConfig) -> None:
    """El silencio da novedad nula y una señal vacía, una envolvente vacía."""
    env = onset_envelope(np.zeros(SR // 2, dtype=np.float32), SR, cfg)
    assert env.size == 1 + (SR // 2) // cfg.hop_length
    assert not np.any(env)
    assert onset_envelope(np.zeros(0, dtype=np.float32), SR, cfg).size == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detect_onsets_notes_separated_by_silence(cfg: SegmentationConfig, seed: int) -> None:
    """Con el δ por defecto se detectan todas las notas y los segmentos conservados son exactamente ellas.

    El δ por defecto (0.05) es algo más sensible que el de máxima precisión
    (0.07, ver la prueba siguiente) para no perder re-ataques débiles; a
    cambio, el apagado (release) de una nota puede dar un onset de más dentro
    del silencio siguiente. Ese onset produce un segmento de pocos ms que
    :func:`segment_audio` descarta ("corto" o "silencio"), así que no llega
    a la tablatura.
    """
    y, onsets, _ = _notes_with_gaps(seed)
    est = detect_onsets(y, SR, cfg)
    assert isinstance(est, np.ndarray) and est.ndim == 1
    assert np.all(np.diff(est) > 0), "los onsets deben estar ordenados y sin repetidos"
    assert all(np.min(np.abs(est - t0)) < ONSET_TOL_S for t0 in onsets)   # ninguna nota perdida
    kept = kept_segments(segment_audio(y, SR, cfg, onsets_s=est))
    assert len(kept) == onsets.size
    assert np.max(np.abs(np.array([s.start_s for s in kept]) - onsets)) < ONSET_TOL_S


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detect_onsets_conservative_delta_has_no_false_onsets(seed: int) -> None:
    """Con δ = 0.07 la novedad log-mel no da NINGÚN onset falso (ni en los apagados)."""
    cfg = SegmentationConfig(onset_delta=0.07)
    y, onsets, _ = _notes_with_gaps(seed)
    est = detect_onsets(y, SR, cfg)
    assert est.size == onsets.size
    assert np.max(np.abs(est - onsets)) < ONSET_TOL_S


@pytest.mark.parametrize(
    ("f0", "continuous_phase"),
    [(41.2, False), (55.0, False), (98.0, False), (146.8, False), (41.2, True), (55.0, True), (98.0, True)],
)
def test_detect_onsets_repeated_same_pitch(cfg: SegmentationConfig, f0: float, continuous_phase: bool) -> None:
    """Re-ataques del mismo pitch sin silencio: el caso que motivó la novedad
    log-mel restringida a 1 kHz con piso de −40 dB."""
    y, onsets = _repeated_notes(seed=3, f0=f0, continuous_phase=continuous_phase)
    est = detect_onsets(y, SR, cfg)
    assert est.size == onsets.size
    assert np.max(np.abs(est - onsets)) < ONSET_TOL_S


def test_detect_onsets_silence_and_empty(cfg: SegmentationConfig) -> None:
    """Sin señal (silencio o vacía) no hay onsets."""
    for y in (np.zeros(SR, dtype=np.float32), np.zeros(0, dtype=np.float32)):
        est = detect_onsets(y, SR, cfg)
        assert isinstance(est, np.ndarray) and est.size == 0


def test_detect_onsets_accepts_precomputed_envelope(cfg: SegmentationConfig) -> None:
    """Pasar la envolvente ya calculada da los mismos onsets."""
    y, _, _ = _notes_with_gaps(seed=1)
    env = onset_envelope(y, SR, cfg)
    np.testing.assert_array_equal(detect_onsets(y, SR, cfg, envelope=env), detect_onsets(y, SR, cfg))


def test_wait_suppresses_close_onsets(cfg: SegmentationConfig) -> None:
    """Con una separación mínima mayor que la distancia entre notas se pierden onsets."""
    y, onsets, _ = _notes_with_gaps(seed=0)
    cfg.onset_wait_s = 1.0
    est = detect_onsets(y, SR, cfg)
    assert 0 < est.size < onsets.size
    assert np.all(np.diff(est) >= 0.9)  # (el backtrack puede acercarlos unos frames)


# ---------------------------------------------------------------------------
# Segmentación
# ---------------------------------------------------------------------------


def test_segment_audio_notes_with_gaps(cfg: SegmentationConfig) -> None:
    """Notas separadas por silencios: un segmento por nota, con inicio en el onset y fin recortado al final real."""
    y, onsets, offsets = _notes_with_gaps(seed=0)
    segs = segment_audio(y, SR, cfg)
    assert [s.index for s in segs] == list(range(len(segs)))
    assert len(segs) == onsets.size
    assert all(s.kept and s.reason == "ok" for s in segs)
    for seg, t_on, t_off in zip(segs, onsets, offsets):
        assert abs(seg.start_s - t_on) < ONSET_TOL_S
        # El fin se recorta al final real de la nota (no al siguiente onset).
        assert -0.01 < seg.end_s - t_off < 0.04
        assert seg.start_sample == round(seg.start_s * SR)
        assert seg.end_sample == round(seg.end_s * SR)
        assert cfg.rms_threshold_db < seg.rms_db <= 0.0
        assert seg.duration_s >= cfg.min_duration_s
    for a, b in zip(segs, segs[1:]):
        assert a.end_sample <= b.start_sample


def test_segment_audio_repeated_notes_are_not_trimmed(cfg: SegmentationConfig) -> None:
    """En notas ligadas (re-ataque sin silencio) cada segmento llega hasta el siguiente onset."""
    y, onsets = _repeated_notes(seed=3, f0=55.0)
    segs = segment_audio(y, SR, cfg)
    assert len(segs) == onsets.size and all(s.kept for s in segs)
    for a, b in zip(segs, segs[1:]):
        assert a.end_sample == b.start_sample


def test_segment_audio_marks_silence(cfg: SegmentationConfig) -> None:
    """Un onset dentro de un silencio largo produce un segmento ``silencio``."""
    rng = np.random.default_rng(5)
    a = _bass_note(55.0, 0.4, rng)
    b = _bass_note(73.42, 0.4, rng)
    y = _finish([np.zeros(int(0.2 * SR)), a, np.zeros(SR), b, np.zeros(int(0.2 * SR))])
    t_b = 0.2 + a.size / SR + 1.0
    onsets = np.array([0.2, 0.2 + a.size / SR + 0.3, t_b])
    segs = segment_audio(y, SR, cfg, onsets_s=onsets)
    assert [(s.index, s.kept, s.reason) for s in segs] == [(0, True, "ok"), (1, False, "silencio"), (2, True, "ok")]
    # El segmento silencioso conserva su intervalo bruto (para dibujarlo en gris).
    assert segs[1].start_s == pytest.approx(onsets[1], abs=1 / SR)
    assert segs[1].end_s == pytest.approx(t_b, abs=1 / SR)
    assert segs[1].rms_db < cfg.rms_threshold_db
    assert kept_segments(segs) == [segs[0], segs[2]]


def test_segment_audio_marks_short(cfg: SegmentationConfig) -> None:
    """Dos onsets separados 30 ms (< 60 ms de duración mínima) → segmento ``corto``."""
    y, onsets, _ = _notes_with_gaps(seed=1, f0s=(55.0, 98.0))
    spurious = onsets[0] + 0.03
    segs = segment_audio(y, SR, cfg, onsets_s=np.array([onsets[0], spurious, onsets[1]]))
    assert [s.reason for s in segs] == ["corto", "ok", "ok"]
    assert segs[0].kept is False and segs[0].duration_s < cfg.min_duration_s
    assert [s.index for s in segs] == [0, 1, 2]
    assert [s.index for s in kept_segments(segs)] == [1, 2]


def test_segment_audio_without_onsets_uses_single_segment(cfg: SegmentationConfig) -> None:
    """Sin onsets, la nota se toma como un único segmento desde el primer frame sobre el umbral."""
    rng = np.random.default_rng(7)
    note = _bass_note(98.0, 0.5, rng)
    y = _finish([np.zeros(int(0.5 * SR)), note, np.zeros(int(0.3 * SR))])
    segs = segment_audio(y, SR, cfg, onsets_s=np.array([]))
    assert len(segs) == 1
    seg = segs[0]
    assert seg.index == 0 and seg.kept
    assert abs(seg.start_s - 0.5) < ONSET_TOL_S  # primer frame sobre el umbral
    assert abs(seg.end_s - 1.0) < 0.04


def test_segment_audio_silent_or_empty_signal(cfg: SegmentationConfig) -> None:
    """Una señal silenciosa o vacía no produce segmentos."""
    assert segment_audio(np.zeros(SR, dtype=np.float32), SR, cfg) == []
    assert segment_audio(np.zeros(0, dtype=np.float32), SR, cfg) == []


def test_segment_audio_sorts_and_filters_onsets(cfg: SegmentationConfig) -> None:
    """Onsets desordenados, repetidos o fuera de la señal no rompen la segmentación."""
    y, onsets, _ = _notes_with_gaps(seed=2, f0s=(55.0, 98.0, 146.8))
    messy = np.array([onsets[2], onsets[0], onsets[1], onsets[0], 100.0])
    segs = segment_audio(y, SR, cfg, onsets_s=messy)
    assert [s.index for s in segs] == [0, 1, 2]
    np.testing.assert_allclose([s.start_s for s in segs], onsets, atol=1 / SR)


def test_segment_audio_detects_onsets_when_none_given(cfg: SegmentationConfig) -> None:
    """Sin onsets explícitos, segment_audio los detecta igual que detect_onsets."""
    y, _, _ = _notes_with_gaps(seed=0)
    auto = segment_audio(y, SR, cfg)
    manual = segment_audio(y, SR, cfg, onsets_s=detect_onsets(y, SR, cfg))
    assert [(s.start_sample, s.end_sample, s.reason) for s in auto] == [
        (s.start_sample, s.end_sample, s.reason) for s in manual
    ]


def test_segment_audio_logs_summary(cfg: SegmentationConfig, caplog: pytest.LogCaptureFixture) -> None:
    """El log resume onsets, segmentos conservados, silencios y cortos."""
    y, onsets, _ = _notes_with_gaps(seed=1, f0s=(55.0, 98.0))
    with caplog.at_level(logging.INFO, logger="src.segmentation"):
        segment_audio(y, SR, cfg, onsets_s=np.array([onsets[0], onsets[0] + 0.03, onsets[1]]))
    assert "3 onsets, 2 segmentos conservados, 0 silencios, 1 cortos" in caplog.text


def test_segment_properties() -> None:
    """Duración en s y MIDI (None sin f0) de un Segment."""
    seg = Segment(index=0, start_s=1.0, end_s=1.25, start_sample=22050, end_sample=27562, rms_db=-3.0)
    assert seg.duration_s == pytest.approx(0.25)
    assert seg.midi is None
    seg.f0_hz = 41.2
    assert seg.midi == pytest.approx(28.0, abs=0.01)
