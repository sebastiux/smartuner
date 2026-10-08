"""Pruebas de la etapa 3 (pasa-bajas de fase cero y normalización)."""

from __future__ import annotations

import doctest

import numpy as np
import pytest
from scipy.signal import butter, correlate, correlation_lags, sosfilt

from src import preprocessing
from src.config import PreprocessConfig
from src.preprocessing import PreprocessResult, lowpass, normalize_peak, preprocess

SR = 22050


def _tone(freq: float, dur: float = 1.0, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    """Seno de ``freq`` Hz, ``dur`` s y amplitud ``amp`` (float32)."""
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms_db(y: np.ndarray) -> float:
    """Nivel RMS de ``y`` en dB (0 dB = RMS 1)."""
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def _attenuation_db(freq: float, cutoff: float = 400.0) -> float:
    """Atenuación (dB) de un tono puro, medida lejos de los bordes."""
    y = _tone(freq)
    out = lowpass(y, SR, cutoff)
    core = slice(2000, -2000)  # se ignoran los extremos (transitorios del relleno)
    return _rms_db(y[core]) - _rms_db(out[core])


# ---------------------------------------------------------------------------
# lowpass
# ---------------------------------------------------------------------------


def test_lowpass_attenuates_above_cutoff() -> None:
    """El pasa-bajas de 400 Hz atenúa más de 20 dB un tono de 1500 Hz."""
    assert _attenuation_db(1500.0) > 20.0


def test_lowpass_passes_below_cutoff() -> None:
    """Un tono de 100 Hz (banda de paso) pasa con menos de 1 dB de cambio."""
    assert abs(_attenuation_db(100.0)) < 1.0


def test_lowpass_is_zero_phase_no_delay() -> None:
    """La correlación cruzada entrada/salida es máxima en el lag 0 (no hay retardo)."""
    rng = np.random.default_rng(1234)
    y = rng.standard_normal(SR).astype(np.float32)  # ruido blanco: excita todas las frecuencias
    out = lowpass(y, SR, 400.0)
    lags = correlation_lags(len(y), len(out), mode="full")
    xcorr = correlate(out, y, mode="full")
    assert lags[np.argmax(xcorr)] == 0

    # Control: el mismo Butterworth aplicado de forma causal (una sola pasada)
    # SÍ retrasa la señal; así se comprueba que la prueba detecta el retardo.
    causal = sosfilt(butter(4, 400.0, fs=SR, output="sos"), y)
    assert lags[np.argmax(correlate(causal, y, mode="full"))] > 0


def test_lowpass_impulse_response_is_symmetric() -> None:
    """Fase cero ⇔ respuesta al impulso simétrica alrededor del instante del impulso."""
    n = 4001
    impulse = np.zeros(n)
    impulse[n // 2] = 1.0
    h = lowpass(impulse, SR, 400.0)
    assert np.argmax(h) == n // 2
    np.testing.assert_allclose(h, h[::-1], atol=1e-10)


@pytest.mark.parametrize("n", [22050, 12345, 7, 2, 1])
def test_lowpass_preserves_length_and_dtype(n: int) -> None:
    """El filtro conserva la longitud y el tipo float32, incluso en señales muy cortas."""
    y = np.random.default_rng(0).uniform(-1, 1, n).astype(np.float32)
    out = lowpass(y, SR, 400.0)
    assert out.shape == y.shape
    assert out.dtype == np.float32


def test_lowpass_above_nyquist_returns_copy() -> None:
    """Un corte ≥ Nyquist no filtra: devuelve una copia idéntica."""
    y = _tone(3000.0)
    out = lowpass(y, SR, SR / 2)
    assert out is not y
    np.testing.assert_array_equal(out, y)


def test_lowpass_does_not_modify_input() -> None:
    """El filtro no modifica el arreglo de entrada."""
    y = _tone(1500.0)
    original = y.copy()
    lowpass(y, SR, 400.0)
    np.testing.assert_array_equal(y, original)


def test_lowpass_rejects_invalid_cutoff() -> None:
    """Un corte ≤ 0 Hz es un error."""
    with pytest.raises(ValueError):
        lowpass(_tone(100.0), SR, 0.0)


# ---------------------------------------------------------------------------
# normalize_peak
# ---------------------------------------------------------------------------


def test_normalize_peak_sets_peak() -> None:
    """normalize_peak lleva el pico a 0.99 aplicando solo una ganancia (sin deformar la onda ni tocar la entrada)."""
    y = np.array([0.1, -0.4, 0.2], dtype=np.float32)
    original = y.copy()
    out = normalize_peak(y)
    assert np.max(np.abs(out)) == pytest.approx(0.99, abs=1e-6)
    assert out.dtype == np.float32
    # La forma de onda no cambia: solo una ganancia.
    np.testing.assert_allclose(out / out[1], y / y[1], rtol=1e-6)
    np.testing.assert_array_equal(y, original)  # entrada intacta


def test_normalize_peak_custom_peak() -> None:
    """normalize_peak acepta otro pico objetivo."""
    out = normalize_peak(_tone(55.0, amp=0.1), peak=0.5)
    assert np.max(np.abs(out)) == pytest.approx(0.5, abs=1e-6)


def test_normalize_peak_null_signal_unchanged() -> None:
    """Una señal nula (o vacía) no se normaliza: no hay división entre 0."""
    zeros = np.zeros(100, dtype=np.float32)
    out = normalize_peak(zeros)
    np.testing.assert_array_equal(out, zeros)
    assert np.all(np.isfinite(out))
    assert normalize_peak(np.zeros(0)).size == 0


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_returns_two_aligned_signals() -> None:
    """preprocess devuelve y_analysis (filtrada + normalizada) y y_spectral (solo normalizada, conserva armónicos agudos)."""
    low, high = _tone(55.0, amp=0.3), _tone(1500.0, amp=0.3)
    y = low + high
    res = preprocess(y, SR, PreprocessConfig(lowpass_hz=400.0, filter_order=4, normalize=True))
    assert isinstance(res, PreprocessResult)
    assert res.sr == SR
    assert len(res.y_analysis) == len(res.y_spectral) == len(y)

    # y_spectral: solo normalizada (conserva el armónico agudo).
    np.testing.assert_allclose(res.y_spectral, normalize_peak(y), atol=1e-6)
    # y_analysis: filtrada (sin el tono de 1500 Hz) y normalizada.
    assert np.max(np.abs(res.y_analysis)) == pytest.approx(0.99, abs=1e-6)
    expected = normalize_peak(low)
    core = slice(2000, -2000)
    np.testing.assert_allclose(res.y_analysis[core], expected[core], atol=0.02)


def test_preprocess_without_normalization() -> None:
    """Con normalize=False ninguna señal se amplifica."""
    y = _tone(80.0, amp=0.2)
    res = preprocess(y, SR, PreprocessConfig(normalize=False))
    np.testing.assert_array_equal(res.y_spectral, y)
    assert res.y_spectral is not y
    assert np.max(np.abs(res.y_analysis)) < 0.25  # no se amplificó


def test_doctests_pass() -> None:
    """Los ejemplos de los docstrings de src.preprocessing se ejecutan sin fallos."""
    result = doctest.testmod(preprocessing)
    assert result.failed == 0
