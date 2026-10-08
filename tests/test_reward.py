"""Pruebas de la recompensa del bandit: template armónico y saliencia S = max(S⁺ − β·S⁻, 0).

Se usan dos tipos de espectros sintéticos:

* **Matrices hechas a mano** sobre una rejilla lineal de 0.5 Hz, con picos
  exactamente en los armónicos h·f₀. Permiten comparar con los valores
  teóricos calculados a mano a partir de las ecuaciones del encabezado de
  :mod:`src.environment` (pruebas "de pizarra", útiles para la exposición).
* **CQT de tonos armónicos sintetizados** con numpy (sumas de senos), que
  incluyen la dispersión real de los picos entre bins vecinos.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import EnvConfig
from src.environment import (
    Arm,
    HarmonicTemplate,
    SILENCE_FLOOR,
    candidate_arms,
    compute_spectrum,
    harmonic_template,
    salience,
    salience_components,
)

SR = 22050

#: Perfil de un bajo con la fundamental DÉBIL: a₁ = 0.3, a₂ = 1 y luego
#: a_h = 1/(h−1) para h ≥ 3. Es el caso típico que provoca errores de octava.
WEAK_FUNDAMENTAL = [0.3, 1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9]

#: Perfil "normal" de bajo: armónicos que decaen desde la fundamental.
BASS_PROFILE = [1.0, 0.8, 0.5, 0.35, 0.25, 0.15]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _tone(f0: float, amps: list[float], dur: float = 2.0, peak: float = 0.5) -> np.ndarray:
    """Tono armónico Σ_h a_h·sin(2π·h·f0·t), normalizado a ``peak``."""
    t = np.arange(int(SR * dur)) / SR
    y = sum(a * np.sin(2 * np.pi * (h + 1) * f0 * t) for h, a in enumerate(amps))
    return (peak * y / np.max(np.abs(y))).astype(np.float32)


def _cqt_frames(f0: float, amps: list[float], cfg: EnvConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CQT del tono y frames centrales (lejos de los bordes, donde las
    ventanas largas de la CQT grave mezclan silencio)."""
    spec = compute_spectrum(_tone(f0, amps), SR, cfg or EnvConfig())
    return spec.mag[:, 40:-40], spec.freqs_hz


#: Rejilla lineal de 0.5 Hz (0.5 … 2000 Hz): los múltiplos de 27.5 Hz caen
#: exactamente en un bin, así que los armónicos de A1 = 55 Hz y sus huecos
#: (h−½)·55 se leen sin error de interpolación.
GRID_HZ = np.arange(1, 4001) * 0.5


def _comb_spectrum(f0: float, amps: list[float]) -> np.ndarray:
    """Espectro de un frame (forma ``(n_bins, 1)``) con picos a_h en h·f0."""
    mag = np.zeros((GRID_HZ.size, 1))
    for h, a in enumerate(amps, start=1):
        mag[int(round(h * f0 / 0.5)) - 1, 0] = a
    return mag


def _expected(amps: list[float], positions: np.ndarray, f0: float, n: int) -> float:
    """Σ_h w_h·X̃(posición_h) con X̃ = a/max(a): la fórmula "de pizarra"."""
    a = np.asarray(amps) / max(amps)
    w = (1.0 / np.arange(1, n + 1)) / np.sum(1.0 / np.arange(1, n + 1))
    total = 0.0
    for wh, x in zip(w, positions):
        ratio = x / f0  # número de armónico real que cae en x (si es entero)
        if abs(ratio - round(ratio)) < 1e-9 and 1 <= round(ratio) <= len(a):
            total += wh * a[int(round(ratio)) - 1]
    return total


def _reference_salience(mag: np.ndarray, freqs: np.ndarray, arm_freqs: np.ndarray, cfg: EnvConfig,
                        n_points: int = 5) -> np.ndarray:
    """Implementación de referencia con bucles (lenta pero obvia) para validar la vectorizada."""
    out = np.zeros((mag.shape[1], arm_freqs.size))
    valid = freqs > 0
    log_f = np.log2(freqs[valid])
    offsets = np.linspace(-cfg.tolerance_semitones, cfg.tolerance_semitones, n_points)
    h = np.arange(1, cfg.n_harmonics + 1)
    w = (1.0 / h) / np.sum(1.0 / h)
    for j in range(mag.shape[1]):
        col = mag[:, j].astype(float)
        if col.max() <= SILENCE_FLOOR:
            continue
        xn = (col / col.max())[valid]

        def read(x: float) -> float:
            pts = np.log2(x * 2.0 ** (offsets / 12.0))
            return float(np.max(np.interp(pts, log_f, xn, left=0.0, right=0.0)))

        for k, f in enumerate(arm_freqs):
            s_plus = sum(w[i] * read(hh * f) for i, hh in enumerate(h))
            s_minus = sum(w[i] * read((hh - 0.5) * f) for i, hh in enumerate(h))
            out[j, k] = max(s_plus - cfg.beta * s_minus, 0.0)
    return out


# ---------------------------------------------------------------------------
# Template armónico
# ---------------------------------------------------------------------------


def test_harmonic_template_positions_and_weights() -> None:
    t = harmonic_template(55.0, 5)
    assert isinstance(t, HarmonicTemplate)
    assert t.f0_hz == 55.0
    np.testing.assert_allclose(t.harmonic_hz, [55, 110, 165, 220, 275])
    np.testing.assert_allclose(t.interharmonic_hz, [27.5, 82.5, 137.5, 192.5, 247.5])
    # w_h = 1/h normalizados: suman 1 y decrecen con h.
    assert t.weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(t.weights) < 0)
    np.testing.assert_allclose(t.weights * np.arange(1, 6), t.weights[0])


@pytest.mark.parametrize("f0, n", [(0.0, 5), (-10.0, 5), (55.0, 0)])
def test_harmonic_template_rejects_invalid(f0: float, n: int) -> None:
    with pytest.raises(ValueError):
        harmonic_template(f0, n)


# ---------------------------------------------------------------------------
# Saliencia: fórmulas exactas en espectros hechos a mano
# ---------------------------------------------------------------------------


def test_salience_matches_hand_computed_formula() -> None:
    """S⁺ y S⁻ coinciden con la fórmula del encabezado calculada a mano."""
    f0, n = 55.0, 5
    cfg = EnvConfig(n_harmonics=n, beta=0.5)
    mag = _comb_spectrum(f0, BASS_PROFILE)
    cands = np.array([f0, 2 * f0, f0 / 2])
    s_plus, s_minus = salience_components(mag, GRID_HZ, cands, cfg)
    for k, f in enumerate(cands):
        t = harmonic_template(f, n)
        assert s_plus[0, k] == pytest.approx(_expected(BASS_PROFILE, t.harmonic_hz, f0, n))
        assert s_minus[0, k] == pytest.approx(_expected(BASS_PROFILE, t.interharmonic_hz, f0, n))
    s = salience(mag, GRID_HZ, cands, cfg)
    np.testing.assert_allclose(s, np.maximum(s_plus - 0.5 * s_minus, 0.0))


def test_beta_zero_is_pure_harmonic_sum() -> None:
    """Con β = 0 la saliencia es exactamente S⁺ (suma armónica pura)."""
    rng = np.random.default_rng(3)
    mag = rng.random((GRID_HZ.size, 4))
    cands = np.array([41.2, 55.0, 98.0])
    cfg = EnvConfig(beta=0.0)
    s_plus, _ = salience_components(mag, GRID_HZ, cands, cfg)
    np.testing.assert_allclose(salience(mag, GRID_HZ, cands, cfg), s_plus)


def test_octave_error_handmade_beta_fixes_it() -> None:
    """PRUEBA DIDÁCTICA: el término inter-armónico β·S⁻ corrige el error de octava.

    Nota A1 (f₀ = 55 Hz) con fundamental débil: a = [0.3, 1, 1/2, 1/3, ...].
    Con N = 5 y w_h ∝ 1/h (Σ 1/h = 2.2833):

    * Candidato correcto f₀ = 55 Hz:
      S⁺ = (1·0.3 + ½·1 + ⅓·½ + ¼·⅓ + ⅕·¼)/2.2833 ≈ 0.482;  S⁻ = 0
      (sus huecos 27.5, 82.5, 137.5... Hz están vacíos).
    * Candidato una octava arriba 2f₀ = 110 Hz: sus armónicos 110, 220,
      330... son los armónicos PARES de la nota, todos presentes:
      S⁺ = (1·1 + ½·⅓ + ⅓·⅕ + ¼·⅐ + ⅕·⅑)/2.2833 ≈ 0.566 > 0.482  ← ¡gana con β = 0!
      Sus huecos (h−½)·110 = 55, 165, 275... son los armónicos IMPARES:
      S⁻ = (1·0.3 + ½·½ + ⅓·¼ + ¼·⅙ + ⅕·⅛)/2.2833 ≈ 0.307.
    * Con β = 0.5: S(110) = 0.566 − 0.5·0.307 ≈ 0.412 < 0.482 = S(55) → gana el correcto.
    """
    mag = _comb_spectrum(55.0, WEAK_FUNDAMENTAL)
    cands = np.array([55.0, 110.0])

    s0 = salience(mag, GRID_HZ, cands, EnvConfig(beta=0.0))[0]
    assert s0[0] == pytest.approx(0.4818, abs=1e-3)
    assert s0[1] == pytest.approx(0.5655, abs=1e-3)
    assert s0[1] > s0[0], "con β=0 la suma armónica pura prefiere la octava superior"

    s5 = salience(mag, GRID_HZ, cands, EnvConfig(beta=0.5))[0]
    assert s5[0] == pytest.approx(0.4818, abs=1e-3)  # el correcto no pierde nada (S⁻ = 0)
    assert s5[1] == pytest.approx(0.5655 - 0.5 * 0.3066, abs=1e-3)
    assert s5[0] > s5[1], "con β=0.5 el candidato correcto gana"


def test_octave_error_cqt_beta_fixes_it() -> None:
    """Lo mismo que la prueba anterior pero con la CQT de un tono sintetizado.

    La CQT de librosa amplifica algo los graves (filtros más largos), así que
    la fundamental débil "pesa" un poco más que en la matriz a mano; aun así,
    con β = 0 la octava superior iguala o supera al candidato correcto y con
    β = 0.5 el correcto gana con holgura.
    """
    f0 = 55.0
    mag, freqs = _cqt_frames(f0, WEAK_FUNDAMENTAL)
    cands = np.array([f0, 2 * f0])
    s0 = salience(mag, freqs, cands, EnvConfig(beta=0.0)).mean(axis=0)
    s5 = salience(mag, freqs, cands, EnvConfig(beta=0.5)).mean(axis=0)
    assert s0[1] >= s0[0] - 0.02  # β=0: la octava arriba iguala o supera al correcto
    assert s5[0] > s5[1] + 0.1    # β=0.5: el correcto gana claramente
    # β solo resta: el correcto (sin energía entre armónicos) apenas cambia.
    assert s5[0] == pytest.approx(s0[0], abs=0.02)


def test_lower_octave_loses_through_empty_harmonics() -> None:
    """El error de octava hacia ABAJO ya lo castiga S⁺ (la mitad de su peine cae en zonas vacías)."""
    mag = _comb_spectrum(110.0, BASS_PROFILE)
    s = salience(mag, GRID_HZ, np.array([110.0, 55.0]), EnvConfig(beta=0.0))[0]
    assert s[0] > 1.5 * s[1]


def test_out_of_range_positions_read_zero() -> None:
    """Las posiciones fuera del eje de frecuencias aportan 0 (no se extrapola)."""
    freqs = np.arange(1.0, 151.0)  # espectro que solo llega a 150 Hz
    mag = np.zeros((freqs.size, 1))
    mag[[54, 109], 0] = [1.0, 0.5]  # 55 y 110 Hz
    cfg = EnvConfig(n_harmonics=5, beta=0.0, tolerance_semitones=0.0)
    w = harmonic_template(55.0, 5).weights
    s = salience(mag, freqs, np.array([55.0]), cfg)
    assert s[0, 0] == pytest.approx(w[0] * 1.0 + w[1] * 0.5)  # 165, 220, 275 Hz → 0


def test_vectorized_matches_reference_loop() -> None:
    """La implementación vectorizada (gather) coincide con un bucle ingenuo."""
    rng = np.random.default_rng(11)
    cfg = EnvConfig(beta=0.7, tolerance_semitones=0.33, n_harmonics=6)
    arm_freqs = np.array([Arm(s, f).freq_hz for s in "EADG" for f in range(13)])
    # Eje CQT (logarítmico) y eje STFT (lineal, con bin DC de 0 Hz).
    cqt_freqs = 32.70 * 2.0 ** (np.arange(216) / 36.0)
    stft_freqs = np.arange(2049) * SR / 4096
    for freqs in (cqt_freqs, stft_freqs):
        mag = rng.random((freqs.size, 6)) ** 4
        mag[:, 2] = 0.0  # un frame silencioso
        np.testing.assert_allclose(salience(mag, freqs, arm_freqs, cfg),
                                   _reference_salience(mag, freqs, arm_freqs, cfg), atol=1e-12)


# ---------------------------------------------------------------------------
# Propiedades generales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0])
def test_salience_in_unit_interval(beta: float) -> None:
    rng = np.random.default_rng(5)
    freqs = 32.70 * 2.0 ** (np.arange(216) / 36.0)
    mag = rng.random((216, 30)) * rng.random(30) * 10.0
    arm_freqs = np.array([Arm(s, f).freq_hz for s in "EADG" for f in range(13)])
    s = salience(mag, freqs, arm_freqs, EnvConfig(beta=beta))
    assert s.shape == (30, 52)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_salience_of_real_tone_in_unit_interval() -> None:
    mag, freqs = _cqt_frames(73.42, BASS_PROFILE)
    arm_freqs = np.array([Arm(s, f).freq_hz for s in "EADG" for f in range(13)])
    s = salience(mag, freqs, arm_freqs, EnvConfig(beta=0.0))
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert s.max() > 0.5


def test_silent_frame_gives_zero() -> None:
    mag = _comb_spectrum(55.0, BASS_PROFILE).repeat(3, axis=1)
    mag[:, 1] = 0.0                       # silencio digital
    mag[:, 2] = SILENCE_FLOOR * 1e-3      # "casi" silencio (por debajo del umbral)
    s = salience(mag, GRID_HZ, np.array([55.0, 110.0]), EnvConfig())
    assert s[0, 0] > 0.5
    np.testing.assert_array_equal(s[1:], 0.0)


def test_salience_is_volume_invariant() -> None:
    """Normalizar por el máximo del frame hace que el volumen no importe."""
    mag, freqs = _cqt_frames(55.0, BASS_PROFILE)
    cands = np.array([a.freq_hz for a in candidate_arms(55.0, 2)])
    cfg = EnvConfig()
    np.testing.assert_allclose(salience(mag, freqs, cands, cfg), salience(0.01 * mag, freqs, cands, cfg),
                               rtol=1e-6, atol=1e-9)


def test_single_frame_and_empty_inputs() -> None:
    cfg = EnvConfig()
    one = _comb_spectrum(55.0, BASS_PROFILE)[:, 0]  # 1-D = un frame
    assert salience(one, GRID_HZ, np.array([55.0]), cfg).shape == (1, 1)
    assert salience(np.zeros((GRID_HZ.size, 0)), GRID_HZ, np.array([55.0]), cfg).shape == (0, 1)
    with pytest.raises(ValueError):
        salience(np.zeros((10, 2)), GRID_HZ, np.array([55.0]), cfg)  # bins incoherentes


# ---------------------------------------------------------------------------
# Saliencia en tonos sintetizados (CQT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [Arm("E", 0), Arm("E", 1), Arm("A", 3), Arm("D", 5), Arm("G", 9), Arm("G", 12)],
                         ids=lambda a: a.label)
def test_correct_pitch_has_highest_salience_among_candidates(arm: Arm) -> None:
    """Entre los candidatos a ±2 semitonos, los brazos con el pitch correcto
    (en cualquier cuerda) tienen la mayor saliencia media, con mucha ventaja."""
    mag, freqs = _cqt_frames(arm.freq_hz, BASS_PROFILE)
    cands = candidate_arms(arm.freq_hz, 2)
    s = salience(mag, freqs, np.array([a.freq_hz for a in cands]), EnvConfig()).mean(axis=0)
    correct = np.array([a.midi == arm.midi for a in cands])
    assert correct.any() and (~correct).any()
    assert cands[int(np.argmax(s))].midi == arm.midi
    assert s[correct].min() > 5 * s[~correct].max()


def test_tuning_tolerance_absorbs_detuning() -> None:
    """Un tono desafinado 20 cents sigue dando una saliencia alta a su nota.

    Con tolerancia 0 la lectura cae justo al lado del pico y la saliencia
    baja: la ventana ±``tolerance_semitones`` existe para absorber la
    desafinación y la inarmonicidad de las cuerdas reales.
    """
    nominal = 55.0
    neighbors = nominal * 2.0 ** (np.array([-2, -1, 0, 1, 2]) / 12.0)
    cfg = EnvConfig(tolerance_semitones=0.33)
    tuned = salience(*_cqt_frames(nominal, BASS_PROFILE), neighbors, cfg).mean(axis=0)
    for cents in (20.0, -20.0):
        detuned_mag, freqs = _cqt_frames(nominal * 2.0 ** (cents / 1200.0), BASS_PROFILE)
        s = salience(detuned_mag, freqs, neighbors, cfg).mean(axis=0)
        assert s[2] >= 0.95 * tuned[2]
        assert s[2] > 2 * np.delete(s, 2).max()
        strict = salience(detuned_mag, freqs, neighbors, EnvConfig(tolerance_semitones=0.0)).mean(axis=0)
        assert strict[2] < s[2] - 0.03
