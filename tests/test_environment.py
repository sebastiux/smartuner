"""Pruebas de la etapa 6: brazos, poda, espectro, datos por segmento y entorno bandit."""

from __future__ import annotations

import doctest
import logging

import numpy as np
import pytest

from src import environment
from src.config import EnvConfig
from src.environment import (
    Arm,
    BanditEnvironment,
    PullResult,
    SegmentBanditData,
    Spectrum,
    all_arms,
    build_all_segment_data,
    build_segment_data,
    candidate_arms,
    compute_spectrum,
    fret_frequency,
    next_hand_fret,
    playability_penalty,
    stft_max_frequency,
)
from src.segmentation import Segment

SR = 22050


def _tone(f0: float, dur: float = 2.0, amps: tuple[float, ...] = (1.0, 0.8, 0.5, 0.35, 0.25)) -> np.ndarray:
    """Tono armónico Σ_h a_h·sin(2π·h·f0·t) normalizado a pico 0.5 (float32)."""
    t = np.arange(int(SR * dur)) / SR
    y = sum(a * np.sin(2 * np.pi * (h + 1) * f0 * t) for h, a in enumerate(amps))
    return (0.5 * y / np.max(np.abs(y))).astype(np.float32)


def _segment(start: float = 0.0, end: float = 1.0, f0: float | None = 55.0, index: int = 0,
             kept: bool = True) -> Segment:
    """Segmento mínimo de ``start`` a ``end`` segundos con f0 opcional (conservado o descartado)."""
    return Segment(index=index, start_s=start, end_s=end, start_sample=int(start * SR),
                   end_sample=int(end * SR), rms_db=-6.0, kept=kept, reason="ok" if kept else "silencio",
                   f0_hz=f0, voiced_ratio=1.0 if f0 else 0.0)


def _data(arms: list[Arm], sal: np.ndarray, position: int = 0) -> SegmentBanditData:
    """SegmentBanditData con la matriz de saliencia ``sal`` dada a mano (frames × brazos)."""
    return SegmentBanditData(segment=_segment(), position=position, arms=arms,
                             frame_indices=np.arange(sal.shape[0]), salience=np.asarray(sal, dtype=float))


@pytest.fixture(scope="module")
def a1_spectrum() -> Spectrum:
    """CQT de un A1 (55 Hz) sintetizado de 2 s."""
    return compute_spectrum(_tone(55.0), SR, EnvConfig())


def test_module_doctests() -> None:
    """Los ejemplos de los docstrings de src.environment se ejecutan sin fallos."""
    result = doctest.testmod(environment)
    assert result.failed == 0 and result.attempted > 0


# ---------------------------------------------------------------------------
# Brazos
# ---------------------------------------------------------------------------


def test_fret_frequency_formula() -> None:
    """f(cuerda, traste) = f_cuerda·2^(traste/12): un traste = un semitono, 12 trastes = una octava."""
    assert fret_frequency("A", 0) == pytest.approx(55.0)
    assert fret_frequency("E", 5) == pytest.approx(55.0, abs=0.01)
    assert fret_frequency("G", 12) == pytest.approx(196.0)
    assert Arm("D", 7).freq_hz == pytest.approx(73.42 * 2 ** (7 / 12))
    # Un traste = un semitono = factor 2^(1/12).
    assert fret_frequency("E", 1) / fret_frequency("E", 0) == pytest.approx(2 ** (1 / 12))


def test_arm_properties() -> None:
    """Etiqueta 'D-3', MIDI = MIDI de la cuerda al aire + traste e índice de cuerda de grave a aguda."""
    arm = Arm("D", 3)
    assert arm.label == "D-3"
    assert arm.midi == 41
    assert arm.string_index == 2
    assert Arm("E", 0).midi == 28 and Arm("G", 12).midi == 55


def test_all_arms_count_and_order() -> None:
    """4 cuerdas × 13 trastes = 52 brazos únicos, ordenados por cuerda (E, A, D, G) y traste."""
    arms = all_arms()
    assert len(arms) == 52
    assert len(set(arms)) == 52
    assert [a.string for a in arms[::13]] == ["E", "A", "D", "G"]
    assert [a.fret for a in arms[:13]] == list(range(13))
    assert len(all_arms(5)) == 24
    with pytest.raises(ValueError):
        all_arms(-1)


def test_candidate_arms_k0_same_pitch_positions() -> None:
    """Con k = 0 solo quedan las posiciones del mismo pitch (A1: E-5 y A-0), ordenadas por cuerda."""
    arms = candidate_arms(55.0, 0)
    assert {a.label for a in arms} == {"A-0", "E-5"}
    assert [a.label for a in arms] == ["E-5", "A-0"]  # mismo MIDI → orden por cuerda (E antes que A)


def test_candidate_arms_k2_midis_and_order() -> None:
    """Con k = 2 quedan TODAS las posiciones a ±2 semitonos de la f0, ordenadas por (MIDI, cuerda)."""
    arms = candidate_arms(55.0, 2)
    assert {a.midi for a in arms} == {31, 32, 33, 34, 35}
    keys = [(a.midi, a.string_index) for a in arms]
    assert keys == sorted(keys)
    # Todas las posiciones de esos pitches entre los trastes 0..12.
    expected = {a for a in all_arms() if 31 <= a.midi <= 35}
    assert set(arms) == expected


def test_candidate_arms_uses_nearest_semitone() -> None:
    """Una f0 desafinada (+30 cents) apunta igualmente a su nota temperada."""
    detuned = 55.0 * 2 ** (30 / 1200)
    assert {a.label for a in candidate_arms(detuned, 0)} == {"A-0", "E-5"}


@pytest.mark.parametrize("f0, k", [(55.0, None), (None, 2), (None, None), (2000.0, 2), (10.0, 0)])
def test_candidate_arms_falls_back_to_all(f0: float | None, k: int | None) -> None:
    """Sin f0, sin poda (k = None) o con f0 fuera del rango del bajo se usan los 52 brazos."""
    assert len(candidate_arms(f0, k)) == 52


def test_candidate_arms_rejects_negative_k() -> None:
    """Un radio de poda negativo es un error."""
    with pytest.raises(ValueError):
        candidate_arms(55.0, -1)


# ---------------------------------------------------------------------------
# Penalización de tocabilidad
# ---------------------------------------------------------------------------


def test_playability_penalty_values() -> None:
    """Penalización λ·|traste − traste_previo|/12: valores a mano; 0 sin posición previa o con λ = 0."""
    pen = playability_penalty(np.array([0, 5, 7, 12]), 5, 0.12)
    np.testing.assert_allclose(pen, [0.05, 0.0, 0.02, 0.07])
    np.testing.assert_array_equal(playability_penalty(np.array([0, 5, 7]), None, 0.12), 0.0)
    np.testing.assert_array_equal(playability_penalty(np.array([0, 5, 7]), 3, 0.0), 0.0)


def test_open_string_free_removes_penalty_and_keeps_hand() -> None:
    """Con open_string_free la cuerda al aire no penaliza y no mueve la mano; sin la opción, la mano va al traste 0."""
    arms = [Arm("A", 0), Arm("E", 5), Arm("D", 2)]
    data = _data(arms, np.full((4, 3), 0.5))
    rng = np.random.default_rng(0)
    literal = BanditEnvironment(data, prev_fret=7, lam=0.12, rng=rng)
    free = BanditEnvironment(data, prev_fret=7, lam=0.12, rng=rng, open_string_free=True)
    np.testing.assert_allclose(literal.penalties, [0.07, 0.02, 0.05])
    np.testing.assert_allclose(free.penalties, [0.0, 0.02, 0.05])
    assert free.optimal_arms.tolist() == [0]

    cfg_free = EnvConfig(open_string_free=True)
    assert next_hand_fret(Arm("A", 0), 7, cfg_free) == 7
    assert next_hand_fret(Arm("A", 0), None, cfg_free) is None
    assert next_hand_fret(Arm("A", 0), 7, EnvConfig()) == 0
    assert next_hand_fret(Arm("E", 5), 7, cfg_free) == 5


def test_from_config_reads_parameters() -> None:
    """BanditEnvironment.from_config toma λ, el ruido σ y open_string_free de EnvConfig."""
    data = _data([Arm("A", 0), Arm("E", 5)], np.full((3, 2), 0.5))
    cfg = EnvConfig(lam=0.24, noise_std=0.05, open_string_free=True)
    env = BanditEnvironment.from_config(data, 5, cfg, np.random.default_rng(0))
    assert env.lam == 0.24 and env.noise_std == 0.05 and env.open_string_free
    np.testing.assert_allclose(env.penalties, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Entorno bandit
# ---------------------------------------------------------------------------


def _random_env(seed: int = 1, noise_std: float = 0.0, prev_fret: int | None = 3) -> BanditEnvironment:
    """Entorno de 5 brazos con saliencia aleatoria (40 frames), λ = 0.3 y semilla ``seed``."""
    rng = np.random.default_rng(100 + seed)
    arms = [Arm("E", 3), Arm("A", 0), Arm("A", 2), Arm("D", 1), Arm("E", 7)]
    sal = rng.random((40, len(arms)))
    return BanditEnvironment(_data(arms, sal), prev_fret=prev_fret, lam=0.3,
                             rng=np.random.default_rng(seed), noise_std=noise_std)


def test_true_means_are_mean_salience_minus_penalty() -> None:
    """μ_a = media de la saliencia sobre los frames − penalización (exacto, sin simular) y μ* = max μ."""
    env = _random_env()
    np.testing.assert_allclose(env.true_means, env.data.salience.mean(axis=0) - env.penalties)
    assert env.best_mean == pytest.approx(env.true_means.max())
    assert env.n_arms == 5
    assert env.arms == env.data.arms


@pytest.mark.parametrize("noise_std", [0.0, 0.2])
def test_true_means_match_empirical_mean_law_of_large_numbers(noise_std: float) -> None:
    """Ley de grandes números: la media de muchos pulls converge a μ_a exacto."""
    env = _random_env(seed=4, noise_std=noise_std)
    n = 20000
    for a in range(env.n_arms):
        rewards = np.array([env.pull(a) for _ in range(n)])
        sigma = np.sqrt(env.data.salience[:, a].var() + noise_std ** 2)
        assert abs(rewards.mean() - env.true_means[a]) < 5 * sigma / np.sqrt(n)


def test_same_seed_same_frame_sequence_for_any_arm() -> None:
    """Números aleatorios comunes: la secuencia de frames no depende del brazo jalado."""
    for noise_std in (0.0, 0.1):
        env_a = _random_env(seed=7, noise_std=noise_std)
        env_b = _random_env(seed=7, noise_std=noise_std)
        frames_a = [env_a.pull_detailed(0).frame_index for _ in range(200)]
        frames_b = [env_b.pull_detailed(t % env_b.n_arms).frame_index for t in range(200)]
        assert frames_a == frames_b
        assert len(set(frames_a)) > 1


def test_pull_and_pull_detailed_consume_rng_identically() -> None:
    """pull y pull_detailed consumen el generador igual: misma recompensa, con frame, saliencia, penalización y ruido coherentes."""
    env_fast = _random_env(seed=9, noise_std=0.1)
    env_slow = _random_env(seed=9, noise_std=0.1)
    for t in range(100):
        a = (3 * t) % env_fast.n_arms
        r = env_fast.pull(a)
        res = env_slow.pull_detailed(a)
        assert isinstance(res, PullResult)
        assert r == pytest.approx(res.reward)
        assert res.arm_index == a
        assert res.salience == env_slow.data.salience[res.frame_index, a]
        assert res.penalty == env_slow.penalties[a]
        assert res.reward == pytest.approx(res.salience - res.penalty + res.noise)


def test_no_noise_means_reward_is_salience_minus_penalty() -> None:
    """Sin ruido extra la recompensa es exactamente S_j(f_a) − penalización."""
    env = _random_env(seed=2)
    for _ in range(50):
        res = env.pull_detailed(1)
        assert res.noise == 0.0
        assert res.reward == pytest.approx(res.salience - res.penalty)


def test_regret_nonnegative_and_zero_for_optimal() -> None:
    """El regret Δ_a = μ* − μ_a es ≥ 0 y vale 0 exactamente en los brazos óptimos."""
    env = _random_env(seed=5)
    regrets = np.array([env.regret_of(a) for a in range(env.n_arms)])
    assert np.all(regrets >= 0)
    for a in range(env.n_arms):
        assert env.is_optimal(a) == (a in env.optimal_arms)
        assert (regrets[a] == 0.0) == env.is_optimal(a)
        assert regrets[a] == pytest.approx(env.best_mean - env.true_means[a])


def test_optimal_arms_include_exact_ties() -> None:
    """Brazos con μ empatado son todos óptimos (regret 0); el resto tiene Δ > 0."""
    sal = np.array([[0.5, 0.5, 0.2], [0.7, 0.7, 0.9]])
    env = BanditEnvironment(_data([Arm("A", 0), Arm("A", 0), Arm("D", 0)], sal), None, 0.1,
                            np.random.default_rng(0))
    assert env.optimal_arms.tolist() == [0, 1]
    assert env.regret_of(0) == 0.0 and env.regret_of(1) == 0.0
    assert env.regret_of(2) == pytest.approx(0.05)


def test_invalid_arm_index_raises() -> None:
    """Jalar un brazo fuera de 0..K−1 lanza IndexError (no se indexa en silencio)."""
    env = _random_env()
    with pytest.raises(IndexError):
        env.pull(env.n_arms)
    with pytest.raises(IndexError):
        env.pull_detailed(-1)


def test_same_pitch_optimum_is_least_movement_synthetic() -> None:
    """A-0 y E-5 tienen la misma saliencia: solo la penalización decide."""
    sal = np.array([[0.8, 0.8], [0.6, 0.6], [0.7, 0.7]])
    data = _data([Arm("E", 5), Arm("A", 0)], sal)
    rng = np.random.default_rng(0)
    assert BanditEnvironment(data, 4, 0.1, rng).optimal_arms.tolist() == [0]  # mano en 4 → E-5
    assert BanditEnvironment(data, 1, 0.1, rng).optimal_arms.tolist() == [1]  # mano en 1 → A-0
    assert BanditEnvironment(data, None, 0.1, rng).optimal_arms.tolist() == [0, 1]  # sin info → empate


def test_same_pitch_optimum_is_least_movement_real_spectrum(a1_spectrum: Spectrum) -> None:
    """Con un A1 sintetizado y k=0 (brazos E-5 y A-0) el óptimo sigue a la mano."""
    data = build_segment_data(a1_spectrum, _segment(0.3, 1.7, 55.0), 0, EnvConfig(k_semitones=0))
    labels = [a.label for a in data.arms]
    assert labels == ["E-5", "A-0"]
    # Saliencias IDÉNTICAS (se calculan una vez por pitch): el espectro no distingue la cuerda.
    np.testing.assert_array_equal(data.salience[:, 0], data.salience[:, 1])
    for prev, best in ((5, "E-5"), (7, "E-5"), (0, "A-0"), (1, "A-0")):
        env = BanditEnvironment(data, prev, 0.1, np.random.default_rng(0))
        assert [labels[i] for i in env.optimal_arms] == [best]


def test_same_pitch_twins_tie_exactly_without_penalty(a1_spectrum: Spectrum) -> None:
    """Regresión: con λ = 0 (o penalizaciones iguales) las posiciones gemelas son TODAS óptimas.

    Antes la saliencia se calculaba con la frecuencia de cada cuerda (55.0 Hz
    frente a 54.995 Hz por el redondeo de OPEN_STRING_HZ) y las gemelas
    diferían en ~10⁻⁴: siempre había un único óptimo, decidido por un
    artefacto numérico.
    """
    data = build_segment_data(a1_spectrum, _segment(0.3, 1.7, 55.0), 0, EnvConfig(k_semitones=None))
    by_midi: dict[int, list[int]] = {}
    for i, arm in enumerate(data.arms):
        by_midi.setdefault(arm.midi, []).append(i)
    for idx in by_midi.values():
        for i in idx[1:]:
            np.testing.assert_array_equal(data.salience[:, i], data.salience[:, idx[0]])
    env = BanditEnvironment(data, 3, 0.0, np.random.default_rng(0))
    assert sorted(data.arms[i].label for i in env.optimal_arms) == ["A-0", "E-5"]
    assert env.gaps[env.optimal_arms].tolist() == [0.0, 0.0]
    # Sin posición previa de la mano no hay penalización: las gemelas vuelven a empatar.
    env_none = BanditEnvironment(data, None, 0.1, np.random.default_rng(0))
    assert sorted(data.arms[i].label for i in env_none.optimal_arms) == ["A-0", "E-5"]
    # Penalizaciones iguales también empatan: D2 en E-10 y en D-0 con la mano en el traste 5.
    env_5 = BanditEnvironment(data, 5, 0.1, np.random.default_rng(0))
    labels = [a.label for a in data.arms]
    assert env_5.true_means[labels.index("E-10")] == env_5.true_means[labels.index("D-0")]


# ---------------------------------------------------------------------------
# Espectro
# ---------------------------------------------------------------------------


def test_compute_spectrum_cqt_peak_near_f0() -> None:
    """La CQT tiene bins logarítmicos crecientes desde cqt_fmin, hop de 256 muestras y su pico a < ½ semitono de la f0."""
    f0 = 98.0
    cfg = EnvConfig(spectrum="cqt")
    y = _tone(f0, amps=(1.0, 0.3, 0.1))
    spec = compute_spectrum(y, SR, cfg)
    assert spec.kind == "cqt" and spec.sr == SR and spec.hop_length == 256
    assert spec.mag.dtype == np.float32
    assert spec.mag.shape[0] == cfg.n_octaves * cfg.bins_per_octave == spec.freqs_hz.size
    assert spec.mag.shape[1] == spec.times_s.size
    assert np.all(np.diff(spec.freqs_hz) > 0)
    assert spec.freqs_hz[0] == pytest.approx(cfg.cqt_fmin_hz)
    assert spec.times_s[1] - spec.times_s[0] == pytest.approx(256 / SR)
    peak_hz = spec.freqs_hz[np.argmax(spec.mag[:, spec.mag.shape[1] // 2])]
    assert abs(12 * np.log2(peak_hz / f0)) < 0.5  # a menos de medio semitono


def test_compute_spectrum_stft() -> None:
    """La STFT guarda solo la banda útil (hasta stft_max_frequency), con bins de sr/n_fft Hz y el pico en la f0."""
    f0 = 110.0
    cfg = EnvConfig(spectrum="stft", n_fft=4096)
    spec = compute_spectrum(_tone(f0, amps=(1.0, 0.3)), SR, cfg, hop_length=512)
    assert spec.kind == "stft" and spec.hop_length == 512
    assert spec.mag.dtype == np.float32
    # Solo se guarda la banda que lee la recompensa (hasta ≈ 1.2 kHz, no hasta sr/2).
    assert spec.mag.shape[0] == spec.freqs_hz.size < 4096 // 2 + 1
    assert spec.freqs_hz[0] == 0.0
    assert spec.freqs_hz[-1] >= stft_max_frequency(cfg) > spec.freqs_hz[-2]
    np.testing.assert_allclose(np.diff(spec.freqs_hz), SR / 4096)
    peak_hz = spec.freqs_hz[np.argmax(spec.mag[:, spec.mag.shape[1] // 2])]
    assert abs(peak_hz - f0) <= SR / 4096
    # La saliencia funciona igual sobre el eje lineal de la STFT.
    data = build_segment_data(spec, _segment(0.3, 1.7, f0), 0, cfg)
    assert data.arms[int(np.argmax(data.mean_salience))].midi == 45


def test_compute_spectrum_rejects_unknown_kind() -> None:
    """Un tipo de espectro desconocido ('mel') es un error."""
    with pytest.raises(ValueError):
        compute_spectrum(_tone(55.0, dur=0.5), SR, EnvConfig(spectrum="mel"))


# ---------------------------------------------------------------------------
# Datos por segmento
# ---------------------------------------------------------------------------


def _toy_spectrum(n_frames: int = 100, hop_s: float = 0.01) -> Spectrum:
    """Espectro mínimo (eje de 1 Hz) con un A1 en todos los frames."""
    freqs = np.arange(1.0, 1001.0)
    mag = np.zeros((freqs.size, n_frames), dtype=np.float32)
    mag[[54, 109, 164], :] = np.array([1.0, 0.5, 0.3], dtype=np.float32)[:, None]
    return Spectrum(mag=mag, freqs_hz=freqs, times_s=np.arange(n_frames) * hop_s,
                    hop_length=220, sr=SR, kind="cqt")


def test_build_segment_data_skips_attack() -> None:
    """build_segment_data salta los primeros attack_skip_s de la nota y poda a ±k semitonos; el mejor brazo tiene el pitch correcto."""
    spec = _toy_spectrum()
    cfg = EnvConfig(attack_skip_s=0.03, k_semitones=2)
    data = build_segment_data(spec, _segment(0.195, 0.495, 55.0), 3, cfg)
    np.testing.assert_array_equal(data.frame_indices, np.arange(23, 50))  # [0.225, 0.495)
    assert data.position == 3
    assert data.salience.shape == (27, data.n_arms)
    assert {a.midi for a in data.arms} == {31, 32, 33, 34, 35}
    np.testing.assert_array_equal(data.frets, [a.fret for a in data.arms])
    best = data.arms[int(np.argmax(data.mean_salience))]
    assert best.midi == 33


def test_build_segment_data_frame_fallbacks() -> None:
    """Notas más cortas que el ataque usan [inicio, fin) y, si no hay frames dentro, el frame más cercano al centro."""
    spec = _toy_spectrum()
    cfg = EnvConfig(attack_skip_s=0.03)
    # Segmento más corto que el ataque: se usan los frames de [inicio, fin).
    short = build_segment_data(spec, _segment(0.195, 0.225, 55.0), 0, cfg)
    np.testing.assert_array_equal(short.frame_indices, [20, 21, 22])
    # Segmento entre dos frames: se usa el frame más cercano al centro.
    tiny = build_segment_data(spec, _segment(0.2012, 0.2048, 55.0), 0, cfg)
    np.testing.assert_array_equal(tiny.frame_indices, [20])
    assert tiny.salience.shape[0] == 1


def test_build_segment_data_unknown_f0_uses_all_arms() -> None:
    """Sin f0 (pYIN no detectó voz) no se poda: el segmento tiene los 52 brazos."""
    data = build_segment_data(_toy_spectrum(), _segment(0.1, 0.5, None), 0, EnvConfig())
    assert data.n_arms == 52
    assert data.arms == all_arms()


def test_build_segment_data_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    """El log INFO de cada segmento dice f0, brazos candidatos y el brazo de mayor saliencia media."""
    with caplog.at_level(logging.INFO, logger="src.environment"):
        build_segment_data(_toy_spectrum(), _segment(0.1, 0.5, 55.0), 7, EnvConfig(k_semitones=0))
    msg = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Segmento 7"))
    assert "f0=55.0 Hz" in msg
    assert "2 brazos candidatos (E-5, A-0)" in msg
    assert "mejor saliencia media:" in msg


def test_build_all_segment_data_only_kept_consecutive_positions() -> None:
    """Solo los segmentos conservados generan datos bandit, con posiciones consecutivas 0, 1, 2..."""
    spec = _toy_spectrum()
    segments = [
        _segment(0.00, 0.20, 55.0, index=0),
        _segment(0.20, 0.30, None, index=1, kept=False),
        _segment(0.30, 0.60, 73.42, index=2),
        _segment(0.60, 0.65, 55.0, index=3, kept=False),
        _segment(0.65, 0.95, 98.0, index=4),
    ]
    data = build_all_segment_data(spec, segments, EnvConfig())
    assert [d.position for d in data] == [0, 1, 2]
    assert [d.segment.index for d in data] == [0, 2, 4]
    assert build_all_segment_data(spec, [], EnvConfig()) == []


def test_environment_from_real_segment(a1_spectrum: Spectrum) -> None:
    """Integración: el óptimo de un A1 real con k=2 es una posición de MIDI 33."""
    cfg = EnvConfig()
    data = build_segment_data(a1_spectrum, _segment(0.3, 1.7, 55.3), 0, cfg)
    env = BanditEnvironment.from_config(data, cfg.initial_hand_fret, cfg, np.random.default_rng(0))
    assert all(data.arms[i].midi == 33 for i in env.optimal_arms)
    assert [data.arms[i].label for i in env.optimal_arms] == ["A-0"]  # mano en el traste 0
    rewards = [env.pull(int(env.optimal_arms[0])) for _ in range(50)]
    assert np.mean(rewards) > 0.5
