"""Pruebas de los experimentos: bucle bandit, cadenas, oráculo, evaluación, experimento completo,
ejecución en vivo y CLI.

Casi todas usan entornos sintéticos pequeños construidos a mano
(:class:`SegmentBanditData` con matrices de saliencia de numpy), así que son
rápidas y no dependen del audio. La prueba de extremo a extremo de la CLI
(``main.py --cli analyze``) está marcada como ``slow`` y requiere ffmpeg.
"""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from src import io_audio
from src.agents import make_agent
from src.config import ALGORITHMS, PROJECT_ROOT, CancelledError, Config
from src.environment import Arm, BanditEnvironment, SegmentBanditData, Spectrum
from src.experiments import (
    OUTCOME_EXACT,
    OUTCOME_LABELS,
    OUTCOME_MISSED,
    OUTCOME_WRONG_PITCH,
    OUTCOME_WRONG_POSITION,
    SUMMARY_COLUMNS,
    TIMING_COLUMNS,
    AccuracyResult,
    ExperimentResult,
    LiveSession,
    chain_value,
    effective_sweep_runs,
    estimate_experiment_seconds,
    match_segments_to_gt,
    oracle_arms,
    oracle_viterbi_arms,
    run_chain,
    run_example_segment,
    run_experiment,
    run_lambda_sweep,
    run_segment,
    run_sensitivity,
    score_arms,
    segment_rngs,
    transcribe,
)
from src.pipeline import AnalysisResult
from src.pitch import PitchTrack
from src.segmentation import Segment
from src.synth_dataset import GTNote

needs_ffmpeg = pytest.mark.skipif(io_audio.find_ffmpeg() is None, reason="ffmpeg no instalado")


# ---------------------------------------------------------------------------
# Construcción de entornos sintéticos
# ---------------------------------------------------------------------------


def make_segment(position: int, start_s: float, dur_s: float = 0.4, f0_hz: float | None = None) -> Segment:
    """Segmento conservado mínimo."""
    return Segment(index=position, start_s=start_s, end_s=start_s + dur_s, start_sample=int(start_s * 22050),
                   end_sample=int((start_s + dur_s) * 22050), rms_db=-3.0, f0_hz=f0_hz)


def make_data(position: int, arms: list[Arm], means: list[float], n_frames: int = 40, noise: float = 0.15,
              seed: int = 0, start_s: float | None = None) -> SegmentBanditData:
    """Segmento bandit sintético: saliencia[frame, brazo] = μ_brazo + ruido, recortada a [0, 1]."""
    rng = np.random.default_rng(seed)
    sal = np.asarray(means, dtype=float)[None, :] + noise * rng.standard_normal((n_frames, len(arms)))
    sal = np.clip(sal, 0.0, 1.0)
    start = 0.5 + 0.5 * position if start_s is None else start_s
    return SegmentBanditData(make_segment(position, start), position, list(arms), np.arange(n_frames), sal)


def make_chain_data() -> list[SegmentBanditData]:
    """Tres segmentos: A1 (A-0 vs E-5), C2 (A-3 vs E-8) y D2 (D-0 vs A-5), con un brazo de otro pitch."""
    return [
        make_data(0, [Arm("E", 5), Arm("A", 0), Arm("A", 1)], [0.70, 0.70, 0.25], seed=1),
        make_data(1, [Arm("E", 8), Arm("A", 3), Arm("A", 2)], [0.65, 0.65, 0.30], seed=2),
        make_data(2, [Arm("A", 5), Arm("D", 0), Arm("D", 1)], [0.60, 0.60, 0.20], seed=3),
    ]


def make_gt(data: list[SegmentBanditData], labels: list[str]) -> list[GTNote]:
    """Ground truth alineado con los segmentos (mismo onset)."""
    notes = []
    for d, label in zip(data, labels):
        string, fret = label.split("-")
        arm = Arm(string, int(fret))
        notes.append(GTNote(onset_s=d.segment.start_s, offset_s=d.segment.end_s, midi=arm.midi,
                            string=string, fret=int(fret)))
    return notes


def make_analysis(segment_data: list[SegmentBanditData], gt: list[GTNote] | None) -> AnalysisResult:
    """AnalysisResult mínimo (sin audio real) para probar los experimentos."""
    empty = np.zeros(0)
    return AnalysisResult(
        path=Path("sintetico.mp3"), source_path=Path("sintetico.mp3"), sr=22050,
        y_raw=np.zeros(22050), y_analysis=np.zeros(22050), y_spectral=np.zeros(22050),
        onsets_s=np.array([d.segment.start_s for d in segment_data]),
        segments=[d.segment for d in segment_data],
        pitch_track=PitchTrack(empty, empty, empty, empty.astype(bool)),
        spectrum=Spectrum(np.zeros((1, 1), dtype=np.float32), np.array([55.0]), np.array([0.0]), 256, 22050, "cqt"),
        segment_data=segment_data, ground_truth=gt, config=Config(),
    )


def small_config(**experiment: object) -> Config:
    """Configuración pequeña para pruebas rápidas."""
    cfg = Config()
    cfg.env.budget = 40
    cfg.env.lam = 0.1
    cfg.env.initial_hand_fret = 0
    cfg.experiment.n_runs = 4
    cfg.experiment.sweep_runs = 2
    cfg.experiment.sweep_epsilon = [0.05, 0.3]
    cfg.experiment.sweep_q0 = [1.0, 5.0]
    cfg.experiment.sweep_c = [0.5, 2.0]
    cfg.experiment.sweep_tau = [0.05, 0.5]
    cfg.experiment.sweep_lambda = [0.0, 0.2]
    cfg.experiment.onset_tolerance_s = 0.05
    for key, value in experiment.items():
        setattr(cfg.experiment, key, value)
    return cfg


# ---------------------------------------------------------------------------
# run_segment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_run_segment_trace_is_consistent(algorithm: str) -> None:
    """La traza de run_segment es coherente: formas (T,), Q final = última fila de q_history, conteos = histograma de brazos y regret = brecha μ* − μ_a."""
    data = make_data(0, [Arm("A", 0), Arm("A", 1), Arm("A", 2), Arm("A", 3)], [0.2, 0.8, 0.5, 0.4])
    env = BanditEnvironment(data, prev_fret=None, lam=0.0, rng=np.random.default_rng(0))
    agent = make_agent(algorithm, data.n_arms, Config().agent, rng=np.random.default_rng(1))
    run = run_segment(agent, env, budget=60, record_q=True)

    assert run.arms.shape == run.rewards.shape == run.regrets.shape == run.optimal.shape == (60,)
    assert run.q_history is not None and run.q_history.shape == (60, 4)
    np.testing.assert_array_equal(run.q_history[-1], run.final_q)
    assert run.final_counts.sum() == 60
    np.testing.assert_array_equal(np.bincount(run.arms, minlength=4), run.final_counts)
    # Regret instantáneo = brecha μ* − μ_a del brazo jalado (≥ 0, 0 solo en óptimos).
    np.testing.assert_allclose(run.regrets, env.gaps[run.arms])
    assert np.all(run.regrets >= 0)
    np.testing.assert_array_equal(run.optimal, run.regrets == 0)
    assert run.recommended == agent.recommend("most_pulled")
    # Cada recompensa es una celda de la matriz de saliencia (λ=0, sin ruido).
    for a, r in zip(run.arms, run.rewards):
        assert np.any(np.isclose(data.salience[:, a], r))


def test_run_segment_resets_agent_and_validates() -> None:
    """run_segment reinicia el agente (no acumula conteos entre llamadas) y rechaza T < 1 o un K distinto del entorno."""
    data = make_data(0, [Arm("A", 0), Arm("A", 1)], [0.9, 0.1])
    env = BanditEnvironment(data, prev_fret=None, lam=0.0, rng=np.random.default_rng(0))
    agent = make_agent("egreedy", 2, Config().agent, rng=np.random.default_rng(1))
    first = run_segment(agent, env, budget=30)
    second = run_segment(agent, env, budget=30)   # el agente se reinicia: no acumula conteos
    assert first.final_counts.sum() == second.final_counts.sum() == 30
    assert first.q_history is None
    with pytest.raises(ValueError):
        run_segment(agent, env, budget=0)
    with pytest.raises(ValueError):
        run_segment(make_agent("ucb1", 3, Config().agent), env, budget=10)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_run_segment_finds_clear_best_arm(algorithm: str) -> None:
    """Con un brazo claramente mejor (0.9 frente a ≤ 0.3) todos los algoritmos lo recomiendan y lo jalan ≥ 60 % al final."""
    data = make_data(0, [Arm("A", 0), Arm("A", 1), Arm("A", 2)], [0.2, 0.9, 0.3], noise=0.05)
    env = BanditEnvironment(data, prev_fret=None, lam=0.0, rng=np.random.default_rng(3))
    agent = make_agent(algorithm, 3, Config().agent, rng=np.random.default_rng(4))
    run = run_segment(agent, env, budget=300)
    assert run.recommended == 1
    assert run.optimal[-30:].mean() >= 0.6


# ---------------------------------------------------------------------------
# run_chain: traste previo, reproducibilidad y números aleatorios comunes
# ---------------------------------------------------------------------------


def test_run_chain_shapes_and_hand_position_chain() -> None:
    """run_chain encadena la mano: el traste recomendado en s es el traste previo de s+1 y μ incluye esa penalización."""
    data = make_chain_data()
    cfg = small_config()
    cfg.env.initial_hand_fret = 7
    chain = run_chain("ucb1", data, cfg, run_index=0)
    assert chain.algorithm == "ucb1"
    assert chain.rewards.shape == chain.regrets.shape == chain.optimal.shape == (3, cfg.env.budget)
    assert chain.optimal.dtype == bool
    assert len(chain.arms) == len(chain.prev_frets) == len(chain.final_q) == len(chain.true_means) == 3
    assert chain.elapsed_s > 0
    # El traste recomendado en s es el traste previo de s+1.
    assert chain.prev_frets[0] == 7
    for s in range(2):
        assert chain.prev_frets[s + 1] == chain.arms[s].fret
    # μ de cada segmento = saliencia media − λ|traste − previo|/12.
    for s, d in enumerate(data):
        expected = d.mean_salience - cfg.env.lam * np.abs(d.frets - chain.prev_frets[s]) / 12.0
        np.testing.assert_allclose(chain.true_means[s], expected)
    assert np.all(chain.regrets >= 0)
    assert all(arm in d.arms for arm, d in zip(chain.arms, data))


def test_run_chain_is_reproducible_and_seed_dependent() -> None:
    """Misma semilla y corrida → cadena idéntica; otra corrida u otra semilla maestra → otra cadena."""
    data = make_chain_data()
    cfg = small_config()
    a = run_chain("softmax", data, cfg, run_index=2)
    b = run_chain("softmax", data, cfg, run_index=2)
    np.testing.assert_array_equal(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.regrets, b.regrets)
    assert a.arms == b.arms
    c = run_chain("softmax", data, cfg, run_index=3)
    assert not np.array_equal(a.rewards, c.rewards)
    cfg2 = copy.deepcopy(cfg)
    cfg2.experiment.seed += 1
    d = run_chain("softmax", data, cfg2, run_index=2)
    assert not np.array_equal(a.rewards, d.rewards)


@pytest.mark.parametrize("noise_std", [0.0, 0.05])
def test_common_random_numbers_same_frames_for_all_algorithms(noise_std: float) -> None:
    """Si todos los brazos valen lo mismo en cada frame (y λ=0), la recompensa depende solo
    del frame muestreado (y del ruido): todos los algoritmos deben ver EXACTAMENTE la misma
    secuencia de recompensas."""
    rng = np.random.default_rng(0)
    data = []
    for s in range(3):
        column = rng.uniform(0.1, 0.9, size=50)
        sal = np.repeat(column[:, None], 4, axis=1)
        arms = [Arm("A", f) for f in range(4)]
        data.append(SegmentBanditData(make_segment(s, 0.5 + s), s, arms, np.arange(50), sal))
    cfg = small_config()
    cfg.env.lam = 0.0
    cfg.env.noise_std = noise_std
    rewards = {algo: run_chain(algo, data, cfg, run_index=5).rewards for algo in ALGORITHMS}
    for algo in ALGORITHMS[1:]:
        np.testing.assert_array_equal(rewards[algo], rewards[ALGORITHMS[0]])


def test_segment_rngs_share_environment_and_separate_agents() -> None:
    """Números aleatorios comunes: el generador del ENTORNO es igual para todos los algoritmos y el del agente no."""
    env_a, ag_a = segment_rngs(42, 1, 2, "egreedy")
    env_b, ag_b = segment_rngs(42, 1, 2, "ucb1")
    np.testing.assert_array_equal(env_a.random(5), env_b.random(5))
    assert not np.array_equal(ag_a.random(5), ag_b.random(5))
    with pytest.raises(ValueError):
        segment_rngs(42, 0, 0, "desconocido")


def test_run_chain_and_transcribe_can_be_cancelled() -> None:
    """Con ``cancel`` activo, run_chain/transcribe lanzan CancelledError antes del siguiente segmento (GUI: «Cancelar»)."""
    import threading

    from src.config import CancelledError
    from src.experiments import transcribe

    data = make_chain_data()
    cancel = threading.Event()
    assert len(run_chain("ucb1", data, small_config(), cancel=cancel).arms) == 3  # sin cancelar, completa
    cancel.set()
    with pytest.raises(CancelledError, match="segmento 0 de 3"):
        run_chain("ucb1", data, small_config(), cancel=cancel)
    with pytest.raises(CancelledError):
        transcribe(make_analysis(data, None), "softmax", small_config(), cancel=cancel)


def test_run_chain_rejects_unknown_algorithm() -> None:
    """Un algoritmo desconocido en run_chain es un error con mensaje en español."""
    with pytest.raises(ValueError, match="desconocido"):
        run_chain("thompson", make_chain_data(), small_config())


def test_run_chain_open_string_free_keeps_hand() -> None:
    """Con open_string_free, tocar una cuerda al aire deja la mano donde estaba para el segmento siguiente."""
    data = [make_data(0, [Arm("A", 0)], [0.7]), make_data(1, [Arm("D", 4)], [0.7])]
    cfg = small_config()
    cfg.env.initial_hand_fret = 5
    cfg.env.open_string_free = True
    chain = run_chain("egreedy", data, cfg)
    assert chain.prev_frets == [5, 5]   # la cuerda al aire no mueve la mano


# ---------------------------------------------------------------------------
# Oráculo
# ---------------------------------------------------------------------------


def _tie_data(position: int = 0) -> SegmentBanditData:
    """E-5 y A-0 con EXACTAMENTE la misma saliencia (mismo pitch)."""
    sal = np.array([[0.8, 0.8, 0.1], [0.6, 0.6, 0.2]])
    return SegmentBanditData(make_segment(position, 0.5 + position), position,
                             [Arm("E", 5), Arm("A", 0), Arm("A", 2)], np.arange(2), sal)


def test_oracle_uses_playability_and_tie_breaks() -> None:
    """El oráculo miope usa la penalización de tocabilidad y desempata (λ = 0) por menor movimiento y luego menor traste."""
    cfg = small_config()
    cfg.env.lam = 0.1
    cfg.env.initial_hand_fret = 5
    assert oracle_arms([_tie_data()], cfg) == [Arm("E", 5)]       # λ>0: E-5 no mueve la mano
    cfg.env.initial_hand_fret = 0
    assert oracle_arms([_tie_data()], cfg) == [Arm("A", 0)]
    # λ = 0: empate exacto en μ → menor movimiento.
    cfg.env.lam = 0.0
    cfg.env.initial_hand_fret = 4
    assert oracle_arms([_tie_data()], cfg) == [Arm("E", 5)]       # |5−4| < |0−4|
    # Mismo movimiento desde el traste 4 (|5−4| = |3−4| = 1): gana el menor traste.
    sal = np.array([[0.5, 0.5]])
    data = SegmentBanditData(make_segment(0, 0.5), 0, [Arm("D", 5), Arm("A", 3)], np.arange(1), sal)
    assert oracle_arms([data], cfg) == [Arm("A", 3)]


def test_oracle_chain_carries_hand_position() -> None:
    """El oráculo miope propaga su propia posición de la mano: tras A-7, un empate de saliencia lo resuelve A-7."""
    cfg = small_config()
    cfg.env.lam = 0.5
    cfg.env.initial_hand_fret = 0
    # Segmento 0 claramente A-7; en el segmento 1 hay empate de saliencia entre D-2 y A-7:
    # con la mano en 7 gana A-7.
    d0 = SegmentBanditData(make_segment(0, 0.5), 0, [Arm("A", 7), Arm("A", 1)], np.arange(1), np.array([[0.9, 0.1]]))
    d1 = SegmentBanditData(make_segment(1, 1.0), 1, [Arm("D", 2), Arm("A", 7)], np.arange(1), np.array([[0.7, 0.7]]))
    assert oracle_arms([d0, d1], cfg) == [Arm("A", 7), Arm("A", 7)]


def _brute_force_chain(data: list[SegmentBanditData], cfg: Config) -> float:
    """Máximo de Σ_s μ_s(a_s | traste previo) probando TODAS las combinaciones (referencia obvia)."""
    import itertools

    return max(chain_value(list(combo), data, cfg) for combo in itertools.product(*(d.arms for d in data)))


@pytest.mark.parametrize("open_string_free", [False, True])
def test_viterbi_oracle_is_optimal_chain(open_string_free: bool) -> None:
    """El oráculo de cadena alcanza el máximo de Σ μ (fuerza bruta) y nunca queda por debajo del miope."""
    rng = np.random.default_rng(7)
    pool = [Arm("E", 5), Arm("A", 0), Arm("A", 7), Arm("D", 2), Arm("E", 12), Arm("G", 0), Arm("D", 9)]
    for trial in range(12):
        data = []
        for pos in range(4):
            arms = [pool[i] for i in rng.choice(len(pool), size=3, replace=False)]
            means = rng.choice([0.5, 0.6, 0.7], size=3).tolist()          # muchos empates de saliencia
            data.append(make_data(pos, arms, means, n_frames=5, noise=0.0, seed=trial))
        cfg = small_config()
        cfg.env.lam = float(rng.choice([0.0, 0.3, 1.0]))
        cfg.env.initial_hand_fret = int(rng.integers(0, 13))
        cfg.env.open_string_free = open_string_free
        best = _brute_force_chain(data, cfg)
        viterbi = oracle_viterbi_arms(data, cfg)
        assert all(a in d.arms for a, d in zip(viterbi, data))
        assert chain_value(viterbi, data, cfg) == pytest.approx(best, abs=1e-9)
        assert chain_value(oracle_arms(data, cfg), data, cfg) <= best + 1e-9


def test_viterbi_oracle_beats_myopic_when_hand_position_matters() -> None:
    """Caso didáctico: el miope se acerca a la mano ahora y paga un salto después."""
    cfg = small_config()
    cfg.env.lam = 1.2
    cfg.env.initial_hand_fret = 3
    d0 = SegmentBanditData(make_segment(0, 0.5), 0, [Arm("E", 5), Arm("A", 0)], np.arange(1), np.array([[0.8, 0.8]]))
    d1 = SegmentBanditData(make_segment(1, 1.0), 1, [Arm("D", 0)], np.arange(1), np.array([[0.8]]))
    assert oracle_arms([d0, d1], cfg) == [Arm("E", 5), Arm("D", 0)]
    assert oracle_viterbi_arms([d0, d1], cfg) == [Arm("A", 0), Arm("D", 0)]
    assert chain_value([Arm("A", 0), Arm("D", 0)], [d0, d1], cfg) == pytest.approx(1.3)
    assert oracle_viterbi_arms([], cfg) == []
    with pytest.raises(ValueError):
        chain_value([Arm("G", 3)], [d0], cfg)       # no es un brazo candidato


# ---------------------------------------------------------------------------
# Emparejamiento y precisión
# ---------------------------------------------------------------------------


def test_match_segments_basic_and_missed_note() -> None:
    """Cada nota GT se empareja con el segmento a ≤ tolerancia; las notas sin segmento quedan en None."""
    segs = [make_segment(i, t) for i, t in enumerate([0.52, 1.03, 2.40])]
    gt = [GTNote(0.50, 0.9, 33, "A", 0), GTNote(1.00, 1.4, 35, "A", 2), GTNote(1.70, 2.1, 38, "D", 0)]
    assert match_segments_to_gt(segs, gt, 0.05) == [0, 1, None]   # la nota de 1.70 s no se detectó
    assert match_segments_to_gt(segs, gt, 0.01) == [None, None, None]
    assert match_segments_to_gt([], gt, 0.05) == [None, None, None]
    assert match_segments_to_gt(segs, [], 0.05) == []


def test_match_segments_is_maximum_one_to_one() -> None:
    """Regresión: el emparejamiento es MÁXIMO (como mir_eval), no greedy por distancia.

    Notas a (1.00 s) y b (1.04 s); segmentos en 1.03 y 1.07 s. El greedy daba
    el segmento de 1.03 a b (el par más cercano, 10 ms) y dejaba a sin pareja
    (1.07 está a 70 ms > 50 ms). El emparejamiento máximo empareja las dos
    notas: a ↔ 1.03 (30 ms) y b ↔ 1.07 (30 ms).
    """
    segs = [make_segment(0, 1.03), make_segment(1, 1.07)]
    gt = [GTNote(1.00, 1.2, 33, "A", 0), GTNote(1.04, 1.3, 33, "A", 0)]
    assert match_segments_to_gt(segs, gt, 0.05) == [0, 1]
    # Con más tolerancia hay dos emparejamientos completos; gana el de menor distancia total
    # (30 + 30 ms frente a 70 + 10 ms).
    assert match_segments_to_gt(segs, gt, 0.08) == [0, 1]
    # Ningún segmento se usa dos veces.
    many = match_segments_to_gt([make_segment(0, 1.0)], gt, 0.1)
    assert sorted(m for m in many if m is not None) == [0]


def test_match_segments_fast_sixteenths_with_onset_lag() -> None:
    """Semicorcheas a 140 bpm (IOI 0.107 s) con onsets ≈ 60 ms tarde: ninguna nota se pierde."""
    segs = [make_segment(0, 0.066), make_segment(1, 0.170)]
    gt = [GTNote(0.0, 0.1, 33, "A", 0), GTNote(0.107, 0.2, 35, "A", 2)]
    assert match_segments_to_gt(segs, gt, 0.08) == [0, 1]


def test_score_arms_counts_extra_segments_in_precision_and_f1() -> None:
    """Regresión: un segmento espurio no cambia el recall pero sí la precisión por segmento y el F1."""
    gt = [GTNote(0.5, 0.9, 33, "A", 0), GTNote(1.0, 1.4, 35, "A", 2)]
    segs = [make_segment(0, 0.52), make_segment(1, 0.70), make_segment(2, 1.02)]
    matches = match_segments_to_gt(segs, gt, 0.08)
    assert matches == [0, 2]
    res = score_arms([Arm("A", 0), Arm("G", 12), Arm("A", 2)], matches, gt)
    assert res.pitch == res.position == 1.0                 # recall: las 2 notas GT, acertadas
    assert res.n_segments == 3 and res.n_matched == 2 and res.n_extra == 1 and res.n_missed == 0
    assert res.position_precision == pytest.approx(2 / 3)   # 2 de 3 notas de la tablatura
    assert res.position_f1 == pytest.approx(0.8)            # 2·2 / (2 + 3)
    assert res.pitch_f1 == pytest.approx(0.8)
    # Sin segmentos de más, F1 = recall = precisión.
    clean = score_arms([Arm("A", 0), Arm("A", 2)], [0, 1], gt)
    assert clean.n_extra == 0 and clean.position_f1 == pytest.approx(1.0)
    # AccuracyResult construido sin n_segments (resultados antiguos): precisión y F1 desconocidos.
    legacy = AccuracyResult(1.0, 1.0, np.array([OUTCOME_EXACT, OUTCOME_EXACT]))
    assert legacy.n_extra is None and np.isnan(legacy.position_f1) and np.isnan(legacy.pitch_precision)


def test_score_arms_outcomes() -> None:
    """score_arms clasifica cada nota GT (exacta, otra posición, pitch incorrecto, no detectada) y calcula los recall."""
    gt = [
        GTNote(0.5, 0.9, 33, "A", 0),   # exacta
        GTNote(1.0, 1.4, 33, "A", 0),   # mismo pitch, otra posición (E-5)
        GTNote(1.5, 1.9, 38, "D", 0),   # pitch incorrecto (A-4 = MIDI 37)
        GTNote(2.0, 2.4, 40, "D", 2),   # no detectada
    ]
    arms = [Arm("A", 0), Arm("E", 5), Arm("A", 4)]
    res = score_arms(arms, [0, 1, 2, None], gt)
    assert res.outcomes.tolist() == [OUTCOME_EXACT, OUTCOME_WRONG_POSITION, OUTCOME_WRONG_PITCH, OUTCOME_MISSED]
    assert res.pitch == pytest.approx(2 / 4)
    assert res.position == pytest.approx(1 / 4)
    assert set(OUTCOME_LABELS) == {0, 1, 2, 3}


def test_score_arms_edge_cases() -> None:
    """Sin ground truth las precisiones son NaN; longitudes incoherentes son un error."""
    empty = score_arms([], [], [])
    assert np.isnan(empty.pitch) and np.isnan(empty.position) and empty.outcomes.size == 0
    with pytest.raises(ValueError):
        score_arms([Arm("A", 0)], [0], [])


# ---------------------------------------------------------------------------
# Transcripción y experimento completo
# ---------------------------------------------------------------------------


def test_transcribe_builds_notes_from_chain() -> None:
    """transcribe convierte las posiciones recomendadas de una cadena en notas de tablatura con sus tiempos."""
    data = make_chain_data()
    tr = transcribe(make_analysis(data, None), "egreedy", small_config())
    assert tr.algorithm == "egreedy"
    assert [n.label for n in tr.notes] == [a.label for a in tr.chain.arms]
    assert [n.position for n in tr.notes] == [0, 1, 2]
    assert tr.notes[1].start_s == pytest.approx(data[1].segment.start_s)


#: (resultado, configuración usada, copia previa de la configuración, fracciones de progreso, segmentos).
ExperimentFixture = tuple[ExperimentResult, Config, Config, list[float], list[SegmentBanditData]]


@pytest.fixture(scope="module")
def experiment_result() -> ExperimentFixture:
    """Experimento pequeño (3 segmentos, 4 corridas, barridos) compartido por las pruebas del módulo."""
    data = make_chain_data()
    gt = make_gt(data, ["A-0", "A-3", "D-0"])
    cfg = small_config()
    cfg_before = copy.deepcopy(cfg)
    fractions: list[float] = []
    res = run_experiment(make_analysis(data, gt), cfg, progress=lambda f, m: fractions.append(f))
    return res, cfg, cfg_before, fractions, data


def test_experiment_curves_and_shapes(experiment_result: ExperimentFixture) -> None:
    """Las curvas tienen forma (corridas, T), el regret acumulado es ≥ 0 y no decrece, y run_experiment no muta la configuración."""
    res, cfg, cfg_before, _, data = experiment_result
    runs, T, S = cfg.experiment.n_runs, cfg.env.budget, len(data)
    assert res.algorithms == list(ALGORITHMS)
    assert (res.n_runs, res.budget, res.n_segments) == (runs, T, S)
    for algo in ALGORITHMS:
        assert res.reward_curves[algo].shape == (runs, T)
        assert res.regret_curves[algo].shape == (runs, T)
        assert res.optimal_curves[algo].shape == (runs, T)
        assert res.choices[algo].shape == (runs, S, 2)
        assert res.elapsed_s[algo].shape == (runs,)
        # Regret acumulado: ≥ 0 y no decreciente.
        regret = res.regret_curves[algo]
        assert np.all(regret >= 0)
        assert np.all(np.diff(regret, axis=1) >= -1e-12)
        assert np.all((res.optimal_curves[algo] >= 0) & (res.optimal_curves[algo] <= 1))
        assert len(res.accuracy[algo]) == runs
    assert cfg.to_dict() == cfg_before.to_dict()      # run_experiment no muta la configuración
    assert res.config.to_dict() == cfg.to_dict() and res.config is not cfg


def test_experiment_curves_match_chains(experiment_result: ExperimentFixture) -> None:
    """Las curvas de la corrida r son exactamente las de run_chain(r) promediadas sobre segmentos."""
    res, cfg, _, _, data = experiment_result
    chain = run_chain("ucb1", data, cfg, run_index=1)
    np.testing.assert_allclose(res.reward_curves["ucb1"][1], chain.rewards.mean(axis=0))
    np.testing.assert_allclose(res.regret_curves["ucb1"][1], np.cumsum(chain.regrets, axis=1).mean(axis=0))
    np.testing.assert_allclose(res.optimal_curves["ucb1"][1], chain.optimal.mean(axis=0))
    assert [tuple(c) for c in res.choices["ucb1"][1]] == [(a.string_index, a.fret) for a in chain.arms]


def test_experiment_oracle_example_and_sweeps(experiment_result: ExperimentFixture) -> None:
    """El experimento guarda el oráculo, el segmento ejemplo (conteos y Q medias) y los barridos con las formas correctas."""
    res, cfg, _, _, data = experiment_result
    assert res.oracle_arms == oracle_arms(data, cfg)
    assert res.oracle_accuracy is not None and res.oracle_accuracy.pitch == pytest.approx(1.0)
    assert res.gt_notes is not None and len(res.gt_notes) == 3

    ex = res.example
    assert ex is not None
    assert ex.position == 0                 # todos tienen 3 brazos → el primero
    assert ex.arm_labels == [a.label for a in data[0].arms]
    assert ex.true_means.shape == (3,)
    for algo in ALGORITHMS:
        assert ex.counts[algo].shape == (cfg.experiment.n_runs, 3)
        assert np.all(ex.counts[algo].sum(axis=1) == cfg.env.budget)
        assert ex.q_mean[algo].shape == ex.q_std[algo].shape == (cfg.env.budget, 3)

    assert [s.param for s in res.sweeps] == ["agent.epsilon", "agent.q0", "agent.ucb_c", "agent.tau"]
    assert [s.algorithm for s in res.sweeps] == list(ALGORITHMS)
    for sweep in res.sweeps:
        assert sweep.final_reward.shape == sweep.mean_reward.shape == sweep.final_regret.shape == (2, 2)
        assert np.all(sweep.final_regret >= 0)

    lam = res.lambda_sweep
    assert lam is not None and lam.values == [0.0, 0.2]
    for algo in ALGORITHMS:
        assert lam.position_acc[algo].shape == lam.pitch_acc[algo].shape == (2, 2)
    assert lam.oracle_position_acc.shape == lam.oracle_pitch_acc.shape == (2,)


def test_experiment_progress_is_monotonic(experiment_result: ExperimentFixture) -> None:
    """El progreso informado es monótono, está en [0, 1] y termina en 1."""
    fractions = experiment_result[3]
    assert fractions and fractions[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-12 for a, b in zip(fractions, fractions[1:]))
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_summary_rows_and_csv(experiment_result: ExperimentFixture, tmp_path: Path) -> None:
    """La tabla resumen y los CSV (resumen, tiempos, curvas) tienen las columnas y los valores esperados."""
    res = experiment_result[0]
    rows = res.summary_rows()
    assert [r["algorithm"] for r in rows] == list(ALGORITHMS) + ["oracle", "oracle_viterbi"]
    for row in rows:
        assert set(SUMMARY_COLUMNS) <= set(row)
    egreedy = rows[0]
    assert egreedy["label"] == "ε-greedy"
    assert egreedy["mean_reward"] == pytest.approx(res.reward_curves["egreedy"].mean())
    assert egreedy["final_regret"] == pytest.approx(res.regret_curves["egreedy"][:, -1].mean())
    n_final = max(1, int(np.ceil(0.1 * res.budget)))
    assert egreedy["optimal_pct"] == pytest.approx(100 * res.optimal_curves["egreedy"][:, -n_final:].mean())
    assert egreedy["pitch_acc"] == pytest.approx(np.mean([a.pitch for a in res.accuracy["egreedy"]]))
    assert egreedy["ms_per_segment"] > 0
    assert egreedy["n_extra"] == 0
    assert egreedy["position_f1"] == pytest.approx(np.mean([a.position_f1 for a in res.accuracy["egreedy"]]))
    oracle = rows[-2]
    assert oracle["final_regret"] == 0.0 and oracle["optimal_pct"] == 100.0 and oracle["mean_reward"] is None
    viterbi = rows[-1]
    assert viterbi["final_regret"] is None and viterbi["optimal_pct"] is None
    assert viterbi["position_acc"] == pytest.approx(res.viterbi_accuracy.position)

    summary = Path(res.to_csv(str(tmp_path / "sub" / "summary.csv")))
    with summary.open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 6 and list(table[0]) == list(SUMMARY_COLUMNS)
    assert table[-1]["mean_reward"] == ""
    assert "ms_per_segment" not in table[0]          # el tiempo de pared va aparte (no es reproducible)

    timing = Path(res.timing_to_csv(str(tmp_path / "timing.csv")))
    with timing.open(encoding="utf-8") as fh:
        times = list(csv.DictReader(fh))
    assert [t["algorithm"] for t in times] == list(ALGORITHMS)
    assert list(times[0]) == list(TIMING_COLUMNS)
    assert float(times[0]["ms_per_segment"]) == pytest.approx(egreedy["ms_per_segment"], rel=1e-4)

    curves = Path(res.curves_to_csv(str(tmp_path / "curves.csv")))
    with curves.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["algorithm", "pull", "reward_mean", "reward_std", "regret_mean",
                                     "regret_std", "optimal_mean", "optimal_std"]
        lines = list(reader)
    assert len(lines) == 4 * res.budget
    assert lines[0]["pull"] == "1" and lines[res.budget - 1]["pull"] == str(res.budget)


def test_experiment_csvs_are_reproducible(experiment_result: ExperimentFixture, tmp_path: Path) -> None:
    """Misma semilla y configuración → summary.csv y curves.csv idénticos byte a byte."""
    res, cfg, _, _, data = experiment_result
    gt = make_gt(data, ["A-0", "A-3", "D-0"])
    again = run_experiment(make_analysis(data, gt), cfg)
    for name, writer in (("summary.csv", "to_csv"), ("curves.csv", "curves_to_csv")):
        a = getattr(res, writer)(str(tmp_path / "a" / name))
        b = getattr(again, writer)(str(tmp_path / "b" / name))
        assert Path(a).read_bytes() == Path(b).read_bytes()


def test_modal_arms(experiment_result: ExperimentFixture) -> None:
    """modal_arms devuelve la posición más frecuente entre corridas para cada segmento (desempate: la primera en aparecer)."""
    res, _, _, _, data = experiment_result
    for algo in ALGORITHMS:
        modal = res.modal_arms(algo)
        assert len(modal) == len(data)
        assert all(arm in d.arms for arm, d in zip(modal, data))
    # Caso hecho a mano: (A, 0) dos veces vs (E, 5) una vez → A-0.
    res2 = copy.copy(res)
    res2.choices = {"egreedy": np.array([[[1, 0]], [[0, 5]], [[1, 0]]])}
    assert res2.modal_arms("egreedy") == [Arm("A", 0)]


def test_experiment_rejects_segment_data_from_other_env_config() -> None:
    """Regresión: si cfg.env cambia la saliencia o los brazos, run_experiment no corre con datos viejos."""
    data = make_chain_data()
    analysis = make_analysis(data, None)
    for key, value in (("env.k_semitones", None), ("env.beta", 0.0), ("env.n_fft", 4096),
                       ("pitch.min_voiced_ratio", 0.5)):
        cfg = small_config(run_sweeps=False, run_lambda_sweep=False)
        cfg.set(key, value)
        with pytest.raises(ValueError, match=key.split(".")[1]):
            run_experiment(analysis, cfg)
    # λ, T, la mano inicial y el ruido solo afectan al entorno: se aceptan.
    cfg = small_config(run_sweeps=False, run_lambda_sweep=False, n_runs=1)
    cfg.env.lam, cfg.env.noise_std = 0.3, 0.01
    assert run_experiment(analysis, cfg).config.env.lam == 0.3


def test_effective_sweep_runs_and_duration_estimate() -> None:
    """Los barridos reducen corridas en pistas largas y la duración estimada crece linealmente con S."""
    cfg = Config()
    assert effective_sweep_runs(cfg, 10) == effective_sweep_runs(cfg, 50) == 30
    assert effective_sweep_runs(cfg, 100) == 15 and effective_sweep_runs(cfg, 1000) == 5
    cfg.experiment.sweep_runs = 3
    assert effective_sweep_runs(cfg, 1000) == 3                 # nunca más que las pedidas
    est16, est32 = estimate_experiment_seconds(16, Config()), estimate_experiment_seconds(32, Config())
    assert est32["main"] == pytest.approx(2 * est16["main"])
    assert est16["total"] == pytest.approx(sum(v for k, v in est16.items() if k != "total"))
    assert estimate_experiment_seconds(16, Config(), with_lambda=False)["lambda"] == 0.0


def test_lambda_sweep_uses_fewer_runs_on_long_tracks() -> None:
    """Con 200 segmentos el barrido de λ usa 5 corridas por valor (no 10) e incluye el oráculo de cadena."""
    data = [make_data(i, [Arm("E", 5), Arm("A", 0)], [0.7, 0.7], n_frames=4, seed=i, start_s=0.5 + 0.5 * i)
            for i in range(200)]
    gt = make_gt(data, ["A-0"] * 200)
    cfg = small_config(sweep_runs=10, sweep_lambda=[0.0, 0.5], algorithms=["ucb1"])
    cfg.env.budget = 10
    res = run_lambda_sweep(make_analysis(data, gt), cfg)
    assert res is not None and res.position_acc["ucb1"].shape == (2, 5)
    assert res.viterbi_position_acc is not None and res.viterbi_position_acc.shape == (2,)
    cfg.experiment.sweep_lambda = []
    assert run_lambda_sweep(make_analysis(data, gt), cfg) is None


def test_experiment_without_ground_truth() -> None:
    """Sin ground truth no hay precisiones, oráculos en la tabla ni barrido de λ, pero el resto del experimento funciona."""
    data = make_chain_data()
    cfg = small_config(run_sweeps=False, algorithms=["ucb1", "egreedy"])
    res = run_experiment(make_analysis(data, None), cfg)
    assert res.algorithms == ["egreedy", "ucb1"]         # orden fijo de ALGORITHMS
    assert res.accuracy == {} and res.oracle_accuracy is None and res.gt_notes is None
    assert res.lambda_sweep is None and res.sweeps == []
    rows = res.summary_rows()
    assert [r["algorithm"] for r in rows] == ["egreedy", "ucb1"]
    assert all(r["pitch_acc"] is None and r["position_acc"] is None for r in rows)
    assert run_lambda_sweep(make_analysis(data, None), cfg) is None


def test_experiment_cancel_and_errors() -> None:
    """Activar ``cancel`` detiene el experimento con CancelledError (antes de empezar o tras la primera cadena)."""
    data = make_chain_data()
    cfg = small_config(run_sweeps=False, run_lambda_sweep=False)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        run_experiment(make_analysis(data, None), cfg, cancel=cancel)

    cancel2 = threading.Event()
    calls: list[float] = []

    def progress(fraction: float, message: str) -> None:
        """Callback de progreso que cancela el experimento en cuanto se informa la primera cadena."""
        calls.append(fraction)
        cancel2.set()        # cancelar después de la primera cadena

    with pytest.raises(CancelledError):
        run_experiment(make_analysis(data, None), cfg, progress=progress, cancel=cancel2)
    assert len(calls) == 1

    with pytest.raises(ValueError):
        run_experiment(make_analysis([], None), cfg)
    with pytest.raises(ValueError):
        run_experiment(make_analysis(data, None), small_config(algorithms=["thompson"]))


def test_run_sensitivity_does_not_mutate_config_and_uses_values() -> None:
    """Los barridos usan una COPIA de la configuración con cada valor y reproducen exactamente la cadena equivalente."""
    data = make_chain_data()
    cfg = small_config(algorithms=["egreedy", "ucb1"])
    before = cfg.to_dict()
    sweeps = run_sensitivity(data, cfg)
    assert cfg.to_dict() == before
    assert [s.param for s in sweeps] == ["agent.epsilon", "agent.ucb_c"]
    eps = sweeps[0]
    assert eps.values == [0.05, 0.3]
    # El primer valor del barrido reproduce exactamente una cadena con ese ε.
    cfg_v = copy.deepcopy(cfg)
    cfg_v.agent.epsilon = 0.05
    chain = run_chain("egreedy", data, cfg_v, run_index=1)
    assert eps.mean_reward[0, 1] == pytest.approx(chain.rewards.mean())
    assert eps.final_regret[0, 1] == pytest.approx(chain.regrets.sum(axis=1).mean())


def test_run_example_segment_defaults_and_errors() -> None:
    """El segmento ejemplo por defecto es el de más brazos; un índice inexistente cae al de más brazos o es un error si se pide explícitamente."""
    data = make_chain_data() + [make_data(3, [Arm("A", f) for f in range(5)], [0.1, 0.2, 0.9, 0.3, 0.4])]
    cfg = small_config()
    cfg.experiment.n_runs = 3
    ex = run_example_segment(data, cfg)
    assert ex.position == 3                               # el de más brazos
    assert ex.optimal_arms.tolist() == [int(np.argmax(ex.true_means))]
    cfg.experiment.example_segment = 1
    assert run_example_segment(data, cfg).position == 1
    cfg.experiment.example_segment = 99                   # inexistente → el de más brazos
    assert run_example_segment(data, cfg).position == 3
    with pytest.raises(IndexError):
        run_example_segment(data, cfg, position=10)
    with pytest.raises(ValueError):
        run_example_segment([], cfg)


# ---------------------------------------------------------------------------
# LiveSession
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_live_session_steps_until_done(algorithm: str) -> None:
    """La ejecución en vivo hace exactamente T pasos con texto explicativo, Q coherente y regret acumulado; después lanza RuntimeError."""
    data = make_chain_data()[0]
    cfg = small_config()
    cfg.env.budget = 15
    live = LiveSession(data, algorithm, cfg, prev_fret=0, seed=1)
    assert live.t == 0 and not live.done and live.events == []
    events = [live.step() for _ in range(15)]
    assert live.done and live.t == 15 and len(live.events) == 15
    assert [e.t for e in events] == list(range(1, 16))
    with pytest.raises(RuntimeError):
        live.step()

    for e in events:
        assert e.text.startswith(f"t={e.t} | ")
        assert f"→ {e.arm_label} |" in e.text and "frame #" in e.text and "| Q " in e.text
        assert e.reward == pytest.approx(e.salience - e.penalty)
        assert e.arm_label == data.arms[e.arm_index].label
        assert e.decision.arm == e.arm_index
    # Q después de cada pull = regla incremental aplicada a Q antes.
    q_final = live.agent.q
    last_by_arm = {e.arm_index: e for e in events}
    for arm, e in last_by_arm.items():
        assert q_final[arm] == pytest.approx(e.q_after)
    gaps = live.env.gaps
    assert events[-1].cumulative_regret == pytest.approx(sum(gaps[e.arm_index] for e in events))
    assert all(e.is_optimal == (gaps[e.arm_index] == 0) for e in events)
    assert live.recommended_arm() == data.arms[live.agent.recommend(cfg.agent.recommend)]


def test_live_session_reset_replays_and_matches_chain() -> None:
    """reset() repite la misma secuencia y la sesión con semilla r reproduce el segmento de la corrida r del experimento."""
    data = make_chain_data()
    cfg = small_config()
    cfg.env.budget = 25
    live = LiveSession(data[0], "softmax", cfg, prev_fret=cfg.env.initial_hand_fret, seed=2)
    first = [live.step().text for _ in range(10)]
    live.reset()
    assert live.t == 0 and live.events == [] and live.cumulative_regret == 0.0
    while not live.done:
        live.step()
    assert [e.text for e in live.events[:10]] == first
    # La sesión con semilla r reproduce el segmento 0 de la corrida r del experimento.
    chain = run_chain("softmax", data, cfg, run_index=2)
    np.testing.assert_allclose([e.reward for e in live.events], chain.rewards[0])


def test_live_session_log_shows_master_seed_and_run(caplog: pytest.LogCaptureFixture) -> None:
    """Regresión: el log dice la semilla MAESTRA (la que muestra En vivo) y aparte la corrida, no «semilla 0»."""
    import logging

    cfg = small_config()
    cfg.experiment.seed = 42
    with caplog.at_level(logging.INFO, logger="src.experiments"):
        LiveSession(make_chain_data()[0], "ucb1", cfg, prev_fret=0, seed=0)
    assert "semilla 42, corrida 0" in caplog.text


def test_live_session_text_example_format() -> None:
    """El texto de un pull sigue el formato 't=… | UCB1 → … | frame #…: S=… − pen … = r … | Q …→…'."""
    data = make_chain_data()[0]
    cfg = small_config()
    live = LiveSession(data, "ucb1", cfg, prev_fret=0)
    ev = live.step()
    assert ev.decision.kind == "init"
    assert ev.text.startswith("t=1 | UCB1 → ")
    assert "pull inicial obligatorio" in ev.text and "UCB1: " not in ev.text
    assert f"S={ev.salience:.2f} − pen {ev.penalty:.2f} = r {ev.reward:.2f}" in ev.text
    assert f"Q {ev.q_before:.2f}→{ev.q_after:.2f}" in ev.text
    with pytest.raises(ValueError):
        LiveSession(data, "thompson", cfg, prev_fret=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_parser_and_table() -> None:
    """El parser de la CLI reconoce subcomandos y opciones, y format_summary_table alinea medias ± std y '—'."""
    import main as cli

    args = cli.build_parser().parse_args(["--cli", "analyze", "x.mp3", "--algo", "softmax", "--seed", "7", "--quiet"])
    assert (args.command, args.audio, args.algo, args.seed, args.quiet) == ("analyze", "x.mp3", "softmax", 7, True)
    args = cli.build_parser().parse_args(["--cli", "experiment", "x.mp3", "--runs", "5", "--no-sweeps"])
    assert args.runs == 5 and args.no_sweeps and not args.no_lambda
    assert cli.main(["--cli"]) == 2                       # sin subcomando: ayuda y código 2
    rows = [{"label": "UCB1", "mean_reward": 0.5, "mean_reward_std": 0.01, "final_regret": 3.0,
             "final_regret_std": 0.5, "optimal_pct": 80.0, "optimal_pct_std": 2.0, "pitch_acc": None,
             "pitch_acc_std": None, "position_acc": 0.75, "position_acc_std": 0.05, "ms_per_segment": 1.2,
             "ms_per_segment_std": 0.1}]
    table = cli.format_summary_table(rows)
    assert "UCB1" in table and "75.0 ± 5.0 %" in table and "—" in table
    assert "F1 posición" in table.splitlines()[0]
    # La ayuda de --runs y --budget muestra los valores de la configuración por defecto.
    help_text = cli.build_parser()._subparsers._group_actions[0].choices["experiment"].format_help()
    assert f"{Config().experiment.n_runs})" in help_text and f"{Config().env.budget})" in help_text


def test_gui_mode_separates_options_from_the_audio_path(monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture) -> None:
    """Regresión: sin --cli, «--quiet»/«--verbose» ajustan la consola y NO se pasan a la GUI como ruta del audio."""
    import logging

    import main as cli

    launched: list[list[str]] = []
    monkeypatch.setattr(cli, "launch_gui", lambda argv: launched.append(list(argv)) or 0)
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    try:
        assert cli.main(["--quiet", "C:/x/money.mp3"]) == 0
        assert launched[-1] == ["C:/x/money.mp3"]
        assert root.level == logging.INFO  # la pestaña Log sigue recibiendo INFO
        assert all(h.level == logging.WARNING for h in root.handlers)  # la consola, solo avisos
        assert cli.main(["--verbose"]) == 0 and launched[-1] == []
        assert root.level == logging.DEBUG
        assert cli.main(["pista.mp3"]) == 0 and launched[-1] == ["pista.mp3"]
        assert cli.main(["--runs", "5"]) == 2  # opción de la CLI sin --cli: se explica, no se abre la GUI
        assert "--cli" in capsys.readouterr().err and len(launched) == 3
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def test_cli_reports_config_and_output_errors_without_traceback(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Regresión: configuración inválida, --config carpeta y --out archivo → "Error: …" y código 1, sin traza.

    Los errores se detectan ANTES del análisis (el audio ni siquiera existe aquí).
    """
    import main as cli

    bad_cfg = tmp_path / "malo.json"
    bad_cfg.write_text(json.dumps({"env": {"lam": "0.1"}}), encoding="utf-8")
    assert cli.main(["--cli", "analyze", "x.mp3", "--config", str(bad_cfg), "--quiet"]) == 1
    assert "env.lam debe ser un número" in capsys.readouterr().err
    assert cli.main(["--cli", "analyze", "x.mp3", "--config", str(tmp_path), "--quiet"]) == 1
    assert "es una carpeta" in capsys.readouterr().err
    out_file = tmp_path / "ocupado.txt"
    out_file.write_text("ya existo", encoding="utf-8")
    assert cli.main(["--cli", "analyze", "x.mp3", "--out", str(out_file), "--quiet"]) == 1
    assert "--out apunta a un archivo existente" in capsys.readouterr().err
    assert cli.main(["--cli", "experiment", "x.mp3", "--runs", "0", "--quiet"]) == 1
    assert "experiment.n_runs" in capsys.readouterr().err


def test_cli_experiment_failure_leaves_no_output_folder(tmp_path: Path) -> None:
    """Regresión: si el análisis falla (audio inexistente), cmd_experiment no deja una carpeta vacía."""
    import main as cli

    out = tmp_path / "res_exp"
    assert cli.main(["--cli", "experiment", str(tmp_path / "no_existe.mp3"), "--out", str(out), "--quiet"]) == 1
    assert not out.exists()


@pytest.mark.slow
@needs_ffmpeg
def test_cli_analyze_end_to_end(tmp_path: Path) -> None:
    """``main.py --cli analyze`` imprime la tablatura y la precisión y exporta TXT/JSON/CSV (prueba lenta, requiere ffmpeg)."""
    audio = PROJECT_ROOT / "data" / "synthetic" / "linea_simple.mp3"
    if not audio.is_file():
        pytest.skip("falta data/synthetic/linea_simple.mp3 (python main.py --cli dataset)")
    out = tmp_path / "res"
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), "--cli", "analyze", str(audio), "--algo", "ucb1",
         "--seed", "3", "--out", str(out), "--quiet"],
        capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout
    for line in ("G|", "D|", "A|", "E|", "t(s)"):
        assert line in stdout
    assert "Precisión de UCB1: pitch" in stdout
    for ext in ("txt", "json", "csv"):
        assert (out / f"linea_simple_ucb1.{ext}").is_file()
    with (out / "linea_simple_ucb1.csv").open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) >= 14     # 16 notas en el ground truth
