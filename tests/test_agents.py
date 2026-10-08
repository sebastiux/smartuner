"""Pruebas de la etapa 7 (agentes bandit: ε-greedy, optimista, UCB1 y Softmax).

Cada prueba verifica una propiedad teórica de Sutton & Barto (cap. 2) o un
detalle de la interfaz que usan :mod:`src.experiments` y la GUI. Todas son
deterministas (semillas fijas) y rápidas.
"""

from __future__ import annotations

import doctest
import math
from collections.abc import Callable
import re
import time

import numpy as np
import pytest

from src import agents as agents_module
from src.agents import (
    BanditAgent,
    Decision,
    EpsilonGreedyAgent,
    OptimisticGreedyAgent,
    SoftmaxAgent,
    UCB1Agent,
    make_agent,
)
from src.config import ALGORITHMS, AgentConfig


def _rng(seed: int) -> np.random.Generator:
    """Generador de números aleatorios con semilla fija (pruebas deterministas)."""
    return np.random.default_rng(seed)


def _chi_square(counts: np.ndarray, probs: np.ndarray) -> float:
    """Estadístico χ² = Σ (observado − esperado)² / esperado."""
    expected = probs * counts.sum()
    return float(np.sum((counts - expected) ** 2 / expected))


#: Valor crítico de χ² con 4 grados de libertad para p = 0.001 (≈ 18.47). Las
#: pruebas usan un umbral más laxo (25) para no ser frágiles.
CHI2_LAX_DF4 = 25.0


# ---------------------------------------------------------------------------
# Regla incremental Q ← Q + α(r − Q)
# ---------------------------------------------------------------------------


def test_incremental_mean_equals_sample_mean() -> None:
    """Con α = 1/n la regla incremental reproduce la media muestral exacta."""
    rng = _rng(0)
    agent = EpsilonGreedyAgent(n_arms=4, epsilon=0.0, rng=_rng(1))
    arms = rng.integers(0, 4, size=500)
    rewards = rng.normal(0.5, 0.3, size=500)
    for a, r in zip(arms, rewards):
        agent.update(int(a), float(r))
    for a in range(4):
        mask = arms == a
        assert agent.counts[a] == mask.sum()
        np.testing.assert_allclose(agent.q[a], rewards[mask].mean(), rtol=0, atol=1e-12)
    assert agent.t == 500
    assert agent.counts.dtype.kind == "i"


def test_constant_alpha_is_exponential_recency_weighted_average() -> None:
    """Con α constante: Q_n = (1−α)ⁿ·Q₀ + Σ_i α(1−α)^{n−i}·r_i (fórmula cerrada)."""
    alpha, q0 = 0.1, 2.0
    rewards = _rng(3).uniform(0.0, 1.0, size=60)
    agent = OptimisticGreedyAgent(n_arms=2, q0=q0, alpha=alpha, rng=_rng(0))
    for r in rewards:
        agent.update(0, float(r))
    n = len(rewards)
    weights = alpha * (1 - alpha) ** (n - np.arange(1, n + 1))
    closed_form = (1 - alpha) ** n * q0 + np.sum(weights * rewards)
    np.testing.assert_allclose(agent.q[0], closed_form, rtol=1e-12)
    # Los pesos (incluido el de Q₀) suman 1: es un promedio ponderado.
    np.testing.assert_allclose((1 - alpha) ** n + weights.sum(), 1.0)
    # El brazo no jalado conserva su Q₀ y su conteo 0.
    assert agent.q[1] == q0 and agent.counts[1] == 0


def test_sample_average_forgets_q0_after_one_pull() -> None:
    """Con α = 1/n el primer paso es 1: Q ← r y el optimismo de Q₀ desaparece."""
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, q0=5.0, alpha=None, rng=_rng(0))
    agent.update(1, 0.3)
    assert agent.q[1] == pytest.approx(0.3)


def test_update_validates_arm_and_reward() -> None:
    """update() rechaza brazos fuera de 0..K−1 (también −1) y recompensas NaN sin tocar el estado; acepta tipos numpy."""
    agent = EpsilonGreedyAgent(n_arms=3, rng=_rng(0))
    with pytest.raises(IndexError):
        agent.update(3, 0.5)
    with pytest.raises(IndexError):
        agent.update(-1, 0.5)  # numpy lo aceptaría en silencio como el último brazo
    with pytest.raises(ValueError):
        agent.update(0, float("nan"))
    assert agent.t == 0 and agent.counts.sum() == 0
    agent.update(np.int64(2), np.float32(0.5))  # tipos numpy también valen
    assert agent.counts[2] == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EpsilonGreedyAgent(0),
        lambda: EpsilonGreedyAgent(3, epsilon=1.5),
        lambda: EpsilonGreedyAgent(3, decay=0.0),
        lambda: EpsilonGreedyAgent(3, decay=1.1),
        lambda: EpsilonGreedyAgent(3, alpha=0.0),
        lambda: OptimisticGreedyAgent(3, alpha=2.0),
        lambda: UCB1Agent(3, c=-1.0),
        lambda: SoftmaxAgent(3, tau=0.0),
        lambda: SoftmaxAgent(3, tau_min=-0.1),
    ],
)
def test_invalid_hyperparameters_raise(factory: Callable[[], BanditAgent]) -> None:
    """Hiperparámetros sin sentido (K < 1, ε ∉ [0, 1], α ∉ (0, 1], c < 0, τ ≤ 0...) lanzan ValueError al construir el agente."""
    with pytest.raises(ValueError):
        factory()


# ---------------------------------------------------------------------------
# ε-greedy
# ---------------------------------------------------------------------------


def test_epsilon_zero_always_exploits() -> None:
    """Con ε = 0 el agente es puramente greedy: siempre elige argmax Q y lo marca como 'exploit'."""
    agent = EpsilonGreedyAgent(n_arms=5, epsilon=0.0, rng=_rng(0))
    for a, r in enumerate([0.1, 0.7, 0.3, 0.2, 0.5]):
        agent.update(a, r)
    for _ in range(200):
        arm = agent.select_arm()
        assert arm == 1
        assert agent.last_decision is not None
        assert agent.last_decision.kind == "exploit"
        agent.update(arm, 0.7)


def test_epsilon_one_is_uniform() -> None:
    """Con ε = 1 elige al azar uniforme (prueba χ²) aunque un brazo tenga Q claramente mayor: exploración 'ciega' al valor."""
    k, n = 5, 5000
    agent = EpsilonGreedyAgent(n_arms=k, epsilon=1.0, rng=_rng(7))
    agent.update(0, 1.0)  # aunque haya un brazo claramente mejor
    counts = np.zeros(k)
    for _ in range(n):
        arm = agent.select_arm()
        assert agent.last_decision is not None and agent.last_decision.kind == "explore"
        counts[arm] += 1
    assert _chi_square(counts, np.full(k, 1 / k)) < CHI2_LAX_DF4


def test_epsilon_greedy_exploration_rate() -> None:
    """La fracción de pulls exploratorios es ≈ ε."""
    agent = EpsilonGreedyAgent(n_arms=4, epsilon=0.2, rng=_rng(11))
    kinds = []
    for _ in range(4000):
        agent.select_arm()
        assert agent.last_decision is not None
        kinds.append(agent.last_decision.kind)
    assert abs(np.mean(np.array(kinds) == "explore") - 0.2) < 0.03


def test_argmax_ties_are_uniform() -> None:
    """Los empates en argmax Q se rompen al azar de forma uniforme, nunca por el índice más bajo (no sesga la exploración)."""
    agent = EpsilonGreedyAgent(n_arms=5, rng=_rng(5))
    values = np.array([0.3, 0.9, 0.1, 0.9, 0.9])
    counts = np.zeros(5)
    for _ in range(6000):
        counts[agent._argmax_random_tie(values)] += 1
    assert counts[0] == counts[2] == 0
    np.testing.assert_allclose(counts[[1, 3, 4]] / 6000, 1 / 3, atol=0.03)


def test_argmax_ties_initial_q_are_uniform() -> None:
    """Al inicio (todas las Q = Q₀) ε-greedy no se sesga hacia los índices bajos."""
    k = 5
    counts = np.zeros(k)
    for seed in range(2000):
        agent = EpsilonGreedyAgent(n_arms=k, epsilon=0.0, rng=_rng(seed))
        counts[agent.select_arm()] += 1
    assert _chi_square(counts, np.full(k, 1 / k)) < CHI2_LAX_DF4
    assert agent.last_decision is not None
    assert "empate" in agent.last_decision.text


def test_argmax_handles_inf_and_nan() -> None:
    """El argmax con desempate aleatorio maneja +∞ (brazos sin probar), −∞ y NaN sin elegir nunca un NaN si hay valores válidos."""
    agent = EpsilonGreedyAgent(n_arms=2, rng=_rng(0))
    inf = np.inf
    assert agent._argmax_random_tie(np.array([0.5, inf, 0.2])) == 1
    seen = {agent._argmax_random_tie(np.array([inf, 0.1, inf])) for _ in range(100)}
    assert seen == {0, 2}
    assert agent._argmax_random_tie(np.array([np.nan, 0.4, 0.9, np.nan])) == 2
    assert agent._argmax_random_tie(np.array([-inf, -inf, -1.0])) == 2
    seen = {agent._argmax_random_tie(np.array([-inf, -inf])) for _ in range(100)}
    assert seen == {0, 1}
    assert agent._argmax_random_tie(np.array([np.nan, np.nan])) in (0, 1)


def test_epsilon_decay_schedule() -> None:
    """ε decae como ε_t = max(ε_min, ε₀·dᵗ) pull a pull y se queda en ε_min."""
    eps0, d, eps_min = 0.5, 0.9, 0.05
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=eps0, decay=d, epsilon_min=eps_min, rng=_rng(0))
    for t in range(60):
        assert agent.t == t
        assert agent.epsilon == pytest.approx(max(eps_min, eps0 * d**t))
        agent.update(agent.select_arm(), 0.5)
    assert agent.epsilon == eps_min  # 0.5·0.9⁶⁰ ≈ 9e-4 < ε_min


def test_no_decay_uses_epsilon0_exactly() -> None:
    """Con d = 1 se usa ε₀ tal cual, aunque sea menor que ε_min."""
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.01, decay=1.0, epsilon_min=0.2, rng=_rng(0))
    for _ in range(10):
        agent.update(agent.select_arm(), 0.5)
    assert agent.epsilon == 0.01


# ---------------------------------------------------------------------------
# ε-greedy optimista
# ---------------------------------------------------------------------------


def test_optimistic_explores_all_arms_first_in_bernoulli_bandit() -> None:
    """Q₀ = 2 > recompensa máxima (1): incluso con ε = 0 los K primeros pulls
    recorren los K brazos, y en los primeros pulls cada brazo se prueba varias
    veces gracias a α constante."""
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    k = len(p)
    for seed in range(10):
        env = _rng(100 + seed)
        agent = OptimisticGreedyAgent(n_arms=k, q0=2.0, epsilon=0.0, alpha=0.1, rng=_rng(seed))
        arms = []
        for _ in range(50):
            a = agent.select_arm()
            arms.append(a)
            agent.update(a, float(env.random() < p[a]))
        assert sorted(arms[:k]) == list(range(k)), "los K primeros pulls deben cubrir todos los brazos"
        assert np.bincount(arms, minlength=k).min() >= 3


def test_optimistic_is_epsilon_greedy_with_constant_alpha() -> None:
    """El agente optimista es un ε-greedy con Q₀ = 2, α = 0.1 constante y ε = 0, y explica su elección por el optimismo."""
    agent = OptimisticGreedyAgent(n_arms=4, rng=_rng(0))
    assert isinstance(agent, EpsilonGreedyAgent)
    assert agent.name == "optimistic"
    assert agent.q0 == 2.0 and agent.alpha == 0.1 and agent.epsilon == 0.0
    np.testing.assert_array_equal(agent.q, np.full(4, 2.0))
    agent.select_arm()
    assert agent.last_decision is not None
    assert "aún sin probar" in agent.last_decision.text


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------


def _hand_ucb(q: list[float], n: list[int], t: int, c: float) -> list[float]:
    """Índice UCB1 calculado "a mano", brazo por brazo."""
    return [q[a] + c * math.sqrt(math.log(t) / n[a]) for a in range(len(q))]


def test_ucb1_pulls_each_arm_once_then_follows_index() -> None:
    """UCB1 jala cada brazo una vez (bono +∞ con n = 0) y luego sigue argmax Q + c·√(ln t / n), comprobado con sumas y conteos a mano."""
    k, c = 4, 1.414
    mu = [0.2, 0.6, 0.4, 0.5]
    env = _rng(1)
    agent = UCB1Agent(n_arms=k, c=c, rng=_rng(2))
    first = []
    for _ in range(k):
        a = agent.select_arm()
        assert agent.last_decision is not None and agent.last_decision.kind == "init"
        first.append(a)
        agent.update(a, float(env.normal(mu[a], 0.1)))
    assert sorted(first) == list(range(k))

    # A partir de aquí el brazo elegido es el argmax del índice calculado a mano
    # (sumas y conteos propios; tras un pull por brazo, Q = suma = recompensa).
    sums, n = agent.q.tolist(), [1] * k
    for _ in range(200):
        hand = _hand_ucb([sums[a] / n[a] for a in range(k)], n, sum(n), c)
        a = agent.select_arm()
        assert a == int(np.argmax(hand))
        assert agent.last_decision is not None
        np.testing.assert_allclose(agent.last_decision.scores, hand, rtol=1e-12)
        np.testing.assert_allclose(agent.ucb_values(), hand, rtol=1e-12)
        assert agent.last_decision.kind in ("explore", "exploit")
        r = float(env.normal(mu[a], 0.1))
        agent.update(a, r)
        sums[a] += r
        n[a] += 1


def test_ucb_values_infinite_for_unpulled_arms() -> None:
    """El índice UCB es +∞ para brazos sin probar y Q + 0 con t = 1 (ln 1 = 0); decision_scores() lo expone con su nombre."""
    agent = UCB1Agent(n_arms=3, c=1.0, rng=_rng(0))
    assert np.all(np.isinf(agent.ucb_values()))
    agent.update(1, 0.5)
    values = agent.ucb_values()
    assert np.isinf(values[0]) and np.isinf(values[2])
    assert values[1] == pytest.approx(0.5)  # t = 1 → ln 1 = 0 → bono 0
    scores, name = agent.decision_scores()
    np.testing.assert_array_equal(scores, values)
    assert name == "Q + c·√(ln t / n)"


def test_ucb1_initial_order_is_random() -> None:
    """El orden de los pulls iniciales de UCB1 es aleatorio (depende de la semilla), no siempre el brazo 0."""
    firsts = {UCB1Agent(n_arms=4, rng=_rng(s)).select_arm() for s in range(40)}
    assert firsts == {0, 1, 2, 3}


def test_ucb1_with_c_zero_is_greedy() -> None:
    """Con c = 0 el bono desaparece y UCB1 se reduce a greedy puro (argmax Q)."""
    agent = UCB1Agent(n_arms=3, c=0.0, rng=_rng(0))
    for a, r in enumerate([0.2, 0.9, 0.4]):
        agent.update(a, r)
    assert agent.select_arm() == 1
    assert agent.last_decision is not None and agent.last_decision.kind == "exploit"


# ---------------------------------------------------------------------------
# Softmax (Boltzmann)
# ---------------------------------------------------------------------------


def _softmax_with_q(q: list[float], tau: float, seed: int = 0, **kwargs: float) -> SoftmaxAgent:
    """Agente Softmax cuyas Q valen exactamente ``q`` (un pull por brazo con α = 1/n)."""
    agent = SoftmaxAgent(n_arms=len(q), tau=tau, rng=_rng(seed), **kwargs)  # type: ignore[arg-type]
    for a, value in enumerate(q):
        agent.update(a, value)  # α = 1/n → Q = valor exacto tras un pull
    return agent


def test_softmax_probabilities_sum_to_one_and_follow_boltzmann() -> None:
    """π(a) = e^(Q_a/τ)/Σ_b e^(Q_b/τ): suma 1, coincide con la fórmula y conserva el orden de las Q."""
    q = [0.1, 0.5, 0.3, 0.9]
    agent = _softmax_with_q(q, tau=0.2)
    p = agent.probabilities()
    assert p.shape == (4,)
    assert p.sum() == pytest.approx(1.0)
    expected = np.exp(np.array(q) / 0.2) / np.exp(np.array(q) / 0.2).sum()
    np.testing.assert_allclose(p, expected, rtol=1e-12)
    # Brazos con más Q tienen más probabilidad (exploración graduada por valor).
    assert list(np.argsort(p)) == list(np.argsort(q))


def test_softmax_low_temperature_is_greedy_and_high_is_uniform() -> None:
    """τ → 0 concentra la probabilidad en argmax Q (greedy) y τ → ∞ la reparte uniforme."""
    q = [0.2, 0.5, 0.45, 0.1]
    cold = _softmax_with_q(q, tau=1e-4).probabilities()
    assert cold[1] > 0.999
    hot = _softmax_with_q(q, tau=1e4).probabilities()
    np.testing.assert_allclose(hot, 0.25, atol=1e-4)


def test_softmax_no_overflow_with_huge_q() -> None:
    """Restar max(Q) antes de exponenciar evita el overflow con Q = 10⁴ y τ = 0.1 (π finita y normalizada)."""
    agent = _softmax_with_q([1e4, 0.0, 9999.9], tau=0.1)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        p = agent.probabilities()
        arm = agent.select_arm()
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)
    assert p[1] == 0.0 and p[0] > p[2] > 0
    assert arm in (0, 2)


def test_softmax_sampling_matches_probabilities() -> None:
    """Las frecuencias de los brazos muestreados siguen π (prueba χ²): el muestreo por transformada inversa es correcto."""
    q = [0.0, 0.1, 0.2, 0.3, 0.4]
    agent = _softmax_with_q(q, tau=0.15, seed=3)
    p = agent.probabilities()
    counts = np.zeros(5)
    for _ in range(6000):
        counts[agent.select_arm()] += 1  # sin update: π fija
    assert _chi_square(counts, p) < CHI2_LAX_DF4
    assert agent.last_decision is not None and agent.last_decision.kind == "sample"


def test_softmax_never_samples_zero_probability_arm() -> None:
    """Un brazo con π = 0 por underflow (e^-1000) nunca se muestrea."""
    agent = _softmax_with_q([1.0, 0.0], tau=1e-3)  # π = [1, e^-1000 = 0]
    assert agent.probabilities()[1] == 0.0
    assert all(agent.select_arm() == 0 for _ in range(500))


def test_tau_annealing_schedule() -> None:
    """τ decae como τ_t = max(τ_min, τ₀·dᵗ); sin decaimiento (d = 1) se usa τ₀ exacta."""
    tau0, d, tau_min = 1.0, 0.8, 0.05
    agent = SoftmaxAgent(n_arms=3, tau=tau0, decay=d, tau_min=tau_min, rng=_rng(0))
    for t in range(40):
        assert agent.tau == pytest.approx(max(tau_min, tau0 * d**t))
        agent.update(agent.select_arm(), 0.5)
    assert agent.tau == tau_min
    constant = SoftmaxAgent(n_arms=3, tau=0.005, decay=1.0, tau_min=0.01, rng=_rng(0))
    constant.update(0, 1.0)
    assert constant.tau == 0.005  # d = 1 → τ₀ exacto, sin aplicar τ_min


def test_softmax_zero_tau_after_underflow_falls_back_to_greedy() -> None:
    """Si el annealing lleva τ a 0 en float64, Softmax se comporta como greedy en vez de dividir entre 0."""
    agent = SoftmaxAgent(n_arms=3, tau=0.1, decay=0.5, tau_min=0.0, rng=_rng(0))
    agent.update(2, 1.0)
    agent.t = 5000  # 0.1·0.5⁵⁰⁰⁰ = 0 en float64
    assert agent.tau == 0.0
    np.testing.assert_array_equal(agent.probabilities(), [0.0, 0.0, 1.0])
    assert agent.select_arm() == 2


# ---------------------------------------------------------------------------
# Decisiones explicadas (lo que muestra la GUI)
# ---------------------------------------------------------------------------


def test_decision_texts_epsilon_greedy() -> None:
    """ε-greedy explica cada decisión en español: 'u < ε → explora' o 'u ≥ ε → explota' con la Q del brazo."""
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.5, rng=_rng(4))
    agent.update(1, 0.72)
    texts: dict[str, str] = {}
    for _ in range(50):
        agent.select_arm()
        d = agent.last_decision
        assert isinstance(d, Decision)
        texts[d.kind] = d.text
    assert re.fullmatch(r"u=\S+ < ε=0\.50(00)? → explora: brazo [0-2] al azar", texts["explore"])
    assert re.fullmatch(r"u=\S+ ≥ ε=0\.50(00)? → explota: brazo 1 tiene el mayor Q \(0\.72\)", texts["exploit"])


def test_decision_text_comparison_never_looks_wrong() -> None:
    """Con u = 0.0996 y ε = 0.1 el texto no debe decir "u=0.10 < ε=0.10"."""
    assert agents_module._fmt_pair(0.0996, 0.1) == ("0.0996", "0.1000")
    assert agents_module._fmt_pair(0.03, 0.1) == ("0.03", "0.10")


def test_decision_texts_ucb1_and_softmax() -> None:
    """UCB1 explica índice = Q + bono con los números del pull y Softmax la probabilidad π del brazo muestreado."""
    ucb = UCB1Agent(n_arms=2, c=1.0, rng=_rng(0))
    ucb.select_arm()
    assert ucb.last_decision is not None
    assert re.fullmatch(r"UCB1: pull inicial obligatorio del brazo [01] .*", ucb.last_decision.text)
    ucb.update(0, 0.7)
    ucb.update(1, 0.2)
    ucb.select_arm()
    d = ucb.last_decision
    assert d is not None
    # t = 2, n = 1: bono = √(ln 2) = 0.83 → índice 1.53 = Q 0.70 + bono 0.83
    assert d.text.startswith("UCB1: brazo 0 con índice 1.53 = Q 0.70 + bono 0.83")
    assert d.kind == "exploit"

    soft = _softmax_with_q([0.5, 0.0], tau=0.1, seed=0)
    soft.select_arm()
    d = soft.last_decision
    assert d is not None
    assert re.fullmatch(r"Softmax \(τ=0\.10\): muestrea brazo [01] con π=\d\.\d\d .*", d.text)


def test_ucb1_labels_bonus_driven_choice_as_explore() -> None:
    """Si UCB1 elige un brazo que NO es argmax Q (gana por el bono), la decisión se marca como 'explore'."""
    agent = UCB1Agent(n_arms=2, c=2.0, rng=_rng(0))
    for _ in range(50):
        agent.update(0, 0.6)  # brazo 0 muy probado
    agent.update(1, 0.5)      # brazo 1 con un solo pull: bono enorme
    assert agent.select_arm() == 1
    assert agent.last_decision is not None and agent.last_decision.kind == "explore"
    assert "bono" in agent.last_decision.text


@pytest.mark.parametrize(
    ("name", "score_name"),
    [
        ("egreedy", "Q estimado"),
        ("optimistic", "Q estimado"),
        ("ucb1", "Q + c·√(ln t / n)"),
        ("softmax", "π(a) = e^(Q/τ) / Σ e^(Q/τ)"),
    ],
)
def test_decision_scores_and_snapshot(name: str, score_name: str) -> None:
    """Cada agente expone sus puntuaciones (Q, índice UCB o π) y la decisión guarda una COPIA que no cambia al actualizar."""
    agent = make_agent(name, 4, AgentConfig(), rng=_rng(0))
    for _ in range(10):
        agent.update(agent.select_arm(), 0.5)
    scores, label = agent.decision_scores()
    assert label == score_name
    assert scores.shape == (4,)
    agent.select_arm()
    d = agent.last_decision
    assert d is not None and d.score_name == score_name
    assert d.scores is not None and d.scores.shape == (4,)
    snapshot = d.scores.copy()
    agent.update(d.arm, 0.0)
    # Las actualizaciones posteriores no modifican lo que se mostró al decidir.
    np.testing.assert_array_equal(d.scores, snapshot)
    if name == "softmax":
        assert scores.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# reset, recommend, make_agent, reproducibilidad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALGORITHMS)
def test_reset_restores_initial_state(name: str) -> None:
    """reset() devuelve Q, conteos, t, ε/τ y la última decisión al estado inicial (Q₀)."""
    cfg = AgentConfig(epsilon_decay=0.95, tau_decay=0.95)
    agent = make_agent(name, 5, cfg, rng=_rng(0))
    fresh = make_agent(name, 5, cfg, rng=_rng(0))
    for _ in range(30):
        agent.update(agent.select_arm(), 0.3)
    agent.reset()
    np.testing.assert_array_equal(agent.q, fresh.q)
    np.testing.assert_array_equal(agent.counts, np.zeros(5))
    assert agent.t == 0
    assert agent.last_decision is None
    np.testing.assert_array_equal(agent.decision_scores()[0], fresh.decision_scores()[0])
    if isinstance(agent, EpsilonGreedyAgent):
        assert agent.epsilon == agent.epsilon0
    if isinstance(agent, SoftmaxAgent):
        assert agent.tau == agent.tau0


def test_recommend_modes() -> None:
    """recommend('most_pulled') elige el brazo más jalado (desempate por Q) y 'greedy' el argmax Q; un modo desconocido es un error."""
    agent = EpsilonGreedyAgent(n_arms=4, epsilon=0.0, rng=_rng(0))
    for arm, r in [(0, 0.5), (0, 0.5), (0, 0.5), (2, 0.9), (3, 0.6), (3, 0.6), (3, 0.6)]:
        agent.update(arm, r)
    # most_pulled: empate de conteos (0 y 3, 3 pulls) → gana el de mayor Q (3).
    assert agent.recommend("most_pulled") == 3
    assert agent.recommend() == 3
    # greedy: argmax Q (brazo 2, aunque tenga un solo pull).
    assert agent.recommend("greedy") == 2
    agent.update(1, 0.9)  # empate en Q (brazos 1 y 2) y en conteos → índice más bajo
    assert agent.recommend("greedy") == 1
    with pytest.raises(ValueError, match="most_pulled"):
        agent.recommend("mayoria")


def test_recommend_does_not_consume_rng() -> None:
    """recommend() no consume el generador: consultar la recomendación no cambia la trayectoria del agente."""
    a = EpsilonGreedyAgent(n_arms=3, epsilon=0.5, rng=_rng(9))
    b = EpsilonGreedyAgent(n_arms=3, epsilon=0.5, rng=_rng(9))
    seq_a, seq_b = [], []
    for _ in range(100):
        a.recommend("most_pulled")
        a.recommend("greedy")
        seq_a.append(a.select_arm())
        seq_b.append(b.select_arm())
        a.update(seq_a[-1], 0.5)
        b.update(seq_b[-1], 0.5)
    assert seq_a == seq_b


def test_make_agent_builds_each_algorithm_from_config() -> None:
    """make_agent construye cada algoritmo con los hiperparámetros de AgentConfig (ε, Q₀, α, c, τ y sus decaimientos)."""
    cfg = AgentConfig(
        epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.02,
        q0=3.0, optimistic_epsilon=0.05, optimistic_alpha=0.2,
        ucb_c=0.7, tau=0.3, tau_decay=0.98, tau_min=0.03,
    )
    eg = make_agent("egreedy", 6, cfg, rng=_rng(0))
    assert type(eg) is EpsilonGreedyAgent and eg.name == "egreedy"
    assert (eg.epsilon0, eg.decay, eg.epsilon_min, eg.q0, eg.alpha) == (0.2, 0.99, 0.02, 0.0, None)
    opt = make_agent("optimistic", 6, cfg, rng=_rng(0))
    assert type(opt) is OptimisticGreedyAgent and opt.name == "optimistic"
    assert (opt.q0, opt.epsilon, opt.alpha) == (3.0, 0.05, 0.2)
    np.testing.assert_array_equal(opt.q, np.full(6, 3.0))
    ucb = make_agent("ucb1", 6, cfg, rng=_rng(0))
    assert type(ucb) is UCB1Agent and ucb.c == 0.7 and ucb.name == "ucb1"
    soft = make_agent("softmax", 6, cfg, rng=_rng(0))
    assert type(soft) is SoftmaxAgent and soft.name == "softmax"
    assert (soft.tau0, soft.decay, soft.tau_min) == (0.3, 0.98, 0.03)
    for agent in (eg, opt, ucb, soft):
        assert agent.n_arms == 6
    # Sin rng se crea uno propio.
    assert isinstance(make_agent("ucb1", 2, cfg).rng, np.random.Generator)
    # Los nombres de los agentes coinciden con las claves de config.ALGORITHMS.
    assert tuple(make_agent(n, 2, cfg).name for n in ALGORITHMS) == ALGORITHMS


def test_make_agent_unknown_name_raises() -> None:
    """Un nombre de algoritmo desconocido lanza ValueError con mensaje en español."""
    with pytest.raises(ValueError, match="Algoritmo desconocido"):
        make_agent("thompson", 3, AgentConfig())


@pytest.mark.parametrize("name", ALGORITHMS)
def test_same_seed_reproduces_trajectory(name: str) -> None:
    """Misma semilla → misma trayectoria (brazos, textos y Q finales); otra semilla → otra trayectoria."""
    def trajectory(seed: int) -> tuple[list[int], list[str], np.ndarray]:
        """Brazos, textos explicativos y Q finales de 300 pulls con la semilla ``seed``."""
        agent = make_agent(name, 6, AgentConfig(), rng=_rng(seed))
        env = _rng(1000)
        arms, texts = [], []
        for _ in range(300):
            a = agent.select_arm()
            arms.append(a)
            assert agent.last_decision is not None
            texts.append(agent.last_decision.text)
            agent.update(a, float(env.normal(0.1 * a, 0.3)))
        return arms, texts, agent.q.copy()

    arms1, texts1, q1 = trajectory(42)
    arms2, texts2, q2 = trajectory(42)
    assert arms1 == arms2 and texts1 == texts2
    np.testing.assert_array_equal(q1, q2)
    arms3, _, _ = trajectory(43)
    assert arms3 != arms1


# ---------------------------------------------------------------------------
# Convergencia y rendimiento
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALGORITHMS)
def test_converges_to_optimal_arm_in_gaussian_bandit(name: str) -> None:
    """Bandit gaussiano de 5 brazos con brechas 0.2 (medias centradas en 0, como
    el testbed de Sutton & Barto; σ = 0.3): con los hiperparámetros por defecto,
    en los últimos 200 de 1000 pulls se elige el óptimo > 80 % de las veces
    (promedio sobre 20 semillas)."""
    mu = np.array([0.0, 0.4, -0.2, 0.2, -0.4])  # óptimo: brazo 1 (no el primero ni el último)
    optimal = int(np.argmax(mu))
    fractions = []
    for seed in range(20):
        agent = make_agent(name, len(mu), AgentConfig(), rng=np.random.default_rng([seed, 1]))
        noise = np.random.default_rng([seed, 0]).normal(0.0, 0.3, size=1000)
        arms = np.empty(1000, dtype=int)
        for t in range(1000):
            a = agent.select_arm()
            arms[t] = a
            agent.update(a, mu[a] + noise[t])
        fractions.append(np.mean(arms[-200:] == optimal))
    assert np.mean(fractions) > 0.8


@pytest.mark.parametrize("name", ALGORITHMS)
def test_pull_cost_is_small(name: str) -> None:
    """Guarda laxa de rendimiento: 10 000 pulls (K = 10) en < 1 s (< 100 µs/pull;
    lo medido es ≈ 5–11 µs/pull)."""
    agent = make_agent(name, 10, AgentConfig(), rng=_rng(0))
    rewards = _rng(1).random(10_000)
    start = time.perf_counter()
    for r in rewards:
        agent.update(agent.select_arm(), float(r))
    assert time.perf_counter() - start < 1.0


def test_module_doctests() -> None:
    """Los ejemplos de los docstrings de src.agents se ejecutan sin fallos."""
    result = doctest.testmod(agents_module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0


@pytest.mark.parametrize("name", ["egreedy", "optimistic", "ucb1", "softmax"])
def test_explain_false_makes_identical_choices(name: str) -> None:
    """``explain=False`` (modo rápido de los experimentos) elige exactamente igual
    que con explicación: el texto no consume el generador."""
    rewards_rng = np.random.default_rng(123)
    means = np.array([0.2, 0.5, 0.45, 0.1, 0.3])
    rewards = means[None, :] + 0.1 * rewards_rng.standard_normal((400, means.size))
    traces = []
    for explain in (True, False):
        agent = make_agent(name, means.size, AgentConfig(), rng=np.random.default_rng(7), explain=explain)
        arms = []
        for t in range(400):
            a = agent.select_arm()
            assert (agent.last_decision is not None) == explain
            agent.update(a, float(rewards[t, a]))
            arms.append(a)
        traces.append((arms, agent.q.copy(), agent.recommend()))
    assert traces[0][0] == traces[1][0]
    np.testing.assert_array_equal(traces[0][1], traces[1][1])
    assert traces[0][2] == traces[1][2]
