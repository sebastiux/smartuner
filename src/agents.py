"""Etapa 7 — Agentes bandit implementados desde cero con numpy. [CONTRATO: implementar]

Interfaz común (:class:`BanditAgent`): ``select_arm()``, ``update(arm, reward)``,
``reset()``. Además cada agente expone ``last_decision`` (:class:`Decision`)
con la explicación de su última elección, que la GUI muestra paso a paso, y
``decision_scores()`` con el criterio que usa para decidir (índice UCB,
probabilidades softmax, etc.).

Regla de actualización incremental (todos los agentes)::

    Q_{n+1} = Q_n + α_n · (r_n − Q_n)

con α_n = 1/n (media muestral exacta, sin guardar el historial) o α
constante (promedio exponencial; olvida gradualmente el pasado y el valor
inicial). Los empates en argmax se rompen al azar con el ``rng`` del agente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config import AgentConfig


@dataclass
class Decision:
    """Explicación de una elección del agente.

    Attributes
    ----------
    arm : int
        Brazo elegido.
    kind : str
        ``"init"`` (pull inicial obligatorio, UCB1), ``"explore"``,
        ``"exploit"`` o ``"sample"`` (muestreo de la distribución softmax).
    scores : np.ndarray | None
        Criterio usado para decidir (forma ``(K,)``): Q para ε-greedy, índice
        UCB para UCB1, probabilidades π para Softmax.
    score_name : str
        Nombre legible del criterio (p. ej. ``"Q + c·√(ln t / n)"``).
    text : str
        Frase en español para el log/GUI, p. ej.
        ``"u=0.03 < ε=0.10 → explora el brazo 2 al azar"``.
    """

    arm: int
    kind: str
    scores: np.ndarray | None
    score_name: str
    text: str


class BanditAgent(ABC):
    """Clase base de los agentes.

    Parameters
    ----------
    n_arms : int
        Número de brazos K.
    rng : np.random.Generator | None
        Generador propio del agente (exploración y desempates).
    q0 : float
        Valor inicial de las estimaciones Q.
    alpha : float | None
        Tamaño de paso constante; ``None`` = 1/n (media muestral).

    Attributes
    ----------
    q : np.ndarray
        Estimaciones Q_a, forma ``(K,)``.
    counts : np.ndarray
        Número de pulls de cada brazo n_a (int), forma ``(K,)``.
    t : int
        Pulls totales realizados.
    last_decision : Decision | None
        Explicación de la última llamada a :meth:`select_arm`.
    """

    #: Identificador interno (clave de :data:`src.config.ALGORITHMS`).
    name: str = "base"

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None, q0: float = 0.0, alpha: float | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_arm(self) -> int:
        """Elige el siguiente brazo a jalar (y actualiza ``last_decision``)."""

    def update(self, arm: int, reward: float) -> None:
        """Actualiza n_a y Q_a con la regla incremental del encabezado; incrementa t."""
        raise NotImplementedError

    def reset(self) -> None:
        """Vuelve al estado inicial (Q = q0, conteos a 0, t = 0, hiperparámetros con decaimiento a su valor inicial)."""
        raise NotImplementedError

    def recommend(self, mode: str = "most_pulled") -> int:
        """Brazo recomendado al final del presupuesto.

        ``"most_pulled"``: argmax de ``counts`` (desempate por mayor Q).
        ``"greedy"``: argmax de ``q`` (desempate por más pulls).
        """
        raise NotImplementedError

    def decision_scores(self) -> tuple[np.ndarray, str]:
        """Criterio de decisión ACTUAL (forma ``(K,)``) y su nombre, para la gráfica en vivo."""
        raise NotImplementedError

    def _argmax_random_tie(self, values: np.ndarray) -> int:
        """argmax rompiendo empates uniformemente al azar."""
        raise NotImplementedError


class EpsilonGreedyAgent(BanditAgent):
    """ε-greedy: con probabilidad ε explora un brazo uniforme, si no explota argmax Q.

    ε puede decaer: ε_t = max(ε_min, ε₀·dᵗ).
    """

    name = "egreedy"

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        decay: float = 1.0,
        epsilon_min: float = 0.0,
        rng: np.random.Generator | None = None,
        q0: float = 0.0,
        alpha: float | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def epsilon(self) -> float:
        """ε vigente en el pull actual."""
        raise NotImplementedError

    def select_arm(self) -> int:
        raise NotImplementedError


class OptimisticGreedyAgent(EpsilonGreedyAgent):
    """ε-greedy optimista: Q₀ alto (> recompensa máxima) y α constante.

    Con Q₀ = 2 y recompensas ≤ 1, cada brazo probado baja su Q por debajo de
    los no probados, así que un agente greedy (ε=0) recorre todos los brazos
    antes de estabilizarse: exploración dirigida "gratis" al inicio.
    """

    name = "optimistic"

    def __init__(
        self,
        n_arms: int,
        q0: float = 2.0,
        epsilon: float = 0.0,
        alpha: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        raise NotImplementedError


class UCB1Agent(BanditAgent):
    """UCB1: primero jala cada brazo una vez; luego argmax Q_a + c·√(ln t / n_a)."""

    name = "ucb1"

    def __init__(self, n_arms: int, c: float = 1.414, rng: np.random.Generator | None = None) -> None:
        raise NotImplementedError

    def ucb_values(self) -> np.ndarray:
        """Índice UCB de cada brazo (``inf`` para brazos no jalados)."""
        raise NotImplementedError

    def select_arm(self) -> int:
        raise NotImplementedError


class SoftmaxAgent(BanditAgent):
    """Softmax/Boltzmann: muestrea a con π(a) = exp(Q_a/τ) / Σ_b exp(Q_b/τ).

    Se resta max(Q) antes de exponenciar (estabilidad numérica). τ puede
    decaer (annealing): τ_t = max(τ_min, τ₀·dᵗ).
    """

    name = "softmax"

    def __init__(
        self,
        n_arms: int,
        tau: float = 0.1,
        decay: float = 1.0,
        tau_min: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def tau(self) -> float:
        """τ vigente en el pull actual."""
        raise NotImplementedError

    def probabilities(self) -> np.ndarray:
        """Distribución de Boltzmann actual π, forma ``(K,)``, suma 1."""
        raise NotImplementedError

    def select_arm(self) -> int:
        raise NotImplementedError


def make_agent(name: str, n_arms: int, cfg: AgentConfig, rng: np.random.Generator | None = None) -> BanditAgent:
    """Fábrica: crea el agente ``name`` (``"egreedy"``, ``"optimistic"``, ``"ucb1"``,
    ``"softmax"``) con los hiperparámetros de ``cfg``. Lanza ``ValueError`` si el nombre no existe."""
    raise NotImplementedError
