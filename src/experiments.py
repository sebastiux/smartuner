"""Experimentos: ejecución de los agentes sobre los entornos bandit y comparación.

Papel en el pipeline
--------------------
:mod:`src.pipeline` convierte cada nota en un problema bandit
(:class:`src.environment.SegmentBanditData`) y :mod:`src.agents` define cómo
decide cada algoritmo. Este módulo los PONE A TRABAJAR juntos y mide qué tan
bien lo hacen:

* :func:`run_segment` — el bucle bandit básico de Sutton & Barto sobre UN segmento.
* :func:`run_chain` / :func:`transcribe` — un algoritmo recorre toda la pista y
  produce una tablatura.
* :func:`run_experiment` — el experimento comparativo (≥ 100 corridas por
  algoritmo, curvas de aprendizaje, precisión contra el ground truth, segmento
  ejemplo y barridos de hiperparámetros).
* :class:`LiveSession` — ejecución pull a pull con explicación en español
  (pestaña "Ejecución en vivo" de la GUI).

Conceptos clave
---------------
* **Cadena** (:func:`run_chain`): un agente recorre los segmentos EN ORDEN; en
  cada uno empieza sin conocimiento previo (``reset``), gasta T pulls y
  recomienda una posición. El traste recomendado se vuelve el ``traste_previo``
  del siguiente segmento (penalización de tocabilidad)::

      traste_previo₀ = cfg.env.initial_hand_fret
      para s = 0..S−1:
          entorno_s ← BanditEnvironment(datos_s, traste_previo_s)
          agente.reset(); T × (elegir, jalar, aprender); brazo_s ← recomendar
          traste_previo_{s+1} ← traste(brazo_s)

* **Números aleatorios comunes** (decisión de diseño 7): en la corrida r, el
  entorno del segmento s usa ``np.random.default_rng([seed, r, s])`` para
  TODOS los algoritmos, y el agente usa
  ``default_rng([seed, r, s, 1 + índice_algoritmo])``. El entorno consume una
  única llamada a su generador por pull (el frame), así que en la corrida r el
  pull t de CUALQUIER algoritmo ve el mismo frame j_t: las diferencias entre
  algoritmos no se deben a la suerte del muestreo de frames, sino a sus
  decisiones (técnica clásica de reducción de varianza en simulación).
* **Regret** (pseudo-regret): R_T = Σ_{t=1..T} (μ* − μ_{a_t}), con μ exactos
  del entorno. A diferencia de "recompensa máxima − recompensa obtenida", no
  incluye el ruido del muestreo: mide solo el costo de las DECISIONES. Es ≥ 0
  y no decrece con t; un buen algoritmo lo hace crecer como log T.
* **Oráculo miope** (:func:`oracle_arms`): elige argmax μ en cada segmento,
  dado el traste previo que dejó SU PROPIA elección anterior. Equivale a un
  bandit con T → ∞ (conoce μ sin explorar) que recorre la misma cadena
  greedy que los agentes. Es el techo de RECOMPENSA por segmento y de % de
  pulls óptimos (regret 0), y si se equivoca de pitch el error está en la
  recompensa (espectro, β, λ), no en el algoritmo bandit. NO es un techo de
  precisión contra el ground truth: al decidir segmento a segmento puede
  dejar la mano en un sitio que encarece las notas siguientes, y un bandit
  que se aparta de argmax μ (por exploración) puede acertar MÁS posiciones
  (p. ej. en ``cromatica``: oráculo miope 23.8 % de posición, Softmax ≈ 56 %).
* **Oráculo de cadena** (:func:`oracle_viterbi_arms`): el verdadero óptimo
  del objetivo de la cadena Σ_s μ_s(a_s | traste_previo_s), calculado con
  programación dinámica (Viterbi) sobre la posición de la mano. Tampoco es
  un techo de precisión: optimiza el MODELO de tocabilidad λ·|Δtraste|/12,
  no la digitación del ground truth. En ``cromatica`` la cadena miope y la
  óptima empatan en Σμ (las dos mueven la mano 26 trastes en total) y solo
  el desempate decide: 23.8 % frente a 71.4 % de posición; en
  ``riff_saltos`` la cadena óptima (Σμ mayor) acierta MENOS posiciones que
  la miope (55 % frente a 80 %). Que un mejor Σμ no implique mejor
  precisión indica que el modelo de tocabilidad no coincide con la
  digitación real (que prefiere cuerdas al aire y posiciones bajas).
* **Precisión** (en realidad *recall*: aciertos / notas del GT): cada nota
  del ground truth se empareja con un segmento cuyo onset está a
  ≤ ``onset_tolerance_s`` (emparejamiento bipartito máximo, como
  ``mir_eval``); precisión de pitch = fracción de notas GT cuyo segmento
  recibió el MIDI correcto; de posición = (cuerda, traste) correctos. Las
  notas GT sin segmento cuentan como error. Los segmentos SIN nota GT
  (onsets falsos, notas partidas) no afectan a esta cifra; por eso
  :class:`AccuracyResult` también da la precisión por segmento
  (aciertos / segmentos) y el F1 = 2·aciertos / (notas GT + segmentos).

Rendimiento
-----------
El bucle caliente (:func:`run_segment`) no crea objetos propios por pull: usa
``env.pull`` (que devuelve un ``float``; ``pull_detailed`` solo se usa en la
ejecución en vivo) y guarda brazos y recompensas en listas; el regret y la
optimalidad se calculan al final de forma vectorizada (``gaps[brazos]``).
Además, los agentes de los experimentos se crean con ``explain=False``: eligen
exactamente igual (mismo consumo del generador), pero no redactan la frase
explicativa de cada decisión, que era la mitad del coste de un pull. Coste
medido por pull: ≈ 5 µs (ε-greedy y optimista), 8 µs (UCB1) y 11 µs (Softmax).

Tiempos con la configuración por defecto (T = 500, 100 corridas, barridos con
30 corridas por valor), medidos en la máquina de desarrollo:

=====================================  ==========================  ==========================
Etapa                                  ``linea_simple`` (16 seg.)  ``cromatica`` (21 seg.)
=====================================  ==========================  ==========================
Experimento principal (4 × 100)        24 s                        30 s
Barridos de ε, Q₀, c y τ (22 × 30)     38 s                        49 s
Barrido de λ (6 × 4 × 30)              42 s                        53 s
Total (incluido el segmento ejemplo)   ≈ 106 s                     ≈ 133 s
=====================================  ==========================  ==========================

El coste crece LINEALMENTE con el número de notas (≈ 6.4 s por nota con todo
activado): una canción de 4 minutos (≈ 320 notas) tardaría más de media hora.
Por eso :func:`run_experiment` anuncia al empezar la duración estimada
(:func:`estimate_experiment_seconds`, WARNING si pasa de 10 min) y, con más de
:data:`SWEEP_MAX_SEGMENTS` = 50 notas, los barridos usan menos corridas por
valor (:func:`effective_sweep_runs`: el trabajo de cada punto queda como el de
30 corridas sobre 50 notas). El experimento principal conserva siempre sus
``n_runs`` corridas.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.agents import BanditAgent, Decision, make_agent
from src.config import ALGO_LABELS, ALGORITHMS, STRING_ORDER, CancelledError, Config, ProgressCallback
from src.environment import Arm, BanditEnvironment, SegmentBanditData, next_hand_fret
from src.pipeline import AnalysisResult
from src.segmentation import Segment
from src.synth_dataset import GTNote
from src.tab import TabNote, notes_from_choices

logger = logging.getLogger(__name__)

#: Códigos de resultado por nota del ground truth (heatmap).
OUTCOME_MISSED = 0          # la nota no tiene segmento emparejado
OUTCOME_WRONG_PITCH = 1     # pitch incorrecto
OUTCOME_WRONG_POSITION = 2  # pitch correcto, posición distinta
OUTCOME_EXACT = 3           # cuerda y traste correctos
OUTCOME_LABELS: dict[int, str] = {
    OUTCOME_MISSED: "No detectada",
    OUTCOME_WRONG_PITCH: "Pitch incorrecto",
    OUTCOME_WRONG_POSITION: "Pitch correcto, otra posición",
    OUTCOME_EXACT: "Posición exacta",
}

#: Fracción final del presupuesto usada para "% óptimo" y "recompensa final".
FINAL_FRACTION: float = 0.1

#: Barridos de sensibilidad: (parámetro de Config, algoritmo afectado, lista de valores en ExperimentConfig).
SWEEP_SPECS: tuple[tuple[str, str, str], ...] = (
    ("agent.epsilon", "egreedy", "sweep_epsilon"),
    ("agent.q0", "optimistic", "sweep_q0"),
    ("agent.ucb_c", "ucb1", "sweep_c"),
    ("agent.tau", "softmax", "sweep_tau"),
)

#: Símbolo de cada hiperparámetro barrido (para mensajes y gráficas).
SWEEP_SYMBOLS: dict[str, str] = {
    "agent.epsilon": "ε",
    "agent.q0": "Q₀",
    "agent.ucb_c": "c",
    "agent.tau": "τ",
}

#: Coste aproximado de un pull (µs) de cada algoritmo en los experimentos
#: (``explain=False``), medido en la máquina de desarrollo. Solo se usa para
#: ANUNCIAR la duración estimada del experimento.
PULL_COST_US: dict[str, float] = {"egreedy": 5.0, "optimistic": 5.0, "ucb1": 8.0, "softmax": 11.0}

#: Los barridos (de sensibilidad y de λ) usan como mucho el trabajo de
#: ``sweep_runs`` corridas sobre este número de segmentos: en pistas largas
#: se reducen las corridas (ver :func:`effective_sweep_runs`).
SWEEP_MAX_SEGMENTS: int = 50

#: Mínimo de corridas por punto de barrido cuando se reducen.
MIN_SWEEP_RUNS: int = 5

#: Duración estimada (s) a partir de la cual se avisa con un WARNING.
LONG_EXPERIMENT_S: float = 600.0

#: Parámetros de :class:`src.config.EnvConfig` que determinan los brazos, los
#: frames o la saliencia de cada segmento (los datos bandit precalculados).
#: Los demás (λ, T, ruido, mano inicial, cuerdas al aire gratis) solo afectan
#: al entorno y se aplican en cada corrida.
SEGMENT_DATA_ENV_PARAMS: tuple[str, ...] = (
    "n_frets", "k_semitones", "n_harmonics", "beta", "tolerance_semitones", "attack_skip_s", "spectrum",
)

#: Nombre corto de cada algoritmo para el log de la ejecución en vivo.
ALGO_SHORT: dict[str, str] = {
    "egreedy": "ε-greedy",
    "optimistic": "Optimista",
    "ucb1": "UCB1",
    "softmax": "Softmax",
}


def _algorithm_index(algorithm: str) -> int:
    """Índice de ``algorithm`` en :data:`src.config.ALGORITHMS` (fija la semilla del agente).

    Raises
    ------
    ValueError
        Si el algoritmo no existe.
    """
    try:
        return ALGORITHMS.index(algorithm)
    except ValueError:
        raise ValueError(f"Algoritmo desconocido: '{algorithm}'. Opciones: {', '.join(ALGORITHMS)}") from None


def _n_final(budget: int) -> int:
    """Número de pulls del último 10 % del presupuesto (al menos 1).

    Examples
    --------
    >>> _n_final(300), _n_final(5)
    (30, 1)
    """
    return max(1, int(math.ceil(FINAL_FRACTION * budget)))


def _check_cancel(cancel: threading.Event | None, what: str) -> None:
    """Lanza :class:`CancelledError` si el usuario pidió cancelar."""
    if cancel is not None and cancel.is_set():
        logger.info("Cancelado por el usuario: %s", what)
        raise CancelledError(f"Cancelado por el usuario ({what}).")


class _WorkProgress:
    """Progreso ponderado por trabajo (número de pulls) de un proceso largo.

    Parameters
    ----------
    total_units : float
        Trabajo total (p. ej. pulls totales).
    progress : ProgressCallback | None
        Callback ``progress(fracción, mensaje)``.
    """

    def __init__(self, total_units: float, progress: ProgressCallback | None) -> None:
        self.total = max(float(total_units), 1.0)
        self.done = 0.0
        self._progress = progress

    def advance(self, units: float, message: str) -> None:
        """Suma ``units`` de trabajo hecho e informa."""
        self.done += units
        if self._progress is not None:
            self._progress(min(self.done / self.total, 1.0), message)

    def sub(self, units: float) -> ProgressCallback:
        """Callback para un subproceso que informa su propio 0→1 sobre ``units`` de trabajo.

        Al terminar el subproceso hay que llamar a ``advance(units, ...)``.
        """
        start = self.done

        def report(fraction: float, message: str) -> None:
            """Traduce el 0→1 del subproceso a la fracción global del trabajo."""
            if self._progress is not None:
                f = (start + min(max(fraction, 0.0), 1.0) * units) / self.total
                self._progress(min(f, 1.0), message)

        return report


# ---------------------------------------------------------------------------
# Un segmento / una cadena
# ---------------------------------------------------------------------------


@dataclass
class SegmentRun:
    """Traza de un agente sobre un segmento (T pulls).

    Attributes
    ----------
    arms : np.ndarray
        Brazo jalado en cada pull, forma ``(T,)``.
    rewards : np.ndarray
        Recompensa de cada pull, ``(T,)``.
    regrets : np.ndarray
        Pseudo-regret instantáneo μ* − μ_{a_t}, ``(T,)``.
    optimal : np.ndarray
        Booleano: el brazo jalado era óptimo, ``(T,)``.
    q_history : np.ndarray | None
        Q de todos los brazos después de cada pull, ``(T, K)`` (si se pidió).
    final_q, final_counts : np.ndarray
        Estado final del agente, ``(K,)``.
    recommended : int
        Brazo recomendado (``agent.recommend(cfg.agent.recommend)``).
    """

    arms: np.ndarray
    rewards: np.ndarray
    regrets: np.ndarray
    optimal: np.ndarray
    q_history: np.ndarray | None
    final_q: np.ndarray
    final_counts: np.ndarray
    recommended: int


def run_segment(agent: BanditAgent, env: BanditEnvironment, budget: int, recommend: str = "most_pulled", record_q: bool = False) -> SegmentRun:
    """Bucle bandit básico: ``reset`` → T × (select_arm, pull, update) → recommend.

    Es, literalmente, el bucle de Sutton & Barto (2018, §2.4)::

        agente.reset()                      # Q = Q₀, N = 0, t = 0
        repetir T veces:
            A ← agente.select_arm()         # exploración / explotación
            R ← entorno.pull(A)             # frame al azar: S_j(f_A) − penalización
            agente.update(A, R)             # Q(A) ← Q(A) + α·(R − Q(A))
        recomendado ← agente.recommend()    # brazo más jalado (o argmax Q)

    Parameters
    ----------
    agent : BanditAgent
        Agente (se reinicia al empezar).
    env : BanditEnvironment
        Entorno del segmento (su generador avanza con cada pull).
    budget : int
        Presupuesto T de pulls (≥ 1).
    recommend : str, optional
        ``"most_pulled"`` o ``"greedy"`` (ver :meth:`BanditAgent.recommend`).
    record_q : bool, optional
        Si es True guarda Q de todos los brazos tras cada pull (``q_history``).

    Returns
    -------
    SegmentRun
        Brazos, recompensas, regret instantáneo, optimalidad y estado final.

    Raises
    ------
    ValueError
        Si ``budget < 1`` o si el agente y el entorno tienen distinto K.

    Notes
    -----
    El regret instantáneo es la brecha Δ_{a_t} = μ* − μ_{a_t} del brazo
    jalado (``env.gaps``): se calcula al final, vectorizado, a partir de los
    brazos elegidos (los μ son constantes durante el segmento).

    Examples
    --------
    >>> from src.config import AgentConfig
    >>> from src.segmentation import Segment
    >>> seg = Segment(index=0, start_s=0.0, end_s=1.0, start_sample=0, end_sample=22050, rms_db=0.0)
    >>> sal = np.array([[0.9, 0.2], [0.7, 0.4]])            # 2 frames × 2 brazos
    >>> data = SegmentBanditData(seg, 0, [Arm("A", 0), Arm("A", 1)], np.arange(2), sal)
    >>> env = BanditEnvironment(data, prev_fret=None, lam=0.0, rng=np.random.default_rng(0))
    >>> agent = make_agent("ucb1", 2, AgentConfig(), rng=np.random.default_rng(1))
    >>> run = run_segment(agent, env, budget=50)
    >>> run.recommended, bool(run.regrets.min() >= 0)
    (0, True)
    """
    budget = int(budget)
    if budget < 1:
        raise ValueError(f"El presupuesto T debe ser ≥ 1 (recibido {budget})")
    if agent.n_arms != env.n_arms:
        raise ValueError(f"El agente tiene {agent.n_arms} brazos pero el entorno {env.n_arms}")
    agent.reset()

    # Atajos locales: en CPython evitan buscar el atributo en cada iteración.
    select_arm, pull, update = agent.select_arm, env.pull, agent.update
    arms: list[int] = []
    rewards: list[float] = []
    q_history = np.empty((budget, agent.n_arms), dtype=float) if record_q else None
    for t in range(budget):
        a = select_arm()       # 1. el agente decide
        r = pull(a)            # 2. el entorno responde con una recompensa ruidosa
        update(a, r)           # 3. el agente aprende: Q(a) ← Q(a) + α·(r − Q(a))
        arms.append(a)
        rewards.append(r)
        if q_history is not None:
            q_history[t] = agent.q

    arms_arr = np.asarray(arms, dtype=np.int64)
    optimal_mask = np.zeros(env.n_arms, dtype=bool)
    optimal_mask[env.optimal_arms] = True
    return SegmentRun(
        arms=arms_arr,
        rewards=np.asarray(rewards, dtype=float),
        regrets=env.gaps[arms_arr],          # Δ_{a_t} = μ* − μ_{a_t} ≥ 0
        optimal=optimal_mask[arms_arr],
        q_history=q_history,
        final_q=agent.q.copy(),
        final_counts=agent.counts.copy(),
        recommended=int(agent.recommend(recommend)),
    )


@dataclass
class ChainResult:
    """Una corrida de un algoritmo sobre todos los segmentos.

    Attributes
    ----------
    algorithm : str
        Identificador del algoritmo.
    arms : list[Arm]
        Posición recomendada para cada segmento conservado.
    rewards, regrets : np.ndarray
        Forma ``(S, T)``.
    optimal : np.ndarray
        Forma ``(S, T)``, booleano.
    prev_frets : list[int | None]
        traste_previo usado en cada segmento.
    final_q, final_counts : list[np.ndarray]
        Estado final del agente en cada segmento.
    true_means : list[np.ndarray]
        μ de cada segmento (dependen de prev_fret).
    elapsed_s : float
        Tiempo de pared de la cadena completa (s).
    """

    algorithm: str
    arms: list[Arm]
    rewards: np.ndarray
    regrets: np.ndarray
    optimal: np.ndarray
    prev_frets: list[int | None]
    final_q: list[np.ndarray]
    final_counts: list[np.ndarray]
    true_means: list[np.ndarray]
    elapsed_s: float


def segment_rngs(seed: int, run_index: int, segment: int, algorithm: str) -> tuple[np.random.Generator, np.random.Generator]:
    """Generadores (entorno, agente) del segmento ``segment`` en la corrida ``run_index``.

    * Entorno: ``default_rng([seed, run_index, segment])`` — IGUAL para todos
      los algoritmos (números aleatorios comunes).
    * Agente: ``default_rng([seed, run_index, segment, 1 + índice_algoritmo])``
      — propio de cada algoritmo e independiente del entorno.

    Parameters
    ----------
    seed : int
        Semilla maestra (``cfg.experiment.seed``).
    run_index : int
        Índice de la corrida r.
    segment : int
        Posición s del segmento entre los conservados.
    algorithm : str
        Algoritmo (fija el último elemento de la semilla del agente).

    Returns
    -------
    tuple[np.random.Generator, np.random.Generator]
        ``(rng_entorno, rng_agente)``.

    Examples
    --------
    >>> e1, _ = segment_rngs(42, 3, 5, "egreedy")
    >>> e2, _ = segment_rngs(42, 3, 5, "softmax")
    >>> int(e1.integers(1000)) == int(e2.integers(1000))      # mismo entorno
    True
    """
    algo_idx = _algorithm_index(algorithm)
    env_rng = np.random.default_rng([int(seed), int(run_index), int(segment)])
    agent_rng = np.random.default_rng([int(seed), int(run_index), int(segment), 1 + algo_idx])
    return env_rng, agent_rng


def run_chain(algorithm: str, segment_data: list[SegmentBanditData], cfg: Config, run_index: int = 0) -> ChainResult:
    """Ejecuta ``algorithm`` sobre todos los segmentos en orden (una corrida).

    Para cada segmento s (ver encabezado del módulo):

    1. Entorno con ``traste_previo`` = posición de la mano tras el segmento
       anterior (``cfg.env.initial_hand_fret`` en el primero) y generador
       ``default_rng([cfg.experiment.seed, run_index, s])``.
    2. Agente NUEVO (sin memoria del segmento anterior) con generador
       ``default_rng([seed, run_index, s, 1 + índice_algoritmo])``.
    3. :func:`run_segment` con T = ``cfg.env.budget`` pulls y recomendación
       ``cfg.agent.recommend``.
    4. La mano pasa al traste recomendado (:func:`src.environment.next_hand_fret`).

    Parameters
    ----------
    algorithm : str
        ``"egreedy"``, ``"optimistic"``, ``"ucb1"`` o ``"softmax"``.
    segment_data : list[SegmentBanditData]
        Datos bandit de los segmentos conservados, en orden temporal.
    cfg : Config
        Usa ``env`` (λ, T, ruido, mano inicial), ``agent`` y ``experiment.seed``.
    run_index : int, optional
        Índice de la corrida r (fija las semillas).

    Returns
    -------
    ChainResult
        Posiciones recomendadas, trazas ``(S, T)`` y tiempo de pared.

    Raises
    ------
    ValueError
        Si el algoritmo no existe.
    """
    _algorithm_index(algorithm)  # valida antes de empezar
    budget = int(cfg.env.budget)
    seed = int(cfg.experiment.seed)
    n_seg = len(segment_data)
    rewards = np.empty((n_seg, budget), dtype=float)
    regrets = np.empty((n_seg, budget), dtype=float)
    optimal = np.empty((n_seg, budget), dtype=bool)
    arms: list[Arm] = []
    prev_frets: list[int | None] = []
    final_q: list[np.ndarray] = []
    final_counts: list[np.ndarray] = []
    true_means: list[np.ndarray] = []

    prev: int | None = cfg.env.initial_hand_fret
    t0 = time.perf_counter()
    for s, data in enumerate(segment_data):
        env_rng, agent_rng = segment_rngs(seed, run_index, s, algorithm)
        env = BanditEnvironment.from_config(data, prev, cfg.env, env_rng)
        agent = make_agent(algorithm, data.n_arms, cfg.agent, agent_rng, explain=False)
        run = run_segment(agent, env, budget, cfg.agent.recommend)
        arm = data.arms[run.recommended]

        rewards[s] = run.rewards
        regrets[s] = run.regrets
        optimal[s] = run.optimal
        arms.append(arm)
        prev_frets.append(prev)
        final_q.append(run.final_q)
        final_counts.append(run.final_counts)
        true_means.append(env.true_means)
        prev = next_hand_fret(arm, prev, cfg.env)
    elapsed = time.perf_counter() - t0
    logger.debug("Cadena %s (corrida %d): %d segmentos × T=%d en %.3f s", algorithm, run_index, n_seg, budget, elapsed)
    return ChainResult(algorithm=algorithm, arms=arms, rewards=rewards, regrets=regrets, optimal=optimal,
                       prev_frets=prev_frets, final_q=final_q, final_counts=final_counts,
                       true_means=true_means, elapsed_s=elapsed)


def _oracle_chain(segment_data: list[SegmentBanditData], cfg: Config) -> tuple[list[Arm], list[int | None]]:
    """Cadena del oráculo: brazos elegidos y traste previo usado en cada segmento."""
    arms: list[Arm] = []
    prev_frets: list[int | None] = []
    prev: int | None = cfg.env.initial_hand_fret
    for data in segment_data:
        # El generador no se usa (el oráculo no jala brazos): solo se leen los μ exactos.
        env = BanditEnvironment.from_config(data, prev, cfg.env, np.random.default_rng(0))
        mu = env.true_means
        frets = data.frets
        # Movimiento de la mano |traste − traste_previo| (0 si no hay posición previa o si
        # la cuerda al aire es "gratis"), igual que en la penalización del entorno.
        moves = np.zeros(frets.size, dtype=int) if prev is None else np.abs(frets - prev)
        if cfg.env.open_string_free:
            moves[frets == 0] = 0
        # Entre los brazos óptimos (μ = μ* con tolerancia), desempate:
        # 1) menor movimiento de la mano, 2) menor traste, 3) menor índice.
        best = min(env.optimal_arms.tolist(), key=lambda i: (int(moves[i]), int(frets[i]), i))
        arm = data.arms[best]
        arms.append(arm)
        prev_frets.append(prev)
        logger.debug("Oráculo, segmento %d: %s (μ=%.3f, traste previo %s)", data.position, arm.label, mu[best], prev)
        prev = next_hand_fret(arm, prev, cfg.env)
    return arms, prev_frets


def oracle_arms(segment_data: list[SegmentBanditData], cfg: Config) -> list[Arm]:
    """Cadena del oráculo MIOPE: en cada segmento argmax μ dado su propio traste previo.

    El oráculo conoce los μ_a EXACTOS de cada entorno (media de la saliencia
    sobre todos los frames menos la penalización), así que no necesita
    explorar: es lo que haría un bandit con T → ∞ recorriendo la misma
    cadena greedy que los agentes. Empates en μ (posiciones gemelas con
    λ = 0 o con la misma penalización): menor movimiento
    |traste − traste_previo|, luego menor traste.

    Es el techo de recompensa por segmento (regret 0, 100 % de pulls
    óptimos), pero NO de precisión contra el ground truth: decide segmento a
    segmento y no maximiza la suma de la cadena (ver
    :func:`oracle_viterbi_arms`, el óptimo de toda la cadena).

    Parameters
    ----------
    segment_data : list[SegmentBanditData]
        Segmentos conservados, en orden.
    cfg : Config
        Usa ``env`` (λ, mano inicial, cuerdas al aire gratis).

    Returns
    -------
    list[Arm]
        Posición óptima de cada segmento.

    Examples
    --------
    >>> from src.segmentation import Segment
    >>> seg = Segment(index=0, start_s=0.0, end_s=1.0, start_sample=0, end_sample=1, rms_db=0.0)
    >>> data = SegmentBanditData(seg, 0, [Arm("E", 5), Arm("A", 0)], np.arange(1), np.array([[0.8, 0.8]]))
    >>> cfg = Config(); cfg.env.initial_hand_fret = 5
    >>> [a.label for a in oracle_arms([data], cfg)]       # misma saliencia: E-5 no mueve la mano
    ['E-5']
    """
    return _oracle_chain(segment_data, cfg)[0]


#: Tolerancia absoluta al comparar sumas de μ entre caminos de la cadena
#: (las sumas en coma flotante de caminos empatados difieren en ~1e-16).
CHAIN_TOL: float = 1e-9


def oracle_viterbi_arms(segment_data: list[SegmentBanditData], cfg: Config) -> list[Arm]:
    """Oráculo de CADENA: las posiciones que maximizan Σ_s μ_s(a_s | traste_previo_s).

    El oráculo miope (:func:`oracle_arms`) elige en cada segmento el mejor
    brazo dado el traste previo, sin mirar adelante; puede dejar la mano en
    un sitio que encarece las notas siguientes. Este oráculo resuelve el
    problema de toda la pista con programación dinámica (algoritmo de
    Viterbi) sobre el ESTADO = posición de la mano (traste 0..n_frets)::

        V₀(p) = 0 si p = cfg.env.initial_hand_fret (−∞ en otro caso)
        para cada segmento s y cada brazo a:
            p' = siguiente_mano(a, p)                  (= traste(a), o p si la
                                                        cuerda al aire es gratis)
            V_{s+1}(p') = max_p [ V_s(p) + mean_salience_s[a] − pen(a, p) ]
        óptimo = max_p V_S(p)  → se reconstruye el camino hacia atrás

    con pen(a, p) = λ·|traste_a − p|/12 (0 para las cuerdas al aire si
    ``open_string_free``), es decir, μ_s(a | p) exactamente como en
    :class:`src.environment.BanditEnvironment`. Coste O(S · K · n_frets).

    Parameters
    ----------
    segment_data : list[SegmentBanditData]
        Segmentos conservados, en orden.
    cfg : Config
        Usa ``env`` (λ, mano inicial, cuerdas al aire gratis).

    Returns
    -------
    list[Arm]
        Posición de cada segmento en el camino óptimo.

    Notes
    -----
    Empates (sumas iguales salvo :data:`CHAIN_TOL`): menor movimiento total
    de la mano Σ|Δtraste| y, si también empatan, el primer camino encontrado:
    los estados se recorren en orden creciente de traste, así que gana el que
    llega desde la posición de mano MÁS BAJA. Los empates son habituales:
    con saliencias idénticas en las posiciones gemelas, Σμ = Σ saliencia −
    (λ/12)·movimiento total, y caminos distintos pueden mover la mano lo
    mismo (en ``cromatica`` la cadena miope y la de Viterbi suman 26 trastes).

    Es el óptimo del MODELO, no de la precisión: no tiene por qué acertar
    más posiciones del ground truth que el oráculo miope.

    Examples
    --------
    La mano empieza en el traste 3; la primera nota (A1) se puede tocar en
    E-5 o en A-0 con la misma saliencia, y la segunda solo en D-0. El miope
    elige E-5 (a 2 trastes de la mano, frente a 3) y después paga el salto
    5 → 0; el de cadena acepta mover 3 trastes ahora para no moverse después:

    >>> from src.segmentation import Segment
    >>> seg = Segment(index=0, start_s=0.0, end_s=1.0, start_sample=0, end_sample=1, rms_db=0.0)
    >>> d0 = SegmentBanditData(seg, 0, [Arm("E", 5), Arm("A", 0)], np.arange(1), np.array([[0.8, 0.8]]))
    >>> d1 = SegmentBanditData(seg, 1, [Arm("D", 0)], np.arange(1), np.array([[0.8]]))
    >>> cfg = Config(); cfg.env.lam = 1.2; cfg.env.initial_hand_fret = 3
    >>> myopic, chain = oracle_arms([d0, d1], cfg), oracle_viterbi_arms([d0, d1], cfg)
    >>> [a.label for a in myopic], [a.label for a in chain]
    (['E-5', 'D-0'], ['A-0', 'D-0'])
    >>> round(chain_value(myopic, [d0, d1], cfg), 3), round(chain_value(chain, [d0, d1], cfg), 3)
    (0.9, 1.3)
    """
    if not segment_data:
        return []
    # Estado: posición de la mano → (V, movimiento acumulado). Las retro-referencias
    # de cada segmento guardan, para cada estado nuevo, (estado anterior, brazo).
    states: dict[int | None, tuple[float, int]] = {cfg.env.initial_hand_fret: (0.0, 0)}
    back: list[dict[int | None, tuple[int | None, int]]] = []
    dummy_rng = np.random.default_rng(0)  # el oráculo no jala brazos: solo lee μ exactos

    def order(p: int | None) -> int:
        """Clave de orden de los estados: None (sin posición) antes que cualquier traste."""
        return -1 if p is None else int(p)

    for data in segment_data:
        frets = data.frets
        new_states: dict[int | None, tuple[float, int]] = {}
        new_back: dict[int | None, tuple[int | None, int]] = {}
        for p in sorted(states, key=order):
            value, moved = states[p]
            mu = BanditEnvironment.from_config(data, p, cfg.env, dummy_rng).true_means   # μ_s(· | p)
            for i, arm in enumerate(data.arms):
                q = next_hand_fret(arm, p, cfg.env)
                step = 0 if (p is None or q == p) else abs(int(frets[i]) - int(p))
                cand = (value + float(mu[i]), moved + step)
                best = new_states.get(q)
                if (best is None or cand[0] > best[0] + CHAIN_TOL
                        or (abs(cand[0] - best[0]) <= CHAIN_TOL and cand[1] < best[1])):
                    new_states[q] = cand
                    new_back[q] = (p, i)
        states = new_states
        back.append(new_back)

    # Mejor estado final (mayor V; empates: menor movimiento, luego menor traste).
    final = sorted(states, key=order)
    best_q = final[0]
    for q in final[1:]:
        v, m = states[q]
        bv, bm = states[best_q]
        if v > bv + CHAIN_TOL or (abs(v - bv) <= CHAIN_TOL and m < bm):
            best_q = q
    # Reconstrucción hacia atrás.
    choices: list[int] = []
    q = best_q
    for s in range(len(segment_data) - 1, -1, -1):
        p, i = back[s][q]
        choices.append(i)
        q = p
    choices.reverse()
    arms = [d.arms[i] for d, i in zip(segment_data, choices)]
    logger.debug("Oráculo de cadena (Viterbi): Σμ = %.4f, movimiento total %d trastes", states[best_q][0],
                 states[best_q][1])
    return arms


def chain_value(arms: list[Arm], segment_data: list[SegmentBanditData], cfg: Config) -> float:
    """Valor de una cadena de posiciones: Σ_s μ_s(a_s | traste_previo_s).

    Es el objetivo que maximiza :func:`oracle_viterbi_arms`; sirve para
    comparar cadenas (la del oráculo miope nunca supera a la de Viterbi).

    Parameters
    ----------
    arms : list[Arm]
        Posición elegida en cada segmento (deben estar entre sus brazos candidatos).
    segment_data : list[SegmentBanditData]
        Segmentos conservados, en orden.
    cfg : Config
        Usa ``env`` (λ, mano inicial, cuerdas al aire gratis).

    Returns
    -------
    float
        Suma de los μ exactos de cada elección dado el traste previo que deja la anterior.

    Raises
    ------
    ValueError
        Si las longitudes no coinciden o una posición no es candidata de su segmento.
    """
    if len(arms) != len(segment_data):
        raise ValueError(f"Hay {len(arms)} posiciones para {len(segment_data)} segmentos")
    total = 0.0
    prev: int | None = cfg.env.initial_hand_fret
    for arm, data in zip(arms, segment_data):
        if arm not in data.arms:
            raise ValueError(f"{arm.label} no es un brazo candidato del segmento {data.position}")
        env = BanditEnvironment.from_config(data, prev, cfg.env, np.random.default_rng(0))
        total += float(env.true_means[data.arms.index(arm)])
        prev = next_hand_fret(arm, prev, cfg.env)
    return total


# ---------------------------------------------------------------------------
# Transcripción (una corrida, para la pestaña Tablatura y la CLI)
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Tablatura producida por un algoritmo en una corrida.

    Attributes
    ----------
    algorithm : str
        Algoritmo usado.
    notes : list[TabNote]
        Una nota por segmento conservado.
    chain : ChainResult
        Traza completa de la corrida.
    """

    algorithm: str
    notes: list[TabNote]
    chain: ChainResult


def transcribe(analysis: AnalysisResult, algorithm: str, cfg: Config, run_index: int = 0) -> TranscriptionResult:
    """Ejecuta una cadena y convierte las posiciones recomendadas en :class:`TabNote`.

    Parameters
    ----------
    analysis : AnalysisResult
        Resultado de :func:`src.pipeline.analyze` (usa ``segment_data``).
    algorithm : str
        Algoritmo.
    cfg : Config
        Configuración (agente, λ, T, semilla).
    run_index : int, optional
        Corrida (semillas).

    Returns
    -------
    TranscriptionResult
        Notas de la tablatura y la cadena que las produjo.
    """
    chain = run_chain(algorithm, analysis.segment_data, cfg, run_index=run_index)
    segments = [d.segment for d in analysis.segment_data]
    notes = notes_from_choices(segments, chain.arms)
    logger.info("Transcripción con %s: %d notas en %.0f ms (%s)", ALGO_LABELS.get(algorithm, algorithm),
                len(notes), 1000.0 * chain.elapsed_s, " ".join(n.label for n in notes[:12])
                + (" …" if len(notes) > 12 else ""))
    return TranscriptionResult(algorithm=algorithm, notes=notes, chain=chain)


# ---------------------------------------------------------------------------
# Evaluación contra ground truth
# ---------------------------------------------------------------------------


def match_segments_to_gt(segments: list[Segment], gt: list[GTNote], tolerance_s: float) -> list[int | None]:
    """Empareja cada nota GT con un segmento detectado (uno a uno, emparejamiento MÁXIMO).

    Es un problema de emparejamiento bipartito (notas GT ↔ segmentos) donde
    solo se admiten pares con |onset_GT − inicio_segmento| ≤ ``tolerance_s``.
    Se busca, como ``mir_eval``, el emparejamiento con el MAYOR número de
    pares y, entre ellos, el de menor distancia total. Se resuelve con el
    algoritmo húngaro (``scipy.optimize.linear_sum_assignment``) sobre la
    matriz de costes::

        coste[k, j] = |onset_k − inicio_j|   si ≤ tolerancia
                    = C (grande)              si no

    con C mayor que la suma de cualquier conjunto de distancias admisibles:
    así cada par admisible extra ahorra más de lo que puede costar
    reorganizar los demás. Al final se descartan los pares con coste C.

    ¿Por qué no basta un greedy por distancia? Con semicorcheas rápidas un
    segmento puede quedar dentro de la tolerancia de dos notas: aceptar
    primero el par más cercano puede dejar sin pareja a la otra nota aunque
    exista una asignación que empareje las dos (ver el ejemplo).

    Parameters
    ----------
    segments : list[Segment]
        Segmentos CONSERVADOS, en orden (su índice en esta lista es la
        "posición" usada por experimentos y tablatura).
    gt : list[GTNote]
        Notas del ground truth.
    tolerance_s : float
        Distancia máxima de onset (s).

    Returns
    -------
    list[int | None]
        Para cada nota GT, la posición del segmento emparejado o None.

    Examples
    --------
    >>> segs = [Segment(i, t, t + 0.4, 0, 1, -3.0) for i, t in enumerate([0.51, 1.02, 1.30])]
    >>> gt = [GTNote(0.5, 0.9, 33, "A", 0), GTNote(1.0, 1.4, 35, "A", 2), GTNote(2.0, 2.4, 38, "D", 0)]
    >>> match_segments_to_gt(segs, gt, 0.05)
    [0, 1, None]

    Semicorcheas a 140 bpm (0.107 s) con onsets detectados ≈ 60 ms tarde: el
    par más cercano (segunda nota ↔ primer segmento, a 41 ms) dejaría sin
    pareja a la primera nota; el emparejamiento máximo empareja las dos:

    >>> segs = [Segment(0, 0.066, 0.2, 0, 1, -3.0), Segment(1, 0.170, 0.3, 0, 1, -3.0)]
    >>> gt = [GTNote(0.0, 0.1, 33, "A", 0), GTNote(0.107, 0.2, 35, "A", 2)]
    >>> match_segments_to_gt(segs, gt, 0.08)
    [0, 1]
    """
    from scipy.optimize import linear_sum_assignment  # import diferido (scipy tarda en cargarse)

    matches: list[int | None] = [None] * len(gt)
    if not gt or not segments:
        return matches
    gt_onsets = np.array([n.onset_s for n in gt], dtype=float)
    seg_onsets = np.array([s.start_s for s in segments], dtype=float)
    # Matriz de distancias |onset_GT − inicio_segmento|, forma (n_gt, n_segmentos).
    dist = np.abs(gt_onsets[:, None] - seg_onsets[None, :])
    admissible = dist <= tolerance_s
    if not admissible.any():
        return matches
    # C supera cualquier suma de distancias admisibles (como mucho min(n_gt, n_seg) pares ≤ tolerancia).
    big = float(tolerance_s) * (min(dist.shape) + 1) + 1.0
    cost = np.where(admissible, dist, big)
    rows, cols = linear_sum_assignment(cost)
    for k, j in zip(rows.tolist(), cols.tolist()):
        if admissible[k, j]:
            matches[k] = j
    return matches


@dataclass
class AccuracyResult:
    """Precisión de una transcripción contra el ground truth.

    ``pitch`` y ``position`` se miden sobre las notas del GT (en la jerga de
    recuperación de información son un *recall*: aciertos / notas GT). No
    ven los segmentos de más (onsets falsos, notas partidas en varios
    segmentos), que sí aparecen en la tablatura; para eso están la
    precisión POR SEGMENTO (aciertos / segmentos) y el F1, como en
    ``mir_eval``::

        recall    = aciertos / n_GT
        precisión = aciertos / n_segmentos
        F1        = 2·P·R / (P + R) = 2·aciertos / (n_GT + n_segmentos)

    Attributes
    ----------
    pitch : float
        Fracción de notas GT con MIDI correcto (recall de pitch).
    position : float
        Fracción de notas GT con cuerda y traste correctos (recall de posición).
    outcomes : np.ndarray
        Código ``OUTCOME_*`` por nota GT, forma ``(n_gt,)``.
    n_segments : int | None
        Número de segmentos (notas de la tablatura) evaluados; None si se
        desconoce (entonces la precisión por segmento y el F1 son NaN).

    Examples
    --------
    3 notas en la tablatura (una de ellas espuria) para 2 notas GT, ambas
    acertadas: recall 100 %, pero precisión 2/3 y F1 0.8.

    >>> res = AccuracyResult(pitch=1.0, position=1.0, outcomes=np.array([3, 3]), n_segments=3)
    >>> res.n_extra, round(res.position_precision, 3), round(res.position_f1, 3)
    (1, 0.667, 0.8)
    """

    pitch: float
    position: float
    outcomes: np.ndarray
    n_segments: int | None = None

    @property
    def n_matched(self) -> int:
        """Notas GT emparejadas con algún segmento."""
        return int(np.count_nonzero(np.asarray(self.outcomes) != OUTCOME_MISSED))

    @property
    def n_missed(self) -> int:
        """Notas GT sin segmento (no detectadas)."""
        return int(np.asarray(self.outcomes).size) - self.n_matched

    @property
    def n_extra(self) -> int | None:
        """Segmentos sin nota GT (falsos positivos de la segmentación); None si se desconoce."""
        return None if self.n_segments is None else int(self.n_segments) - self.n_matched

    def _counts(self, minimum_outcome: int) -> int:
        """Notas GT con código ≥ ``minimum_outcome`` (aciertos de pitch o de posición)."""
        return int(np.count_nonzero(np.asarray(self.outcomes) >= minimum_outcome))

    def _precision(self, hits: int) -> float:
        """aciertos / n_segmentos (NaN si no se conoce o no hay segmentos)."""
        if not self.n_segments:
            return float("nan")
        return hits / float(self.n_segments)

    def _f1(self, hits: int) -> float:
        """2·aciertos / (n_GT + n_segmentos) (NaN si se desconoce n_segmentos o no hay nada)."""
        if self.n_segments is None or np.asarray(self.outcomes).size + self.n_segments == 0:
            return float("nan")
        return 2.0 * hits / float(np.asarray(self.outcomes).size + self.n_segments)

    @property
    def pitch_precision(self) -> float:
        """Fracción de segmentos con el pitch de su nota GT (los segmentos de más cuentan como error)."""
        return self._precision(self._counts(OUTCOME_WRONG_POSITION))

    @property
    def position_precision(self) -> float:
        """Fracción de segmentos con la cuerda y el traste de su nota GT."""
        return self._precision(self._counts(OUTCOME_EXACT))

    @property
    def pitch_f1(self) -> float:
        """F1 de pitch = 2·aciertos / (n_GT + n_segmentos)."""
        return self._f1(self._counts(OUTCOME_WRONG_POSITION))

    @property
    def position_f1(self) -> float:
        """F1 de posición = 2·aciertos / (n_GT + n_segmentos)."""
        return self._f1(self._counts(OUTCOME_EXACT))


def score_arms(arms: list[Arm], matches: list[int | None], gt: list[GTNote]) -> AccuracyResult:
    """Compara las posiciones elegidas (por posición de segmento) con el ground truth.

    Para cada nota GT k:

    * sin segmento emparejado → :data:`OUTCOME_MISSED` (cuenta como error);
    * MIDI del brazo elegido ≠ MIDI real → :data:`OUTCOME_WRONG_PITCH`;
    * mismo MIDI pero otra (cuerda, traste) → :data:`OUTCOME_WRONG_POSITION`;
    * misma cuerda y traste → :data:`OUTCOME_EXACT`.

    Parameters
    ----------
    arms : list[Arm]
        Posición elegida para cada segmento conservado.
    matches : list[int | None]
        Salida de :func:`match_segments_to_gt`.
    gt : list[GTNote]
        Notas del ground truth.

    Returns
    -------
    AccuracyResult
        Precisión (recall) de pitch, de posición, código por nota (NaN si
        ``gt`` está vacío) y ``n_segments = len(arms)`` para la precisión
        por segmento y el F1.

    Raises
    ------
    ValueError
        Si ``matches`` y ``gt`` tienen distinta longitud.

    Examples
    --------
    >>> gt = [GTNote(0.5, 0.9, 33, "A", 0), GTNote(1.0, 1.4, 33, "A", 0), GTNote(1.5, 1.9, 38, "D", 0)]
    >>> res = score_arms([Arm("A", 0), Arm("E", 5)], [0, 1, None], gt)
    >>> res.outcomes.tolist(), round(res.pitch, 3), round(res.position, 3)
    ([3, 2, 0], 0.667, 0.333)
    >>> res.n_extra, round(res.pitch_precision, 3), round(res.pitch_f1, 3)
    (0, 1.0, 0.8)
    """
    if len(matches) != len(gt):
        raise ValueError(f"matches tiene {len(matches)} entradas pero el ground truth {len(gt)} notas")
    outcomes = np.empty(len(gt), dtype=np.int64)
    for k, (note, m) in enumerate(zip(gt, matches)):
        if m is None:
            outcomes[k] = OUTCOME_MISSED
            continue
        arm = arms[m]
        if arm.midi != note.midi:
            outcomes[k] = OUTCOME_WRONG_PITCH
        elif arm.string != note.string or arm.fret != note.fret:
            outcomes[k] = OUTCOME_WRONG_POSITION
        else:
            outcomes[k] = OUTCOME_EXACT
    if outcomes.size == 0:
        return AccuracyResult(pitch=float("nan"), position=float("nan"), outcomes=outcomes, n_segments=len(arms))
    pitch = float(np.mean(outcomes >= OUTCOME_WRONG_POSITION))   # pitch correcto (exacto o no)
    position = float(np.mean(outcomes == OUTCOME_EXACT))
    return AccuracyResult(pitch=pitch, position=position, outcomes=outcomes, n_segments=len(arms))


# ---------------------------------------------------------------------------
# Experimento completo
# ---------------------------------------------------------------------------


@dataclass
class ExampleSegmentResult:
    """Datos para las gráficas de un solo segmento (pulls por brazo, evolución de Q).

    Se ejecuta aislado, con ``prev_fret`` fijo (el de la cadena del oráculo), para
    que μ sea el mismo en todas las corridas.

    Attributes
    ----------
    position : int
        Segmento (entre los conservados).
    arm_labels : list[str]
        Etiquetas de los brazos.
    true_means : np.ndarray
        μ de cada brazo, ``(K,)``.
    optimal_arms : np.ndarray
        Índices de brazos óptimos.
    prev_fret : int | None
        traste_previo usado.
    counts : dict[str, np.ndarray]
        Por algoritmo, pulls por brazo de cada corrida, ``(runs, K)``.
    q_mean, q_std : dict[str, np.ndarray]
        Por algoritmo, media y desviación entre corridas de Q tras cada pull, ``(T, K)``.
    """

    position: int
    arm_labels: list[str]
    true_means: np.ndarray
    optimal_arms: np.ndarray
    prev_fret: int | None
    counts: dict[str, np.ndarray]
    q_mean: dict[str, np.ndarray]
    q_std: dict[str, np.ndarray]


@dataclass
class SweepResult:
    """Barrido de un hiperparámetro de un algoritmo.

    Attributes
    ----------
    param : str
        Clave ``"agent.epsilon"``, ``"agent.q0"``, ``"agent.ucb_c"`` o ``"agent.tau"``.
    algorithm : str
        Algoritmo afectado.
    values : list[float]
        Valores evaluados.
    final_reward : np.ndarray
        Recompensa media en el último 10 % de pulls, por valor y corrida, ``(V, runs)``.
    mean_reward : np.ndarray
        Recompensa media en todo el presupuesto, ``(V, runs)``.
    final_regret : np.ndarray
        Regret acumulado al final (medio por segmento), ``(V, runs)``.
    """

    param: str
    algorithm: str
    values: list[float]
    final_reward: np.ndarray
    mean_reward: np.ndarray
    final_regret: np.ndarray


@dataclass
class LambdaSweepResult:
    """Efecto de λ en la precisión.

    Attributes
    ----------
    values : list[float]
        Valores de λ.
    position_acc, pitch_acc : dict[str, np.ndarray]
        Por algoritmo, ``(V, runs)``.
    oracle_position_acc, oracle_pitch_acc : np.ndarray
        Precisión del oráculo miope (argmax μ por segmento), ``(V,)``.
    viterbi_position_acc, viterbi_pitch_acc : np.ndarray | None
        Precisión del oráculo de cadena (:func:`oracle_viterbi_arms`), ``(V,)``;
        None en resultados antiguos.
    """

    values: list[float]
    position_acc: dict[str, np.ndarray]
    pitch_acc: dict[str, np.ndarray]
    oracle_position_acc: np.ndarray
    oracle_pitch_acc: np.ndarray
    viterbi_position_acc: np.ndarray | None = None
    viterbi_pitch_acc: np.ndarray | None = None


def _mean_std(values: np.ndarray | list[float]) -> tuple[float, float]:
    """Media y desviación estándar (poblacional) de ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


@dataclass
class ExperimentResult:
    """Todos los resultados del experimento comparativo.

    Attributes
    ----------
    config : Config
        Configuración usada.
    algorithms : list[str]
        Algoritmos comparados (orden de :data:`src.config.ALGORITHMS`).
    n_runs, budget, n_segments : int
        Dimensiones del experimento.
    reward_curves : dict[str, np.ndarray]
        Recompensa por pull promediada sobre segmentos, ``(runs, T)``.
    regret_curves : dict[str, np.ndarray]
        Regret ACUMULADO vs pulls, promediado sobre segmentos, ``(runs, T)``.
    optimal_curves : dict[str, np.ndarray]
        Fracción de segmentos en que el pull t fue óptimo, ``(runs, T)``.
    choices : dict[str, np.ndarray]
        Posición recomendada por corrida y segmento, ``(runs, S, 2)`` con
        (índice de cuerda, traste).
    elapsed_s : dict[str, np.ndarray]
        Tiempo por corrida (s), ``(runs,)``.
    accuracy : dict[str, list[AccuracyResult]]
        Por algoritmo, una entrada por corrida (vacío si no hay ground truth).
    oracle_arms : list[Arm]
        Cadena del oráculo miope (:func:`oracle_arms`).
    oracle_accuracy : AccuracyResult | None
        Precisión del oráculo miope (None sin ground truth).
    gt_notes : list[GTNote] | None
        Ground truth usado.
    example : ExampleSegmentResult | None
        Segmento ejemplo.
    sweeps : list[SweepResult]
        Barridos de sensibilidad (vacío si se desactivaron).
    lambda_sweep : LambdaSweepResult | None
        Barrido de λ (None si se desactivó o no hay ground truth).
    viterbi_arms : list[Arm]
        Cadena del oráculo de cadena (:func:`oracle_viterbi_arms`).
    viterbi_accuracy : AccuracyResult | None
        Precisión del oráculo de cadena (None sin ground truth).
    """

    config: Config
    algorithms: list[str]
    n_runs: int
    budget: int
    n_segments: int
    reward_curves: dict[str, np.ndarray]
    regret_curves: dict[str, np.ndarray]
    optimal_curves: dict[str, np.ndarray]
    choices: dict[str, np.ndarray]
    elapsed_s: dict[str, np.ndarray]
    accuracy: dict[str, list[AccuracyResult]] = field(default_factory=dict)
    oracle_arms: list[Arm] = field(default_factory=list)
    oracle_accuracy: AccuracyResult | None = None
    gt_notes: list[GTNote] | None = None
    example: ExampleSegmentResult | None = None
    sweeps: list[SweepResult] = field(default_factory=list)
    lambda_sweep: LambdaSweepResult | None = None
    viterbi_arms: list[Arm] = field(default_factory=list)
    viterbi_accuracy: AccuracyResult | None = None

    def modal_arms(self, algorithm: str) -> list[Arm]:
        """Posición más frecuente entre corridas para cada segmento.

        Parameters
        ----------
        algorithm : str
            Algoritmo (clave de ``choices``).

        Returns
        -------
        list[Arm]
            Una posición por segmento. En caso de empate gana la que apareció
            primero (en la corrida de menor índice).
        """
        choices = self.choices[algorithm]  # (runs, S, 2)
        modal: list[Arm] = []
        for s in range(choices.shape[1]):
            counter = Counter((int(si), int(fret)) for si, fret in choices[:, s, :])
            (si, fret), _ = counter.most_common(1)[0]   # most_common respeta el orden de aparición en empates
            modal.append(Arm(STRING_ORDER[si], fret))
        return modal

    def summary_rows(self) -> list[dict[str, object]]:
        """Tabla comparativa final, una fila por algoritmo (+ los dos oráculos si hay GT).

        Claves: ``algorithm``, ``label``, ``mean_reward``, ``mean_reward_std``,
        ``final_regret``, ``final_regret_std``, ``optimal_pct`` (último 10 % de
        pulls), ``optimal_pct_std``, ``pitch_acc``, ``pitch_acc_std``,
        ``position_acc``, ``position_acc_std``, ``pitch_f1``, ``pitch_f1_std``,
        ``position_f1``, ``position_f1_std``, ``n_extra``, ``ms_per_segment``,
        ``ms_per_segment_std``. Las precisiones son None sin ground truth.

        Returns
        -------
        list[dict[str, object]]
            Medias y desviaciones ENTRE CORRIDAS de:

            * ``mean_reward``: recompensa media por pull (todo el presupuesto,
              promediada sobre segmentos);
            * ``final_regret``: regret acumulado al final de los T pulls, medio
              por segmento;
            * ``optimal_pct``: % de pulls óptimos en el último 10 % del presupuesto;
            * ``pitch_acc`` / ``position_acc``: fracciones en [0, 1] de las
              notas del GT acertadas (recall);
            * ``pitch_f1`` / ``position_f1``: F1 = 2·aciertos / (notas GT +
              segmentos), que también penaliza los segmentos de más;
            * ``n_extra``: segmentos sin nota GT (igual en todas las corridas);
            * ``ms_per_segment``: tiempo de pared por segmento (ms).

            La fila del oráculo miope (``algorithm == "oracle"``) tiene regret
            0, 100 % óptimo y su precisión; recompensa y tiempo quedan en None.
            La del oráculo de cadena (``"oracle_viterbi"``) solo tiene
            precisiones: su elección puede no ser argmax μ en cada segmento, así
            que regret y % óptimo (definidos por segmento) no aplican.
        """
        rows: list[dict[str, object]] = []
        n_final = _n_final(self.budget)
        n_seg = max(self.n_segments, 1)
        for algo in self.algorithms:
            mean_reward = self.reward_curves[algo].mean(axis=1)
            final_regret = self.regret_curves[algo][:, -1]
            optimal_pct = 100.0 * self.optimal_curves[algo][:, -n_final:].mean(axis=1)
            ms = 1000.0 * self.elapsed_s[algo] / n_seg
            row: dict[str, object] = {"algorithm": algo, "label": ALGO_LABELS.get(algo, algo)}
            row["mean_reward"], row["mean_reward_std"] = _mean_std(mean_reward)
            row["final_regret"], row["final_regret_std"] = _mean_std(final_regret)
            row["optimal_pct"], row["optimal_pct_std"] = _mean_std(optimal_pct)
            row.update(_accuracy_columns(self.accuracy.get(algo) or []))
            row["ms_per_segment"], row["ms_per_segment_std"] = _mean_std(ms)
            rows.append(row)
        if self.oracle_accuracy is not None:
            row = {
                "algorithm": "oracle", "label": ALGO_LABELS["oracle"],
                "mean_reward": None, "mean_reward_std": None,
                "final_regret": 0.0, "final_regret_std": 0.0,
                "optimal_pct": 100.0, "optimal_pct_std": 0.0,
            }
            row.update(_accuracy_columns([self.oracle_accuracy]))
            row["ms_per_segment"] = row["ms_per_segment_std"] = None
            rows.append(row)
        if self.viterbi_accuracy is not None:
            row = {
                "algorithm": "oracle_viterbi", "label": ALGO_LABELS["oracle_viterbi"],
                "mean_reward": None, "mean_reward_std": None,
                "final_regret": None, "final_regret_std": None,
                "optimal_pct": None, "optimal_pct_std": None,
            }
            row.update(_accuracy_columns([self.viterbi_accuracy]))
            row["ms_per_segment"] = row["ms_per_segment_std"] = None
            rows.append(row)
        return rows

    def to_csv(self, path: str) -> str:
        """Escribe :meth:`summary_rows` en CSV (columnas :data:`SUMMARY_COLUMNS`).

        Los valores None se escriben como celdas vacías y los números con 6
        decimales. Se crean las carpetas intermedias.

        El tiempo de pared (``ms_per_segment``) NO se incluye: depende de la
        máquina y de la carga del sistema, así que dos ejecuciones con la misma
        semilla darían archivos distintos. Este CSV es 100 % reproducible; los
        tiempos se guardan aparte con :meth:`timing_to_csv`.

        Parameters
        ----------
        path : str
            Ruta del CSV (p. ej. ``"results/x/summary.csv"``).

        Returns
        -------
        str
            Ruta escrita.
        """
        rows = self.summary_rows()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        columns = list(SUMMARY_COLUMNS)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(row.get(k)) for k in columns})
        logger.info("Tabla comparativa guardada en %s", out)
        return str(out)

    def timing_to_csv(self, path: str) -> str:
        """Escribe el tiempo de cómputo por algoritmo (columnas :data:`TIMING_COLUMNS`).

        Una fila por algoritmo: media y desviación entre corridas del tiempo
        de pared por segmento (ms) y tiempo total del experimento principal
        (s). Al contrario que :meth:`to_csv`, NO es reproducible bit a bit.

        Parameters
        ----------
        path : str
            Ruta del CSV (se crean las carpetas intermedias).

        Returns
        -------
        str
            Ruta escrita.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        n_seg = max(self.n_segments, 1)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(TIMING_COLUMNS)
            for algo in self.algorithms:
                ms = 1000.0 * self.elapsed_s[algo] / n_seg
                mean, std = _mean_std(ms)
                writer.writerow([algo, ALGO_LABELS.get(algo, algo), _csv_value(mean), _csv_value(std),
                                 _csv_value(float(self.elapsed_s[algo].sum()))])
        logger.info("Tiempos de cómputo guardados en %s", out)
        return str(out)

    def curves_to_csv(self, path: str) -> str:
        """Escribe las curvas de aprendizaje en un CSV largo (una fila por algoritmo y pull).

        Columnas: ``algorithm, pull, reward_mean, reward_std, regret_mean,
        regret_std, optimal_mean, optimal_std``; ``pull`` = 1..T, media y
        desviación entre corridas; ``optimal_*`` es una fracción en [0, 1].
        Reproducible con la misma semilla.

        Parameters
        ----------
        path : str
            Ruta del CSV (se crean las carpetas intermedias).

        Returns
        -------
        str
            Ruta escrita.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        columns = ["algorithm", "pull", "reward_mean", "reward_std", "regret_mean", "regret_std",
                   "optimal_mean", "optimal_std"]
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for algo in self.algorithms:
                curves = (self.reward_curves[algo], self.regret_curves[algo], self.optimal_curves[algo])
                stats = [(c.mean(axis=0), c.std(axis=0)) for c in curves]
                for t in range(self.budget):
                    row: list[object] = [algo, t + 1]
                    for mean, std in stats:
                        row += [_csv_value(float(mean[t])), _csv_value(float(std[t]))]
                    writer.writerow(row)
        logger.info("Curvas de aprendizaje guardadas en %s", out)
        return str(out)


#: Columnas de :meth:`ExperimentResult.to_csv` (orden de :meth:`ExperimentResult.summary_rows`).
#: (Sin el tiempo de pared, que no es reproducible: ver :data:`TIMING_COLUMNS`.)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "algorithm", "label", "mean_reward", "mean_reward_std", "final_regret", "final_regret_std",
    "optimal_pct", "optimal_pct_std", "pitch_acc", "pitch_acc_std", "position_acc", "position_acc_std",
    "pitch_f1", "pitch_f1_std", "position_f1", "position_f1_std", "n_extra",
)


def _accuracy_columns(accs: list[AccuracyResult]) -> dict[str, object]:
    """Columnas de precisión de :meth:`ExperimentResult.summary_rows` (media y std entre corridas).

    Parameters
    ----------
    accs : list[AccuracyResult]
        Una entrada por corrida (vacía si no hay ground truth).

    Returns
    -------
    dict[str, object]
        ``pitch_acc``, ``position_acc``, ``pitch_f1``, ``position_f1`` (con su
        ``_std``) y ``n_extra``; todos None si ``accs`` está vacía.
    """
    keys = ("pitch_acc", "position_acc", "pitch_f1", "position_f1")
    if not accs:
        out: dict[str, object] = {k: None for k in keys}
        out.update({f"{k}_std": None for k in keys})
        out["n_extra"] = None
        return out
    out = {}
    for key, values in (("pitch_acc", [a.pitch for a in accs]), ("position_acc", [a.position for a in accs]),
                        ("pitch_f1", [a.pitch_f1 for a in accs]), ("position_f1", [a.position_f1 for a in accs])):
        mean, std = _mean_std(values)
        out[key], out[f"{key}_std"] = (None, None) if math.isnan(mean) else (mean, std)
    out["n_extra"] = accs[0].n_extra
    return out

#: Columnas de :meth:`ExperimentResult.timing_to_csv`.
TIMING_COLUMNS: tuple[str, ...] = ("algorithm", "label", "ms_per_segment", "ms_per_segment_std", "total_s")


def _csv_value(value: object) -> object:
    """Celda de CSV: None → vacío, floats con 6 decimales."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, 6)
    return value


def _default_example_position(segment_data: list[SegmentBanditData], cfg: Config) -> int:
    """Segmento ejemplo: ``cfg.experiment.example_segment`` o el de más brazos (el primero si empatan)."""
    wanted = cfg.experiment.example_segment
    if wanted is not None:
        if 0 <= int(wanted) < len(segment_data):
            return int(wanted)
        logger.warning("El segmento ejemplo %s no existe (hay %d); se usa el de más brazos", wanted, len(segment_data))
    n_arms = [d.n_arms for d in segment_data]
    return int(np.argmax(n_arms))


def run_example_segment(segment_data: list[SegmentBanditData], cfg: Config, position: int | None = None,
                        progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> ExampleSegmentResult:
    """Corre todos los algoritmos ``cfg.experiment.n_runs`` veces sobre un único segmento.

    Por defecto el segmento es ``cfg.experiment.example_segment`` o, si es
    None, el de más brazos (el problema más interesante). El ``traste_previo``
    es el de la cadena del oráculo, fijo en todas las corridas para que los μ
    no cambien. Las semillas son las mismas que en :func:`run_chain` para ese
    segmento (``[seed, r, posición]`` el entorno y ``[seed, r, posición, 1 +
    índice_algoritmo]`` el agente): números aleatorios comunes.

    Parameters
    ----------
    segment_data : list[SegmentBanditData]
        Segmentos conservados.
    cfg : Config
        Usa ``env``, ``agent`` y ``experiment`` (``n_runs``, ``seed``,
        ``algorithms``, ``example_segment``).
    position : int | None, optional
        Segmento a usar (posición entre los conservados).
    progress : ProgressCallback | None, optional
        ``progress(fracción, mensaje)`` tras cada corrida.
    cancel : threading.Event | None, optional
        Se comprueba entre corridas.

    Returns
    -------
    ExampleSegmentResult
        Conteos por brazo ``(runs, K)`` y media/desviación de Q ``(T, K)`` por algoritmo.

    Raises
    ------
    ValueError
        Si no hay segmentos.
    IndexError
        Si ``position`` está fuera de rango.
    src.config.CancelledError
        Si ``cancel`` se activa.
    """
    if not segment_data:
        raise ValueError("No hay segmentos conservados: no se puede elegir un segmento ejemplo.")
    if position is None:
        position = _default_example_position(segment_data, cfg)
    elif not 0 <= int(position) < len(segment_data):
        raise IndexError(f"Segmento ejemplo {position} fuera de rango (hay {len(segment_data)})")
    position = int(position)
    data = segment_data[position]
    prev_fret = _oracle_chain(segment_data, cfg)[1][position]
    algorithms = _selected_algorithms(cfg)
    n_runs = int(cfg.experiment.n_runs)
    budget = int(cfg.env.budget)
    reference_env = BanditEnvironment.from_config(data, prev_fret, cfg.env, np.random.default_rng(0))
    logger.info("Segmento ejemplo %d: %d brazos, traste previo %s, óptimo(s) %s", position, data.n_arms, prev_fret,
                ", ".join(data.arms[i].label for i in reference_env.optimal_arms))

    counts: dict[str, np.ndarray] = {}
    q_mean: dict[str, np.ndarray] = {}
    q_std: dict[str, np.ndarray] = {}
    total = max(len(algorithms) * n_runs, 1)
    done = 0
    for algo in algorithms:
        algo_counts = np.empty((n_runs, data.n_arms), dtype=np.int64)
        q_runs = np.empty((n_runs, budget, data.n_arms), dtype=float)
        for r in range(n_runs):
            _check_cancel(cancel, "segmento ejemplo")
            env_rng, agent_rng = segment_rngs(cfg.experiment.seed, r, position, algo)
            env = BanditEnvironment.from_config(data, prev_fret, cfg.env, env_rng)
            agent = make_agent(algo, data.n_arms, cfg.agent, agent_rng, explain=False)
            run = run_segment(agent, env, budget, cfg.agent.recommend, record_q=True)
            algo_counts[r] = run.final_counts
            q_runs[r] = run.q_history
            done += 1
            if progress is not None:
                progress(done / total, f"Segmento ejemplo: {ALGO_LABELS[algo]} {r + 1}/{n_runs}")
        counts[algo] = algo_counts
        q_mean[algo] = q_runs.mean(axis=0)
        q_std[algo] = q_runs.std(axis=0)
    return ExampleSegmentResult(
        position=position,
        arm_labels=[a.label for a in data.arms],
        true_means=reference_env.true_means.copy(),
        optimal_arms=reference_env.optimal_arms.copy(),
        prev_fret=prev_fret,
        counts=counts,
        q_mean=q_mean,
        q_std=q_std,
    )


def _selected_algorithms(cfg: Config) -> list[str]:
    """Algoritmos de ``cfg.experiment.algorithms`` en el orden fijo de :data:`ALGORITHMS`.

    Raises
    ------
    ValueError
        Si hay algún nombre desconocido o la lista queda vacía.
    """
    wanted = list(cfg.experiment.algorithms)
    unknown = [a for a in wanted if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"Algoritmos desconocidos: {', '.join(unknown)}. Opciones: {', '.join(ALGORITHMS)}")
    selected = [a for a in ALGORITHMS if a in wanted]
    if not selected:
        raise ValueError("No hay algoritmos seleccionados para el experimento.")
    return selected


def _sweep_plan(cfg: Config) -> list[tuple[str, str, list[float]]]:
    """(parámetro, algoritmo, valores) de cada barrido que se ejecutará."""
    selected = set(_selected_algorithms(cfg))
    plan: list[tuple[str, str, list[float]]] = []
    for param, algo, list_name in SWEEP_SPECS:
        values = [float(v) for v in getattr(cfg.experiment, list_name)]
        if algo in selected and values:
            plan.append((param, algo, values))
    return plan


def effective_sweep_runs(cfg: Config, n_segments: int) -> int:
    """Corridas por punto de los barridos, reducidas en pistas largas.

    El coste de un barrido es proporcional a corridas × segmentos. Con más de
    :data:`SWEEP_MAX_SEGMENTS` segmentos se usan

        corridas = min(sweep_runs, max(MIN_SWEEP_RUNS, ⌈sweep_runs · 50 / S⌉))

    de modo que el trabajo (y la precisión de la media, que promedia
    corridas × segmentos) queda como con 50 segmentos. Una canción de 320
    notas pasa de 30 a 5 corridas por punto: los barridos tardan como los de
    una pista de 50 notas en lugar de ≈ 25 minutos.

    Parameters
    ----------
    cfg : Config
        Usa ``experiment.sweep_runs``.
    n_segments : int
        Número S de segmentos conservados.

    Returns
    -------
    int
        Corridas por valor barrido (≥ 1).

    Examples
    --------
    >>> cfg = Config()
    >>> effective_sweep_runs(cfg, 16), effective_sweep_runs(cfg, 80), effective_sweep_runs(cfg, 320)
    (30, 19, 5)
    """
    runs = max(int(cfg.experiment.sweep_runs), 1)
    if n_segments <= SWEEP_MAX_SEGMENTS:
        return runs
    return min(runs, max(MIN_SWEEP_RUNS, int(math.ceil(runs * SWEEP_MAX_SEGMENTS / n_segments))))


def run_sensitivity(segment_data: list[SegmentBanditData], cfg: Config, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> list[SweepResult]:
    """Barridos ε (egreedy), Q₀ (optimistic), c (ucb1) y τ (softmax) con ``sweep_runs`` corridas.

    Para cada parámetro y valor v se hace una COPIA de ``cfg`` con ese valor
    (``cfg`` nunca se modifica) y se ejecutan ``cfg.experiment.sweep_runs``
    cadenas completas (corridas 0..sweep_runs−1, mismas semillas para todos
    los valores: las diferencias se deben solo al hiperparámetro). Solo se
    barren los algoritmos de ``cfg.experiment.algorithms``. Con más de
    :data:`SWEEP_MAX_SEGMENTS` segmentos las corridas se reducen
    (:func:`effective_sweep_runs`).

    Parameters
    ----------
    segment_data : list[SegmentBanditData]
        Segmentos conservados.
    cfg : Config
        Configuración base (valores en ``experiment.sweep_epsilon``, ``sweep_q0``,
        ``sweep_c`` y ``sweep_tau``).
    progress : ProgressCallback | None, optional
        ``progress(fracción, mensaje)`` tras cada cadena.
    cancel : threading.Event | None, optional
        Se comprueba entre cadenas.

    Returns
    -------
    list[SweepResult]
        Un resultado por parámetro barrido (orden ε, Q₀, c, τ).

    Raises
    ------
    src.config.CancelledError
        Si ``cancel`` se activa.
    """
    runs = effective_sweep_runs(cfg, len(segment_data))
    if runs < int(cfg.experiment.sweep_runs):
        logger.info("Barridos de sensibilidad: %d segmentos → %d corridas por valor (en lugar de %d) para "
                    "acotar el tiempo", len(segment_data), runs, cfg.experiment.sweep_runs)
    plan = _sweep_plan(cfg)
    total = max(sum(len(values) for _, _, values in plan) * runs, 1)
    done = 0
    n_final = _n_final(int(cfg.env.budget))
    results: list[SweepResult] = []
    for param, algo, values in plan:
        symbol = SWEEP_SYMBOLS.get(param, param)
        t0 = time.perf_counter()
        final_reward = np.empty((len(values), runs))
        mean_reward = np.empty((len(values), runs))
        final_regret = np.empty((len(values), runs))
        for v_idx, value in enumerate(values):
            cfg_v = copy.deepcopy(cfg)
            cfg_v.set(param, value)
            for r in range(runs):
                _check_cancel(cancel, f"barrido de {param}")
                chain = run_chain(algo, segment_data, cfg_v, run_index=r)
                final_reward[v_idx, r] = chain.rewards[:, -n_final:].mean()
                mean_reward[v_idx, r] = chain.rewards.mean()
                final_regret[v_idx, r] = chain.regrets.sum(axis=1).mean()
                done += 1
                if progress is not None:
                    progress(done / total, f"Barrido {symbol}={value:g} ({ALGO_LABELS[algo]}): corrida {r + 1}/{runs}")
        results.append(SweepResult(param=param, algorithm=algo, values=list(values), final_reward=final_reward,
                                   mean_reward=mean_reward, final_regret=final_regret))
        best = int(np.argmax(final_reward.mean(axis=1)))
        logger.info("Barrido de %s (%s): %s = %s → recompensa final %s; mejor %s = %g (%.1f s)", symbol,
                    ALGO_LABELS[algo], symbol, ", ".join(f"{v:g}" for v in values),
                    ", ".join(f"{m:.3f}" for m in final_reward.mean(axis=1)), symbol, values[best],
                    time.perf_counter() - t0)
    return results


def run_lambda_sweep(analysis: AnalysisResult, cfg: Config, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> LambdaSweepResult | None:
    """Precisión de pitch/posición vs λ para cada algoritmo y el oráculo (None sin GT).

    λ solo cambia la penalización de tocabilidad (no la saliencia), así que se
    reutilizan los ``segment_data`` del análisis. Para cada λ se ejecutan
    ``sweep_runs`` cadenas por algoritmo (reducidas en pistas largas, ver
    :func:`effective_sweep_runs`; sobre una copia de ``cfg``) y las cadenas
    de los dos oráculos (miope y de cadena).

    Parameters
    ----------
    analysis : AnalysisResult
        Análisis con ``ground_truth``.
    cfg : Config
        Usa ``experiment.sweep_lambda``, ``sweep_runs``, ``onset_tolerance_s``
        y ``algorithms``.
    progress : ProgressCallback | None, optional
        ``progress(fracción, mensaje)`` tras cada cadena.
    cancel : threading.Event | None, optional
        Se comprueba entre cadenas.

    Returns
    -------
    LambdaSweepResult | None
        None si el análisis no tiene ground truth o no hay valores de λ.

    Raises
    ------
    src.config.CancelledError
        Si ``cancel`` se activa.
    """
    gt = analysis.ground_truth
    if not gt:
        logger.info("Barrido de λ omitido: no hay ground truth para medir la precisión")
        return None
    segment_data = analysis.segment_data
    values = [float(v) for v in cfg.experiment.sweep_lambda]
    if not values:
        logger.info("Barrido de λ omitido: la lista de valores está vacía")
        return None
    algorithms = _selected_algorithms(cfg)
    runs = effective_sweep_runs(cfg, len(segment_data))
    matches = match_segments_to_gt([d.segment for d in segment_data], gt, cfg.experiment.onset_tolerance_s)
    position_acc = {a: np.empty((len(values), runs)) for a in algorithms}
    pitch_acc = {a: np.empty((len(values), runs)) for a in algorithms}
    oracle_position = np.empty(len(values))
    oracle_pitch = np.empty(len(values))
    viterbi_position = np.empty(len(values))
    viterbi_pitch = np.empty(len(values))
    total = max(len(values) * len(algorithms) * runs, 1)
    done = 0
    t0 = time.perf_counter()
    for v_idx, lam in enumerate(values):
        cfg_v = copy.deepcopy(cfg)
        cfg_v.env.lam = lam
        oracle_acc = score_arms(oracle_arms(segment_data, cfg_v), matches, gt)
        oracle_position[v_idx], oracle_pitch[v_idx] = oracle_acc.position, oracle_acc.pitch
        viterbi_acc = score_arms(oracle_viterbi_arms(segment_data, cfg_v), matches, gt)
        viterbi_position[v_idx], viterbi_pitch[v_idx] = viterbi_acc.position, viterbi_acc.pitch
        for algo in algorithms:
            for r in range(runs):
                _check_cancel(cancel, "barrido de λ")
                chain = run_chain(algo, segment_data, cfg_v, run_index=r)
                acc = score_arms(chain.arms, matches, gt)
                position_acc[algo][v_idx, r] = acc.position
                pitch_acc[algo][v_idx, r] = acc.pitch
                done += 1
                if progress is not None:
                    progress(done / total, f"Barrido λ={lam:g} ({ALGO_LABELS[algo]}): corrida {r + 1}/{runs}")
        logger.info("λ=%g: precisión de posición — oráculo miope %.0f %%, oráculo de cadena %.0f %%, %s", lam,
                    100 * oracle_acc.position, 100 * viterbi_acc.position,
                    ", ".join(f"{ALGO_LABELS[a]} {100 * position_acc[a][v_idx].mean():.0f} %" for a in algorithms))
    logger.info("Barrido de λ terminado en %.1f s", time.perf_counter() - t0)
    return LambdaSweepResult(values=values, position_acc=position_acc, pitch_acc=pitch_acc,
                             oracle_position_acc=oracle_position, oracle_pitch_acc=oracle_pitch,
                             viterbi_position_acc=viterbi_position, viterbi_pitch_acc=viterbi_pitch)


def _check_analysis_config(analysis: AnalysisResult, cfg: Config) -> Config:
    """Comprueba que ``analysis.segment_data`` se calculó con los parámetros de ``cfg``.

    Los brazos, los frames y la saliencia de cada segmento dependen de
    :data:`SEGMENT_DATA_ENV_PARAMS`, de los parámetros del espectro y de
    ``pitch.min_voiced_ratio``: si ``cfg`` difiere de ``analysis.config`` en
    alguno, el experimento correría sobre datos de OTRA configuración y el
    resultado quedaría mal etiquetado.

    Parameters
    ----------
    analysis : AnalysisResult
        Análisis (su ``config`` es la usada para calcular ``segment_data``).
    cfg : Config
        Configuración pedida para el experimento (no se modifica).

    Returns
    -------
    Config
        Copia de ``cfg`` cuyas secciones de las etapas 1–5 (audio,
        preprocesamiento, segmentación y pitch) son las REALMENTE usadas por
        el análisis (con un aviso en el log si ``cfg`` pedía otras).

    Raises
    ------
    ValueError
        Si difiere algún parámetro que cambia los datos bandit (hay que
        llamar antes a :func:`src.pipeline.rebuild_segment_data`).
    """
    from src.pipeline import _ignored_changes, _spectrum_changed

    used = analysis.config
    changed = [f"env.{n}" for n in SEGMENT_DATA_ENV_PARAMS if getattr(used.env, n) != getattr(cfg.env, n)]
    if _spectrum_changed(used, cfg):
        changed += [f"env.{n}" for n in ("n_fft", "bins_per_octave", "cqt_fmin_hz", "n_octaves")
                    if getattr(used.env, n) != getattr(cfg.env, n) and f"env.{n}" not in changed]
    if used.pitch.min_voiced_ratio != cfg.pitch.min_voiced_ratio:
        changed.append("pitch.min_voiced_ratio")
    if changed:
        raise ValueError("Los datos bandit del análisis se calcularon con otra configuración ("
                         + ", ".join(f"{k}: {used.get(k)!r} → {cfg.get(k)!r}" for k in changed)
                         + "). Recalcúlalos con src.pipeline.rebuild_segment_data(análisis, cfg) antes del experimento.")
    out = copy.deepcopy(cfg)
    ignored = _ignored_changes(used, cfg)
    if ignored:
        logger.warning("El análisis se hizo con otros parámetros de %s; el experimento usa (y guarda) los del "
                       "análisis", ", ".join(ignored))
        for section in ("audio", "preprocess", "segmentation", "pitch"):
            setattr(out, section, copy.deepcopy(getattr(used, section)))
    return out


def estimate_experiment_seconds(n_segments: int, cfg: Config, with_lambda: bool = True) -> dict[str, float]:
    """Duración estimada (s) de cada parte de :func:`run_experiment`.

    Se multiplica el número de pulls de cada parte por el coste por pull de
    cada algoritmo (:data:`PULL_COST_US`); el coste crece LINEALMENTE con el
    número de segmentos (≈ 1.4 s por segmento solo en el experimento
    principal con 100 corridas y T = 500).

    Parameters
    ----------
    n_segments : int
        Segmentos conservados S.
    cfg : Config
        Usa ``env.budget`` y ``experiment`` (algoritmos, corridas, barridos).
    with_lambda : bool, optional
        Si el barrido de λ se ejecutará (requiere ground truth).

    Returns
    -------
    dict[str, float]
        Claves ``"main"``, ``"example"``, ``"sweeps"``, ``"lambda"`` y ``"total"`` en segundos.

    Examples
    --------
    >>> est = estimate_experiment_seconds(16, Config())
    >>> round(est["main"]), round(est["sweeps"]), round(est["lambda"])
    (23, 38, 42)
    """
    algorithms = _selected_algorithms(cfg)
    budget = int(cfg.env.budget)
    n_runs = int(cfg.experiment.n_runs)
    sweep_runs = effective_sweep_runs(cfg, n_segments)
    us = PULL_COST_US
    main = sum(us.get(a, 10.0) for a in algorithms) * n_runs * n_segments * budget
    example = sum(us.get(a, 10.0) for a in algorithms) * n_runs * budget
    sweeps = (sum(len(v) * us.get(a, 10.0) for _, a, v in _sweep_plan(cfg)) * sweep_runs * n_segments * budget
              if cfg.experiment.run_sweeps else 0.0)
    lam = (len(cfg.experiment.sweep_lambda) * sum(us.get(a, 10.0) for a in algorithms) * sweep_runs
           * n_segments * budget if (cfg.experiment.run_lambda_sweep and with_lambda) else 0.0)
    out = {"main": main / 1e6, "example": example / 1e6, "sweeps": sweeps / 1e6, "lambda": lam / 1e6}
    out["total"] = sum(out.values())
    return out


def run_experiment(analysis: AnalysisResult, cfg: Config, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> ExperimentResult:
    """Experimento completo: ``n_runs`` cadenas por algoritmo, oráculo, precisión,
    segmento ejemplo y (opcionalmente) barridos.

    Pasos:

    1. Para cada algoritmo de ``cfg.experiment.algorithms`` y cada corrida
       r = 0..n_runs−1, una cadena (:func:`run_chain`) con números aleatorios
       comunes. De cada cadena se guardan las curvas promediadas sobre
       segmentos (recompensa por pull, regret ACUMULADO, fracción de pulls
       óptimos), las posiciones elegidas, el tiempo y la precisión (si hay GT).
    2. Cadenas de los dos oráculos (miope y de cadena) y su precisión.
    3. Segmento ejemplo (:func:`run_example_segment`).
    4. Barridos de sensibilidad (si ``run_sweeps``) y de λ (si
       ``run_lambda_sweep`` y hay GT), con menos corridas en pistas largas
       (:func:`effective_sweep_runs`).

    Antes de empezar se comprueba que ``analysis.segment_data`` se calculó
    con los parámetros del entorno de ``cfg`` y se anuncia en el log la
    duración estimada (:func:`estimate_experiment_seconds`; WARNING si
    supera :data:`LONG_EXPERIMENT_S`). El progreso se pondera por trabajo
    (número de pulls) y ``cancel`` se comprueba entre cadenas.

    Parameters
    ----------
    analysis : AnalysisResult
        Resultado de :func:`src.pipeline.analyze` (o de
        :func:`src.pipeline.rebuild_segment_data` si cambió el entorno).
    cfg : Config
        Configuración del experimento (se guarda una copia).
    progress : ProgressCallback | None, optional
        ``progress(fracción, mensaje)``.
    cancel : threading.Event | None, optional
        Evento de cancelación.

    Returns
    -------
    ExperimentResult
        Todos los resultados.

    Raises
    ------
    ValueError
        Si no hay segmentos conservados, algún algoritmo es desconocido o los
        datos bandit del análisis se calcularon con otros parámetros del
        entorno (ver :func:`_check_analysis_config`).
    src.config.CancelledError
        Si ``cancel`` se activa.
    """
    cfg = _check_analysis_config(analysis, cfg)   # copia: cfg del usuario no se modifica
    segment_data = analysis.segment_data
    if not segment_data:
        raise ValueError("El análisis no tiene segmentos conservados: no hay notas sobre las que experimentar.")
    algorithms = _selected_algorithms(cfg)
    n_runs = int(cfg.experiment.n_runs)
    if n_runs < 1:
        raise ValueError(f"El número de corridas debe ser ≥ 1 (recibido {n_runs})")
    budget = int(cfg.env.budget)
    n_seg = len(segment_data)
    gt = analysis.ground_truth or None

    # --- Plan de trabajo (en pulls) para un progreso proporcional al tiempo real.
    chain_units = n_seg * budget
    sweep_runs = effective_sweep_runs(cfg, n_seg)
    main_units = len(algorithms) * n_runs * chain_units
    example_units = len(algorithms) * n_runs * budget
    sweep_units = (sum(len(v) for _, _, v in _sweep_plan(cfg)) * sweep_runs * chain_units
                   if cfg.experiment.run_sweeps else 0)
    do_lambda = bool(cfg.experiment.run_lambda_sweep and gt and cfg.experiment.sweep_lambda)
    lambda_units = (len(cfg.experiment.sweep_lambda) * len(algorithms) * sweep_runs * chain_units
                    if do_lambda else 0)
    work = _WorkProgress(main_units + example_units + sweep_units + lambda_units, progress)
    if do_lambda:
        lambda_text = "sí"
    elif cfg.experiment.run_lambda_sweep:
        lambda_text = "no (sin ground truth)"
    else:
        lambda_text = "no"
    logger.info("Experimento: %d algoritmos × %d corridas × %d segmentos × T=%d pulls (%.2f M pulls); "
                "barridos de sensibilidad: %s; barrido de λ: %s", len(algorithms), n_runs, n_seg, budget,
                main_units / 1e6, "sí" if sweep_units else "no", lambda_text)
    if (sweep_units or lambda_units) and sweep_runs < int(cfg.experiment.sweep_runs):
        logger.info("Pista larga (%d segmentos): los barridos usan %d corridas por valor en lugar de %d",
                    n_seg, sweep_runs, cfg.experiment.sweep_runs)
    est = estimate_experiment_seconds(n_seg, cfg, with_lambda=do_lambda)
    est_text = (f"duración estimada ≈ {_fmt_duration(est['total'])} (principal {_fmt_duration(est['main'])}, "
                f"barridos {_fmt_duration(est['sweeps'])}, λ {_fmt_duration(est['lambda'])})")
    if est["total"] > LONG_EXPERIMENT_S:
        logger.warning("Experimento largo: %s. Para acortarlo: menos corridas (--runs), sin barridos "
                       "(--no-sweeps / --no-lambda) o un fragmento más corto de la pista.", est_text)
    else:
        logger.info("Experimento: %s", est_text)

    matches = match_segments_to_gt([d.segment for d in segment_data], gt, cfg.experiment.onset_tolerance_s) if gt else None
    if gt and matches is not None:
        n_missed = sum(m is None for m in matches)
        n_extra = n_seg - (len(gt) - n_missed)
        logger.info("Emparejamiento con el ground truth (±%.0f ms): %d de %d notas con segmento%s%s",
                    1000 * cfg.experiment.onset_tolerance_s, len(gt) - n_missed, len(gt),
                    f" ({n_missed} sin detectar: cuentan como error)" if n_missed else "",
                    f"; {n_extra} segmentos de más (no cuentan en la precisión por nota, sí en el F1)"
                    if n_extra else "")

    # --- 1. Cadenas principales.
    reward_curves: dict[str, np.ndarray] = {}
    regret_curves: dict[str, np.ndarray] = {}
    optimal_curves: dict[str, np.ndarray] = {}
    choices: dict[str, np.ndarray] = {}
    elapsed: dict[str, np.ndarray] = {}
    accuracy: dict[str, list[AccuracyResult]] = {}
    for algo in algorithms:
        t0 = time.perf_counter()
        rewards = np.empty((n_runs, budget))
        regrets = np.empty((n_runs, budget))
        optimal = np.empty((n_runs, budget))
        algo_choices = np.empty((n_runs, n_seg, 2), dtype=np.int64)
        algo_elapsed = np.empty(n_runs)
        algo_acc: list[AccuracyResult] = []
        for r in range(n_runs):
            _check_cancel(cancel, "experimento comparativo")
            chain = run_chain(algo, segment_data, cfg, run_index=r)
            rewards[r] = chain.rewards.mean(axis=0)                    # media sobre segmentos, por pull
            regrets[r] = np.cumsum(chain.regrets, axis=1).mean(axis=0)  # regret ACUMULADO medio por segmento
            optimal[r] = chain.optimal.mean(axis=0)                    # fracción de segmentos con pull óptimo
            algo_choices[r] = [(a.string_index, a.fret) for a in chain.arms]
            algo_elapsed[r] = chain.elapsed_s
            if gt and matches is not None:
                algo_acc.append(score_arms(chain.arms, matches, gt))
            work.advance(chain_units, f"{ALGO_LABELS[algo]}: corrida {r + 1}/{n_runs}")
        reward_curves[algo], regret_curves[algo], optimal_curves[algo] = rewards, regrets, optimal
        choices[algo], elapsed[algo] = algo_choices, algo_elapsed
        if gt:
            accuracy[algo] = algo_acc
        logger.info("%s: %d corridas en %.1f s", ALGO_LABELS[algo], n_runs, time.perf_counter() - t0)

    # --- 2. Oráculos: miope (argmax μ segmento a segmento) y de cadena (Viterbi).
    oracle = oracle_arms(segment_data, cfg)
    viterbi = oracle_viterbi_arms(segment_data, cfg)
    oracle_accuracy = score_arms(oracle, matches, gt) if gt and matches is not None else None
    viterbi_accuracy = score_arms(viterbi, matches, gt) if gt and matches is not None else None
    logger.info("Oráculos: Σμ de la cadena miope %.3f, de la cadena óptima (Viterbi) %.3f",
                chain_value(oracle, segment_data, cfg), chain_value(viterbi, segment_data, cfg))

    # --- 3. Segmento ejemplo.
    _check_cancel(cancel, "segmento ejemplo")
    example = run_example_segment(segment_data, cfg, progress=work.sub(example_units), cancel=cancel)
    work.advance(example_units, "Segmento ejemplo terminado")

    # --- 4. Barridos.
    sweeps: list[SweepResult] = []
    if cfg.experiment.run_sweeps:
        sweeps = run_sensitivity(segment_data, cfg, progress=work.sub(sweep_units), cancel=cancel)
        work.advance(sweep_units, "Barridos de sensibilidad terminados")
    lambda_sweep: LambdaSweepResult | None = None
    if do_lambda:
        lambda_sweep = run_lambda_sweep(analysis, cfg, progress=work.sub(lambda_units), cancel=cancel)
        work.advance(lambda_units, "Barrido de λ terminado")

    result = ExperimentResult(
        config=cfg, algorithms=algorithms, n_runs=n_runs, budget=budget, n_segments=n_seg,
        reward_curves=reward_curves, regret_curves=regret_curves, optimal_curves=optimal_curves,
        choices=choices, elapsed_s=elapsed, accuracy=accuracy, oracle_arms=oracle,
        oracle_accuracy=oracle_accuracy, gt_notes=gt, example=example, sweeps=sweeps, lambda_sweep=lambda_sweep,
        viterbi_arms=viterbi, viterbi_accuracy=viterbi_accuracy,
    )
    _log_summary(result)
    if progress is not None:
        progress(1.0, "Experimento terminado")
    return result


def _pct(value: object) -> str:
    """Fracción como porcentaje: ``0.95 → "95.0 %"``; None → ``"—"``.

    Examples
    --------
    >>> _pct(0.95), _pct(None)
    ('95.0 %', '—')
    """
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f} %"  # type: ignore[arg-type]


def _fmt_duration(seconds: float) -> str:
    """Duración legible: ``"42 s"``, ``"3.5 min"``.

    Examples
    --------
    >>> _fmt_duration(42.3), _fmt_duration(210.0)
    ('42 s', '3.5 min')
    """
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60.0:.1f} min"


def _log_summary(result: ExperimentResult) -> None:
    """Narra en el log (INFO) la tabla comparativa del experimento.

    Parameters
    ----------
    result : ExperimentResult
        Resultado de :func:`run_experiment`.
    """
    logger.info("Resumen (%d corridas, T=%d, %d segmentos; medias ± desviación entre corridas):",
                result.n_runs, result.budget, result.n_segments)
    for row in result.summary_rows():
        if row["algorithm"] in ("oracle", "oracle_viterbi"):
            logger.info("  %s: precisión de pitch %s, de posición %s (F1 de posición %s)", row["label"],
                        _pct(row["pitch_acc"]), _pct(row["position_acc"]), _pct(row["position_f1"]))
            continue
        acc = ""
        if row["pitch_acc"] is not None:
            acc = (f", precisión pitch {_pct(row['pitch_acc'])} ± {_pct(row['pitch_acc_std'])}, "
                   f"posición {_pct(row['position_acc'])} ± {_pct(row['position_acc_std'])} "
                   f"(F1 {_pct(row['position_f1'])}; {row['n_extra']} segmentos de más)")
        logger.info("  %s: recompensa media %.3f ± %.3f, regret final %.2f ± %.2f por segmento, "
                    "%.1f %% óptimo (último 10 %%)%s, %.2f ms/segmento", row["label"], row["mean_reward"],
                    row["mean_reward_std"], row["final_regret"], row["final_regret_std"], row["optimal_pct"], acc,
                    row["ms_per_segment"])


# ---------------------------------------------------------------------------
# Ejecución paso a paso (pestaña "Ejecución en vivo")
# ---------------------------------------------------------------------------


@dataclass
class PullEvent:
    """Un paso de la ejecución en vivo.

    Attributes
    ----------
    t : int
        Número de pull (1..T).
    arm_index : int
        Brazo jalado.
    arm_label : str
        Etiqueta ``"A-0"``.
    reward, salience, penalty : float
        Recompensa observada y sus componentes.
    frame_index : int
        Frame muestreado (índice dentro del segmento).
    q_before, q_after : float
        Q del brazo antes y después de la actualización.
    decision : Decision
        Explicación del agente.
    cumulative_regret : float
        Regret acumulado hasta este pull.
    is_optimal : bool
        True si el brazo jalado era óptimo.
    text : str
        Frase completa para el log, p. ej. ``"t=37 | UCB1 → D-2 | brazo 2 con
        índice 0.91 = Q 0.70 + bono 0.21 (…) | frame #14: S=0.74 − pen 0.03 =
        r 0.71 | Q 0.62→0.65"``.
    """

    t: int
    arm_index: int
    arm_label: str
    reward: float
    salience: float
    penalty: float
    frame_index: int
    q_before: float
    q_after: float
    decision: Decision
    cumulative_regret: float
    is_optimal: bool
    text: str


def _split_decision_text(algorithm: str, text: str) -> tuple[str, str]:
    """Separa el texto de :class:`Decision` en (encabezado del algoritmo, explicación).

    UCB1 y Softmax empiezan su texto con su nombre (``"UCB1: ..."``,
    ``"Softmax (τ=0.10): ..."``); ese prefijo pasa al encabezado para no
    repetirlo. ε-greedy y el optimista no lo llevan y usan el nombre corto.

    Examples
    --------
    >>> _split_decision_text("ucb1", "UCB1: pull inicial obligatorio del brazo 3")
    ('UCB1', 'pull inicial obligatorio del brazo 3')
    >>> _split_decision_text("egreedy", "u=0.03 < ε=0.10 → explora: brazo 2 al azar")
    ('ε-greedy', 'u=0.03 < ε=0.10 → explora: brazo 2 al azar')
    """
    short = ALGO_SHORT.get(algorithm, algorithm)
    prefix, sep, rest = text.partition(": ")
    if sep and prefix.startswith(short):
        return prefix, rest
    return short, text


class LiveSession:
    """Ejecución pull a pull de un agente en un segmento (la GUI llama a :meth:`step`).

    Cada :meth:`step` hace exactamente lo mismo que una iteración de
    :func:`run_segment`, pero con ``pull_detailed`` para poder explicar el
    pull (frame muestreado, saliencia, penalización) y guardando Q antes y
    después de la actualización.

    Las semillas son las del experimento: entorno
    ``default_rng([cfg.experiment.seed, seed, posición])`` y agente
    ``default_rng([cfg.experiment.seed, seed, posición, 1 + índice_algoritmo])``.
    Por tanto ``LiveSession(..., prev_fret=p, seed=r)`` reproduce pull a pull
    el segmento de la corrida r del experimento (si p es el traste previo de
    esa corrida), y dos algoritmos con la misma semilla ven la misma secuencia
    de frames (números aleatorios comunes).

    Parameters
    ----------
    data : SegmentBanditData
        Segmento.
    algorithm : str
        Algoritmo.
    cfg : Config
        Configuración (hiperparámetros, λ, T). Se guarda una copia.
    prev_fret : int | None
        traste_previo del segmento.
    seed : int
        Semilla de esta sesión (índice de corrida).

    Attributes
    ----------
    agent : BanditAgent
        Agente en ejecución.
    env : BanditEnvironment
        Entorno del segmento.
    budget : int
        Presupuesto T.
    events : list[PullEvent]
        Pulls realizados, en orden.
    cumulative_regret : float
        Regret acumulado hasta ahora.

    Examples
    --------
    >>> from src.segmentation import Segment
    >>> seg = Segment(index=0, start_s=0.0, end_s=1.0, start_sample=0, end_sample=1, rms_db=0.0)
    >>> data = SegmentBanditData(seg, 0, [Arm("A", 0), Arm("A", 1)], np.arange(2), np.array([[0.9, 0.2], [0.7, 0.4]]))
    >>> cfg = Config(); cfg.env.budget = 3
    >>> live = LiveSession(data, "ucb1", cfg, prev_fret=0)
    >>> ev = live.step()
    >>> ev.t, ev.decision.kind, ev.text.startswith("t=1 | UCB1 → ")
    (1, 'init', True)
    """

    def __init__(self, data: SegmentBanditData, algorithm: str, cfg: Config, prev_fret: int | None, seed: int = 0) -> None:
        _algorithm_index(algorithm)
        self.data = data
        self.algorithm = algorithm
        self.cfg = copy.deepcopy(cfg)
        self.prev_fret = prev_fret
        self.seed = int(seed)
        self.budget = int(self.cfg.env.budget)
        self.agent: BanditAgent
        self.env: BanditEnvironment
        self.events: list[PullEvent] = []
        self.cumulative_regret = 0.0
        self.reset()

    @property
    def t(self) -> int:
        """Pulls realizados."""
        return int(self.agent.t)

    @property
    def done(self) -> bool:
        """True si se agotó el presupuesto."""
        return self.t >= self.budget

    def step(self) -> PullEvent:
        """Realiza un pull. Lanza ``RuntimeError`` si ``done``.

        Returns
        -------
        PullEvent
            Detalle del pull con su explicación en español (``text``).

        Raises
        ------
        RuntimeError
            Si ya se gastaron los T pulls (usa :meth:`reset` para empezar de nuevo).
        """
        if self.done:
            raise RuntimeError(f"Presupuesto agotado: ya se hicieron los {self.budget} pulls de este segmento "
                               "(usa reset() para empezar de nuevo).")
        agent, env = self.agent, self.env
        a = agent.select_arm()                       # 1. decidir (y explicar por qué)
        decision = agent.last_decision
        if decision is None:  # todos los agentes la dejan; es solo una salvaguarda
            decision = Decision(a, "exploit", None, "", f"brazo {a}")
        q_before = float(agent.q[a])
        pull = env.pull_detailed(a)                  # 2. el entorno responde (frame j al azar)
        agent.update(a, pull.reward)                 # 3. aprender: Q(a) ← Q(a) + α·(r − Q(a))
        q_after = float(agent.q[a])
        self.cumulative_regret += float(env.gaps[a])  # pseudo-regret: μ* − μ_a
        t = agent.t
        label = self.data.arms[a].label

        head, body = _split_decision_text(self.algorithm, decision.text)
        noise = ""
        if pull.noise != 0.0:
            noise = f" {'+' if pull.noise >= 0 else '−'} ruido {abs(pull.noise):.2f}"
        text = (f"t={t} | {head} → {label} | {body} | frame #{pull.frame_index}: "
                f"S={pull.salience:.2f} − pen {pull.penalty:.2f}{noise} = r {pull.reward:.2f} | "
                f"Q {q_before:.2f}→{q_after:.2f}")
        event = PullEvent(t=t, arm_index=a, arm_label=label, reward=float(pull.reward),
                          salience=float(pull.salience), penalty=float(pull.penalty),
                          frame_index=int(pull.frame_index), q_before=q_before, q_after=q_after,
                          decision=decision, cumulative_regret=self.cumulative_regret,
                          is_optimal=bool(env.is_optimal(a)), text=text)
        self.events.append(event)
        logger.debug("%s", text)
        return event

    def recommended_arm(self) -> Arm:
        """Posición que se recomendaría ahora (``agent.recommend(cfg.agent.recommend)``).

        No consume el generador del agente: se puede consultar en cada paso.

        Returns
        -------
        Arm
            Brazo más jalado hasta ahora (desempate por Q) o argmax Q, según
            ``cfg.agent.recommend``.
        """
        return self.data.arms[self.agent.recommend(self.cfg.agent.recommend)]

    def reset(self) -> None:
        """Reinicia agente, entorno (misma semilla) y eventos."""
        env_rng, agent_rng = segment_rngs(self.cfg.experiment.seed, self.seed, self.data.position, self.algorithm)
        self.env = BanditEnvironment.from_config(self.data, self.prev_fret, self.cfg.env, env_rng)
        self.agent = make_agent(self.algorithm, self.data.n_arms, self.cfg.agent, agent_rng)
        self.events = []
        self.cumulative_regret = 0.0
        logger.info("Ejecución en vivo: %s en el segmento %d (%d brazos, T=%d, traste previo %s, semilla %d)",
                    ALGO_LABELS.get(self.algorithm, self.algorithm), self.data.position, self.data.n_arms,
                    self.budget, self.prev_fret, self.seed)


__all__ = [
    "OUTCOME_MISSED", "OUTCOME_WRONG_PITCH", "OUTCOME_WRONG_POSITION", "OUTCOME_EXACT", "OUTCOME_LABELS",
    "FINAL_FRACTION", "SWEEP_SPECS", "SWEEP_SYMBOLS", "SUMMARY_COLUMNS", "TIMING_COLUMNS", "SegmentRun", "run_segment", "ChainResult",
    "segment_rngs", "run_chain", "oracle_arms", "oracle_viterbi_arms", "chain_value", "CHAIN_TOL",
    "TranscriptionResult", "transcribe", "match_segments_to_gt", "effective_sweep_runs",
    "estimate_experiment_seconds", "PULL_COST_US", "SWEEP_MAX_SEGMENTS", "MIN_SWEEP_RUNS",
    "SEGMENT_DATA_ENV_PARAMS",
    "AccuracyResult", "score_arms", "ExampleSegmentResult", "SweepResult", "LambdaSweepResult",
    "ExperimentResult", "run_example_segment", "run_sensitivity", "run_lambda_sweep", "run_experiment",
    "PullEvent", "LiveSession",
]
