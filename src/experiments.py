"""Experimentos: ejecución de agentes sobre los entornos y comparación. [CONTRATO: implementar]

Conceptos clave
---------------
* **Cadena** (:func:`run_chain`): un agente recorre los segmentos EN ORDEN; en
  cada uno empieza sin conocimiento previo (``reset``), gasta T pulls y
  recomienda una posición. El traste recomendado se vuelve el ``traste_previo``
  del siguiente segmento (penalización de tocabilidad).
* **Números aleatorios comunes**: en la corrida r, el entorno del segmento s
  usa ``np.random.default_rng([seed, r, s])`` para TODOS los algoritmos, y el
  agente usa ``default_rng([seed, r, s, 1 + índice_algoritmo])``. Así las
  diferencias entre algoritmos no se deben a la suerte del muestreo de frames.
* **Regret** (pseudo-regret): Σ_t (μ* − μ_{a_t}), con μ exactos del entorno.
* **Oráculo**: elige argmax μ en cada segmento (con su propia cadena de
  traste previo). Es el techo de precisión alcanzable con esta recompensa.
* **Precisión**: cada nota del ground truth se empareja con el segmento cuyo
  onset está a ≤ ``onset_tolerance_s``; precisión de pitch = fracción de notas
  GT cuyo segmento recibió el MIDI correcto; de posición = (cuerda, traste) correctos.
  Las notas GT sin segmento cuentan como error.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from src.agents import BanditAgent, Decision
from src.config import Config, ProgressCallback
from src.environment import Arm, BanditEnvironment, SegmentBanditData
from src.pipeline import AnalysisResult
from src.segmentation import Segment
from src.synth_dataset import GTNote
from src.tab import TabNote

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
    """Bucle bandit básico: ``reset`` → T × (select_arm, pull, update) → recommend."""
    raise NotImplementedError


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
        Tiempo de pared de la cadena completa.
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


def run_chain(algorithm: str, segment_data: list[SegmentBanditData], cfg: Config, run_index: int = 0) -> ChainResult:
    """Ejecuta ``algorithm`` sobre todos los segmentos en orden (ver encabezado:
    semillas ``[cfg.experiment.seed, run_index, s]``)."""
    raise NotImplementedError


def oracle_arms(segment_data: list[SegmentBanditData], cfg: Config) -> list[Arm]:
    """Cadena del oráculo: en cada segmento argmax μ dado su propio traste previo."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Transcripción (una corrida, para la pestaña Tablatura y la CLI)
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Tablatura producida por un algoritmo en una corrida."""

    algorithm: str
    notes: list[TabNote]
    chain: ChainResult


def transcribe(analysis: AnalysisResult, algorithm: str, cfg: Config, run_index: int = 0) -> TranscriptionResult:
    """Ejecuta una cadena y convierte las posiciones recomendadas en :class:`TabNote`."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Evaluación contra ground truth
# ---------------------------------------------------------------------------


def match_segments_to_gt(segments: list[Segment], gt: list[GTNote], tolerance_s: float) -> list[int | None]:
    """Para cada nota GT, la posición (entre ``segments``, que deben ser los
    conservados) del segmento emparejado o None. Emparejamiento uno a uno,
    greedy por menor distancia de onset."""
    raise NotImplementedError


@dataclass
class AccuracyResult:
    """Precisión de una transcripción.

    Attributes
    ----------
    pitch : float
        Fracción de notas GT con MIDI correcto.
    position : float
        Fracción de notas GT con cuerda y traste correctos.
    outcomes : np.ndarray
        Código ``OUTCOME_*`` por nota GT, forma ``(n_gt,)``.
    """

    pitch: float
    position: float
    outcomes: np.ndarray


def score_arms(arms: list[Arm], matches: list[int | None], gt: list[GTNote]) -> AccuracyResult:
    """Compara las posiciones elegidas (por posición de segmento) con el ground truth."""
    raise NotImplementedError


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
        Precisión del oráculo, ``(V,)``.
    """

    values: list[float]
    position_acc: dict[str, np.ndarray]
    pitch_acc: dict[str, np.ndarray]
    oracle_position_acc: np.ndarray
    oracle_pitch_acc: np.ndarray


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
        Tiempo por corrida, ``(runs,)``.
    accuracy : dict[str, list[AccuracyResult]]
        Por algoritmo, una entrada por corrida (vacío si no hay ground truth).
    oracle_arms : list[Arm]
        Cadena del oráculo.
    oracle_accuracy : AccuracyResult | None
        Precisión del oráculo (None sin ground truth).
    gt_notes : list[GTNote] | None
        Ground truth usado.
    example : ExampleSegmentResult | None
        Segmento ejemplo.
    sweeps : list[SweepResult]
        Barridos de sensibilidad (vacío si se desactivaron).
    lambda_sweep : LambdaSweepResult | None
        Barrido de λ (None si se desactivó o no hay ground truth).
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

    def modal_arms(self, algorithm: str) -> list[Arm]:
        """Posición más frecuente entre corridas para cada segmento."""
        raise NotImplementedError

    def summary_rows(self) -> list[dict[str, object]]:
        """Tabla comparativa final, una fila por algoritmo (+ oráculo si hay GT).

        Claves: ``algorithm``, ``label``, ``mean_reward``, ``mean_reward_std``,
        ``final_regret``, ``final_regret_std``, ``optimal_pct`` (último 10 % de
        pulls), ``optimal_pct_std``, ``pitch_acc``, ``pitch_acc_std``,
        ``position_acc``, ``position_acc_std``, ``ms_per_segment``,
        ``ms_per_segment_std``. Las precisiones son None sin ground truth.
        """
        raise NotImplementedError

    def to_csv(self, path: str) -> str:
        """Escribe :meth:`summary_rows` en CSV y devuelve la ruta."""
        raise NotImplementedError

    def curves_to_csv(self, path: str) -> str:
        """CSV largo con columnas algorithm,pull,reward_mean,reward_std,regret_mean,
        regret_std,optimal_mean,optimal_std."""
        raise NotImplementedError


def run_example_segment(segment_data: list[SegmentBanditData], cfg: Config, position: int | None = None) -> ExampleSegmentResult:
    """Corre todos los algoritmos ``cfg.experiment.n_runs`` veces sobre un único
    segmento (por defecto ``cfg.experiment.example_segment`` o el de más brazos)."""
    raise NotImplementedError


def run_sensitivity(segment_data: list[SegmentBanditData], cfg: Config, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> list[SweepResult]:
    """Barridos ε (egreedy), Q₀ (optimistic), c (ucb1) y τ (softmax) con ``sweep_runs`` corridas."""
    raise NotImplementedError


def run_lambda_sweep(analysis: AnalysisResult, cfg: Config, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> LambdaSweepResult | None:
    """Precisión de pitch/posición vs λ para cada algoritmo y el oráculo (None sin GT)."""
    raise NotImplementedError


def run_experiment(analysis: AnalysisResult, cfg: Config, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> ExperimentResult:
    """Experimento completo: ``n_runs`` cadenas por algoritmo, oráculo, precisión,
    segmento ejemplo y (opcionalmente) barridos. Informa el progreso y respeta
    ``cancel`` entre cadenas (lanza :class:`src.config.CancelledError`)."""
    raise NotImplementedError


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
        Frase completa para el log, p. ej. ``"t=37 | UCB1 eligió D-2 (índice
        0.91) | frame #14 | S=0.74 − pen 0.03 = r 0.71 | Q 0.62 → 0.65"``.
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


class LiveSession:
    """Ejecución pull a pull de un agente en un segmento (la GUI llama a :meth:`step`).

    Parameters
    ----------
    data : SegmentBanditData
        Segmento.
    algorithm : str
        Algoritmo.
    cfg : Config
        Configuración (hiperparámetros, λ, T).
    prev_fret : int | None
        traste_previo del segmento.
    seed : int
        Semilla de esta sesión.

    Attributes
    ----------
    agent : BanditAgent
    env : BanditEnvironment
    budget : int
    events : list[PullEvent]
    """

    def __init__(self, data: SegmentBanditData, algorithm: str, cfg: Config, prev_fret: int | None, seed: int = 0) -> None:
        raise NotImplementedError

    @property
    def t(self) -> int:
        """Pulls realizados."""
        raise NotImplementedError

    @property
    def done(self) -> bool:
        """True si se agotó el presupuesto."""
        raise NotImplementedError

    def step(self) -> PullEvent:
        """Realiza un pull. Lanza ``RuntimeError`` si ``done``."""
        raise NotImplementedError

    def recommended_arm(self) -> Arm:
        """Posición que se recomendaría ahora."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reinicia agente, entorno (misma semilla) y eventos."""
        raise NotImplementedError
