"""Gráficas del sistema (matplotlib orientado a objetos, sin estado global). [CONTRATO: implementar]

Todas las funciones reciben opcionalmente una ``Figure`` existente (la GUI
reutiliza la figura embebida con ``FigureCanvasTkAgg``): si se pasa, se limpia
(``fig.clear()``) y se dibuja dentro; si no, se crea una nueva con
``matplotlib.figure.Figure`` (nunca ``pyplot``, para ser seguro en hilos y
no abrir ventanas). Todas devuelven la figura.

Estilo común: superficie clara, rejilla fina y recesiva, líneas de 2 px,
color FIJO por algoritmo (:data:`src.config.ALGO_COLORS`) más marcador y
estilo de línea como codificación secundaria, título, ejes con unidades y
leyenda siempre presentes. Las bandas de ±1 desviación estándar usan el color
del algoritmo con ~15 % de opacidad.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from src.environment import Arm
from src.experiments import ExperimentResult
from src.pipeline import AnalysisResult

# Tokens de estilo (superficie, tinta, rejilla) compartidos por todas las gráficas.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Colores de estado para el heatmap de aciertos (siempre con etiqueta en la leyenda).
OUTCOME_COLORS: dict[int, str] = {0: "#c3c2b7", 1: "#d03b3b", 2: "#fab219", 3: "#0ca30c"}


def apply_style(fig: Figure) -> None:
    """Aplica el estilo común a todos los ejes de ``fig`` (superficie, rejilla, spines, fuentes)."""
    raise NotImplementedError


def plot_average_reward(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Recompensa promedio por pull (4 algoritmos, banda ±1 std entre corridas)."""
    raise NotImplementedError


def plot_cumulative_regret(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Regret acumulado vs. pulls (media ± std)."""
    raise NotImplementedError


def plot_optimal_action(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """% de selección del brazo óptimo vs. pulls."""
    raise NotImplementedError


def plot_arm_distribution(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Pulls por brazo en el segmento ejemplo (barras agrupadas por algoritmo, brazo óptimo marcado)."""
    raise NotImplementedError


def plot_q_evolution(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Evolución de Q estimados vs. μ real en el segmento ejemplo (2×2, un panel por algoritmo)."""
    raise NotImplementedError


def plot_sensitivity(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Sensibilidad: recompensa final vs ε, Q₀, c y τ (2×2, una curva por parámetro)."""
    raise NotImplementedError


def plot_lambda_effect(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Precisión de posición (y de pitch) vs λ, por algoritmo y oráculo."""
    raise NotImplementedError


def plot_accuracy(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Precisión de pitch y de posición por algoritmo (barras agrupadas ± std, oráculo como referencia)."""
    raise NotImplementedError


def plot_runtime(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Tiempo de ejecución por algoritmo (ms por segmento, media ± std)."""
    raise NotImplementedError


def plot_tab_heatmap(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Heatmap de aciertos/errores por nota GT (columnas) y algoritmo (filas), con la
    posición modal entre corridas; categorías :data:`src.experiments.OUTCOME_LABELS`."""
    raise NotImplementedError


def plot_spectrogram(analysis: AnalysisResult, fig: Figure | None = None, selected: int | None = None) -> Figure:
    """Espectrograma (CQT en dB) con onsets, segmentos descartados y f0 de pYIN superpuestos.

    ``selected``: posición (entre conservados) de un segmento a resaltar.
    """
    raise NotImplementedError


def plot_audio_overview(analysis: AnalysisResult, fig: Figure | None = None, selected: int | None = None) -> Figure:
    """Forma de onda (arriba) + espectrograma con onsets y f0 (abajo), ejes x compartidos.
    Es la figura de la pestaña Audio."""
    raise NotImplementedError


def plot_note_spectrum(
    analysis: AnalysisResult,
    position: int,
    arm: Arm,
    gt_arm: Arm | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Espectro medio del segmento ``position`` con el template armónico del brazo
    elegido (líneas en h·f y, punteadas, en (h−½)·f) y el del ground truth si difiere."""
    raise NotImplementedError


def message_figure(text: str, fig: Figure | None = None) -> Figure:
    """Figura vacía con un mensaje centrado (p. ej. "Requiere ground truth")."""
    raise NotImplementedError


@dataclass(frozen=True)
class PlotSpec:
    """Entrada del catálogo de gráficas comparativas.

    Attributes
    ----------
    key : str
        Nombre de archivo (sin extensión).
    title : str
        Título en español mostrado en la lista de la GUI.
    func : Callable
        Función ``(res, fig=None) -> Figure``.
    description : str
        Qué muestra y cómo interpretarla (tooltip / README).
    """

    key: str
    title: str
    func: Callable[..., Figure]
    description: str


#: Catálogo ordenado de las gráficas comparativas (pestaña Comparación y export).
COMPARISON_PLOTS: list[PlotSpec] = []


def save_figure(fig: Figure, out_dir: str | Path, name: str, formats: tuple[str, ...] = ("png", "pdf"), dpi: int = 150) -> list[Path]:
    """Guarda ``fig`` como ``<out_dir>/<name>.<fmt>`` para cada formato."""
    raise NotImplementedError


def save_all_plots(res: ExperimentResult, analysis: AnalysisResult | None, out_dir: str | Path, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Genera y guarda todas las gráficas comparativas (+ espectrograma si hay ``analysis``)."""
    raise NotImplementedError
