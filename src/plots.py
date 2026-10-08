"""Gráficas del sistema (matplotlib orientado a objetos, sin estado global).

Papel en el pipeline
--------------------
Este módulo es la ÚLTIMA etapa: convierte los resultados del análisis de audio
(:class:`src.pipeline.AnalysisResult`) y del experimento comparativo
(:class:`src.experiments.ExperimentResult`) en figuras para explicar en clase
qué hace cada algoritmo bandit y por qué. No calcula nada del sistema: solo
lee los resultados y los dibuja.

Todas las funciones reciben opcionalmente una ``Figure`` existente (la GUI
reutiliza la figura embebida con ``FigureCanvasTkAgg``): si se pasa, se limpia
(``fig.clear()``) y se dibuja dentro; si no, se crea una nueva con
``matplotlib.figure.Figure`` (nunca ``pyplot``, para ser seguro en hilos y
no abrir ventanas). Todas devuelven la figura.

Sistema visual (común a todas las gráficas)
-------------------------------------------
* **Superficie** clara ``#fcfcfb`` en figura y ejes; tinta de texto
  ``#0b0b0b`` (títulos), secundaria ``#52514e`` (etiquetas de ejes,
  subtítulos) y ``#898781`` para marcas y números de los ejes. Rejilla fina,
  sólida (nunca punteada) y DETRÁS de los datos; sin bordes superior/derecho.
* **Color fijo por algoritmo** (:data:`src.config.ALGO_COLORS`): el color
  sigue al algoritmo, nunca a su posición en un ranking. La identidad nunca
  depende solo del color: cada algoritmo tiene además su marcador
  (:data:`src.config.ALGO_MARKERS`, uno cada ≈ T/10 pulls en las curvas) y su
  estilo de línea (:data:`src.config.ALGO_LINESTYLES`), y la leyenda está
  siempre presente. El texto nunca usa el color de la serie.
* **Incertidumbre**: bandas de ±1 desviación estándar ENTRE CORRIDAS con el
  color del algoritmo al 15 % de opacidad (curvas) o barras de error (barras).
* **Títulos en español y ejes con unidades**; un subtítulo pequeño da el
  contexto (corridas, T, segmentos).
* Nunca doble eje Y: dos magnitudes distintas van en paneles distintos.
* Si falta un dato (sin ground truth, sin barridos...) se dibuja
  :func:`message_figure` con la explicación de qué hacer.

Catálogo
--------
:data:`COMPARISON_PLOTS` lista, en orden, las gráficas comparativas con su
título y una descripción de cómo interpretarlas (la GUI la muestra como
ayuda); :func:`save_all_plots` las exporta todas a PNG y PDF.
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullLocator

from src.config import (
    ALGO_COLORS,
    ALGO_LABELS,
    ALGO_LINESTYLES,
    ALGO_MARKERS,
    ALGORITHMS,
    OPEN_STRING_HZ,
    OPEN_STRING_MIDI,
    STRING_ORDER,
)
from src.environment import Arm, Spectrum, harmonic_template
from src.experiments import (
    OUTCOME_EXACT,
    OUTCOME_LABELS,
    OUTCOME_MISSED,
    OUTCOME_WRONG_PITCH,
    OUTCOME_WRONG_POSITION,
    ExperimentResult,
)
from src.pipeline import AnalysisResult
from src.pitch import midi_to_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokens de estilo
# ---------------------------------------------------------------------------

# Tokens de estilo (superficie, tinta, rejilla) compartidos por todas las gráficas.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Colores de estado para el heatmap de aciertos (siempre con etiqueta en la leyenda).
OUTCOME_COLORS: dict[int, str] = {0: "#c3c2b7", 1: "#d03b3b", 2: "#fab219", 3: "#0ca30c"}

#: Gris neutro (un paso fuera de la superficie) para resaltar zonas sin competir con los datos.
HIGHLIGHT = "#f0efec"

# Colores de "rol" para las figuras de audio, donde no aparece ningún algoritmo.
# Salen de la misma paleta categórica del proyecto, pero de huecos que NO usan
# los algoritmos (así un color nunca sugiere un algoritmo que no está).
#: Forma de onda: paso oscuro del azul secuencial (el mismo tono que el espectrograma).
SIGNAL_COLOR = "#1c5cab"
#: Espectro medio de una nota: tinta neutra, para que los peines de color destaquen sobre él.
SPECTRUM_COLOR = "#3b3a37"
#: Trayectoria f0 de pYIN sobre el espectrograma.
F0_COLOR = "#e34948"
#: Template armónico del brazo elegido (espectro de una nota).
ARM_COLOR = "#4a3aa7"
#: Template armónico del ground truth (espectro de una nota).
GT_COLOR = "#008300"
#: Onsets detectados (líneas verticales finas).
ONSET_COLOR = "#3b3a37"
#: Sombreado de los segmentos descartados.
DISCARDED_COLOR = "#898781"

#: Mapa de color secuencial de un solo tono (superficie → azul oscuro) para el
#: espectrograma: claro = poca energía, oscuro = mucha.
#: El primer tercio de la escala (el ruido de fondo, −70…−45 dB) queda casi
#: blanco para que los armónicos destaquen.
SPECTRO_CMAP = LinearSegmentedColormap.from_list(
    "smartuner_blues",
    [(0.0, SURFACE), (0.35, "#e3eefc"), (0.55, "#9ec5f4"), (0.75, "#3987e5"), (0.9, "#1c5cab"), (1.0, "#0d366b")],
)

# Tamaños (puntos tipográficos) y marcas.
TITLE_SIZE = 13.0
SUBTITLE_SIZE = 9.5
PANEL_TITLE_SIZE = 10.5
LABEL_SIZE = 10.0
TICK_SIZE = 9.0
LEGEND_SIZE = 9.0
ANNOTATION_SIZE = 8.0
LINE_WIDTH = 2.0
BAND_ALPHA = 0.15
MARKER_SIZE = 6.5
GRID_WIDTH = 0.7

#: Tamaño por defecto (pulgadas) de una gráfica simple y de una cuadrícula 2×2.
FIGSIZE_SIMPLE: tuple[float, float] = (9.0, 5.5)
FIGSIZE_GRID: tuple[float, float] = (11.0, 7.5)
FIGSIZE_WIDE: tuple[float, float] = (11.0, 6.5)

#: Fracción del hueco de cada grupo de barras que ocupan las barras (el resto es aire).
GROUP_WIDTH = 0.72
#: Fracción del sub-hueco de cada barra que se rellena (deja un pequeño hueco entre barras vecinas).
BAR_FILL = 0.78
#: Grosor máximo de una barra (pulgadas; ≈ 24 px a 100 ppp): barras delgadas, el resto es aire.
BAR_MAX_IN = 0.24
#: Fondo de las etiquetas de texto sobre los datos (la rejilla y las líneas no las atraviesan).
LABEL_BOX: dict[str, object] = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none")
#: Máximo de brazos dibujados individualmente en las gráficas de un segmento.
MAX_ARM_GROUPS = 10
MAX_Q_ARMS = 7
#: Frecuencia máxima mostrada en el espectrograma (Hz): cubre el 5.º armónico de G-12 (980 Hz).
SPECTRO_FMAX_HZ = 1100.0
#: Rango dinámico del espectrograma (dB bajo el máximo).
SPECTRO_DB_RANGE = 70.0
#: Columnas máximas del espectrograma (más que píxeles no aporta nada y hace lenta la GUI).
SPECTRO_MAX_COLUMNS = 1600
#: Columnas de la envolvente de la onda: con más de 4× este número de muestras se dibuja el
#: mínimo y el máximo de cada bloque en vez de cada muestra.
WAVEFORM_MAX_POINTS = 4000
#: Densidad de onsets (líneas por pulgada) a partir de la cual se aclaran sus líneas.
ONSETS_PER_INCH = 6.0

#: Notas marcadas en el eje de frecuencia: las cuatro cuerdas al aire y las octavas de G.
NOTE_TICKS: tuple[tuple[str, float], ...] = (
    ("E1", OPEN_STRING_HZ["E"]),
    ("A1", OPEN_STRING_HZ["A"]),
    ("D2", OPEN_STRING_HZ["D"]),
    ("G2", OPEN_STRING_HZ["G"]),
    ("G3", 2 * OPEN_STRING_HZ["G"]),
    ("G4", 4 * OPEN_STRING_HZ["G"]),
    ("G5", 8 * OPEN_STRING_HZ["G"]),
)

#: Mensajes cuando falta un dato (se muestran con :func:`message_figure`).
MSG_NO_RESULTS = "Sin resultados: ejecuta el experimento comparativo para ver esta gráfica."
MSG_NO_GT = ("Requiere ground truth: genera o carga un MP3 sintético con su .gt.json "
             "(el archivo de notas reales junto al audio).")
MSG_NO_EXAMPLE = ("Sin segmento de ejemplo: el experimento no guardó los pulls por brazo "
                  "ni la evolución de Q de un segmento.")
MSG_NO_SWEEPS = ("Sin barridos de sensibilidad: activa «Barridos de sensibilidad» en la "
                 "configuración del experimento y vuelve a ejecutarlo.")
MSG_NO_LAMBDA = ("Sin barrido de λ: activa «Barrido de λ» en la configuración del "
                 "experimento y vuelve a ejecutarlo.")

#: Orden de los paneles de sensibilidad y textos de cada hiperparámetro.
_SWEEP_INFO: dict[str, tuple[str, str]] = {
    "agent.epsilon": ("ε", "ε (probabilidad de explorar)"),
    "agent.q0": ("Q₀", "Q₀ (valor inicial optimista)"),
    "agent.ucb_c": ("c", "c (peso del bono de exploración)"),
    "agent.tau": ("τ", "τ (temperatura de Boltzmann)"),
}


# ---------------------------------------------------------------------------
# Estilo y utilidades comunes
# ---------------------------------------------------------------------------


def _style_axes(ax: Axes, grid: str | None = "both") -> None:
    """Aplica el estilo común a un eje.

    Parameters
    ----------
    ax : Axes
        Eje a estilizar.
    grid : {"both", "x", "y"} or None
        Qué rejilla dibujar (None = ninguna, p. ej. en heatmaps y espectrogramas).
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", colors=TEXT_MUTED, labelcolor=TEXT_MUTED,
                   labelsize=TICK_SIZE, length=3, width=0.8)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(TEXT_SECONDARY)
        label.set_fontsize(LABEL_SIZE)
    # Rejilla: hairline sólida, recesiva y DETRÁS de los datos.
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid is not None:
        ax.grid(True, axis=grid, color=GRID, linewidth=GRID_WIDTH, linestyle="-")
    legend = ax.get_legend()
    if legend is not None:
        _style_legend(legend)


def _style_legend(legend: object) -> None:
    """Leyenda sin marco pesado y con texto en tinta primaria (nunca el color de la serie)."""
    for text in legend.get_texts():  # type: ignore[attr-defined]
        text.set_color(TEXT_PRIMARY)
    title = legend.get_title()  # type: ignore[attr-defined]
    title.set_color(TEXT_SECONDARY)
    title.set_fontsize(LEGEND_SIZE)


def apply_style(fig: Figure, grid: str | None = "both") -> None:
    """Aplica el estilo común a todos los ejes de ``fig`` (superficie, rejilla, spines, fuentes).

    Parameters
    ----------
    fig : Figure
        Figura cuyos ejes se estilizan (los ejes de barras de color solo
        reciben el estilo de texto y marcas).
    grid : {"both", "x", "y"} or None, optional
        Rejilla de los ejes de datos (``"both"`` por defecto). Cada gráfica
        puede ajustarla después eje por eje.

    Notes
    -----
    El estilo se aplica eje por eje (no con ``rcParams``) para no tocar el
    estado global de matplotlib, que comparten la GUI y la CLI.
    """
    fig.set_facecolor(SURFACE)
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None:
            ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_MUTED, labelsize=TICK_SIZE)
            ax.yaxis.label.set_color(TEXT_SECONDARY)
            ax.xaxis.label.set_color(TEXT_SECONDARY)
            continue
        if not ax.axison:
            continue
        _style_axes(ax, grid)
    for legend in fig.legends:
        _style_legend(legend)


def _prepare_figure(fig: Figure | None, figsize: tuple[float, float]) -> Figure:
    """Devuelve una figura limpia con superficie y *constrained layout*.

    Si ``fig`` es None se crea una nueva de tamaño ``figsize`` (pulgadas); si
    no, se limpia y se reutiliza CON SU TAMAÑO (la GUI controla el lienzo).
    """
    if fig is None:
        fig = Figure(figsize=figsize, facecolor=SURFACE)
    else:
        fig.clear()
        fig.set_facecolor(SURFACE)
    fig.set_layout_engine("constrained", h_pad=0.06, w_pad=0.06)
    return fig


def _titles(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """Título (alineado a la izquierda) y subtítulo pequeño de contexto.

    El título es el ``suptitle`` de la figura, que *constrained layout* deja
    siempre fuera de los ejes. El subtítulo se ancla a las líneas en blanco
    que se reservan debajo del título, así nunca se superpone con los datos.
    Si el subtítulo no cabe en el ancho de la figura se parte en varias líneas.
    """
    if subtitle:
        # ≈ caracteres por línea: ancho en puntos / ancho medio de un carácter (≈ 0.55 em).
        max_chars = max(30, int(fig.get_size_inches()[0] * 72.0 / (SUBTITLE_SIZE * 0.55)))
        subtitle = textwrap.fill(subtitle, width=max_chars, break_long_words=False)
    n_lines = subtitle.count("\n") + 1 if subtitle else 0
    sup = fig.suptitle(title + "\n" * n_lines, x=0.012, ha="left", fontsize=TITLE_SIZE,
                       fontweight="bold", color=TEXT_PRIMARY, linespacing=1.35)
    if subtitle:
        fig.add_artist(Annotation(subtitle, xy=(0.0, 0.0), xycoords=sup, xytext=(0.5, 2.0),
                                  textcoords="offset points", ha="left", va="bottom",
                                  fontsize=SUBTITLE_SIZE, color=TEXT_SECONDARY))


def _axis_labels(ax: Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Etiquetas de los ejes en tinta secundaria."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=LABEL_SIZE)


def _panel_title(ax: Axes, text: str) -> None:
    """Título de un panel (en cuadrículas), alineado a la izquierda."""
    ax.set_title(text, loc="left", fontsize=PANEL_TITLE_SIZE, color=TEXT_PRIMARY, fontweight="bold", pad=6)


def _legend(target: Axes | Figure, handles: Sequence[object], labels: Sequence[str] | None = None, **kwargs: object) -> object:
    """Leyenda sin marco con el estilo común (en un eje o en la figura)."""
    if labels is None:
        labels = [h.get_label() for h in handles]  # type: ignore[attr-defined]
    opts: dict[str, object] = dict(frameon=False, fontsize=LEGEND_SIZE, handlelength=3.2, borderaxespad=0.6,
                                   labelcolor=TEXT_PRIMARY)
    opts.update(kwargs)
    legend = target.legend(list(handles), list(labels), **opts)  # type: ignore[arg-type]
    _style_legend(legend)
    return legend


def _percent_axis(ax: Axes, axis: str = "y", top: float = 100.0) -> None:
    """Eje en porcentaje (0–100 %) con ticks cada 20 %."""
    fmt = FuncFormatter(lambda v, _pos: f"{v:.0f} %")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    target.set_major_formatter(fmt)
    if axis == "y":
        ax.set_ylim(0, top)
    else:
        ax.set_xlim(0, top)


def _fmt_value(v: float) -> str:
    """Número compacto para etiquetas de ticks (0.01, 1.414, 10)."""
    return f"{v:g}"


def _relative_luminance(color: str) -> float:
    """Luminancia relativa WCAG de un color (0 = negro, 1 = blanco)."""
    rgb = np.array(to_rgb(color))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def _ink_on(color: str) -> str:
    """Texto blanco o negro, el que más contraste tenga sobre ``color``."""
    lum = _relative_luminance(color)
    contrast_black = (lum + 0.05) / 0.05
    contrast_white = 1.05 / (lum + 0.05)
    return TEXT_PRIMARY if contrast_black >= contrast_white else "#ffffff"


def _neutral_ramp(n: int, dark: str = "#2f2e2b", light: str = "#b9b7ad") -> list[str]:
    """``n`` grises de oscuro a claro (paleta neutra secuencial)."""
    if n <= 0:
        return []
    if n == 1:
        return [dark]
    a, b = np.array(to_rgb(dark)), np.array(to_rgb(light))
    return [to_hex(a + (b - a) * k / (n - 1)) for k in range(n)]


def _grid_shape(n: int) -> tuple[int, int]:
    """Filas y columnas para ``n`` paneles (1 → 1×1, 2 → 1×2, 3–4 → 2×2...)."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    cols = 2 if n <= 4 else 3
    return int(np.ceil(n / cols)), cols


def _panel_axes(fig: Figure, n: int, sharex: bool = False) -> list[Axes]:
    """Crea ``n`` paneles en cuadrícula y oculta los sobrantes."""
    rows, cols = _grid_shape(n)
    axes = fig.subplots(rows, cols, sharex=sharex, squeeze=False).ravel().tolist()
    for ax in axes[n:]:
        ax.set_visible(False)
        ax.set_axis_off()
    return axes[:n]


# ---------------------------------------------------------------------------
# Algoritmos: orden, estilo y curvas
# ---------------------------------------------------------------------------


def _ordered_algorithms(res: ExperimentResult, available: Mapping[str, object] | None = None) -> list[str]:
    """Algoritmos del experimento en el orden fijo de :data:`src.config.ALGORITHMS`.

    Parameters
    ----------
    res : ExperimentResult
        Resultado del experimento.
    available : Mapping, optional
        Si se da, solo se devuelven los algoritmos que son clave de este diccionario.
    """
    algs = list(res.algorithms) if res.algorithms else list(ALGORITHMS)
    if available is not None:
        algs = [a for a in algs if a in available]
    rank = {a: i for i, a in enumerate(ALGORITHMS)}
    return sorted(algs, key=lambda a: rank.get(a, len(rank)))


def _algo_color(alg: str) -> str:
    """Color fijo del algoritmo (gris secundario si no está en la paleta)."""
    return ALGO_COLORS.get(alg, TEXT_SECONDARY)


def _algo_label(alg: str) -> str:
    """Nombre legible del algoritmo en español."""
    return ALGO_LABELS.get(alg, alg)


def _algo_handle(alg: str, label: str | None = None, line: bool = True) -> Line2D:
    """Muestra de leyenda de un algoritmo: color + estilo de línea + marcador."""
    return Line2D([], [], color=_algo_color(alg), linewidth=LINE_WIDTH if line else 0,
                  linestyle=ALGO_LINESTYLES.get(alg, "-") if line else "none",
                  marker=ALGO_MARKERS.get(alg, "o"), markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                  markeredgewidth=1.0, solid_capstyle="round", dash_capstyle="round",
                  label=label if label is not None else _algo_label(alg))


def _markevery(n_points: int, slot: int, n_slots: int) -> slice:
    """Marcadores cada ≈ n/10 puntos, desfasados por algoritmo para que no se tapen."""
    step = max(1, int(round(n_points / 10)))
    offset = int(step * (slot + 0.5) / max(n_slots, 1)) if step > 1 else 0
    return slice(offset, None, step)


def _mean_std(curves: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación estándar entre corridas (std = 0 con una sola corrida)."""
    arr = np.asarray(curves, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :] if axis == 0 else arr[:, None]
    return arr.mean(axis=axis), arr.std(axis=axis)


def _plot_algo_curve(ax: Axes, alg: str, x: np.ndarray, mean: np.ndarray, std: np.ndarray | None,
                     slot: int, n_slots: int, markevery: slice | None = None) -> None:
    """Curva media de un algoritmo con su banda ±1 std (15 % de opacidad) y marcadores."""
    color = _algo_color(alg)
    if std is not None and np.any(std > 0):
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, linewidth=LINE_WIDTH, linestyle=ALGO_LINESTYLES.get(alg, "-"),
            marker=ALGO_MARKERS.get(alg, "o"), markersize=MARKER_SIZE, markeredgecolor=SURFACE,
            markeredgewidth=1.0, markevery=markevery if markevery is not None else _markevery(len(x), slot, n_slots),
            solid_capstyle="round", dash_capstyle="round", solid_joinstyle="round", zorder=3)


def _context(res: ExperimentResult) -> str:
    """Subtítulo con las dimensiones del experimento."""
    return (f"{res.n_runs} corridas · T = {res.budget} pulls por segmento · "
            f"{res.n_segments} segmentos · λ = {_fmt_value(res.config.env.lam)}")


def _pulls_axis(n: int) -> np.ndarray:
    """Eje de pulls t = 1..T."""
    return np.arange(1, n + 1)


# ---------------------------------------------------------------------------
# Datos faltantes
# ---------------------------------------------------------------------------


def missing_data_message(key: str, res: ExperimentResult) -> str | None:
    """Mensaje en español si a ``res`` le falta lo necesario para la gráfica ``key``.

    Parameters
    ----------
    key : str
        Clave de :data:`COMPARISON_PLOTS` (``"accuracy"``, ``"sensitivity"``...).
    res : ExperimentResult
        Resultado del experimento.

    Returns
    -------
    str | None
        None si la gráfica se puede dibujar; si no, qué falta y cómo obtenerlo.
        La GUI lo usa para avisar y :func:`save_all_plots` para omitir la gráfica.
    """
    if key in ("average_reward", "cumulative_regret", "optimal_action", "runtime"):
        curves = {"average_reward": res.reward_curves, "cumulative_regret": res.regret_curves,
                  "optimal_action": res.optimal_curves, "runtime": res.elapsed_s}[key]
        return None if curves else MSG_NO_RESULTS
    if key in ("arm_distribution", "q_evolution"):
        return None if res.example is not None else MSG_NO_EXAMPLE
    if key == "sensitivity":
        return None if res.sweeps else MSG_NO_SWEEPS
    if key == "lambda_effect":
        if res.lambda_sweep is not None:
            return None
        return MSG_NO_GT if not res.gt_notes else MSG_NO_LAMBDA
    if key in ("accuracy", "tab_heatmap"):
        has_acc = any(len(v) > 0 for v in res.accuracy.values())
        if not res.gt_notes or not has_acc:
            return MSG_NO_GT
        if key == "tab_heatmap" and not res.choices:
            return MSG_NO_RESULTS
        return None
    return None


def message_figure(text: str, fig: Figure | None = None) -> Figure:
    """Figura vacía con un mensaje centrado (p. ej. "Requiere ground truth").

    Parameters
    ----------
    text : str
        Mensaje en español: qué falta y qué hacer para obtenerlo.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        La figura con un único eje invisible que contiene el texto.

    Notes
    -----
    El texto se parte en líneas de ≈ 64 caracteres para que siempre quepa.

    Examples
    --------
    >>> fig = message_figure("Requiere ground truth")
    >>> fig.axes[0].texts[0].get_text()
    'Requiere ground truth'
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    ax = fig.add_subplot()
    ax.set_axis_off()
    # Partido a mano (≈ 64 caracteres por línea): ``wrap=True`` solo corta al llegar al borde.
    wrapped = "\n".join(textwrap.fill(line, width=64) for line in text.splitlines()) if text else ""
    ax.text(0.5, 0.5, wrapped, ha="center", va="center", multialignment="center", fontsize=12,
            color=TEXT_SECONDARY, transform=ax.transAxes, linespacing=1.5)
    return fig


# ---------------------------------------------------------------------------
# Curvas de aprendizaje (promedio sobre segmentos y corridas)
# ---------------------------------------------------------------------------


def plot_average_reward(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Recompensa promedio por pull (4 algoritmos, banda ±1 std entre corridas).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``reward_curves[alg]`` (forma ``(corridas, T)``: recompensa del
        pull t promediada sobre los segmentos).
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Una curva por algoritmo (media entre corridas) con su banda ±1 std.

    Notes
    -----
    Es la gráfica clásica de Sutton & Barto (fig. 2.2): un buen algoritmo sube
    rápido (explora poco tiempo) y se estabiliza alto (explota el mejor
    brazo). La asíntota de ε-greedy queda por debajo de μ* porque sigue
    explorando una fracción ε de los pulls.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("average_reward", res)
    if msg:
        return message_figure(msg, fig)
    ax = fig.add_subplot()
    algs = _ordered_algorithms(res, res.reward_curves)
    for i, alg in enumerate(algs):
        mean, std = _mean_std(res.reward_curves[alg])
        _plot_algo_curve(ax, alg, _pulls_axis(mean.size), mean, std, i, len(algs))
    n = max(np.asarray(res.reward_curves[a]).shape[-1] for a in algs)
    ax.set_xlim(1, n)
    _axis_labels(ax, "Pull t (interacciones con el segmento)", "Recompensa media r")
    _titles(fig, "Recompensa media por pull", _context(res) + " · banda = ±1 std entre corridas")
    _legend(ax, [_algo_handle(a) for a in algs], loc="lower right")
    apply_style(fig)
    return fig


def plot_cumulative_regret(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Regret acumulado vs. pulls (media ± std).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``regret_curves[alg]``: Σ_{s≤t} (μ* − μ_{a_s}) promediado sobre
        segmentos, forma ``(corridas, T)``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Una curva por algoritmo; la leyenda incluye el valor final medio.

    Notes
    -----
    El pseudo-regret mide lo que se PIERDE por no jugar siempre el brazo
    óptimo. Si la curva se aplana, el agente dejó de equivocarse; si crece
    como una recta (ε-greedy con ε fijo), el regret es lineal en T; UCB1
    garantiza un crecimiento logarítmico.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("cumulative_regret", res)
    if msg:
        return message_figure(msg, fig)
    ax = fig.add_subplot()
    algs = _ordered_algorithms(res, res.regret_curves)
    handles = []
    n = 1
    for i, alg in enumerate(algs):
        mean, std = _mean_std(res.regret_curves[alg])
        n = max(n, mean.size)
        _plot_algo_curve(ax, alg, _pulls_axis(mean.size), mean, std, i, len(algs))
        handles.append(_algo_handle(alg, f"{_algo_label(alg)} — {mean[-1]:.2f} al final"))
    ax.set_xlim(1, n)
    ax.set_ylim(bottom=0)
    _axis_labels(ax, "Pull t (interacciones con el segmento)", "Regret acumulado Σ(μ* − μₐ)")
    _titles(fig, "Regret acumulado", _context(res) + " · banda = ±1 std entre corridas")
    _legend(ax, handles, loc="upper left")
    apply_style(fig)
    return fig


def plot_optimal_action(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """% de selección del brazo óptimo vs. pulls.

    Parameters
    ----------
    res : ExperimentResult
        Usa ``optimal_curves[alg]``: fracción de segmentos en que el pull t
        fue a un brazo óptimo, forma ``(corridas, T)``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Curvas en porcentaje (0–100 %); la leyenda incluye la media del
        último 10 % de pulls.

    Notes
    -----
    Complementa a la recompensa: dos brazos con μ casi igual (A-0 y E-5 sin
    penalización) dan casi la misma recompensa, pero solo uno es óptimo.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("optimal_action", res)
    if msg:
        return message_figure(msg, fig)
    ax = fig.add_subplot()
    algs = _ordered_algorithms(res, res.optimal_curves)
    handles = []
    n = 1
    for i, alg in enumerate(algs):
        mean, std = _mean_std(100.0 * np.asarray(res.optimal_curves[alg], dtype=float))
        n = max(n, mean.size)
        tail = mean[-max(1, mean.size // 10):].mean()
        _plot_algo_curve(ax, alg, _pulls_axis(mean.size), mean, std, i, len(algs))
        handles.append(_algo_handle(alg, f"{_algo_label(alg)} — {tail:.0f} % al final"))
    ax.set_xlim(1, n)
    _percent_axis(ax, "y", top=102)
    _axis_labels(ax, "Pull t (interacciones con el segmento)", "% de pulls al brazo óptimo")
    _titles(fig, "Selección del brazo óptimo",
            _context(res) + " · «al final» = media del último 10 % de pulls")
    _legend(ax, handles, loc="lower right")
    apply_style(fig)
    return fig


# ---------------------------------------------------------------------------
# Un segmento: pulls por brazo y evolución de Q
# ---------------------------------------------------------------------------


def _ranked_arms(true_means: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices de brazos por μ descendente: (mostrados, resto agrupado)."""
    order = np.argsort(-np.asarray(true_means, dtype=float), kind="stable")
    if order.size > limit + 1:  # agrupar uno solo no ahorra nada
        return order[:limit], order[limit:]
    return order, order[:0]


def _bar_limit(fig: Figure, data_span: float, axes_fraction: float = 0.8, vertical: bool = True) -> float:
    """Grosor máximo de barra en unidades de datos para no pasar de :data:`BAR_MAX_IN` pulgadas.

    Parameters
    ----------
    fig : Figure
        Figura (su tamaño determina cuántas pulgadas mide una unidad de datos).
    data_span : float
        Rango de datos del eje de las categorías (``xlim`` o ``ylim``).
    axes_fraction : float
        Fracción aproximada de la figura que ocupa el eje.
    vertical : bool
        True para barras verticales (se usa el ancho de la figura).
    """
    w, h = fig.get_size_inches()
    axis_in = (w if vertical else h) * axes_fraction
    return BAR_MAX_IN * data_span / max(axis_in, 1e-6)


def _bar_offsets(n_series: int, max_width: float | None = None) -> tuple[np.ndarray, float]:
    """Desplazamiento de cada serie dentro de su grupo y ancho de barra.

    Las barras son delgadas: ocupan :data:`BAR_FILL` de su hueco (queda un
    pequeño hueco entre barras vecinas) y nunca superan ``max_width``; si lo
    harían, el grupo entero se estrecha y el resto del hueco queda como aire.
    """
    slot = GROUP_WIDTH / max(n_series, 1)
    if max_width is not None:
        slot = min(slot, max_width / BAR_FILL)
    offsets = -slot * n_series / 2 + slot * (np.arange(n_series) + 0.5)
    return offsets, slot * BAR_FILL


def _bar_handle(alg: str, label: str | None = None) -> tuple[Patch, Line2D]:
    """Muestra de leyenda para barras: rectángulo del color del algoritmo + su marcador encima."""
    patch = Patch(facecolor=_algo_color(alg), edgecolor="none", label=label or _algo_label(alg))
    marker = Line2D([], [], linestyle="none", marker=ALGO_MARKERS.get(alg, "o"), markersize=5.5,
                    color=_algo_color(alg), markeredgecolor=SURFACE, markeredgewidth=1.0)
    return patch, marker


def _fmt_mu(v: float) -> str:
    """μ con dos decimales sin el feo «-0.00»."""
    text = f"{v:.2f}"
    return "0.00" if text == "-0.00" else text


def plot_arm_distribution(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Pulls por brazo en el segmento ejemplo (barras agrupadas por algoritmo, brazo óptimo marcado).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``example`` (:class:`src.experiments.ExampleSegmentResult`):
        ``counts[alg]`` de forma ``(corridas, K)``, ``true_means`` y
        ``optimal_arms``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Eje x = brazos ordenados por μ (de mayor a menor, etiqueta
        ``"A-0\\nμ=0.82"``); una barra por algoritmo (media de pulls entre
        corridas, error = ±1 std). El brazo óptimo lleva fondo gris y la
        etiqueta «óptimo». Con más de :data:`MAX_ARM_GROUPS` brazos, los de
        menor μ se agrupan en «Otros».

    Notes
    -----
    Muestra CÓMO reparte cada algoritmo su presupuesto: un buen agente
    concentra los pulls en el óptimo y gasta pocos en brazos claramente
    malos; si gasta muchos en el segundo mejor es que la brecha Δ es pequeña.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("arm_distribution", res)
    if msg:
        return message_figure(msg, fig)
    ex = res.example
    assert ex is not None
    mu = np.asarray(ex.true_means, dtype=float)
    shown, rest = _ranked_arms(mu, MAX_ARM_GROUPS)
    optimal = {int(a) for a in np.atleast_1d(ex.optimal_arms)}
    labels = [f"{ex.arm_labels[a]}\nμ={_fmt_mu(mu[a])}" for a in shown]
    if rest.size:
        labels.append(f"Otros\n({rest.size} brazos)")
    algs = _ordered_algorithms(res, ex.counts)
    x = np.arange(len(labels), dtype=float)
    offsets, width = _bar_offsets(len(algs), _bar_limit(fig, len(labels) + 0.2))

    ax = fig.add_subplot()
    n_runs = 0
    budget = 0
    for i, alg in enumerate(algs):
        counts = np.atleast_2d(np.asarray(ex.counts[alg], dtype=float))  # (corridas, K)
        n_runs = max(n_runs, counts.shape[0])
        budget = max(budget, int(round(counts.sum(axis=1).max())))
        values = counts[:, shown]
        if rest.size:
            values = np.column_stack([values, counts[:, rest].sum(axis=1)])
        mean, std = values.mean(axis=0), values.std(axis=0)
        xi = x + offsets[i]
        ax.bar(xi, mean, width=width, color=_algo_color(alg), linewidth=0, zorder=3)
        ax.errorbar(xi, mean, yerr=std, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=2.0,
                    capthick=1.0, zorder=4)
        # El marcador del algoritmo en la punta es la codificación secundaria de la barra.
        ax.plot(xi, mean, linestyle="none", marker=ALGO_MARKERS.get(alg, "o"), markersize=5.5,
                color=_algo_color(alg), markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)

    # Brazo óptimo: fondo gris neutro detrás del grupo y etiqueta en el borde superior.
    top = ax.get_xaxis_transform()
    ax.set_xticks(x, labels)
    optimal_cols = [j for j, a in enumerate(shown) if int(a) in optimal]
    for j in optimal_cols:
        ax.axvspan(j - 0.48, j + 0.48, color=HIGHLIGHT, zorder=0, linewidth=0)
        ax.text(j, 1.0, "óptimo", transform=top, ha="center", va="bottom", fontsize=ANNOTATION_SIZE,
                color=TEXT_SECONDARY, fontweight="bold")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(bottom=0)
    _axis_labels(ax, "Brazo (cuerda-traste) ordenado por μ real", "Pulls al brazo (media entre corridas)")
    prev = "ninguno" if ex.prev_fret is None else str(ex.prev_fret)
    _titles(fig, f"Reparto de pulls entre brazos · segmento {ex.position}",
            f"{n_runs} corridas · T = {budget} pulls · traste previo = {prev} · {mu.size} brazos · "
            "barras de error = ±1 std")
    handles: list[object] = [_bar_handle(a) for a in algs]
    labels_leg = [_algo_label(a) for a in algs] + ["Brazo óptimo (argmax μ)"]
    handles.append(Patch(facecolor=HIGHLIGHT, edgecolor=AXIS, linewidth=0.6))
    _legend(fig, handles, labels_leg, loc="outside lower center", ncol=len(handles), handlelength=1.6,
            handler_map={tuple: HandlerTuple(ndivide=1)})
    apply_style(fig, grid="y")
    # Después del estilo común (que pinta todas las etiquetas en gris): el óptimo en negrita y tinta.
    tick_labels = ax.get_xticklabels()
    for j in optimal_cols:
        tick_labels[j].set_fontweight("bold")
        tick_labels[j].set_color(TEXT_PRIMARY)
    return fig


def _q_panel_title(alg: str, res: ExperimentResult) -> str:
    """Nombre del algoritmo con sus hiperparámetros principales."""
    a = res.config.agent
    params = {
        "egreedy": f"ε = {_fmt_value(a.epsilon)}",
        "optimistic": f"Q₀ = {_fmt_value(a.q0)}, α = {_fmt_value(a.optimistic_alpha)}",
        "ucb1": f"c = {_fmt_value(a.ucb_c)}",
        "softmax": f"τ = {_fmt_value(a.tau)}",
    }.get(alg)
    return f"{_algo_label(alg)} · {params}" if params else _algo_label(alg)


def plot_q_evolution(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Evolución de Q estimados vs. μ real en el segmento ejemplo (2×2, un panel por algoritmo).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``example.q_mean[alg]`` y ``example.q_std[alg]`` (forma ``(T, K)``),
        ``true_means`` y ``optimal_arms``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Un panel por algoritmo con una línea por brazo (Q medio entre
        corridas) y su μ real como línea horizontal punteada del mismo color.
        El brazo óptimo va en el color del algoritmo (con banda ±1 std); los
        demás en grises (más oscuro = mayor μ). Se muestran como máximo
        :data:`MAX_Q_ARMS` brazos (los de mayor μ).

    Notes
    -----
    Muestra la regla incremental Q ← Q + α(r − Q) en acción: cada Q converge a
    su μ solo si el brazo se jala bastante. Los brazos que el agente abandona
    se quedan "congelados" lejos de μ. En el optimista todas las Q empiezan
    en Q₀ y bajan; en Softmax/ε-greedy empiezan en 0.
    """
    fig = _prepare_figure(fig, FIGSIZE_GRID)
    msg = missing_data_message("q_evolution", res)
    if msg:
        return message_figure(msg, fig)
    ex = res.example
    assert ex is not None
    mu = np.asarray(ex.true_means, dtype=float)
    optimal = {int(a) for a in np.atleast_1d(ex.optimal_arms)}
    shown, rest = _ranked_arms(mu, MAX_Q_ARMS - 1)
    others = [int(a) for a in shown if int(a) not in optimal]
    grays = dict(zip(others, _neutral_ramp(len(others))))
    algs = _ordered_algorithms(res, ex.q_mean)
    axes = _panel_axes(fig, len(algs), sharex=True)
    rows, cols = _grid_shape(len(algs))
    n_runs = max((np.atleast_2d(ex.counts[a]).shape[0] for a in ex.counts), default=0)
    dotted = (0, (1, 2))

    for k, (ax, alg) in enumerate(zip(axes, algs)):
        q = np.asarray(ex.q_mean[alg], dtype=float)
        q_std = np.asarray(ex.q_std.get(alg, np.zeros_like(q)), dtype=float)
        t = _pulls_axis(q.shape[0])
        # Primero los grises (detrás), luego el óptimo encima.
        for a in reversed(others):
            ax.axhline(mu[a], color=grays[a], linestyle=dotted, linewidth=1.1, zorder=2)
            ax.plot(t, q[:, a], color=grays[a], linewidth=1.5, solid_capstyle="round", zorder=3)
        color = _algo_color(alg)
        for a in sorted(optimal):
            if q_std.shape == q.shape and np.any(q_std[:, a] > 0):
                ax.fill_between(t, q[:, a] - q_std[:, a], q[:, a] + q_std[:, a], color=color,
                                alpha=BAND_ALPHA, linewidth=0, zorder=2)
            ax.axhline(mu[a], color=color, linestyle=dotted, linewidth=1.4, zorder=4)
            ax.plot(t, q[:, a], color=color, linewidth=2.4, linestyle=ALGO_LINESTYLES.get(alg, "-"),
                    marker=ALGO_MARKERS.get(alg, "o"), markevery=_markevery(t.size, 0, 1), markersize=MARKER_SIZE,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, solid_capstyle="round",
                    dash_capstyle="round", zorder=5)
        ax.set_xlim(1, t.size)
        _panel_title(ax, _q_panel_title(alg, res))
        if k // cols == rows - 1 or k + cols >= len(algs):
            _axis_labels(ax, xlabel="Pull t (interacciones con el segmento)")
    # Una sola etiqueta Y para toda la cuadrícula (en la figura de la GUI no caben cuatro).
    fig.supylabel("Q estimado (media entre corridas)", color=TEXT_SECONDARY, fontsize=LABEL_SIZE)

    # Leyenda común: el óptimo se dibuja con el color de CADA algoritmo (muestra múltiple).
    handles: list[object] = []
    labels: list[str] = []
    for a in sorted(optimal):
        handles.append(tuple(Line2D([], [], color=_algo_color(g), linewidth=2.4,
                                    linestyle=ALGO_LINESTYLES.get(g, "-")) for g in algs))
        labels.append(f"{ex.arm_labels[a]} · μ = {_fmt_mu(mu[a])} (óptimo)")
    for a in others:
        handles.append(Line2D([], [], color=grays[a], linewidth=1.5))
        labels.append(f"{ex.arm_labels[a]} · μ = {_fmt_mu(mu[a])}")
    handles.append(Line2D([], [], color=TEXT_SECONDARY, linestyle=dotted, linewidth=1.3))
    labels.append("μ real (punteada)")
    ncol = max(1, min(4, len(handles), int(fig.get_size_inches()[0] / 2.4)))
    _legend(fig, handles, labels, loc="outside lower center", ncol=ncol,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, handlelength=4.0)

    prev = "ninguno" if ex.prev_fret is None else str(ex.prev_fret)
    hidden = f" (se muestran los {shown.size} de mayor μ)" if rest.size else ""
    _titles(fig, f"Evolución de Q frente a μ real · segmento {ex.position}",
            f"{n_runs} corridas · {mu.size} brazos{hidden} · traste previo = {prev} · brazo óptimo en el "
            "color de cada algoritmo, con banda ±1 std")
    apply_style(fig)
    return fig


# ---------------------------------------------------------------------------
# Barridos: sensibilidad a hiperparámetros y efecto de λ
# ---------------------------------------------------------------------------


def _by_value_and_run(data: np.ndarray) -> np.ndarray:
    """Resultados de un barrido como matriz ``(V valores, corridas)`` (un vector = una corrida)."""
    arr = np.asarray(data, dtype=float)
    return arr[:, None] if arr.ndim == 1 else arr


def _value_axis(ax: Axes, values: np.ndarray, log: bool) -> None:
    """Eje x con ticks EXACTAMENTE en los valores evaluados (lineal, log o symlog si hay un 0)."""
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if log and positive.size >= 2:
        if np.any(values <= 0):
            ax.set_xscale("symlog", linthresh=float(positive.min()), linscale=0.6)
        else:
            ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(values.tolist()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _fmt_value(v)))
    ax.xaxis.set_minor_locator(NullLocator())


def _wants_log(values: Sequence[float], param: str | None = None) -> bool:
    """Escala logarítmica para c y τ, y para cualquier barrido que abarque ≥ 1 década."""
    v = np.asarray(values, dtype=float)
    positive = v[v > 0]
    if positive.size < 2:
        return False
    if param in ("agent.ucb_c", "agent.tau"):
        return True
    return bool(positive.max() / positive.min() >= 10 and param != "agent.epsilon")


def _current_value_line(ax: Axes, value: float, text: str) -> Line2D:
    """Línea vertical gris en el valor actual de la configuración (con su etiqueta)."""
    line = ax.axvline(value, color=TEXT_MUTED, linewidth=1.3, zorder=1)
    ax.annotate(text, xy=(value, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(3, -2),
                textcoords="offset points", ha="left", va="top", fontsize=ANNOTATION_SIZE, color=TEXT_SECONDARY,
                bbox=LABEL_BOX, zorder=7)
    return line


def plot_sensitivity(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Sensibilidad: recompensa final vs ε, Q₀, c y τ (2×2, una curva por parámetro).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``sweeps`` (:class:`src.experiments.SweepResult`): ``values`` y
        ``final_reward`` de forma ``(V, corridas)``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Un panel por barrido (ε de ε-greedy, Q₀ del optimista, c de UCB1, τ de
        Softmax) con la media ± 1 std de la recompensa final; x logarítmica
        para c, τ y Q₀; línea gris vertical = valor actual de la configuración.

    Notes
    -----
    Una curva en forma de "U invertida" es la firma del dilema
    exploración–explotación: poca exploración se queda con un brazo
    subóptimo; demasiada desperdicia pulls en brazos malos.
    """
    fig = _prepare_figure(fig, FIGSIZE_GRID)
    msg = missing_data_message("sensitivity", res)
    if msg:
        return message_figure(msg, fig)
    order = {p: i for i, p in enumerate(_SWEEP_INFO)}
    sweeps = sorted(res.sweeps, key=lambda s: order.get(s.param, len(order)))
    axes = _panel_axes(fig, len(sweeps))
    _rows, cols = _grid_shape(len(sweeps))
    algs_seen: list[str] = []
    n_runs = 0
    for k, (ax, sw) in enumerate(zip(axes, sweeps)):
        symbol, xlabel = _SWEEP_INFO.get(sw.param, (sw.param, sw.param))
        values = np.asarray(sw.values, dtype=float)
        idx = np.argsort(values)
        values = values[idx]
        final = _by_value_and_run(sw.final_reward)[idx]
        n_runs = max(n_runs, final.shape[1])
        mean, std = final.mean(axis=1), final.std(axis=1)
        alg = sw.algorithm
        if alg not in algs_seen:
            algs_seen.append(alg)
        _plot_algo_curve(ax, alg, values, mean, std, 0, 1, markevery=slice(None))
        _value_axis(ax, values, _wants_log(values, sw.param))
        try:
            current = float(res.config.get(sw.param))
        except (AttributeError, ValueError, TypeError):
            current = None
        if current is not None and values.min() <= current <= values.max():
            _current_value_line(ax, current, f"actual: {_fmt_value(current)}")
        best = int(np.argmax(mean))
        _panel_title(ax, f"{symbol} — {_algo_label(alg)} · mejor: {_fmt_value(values[best])}")
        _axis_labels(ax, xlabel=xlabel)
        if k % cols == 0:
            _axis_labels(ax, ylabel="Recompensa media final r")
    rank = {a: i for i, a in enumerate(ALGORITHMS)}
    handles: list[object] = [_algo_handle(a) for a in sorted(algs_seen, key=lambda a: rank.get(a, len(rank)))]
    handles.append(Line2D([], [], color=TEXT_MUTED, linewidth=1.3, label="Valor actual de la configuración"))
    _legend(fig, handles, loc="outside lower center", ncol=len(handles))
    _titles(fig, "Sensibilidad a los hiperparámetros",
            f"Media ± 1 std de {n_runs} corridas por valor · r = recompensa media del último 10 % de pulls "
            f"(promedio sobre {res.n_segments} segmentos)")
    apply_style(fig)
    return fig


def plot_lambda_effect(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Precisión de posición (y de pitch) vs λ, por algoritmo y para los dos oráculos.

    Parameters
    ----------
    res : ExperimentResult
        Usa ``lambda_sweep`` (:class:`src.experiments.LambdaSweepResult`).
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Dos paneles con eje y 0–100 %: precisión de POSICIÓN (izquierda) y de
        PITCH (derecha) frente a λ, una curva por algoritmo (media ± 1 std),
        el oráculo miope en gris oscuro y el oráculo de cadena (Viterbi) en
        gris claro; línea vertical = λ actual.

    Notes
    -----
    λ solo cambia la penalización λ·|Δtraste|/12, que es idéntica para
    posiciones con el mismo pitch salvo por el traste: por eso mueve la
    precisión de posición (A-0 frente a E-5) y casi no la de pitch. Con
    λ = 0 las posiciones gemelas empatan y deciden las reglas de desempate.
    El oráculo de cadena es el óptimo de Σ_s μ_s con el MISMO modelo de
    tocabilidad: si no llega al 100 % de posición (o acierta menos que el
    miope), el modelo λ·|Δtraste|/12 no coincide con la digitación del
    ground truth.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("lambda_effect", res)
    if msg:
        return message_figure(msg, fig)
    ls = res.lambda_sweep
    assert ls is not None
    values = np.asarray(ls.values, dtype=float)
    idx = np.argsort(values)
    values = values[idx]
    algs = _ordered_algorithms(res, ls.position_acc)
    axes = fig.subplots(1, 2, sharey=True).ravel().tolist()
    panels = [("Posición (cuerda y traste correctos)", ls.position_acc, ls.oracle_position_acc,
               ls.viterbi_position_acc),
              ("Pitch (nota MIDI correcta)", ls.pitch_acc, ls.oracle_pitch_acc, ls.viterbi_pitch_acc)]
    n_runs = 0
    for k, (ax, (title, acc, oracle, viterbi)) in enumerate(zip(axes, panels)):
        for i, alg in enumerate(algs):
            if alg not in acc:
                continue
            data = 100.0 * _by_value_and_run(acc[alg])[idx]
            n_runs = max(n_runs, data.shape[1])
            _plot_algo_curve(ax, alg, values, data.mean(axis=1), data.std(axis=1), i, len(algs),
                             markevery=slice(None))
        for key, ref in (("oracle", oracle), ("oracle_viterbi", viterbi)):
            if ref is None:
                continue
            ax.plot(values, 100.0 * np.asarray(ref, dtype=float)[idx], color=_algo_color(key),
                    linewidth=LINE_WIDTH, linestyle=ALGO_LINESTYLES[key], marker=ALGO_MARKERS[key],
                    markersize=MARKER_SIZE, markeredgewidth=1.6, dash_capstyle="round", zorder=4)
        _value_axis(ax, values, _wants_log(values))
        if values.min() <= res.config.env.lam <= values.max():
            _current_value_line(ax, res.config.env.lam, f"λ actual: {_fmt_value(res.config.env.lam)}")
        _percent_axis(ax, "y", top=102)
        _panel_title(ax, title)
        _axis_labels(ax, xlabel="λ (peso de la penalización de tocabilidad)")
        if k == 0:
            _axis_labels(ax, ylabel="Precisión (% de notas del ground truth)")
    handles: list[object] = [_algo_handle(a) for a in algs]
    for key in ("oracle", "oracle_viterbi") if ls.viterbi_position_acc is not None else ("oracle",):
        handles.append(Line2D([], [], color=_algo_color(key), linewidth=LINE_WIDTH,
                              linestyle=ALGO_LINESTYLES[key], marker=ALGO_MARKERS[key],
                              markersize=MARKER_SIZE, markeredgewidth=1.6, label=_algo_label(key)))
    _legend(fig, handles, loc="outside lower center", ncol=min(3, len(handles)))
    _titles(fig, "Efecto de λ en la precisión",
            f"Media ± 1 std de {n_runs} corridas por valor · {len(res.gt_notes or [])} notas del ground truth · "
            "λ penaliza |Δtraste|/12")
    apply_style(fig)
    return fig


# ---------------------------------------------------------------------------
# Precisión y tiempo
# ---------------------------------------------------------------------------


def _accuracy_arrays(res: ExperimentResult, alg: str) -> tuple[np.ndarray, np.ndarray]:
    """Precisión de pitch y de posición (en %) de cada corrida de ``alg``."""
    runs = res.accuracy.get(alg, [])
    pitch = 100.0 * np.array([r.pitch for r in runs], dtype=float)
    position = 100.0 * np.array([r.position for r in runs], dtype=float)
    return pitch, position


def plot_accuracy(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Precisión de pitch y de posición por algoritmo (barras agrupadas ± std, oráculo como referencia).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``accuracy[alg]`` (una :class:`src.experiments.AccuracyResult`
        por corrida) y ``oracle_accuracy``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Dos grupos (pitch, posición) con una barra por algoritmo (media ±
        1 std entre corridas, valor en la punta) y los dos oráculos como
        líneas horizontales de referencia sobre cada grupo (miope a la
        derecha, de cadena a la izquierda); eje y 0–100 %.

    Notes
    -----
    El oráculo miope (argmax μ en cada segmento) es el techo de RECOMPENSA
    y de % de pulls óptimos, no de precisión contra el ground truth: un
    algoritmo puede superarlo en posición si se aparta de argmax μ hacia la
    posición del GT (p. ej. en ``cromatica``: oráculo 23.8 %, Softmax ≈ 56 %),
    lo que indica que el modelo de tocabilidad no coincide con la
    digitación real. Lo que sí es cierto: si el oráculo se equivoca de
    PITCH, el error está en la recompensa (espectro, β), la segmentación o
    pYIN, no en el bandit. El oráculo de cadena (Viterbi) es el óptimo de la
    suma Σ_s μ_s de toda la pista con el mismo modelo; tampoco es un techo de
    precisión (en ``riff_saltos`` acierta menos posiciones que el miope).

    Las barras son el acierto sobre las notas del GT (recall): los segmentos
    de más no cuentan aquí; si los hay, el subtítulo lo indica (el F1 de
    ``summary.csv`` sí los penaliza).
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("accuracy", res)
    if msg:
        return message_figure(msg, fig)
    algs = [a for a in _ordered_algorithms(res, res.accuracy) if res.accuracy.get(a)]
    groups = ["Pitch\n(nota MIDI correcta)", "Posición\n(cuerda y traste correctos)"]
    x = np.arange(len(groups), dtype=float)
    xlim = (-0.6, len(groups) - 0.25)
    offsets, width = _bar_offsets(len(algs), _bar_limit(fig, xlim[1] - xlim[0]))
    half = (offsets[-1] - offsets[0]) / 2 + width / 2 if len(algs) else GROUP_WIDTH / 2
    ax = fig.add_subplot()
    n_runs = 0
    for i, alg in enumerate(algs):
        pitch, position = _accuracy_arrays(res, alg)
        n_runs = max(n_runs, pitch.size)
        means = np.array([pitch.mean(), position.mean()])
        stds = np.array([pitch.std(), position.std()])
        xi = x + offsets[i]
        ax.bar(xi, means, width=width, color=_algo_color(alg), linewidth=0, zorder=3)
        ax.errorbar(xi, means, yerr=stds, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=2.5,
                    capthick=1.0, zorder=4)
        ax.plot(xi, means, linestyle="none", marker=ALGO_MARKERS.get(alg, "o"), markersize=5.5,
                color=_algo_color(alg), markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)
        for xv, m, s in zip(xi, means, stds):
            ax.text(xv, m + s + 1.5, f"{m:.0f} %", ha="center", va="bottom", fontsize=ANNOTATION_SIZE,
                    color=TEXT_PRIMARY, zorder=7, bbox=LABEL_BOX)
    handles: list[object] = [_bar_handle(a) for a in algs]
    leg_labels = [_algo_label(a) for a in algs]
    # Referencias: oráculo miope (etiqueta a la derecha) y de cadena (a la izquierda).
    references = [("oracle", res.oracle_accuracy, "Oráculo miope", 1),
                  ("oracle_viterbi", res.viterbi_accuracy, "Oráculo cadena", -1)]
    for key, acc, short, side in references:
        if acc is None:
            continue
        ref = 100.0 * np.array([acc.pitch, acc.position])
        for xv, v in zip(x, ref):
            ax.hlines(v, xv - half - 0.05, xv + half + 0.05, color=_algo_color(key),
                      linewidth=LINE_WIDTH, linestyle=ALGO_LINESTYLES[key], zorder=6)
            ax.text(xv + side * (half + 0.08), v, f"{short}\n{v:.0f} %", ha="left" if side > 0 else "right",
                    va="center", fontsize=ANNOTATION_SIZE, color=TEXT_SECONDARY, linespacing=1.1)
        handles.append(Line2D([], [], color=_algo_color(key), linewidth=LINE_WIDTH, linestyle=ALGO_LINESTYLES[key]))
        leg_labels.append(_algo_label(key))
    if res.viterbi_accuracy is not None:
        xlim = (xlim[0] - 0.3, xlim[1])   # sitio para las etiquetas de la izquierda
    ax.set_xticks(x, groups)
    ax.set_xlim(*xlim)
    _percent_axis(ax, "y", top=112)
    _axis_labels(ax, ylabel="Precisión (% de notas del ground truth)")
    n_extra = next((a.n_extra for runs in res.accuracy.values() for a in runs if a.n_extra), 0)
    extra_text = f" · {n_extra} segmentos de más (no cuentan aquí)" if n_extra else ""
    _titles(fig, "Precisión de la transcripción",
            f"{n_runs} corridas por algoritmo · {len(res.gt_notes or [])} notas del ground truth · "
            f"barras = media ± 1 std; las notas sin segmento cuentan como error{extra_text}")
    _legend(fig, handles, leg_labels, loc="outside lower center", ncol=min(len(handles), 3), handlelength=2.0,
            handler_map={tuple: HandlerTuple(ndivide=1)})
    apply_style(fig, grid="y")
    ax.tick_params(axis="x", length=0, labelcolor=TEXT_SECONDARY, labelsize=LABEL_SIZE - 0.5)
    return fig


def plot_runtime(res: ExperimentResult, fig: Figure | None = None) -> Figure:
    """Tiempo de ejecución por algoritmo (ms por segmento, media ± std).

    Parameters
    ----------
    res : ExperimentResult
        Usa ``elapsed_s[alg]`` (segundos por corrida de la cadena completa) y
        ``n_segments``.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Barras horizontales (una por algoritmo, en el orden fijo) con la media
        ± 1 std entre corridas y el valor en la punta.

    Notes
    -----
    Incluye select_arm + pull + update × T pulls por segmento. Las diferencias
    vienen del coste de decidir: ε-greedy hace un argmax, UCB1 calcula un bono
    por brazo y Softmax exponencia y muestrea una distribución.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    msg = missing_data_message("runtime", res)
    if msg:
        return message_figure(msg, fig)
    algs = _ordered_algorithms(res, res.elapsed_s)
    n_seg = max(int(res.n_segments), 1)
    ax = fig.add_subplot()
    y = np.arange(len(algs), dtype=float)
    means, stds = [], []
    for alg in algs:
        ms = 1000.0 * np.atleast_1d(np.asarray(res.elapsed_s[alg], dtype=float)) / n_seg
        means.append(ms.mean())
        stds.append(ms.std())
    means_a, stds_a = np.array(means), np.array(stds)
    height = min(0.5, _bar_limit(fig, len(algs), axes_fraction=0.7, vertical=False))
    ax.barh(y, means_a, height=height, color=[_algo_color(a) for a in algs], linewidth=0, zorder=3)
    ax.errorbar(means_a, y, xerr=stds_a, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.0, capsize=3.0,
                capthick=1.0, zorder=4)
    span = float(np.max(means_a + stds_a)) if means_a.size else 1.0
    for yv, m, s in zip(y, means_a, stds_a):
        ax.text(m + s + 0.015 * span, yv, f"{m:.2f} ms", ha="left", va="center", fontsize=ANNOTATION_SIZE + 0.5,
                color=TEXT_PRIMARY)
    ax.set_yticks(y, [_algo_label(a) for a in algs])
    ax.invert_yaxis()  # el primer algoritmo arriba, como en la leyenda de las demás gráficas
    ax.set_xlim(0, span * 1.18 if span > 0 else 1.0)
    _axis_labels(ax, xlabel="Tiempo (ms por segmento)")
    _titles(fig, "Tiempo de ejecución por algoritmo",
            f"Media ± 1 std de {res.n_runs} corridas · cada segmento = T = {res.budget} pulls "
            "(select_arm + pull + update)")
    apply_style(fig, grid="x")
    # Las etiquetas del eje Y nombran cada barra (hacen de leyenda): tinta primaria y sin marcas.
    ax.tick_params(axis="y", length=0, labelcolor=TEXT_PRIMARY, labelsize=LABEL_SIZE)
    return fig


# ---------------------------------------------------------------------------
# Heatmap de aciertos por nota
# ---------------------------------------------------------------------------


def _modal_arms(choices: np.ndarray) -> list[Arm]:
    """Posición más frecuente entre corridas para cada segmento.

    Parameters
    ----------
    choices : np.ndarray
        Forma ``(corridas, S, 2)`` con (índice de cuerda, traste).

    Returns
    -------
    list[Arm]
        Una posición por segmento; los empates se resuelven por el par
        (cuerda, traste) menor (determinista).
    """
    ch = np.asarray(choices, dtype=int)
    if ch.ndim == 2:
        ch = ch[None]
    arms: list[Arm] = []
    for s in range(ch.shape[1]):
        pairs, counts = np.unique(ch[:, s, :], axis=0, return_counts=True)
        si, fret = pairs[int(np.argmax(counts))]
        arms.append(Arm(STRING_ORDER[int(si)], int(fret)))
    return arms


def _outcome(arm: Arm, note: object) -> int:
    """Código ``OUTCOME_*`` de una posición predicha frente a una nota GT emparejada."""
    if arm.midi != note.midi:  # type: ignore[attr-defined]
        return OUTCOME_WRONG_PITCH
    if arm.string == note.string and arm.fret == note.fret:  # type: ignore[attr-defined]
        return OUTCOME_EXACT
    return OUTCOME_WRONG_POSITION


def _infer_gt_matches(res: ExperimentResult) -> list[int | None]:
    """Reconstruye qué segmento se emparejó con cada nota GT usando solo ``res``.

    ``ExperimentResult`` guarda los códigos de resultado por nota GT y las
    posiciones elegidas por segmento, pero no el emparejamiento. Como este
    depende solo de los onsets (es el mismo en todas las corridas):

    1. Las notas con código ``OUTCOME_MISSED`` no tienen segmento.
    2. Las demás se alinean EN ORDEN TEMPORAL con los segmentos (alineación
       monótona por programación dinámica) maximizando cuántas
       (corrida, algoritmo, nota) reproducen exactamente su código guardado.
       Con tantos segmentos como notas emparejadas la alineación es la identidad.

    Returns
    -------
    list[int | None]
        Para cada nota GT, la posición del segmento o None.
    """
    gt = list(res.gt_notes or [])
    n_gt = len(gt)
    sources: list[tuple[np.ndarray, np.ndarray]] = []  # (elecciones (n, S, 2), códigos (n, G))
    for alg in _ordered_algorithms(res, res.choices):
        runs = res.accuracy.get(alg, [])
        if not runs:
            continue
        ch = np.asarray(res.choices[alg], dtype=int)
        out = np.array([np.asarray(r.outcomes, dtype=int) for r in runs])
        n = min(ch.shape[0], out.shape[0])
        if n and out.shape[1] == n_gt:
            sources.append((ch[:n], out[:n]))
    if res.oracle_accuracy is not None and res.oracle_arms:
        oracle_ch = np.array([[a.string_index, a.fret] for a in res.oracle_arms], dtype=int)[None]
        oracle_out = np.asarray(res.oracle_accuracy.outcomes, dtype=int)[None]
        if oracle_out.shape[1] == n_gt:
            sources.append((oracle_ch, oracle_out))
    if not sources or n_gt == 0:
        return [None] * n_gt
    n_seg = min(src[0].shape[1] for src in sources)
    all_out = np.concatenate([src[1] for src in sources], axis=0)
    matched = np.flatnonzero((all_out == OUTCOME_MISSED).mean(axis=0) < 0.5)
    m = matched.size
    if m == 0 or m > n_seg:
        if m > n_seg:
            logger.warning("Hay más notas GT emparejadas (%d) que segmentos (%d); no se puede reconstruir "
                           "el emparejamiento.", m, n_seg)
        return [None] * n_gt

    open_midi = np.array([OPEN_STRING_MIDI[s] for s in STRING_ORDER])
    gt_midi = np.array([gt[g].midi for g in matched])
    gt_string = np.array([STRING_ORDER.index(gt[g].string) for g in matched])
    gt_fret = np.array([gt[g].fret for g in matched])
    agree = np.zeros((m, n_seg))
    for ch, out in sources:
        ch = ch[:, :n_seg]
        midi = open_midi[ch[..., 0]] + ch[..., 1]  # (n, S)
        pitch_ok = midi[:, None, :] == gt_midi[None, :, None]  # (n, M, S)
        pos_ok = pitch_ok & (ch[:, None, :, 0] == gt_string[None, :, None]) & (ch[:, None, :, 1] == gt_fret[None, :, None])
        code = np.where(pos_ok, OUTCOME_EXACT, np.where(pitch_ok, OUTCOME_WRONG_POSITION, OUTCOME_WRONG_PITCH))
        agree += (code == out[:, matched][:, :, None]).sum(axis=0)

    # f[i, j] = mejor coincidencia alineando las i primeras notas con los j primeros segmentos.
    f = np.full((m + 1, n_seg + 1), -np.inf)
    f[0, :] = 0.0
    for i in range(1, m + 1):
        for j in range(i, n_seg + 1):
            f[i, j] = max(f[i, j - 1], f[i - 1, j - 1] + agree[i - 1, j - 1])
    result: list[int | None] = [None] * n_gt
    i, j = m, n_seg
    while i > 0:
        if f[i - 1, j - 1] + agree[i - 1, j - 1] >= f[i, j - 1]:
            result[int(matched[i - 1])] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return result


def plot_tab_heatmap(res: ExperimentResult, fig: Figure | None = None,
                     matches: Sequence[int | None] | None = None) -> Figure:
    """Heatmap de aciertos/errores por nota GT (columnas) y algoritmo (filas), con la
    posición modal entre corridas; categorías :data:`src.experiments.OUTCOME_LABELS`.

    Parameters
    ----------
    res : ExperimentResult
        Usa ``choices``, ``gt_notes``, ``accuracy``, ``oracle_arms``,
        ``oracle_accuracy`` y ``viterbi_arms``.
    fig : Figure, optional
        Figura a reutilizar.
    matches : sequence of int | None, optional
        Segmento emparejado con cada nota GT (como lo devuelve
        :func:`src.experiments.match_segments_to_gt`). Si es None se
        reconstruye a partir de ``res`` (ver :func:`_infer_gt_matches`).

    Returns
    -------
    Figure
        Una fila por algoritmo (más los oráculos miope y de cadena) y una columna por nota GT
        (etiqueta = posición real). Cada celda se colorea según el resultado
        de la POSICIÓN MODAL entre corridas (:data:`OUTCOME_COLORS`) y lleva
        escrita la posición predicha; «—» = nota sin segmento. A la derecha,
        el % de posiciones exactas de cada fila.

    Notes
    -----
    Permite ver QUÉ notas fallan: un error de pitch común a todos los
    algoritmos (y al oráculo) apunta a la recompensa o a pYIN; un amarillo
    solo en algunos algoritmos es una cuestión de exploración o de λ.
    """
    fig = _prepare_figure(fig, FIGSIZE_WIDE)
    msg = missing_data_message("tab_heatmap", res)
    if msg:
        return message_figure(msg, fig)
    gt = list(res.gt_notes or [])
    n_gt = len(gt)
    if matches is None:
        matches = _infer_gt_matches(res)
    rows: list[tuple[str, list[Arm]]] = [(_algo_label(a), _modal_arms(res.choices[a]))
                                         for a in _ordered_algorithms(res, res.choices)]
    if res.oracle_arms:
        rows.append((_algo_label("oracle"), list(res.oracle_arms)))
    if res.viterbi_arms:
        rows.append((_algo_label("oracle_viterbi"), list(res.viterbi_arms)))
    codes = np.zeros((len(rows), n_gt), dtype=int)
    texts: list[list[str]] = []
    for r, (_label, arms) in enumerate(rows):
        row_text = []
        for g, note in enumerate(gt):
            s = matches[g] if g < len(matches) else None
            if s is None or s >= len(arms):
                codes[r, g] = OUTCOME_MISSED
                row_text.append("—")
            else:
                codes[r, g] = _outcome(arms[s], note)
                row_text.append(arms[s].label)
        texts.append(row_text)

    ax = fig.add_subplot()
    cmap = ListedColormap([OUTCOME_COLORS[k] for k in sorted(OUTCOME_COLORS)])
    norm = BoundaryNorm(np.arange(-0.5, len(OUTCOME_COLORS) + 0.5), cmap.N)
    # Bordes del color de la superficie = hueco de 2 px entre celdas vecinas.
    ax.pcolormesh(np.arange(n_gt + 1), np.arange(len(rows) + 1), codes, cmap=cmap, norm=norm,
                  edgecolors=SURFACE, linewidth=2.0)
    ax.set_xlim(0, n_gt)
    ax.set_ylim(len(rows), 0)  # primera fila arriba

    # Texto dentro de cada celda: la posición predicha (o solo el traste si no cabe).
    fig_w = fig.get_size_inches()[0]
    cell_pt = 72.0 * fig_w * 0.80 / max(n_gt, 1)
    if cell_pt >= 24:
        fontsize, short = 7.5, False
    elif cell_pt >= 13:
        fontsize, short = 7.0, True
    else:
        fontsize, short = 5.5, True
    for r in range(len(rows)):
        for g in range(n_gt):
            text = texts[r][g]
            if short and "-" in text:
                text = text.split("-", 1)[1]
            ax.text(g + 0.5, r + 0.5, text, ha="center", va="center", fontsize=fontsize,
                    color=_ink_on(OUTCOME_COLORS[int(codes[r, g])]))
    # % de posiciones exactas por fila, como columna de texto a la derecha.
    right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, -0.15, "Posición\nexacta", transform=right, ha="left", va="bottom", fontsize=ANNOTATION_SIZE,
            color=TEXT_SECONDARY, linespacing=1.1)
    for r in range(len(rows)):
        pct = 100.0 * float(np.mean(codes[r] == OUTCOME_EXACT)) if n_gt else 0.0
        ax.text(1.01, r + 0.5, f"{pct:.0f} %", transform=right, ha="left", va="center", fontsize=LEGEND_SIZE,
                color=TEXT_PRIMARY)

    ax.set_yticks(np.arange(len(rows)) + 0.5, [label for label, _ in rows])
    rotate = n_gt > 24
    ax.set_xticks(np.arange(n_gt) + 0.5, [n.label for n in gt], rotation=90 if rotate else 0)
    apply_style(fig, grid=None)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", labelcolor=TEXT_PRIMARY, labelsize=LABEL_SIZE - 0.5)
    # Las etiquetas X son la posición real (información esencial): tinta secundaria, no gris tenue.
    ax.tick_params(axis="x", labelcolor=TEXT_SECONDARY, labelsize=7.5 if rotate else 8.5)
    _axis_labels(ax, xlabel="Nota del ground truth en orden temporal (etiqueta = posición real cuerda-traste)")
    handles = [Patch(facecolor=OUTCOME_COLORS[k], edgecolor="none", label=OUTCOME_LABELS[k])
               for k in sorted(OUTCOME_LABELS)]
    _legend(fig, handles, loc="outside lower center", ncol=len(handles), handlelength=1.4)
    _titles(fig, "Aciertos por nota del ground truth",
            f"Posición modal entre {res.n_runs} corridas · texto = posición predicha (cuerda-traste) · "
            "— = nota sin segmento detectado")
    return fig


# ---------------------------------------------------------------------------
# Audio: forma de onda, espectrograma y espectro de una nota
# ---------------------------------------------------------------------------


#: Caché de UNA entrada para la CQT de visualización cuando la recompensa usa
#: la STFT: (señal analizada, clave de parámetros, espectro). La GUI redibuja
#: el espectrograma en cada selección de segmento y la CQT de una pista larga
#: tarda segundos; con la caché solo se calcula una vez por análisis.
_DISPLAY_CQT_CACHE: list[tuple[np.ndarray, tuple[object, ...], Spectrum]] = []


def _cached_display_cqt(analysis: AnalysisResult) -> Spectrum:
    """CQT de ``analysis.y_spectral`` para dibujar (se reutiliza si no cambió la señal)."""
    from src.environment import compute_spectrum  # diferido: solo hace falta con STFT

    cfg = replace(analysis.config.env, spectrum="cqt")
    key = (analysis.sr, analysis.spectrum.hop_length, cfg.cqt_fmin_hz, cfg.n_octaves, cfg.bins_per_octave)
    if _DISPLAY_CQT_CACHE:
        y_cached, key_cached, spec_cached = _DISPLAY_CQT_CACHE[0]
        if y_cached is analysis.y_spectral and key_cached == key:
            return spec_cached
    spec = compute_spectrum(analysis.y_spectral, analysis.sr, cfg, hop_length=analysis.spectrum.hop_length)
    _DISPLAY_CQT_CACHE[:] = [(analysis.y_spectral, key, spec)]
    return spec


def _display_cqt(analysis: AnalysisResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CQT en dB (0 dB = máximo) de ``y_spectral`` lista para dibujar.

    Reutiliza ``analysis.spectrum`` si ya es una CQT; si la recompensa usa la
    STFT, calcula una CQT con los mismos parámetros. Recorta a
    :data:`SPECTRO_FMAX_HZ` y reduce las columnas a :data:`SPECTRO_MAX_COLUMNS`
    (máximo por bloques, para no perder ataques).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(db (bins, frames), freqs_hz (bins,), times_s (frames,))``.
    """
    spec = analysis.spectrum
    if spec.kind != "cqt":
        spec = _cached_display_cqt(analysis)
    mag = np.asarray(spec.mag, dtype=float)
    freqs = np.asarray(spec.freqs_hz, dtype=float)
    times = np.asarray(spec.times_s, dtype=float)
    keep = freqs <= SPECTRO_FMAX_HZ
    mag, freqs = mag[keep], freqs[keep]
    n_frames = mag.shape[1]
    if n_frames > SPECTRO_MAX_COLUMNS:
        block = int(np.ceil(n_frames / SPECTRO_MAX_COLUMNS))
        n_cols = n_frames // block
        mag = mag[:, :n_cols * block].reshape(mag.shape[0], n_cols, block).max(axis=2)
        times = times[:n_cols * block].reshape(n_cols, block).mean(axis=1)
    ref = max(float(mag.max()) if mag.size else 0.0, 1e-10)
    # dB relativos al máximo: 20·log10(|X| / max|X|), recortado a −SPECTRO_DB_RANGE dB.
    db = 20.0 * np.log10(np.maximum(mag, 1e-10) / ref)
    return np.clip(db, -SPECTRO_DB_RANGE, 0.0), freqs, times


def _edges(centers: np.ndarray, log: bool = False) -> np.ndarray:
    """Bordes de celda a partir de los centros (en escala log si ``log``)."""
    c = np.log2(centers) if log else np.asarray(centers, dtype=float)
    if c.size == 1:
        d = 1.0
        e = np.array([c[0] - d / 2, c[0] + d / 2])
    else:
        mid = (c[1:] + c[:-1]) / 2
        e = np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
    return 2.0 ** e if log else e


def _note_axis(ax: Axes, fmin: float, fmax: float) -> None:
    """Eje de frecuencia logarítmico con ticks en notas del bajo (E1, A1, D2, G2, G3...)."""
    ax.set_yscale("log")
    ax.set_ylim(fmin, fmax)
    ticks = [(name, hz) for name, hz in NOTE_TICKS if fmin <= hz <= fmax]
    ax.yaxis.set_major_locator(FixedLocator([hz for _n, hz in ticks]))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{name} · {hz:.0f}" for name, hz in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())


def _format_time_freq(x: float, y: float) -> str:
    """Coordenadas que muestra la barra de la GUI sobre el espectrograma."""
    if y > 0:
        midi = 69.0 + 12.0 * np.log2(y / 440.0)
        return f"t = {x:.3f} s, f = {y:.1f} Hz (≈ {midi_to_name(int(round(midi)))})"
    return f"t = {x:.3f} s"


def _draw_spectrogram(ax: Axes, analysis: AnalysisResult) -> QuadMesh:
    """Dibuja la CQT en dB con eje de frecuencia log (Hz) y la f0 de pYIN encima."""
    db, freqs, times = _display_cqt(analysis)
    mesh = ax.pcolormesh(_edges(times).clip(min=0.0), _edges(freqs, log=True), db, cmap=SPECTRO_CMAP,
                         vmin=-SPECTRO_DB_RANGE, vmax=0.0, shading="flat", rasterized=True, zorder=1)
    _note_axis(ax, float(_edges(freqs, log=True)[0]), float(_edges(freqs, log=True)[-1]))
    track = analysis.pitch_track
    if track.n_frames:
        # Halo del color de la superficie: la f0 se lee igual sobre zonas claras y oscuras.
        ax.plot(track.times_s, track.f0_hz, color=F0_COLOR, linewidth=1.8, solid_capstyle="round", zorder=5,
                path_effects=[path_effects.Stroke(linewidth=3.6, foreground=SURFACE), path_effects.Normal()])
    ax.format_coord = _format_time_freq  # type: ignore[method-assign]
    return mesh


def _draw_segments(ax: Axes, analysis: AnalysisResult, selected: int | None, numbers: bool = False) -> None:
    """Onsets (líneas finas), segmentos descartados (gris) y segmento seleccionado (marco)."""
    for seg in analysis.segments:
        if not seg.kept:
            ax.axvspan(seg.start_s, seg.end_s, color=DISCARDED_COLOR, alpha=0.28, linewidth=0, zorder=2)
    onsets = np.asarray(analysis.onsets_s, dtype=float)
    if onsets.size:
        # En pistas largas cientos de líneas taparían el espectrograma: se aclaran según su
        # densidad (líneas por pulgada de eje); al hacer zoom en la GUI siguen ahí.
        per_inch = onsets.size / max(ax.figure.get_size_inches()[0] * 0.8, 1e-6)
        alpha = 0.6 if per_inch <= ONSETS_PER_INCH else max(0.12, 0.6 * ONSETS_PER_INCH / per_inch)
        ax.vlines(onsets, 0, 1, transform=ax.get_xaxis_transform(), color=ONSET_COLOR,
                  linewidth=0.7 if per_inch <= ONSETS_PER_INCH else 0.5, alpha=alpha, zorder=4)
    kept = analysis.kept
    is_selected = selected is not None and 0 <= selected < len(kept)
    if numbers and 0 < len(kept) <= 40:
        top = ax.get_xaxis_transform()
        for pos, seg in enumerate(kept):
            chosen = is_selected and pos == selected
            ax.text((seg.start_s + seg.end_s) / 2, 0.97, str(pos), transform=top, ha="center", va="top",
                    fontsize=7.5 if chosen else 7, fontweight="bold" if chosen else "normal",
                    color=TEXT_PRIMARY if chosen else TEXT_SECONDARY, zorder=6)
    if is_selected:
        # Solo un marco (sin relleno): el gris de fondo está reservado a los segmentos descartados.
        seg = kept[selected]  # type: ignore[index]
        for edge in (seg.start_s, seg.end_s):
            ax.axvline(edge, color=TEXT_PRIMARY, linewidth=2.0, zorder=6)


def _audio_handles(selected: bool) -> list[object]:
    """Leyenda común de las figuras de audio."""
    handles: list[object] = [
        Line2D([], [], color=ONSET_COLOR, linewidth=1.0, label="Onset detectado"),
        Patch(facecolor=DISCARDED_COLOR, alpha=0.35, edgecolor="none", label="Segmento descartado (silencio o corto)"),
        Line2D([], [], color=F0_COLOR, linewidth=1.8, label="f0 de pYIN"),
    ]
    if selected:
        handles.append(Patch(facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=1.6, label="Segmento seleccionado"))
    return handles


def _audio_subtitle(analysis: AnalysisResult) -> str:
    """Subtítulo de las figuras de audio: archivo, duración, onsets y segmentos."""
    kept = len(analysis.kept)
    discarded = len(analysis.segments) - kept
    name = Path(analysis.path).name
    if len(name) > 40:
        name = name[:37] + "…"
    shown = "CQT de la señal sin filtrar"
    if analysis.spectrum.kind != "cqt":
        shown += f" (la recompensa usa la {analysis.spectrum.kind.upper()})"
    return (f"{name} · {analysis.duration_s:.1f} s · {len(analysis.onsets_s)} onsets · "
            f"{kept} segmentos conservados, {discarded} descartados · {shown}")


def _selected_text(analysis: AnalysisResult, selected: int | None) -> str | None:
    """Texto breve del segmento seleccionado (posición, nota y f0)."""
    kept = analysis.kept
    if selected is None or not 0 <= selected < len(kept):
        return None
    seg = kept[selected]
    if seg.f0_hz is None or seg.midi is None:
        return f"Seg. {selected} · f0 desconocida"
    return f"Seg. {selected} · {midi_to_name(int(round(seg.midi)))} ({seg.f0_hz:.1f} Hz)"


def _colorbar(fig: Figure, mesh: QuadMesh, **kwargs: object) -> None:
    """Barra de color del espectrograma (dB relativos al máximo)."""
    cbar = fig.colorbar(mesh, **kwargs)  # type: ignore[arg-type]
    cbar.set_label("dB (0 = máximo)", color=TEXT_SECONDARY, fontsize=LABEL_SIZE - 1)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_MUTED, labelsize=TICK_SIZE - 0.5, length=2)


def plot_spectrogram(analysis: AnalysisResult, fig: Figure | None = None, selected: int | None = None) -> Figure:
    """Espectrograma (CQT en dB) con onsets, segmentos descartados y f0 de pYIN superpuestos.

    ``selected``: posición (entre conservados) de un segmento a resaltar.

    Parameters
    ----------
    analysis : AnalysisResult
        Usa ``spectrum`` (o calcula una CQT de ``y_spectral`` si la recompensa
        usa STFT), ``onsets_s``, ``segments`` y ``pitch_track``.
    fig : Figure, optional
        Figura a reutilizar.
    selected : int, optional
        Posición (entre los segmentos conservados) a resaltar con un marco.

    Returns
    -------
    Figure
        Espectrograma con eje de frecuencia logarítmico en Hz (ticks en
        notas del bajo), barra de color en dB y leyenda.

    Notes
    -----
    En la CQT cada armónico h·f queda a la misma distancia vertical
    (log₂ h octavas) sobre la fundamental, sea cual sea la nota: el "peine"
    de armónicos que mide la recompensa se ve como líneas paralelas.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    ax = fig.add_subplot()
    mesh = _draw_spectrogram(ax, analysis)
    _draw_segments(ax, analysis, selected, numbers=False)
    ax.set_xlim(0, analysis.duration_s)
    _axis_labels(ax, "Tiempo (s)", "Frecuencia (Hz, escala log)")
    _colorbar(fig, mesh, ax=ax, pad=0.01, fraction=0.04)
    title = "Espectrograma con onsets y f0"
    sel = _selected_text(analysis, selected)
    _titles(fig, title + (f" · {sel}" if sel else ""), _audio_subtitle(analysis))
    _legend(fig, _audio_handles(sel is not None), loc="outside lower center", ncol=4, handlelength=2.0)
    apply_style(fig, grid=None)
    return fig


def plot_audio_overview(analysis: AnalysisResult, fig: Figure | None = None, selected: int | None = None) -> Figure:
    """Forma de onda (arriba) + espectrograma con onsets y f0 (abajo), ejes x compartidos.
    Es la figura de la pestaña Audio.

    Parameters
    ----------
    analysis : AnalysisResult
        Usa ``y_spectral``, ``sr`` y lo mismo que :func:`plot_spectrogram`.
    fig : Figure, optional
        Figura a reutilizar.
    selected : int, optional
        Posición (entre los segmentos conservados) a resaltar en ambos paneles.

    Returns
    -------
    Figure
        Dos paneles con el eje de tiempo compartido (al hacer zoom en uno se
        mueve el otro). Sobre la onda se numeran los segmentos conservados
        (si son ≤ 40); si la señal es larga se dibuja su envolvente (mínimo y
        máximo por columna).
    """
    fig = _prepare_figure(fig, FIGSIZE_WIDE)
    # Columna derecha estrecha solo para la barra de color: así ambos paneles tienen el mismo ancho.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2], width_ratios=[1.0, 0.018])
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)
    cax = fig.add_subplot(gs[1, 1])

    y = np.asarray(analysis.y_spectral, dtype=float)
    sr = float(analysis.sr)
    if y.size > WAVEFORM_MAX_POINTS * 4:
        # Envolvente: mínimo y máximo de cada bloque (una columna de píxeles ≈ un bloque).
        block = int(np.ceil(y.size / WAVEFORM_MAX_POINTS))
        n_blocks = y.size // block
        blocks = y[:n_blocks * block].reshape(n_blocks, block)
        t = (np.arange(n_blocks) + 0.5) * block / sr
        ax_wave.fill_between(t, blocks.min(axis=1), blocks.max(axis=1), color=SIGNAL_COLOR, linewidth=0,
                             alpha=0.85, zorder=3)
    elif y.size:
        ax_wave.plot(np.arange(y.size) / sr, y, color=SIGNAL_COLOR, linewidth=0.8, zorder=3)
    ax_wave.axhline(0.0, color=AXIS, linewidth=0.8, zorder=2)
    peak = max(float(np.max(np.abs(y))) if y.size else 1.0, 1e-6)
    ax_wave.set_ylim(-1.08 * peak, 1.08 * peak)
    _draw_segments(ax_wave, analysis, selected, numbers=True)
    _axis_labels(ax_wave, ylabel="Amplitud\n(normalizada)")
    ax_wave.tick_params(axis="x", labelbottom=False)

    mesh = _draw_spectrogram(ax_spec, analysis)
    _draw_segments(ax_spec, analysis, selected, numbers=False)
    ax_spec.set_xlim(0, analysis.duration_s)
    _axis_labels(ax_spec, "Tiempo (s)", "Frecuencia (Hz, escala log)")
    _colorbar(fig, mesh, cax=cax)

    sel = _selected_text(analysis, selected)
    _titles(fig, "Forma de onda y espectrograma" + (f" · {sel}" if sel else ""), _audio_subtitle(analysis))
    _legend(fig, _audio_handles(sel is not None), loc="outside lower center", ncol=4, handlelength=2.0)
    apply_style(fig, grid=None)
    ax_wave.grid(True, axis="y", color=GRID, linewidth=GRID_WIDTH)
    return fig


def plot_note_spectrum(
    analysis: AnalysisResult,
    position: int,
    arm: Arm,
    gt_arm: Arm | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Espectro medio del segmento ``position`` con el template armónico del brazo
    elegido (líneas en h·f y, punteadas, en (h−½)·f) y el del ground truth si difiere.

    Parameters
    ----------
    analysis : AnalysisResult
        Usa ``spectrum`` (el mismo que la recompensa), ``segment_data`` y
        ``config.env.n_harmonics``.
    position : int
        Posición del segmento entre los conservados.
    arm : Arm
        Posición elegida por el algoritmo.
    gt_arm : Arm, optional
        Posición real (ground truth). Su template se dibuja en otro color si
        su frecuencia difiere de la del brazo elegido.
    fig : Figure, optional
        Figura a reutilizar.

    Returns
    -------
    Figure
        Magnitud media normalizada (máximo = 1) frente a la frecuencia (Hz,
        escala log) sobre los mismos frames que muestrea la recompensa.

    Raises
    ------
    IndexError
        Si ``position`` no es un segmento conservado.

    Notes
    -----
    Es la recompensa "a la vista": la saliencia suma la energía bajo las
    líneas continuas (pesos 1/h) y resta β veces la energía bajo las
    punteadas. Un candidato una octava arriba pierde la mitad de sus dientes
    y tiene los huecos sobre armónicos reales.
    """
    fig = _prepare_figure(fig, FIGSIZE_SIMPLE)
    if not 0 <= position < len(analysis.segment_data):
        raise IndexError(f"El segmento {position} no existe (hay {len(analysis.segment_data)} conservados)")
    data = analysis.segment_data[position]
    seg = data.segment
    spec = analysis.spectrum
    freqs = np.asarray(spec.freqs_hz, dtype=float)
    mean = np.asarray(spec.mag, dtype=float)[:, np.asarray(data.frame_indices, dtype=int)].mean(axis=1)
    valid = freqs > 0  # la STFT incluye 0 Hz, que no existe en escala log
    freqs, mean = freqs[valid], mean[valid]
    peak = max(float(mean.max()) if mean.size else 0.0, 1e-12)
    mag = mean / peak

    n_harm = int(analysis.config.env.n_harmonics)
    show_gt = gt_arm is not None and abs(12.0 * np.log2(gt_arm.freq_hz / arm.freq_hz)) > 0.05
    f_top = (n_harm + 0.6) * max(arm.freq_hz, gt_arm.freq_hz if show_gt and gt_arm else 0.0)
    fmin = max(float(freqs.min()) if freqs.size else 20.0, 0.4 * min(arm.freq_hz, gt_arm.freq_hz if gt_arm else arm.freq_hz))
    fmax = min(float(freqs.max()) if freqs.size else 2000.0, f_top)

    ax = fig.add_subplot()
    in_view = (freqs >= fmin * 0.95) & (freqs <= fmax * 1.05)
    ax.fill_between(freqs[in_view], 0, mag[in_view], color=SPECTRUM_COLOR, alpha=0.08, linewidth=0, zorder=2)
    ax.plot(freqs[in_view], mag[in_view], color=SPECTRUM_COLOR, linewidth=LINE_WIDTH, solid_joinstyle="round",
            zorder=4)
    xaxis_tf = ax.get_xaxis_transform()

    def comb(target: Arm, color: str, harmonic_style: object, labels_on_top: bool) -> None:
        """Peine del template: h·f (dientes, se suman) y (h−½)·f (huecos, se restan con β)."""
        tmpl = harmonic_template(target.freq_hz, n_harm)
        for h, f in enumerate(tmpl.harmonic_hz, start=1):
            ax.axvline(f, color=color, linewidth=1.5, linestyle=harmonic_style, alpha=0.9, zorder=3)
            # Número de armónico: arriba (fuera del eje) para el elegido, abajo (dentro) para el GT,
            # así nunca se pisan cuando los dos peines comparten una línea (errores de octava).
            ax.text(f, 1.0 if labels_on_top else 0.015, f"{h}", transform=xaxis_tf, ha="center",
                    va="bottom", fontsize=ANNOTATION_SIZE, color=TEXT_SECONDARY, bbox=LABEL_BOX, zorder=7)
        for f in tmpl.interharmonic_hz:
            if f >= fmin:
                ax.axvline(f, color=color, linewidth=1.2, linestyle=(0, (1, 2)), alpha=0.9, zorder=3)

    comb(arm, ARM_COLOR, "-", True)
    handles: list[object] = [
        Line2D([], [], color=SPECTRUM_COLOR, linewidth=LINE_WIDTH, label="Espectro medio del segmento"),
        Line2D([], [], color=ARM_COLOR, linewidth=1.5, label=f"h·f del brazo elegido {arm.label} ({arm.freq_hz:.1f} Hz)"),
        Line2D([], [], color=ARM_COLOR, linewidth=1.2, linestyle=(0, (1, 2)), label="(h−½)·f del elegido (se resta con β)"),
    ]
    if show_gt and gt_arm is not None:
        comb(gt_arm, GT_COLOR, (0, (5, 2)), False)
        handles += [
            Line2D([], [], color=GT_COLOR, linewidth=1.5, linestyle=(0, (5, 2)),
                   label=f"h·f del ground truth {gt_arm.label} ({gt_arm.freq_hz:.1f} Hz)"),
            Line2D([], [], color=GT_COLOR, linewidth=1.2, linestyle=(0, (1, 2)), label="(h−½)·f del ground truth"),
        ]
    ax.set_xscale("log")
    ax.set_xlim(fmin, fmax)
    nice = [20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000]
    ax.xaxis.set_major_locator(FixedLocator([v for v in nice if fmin <= v <= fmax]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 1.05)
    _axis_labels(ax, "Frecuencia (Hz, escala log)", "Magnitud media (normalizada, máx. = 1)")

    if seg.f0_hz is not None and seg.midi is not None:
        note = f"nota {midi_to_name(int(round(seg.midi)))} · f0 pYIN = {seg.f0_hz:.1f} Hz"
    else:
        note = "f0 de pYIN desconocida"
    if gt_arm is None:
        positions = f"elegido {arm.label}"
    elif gt_arm == arm:
        positions = f"elegido {arm.label} = ground truth"
    elif not show_gt:
        positions = f"elegido {arm.label} · ground truth {gt_arm.label} (misma altura, otra posición)"
    else:
        positions = f"elegido {arm.label} · ground truth {gt_arm.label}"
    _titles(fig, f"Espectro del segmento {position}: {note}",
            f"{positions} · {seg.start_s:.2f}–{seg.end_s:.2f} s · media de {len(data.frame_indices)} frames "
            f"({spec.kind.upper()}) · números = armónico h" + (" (abajo: del ground truth)" if show_gt else ""))
    _legend(fig, handles, loc="outside lower center", ncol=3 if len(handles) > 3 else len(handles),
            handlelength=2.6)
    apply_style(fig, grid="y")
    return fig


# ---------------------------------------------------------------------------
# Catálogo y exportación
# ---------------------------------------------------------------------------


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
COMPARISON_PLOTS: list[PlotSpec] = [
    PlotSpec(
        "average_reward", "Recompensa media por pull", plot_average_reward,
        "Recompensa obtenida en cada pull t, promediada sobre todos los segmentos y corridas (banda = ±1 std "
        "entre corridas). Cuanto antes sube y más alto se estabiliza la curva, mejor equilibra el algoritmo "
        "exploración y explotación; ε-greedy queda algo por debajo porque sigue explorando una fracción ε.",
    ),
    PlotSpec(
        "cumulative_regret", "Regret acumulado", plot_cumulative_regret,
        "Suma de lo perdido por no jugar el brazo óptimo, Σ(μ* − μ_a), promediada sobre segmentos. Una curva "
        "que se aplana indica que el agente ya encontró el mejor brazo; crecer como una recta (regret lineal) "
        "es típico de ε fijo, mientras que UCB1 garantiza crecimiento logarítmico.",
    ),
    PlotSpec(
        "optimal_action", "Selección del brazo óptimo", plot_optimal_action,
        "Porcentaje de segmentos en los que el pull t fue al brazo óptimo (argmax μ). Distingue brazos con "
        "recompensa casi igual pero distinta posición (A-0 frente a E-5): un algoritmo puede tener buena "
        "recompensa media y aun así no fijarse en la posición correcta.",
    ),
    PlotSpec(
        "arm_distribution", "Reparto de pulls entre brazos", plot_arm_distribution,
        "Para un segmento de ejemplo, cuántos de los T pulls dedicó cada algoritmo a cada brazo (ordenados por "
        "μ real; el óptimo, con fondo gris). Un buen agente concentra los pulls en el óptimo y gasta pocos en "
        "brazos claramente malos; muchos pulls al segundo mejor indican una brecha Δ pequeña.",
    ),
    PlotSpec(
        "q_evolution", "Evolución de Q frente a μ real", plot_q_evolution,
        "Un panel por algoritmo: cómo cambian las estimaciones Q de cada brazo pull a pull (media entre "
        "corridas) frente a su valor real μ (línea punteada). Solo los brazos que se jalan a menudo convergen "
        "a su μ; en el optimista todas las Q parten de Q₀ y bajan hasta que el mejor brazo se separa.",
    ),
    PlotSpec(
        "sensitivity", "Sensibilidad a los hiperparámetros", plot_sensitivity,
        "Recompensa media final (último 10 % de pulls) al variar ε, Q₀, c y τ, cada uno en su algoritmo "
        "(media ± 1 std). Una forma de U invertida refleja el dilema exploración–explotación; la línea gris "
        "vertical marca el valor actual de la configuración para ver si está cerca del óptimo.",
    ),
    PlotSpec(
        "lambda_effect", "Efecto de λ en la precisión", plot_lambda_effect,
        "Precisión de posición y de pitch frente a λ, el peso de la penalización de tocabilidad, para cada "
        "algoritmo y para los dos oráculos (miope y de cadena). λ es lo único que separa posiciones con el mismo pitch, así que debería "
        "mover la precisión de posición y dejar casi igual la de pitch. Requiere ground truth.",
    ),
    PlotSpec(
        "accuracy", "Precisión de la transcripción", plot_accuracy,
        "Porcentaje de notas del ground truth con el pitch correcto y con la posición (cuerda y traste) "
        "correcta, por algoritmo (media ± 1 std). Líneas de referencia: el oráculo miope (argmax μ en cada "
        "segmento; techo de recompensa, no de precisión: un bandit puede superarlo en posición si el modelo de "
        "tocabilidad no coincide con la digitación real) y el oráculo de cadena (óptimo de la suma de μ de toda "
        "la pista con el mismo modelo; tampoco es un techo de precisión). Si el oráculo falla el PITCH, el error "
        "está en la recompensa, la segmentación o pYIN.",
    ),
    PlotSpec(
        "runtime", "Tiempo de ejecución", plot_runtime,
        "Milisegundos que tarda cada algoritmo en resolver un segmento (T pulls de select_arm + pull + update), "
        "media ± 1 std entre corridas. Refleja el coste de decidir: un argmax (ε-greedy), un bono por brazo "
        "(UCB1) o exponenciar y muestrear una distribución (Softmax).",
    ),
    PlotSpec(
        "tab_heatmap", "Aciertos por nota", plot_tab_heatmap,
        "Cada columna es una nota del ground truth (etiqueta = posición real) y cada fila un algoritmo; la "
        "celda muestra la posición más elegida entre corridas y su color, si acertó la posición, solo el pitch, "
        "falló el pitch o la nota no se detectó. Errores comunes a todas las filas apuntan a la recompensa o a pYIN.",
    ),
]


def save_figure(fig: Figure, out_dir: str | Path, name: str, formats: tuple[str, ...] = ("png", "pdf"), dpi: int = 150) -> list[Path]:
    """Guarda ``fig`` como ``<out_dir>/<name>.<fmt>`` para cada formato.

    Parameters
    ----------
    fig : Figure
        Figura a guardar.
    out_dir : str | Path
        Carpeta de destino (se crea si no existe, con sus padres).
    name : str
        Nombre del archivo sin extensión.
    formats : tuple[str, ...]
        Formatos (``"png"``, ``"pdf"``, ``"svg"``...).
    dpi : int
        Resolución de los formatos rasterizados (puntos por pulgada).

    Returns
    -------
    list[Path]
        Rutas escritas, en el orden de ``formats``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        ext = fmt.lower().lstrip(".")
        path = out / f"{name}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi, facecolor=fig.get_facecolor())
        paths.append(path)
    logger.info("Gráfica «%s» guardada en %s (%s)", name, out, ", ".join(f.lower().lstrip(".") for f in formats))
    return paths


def save_all_plots(res: ExperimentResult, analysis: AnalysisResult | None, out_dir: str | Path, formats: tuple[str, ...] = ("png", "pdf"),
                   dpi: int = 150) -> list[Path]:
    """Genera y guarda todas las gráficas comparativas (+ espectrograma si hay ``analysis``).

    Parameters
    ----------
    res : ExperimentResult
        Resultado del experimento.
    analysis : AnalysisResult | None
        Análisis de la pista; si se da, se guarda también ``spectrogram``.
    out_dir : str | Path
        Carpeta de destino (se crea si no existe).
    formats : tuple[str, ...]
        Formatos de cada gráfica.
    dpi : int, optional
        Resolución de los formatos rasterizados (puntos por pulgada).

    Returns
    -------
    list[Path]
        Todas las rutas escritas. Las gráficas a las que les falta un dato
        (p. ej. precisión sin ground truth) se omiten y se registra el motivo.
    """
    paths: list[Path] = []
    for spec in COMPARISON_PLOTS:
        msg = missing_data_message(spec.key, res)
        if msg:
            logger.info("Se omite la gráfica «%s»: %s", spec.title, msg)
            continue
        paths += save_figure(spec.func(res), out_dir, spec.key, formats, dpi=dpi)
    if analysis is not None:
        paths += save_figure(plot_spectrogram(analysis), out_dir, "spectrogram", formats, dpi=dpi)
    logger.info("Exportadas %d gráficas (%d archivos) a %s", len(paths) // max(len(formats), 1), len(paths), out_dir)
    return paths
