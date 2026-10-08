"""Pestaña 4 · Tablatura: la salida final del sistema, nota a nota.

Papel en la interfaz
--------------------
Muestra la tablatura de bajo que produjo UN algoritmo bandit (una corrida con
la semilla de la configuración; decisión de diseño 5) y permite explicar en
clase, para cada nota, POR QUÉ el agente eligió esa posición. No calcula nada
por su cuenta: lee ``app.state.transcriptions[alg]`` (un
:class:`src.experiments.TranscriptionResult` por algoritmo), compara con el
ground truth mediante :func:`src.experiments.match_segments_to_gt` y
:func:`src.experiments.score_arms`, dibuja el espectro con
:func:`src.plots.plot_note_spectrum`, genera el texto con
:func:`src.tab.render_ascii` y exporta con :func:`src.tab.export_tab`.

Contenido::

    ┌ Algoritmo (●) ε-greedy (○) Optimista (○) UCB1 (○) Softmax   ☑ Comparar con ground truth
    ├ Zoom [−] ──●── [+] 160 px/s [Ajustar]  ☐ Vista ASCII        Exportar: [TXT] [JSON] [CSV] [Exportar las 4…]
    ├ Aviso (solo si cambió la configuración del entorno o de los agentes: [Re-transcribir])
    ├ Resumen: «UCB1 · 16 notas · pitch 16/16 (100 %) · posición 12/16 (75 %)»     leyenda ✓ ~ ✗ + ⬚
    ├ Tablatura gráfica (desplazamiento horizontal)          ─ o ─  Vista ASCII (texto monoespaciado)
    │   t (s)   0    1    2    3 …                     regla de tiempo
    │     G|───────────────────────────               tablatura del algoritmo: un recuadro por nota
    │     D|──────[2]✓────────────────                 (traste) en x = inicio del segmento; la barra
    │     A|─[0]✓─────[4]~────────────                 fina que sigue al recuadro es su duración
    │     E|───────────────────────────
    │   Ground truth · posiciones reales                 pauta tenue con las notas reales (.gt.json)
    └ Nota seleccionada: «Segmento 3 · 1.80–2.38 s · f0 pYIN 55.1 Hz (A1) · traste previo 0 · elegido A-0 · GT A-0 ✓»
        espectro del segmento + template armónico │ tabla de brazos candidatos (μ, Q final, pulls)

Lectura de la comparación con el ground truth
---------------------------------------------
Cada segmento detectado se empareja uno a uno con una nota real (inicio a ≤
``experiment.onset_tolerance_s``) y recibe uno de estos estados, con su color
de :data:`src.plots.OUTCOME_COLORS` y un símbolo (la información nunca depende
solo del color):

* ✓ posición exacta (cuerda y traste correctos);
* ~ pitch correcto en otra posición (p. ej. A-0 en vez de E-5: lo único que
  las separa es la penalización de tocabilidad λ·|Δtraste|/12);
* ✗ pitch incorrecto (error de la recompensa, de pYIN o de la segmentación);
* + segmento sin nota real (onset falso o nota partida en dos).

En la pauta del ground truth, las notas reales sin segmento («no detectadas»)
se dibujan con borde discontinuo.

Sincronización con las demás pestañas
-------------------------------------
* ``analysis_ready`` / ``transcriptions_ready`` → redibuja (los dos eventos
  llegan seguidos y se agrupan en un solo redibujo con ``after_idle``).
* ``algorithm_selected`` → cambia el algoritmo mostrado (y los botones de radio).
* ``segment_selected`` de otra pestaña → resalta la nota, desplaza la vista
  hasta ella y actualiza el panel inferior. Las selecciones de esta pestaña se
  publican con ``source="tablature"``; como
  :meth:`gui.app.AppState.select_segment` ignora la posición ya seleccionada y
  aquí solo se actúa si la vista no está sincronizada, no hay bucles.
* ``config_changed`` → recalcula el emparejamiento si cambió la tolerancia de
  onset y muestra el aviso «re-transcribir» si cambiaron el entorno, los
  agentes o la semilla.
* ``busy_changed`` → deshabilita la exportación mientras hay una tarea en curso.

La gráfica del espectro (≈ 0.1–0.3 s) se dibuja de forma diferida (agrupa
selecciones rápidas, p. ej. al recorrer las notas con ← y →) y solo cuando la
pestaña está visible.
"""

from __future__ import annotations

import logging
import math
import tkinter as tk
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter import font as tkfont
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from gui.widgets import UI_ACCENT, PlotFrame, Tooltip, show_error
from src import plots
from src.config import ALGO_COLORS, ALGO_LABELS, ALGORITHMS, RESULTS_DIR
from src.environment import Arm
from src.experiments import (
    OUTCOME_EXACT,
    OUTCOME_MISSED,
    OUTCOME_WRONG_PITCH,
    OUTCOME_WRONG_POSITION,
    AccuracyResult,
    match_segments_to_gt,
    score_arms,
)
from src.pitch import midi_to_name
from src.tab import TAB_STRINGS, TabNote, export_tab, render_ascii

if TYPE_CHECKING:  # solo para las anotaciones de tipo (evita importes circulares)
    from gui.app import SmartunerApp
    from src.experiments import TranscriptionResult
    from src.pipeline import AnalysisResult
    from src.synth_dataset import GTNote

logger = logging.getLogger(__name__)

#: Identificador de esta pestaña como origen de los eventos ``segment_selected``.
SOURCE = "tablature"

#: Estado propio de un segmento que no tiene nota real emparejada (no es un ``OUTCOME_*``
#: porque esos códigos describen notas del ground truth, no segmentos).
OUTCOME_EXTRA = -1

#: Símbolo de cada estado (la identidad nunca depende solo del color).
OUTCOME_SYMBOLS: dict[int, str] = {
    OUTCOME_EXACT: "✓",
    OUTCOME_WRONG_POSITION: "~",
    OUTCOME_WRONG_PITCH: "✗",
    OUTCOME_EXTRA: "+",
    OUTCOME_MISSED: "",
}

#: Descripción corta de cada estado (leyenda, tooltips y panel inferior).
OUTCOME_TEXT: dict[int, str] = {
    OUTCOME_EXACT: "Posición exacta",
    OUTCOME_WRONG_POSITION: "Pitch correcto, otra posición",
    OUTCOME_WRONG_PITCH: "Pitch incorrecto",
    OUTCOME_EXTRA: "Segmento sin nota real",
    OUTCOME_MISSED: "Nota real no detectada",
}

#: Color de relleno de cada estado (los de :data:`src.plots.OUTCOME_COLORS`; el segmento de más usa
#: el gris de «no detectada», porque tampoco tiene pareja).
OUTCOME_FILL: dict[int, str] = {
    OUTCOME_EXACT: plots.OUTCOME_COLORS[OUTCOME_EXACT],
    OUTCOME_WRONG_POSITION: plots.OUTCOME_COLORS[OUTCOME_WRONG_POSITION],
    OUTCOME_WRONG_PITCH: plots.OUTCOME_COLORS[OUTCOME_WRONG_PITCH],
    OUTCOME_EXTRA: plots.OUTCOME_COLORS[OUTCOME_MISSED],
    OUTCOME_MISSED: plots.OUTCOME_COLORS[OUTCOME_MISSED],
}

#: Ayuda de cada estado (tooltips de la leyenda).
OUTCOME_HELP: dict[int, str] = {
    OUTCOME_EXACT: "El algoritmo eligió la misma cuerda y el mismo traste que la nota real.",
    OUTCOME_WRONG_POSITION: (
        "El pitch es correcto pero en otra posición del diapasón (p. ej. A-0 en vez de E-5). Las dos posiciones "
        "tienen la misma saliencia armónica: solo las separa la penalización de tocabilidad λ·|Δtraste|/12, así "
        "que el error viene de una brecha Δ muy pequeña o de que el modelo de tocabilidad no coincide con la "
        "digitación real."),
    OUTCOME_WRONG_PITCH: (
        "El algoritmo eligió una nota distinta de la real. Si el oráculo (argmax μ) también falla, el problema está "
        "en la recompensa, en pYIN (poda ±k) o en la segmentación; si no, el agente no exploró lo suficiente."),
    OUTCOME_EXTRA: (
        "Segmento detectado sin nota real a menos de la tolerancia de onset: un onset falso o una nota partida en "
        "dos segmentos. No cuenta en el recall, pero sí en la precisión por segmento."),
    OUTCOME_MISSED: (
        "(En la pauta del ground truth, borde discontinuo.) Nota real sin ningún segmento cuyo inicio esté a menos "
        "de la tolerancia de onset: la segmentación no la detectó y cuenta como error."),
}

#: Zoom horizontal: límites y valor inicial (píxeles por segundo).
MIN_ZOOM = 40.0
MAX_ZOOM = 800.0
DEFAULT_ZOOM = 160.0
#: Factor de los botones − / + (y de Ctrl + rueda del ratón).
ZOOM_STEP = 1.25

#: Separación mínima (px) entre dos marcas con número de la regla de tiempo.
MIN_TICK_PX = 56.0
#: Pasos candidatos de la regla de tiempo (s).
TICK_STEPS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0)

#: x (px) del instante 0 dentro del lienzo y margen derecho tras el final del audio.
X0 = 18
RIGHT_PAD = 36
#: Ancho (px) de la columna fija con los nombres de las cuerdas.
GUTTER_W = 64

#: Altura (px) de la pestaña por debajo de la cual se usa la geometría compacta.
COMPACT_TAB_HEIGHT = 650

#: Ancho (px) del panel inferior por debajo del cual la tabla de brazos oculta la columna «Nota».
NARROW_DETAIL_PX = 1100

#: Alto mínimo (pulgadas) de la figura del espectro para mostrar su título y su subtítulo.
SPECTRUM_TITLE_MIN_IN = 2.6
SPECTRUM_SUBTITLE_MIN_IN = 3.8

#: Fracción máxima de la altura de la pestaña que ocupa la vista ASCII.
ASCII_MAX_FRACTION = 0.45

#: Color de la banda que marca la duración del segmento seleccionado.
SELECTION_BAND = "#e3eefc"
#: Líneas verticales de la regla (muy tenues, detrás de las notas).
GUIDE_COLOR = "#efeee9"
#: Relleno de los recuadros sin comparación y del lienzo.
BOX_FILL = "#ffffff"
#: Aviso «re-transcribir» (ámbar muy claro, tinta oscura).
BANNER_BG = "#fdf3d8"
BANNER_FG = "#5c4400"

#: Formatos de exportación: extensión → nombre legible.
EXPORT_FORMATS: dict[str, str] = {"txt": "Texto (TXT)", "json": "JSON", "csv": "CSV"}

#: Explicación breve de cada algoritmo (tooltips de los botones de radio).
ALGO_HELP: dict[str, str] = {
    "egreedy": "ε-greedy: con probabilidad ε explora un brazo al azar; si no, juega el de mayor Q. "
               "Q se actualiza con la media incremental Q ← Q + (r − Q)/n.",
    "optimistic": "ε-greedy optimista: todas las Q empiezan en Q₀ (mayor que cualquier recompensa) y se "
                  "actualizan con α constante, Q ← Q + α·(r − Q). El optimismo obliga a probar cada brazo al "
                  "principio aunque ε sea 0.",
    "ucb1": "UCB1: juega argmax [Q(a) + c·√(ln t / n(a))]. El bono favorece los brazos poco probados "
            "(optimismo ante la incertidumbre) y se reduce al jalarlos.",
    "softmax": "Softmax (Boltzmann): elige el brazo a con probabilidad P(a) = exp(Q(a)/τ) / Σ exp(Q(b)/τ). "
               "τ alta = casi uniforme (explora); τ baja = casi greedy.",
}

#: Columnas de la tabla de brazos: (id, encabezado, ancho px, alineación, ayuda).
ARM_COLUMNS: tuple[tuple[str, str, int, str, str], ...] = (
    ("arm", "Brazo", 50, "center",
     "Posición candidata cuerda-traste (A-0 = cuerda La al aire). Solo se consideran las posiciones a ±k "
     "semitonos de la f0 de pYIN (poda)."),
    ("freq", "f (Hz)", 54, "center",
     "Frecuencia fundamental del brazo: f = f_cuerda · 2^(traste/12)."),
    ("note", "Nota", 44, "center", "Nombre de la nota del brazo (C4 = MIDI 60)."),
    ("mu", "μ real", 58, "center",
     "Recompensa esperada del brazo en este segmento: saliencia armónica media − λ·|traste − traste previo|/12, "
     "con el traste previo de la cadena de ESTE algoritmo. El agente no la conoce: solo ve muestras ruidosas."),
    ("q", "Q final", 58, "center",
     "Estimación Q(a) del agente al agotar los T pulls. Un brazo poco jalado conserva casi su valor inicial "
     "(Q₀ en el optimista)."),
    ("n", "Pulls", 48, "center",
     "Cuántas de las T veces jaló el agente este brazo. La posición recomendada es la más jalada "
     "(desempate por Q)."),
    ("role", "Papel", 170, "w",
     "elegido = la posición recomendada (la que va a la tablatura) · ★ óptimo = argmax μ (lo que elegiría un "
     "oráculo que conociera μ) · ● real = posición del ground truth."),
)

#: Mensaje guía antes de analizar un audio.
EMPTY_TITLE = "Todavía no hay ninguna tablatura"
EMPTY_TEXT = ("Abre un MP3 (Archivo → Abrir) o genera el dataset sintético. Al terminar el análisis, aquí "
              "aparecerá la tablatura de cada algoritmo, nota a nota.")
#: Texto del panel inferior cuando no hay nota seleccionada.
NO_SELECTION_TEXT = ("Ninguna nota seleccionada. Haz clic en una nota de la tablatura (o usa ← y → sobre ella) "
                     "para ver su espectro, el template armónico del brazo elegido y los brazos candidatos.")
#: Atajos de la tablatura gráfica (tooltip del lienzo y pista).
CANVAS_HINT = "Clic en una nota: ver sus detalles · ← / →: nota anterior / siguiente · rueda: desplazar · Ctrl + rueda: zoom"


# ---------------------------------------------------------------------------
# Utilidades puras (sin widgets; se prueban por separado)
# ---------------------------------------------------------------------------


def nice_time_step(px_per_s: float, min_px: float = MIN_TICK_PX) -> float:
    """Paso «redondo» de la regla de tiempo para que los números no se pisen.

    Parameters
    ----------
    px_per_s : float
        Zoom horizontal (píxeles por segundo).
    min_px : float, optional
        Separación mínima en píxeles entre dos marcas con número.

    Returns
    -------
    float
        El menor paso de :data:`TICK_STEPS` (s) que ocupa al menos ``min_px`` píxeles.

    Examples
    --------
    >>> nice_time_step(160.0), nice_time_step(40.0), nice_time_step(800.0)
    (0.5, 2.0, 0.1)
    """
    for step in TICK_STEPS:
        if step * px_per_s >= min_px:
            return step
    return TICK_STEPS[-1]


def format_seconds(t: float, step: float) -> str:
    """Número de una marca de la regla (sin ceros inútiles).

    Parameters
    ----------
    t : float
        Instante (s).
    step : float
        Paso de la regla (s): decide los decimales.

    Returns
    -------
    str
        p. ej. ``"2"`` con paso 1 s o ``"1.25"`` con paso 0.25 s.

    Examples
    --------
    >>> format_seconds(2.0, 1.0), format_seconds(1.5, 0.5), format_seconds(0.25, 0.25)
    ('2', '1.5', '0.25')
    """
    decimals = 0 if step >= 1 else 1 if math.isclose(step * 10, round(step * 10)) else 2
    return f"{t:.{decimals}f}"


def outcome_of_segments(n_segments: int, matches: list[int | None], outcomes: np.ndarray) -> list[int]:
    """Estado de cada SEGMENTO a partir de los estados de las notas del ground truth.

    Parameters
    ----------
    n_segments : int
        Número de segmentos conservados (notas de la tablatura).
    matches : list[int | None]
        Para cada nota GT, la posición del segmento emparejado (salida de
        :func:`src.experiments.match_segments_to_gt`).
    outcomes : np.ndarray
        Código ``OUTCOME_*`` de cada nota GT (:attr:`AccuracyResult.outcomes`).

    Returns
    -------
    list[int]
        Un código por segmento: el de su nota GT, u :data:`OUTCOME_EXTRA` si
        ninguna nota real se emparejó con él.

    Examples
    --------
    >>> outcome_of_segments(3, [0, None, 2], np.array([3, 0, 1]))
    [3, -1, 1]
    """
    result = [OUTCOME_EXTRA] * n_segments
    for k, pos in enumerate(matches):
        if pos is not None and 0 <= pos < n_segments:
            result[pos] = int(outcomes[k])
    return result


def detail_text(position: int, start_s: float, end_s: float, f0_hz: float | None, prev_fret: int | None,
                chosen: str, gt_label: str | None = None, outcome: int | None = None) -> str:
    """Línea de resumen de la nota seleccionada (panel inferior).

    Parameters
    ----------
    position : int
        Posición del segmento entre los conservados.
    start_s, end_s : float
        Inicio y fin del segmento (s).
    f0_hz : float | None
        f0 estimada por pYIN (Hz) o None si es desconocida.
    prev_fret : int | None
        Traste previo con el que se calculó μ (posición de la mano tras la nota anterior).
    chosen : str
        Etiqueta del brazo elegido (``"A-0"``).
    gt_label : str | None, optional
        Etiqueta de la posición real (None si no hay comparación).
    outcome : int | None, optional
        Estado del segmento (``OUTCOME_*`` u :data:`OUTCOME_EXTRA`); None si no hay comparación.

    Returns
    -------
    str
        Texto con los datos separados por « · ».

    Examples
    --------
    >>> detail_text(3, 1.8, 2.38, 55.1, 0, "A-0", "A-0", OUTCOME_EXACT)
    'Segmento 3 · 1.80–2.38 s · f0 pYIN 55.1 Hz (A1) · traste previo 0 · elegido A-0 · GT A-0 ✓'
    >>> detail_text(0, 0.5, 0.9, None, None, "E-5", None, OUTCOME_EXTRA)
    'Segmento 0 · 0.50–0.90 s · f0 pYIN desconocida · sin traste previo · elegido E-5 · sin nota real (segmento de más)'
    """
    if f0_hz is not None and f0_hz > 0:
        midi = 69.0 + 12.0 * math.log2(f0_hz / 440.0)
        f0 = f"f0 pYIN {f0_hz:.1f} Hz ({midi_to_name(int(round(midi)))})"
    else:
        f0 = "f0 pYIN desconocida"
    prev = f"traste previo {prev_fret}" if prev_fret is not None else "sin traste previo"
    parts = [f"Segmento {position}", f"{start_s:.2f}–{end_s:.2f} s", f0, prev, f"elegido {chosen}"]
    if outcome == OUTCOME_EXTRA:
        parts.append("sin nota real (segmento de más)")
    elif gt_label is not None:
        symbol = OUTCOME_SYMBOLS.get(outcome, "") if outcome is not None else ""
        parts.append(f"GT {gt_label} {symbol}".rstrip())
    return " · ".join(parts)


def gt_tab_notes(gt: list[GTNote]) -> list[TabNote]:
    """Convierte las notas del ground truth en :class:`src.tab.TabNote` (para la vista ASCII).

    Parameters
    ----------
    gt : list[GTNote]
        Notas reales.

    Returns
    -------
    list[TabNote]
        Una nota por nota real, con ``position`` = su índice en el ground truth.
    """
    return [TabNote(position=k, start_s=float(n.onset_s), end_s=float(n.offset_s), string=n.string,
                    fret=int(n.fret), midi=int(n.midi)) for k, n in enumerate(gt)]


def _mix(color: str, other: str, amount: float) -> str:
    """Mezcla lineal de dos colores ``#rrggbb`` (``amount`` = peso de ``color``)."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(amount * x + (1 - amount) * y):02x}" for x, y in zip(a, b))


def _luminance(color: str) -> float:
    """Luminancia relativa WCAG de un color ``#rrggbb`` (0 = negro, 1 = blanco)."""
    channels = []
    for i in (1, 3, 5):
        c = int(color[i:i + 2], 16) / 255.0
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _ink_on(color: str) -> str:
    """Tinta legible (casi negra o blanca) sobre un fondo ``color``."""
    return plots.TEXT_PRIMARY if _luminance(color) > 0.4 else "#ffffff"


def _short_legend_label(label: str) -> str:
    """Versión corta de una etiqueta de la leyenda de :func:`src.plots.plot_note_spectrum`.

    Examples
    --------
    >>> _short_legend_label("h·f del brazo elegido A-4 (69.3 Hz)")
    'h·f elegido A-4'
    >>> _short_legend_label("(h−½)·f del ground truth")
    '(h−½)·f real'
    """
    rules = (
        ("Espectro medio", lambda rest: "Espectro medio"),
        ("h·f del brazo elegido ", lambda rest: "h·f elegido " + rest.split(" (")[0]),
        ("(h−½)·f del elegido", lambda rest: "(h−½)·f elegido (−β)"),
        ("h·f del ground truth ", lambda rest: "h·f real " + rest.split(" (")[0]),
        ("(h−½)·f del ground truth", lambda rest: "(h−½)·f real"),
    )
    for prefix, short in rules:
        if label.startswith(prefix):
            return short(label[len(prefix):])
    return label


def note_spectrum_figure(analysis: AnalysisResult, position: int, arm: Arm, gt_arm: Arm | None = None,
                         fig: Figure | None = None) -> Figure:
    """:func:`src.plots.plot_note_spectrum` adaptada al panel inferior de la pestaña (más pequeño).

    La gráfica de ``src.plots`` está pensada para una figura de ≈ 9 × 5.5
    pulgadas; en el panel inferior (≈ 5–8 × 2–4 pulgadas) la leyenda de tres
    columnas no cabe y la etiqueta vertical larga invade el título. Aquí solo se
    ajusta la PRESENTACIÓN (los datos son los mismos):

    * etiqueta del eje y más corta;
    * en figuras bajas se ocultan el subtítulo y, si hace falta, el título
      (sus datos ya están en la línea de resumen del panel);
    * la leyenda usa etiquetas cortas en figuras pequeñas y tantas columnas
      como quepan en el ancho (medido con el renderizador).

    Parameters
    ----------
    analysis : AnalysisResult
        Análisis del audio.
    position : int
        Segmento.
    arm : Arm
        Brazo elegido por el algoritmo.
    gt_arm : Arm | None, optional
        Posición real (ground truth).
    fig : Figure | None, optional
        Figura a reutilizar (la del :class:`gui.widgets.PlotFrame`).

    Returns
    -------
    Figure
        La figura dibujada.
    """
    fig = plots.plot_note_spectrum(analysis, position, arm, gt_arm=gt_arm, fig=fig)
    width_in, height_in = fig.get_size_inches()
    small = height_in < SPECTRUM_SUBTITLE_MIN_IN or width_in < 7.0
    for ax in fig.axes:
        if ax.get_ylabel():
            ax.yaxis.label.set_text("Magnitud (máx. = 1)" if height_in >= 3.0 else "Magnitud")
    hide_subtitle = height_in < SPECTRUM_SUBTITLE_MIN_IN
    hide_title = height_in < SPECTRUM_TITLE_MIN_IN
    for text in fig.texts:
        if text.get_text().startswith("Espectro del segmento"):
            if hide_subtitle:  # el título reserva líneas en blanco para el subtítulo: se quitan
                text.set_text(text.get_text().rstrip("\n"))
            text.set_fontsize(12 if width_in >= 6.5 else 11)
            text.set_visible(not hide_title)
            text.set_in_layout(not hide_title)
    for artist in fig.artists:
        if isinstance(artist, Annotation):
            artist.set_visible(not hide_subtitle)
    _fit_legend(fig, short_labels=small)
    return fig


def _fit_legend(fig: Figure, short_labels: bool) -> None:
    """Rehace la leyenda de figura con el mayor número de columnas que quepa en su ancho.

    ``Legend.set_ncols`` no recoloca una leyenda ya construida, así que se crea
    de nuevo con los mismos elementos y el estilo de :mod:`src.plots`.
    """
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # lienzo sin renderizador Agg: se deja como está
        return
    available = 0.97 * fig.get_figwidth() * fig.dpi
    for legend in list(fig.legends):
        handles = list(legend.legend_handles)
        labels = [t.get_text() for t in legend.get_texts()]
        if short_labels:
            labels = [_short_legend_label(label) for label in labels]
        legend.remove()
        for ncols in range(len(labels), 0, -1):
            new = fig.legend(handles, labels, loc="outside lower center", ncols=ncols, frameon=False,
                             fontsize=plots.LEGEND_SIZE, handlelength=2.6, labelcolor=plots.TEXT_PRIMARY,
                             columnspacing=1.4)
            if ncols == 1 or new.get_window_extent(renderer).width <= available:
                break
            new.remove()


# ---------------------------------------------------------------------------
# Geometría del lienzo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaffGeometry:
    """Medidas (px) de la tablatura gráfica.

    Attributes
    ----------
    ruler_h : int
        Alto de la regla de tiempo.
    badge_room : int
        Espacio libre sobre la primera cuerda (para los símbolos de estado).
    string_gap : int
        Distancia entre cuerdas de la tablatura del algoritmo.
    box_h : int
        Alto de los recuadros de las notas.
    between : int
        Separación entre las dos pautas (incluye el título «Ground truth»).
    gt_gap : int
        Distancia entre cuerdas de la pauta del ground truth.
    gt_box_h : int
        Alto de los recuadros del ground truth.
    bottom : int
        Margen inferior.
    font_size, small_font_size : int
        Tamaños de letra (puntos) de los trastes y de los textos secundarios.
    """

    ruler_h: int
    badge_room: int
    string_gap: int
    box_h: int
    between: int
    gt_gap: int
    gt_box_h: int
    bottom: int
    font_size: int
    small_font_size: int


#: Geometría normal (ventanas de ≈ 1360×880) y compacta (≈ 1024×700).
NORMAL_GEOMETRY = StaffGeometry(ruler_h=24, badge_room=10, string_gap=27, box_h=20, between=36,
                                gt_gap=17, gt_box_h=15, bottom=12, font_size=10, small_font_size=8)
COMPACT_GEOMETRY = StaffGeometry(ruler_h=20, badge_room=8, string_gap=21, box_h=18, between=28,
                                 gt_gap=13, gt_box_h=12, bottom=6, font_size=9, small_font_size=7)


@dataclass(frozen=True)
class StaffLayout:
    """Posiciones verticales calculadas a partir de una :class:`StaffGeometry`.

    Attributes
    ----------
    main_lines : dict[str, float]
        y de cada cuerda de la tablatura del algoritmo (G arriba, E abajo).
    gt_lines : dict[str, float]
        y de cada cuerda de la pauta del ground truth (vacío si no se muestra).
    gt_heading_y : float | None
        y del título «Ground truth».
    height : int
        Alto total del lienzo.
    """

    main_lines: dict[str, float]
    gt_lines: dict[str, float]
    gt_heading_y: float | None
    height: int


def staff_layout(geometry: StaffGeometry, show_gt: bool) -> StaffLayout:
    """Calcula la posición vertical de las cuerdas de las dos pautas.

    Parameters
    ----------
    geometry : StaffGeometry
        Medidas de la tablatura.
    show_gt : bool
        Si se dibuja la pauta del ground truth debajo.

    Returns
    -------
    StaffLayout
        Posiciones y alto total.

    Examples
    --------
    >>> lay = staff_layout(NORMAL_GEOMETRY, show_gt=False)
    >>> list(lay.main_lines), lay.main_lines["E"] - lay.main_lines["G"] == 3 * NORMAL_GEOMETRY.string_gap
    (['G', 'D', 'A', 'E'], True)
    """
    g = geometry
    first = g.ruler_h + g.badge_room + g.box_h / 2 + 2
    main = {s: first + i * g.string_gap for i, s in enumerate(TAB_STRINGS)}
    bottom_main = main[TAB_STRINGS[-1]] + g.box_h / 2
    if not show_gt:
        return StaffLayout(main, {}, None, int(math.ceil(bottom_main + g.bottom + 4)))
    heading = bottom_main + g.between * 0.45
    gt_first = bottom_main + g.between + g.gt_box_h / 2
    gt = {s: gt_first + i * g.gt_gap for i, s in enumerate(TAB_STRINGS)}
    height = gt[TAB_STRINGS[-1]] + g.gt_box_h / 2 + g.bottom
    return StaffLayout(main, gt, heading, int(math.ceil(height)))


def _round_rect(canvas: tk.Canvas, x0: float, y0: float, x1: float, y1: float, radius: float = 4.0,
                **kwargs: Any) -> int:
    """Rectángulo con esquinas redondeadas (polígono suavizado) en ``canvas``; devuelve su id."""
    r = max(0.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    points = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
              x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# ---------------------------------------------------------------------------
# Tooltip según la zona bajo el puntero
# ---------------------------------------------------------------------------


class _HoverTooltip(Tooltip):
    """Tooltip cuyo texto depende de lo que hay bajo el ratón (una nota del lienzo, un encabezado).

    Parameters
    ----------
    widget : tk.Widget
        Widget al que se asocia.
    key_for : Callable[[int, int], Hashable | None]
        Función ``(x, y)`` (coordenadas del widget) → identificador de la zona
        (None = nada que explicar).
    text_for : Callable[[Hashable], str]
        Texto para una zona.
    """

    def __init__(self, widget: tk.Widget, key_for: Callable[[int, int], Hashable | None],
                 text_for: Callable[[Hashable], str]) -> None:
        """Asocia el tooltip a ``widget`` y sigue el movimiento del ratón."""
        self._key_for = key_for
        self._text_for = text_for
        self._hover: Hashable | None = None
        super().__init__(widget, self._current_text, delay_ms=350, wraplength=360)
        widget.bind("<Motion>", self._on_motion, add="+")

    def hide(self) -> None:
        """Oculta el tooltip (p. ej. al redibujar el lienzo: la nota bajo el puntero cambió)."""
        self._hover = None
        self._hide()

    def _pointer_key(self) -> Hashable | None:
        """Zona bajo el puntero (o None)."""
        w = self.widget
        try:
            return self._key_for(w.winfo_pointerx() - w.winfo_rootx(), w.winfo_pointery() - w.winfo_rooty())
        except tk.TclError:
            return None

    def _current_text(self) -> str:
        """Texto de la zona actual (cadena vacía = no mostrar nada)."""
        key = self._pointer_key()
        return self._text_for(key) if key is not None else ""

    def _on_motion(self, event: tk.Event) -> None:
        """Reinicia la espera cuando el ratón pasa a otra zona."""
        key = self._key_for(event.x, event.y)
        if key != self._hover:
            self._hover = key
            self._hide()
            if key is not None:
                self._schedule()

    def _show(self) -> None:
        """Muestra el texto junto al puntero, sin salirse de la pantalla."""
        text = self._current_text()
        if not text or self._tip is not None:
            return
        w = self.widget
        self._tip = tip = tk.Toplevel(w)
        tip.withdraw()
        tip.wm_overrideredirect(True)
        tk.Label(tip, text=text, justify="left", wraplength=self.wraplength, background="#fffbe8",
                 foreground=plots.TEXT_PRIMARY, relief="solid", borderwidth=1, padx=8, pady=6,
                 font=("TkDefaultFont", 9)).pack()
        tip.update_idletasks()
        x = w.winfo_pointerx() + 14
        y = w.winfo_pointery() + 18
        if y + tip.winfo_reqheight() > w.winfo_screenheight():
            y = w.winfo_pointery() - tip.winfo_reqheight() - 10
        x = min(x, max(0, w.winfo_screenwidth() - tip.winfo_reqwidth() - 4))
        tip.wm_geometry(f"+{x}+{y}")
        tip.deiconify()


# ---------------------------------------------------------------------------
# Pestaña
# ---------------------------------------------------------------------------


class TablatureTab(ttk.Frame):
    """Pestaña 4 · Tablatura: tablatura gráfica/ASCII de un algoritmo, comparación y exportación.

    Parameters
    ----------
    master : tk.Misc
        Contenedor (el ``ttk.Notebook`` de la ventana principal).
    app : SmartunerApp
        Aplicación: da acceso al estado compartido (``app.state``), a la acción
        ``retranscribe`` y a la barra de estado.

    Attributes
    ----------
    algorithm : str
        Algoritmo mostrado (clave de :data:`src.config.ALGORITHMS`).
    selected : int | None
        Segmento seleccionado (posición entre los conservados).
    zoom : float
        Zoom horizontal en píxeles por segundo.
    view_state : str
        Qué muestra la tablatura: ``"empty"`` (mensaje guía), ``"busy"``
        (analizando), ``"message"`` (aviso, p. ej. sin notas) o ``"tab"``.
    compact : bool
        True si se usa la geometría compacta (pestaña de menos de
        :data:`COMPACT_TAB_HEIGHT` px de alto).
    canvas : tk.Canvas
        Tablatura gráfica (con desplazamiento horizontal).
    gutter : tk.Canvas
        Columna fija con los nombres de las cuerdas.
    ascii_text : tk.Text
        Vista ASCII.
    plot : PlotFrame
        Espectro de la nota seleccionada.
    tree : ttk.Treeview
        Brazos candidatos de la nota seleccionada (filas ``"arm<i>"``).
    algo_var, compare_var, ascii_var : tk.Variable
        Estado de los controles de la barra.
    export_buttons : dict[str, ttk.Button]
        Botones de exportación por formato (``"txt"``, ``"json"``, ``"csv"``).
    export_all_menu : tk.Menu
        Menú de «Exportar las 4…» (una entrada por formato y «los tres formatos»).
    """

    def __init__(self, master: tk.Misc, app: SmartunerApp) -> None:
        """Construye la pestaña, se suscribe a los eventos de ``app.state`` y muestra el estado actual."""
        super().__init__(master, padding=(12, 10, 12, 8))
        self.app = app
        self.analysis: AnalysisResult | None = None
        initial = app.state.selected_algorithm
        self.algorithm: str = initial if initial in ALGORITHMS else "ucb1"
        self.selected: int | None = None
        self.zoom: float = DEFAULT_ZOOM
        self.view_state = "empty"
        self.compact = False
        self._busy = bool(app.busy)
        self._layout = staff_layout(NORMAL_GEOMETRY, show_gt=False)
        self._content_width = 0
        self._note_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._note_spans: dict[int, tuple[float, float]] = {}
        self._gt_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._matches: list[int | None] = []
        self._segment_gt: dict[int, int] = {}
        self._segment_outcomes: list[int] = []
        self._accuracy: AccuracyResult | None = None
        self._refresh_job: str | None = None
        self._detail_job: str | None = None
        self._ascii_job: str | None = None
        self._detail_dirty = False
        self._ascii_width = 0
        self._swatches: dict[str, tk.PhotoImage] = {}
        self._wrapped_info: bool | None = None
        self._syncing_scale = False
        self._narrow_table = False
        self._canvas_height = 150

        self._setup_fonts_and_styles()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        self._build_toolbar()
        self._build_banner()
        self._build_info_row()
        self._build_view()
        self._build_detail()

        events = app.state.events
        events.subscribe("analysis_ready", self._on_analysis_ready)
        events.subscribe("transcriptions_ready", self._on_transcriptions_ready)
        events.subscribe("algorithm_selected", self._on_algorithm_selected)
        events.subscribe("segment_selected", self._on_segment_selected)
        events.subscribe("config_changed", self._on_config_changed)
        events.subscribe("busy_changed", self._on_busy_changed)
        self.bind("<Map>", self._on_map, add="+")
        self.bind("<Configure>", self._on_tab_configure, add="+")

        self.analysis = app.state.analysis
        self.selected = app.state.selected_position
        self._recompute_comparison()
        self.refresh()

    # ================================================================ construcción
    def _setup_fonts_and_styles(self) -> None:
        """Fuentes del lienzo y estilos propios (prefijo ``Tabla.`` para no afectar a otras pestañas)."""
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        self._family = family
        self.font_fret = tkfont.Font(self, family=family, size=10, weight="bold")
        self.font_small = tkfont.Font(self, family=family, size=8)
        self.font_badge = tkfont.Font(self, family=family, size=8, weight="bold")
        self.font_label = tkfont.Font(self, family=family, size=10, weight="bold")
        self.font_heading = tkfont.Font(self, family=family, size=9)
        self.font_empty_title = tkfont.Font(self, family=family, size=13, weight="bold")
        self.font_empty = tkfont.Font(self, family=family, size=10)
        self.font_mono = tkfont.Font(self, family=tkfont.nametofont("TkFixedFont").actual("family"), size=10)
        self.font_tree_bold = tkfont.Font(self, family=family, size=9, weight="bold")
        self.font_summary = tkfont.Font(self, family=family, size=10)

        style = ttk.Style(self)
        style.configure("Tabla.TLabelframe", padding=(10, 4, 10, 8))
        style.configure("Tabla.TLabelframe.Label", font=(family, 10, "bold"), foreground=plots.TEXT_PRIMARY)
        style.configure("TablaKey.TLabel", foreground=plots.TEXT_SECONDARY)
        style.configure("TablaSummary.TLabel", foreground=plots.TEXT_PRIMARY, font=self.font_summary)
        style.configure("TablaHint.TLabel", foreground=plots.TEXT_SECONDARY, font=(family, 9))
        style.configure("TablaDetail.TLabel", foreground=plots.TEXT_PRIMARY, font=(family, 10, "bold"))
        style.configure("TablaBar.TButton", padding=(6, 3), width=0)
        style.configure("TablaBar.TMenubutton", padding=(6, 3), width=0)
        style.configure("TablaBanner.TFrame", background=BANNER_BG)
        style.configure("TablaBanner.TLabel", background=BANNER_BG, foreground=BANNER_FG, padding=(10, 6))
        style.configure("Tabla.Treeview", rowheight=21)
        style.configure("Tabla.Treeview.Heading", font=(family, 9, "bold"))

    def _swatch(self, color: str) -> tk.PhotoImage:
        """Cuadrado de color 12×12 px (identidad del algoritmo junto a su botón de radio)."""
        if color not in self._swatches:
            img = tk.PhotoImage(master=self, width=12, height=12)
            img.put(_mix(color, "#000000", 0.75), to=(0, 0, 12, 12))
            img.put(color, to=(1, 1, 11, 11))
            self._swatches[color] = img
        return self._swatches[color]

    def _build_toolbar(self) -> None:
        """Dos filas de controles: algoritmo + comparación; zoom + vista + exportación."""
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        bar.columnconfigure(0, weight=1)

        # --- fila 1: algoritmo y comparación con el ground truth
        row1 = ttk.Frame(bar)
        row1.grid(row=0, column=0, sticky="ew")
        label = ttk.Label(row1, text="Algoritmo:", style="TablaKey.TLabel")
        label.pack(side="left", padx=(0, 6))
        Tooltip(label, "Algoritmo cuya tablatura se muestra. Cada uno resolvió las mismas notas (los mismos "
                       "problemas bandit) con T pulls por nota y la semilla de la configuración. También se "
                       "cambia desde la pestaña «Ejecución en vivo».")
        self.algo_var = tk.StringVar(master=self, value=self.algorithm)
        self.algo_radios: dict[str, ttk.Radiobutton] = {}
        for alg in ALGORITHMS:
            radio = ttk.Radiobutton(row1, text=ALGO_LABELS[alg], image=self._swatch(ALGO_COLORS[alg]),
                                    compound="left", variable=self.algo_var, value=alg,
                                    command=self._on_algo_radio)
            radio.pack(side="left", padx=(0, 10))
            Tooltip(radio, lambda alg=alg: ALGO_HELP[alg] + "\n\n" + self._algo_status(alg))
            self.algo_radios[alg] = radio
        ttk.Separator(row1, orient="vertical").pack(side="left", fill="y", padx=(2, 12), pady=2)
        self.compare_var = tk.BooleanVar(master=self, value=True)
        self.chk_compare = ttk.Checkbutton(row1, text="Comparar con ground truth", variable=self.compare_var,
                                           command=self._on_compare_toggled)
        self.chk_compare.pack(side="left")
        Tooltip(self.chk_compare, self._compare_help)

        # --- fila 2: zoom, vista ASCII y exportación
        row2 = ttk.Frame(bar)
        row2.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        row2.columnconfigure(1, weight=1)
        left = ttk.Frame(row2)
        left.grid(row=0, column=0, sticky="w")
        zoom_label = ttk.Label(left, text="Zoom:", style="TablaKey.TLabel")
        zoom_label.pack(side="left", padx=(0, 4))
        zoom_help = ("Zoom horizontal de la tablatura gráfica, en píxeles por segundo de audio. Más zoom separa las "
                     "notas rápidas; menos zoom muestra más compases a la vez. También: Ctrl + rueda del ratón "
                     "sobre la tablatura.")
        Tooltip(zoom_label, zoom_help)
        self.btn_zoom_out = ttk.Button(left, text="−", width=2, command=lambda: self.set_zoom(self.zoom / ZOOM_STEP))
        self.btn_zoom_out.pack(side="left")
        Tooltip(self.btn_zoom_out, "Alejar (÷ 1.25).")
        self.zoom_scale = ttk.Scale(left, from_=math.log2(MIN_ZOOM), to=math.log2(MAX_ZOOM), orient="horizontal",
                                    length=130, command=self._on_zoom_scale)
        self.zoom_scale.set(math.log2(self.zoom))
        self.zoom_scale.pack(side="left", padx=4)
        Tooltip(self.zoom_scale, zoom_help)
        self.btn_zoom_in = ttk.Button(left, text="+", width=2, command=lambda: self.set_zoom(self.zoom * ZOOM_STEP))
        self.btn_zoom_in.pack(side="left")
        Tooltip(self.btn_zoom_in, "Acercar (× 1.25).")
        self.zoom_label = ttk.Label(left, text="", width=9, anchor="w", style="TablaKey.TLabel")
        self.zoom_label.pack(side="left", padx=(6, 0))
        self.btn_fit = ttk.Button(left, text="Ajustar", style="TablaBar.TButton", command=self.fit_zoom)
        self.btn_fit.pack(side="left", padx=(2, 0))
        Tooltip(self.btn_fit, "Elige el zoom para que toda la pista quepa en el ancho visible (si las notas no "
                              "quedan demasiado juntas).")
        ttk.Separator(left, orient="vertical").pack(side="left", fill="y", padx=12, pady=2)
        self.ascii_var = tk.BooleanVar(master=self, value=False)
        self.chk_ascii = ttk.Checkbutton(left, text="Vista ASCII", variable=self.ascii_var,
                                         command=self._on_ascii_toggled)
        self.chk_ascii.pack(side="left")
        Tooltip(self.chk_ascii, "Muestra la tablatura como texto monoespaciado (src.tab.render_ascii), igual que en "
                                "el archivo TXT exportado: una columna por nota, en orden, con su tiempo de inicio "
                                "encima. El ancho de línea se adapta al de la ventana.")
        self._update_zoom_label()

        right = ttk.Frame(row2)
        right.grid(row=0, column=2, sticky="e")
        export_label = ttk.Label(right, text="Exportar:", style="TablaKey.TLabel")
        export_label.pack(side="left", padx=(0, 4))
        Tooltip(export_label, "Guarda la tablatura del algoritmo mostrado (src.tab.export_tab).")
        self.export_buttons: dict[str, ttk.Button] = {}
        export_help = {
            "txt": "Tablatura ASCII (la misma que la vista ASCII) en un archivo de texto UTF-8.",
            "json": "JSON con los metadatos (algoritmo, audio, T, λ, semilla, precisión) y una entrada por nota: "
                    "posición, inicio/fin (s), cuerda, traste, MIDI, nombre de la nota y f0 de pYIN.",
            "csv": "Tabla CSV con una fila por nota: position, start_s, end_s, string, fret, midi, note, f0_hz "
                   "(se abre con una hoja de cálculo).",
        }
        for fmt in EXPORT_FORMATS:
            button = ttk.Button(right, text=fmt.upper(), style="TablaBar.TButton",
                                command=lambda fmt=fmt: self.export_current(fmt))
            button.pack(side="left", padx=(0, 4))
            Tooltip(button, export_help[fmt])
            self.export_buttons[fmt] = button
        self.export_all_button = ttk.Menubutton(right, text="Exportar las 4…", style="TablaBar.TMenubutton")
        self.export_all_menu = tk.Menu(self.export_all_button, tearoff=False)
        for fmt, name in EXPORT_FORMATS.items():
            self.export_all_menu.add_command(label=f"Como {name}…", command=lambda fmt=fmt: self.export_all((fmt,)))
        self.export_all_menu.add_separator()
        self.export_all_menu.add_command(label="En los tres formatos…",
                                         command=lambda: self.export_all(tuple(EXPORT_FORMATS)))
        self.export_all_button.configure(menu=self.export_all_menu)
        self.export_all_button.pack(side="left", padx=(4, 0))
        Tooltip(self.export_all_button, "Elige una carpeta y guarda la tablatura de los cuatro algoritmos, un archivo "
                                        "por algoritmo (<audio>_tab_<algoritmo>.<formato>), para compararlas o "
                                        "adjuntarlas al informe.")

    def _build_banner(self) -> None:
        """Aviso (oculto por defecto) de que la configuración cambió después de transcribir."""
        banner = ttk.Frame(self, style="TablaBanner.TFrame")
        banner.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        banner.columnconfigure(0, weight=1)
        self.banner_label = ttk.Label(
            banner, style="TablaBanner.TLabel", anchor="w",
            text="⚠  Cambiaste parámetros del entorno, de los agentes o la semilla después de transcribir: esta "
                 "tablatura todavía usa los anteriores.")
        self.banner_label.grid(row=0, column=0, sticky="ew")
        self.banner_label.bind("<Configure>", lambda e: self.banner_label.configure(
            wraplength=max(200, e.width - 24)), add="+")
        self.btn_retranscribe = ttk.Button(banner, text="Re-transcribir", style="TablaBar.TButton",
                                           command=self._on_retranscribe)
        self.btn_retranscribe.grid(row=0, column=1, padx=(0, 8), pady=4)
        Tooltip(self.btn_retranscribe, "Recalcula los entornos bandit con la configuración actual y vuelve a "
                                       "transcribir con los cuatro algoritmos (Ejecutar → Re-transcribir). No vuelve "
                                       "a decodificar ni a segmentar el audio.")
        banner.grid_remove()
        self.banner = banner

    def _build_info_row(self) -> None:
        """Resumen del algoritmo (notas y precisión) y leyenda de estados (o la pista de uso)."""
        info = ttk.Frame(self)
        info.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        info.columnconfigure(0, weight=1)
        self.info_row = info
        self.summary_label = ttk.Label(info, text="", style="TablaSummary.TLabel", anchor="w", justify="left")
        self.summary_label.grid(row=0, column=0, sticky="w")
        Tooltip(self.summary_label, self._summary_help)

        legend = ttk.Frame(info)
        self.legend = legend
        background = ttk.Style(self).lookup("TFrame", "background") or plots.HIGHLIGHT
        for code in (OUTCOME_EXACT, OUTCOME_WRONG_POSITION, OUTCOME_WRONG_PITCH, OUTCOME_EXTRA, OUTCOME_MISSED):
            item = ttk.Frame(legend)
            item.pack(side="left", padx=(10, 0))
            swatch = tk.Canvas(item, width=18, height=16, highlightthickness=0, borderwidth=0, background=background)
            swatch.pack(side="left", padx=(0, 3))
            if code == OUTCOME_MISSED:
                _round_rect(swatch, 2, 3, 16, 14, 3, fill=BOX_FILL, outline=plots.TEXT_MUTED, dash=(3, 2), width=1)
            else:
                fill = OUTCOME_FILL[code]
                swatch.create_oval(2, 1, 16, 15, fill=fill, outline="")
                swatch.create_text(9, 8, text=OUTCOME_SYMBOLS[code], fill=_ink_on(fill), font=self.font_badge)
            text = ttk.Label(item, text=OUTCOME_TEXT[code], style="TablaHint.TLabel")
            text.pack(side="left")
            for widget in (swatch, text):
                Tooltip(widget, OUTCOME_HELP[code])
        self.hint_label = ttk.Label(info, text=CANVAS_HINT, style="TablaHint.TLabel")
        info.bind("<Configure>", lambda _e: self._layout_info_row(), add="+")

    def _build_view(self) -> None:
        """Zona de la tablatura: lienzo gráfico (con columna fija y barra horizontal) o vista ASCII."""
        box = ttk.Frame(self, height=200, width=400)
        box.grid(row=3, column=0, sticky="ew")
        box.grid_propagate(False)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.view_box = box

        # --- vista gráfica: columna fija (cuerdas) + lienzo desplazable
        graphic = tk.Frame(box, background=plots.AXIS, borderwidth=0, highlightthickness=0)
        graphic.grid(row=0, column=0, sticky="nsew")
        graphic.columnconfigure(1, weight=1)
        graphic.rowconfigure(0, weight=1)
        self.graphic_frame = graphic
        self.gutter = tk.Canvas(graphic, width=GUTTER_W, height=150, background=plots.SURFACE,
                                highlightthickness=0, borderwidth=0)
        self.gutter.grid(row=0, column=0, sticky="ns", padx=(1, 0), pady=(1, 0))
        self.canvas = tk.Canvas(graphic, height=150, width=300, background=plots.SURFACE, highlightthickness=0,
                                borderwidth=0, xscrollincrement=1, takefocus=True)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(0, 1), pady=(1, 0))
        self.hscroll = ttk.Scrollbar(graphic, orient="horizontal", command=self._xview)
        self.hscroll.grid(row=1, column=0, columnspan=2, sticky="ew", padx=1, pady=(0, 1))
        self.canvas.configure(xscrollcommand=self._on_xscroll)
        self.canvas.tag_bind("note", "<Button-1>", self._on_note_click)
        self.canvas.tag_bind("gtnote", "<Button-1>", self._on_gt_click)
        for tag in ("note", "gtnote"):
            self.canvas.tag_bind(tag, "<Enter>", lambda _e: self.canvas.configure(cursor="hand2"))
            self.canvas.tag_bind(tag, "<Leave>", lambda _e: self.canvas.configure(cursor=""))
        self.canvas.bind("<Button-1>", lambda _e: self.canvas.focus_set(), add="+")
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(sequence, self._on_wheel, add="+")
            self.gutter.bind(sequence, self._on_wheel, add="+")
        for sequence in ("<Control-MouseWheel>", "<Control-Button-4>", "<Control-Button-5>"):
            self.canvas.bind(sequence, self._on_ctrl_wheel)
        self.canvas.bind("<Left>", lambda _e: self.step_selection(-1))
        self.canvas.bind("<Right>", lambda _e: self.step_selection(+1))
        self.canvas.bind("<Home>", lambda _e: self._select_edge(first=True))
        self.canvas.bind("<End>", lambda _e: self._select_edge(first=False))
        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")
        self._canvas_tip = _HoverTooltip(self.canvas, self._canvas_key, self._canvas_tooltip)

        # Botones del estado vacío (se colocan dentro del lienzo con create_window).
        buttons = ttk.Frame(self.canvas)
        self.btn_empty_open = ttk.Button(buttons, text="Abrir MP3…", style="Accent.TButton",
                                         command=lambda: self.app.open_audio())
        self.btn_empty_open.pack(side="left", padx=4)
        self.btn_empty_dataset = ttk.Button(buttons, text="Generar dataset sintético…",
                                            command=lambda: self.app.generate_dataset())
        self.btn_empty_dataset.pack(side="left", padx=4)
        Tooltip(self.btn_empty_open, "Elige un MP3 con un bajo aislado; se analiza y se transcribe con los cuatro "
                                     "algoritmos.")
        Tooltip(self.btn_empty_dataset, "Crea piezas de ejemplo con su ground truth (.gt.json) para comparar la "
                                        "tablatura con las posiciones reales.")
        self.empty_buttons = buttons

        # --- vista ASCII
        ascii_frame = ttk.Frame(box)
        ascii_frame.grid(row=0, column=0, sticky="nsew")
        ascii_frame.columnconfigure(0, weight=1)
        ascii_frame.rowconfigure(0, weight=1)
        self.ascii_frame = ascii_frame
        self.ascii_text = tk.Text(ascii_frame, wrap="none", font=self.font_mono, background=plots.SURFACE,
                                  foreground=plots.TEXT_PRIMARY, borderwidth=0, highlightthickness=1,
                                  highlightbackground=plots.AXIS, highlightcolor=plots.AXIS, padx=12, pady=8,
                                  height=8, width=40)
        self.ascii_text.grid(row=0, column=0, sticky="nsew")
        ascii_scroll = ttk.Scrollbar(ascii_frame, orient="vertical", command=self.ascii_text.yview)
        ascii_scroll.grid(row=0, column=1, sticky="ns")
        self.ascii_text.configure(yscrollcommand=ascii_scroll.set, state="disabled")
        self.ascii_text.tag_configure("title", font=self.font_label, foreground=plots.TEXT_PRIMARY)
        self.ascii_text.tag_configure("gt", foreground=plots.TEXT_SECONDARY)
        self.ascii_text.bind("<Configure>", self._on_ascii_configure, add="+")
        ascii_frame.grid_remove()

    def _build_detail(self) -> None:
        """Panel inferior: resumen de la nota, espectro con su template y tabla de brazos candidatos."""
        frame = ttk.LabelFrame(self, text="Nota seleccionada", style="Tabla.TLabelframe")
        frame.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        self.detail_frame = frame

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        header.columnconfigure(0, weight=1)
        self.detail_label = ttk.Label(header, text=NO_SELECTION_TEXT, style="TablaHint.TLabel", anchor="w",
                                      justify="left")
        self.detail_label.grid(row=0, column=0, sticky="ew")
        self.detail_outcome = tk.Label(header, text="", font=self.font_badge, padx=8, pady=1)
        header.bind("<Configure>", lambda e: self.detail_label.configure(wraplength=max(200, e.width - 200)),
                    add="+")

        # El cuerpo NO propaga el tamaño pedido de la figura y la tabla: así la pestaña nunca
        # pide más alto que la ventana (a 1024×700 la barra de estado quedaría fuera) y el
        # panel se encoge o crece con el espacio que dejan la barra y la tablatura.
        body = ttk.Frame(frame, height=150, width=600)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_propagate(False)
        body.columnconfigure(0, weight=1, minsize=360)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)
        body.bind("<Configure>", self._on_detail_body_configure, add="+")

        # La figura va dentro de un marco que NO propaga su tamaño pedido: así el lienzo de
        # matplotlib nunca obliga a la ventana a crecer (la pestaña cabe a 1024×700).
        holder = ttk.Frame(body, width=420, height=160)
        holder.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        holder.pack_propagate(False)
        self.plot = PlotFrame(holder, figsize=(5.4, 2.4))
        self.plot.pack(fill="both", expand=True)
        self.plot_holder = holder
        self._plot_size: tuple[int, int] = (0, 0)
        self.plot.canvas.get_tk_widget().bind("<Configure>", self._on_plot_configure, add="+")

        table = ttk.Frame(body)
        table.grid(row=0, column=1, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(1, weight=1)
        self.table_caption = ttk.Label(table, text="Brazos candidatos", style="TablaKey.TLabel", anchor="w")
        self.table_caption.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        Tooltip(self.table_caption, "Cada fila es un brazo del problema bandit de esta nota (las posiciones "
                                    "cuerda-traste a ±k semitonos de la f0 de pYIN), ordenadas por μ real. Datos "
                                    "de la corrida que produjo la tablatura (src.experiments.ChainResult).")
        columns = [c[0] for c in ARM_COLUMNS]
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=4, selectmode="none",
                                 style="Tabla.Treeview")
        for cid, heading, width, anchor, _help in ARM_COLUMNS:
            self.tree.heading(cid, text=heading, anchor="center")
            self.tree.column(cid, width=width, minwidth=40, anchor=anchor, stretch=cid == "role")
        self.tree.tag_configure("chosen", background="#dbe8fb")
        self.tree.tag_configure("optimal", font=self.font_tree_bold)
        self.tree.grid(row=1, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        _HoverTooltip(self.tree, self._tree_key, self._tree_tooltip)
        self.table_legend = ttk.Label(table, style="TablaHint.TLabel", anchor="w", justify="left",
                                      text="Fondo azul: brazo elegido (el más jalado) · ★ óptimo = argmax μ · "
                                           "● real = posición del ground truth")
        self.table_legend.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._wrap_table_labels()

    # ============================================================ datos actuales
    def transcription(self, algorithm: str | None = None) -> TranscriptionResult | None:
        """Transcripción mostrada (o la de ``algorithm``), si es coherente con el análisis actual.

        Parameters
        ----------
        algorithm : str | None, optional
            Algoritmo (por defecto el mostrado).

        Returns
        -------
        TranscriptionResult | None
            None si no hay análisis, si falta ese algoritmo o si su número de
            notas no coincide con los segmentos conservados del análisis.
        """
        tr = self.app.state.transcriptions.get(algorithm or self.algorithm)
        if tr is None or self.analysis is None or len(tr.notes) != len(self.analysis.segment_data):
            return None
        return tr

    @property
    def notes(self) -> list[TabNote]:
        """Notas de la tablatura mostrada (lista vacía si no hay)."""
        tr = self.transcription()
        return list(tr.notes) if tr is not None else []

    @property
    def ground_truth(self) -> list[GTNote] | None:
        """Notas reales del audio actual (None si no hay ``.gt.json``)."""
        return self.analysis.ground_truth if self.analysis is not None else None

    @property
    def comparing(self) -> bool:
        """True si se muestra la comparación con el ground truth (casilla activa y hay GT)."""
        return bool(self.compare_var.get()) and bool(self.ground_truth)

    def outcome(self, position: int) -> int | None:
        """Estado del segmento ``position`` frente al ground truth.

        Parameters
        ----------
        position : int
            Posición del segmento entre los conservados.

        Returns
        -------
        int | None
            ``OUTCOME_EXACT``, ``OUTCOME_WRONG_POSITION``, ``OUTCOME_WRONG_PITCH``
            u :data:`OUTCOME_EXTRA`; None si no hay comparación.
        """
        if not self.comparing or not 0 <= position < len(self._segment_outcomes):
            return None
        return self._segment_outcomes[position]

    def gt_note_for(self, position: int) -> GTNote | None:
        """Nota real emparejada con el segmento ``position`` (None si no hay o no se compara)."""
        gt = self.ground_truth
        k = self._segment_gt.get(position)
        return gt[k] if self.comparing and gt is not None and k is not None else None

    def _recompute_comparison(self) -> None:
        """Empareja segmentos y notas reales y puntúa la transcripción mostrada (src.experiments)."""
        self._matches, self._segment_gt, self._segment_outcomes, self._accuracy = [], {}, [], None
        gt = self.ground_truth
        tr = self.transcription()
        if self.analysis is None or not gt:
            return
        segments = [d.segment for d in self.analysis.segment_data]
        tolerance = float(self.app.state.config.experiment.onset_tolerance_s)
        self._matches = match_segments_to_gt(segments, gt, tolerance)
        self._segment_gt = {pos: k for k, pos in enumerate(self._matches) if pos is not None}
        if tr is not None:
            self._accuracy = score_arms(tr.chain.arms, self._matches, gt)
            self._segment_outcomes = outcome_of_segments(len(segments), self._matches, self._accuracy.outcomes)

    def _config_stale(self) -> bool:
        """True si el entorno, los agentes o la semilla cambiaron desde que se transcribió."""
        if self.analysis is None or not self.app.state.transcriptions:
            return False
        used, current = self.analysis.config, self.app.state.config
        return (current.env != used.env or current.agent != used.agent
                or current.experiment.seed != used.experiment.seed)

    # ================================================================ refresco
    def refresh(self) -> None:
        """Redibuja todo (resumen, tablatura o vista ASCII, panel inferior y controles)."""
        if self._refresh_job is not None:
            self.after_cancel(self._refresh_job)
            self._refresh_job = None
        n_notes = len(self.analysis.segment_data) if self.analysis is not None else 0
        if self.selected is not None and not 0 <= self.selected < n_notes:
            self.selected = None
        self.algo_var.set(self.algorithm)
        self._update_controls()
        self._update_summary()
        self._redraw_canvas()
        if self.ascii_var.get():
            self._render_ascii()
        self._update_detail()

    def _request_refresh(self) -> None:
        """Agrupa varios eventos seguidos (análisis + transcripciones) en un solo redibujo."""
        if self._refresh_job is None:
            self._refresh_job = self.after_idle(self.refresh)

    def _update_controls(self) -> None:
        """Habilita o deshabilita los controles según haya datos, ground truth o una tarea en curso."""
        has_tab = self.transcription() is not None
        has_gt = bool(self.ground_truth)
        self.chk_compare.state(["!disabled"] if has_gt else ["disabled"])
        export_ok = has_tab and not self._busy
        for button in self.export_buttons.values():
            button.state(["!disabled"] if export_ok else ["disabled"])
        all_ok = (self.analysis is not None and not self._busy
                  and any(self.transcription(a) is not None for a in ALGORITHMS))
        self.export_all_button.state(["!disabled"] if all_ok else ["disabled"])
        for alg, radio in self.algo_radios.items():
            radio.state(["!disabled"] if self.analysis is None or self.transcription(alg) is not None
                        else ["disabled"])
        self.btn_retranscribe.state(["disabled"] if self._busy else ["!disabled"])
        zoom_ok = has_tab and not self.ascii_var.get()
        for widget in (self.zoom_scale, self.btn_zoom_in, self.btn_zoom_out, self.btn_fit):
            widget.state(["!disabled"] if zoom_ok else ["disabled"])
        if self._config_stale():
            self.banner.grid()
        else:
            self.banner.grid_remove()

    def _update_summary(self) -> None:
        """Texto de resumen: algoritmo, número de notas y precisión frente al ground truth."""
        tr = self.transcription()
        name = ALGO_LABELS.get(self.algorithm, self.algorithm)
        if self.analysis is None:
            text = ("Analizando el audio…" if self._busy else
                    "Sin audio analizado: la tablatura aparecerá aquí después del análisis.")
        elif tr is None:
            text = f"{name}: sin transcripción para este audio (Ejecutar → Re-transcribir)."
        else:
            n = len(tr.notes)
            parts = [f"{name}", f"{n} nota{'s' if n != 1 else ''}"]
            gt = self.ground_truth
            if not gt:
                parts.append("sin ground truth (abre un MP3 del dataset sintético para comparar)")
            elif self.comparing and self._accuracy is not None:
                acc = self._accuracy
                n_gt = len(gt)
                pitch_ok = int(np.count_nonzero(acc.outcomes >= OUTCOME_WRONG_POSITION))
                exact = int(np.count_nonzero(acc.outcomes == OUTCOME_EXACT))
                parts.append(f"pitch {pitch_ok}/{n_gt} ({100 * acc.pitch:.0f} %)")
                parts.append(f"posición {exact}/{n_gt} ({100 * acc.position:.0f} %)")
                if acc.n_missed:
                    parts.append(f"{acc.n_missed} nota{'s' if acc.n_missed != 1 else ''} real"
                                 f"{'es' if acc.n_missed != 1 else ''} sin detectar")
                if acc.n_extra:
                    parts.append(f"{acc.n_extra} segmento{'s' if acc.n_extra != 1 else ''} de más")
            else:
                parts.append(f"ground truth disponible ({len(gt)} notas reales; activa «Comparar»)")
            text = " · ".join(parts)
        self.summary_label.configure(text=text)
        self._wrapped_info = None
        self._layout_info_row()

    def _layout_info_row(self) -> None:
        """Coloca la leyenda (o la pista de uso) a la derecha del resumen, o debajo si no cabe."""
        width = self.info_row.winfo_width()
        if width <= 1:
            width = 1300
        side = self.legend if self.comparing and self.view_state == "tab" else self.hint_label
        other = self.hint_label if side is self.legend else self.legend
        other.grid_remove()
        if self.view_state != "tab":
            side.grid_remove()
            self.summary_label.configure(wraplength=max(200, width - 8))
            return
        summary_w = self.font_summary.measure(self.summary_label.cget("text"))
        wrapped = summary_w + side.winfo_reqwidth() + 24 > width
        if wrapped:
            side.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.summary_label.configure(wraplength=max(200, width - 8))
        else:
            side.grid(row=0, column=1, sticky="e")
            self.summary_label.configure(wraplength=max(200, width - side.winfo_reqwidth() - 16))
        self._wrapped_info = wrapped

    # ======================================================= tablatura gráfica
    @property
    def geometry(self) -> StaffGeometry:
        """Geometría vigente (normal o compacta)."""
        return COMPACT_GEOMETRY if self.compact else NORMAL_GEOMETRY

    def x_of(self, t: float) -> float:
        """Coordenada x del lienzo (px) del instante ``t`` (s) con el zoom actual."""
        return X0 + float(t) * self.zoom

    def note_bbox(self, position: int) -> tuple[float, float, float, float] | None:
        """Recuadro ``(x0, y0, x1, y1)`` de la nota ``position`` en coordenadas del lienzo (None si no se dibujó)."""
        return self._note_boxes.get(position)

    def gt_bbox(self, index: int) -> tuple[float, float, float, float] | None:
        """Recuadro de la nota real ``index`` en la pauta del ground truth (None si no se dibujó)."""
        return self._gt_boxes.get(index)

    def _visible_width(self) -> int:
        """Ancho visible del lienzo (px), con un valor razonable si aún no se dibujó."""
        width = self.canvas.winfo_width()
        return width if width > 10 else 1200

    def _redraw_canvas(self) -> None:
        """Dibuja de nuevo la tablatura gráfica (o el estado vacío) y ajusta la altura de la vista."""
        canvas, gutter = self.canvas, self.gutter
        self._canvas_tip.hide()
        canvas.delete("all")
        gutter.delete("all")
        self._note_boxes, self._note_spans, self._gt_boxes = {}, {}, {}
        tr = self.transcription()
        show_gt = self.comparing and tr is not None
        self._layout = layout = staff_layout(self.geometry, show_gt)
        if self.analysis is None or tr is None or not tr.notes:
            self._draw_message_state(tr)
            self._layout_info_row()
            return
        self.view_state = "tab"
        duration = max(self.analysis.duration_s, max(n.end_s for n in tr.notes))
        self._content_width = int(self.x_of(duration) + RIGHT_PAD)
        height = layout.height
        self._set_view_height(height)
        scroll_w = max(self._content_width, self._visible_width())
        canvas.configure(scrollregion=(0, 0, scroll_w, height))
        self._draw_ruler(duration, height)
        self._draw_staff(layout.main_lines, main=True)
        self._draw_gutter(layout)
        for note in tr.notes:
            self._draw_note(note)
        if show_gt:
            self._draw_staff(layout.gt_lines, main=False)
            canvas.create_text(X0, layout.gt_heading_y, anchor="w", tags=("sticky",),
                               text="Ground truth · posiciones reales (archivo .gt.json)",
                               fill=plots.TEXT_SECONDARY, font=self.font_heading)
            for k, gt_note in enumerate(self.ground_truth or []):
                self._draw_gt_note(k, gt_note)
        self._draw_selection()
        self._update_sticky()
        self._layout_info_row()

    def _set_view_height(self, height: int) -> None:
        """Ajusta la altura del lienzo y de la zona de la tablatura (ver :meth:`_apply_view_height`)."""
        self._canvas_height = height
        self.canvas.configure(height=height)
        self.gutter.configure(height=height)
        self._apply_view_height()

    def _apply_view_height(self) -> None:
        """Altura de la zona de la tablatura: la del lienzo (+ barra) o, en vista ASCII, la de su texto.

        La vista ASCII crece hasta mostrar todo el texto, sin pasar de
        :data:`ASCII_MAX_FRACTION` de la altura de la pestaña (después aparece la
        barra vertical); nunca es más baja que la vista gráfica.
        """
        bar_h = max(self.hscroll.winfo_reqheight(), 12)
        height = self._canvas_height + bar_h + 3
        if self.ascii_var.get():
            n_lines = int(self.ascii_text.index("end-1c").split(".")[0])
            text_h = n_lines * self.font_mono.metrics("linespace") + 2 * int(self.ascii_text.cget("pady")) + 6
            tab_h = self.winfo_height() if self.winfo_height() > 1 else 800
            height = max(height, min(text_h, int(ASCII_MAX_FRACTION * tab_h)))
        if int(self.view_box.cget("height")) != height:
            self.view_box.configure(height=height)

    def _draw_message_state(self, tr: TranscriptionResult | None) -> None:
        """Estado sin tablatura: mensaje guía (sin audio), «analizando…» o aviso."""
        canvas = self.canvas
        lay = staff_layout(self.geometry, show_gt=False)
        height = max(lay.height, 150)
        self._set_view_height(height)
        width = self._visible_width()
        canvas.configure(scrollregion=(0, 0, width, height))
        canvas.xview_moveto(0.0)
        self._content_width = width
        if self.analysis is None and not self._busy:
            self.view_state = "empty"
            title, text = EMPTY_TITLE, EMPTY_TEXT
        elif self.analysis is None:
            self.view_state = "busy"
            title, text = "Analizando el audio…", "La tablatura aparecerá aquí en cuanto termine la transcripción."
        elif tr is None:
            self.view_state = "message"
            name = ALGO_LABELS.get(self.algorithm, self.algorithm)
            title = f"Sin transcripción de {name}"
            text = ("Este algoritmo no tiene tablatura para el audio actual. Usa Ejecutar → Re-transcribir con la "
                    "configuración actual.")
        else:
            self.view_state = "message"
            title = "No se detectaron notas"
            text = ("El análisis no conservó ningún segmento (todo era silencio o demasiado corto). Revisa el umbral "
                    "de silencio y la duración mínima en la pestaña Configuración y vuelve a analizar.")
        cx = width / 2
        canvas.create_text(cx, height / 2 - 26, text=title, fill=plots.TEXT_PRIMARY, font=self.font_empty_title,
                           tags=("message",))
        canvas.create_text(cx, height / 2 + 4, text=text, fill=plots.TEXT_SECONDARY, font=self.font_empty,
                           width=min(720, width - 40), justify="center", tags=("message",))
        if self.view_state == "empty":
            canvas.create_window(cx, height / 2 + 46, window=self.empty_buttons, tags=("message",))
        for s, y in lay.main_lines.items():
            self.gutter.create_text(GUTTER_W - 12, y, text=s, anchor="e", fill=plots.TEXT_MUTED, font=self.font_label)

    def _draw_ruler(self, duration: float, height: int) -> None:
        """Regla de tiempo (s) arriba y líneas guía verticales muy tenues."""
        canvas, g = self.canvas, self.geometry
        step = nice_time_step(self.zoom)
        minor = step / 5 if step * self.zoom / 5 >= 9 else step / 2
        base = g.ruler_h - 1
        canvas.create_line(0, base, self._content_width, base, fill=plots.AXIS, tags=("ruler",))
        top_staff = self._layout.main_lines[TAB_STRINGS[0]] - g.box_h / 2 - 2
        n_minor = int(duration / minor + 1e-9)
        for i in range(n_minor + 1):
            t = i * minor
            x = self.x_of(t)
            is_major = math.isclose(t / step, round(t / step), abs_tol=1e-6)
            if is_major:
                canvas.create_line(x, top_staff, x, height, fill=GUIDE_COLOR, tags=("guide",))
                canvas.create_line(x, base - 6, x, base, fill=plots.TEXT_MUTED, tags=("ruler",))
                canvas.create_text(x, base - 7, text=format_seconds(t, step), anchor="s", fill=plots.TEXT_SECONDARY,
                                   font=self.font_small, tags=("ruler",))
            else:
                canvas.create_line(x, base - 3, x, base, fill=plots.AXIS, tags=("ruler",))
        canvas.tag_lower("guide")

    def _draw_staff(self, lines: dict[str, float], main: bool) -> None:
        """Cuatro líneas de cuerda (G, D, A, E) y la barra final."""
        color = plots.AXIS if main else plots.GRID
        end = self._content_width - RIGHT_PAD / 2
        for y in lines.values():
            self.canvas.create_line(0, y, end, y, fill=color, width=1, tags=("staff",))
        ys = list(lines.values())
        self.canvas.create_line(end, ys[0], end, ys[-1], fill=color, width=2 if main else 1, tags=("staff",))

    def _draw_gutter(self, layout: StaffLayout) -> None:
        """Columna fija: «t (s)», nombres de cuerda, barra inicial y color del algoritmo."""
        gutter, g = self.gutter, self.geometry
        gutter.create_text(GUTTER_W - 10, g.ruler_h - 8, text="t (s)", anchor="se", fill=plots.TEXT_SECONDARY,
                           font=self.font_small)
        gutter.create_line(0, g.ruler_h - 1, GUTTER_W, g.ruler_h - 1, fill=plots.AXIS)
        ys = list(layout.main_lines.values())
        color = ALGO_COLORS.get(self.algorithm, plots.TEXT_SECONDARY)
        gutter.create_line(8, ys[0] - g.box_h / 2, 8, ys[-1] + g.box_h / 2, fill=color, width=5, capstyle="round")
        for s, y in layout.main_lines.items():
            gutter.create_text(GUTTER_W - 12, y, text=s, anchor="e", fill=plots.TEXT_PRIMARY, font=self.font_label)
            gutter.create_line(GUTTER_W - 6, y, GUTTER_W, y, fill=plots.AXIS)
        gutter.create_line(GUTTER_W - 1, ys[0], GUTTER_W - 1, ys[-1], fill=plots.TEXT_SECONDARY, width=2)
        if layout.gt_lines:
            gys = list(layout.gt_lines.values())
            for s, y in layout.gt_lines.items():
                gutter.create_text(GUTTER_W - 12, y, text=s, anchor="e", fill=plots.TEXT_MUTED, font=self.font_small)
                gutter.create_line(GUTTER_W - 6, y, GUTTER_W, y, fill=plots.GRID)
            gutter.create_line(GUTTER_W - 1, gys[0], GUTTER_W - 1, gys[-1], fill=plots.AXIS, width=1)
            gutter.create_text(8, (gys[0] + gys[-1]) / 2, text="Real", angle=90, fill=plots.TEXT_MUTED,
                               font=self.font_small)

    def _box_width(self, fret: int, font: tkfont.Font, minimum: int) -> float:
        """Ancho (px) del recuadro de un traste (el número más un margen)."""
        return max(float(minimum), font.measure(str(fret)) + 10.0)

    def _draw_note(self, note: TabNote) -> None:
        """Recuadro de una nota (traste) en su cuerda, con su duración y, si se compara, su estado."""
        canvas, g = self.canvas, self.geometry
        y = self._layout.main_lines[note.string]
        x0 = self.x_of(note.start_s)
        x_end = self.x_of(note.end_s)
        w = self._box_width(note.fret, self.font_fret, g.box_h + 2)
        box = (x0, y - g.box_h / 2, x0 + w, y + g.box_h / 2)
        tags = ("note", f"pos:{note.position}")
        outcome = self.outcome(note.position)
        if outcome is None:
            accent = ALGO_COLORS.get(self.algorithm, plots.TEXT_SECONDARY)
            fill, outline = BOX_FILL, accent
        else:
            accent = OUTCOME_FILL[outcome]
            fill, outline = _mix(accent, "#ffffff", 0.22), _mix(accent, "#000000", 0.85)
        if x_end > box[2] + 2:  # duración del segmento: barra fina tras el recuadro
            canvas.create_line(box[2], y, x_end, y, fill=_mix(accent, "#ffffff", 0.45), width=4,
                               capstyle="round", tags=tags)
        _round_rect(canvas, *box, radius=4, fill=fill, outline=outline, width=2, tags=tags)
        canvas.create_text((box[0] + box[2]) / 2, y, text=str(note.fret), fill=plots.TEXT_PRIMARY,
                           font=self.font_fret, tags=tags)
        if outcome is not None:  # símbolo de estado en la esquina superior derecha
            r = 7 if not self.compact else 6
            cx, cy = box[2] - 1, box[1] - 1
            badge_fill = OUTCOME_FILL[outcome]
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=badge_fill, outline=plots.SURFACE, width=1.5,
                               tags=tags + ("badge",))
            canvas.create_text(cx, cy, text=OUTCOME_SYMBOLS[outcome], fill=_ink_on(badge_fill),
                               font=self.font_badge, tags=tags + ("badge",))
        self._note_boxes[note.position] = box
        self._note_spans[note.position] = (x0, max(x_end, box[2]))

    def _draw_gt_note(self, index: int, note: GTNote) -> None:
        """Recuadro tenue de una nota real (borde discontinuo si ningún segmento la detectó)."""
        canvas, g = self.canvas, self.geometry
        y = self._layout.gt_lines[note.string]
        x0 = self.x_of(note.onset_s)
        w = self._box_width(note.fret, self.font_small, g.gt_box_h + 4)
        box = (x0, y - g.gt_box_h / 2, x0 + w, y + g.gt_box_h / 2)
        tags = ("gtnote", f"gt:{index}")
        missed = index < len(self._matches) and self._matches[index] is None
        x_end = self.x_of(note.offset_s)
        if x_end > box[2] + 2:
            canvas.create_line(box[2], y, x_end, y, fill=plots.GRID, width=3, capstyle="round", tags=tags)
        if missed:
            _round_rect(canvas, *box, radius=3, fill=BOX_FILL, outline=plots.TEXT_MUTED, width=1, dash=(3, 2),
                        tags=tags + ("missed",))
        else:
            _round_rect(canvas, *box, radius=3, fill="#f4f3ef", outline=plots.AXIS, width=1, tags=tags)
        canvas.create_text((box[0] + box[2]) / 2, y, text=str(note.fret), font=self.font_small,
                           fill=plots.TEXT_MUTED if missed else plots.TEXT_SECONDARY, tags=tags)
        self._gt_boxes[index] = box

    def _draw_selection(self) -> None:
        """Resalta la nota seleccionada: banda de su duración, contorno azul y su nota real."""
        canvas, g = self.canvas, self.geometry
        canvas.delete("selection")
        pos = self.selected
        if pos is None or pos not in self._note_boxes:
            return
        x0, y0, x1, y1 = self._note_boxes[pos]
        span = self._note_spans.get(pos, (x0, x1))
        band_top = g.ruler_h
        band_bottom = self._layout.height
        band = canvas.create_rectangle(span[0] - 3, band_top, span[1] + 3, band_bottom, fill=SELECTION_BAND,
                                       outline="", tags=("selection", "band"))
        canvas.tag_lower(band)
        _round_rect(canvas, x0 - 3, y0 - 3, x1 + 3, y1 + 3, radius=6, fill="", outline=UI_ACCENT, width=3,
                    tags=("selection",))
        gt_note = self._segment_gt.get(pos)
        if self.comparing and gt_note is not None and gt_note in self._gt_boxes:
            gx0, gy0, gx1, gy1 = self._gt_boxes[gt_note]
            _round_rect(canvas, gx0 - 2, gy0 - 2, gx1 + 2, gy1 + 2, radius=4, fill="", outline=UI_ACCENT, width=2,
                        tags=("selection",))
            canvas.create_line((x0 + x1) / 2, y1 + 4, (gx0 + gx1) / 2, gy0 - 3, fill=UI_ACCENT, dash=(2, 3),
                               width=1, tags=("selection",))
        canvas.tag_raise("badge")

    # --------------------------------------------------- desplazamiento y zoom
    def _xview(self, *args: Any) -> None:
        """Comando de la barra horizontal."""
        self.canvas.xview(*args)

    def _on_xscroll(self, first: str, last: str) -> None:
        """Actualiza la barra y mantiene los títulos «pegados» al borde izquierdo visible."""
        self.hscroll.set(first, last)
        self._update_sticky()

    def _update_sticky(self) -> None:
        """Mueve los textos con la etiqueta ``sticky`` al borde izquierdo de la zona visible."""
        left = self.canvas.canvasx(0)
        for item in self.canvas.find_withtag("sticky"):
            _x, y = self.canvas.coords(item)
            self.canvas.coords(item, left + X0, y)

    def visible_range(self) -> tuple[float, float]:
        """Rango horizontal visible del lienzo ``(x_izquierda, x_derecha)`` en coordenadas del lienzo."""
        left = self.canvas.canvasx(0)
        return left, left + self._visible_width()

    def scroll_to(self, position: int, force: bool = False) -> None:
        """Desplaza la vista para que la nota ``position`` quede visible (a un tercio desde la izquierda).

        Parameters
        ----------
        position : int
            Segmento.
        force : bool, optional
            Si es True se recoloca aunque ya sea visible.
        """
        box = self._note_boxes.get(position)
        if box is None:
            return
        left, right = self.visible_range()
        margin = 30
        if not force and box[0] - margin >= left and box[2] + margin <= right:
            return
        total = max(self._content_width, self._visible_width())
        target = max(0.0, box[0] - self._visible_width() * 0.33)
        self.canvas.xview_moveto(target / total)
        self._update_sticky()

    def set_zoom(self, px_per_s: float, anchor_time: float | None = None) -> None:
        """Cambia el zoom horizontal manteniendo fijo en pantalla un instante de referencia.

        Parameters
        ----------
        px_per_s : float
            Nuevo zoom (se limita a [:data:`MIN_ZOOM`, :data:`MAX_ZOOM`]).
        anchor_time : float | None, optional
            Instante (s) que no se mueve en pantalla. Por defecto, el inicio de
            la nota seleccionada si está visible; si no, el centro de la vista.
        """
        new = float(min(MAX_ZOOM, max(MIN_ZOOM, px_per_s)))
        if math.isclose(new, self.zoom, rel_tol=1e-4):
            self._sync_zoom_scale()
            return
        left, right = self.visible_range()
        screen_x = (right - left) / 2
        if anchor_time is None:
            box = self._note_boxes.get(self.selected) if self.selected is not None else None
            if box is not None and left <= box[0] <= right:
                anchor_time = (box[0] - X0) / self.zoom
                screen_x = box[0] - left
            else:
                anchor_time = (left + screen_x - X0) / self.zoom
        else:
            screen_x = min(max(self.x_of(anchor_time) - left, 0.0), right - left)
        self.zoom = new
        self._sync_zoom_scale()
        self._update_zoom_label()
        if self.view_state == "tab" or self.transcription() is not None:
            self._redraw_canvas()
            total = max(self._content_width, self._visible_width())
            self.canvas.xview_moveto(max(0.0, self.x_of(max(anchor_time, 0.0)) - screen_x) / total)
            self._update_sticky()

    def fit_zoom(self) -> None:
        """Zoom para que toda la pista quepa en el ancho visible (botón «Ajustar»)."""
        if self.analysis is None:
            return
        tr = self.transcription()
        duration = max([self.analysis.duration_s] + ([n.end_s for n in tr.notes] if tr else []))
        if duration <= 0:
            return
        self.set_zoom((self._visible_width() - X0 - RIGHT_PAD) / duration, anchor_time=0.0)
        self.canvas.xview_moveto(0.0)

    def _sync_zoom_scale(self) -> None:
        """Pone el deslizador en la posición del zoom actual (escala logarítmica)."""
        self._syncing_scale = True
        try:
            self.zoom_scale.set(math.log2(self.zoom))
        finally:
            self._syncing_scale = False

    def _update_zoom_label(self) -> None:
        """Texto «160 px/s» junto al deslizador."""
        self.zoom_label.configure(text=f"{self.zoom:.0f} px/s")

    def _on_zoom_scale(self, value: str) -> None:
        """Deslizador del zoom (escala log₂ de px/s)."""
        if self._syncing_scale:
            return
        try:
            self.set_zoom(2.0 ** float(value))
        except ValueError:
            pass

    def _on_wheel(self, event: tk.Event) -> str:
        """Rueda del ratón sobre la tablatura: desplazamiento horizontal."""
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.xview_scroll(-60, "units")
        else:
            self.canvas.xview_scroll(60, "units")
        return "break"

    def _on_ctrl_wheel(self, event: tk.Event) -> str:
        """Ctrl + rueda: zoom alrededor del instante bajo el puntero."""
        anchor = (self.canvas.canvasx(event.x) - X0) / self.zoom
        up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        self.set_zoom(self.zoom * (ZOOM_STEP if up else 1 / ZOOM_STEP), anchor_time=max(anchor, 0.0))
        return "break"

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        """Al cambiar el ancho visible se recalcula la región desplazable (y el estado vacío se recentra)."""
        if self.view_state == "tab":
            height = self._layout.height
            self.canvas.configure(scrollregion=(0, 0, max(self._content_width, self._visible_width()), height))
            self._update_sticky()
        else:
            self._redraw_canvas()

    def _on_tab_configure(self, _event: tk.Event) -> None:
        """Cambia a la geometría compacta en ventanas bajas (≈ 1024×700) y viceversa."""
        height = self.winfo_height()
        if height <= 1:
            return
        compact = height < COMPACT_TAB_HEIGHT
        if compact != self.compact:
            self.compact = compact
            self._apply_compact()

    def _apply_compact(self) -> None:
        """Aplica la geometría (fuentes del lienzo y barra de la figura) y redibuja."""
        g = self.geometry
        self.font_fret.configure(size=g.font_size)
        self.font_small.configure(size=g.small_font_size)
        self.font_badge.configure(size=g.small_font_size)
        if self.plot.toolbar is not None:
            if self.compact:
                self.plot.toolbar.pack_forget()
            else:
                self.plot.toolbar.pack(side="bottom", fill="x", before=self.plot.canvas.get_tk_widget())
        self._redraw_canvas()
        if self.selected is not None:
            self.scroll_to(self.selected)
        self._schedule_detail_plot()

    # ============================================================== selección
    def _canvas_key(self, x: int, y: int) -> Hashable | None:
        """Nota (``("pos", n)``) o nota real (``("gt", k)``) bajo el punto ``(x, y)`` del lienzo."""
        cx, cy = self.canvas.canvasx(x), self.canvas.canvasy(y)
        for item in reversed(self.canvas.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1)):
            for tag in self.canvas.gettags(item):
                if tag.startswith("pos:"):
                    return ("pos", int(tag[4:]))
                if tag.startswith("gt:"):
                    return ("gt", int(tag[3:]))
        return None

    def _current_item_key(self) -> Hashable | None:
        """Nota bajo el puntero según la etiqueta ``current`` del lienzo."""
        for tag in self.canvas.gettags("current"):
            if tag.startswith("pos:"):
                return ("pos", int(tag[4:]))
            if tag.startswith("gt:"):
                return ("gt", int(tag[3:]))
        return None

    def _on_note_click(self, event: tk.Event) -> None:
        """Clic en una nota de la tablatura: la selecciona (y avisa a las demás pestañas)."""
        key = self._current_item_key() or self._canvas_key(event.x, event.y)
        if key is not None and key[0] == "pos":
            self.canvas.focus_set()
            self.select(int(key[1]))

    def _on_gt_click(self, event: tk.Event) -> None:
        """Clic en una nota real: selecciona el segmento emparejado con ella (si existe)."""
        key = self._current_item_key() or self._canvas_key(event.x, event.y)
        if key is None or key[0] != "gt":
            return
        k = int(key[1])
        pos = self._matches[k] if k < len(self._matches) else None
        self.canvas.focus_set()
        if pos is not None:
            self.select(pos)
        else:
            self.app.status.set_message(f"La nota real {k} no tiene segmento: la segmentación no la detectó.")

    def select(self, position: int | None, publish: bool = True, scroll: bool = True) -> None:
        """Selecciona una nota: resalta, desplaza la vista, actualiza el panel inferior y avisa.

        Parameters
        ----------
        position : int | None
            Segmento (None = ninguno).
        publish : bool, optional
            Si es True se publica con ``app.state.select_segment(position, "tablature")``.
        scroll : bool, optional
            Si es True la vista se desplaza hasta la nota si no está visible.
        """
        n_notes = len(self.notes)
        if position is not None and not 0 <= position < n_notes:
            return
        self.selected = position
        self._draw_selection()
        if scroll and position is not None:
            self.scroll_to(position)
        self._update_detail()
        if publish:
            self.app.state.select_segment(position, source=SOURCE)

    def step_selection(self, delta: int) -> str:
        """Selecciona la nota anterior (``delta`` < 0) o siguiente (> 0); devuelve ``"break"``."""
        n = len(self.notes)
        if n == 0:
            return "break"
        current = self.selected if self.selected is not None else (-1 if delta > 0 else n)
        self.select(int(min(n - 1, max(0, current + delta))))
        return "break"

    def _select_edge(self, first: bool) -> str:
        """Selecciona la primera o la última nota (teclas Inicio / Fin)."""
        n = len(self.notes)
        if n:
            self.select(0 if first else n - 1)
        return "break"

    def _canvas_tooltip(self, key: Hashable) -> str:
        """Texto del tooltip de una nota del lienzo o de una nota real."""
        kind, index = key  # type: ignore[misc]
        if self.analysis is None:
            return ""
        if kind == "gt":
            gt = self.ground_truth or []
            if not 0 <= index < len(gt):
                return ""
            note = gt[index]
            pos = self._matches[index] if index < len(self._matches) else None
            found = (f"Detectada como segmento {pos} (clic para seleccionarlo)." if pos is not None else
                     "No detectada: ningún segmento empieza a menos de la tolerancia de onset "
                     f"({self.app.state.config.experiment.onset_tolerance_s:.2f} s).")
            return (f"Nota real {index} (ground truth): {note.label} · {midi_to_name(note.midi)} · "
                    f"{note.onset_s:.2f}–{note.offset_s:.2f} s\n{found}")
        notes = self.notes
        if not 0 <= index < len(notes):
            return ""
        note = notes[index]
        arm = Arm(note.string, note.fret)
        lines = [f"Segmento {index} · {note.label} ({note.note_name}, {arm.freq_hz:.1f} Hz)",
                 f"{note.start_s:.2f}–{note.end_s:.2f} s · "
                 + (f"f0 pYIN {note.f0_hz:.1f} Hz" if note.f0_hz else "f0 pYIN desconocida")]
        outcome = self.outcome(index)
        gt_note = self.gt_note_for(index)
        if outcome == OUTCOME_EXTRA:
            lines.append("Sin nota real: segmento de más.")
        elif outcome is not None and gt_note is not None:
            lines.append(f"Real: {gt_note.label} → {OUTCOME_SYMBOLS[outcome]} {OUTCOME_TEXT[outcome].lower()}")
        lines.append("Clic: ver su espectro y los brazos candidatos.")
        return "\n".join(lines)

    # =========================================================== vista ASCII
    def _on_ascii_toggled(self) -> None:
        """Alterna entre la tablatura gráfica y la vista ASCII."""
        self._update_controls()  # el zoom no se aplica a la vista ASCII
        if self.ascii_var.get():
            self.graphic_frame.grid_remove()
            self.ascii_frame.grid()
            self.update_idletasks()
            self._render_ascii()
        else:
            self.ascii_frame.grid_remove()
            self.graphic_frame.grid()
            self._apply_view_height()
            if self.selected is not None:
                self.after_idle(lambda: self.selected is not None and self.scroll_to(self.selected))

    def ascii_line_width(self) -> int:
        """Ancho de línea (caracteres) de la vista ASCII según el ancho del widget."""
        width = self.ascii_text.winfo_width()
        if width <= 10:
            width = max(self.view_box.winfo_width(), 800)
        usable = width - 2 * int(self.ascii_text.cget("padx")) - 8
        return max(40, usable // max(1, self.font_mono.measure("0")))

    def ascii_content(self) -> str:
        """Texto de la vista ASCII: tablatura del algoritmo y, si se compara, la del ground truth."""
        tr = self.transcription()
        if self.analysis is None:
            return EMPTY_TEXT
        if tr is None:
            return f"Sin transcripción de {ALGO_LABELS.get(self.algorithm, self.algorithm)} para este audio."
        width = self.ascii_line_width()
        name = ALGO_LABELS.get(self.algorithm, self.algorithm)
        blocks = [f"{name} — {Path(self.analysis.path).name}\n" + render_ascii(tr.notes, line_width=width)]
        if self.comparing:
            blocks.append("Ground truth (posiciones reales)\n"
                          + render_ascii(gt_tab_notes(self.ground_truth or []), line_width=width))
        return "\n\n".join(blocks)

    def _render_ascii(self) -> None:
        """Escribe :meth:`ascii_content` en el widget de texto (títulos en negrita, GT en gris)."""
        self._ascii_job = None
        text = self.ascii_content()
        self._ascii_width = self.ascii_line_width()
        widget = self.ascii_text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        in_gt = False
        has_title = self.transcription() is not None
        for i, line in enumerate(text.split("\n")):
            if line.startswith("Ground truth"):
                in_gt = True
            tags: tuple[str, ...] = ()
            if (i == 0 and has_title) or line.startswith("Ground truth"):
                tags = ("title",)
            elif in_gt:
                tags = ("gt",)
            widget.insert("end", ("\n" if i else "") + line, tags)
        widget.configure(state="disabled")
        self._apply_view_height()

    def _on_ascii_configure(self, _event: tk.Event) -> None:
        """Re-renderiza la vista ASCII (diferido) si cambia el número de caracteres por línea."""
        if not self.ascii_var.get() or self.ascii_line_width() == self._ascii_width:
            return
        if self._ascii_job is not None:
            self.after_cancel(self._ascii_job)
        self._ascii_job = self.after(80, self._render_ascii)

    # ======================================================== panel inferior
    def _update_detail(self) -> None:
        """Rellena el resumen y la tabla de la nota seleccionada y programa la gráfica."""
        tr = self.transcription()
        pos = self.selected
        self.tree.delete(*self.tree.get_children())
        self.detail_outcome.grid_remove()
        if self.analysis is None or tr is None or pos is None or not 0 <= pos < len(tr.notes):
            self.detail_label.configure(text=NO_SELECTION_TEXT, style="TablaHint.TLabel")
            self.table_caption.configure(text="Brazos candidatos")
            self._schedule_detail_plot()
            return
        data = self.analysis.segment_data[pos]
        chain = tr.chain
        seg = data.segment
        chosen = chain.arms[pos]
        prev = chain.prev_frets[pos] if pos < len(chain.prev_frets) else None
        gt_note = self.gt_note_for(pos)
        outcome = self.outcome(pos)
        self.detail_label.configure(
            text=detail_text(pos, seg.start_s, seg.end_s, seg.f0_hz, prev, chosen.label,
                             gt_note.label if gt_note is not None else None, outcome),
            style="TablaDetail.TLabel")
        if outcome is not None:
            fill = OUTCOME_FILL[outcome]
            self.detail_outcome.configure(text=f"{OUTCOME_SYMBOLS[outcome]}  {OUTCOME_TEXT[outcome]}",
                                          background=fill, foreground=_ink_on(fill))
            self.detail_outcome.grid(row=0, column=1, sticky="e", padx=(8, 0))
        k = self.analysis.config.env.k_semitones
        poda = f"±{k} semitonos de la f0" if k is not None else "sin poda: los 52 brazos"
        self.table_caption.configure(
            text=f"Brazos candidatos ({data.n_arms}; {poda}) · T = {chain.rewards.shape[1]} pulls · "
                 f"{ALGO_LABELS.get(self.algorithm, self.algorithm)}")
        self._fill_arm_table(pos)
        self._schedule_detail_plot()

    def arm_rows(self, position: int) -> list[dict[str, Any]]:
        """Datos de la tabla de brazos del segmento ``position`` (ordenados por μ real, de mayor a menor).

        Parameters
        ----------
        position : int
            Segmento.

        Returns
        -------
        list[dict[str, Any]]
            Una entrada por brazo con ``arm`` (:class:`Arm`), ``mu``, ``q``,
            ``pulls``, ``chosen``, ``optimal`` y ``gt`` (booleanos). Lista vacía
            si no hay transcripción.
        """
        tr = self.transcription()
        if tr is None or self.analysis is None or not 0 <= position < len(tr.notes):
            return []
        data = self.analysis.segment_data[position]
        chain = tr.chain
        mu = np.asarray(chain.true_means[position], dtype=float)
        q = np.asarray(chain.final_q[position], dtype=float)
        counts = np.asarray(chain.final_counts[position])
        chosen = chain.arms[position]
        gt_note = self.gt_note_for(position)
        best = float(mu.max()) if mu.size else 0.0
        rows = []
        for i, arm in enumerate(data.arms):
            rows.append({
                "index": i, "arm": arm, "mu": float(mu[i]), "q": float(q[i]), "pulls": int(counts[i]),
                "chosen": arm == chosen, "optimal": bool(math.isclose(float(mu[i]), best, abs_tol=1e-12)),
                "gt": gt_note is not None and arm.string == gt_note.string and arm.fret == gt_note.fret,
            })
        rows.sort(key=lambda r: (-r["mu"], r["index"]))
        return rows

    def _fill_arm_table(self, position: int) -> None:
        """Una fila por brazo candidato (:meth:`arm_rows`); resalta el elegido y el óptimo."""
        for row in self.arm_rows(position):
            arm: Arm = row["arm"]
            roles = []
            if row["chosen"]:
                roles.append("elegido")
            if row["optimal"]:
                roles.append("★ óptimo")
            if row["gt"]:
                roles.append("● real")
            tags = tuple(t for t, on in (("chosen", row["chosen"]), ("optimal", row["optimal"])) if on)
            self.tree.insert("", "end", iid=f"arm{row['index']}", tags=tags, values=(
                arm.label, f"{arm.freq_hz:.1f}", midi_to_name(arm.midi), f"{row['mu']:.3f}", f"{row['q']:.3f}",
                str(row["pulls"]), " · ".join(roles)))
        chosen_iid = next((iid for iid in self.tree.get_children() if "chosen" in self.tree.item(iid, "tags")), None)
        if chosen_iid is not None:
            self.tree.see(chosen_iid)

    def _tree_key(self, x: int, y: int) -> Hashable | None:
        """Encabezado de columna bajo el puntero (para su tooltip)."""
        if self.tree.identify_region(x, y) not in ("heading", "separator"):
            return None
        column = self.tree.identify_column(x)  # "#n": n-ésima columna VISIBLE (1 = la primera)
        try:
            index = int(column.lstrip("#")) - 1
        except ValueError:
            return None
        shown = self._shown_arm_columns()
        return ("heading", shown[index]) if 0 <= index < len(shown) else None

    def _shown_arm_columns(self) -> list[str]:
        """Identificadores de las columnas visibles de la tabla de brazos, en orden."""
        return [c[0] for c in ARM_COLUMNS if not (self._narrow_table and c[0] == "note")]

    def _tree_tooltip(self, key: Hashable) -> str:
        """Ayuda de una columna de la tabla de brazos."""
        _kind, column = key  # type: ignore[misc]
        return next((c[4] for c in ARM_COLUMNS if c[0] == column), "")

    def _schedule_detail_plot(self, delay_ms: int = 40) -> None:
        """Programa el dibujo del espectro (diferido; solo si la pestaña está visible).

        Parameters
        ----------
        delay_ms : int, optional
            Espera (ms): agrupa selecciones o redimensionados muy seguidos.
        """
        if not self.winfo_ismapped():
            self._detail_dirty = True
            return
        if self._detail_job is not None:
            self.after_cancel(self._detail_job)
        self._detail_job = self.after(delay_ms, self._draw_detail_plot)

    def _on_plot_configure(self, event: tk.Event) -> None:
        """Si el lienzo de la figura cambia mucho de tamaño, se vuelve a dibujar el espectro.

        :func:`src.plots.plot_note_spectrum` parte el subtítulo según el ancho de la
        figura y aquí se eligen las columnas de la leyenda; al redimensionar la ventana
        (o al aparecer el aviso) hay que recalcularlos.
        """
        old_w, old_h = self._plot_size
        if old_w <= 1 or (abs(event.width - old_w) < 40 and abs(event.height - old_h) < 30):
            return
        if self.selected is not None and self.transcription() is not None:
            self._schedule_detail_plot(delay_ms=150)

    def _on_detail_body_configure(self, event: tk.Event) -> None:
        """En paneles estrechos (≈ 1024 px) se oculta la columna «Nota» para dejar sitio al espectro."""
        narrow = event.width < NARROW_DETAIL_PX
        if narrow != self._narrow_table:
            self._narrow_table = narrow
            self.tree.configure(displaycolumns=self._shown_arm_columns() if narrow else "#all")
            self._wrap_table_labels()

    def _wrap_table_labels(self) -> None:
        """Parte el título y la leyenda de la tabla al ancho de sus columnas visibles.

        El ancho se calcula a partir de las columnas (no del propio texto ni de un
        ``<Configure>``): la columna de la tabla toma su ancho natural, y si el texto
        dependiera de ese ancho, Tk entraría en un bucle de geometría.
        """
        shown = set(self._shown_arm_columns())
        width = sum(c[2] for c in ARM_COLUMNS if c[0] in shown) + 14
        self.table_caption.configure(wraplength=width)
        self.table_legend.configure(wraplength=width)

    @property
    def detail_pending(self) -> bool:
        """True si la gráfica del espectro tiene un redibujo pendiente."""
        return self._detail_job is not None or self._detail_dirty

    def _draw_detail_plot(self) -> None:
        """Espectro del segmento con el template armónico del brazo elegido (src.plots.plot_note_spectrum)."""
        self._detail_job = None
        self._detail_dirty = False
        tr = self.transcription()
        pos = self.selected
        if self.analysis is None:
            self.plot.clear("Aquí verás el espectro de la nota seleccionada.")
            return
        if tr is None or pos is None or not 0 <= pos < len(tr.notes):
            self.plot.clear("Selecciona una nota para ver su espectro medio y el «peine» de armónicos\n"
                            "del brazo elegido: la recompensa suma la energía bajo los dientes.")
            return
        gt_note = self.gt_note_for(pos)
        gt_arm = Arm(gt_note.string, gt_note.fret) if gt_note is not None else None
        widget = self.plot.canvas.get_tk_widget()
        self._plot_size = (widget.winfo_width(), widget.winfo_height())
        self.plot.show(note_spectrum_figure, self.analysis, pos, tr.chain.arms[pos], gt_arm=gt_arm)

    # ================================================================ eventos
    def _on_analysis_ready(self, analysis: AnalysisResult | None = None, **_kwargs: Any) -> None:
        """Nuevo análisis: se olvida la selección y se redibuja (agrupado con las transcripciones)."""
        self.analysis = analysis if analysis is not None else self.app.state.analysis
        self.selected = self.app.state.selected_position
        self._recompute_comparison()
        self._request_refresh()

    def _on_transcriptions_ready(self, **_kwargs: Any) -> None:
        """Nuevas transcripciones (análisis o re-transcripción): se vuelve a puntuar y se redibuja."""
        self.analysis = self.app.state.analysis
        self._recompute_comparison()
        self._request_refresh()

    def _on_algorithm_selected(self, algorithm: str = "", **_kwargs: Any) -> None:
        """Otra pestaña cambió el algoritmo mostrado."""
        if algorithm in ALGORITHMS and algorithm != self.algorithm:
            self._show_algorithm(algorithm)

    def _on_algo_radio(self) -> None:
        """Botón de radio del algoritmo: lo muestra y avisa a las demás pestañas."""
        algorithm = self.algo_var.get()
        if algorithm != self.algorithm:
            self._show_algorithm(algorithm)
        self.app.state.select_algorithm(algorithm)

    def _show_algorithm(self, algorithm: str) -> None:
        """Cambia el algoritmo mostrado conservando la nota seleccionada y la posición de la vista."""
        left = self.canvas.canvasx(0)
        self.algorithm = algorithm
        self._recompute_comparison()
        self.refresh()
        total = max(self._content_width, self._visible_width())
        self.canvas.xview_moveto(left / total)
        self._update_sticky()

    def _on_segment_selected(self, position: int | None = None, source: str = "", **_kwargs: Any) -> None:
        """Selección desde otra pestaña: resaltar la nota y desplazar la vista hasta ella."""
        if position == self.selected:
            return  # ya sincronizado (p. ej. el eco de una selección hecha aquí)
        self.select(position, publish=False, scroll=True)

    def _on_config_changed(self, key: str | None = None, **_kwargs: Any) -> None:
        """Tolerancia de onset → nuevo emparejamiento; entorno/agentes/semilla → aviso «re-transcribir»."""
        if self.analysis is not None and key in (None, "experiment.onset_tolerance_s"):
            self._recompute_comparison()
            self.refresh()
        else:
            self._update_controls()

    def _on_busy_changed(self, busy: bool = False, **_kwargs: Any) -> None:
        """Deshabilita la exportación mientras hay una tarea en segundo plano."""
        self._busy = bool(busy)
        self._update_controls()
        if self.analysis is None:
            self._update_summary()
            self._redraw_canvas()

    def _on_map(self, _event: tk.Event) -> None:
        """Al mostrarse la pestaña, dibuja la gráfica pendiente y recoloca la vista."""
        if self._detail_dirty:
            self._schedule_detail_plot()
        if self.view_state == "tab" and self.selected is not None:
            self.after_idle(lambda: self.selected is not None and self.scroll_to(self.selected))

    def _on_compare_toggled(self) -> None:
        """Casilla «Comparar con ground truth»."""
        self._recompute_comparison()
        self.refresh()
        if self.selected is not None:
            self.scroll_to(self.selected)

    def _on_retranscribe(self) -> None:
        """Botón del aviso: re-transcribir con la configuración actual (acción de la aplicación)."""
        self.app.retranscribe()

    # ============================================================= exportación
    def _export_stem(self) -> str:
        """Nombre base de los archivos exportados (el del audio)."""
        return Path(self.analysis.path).stem if self.analysis is not None else "tablatura"

    def _export_dir(self) -> str:
        """Carpeta inicial de los diálogos de exportación."""
        if RESULTS_DIR.exists():
            return str(RESULTS_DIR)
        return str(Path(self.analysis.path).parent) if self.analysis is not None else str(Path.cwd())

    def export_meta(self, algorithm: str) -> dict[str, Any]:
        """Metadatos de la exportación (van completos al JSON; ``title`` encabeza el TXT).

        Parameters
        ----------
        algorithm : str
            Algoritmo exportado.

        Returns
        -------
        dict[str, Any]
            Título, algoritmo, audio, T, λ, semilla y, si hay ground truth, la
            precisión de pitch y de posición (recall sobre las notas reales).
        """
        name = ALGO_LABELS.get(algorithm, algorithm)
        meta: dict[str, Any] = {"title": f"Smartuner · {name}", "algorithm": algorithm, "algorithm_label": name}
        if self.analysis is not None:
            cfg = self.analysis.config
            meta["title"] += f" · {Path(self.analysis.path).name}"
            meta.update({"audio": str(self.analysis.path), "budget_T": int(cfg.env.budget),
                         "lambda": float(cfg.env.lam), "seed": int(cfg.experiment.seed), "run_index": 0})
            tr = self.transcription(algorithm)
            gt = self.ground_truth
            if tr is not None and gt:
                segments = [d.segment for d in self.analysis.segment_data]
                acc = score_arms(tr.chain.arms, match_segments_to_gt(
                    segments, gt, float(self.app.state.config.experiment.onset_tolerance_s)), gt)
                meta.update({"pitch_accuracy": round(acc.pitch, 4), "position_accuracy": round(acc.position, 4),
                             "n_ground_truth_notes": len(gt)})
        return meta

    def export_current(self, fmt: str, path: str | Path | None = None) -> Path | None:
        """Exporta la tablatura del algoritmo mostrado (botones TXT / JSON / CSV).

        Parameters
        ----------
        fmt : str
            ``"txt"``, ``"json"`` o ``"csv"``.
        path : str | Path | None, optional
            Archivo de salida; si es None se pregunta con un diálogo.

        Returns
        -------
        Path | None
            Ruta escrita, o None si se canceló o falló.
        """
        tr = self.transcription()
        if tr is None:
            self.app.status.set_message("No hay tablatura que exportar: primero analiza un audio.")
            return None
        if path is None:
            path = filedialog.asksaveasfilename(
                parent=self, title=f"Exportar tablatura de {ALGO_LABELS.get(self.algorithm, self.algorithm)} "
                                   f"({fmt.upper()})",
                defaultextension=f".{fmt}", initialdir=self._export_dir(),
                initialfile=f"{self._export_stem()}_tab_{self.algorithm}.{fmt}",
                filetypes=[(EXPORT_FORMATS[fmt], f"*.{fmt}"), ("Todos los archivos", "*.*")])
            if not path:
                return None
        target = Path(path)
        if target.suffix.lower() != f".{fmt}":
            target = target.with_name(target.name + f".{fmt}")
        try:
            written = export_tab(tr.notes, target, meta=self.export_meta(self.algorithm))
        except Exception as exc:  # noqa: BLE001 - se informa al usuario
            show_error(self, "No se pudo exportar la tablatura", exc)
            return None
        self.app.status.set_message(f"Tablatura de {ALGO_LABELS.get(self.algorithm, self.algorithm)} guardada en "
                                    f"{written}")
        return written

    def export_all(self, formats: tuple[str, ...] = ("txt",), folder: str | Path | None = None) -> list[Path]:
        """Exporta la tablatura de los cuatro algoritmos a una carpeta (un archivo por algoritmo y formato).

        Parameters
        ----------
        formats : tuple[str, ...], optional
            Formatos (``"txt"``, ``"json"``, ``"csv"``).
        folder : str | Path | None, optional
            Carpeta de salida; si es None se pregunta con un diálogo.

        Returns
        -------
        list[Path]
            Archivos escritos (vacía si se canceló o no había tablaturas).
        """
        algorithms = [a for a in ALGORITHMS if self.transcription(a) is not None]
        if not algorithms:
            self.app.status.set_message("No hay tablaturas que exportar: primero analiza un audio.")
            return []
        if folder is None:
            folder = filedialog.askdirectory(parent=self, title="Carpeta para las tablaturas de los 4 algoritmos",
                                             initialdir=self._export_dir(), mustexist=False)
            if not folder:
                return []
        written: list[Path] = []
        try:
            for alg in algorithms:
                tr = self.transcription(alg)
                assert tr is not None
                for fmt in formats:
                    path = Path(folder) / f"{self._export_stem()}_tab_{alg}.{fmt}"
                    written.append(export_tab(tr.notes, path, meta=self.export_meta(alg)))
        except Exception as exc:  # noqa: BLE001 - se informa al usuario
            show_error(self, "No se pudieron exportar las tablaturas", exc)
            return written
        logger.info("Tablaturas de %d algoritmos exportadas a %s (%d archivos)", len(algorithms), folder,
                    len(written))
        self.app.status.set_message(f"Se guardaron {len(written)} archivos de tablatura en {folder}")
        return written

    # ================================================================ ayudas
    def _algo_status(self, algorithm: str) -> str:
        """Línea de estado de un algoritmo para su tooltip (notas y precisión si hay GT)."""
        tr = self.transcription(algorithm)
        if tr is None:
            return "Todavía no hay tablatura de este algoritmo."
        text = f"Tablatura actual: {len(tr.notes)} notas."
        gt = self.ground_truth
        if gt and self.analysis is not None:
            segments = [d.segment for d in self.analysis.segment_data]
            acc = score_arms(tr.chain.arms, match_segments_to_gt(
                segments, gt, float(self.app.state.config.experiment.onset_tolerance_s)), gt)
            text += f" Pitch {100 * acc.pitch:.0f} %, posición {100 * acc.position:.0f} % (frente al ground truth)."
        return text

    def _compare_help(self) -> str:
        """Ayuda de la casilla «Comparar con ground truth» (depende de si hay GT)."""
        base = ("Colorea cada nota según la posición real (✓ exacta, ~ pitch correcto en otra posición, ✗ pitch "
                "incorrecto, + segmento sin nota real) y dibuja debajo una pauta con las posiciones reales. Cada "
                "segmento se empareja con la nota real cuyo inicio está a ≤ la tolerancia de onset del experimento.")
        if not self.ground_truth:
            return base + "\n\nEste audio no tiene ground truth (archivo <nombre>.gt.json junto al MP3)."
        return base

    def _summary_help(self) -> str:
        """Ayuda del resumen (qué significan las cifras)."""
        return ("pitch = notas reales cuyo segmento recibió la nota correcta (en cualquier posición); posición = "
                "notas reales con la cuerda y el traste exactos. Ambas son recall: aciertos / notas del ground "
                "truth (una nota real sin detectar cuenta como error). Es UNA corrida; la pestaña Comparación "
                "promedia muchas.")
