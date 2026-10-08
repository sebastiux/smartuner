"""Pestaña 5 · Comparación: el experimento que responde «¿qué algoritmo bandit es mejor?».

Papel en la interfaz
--------------------
Es la pestaña con la que se cierra la explicación en clase: muestra los
resultados del experimento comparativo (:func:`src.experiments.run_experiment`)
de los cuatro algoritmos —ε-greedy, ε-greedy optimista, UCB1 y Softmax— con
números aleatorios comunes. No calcula nada por su cuenta: lanza el
experimento con la ACCIÓN ``app.run_experiment``, dibuja con las funciones del
catálogo :data:`src.plots.COMPARISON_PLOTS` (más
:func:`src.plots.plot_spectrogram`), rellena la tabla con
:meth:`src.experiments.ExperimentResult.summary_rows` y exporta con
:func:`src.plots.save_all_plots` y los métodos ``to_csv`` /
``curves_to_csv`` / ``timing_to_csv`` del resultado.

Contenido::

    ┌──────────────────────────────┬──────────────────────────────────────────────────┐
    │ [▶ Ejecutar experimento      │ [◀][▶] Gráfica 3 de 11  [Guardar gráfica actual…] │
    │   completo]                  │                         [Exportar todas…]         │
    │ estado (vigente / desfasado) │ ┌──────────────────────────────────────────────┐ │
    │                              │ │  Figura de matplotlib (src.plots)            │ │
    │ Configuración del experimento│ │  + barra de herramientas (zoom, desplazar)   │ │
    │   corridas · semilla, T,     │ └──────────────────────────────────────────────┘ │
    │   algoritmos, barridos, λ,   │ ▾ Tabla comparativa final   [Exportar tabla CSV…] │
    │   tiempo estimado            │                 [… curvas CSV…] [… tiempos CSV…]  │
    │   Editar en Configuración →  │ ■ ε-greedy       0.530 ± 0.008  39.8 ± 4.4  …     │
    │ Gráficas (lista, en gris las │ □ Oráculo miope  —              0.00        …     │
    │   que aún no tienen datos)   │ nota: media ± desviación entre corridas · ★ mejor │
    │ Cómo leer esta gráfica       │                                                  │
    └──────────────────────────────┴──────────────────────────────────────────────────┘

La columna izquierda ocupa toda la altura (la descripción necesita sitio para
leerse en clase); la tabla va bajo la figura. Antes del experimento la figura
muestra un mensaje guía (abrir un MP3 o generar el dataset, y la duración
estimada) y la tabla queda vacía.

En ventanas bajas (≈ 1024×700) la pestaña entra en modo compacto: la tabla
empieza plegada (el título «▸ Tabla comparativa final» la despliega; la
elección del usuario se respeta), su nota pasa a un tooltip «ⓘ», las figuras
pierden el subtítulo cuando el lienzo es bajo y los botones CSV se acortan si
no caben. En ventanas estrechas la tabla se desplaza en horizontal.

Sincronización con las demás pestañas
-------------------------------------
* ``experiment_ready`` → rellena la tabla, marca qué gráficas tienen datos y
  muestra la primera (si ya había resultados, conserva la gráfica elegida).
* ``analysis_ready`` → limpia todo: el experimento anterior ya no corresponde
  al audio (o a los datos bandit) vigentes.
* ``config_changed`` → actualiza el resumen de la configuración, la duración
  estimada y el aviso «resultados con otra configuración».
* ``busy_changed`` → deshabilita «Ejecutar» y las exportaciones que lanzan
  tareas mientras hay otra en curso.
* ``segment_selected`` → resalta el segmento en el espectrograma y la columna
  correspondiente en el mapa de aciertos. Un clic en esas gráficas selecciona
  el segmento en todas las pestañas con ``source="compare"``; la pestaña solo
  redibuja si su vista no está ya sincronizada, así que no hay bucles.
* ``algorithm_selected`` → resalta la fila del algoritmo en la tabla; un clic
  en una fila de algoritmo lo selecciona en las demás pestañas.

Las gráficas solo se dibujan cuando la pestaña está visible (al volver a ella
se dibuja lo pendiente) y el mensaje guía y el mapa de aciertos —cuyo tamaño de
letra depende del ancho— se recalculan al redimensionar la ventana.
"""

from __future__ import annotations

import contextlib
import logging
import math
import textwrap
import threading
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter import font as tkfont
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Annotation

from gui.widgets import UI_ACCENT, PlotFrame, Tooltip, show_error
from src import plots
from src.config import ALGO_COLORS, ALGORITHMS, RESULTS_DIR, Config, ProgressCallback
from src.experiments import (
    ALGO_SHORT,
    SWEEP_SPECS,
    SWEEP_SYMBOLS,
    effective_sweep_runs,
    estimate_experiment_seconds,
    match_segments_to_gt,
)

if TYPE_CHECKING:  # solo para las anotaciones de tipo (evita importes circulares)
    from gui.app import SmartunerApp
    from src.experiments import ExperimentResult
    from src.pipeline import AnalysisResult

logger = logging.getLogger(__name__)

#: Identificador de esta pestaña como origen de los eventos ``segment_selected``.
SOURCE = "compare"

#: Clave de la gráfica extra (no pertenece al catálogo comparativo: usa el análisis).
SPECTROGRAM_KEY = "spectrogram"

#: Entrada de la lista para el espectrograma (``func`` recibe el ANÁLISIS, no el experimento).
SPECTROGRAM_SPEC = plots.PlotSpec(
    SPECTROGRAM_KEY, "Espectrograma con onsets y f0", plots.plot_spectrogram,
    "CQT de la pista en dB: el eje vertical es logarítmico, así que los armónicos h·f0 de cada nota "
    "forman un «peine» de líneas paralelas, que es lo que mide la recompensa. Las líneas verticales "
    "finas son los onsets detectados, las franjas grises los segmentos descartados (silencio o "
    "demasiado cortos) y la curva, la f0 de pYIN. Cada segmento conservado es un problema bandit "
    "independiente. Haz clic sobre una nota para seleccionarla en todas las pestañas.",
)

#: Duración orientativa del experimento con la configuración por defecto (pista de ≈ 16 notas).
DEFAULT_DURATION_TEXT = "≈2 min con la configuración por defecto"

#: Resolución (puntos por pulgada) de las gráficas exportadas en formatos rasterizados.
EXPORT_DPI = 150

#: Ancho (px) de la columna izquierda (botón, resumen, lista y descripción).
LEFT_COLUMN_PX = 300

#: Ancho máximo (px) de los textos de la columna izquierda.
LEFT_WRAP_PX = LEFT_COLUMN_PX - 20

#: Por debajo de esta altura (px) de la columna derecha (ventanas de ≈ 700 px) la pestaña
#: pasa a modo compacto: la tabla empieza plegada (salvo que el usuario la haya abierto o
#: cerrado a mano) y, si se muestra, su nota pasa a un tooltip «ⓘ».
COMPACT_RIGHT_PX = 720

#: Por debajo de esta altura (px) del lienzo se quita el subtítulo de las figuras (sus datos
#: —corridas, T, segmentos— ya están en la nota de la tabla) para dejar sitio a los ejes.
COMPACT_CANVAS_PX = 420

#: Filas visibles mínimas de la lista de gráficas (crece con la ventana hasta mostrarlas todas).
LIST_MIN_ROWS = 6

#: Espera (ms) antes de recalcular una figura tras redimensionar la ventana.
REPLOT_DELAY_MS = 250

#: Cambio mínimo de tamaño (px) del lienzo que provoca recalcular la figura.
RESIZE_TOLERANCE_PX = 24

#: Claves de las filas de referencia de :meth:`ExperimentResult.summary_rows`.
ORACLES: tuple[str, ...] = ("oracle", "oracle_viterbi")

#: Colores de los avisos (ámbar oscuro sobre ámbar muy claro, como el resto de la GUI).
WARN_FG = "#7a5200"

#: Fondo de las filas de los oráculos (referencias neutras, recesivas).
ORACLE_ROW_BG = "#f4f3ef"

#: Fondo de la fila del algoritmo seleccionado en las vistas de un solo algoritmo.
SELECTED_ROW_BG = "#dbe9fb"

#: Marca del mejor algoritmo en cada columna de la tabla.
BEST_MARK = " ★"

#: Ayuda de cada dato del resumen de configuración: (clave, etiqueta, ayuda).
PLAN_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("runs", "Corridas",
     "Repeticiones independientes del experimento principal; las curvas y la tabla muestran la media ± la "
     "desviación estándar ENTRE corridas. Semilla maestra: con la misma semilla y configuración el "
     "experimento da exactamente los mismos resultados, y los cuatro algoritmos ven la misma secuencia de "
     "frames (números aleatorios comunes)."),
    ("budget", "Presupuesto T",
     "Pulls por segmento (env.budget): cuántas veces puede probar el agente una posición (cuerda, traste) "
     "antes de recomendar una."),
    ("algorithms", "Algoritmos",
     "Algoritmos incluidos (casillas «Incluir en el experimento» de «2 · Configuración»). El cuadro es "
     "su color en todas las gráficas."),
    ("sweeps", "Barridos",
     "Barridos de sensibilidad: se repite el experimento variando ε (ε-greedy), Q₀ (optimista), c (UCB1) "
     "y τ (Softmax), cada uno con «corridas por valor» repeticiones (menos en pistas largas)."),
    ("lambda", "Barrido de λ",
     "Repite el experimento con varios λ (peso de la penalización de tocabilidad) para ver cómo cambia "
     "la precisión de posición. Requiere ground truth (.gt.json junto al audio)."),
    ("estimate", "Tiempo estimado",
     "Estimación a partir del coste medido por pull de cada algoritmo (src.experiments."
     "estimate_experiment_seconds). Crece linealmente con el número de notas."),
)


@dataclass(frozen=True)
class TableColumn:
    """Columna de la tabla comparativa.

    Attributes
    ----------
    key : str
        Clave de la media en :meth:`ExperimentResult.summary_rows` (la
        desviación es ``key + "_std"``).
    heading : str
        Encabezado en español.
    width : int
        Ancho inicial (px).
    kind : str
        Formato: ``"reward"`` (3 decimales), ``"regret"``, ``"pct"`` (ya en
        %), ``"frac"`` (fracción 0–1 mostrada en %), ``"count"`` (entero) o
        ``"ms"`` (milisegundos).
    better : str | None
        ``"high"`` o ``"low"`` si la columna tiene un «mejor» valor (se marca
        con ★); None si no.
    help : str
        Explicación (tooltip del encabezado).
    needs_gt : bool
        Si es True la columna solo se muestra cuando hay ground truth.
    extra : bool
        Si es True solo se muestra cuando hay segmentos de más (con 0
        segmentos de más el F1 coincide con la precisión).
    """

    key: str
    heading: str
    width: int
    kind: str
    better: str | None
    help: str
    needs_gt: bool = False
    extra: bool = False


#: Ayuda de la primera columna (algoritmo, con el cuadro de su color).
ALGORITHM_COLUMN_HELP = (
    "Algoritmo bandit; el cuadro es su color en todas las gráficas. En gris, los oráculos de referencia, "
    "que conocen μ: el miope elige argmax μ en cada segmento y el de cadena (Viterbi) maximiza la suma de "
    "μ de toda la pista con el mismo modelo de tocabilidad. Son techos de RECOMPENSA, no de precisión. "
    "Clic en una fila de algoritmo = mostrarlo en «Ejecución en vivo» y «Tablatura»."
)

#: Columnas de la tabla comparativa, en orden.
TABLE_COLUMNS: tuple[TableColumn, ...] = (
    TableColumn("mean_reward", "Recompensa media", 146, "reward", "high",
                "Recompensa media por pull en todo el presupuesto T, promediada sobre los segmentos. Más alta = "
                "el agente pasó más pulls en brazos buenos. Los oráculos no tienen: no exploran."),
    TableColumn("final_regret", "Regret final", 104, "regret", "low",
                "Regret acumulado tras los T pulls, medio por segmento: Σₜ (μ* − μ_aₜ). Mide lo perdido por "
                "explorar o equivocarse: más bajo = mejor. El oráculo miope vale 0 por definición; el de cadena "
                "no elige argmax μ en cada segmento, así que no se mide así."),
    TableColumn("optimal_pct", "% pulls óptimos", 126, "pct", "high",
                "Porcentaje de pulls al brazo óptimo (argmax μ) en el último 10 % del presupuesto: indica si el "
                "agente terminó fijándose en la mejor posición."),
    TableColumn("pitch_acc", "Precisión pitch", 122, "frac", "high",
                "Notas del ground truth cuyo segmento recibió la nota (MIDI) correcta: aciertos / notas reales "
                "(recall). Las notas no detectadas cuentan como error. Requiere ground truth.", needs_gt=True),
    TableColumn("position_acc", "Precisión posición", 140, "frac", "high",
                "Notas del ground truth con la cuerda Y el traste correctos. Es la métrica de la tablatura: "
                "posiciones con el mismo pitch solo se distinguen por la penalización de tocabilidad λ.",
                needs_gt=True),
    TableColumn("pitch_f1", "F1 pitch", 100, "frac", "high",
                "F1 = 2·aciertos / (notas reales + segmentos): como la precisión, pero también penaliza los "
                "segmentos de más (onsets falsos, notas partidas).", needs_gt=True, extra=True),
    TableColumn("position_f1", "F1 posición", 108, "frac", "high",
                "F1 de la posición (cuerda y traste): 2·aciertos / (notas reales + segmentos).",
                needs_gt=True, extra=True),
    TableColumn("ms_per_segment", "Tiempo / segmento", 140, "ms", "low",
                "Tiempo de pared medio para resolver un segmento (T pulls de elegir brazo + recompensa + "
                "actualizar Q), en milisegundos. Depende de la máquina, por eso no va en el CSV reproducible de "
                "la tabla: se exporta aparte con «Exportar tiempos CSV…»."),
)

#: Tarjetas del mensaje guía: qué se verá tras el experimento.
EMPTY_CARDS: tuple[tuple[str, str], ...] = (
    ("Curvas de aprendizaje", "recompensa, regret y\n% de pulls al brazo óptimo"),
    ("Segmento de ejemplo", "pulls por brazo y\nevolución de Q frente a μ"),
    ("Barridos", "sensibilidad a ε, Q₀, c y τ;\nefecto de λ en la precisión"),
    ("Frente al ground truth", "precisión de la tablatura\ny aciertos nota a nota"),
)


# ---------------------------------------------------------------------------
# Formato (funciones puras: se prueban sin ventana)
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """True si ``value`` es un número finito (no None ni NaN)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _decimals(kind: str, value: float) -> int:
    """Decimales con que se muestra un valor de la columna de tipo ``kind``."""
    if kind == "reward":
        return 3
    if kind in ("regret", "ms"):
        return 2 if abs(value) < 1.0 else 1
    if kind == "count":
        return 0
    return 1  # porcentajes


def display_value(value: object, kind: str) -> float | None:
    """Valor tal como se muestra (escalado a % y redondeado), o None si falta.

    Parameters
    ----------
    value : object
        Media de :meth:`ExperimentResult.summary_rows` (puede ser None).
    kind : str
        Tipo de columna (ver :class:`TableColumn`).

    Returns
    -------
    float | None
        Valor redondeado a los decimales mostrados; sirve para decidir
        empates al marcar el mejor algoritmo.

    Examples
    --------
    >>> display_value(0.98504, "frac"), display_value(None, "reward")
    (98.5, None)
    """
    if not _is_number(value):
        return None
    v = float(value) * (100.0 if kind == "frac" else 1.0)  # type: ignore[arg-type]
    return round(v, _decimals(kind, v))


def format_cell(mean: object, std: object, kind: str, with_std: bool = True) -> str:
    """Texto de una celda: ``"0.712 ± 0.031"``, ``"98.5 ± 1.2 %"``, ``"4.1 ± 0.2 ms"``.

    Parameters
    ----------
    mean, std : object
        Media y desviación estándar entre corridas (None = no aplica).
    kind : str
        Tipo de columna (ver :class:`TableColumn`): ``"frac"`` se multiplica por 100.
    with_std : bool, optional
        Si es False se omite «± std» (filas deterministas, como los oráculos).

    Returns
    -------
    str
        Texto formateado; ``"—"`` si la media falta.

    Examples
    --------
    >>> format_cell(0.71234, 0.0312, "reward")
    '0.712 ± 0.031'
    >>> format_cell(0.985, 0.012, "frac"), format_cell(73.2, 5.71, "pct")
    ('98.5 ± 1.2 %', '73.2 ± 5.7 %')
    >>> format_cell(4.123, 0.21, "ms"), format_cell(None, None, "ms"), format_cell(1.0, 0.0, "frac", with_std=False)
    ('4.1 ± 0.2 ms', '—', '100.0 %')
    >>> format_cell(3, None, "count"), format_cell(39.78, 4.39, "regret")
    ('3', '39.8 ± 4.4')
    """
    if not _is_number(mean):
        return "—"
    scale = 100.0 if kind == "frac" else 1.0
    m = float(mean) * scale  # type: ignore[arg-type]
    if kind == "count":
        return str(int(round(m)))
    digits = _decimals(kind, m)
    unit = {"pct": " %", "frac": " %", "ms": " ms"}.get(kind, "")
    text = f"{m:.{digits}f}"
    if with_std and _is_number(std):
        text += f" ± {float(std) * scale:.{digits}f}"  # type: ignore[arg-type]
    return text + unit


def best_algorithms(rows: Sequence[dict[str, object]], column: TableColumn) -> set[str]:
    """Algoritmos con el mejor valor MOSTRADO de ``column`` (excluye los oráculos).

    Parameters
    ----------
    rows : Sequence[dict[str, object]]
        Filas de :meth:`ExperimentResult.summary_rows`.
    column : TableColumn
        Columna a evaluar (``better`` decide si gana el mayor o el menor).

    Returns
    -------
    set[str]
        Claves de los algoritmos ganadores (todos los empatados al redondear).
        Vacío si la columna no tiene «mejor», hay menos de dos algoritmos con
        valor o todos empatan (marcarlos no informaría nada).

    Examples
    --------
    >>> col = TABLE_COLUMNS[1]                                   # regret: menor es mejor
    >>> rows = [{"algorithm": "egreedy", "final_regret": 40.0}, {"algorithm": "ucb1", "final_regret": 21.2},
    ...         {"algorithm": "oracle", "final_regret": 0.0}]
    >>> best_algorithms(rows, col)
    {'ucb1'}
    """
    if column.better is None:
        return set()
    shown = [(str(r["algorithm"]), display_value(r.get(column.key), column.kind))
             for r in rows if r.get("algorithm") not in ORACLES]
    shown = [(a, v) for a, v in shown if v is not None]
    if len(shown) < 2:
        return set()
    values = [v for _a, v in shown]
    target = max(values) if column.better == "high" else min(values)
    if all(v == target for v in values):
        return set()
    return {a for a, v in shown if v == target}


def format_duration(seconds: float) -> str:
    """Duración legible: ``"42 s"`` o ``"3.5 min"``.

    Examples
    --------
    >>> format_duration(13.1), format_duration(104.3)
    ('13 s', '1.7 min')
    """
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60.0:.1f} min"


def estimate_text(cfg: Config, analysis: AnalysisResult | None) -> str:
    """Duración estimada del experimento con ``cfg`` (texto para el resumen).

    Parameters
    ----------
    cfg : Config
        Configuración vigente.
    analysis : AnalysisResult | None
        Análisis vigente (da el número de notas y si hay ground truth).

    Returns
    -------
    str
        ``"≈ 1.7 min (16 notas)"``; sin análisis, el coste por nota.
    """
    try:
        if analysis is None:
            per_note = estimate_experiment_seconds(1, cfg, with_lambda=True)["total"]
            return f"≈ {per_note:.1f} s por nota"
        n_notes = len(analysis.segment_data)
        est = estimate_experiment_seconds(n_notes, cfg, with_lambda=bool(analysis.ground_truth))
    except Exception:  # noqa: BLE001 - p. ej. sin algoritmos seleccionados
        return "no disponible con la configuración actual"
    return f"≈ {format_duration(est['total'])} ({n_notes} notas)"


def experiment_plan(cfg: Config, analysis: AnalysisResult | None) -> dict[str, str]:
    """Resumen legible de lo que hará el experimento con ``cfg``.

    Parameters
    ----------
    cfg : Config
        Configuración vigente (``env.budget`` y la sección ``experiment``).
    analysis : AnalysisResult | None
        Análisis vigente: con él se calculan las corridas efectivas de los
        barridos, si habrá barrido de λ (requiere ground truth) y la duración.

    Returns
    -------
    dict[str, str]
        Un texto por clave de :data:`PLAN_FIELDS`.

    Examples
    --------
    >>> plan = experiment_plan(Config(), None)
    >>> plan["runs"], plan["budget"], plan["sweeps"], plan["lambda"]
    ('100 · semilla 42', '500 pulls por nota', 'ε, Q₀, c, τ · 30 corridas', 'sí · 6 valores')
    """
    exp = cfg.experiment
    selected = [a for a in ALGORITHMS if a in exp.algorithms]
    plan: dict[str, str] = {
        "runs": f"{exp.n_runs} · semilla {exp.seed}",
        "budget": f"{cfg.env.budget} pulls por nota",
        "algorithms": ", ".join(ALGO_SHORT.get(a, a) for a in selected) or "ninguno (elige al menos uno)",
    }
    symbols = [SWEEP_SYMBOLS[param] for param, algo, list_name in SWEEP_SPECS
               if algo in selected and getattr(exp, list_name)]
    if exp.run_sweeps and symbols:
        runs = effective_sweep_runs(cfg, len(analysis.segment_data)) if analysis is not None else exp.sweep_runs
        plan["sweeps"] = f"{', '.join(symbols)} · {runs} corridas"
    else:
        plan["sweeps"] = "no"
    if not exp.run_lambda_sweep:
        plan["lambda"] = "no"
    elif analysis is not None and not analysis.ground_truth:
        plan["lambda"] = "no (el audio no tiene ground truth)"
    else:
        plan["lambda"] = f"sí · {len(exp.sweep_lambda)} valores"
    plan["estimate"] = estimate_text(cfg, analysis)
    return plan


def _drop_subtitle(fig: Figure) -> None:
    """Quita el subtítulo de una figura de :mod:`src.plots` (modo compacto).

    :func:`src.plots._titles` reserva líneas en blanco bajo el título
    (``suptitle``) y ancla ahí el subtítulo con una ``Annotation``; se quitan
    ambas cosas para que los ejes ganen ese espacio. Las figuras exportadas no
    se tocan (se dibujan de nuevo, completas).
    """
    sup = getattr(fig, "_suptitle", None)
    if sup is None:
        return
    for artist in list(fig.artists):
        if isinstance(artist, Annotation) and getattr(artist, "xycoords", None) is sup:
            artist.remove()
    sup.set_text(sup.get_text().rstrip("\n"))


def _autoscroll(scrollbar: ttk.Scrollbar) -> Callable[[str, str], None]:
    """``*scrollcommand`` que muestra la barra solo cuando el contenido no cabe.

    La barra debe estar colocada con ``grid`` (``grid_remove`` recuerda sus opciones).
    """

    def set_(first: str, last: str) -> None:
        if float(first) <= 0.0 and float(last) >= 1.0:
            scrollbar.grid_remove()
        else:
            scrollbar.grid()
        scrollbar.set(first, last)

    return set_


class _HeadingTooltip(Tooltip):
    """Tooltip de la tabla que explica la columna bajo el ratón (solo en los encabezados).

    Parameters
    ----------
    tree : ttk.Treeview
        Tabla a la que se asocia.
    help_for : Callable[[str], str]
        Función ``columna → texto`` (``"#0"`` es la columna del algoritmo).
    """

    def __init__(self, tree: ttk.Treeview, help_for: Callable[[str], str]) -> None:
        """Asocia el tooltip a ``tree`` y sigue el movimiento del ratón."""
        self._tree = tree
        self._help_for = help_for
        self._hover: str | None = None
        super().__init__(tree, self._current_text, wraplength=340)
        tree.bind("<Motion>", self._on_motion, add="+")

    def column_under_pointer(self) -> str | None:
        """Columna cuyo encabezado está bajo el puntero (None fuera de los encabezados)."""
        tree = self._tree
        x = tree.winfo_pointerx() - tree.winfo_rootx()
        y = tree.winfo_pointery() - tree.winfo_rooty()
        return self.column_at(x, y)

    def column_at(self, x: int, y: int) -> str | None:
        """Columna del encabezado en ``(x, y)`` (px relativos a la tabla) o None.

        Parameters
        ----------
        x, y : int
            Coordenadas dentro de la tabla (px).

        Returns
        -------
        str | None
            ``"#0"`` para la columna del algoritmo, la clave de la columna en
            las demás, o None si el punto no está sobre un encabezado.
        """
        tree = self._tree
        if tree.identify_region(x, y) not in ("heading", "separator"):
            return None
        column = tree.identify_column(x)
        if column == "#0":
            return "#0"
        try:
            index = int(column.lstrip("#")) - 1
        except ValueError:
            return None
        shown = _display_columns(tree)
        return shown[index] if 0 <= index < len(shown) else None

    def _current_text(self) -> str:
        """Texto de la columna actual (vacío = no mostrar nada)."""
        column = self.column_under_pointer()
        return self._help_for(column) if column else ""

    def _on_motion(self, _event: tk.Event) -> None:
        """Reinicia el tooltip cuando el ratón pasa a otro encabezado."""
        column = self.column_under_pointer()
        if column != self._hover:
            self._hover = column
            self._hide()
            if column:
                self._schedule()

    def _show(self) -> None:
        """Muestra el texto junto al puntero, siempre dentro de la pantalla."""
        text = self._current_text()
        if not text or self._tip is not None:
            return
        tree = self._tree
        self._tip = tip = tk.Toplevel(tree)
        tip.withdraw()
        tip.wm_overrideredirect(True)
        tk.Label(tip, text=text, justify="left", wraplength=self.wraplength, background="#fffbe8",
                 foreground=plots.TEXT_PRIMARY, relief="solid", borderwidth=1, padx=8, pady=6,
                 font=("TkDefaultFont", 9)).pack()
        tip.update_idletasks()
        x = tree.winfo_pointerx() + 14
        y = tree.winfo_pointery() + 18
        if y + tip.winfo_reqheight() > tree.winfo_screenheight():
            y = tree.winfo_pointery() - tip.winfo_reqheight() - 10
        x = min(x, max(0, tree.winfo_screenwidth() - tip.winfo_reqwidth() - 4))
        tip.wm_geometry(f"+{x}+{y}")
        tip.deiconify()


def _display_columns(tree: ttk.Treeview) -> list[str]:
    """Columnas visibles de ``tree`` en orden (resuelve ``displaycolumns = "#all"``)."""
    shown = tree.cget("displaycolumns")
    shown = list(tree.tk.splitlist(shown) if isinstance(shown, str) else shown)
    if not shown or shown == ["#all"]:
        columns = tree.cget("columns")
        return list(tree.tk.splitlist(columns) if isinstance(columns, str) else columns)
    return [str(c) for c in shown]


# ---------------------------------------------------------------------------
# Pestaña
# ---------------------------------------------------------------------------


class CompareTab(ttk.Frame):
    """Pestaña 5 · Comparación: experimento comparativo, gráficas, tabla y exportación.

    Parameters
    ----------
    master : tk.Misc
        Contenedor (el ``ttk.Notebook`` de la ventana principal).
    app : SmartunerApp
        Aplicación: estado compartido (``app.state``), acciones
        (``run_experiment``, ``show_tab``, ``run_task``) y barra de estado.

    Attributes
    ----------
    result : ExperimentResult | None
        Experimento mostrado (None antes de ejecutarlo o tras un análisis nuevo).
    plot_specs : list[src.plots.PlotSpec]
        Gráficas de la lista, en orden: el catálogo comparativo y el espectrograma.
    plot_state : str
        Qué muestra la figura: ``"empty"`` (mensaje guía), ``"plot"``,
        ``"missing"`` (la gráfica explica qué dato falta) o ``"error"``.
    drawn_key : str | None
        Clave de la gráfica dibujada.
    drawn_selection : int | None
        Segmento resaltado en la figura (espectrograma o mapa de aciertos).
    table_rows : list[dict[str, str]]
        Textos mostrados en la tabla: ``{"algorithm", "label", <columna>: texto}``.
    plan_values : dict[str, str]
        Textos del resumen de la configuración (ver :func:`experiment_plan`).
    table_visible : bool
        Si la tabla está desplegada (ver :meth:`toggle_table`).
    compact : bool
        Modo compacto (ventana baja, ver :data:`COMPACT_RIGHT_PX`).
    """

    def __init__(self, master: tk.Misc, app: SmartunerApp) -> None:
        """Construye la pestaña y se suscribe a los eventos de ``app.state.events``."""
        super().__init__(master, padding=10)
        self.app = app
        self.result: ExperimentResult | None = None
        self.plot_specs: list[plots.PlotSpec] = [*plots.COMPARISON_PLOTS, SPECTROGRAM_SPEC]
        self.plot_state = "empty"
        self.drawn_key: str | None = None
        self.drawn_selection: int | None = None
        self.table_rows: list[dict[str, str]] = []
        self.plan_values: dict[str, str] = {}
        self._busy = bool(app.busy)
        self._redraw_pending = False
        self._draw_after: str | None = None
        self._replot_after: str | None = None
        self._drawn_size: tuple[int, int] = (0, 0)
        self._drawn_compact = False
        self._table_choice: bool | None = None  # None = automático (plegada solo en modo compacto)
        self._drawn_call: tuple[Callable[..., Figure], tuple[Any, ...], dict[str, Any]] | None = None
        self._matches: list[int | None] | None = None
        self._heat_overlay: list[Any] = []
        self._last_dir: Path = RESULTS_DIR
        self._swatches: dict[str, tk.PhotoImage] = {}

        self._setup_styles()
        self._make_swatches()
        self.columnconfigure(0, minsize=LEFT_COLUMN_PX)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.compact = False
        self._build_plot_area(right)
        self._build_table(right)
        right.bind("<Configure>", self._on_right_configure, add="+")

        events = app.state.events
        events.subscribe("analysis_ready", self._on_analysis_ready)
        events.subscribe("experiment_ready", self._on_experiment_ready)
        events.subscribe("segment_selected", self._on_segment_selected)
        events.subscribe("algorithm_selected", self._on_algorithm_selected)
        events.subscribe("config_changed", self._on_config_changed)
        events.subscribe("busy_changed", self._on_busy_changed)
        self.bind("<Map>", self._on_map, add="+")

        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self._fill_table()
        self._refresh_static()
        if app.state.experiment is not None:
            self._on_experiment_ready(result=app.state.experiment)
        else:
            self._schedule_draw()

    # ================================================================ construcción
    def _setup_styles(self) -> None:
        """Estilos propios de la pestaña (tabla, avisos, enlace, contador)."""
        style = ttk.Style(self)
        self._frame_bg = style.lookup("TFrame", "background") or "#dcdad5"
        self._field_bg = style.lookup("Treeview", "fieldbackground") or "white"
        style.configure("Compare.Treeview", rowheight=23)
        style.map("Compare.Treeview", background=[("selected", SELECTED_ROW_BG)],
                  foreground=[("selected", plots.TEXT_PRIMARY)])
        style.configure("CmpWarn.TLabel", foreground=WARN_FG)
        style.configure("CmpKey.TLabel", foreground=plots.TEXT_SECONDARY)
        style.configure("CmpCounter.TLabel", font=("TkDefaultFont", 10, "bold"))
        # «Enlace»: botón plano con texto de acento (sigue siendo un botón: se puede invocar y enfocar).
        style.configure("CmpLink.TButton", foreground=UI_ACCENT, background=self._frame_bg, borderwidth=0,
                        relief="flat", padding=(0, 1), focuscolor=self._frame_bg,
                        font=("TkDefaultFont", 9, "underline"))
        style.map("CmpLink.TButton", foreground=[("disabled", plots.TEXT_MUTED), ("active", "#1c5cab")],
                  background=[("active", self._frame_bg), ("pressed", self._frame_bg)])
        style.configure("CmpTitle.TButton", foreground=plots.TEXT_PRIMARY, background=self._frame_bg, borderwidth=0,
                        relief="flat", padding=(0, 2), focuscolor=self._frame_bg, font=("TkDefaultFont", 11, "bold"))
        style.map("CmpTitle.TButton", foreground=[("active", UI_ACCENT)],
                  background=[("active", self._frame_bg), ("pressed", self._frame_bg)])
        base = tkfont.nametofont("TkDefaultFont")
        self._bold_font = base.copy()
        self._bold_font.configure(weight="bold")

    def _make_swatches(self) -> None:
        """Cuadros de color de cada algoritmo (hueco para los oráculos, que son referencias)."""
        for key in (*ALGORITHMS, *ORACLES):
            image = tk.PhotoImage(master=self, width=12, height=12)
            image.put(ALGO_COLORS.get(key, plots.TEXT_MUTED), to=(0, 0, 12, 12))
            if key in ORACLES:
                image.put(plots.SURFACE, to=(3, 3, 9, 9))
            self._swatches[key] = image

    def _build_left(self) -> None:
        """Columna izquierda (toda la altura): ejecutar, resumen de la configuración, lista y descripción."""
        left = ttk.Frame(self, padding=(0, 0, 12, 0))
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(left, text="▶  Ejecutar experimento completo", style="Accent.TButton",
                                     command=self.run_experiment)
        self.run_button.grid(row=0, column=0, sticky="ew", ipady=4)
        Tooltip(self.run_button, self._run_help)
        self.run_status = ttk.Label(left, text="", style="Muted.TLabel", wraplength=LEFT_WRAP_PX, justify="left")
        self.run_status.grid(row=1, column=0, sticky="ew", pady=(6, 12))

        ttk.Label(left, text="Configuración del experimento", style="Header.TLabel").grid(row=2, column=0, sticky="w")
        summary = ttk.Frame(left, padding=(0, 4, 0, 0))
        summary.grid(row=3, column=0, sticky="ew")
        summary.columnconfigure(1, weight=1)
        self.plan_labels: dict[str, ttk.Label] = {}
        for row, (key, label, help_text) in enumerate(PLAN_FIELDS):
            key_label = ttk.Label(summary, text=label, style="CmpKey.TLabel")
            key_label.grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=1)
            Tooltip(key_label, help_text)
            if key == "algorithms":
                value: ttk.Widget = ttk.Frame(summary)
                self.algorithms_frame = value
            else:
                value = ttk.Label(summary, text="", wraplength=LEFT_WRAP_PX - 112, justify="left")
                self.plan_labels[key] = value  # type: ignore[assignment]
            value.grid(row=row, column=1, sticky="w", pady=1)
            Tooltip(value, help_text)
        self.edit_config_button = ttk.Button(left, text="Editar en Configuración →", style="CmpLink.TButton",
                                             cursor="hand2", command=self.edit_config)
        self.edit_config_button.grid(row=4, column=0, sticky="w", pady=(4, 12))
        Tooltip(self.edit_config_button, "Abre «2 · Configuración», donde se cambian las corridas, la semilla, "
                                         "los algoritmos incluidos y los barridos del experimento.")

        ttk.Label(left, text="Gráficas", style="Header.TLabel").grid(row=5, column=0, sticky="w")
        list_frame = ttk.Frame(left, padding=(0, 4, 0, 12))
        list_frame.grid(row=6, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(
            list_frame, height=LIST_MIN_ROWS, width=1, exportselection=False, activestyle="none",
            relief="flat", borderwidth=0, highlightthickness=1, highlightbackground=plots.AXIS,
            highlightcolor=UI_ACCENT, background=plots.SURFACE, foreground=plots.TEXT_PRIMARY,
            selectbackground=UI_ACCENT, selectforeground="white", selectborderwidth=0, font="TkDefaultFont",
        )
        for i, spec in enumerate(self.plot_specs):
            self.listbox.insert("end", f" {i + 1:>2}. {spec.title}")
        self.listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=_autoscroll(list_scroll))
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        Tooltip(self.listbox, "Elige la gráfica a mostrar (también con las flechas ↑ ↓ o con ◀ ▶). En gris, las "
                              "que aún no tienen datos; abajo se explica qué falta.")

        ttk.Label(left, text="Cómo leer esta gráfica", style="Header.TLabel").grid(row=7, column=0, sticky="w")
        desc_frame = ttk.Frame(left, padding=(0, 4, 0, 0))
        desc_frame.grid(row=8, column=0, sticky="nsew")
        desc_frame.columnconfigure(0, weight=1)
        desc_frame.rowconfigure(0, weight=1)
        self.description = tk.Text(
            desc_frame, width=1, height=3, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            background=self._frame_bg, foreground=plots.TEXT_PRIMARY, font="TkDefaultFont", padx=0, pady=0,
            spacing1=0, spacing2=1, spacing3=4, cursor="arrow", takefocus=0,
        )
        self.description.tag_configure("warn", foreground=WARN_FG, font=self._bold_font)
        self.description.tag_configure("note", foreground=plots.TEXT_SECONDARY)
        self.description.grid(row=0, column=0, sticky="nsew")
        desc_scroll = ttk.Scrollbar(desc_frame, orient="vertical", command=self.description.yview)
        desc_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        self.description.configure(yscrollcommand=_autoscroll(desc_scroll), state="disabled")
        left.rowconfigure(6, weight=1)
        left.rowconfigure(8, weight=1)

    def _build_plot_area(self, right: ttk.Frame) -> None:
        """Arriba a la derecha: navegación, exportación de gráficas y figura embebida."""
        header = ttk.Frame(right)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.prev_button = ttk.Button(header, text="◀", width=3, command=lambda: self.step_plot(-1))
        self.next_button = ttk.Button(header, text="▶", width=3, command=lambda: self.step_plot(1))
        self.export_all_button = ttk.Button(header, text="Exportar todas las gráficas…",
                                            command=self.export_all_plots)
        self.save_plot_button = ttk.Button(header, text="Guardar gráfica actual…", command=self.save_current_plot)
        self.plot_counter = ttk.Label(header, text="", style="CmpCounter.TLabel")
        # Orden de empaquetado = prioridad: si falta ancho, se recorta antes el contador que los botones.
        self.prev_button.pack(side="left")
        self.next_button.pack(side="left", padx=(4, 0))
        self.export_all_button.pack(side="right")
        self.save_plot_button.pack(side="right", padx=(0, 6))
        self.plot_counter.pack(side="left", padx=(10, 0))
        Tooltip(self.prev_button, "Gráfica anterior de la lista.")
        Tooltip(self.next_button, "Gráfica siguiente de la lista.")
        Tooltip(self.export_all_button, self._export_all_help)
        Tooltip(self.save_plot_button, self._save_plot_help)

        self.plot = PlotFrame(right, figsize=(5.0, 2.6))
        self.plot.grid(row=1, column=0, sticky="nsew")
        self.plot.canvas.mpl_connect("button_press_event", self._on_figure_click)
        self.plot.canvas.get_tk_widget().bind("<Configure>", self._on_canvas_configure, add="+")

    def _build_table(self, right: ttk.Frame) -> None:
        """Abajo a la derecha: tabla comparativa final, nota explicativa y exportación CSV."""
        frame = ttk.Frame(right)
        frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        frame.columnconfigure(0, weight=1)
        head = ttk.Frame(frame)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.export_timing_button = ttk.Button(head, command=self.export_timing_csv)
        self.export_curves_button = ttk.Button(head, command=self.export_curves_csv)
        self.export_table_button = ttk.Button(head, command=self.export_table_csv)
        self.export_label = ttk.Label(head, text="", style="CmpKey.TLabel")
        self.table_toggle = ttk.Button(head, text="", style="CmpTitle.TButton", cursor="hand2",
                                       command=self.toggle_table)
        # Orden de empaquetado = prioridad cuando falta ancho (primero los botones).
        self.export_timing_button.pack(side="right")
        self.export_curves_button.pack(side="right", padx=(0, 6))
        self.export_table_button.pack(side="right", padx=(0, 6))
        self.export_label.pack(side="right", padx=(0, 6))
        self.table_toggle.pack(side="left")
        self.table_info = ttk.Label(head, text="ⓘ", style="CmpKey.TLabel", cursor="question_arrow",
                                    font=("TkDefaultFont", 12))
        Tooltip(self.table_info, lambda: self.table_note.cget("text"), wraplength=420)
        Tooltip(self.table_toggle, lambda: ("Clic para plegar la tabla y dejar más espacio a la gráfica (útil al "
                                            "proyectar en clase)." if self.table_visible else
                                            "La tabla está plegada para dejar sitio a la gráfica: clic para verla."))
        Tooltip(self.export_timing_button, "Guarda el tiempo de cómputo por algoritmo (ms por segmento y total) en "
                                           "CSV. Depende de la máquina, por eso va aparte de la tabla.")
        Tooltip(self.export_curves_button, "Guarda las curvas de aprendizaje (recompensa, regret acumulado y "
                                           "fracción de pulls óptimos; media y desviación entre corridas) con una "
                                           "fila por algoritmo y pull, para rehacer las gráficas en otro programa.")
        Tooltip(self.export_table_button, "Guarda la tabla comparativa (medias y desviaciones con 6 decimales; "
                                          "precisiones como fracciones 0–1). Es 100 % reproducible con la misma "
                                          "semilla: no incluye el tiempo de pared.")
        self._set_export_labels(short=False)
        self._full_header_px = (sum(w.winfo_reqwidth() for w in (self.export_timing_button, self.export_curves_button,
                                                                 self.export_table_button)) + 12)
        self._short_header: bool | None = False
        head.bind("<Configure>", self._on_table_head_configure, add="+")
        self.table_visible = True
        self._update_table_toggle()

        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=1, column=0, sticky="ew")
        tree_frame.columnconfigure(0, weight=1)
        self.table = ttk.Treeview(tree_frame, columns=[c.key for c in TABLE_COLUMNS], show="tree headings",
                                  height=len(ALGORITHMS) + len(ORACLES), selectmode="browse",
                                  style="Compare.Treeview")
        self.table.heading("#0", text="Algoritmo", anchor="w")
        self.table.column("#0", width=216, minwidth=200, stretch=False, anchor="w")
        for col in TABLE_COLUMNS:
            self.table.heading(col.key, text=col.heading, anchor="center")
            self.table.column(col.key, width=col.width, minwidth=col.width - 14, stretch=True, anchor="center")
        self.table.tag_configure("oracle", foreground=plots.TEXT_SECONDARY, background=ORACLE_ROW_BG)
        self.table.grid(row=0, column=0, sticky="ew")
        table_scroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.table.xview)
        table_scroll.grid(row=1, column=0, sticky="ew")
        self.table.configure(xscrollcommand=_autoscroll(table_scroll))
        self.table.bind("<<TreeviewSelect>>", self._on_table_select)
        self.table_tooltip = _HeadingTooltip(self.table, self._column_help)
        self.table_message = tk.Label(tree_frame, text="La tabla aparecerá aquí al terminar el experimento.",
                                      background=self._field_bg, foreground=plots.TEXT_SECONDARY,
                                      font=("TkDefaultFont", 10))
        self.table_frame = tree_frame
        self.table_note = ttk.Label(frame, text="", style="Muted.TLabel", justify="left")
        self.table_note.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        frame.bind("<Configure>", lambda e: self.table_note.configure(wraplength=max(200, e.width - 8)), add="+")

    # ================================================================ consultas
    @property
    def plot_keys(self) -> list[str]:
        """Claves de las gráficas de la lista, en orden."""
        return [spec.key for spec in self.plot_specs]

    @property
    def current_index(self) -> int:
        """Posición de la gráfica elegida en la lista (0 si no hay selección)."""
        selection = self.listbox.curselection()
        return int(selection[0]) if selection else 0

    @property
    def current_spec(self) -> plots.PlotSpec:
        """Entrada del catálogo de la gráfica elegida."""
        return self.plot_specs[self.current_index]

    @property
    def current_key(self) -> str:
        """Clave de la gráfica elegida (``"average_reward"``, ``"spectrogram"``…)."""
        return self.current_spec.key

    @property
    def redraw_pending(self) -> bool:
        """True si hay un redibujo esperando a que la pestaña sea visible o a que Tk quede libre."""
        return self._redraw_pending or self._draw_after is not None

    def missing_reason(self, key: str) -> str | None:
        """Por qué la gráfica ``key`` no se puede dibujar ahora (None si se puede).

        Parameters
        ----------
        key : str
            Clave de :attr:`plot_keys`.

        Returns
        -------
        str | None
            Mensaje en español: qué falta y cómo obtenerlo.
        """
        if key == SPECTROGRAM_KEY:
            return None if self.app.state.analysis is not None else "Abre y analiza un audio para ver su espectrograma."
        if self.result is None:
            return "Disponible después de ejecutar el experimento."
        return plots.missing_data_message(key, self.result)

    def is_stale(self) -> bool:
        """True si el experimento mostrado se calculó con otra configuración (entorno, agentes o experimento)."""
        result = self.result
        if result is None:
            return False
        used = getattr(result, "config", None)
        cfg = self.app.state.config
        return used is None or any(getattr(used, s) != getattr(cfg, s) for s in ("env", "agent", "experiment"))

    # ================================================================ selección de gráfica
    def select_plot(self, which: int | str) -> None:
        """Elige la gráfica ``which`` (índice de la lista o clave) y la dibuja.

        Parameters
        ----------
        which : int | str
            Índice en :attr:`plot_keys` o clave (``"cumulative_regret"``...).

        Raises
        ------
        ValueError
            Si la clave no existe.
        """
        index = self.plot_keys.index(which) if isinstance(which, str) else int(which)
        index = max(0, min(index, len(self.plot_specs) - 1))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)
        self._on_plot_changed()

    def step_plot(self, delta: int) -> None:
        """Pasa a la gráfica anterior (``delta = -1``) o siguiente (``+1``) de la lista."""
        target = self.current_index + int(delta)
        if 0 <= target < len(self.plot_specs):
            self.select_plot(target)

    def _on_list_select(self, _event: tk.Event | None = None) -> None:
        """Clic o flechas en la lista de gráficas."""
        if self.listbox.curselection():
            self._on_plot_changed()

    def _on_plot_changed(self) -> None:
        """Actualiza la descripción, la navegación y dibuja la gráfica elegida."""
        self._update_description()
        self._update_header()
        self._schedule_draw()

    # ================================================================ eventos
    def _on_analysis_ready(self, analysis: AnalysisResult | None = None, **_kwargs: Any) -> None:
        """Nuevo análisis: el experimento anterior ya no corresponde; se limpia todo."""
        self.result = None
        self._matches = None
        self.drawn_selection = None
        self._fill_table()
        self._refresh_static()
        self._schedule_draw()

    def _on_experiment_ready(self, result: ExperimentResult | None = None, **_kwargs: Any) -> None:
        """Nuevos resultados: tabla, lista y primera gráfica (o la que ya estaba elegida)."""
        if result is None:
            return
        first = self.result is None
        self.result = result
        self._matches = None
        self._fill_table()
        self._refresh_static()
        if first:
            self.select_plot(0)
        else:
            self._schedule_draw()
        logger.info("Comparación: %d corridas × %d segmentos listas (%d gráficas en la lista)",
                    result.n_runs, result.n_segments, len(self.plot_specs))

    def _on_segment_selected(self, position: int | None = None, source: str = "", **_kwargs: Any) -> None:
        """Resalta el segmento en el espectrograma o en el mapa de aciertos (si se ven).

        Si ya está resaltado (p. ej. el evento lo originó un clic en esta
        pestaña y la vista se actualizó) no se hace nada: así no hay bucles.
        """
        if position == self.drawn_selection or self.plot_state != "plot":
            return
        if self.drawn_key == SPECTROGRAM_KEY:
            self._schedule_draw()
        elif self.drawn_key == "tab_heatmap":
            self._update_heat_overlay()
            self.plot.draw()

    def _on_algorithm_selected(self, algorithm: str = "", **_kwargs: Any) -> None:
        """Resalta la fila del algoritmo que muestran las vistas de un solo algoritmo."""
        self._sync_table_selection()

    def _on_config_changed(self, key: str | None = None, **_kwargs: Any) -> None:
        """Actualiza el resumen, la duración estimada y el aviso de resultados desfasados."""
        self._refresh_static()
        if self.plot_state == "empty":
            self._schedule_draw()

    def _on_busy_changed(self, busy: bool = False, **_kwargs: Any) -> None:
        """Deshabilita las acciones que lanzan tareas mientras hay una en curso."""
        self._busy = bool(busy)
        self._update_run_status()
        self._update_buttons()
        if self.plot_state == "empty":
            self._schedule_draw()

    def _on_map(self, _event: tk.Event) -> None:
        """Al mostrarse la pestaña, dibuja lo que quedó pendiente mientras estaba oculta."""
        if self._redraw_pending:
            self._schedule_draw()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Recalcula (con retardo) las figuras cuyo diseño depende del tamaño en píxeles.

        Son el mensaje guía, el mapa de aciertos (su letra depende del ancho) y
        cualquier figura cuando el lienzo cruza :data:`COMPACT_CANVAS_PX`
        (se quita o se repone el subtítulo).
        """
        w0, h0 = self._drawn_size
        resized = abs(event.width - w0) > RESIZE_TOLERANCE_PX or abs(event.height - h0) > RESIZE_TOLERANCE_PX
        compact_changed = (event.height < COMPACT_CANVAS_PX) != self._drawn_compact
        if not compact_changed and not (resized and (self.plot_state == "empty" or self.drawn_key == "tab_heatmap")):
            return
        if self._replot_after is not None:
            self.after_cancel(self._replot_after)
        self._replot_after = self.after(REPLOT_DELAY_MS, self._replot_after_resize)

    def _replot_after_resize(self) -> None:
        """Redibuja tras redimensionar (si la pestaña sigue visible)."""
        self._replot_after = None
        if self.winfo_ismapped():
            self.draw_current()

    def _on_table_select(self, _event: tk.Event | None = None) -> None:
        """Clic en una fila: un algoritmo pasa a ser el de las vistas de un solo algoritmo.

        Las filas de los oráculos no son algoritmos: se vuelve a resaltar el
        algoritmo vigente. Como :meth:`AppState.select_algorithm` ignora el
        algoritmo ya elegido, resaltar la fila desde el evento no crea bucles.
        """
        selection = self.table.selection()
        if not selection:
            return
        item = selection[0]
        if item in ALGORITHMS:
            self.app.state.select_algorithm(item)
        else:
            self._sync_table_selection()

    def _on_figure_click(self, event: Any) -> None:
        """Clic en el espectrograma (una nota) o en el mapa de aciertos (una columna) → selecciona el segmento."""
        toolbar = self.plot.toolbar
        if toolbar is not None and getattr(toolbar.mode, "value", toolbar.mode):
            return  # zoom o desplazamiento activos: el clic es para la herramienta
        if self.plot_state != "plot" or event.inaxes is None or event.xdata is None or event.button != 1:
            return
        axes = self.plot.figure.axes
        if not axes or event.inaxes is not axes[0]:
            return
        position: int | None = None
        if self.drawn_key == SPECTROGRAM_KEY:
            position = self.position_at_time(float(event.xdata))
        elif self.drawn_key == "tab_heatmap":
            matches = self._gt_matches()
            column = int(math.floor(float(event.xdata)))
            if matches is not None and 0 <= column < len(matches):
                position = matches[column]
        if position is not None:
            self.app.state.select_segment(position, source=SOURCE)

    def position_at_time(self, t: float) -> int | None:
        """Segmento conservado que contiene el instante ``t`` (s), o None.

        Parameters
        ----------
        t : float
            Tiempo en segundos desde el inicio de la pista.

        Returns
        -------
        int | None
            Posición entre los segmentos conservados.
        """
        analysis = self.app.state.analysis
        if analysis is None:
            return None
        for position, segment in enumerate(analysis.kept):
            if segment.start_s <= t < segment.end_s:
                return position
        return None

    # ================================================================ acciones
    def toggle_table(self) -> None:
        """Oculta o muestra la tabla comparativa (la gráfica ocupa el espacio libre).

        La elección del usuario se respeta a partir de entonces (ya no se pliega
        ni se despliega sola al cambiar el tamaño de la ventana).
        """
        self._table_choice = not self.table_visible
        self.table_visible = self._table_choice
        self._layout_table()
        self._update_table_toggle()

    def _layout_table(self) -> None:
        """Muestra u oculta la tabla y su nota (en modo compacto, la nota va en el tooltip «ⓘ»)."""
        if self.table_visible:
            self.table_frame.grid()
        else:
            self.table_frame.grid_remove()
        if self.table_visible and not self.compact:
            self.table_note.grid()
        else:
            self.table_note.grid_remove()
        if self.table_visible and self.compact:
            self.table_info.pack(side="left", padx=(6, 0), after=self.table_toggle)
        else:
            self.table_info.pack_forget()

    def _on_right_configure(self, event: tk.Event) -> None:
        """Modo compacto en ventanas bajas: tabla plegada (si el usuario no eligió) y nota en «ⓘ»."""
        compact = event.height < COMPACT_RIGHT_PX
        if compact != self.compact:
            self.compact = compact
            if self._table_choice is None:
                self.table_visible = not compact
            self._layout_table()
            self._update_table_toggle()

    def _update_table_toggle(self) -> None:
        """Texto del título plegable de la tabla (▾ abierta, ▸ plegada)."""
        self.table_toggle.configure(text=("▾ " if self.table_visible else "▸ ") + "Tabla comparativa final")

    def _set_export_labels(self, short: bool) -> None:
        """Textos de los botones CSV: completos o, en ventanas estrechas, «Exportar: [Tabla CSV…]…»."""
        if short:
            texts = ("Tabla CSV…", "Curvas CSV…", "Tiempos CSV…")
        else:
            texts = ("Exportar tabla CSV…", "Exportar curvas CSV…", "Exportar tiempos CSV…")
        for button, text in zip((self.export_table_button, self.export_curves_button, self.export_timing_button),
                                texts, strict=True):
            button.configure(text=text)
        self.export_label.configure(text="Exportar:" if short else "")

    def _on_table_head_configure(self, event: tk.Event) -> None:
        """Acorta los textos de exportación si no caben junto al título."""
        short = event.width < self.table_toggle.winfo_reqwidth() + self._full_header_px + 24
        if short != self._short_header:
            self._short_header = short
            self._set_export_labels(short)

    def edit_config(self) -> None:
        """Muestra la pestaña «2 · Configuración» (donde se editan los parámetros del experimento)."""
        self.app.show_tab("config")

    def run_experiment(self) -> None:
        """Valida la configuración y lanza el experimento completo (``app.run_experiment``)."""
        if self.app.state.analysis is None or self._busy:
            return
        try:
            self.app.state.config.validate()
        except ValueError as exc:
            show_error(self.winfo_toplevel(), "Configuración inválida", exc)
            return
        self.app.run_experiment()

    def save_current_plot(self, path: str | Path | None = None) -> bool:
        """Guarda la gráfica visible (PNG, PDF o SVG según la extensión), con el zoom actual.

        La figura se vuelve a dibujar en una figura nueva del mismo tamaño en
        un hilo trabajador (``app.run_task``): ``savefig`` tarda ≈ 0.3–0.5 s y
        no debe congelar la ventana. Si el usuario hizo zoom con la barra de
        herramientas, se copian los límites de los ejes.

        Parameters
        ----------
        path : str | Path | None
            Archivo de destino; si es None se pregunta con un diálogo.

        Returns
        -------
        bool
            True si se lanzó la tarea de guardado.
        """
        if self.plot_state != "plot" or self._drawn_call is None or self._busy:
            return False
        if path is None:
            chosen = filedialog.asksaveasfilename(
                parent=self, title="Guardar gráfica actual", initialdir=str(self._initial_dir()),
                initialfile=f"{self.drawn_key}.png", defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            )
            if not chosen:
                return False
            path = chosen
        target = Path(path)
        ext = target.suffix.lower().lstrip(".")
        if not ext:
            target, ext = target.with_suffix(".png"), "png"
        if ext not in self.plot.figure.canvas.get_supported_filetypes():
            show_error(self.winfo_toplevel(), "Formato no soportado",
                       f"No se puede guardar como «.{ext}». Usa .png, .pdf o .svg.")
            return False
        fn, args, kwargs = self._drawn_call
        size = tuple(float(v) for v in self.plot.figure.get_size_inches())
        limits = [(ax.get_xlim(), ax.get_ylim()) for ax in self.plot.figure.axes]

        def task(progress: ProgressCallback, cancel: threading.Event) -> Path:
            progress(0.1, "Dibujando la gráfica…")
            fig = Figure(figsize=size, dpi=100)
            fn(*args, fig=fig, **kwargs)
            if len(fig.axes) == len(limits):  # mismo diseño: se conserva el zoom de la vista
                for ax, (xlim, ylim) in zip(fig.axes, limits, strict=True):
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            progress(0.5, f"Guardando {target.name}…")
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, format=ext, dpi=EXPORT_DPI, facecolor=fig.get_facecolor())
            return target

        def done(written: Path) -> None:
            self._last_dir = written.parent
            self.app.status.set_message(f"Gráfica guardada en {written}")
            logger.info("Gráfica «%s» guardada en %s", self.current_spec.title, written)

        return self.app.run_task("Guardando la gráfica", task, done, error_title="No se pudo guardar la gráfica")

    def export_all_plots(self, folder: str | Path | None = None, formats: tuple[str, ...] = ("png", "pdf")) -> bool:
        """Exporta todas las gráficas comparativas y el espectrograma (PNG y PDF) a una carpeta.

        Llama a :func:`src.plots.save_all_plots` en un hilo trabajador
        (≈ 6 s); las gráficas sin datos (p. ej. sin ground truth) se omiten.

        Parameters
        ----------
        folder : str | Path | None
            Carpeta de destino (se crea si no existe); None = preguntar.
        formats : tuple[str, ...], optional
            Formatos de cada gráfica (PNG y PDF por defecto).

        Returns
        -------
        bool
            True si se lanzó la tarea.
        """
        result = self.result
        if result is None or self._busy:
            return False
        if folder is None:
            folder = filedialog.askdirectory(parent=self, title="Carpeta para exportar todas las gráficas",
                                             initialdir=str(self._initial_dir()), mustexist=False)
            if not folder:
                return False
        out_dir = Path(folder)
        analysis = self.app.state.analysis

        def task(progress: ProgressCallback, cancel: threading.Event) -> list[Path]:
            progress(0.05, f"Exportando las gráficas ({' y '.join(f.upper() for f in formats)})…")
            return plots.save_all_plots(result, analysis, out_dir, formats=formats)

        def done(paths: list[Path]) -> None:
            self._last_dir = out_dir
            n_plots = len({p.stem for p in paths})
            kinds = " y ".join(f.upper() for f in formats)
            self.app.status.set_message(f"Se exportaron {n_plots} gráficas ({len(paths)} archivos {kinds}) a {out_dir}")

        return self.app.run_task("Exportando todas las gráficas", task, done,
                                 error_title="No se pudieron exportar las gráficas")

    def export_table_csv(self, path: str | Path | None = None) -> Path | None:
        """Guarda la tabla comparativa con :meth:`ExperimentResult.to_csv`.

        Parameters
        ----------
        path : str | Path | None
            Archivo CSV; None = preguntar.

        Returns
        -------
        Path | None
            Ruta escrita (None si se canceló o falló).
        """
        return self._export_csv("la tabla comparativa", "summary.csv", "to_csv", path)

    def export_curves_csv(self, path: str | Path | None = None) -> Path | None:
        """Guarda las curvas de aprendizaje con :meth:`ExperimentResult.curves_to_csv`.

        Parameters
        ----------
        path : str | Path | None
            Archivo CSV; None = preguntar.

        Returns
        -------
        Path | None
            Ruta escrita (None si se canceló o falló).
        """
        return self._export_csv("las curvas de aprendizaje", "curves.csv", "curves_to_csv", path)

    def export_timing_csv(self, path: str | Path | None = None) -> Path | None:
        """Guarda los tiempos de cómputo con :meth:`ExperimentResult.timing_to_csv`.

        Parameters
        ----------
        path : str | Path | None
            Archivo CSV; None = preguntar.

        Returns
        -------
        Path | None
            Ruta escrita (None si se canceló o falló).
        """
        return self._export_csv("los tiempos de cómputo", "timing.csv", "timing_to_csv", path)

    def _export_csv(self, what: str, default_name: str, method: str, path: str | Path | None) -> Path | None:
        """Exporta un CSV del resultado con su método ``method`` (es rápido: hilo de la GUI).

        Parameters
        ----------
        what : str
            Descripción para los mensajes («la tabla comparativa»).
        default_name : str
            Nombre propuesto en el diálogo.
        method : str
            Método de :class:`ExperimentResult` que escribe el archivo.
        path : str | Path | None
            Destino; None = preguntar.

        Returns
        -------
        Path | None
            Ruta escrita, o None si no hay resultados, se canceló o falló.
        """
        result = self.result
        writer = getattr(result, method, None) if result is not None else None
        if writer is None:
            return None
        if path is None:
            chosen = filedialog.asksaveasfilename(
                parent=self, title=f"Exportar {what}", initialdir=str(self._initial_dir()),
                initialfile=default_name, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            )
            if not chosen:
                return None
            path = chosen
        try:
            written = Path(writer(str(path)))
        except Exception as exc:  # noqa: BLE001 - p. ej. carpeta sin permisos
            show_error(self.winfo_toplevel(), "No se pudo exportar", exc)
            return None
        self._last_dir = written.parent
        self.app.status.set_message(f"Se guardó {what} en {written}")
        return written

    def _initial_dir(self) -> Path:
        """Carpeta inicial de los diálogos (la última usada, o ``results/``)."""
        for candidate in (self._last_dir, RESULTS_DIR, RESULTS_DIR.parent):
            if candidate.is_dir():
                return candidate
        return Path.cwd()

    # ================================================================ dibujo
    def _schedule_draw(self) -> None:
        """Pide redibujar: en cuanto Tk quede libre si la pestaña se ve; si no, al mostrarse."""
        if not self.winfo_ismapped():
            self._redraw_pending = True
            return
        self._redraw_pending = False
        if self._draw_after is None:
            self._draw_after = self.after_idle(self.draw_current)

    def draw_current(self) -> None:
        """Dibuja YA la gráfica elegida (o el mensaje guía si aún no hay datos)."""
        if self._draw_after is not None:
            self.after_cancel(self._draw_after)
            self._draw_after = None
        self._redraw_pending = False
        spec = self.current_spec
        analysis = self.app.state.analysis
        self._heat_overlay = []
        self.drawn_selection = None
        if spec.key == SPECTROGRAM_KEY:
            if analysis is None:
                self._draw_empty_state(spec.key)
                return
            selected = self._valid_selection()
            self._plot(plots.plot_spectrogram, (analysis,), {"selected": selected})
            self.drawn_selection = selected
        elif self.result is None:
            self._draw_empty_state(spec.key)
            return
        else:
            missing = plots.missing_data_message(spec.key, self.result)
            kwargs: dict[str, Any] = {}
            if spec.key == "tab_heatmap" and missing is None:
                matches = self._gt_matches()
                if matches is not None:
                    kwargs["matches"] = matches
            self._plot(spec.func, (self.result,), kwargs, missing=missing is not None)
            if spec.key == "tab_heatmap" and self.plot_state == "plot":
                self._update_heat_overlay()
        self.drawn_key = spec.key
        self._finish_draw()

    def _plot(self, fn: Callable[..., Figure], args: tuple[Any, ...], kwargs: dict[str, Any],
              missing: bool = False) -> None:
        """Llama a ``fn(*args, fig=figura, **kwargs)``; si falla, muestra el error en la figura.

        Parameters
        ----------
        fn : Callable[..., Figure]
            Función de :mod:`src.plots`.
        args, kwargs : tuple, dict
            Argumentos (el resultado o el análisis, y opciones).
        missing : bool
            True si ``fn`` dibujará un mensaje de «falta un dato».
        """
        fig = self.plot.figure
        try:
            fn(*args, fig=fig, **kwargs)
        except Exception as exc:  # noqa: BLE001 - una gráfica rota no debe romper la pestaña
            logger.exception("No se pudo dibujar la gráfica %s", getattr(fn, "__name__", fn))
            plots.message_figure(f"No se pudo dibujar la gráfica: {exc}", fig=fig)
            self.plot_state = "error"
            self._drawn_call = None
            return
        self.plot_state = "missing" if missing else "plot"
        self._drawn_call = (fn, args, kwargs) if not missing else None
        if self.plot.canvas.get_tk_widget().winfo_height() < COMPACT_CANVAS_PX:
            _drop_subtitle(fig)

    def _finish_draw(self) -> None:
        """Reinicia la barra de herramientas, refresca el lienzo y los controles."""
        widget = self.plot.canvas.get_tk_widget()
        self._drawn_size = (int(widget.winfo_width()), int(widget.winfo_height()))
        self._drawn_compact = self._drawn_size[1] < COMPACT_CANVAS_PX
        if self.plot.toolbar is not None:
            self.plot.toolbar.update()  # la vista «inicio» pasa a ser la de la nueva gráfica
        self.plot.draw()
        self._update_buttons()

    def _valid_selection(self) -> int | None:
        """Segmento seleccionado en la aplicación, si existe en el análisis vigente."""
        position = self.app.state.selected_position
        analysis = self.app.state.analysis
        if position is None or analysis is None or not 0 <= position < len(analysis.kept):
            return None
        return int(position)

    def _gt_matches(self) -> list[int | None] | None:
        """Segmento emparejado con cada nota del ground truth (caché por experimento).

        Usa :func:`src.experiments.match_segments_to_gt` con la tolerancia del
        experimento; None si no hay ground truth o el análisis vigente no es
        el del experimento.
        """
        if self._matches is not None:
            return self._matches
        result, analysis = self.result, self.app.state.analysis
        if result is None or analysis is None or not result.gt_notes:
            return None
        kept = analysis.kept
        if len(kept) != result.n_segments:
            return None
        self._matches = match_segments_to_gt(kept, list(result.gt_notes), result.config.experiment.onset_tolerance_s)
        return self._matches

    def _update_heat_overlay(self) -> None:
        """Enmarca en el mapa de aciertos la(s) columna(s) del segmento seleccionado."""
        for artist in self._heat_overlay:
            with contextlib.suppress(ValueError, NotImplementedError):  # ya no estaba en la figura
                artist.remove()
        self._heat_overlay = []
        position = self._valid_selection()
        self.drawn_selection = position
        matches = self._gt_matches()
        axes = self.plot.figure.axes
        if position is None or matches is None or not axes:
            return
        ax = axes[0]
        n_rows = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
        for column, segment in enumerate(matches):
            if segment == position:
                frame = Rectangle((column, 0), 1, n_rows, fill=False, edgecolor=plots.TEXT_PRIMARY, linewidth=2.4,
                                  zorder=6, clip_on=False)
                ax.add_patch(frame)
                self._heat_overlay.append(frame)

    def _draw_empty_state(self, key: str) -> None:
        """Mensaje guía en la figura (qué hacer y qué se verá), dimensionado en píxeles.

        Las posiciones se calculan con el tamaño real del lienzo para que nada
        se solape en ventanas pequeñas: se reduce la letra, las tarjetas pasan
        a 2 × 2 y, si no caben, se omiten.

        Parameters
        ----------
        key : str
            Gráfica elegida (el espectrograma tiene su propio mensaje).
        """
        self.plot_state = "empty"
        self.drawn_key = key
        self._drawn_call = None
        fig = self.plot.figure
        fig.clear()
        fig.set_layout_engine("none")
        fig.set_facecolor(plots.SURFACE)
        width_px, height_px = (float(v) for v in fig.get_size_inches() * fig.dpi)
        scale = 1.0 if width_px >= 760 and height_px >= 430 else 0.86
        px_per_pt = fig.dpi / 72.0
        title, body = self._empty_texts(key)
        title_size, body_size, card_size, foot_size = 17 * scale, 12 * scale, 10 * scale, 9.5 * scale

        def wrap(text: str, size: float, width_fraction: float = 0.86) -> str:
            """Parte ``text`` en líneas que quepan en ``width_fraction`` del ancho."""
            chars = max(20, int(width_fraction * width_px / (0.55 * size * px_per_pt)))
            return "\n".join(textwrap.fill(line, width=chars) for line in text.splitlines())

        def height_of(text: str, size: float, spacing: float) -> float:
            """Alto aproximado (px) de un texto de varias líneas."""
            return (text.count("\n") + 1) * size * spacing * px_per_pt

        title = wrap(title, title_size)
        body = wrap(body, body_size)
        footer = wrap("Los algoritmos se comparan con números aleatorios comunes: en cada corrida todos ven la misma "
                      "secuencia de frames, así que las diferencias se deben al algoritmo y no al azar.",
                      foot_size, 0.8)
        # Tarjetas de igual tamaño: el ancho sale del texto más largo; 4 en fila, 2 × 2 o ninguna.
        pad_px = 12 * scale
        char_px = 0.5 * card_size * px_per_pt
        longest = max(len(line) for card in EMPTY_CARDS for part in card for line in part.splitlines())
        card_w = longest * char_px + 2 * pad_px
        detail_lines = max(card[1].count("\n") + 1 for card in EMPTY_CARDS)
        card_h = (1 + detail_lines) * card_size * 1.45 * px_per_pt + 2 * pad_px
        card_gap = 14 * scale
        n_cols = next((n for n in (4, 2) if n * card_w + (n - 1) * card_gap <= 0.94 * width_px), 0)
        gap = 20 * scale
        blocks: list[tuple[str, float]] = [("title", height_of(title, title_size, 1.3)),
                                           ("body", height_of(body, body_size, 1.55))]
        total = sum(h for _k, h in blocks) + gap
        if key != SPECTROGRAM_KEY and n_cols:
            n_card_rows = math.ceil(len(EMPTY_CARDS) / n_cols)
            cards_h = n_card_rows * card_h + (n_card_rows - 1) * card_gap
            if total + cards_h + gap + 20 <= height_px:
                blocks.append(("cards", cards_h))
                total += cards_h + gap
                foot_h = height_of(footer, foot_size, 1.5)
                if total + foot_h + 20 <= height_px:
                    blocks.append(("footer", foot_h))
                    total += foot_h + gap
        top = max((height_px - (total - gap)) / 2.0, 6.0)

        def y_of(center_px: float) -> float:
            """Píxeles desde arriba → fracción de la figura."""
            return 1.0 - center_px / height_px

        for kind, block_h in blocks:
            center = y_of(top + block_h / 2.0)
            if kind == "title":
                fig.text(0.5, center, title, ha="center", va="center", fontsize=title_size, fontweight="bold",
                         color=plots.TEXT_PRIMARY, linespacing=1.3, multialignment="center")
            elif kind == "body":
                fig.text(0.5, center, body, ha="center", va="center", fontsize=body_size,
                         color=plots.TEXT_SECONDARY, linespacing=1.55, multialignment="center")
            elif kind == "cards":
                row_w = n_cols * card_w + (n_cols - 1) * card_gap
                left_px = (width_px - row_w) / 2.0
                for i, (card_title, detail) in enumerate(EMPTY_CARDS):
                    r, c = divmod(i, n_cols)
                    x0 = left_px + c * (card_w + card_gap)
                    y_top = top + r * (card_h + card_gap)
                    # Caja redondeada en fracciones de figura; mutation_aspect corrige la proporción W/H
                    # para que las esquinas sean circulares (radio ≈ 8 px).
                    fig.add_artist(FancyBboxPatch(
                        (x0 / width_px, 1.0 - (y_top + card_h) / height_px), card_w / width_px, card_h / height_px,
                        boxstyle=f"round,pad=0,rounding_size={8.0 / width_px:.5f}",
                        mutation_aspect=width_px / height_px,
                        transform=fig.transFigure, facecolor=plots.HIGHLIGHT, edgecolor=plots.AXIS, linewidth=0.9))
                    # Una línea por texto (título en negrita + detalle): interlineado uniforme.
                    cx = (x0 + card_w / 2.0) / width_px
                    line_px = card_size * 1.45 * px_per_pt
                    for j, line in enumerate([card_title, *detail.splitlines()]):
                        fig.text(cx, y_of(y_top + pad_px + line_px * (j + 0.5)), line, ha="center", va="center",
                                 fontsize=card_size, fontweight="bold" if j == 0 else "normal",
                                 color=plots.TEXT_PRIMARY if j == 0 else plots.TEXT_SECONDARY)
            else:
                fig.text(0.5, center, footer, ha="center", va="center", fontsize=foot_size,
                         color=plots.TEXT_MUTED, linespacing=1.5, multialignment="center")
            top += block_h + gap
        self._finish_draw()

    def _empty_texts(self, key: str) -> tuple[str, str]:
        """Título y cuerpo del mensaje guía según el estado (sin audio, sin experimento, ocupado)."""
        analysis = self.app.state.analysis
        if self._busy:
            return ("Hay una tarea en segundo plano…",
                    "Sigue el avance en la barra de estado (abajo). Cuando termine el experimento, las gráficas "
                    "comparativas aparecerán aquí.")
        if key == SPECTROGRAM_KEY:
            return ("Todavía no hay ningún audio analizado",
                    "Abre un MP3 (Archivo → Abrir) o genera el dataset sintético para ver su espectrograma.")
        if analysis is None:
            return ("Todavía no hay resultados del experimento",
                    "1 · Abre un MP3 (Archivo → Abrir) o genera el dataset sintético.\n"
                    f"2 · Pulsa «▶ Ejecutar experimento completo» ({DEFAULT_DURATION_TEXT}).")
        cfg = self.app.state.config
        name = Path(analysis.path).name
        n_notes = len(analysis.segment_data)
        estimate = self.plan_values.get("estimate") or estimate_text(cfg, analysis)
        duration = estimate.split(" (")[0]
        return ("Ejecuta el experimento para ver las gráficas comparativas",
                f"Pulsa «▶ Ejecutar experimento completo» (a la izquierda): {cfg.experiment.n_runs} corridas de cada "
                f"algoritmo sobre las {n_notes} notas de «{name}».\n"
                f"Con la configuración actual tardará {duration}.")

    # ================================================================ vistas auxiliares
    def _refresh_static(self) -> None:
        """Actualiza todo lo que no es la figura: resumen, estado, lista, descripción y botones."""
        self._update_plan()
        self._update_run_status()
        self._refresh_list_colors()
        self._update_description()
        self._update_header()
        self._update_buttons()

    def _update_plan(self) -> None:
        """Resumen de la configuración del experimento (y cuadros de color de los algoritmos)."""
        cfg = self.app.state.config
        self.plan_values = experiment_plan(cfg, self.app.state.analysis)
        for key, label in self.plan_labels.items():
            label.configure(text=self.plan_values.get(key, ""))
        for child in self.algorithms_frame.winfo_children():
            child.destroy()
        selected = [a for a in ALGORITHMS if a in cfg.experiment.algorithms]
        if not selected:
            ttk.Label(self.algorithms_frame, text=self.plan_values["algorithms"], style="CmpWarn.TLabel",
                      wraplength=LEFT_WRAP_PX - 100).grid(row=0, column=0, sticky="w")
        for i, algo in enumerate(selected):
            ttk.Label(self.algorithms_frame, text=ALGO_SHORT.get(algo, algo), image=self._swatches[algo],
                      compound="left", padding=(0, 0, 10, 0)).grid(row=i // 2, column=i % 2, sticky="w")

    def _update_run_status(self) -> None:
        """Texto bajo el botón principal: qué falta, si hay tarea en curso o si los resultados están vigentes."""
        analysis, result = self.app.state.analysis, self.result
        style = "Muted.TLabel"
        if analysis is None:
            text = "Primero abre un audio (Archivo → Abrir o pestaña «1 · Audio»)."
        elif self._busy:
            text = "Hay una tarea en curso: espera a que termine o cancélala (barra de estado)."
        elif result is None:
            text = f"Listo para ejecutar sobre las {len(analysis.segment_data)} notas de «{Path(analysis.path).name}»."
        elif self.is_stale():
            text = "⚠ Resultados calculados con otra configuración: vuelve a ejecutar el experimento."
            style = "CmpWarn.TLabel"
        else:
            text = (f"✓ Resultados vigentes: {result.n_runs} corridas × {result.n_segments} segmentos, "
                    f"T = {result.budget}.")
        self.run_status.configure(text=text, style=style)

    def _refresh_list_colors(self) -> None:
        """Gris para las gráficas que aún no tienen datos."""
        for i, spec in enumerate(self.plot_specs):
            available = self.missing_reason(spec.key) is None
            self.listbox.itemconfigure(i, foreground=plots.TEXT_PRIMARY if available else plots.TEXT_MUTED)

    def _update_description(self) -> None:
        """«Cómo leer esta gráfica»: qué falta (si falta algo) y la descripción del catálogo."""
        spec = self.current_spec
        text = self.description
        text.configure(state="normal")
        text.delete("1.0", "end")
        reason = self.missing_reason(spec.key)
        if reason:
            text.insert("end", f"⚠ {reason}\n", "warn")
        text.insert("end", spec.description)
        result = self.result
        if spec.key in ("arm_distribution", "q_evolution") and result is not None and result.example is not None:
            example = result.example
            text.insert("end", f"\nSegmento de ejemplo: n.º {example.position} ({len(example.arm_labels)} brazos "
                               "candidatos).", "note")
        elif spec.key == "tab_heatmap" and reason is None:
            text.insert("end", "\nClic en una columna = seleccionar esa nota en las demás pestañas.", "note")
        text.configure(state="disabled")
        text.yview_moveto(0.0)

    def _update_header(self) -> None:
        """Contador «Gráfica i de n» y botones ◀ ▶."""
        index, n = self.current_index, len(self.plot_specs)
        self.listbox.see(index)
        self.plot_counter.configure(text=f"Gráfica {index + 1} de {n}")
        self.prev_button.configure(state="normal" if index > 0 else "disabled")
        self.next_button.configure(state="normal" if index < n - 1 else "disabled")

    def _update_buttons(self) -> None:
        """Habilita o deshabilita las acciones según haya análisis, resultados o una tarea en curso."""
        has_result = self.result is not None

        def enable(widget: ttk.Button, on: bool) -> None:
            widget.configure(state="normal" if on else "disabled")

        enable(self.run_button, self.app.state.analysis is not None and not self._busy)
        enable(self.save_plot_button, self.plot_state == "plot" and not self._busy)
        enable(self.export_all_button, has_result and not self._busy)
        enable(self.export_table_button, has_result)
        enable(self.export_curves_button, has_result)
        enable(self.export_timing_button, has_result and hasattr(self.result, "timing_to_csv"))

    def _fill_table(self) -> None:
        """Rellena la tabla comparativa desde :meth:`ExperimentResult.summary_rows`."""
        table = self.table
        table.delete(*table.get_children())
        self.table_rows = []
        result = self.result
        if result is None:
            table.configure(height=len(ALGORITHMS), displaycolumns=[c.key for c in TABLE_COLUMNS if not c.extra])
            self.table_message.place(relx=0.5, rely=0.58, anchor="center")
            self.table_note.configure(text="Pasa el ratón por los encabezados para ver qué mide cada columna.")
            return
        self.table_message.place_forget()
        rows = result.summary_rows()
        has_gt = any(r.get("pitch_acc") is not None for r in rows)
        has_extra = has_gt and any((r.get("n_extra") or 0) > 0 for r in rows)
        shown = [c for c in TABLE_COLUMNS if (has_gt or not c.needs_gt) and (has_extra or not c.extra)]
        table.configure(displaycolumns=[c.key for c in shown], height=max(len(ALGORITHMS), len(rows)))
        best = {c.key: best_algorithms(rows, c) for c in TABLE_COLUMNS}
        for row in rows:
            algo = str(row["algorithm"])
            oracle = algo in ORACLES
            cells: dict[str, str] = {"algorithm": algo, "label": str(row["label"])}
            values = []
            for col in TABLE_COLUMNS:
                text = format_cell(row.get(col.key), row.get(f"{col.key}_std"), col.kind, with_std=not oracle)
                if algo in best[col.key]:
                    text += BEST_MARK
                cells[col.key] = text
                values.append(text)
            table.insert("", "end", iid=algo, text=f" {row['label']}", image=self._swatches.get(algo, ""),
                         values=values, tags=("oracle",) if oracle else ("algorithm",))
            self.table_rows.append(cells)
        note = (f"Media ± desviación estándar entre {result.n_runs} corridas (T = {result.budget} pulls, "
                f"{result.n_segments} segmentos) · ★ mejor algoritmo · en gris, oráculos que conocen μ · pasa el "
                "ratón por los encabezados para ver qué mide cada columna.")
        if not has_gt:
            note += " Sin ground truth no hay columnas de precisión."
        elif has_extra:
            n_extra = next((r.get("n_extra") for r in rows if r.get("n_extra") is not None), 0)
            note += f" {n_extra} segmentos sin nota real: cuentan en el F1, no en la precisión."
        else:
            note += " F1 omitido: ningún segmento sobra, coincide con la precisión."
        self.table_note.configure(text=note)
        self._sync_table_selection()

    def _sync_table_selection(self) -> None:
        """Resalta en la tabla el algoritmo elegido en la aplicación (si tiene fila)."""
        algorithm = self.app.state.selected_algorithm
        if self.table.exists(algorithm):
            if tuple(self.table.selection()) != (algorithm,):
                self.table.selection_set(algorithm)
        elif self.table.selection():
            self.table.selection_remove(*self.table.selection())

    def _column_help(self, column: str) -> str:
        """Explicación de la columna ``column`` de la tabla (tooltip del encabezado)."""
        if column == "#0":
            return ALGORITHM_COLUMN_HELP
        for col in TABLE_COLUMNS:
            if col.key == column:
                return col.help
        return ""

    def _run_help(self) -> str:
        """Tooltip del botón principal (o el motivo por el que está deshabilitado)."""
        if self.app.state.analysis is None:
            return "Primero abre y analiza un audio: el experimento se ejecuta sobre sus notas."
        if self._busy:
            return "Hay una tarea en curso; espera a que termine (o cancélala en la barra de estado)."
        return ("Ejecuta el experimento comparativo con la configuración resumida debajo: las corridas de cada "
                "algoritmo, los oráculos, el segmento de ejemplo y los barridos. Puedes cancelarlo desde la barra "
                "de estado. Duración estimada: " + self.plan_values.get("estimate", "—") + ".")

    def _save_plot_help(self) -> str:
        """Tooltip de «Guardar gráfica actual…»."""
        if self.plot_state != "plot":
            return "No hay ninguna gráfica con datos a la vista."
        return ("Guarda la gráfica tal como se ve (con el zoom actual) a 150 ppp. El formato depende de la "
                "extensión: .png, .pdf o .svg.")

    def _export_all_help(self) -> str:
        """Tooltip de «Exportar todas las gráficas…»."""
        if self.result is None:
            return "Disponible después de ejecutar el experimento."
        return ("Guarda en una carpeta todas las gráficas comparativas y el espectrograma, cada una en PNG y PDF "
                "(se omiten las que no tienen datos, p. ej. la precisión sin ground truth).")

    def destroy(self) -> None:
        """Cancela los redibujos programados antes de destruir la pestaña."""
        for after_id in (self._draw_after, self._replot_after):
            if after_id is not None:
                with contextlib.suppress(tk.TclError):
                    self.after_cancel(after_id)
        self._draw_after = self._replot_after = None
        super().destroy()
