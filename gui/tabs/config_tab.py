"""Pestaña 2 · Configuración: todos los hiperparámetros del sistema en un solo lugar.

Papel en la GUI
---------------
Esta pestaña es el "panel de control" de Smartuner. Muestra, agrupados por
etapa del pipeline, TODOS los parámetros editables de
:data:`src.config.PARAM_SPECS` (etiqueta, unidad, rango y tooltip salen de ahí,
así que la GUI, la CLI y la documentación nunca divergen):

* **Preprocesamiento, segmentación y pitch** (etapas 3–5): cambiarlos obliga a
  volver a analizar el audio (decodificar, filtrar, segmentar, pYIN).
* **Entorno bandit** (etapa 6): poda de brazos ±k, template armónico (N, β),
  penalización de tocabilidad λ, presupuesto T, tipo de espectro.
* **Algoritmos** (etapa 7): una tarjeta por agente con su color, su regla de
  decisión y sus hiperparámetros, y la casilla «Incluir en el experimento».
* **Experimento**: corridas, semilla, barridos de sensibilidad y de λ, con la
  duración estimada del experimento.

A la derecha, un panel opcional de **Fórmulas** (recompensa y reglas de los
agentes, renderizadas con *mathtext* de matplotlib) pensado para proyectarse
en clase; debajo de cada fórmula se muestran los valores vigentes.

Flujo de datos
--------------
La pestaña NO contiene lógica del sistema: cada cambio válido se escribe en
``app.state.config`` y se anuncia con el evento ``config_changed``. Comparando
la configuración vigente con la que usó el último análisis
(``app.state.analysis.config``) la pestaña sabe qué hace falta para aplicar
los cambios y lo indica con un aviso:

=====================================  ===============================================
Cambió...                              Aviso y acción
=====================================  ===============================================
audio / preprocess / segmentation /    «Requiere re-analizar el audio»
pitch (etapas 1–5)                     → [Analizar de nuevo] (``app.analyze``)
env / agent (o la fracción con voz     «Requiere re-transcribir»
mínima, que se re-anota barato)        → [Aplicar y re-transcribir] (``app.retranscribe``)
experiment                             Se aplica al ejecutar el experimento
=====================================  ===============================================

Como el aviso se calcula por diferencia (y no por "se tocó algo"), desaparece
solo si el usuario devuelve un parámetro a su valor original, y se actualiza
al recibir ``analysis_ready`` / ``transcriptions_ready``.

Eventos
-------
Escucha ``config_changed`` (refresca los campos si el cambio vino de fuera:
cargar JSON, restaurar valores por defecto u otra pestaña), ``analysis_ready``,
``transcriptions_ready``, ``experiment_ready``, ``busy_changed`` (deshabilita
las acciones mientras hay una tarea) y ``algorithm_selected`` (marca la
tarjeta del algoritmo que muestran las vistas de un solo algoritmo). No usa
``segment_selected``: la configuración no depende del segmento elegido.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
import math
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser

from gui.widgets import UI_ACCENT, UI_ERROR, UI_SURFACE, UI_TEXT, UI_TEXT_SECONDARY, ParamField, ScrollableFrame, Tooltip, show_error
from src.config import ALGO_COLORS, ALGO_LABELS, ALGORITHMS, PARAM_SPECS, Config
from src.experiments import SWEEP_SPECS, estimate_experiment_seconds
from src.plots import GRID, TEXT_MUTED

logger = logging.getLogger(__name__)

#: Nombre de esta pestaña como origen de eventos (``source=``).
TAB_SOURCE = "config"

# ---------------------------------------------------------------------------
# Qué exige cada cambio
# ---------------------------------------------------------------------------

#: Secciones de :class:`src.config.Config` que se aplican en las etapas 1–5
#: (decodificación, separación, filtro, onsets, pYIN): cambiarlas exige
#: volver a analizar el audio.
REANALYZE_SECTIONS: tuple[str, ...] = ("audio", "preprocess", "segmentation", "pitch")

#: Secciones que :func:`src.pipeline.rebuild_segment_data` aplica sin volver a
#: decodificar ni ejecutar pYIN: basta con re-transcribir.
RETRANSCRIBE_SECTIONS: tuple[str, ...] = ("env", "agent")

#: Parámetros de las etapas 1–5 que, sin embargo, se aplican al re-transcribir
#: (la f0 de cada segmento se re-anota a partir de la trayectoria de pYIN ya
#: guardada; ver :func:`src.pipeline.rebuild_segment_data`).
RETRANSCRIBE_EXCEPTIONS: tuple[str, ...] = ("pitch.min_voiced_ratio",)


def config_differences(reference: Config, current: Config) -> tuple[list[str], list[str]]:
    """Clasifica los parámetros que difieren entre dos configuraciones.

    Parameters
    ----------
    reference : Config
        Configuración con que se calculó el análisis vigente
        (``app.state.analysis.config``).
    current : Config
        Configuración que el usuario está editando (``app.state.config``).

    Returns
    -------
    reanalyze : list[str]
        Claves ``"seccion.campo"`` de las etapas 1–5 que cambiaron y exigen
        volver a analizar el audio.
    retranscribe : list[str]
        Claves de ``env``/``agent`` (y :data:`RETRANSCRIBE_EXCEPTIONS`) que
        cambiaron y se aplican re-transcribiendo.

    Notes
    -----
    Se comparan TODOS los campos de las dataclasses (no solo los de
    :data:`PARAM_SPECS`), porque un JSON cargado puede cambiar campos sin
    control en la GUI (p. ej. ``segmentation.hop_length``). La sección
    ``experiment`` no aparece: solo afecta a :func:`src.experiments.run_experiment`.

    Examples
    --------
    >>> ref, cur = Config(), Config()
    >>> cur.env.lam = 0.3; cur.preprocess.lowpass_hz = 300.0; cur.pitch.min_voiced_ratio = 0.5
    >>> config_differences(ref, cur)
    (['preprocess.lowpass_hz'], ['pitch.min_voiced_ratio', 'env.lam'])
    """
    reanalyze: list[str] = []
    retranscribe: list[str] = []
    for section in REANALYZE_SECTIONS + RETRANSCRIBE_SECTIONS:
        old_sec, new_sec = getattr(reference, section), getattr(current, section)
        for f in dataclasses.fields(old_sec):
            key = f"{section}.{f.name}"
            if getattr(old_sec, f.name) == getattr(new_sec, f.name):
                continue
            if section in RETRANSCRIBE_SECTIONS or key in RETRANSCRIBE_EXCEPTIONS:
                retranscribe.append(key)
            else:
                reanalyze.append(key)
    return reanalyze, retranscribe


def format_value(key: str, value: Any) -> str:
    """Texto corto de un valor de configuración para los avisos.

    Parameters
    ----------
    key : str
        Clave ``"seccion.campo"`` (para buscar la unidad en :data:`PARAM_SPECS`).
    value : Any
        Valor a mostrar.

    Returns
    -------
    str
        ``"0.1"``, ``"400 Hz"``, ``"sí"``, ``"vacío"``, ``"0.1, 0.2"``...

    Examples
    --------
    >>> format_value("preprocess.lowpass_hz", 400.0), format_value("env.k_semitones", None)
    ('400 Hz', 'vacío')
    """
    if value is None:
        return "vacío"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
    text = f"{value:g}" if isinstance(value, float) else str(value)
    spec = PARAM_SPECS.get(key)
    return f"{text} {spec.unit}" if spec is not None and spec.unit else text


def describe_changes(keys: Sequence[str], reference: Config, current: Config, limit: int = 4) -> str:
    """Lista legible de cambios: ``"λ tocabilidad (0.1 → 0.3), ε (0.1 → 0.2)"``.

    Parameters
    ----------
    keys : Sequence[str]
        Claves que cambiaron (de :func:`config_differences`).
    reference, current : Config
        Valores de antes y de ahora.
    limit : int, optional
        Máximo de cambios detallados; el resto se resume como "y N más".

    Returns
    -------
    str
        Texto en español para el aviso.

    Examples
    --------
    >>> cur = Config(); cur.env.lam = 0.3
    >>> describe_changes(["env.lam"], Config(), cur)
    'λ tocabilidad (0.1 → 0.3)'
    """
    parts = []
    for key in keys[:limit]:
        spec = PARAM_SPECS.get(key)
        label = spec.label if spec is not None else key
        parts.append(f"{label} ({format_value(key, reference.get(key))} → {format_value(key, current.get(key))})")
    if len(keys) > limit:
        parts.append(f"y {len(keys) - limit} más")
    return ", ".join(parts)


def validation_problems(cfg: Config) -> list[str]:
    """Problemas que encuentra :meth:`src.config.Config.validate` (lista vacía si es válida).

    Parameters
    ----------
    cfg : Config
        Configuración a comprobar.

    Returns
    -------
    list[str]
        Un mensaje en español por problema.

    Examples
    --------
    >>> validation_problems(Config())
    []
    """
    try:
        cfg.validate()
    except ValueError as exc:
        lines = [line.strip().lstrip("•").strip() for line in str(exc).splitlines()[1:]]
        return [line for line in lines if line] or [str(exc)]
    return []


def format_seconds(seconds: float) -> str:
    """Duración legible: ``"42 s"`` o ``"3.5 min"``.

    Examples
    --------
    >>> format_seconds(42.3), format_seconds(210.0)
    ('42 s', '3.5 min')
    """
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60.0:.1f} min"


# ---------------------------------------------------------------------------
# Disposición de los parámetros
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Group:
    """Subgrupo de parámetros dentro de una sección (título, claves y nota opcional)."""

    title: str
    keys: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class _Section:
    """Sección (``LabelFrame``) de la pestaña: prefijos de clave que agrupa y sus subgrupos."""

    key: str
    title: str
    prefixes: tuple[str, ...]
    note: str
    groups: tuple[_Group, ...]


#: Secciones de parámetros "planos" (los algoritmos tienen su propia sección).
#: Las claves de PARAM_SPECS que no aparezcan aquí se añaden automáticamente en
#: un subgrupo «Otros parámetros» de la sección de su prefijo.
SECTIONS: tuple[_Section, ...] = (
    _Section(
        "analysis", "Preprocesamiento, segmentación y pitch (requieren re-analizar)",
        ("audio", "preprocess", "segmentation", "pitch"),
        "Etapas 3–5: se aplican volviendo a analizar el audio (filtrar, detectar onsets y estimar la f0 con "
        "pYIN). Excepción: «Fracción con voz mínima» solo requiere re-transcribir.",
        (
            _Group("Preprocesamiento: pasa-bajas (solo para onsets y pYIN)",
                   ("preprocess.lowpass_hz", "preprocess.filter_order")),
            _Group("Segmentación por onsets",
                   ("segmentation.onset_delta", "segmentation.onset_wait_s",
                    "segmentation.rms_threshold_db", "segmentation.min_duration_s")),
            _Group("Pitch con pYIN (f0 de cada nota)",
                   ("pitch.max_transition_rate", "pitch.switch_prob", "pitch.min_voiced_ratio")),
        ),
    ),
    _Section(
        "env", "Entorno bandit (recompensa, poda, presupuesto)", ("env",),
        "Etapa 6: cada nota es un problema bandit cuyos brazos son posiciones (cuerda, traste). Se aplican "
        "re-transcribiendo, sin volver a decodificar ni ejecutar pYIN.",
        (
            _Group("Brazos candidatos y presupuesto", ("env.k_semitones", "env.budget")),
            _Group("Recompensa: saliencia armónica S = max(S⁺ − β·S⁻, 0)",
                   ("env.n_harmonics", "env.beta", "env.tolerance_semitones", "env.attack_skip_s", "env.noise_std")),
            _Group("Tocabilidad: r = S − λ·|Δtraste|/12",
                   ("env.lam", "env.initial_hand_fret", "env.open_string_free")),
            _Group("Espectro usado por la recompensa", ("env.spectrum", "env.n_fft", "env.bins_per_octave"),
                   "«Tamaño FFT» solo se usa con STFT y «Bins por octava» solo con CQT: el que no aplica "
                   "queda desactivado."),
        ),
    ),
    _Section(
        "experiment", "Experimento", ("experiment",),
        "Se aplican al ejecutar el experimento comparativo (Ejecutar → Ejecutar experimento completo); no "
        "cambian la tablatura actual.",
        (
            _Group("Corridas y reproducibilidad",
                   ("experiment.n_runs", "experiment.seed", "experiment.onset_tolerance_s")),
            _Group("Barridos de sensibilidad (ε, Q₀, c, τ)",
                   ("experiment.run_sweeps", "experiment.sweep_runs", "experiment.sweep_epsilon",
                    "experiment.sweep_q0", "experiment.sweep_c", "experiment.sweep_tau"),
                   "Cada lista se barre solo si su algoritmo está incluido en el experimento."),
            _Group("Barrido de λ (tocabilidad; requiere ground truth)",
                   ("experiment.run_lambda_sweep", "experiment.sweep_lambda")),
        ),
    ),
)

#: Hiperparámetros de cada algoritmo (tarjetas de la sección «Algoritmos»).
ALGO_PARAMS: dict[str, tuple[str, ...]] = {
    "egreedy": ("agent.epsilon", "agent.epsilon_decay", "agent.epsilon_min"),
    "optimistic": ("agent.q0", "agent.optimistic_epsilon", "agent.optimistic_alpha"),
    "ucb1": ("agent.ucb_c",),
    "softmax": ("agent.tau", "agent.tau_decay", "agent.tau_min"),
}

#: Regla de decisión de cada algoritmo (texto plano) y cómo explora.
ALGO_RULES: dict[str, tuple[str, str]] = {
    "egreedy": ("a = argmax Q(a) con prob. 1−ε;  aleatorio con prob. ε",
                "Explora al azar y «a ciegas»: un brazo pésimo se prueba tanto como uno casi óptimo."),
    "optimistic": ("Q₀ alto ⇒ exploración inicial;  Q ← Q + α(r − Q)",
                   "Cada brazo «decepciona» al probarlo (r < Q₀), así que al principio los recorre todos."),
    "ucb1": ("a = argmax [ Q(a) + c·√(ln t / n(a)) ]",
             "Optimismo ante la incertidumbre: bono grande para los brazos poco probados."),
    "softmax": ("π(a) = e^(Q(a)/τ) / Σ_b e^(Q(b)/τ)",
                "Exploración graduada por el valor: los brazos buenos se prueban más que los malos."),
}

#: Paleta de los avisos: (fondo, borde/acento, texto).
_BANNER_COLORS: dict[str, tuple[str, str, str]] = {
    "info": ("#e9f1fb", UI_ACCENT, "#16406f"),
    "warn": ("#fdf3dc", "#e09b00", "#5a3d00"),
    "ok": ("#e7f4ec", "#1baf7a", "#174d33"),
    "error": ("#fbeaea", UI_ERROR, "#7a1b1b"),
}

#: Color del texto de un control desactivado (el gris de las marcas de los ejes de src.plots).
_DISABLED_TEXT = TEXT_MUTED

#: Color de la etiqueta de un parámetro cambiado que aún no se aplicó al análisis.
_PENDING_TEXT = "#b25d00"

#: Color de la línea «Ahora: ...» bajo cada fórmula.
_VALUE_TEXT = "#1c5cab"


def _fmt(value: float) -> str:
    """Número corto (``0.1``, ``1.414``, ``500``)."""
    return f"{value:g}"


def _decay_text(symbol: str, value0: float, decay: float, minimum: float) -> str:
    """Texto de un parámetro con decaimiento: ``"ε = 0.1 (constante)"`` o con d y mínimo."""
    if decay >= 1.0:
        return f"{symbol} = {_fmt(value0)} (constante, d = 1)"
    return f"{symbol}₀ = {_fmt(value0)}, d = {_fmt(decay)}, {symbol}_min = {_fmt(minimum)}"


@dataclass(frozen=True)
class _FormulaBlock:
    """Bloque del panel de fórmulas: título, ecuaciones (mathtext), explicación y valores vigentes."""

    title: str
    algorithm: str | None
    formulas: tuple[str, ...]
    explanation: str
    values: Callable[[Config], str] | None = None


#: Contenido del panel «Fórmulas» (de arriba abajo).
FORMULA_BLOCKS: tuple[_FormulaBlock, ...] = (
    _FormulaBlock(
        "Brazos: posiciones del diapasón", None,
        (r"$f(\mathrm{cuerda},\,\mathrm{traste}) = f_{\mathrm{cuerda}}\cdot 2^{\,\mathrm{traste}/12}$",),
        "Cada traste sube un semitono. Solo hay brazos a ±k semitonos de la f0 que estimó pYIN.",
        lambda c: "Ahora: " + ("k vacío → 52 brazos" if c.env.k_semitones is None
                               else f"k = ±{c.env.k_semitones} semitonos"),
    ),
    _FormulaBlock(
        "Recompensa de un pull", None,
        (r"$S^{+}(f) = \sum_{h=1}^{N} \bar{w}_h\,\tilde{X}(h\,f)$",
         r"$S^{-}(f) = \sum_{h=1}^{N} \bar{w}_h\,\tilde{X}\left((h-\frac{1}{2})\,f\right)$",
         r"$\bar{w}_h = \frac{1}{h}\ /\ \sum_{k=1}^{N} \frac{1}{k}$",
         r"$S(f) = \max\left(S^{+}(f) - \beta\,S^{-}(f),\ 0\right)$",
         r"$r = S(f_a) - \frac{\lambda}{12}\,|\mathrm{traste}_a - \mathrm{traste}_{\mathrm{prev}}|$"),
        "En cada pull se elige un frame al azar de la nota y se normaliza su espectro (máximo = 1). S⁺ "
        "mide la energía en los armónicos h·f y S⁻ la energía ENTRE armónicos, que delata los errores de "
        "octava. λ favorece la posición que menos mueve la mano.",
        lambda c: (f"Ahora: N = {c.env.n_harmonics}, β = {_fmt(c.env.beta)}, λ = {_fmt(c.env.lam)}, "
                   f"T = {c.env.budget} pulls"),
    ),
    _FormulaBlock(
        "Actualización incremental de Q", None,
        (r"$Q(a) \leftarrow Q(a) + \alpha\,\left[\,r - Q(a)\,\right]$",),
        "Con α = 1/n(a), Q es la media de las recompensas del brazo (ε-greedy, UCB1, Softmax). Con α "
        "constante pesa más lo reciente (optimista).",
        lambda c: f"Ahora: α del optimista = {_fmt(c.agent.optimistic_alpha)}",
    ),
    _FormulaBlock(
        "ε-greedy", "egreedy",
        (r"$A_t = \operatorname{argmax}_a\,Q(a)\quad \mathrm{con\ prob.}\ 1-\varepsilon$",
         r"$A_t \sim \mathrm{Uniforme}\quad \mathrm{con\ prob.}\ \varepsilon$",
         r"$\varepsilon_t = \max\left(\varepsilon_{\min},\ \varepsilon_0\, d^{\,t}\right)$"),
        "Exploración ciega: una fracción ε de los pulls va a un brazo cualquiera.",
        lambda c: "Ahora: " + _decay_text("ε", c.agent.epsilon, c.agent.epsilon_decay, c.agent.epsilon_min),
    ),
    _FormulaBlock(
        "ε-greedy optimista", "optimistic",
        (r"$Q_1(a) = Q_0 > 1 \geq r$",
         r"$Q(a) \leftarrow Q(a) + \alpha\,\left[\,r - Q(a)\,\right],\quad \alpha\ \mathrm{constante}$"),
        "Todo brazo no probado parece excelente; al probarlo «decepciona» y el argmax pasa a otro. Con "
        "α = 1/n el optimismo desaparecería tras un solo pull.",
        lambda c: (f"Ahora: Q₀ = {_fmt(c.agent.q0)}, α = {_fmt(c.agent.optimistic_alpha)}, "
                   f"ε = {_fmt(c.agent.optimistic_epsilon)}"),
    ),
    _FormulaBlock(
        "UCB1", "ucb1",
        (r"$A_t = \operatorname{argmax}_a \left[ Q(a) + c\,\sqrt{\frac{\ln t}{n(a)}} \right]$",),
        "El bono crece con el tiempo t y baja al probar el brazo: todo brazo se revisa, cada vez con menos "
        "frecuencia. Primero se jala cada brazo una vez (n = 0 → bono infinito).",
        lambda c: f"Ahora: c = {_fmt(c.agent.ucb_c)}  (UCB1 original: √2 ≈ 1.414)",
    ),
    _FormulaBlock(
        "Softmax (Boltzmann)", "softmax",
        (r"$\pi(a) = e^{Q(a)/\tau}\ /\ \sum_{b} e^{Q(b)/\tau}$",
         r"$\tau_t = \max\left(\tau_{\min},\ \tau_0\, d^{\,t}\right)$"),
        "τ alta → casi uniforme; τ → 0 → greedy. τ está en unidades de recompensa: importa ΔQ/τ.",
        lambda c: "Ahora: " + _decay_text("τ", c.agent.tau, c.agent.tau_decay, c.agent.tau_min),
    ),
    _FormulaBlock(
        "Regret acumulado", None,
        (r"$L_T = \sum_{t=1}^{T} \left(\mu^{*} - \mu_{A_t}\right)$",),
        "μ* es el valor real del mejor brazo de la nota. El agente nunca ve μ, solo muestras r; μ se "
        "calcula exactamente promediando la saliencia de todos los frames.",
        None,
    ),
)


def render_math_png(tex: str, size: float = 13.0, dpi: float = 96.0, color: str = UI_TEXT,
                    background: str = UI_SURFACE, pad_px: int = 3) -> bytes:
    """Renderiza una fórmula de *mathtext* (sintaxis de LaTeX) a PNG.

    Parameters
    ----------
    tex : str
        Expresión entre ``$...$`` (p. ej. ``r"$Q \\leftarrow Q + \\alpha(r - Q)$"``).
    size : float, optional
        Tamaño de letra en puntos.
    dpi : float, optional
        Resolución en píxeles por pulgada (la de la pantalla, para que el
        tamaño coincida con el de las fuentes de Tk).
    color, background : str, optional
        Color de la tinta y del fondo (hex).
    pad_px : int, optional
        Margen alrededor de la fórmula en píxeles.

    Returns
    -------
    bytes
        Imagen PNG ajustada a la fórmula.

    Raises
    ------
    ValueError
        Si ``tex`` no es una expresión de mathtext válida.

    Notes
    -----
    Se usa la familia matemática Computer Modern (``"cm"``), la de LaTeX, y
    una ``Figure`` sin ``pyplot`` (seguro y sin ventanas). El tamaño de la
    figura se calcula con :class:`matplotlib.mathtext.MathTextParser`.
    """
    prop = FontProperties(size=size, math_fontfamily="cm")
    width, height, depth, _, _ = MathTextParser("path").parse(tex, dpi=72, prop=prop)
    pad_pt = pad_px * 72.0 / dpi
    fig = Figure(figsize=((width + 2 * pad_pt) / 72.0, (height + 2 * pad_pt) / 72.0), dpi=dpi, facecolor=background)
    FigureCanvasAgg(fig)
    fig.text(pad_pt / (width + 2 * pad_pt), (depth + pad_pt) / (height + 2 * pad_pt), tex,
             fontproperties=prop, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=background)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Widgets auxiliares de la pestaña
# ---------------------------------------------------------------------------


class _ResponsiveGrid(ttk.Frame):
    """Marco que reparte sus hijos en tantas columnas como quepan en su ancho.

    Parameters
    ----------
    master : tk.Misc
        Contenedor.
    max_columns : int
        Máximo de columnas.
    padx, pady : int
        Separación entre celdas en píxeles.
    sticky : str
        Cómo se ajusta cada hijo a su celda (``"nw"``: tamaño natural;
        ``"nsew"``: ocupa toda la celda, p. ej. tarjetas del mismo alto).
    style : str
        Estilo ttk del marco.

    Notes
    -----
    El ancho lo impone el contenedor (el lienzo desplazable), no el
    contenido, así que recolocar los hijos al cambiar el número de columnas no
    provoca bucles de ``<Configure>``.
    """

    def __init__(self, master: tk.Misc, max_columns: int = 2, padx: int = 10, pady: int = 3, sticky: str = "nw",
                 style: str = "TFrame") -> None:
        super().__init__(master, style=style)
        self.max_columns = max_columns
        self.padx = padx
        self.pady = pady
        self.sticky = sticky
        self.items: list[tk.Widget] = []
        self.columns = 0
        self.bind("<Configure>", lambda e: self.relayout(e.width), add="+")

    def add(self, widget: tk.Widget) -> None:
        """Añade ``widget`` (hijo de este marco) a la rejilla."""
        self.items.append(widget)
        self.columns = 0  # fuerza recolocar
        self.relayout(self.winfo_width())

    def relayout(self, width: int) -> None:
        """Recoloca los hijos si cambia el número de columnas que caben en ``width`` píxeles.

        Parameters
        ----------
        width : int
            Ancho disponible en píxeles.
        """
        if not self.items:
            return
        # n columnas ocupan n·ancho + (n−1)·padx (la última no lleva separación).
        cell = max(w.winfo_reqwidth() for w in self.items) + self.padx
        columns = max(1, min(self.max_columns, len(self.items), (int(width) + self.padx) // max(cell, 1)))
        if columns == self.columns:
            return
        self.columns = columns
        for col in range(self.max_columns):
            self.columnconfigure(col, weight=1 if col < columns else 0, uniform="cell" if col < columns else "")
        for i, widget in enumerate(self.items):
            last = i % columns == columns - 1
            widget.grid(row=i // columns, column=i % columns, sticky=self.sticky,
                        padx=(0, 0 if last else self.padx), pady=self.pady)


class _Banner(ttk.Frame):
    """Aviso de color con título, detalle y botón de acción opcional.

    Parameters
    ----------
    master : tk.Misc
        Contenedor.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, style="CfgBanner.info.TFrame", padding=(0, 0, 10, 0))
        self.kind = "info"
        self.strip = tk.Frame(self, width=5, background=UI_ACCENT)
        self.strip.pack(side="left", fill="y", padx=(0, 10))
        self.button = ttk.Button(self, style="Accent.TButton")
        self.button.pack(side="right", padx=(10, 0), pady=8)
        self.body = ttk.Frame(self, style="CfgBanner.info.TFrame")
        self.body.pack(side="left", fill="both", expand=True, pady=7)
        self.title = ttk.Label(self.body, style="CfgBannerTitle.info.TLabel")
        self.title.pack(anchor="w")
        self.detail = ttk.Label(self.body, style="CfgBanner.info.TLabel", justify="left", wraplength=900)
        self.detail.pack(anchor="w", fill="x")
        self.body.bind("<Configure>", lambda e: self.detail.configure(wraplength=max(200, e.width - 8)), add="+")
        self._command: Callable[[], None] | None = None
        self.button.configure(command=self._invoke)
        self.button_help = ""
        Tooltip(self.button, lambda: self.button_help)

    def _invoke(self) -> None:
        """Ejecuta la acción asociada al botón."""
        if self._command is not None:
            self._command()

    def set(self, kind: str, title: str, detail: str = "", button: str = "",
            command: Callable[[], None] | None = None, enabled: bool = True, help_text: str = "") -> None:
        """Cambia el contenido del aviso.

        Parameters
        ----------
        kind : str
            ``"info"``, ``"warn"``, ``"ok"`` o ``"error"`` (colores de :data:`_BANNER_COLORS`).
        title : str
            Primera línea, en negrita.
        detail : str, optional
            Explicación (se ajusta al ancho).
        button : str, optional
            Texto del botón; vacío = sin botón.
        command : Callable[[], None] | None, optional
            Acción del botón.
        enabled : bool, optional
            Si es False el botón aparece deshabilitado (p. ej. con una tarea en curso).
        help_text : str, optional
            Tooltip del botón (qué hará exactamente).
        """
        self.kind = kind
        self.configure(style=f"CfgBanner.{kind}.TFrame")
        self.body.configure(style=f"CfgBanner.{kind}.TFrame")
        self.strip.configure(background=_BANNER_COLORS[kind][1])
        self.title.configure(text=title, style=f"CfgBannerTitle.{kind}.TLabel")
        self.detail.configure(text=detail, style=f"CfgBanner.{kind}.TLabel")
        if detail:
            self.detail.pack(anchor="w", fill="x")
        else:
            self.detail.pack_forget()
        self._command = command
        self.button_help = help_text
        if button:
            self.button.configure(text=button, state="normal" if enabled else "disabled")
            # before=body: el botón se coloca antes que el texto, así nunca se queda sin espacio.
            self.button.pack(side="right", padx=(10, 0), pady=8, before=self.body)
        else:
            self.button.pack_forget()


# ---------------------------------------------------------------------------
# La pestaña
# ---------------------------------------------------------------------------


class ConfigTab(ttk.Frame):
    """Pestaña «2 · Configuración»: edición de todos los hiperparámetros.

    Parameters
    ----------
    master : tk.Misc
        El ``ttk.Notebook`` de la ventana principal.
    app : gui.app.SmartunerApp
        Aplicación (estado compartido ``app.state`` y acciones ``app.analyze``,
        ``app.retranscribe``, ``app.run_experiment``, ``app.save_config``...).

    Attributes
    ----------
    fields : dict[str, ParamField]
        Un control por clave de :data:`src.config.PARAM_SPECS`.
    algo_vars : dict[str, tk.BooleanVar]
        Casillas «Incluir en el experimento» por algoritmo.
    status_kind : str
        Estado del aviso principal: ``"empty"`` (sin audio), ``"reanalyze"``,
        ``"retranscribe"`` o ``"synced"``.
    problems : list[str]
        Problemas de validación de la configuración vigente (vacía si es válida).
    pending_keys : set[str]
        Parámetros cuyo valor difiere del usado en el análisis vigente (se
        resaltan en naranja); vacío si no hay análisis.
    formulas_visible : tk.BooleanVar
        Si el panel de fórmulas está visible.
    formula_images : list[tk.PhotoImage]
        Fórmulas renderizadas (se guardan para que Tk no las libere).

    Examples
    --------
    Uso desde la aplicación (la crea :class:`gui.app.SmartunerApp`)::

        tab = ConfigTab(app.notebook, app)
        tab.fields["env.lam"].var.set("0.3")
        tab.fields["env.lam"]._changed()      # como si el usuario pulsara Enter
        tab.status_kind                       # "retranscribe" si había un análisis
    """

    def __init__(self, master: tk.Misc, app: Any) -> None:
        super().__init__(master, padding=(12, 10, 12, 8))
        self.app = app
        self.fields: dict[str, ParamField] = {}
        self.algo_vars: dict[str, tk.BooleanVar] = {}
        self.status_kind = "empty"
        self.problems: list[str] = validation_problems(app.state.config)
        self.formulas_visible = tk.BooleanVar(value=True)
        self.formula_images: list[tk.PhotoImage] = []
        self._busy = bool(getattr(app, "busy", False))
        self._emitting_key: str | None = None
        self._algo_tags: dict[str, ttk.Label] = {}
        self._formula_values: list[tuple[ttk.Label, Callable[[Config], str]]] = []
        self._sections: dict[str, ttk.LabelFrame] = {}
        self.pending_keys: set[str] = set()

        self._setup_styles()
        self._build_header()
        self._build_banners()
        self._build_body()

        events = app.state.events
        events.subscribe("config_changed", self._on_config_changed)
        events.subscribe("analysis_ready", self._on_analysis_ready)
        events.subscribe("transcriptions_ready", self._on_transcriptions_ready)
        events.subscribe("experiment_ready", self._on_experiment_ready)
        events.subscribe("busy_changed", self._on_busy_changed)
        events.subscribe("algorithm_selected", self._on_algorithm_selected)

        self._on_algorithm_selected(app.state.selected_algorithm)
        self._update_derived()

    # ------------------------------------------------------------ estilos
    def _setup_styles(self) -> None:
        """Define los estilos ttk propios de la pestaña (prefijo ``Cfg``)."""
        style = ttk.Style(self)
        base = tkfont.nametofont("TkDefaultFont")
        family, size = base.actual("family"), int(base.actual("size")) or 9
        size = abs(size)
        self._family = family
        self._base_size = size
        self._bold_font = (family, size, "bold")
        self._bg = style.lookup("TFrame", "background") or "#dcdad5"
        style.configure("CfgTitle.TLabel", font=(family, size + 4, "bold"), foreground=UI_TEXT)
        style.configure("CfgSection.TLabelframe", padding=(12, 6, 12, 10))
        style.configure("CfgSection.TLabelframe.Label", font=(family, size + 2, "bold"), foreground=UI_TEXT)
        style.configure("CfgGroup.TLabel", font=(family, size, "bold"), foreground=UI_TEXT_SECONDARY)
        style.configure("CfgNote.TLabel", foreground=UI_TEXT_SECONDARY)
        style.configure("CfgError.TLabel", foreground=UI_ERROR)
        style.configure("CfgRule.TLabel", font=(family, size + 1), foreground=UI_TEXT)
        style.configure("CfgAlgoName.TLabel", font=(family, size + 1, "bold"), foreground=UI_TEXT)
        style.configure("CfgTag.TLabel", font=(family, size - 1, "bold"), foreground=UI_ACCENT)
        for kind, (bg, _accent, fg) in _BANNER_COLORS.items():
            style.configure(f"CfgBanner.{kind}.TFrame", background=bg)
            style.configure(f"CfgBanner.{kind}.TLabel", background=bg, foreground=fg)
            style.configure(f"CfgBannerTitle.{kind}.TLabel", background=bg, foreground=fg,
                            font=(family, size + 1, "bold"))
        style.configure("CfgCard.TFrame", background=UI_SURFACE)
        style.configure("CfgCard.TLabel", background=UI_SURFACE, foreground=UI_TEXT)
        style.configure("CfgCardTitle.TLabel", background=UI_SURFACE, foreground=UI_TEXT,
                        font=(family, size + 1, "bold"))
        style.configure("CfgCardHeader.TLabel", background=UI_SURFACE, foreground=UI_TEXT,
                        font=(family, size + 4, "bold"))
        style.configure("CfgCardMuted.TLabel", background=UI_SURFACE, foreground=UI_TEXT_SECONDARY)
        style.configure("CfgCardValue.TLabel", background=UI_SURFACE, foreground=_VALUE_TEXT,
                        font=(family, size, "bold"))
        style.configure("CfgCard.TSeparator", background=GRID)

    # ---------------------------------------------------------- cabecera
    def _build_header(self) -> None:
        """Título, explicación y botones Guardar/Cargar/Restaurar/Fórmulas."""
        header = ttk.Frame(self)
        header.pack(side="top", fill="x")
        header.columnconfigure(0, weight=1)
        titles = ttk.Frame(header)
        titles.grid(row=0, column=0, sticky="ew")
        ttk.Label(titles, text="Configuración de hiperparámetros", style="CfgTitle.TLabel").pack(anchor="w")
        subtitle = ttk.Label(
            titles, style="CfgNote.TLabel", justify="left",
            text="Cada cambio válido se aplica al instante a la configuración. Pasa el ratón sobre un "
                 "parámetro para ver qué hace y su rango.")
        subtitle.pack(anchor="w", fill="x", pady=(2, 0))
        self._wrap(subtitle, titles, 8)

        buttons = ttk.Frame(header)
        buttons.grid(row=0, column=1, sticky="ne", padx=(12, 0))
        self.save_button = ttk.Button(buttons, text="Guardar JSON…", command=self.save_json)
        self.load_button = ttk.Button(buttons, text="Cargar JSON…", command=self.load_json)
        self.restore_button = ttk.Button(buttons, text="Restaurar valores por defecto", command=self.restore_defaults)
        self.formulas_check = ttk.Checkbutton(buttons, text="Mostrar fórmulas", variable=self.formulas_visible,
                                              command=self._apply_formulas_visibility)
        for i, widget in enumerate((self.save_button, self.load_button, self.restore_button)):
            widget.grid(row=0, column=i, padx=(0 if i == 0 else 6, 0))
        self.formulas_check.grid(row=0, column=3, padx=(14, 0))
        Tooltip(self.save_button, "Guarda TODA la configuración vigente en un archivo JSON (el mismo formato que "
                                  "usa la CLI: python main.py --cli --config archivo.json).")
        Tooltip(self.load_button, "Carga una configuración desde JSON. Se valida antes de aplicarla: si tiene "
                                  "valores fuera de rango no se cambia nada.")
        Tooltip(self.restore_button, "Vuelve a los valores por defecto de src/config.py (los calibrados con el "
                                     "dataset). Pide confirmación.")
        Tooltip(self.formulas_check, "Muestra u oculta el panel de fórmulas (recompensa y reglas de los cuatro "
                                     "agentes), pensado para proyectar en clase.")

    # ------------------------------------------------------------ avisos
    def _build_banners(self) -> None:
        """Aviso de estado (re-analizar / re-transcribir / al día) y aviso de validación."""
        self.banner_area = ttk.Frame(self)
        self.banner_area.pack(side="top", fill="x", pady=(10, 8))
        self.status_banner = _Banner(self.banner_area)
        self.status_banner.pack(side="top", fill="x")
        self.validation_banner = _Banner(self.banner_area)

    # -------------------------------------------------------------- cuerpo
    def _build_body(self) -> None:
        """Panel dividido: parámetros desplazables (izquierda) y fórmulas (derecha)."""
        self.paned = ttk.Panedwindow(self, orient="horizontal")
        self.paned.pack(side="top", fill="both", expand=True)

        self.scroller = ScrollableFrame(self.paned)
        self.scroller.canvas.configure(background=self._bg, width=760)
        self.paned.add(self.scroller, weight=1)
        inner = self.scroller.inner
        for section in SECTIONS[:2]:
            self._build_section(inner, section)
        self._build_algorithms(inner)
        self._build_section(inner, SECTIONS[2])
        self._build_leftovers(inner)
        self._build_experiment_footer()

        self.formula_panel = tk.Frame(self.paned, background=GRID, padx=1, pady=1)
        self._build_formulas(self.formula_panel)
        self.paned.add(self.formula_panel, weight=0)

    def _wrap(self, label: ttk.Label, container: tk.Widget, margin: int = 0) -> None:
        """Ajusta el ``wraplength`` de ``label`` al ancho de ``container`` al redimensionar."""
        # 8 px de holgura: con wraplength igual al ancho exacto, ttk recorta la última letra.
        container.bind("<Configure>", lambda e: label.configure(wraplength=max(160, e.width - margin - 8)), add="+")

    def _section_frame(self, parent: tk.Widget, title: str, note: str) -> ttk.LabelFrame:
        """Crea un ``LabelFrame`` de sección con su nota explicativa."""
        frame = ttk.LabelFrame(parent, text=title, style="CfgSection.TLabelframe")
        frame.pack(side="top", fill="x", padx=(0, 12), pady=(0, 12))
        if note:
            label = ttk.Label(frame, text=note, style="CfgNote.TLabel", justify="left")
            label.pack(side="top", anchor="w", fill="x", pady=(0, 4))
            self._wrap(label, frame, 28)
        return frame

    def _label_chars(self, keys: Sequence[str]) -> tuple[int, int]:
        """Ancho (en caracteres «0») de etiqueta y de unidad que necesitan las claves ``keys``."""
        font = tkfont.nametofont("TkDefaultFont")
        zero = max(font.measure("0"), 1)
        specs = [PARAM_SPECS[k] for k in keys if k in PARAM_SPECS]
        label_px = max((font.measure(s.label) for s in specs), default=zero * 10)
        unit_px = max((font.measure(s.unit) for s in specs), default=0)
        return math.ceil(label_px / zero) + 1, max(2, math.ceil(unit_px / zero) + 1)

    def _add_group(self, parent: tk.Widget, group: _Group, widths: tuple[int, int]) -> None:
        """Añade un subgrupo: título, nota opcional y sus campos en rejilla adaptable."""
        keys = [k for k in group.keys if k in PARAM_SPECS and k not in self.fields]
        if not keys:
            return
        ttk.Label(parent, text=group.title, style="CfgGroup.TLabel").pack(side="top", anchor="w", pady=(8, 2))
        if group.note:
            note = ttk.Label(parent, text=group.note, style="CfgNote.TLabel", justify="left")
            note.pack(side="top", anchor="w", fill="x")
            self._wrap(note, parent, 28)
        grid = _ResponsiveGrid(parent, max_columns=2)
        grid.pack(side="top", fill="x")
        for key in keys:
            grid.add(self._make_field(grid, key, widths))

    def _build_section(self, parent: tk.Widget, section: _Section) -> None:
        """Construye una sección de :data:`SECTIONS` con sus subgrupos (y los parámetros sobrantes)."""
        frame = self._section_frame(parent, section.title, section.note)
        self._sections[section.key] = frame
        listed = [k for g in section.groups for k in g.keys]
        extra = tuple(k for k in PARAM_SPECS if k.split(".", 1)[0] in section.prefixes and k not in listed)
        widths = self._label_chars(listed + list(extra))
        for group in section.groups:
            self._add_group(frame, group, widths)
        if extra:
            self._add_group(frame, _Group("Otros parámetros", extra), widths)

    def _build_leftovers(self, parent: tk.Widget) -> None:
        """Sección «Otros» con las claves de PARAM_SPECS sin sección conocida (por compatibilidad futura)."""
        known = {p for s in SECTIONS for p in s.prefixes} | {"agent"}
        extra = tuple(k for k in PARAM_SPECS if k not in self.fields and k.split(".", 1)[0] not in known)
        if extra:
            frame = self._section_frame(parent, "Otros parámetros", "")
            self._add_group(frame, _Group("Parámetros sin sección", extra), self._label_chars(extra))

    def _make_field(self, parent: tk.Widget, key: str, widths: tuple[int, int]) -> ParamField:
        """Crea el :class:`ParamField` de ``key`` y lo registra en :attr:`fields`.

        Parameters
        ----------
        parent : tk.Widget
            Contenedor.
        key : str
            Clave de :data:`PARAM_SPECS`.
        widths : tuple[int, int]
            Ancho de la etiqueta y de la unidad (en caracteres) para alinear la sección.

        Returns
        -------
        ParamField
            El control creado.
        """
        field = ParamField(parent, key, PARAM_SPECS[key], self.app.state.config.get(key), on_change=self._on_param_change)
        # Etiquetas y unidades con el ancho que necesita la sección (ParamField usa
        # 26 y 4 caracteres fijos, que cortarían «Prob. de cambio voz/sin voz (pYIN)»
        # o «muestras»).
        field.label.configure(width=widths[0])
        field.unit.configure(width=widths[1])
        kind = PARAM_SPECS[key].kind
        if kind == "float_list":
            field.control.configure(width=26)
        elif kind in ("int", "float", "int_or_none"):
            field.control.configure(width=10)
        self._guard_wheel(field.control)
        self.fields[key] = field
        return field

    def _guard_wheel(self, widget: tk.Widget) -> None:
        """Hace que la rueda del ratón desplace la página en vez de cambiar el valor del control.

        Los ``Spinbox`` y ``Combobox`` de ttk cambian su valor con la rueda: en
        un formulario desplazable, eso modificaría parámetros sin querer.
        """
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self._scroll_page)

    def _scroll_page(self, event: tk.Event) -> str:
        """Desplaza la lista de parámetros y corta la acción por defecto del control."""
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.scroller.canvas.yview_scroll(delta, "units")
        return "break"

    # ---------------------------------------------------------- algoritmos
    def _build_algorithms(self, parent: tk.Widget) -> None:
        """Sección «Algoritmos»: una tarjeta por agente y la recomendación final."""
        frame = self._section_frame(
            parent, "Algoritmos",
            "Etapa 7: los cuatro agentes bandit. La tablatura se calcula siempre con los cuatro; la casilla "
            "decide cuáles entran en el experimento comparativo (al menos uno).")
        self._sections["agent"] = frame
        agent_keys = [k for keys in ALGO_PARAMS.values() for k in keys]
        other = [k for k in PARAM_SPECS if k.startswith("agent.") and k not in agent_keys]
        widths = self._label_chars(agent_keys)
        grid = _ResponsiveGrid(frame, max_columns=2, padx=12, pady=5, sticky="nsew")
        grid.pack(side="top", fill="x", pady=(4, 0))
        for algo in ALGORITHMS:
            grid.add(self._build_algorithm_card(grid, algo, widths))
        self.algo_message = ttk.Label(frame, text="", style="CfgError.TLabel")
        self.algo_message.pack(side="top", anchor="w")
        if other:
            self._add_group(frame, _Group("Al agotar el presupuesto T", tuple(other),
                                          "Qué posición se reporta en la tablatura: el brazo más jalado "
                                          "(robusto) o el de mayor Q."), self._label_chars(other))

    def _build_algorithm_card(self, parent: tk.Widget, algo: str, widths: tuple[int, int]) -> ttk.LabelFrame:
        """Tarjeta de un algoritmo: color, nombre, casilla, regla y parámetros.

        Parameters
        ----------
        parent : tk.Widget
            Rejilla de tarjetas.
        algo : str
            Identificador de :data:`src.config.ALGORITHMS`.
        widths : tuple[int, int]
            Ancho de etiqueta y unidad de los campos.

        Returns
        -------
        ttk.LabelFrame
            La tarjeta.
        """
        header = ttk.Frame(parent)
        swatch = tk.Canvas(header, width=14, height=14, highlightthickness=0, borderwidth=0, background=self._bg)
        swatch.create_rectangle(1, 1, 13, 13, fill=ALGO_COLORS[algo], outline=ALGO_COLORS[algo])
        swatch.pack(side="left", padx=(0, 6))
        ttk.Label(header, text=ALGO_LABELS[algo], style="CfgAlgoName.TLabel").pack(side="left")
        tag = ttk.Label(header, text="· en las vistas de un algoritmo", style="CfgTag.TLabel")
        self._algo_tags[algo] = tag
        Tooltip(tag, "Algoritmo que muestran ahora «Ejecución en vivo» y «Tablatura».")
        card = ttk.LabelFrame(parent, labelwidget=header, padding=(10, 4, 10, 8))
        header.lift(card)

        var = tk.BooleanVar(value=algo in self.app.state.config.experiment.algorithms)
        self.algo_vars[algo] = var
        check = ttk.Checkbutton(card, text="Incluir en el experimento", variable=var,
                                command=lambda a=algo: self._on_algorithm_toggle(a))
        check.pack(side="top", anchor="w")
        Tooltip(check, f"Si se marca, {ALGO_LABELS[algo]} entra en el experimento comparativo (curvas, "
                       "precisión, barridos). Debe quedar al menos un algoritmo marcado.")
        rule, how = ALGO_RULES[algo]
        rule_label = ttk.Label(card, text=rule, style="CfgRule.TLabel")
        rule_label.pack(side="top", anchor="w", pady=(4, 0))
        Tooltip(rule_label, "Regla de decisión del agente (ver el panel «Fórmulas»).")
        how_label = ttk.Label(card, text=how, style="CfgNote.TLabel", justify="left")
        how_label.pack(side="top", anchor="w", fill="x", pady=(0, 4))
        # Ancho fijo: si dependiera del ancho de la tarjeta, la tarjeta pediría más
        # ancho al ensancharse y la rejilla nunca volvería a dos columnas.
        how_label.configure(wraplength=330)
        for key in ALGO_PARAMS[algo]:
            if key in PARAM_SPECS:
                self._make_field(card, key, widths).pack(side="top", anchor="w", pady=2)
        return card

    def _build_experiment_footer(self) -> None:
        """Duración estimada, vigencia de los resultados y botón «Ejecutar experimento»."""
        frame = self._sections["experiment"]
        footer = ttk.Frame(frame)
        footer.pack(side="top", fill="x", pady=(10, 0))
        ttk.Separator(footer).pack(side="top", fill="x", pady=(0, 8))
        self.run_experiment_button = ttk.Button(footer, text="Ejecutar experimento", command=self.run_experiment)
        self.run_experiment_button.pack(side="right", anchor="n", padx=(12, 0))
        Tooltip(self.run_experiment_button, "Ejecuta el experimento comparativo con esta configuración (igual que "
                                            "Ejecutar → Ejecutar experimento completo). Los resultados aparecen "
                                            "en «5 · Comparación».")
        texts = ttk.Frame(footer)
        texts.pack(side="left", fill="x", expand=True)
        self.estimate_label = ttk.Label(texts, text="", justify="left")
        self.estimate_label.pack(side="top", anchor="w", fill="x")
        self.experiment_state_label = ttk.Label(texts, text="", style="CfgNote.TLabel", justify="left")
        self.experiment_state_label.pack(side="top", anchor="w", fill="x")
        self._wrap(self.estimate_label, texts, 4)
        self._wrap(self.experiment_state_label, texts, 4)

    # ------------------------------------------------------------ fórmulas
    def _build_formulas(self, container: tk.Frame) -> None:
        """Panel «Fórmulas»: ecuaciones renderizadas con mathtext y valores vigentes."""
        scroller = ScrollableFrame(container)
        scroller.canvas.configure(background=UI_SURFACE, width=390)
        scroller.inner.configure(style="CfgCard.TFrame", padding=(16, 12, 12, 16))
        scroller.pack(fill="both", expand=True)
        self.formula_scroller = scroller
        inner = scroller.inner
        ttk.Label(inner, text="Fórmulas", style="CfgCardHeader.TLabel").pack(anchor="w")
        intro = ttk.Label(inner, style="CfgCardMuted.TLabel", justify="left",
                          text="Para proyectar en clase. La línea «Ahora» sigue la configuración vigente.")
        intro.pack(anchor="w", fill="x", pady=(0, 6))
        self._wrap(intro, inner, 32)
        dpi = float(self.winfo_fpixels("1i")) or 96.0
        for i, block in enumerate(FORMULA_BLOCKS):
            if i:
                ttk.Separator(inner, style="CfgCard.TSeparator").pack(fill="x", pady=8)
            title_row = ttk.Frame(inner, style="CfgCard.TFrame")
            title_row.pack(anchor="w", fill="x")
            if block.algorithm is not None:
                swatch = tk.Canvas(title_row, width=12, height=12, highlightthickness=0, borderwidth=0,
                                   background=UI_SURFACE)
                color = ALGO_COLORS[block.algorithm]
                swatch.create_rectangle(1, 1, 11, 11, fill=color, outline=color)
                swatch.pack(side="left", padx=(0, 6))
            ttk.Label(title_row, text=block.title, style="CfgCardTitle.TLabel").pack(side="left")
            for tex in block.formulas:
                self._formula_label(inner, tex, dpi).pack(anchor="w", pady=1, padx=(4, 0))
            explanation = ttk.Label(inner, text=block.explanation, style="CfgCardMuted.TLabel", justify="left")
            explanation.pack(anchor="w", fill="x", pady=(2, 0))
            self._wrap(explanation, inner, 32)
            if block.values is not None:
                value_label = ttk.Label(inner, text="", style="CfgCardValue.TLabel", justify="left")
                value_label.pack(anchor="w", fill="x", pady=(2, 0))
                self._wrap(value_label, inner, 32)
                self._formula_values.append((value_label, block.values))

    def _formula_label(self, parent: tk.Widget, tex: str, dpi: float) -> ttk.Label:
        """Etiqueta con la fórmula renderizada (o su texto fuente si mathtext falla)."""
        try:
            png = render_math_png(tex, size=self._base_size + 4, dpi=dpi)
            image = tk.PhotoImage(master=self, data=base64.b64encode(png).decode("ascii"), format="png")
        except Exception:  # noqa: BLE001 - una fórmula rota no debe impedir usar la pestaña
            logger.warning("No se pudo renderizar la fórmula %s", tex, exc_info=True)
            return ttk.Label(parent, text=tex.strip("$"), style="CfgCard.TLabel", font="TkFixedFont")
        self.formula_images.append(image)
        return ttk.Label(parent, image=image, style="CfgCard.TLabel")

    def _apply_formulas_visibility(self) -> None:
        """Muestra u oculta el panel de fórmulas según :attr:`formulas_visible`."""
        panes = [str(p) for p in self.paned.panes()]
        shown = str(self.formula_panel) in panes
        if self.formulas_visible.get() and not shown:
            self.paned.add(self.formula_panel, weight=0)
        elif not self.formulas_visible.get() and shown:
            self.paned.forget(self.formula_panel)

    def set_formulas_visible(self, visible: bool) -> None:
        """Muestra (``True``) u oculta (``False``) el panel de fórmulas.

        Parameters
        ----------
        visible : bool
            Visibilidad deseada.
        """
        self.formulas_visible.set(bool(visible))
        self._apply_formulas_visibility()

    # ----------------------------------------------------- edición de valores
    def _on_param_change(self, key: str, value: Any) -> None:
        """Escribe un valor VÁLIDO de un :class:`ParamField` en la configuración.

        Si el valor rompe una restricción de :meth:`Config.validate` que no cabe
        en el rango del control (p. ej. una lista de barrido con valores fuera
        del rango del parámetro), se deshace y se muestra el error bajo el campo.

        Parameters
        ----------
        key : str
            Clave ``"seccion.campo"``.
        value : Any
            Valor ya convertido y comprobado contra su rango por ``ParamField``.
        """
        cfg = self.app.state.config
        old = cfg.get(key)
        if value == old and type(value) is type(old):
            return  # p. ej. el foco salió del campo sin cambiarlo
        before = validation_problems(cfg)
        cfg.set(key, value)
        after = validation_problems(cfg)
        new = [p for p in after if p not in before]
        if new:
            cfg.set(key, old)
            message = new[0]
            if message.startswith(key):
                message = message[len(key):].lstrip(" :")
                message = message[:1].upper() + message[1:]
            # ParamField no expone un método público para mostrar errores externos.
            self.fields[key]._show_error(message)
            logger.warning("Valor rechazado para %s: %s", key, "; ".join(new))
            return
        logger.info("Configuración: %s = %s", key, format_value(key, value))
        self._emit_change(key)

    def _on_algorithm_toggle(self, algo: str) -> None:
        """Actualiza ``experiment.algorithms`` (orden de :data:`ALGORITHMS`, al menos uno).

        Parameters
        ----------
        algo : str
            Algoritmo cuya casilla cambió.
        """
        selected = [a for a in ALGORITHMS if self.algo_vars[a].get()]
        if not selected:
            self.algo_vars[algo].set(True)
            self.algo_message.configure(text="⚠ Debe quedar al menos un algoritmo incluido en el experimento.")
            self.bell()
            return
        self.algo_message.configure(text="")
        if selected == list(self.app.state.config.experiment.algorithms):
            return
        self.app.state.config.set("experiment.algorithms", selected)
        logger.info("Configuración: algoritmos del experimento = %s", ", ".join(ALGO_LABELS[a] for a in selected))
        self._emit_change("experiment.algorithms")

    def _emit_change(self, key: str) -> None:
        """Emite ``config_changed`` marcando que el cambio salió de esta pestaña."""
        self._emitting_key = key
        try:
            self.app.state.events.emit("config_changed", key=key)
        finally:
            self._emitting_key = None

    # ------------------------------------------------------------ acciones
    def save_json(self) -> None:
        """Guarda la configuración en JSON (diálogo de :meth:`SmartunerApp.save_config`)."""
        self.app.save_config()

    def load_json(self) -> None:
        """Carga una configuración desde JSON (:meth:`SmartunerApp.load_config` emite ``config_changed``)."""
        self.app.load_config()

    def restore_defaults(self, confirm: bool = True) -> bool:
        """Vuelve a los valores por defecto (``Config()``) y avisa a todas las pestañas.

        Parameters
        ----------
        confirm : bool, optional
            Si es True (botón) se pide confirmación; los tests pasan False.

        Returns
        -------
        bool
            True si se restauró.
        """
        if confirm and not messagebox.askyesno(
                "Restaurar valores por defecto",
                "¿Volver a los valores por defecto de todos los parámetros?\n\n"
                "Los cambios que no hayas guardado en un JSON se perderán.",
                parent=self.winfo_toplevel()):
            return False
        self.app.state.config = Config()
        logger.info("Configuración restaurada a los valores por defecto")
        self.app.state.events.emit("config_changed", key=None)
        return True

    def reanalyze(self) -> None:
        """Vuelve a analizar el audio vigente con la configuración actual (``app.analyze``)."""
        analysis = self.app.state.analysis
        if analysis is None:
            self.app.open_audio()
            return
        if self._refuse_if_invalid():
            return
        self.app.analyze(analysis.path)

    def retranscribe(self) -> None:
        """Recalcula los entornos y re-transcribe con la configuración actual (``app.retranscribe``)."""
        if self._refuse_if_invalid():
            return
        self.app.retranscribe()

    def run_experiment(self) -> None:
        """Ejecuta el experimento comparativo (``app.run_experiment``)."""
        if self._refuse_if_invalid():
            return
        self.app.run_experiment()

    def _refuse_if_invalid(self) -> bool:
        """Muestra los problemas y devuelve True si la configuración no es válida."""
        if not self.problems:
            return False
        show_error(self.winfo_toplevel(), "Configuración inválida", "Corrige primero:\n• " + "\n• ".join(self.problems))
        return True

    # ------------------------------------------------------------- eventos
    def _on_config_changed(self, key: str | None = None, **_: Any) -> None:
        """Sincroniza la vista con la configuración tras ``config_changed``.

        Parameters
        ----------
        key : str | None
            Parámetro que cambió (None = varios: JSON cargado, valores por defecto).
        """
        if key is None or key != self._emitting_key:
            self.refresh_fields()
        self._update_derived()

    def _on_analysis_ready(self, analysis: Any = None, **_: Any) -> None:
        """Nuevo análisis: el aviso de cambios pendientes se recalcula contra su configuración."""
        self._update_derived()

    def _on_transcriptions_ready(self, transcriptions: Any = None, **_: Any) -> None:
        """Nuevas transcripciones: actualiza el aviso de cambios pendientes."""
        self._update_derived()

    def _on_experiment_ready(self, result: Any = None, **_: Any) -> None:
        """Nuevos resultados del experimento: actualiza la línea de vigencia."""
        self._update_experiment_info()

    def _on_busy_changed(self, busy: bool = False, **_: Any) -> None:
        """Deshabilita las acciones que lanzan tareas mientras hay una en curso."""
        self._busy = bool(busy)
        self._update_status()
        self._update_experiment_info()

    def _on_algorithm_selected(self, algorithm: str = "", **_: Any) -> None:
        """Marca la tarjeta del algoritmo que muestran las vistas de un solo algoritmo."""
        for algo, tag in self._algo_tags.items():
            if algo == algorithm:
                tag.pack(side="left", padx=(6, 0))
            else:
                tag.pack_forget()

    # -------------------------------------------------- vista sincronizada
    def refresh_fields(self) -> None:
        """Muestra en cada control el valor vigente de ``app.state.config`` (sin emitir eventos)."""
        cfg = self.app.state.config
        for key, field in self.fields.items():
            field.set(cfg.get(key))
        for algo, var in self.algo_vars.items():
            var.set(algo in cfg.experiment.algorithms)
        self.algo_message.configure(text="")

    def pending_changes(self) -> tuple[list[str], list[str]]:
        """Cambios aún no aplicados al análisis vigente.

        Returns
        -------
        reanalyze : list[str]
            Claves que exigen volver a analizar el audio.
        retranscribe : list[str]
            Claves que se aplican re-transcribiendo.
            Ambas listas están vacías si no hay análisis.
        """
        analysis = self.app.state.analysis
        if analysis is None:
            return [], []
        return config_differences(analysis.config, self.app.state.config)

    def _update_derived(self) -> None:
        """Recalcula todo lo que depende de la configuración: validación, avisos, dependencias y fórmulas."""
        self.problems = validation_problems(self.app.state.config)
        self._update_field_styles()
        self._update_status()
        self._update_validation()
        self._update_formula_values()
        self._update_experiment_info()

    def _update_field_styles(self) -> None:
        """Desactiva los campos que no se usan y resalta en naranja los cambios sin aplicar.

        Un campo se desactiva si la configuración vigente no lo usa (p. ej.
        ``env.n_fft`` con espectro CQT, ``agent.epsilon_min`` sin decaimiento o
        una lista de barrido de un algoritmo excluido). Su etiqueta se pinta de
        naranja y en negrita si el valor difiere del usado en el análisis vigente.
        """
        cfg = self.app.state.config
        exp = cfg.experiment
        enabled: dict[str, bool] = {
            "env.n_fft": cfg.env.spectrum == "stft",
            "env.bins_per_octave": cfg.env.spectrum == "cqt",
            "agent.epsilon_min": cfg.agent.epsilon_decay < 1.0,
            "agent.tau_min": cfg.agent.tau_decay < 1.0,
            "experiment.sweep_runs": exp.run_sweeps or exp.run_lambda_sweep,
            "experiment.sweep_lambda": exp.run_lambda_sweep,
        }
        for _param, algo, list_name in SWEEP_SPECS:
            enabled[f"experiment.{list_name}"] = exp.run_sweeps and algo in exp.algorithms
        reanalyze, retranscribe = self.pending_changes()
        self.pending_keys = set(reanalyze) | set(retranscribe)
        for key, field in self.fields.items():
            on = enabled.get(key, True)
            if key in enabled:
                field.control.state(["!disabled"] if on else ["disabled"])
            pending = on and key in self.pending_keys
            field.label.configure(foreground=_DISABLED_TEXT if not on else _PENDING_TEXT if pending else UI_TEXT,
                                  font=self._bold_font if pending else "TkDefaultFont")

    def _update_status(self) -> None:
        """Actualiza el aviso principal (sin audio / re-analizar / re-transcribir / al día)."""
        state = self.app.state
        can_run = not self._busy and not self.problems
        busy_note = " (espera a que termine la tarea en curso)" if self._busy else ""
        analysis = state.analysis
        if analysis is None:
            self.status_kind = "empty"
            self.status_banner.set(
                "info", "Aún no hay audio analizado",
                "Abre un MP3 (Archivo → Abrir) o genera el dataset sintético (Archivo → Generar dataset "
                "sintético…): se analizará con estos parámetros. Puedes ajustarlos antes o después." + busy_note,
                "Abrir audio…", self.app.open_audio, enabled=not self._busy,
                help_text="Elige un archivo de audio (MP3, WAV...) y lo analiza con la configuración actual.")
            return
        reanalyze, retranscribe = self.pending_changes()
        name = getattr(analysis.path, "name", str(analysis.path))
        if reanalyze:
            self.status_kind = "reanalyze"
            detail = (f"Cambiaste parámetros de las etapas 1–5: {describe_changes(reanalyze, analysis.config, state.config)}. "
                      f"Hay que volver a decodificar, filtrar, segmentar y estimar el pitch de «{name}».")
            if retranscribe:
                detail += f" También se aplicarán: {describe_changes(retranscribe, analysis.config, state.config)}."
            detail += " Los parámetros cambiados se marcan en naranja."
            self.status_banner.set("warn", "Requiere re-analizar el audio", detail + busy_note,
                                   "Analizar de nuevo", self.reanalyze, enabled=can_run,
                                   help_text="Repite las etapas 1–6 sobre el mismo archivo con la configuración "
                                             "actual y vuelve a transcribir con los 4 algoritmos.")
        elif retranscribe:
            self.status_kind = "retranscribe"
            detail = (f"Cambiaste parámetros del entorno o de los agentes: "
                      f"{describe_changes(retranscribe, analysis.config, state.config)}. Se recalculan los "
                      "problemas bandit (sin volver a ejecutar pYIN) y se transcribe de nuevo con los 4 algoritmos. "
                      "Los parámetros cambiados se marcan en naranja.")
            self.status_banner.set("warn", "Requiere re-transcribir", detail + busy_note,
                                   "Aplicar y re-transcribir", self.retranscribe, enabled=can_run,
                                   help_text="Recalcula solo la etapa 6 (brazos y recompensas) y transcribe de "
                                             "nuevo con los 4 algoritmos. Mucho más rápido que re-analizar.")
        else:
            self.status_kind = "synced"
            self.status_banner.set(
                "ok", "Configuración aplicada",
                f"La tablatura de «{name}» se calculó con estos parámetros. Los del experimento se aplican "
                "al ejecutarlo." + busy_note)

    def _update_validation(self) -> None:
        """Muestra u oculta el aviso rojo con los problemas de :meth:`Config.validate`."""
        if self.problems:
            self.validation_banner.set("error", "Configuración inválida",
                                       "\n".join(f"• {p}" for p in self.problems))
            self.validation_banner.pack(side="top", fill="x", pady=(6, 0))
        else:
            self.validation_banner.pack_forget()

    def _update_formula_values(self) -> None:
        """Actualiza las líneas «Ahora: ...» del panel de fórmulas."""
        cfg = self.app.state.config
        for label, fn in self._formula_values:
            try:
                label.configure(text=fn(cfg))
            except Exception:  # noqa: BLE001 - un valor raro no debe romper la vista
                label.configure(text="")

    def _update_experiment_info(self) -> None:
        """Duración estimada del experimento y vigencia de los resultados de «Comparación»."""
        state = self.app.state
        cfg = state.config
        analysis = state.analysis
        if analysis is None:
            try:
                per_note = estimate_experiment_seconds(1, cfg, with_lambda=True)
            except Exception:  # noqa: BLE001 - p. ej. una configuración inválida
                estimate = "Duración estimada: abre un audio para calcularla."
            else:
                # El coste crece linealmente con el número de notas (segmentos).
                estimate = (f"Duración estimada: ≈ {per_note['total']:.1f} s por nota con esta configuración "
                            f"(≈ {per_note['main']:.1f} s sin barridos). Abre un audio para calcular el total.")
        else:
            n_segments = len(analysis.segment_data)
            with_gt = analysis.ground_truth is not None
            try:
                est = estimate_experiment_seconds(n_segments, cfg, with_lambda=with_gt)
            except Exception:  # noqa: BLE001 - p. ej. una configuración inválida
                estimate = "Duración estimada: no disponible con la configuración actual."
            else:
                estimate = (f"Duración estimada con {n_segments} notas: ≈ {format_seconds(est['total'])} "
                            f"(principal {format_seconds(est['main'])}, barridos {format_seconds(est['sweeps'])}, "
                            f"λ {format_seconds(est['lambda'])}).")
                if cfg.experiment.run_lambda_sweep and not with_gt:
                    estimate += " El barrido de λ se omitirá: este audio no tiene ground truth (.gt.json)."
        self.estimate_label.configure(text=estimate)

        result = state.experiment
        if result is None:
            status = "Aún no hay resultados del experimento para este audio."
        else:
            used = getattr(result, "config", None)
            stale = used is None or any(getattr(used, s) != getattr(cfg, s) for s in ("env", "agent", "experiment"))
            status = ("Los resultados de «5 · Comparación» se calcularon con otra configuración: vuelve a ejecutar "
                      "el experimento para actualizarlos." if stale else
                      "Los resultados de «5 · Comparación» corresponden a esta configuración.")
        self.experiment_state_label.configure(text=status)
        enabled = analysis is not None and not self._busy and not self.problems
        self.run_experiment_button.configure(state="normal" if enabled else "disabled")
