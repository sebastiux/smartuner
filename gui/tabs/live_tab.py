"""Pestaña 3 · Ejecución en vivo: cómo «piensa» cada algoritmo bandit, pull a pull.

Papel en la interfaz
--------------------
Es la pestaña central para explicar en clase el aprendizaje por refuerzo del
proyecto. Toma UNA nota (un segmento = un problema Multi-Armed Bandit
independiente) y UN algoritmo (decisión de diseño 5) y ejecuta el bucle de
Sutton & Barto paso a paso con :class:`src.experiments.LiveSession`::

    elegir brazo (y explicar por qué) → el entorno devuelve r → Q(a) ← Q(a) + α·(r − Q(a))

Cada paso produce un :class:`src.experiments.PullEvent` con la explicación en
español que se muestra en el registro. La pestaña NO implementa nada del
algoritmo: solo llama a ``LiveSession.step`` / ``reset`` /
``recommended_arm`` y dibuja el estado del agente (``agent.q``,
``agent.counts``, ``agent.decision_scores()``) y del entorno
(``env.true_means``, ``env.optimal_arms``).

Contenido::

    ┌ Segmento ◀ [n.º 3 · 1.80 s · f0 55.0 Hz (A1) · 6 brazos] ▶  Algoritmo [UCB1]  Semilla [42]  Traste previo: 4 ┐
    ├ [Paso] [Auto ▶] [Pausa ⏸] [Detener ■] [Reiniciar ↺]   Velocidad ──●── 20 pulls/s        t = 37 / 500 ▓▓░░ ┤
    ├ Aviso (solo si cambió la configuración: «se aplicará al reiniciar»)                                      ┤
    ├ (a) Q estimado vs. μ real │ (b) Pulls por brazo        │ Estado del agente (recomendado, óptimo, regret)  ┤
    │ (c) Recompensa por pull   │ (d) Criterio de decisión   │ Registro de decisiones (un PullEvent por línea)  │
    └ leyenda común                                          │ ¿Qué muestra el panel (d)?                       ┘

Con la ventana estrecha (1024 px) la primera fila se parte en dos (semilla y
traste previo debajo), los títulos usan su versión corta, las etiquetas de
los brazos se giran y la leyenda usa menos columnas: nada queda cortado.

Semilla y reproducibilidad
--------------------------
La semilla de la pestaña sustituye a ``cfg.experiment.seed`` en la copia de la
configuración de la sesión, y la sesión usa el índice de corrida 0. Con la
semilla por defecto y el traste previo de la tablatura, la ejecución en vivo
reproduce EXACTAMENTE, pull a pull, la corrida que produjo la tablatura de la
pestaña 4 (mismos generadores; ver :func:`src.experiments.segment_rngs`).

Sincronización con las demás pestañas
-------------------------------------
* ``analysis_ready`` / ``transcriptions_ready`` → nueva sesión con el segmento
  seleccionado (o el 0). Ambos eventos llegan seguidos, así que se agrupan en
  una sola reconstrucción (``after_idle``).
* ``segment_selected`` de OTRA pestaña → detiene el modo automático y carga
  ese segmento. Las selecciones de esta pestaña se publican con
  ``source="live"`` y se ignoran al volver.
* ``algorithm_selected`` → cambia el algoritmo (y el combobox).
* ``config_changed`` → la sesión en curso NO cambia: aparece el aviso «se
  aplicará al reiniciar» (o «hay que re-transcribir» si cambiaron los brazos
  o la recompensa).
* ``busy_changed`` → pausa el modo automático mientras hay una tarea pesada.

Eficiencia
----------
Los artistas de matplotlib (barras, puntos, líneas) se crean UNA vez por
sesión y en cada paso solo se actualizan sus datos (``set_height``,
``set_data``). En modo automático se dan varios pulls por tick (la velocidad
se mide con el reloj, no por ticks), se redibuja como mucho a 25 fps con
``root.after`` y se usa *blitting*: el fondo estático (ejes, rejilla, textos)
se guarda tras un dibujado completo y en cada tick solo se repintan los
artistas que cambian. Solo hay un dibujado completo cuando cambia la escala de
un eje (escalones «redondos») o el tamaño de la ventana. El *constrained
layout* se calcula en esos dibujados completos y queda congelado en los
demás.
"""

from __future__ import annotations

import copy
import logging
import math
import time
import tkinter as tk
from collections.abc import Sequence
from dataclasses import fields
from tkinter import font as tkfont
from tkinter import ttk
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import DrawEvent, ResizeEvent
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, MaxNLocator

from gui.widgets import PlotFrame, Tooltip
from src import plots
from src.config import ALGO_COLORS, ALGO_LABELS, ALGORITHMS, PARAM_SPECS, Config
from src.experiments import ALGO_SHORT, SEGMENT_DATA_ENV_PARAMS, LiveSession, PullEvent
from src.pitch import hz_to_midi, midi_to_name

if TYPE_CHECKING:  # solo para anotaciones (evita importes circulares)
    from gui.app import SmartunerApp
    from src.agents import BanditAgent
    from src.environment import SegmentBanditData
    from src.pipeline import AnalysisResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Identificador de esta pestaña como origen de los eventos ``segment_selected``.
SOURCE = "live"

#: Velocidad del modo automático (pulls por segundo): mínimo, máximo y valor inicial.
MIN_SPEED = 1
MAX_SPEED = 200
DEFAULT_SPEED = 20

#: Intervalo mínimo entre redibujados del modo automático (ms): 40 ms = 25 fps.
FRAME_MS = 40

#: Pausa mínima (ms) entre el final de un tick y el siguiente (Tk atiende eventos y dibujados).
MIN_TICK_GAP_MS = 5

#: Tiempo máximo (s) que se «recupera» en un tick si el anterior se retrasó
#: (p. ej. por un dibujado completo): evita ráfagas enormes de pulls.
MAX_CATCHUP_S = 0.25

#: Líneas máximas del registro de decisiones (las más antiguas se descartan).
MAX_LOG_LINES = 500

#: Ancho pedido (px) del panel derecho (estado, registro y explicación).
SIDE_WIDTH = 380

#: Por debajo de este ancho de pestaña (px) el panel derecho se estrecha.
NARROW_WIDTH = 1180
SIDE_WIDTH_NARROW = 320

#: Separación del *constrained layout* (pulgadas y fracción) de la figura 2×2.
LAYOUT_PADS: dict[str, float] = {"h_pad": 0.05, "w_pad": 0.06, "hspace": 0.05, "wspace": 0.04}

#: Fracción del hueco de cada brazo que ocupa su barra.
BAR_WIDTH = 0.72

#: Opacidad de las barras de Q de los brazos aún sin probar (Q = Q₀).
UNPULLED_ALPHA = 0.35

#: Grosor (pt) del borde que resalta el brazo jalado en el último paso.
HIGHLIGHT_LW = 2.0

#: Rayado de los brazos sin probar en el índice UCB (+∞).
UNTRIED_HATCH = "////"

#: Títulos de los paneles fijos: versión normal y corta (paneles estrechos).
PANEL_TITLES: dict[str, tuple[str, str]] = {
    "a": ("(a) Q estimado por brazo vs. μ real", "(a) Q estimado vs. μ real"),
    "b": ("(b) Pulls por brazo", "(b) Pulls por brazo"),
    "c": ("(c) Recompensa por pull", "(c) Recompensa por pull"),
}

#: Tamaño (pt) de los textos de la leyenda común y del título de la figura.
LEGEND_FONT_SIZE = 8.5
SUPTITLE_SIZE = 11.5

#: Opciones de la leyenda común (en unidades de tamaño de fuente, como en matplotlib).
LEGEND_OPTS: dict[str, float] = {"handlelength": 2.0, "handletextpad": 0.8, "columnspacing": 1.4, "borderpad": 0.4}

#: Máximo de columnas de la leyenda común (con la ventana ancha cabe en dos filas).
LEGEND_MAX_COLS = 5

#: Escalones «redondos» (mantisas) para los límites de los ejes dinámicos:
#: así la escala cambia pocas veces y casi todos los ticks se pintan con *blitting*.
COUNT_MANTISSAS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
INDEX_MANTISSAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

#: Fondo y tinta del aviso de configuración pendiente (ámbar claro, como en la pestaña Audio).
BANNER_BG = "#fdf3d8"
BANNER_FG = "#5c4400"

#: Fondo de la línea más reciente del registro.
LATEST_BG = "#e8f0fb"

#: Texto corto de cada tipo de decisión (``Decision.kind``).
KIND_TEXT: dict[str, str] = {
    "init": "pull inicial",
    "explore": "explora",
    "exploit": "explota",
    "sample": "muestreo de π",
}

#: Mensaje guía antes de analizar un audio.
EMPTY_TEXT = ("Abre un MP3 (Archivo → Abrir) o genera el dataset sintético\n"
              "(Archivo → Generar dataset sintético…) para empezar.")
EMPTY_DETAIL = ("Aquí verás, pull a pull, cómo decide un algoritmo bandit en una nota: qué brazo "
                "(cuerda-traste) prueba, qué recompensa recibe, cómo cambian sus estimaciones Q y "
                "por qué elige lo que elige.")
BUSY_TEXT = "Procesando el audio… sigue el avance en la barra de estado (abajo)."
NO_SEGMENTS_TEXT = ("El análisis no conservó ninguna nota (todos los segmentos son silencio o "
                    "demasiado cortos).\nRevisa el umbral de silencio y la duración mínima en la "
                    "pestaña Configuración o abre otro audio.")

#: Texto inicial del registro.
LOG_HINT = ("Pulsa «Paso» para dar un pull o «Auto ▶» para verlos seguidos. Cada línea "
            "explica un pull: t | algoritmo → brazo | por qué lo eligió | frame muestreado: "
            "saliencia S − penalización = recompensa r | Q antes→después.")

#: Ayudas (tooltips) de los controles.
HELP: dict[str, str] = {
    "segment": ("Nota (segmento) que se resuelve como problema bandit. Formato: n.º entre las notas "
                "conservadas · inicio (s) · f0 estimada por pYIN (nota) · número de brazos candidatos "
                "(posiciones cuerda-traste a ±k semitonos). La selección se comparte con las demás pestañas."),
    "prev": "Nota anterior (◀).",
    "next": "Nota siguiente (▶).",
    "algorithm": ("Algoritmo que se ejecuta (uno a la vez). Cambiarlo inicia una sesión nueva en la misma "
                  "nota y se refleja en las demás pestañas."),
    "seed": ("Semilla de los números aleatorios de la sesión (frames muestreados y decisiones al azar del "
             "agente). Con la semilla del experimento (por defecto) la ejecución reproduce pull a pull la "
             "corrida que generó la tablatura (pestaña 4). Con la misma semilla, los cuatro algoritmos ven "
             "la misma secuencia de frames (números aleatorios comunes)."),
    "step": "Da UN pull: el agente elige un brazo, el entorno devuelve la recompensa y el agente actualiza Q.",
    "auto": "Da pulls seguidos a la velocidad elegida hasta agotar el presupuesto T (o hasta pulsar Pausa).",
    "pause": "Pausa el modo automático conservando el estado (Auto ▶ continúa desde aquí).",
    "stop": ("Detiene y vuelve a t = 0 con la MISMA sesión: misma configuración y semilla, así que la "
             "repetición es idéntica."),
    "restart": ("Crea una sesión nueva con la configuración ACTUAL (aplica los cambios pendientes de la "
                "pestaña Configuración), la semilla y el traste previo vigentes."),
    "speed": ("Velocidad del modo automático en pulls por segundo (escala logarítmica, 1–200). A más de "
              "25 pulls/s se dan varios pulls entre dos redibujados."),
    "t": "Pulls realizados / presupuesto T de la nota (env.budget).",
    "recommended": ("Posición que se escribiría en la tablatura si el presupuesto acabara ahora: el brazo "
                    "más jalado (desempate por Q) o el de mayor Q, según «Recomendación final»."),
    "optimal": ("Brazo(s) con la mayor recompensa media real μ. El entorno la conoce porque promedia la "
                "saliencia de TODOS los frames; el agente no: solo ve recompensas sueltas."),
    "last": "Brazo jalado en el último paso, su recompensa r y el tipo de decisión.",
    "regret": ("Regret acumulado = Σ_t (μ* − μ_{a_t}): lo que se pierde, en media, por no haber jalado "
               "siempre el brazo óptimo. Crece rápido mientras se explora y se aplana al converger."),
    "pct": "Porcentaje de los pulls realizados que fueron al brazo óptimo (curva «% acción óptima»).",
    "log": ("Registro de decisiones: una línea por pull con la explicación que genera el agente. La línea "
            "resaltada es la más reciente. Si subes para leer, el registro deja de desplazarse solo."),
}


# ---------------------------------------------------------------------------
# Funciones puras (sin Tk; se prueban sin pantalla)
# ---------------------------------------------------------------------------


def speed_from_scale(value: float) -> int:
    """Convierte la posición de la escala logarítmica en pulls por segundo.

    Parameters
    ----------
    value : float
        Posición de la escala: log₁₀(velocidad), entre 0 y log₁₀(200).

    Returns
    -------
    int
        Velocidad en pulls/s, redondeada y acotada a [1, 200].

    Examples
    --------
    >>> speed_from_scale(0.0), speed_from_scale(1.0), speed_from_scale(math.log10(200))
    (1, 10, 200)
    """
    speed = int(round(10.0 ** float(value)))
    return max(MIN_SPEED, min(MAX_SPEED, speed))


def scale_from_speed(speed: float) -> float:
    """Inversa de :func:`speed_from_scale`: posición de la escala para ``speed`` pulls/s.

    Parameters
    ----------
    speed : float
        Velocidad en pulls/s (se acota a [1, 200]).

    Returns
    -------
    float
        log₁₀(velocidad).

    Examples
    --------
    >>> scale_from_speed(100)
    2.0
    """
    return math.log10(max(MIN_SPEED, min(MAX_SPEED, float(speed))))


def tick_interval_ms(speed: float) -> int:
    """Intervalo entre ticks del modo automático.

    Hasta 25 pulls/s se da un pull por tick (intervalo 1000/velocidad ms); por
    encima, el intervalo queda en :data:`FRAME_MS` (25 fps) y en cada tick se
    dan varios pulls.

    Parameters
    ----------
    speed : float
        Velocidad en pulls/s.

    Returns
    -------
    int
        Milisegundos entre ticks.

    Examples
    --------
    >>> tick_interval_ms(1), tick_interval_ms(20), tick_interval_ms(200)
    (1000, 50, 40)
    """
    return max(FRAME_MS, int(round(1000.0 / max(float(speed), 1e-9))))


def nice_ceiling(value: float, mantissas: Sequence[float] = INDEX_MANTISSAS) -> float:
    """Menor número «redondo» m·10ᵏ (m en ``mantissas``) mayor o igual que ``value``.

    Parameters
    ----------
    value : float
        Valor a cubrir (> 0; los valores ≤ 0 devuelven la primera mantisa).
    mantissas : Sequence[float]
        Mantisas permitidas en [1, 10], en orden creciente.

    Returns
    -------
    float
        Límite redondo.

    Examples
    --------
    >>> nice_ceiling(37, COUNT_MANTISSAS), nice_ceiling(0.7), nice_ceiling(2.2)
    (50.0, 0.8, 3.0)
    """
    if not math.isfinite(value) or value <= 0:
        return float(mantissas[0])
    exponent = math.floor(math.log10(value))
    for k in (exponent - 1, exponent, exponent + 1):
        for m in mantissas:
            candidate = m * 10.0 ** k
            if candidate >= value * (1 - 1e-12):
                return float(round(candidate, 12))
    return float(mantissas[-1] * 10.0 ** (exponent + 1))


def count_axis_top(max_count: int, budget: int) -> float:
    """Límite superior del eje de pulls por brazo (escalones 5, 10, 20, 50, 100…, sin pasar de T).

    Parameters
    ----------
    max_count : int
        Mayor número de pulls de un brazo hasta ahora.
    budget : int
        Presupuesto T (ningún brazo puede superarlo).

    Returns
    -------
    float
        Límite del eje.

    Examples
    --------
    >>> count_axis_top(0, 500), count_axis_top(7, 500), count_axis_top(480, 500), count_axis_top(250, 300)
    (5.0, 10.0, 500.0, 300.0)
    """
    top = nice_ceiling(max(int(max_count), 5), COUNT_MANTISSAS)
    return float(min(top, max(int(budget), 1))) if max_count <= budget else top


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Media móvil causal: el punto t promedia los últimos ``window`` valores (o los que haya).

    Parameters
    ----------
    values : np.ndarray
        Serie, forma ``(n,)``.
    window : int
        Ventana W ≥ 1 (en pulls).

    Returns
    -------
    np.ndarray
        Forma ``(n,)``: mₜ = (1/min(t, W))·Σ_{i=t−W+1..t} xᵢ.

    Examples
    --------
    >>> moving_average(np.array([1.0, 3.0, 5.0, 7.0]), 2).tolist()
    [1.0, 2.0, 4.0, 6.0]
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x.copy()
    w = max(int(window), 1)
    csum = np.cumsum(x)
    out = np.empty_like(x)
    head = min(w, x.size)
    out[:head] = csum[:head] / np.arange(1, head + 1)
    if x.size > w:
        out[w:] = (csum[w:] - csum[:-w]) / w
    return out


def moving_window(budget: int) -> int:
    """Ventana de la media móvil de la recompensa: T/25 pulls, entre 5 y 50.

    Examples
    --------
    >>> moving_window(500), moving_window(50), moving_window(5000)
    (20, 5, 50)
    """
    return int(max(5, min(50, int(budget) // 25)))


def note_name(f0_hz: float | None) -> str:
    """Nombre de la nota más cercana a ``f0_hz`` (``"—"`` si es desconocida).

    Examples
    --------
    >>> note_name(55.0), note_name(None)
    ('A1', '—')
    """
    if f0_hz is None or not math.isfinite(f0_hz) or f0_hz <= 0:
        return "—"
    return midi_to_name(int(round(hz_to_midi(float(f0_hz)))))


def segment_label(data: SegmentBanditData) -> str:
    """Texto del selector de segmento: ``"n.º 3 · 1.80 s · f0 55.0 Hz (A1) · 6 brazos"``.

    Parameters
    ----------
    data : SegmentBanditData
        Datos bandit del segmento.

    Returns
    -------
    str
        Posición, inicio (s), f0 (Hz y nota) y número de brazos.
    """
    seg = data.segment
    f0 = seg.f0_hz
    f0_text = f"f0 {f0:.1f} Hz ({note_name(f0)})" if f0 else "f0 desconocida"
    arms = "1 brazo" if data.n_arms == 1 else f"{data.n_arms} brazos"
    return f"n.º {data.position} · {seg.start_s:.2f} s · {f0_text} · {arms}"


def reward_range(session: LiveSession) -> tuple[float, float]:
    """Rango de recompensas POSIBLES de la sesión (para fijar los ejes de valor).

    Cada recompensa es r = S_j(f_a) − pen_a (+ ruido) para algún frame j y
    brazo a, así que todas caen entre el mínimo y el máximo de esa matriz (con
    ruido se añaden ±3σ). Las estimaciones Q son promedios de recompensas (y
    de Q₀), de modo que tampoco salen de ese rango ampliado con Q₀: los ejes
    de valor se pueden fijar una vez por sesión.

    Parameters
    ----------
    session : LiveSession
        Sesión en curso.

    Returns
    -------
    tuple[float, float]
        ``(r_min, r_max)``.
    """
    rewards = session.data.salience - session.env.penalties[np.newaxis, :]
    lo, hi = float(np.min(rewards)), float(np.max(rewards))
    noise = float(session.env.noise_std)
    if noise > 0:
        lo, hi = lo - 3.0 * noise, hi + 3.0 * noise
    return lo, hi


def value_limits(r_min: float, r_max: float, q0: float) -> tuple[float, float]:
    """Límites del eje de valores (Q, μ, r): rango de recompensas ∪ {0, Q₀} con margen.

    Examples
    --------
    >>> [round(v, 3) for v in value_limits(0.1, 0.9, 0.0)]
    [0.0, 0.954]
    >>> [round(v, 3) for v in value_limits(-0.2, 0.9, 2.0)]       # Q₀ = 2 (optimista) entra en el eje
    [-0.332, 2.132]
    """
    lo = min(0.0, r_min, q0)
    hi = max(r_max, q0, lo + 0.1)
    pad = 0.06 * (hi - lo)
    return (lo - pad if lo < 0 else 0.0), hi + pad


def criterion_title(algorithm: str, agent: BanditAgent, compact: bool = False) -> str:
    """Título del panel (d) con el hiperparámetro VIGENTE (ε o τ pueden decaer con t).

    Parameters
    ----------
    algorithm : str
        Identificador del algoritmo.
    agent : BanditAgent
        Agente en ejecución (se leen ``epsilon``, ``q0``, ``c`` o ``tau``).
    compact : bool, optional
        Versión corta (sin la fórmula) para paneles estrechos (ventana de 1024 px).

    Returns
    -------
    str
        P. ej. ``"(d) Criterio: Q estimado   (ε = 0.1)"``.

    Examples
    --------
    >>> from src.agents import UCB1Agent
    >>> criterion_title("ucb1", UCB1Agent(n_arms=3, c=1.414))
    '(d) Índice UCB = Q + c·√(ln t / n)   (c = 1.41)'
    >>> criterion_title("ucb1", UCB1Agent(n_arms=3, c=1.414), compact=True)
    '(d) Índice UCB   (c = 1.41)'
    """
    if algorithm == "ucb1":
        c = float(getattr(agent, "c", 0.0))
        return f"(d) Índice UCB   (c = {c:.2f})" if compact else f"(d) Índice UCB = Q + c·√(ln t / n)   (c = {c:.2f})"
    if algorithm == "softmax":
        tau = float(getattr(agent, "tau", 0.0))
        return f"(d) Probabilidad π(a)   (τ = {tau:.3g})" if compact else \
            f"(d) Probabilidad π(a) de cada brazo   (τ = {tau:.3g})"
    eps = float(getattr(agent, "epsilon", 0.0))
    if algorithm == "optimistic":
        if compact:
            return f"(d) Q estimado   (ε = {eps:.2g}, Q₀ = {agent.q0:g})"
        return f"(d) Criterio: Q estimado, greedy   (ε = {eps:.2g}, Q₀ = {agent.q0:g})"
    return f"(d) Q estimado   (ε = {eps:.3g})" if compact else f"(d) Criterio: Q estimado   (ε = {eps:.3g})"


def criterion_ylabel(algorithm: str) -> str:
    """Etiqueta (con unidades) del eje Y del panel (d)."""
    if algorithm == "ucb1":
        return "índice (recompensa)"
    if algorithm == "softmax":
        return "probabilidad π(a)"
    return "Q (recompensa)"


def criterion_explanation(algorithm: str, cfg: Config) -> str:
    """Explicación breve (2–3 líneas) de qué muestra el panel (d) para ``algorithm``.

    Parameters
    ----------
    algorithm : str
        Identificador del algoritmo.
    cfg : Config
        Configuración de la sesión (para citar sus hiperparámetros).

    Returns
    -------
    str
        Texto en español para el recuadro inferior del panel derecho.
    """
    a = cfg.agent
    if algorithm == "egreedy":
        decay = "" if a.epsilon_decay == 1.0 else f" (decae ×{a.epsilon_decay:g} por pull hasta {a.epsilon_min:g})"
        return (f"ε-greedy decide con Q: con probabilidad 1 − ε elige la barra más alta (explota) y con "
                f"probabilidad ε = {a.epsilon:g}{decay} un brazo cualquiera al azar (explora), sin mirar "
                "su valor.")
    if algorithm == "optimistic":
        return (f"El optimista empieza con Q₀ = {a.q0:g}, más que cualquier recompensa, y elige siempre la "
                f"Q más alta. Cada brazo probado «decepciona» y su Q baja (α = {a.optimistic_alpha:g}), "
                "así que recorre todos los brazos al principio sin necesidad de azar.")
    if algorithm == "ucb1":
        return (f"UCB1 elige la barra más alta del índice Q + c·√(ln t / n) con c = {a.ucb_c:g}. El bono es "
                "grande en los brazos poco probados (incertidumbre); los no probados valen +∞ (barra "
                "rayada) y se prueban primero.")
    if algorithm == "softmax":
        decay = "" if a.tau_decay == 1.0 else f", que se enfría ×{a.tau_decay:g} por pull"
        return (f"Softmax sortea el brazo con probabilidad π(a) ∝ e^(Q(a)/τ) (τ = {a.tau:g}{decay}): los "
                "brazos buenos salen a menudo y los malos casi nunca. τ alta → casi uniforme; τ baja → "
                "casi greedy.")
    return ""


def data_param_keys(cfg: Config) -> list[str]:
    """Parámetros que cambian los DATOS bandit (brazos, frames, saliencia) del análisis.

    Si alguno cambia hay que recalcular la etapa 6 (Ejecutar → Re-transcribir):
    no basta con reiniciar la sesión.

    Parameters
    ----------
    cfg : Config
        Configuración vigente (decide qué parámetros del espectro aplican).

    Returns
    -------
    list[str]
        Claves ``"seccion.campo"``.
    """
    keys = [f"env.{name}" for name in SEGMENT_DATA_ENV_PARAMS]
    spectrum = ("bins_per_octave", "cqt_fmin_hz", "n_octaves") if str(cfg.env.spectrum).lower() == "cqt" else ("n_fft",)
    keys += [f"env.{name}" for name in spectrum if f"env.{name}" not in keys]
    keys.append("pitch.min_voiced_ratio")
    return keys


def pending_changes(session_cfg: Config, current: Config, analysis_cfg: Config | None) -> tuple[list[str], list[str]]:
    """Cambios de configuración que la sesión en curso todavía no usa.

    Parameters
    ----------
    session_cfg : Config
        Configuración con la que se creó la sesión.
    current : Config
        Configuración vigente de la aplicación.
    analysis_cfg : Config | None
        Configuración con la que se calcularon los datos bandit del análisis.

    Returns
    -------
    restart : list[str]
        Parámetros de entorno/agente que se aplicarán al reiniciar la sesión.
    retranscribe : list[str]
        Parámetros que exigen recalcular los datos bandit (re-transcribir).

    Examples
    --------
    >>> old, new = Config(), Config()
    >>> new.env.lam = 0.3; new.env.k_semitones = 3
    >>> pending_changes(old, new, old)
    (['env.lam'], ['env.k_semitones'])
    """
    data_keys = data_param_keys(current)
    retranscribe = []
    if analysis_cfg is not None:
        retranscribe = [k for k in data_keys if analysis_cfg.get(k) != current.get(k)]
    restart = []
    for section in ("env", "agent"):
        for f in fields(getattr(current, section)):
            key = f"{section}.{f.name}"
            if key not in data_keys and session_cfg.get(key) != current.get(key):
                restart.append(key)
    return restart, retranscribe


def format_change(key: str, old: Any, new: Any) -> str:
    """Describe un cambio ``"etiqueta (antes → después)"`` con la etiqueta de PARAM_SPECS.

    Examples
    --------
    >>> format_change("env.budget", 500, 300)
    'T pulls por segmento (500 → 300)'
    """
    spec = PARAM_SPECS.get(key)
    label = spec.label if spec is not None else key

    def fmt(v: Any) -> str:
        """Valor legible: «vacío», «sí»/«no» o el número sin ceros sobrantes."""
        if v is None:
            return "vacío"
        if isinstance(v, bool):
            return "sí" if v else "no"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    return f"{label} ({fmt(old)} → {fmt(new)})"


def legend_columns(label_widths: Sequence[float], available_px: float, font_px: float,
                   max_cols: int = LEGEND_MAX_COLS) -> int:
    """Mayor número de columnas (≤ ``max_cols``) con el que la leyenda cabe en ``available_px``.

    Reproduce cómo reparte matplotlib las entradas (``np.array_split``: las
    primeras columnas reciben una entrada más) y suma, por columna, el ancho
    de su texto más largo + muestra + separaciones (:data:`LEGEND_OPTS`).

    Parameters
    ----------
    label_widths : Sequence[float]
        Ancho (px) del texto de cada entrada, en orden.
    available_px : float
        Ancho disponible (px), normalmente el de la figura.
    font_px : float
        Tamaño de la fuente de la leyenda en píxeles.
    max_cols : int, optional
        Máximo de columnas.

    Returns
    -------
    int
        Columnas (al menos 1).

    Examples
    --------
    >>> legend_columns([100.0] * 8, 1000.0, 12.0), legend_columns([100.0] * 8, 500.0, 12.0)
    (5, 3)
    """
    widths = np.asarray(label_widths, dtype=float)
    if widths.size == 0:
        return 1
    per_entry = (LEGEND_OPTS["handlelength"] + LEGEND_OPTS["handletextpad"]) * font_px
    for ncol in range(min(max_cols, widths.size), 1, -1):
        columns = [c for c in np.array_split(widths, ncol) if c.size]
        total = sum(float(c.max()) + per_entry for c in columns)
        total += (len(columns) - 1) * LEGEND_OPTS["columnspacing"] * font_px + 2 * LEGEND_OPTS["borderpad"] * font_px
        if total <= available_px:
            return ncol
    return 1


def panel_title(ax: Axes, text: str) -> Text:
    """Pone el título de un panel con el estilo de :mod:`src.plots` y DEVUELVE el texto.

    Es el título de la izquierda (``loc="left"``), como en ``src.plots``; se
    guarda la referencia para cambiarlo después (el ε o la τ vigentes del
    panel (d)) sin crear un segundo título encima.

    Parameters
    ----------
    ax : Axes
        Panel.
    text : str
        Título.

    Returns
    -------
    Text
        El artista del título.
    """
    return ax.set_title(text, loc="left", fontsize=plots.PANEL_TITLE_SIZE, color=plots.TEXT_PRIMARY,
                        fontweight="bold", pad=6)


def arm_tick_labels(data: SegmentBanditData, optimal: Sequence[int]) -> list[str]:
    """Etiquetas del eje x de los paneles de barras: ``"A-0"`` y ``"A-0★"`` para los óptimos.

    Parameters
    ----------
    data : SegmentBanditData
        Segmento (da las etiquetas de los brazos).
    optimal : Sequence[int]
        Índices de los brazos óptimos (mayor μ).

    Returns
    -------
    list[str]
        Una etiqueta por brazo.
    """
    best = {int(i) for i in optimal}
    return [arm.label + ("★" if i in best else "") for i, arm in enumerate(data.arms)]


# ---------------------------------------------------------------------------
# Figura 2×2 con artistas persistentes
# ---------------------------------------------------------------------------


class LivePlot:
    """Figura 2×2 de la ejecución en vivo; crea los artistas una vez y luego solo actualiza datos.

    Paneles:

    * (a) Q estimado por brazo (barras del color del algoritmo) y μ real
      (rombos negros). Brazos sin probar en tono claro; el brazo jalado en el
      último paso con borde grueso; óptimos con ★ en la etiqueta.
    * (b) Pulls por brazo n_a.
    * (c) Recompensa de cada pull (relleno de color = pull al brazo óptimo;
      gris = otro brazo), media móvil y μ* (línea discontinua).
    * (d) Criterio de decisión del agente (``agent.decision_scores()``): Q
      para ε-greedy y optimista, índice UCB (los +∞ de brazos sin probar se
      dibujan rayados hasta el borde) o probabilidades π de Softmax (eje 0–1).

    Parameters
    ----------
    figure : Figure
        Figura embebida (la de :class:`gui.widgets.PlotFrame`).

    Attributes
    ----------
    algorithm : str
        Algoritmo de la sesión dibujada.
    q_bars, count_bars, score_bars : list[Rectangle]
        Barras de los paneles (a), (b) y (d), una por brazo.
    """

    def __init__(self, figure: Figure) -> None:
        """Guarda la figura; los ejes y artistas se crean en :meth:`build` (uno por sesión)."""
        self.figure = figure
        self.algorithm = ""
        self.color = plots.TEXT_SECONDARY
        self.n_arms = 0
        self.budget = 1
        self.window = 20
        self.ax_q: Any = None
        self.ax_n: Any = None
        self.ax_r: Any = None
        self.ax_d: Any = None
        self.q_bars: list[Rectangle] = []
        self.count_bars: list[Rectangle] = []
        self.score_bars: list[Rectangle] = []
        self.inf_texts: list[Text] = []
        self.mu_markers: Line2D | None = None
        self.points_optimal: Line2D | None = None
        self.points_other: Line2D | None = None
        self.average_line: Line2D | None = None
        self.titles: dict[str, Text] = {}
        self.suptitle: Text | None = None
        self.suptitle_texts: tuple[str, str] = ("", "")
        self.legend: Any = None
        self.legend_ncol = LEGEND_MAX_COLS
        self.compact_d = False
        self.count_top = 0.0
        self.index_top = 0.0
        self.value_lims = (0.0, 1.0)
        self.has_session = False
        self._agent: BanditAgent | None = None
        self._handles: list[Artist] = []

    # ----------------------------------------------------------- construcción
    def show_message(self, text: str) -> None:
        """Limpia la figura y muestra ``text`` centrado (sin sesión que dibujar).

        Parameters
        ----------
        text : str
            Mensaje en español.
        """
        fig = self.figure
        fig.clear()
        fig.set_layout_engine("none")
        fig.set_facecolor(plots.SURFACE)
        fig.text(0.5, 0.5, text, ha="center", va="center", color=plots.TEXT_SECONDARY, fontsize=11, wrap=True)
        self.has_session = False
        self.q_bars, self.count_bars, self.score_bars, self.inf_texts = [], [], [], []
        self.titles, self.suptitle, self.legend, self._agent = {}, None, None, None

    def build(self, session: LiveSession, title: str, short_title: str | None = None) -> None:
        """Crea ejes y artistas para ``session`` (una vez por sesión).

        Parameters
        ----------
        session : LiveSession
            Sesión recién creada (t = 0).
        title : str
            Título de la figura (contexto: algoritmo, segmento, semilla…).
        short_title : str | None
            Versión corta del título para figuras estrechas (None = la misma).
        """
        fig = self.figure
        fig.clear()
        fig.set_facecolor(plots.SURFACE)
        fig.set_layout_engine("constrained", **LAYOUT_PADS)
        alg = session.algorithm
        self.algorithm = alg
        self.color = ALGO_COLORS.get(alg, plots.TEXT_SECONDARY)
        data, env, agent = session.data, session.env, session.agent
        self._agent = agent
        k = data.n_arms
        self.n_arms = k
        self.budget = int(session.budget)
        self.window = moving_window(self.budget)
        x = np.arange(k)
        r_min, r_max = reward_range(session)
        self.value_lims = value_limits(r_min, r_max, float(agent.q0))

        axes = fig.subplots(2, 2)
        self.ax_q, self.ax_n, self.ax_r, self.ax_d = axes.flat
        zeros = np.zeros(k)

        # (a) Q estimado (barras) y μ real (rombos, dibujados ENCIMA de las barras).
        ax = self.ax_q
        if alg == "optimistic":
            ax.axhline(float(agent.q0), color=plots.TEXT_SECONDARY, linewidth=1.0, linestyle=":", zorder=1)
        self.q_bars = list(ax.bar(x, zeros, width=BAR_WIDTH, color=self.color, linewidth=0, zorder=2))
        (self.mu_markers,) = ax.plot(x, env.true_means, linestyle="none", marker="D", markersize=6.0,
                                     color=plots.TEXT_PRIMARY, markeredgecolor=plots.SURFACE,
                                     markeredgewidth=0.8, zorder=4)
        ax.set_ylim(*self.value_lims)
        self.titles["a"] = panel_title(ax, PANEL_TITLES["a"][0])
        plots._axis_labels(ax, ylabel="valor (recompensa)")

        # (b) Pulls por brazo.
        ax = self.ax_n
        self.count_bars = list(ax.bar(x, zeros, width=BAR_WIDTH, color=self.color, linewidth=0, zorder=2))
        self.count_top = count_axis_top(0, self.budget)
        ax.set_ylim(0, self.count_top * 1.05)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        self.titles["b"] = panel_title(ax, PANEL_TITLES["b"][0])
        plots._axis_labels(ax, ylabel="pulls (n)")

        # (c) Recompensa por pull: puntos, media móvil y μ*.
        ax = self.ax_r
        ax.axhline(float(env.best_mean), color=plots.TEXT_SECONDARY, linewidth=1.2, linestyle="--", zorder=1)
        (self.points_other,) = ax.plot([], [], linestyle="none", marker="o", markersize=3.0,
                                       color=plots.TEXT_MUTED, alpha=0.6, markeredgewidth=0, zorder=2)
        (self.points_optimal,) = ax.plot([], [], linestyle="none", marker="o", markersize=3.4,
                                         color=self.color, markeredgewidth=0, zorder=3)
        (self.average_line,) = ax.plot([], [], color=plots.TEXT_PRIMARY, linewidth=1.8, zorder=4)
        ax.set_xlim(0, self.budget)
        r_lo, r_hi = min(0.0, r_min), r_max
        pad = 0.06 * max(r_hi - r_lo, 0.1)
        ax.set_ylim(r_lo - (pad if r_lo < 0 else 0.0), r_hi + pad)
        self.titles["c"] = panel_title(ax, PANEL_TITLES["c"][0])
        plots._axis_labels(ax, xlabel="pull t", ylabel="recompensa r")

        # (d) Criterio de decisión.
        ax = self.ax_d
        self.score_bars = list(ax.bar(x, zeros, width=BAR_WIDTH, color=self.color, linewidth=0, zorder=2))
        if alg == "softmax":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
        elif alg == "ucb1":
            self.index_top = nice_ceiling(max(self.value_lims[1], 1.0))
            ax.set_ylim(self.value_lims[0], self.index_top)
        else:
            ax.set_ylim(*self.value_lims)
        # «∞» sobre las barras rayadas de UCB1 (x en datos, y en fracción del eje: no
        # dependen de la escala). Se muestran solo mientras el brazo no se ha probado.
        self.inf_texts = []
        if alg == "ucb1":
            self.inf_texts = [
                ax.text(i, 0.985, "∞", transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=10, fontweight="bold", color=self.color, zorder=5, visible=False,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": plots.SURFACE, "edgecolor": "none"})
                for i in range(k)
            ]
        self.compact_d = False
        self.titles["d"] = panel_title(ax, criterion_title(alg, agent))
        plots._axis_labels(ax, xlabel="brazo (cuerda-traste)", ylabel=criterion_ylabel(alg))

        for bar_ax in (self.ax_q, self.ax_n, self.ax_d):
            bar_ax.set_xlim(-0.6, k - 0.4)
        plots.apply_style(fig, grid="y")
        self.ax_r.grid(True, axis="both", color=plots.GRID, linewidth=plots.GRID_WIDTH)
        self.suptitle_texts = (title, short_title or title)
        self.suptitle = fig.suptitle(title, x=0.01, ha="left", fontsize=SUPTITLE_SIZE, fontweight="bold",
                                     color=plots.TEXT_PRIMARY)
        self._handles = self._legend_handles(session)
        self.legend = None
        self._set_arm_ticks(arm_tick_labels(data, env.optimal_arms), env.optimal_arms)   # crea la leyenda
        self.has_session = True
        self.update(session)

    def _legend_handles(self, session: LiveSession) -> list[Artist]:
        """Muestras de la leyenda común (en el orden de los paneles).

        Parameters
        ----------
        session : LiveSession
            Sesión dibujada (da Q₀ del optimista).

        Returns
        -------
        list[Artist]
            Muestras con su etiqueta (``get_label()``).
        """
        alg = self.algorithm
        handles: list[Artist] = [
            Patch(facecolor=self.color, edgecolor="none", label=f"{ALGO_LABELS.get(alg, alg)}"),
            Patch(facecolor="none", edgecolor=plots.TEXT_PRIMARY, linewidth=HIGHLIGHT_LW, label="último brazo jalado"),
            Line2D([], [], linestyle="none", marker="D", markersize=6, color=plots.TEXT_PRIMARY,
                   label="μ real (oculto al agente)"),
            Line2D([], [], linestyle="none", marker="*", markersize=9, color=plots.TEXT_PRIMARY,
                   label="★ brazo óptimo (mayor μ)"),
        ]
        if alg == "optimistic":
            handles.append(Line2D([], [], color=plots.TEXT_SECONDARY, linewidth=1.0, linestyle=":",
                                  label=f"Q₀ = {session.agent.q0:g} (barra clara: sin probar)"))
        if alg == "ucb1":
            handles.append(Patch(facecolor=plots.SURFACE, edgecolor=self.color, hatch=UNTRIED_HATCH,
                                 linewidth=0.8, label="∞ sin probar: índice +∞"))
        handles += [
            Line2D([], [], linestyle="none", marker="o", markersize=5, color=self.color, markeredgewidth=0,
                   label="r de un pull al óptimo"),
            Line2D([], [], linestyle="none", marker="o", markersize=5, color=plots.TEXT_MUTED, alpha=0.6,
                   markeredgewidth=0, label="r de otro brazo"),
            Line2D([], [], color=plots.TEXT_PRIMARY, linewidth=1.8, label=f"media móvil ({self.window} pulls)"),
            Line2D([], [], color=plots.TEXT_SECONDARY, linewidth=1.2, linestyle="--", label="μ* (mejor media real)"),
        ]
        return handles

    def _set_arm_ticks(self, labels: list[str], optimal: Sequence[int]) -> None:
        """Etiquetas de brazo en los paneles de barras (★ y negrita en los óptimos).

        Parameters
        ----------
        labels : list[str]
            Una etiqueta por brazo (ver :func:`arm_tick_labels`).
        optimal : Sequence[int]
            Índices de los brazos óptimos.
        """
        best = {int(i) for i in optimal}
        x = np.arange(len(labels))
        for ax in (self.ax_q, self.ax_n, self.ax_d):
            ax.set_xticks(x, labels)
            for i, text in enumerate(ax.get_xticklabels()):
                if i in best:
                    text.set_color(plots.TEXT_PRIMARY)
                    text.set_fontweight("bold")
        self.apply_responsive_layout()

    # ----------------------------------------------------------- adaptación al tamaño
    def _text_width(self, text: str, size: float, bold: bool = False) -> float:
        """Ancho (px) que ocupará ``text`` con la fuente de las gráficas.

        Parameters
        ----------
        text : str
            Texto a medir.
        size : float
            Tamaño de fuente (pt).
        bold : bool
            Negrita.

        Returns
        -------
        float
            Ancho en píxeles de la figura (mide el *renderer* Agg real).
        """
        renderer = self.figure.canvas.get_renderer()
        prop = FontProperties(size=size, weight="bold" if bold else "normal")
        width, _height, _descent = renderer.get_text_width_height_descent(text, prop, ismath=False)
        return float(width)

    def apply_responsive_layout(self) -> None:
        """Adapta textos y leyenda al tamaño ACTUAL de la figura (al crearla y al redimensionar).

        * Etiquetas de brazo: horizontales si caben; si no, giradas 45° o 90°.
        * Títulos de los paneles y de la figura: versión corta si la larga no cabe.
        * Leyenda común: tantas columnas como quepan (máx. :data:`LEGEND_MAX_COLS`).
        """
        if self.ax_q is None or not self.q_bars:
            return
        fig = self.figure
        fig_px = float(fig.get_figwidth() * fig.dpi)
        panel_px = 0.5 * fig_px - 75.0                           # ancho aproximado de un panel
        # 1. Etiquetas de los brazos.
        longest = max((len(t.get_text()) for t in self.ax_q.get_xticklabels()), default=3)
        needed = self.n_arms * (longest * 6.2 + 6)               # ≈ 6 px por carácter a 8 pt
        if needed <= panel_px:
            rotation, ha, size = 0.0, "center", 8.5
        elif self.n_arms <= 24:
            rotation, ha, size = 45.0, "right", 8.0
        else:
            rotation, ha, size = 90.0, "center", 7.0
        for ax in (self.ax_q, self.ax_n, self.ax_d):
            for text in ax.get_xticklabels():
                text.set_rotation(rotation)
                text.set_horizontalalignment(ha)
                text.set_rotation_mode("anchor" if rotation == 45.0 else "default")
                text.set_fontsize(size)
        # 2. Títulos: la versión larga solo si cabe en el panel (y la figura).
        title_size = plots.PANEL_TITLE_SIZE
        for key, (long_text, short_text) in PANEL_TITLES.items():
            if key in self.titles:
                fits = self._text_width(long_text, title_size, bold=True) <= panel_px + 10
                self.titles[key].set_text(long_text if fits else short_text)
        if self._agent is not None and "d" in self.titles:
            long_text = criterion_title(self.algorithm, self._agent)
            self.compact_d = self._text_width(long_text, title_size, bold=True) > panel_px + 10
            self.titles["d"].set_text(criterion_title(self.algorithm, self._agent, compact=self.compact_d))
        if self.suptitle is not None:
            long_text, short_text = self.suptitle_texts
            fits = self._text_width(long_text, SUPTITLE_SIZE, bold=True) <= fig_px - 24
            self.suptitle.set_text(long_text if fits else short_text)
        # 3. Leyenda: se rehace solo si cambia el número de columnas.
        font_px = LEGEND_FONT_SIZE * fig.dpi / 72.0
        widths = [self._text_width(h.get_label(), LEGEND_FONT_SIZE) for h in self._handles]
        ncol = legend_columns(widths, fig_px - 16.0, font_px)
        if self.legend is None or ncol != self.legend_ncol:
            if self.legend is not None:
                self.legend.remove()
            self.legend_ncol = ncol
            self.legend = plots._legend(fig, self._handles, loc="outside lower center", ncol=ncol,
                                        fontsize=LEGEND_FONT_SIZE, **LEGEND_OPTS)

    # ----------------------------------------------------------- actualización
    def _bar_style(self, rect: Rectangle, highlighted: bool) -> None:
        """Estilo normal de una barra (relleno del algoritmo; borde grueso si es el último pull)."""
        rect.set_hatch(None)
        rect.set_facecolor(self.color)
        if highlighted:
            rect.set_edgecolor(plots.TEXT_PRIMARY)
            rect.set_linewidth(HIGHLIGHT_LW)
        else:
            rect.set_edgecolor("none")
            rect.set_linewidth(0.0)

    def update(self, session: LiveSession, allow_shrink: bool = True) -> bool:
        """Copia el estado de la sesión a los artistas (sin dibujar).

        Parameters
        ----------
        session : LiveSession
            Sesión cuyo estado se muestra.
        allow_shrink : bool, optional
            Si es False, la escala del índice UCB solo puede crecer (en modo
            automático: cada cambio de escala exige un dibujado completo).

        Returns
        -------
        bool
            True si cambió la escala de algún eje (hace falta un dibujado
            completo, no basta con repintar los artistas).
        """
        if not self.has_session:
            return False
        agent = session.agent
        q, counts = agent.q, agent.counts
        events = session.events
        last = events[-1].arm_index if events else -1
        changed = False

        # (a) Q estimado: claro mientras el brazo no se ha probado (Q = Q₀).
        for i, rect in enumerate(self.q_bars):
            rect.set_height(float(q[i]))
            rect.set_alpha(1.0 if counts[i] > 0 else UNPULLED_ALPHA)
            self._bar_style(rect, i == last)

        # (b) Pulls por brazo, con escala en escalones redondos.
        for i, rect in enumerate(self.count_bars):
            rect.set_height(float(counts[i]))
            self._bar_style(rect, i == last)
        top = count_axis_top(int(counts.max()) if counts.size else 0, self.budget)
        if top > self.count_top:            # solo crece: sin saltos de escala hacia atrás
            self.set_count_top(top)
            changed = True

        # (c) Recompensas: relleno de color si el pull fue al brazo óptimo.
        n = len(events)
        if n:
            t = np.arange(1, n + 1)
            r = np.fromiter((e.reward for e in events), dtype=float, count=n)
            opt = np.fromiter((e.is_optimal for e in events), dtype=bool, count=n)
            self.points_optimal.set_data(t[opt], r[opt])
            self.points_other.set_data(t[~opt], r[~opt])
            self.average_line.set_data(t, moving_average(r, self.window))
        else:
            for line in (self.points_optimal, self.points_other, self.average_line):
                line.set_data([], [])

        # (d) Criterio de decisión vigente (el que usará el agente en el PRÓXIMO pull).
        scores, _name = agent.decision_scores()
        scores = np.asarray(scores, dtype=float)
        if self.algorithm == "ucb1":
            finite = scores[np.isfinite(scores)]
            need = max(float(finite.max()) if finite.size else 0.0, self.value_lims[1])
            if need > self.index_top or (allow_shrink and need < 0.45 * self.index_top):
                new_top = nice_ceiling(need * 1.05)
                if new_top != self.index_top:
                    self.index_top = new_top
                    self.ax_d.set_ylim(self.value_lims[0], new_top)
                    changed = True
        d_top = self.ax_d.get_ylim()[1]
        for i, rect in enumerate(self.score_bars):
            value = float(scores[i])
            untried = math.isinf(value) and value > 0
            if untried:
                # Brazo sin probar en UCB1: índice +∞ → barra rayada hasta el borde.
                rect.set_height(d_top)
                rect.set_facecolor(plots.SURFACE)
                rect.set_edgecolor(self.color)
                rect.set_hatch(UNTRIED_HATCH)
                rect.set_linewidth(0.8)
            else:
                rect.set_height(value if math.isfinite(value) else 0.0)
                self._bar_style(rect, i == last)
            if i < len(self.inf_texts):
                self.inf_texts[i].set_visible(untried)
        if "d" in self.titles:
            self.titles["d"].set_text(criterion_title(self.algorithm, agent, compact=self.compact_d))
        return changed

    def set_count_top(self, top: float) -> None:
        """Fija el límite superior del eje de pulls (panel b).

        Parameters
        ----------
        top : float
            Límite en pulls (se deja un 5 % de aire encima).
        """
        self.count_top = float(top)
        self.ax_n.set_ylim(0, self.count_top * 1.05)

    def reset_count_axis(self) -> None:
        """Vuelve el eje de pulls a su escala inicial (al detener la sesión y volver a t = 0)."""
        if self.has_session:
            self.set_count_top(count_axis_top(0, self.budget))

    # ----------------------------------------------------------------- blitting
    def dynamic_artists(self) -> list[Artist]:
        """Artistas que cambian en cada pull, en orden de dibujo.

        Returns
        -------
        list[Artist]
            Barras, rombos de μ (encima de las barras), puntos, media móvil,
            marcas «∞» de UCB1 y el título del panel (d) (ε o τ pueden decaer).
        """
        if not self.has_session:
            return []
        artists: list[Artist] = [*self.q_bars, self.mu_markers, *self.count_bars, self.points_other,
                                 self.points_optimal, self.average_line, *self.score_bars, *self.inf_texts,
                                 self.titles.get("d")]
        return [a for a in artists if a is not None]

    def set_animated(self, animated: bool) -> None:
        """Marca los artistas dinámicos como animados (fuera del fondo estático) o no.

        Parameters
        ----------
        animated : bool
            True durante el modo automático (*blitting*); False en reposo, para
            que un dibujado normal (o «guardar figura») los incluya.
        """
        for artist in self.dynamic_artists():
            artist.set_animated(animated)

    def draw_animated(self) -> None:
        """Pinta los artistas dinámicos sobre el fondo restaurado (solo en *blitting*)."""
        for artist in self.dynamic_artists():
            artist.axes.draw_artist(artist)


# ---------------------------------------------------------------------------
# Pestaña
# ---------------------------------------------------------------------------


class LiveTab(ttk.Frame):
    """Pestaña «3 · Ejecución en vivo»: un algoritmo, una nota, pull a pull.

    Parameters
    ----------
    master : tk.Misc
        El ``ttk.Notebook`` de la ventana principal.
    app : SmartunerApp
        Aplicación (estado compartido, bus de eventos y acciones).

    Attributes
    ----------
    session : LiveSession | None
        Sesión en curso (None antes de analizar un audio).
    position : int
        Segmento mostrado (posición entre los conservados).
    algorithm : str
        Algoritmo de la sesión.
    speed : int
        Velocidad del modo automático (pulls/s).
    running : bool
        True mientras el modo automático está activo.
    live_plot : LivePlot
        Gestor de la figura 2×2.
    """

    def __init__(self, master: tk.Misc, app: SmartunerApp) -> None:
        """Construye los controles, se suscribe a los eventos y muestra la sesión o el mensaje guía."""
        super().__init__(master, padding=(12, 10, 12, 8))
        self.app = app
        self.session: LiveSession | None = None
        self.position = 0
        self.algorithm = app.state.selected_algorithm if app.state.selected_algorithm in ALGORITHMS else "ucb1"
        self.speed = DEFAULT_SPEED
        self.running = False
        self.prev_fret_source = ""
        self._session_cfg: Config | None = None
        self._session_seed: int | None = None
        self._seed_follows_config = True
        self._auto_job: str | None = None
        self._rebuild_job: str | None = None
        self._last_tick = 0.0
        self._credit = 0.0
        self._blitting = False
        self._background: Any = None
        self._full_pending = False
        self._dirty = False
        self._finished = False
        self._busy = False
        self._log_has_hint = False
        self._selection_job: str | None = None
        self._log_follow = True

        self._setup_styles()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self._build_selection_bar()
        self._build_playback_bar()
        self._build_banner()
        self._build_body()
        self._build_empty_state()

        events = app.state.events
        events.subscribe("analysis_ready", self._on_analysis_ready)
        events.subscribe("transcriptions_ready", self._on_transcriptions_ready)
        events.subscribe("segment_selected", self._on_segment_selected)
        events.subscribe("algorithm_selected", self._on_algorithm_selected)
        events.subscribe("config_changed", self._on_config_changed)
        events.subscribe("busy_changed", self._on_busy_changed)
        self.bind("<Map>", self._on_map, add="+")

        if app.state.analysis is not None:  # la pestaña se crea después de analizar
            self._rebuild_now()
        else:
            self._show_empty()

    # ================================================================ construcción
    def _setup_styles(self) -> None:
        """Estilos propios (prefijo ``Live.``) para no afectar a las demás pestañas."""
        style = ttk.Style(self)
        style.configure("Live.TLabelframe", padding=(10, 6, 10, 8))
        style.configure("Live.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"), foreground=plots.TEXT_PRIMARY)
        style.configure("LiveKey.TLabel", foreground=plots.TEXT_SECONDARY)
        style.configure("LiveTitle.TLabel", font=("TkDefaultFont", 10, "bold"), foreground=plots.TEXT_PRIMARY)
        style.configure("LiveValue.TLabel", foreground=plots.TEXT_PRIMARY, font=("TkDefaultFont", 10, "bold"))
        style.configure("LiveMuted.TLabel", foreground=plots.TEXT_SECONDARY, font=("TkDefaultFont", 9))
        style.configure("LiveCounter.TLabel", foreground=plots.TEXT_PRIMARY, font=("TkDefaultFont", 12, "bold"))
        style.configure("LiveBar.TButton", padding=(8, 4), width=0)
        style.configure("LiveBar.Accent.TButton", padding=(8, 4), width=0)
        style.configure("LiveNav.TButton", padding=(4, 2), width=2)
        style.configure("LiveEmpty.TLabel", foreground=plots.TEXT_PRIMARY, font=("TkDefaultFont", 13, "bold"),
                        justify="center")
        style.configure("LiveEmptyDetail.TLabel", foreground=plots.TEXT_SECONDARY, font=("TkDefaultFont", 10),
                        justify="center")

    def _build_selection_bar(self) -> None:
        """Primera fila: segmento (◀ combo ▶), algoritmo, semilla y traste previo.

        Cada control con su etiqueta va en un grupo (``ttk.Frame``); los grupos
        se colocan con :meth:`_layout_selection_bar`, que los reparte en dos
        filas cuando la ventana es estrecha (1024 px) para que nada se corte.
        """
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.selection_bar = bar

        # Segmento: ◀ [combo] ▶
        seg_group = ttk.Frame(bar)
        ttk.Label(seg_group, text="Segmento:", style="LiveKey.TLabel").pack(side="left", padx=(0, 4))
        self.prev_button = ttk.Button(seg_group, text="◀", style="LiveNav.TButton", command=lambda: self._shift_segment(-1))
        self.prev_button.pack(side="left")
        Tooltip(self.prev_button, HELP["prev"])
        self.segment_var = tk.StringVar()
        self.segment_combo = ttk.Combobox(seg_group, textvariable=self.segment_var, state="readonly", width=38)
        self.segment_combo.pack(side="left", padx=2)
        self.segment_combo.bind("<<ComboboxSelected>>", self._on_segment_combo)
        Tooltip(self.segment_combo, HELP["segment"])
        self.next_button = ttk.Button(seg_group, text="▶", style="LiveNav.TButton", command=lambda: self._shift_segment(1))
        self.next_button.pack(side="left")
        Tooltip(self.next_button, HELP["next"])

        # Algoritmo (muestra las etiquetas legibles; internamente se usan las claves).
        algo_group = ttk.Frame(bar)
        ttk.Label(algo_group, text="Algoritmo:", style="LiveKey.TLabel").pack(side="left", padx=(0, 4))
        self.algo_var = tk.StringVar(value=ALGO_LABELS[self.algorithm])
        self.algo_combo = ttk.Combobox(algo_group, textvariable=self.algo_var, state="readonly", width=19,
                                       values=[ALGO_LABELS[a] for a in ALGORITHMS])
        self.algo_combo.pack(side="left")
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_combo)
        Tooltip(self.algo_combo, HELP["algorithm"])

        # Semilla.
        seed_group = ttk.Frame(bar)
        seed_label = ttk.Label(seed_group, text="Semilla:", style="LiveKey.TLabel")
        seed_label.pack(side="left", padx=(0, 4))
        spec = PARAM_SPECS["experiment.seed"]
        self.seed_var = tk.StringVar(value=str(self.app.state.config.experiment.seed))
        self.seed_spin = ttk.Spinbox(seed_group, textvariable=self.seed_var, width=7, from_=spec.minimum or 0,
                                     to=spec.maximum or 2**31 - 1, increment=1, command=self._on_seed_entered)
        self.seed_spin.pack(side="left")
        self.seed_spin.bind("<Return>", lambda _e: self._on_seed_entered())
        self.seed_spin.bind("<FocusOut>", lambda _e: self._on_seed_entered())
        Tooltip(seed_label, HELP["seed"])
        Tooltip(self.seed_spin, HELP["seed"])

        # Traste previo usado (solo lectura) y de dónde sale.
        prev_group = ttk.Frame(bar)
        prev_key = ttk.Label(prev_group, text="Traste previo:", style="LiveKey.TLabel")
        prev_key.pack(side="left", padx=(0, 4))
        self.prev_fret_label = ttk.Label(prev_group, text="—", style="LiveValue.TLabel")
        self.prev_fret_label.pack(side="left")
        self.prev_fret_source_label = ttk.Label(prev_group, text="", style="LiveMuted.TLabel")
        self.prev_fret_source_label.pack(side="left", padx=(6, 0))
        for widget in (prev_key, self.prev_fret_label, self.prev_fret_source_label):
            Tooltip(widget, self._prev_fret_help)

        self._selection_groups = (seg_group, algo_group, seed_group, prev_group)
        self._selection_rows = 0                    # 0 = aún sin colocar; 1 o 2 filas
        self._layout_selection_bar(1 << 16)
        bar.bind("<Configure>", lambda e: self._layout_selection_bar(e.width), add="+")

    def _layout_selection_bar(self, width: int) -> None:
        """Coloca los grupos de la primera fila en una o dos filas según el ancho disponible.

        Con la ventana ancha todo va en una fila; si no cabe (≈ 1024 px), la
        semilla y el traste previo pasan a una segunda fila. Solo se vuelve a
        colocar cuando cambia la decisión (evita bucles de ``<Configure>``).

        Parameters
        ----------
        width : int
            Ancho de la barra en píxeles.
        """
        gap = 16
        groups = self._selection_groups
        needed = sum(g.winfo_reqwidth() for g in groups) + gap * (len(groups) - 1)
        rows = 1 if needed <= width else 2
        if rows == self._selection_rows:
            return
        self._selection_rows = rows
        for group in groups:
            group.grid_forget()
        if rows == 1:
            for col, group in enumerate(groups):
                group.grid(row=0, column=col, sticky="w", padx=(0, gap if col < len(groups) - 1 else 0))
        else:
            seg_group, algo_group, seed_group, prev_group = groups
            seg_group.grid(row=0, column=0, sticky="w", padx=(0, gap))
            algo_group.grid(row=0, column=1, columnspan=2, sticky="w")
            seed_group.grid(row=1, column=0, sticky="w", pady=(6, 0))
            prev_group.grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))

    def _relayout_selection_bar(self) -> None:
        """Revisa la colocación de la primera fila cuando cambia un texto (tras calcular tamaños)."""
        if self._selection_job is None:
            self._selection_job = self.after_idle(self._relayout_selection_bar_now)

    def _relayout_selection_bar_now(self) -> None:
        """Aplica :meth:`_layout_selection_bar` con el ancho real de la barra."""
        self._selection_job = None
        width = self.selection_bar.winfo_width()
        if width > 1:
            self._layout_selection_bar(width)

    @property
    def selection_rows(self) -> int:
        """Filas que ocupa la barra de selección (1 con ventana ancha, 2 si es estrecha)."""
        return self._selection_rows

    def _build_playback_bar(self) -> None:
        """Segunda fila: botones de ejecución, velocidad y contador t / T."""
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.playback_bar = bar
        specs = (
            ("step_button", "Paso", "step", "LiveBar.Accent.TButton", lambda: self.step(1)),
            ("auto_button", "Auto ▶", "auto", "LiveBar.TButton", self.start_auto),
            ("pause_button", "Pausa ⏸", "pause", "LiveBar.TButton", self.pause),
            ("stop_button", "Detener ■", "stop", "LiveBar.TButton", self.stop),
            ("restart_button", "Reiniciar ↺", "restart", "LiveBar.TButton", self.restart),
        )
        for col, (attr, text, help_key, style, command) in enumerate(specs):
            button = ttk.Button(bar, text=text, style=style, command=command)
            button.grid(row=0, column=col, padx=(0, 6))
            Tooltip(button, HELP[help_key])
            setattr(self, attr, button)
        col = len(specs)
        ttk.Separator(bar, orient="vertical").grid(row=0, column=col, sticky="ns", padx=(8, 12))
        col += 1
        speed_key = ttk.Label(bar, text="Velocidad:", style="LiveKey.TLabel")
        speed_key.grid(row=0, column=col, padx=(0, 4))
        col += 1
        self.speed_scale = ttk.Scale(bar, from_=0.0, to=scale_from_speed(MAX_SPEED), orient="horizontal", length=140)
        self.speed_scale.set(scale_from_speed(self.speed))
        self.speed_scale.grid(row=0, column=col)
        col += 1
        self.speed_label = ttk.Label(bar, text=f"{self.speed} pulls/s", width=11, style="LiveKey.TLabel")
        self.speed_label.grid(row=0, column=col, padx=(6, 0))
        self.speed_scale.configure(command=self._on_speed_scale)   # tras crear la etiqueta que actualiza
        for widget in (speed_key, self.speed_scale, self.speed_label):
            Tooltip(widget, HELP["speed"])
        col += 1
        bar.columnconfigure(col, weight=1)
        col += 1
        self.t_label = ttk.Label(bar, text="t = 0 / 0", style="LiveCounter.TLabel")
        self.t_label.grid(row=0, column=col, padx=(8, 8))
        col += 1
        self.progress = ttk.Progressbar(bar, length=120, mode="determinate", maximum=1.0)
        self.progress.grid(row=0, column=col)
        for widget in (self.t_label, self.progress):
            Tooltip(widget, HELP["t"])

    def _build_banner(self) -> None:
        """Aviso (oculto por defecto) de configuración pendiente, con su botón de acción."""
        self.banner = tk.Frame(self, background=BANNER_BG, padx=10, pady=6)
        self.banner.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.banner.columnconfigure(0, weight=1)
        self.banner_label = tk.Label(self.banner, text="", background=BANNER_BG, foreground=BANNER_FG,
                                     anchor="w", justify="left", font=("TkDefaultFont", 9), wraplength=900)
        self.banner_label.grid(row=0, column=0, sticky="ew")
        self.banner_button = ttk.Button(self.banner, text="Reiniciar ahora ↺", style="LiveBar.TButton",
                                        command=self._on_banner_action)
        self.banner_button.grid(row=0, column=1, padx=(10, 0))
        self.banner_action = "restart"
        self.banner.bind("<Configure>", lambda e: self.banner_label.configure(wraplength=max(200, e.width - 190)),
                         add="+")
        self.banner.grid_remove()

    def _build_body(self) -> None:
        """Figura 2×2 a la izquierda y panel de estado/registro/explicación a la derecha."""
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.grid(row=3, column=0, sticky="nsew")
        self.body = paned

        # Marco que NO propaga el tamaño pedido por el lienzo de matplotlib: así la
        # pestaña nunca pide más que la ventana (la barra de estado no se sale).
        holder = ttk.Frame(paned, width=560, height=300)
        holder.pack_propagate(False)
        self.plot = PlotFrame(holder, figsize=(9.2, 6.4))
        self.plot.pack(fill="both", expand=True)
        self.live_plot = LivePlot(self.plot.figure)
        canvas = self.plot.canvas
        canvas.mpl_connect("draw_event", self._on_draw_event)
        canvas.mpl_connect("resize_event", self._on_resize_event)
        paned.add(holder, weight=1)

        side = ttk.Frame(paned, width=SIDE_WIDTH, height=300, padding=(10, 0, 0, 0))
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)
        self.side = side
        paned.add(side, weight=0)
        self._build_status_box(side)
        self._build_log_box(side)
        self._build_explanation_box(side)
        self.bind("<Configure>", self._on_tab_configure, add="+")

    def _build_status_box(self, parent: ttk.Frame) -> None:
        """Recuadro «Estado del agente»: recomendado, óptimo, último pull, regret y % óptimo."""
        box = ttk.LabelFrame(parent, text="Estado del agente", style="Live.TLabelframe")
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)
        self.status_box = box
        rows = (
            ("recommended", "Recomienda ahora:"),
            ("optimal", "Óptimo según μ:"),
            ("last", "Último pull:"),
            ("regret", "Regret acumulado:"),
            ("pct", "Pulls al óptimo:"),
        )
        self.status_values: dict[str, ttk.Label] = {}
        names: list[ttk.Label] = []
        for row, (key, text) in enumerate(rows):
            name = ttk.Label(box, text=text, style="LiveKey.TLabel")
            name.grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=1)
            value = ttk.Label(box, text="—", style="LiveValue.TLabel", anchor="w", justify="left", wraplength=220)
            value.grid(row=row, column=1, sticky="ew", pady=1)
            Tooltip(name, HELP[key])
            Tooltip(value, HELP[key])
            self.status_values[key] = value
            names.append(name)

        def wrap_values(event: tk.Event) -> None:
            """Ajusta el ancho de línea de los valores al ancho del recuadro (no se cortan)."""
            key_width = max(n.winfo_reqwidth() for n in names)
            for label in self.status_values.values():
                label.configure(wraplength=max(120, event.width - key_width - 36))

        box.bind("<Configure>", wrap_values, add="+")
        self.result_label = tk.Label(box, text="", anchor="w", justify="left", wraplength=320,
                                     background=plots.HIGHLIGHT, foreground=plots.TEXT_PRIMARY,
                                     font=("TkDefaultFont", 9), padx=8, pady=5)
        self.result_label.grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.result_label.grid_remove()
        box.bind("<Configure>", lambda e: self.result_label.configure(wraplength=max(160, e.width - 40)), add="+")

    def _build_log_box(self, parent: ttk.Frame) -> None:
        """Registro de decisiones: un ``PullEvent.text`` por línea, fuente monoespaciada."""
        # El título es un Label propio: su tooltip sale al pasar por el título, no al leer el registro.
        title = ttk.Label(parent, text="Registro de decisiones (un pull por línea)", style="LiveTitle.TLabel")
        box = ttk.LabelFrame(parent, labelwidget=title, style="Live.TLabelframe")
        box.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        base = tkfont.nametofont("TkFixedFont")
        self.log_font = base.copy()
        self.log_font.configure(size=9)
        self.log_font_bold = base.copy()
        self.log_font_bold.configure(size=9, weight="bold")
        self.log = tk.Text(box, wrap="word", height=8, width=40, font=self.log_font, relief="flat",
                           borderwidth=0, background=plots.SURFACE, foreground=plots.TEXT_PRIMARY,
                           padx=6, pady=4, spacing3=3, cursor="arrow", highlightthickness=0)
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        indent = self.log_font.measure("    ")
        self.log.tag_configure("event", lmargin2=indent)
        self.log.tag_configure("head", font=self.log_font_bold)
        self.log.tag_configure("latest", background=LATEST_BG)
        self.log.tag_configure("hint", foreground=plots.TEXT_SECONDARY, lmargin2=0)
        self.log.tag_configure("summary", font=self.log_font_bold, background=plots.HIGHLIGHT, lmargin2=indent)
        self.log.configure(state="disabled")
        self.log.bind("<Configure>", self._on_log_configure, add="+")
        Tooltip(title, HELP["log"])

    def _build_explanation_box(self, parent: ttk.Frame) -> None:
        """Recuadro inferior: qué muestra el panel (d) para el algoritmo elegido."""
        box = ttk.LabelFrame(parent, text="¿Qué muestra el panel (d)?", style="Live.TLabelframe")
        box.grid(row=2, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.explanation_label = ttk.Label(box, text="", style="LiveMuted.TLabel", justify="left", wraplength=320)
        self.explanation_label.grid(row=0, column=0, sticky="ew")
        box.bind("<Configure>", lambda e: self.explanation_label.configure(wraplength=max(160, e.width - 28)),
                 add="+")

    def _build_empty_state(self) -> None:
        """Mensaje guía (en lugar de la figura) mientras no hay un audio analizado."""
        frame = tk.Frame(self, background=plots.SURFACE, highlightthickness=1, highlightbackground=plots.GRID)
        frame.grid(row=3, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)
        self.empty_frame = frame
        self.empty_label = ttk.Label(frame, text=EMPTY_TEXT, style="LiveEmpty.TLabel", background=plots.SURFACE,
                                     anchor="center")
        self.empty_label.grid(row=1, column=0, pady=(0, 10))
        self.empty_detail = ttk.Label(frame, text=EMPTY_DETAIL, style="LiveEmptyDetail.TLabel",
                                      background=plots.SURFACE, wraplength=620, anchor="center")
        self.empty_detail.grid(row=2, column=0, pady=(0, 14))
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0)
        self.empty_open_button = ttk.Button(buttons, text="Abrir MP3…", style="LiveBar.Accent.TButton",
                                            command=self.app.open_audio)
        self.empty_open_button.grid(row=0, column=0, padx=6)
        self.empty_dataset_button = ttk.Button(buttons, text="Generar dataset sintético…", style="LiveBar.TButton",
                                               command=self.app.generate_dataset)
        self.empty_dataset_button.grid(row=0, column=1, padx=6)
        Tooltip(self.empty_open_button, "Elige un MP3 con el bajo aislado; se analiza y se transcribe con los cuatro algoritmos.")
        Tooltip(self.empty_dataset_button, "Crea piezas de prueba (MIDI → audio → MP3) con su tablatura real (ground truth).")
        frame.grid_remove()

    # ================================================================ estado vacío
    def _show_empty(self, message: str | None = None) -> None:
        """Oculta la figura y muestra el mensaje guía; deshabilita los controles.

        Parameters
        ----------
        message : str | None
            Texto principal (por defecto :data:`EMPTY_TEXT`, o :data:`BUSY_TEXT`
            si hay una tarea en curso).
        """
        self._stop_auto(redraw=False)
        self.session = None
        self.live_plot.show_message("")
        has_analysis = self.app.state.analysis is not None
        if message is None:
            message = BUSY_TEXT if self._busy else EMPTY_TEXT
        self.empty_label.configure(text=message)
        if has_analysis:
            self.empty_detail.configure(text="")
        else:
            self.empty_detail.configure(text=EMPTY_DETAIL)
        state = ["disabled"] if self._busy else ["!disabled"]
        self.empty_open_button.state(state)
        self.empty_dataset_button.state(state)
        self.body.grid_remove()
        self.empty_frame.grid()
        self.segment_combo.configure(values=[])
        self.segment_var.set("")
        self.prev_fret_label.configure(text="—")
        self.prev_fret_source_label.configure(text="")
        self.t_label.configure(text="t = 0 / 0")
        self.progress.configure(value=0.0)
        self.banner.grid_remove()
        self._update_controls()

    def _show_body(self) -> None:
        """Muestra la figura y el panel derecho (hay un análisis con notas)."""
        if not self.body.winfo_manager():
            self.empty_frame.grid_remove()
            self.body.grid()

    @property
    def showing_empty_state(self) -> bool:
        """True si la pestaña muestra el mensaje guía en lugar de la figura."""
        return bool(self.empty_frame.winfo_manager())

    # ============================================================ sesión
    def _segment_data(self) -> list[SegmentBanditData]:
        """Datos bandit de los segmentos conservados del análisis vigente (lista vacía si no hay)."""
        analysis: AnalysisResult | None = self.app.state.analysis
        if analysis is None:
            return []
        return list(analysis.segment_data)

    def _valid_position(self, position: int | None) -> int | None:
        """``position`` si es un segmento válido del análisis vigente; si no, None."""
        if position is None:
            return None
        n = len(self._segment_data())
        return int(position) if 0 <= int(position) < n else None

    def prev_fret_for(self, algorithm: str, position: int) -> tuple[int | None, str]:
        """Traste previo (posición de la mano antes de la nota) y de dónde sale.

        Es el que usó la tablatura de ``algorithm`` en esa nota
        (``transcriptions[alg].chain.prev_frets[pos]``): la mano quedó en el
        traste elegido para la nota anterior. Sin tablatura (o en la primera
        nota) es ``cfg.env.initial_hand_fret``.

        Parameters
        ----------
        algorithm : str
            Algoritmo.
        position : int
            Posición del segmento.

        Returns
        -------
        prev_fret : int | None
            Traste previo (None = sin penalización de movimiento).
        source : str
            Descripción breve del origen (para la etiqueta y el tooltip).
        """
        transcription = self.app.state.transcriptions.get(algorithm)
        n_segments = len(self._segment_data())
        if transcription is not None:
            prev = list(transcription.chain.prev_frets)
            if len(prev) == n_segments and 0 <= position < len(prev):
                if position == 0:
                    return prev[0], "posición inicial de la mano"
                return prev[position], f"según la tablatura de {ALGO_SHORT.get(algorithm, algorithm)}"
        return self.app.state.config.env.initial_hand_fret, "posición inicial (sin tablatura)"

    def _prev_fret_help(self) -> str:
        """Tooltip dinámico del traste previo."""
        base = ("Posición de la mano ANTES de esta nota. Cada brazo paga la penalización de tocabilidad "
                "λ·|traste − traste previo|/12, así que cambia qué brazo es el óptimo. ")
        if self.prev_fret_source.startswith("según"):
            return base + ("Es el traste que eligió el mismo algoritmo para la nota anterior en su tablatura "
                           "(pestaña 4).")
        return base + "Es la posición inicial de la mano de la configuración (env.initial_hand_fret)."

    def _read_seed(self) -> int | None:
        """Semilla escrita en el control, validada (None si es inválida)."""
        spec = PARAM_SPECS["experiment.seed"]
        try:
            seed = int(float(str(self.seed_var.get()).strip()))
        except ValueError:
            return None
        lo = int(spec.minimum or 0)
        hi = int(spec.maximum or 2**31 - 1)
        return seed if lo <= seed <= hi else None

    @staticmethod
    def figure_titles(session: LiveSession) -> tuple[str, str]:
        """Título de la figura con el contexto de la sesión (versión larga y corta).

        Parameters
        ----------
        session : LiveSession
            Sesión dibujada.

        Returns
        -------
        tuple[str, str]
            P. ej. ``("UCB1 · nota n.º 3 (f0 55.0 Hz ≈ A1) · traste previo 0 · semilla 42 · T = 500 pulls",
            "UCB1 · nota n.º 3 (A1) · traste previo 0 · semilla 42")``.
        """
        seg = session.data.segment
        name = ALGO_LABELS.get(session.algorithm, session.algorithm)
        f0 = f"f0 {seg.f0_hz:.1f} Hz ≈ {note_name(seg.f0_hz)}" if seg.f0_hz else "f0 desconocida"
        prev = "ninguno" if session.prev_fret is None else str(session.prev_fret)
        seed = session.cfg.experiment.seed
        long_title = (f"{name} · nota n.º {session.data.position} ({f0}) · traste previo {prev} · "
                      f"semilla {seed} · T = {session.budget} pulls")
        short_title = (f"{name} · nota n.º {session.data.position} ({note_name(seg.f0_hz)}) · "
                       f"traste previo {prev} · semilla {seed}")
        return long_title, short_title

    def _new_session(self) -> None:
        """Crea una sesión para (segmento, algoritmo, semilla) actuales con la configuración vigente."""
        self._stop_auto(redraw=False)
        segments = self._segment_data()
        if not segments:
            self._show_empty(NO_SEGMENTS_TEXT if self.app.state.analysis is not None else None)
            return
        self.position = min(max(int(self.position), 0), len(segments) - 1)
        seed = self._read_seed()
        if seed is None:
            seed = self._session_seed if self._session_seed is not None else self.app.state.config.experiment.seed
            self.seed_var.set(str(seed))
        cfg = copy.deepcopy(self.app.state.config)
        cfg.experiment.seed = int(seed)
        prev, source = self.prev_fret_for(self.algorithm, self.position)
        self.prev_fret_source = source
        try:
            session = LiveSession(segments[self.position], self.algorithm, cfg, prev, seed=0)
        except Exception as exc:  # noqa: BLE001 - p. ej. configuración inválida: se informa sin romper la GUI
            logger.exception("No se pudo crear la sesión en vivo")
            self.session = None
            self._show_body()
            self.live_plot.show_message(f"No se pudo crear la sesión en vivo:\n{exc}")
            self.plot.draw()
            self._update_controls()
            return
        self.session = session
        self._session_cfg = copy.deepcopy(self.app.state.config)
        self._session_seed = int(seed)
        self._finished = False
        self._blitting = False
        self._background = None
        self._show_body()
        self._fill_segment_combo()
        self.prev_fret_label.configure(text="ninguno" if prev is None else str(prev))
        self.prev_fret_source_label.configure(text=f"({source})")
        self._relayout_selection_bar()
        self.explanation_label.configure(text=criterion_explanation(self.algorithm, cfg))
        self.result_label.grid_remove()
        self.live_plot.build(session, *self.figure_titles(session))
        self._clear_log()
        self._refresh_status()
        self._update_banner()
        self._update_controls()
        self._render(relayout=True)

    def _fill_segment_combo(self) -> None:
        """Rellena el selector de segmento y muestra el actual."""
        labels = [segment_label(d) for d in self._segment_data()]
        if list(self.segment_combo.cget("values")) != labels:
            self.segment_combo.configure(values=labels)
        if 0 <= self.position < len(labels):
            self.segment_combo.current(self.position)

    def _rebuild_now(self) -> None:
        """Reconstruye la sesión tras un análisis o transcripción nuevos (segmento seleccionado o 0)."""
        self._rebuild_job = None
        segments = self._segment_data()
        if self.app.state.analysis is None:
            self._show_empty()
            return
        if not segments:
            self._show_empty(NO_SEGMENTS_TEXT)
            return
        selected = self._valid_position(self.app.state.selected_position)
        self.position = selected if selected is not None else 0
        self._new_session()

    def _request_rebuild(self) -> None:
        """Agrupa en una sola reconstrucción los eventos que llegan seguidos (análisis + transcripción)."""
        self._stop_auto(redraw=False)
        if self._rebuild_job is None:
            self._rebuild_job = self.after_idle(self._rebuild_now)

    # ============================================================ acciones públicas
    def step(self, n: int = 1) -> list[PullEvent]:
        """Da ``n`` pulls (o los que queden) y actualiza la vista.

        Parameters
        ----------
        n : int
            Número de pulls.

        Returns
        -------
        list[PullEvent]
            Eventos producidos (vacía si no hay sesión o ya terminó).
        """
        session = self.session
        if session is None or session.done:
            return []
        events: list[PullEvent] = []
        for _ in range(max(int(n), 0)):
            if session.done:
                break
            events.append(session.step())
        self._after_pulls(events)
        return events

    def run_to_end(self) -> None:
        """Da todos los pulls que quedan de golpe (usado por las pruebas y para ir al final)."""
        if self.session is not None:
            self.step(self.session.budget - self.session.t)

    def start_auto(self) -> None:
        """Activa el modo automático (pulls seguidos a ``speed`` pulls/s)."""
        session = self.session
        if session is None or session.done or self.running:
            return
        self.running = True
        self._credit = 1.0                      # el primer pull ocurre en el primer tick
        self._last_tick = time.perf_counter()
        # Blitting: los artistas dinámicos salen del fondo estático.
        self._blitting = True
        self._background = None
        self.live_plot.set_animated(True)
        # El eje de pulls pasa a [0, T] durante la animación: las barras «crecen» hacia su
        # fracción final del presupuesto y no hay re-escalados (dibujados completos) a mitad.
        relayout = self.live_plot.count_top < session.budget
        if relayout:
            self.live_plot.set_count_top(session.budget)
        self._render(relayout=relayout, force_full=True)
        self._update_controls()
        self._auto_job = self.after(0, self._auto_tick)

    def pause(self) -> None:
        """Pausa el modo automático conservando el estado de la sesión."""
        self._stop_auto(redraw=True)

    def stop(self) -> None:
        """Detiene y vuelve a t = 0 con la MISMA sesión (misma semilla y configuración)."""
        self._stop_auto(redraw=False)
        if self.session is None:
            return
        self.session.reset()
        self._finished = False
        self.result_label.grid_remove()
        self.live_plot.reset_count_axis()
        self.live_plot.update(self.session)
        self._clear_log()
        self._refresh_status()
        self._update_controls()
        self._render(relayout=True)

    def restart(self) -> None:
        """Crea una sesión nueva con la configuración ACTUAL (aplica cambios pendientes)."""
        if self.app.state.analysis is None:
            return
        self._new_session()

    def select_position(self, position: int, broadcast: bool = True) -> None:
        """Carga el segmento ``position`` (detiene el modo automático).

        Parameters
        ----------
        position : int
            Posición entre los segmentos conservados.
        broadcast : bool
            Si es True se publica la selección a las demás pestañas
            (``source="live"``).
        """
        valid = self._valid_position(position)
        if valid is None:
            return
        self._stop_auto(redraw=False)
        self.position = valid
        if broadcast:
            self.app.state.select_segment(valid, source=SOURCE)
        self._new_session()

    def set_algorithm(self, algorithm: str) -> None:
        """Cambia el algoritmo (nueva sesión) y lo publica a las demás pestañas.

        Parameters
        ----------
        algorithm : str
            Identificador (``"egreedy"``, ``"optimistic"``, ``"ucb1"`` o ``"softmax"``).
        """
        if algorithm not in ALGORITHMS:
            return
        self.algo_var.set(ALGO_LABELS[algorithm])
        changed = self.session is None or self.session.algorithm != algorithm
        self.algorithm = algorithm
        if changed and self.app.state.analysis is not None:
            self._new_session()          # antes de publicar: el eco del evento ya no cambia nada
        self.app.state.select_algorithm(algorithm)

    def set_seed(self, seed: int) -> None:
        """Escribe ``seed`` en el control y, si cambió, crea una sesión nueva.

        Parameters
        ----------
        seed : int
            Semilla (≥ 0).
        """
        self.seed_var.set(str(int(seed)))
        self._on_seed_entered()

    def set_speed(self, speed: float) -> None:
        """Fija la velocidad del modo automático (pulls/s, acotada a 1–200).

        Parameters
        ----------
        speed : float
            Pulls por segundo.
        """
        self.speed_scale.set(scale_from_speed(speed))
        self._on_speed_scale(str(scale_from_speed(speed)))

    # ============================================================ modo automático
    def _auto_tick(self) -> None:
        """Un tick del modo automático: da los pulls que «tocan» según el reloj y redibuja.

        La velocidad se mide con el reloj: cada tick suma velocidad·Δt pulls
        de crédito y da la parte entera. Así, aunque un dibujado se retrase,
        la velocidad media se mantiene (con Δt acotado a :data:`MAX_CATCHUP_S`).
        El siguiente tick se programa descontando lo que tardó este (pulls +
        registro + *blitting*), para sostener ≈ 25 redibujados por segundo.
        """
        self._auto_job = None
        session = self.session
        if not self.running or session is None:
            return
        now = time.perf_counter()
        elapsed = min(now - self._last_tick, MAX_CATCHUP_S)
        self._last_tick = now
        self._credit += elapsed * self.speed
        n = int(self._credit)
        if n > 0:
            self._credit -= n
            self.step(n)
        if not self.running or self.session is None or self.session.done:
            return
        work_ms = (time.perf_counter() - now) * 1000.0
        # Al menos MIN_TICK_GAP_MS de respiro: deja a Tk procesar el ratón y los dibujados diferidos.
        delay = max(MIN_TICK_GAP_MS, int(round(tick_interval_ms(self.speed) - work_ms)))
        self._auto_job = self.after(delay, self._auto_tick)

    def _stop_auto(self, redraw: bool = True) -> None:
        """Detiene el modo automático y vuelve al dibujado normal (sin *blitting*).

        Parameters
        ----------
        redraw : bool
            Si es True se redibuja la figura completa (con los artistas ya no animados).
        """
        if self._auto_job is not None:
            try:
                self.after_cancel(self._auto_job)
            except tk.TclError:
                pass
            self._auto_job = None
        self.running = False
        if self._blitting:
            self._blitting = False
            self._background = None
            self.live_plot.set_animated(False)
            if redraw and self.session is not None:
                self._render(relayout=False, force_full=True)
        self._update_controls()

    # ============================================================ tras cada pull
    def _after_pulls(self, events: list[PullEvent]) -> None:
        """Actualiza registro, estado y figura tras uno o varios pulls."""
        if not events:
            return
        self._append_log(events)
        self._refresh_status()
        self._render()
        if self.session is not None and self.session.done:
            self._on_budget_exhausted()
        else:
            self._update_controls()

    def _on_budget_exhausted(self) -> None:
        """Fin del presupuesto: detiene Auto, muestra el resumen y lo registra en el log (INFO)."""
        session = self.session
        if session is None or self._finished:
            return
        self._finished = True
        self._stop_auto(redraw=True)
        arm = session.recommended_arm()
        index = session.data.arms.index(arm)
        optimal = [session.data.arms[i].label for i in session.env.optimal_arms]
        hit = session.env.is_optimal(index)
        n_opt = sum(1 for e in session.events if e.is_optimal)
        name = ALGO_LABELS.get(session.algorithm, session.algorithm)
        logger.info("En vivo: %s recomendó %s en el segmento %d tras %d pulls (óptimo: %s, regret %.1f)",
                    name, arm.label, session.data.position, session.t, ", ".join(optimal),
                    session.cumulative_regret)
        verdict = "✓ es el óptimo" if hit else f"✗ el óptimo era {', '.join(optimal)}"
        summary = (f"Terminado: {name} recomienda {arm.label} ({verdict}). Regret {session.cumulative_regret:.2f} "
                   f"en {session.t} pulls; {100.0 * n_opt / max(session.t, 1):.0f} % de los pulls al óptimo.")
        self.result_label.configure(text=summary)
        self.result_label.grid()
        self._append_summary(summary)
        self._update_controls()

    # ============================================================ dibujo
    def _render(self, relayout: bool = False, force_full: bool = False) -> None:
        """Lleva el estado de la sesión a la figura con el método más barato posible.

        * Modo automático con fondo guardado y sin cambio de escala →
          *blitting* (solo se repintan los artistas dinámicos).
        * Si no → dibujado completo diferido (``draw_idle``); con ``relayout``
          se recalcula también el *constrained layout*.

        Parameters
        ----------
        relayout : bool
            Recalcular la disposición (nueva sesión, cambio de escala o de tamaño).
        force_full : bool
            Forzar un dibujado completo aunque se pudiera hacer *blitting*.
        """
        if self.session is None:
            return
        changed = self.live_plot.update(self.session, allow_shrink=not self._blitting)
        if not self.winfo_ismapped():
            self._dirty = True            # se dibuja al volver a la pestaña (<Map>)
            return
        if (self._blitting and not (changed or relayout or force_full) and self._background is not None
                and not self._full_pending):
            self._blit()
            return
        # Un cambio de escala en plena animación no recalcula la disposición (≈ 40 % más
        # barato): los números del eje apenas cambian de ancho y sobra margen entre paneles.
        self._full_draw(relayout=relayout or (changed and not self._blitting))

    def _full_draw(self, relayout: bool) -> None:
        """Pide un dibujado completo (diferido); con ``relayout`` reactiva el *constrained layout*."""
        if relayout:
            self.plot.figure.set_layout_engine("constrained", **LAYOUT_PADS)
        self._full_pending = True
        self.plot.canvas.draw_idle()

    def _blit(self) -> None:
        """Repinta solo los artistas dinámicos sobre el fondo estático guardado."""
        canvas = self.plot.canvas
        canvas.restore_region(self._background)
        self.live_plot.draw_animated()
        canvas.blit(self.plot.figure.bbox)

    def _on_draw_event(self, event: DrawEvent) -> None:
        """Tras cada dibujado completo: congela la disposición y guarda el fondo para el *blitting*.

        Parameters
        ----------
        event : DrawEvent
            Evento de matplotlib (se ignora el de «guardar figura»).
        """
        canvas = self.plot.canvas
        if getattr(event, "canvas", canvas) is not canvas or canvas.is_saving():
            return
        self._full_pending = False
        fig = self.plot.figure
        if isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
            # La disposición ya está calculada: congelarla ahorra ≈ 40 % de cada dibujado.
            fig.set_layout_engine("none")
        if self._blitting and self.session is not None and self.live_plot.has_session:
            self._background = canvas.copy_from_bbox(fig.bbox)
            self.live_plot.draw_animated()

    def _on_resize_event(self, _event: ResizeEvent) -> None:
        """El lienzo cambió de tamaño: el fondo deja de valer y hay que recalcular la disposición."""
        self._background = None
        self._full_pending = True
        if self.live_plot.has_session:
            self.live_plot.apply_responsive_layout()
            self.plot.figure.set_layout_engine("constrained", **LAYOUT_PADS)

    def _on_map(self, event: tk.Event) -> None:
        """Al mostrarse la pestaña, dibuja lo que quedó pendiente mientras estaba oculta."""
        if event.widget is not self:
            return
        if self._dirty and self.session is not None:
            self._dirty = False
            self._render(relayout=True, force_full=True)

    def _on_tab_configure(self, event: tk.Event) -> None:
        """Ajusta el ancho del panel derecho a ventanas estrechas (1024 px)."""
        if event.widget is not self:
            return
        width = SIDE_WIDTH_NARROW if event.width < NARROW_WIDTH else SIDE_WIDTH
        if int(self.side.cget("width")) != width:
            self.side.configure(width=width)

    # ============================================================ panel derecho
    def _clear_log(self) -> None:
        """Vacía el registro y muestra la pista inicial."""
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", LOG_HINT, ("hint",))
        self.log.configure(state="disabled")
        self._log_has_hint = True
        self._log_follow = True

    def _append_log(self, events: list[PullEvent]) -> None:
        """Añade un ``PullEvent.text`` por línea (encabezado en negrita) y resalta el último.

        Solo se desplaza al final si el usuario no subió a leer; se conservan
        las últimas :data:`MAX_LOG_LINES` líneas.

        Parameters
        ----------
        events : list[PullEvent]
            Eventos nuevos, en orden.
        """
        if not events:
            return
        log = self.log
        at_bottom = log.yview()[1] >= 0.999
        log.configure(state="normal")
        if self._log_has_hint:
            log.delete("1.0", "end")
            self._log_has_hint = False
        log.tag_remove("latest", "1.0", "end")
        args: list[Any] = []
        for event in events:
            parts = event.text.split(" | ", 2)          # "t=37" | "UCB1 → D-2" | resto
            head = " | ".join(parts[:2])
            body = (" | " + parts[2]) if len(parts) > 2 else ""
            args += [head, ("event", "head"), body + "\n", ("event",)]
        log.insert("end", *args)
        log.tag_add("latest", "end-2l linestart", "end-1c")
        lines = int(log.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES + 1:
            log.delete("1.0", f"{lines - MAX_LOG_LINES}.0")
        log.configure(state="disabled")
        self._log_follow = at_bottom
        if at_bottom:
            log.see("end")

    def _append_summary(self, text: str) -> None:
        """Añade la línea de resumen final al registro y la deja a la vista.

        Parameters
        ----------
        text : str
            Resumen de la sesión.
        """
        log = self.log
        log.configure(state="normal")
        log.tag_remove("latest", "1.0", "end")
        log.insert("end", text + "\n", ("summary",))
        log.configure(state="disabled")
        log.see("end")
        self._log_follow = True

    def _on_log_configure(self, _event: tk.Event) -> None:
        """El registro cambió de tamaño (p. ej. al aparecer el resumen): si seguía el final, lo mantiene a la vista."""
        if self._log_follow:
            self.log.see("end")

    @property
    def log_lines(self) -> list[str]:
        """Líneas del registro (sin la pista inicial), para pruebas y depuración."""
        if self._log_has_hint:
            return []
        text = self.log.get("1.0", "end-1c")
        return [line for line in text.split("\n") if line]

    def _refresh_status(self) -> None:
        """Actualiza contador t / T, barra de progreso y el recuadro «Estado del agente»."""
        session = self.session
        values = self.status_values
        if session is None:
            for label in values.values():
                label.configure(text="—")
            return
        t, budget = session.t, session.budget
        self.t_label.configure(text=f"t = {t} / {budget}")
        self.progress.configure(value=t / max(budget, 1))
        env, data = session.env, session.data
        optimal = ", ".join(data.arms[i].label for i in env.optimal_arms)
        values["optimal"].configure(text=f"{optimal}  (μ* = {env.best_mean:.2f})")
        if t == 0:
            values["recommended"].configure(text="— (aún sin pulls)")
            values["last"].configure(text="—")
            values["regret"].configure(text="0.00")
            values["pct"].configure(text="—")
            return
        arm = session.recommended_arm()
        mark = "✓ óptimo" if env.is_optimal(data.arms.index(arm)) else "✗ no es el óptimo"
        values["recommended"].configure(text=f"{arm.label}  {mark}")
        last = session.events[-1]
        kind = KIND_TEXT.get(last.decision.kind, last.decision.kind)
        values["last"].configure(text=f"{last.arm_label} · r = {last.reward:.2f} · {kind}")
        values["regret"].configure(text=f"{session.cumulative_regret:.2f}")
        n_opt = sum(1 for e in session.events if e.is_optimal)
        values["pct"].configure(text=f"{100.0 * n_opt / t:.0f} %  ({n_opt} de {t})")

    # ============================================================ controles
    def _update_controls(self) -> None:
        """Habilita/deshabilita los controles según haya sesión, esté corriendo o haya terminado."""
        session = self.session
        has = session is not None
        done = has and session.done
        n_segments = len(self._segment_data()) if self.app.state.analysis is not None else 0

        def state(widget: ttk.Widget, enabled: bool) -> None:
            """Habilita (``enabled``) o deshabilita un widget ttk."""
            widget.state(["!disabled"] if enabled else ["disabled"])

        state(self.step_button, has and not done and not self.running)
        state(self.auto_button, has and not done and not self.running)
        state(self.pause_button, has and self.running)
        state(self.stop_button, has and (self.running or session.t > 0))
        state(self.restart_button, self.app.state.analysis is not None and n_segments > 0)
        state(self.prev_button, has and self.position > 0)
        state(self.next_button, has and self.position < n_segments - 1)
        self.segment_combo.state(["!disabled", "readonly"] if n_segments else ["disabled"])
        self.algo_combo.state(["!disabled", "readonly"])
        state(self.seed_spin, True)
        self.auto_button.configure(text="Auto ▶" if not self.running else "Auto ▶ …")

    def _shift_segment(self, delta: int) -> None:
        """Botones ◀ / ▶: segmento anterior o siguiente."""
        self.select_position(self.position + int(delta))

    def _on_segment_combo(self, _event: tk.Event | None = None) -> None:
        """Selección en el combobox de segmentos."""
        index = self.segment_combo.current()
        if index >= 0 and index != self.position:
            self.select_position(index)

    def _on_algo_combo(self, _event: tk.Event | None = None) -> None:
        """Selección en el combobox de algoritmos (muestra etiquetas, guarda claves)."""
        label = self.algo_var.get()
        for algo in ALGORITHMS:
            if ALGO_LABELS[algo] == label:
                self.set_algorithm(algo)
                return

    def _on_seed_entered(self) -> None:
        """Semilla editada (flechas, Enter o salir del control): valida y crea una sesión nueva si cambió."""
        seed = self._read_seed()
        if seed is None:
            # Valor inválido: se restaura el de la sesión.
            fallback = self._session_seed if self._session_seed is not None else self.app.state.config.experiment.seed
            self.seed_var.set(str(fallback))
            return
        self._seed_follows_config = seed == self.app.state.config.experiment.seed
        if self.session is not None and seed != self._session_seed:
            self._new_session()

    def _on_speed_scale(self, value: str) -> None:
        """Movimiento de la escala de velocidad (logarítmica)."""
        self.speed = speed_from_scale(float(value))
        self.speed_label.configure(text=f"{self.speed} pulls/s")

    def _on_banner_action(self) -> None:
        """Botón del aviso: reiniciar la sesión o re-transcribir (acción de la aplicación)."""
        if self.banner_action == "retranscribe":
            self.app.retranscribe()
        else:
            self.restart()

    def _update_banner(self) -> None:
        """Muestra u oculta el aviso de configuración pendiente."""
        if self.session is None or self._session_cfg is None:
            self.banner.grid_remove()
            return
        current = self.app.state.config
        analysis = self.app.state.analysis
        restart, retranscribe = pending_changes(self._session_cfg, current, getattr(analysis, "config", None))
        seed = self._read_seed()
        notes: list[str] = []
        if retranscribe:
            changes = ", ".join(format_change(k, analysis.config.get(k), current.get(k)) for k in retranscribe[:3])
            more = f" y {len(retranscribe) - 3} más" if len(retranscribe) > 3 else ""
            text = (f"⚠ Cambiaron parámetros de los brazos o de la recompensa: {changes}{more}. Hay que "
                    "re-transcribir (Ejecutar → Re-transcribir) para recalcular las notas; después la "
                    "sesión se reinicia sola.")
            self.banner_action = "retranscribe"
            self.banner_button.configure(text="Re-transcribir")
        elif restart or (seed is not None and seed != self._session_seed):
            for key in restart[:4]:
                notes.append(format_change(key, self._session_cfg.get(key), current.get(key)))
            if seed is not None and seed != self._session_seed:
                notes.append(f"Semilla ({self._session_seed} → {seed})")
            more = f" y {len(restart) - 4} más" if len(restart) > 4 else ""
            text = (f"⚠ La configuración cambió: {', '.join(notes)}{more}. La sesión en curso sigue con la "
                    "anterior; se aplicará al reiniciar (↺).")
            self.banner_action = "restart"
            self.banner_button.configure(text="Reiniciar ahora ↺")
        else:
            self.banner.grid_remove()
            return
        self.banner_label.configure(text=text)
        self.banner.grid()

    @property
    def banner_visible(self) -> bool:
        """True si el aviso de configuración pendiente está visible."""
        return bool(self.banner.winfo_manager())

    # ============================================================ eventos del bus
    def _on_analysis_ready(self, analysis: Any = None, **_kwargs: Any) -> None:
        """Nuevo análisis: nueva sesión con el segmento seleccionado (o el 0)."""
        self._request_rebuild()

    def _on_transcriptions_ready(self, transcriptions: Any = None, **_kwargs: Any) -> None:
        """Nuevas tablaturas: cambian los trastes previos → nueva sesión (agrupada con el análisis)."""
        self._request_rebuild()

    def _on_segment_selected(self, position: int | None = None, source: str = "", **_kwargs: Any) -> None:
        """Selección de otra pestaña: carga ese segmento (y detiene Auto).

        Parameters
        ----------
        position : int | None
            Posición seleccionada.
        source : str
            Pestaña de origen; las selecciones propias (``"live"``) se ignoran.
        """
        if source == SOURCE or self._rebuild_job is not None:
            return
        valid = self._valid_position(position)
        if valid is None:
            return
        if self.session is not None and valid == self.position and self.session.data.position == valid:
            return
        self.select_position(valid, broadcast=False)

    def _on_algorithm_selected(self, algorithm: str = "", **_kwargs: Any) -> None:
        """Algoritmo elegido en otra pestaña: se refleja en el combobox y en la sesión."""
        if algorithm not in ALGORITHMS:
            return
        self.algo_var.set(ALGO_LABELS[algorithm])
        if self.session is not None and self.session.algorithm == algorithm:
            return
        self.algorithm = algorithm
        if self.app.state.analysis is not None and self._rebuild_job is None:
            self._new_session()

    def _on_config_changed(self, key: str | None = None, **_kwargs: Any) -> None:
        """Cambio de configuración: la sesión sigue igual y se avisa de que se aplicará al reiniciar.

        Si la semilla del control seguía a la del experimento, se actualiza
        (la sesión nueva la usará al reiniciar).

        Parameters
        ----------
        key : str | None
            Parámetro modificado o None si cambiaron varios.
        """
        if key in (None, "experiment.seed") and self._seed_follows_config:
            self.seed_var.set(str(self.app.state.config.experiment.seed))
        self._update_banner()

    def _on_busy_changed(self, busy: bool = False, **_kwargs: Any) -> None:
        """Tarea en segundo plano: pausa el modo automático (no compite con el análisis)."""
        self._busy = bool(busy)
        if self._busy and self.running:
            self.pause()
        if self.showing_empty_state and self.app.state.analysis is None:
            self._show_empty()

    # ============================================================ limpieza
    def destroy(self) -> None:
        """Cancela los temporizadores pendientes antes de destruir la pestaña."""
        for job in (self._auto_job, self._rebuild_job, self._selection_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        self._auto_job = self._rebuild_job = self._selection_job = None
        super().destroy()


__all__ = [
    "LiveTab", "LivePlot", "SOURCE", "speed_from_scale", "scale_from_speed", "tick_interval_ms", "nice_ceiling",
    "count_axis_top", "moving_average", "moving_window", "segment_label", "reward_range", "value_limits",
    "criterion_title", "criterion_explanation", "pending_changes", "format_change", "arm_tick_labels",
    "data_param_keys", "note_name", "legend_columns", "panel_title",
]
