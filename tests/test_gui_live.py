"""Pruebas de la pestaña 3 · Ejecución en vivo (:mod:`gui.tabs.live_tab`).

Dos grupos:

* **Funciones puras** del módulo (escala logarítmica de velocidad, escalones de
  los ejes, media móvil, columnas de la leyenda, cambios de configuración
  pendientes, títulos del panel (d)) y sus ejemplos ``>>>``. Solo necesitan
  tkinter importable, no una pantalla.
* **La pestaña real** dentro de :class:`gui.app.SmartunerApp`, con un análisis
  REAL pequeño (``data/synthetic/linea_simple.mp3``: 11 s, 16 notas) hecho con
  la acción de la aplicación (hilo trabajador + eventos ``analysis_ready`` y
  ``transcriptions_ready``). Se ejercitan los controles (botones, combobox,
  semilla, velocidad, modo automático) y se verifica ESTADO —sesión, registro,
  alturas de las barras, botones habilitados, avisos—, no píxeles.

Las pruebas con ventana necesitan un servidor X; sin ``DISPLAY`` se omiten::

    xvfb-run -a .venv/bin/python -m pytest tests/test_gui_live.py -q

Para que el archivo tarde poco, todas las pruebas con análisis comparten una
ventana y un único análisis (fixtures de módulo); cada prueba restablece la
configuración, el algoritmo, el segmento y la semilla.
"""

from __future__ import annotations

import doctest
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import tkinter  # noqa: F401

    HAS_TK = True
except ImportError:  # pragma: no cover - depende de la instalación de Python
    HAS_TK = False

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "data" / "synthetic" / "linea_simple.mp3"

pytestmark = pytest.mark.skipif(not HAS_TK, reason="tkinter no está disponible")

#: Marcas de las pruebas que abren la ventana (pantalla, ffmpeg y el MP3 de prueba).
needs_gui = [
    pytest.mark.gui,
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="sin servidor X (DISPLAY); usa xvfb-run"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado"),
    pytest.mark.skipif(not AUDIO.exists(), reason=f"falta {AUDIO.name} (genera el dataset sintético)"),
]


def gui_test(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Aplica las marcas :data:`needs_gui` a una prueba."""
    for mark in reversed(needs_gui):
        fn = mark(fn)
    return fn


# ---------------------------------------------------------------------------
# Funciones puras (sin ventana)
# ---------------------------------------------------------------------------


def test_module_doctests() -> None:
    """Los ejemplos ``>>>`` de los docstrings del módulo se cumplen."""
    import gui.tabs.live_tab as live_tab

    result = doctest.testmod(live_tab, optionflags=doctest.ELLIPSIS)
    assert result.attempted >= 10
    assert result.failed == 0


def test_speed_scale_is_logarithmic_and_clamped() -> None:
    """La escala de velocidad es log₁₀ (1–200 pulls/s) y su inversa es coherente."""
    from gui.tabs.live_tab import MAX_SPEED, MIN_SPEED, scale_from_speed, speed_from_scale

    for speed in (1, 2, 5, 10, 20, 50, 100, 200):
        assert speed_from_scale(scale_from_speed(speed)) == speed
    assert speed_from_scale(-5.0) == MIN_SPEED
    assert speed_from_scale(10.0) == MAX_SPEED
    assert scale_from_speed(1000) == pytest.approx(math.log10(MAX_SPEED))


def test_tick_interval_caps_redraws_at_25_fps() -> None:
    """Hasta 25 pulls/s hay un tick por pull; por encima el tick queda en 40 ms (≤ 25 fps)."""
    from gui.tabs.live_tab import FRAME_MS, tick_interval_ms

    assert tick_interval_ms(1) == 1000
    assert tick_interval_ms(10) == 100
    assert tick_interval_ms(25) == FRAME_MS == 40
    assert tick_interval_ms(200) == FRAME_MS


def test_moving_average_matches_definition() -> None:
    """La media móvil causal coincide con su definición punto a punto."""
    from gui.tabs.live_tab import moving_average

    rng = np.random.default_rng(0)
    x = rng.normal(size=37)
    w = 5
    expected = np.array([x[max(0, t - w + 1): t + 1].mean() for t in range(x.size)])
    np.testing.assert_allclose(moving_average(x, w), expected)
    assert moving_average(np.array([]), 3).size == 0


def test_axis_steps_are_round_and_bounded() -> None:
    """Los límites de los ejes dinámicos son números redondos y el de pulls no pasa de T."""
    from gui.tabs.live_tab import COUNT_MANTISSAS, count_axis_top, nice_ceiling

    for value in (0.07, 0.7, 1.3, 2.2, 7.0, 37.0, 480.0):
        top = nice_ceiling(value)
        assert top >= value
        mantissa = top / 10 ** math.floor(math.log10(top))
        assert round(mantissa, 9) in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    assert nice_ceiling(37, COUNT_MANTISSAS) == 50.0
    assert count_axis_top(0, 500) == 5.0
    assert count_axis_top(480, 500) == 500.0          # 1000 > T → se queda en T
    assert count_axis_top(120, 1000) == 200.0


def test_value_limits_include_zero_and_q0() -> None:
    """El eje de valores contiene 0, todas las recompensas posibles y Q₀ (optimista)."""
    from gui.tabs.live_tab import value_limits

    lo, hi = value_limits(-0.2, 0.9, 2.0)
    assert lo < -0.2 and hi > 2.0
    lo, hi = value_limits(0.1, 0.8, 0.0)
    assert lo == 0.0 and hi > 0.8


def test_legend_columns_fit_available_width() -> None:
    """La leyenda usa tantas columnas como quepan, entre 1 y el máximo."""
    from gui.tabs.live_tab import LEGEND_MAX_COLS, legend_columns

    widths = [120.0, 90.0, 150.0, 140.0, 130.0, 110.0, 100.0, 140.0, 120.0]
    wide = legend_columns(widths, 1000.0, 12.0)
    narrow = legend_columns(widths, 600.0, 12.0)
    assert wide == LEGEND_MAX_COLS
    assert 1 <= narrow < wide
    assert legend_columns(widths, 50.0, 12.0) == 1
    assert legend_columns([], 500.0, 12.0) == 1


def test_criterion_titles_show_current_hyperparameter() -> None:
    """El título del panel (d) muestra el ε / τ VIGENTE (con decaimiento) y tiene versión corta."""
    from gui.tabs.live_tab import criterion_title
    from src.agents import EpsilonGreedyAgent, SoftmaxAgent

    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.5, decay=0.5, epsilon_min=0.1)
    assert "ε = 0.5" in criterion_title("egreedy", agent)
    agent.t = 2                                         # ε₂ = max(0.1, 0.5·0.5²) = 0.125
    assert "ε = 0.125" in criterion_title("egreedy", agent)
    softmax = SoftmaxAgent(n_arms=3, tau=0.2)
    full, short = criterion_title("softmax", softmax), criterion_title("softmax", softmax, compact=True)
    assert "τ = 0.2" in full and "τ = 0.2" in short and len(short) < len(full)


def test_criterion_explanation_for_every_algorithm() -> None:
    """Cada algoritmo tiene su explicación del panel (d), con sus hiperparámetros."""
    from gui.tabs.live_tab import criterion_explanation
    from src.config import ALGORITHMS, Config

    cfg = Config()
    for algo in ALGORITHMS:
        text = criterion_explanation(algo, cfg)
        assert 60 < len(text) < 400
    assert f"{cfg.agent.ucb_c:g}" in criterion_explanation("ucb1", cfg)
    assert f"{cfg.agent.q0:g}" in criterion_explanation("optimistic", cfg)


def test_pending_changes_split_restart_and_retranscribe() -> None:
    """λ y T se aplican al reiniciar; k (brazos) exige re-transcribir."""
    from gui.tabs.live_tab import format_change, pending_changes
    from src.config import Config

    old, new = Config(), Config()
    assert pending_changes(old, new, old) == ([], [])
    new.env.lam = 0.3
    new.env.budget = 300
    new.agent.epsilon = 0.2
    new.env.k_semitones = 4
    restart, retranscribe = pending_changes(old, new, old)
    assert set(restart) == {"env.lam", "env.budget", "agent.epsilon"}
    assert retranscribe == ["env.k_semitones"]
    assert format_change("env.budget", 500, 300) == "T pulls por segmento (500 → 300)"


# ---------------------------------------------------------------------------
# Utilidades de las pruebas con ventana
# ---------------------------------------------------------------------------


def pump(app: Any, seconds: float = 0.15) -> None:
    """Procesa eventos de tkinter durante ``seconds`` (incluye ``after`` y el sondeo de tareas)."""
    end = time.time() + seconds
    while time.time() < end:
        app.root.update()
        time.sleep(0.005)


def wait_until(app: Any, condition: Callable[[], bool], timeout: float = 60.0) -> None:
    """Procesa eventos hasta que ``condition()`` sea cierta (o falla tras ``timeout`` s)."""
    end = time.time() + timeout
    while not condition():
        if time.time() > end:
            raise TimeoutError("La condición no se cumplió a tiempo")
        app.root.update()
        time.sleep(0.005)


def settle(tab: Any, timeout: float = 10.0) -> None:
    """Procesa eventos hasta que la pestaña no tenga reconstrucciones ni dibujados completos pendientes."""
    app = tab.app
    app.root.update()
    wait_until(app, lambda: tab._rebuild_job is None and not tab._full_pending, timeout=timeout)
    app.root.update()


def disabled(widget: Any) -> bool:
    """True si el widget ttk está deshabilitado."""
    return bool(widget.instate(["disabled"]))


def event_lines(tab: Any) -> list[str]:
    """Líneas del registro que corresponden a un pull (empiezan por ``t=``)."""
    return [line for line in tab.log_lines if line.startswith("t=")]


def rewards(tab: Any) -> np.ndarray:
    """Recompensas de los pulls realizados en la sesión en curso."""
    return np.array([e.reward for e in tab.session.events])


@pytest.fixture(scope="module")
def app() -> Iterator[Any]:
    """Una ventana compartida por las pruebas del módulo, en la pestaña «Ejecución en vivo».

    Es la aplicación real (:class:`gui.app.SmartunerApp`: menú, bus de eventos,
    acciones y tareas en segundo plano), pero solo con ESTA pestaña: las demás
    tienen sus propias pruebas, y así un fallo en otra pestaña no contamina
    estas (todas se comunican solo por ``app.state.events``).
    """
    from tkinter import messagebox

    from gui.app import SmartunerApp

    # Ningún diálogo modal debe bloquear las pruebas.
    originals = {name: getattr(messagebox, name) for name in ("showinfo", "showerror", "showwarning")}
    for name in originals:
        setattr(messagebox, name, lambda *a, **k: None)
    only_live = tuple(spec for spec in SmartunerApp.TAB_SPECS if spec[0] == "live")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SmartunerApp, "TAB_SPECS", only_live)
        application = SmartunerApp()
    application.root.geometry("1360x880+0+0")
    application.show_tab("live")
    pump(application, 0.4)
    yield application
    application.quit()
    for name, fn in originals.items():
        setattr(messagebox, name, fn)


@pytest.fixture(scope="module")
def analyzed(app: Any) -> Any:
    """La misma ventana tras analizar ``linea_simple.mp3`` con la acción de la aplicación (≈ 3 s)."""
    app.open_audio(AUDIO)
    wait_until(app, lambda: not app.busy and app.state.analysis is not None, timeout=120)
    tab = app.tabs["live"]
    wait_until(app, lambda: tab.session is not None, timeout=10)
    settle(tab)
    return app


@pytest.fixture
def tab(analyzed: Any) -> Iterator[Any]:
    """La pestaña con estado limpio: configuración por defecto, UCB1, segmento 0, semilla 42, t = 0."""
    from src.config import Config

    app = analyzed
    live = app.tabs["live"]
    live.pause()
    app.root.geometry("1360x880+0+0")
    app.state.config = Config()
    app.state.events.emit("config_changed", key=None)
    app.state.events.emit("busy_changed", busy=False)
    live.set_speed(20)
    live.seed_var.set(str(app.state.config.experiment.seed))
    live.set_algorithm("ucb1")
    live.select_position(0)                                    # siempre crea una sesión nueva (t = 0)
    assert live._rebuild_job is None
    assert live.session is not None and live.session.t == 0
    yield live
    live.pause()


# ---------------------------------------------------------------------------
# Pruebas con ventana
# ---------------------------------------------------------------------------


@gui_test
def test_empty_state_before_analysis(app: Any) -> None:
    """Antes de analizar: mensaje guía, sin sesión y controles de ejecución deshabilitados."""
    from gui.tabs.live_tab import BUSY_TEXT

    tab = app.tabs["live"]
    if app.state.analysis is not None:
        pytest.skip("la ventana compartida ya tiene un análisis")
    assert tab.session is None
    assert tab.showing_empty_state
    assert "Abre un MP3 (Archivo → Abrir)" in tab.empty_label.cget("text")
    for button in (tab.step_button, tab.auto_button, tab.pause_button, tab.stop_button, tab.restart_button):
        assert disabled(button)
    assert tab.t_label.cget("text") == "t = 0 / 0"
    # Mientras hay una tarea en curso el mensaje lo dice; al terminar vuelve el mensaje guía.
    app.state.events.emit("busy_changed", busy=True)
    assert tab.empty_label.cget("text") == BUSY_TEXT
    assert disabled(tab.empty_open_button)
    app.state.events.emit("busy_changed", busy=False)
    assert "Abre un MP3" in tab.empty_label.cget("text")
    assert not disabled(tab.empty_open_button)
    # Las acciones sin sesión no fallan.
    assert tab.step() == []
    tab.start_auto()
    assert not tab.running


@gui_test
def test_session_after_analysis(tab: Any) -> None:
    """Tras analizar hay una sesión en el segmento 0 con selector, semilla y traste previo rellenos."""
    app = tab.app
    data = app.state.analysis.segment_data
    assert not tab.showing_empty_state
    assert tab.session.algorithm == app.state.selected_algorithm == "ucb1"
    assert tab.session.data.position == tab.position == 0
    assert tab.t_label.cget("text") == f"t = 0 / {app.state.config.env.budget}"
    labels = list(tab.segment_combo.cget("values"))
    assert len(labels) == len(data)
    assert re.fullmatch(r"n\.º 0 · \d+\.\d\d s · f0 \d+\.\d Hz \([A-G]#?-?\d\) · \d+ brazos", labels[0])
    assert labels[0].endswith(f"· {data[0].n_arms} brazos")
    assert tab.segment_var.get() == labels[0]
    assert tab.seed_var.get() == str(app.state.config.experiment.seed)
    prev = app.state.transcriptions["ucb1"].chain.prev_frets[0]
    assert tab.prev_fret_label.cget("text") == str(prev)
    assert "UCB1" in tab.explanation_label.cget("text")
    assert not disabled(tab.step_button) and not disabled(tab.auto_button)
    assert disabled(tab.pause_button) and disabled(tab.stop_button) and disabled(tab.prev_button)
    assert not disabled(tab.next_button)
    assert tab.log_lines == []                                  # solo la pista inicial
    assert not tab.banner_visible


@gui_test
def test_step_button_updates_log_status_and_figure(tab: Any) -> None:
    """«Paso» da un pull: registro, estado y barras reflejan el agente (UCB1 sin probar → rayadas e ∞)."""
    from gui.tabs.live_tab import HIGHLIGHT_LW, UNTRIED_HATCH

    for _ in range(3):
        tab.step_button.invoke()
    pump(tab.app, 0.2)
    session = tab.session
    assert session.t == 3
    assert tab.t_label.cget("text") == f"t = 3 / {session.budget}"
    lines = event_lines(tab)
    assert len(lines) == 3
    assert lines[-1].startswith("t=3 | UCB1 → ")
    assert lines[-1] == session.events[-1].text
    assert tab.status_values["recommended"].cget("text").startswith(session.recommended_arm().label)
    assert tab.status_values["regret"].cget("text") == f"{session.cumulative_regret:.2f}"
    assert not disabled(tab.stop_button)

    plot = tab.live_plot
    agent = session.agent
    np.testing.assert_allclose([r.get_height() for r in plot.count_bars], agent.counts)
    np.testing.assert_allclose([r.get_height() for r in plot.q_bars], agent.q)
    last = session.events[-1].arm_index
    assert plot.q_bars[last].get_linewidth() == pytest.approx(HIGHLIGHT_LW)
    untried = np.flatnonzero(agent.counts == 0)
    assert untried.size == session.data.n_arms - 3
    for i in range(session.data.n_arms):
        assert (plot.score_bars[i].get_hatch() == UNTRIED_HATCH) == (i in untried)
        assert plot.inf_texts[i].get_visible() == (i in untried)
    # Etiquetas: brazo «A-0» y ★ en los óptimos.
    ticks = [t.get_text() for t in plot.ax_q.get_xticklabels()]
    for i in session.env.optimal_arms:
        assert ticks[i] == session.data.arms[i].label + "★"
    assert plot.titles["d"].get_text().startswith("(d) Índice UCB")
    assert plot.ax_d.get_title() == ""                          # sin título central duplicado


@gui_test
def test_run_to_end_reproduces_tablature_and_logs_summary(tab: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Con la semilla del experimento, la ejecución en vivo repite pull a pull la corrida de la tablatura."""
    from src.config import ALGORITHMS

    app = tab.app
    position = 2
    caplog.set_level(logging.INFO, logger="gui.tabs.live_tab")
    for algo in ALGORITHMS:
        tab.set_algorithm(algo)
        tab.select_position(position)
        tab.run_to_end()
        chain = app.state.transcriptions[algo].chain
        session = tab.session
        assert session.done and session.prev_fret == chain.prev_frets[position]
        np.testing.assert_allclose(rewards(tab), chain.rewards[position])
        assert session.recommended_arm() == chain.arms[position]
        # Fin del presupuesto: Paso/Auto deshabilitados y resumen visible.
        assert disabled(tab.step_button) and disabled(tab.auto_button)
        assert tab.result_label.winfo_manager()
        assert tab.result_label.cget("text").startswith("Terminado:")
        assert tab.log_lines[-1].startswith("Terminado:")
    messages = [r.getMessage() for r in caplog.records if r.name == "gui.tabs.live_tab"]
    pattern = (r"En vivo: .+ recomendó [EADG]-\d+ en el segmento 2 tras 500 pulls "
               r"\(óptimo: [EADG]-\d+(, [EADG]-\d+)*, regret \d+\.\d\)")
    assert sum(bool(re.fullmatch(pattern, m)) for m in messages) == len(ALGORITHMS)


@gui_test
def test_log_keeps_last_500_pulls(tab: Any) -> None:
    """Con T = 700 el registro conserva solo los últimos 500 pulls (más el resumen)."""
    from gui.tabs.live_tab import MAX_LOG_LINES

    app = tab.app
    app.state.config.env.budget = 700
    app.state.events.emit("config_changed", key="env.budget")
    tab.restart()
    tab.run_to_end()
    pump(app, 0.1)
    lines = event_lines(tab)
    assert len(lines) == MAX_LOG_LINES
    assert lines[0].startswith("t=201 | ") and lines[-1].startswith("t=700 | ")
    assert tab.log_lines[-1].startswith("Terminado:")
    assert tab.log.yview()[1] == pytest.approx(1.0)             # el final queda a la vista


@gui_test
def test_auto_pause_stop_and_speed(tab: Any) -> None:
    """Auto avanza solo (fluido a 200 pulls/s, con blitting), Pausa congela, Detener vuelve a t = 0."""
    app = tab.app
    counts = {"blit": 0, "full": 0}
    original_blit, original_draw = tab._blit, tab.plot.canvas.draw

    def blit() -> None:
        counts["blit"] += 1
        original_blit()

    def draw(*args: Any, **kwargs: Any) -> Any:
        counts["full"] += 1
        return original_draw(*args, **kwargs)

    tab._blit = blit
    tab.plot.canvas.draw = draw
    try:
        tab.set_speed(200)
        assert tab.speed == 200 and tab.speed_label.cget("text") == "200 pulls/s"
        tab.auto_button.invoke()
        assert tab.running and not disabled(tab.pause_button) and disabled(tab.step_button)
        wait_until(app, lambda: tab.session.t >= 60, timeout=10)
        tab.pause_button.invoke()
        paused_at = tab.session.t
        assert not tab.running and not disabled(tab.step_button)
        pump(app, 0.3)
        assert tab.session.t == paused_at
        assert len(event_lines(tab)) == paused_at

        # Detener: misma sesión (misma semilla) vuelta a t = 0.
        session = tab.session
        first = rewards(tab)[:20].copy()
        tab.stop_button.invoke()
        assert tab.session is session and session.t == 0 and tab.log_lines == []
        assert disabled(tab.stop_button)

        # Auto hasta el final (T = 150 para ir rápido): la repetición es idéntica y la
        # velocidad real ronda la pedida.
        app.state.config.env.budget = 150
        app.state.events.emit("config_changed", key="env.budget")
        tab.restart()
        session = tab.session
        settle(tab)
        counts.update(blit=0, full=0)
        start = time.perf_counter()
        tab.auto_button.invoke()
        wait_until(app, lambda: not tab.running, timeout=20)
        elapsed = time.perf_counter() - start
        assert session.done
        np.testing.assert_allclose(rewards(tab)[:20], first)
        assert session.budget / elapsed > 100                    # ≥ 100 pulls/s efectivos (pedidos: 200)
        assert counts["blit"] >= 6                               # se repinta con blitting…
        assert counts["full"] <= 5                               # …y casi nunca la figura entera
    finally:
        tab._blit = original_blit
        tab.plot.canvas.draw = original_draw


@gui_test
def test_speed_scale_command(tab: Any) -> None:
    """Mover la escala (logarítmica) cambia la velocidad y su etiqueta."""
    from gui.tabs.live_tab import scale_from_speed

    tab.speed_scale.set(scale_from_speed(50))
    tab._on_speed_scale(str(tab.speed_scale.get()))           # lo que hace Tk al arrastrar
    assert tab.speed == 50
    assert tab.speed_label.cget("text") == "50 pulls/s"
    tab.set_speed(1000)
    assert tab.speed == 200


@gui_test
def test_algorithm_combo_and_external_selection(tab: Any) -> None:
    """El combobox cambia el algoritmo (y lo publica); otra pestaña también puede cambiarlo."""
    from src.config import ALGO_LABELS

    app = tab.app
    tab.step(5)
    tab.algo_var.set(ALGO_LABELS["softmax"])
    tab.algo_combo.event_generate("<<ComboboxSelected>>")
    pump(app, 0.1)
    assert tab.session.algorithm == "softmax" and tab.session.t == 0
    assert app.state.selected_algorithm == "softmax"
    assert tab.live_plot.ax_d.get_ylim() == pytest.approx((0.0, 1.05))
    assert "τ =" in tab.live_plot.titles["d"].get_text()
    assert "Softmax" in tab.explanation_label.cget("text")
    np.testing.assert_allclose(sum(r.get_height() for r in tab.live_plot.score_bars), 1.0)

    app.state.select_algorithm("egreedy")                     # p. ej. desde la pestaña Tablatura
    pump(app, 0.1)
    assert tab.algo_var.get() == ALGO_LABELS["egreedy"]
    assert tab.session.algorithm == "egreedy"
    assert "ε = 0.1" in tab.live_plot.titles["d"].get_text()


@gui_test
def test_epsilon_in_title_follows_decay(tab: Any) -> None:
    """Con decaimiento, el título del panel (d) muestra el ε vigente tras cada pull."""
    app = tab.app
    app.state.config.agent.epsilon_decay = 0.99
    app.state.events.emit("config_changed", key="agent.epsilon_decay")
    tab.set_algorithm("egreedy")
    tab.step(50)
    eps = tab.session.agent.epsilon
    assert eps == pytest.approx(max(app.state.config.agent.epsilon_min, 0.1 * 0.99 ** 50))
    assert f"ε = {eps:.3g}" in tab.live_plot.titles["d"].get_text()


@gui_test
def test_segment_navigation_is_shared(tab: Any) -> None:
    """◀ ▶ y el combobox publican la selección; la de otra pestaña carga ese segmento y detiene Auto."""
    app = tab.app
    tab.next_button.invoke()
    assert tab.position == tab.session.data.position == app.state.selected_position == 1
    tab.prev_button.invoke()
    assert tab.position == 0 and app.state.selected_position == 0

    tab.segment_combo.current(3)
    tab.segment_combo.event_generate("<<ComboboxSelected>>")
    pump(app, 0.1)
    assert tab.position == tab.session.data.position == app.state.selected_position == 3

    # Eco de su propia selección: no se recrea la sesión.
    session = tab.session
    app.state.events.emit("segment_selected", position=3, source="live")
    assert tab.session is session

    # Selección desde otra pestaña con Auto en marcha.
    tab.set_speed(20)
    tab.start_auto()
    assert tab.running
    app.state.select_segment(5, source="audio")
    pump(app, 0.1)
    assert not tab.running
    assert tab.position == tab.session.data.position == 5 and tab.session.t == 0
    assert tab.segment_var.get().startswith("n.º 5 · ")
    last = len(app.state.analysis.segment_data) - 1
    tab.select_position(last)
    assert disabled(tab.next_button) and not disabled(tab.prev_button)


@gui_test
def test_seed_spinbox_creates_new_session(tab: Any) -> None:
    """Otra semilla = otra secuencia de frames; un valor inválido se rechaza."""
    app = tab.app
    tab.step(30)
    original = rewards(tab).copy()
    tab.seed_spin.focus_force()
    app.root.update()
    tab.seed_var.set("7")
    tab.seed_spin.event_generate("<Return>")
    pump(app, 0.1)
    assert tab.session.cfg.experiment.seed == 7 and tab.session.t == 0
    tab.step(30)
    assert not np.allclose(rewards(tab), original)
    assert "semilla 7" in tab.live_plot.suptitle.get_text()

    session = tab.session
    tab.seed_var.set("abc")
    tab.seed_spin.event_generate("<Return>")
    assert tab.seed_var.get() == "7" and tab.session is session
    tab.set_seed(app.state.config.experiment.seed)
    tab.step(30)
    np.testing.assert_allclose(rewards(tab), original)        # misma semilla → mismos pulls


@gui_test
def test_prev_fret_comes_from_tablature(tab: Any) -> None:
    """El traste previo es el de la tablatura del algoritmo; sin tablatura, la mano inicial."""
    app = tab.app
    position = 4
    for algo in ("ucb1", "softmax"):
        tab.set_algorithm(algo)
        tab.select_position(position)
        expected = app.state.transcriptions[algo].chain.prev_frets[position]
        assert tab.session.prev_fret == expected
        assert tab.prev_fret_label.cget("text") == ("ninguno" if expected is None else str(expected))
        assert "según la tablatura" in tab.prev_fret_source_label.cget("text")

    saved = app.state.transcriptions
    try:
        app.state.transcriptions = {}
        tab.restart()
        initial = app.state.config.env.initial_hand_fret
        assert tab.session.prev_fret == initial
        assert tab.prev_fret_label.cget("text") == str(initial)
        assert "sin tablatura" in tab.prev_fret_source_label.cget("text")
    finally:
        app.state.transcriptions = saved


@gui_test
def test_config_change_applies_on_restart(tab: Any) -> None:
    """Un cambio de λ no toca la sesión en curso: aviso «se aplicará al reiniciar» y Reiniciar lo aplica."""
    from src.config import Config

    app = tab.app
    tab.step(10)
    app.state.config.env.lam = 0.3
    app.state.events.emit("config_changed", key="env.lam")
    pump(app, 0.1)
    assert tab.banner_visible
    assert "se aplicará al reiniciar" in tab.banner_label.cget("text")
    assert tab.session.cfg.env.lam == pytest.approx(Config().env.lam) and tab.session.t == 10

    # Un parámetro de los brazos exige re-transcribir (el botón del aviso cambia de acción).
    default_k = app.state.config.env.k_semitones
    app.state.config.env.k_semitones = (default_k or 2) + 2
    app.state.events.emit("config_changed", key="env.k_semitones")
    assert tab.banner_action == "retranscribe"
    assert "re-transcribir" in tab.banner_label.cget("text").lower()
    app.state.config.env.k_semitones = default_k
    app.state.events.emit("config_changed", key="env.k_semitones")
    assert tab.banner_action == "restart"

    tab.restart_button.invoke()
    pump(app, 0.1)
    assert tab.session.cfg.env.lam == pytest.approx(0.3) and tab.session.t == 0
    assert not tab.banner_visible


@gui_test
def test_new_analysis_resets_session(tab: Any) -> None:
    """``analysis_ready`` + ``transcriptions_ready`` reinician en el segmento seleccionado (o el 0)."""
    app = tab.app
    tab.select_position(6)
    tab.step(25)
    app.state.selected_position = 3
    app.state.events.emit("analysis_ready", analysis=app.state.analysis)
    app.state.events.emit("transcriptions_ready", transcriptions=app.state.transcriptions)
    settle(tab)
    assert tab.position == tab.session.data.position == 3 and tab.session.t == 0

    app.state.selected_position = None
    app.state.events.emit("analysis_ready", analysis=app.state.analysis)
    app.state.events.emit("transcriptions_ready", transcriptions=app.state.transcriptions)
    settle(tab)
    assert tab.position == 0 and tab.session.t == 0


@gui_test
def test_busy_task_pauses_auto(tab: Any) -> None:
    """Si arranca una tarea pesada en segundo plano, el modo automático se pausa."""
    app = tab.app
    tab.start_auto()
    assert tab.running
    app.state.events.emit("busy_changed", busy=True)
    assert not tab.running
    app.state.events.emit("busy_changed", busy=False)
    assert not disabled(tab.auto_button)


@gui_test
def test_hidden_tab_defers_drawing(tab: Any) -> None:
    """Con la pestaña oculta los pulls no dibujan; al volver se dibuja lo pendiente."""
    from tkinter import ttk

    app = tab.app
    other = ttk.Frame(app.notebook)                            # otra pestaña cualquiera
    app.notebook.add(other, text="otra")
    try:
        app.notebook.select(other)
        pump(app, 0.2)
        tab.step(5)
        assert tab._dirty
        app.show_tab("live")
        pump(app, 0.3)
        assert not tab._dirty
        assert tab.session.t == 5
    finally:
        app.notebook.forget(other)
        other.destroy()


@gui_test
def test_narrow_window_layout(tab: Any) -> None:
    """A 1024×700 la primera fila pasa a dos filas sin cortes y los títulos usan su versión corta."""
    app = tab.app
    tab.step(4)
    app.root.geometry("1024x700+0+0")
    wait_until(app, lambda: tab.selection_rows == 2 and tab.winfo_width() < 1100, timeout=5)
    pump(app, 0.2)
    settle(tab)
    assert tab.selection_rows == 2
    bar_width = tab.selection_bar.winfo_width()
    for group in tab._selection_groups:
        assert group.winfo_ismapped()
        assert group.winfo_x() + group.winfo_width() <= bar_width
    plot = tab.live_plot
    assert plot.titles["d"].get_text() == "(d) Índice UCB   (c = 1.41)"
    assert plot.legend_ncol < 5
    assert tab.side.winfo_width() > 250
    assert tab.plot.winfo_width() > 450

    app.root.geometry("1360x880+0+0")
    wait_until(app, lambda: tab.selection_rows == 1 and tab.winfo_width() > 1300, timeout=5)
    pump(app, 0.2)
    settle(tab)
    assert tab.selection_rows == 1
    assert plot.titles["d"].get_text().startswith("(d) Índice UCB = Q + c·√(ln t / n)")
    assert plot.legend_ncol == 5
