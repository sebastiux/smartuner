"""Pruebas de la pestaña 5 · Comparación (:mod:`gui.tabs.compare_tab`).

Abren la aplicación real (:class:`gui.app.SmartunerApp`), analizan
``data/synthetic/linea_simple.mp3`` (11 s, 16 notas con ground truth) y
ejecutan un experimento PEQUEÑO desde el botón de la pestaña (4 corridas,
T = 100, barridos de 2 valores con 2 corridas por valor). Verifican ESTADO —no píxeles—:
tabla contra :meth:`ExperimentResult.summary_rows`, gráficas dibujadas,
emparejamiento con el ground truth del mapa de aciertos, sincronización de
segmento y algoritmo con las demás pestañas, archivos exportados, estados
vacíos, modo compacto y avisos.

Necesitan tkinter y un servidor X; sin ``DISPLAY`` se omiten. Para ejecutarlas::

    xvfb-run -a .venv/bin/python -m pytest tests/test_gui_compare.py -q

Para que el archivo tarde poco, casi todas las pruebas comparten una ventana,
un análisis y un experimento (fixture de módulo).
"""

from __future__ import annotations

import copy
import csv
import doctest
import os
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

try:
    import tkinter  # noqa: F401

    HAS_TK = True
except ImportError:  # pragma: no cover - depende de la instalación de Python
    HAS_TK = False

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "data" / "synthetic" / "linea_simple.mp3"

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not HAS_TK, reason="tkinter no está disponible"),
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="sin servidor X (DISPLAY); usa xvfb-run"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado"),
    pytest.mark.skipif(not AUDIO.exists(), reason=f"falta {AUDIO.name} (genera el dataset sintético)"),
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def pump(app: Any, seconds: float = 0.1) -> None:
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
        time.sleep(0.01)


def wait_idle(app: Any, timeout: float = 120.0) -> None:
    """Espera a que termine la tarea en segundo plano y procesa lo pendiente."""
    pump(app, 0.05)
    wait_until(app, lambda: not app.busy, timeout=timeout)
    pump(app, 0.2)


def new_app(geometry: str = "1360x880+0+0") -> Any:
    """Crea la aplicación con la pestaña Comparación visible."""
    from gui.app import SmartunerApp

    app = SmartunerApp()
    app.root.geometry(geometry)
    app.show_tab("compare")
    pump(app, 0.4)
    return app


def tab_of(app: Any) -> Any:
    """La pestaña Comparación de ``app``."""
    return app.tabs["compare"]


def shown(tab: Any) -> None:
    """Muestra la pestaña y dibuja lo pendiente (el dibujo es diferido)."""
    tab.app.show_tab("compare")
    pump(tab.app, 0.15)
    if tab.redraw_pending:
        tab.draw_current()


def figure_text(tab: Any) -> str:
    """Todo el texto escrito en la figura, con los saltos de línea del ajuste convertidos en espacios."""
    return " ".join(" ".join(text.get_text() for text in tab.plot.figure.texts).split())


def display_columns(tab: Any) -> list[str]:
    """Columnas visibles de la tabla."""
    from gui.tabs.compare_tab import _display_columns

    return _display_columns(tab.table)


def click_figure(tab: Any, x_data: float, y_data: float) -> None:
    """Simula un clic izquierdo en el punto de datos ``(x, y)`` del primer eje de la figura."""
    canvas = tab.plot.canvas
    canvas.draw()  # constrained layout coloca los ejes al dibujar
    pump(tab.app, 0.05)
    ax = tab.plot.figure.axes[0]
    x_disp, y_disp = ax.transData.transform((x_data, y_data))
    widget = canvas.get_tk_widget()
    x_tk, y_tk = int(round(x_disp)), int(round(widget.winfo_height() - y_disp))
    widget.event_generate("<Motion>", x=x_tk, y=y_tk)
    widget.event_generate("<ButtonPress-1>", x=x_tk, y=y_tk)
    widget.event_generate("<ButtonRelease-1>", x=x_tk, y=y_tk)
    pump(tab.app, 0.1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app() -> Iterator[Any]:
    """Aplicación con ``linea_simple.mp3`` analizado y una configuración de experimento pequeña."""
    application = new_app()
    cfg = application.state.config
    cfg.experiment.n_runs = 4
    cfg.experiment.sweep_runs = 2
    cfg.env.budget = 100
    # Dos valores por barrido: basta para dibujar las curvas de sensibilidad y de λ.
    cfg.experiment.sweep_epsilon = [0.05, 0.2]
    cfg.experiment.sweep_q0 = [1.0, 5.0]
    cfg.experiment.sweep_c = [0.5, 2.0]
    cfg.experiment.sweep_tau = [0.03, 0.3]
    cfg.experiment.sweep_lambda = [0.0, 0.5]
    application.state.events.emit("config_changed", key=None)
    application.open_audio(AUDIO)
    wait_until(application, lambda: not application.busy and application.state.analysis is not None, timeout=120)
    application.show_tab("compare")
    pump(application, 0.3)
    yield application
    application.quit()


@pytest.fixture(scope="module")
def experiment(app: Any) -> Any:
    """Ejecuta el experimento con el botón principal de la pestaña (una vez para todo el archivo)."""
    tab = tab_of(app)
    busy_events: list[bool] = []
    app.state.events.subscribe("busy_changed", lambda busy=False, **_k: busy_events.append(bool(busy)))
    assert tab.run_button.instate(["!disabled"])
    tab.run_button.invoke()
    assert app.busy  # la tarea se lanzó en segundo plano
    assert tab.run_button.instate(["disabled"])  # busy_changed deshabilita el botón mientras corre
    wait_idle(app, timeout=180)
    assert busy_events[:2] == [True, False]
    assert app.state.experiment is not None
    return app.state.experiment


@pytest.fixture()
def tab(app: Any, experiment: Any) -> Iterator[Any]:
    """La pestaña con el experimento vigente, la tabla desplegada y sin segmento seleccionado."""
    t = tab_of(app)
    if t.result is not experiment:
        app.state.events.emit("experiment_ready", result=experiment)
    if not t.table_visible:
        t.toggle_table()
    app.state.select_segment(None, source="test")
    shown(t)
    yield t
    app.state.select_segment(None, source="test")


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------


def test_docstring_examples() -> None:
    """Los ejemplos ``>>>`` del módulo (formato de celdas, mejor algoritmo, resumen) se cumplen."""
    import gui.tabs.compare_tab as module

    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0


def test_cell_format_and_best_marks() -> None:
    """Formato «media ± std» por tipo de columna y ★ solo en el mejor algoritmo (nunca en los oráculos)."""
    from gui.tabs.compare_tab import TABLE_COLUMNS, best_algorithms, format_cell

    assert format_cell(0.71234, 0.03111, "reward") == "0.712 ± 0.031"
    assert format_cell(0.985, 0.012, "frac") == "98.5 ± 1.2 %"
    assert format_cell(4.13, 0.24, "ms") == "4.1 ± 0.2 ms"
    assert format_cell(0.5, 0.04, "ms") == "0.50 ± 0.04 ms"
    assert format_cell(float("nan"), 1.0, "reward") == "—"
    assert format_cell(1.0, 0.0, "frac", with_std=False) == "100.0 %"
    by_key = {c.key: c for c in TABLE_COLUMNS}
    rows = [
        {"algorithm": "egreedy", "mean_reward": 0.50, "final_regret": 40.0, "pitch_acc": 1.0},
        {"algorithm": "ucb1", "mean_reward": 0.56, "final_regret": 21.0, "pitch_acc": 1.0},
        {"algorithm": "softmax", "mean_reward": 0.5601, "final_regret": 21.04, "pitch_acc": 1.0},
        {"algorithm": "oracle", "mean_reward": None, "final_regret": 0.0, "pitch_acc": 1.0},
    ]
    # Empates al redondear (0.560 = 0.560; 21.0 = 21.0) → ambos llevan ★; el oráculo nunca.
    assert best_algorithms(rows, by_key["mean_reward"]) == {"ucb1", "softmax"}
    assert best_algorithms(rows, by_key["final_regret"]) == {"ucb1", "softmax"}
    # Todos iguales → no se marca a nadie.
    assert best_algorithms(rows, by_key["pitch_acc"]) == set()


# ---------------------------------------------------------------------------
# Estados vacíos y modo compacto (ventanas propias)
# ---------------------------------------------------------------------------


def test_empty_state_without_audio() -> None:
    """Sin audio: mensaje guía, «Ejecutar» y exportaciones deshabilitados, tabla vacía y lista completa."""
    from src import plots

    application = new_app()
    try:
        t = tab_of(application)
        shown(t)
        assert t.plot_state == "empty"
        texts = figure_text(t)
        assert "Abre un MP3" in texts and "≈2 min con la configuración por defecto" in texts
        assert t.run_button.instate(["disabled"])
        for button in (t.save_plot_button, t.export_all_button, t.export_table_button, t.export_curves_button,
                       t.export_timing_button):
            assert button.instate(["disabled"])
        assert t.table.get_children() == ()
        assert t.table_message.winfo_ismapped()
        assert t.listbox.size() == len(plots.COMPARISON_PLOTS) + 1
        assert t.plot_keys == [s.key for s in plots.COMPARISON_PLOTS] + ["spectrogram"]
        assert "Primero abre un audio" in t.run_status.cget("text")
        assert "por nota" in t.plan_values["estimate"]
        assert t.description.get("1.0", "end").startswith("⚠ Disponible después de ejecutar el experimento.")
        # Todas las gráficas en gris (sin datos).
        assert all(str(t.listbox.itemcget(i, "foreground")) == plots.TEXT_MUTED for i in range(t.listbox.size()))
        # El espectrograma necesita un análisis: también muestra su mensaje guía.
        t.select_plot("spectrogram")
        t.draw_current()
        assert t.plot_state == "empty"
        assert "audio analizado" in figure_text(t)
        # Con una tarea en curso el mensaje cambia y vuelve al terminar.
        application.state.events.emit("busy_changed", busy=True)
        t.draw_current()
        assert "segundo plano" in figure_text(t)
        application.state.events.emit("busy_changed", busy=False)
        # «Editar en Configuración» lleva a la pestaña 2.
        t.edit_config_button.invoke()
        pump(application, 0.1)
        assert application.notebook.select() == str(application.tabs["config"])
    finally:
        application.quit()


def test_compact_window_keeps_status_bar_and_folds_table() -> None:
    """A 1024×700 la pestaña cabe (la barra de estado sigue visible) y la tabla empieza plegada."""
    application = new_app("1024x700+0+0")
    try:
        t = tab_of(application)
        pump(application, 0.5)
        assert application.status.winfo_ismapped()
        assert t.winfo_reqheight() <= application.notebook.winfo_height()
        assert t.compact and not t.table_visible
        assert not t.table_frame.winfo_ismapped()
        assert "▸" in t.table_toggle.cget("text")
        # El usuario la despliega: su elección se respeta y la nota pasa al tooltip «ⓘ».
        t.table_toggle.invoke()
        pump(application, 0.3)
        assert t.table_visible and t.table_frame.winfo_ismapped()
        assert t.table_info.winfo_ismapped() and not t.table_note.winfo_ismapped()
        # Botones CSV cortos con el rótulo «Exportar:» cuando no caben los textos completos.
        assert t.export_label.cget("text") == "Exportar:"
        assert t.export_table_button.cget("text") == "Tabla CSV…"
        application.root.geometry("1360x880+0+0")
        pump(application, 0.5)
        assert not t.compact and t.table_visible and t.table_note.winfo_ismapped()
        assert t.export_table_button.cget("text") == "Exportar tabla CSV…"
    finally:
        application.quit()


# ---------------------------------------------------------------------------
# Con análisis, antes del experimento
# ---------------------------------------------------------------------------


def test_ready_before_experiment(app: Any) -> None:
    """Tras analizar: «Ejecutar» habilitado, estimación con el número de notas y espectrograma disponible."""
    t = tab_of(app)
    if app.state.experiment is not None:  # por si el orden de las pruebas cambia
        app.state.events.emit("analysis_ready", analysis=app.state.analysis)
    shown(t)
    n_notes = len(app.state.analysis.segment_data)
    assert t.result is None
    assert t.run_button.instate(["!disabled"])
    assert t.plot_state == "empty"
    assert "Ejecuta el experimento" in figure_text(t)
    assert f"({n_notes} notas)" in t.plan_values["estimate"]
    assert t.plan_values["runs"] == "4 · semilla 42"
    assert t.plan_values["budget"] == "100 pulls por nota"
    assert t.plan_values["sweeps"] == "ε, Q₀, c, τ · 2 corridas"
    assert t.plan_values["lambda"].startswith("sí")
    assert f"{n_notes} notas" in t.run_status.cget("text")
    t.select_plot("spectrogram")
    t.draw_current()
    assert t.plot_state == "plot" and t.drawn_key == "spectrogram"
    assert t.save_plot_button.instate(["!disabled"])
    t.select_plot(0)


# ---------------------------------------------------------------------------
# Con el experimento
# ---------------------------------------------------------------------------


def test_experiment_populates_table_and_first_plot(app: Any, experiment: Any, tab: Any) -> None:
    """``experiment_ready`` rellena la tabla desde ``summary_rows`` y muestra la primera gráfica."""
    from gui.tabs.compare_tab import BEST_MARK, ORACLES, TABLE_COLUMNS, format_cell
    from src.config import ALGORITHMS

    assert app.notebook.select() == str(tab)  # app.run_experiment muestra esta pestaña al terminar
    assert tab.result is experiment
    assert tab.current_key == "average_reward" and tab.plot_state == "plot"
    rows = experiment.summary_rows()
    assert list(tab.table.get_children()) == [r["algorithm"] for r in rows]
    assert [r["algorithm"] for r in rows] == [*ALGORITHMS, *ORACLES]  # hay ground truth → dos oráculos
    for row, shown_row in zip(rows, tab.table_rows, strict=True):
        oracle = row["algorithm"] in ORACLES
        for col in TABLE_COLUMNS:
            expected = format_cell(row.get(col.key), row.get(f"{col.key}_std"), col.kind, with_std=not oracle)
            assert shown_row[col.key].removesuffix(BEST_MARK) == expected
        assert tab.table.item(row["algorithm"], "text").strip() == row["label"]
        assert ("oracle" in tab.table.item(row["algorithm"], "tags")) == oracle
    # El ★ de la recompensa media está en el algoritmo de mayor media.
    best = max((r for r in rows if r["algorithm"] not in ORACLES), key=lambda r: r["mean_reward"])
    assert tab.table_rows[rows.index(best)]["mean_reward"].endswith(BEST_MARK)
    # Sin segmentos de más el F1 coincide con la precisión y se omite.
    if all(not r.get("n_extra") for r in rows):
        assert "pitch_f1" not in display_columns(tab) and "pitch_acc" in display_columns(tab)
    assert f"{experiment.n_runs} corridas" in tab.table_note.cget("text")
    assert "Resultados vigentes" in tab.run_status.cget("text")
    for button in (tab.save_plot_button, tab.export_all_button, tab.export_table_button, tab.export_curves_button,
                   tab.export_timing_button):
        assert button.instate(["!disabled"])


def test_every_plot_draws_with_its_description(app: Any, tab: Any) -> None:
    """Cada gráfica de la lista se dibuja (con datos) y su descripción aparece en «Cómo leer esta gráfica»."""
    n = len(tab.plot_specs)
    for i, spec in enumerate(tab.plot_specs):
        tab.select_plot(i)
        tab.draw_current()
        assert tab.drawn_key == spec.key
        assert tab.plot_state == "plot", spec.key
        assert tab.plot.figure.axes
        assert spec.description[:60] in tab.description.get("1.0", "end").replace("\n", " ")
        assert tab.plot_counter.cget("text") == f"Gráfica {i + 1} de {n}"
    # ◀ ▶: al final ▶ está deshabilitado; ◀ retrocede.
    assert tab.next_button.instate(["disabled"])
    tab.prev_button.invoke()
    assert tab.current_index == n - 2
    tab.select_plot(0)
    assert tab.prev_button.instate(["disabled"])
    tab.next_button.invoke()
    assert tab.current_key == "cumulative_regret"


def test_heatmap_uses_gt_matches_and_highlights_selection(app: Any, experiment: Any, tab: Any) -> None:
    """El mapa de aciertos recibe ``matches`` de ``match_segments_to_gt`` y enmarca el segmento elegido."""
    from src.experiments import match_segments_to_gt

    tab.select_plot("tab_heatmap")
    tab.draw_current()
    expected = match_segments_to_gt(app.state.analysis.kept, experiment.gt_notes,
                                    experiment.config.experiment.onset_tolerance_s)
    fn, _args, kwargs = tab._drawn_call
    assert kwargs["matches"] == expected
    # Selección desde otra pestaña → marco en la(s) columna(s) de ese segmento, sin redibujar la figura.
    app.state.select_segment(3, source="audio")
    pump(app, 0.1)
    assert tab.drawn_selection == 3
    assert len(tab._heat_overlay) == expected.count(3) >= 1
    # Clic en otra columna → selecciona su segmento en toda la aplicación.
    column = next(g for g, s in enumerate(expected) if s is not None and s != 3)
    click_figure(tab, column + 0.5, 0.5)
    assert app.state.selected_position == expected[column]
    assert tab.drawn_selection == expected[column]


def test_spectrogram_click_and_external_selection(app: Any, tab: Any) -> None:
    """Clic en una nota del espectrograma → ``segment_selected`` con ``source="compare"``; y al revés."""
    received: list[tuple[Any, str]] = []
    app.state.events.subscribe("segment_selected",
                               lambda position=None, source="", **_k: received.append((position, source)))
    tab.select_plot("spectrogram")
    tab.draw_current()
    seg = app.state.analysis.kept[4]
    click_figure(tab, (seg.start_s + seg.end_s) / 2, 100.0)
    assert app.state.selected_position == 4
    assert (4, "compare") in received
    if tab.redraw_pending:
        tab.draw_current()
    assert tab.drawn_selection == 4
    assert tab._drawn_call[2]["selected"] == 4
    # Selección desde otra pestaña: se redibuja con el nuevo segmento resaltado.
    app.state.select_segment(7, source="tablature")
    pump(app, 0.2)
    if tab.redraw_pending:
        tab.draw_current()
    assert tab.drawn_selection == 7
    assert tab.position_at_time(seg.start_s + 1e-3) == 4


def test_table_row_selects_algorithm_everywhere(app: Any, tab: Any) -> None:
    """Clic en una fila → ``select_algorithm``; los oráculos no son algoritmos; ``algorithm_selected`` → fila."""
    tab.table.selection_set("softmax")
    pump(app, 0.1)
    assert app.state.selected_algorithm == "softmax"
    tab.table.selection_set("oracle")
    pump(app, 0.1)
    assert tab.table.selection() == ("softmax",)
    assert app.state.selected_algorithm == "softmax"
    app.state.select_algorithm("egreedy")
    pump(app, 0.1)
    assert tab.table.selection() == ("egreedy",)


def test_exports(app: Any, experiment: Any, tab: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tabla, curvas y tiempos en CSV; gráfica actual en PNG/PDF; todas las gráficas en una carpeta."""
    import gui.tabs.compare_tab as module
    from src.experiments import SUMMARY_COLUMNS, TIMING_COLUMNS

    written = tab.export_table_csv(tmp_path / "tabla.csv")
    with open(written, encoding="utf-8") as fh:
        table_rows = list(csv.DictReader(fh))
    assert tuple(table_rows[0].keys()) == SUMMARY_COLUMNS
    assert [r["algorithm"] for r in table_rows] == [r["algorithm"] for r in experiment.summary_rows()]

    with open(tab.export_curves_csv(tmp_path / "curvas.csv"), encoding="utf-8") as fh:
        curves = list(csv.reader(fh))
    assert len(curves) == 1 + len(experiment.algorithms) * experiment.budget
    with open(tab.export_timing_csv(tmp_path / "tiempos.csv"), encoding="utf-8") as fh:
        timing = list(csv.reader(fh))
    assert tuple(timing[0]) == TIMING_COLUMNS and len(timing) == 1 + len(experiment.algorithms)
    assert str(tmp_path) in app.status.message.cget("text")

    tab.select_plot("cumulative_regret")
    tab.draw_current()
    for name in ("regret.png", "regret.pdf"):
        assert tab.save_current_plot(tmp_path / name)
        wait_idle(app)
        assert (tmp_path / name).stat().st_size > 1000
    assert (tmp_path / "regret.png").read_bytes()[:4] == b"\x89PNG"
    assert (tmp_path / "regret.pdf").read_bytes()[:4] == b"%PDF"
    # Extensión no soportada → aviso (sin escribir nada).
    errors: list[tuple[Any, ...]] = []
    monkeypatch.setattr(module, "show_error", lambda *args, **_kw: errors.append(args))
    assert not tab.save_current_plot(tmp_path / "regret.xyz")
    assert errors and not (tmp_path / "regret.xyz").exists()

    assert tab.export_all_plots(tmp_path / "todas", formats=("png",))
    wait_idle(app, timeout=120)
    names = sorted(p.name for p in (tmp_path / "todas").iterdir())
    assert names == sorted(f"{key}.png" for key in tab.plot_keys)
    assert "Se exportaron 11 gráficas" in app.status.message.cget("text")


def test_table_headings_explain_each_column(app: Any, tab: Any) -> None:
    """Cada encabezado visible tiene su explicación (tooltip) y ``column_at`` identifica la columna."""
    from gui.tabs.compare_tab import ALGORITHM_COLUMN_HELP, TABLE_COLUMNS

    tree = tab.table
    pump(app, 0.1)
    x = 5
    assert tab.table_tooltip.column_at(x, 8) == "#0"
    assert tab._column_help("#0") == ALGORITHM_COLUMN_HELP
    x += int(tree.column("#0", "width"))
    helps = {c.key: c.help for c in TABLE_COLUMNS}
    for key in display_columns(tab):
        width = int(tree.column(key, "width"))
        if x + width // 2 > tree.winfo_width():
            break  # columna fuera de la vista (desplazamiento horizontal)
        assert tab.table_tooltip.column_at(x + width // 2, 8) == key
        assert tab._column_help(key) == helps[key] != ""
        x += width
    # Fuera de los encabezados (sobre las filas) no hay tooltip de columna.
    assert tab.table_tooltip.column_at(40, 60) is None


def test_busy_and_stale_config(app: Any, tab: Any) -> None:
    """Con una tarea en curso se deshabilitan las acciones; si cambia la configuración se avisa."""
    events = app.state.events
    events.emit("busy_changed", busy=True)
    assert tab.run_button.instate(["disabled"])
    assert tab.export_all_button.instate(["disabled"]) and tab.save_plot_button.instate(["disabled"])
    assert tab.export_table_button.instate(["!disabled"])  # los CSV son instantáneos
    assert "tarea en curso" in tab.run_status.cget("text")
    events.emit("busy_changed", busy=False)
    assert tab.run_button.instate(["!disabled"])

    cfg = app.state.config
    old = cfg.experiment.n_runs
    try:
        cfg.experiment.n_runs = old + 1
        events.emit("config_changed", key="experiment.n_runs")
        assert tab.is_stale()
        assert "otra configuración" in tab.run_status.cget("text")
        assert str(tab.run_status.cget("style")) == "CmpWarn.TLabel"
        assert tab.plan_values["runs"].startswith(f"{old + 1} ")
    finally:
        cfg.experiment.n_runs = old
        events.emit("config_changed", key="experiment.n_runs")
    assert not tab.is_stale()


def test_missing_data_without_ground_truth(app: Any, experiment: Any, tab: Any) -> None:
    """Sin ground truth: gráficas de precisión en gris con su motivo, sin columnas de precisión ni oráculos."""
    from dataclasses import replace

    from src import plots

    no_gt = replace(experiment, gt_notes=None, accuracy={}, oracle_accuracy=None, viterbi_accuracy=None,
                    lambda_sweep=None)
    try:
        app.state.events.emit("experiment_ready", result=no_gt)
        shown(tab)
        for key in ("accuracy", "tab_heatmap", "lambda_effect"):
            index = tab.plot_keys.index(key)
            assert str(tab.listbox.itemcget(index, "foreground")) == plots.TEXT_MUTED
        tab.select_plot("accuracy")
        tab.draw_current()
        assert tab.plot_state == "missing"
        assert tab.save_plot_button.instate(["disabled"])
        assert tab.description.get("1.0", "end").startswith("⚠ " + plots.MSG_NO_GT[:30])
        assert list(tab.table.get_children()) == list(experiment.algorithms)
        assert "pitch_acc" not in display_columns(tab)
        assert "Sin ground truth" in tab.table_note.cget("text")
    finally:
        app.state.events.emit("experiment_ready", result=experiment)
    assert tab.result is experiment


def test_extra_segments_show_f1_columns(app: Any, experiment: Any, tab: Any) -> None:
    """Si hay segmentos sin nota real, se muestran las columnas F1 y la nota lo explica."""
    rows = experiment.summary_rows()
    for row in rows:
        row["n_extra"] = 2
    with_extra = copy.copy(experiment)
    with_extra.summary_rows = lambda: copy.deepcopy(rows)  # type: ignore[method-assign]
    try:
        app.state.events.emit("experiment_ready", result=with_extra)
        assert "pitch_f1" in display_columns(tab) and "position_f1" in display_columns(tab)
        assert "2 segmentos sin nota real" in tab.table_note.cget("text")
    finally:
        app.state.events.emit("experiment_ready", result=experiment)


def test_new_analysis_clears_results(app: Any, experiment: Any, tab: Any) -> None:
    """``analysis_ready`` deja la pestaña vacía (el experimento ya no corresponde); un experimento la rellena."""
    tab.select_plot("runtime")
    analysis = app.state.analysis
    app.state.events.emit("analysis_ready", analysis=analysis)
    shown(tab)
    assert tab.result is None and tab.plot_state == "empty"
    assert tab.table.get_children() == () and tab.table_rows == []
    assert tab.export_table_button.instate(["disabled"])
    assert tab.export_table_csv("no-se-usa.csv") is None
    app.state.events.emit("experiment_ready", result=experiment)
    shown(tab)
    assert tab.result is experiment
    assert tab.current_key == "average_reward" and tab.plot_state == "plot"
