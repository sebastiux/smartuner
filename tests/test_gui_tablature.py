"""Pruebas de la pestaña 4 · Tablatura (:mod:`gui.tabs.tablature_tab`).

Abren la aplicación real (:class:`gui.app.SmartunerApp`) y ejercitan los
controles de la pestaña con un análisis REAL pequeño
(``data/synthetic/linea_simple.mp3``, 11 s, 16 notas con ground truth).
Verifican ESTADO —recuadros dibujados en el lienzo y su posición, segmento
seleccionado en ``app.state``, algoritmo, comparación con el ground truth
(contra :mod:`src.experiments`), texto ASCII (contra :func:`src.tab.render_ascii`),
archivos exportados, filas de la tabla de brazos—, no píxeles.

Necesitan tkinter y un servidor X; sin ``DISPLAY`` se omiten. Para ejecutarlas::

    xvfb-run -a .venv/bin/python -m pytest tests/test_gui_tablature.py -q

Para que el archivo tarde poco, casi todas las pruebas comparten una ventana y
un único análisis (fixture de módulo); cada prueba restablece algoritmo,
comparación, vista, zoom y selección.
"""

from __future__ import annotations

import csv
import doctest
import json
import math
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


def new_app(geometry: str = "1360x880+0+0") -> Any:
    """Crea la aplicación con la pestaña Tablatura visible."""
    from gui.app import SmartunerApp

    app = SmartunerApp()
    app.root.geometry(geometry)
    app.show_tab("tab")
    pump(app, 0.3)
    return app


def tab_of(app: Any) -> Any:
    """La pestaña Tablatura de ``app``."""
    return app.tabs["tab"]


def labels_on_canvas(tab: Any) -> dict[int, str]:
    """Traste dibujado en el recuadro de cada nota: {posición: texto}."""
    canvas = tab.canvas
    result: dict[int, str] = {}
    for item in canvas.find_withtag("note"):
        if canvas.type(item) != "text" or "badge" in canvas.gettags(item):
            continue
        pos = next(int(t[4:]) for t in canvas.gettags(item) if t.startswith("pos:"))
        result[pos] = canvas.itemcget(item, "text")
    return result


def click_note(tab: Any, position: int) -> None:
    """Simula un clic izquierdo en el centro del recuadro de la nota ``position``."""
    tab.scroll_to(position, force=True)
    pump(tab.app, 0.05)
    x0, y0, x1, y1 = tab.note_bbox(position)
    x = int((x0 + x1) / 2 - tab.canvas.canvasx(0))
    y = int((y0 + y1) / 2 - tab.canvas.canvasy(0))
    tab.canvas.event_generate("<Motion>", x=x, y=y)
    tab.canvas.event_generate("<Button-1>", x=x, y=y)
    tab.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
    pump(tab.app, 0.05)


def expected_outcomes(app: Any, algorithm: str) -> list[int]:
    """Estado esperado de cada segmento, calculado directamente con :mod:`src.experiments`."""
    from gui.tabs.tablature_tab import outcome_of_segments
    from src.experiments import match_segments_to_gt, score_arms

    analysis = app.state.analysis
    segments = [d.segment for d in analysis.segment_data]
    matches = match_segments_to_gt(segments, analysis.ground_truth, app.state.config.experiment.onset_tolerance_s)
    acc = score_arms(app.state.transcriptions[algorithm].chain.arms, matches, analysis.ground_truth)
    return outcome_of_segments(len(segments), matches, acc.outcomes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app() -> Iterator[Any]:
    """Aplicación con ``linea_simple.mp3`` analizado y transcrito (una vez para todo el archivo)."""
    application = new_app()
    application.open_audio(AUDIO)
    wait_until(application, lambda: not application.busy and len(application.state.transcriptions) == 4,
               timeout=120)
    application.show_tab("tab")
    pump(application, 0.3)
    yield application
    application.quit()


@pytest.fixture()
def tab(app: Any) -> Iterator[Any]:
    """La pestaña en un estado conocido: UCB1, comparación activa, vista gráfica, zoom inicial, sin selección."""
    from gui.tabs.tablature_tab import DEFAULT_ZOOM

    t = tab_of(app)
    app.show_tab("tab")
    app.state.select_algorithm("ucb1")
    if not t.compare_var.get():
        t.chk_compare.invoke()
    if t.ascii_var.get():
        t.chk_ascii.invoke()
    t.set_zoom(DEFAULT_ZOOM)
    app.state.select_segment(None, source="test")
    pump(app, 0.1)
    yield t
    app.state.select_segment(None, source="test")


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------


def test_docstring_examples() -> None:
    """Los ejemplos ``>>>`` del módulo de la pestaña (utilidades puras) se cumplen."""
    import gui.tabs.tablature_tab as module

    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0


def test_empty_state_before_analysis() -> None:
    """Sin audio la pestaña muestra el mensaje guía, botones para empezar y la exportación deshabilitada."""
    application = new_app()
    try:
        t = tab_of(application)
        assert t.view_state == "empty"
        texts = [t.canvas.itemcget(i, "text") for i in t.canvas.find_withtag("message")
                 if t.canvas.type(i) == "text"]
        assert any("Abre un MP3" in text for text in texts)
        assert t.canvas.find_withtag("message") and not t.canvas.find_withtag("note")
        assert t.btn_empty_open.winfo_ismapped()
        assert all(b.instate(["disabled"]) for b in t.export_buttons.values())
        assert t.export_all_button.instate(["disabled"])
        assert t.chk_compare.instate(["disabled"])
        assert t.zoom_scale.instate(["disabled"])
        assert "Ninguna nota seleccionada" in t.detail_label.cget("text")
        assert t.tree.get_children() == ()
        # Con una tarea en curso y aún sin análisis, el lienzo lo dice.
        application.state.events.emit("busy_changed", busy=True)
        assert t.view_state == "busy"
        application.state.events.emit("busy_changed", busy=False)
        assert t.view_state == "empty"
    finally:
        application.quit()


def test_draws_one_box_per_note_on_its_string(app: Any, tab: Any) -> None:
    """Cada nota de la transcripción es un recuadro con su traste, en la línea de su cuerda y en x = inicio."""
    from gui.tabs.tablature_tab import X0

    notes = app.state.transcriptions["ucb1"].notes
    assert tab.view_state == "tab"
    assert len(notes) == 16
    assert labels_on_canvas(tab) == {n.position: str(n.fret) for n in notes}
    for note in notes:
        x0, y0, x1, y1 = tab.note_bbox(note.position)
        assert x0 == pytest.approx(X0 + note.start_s * tab.zoom)
        assert (y0 + y1) / 2 == pytest.approx(tab._layout.main_lines[note.string])
    # G arriba y E abajo, como en cualquier tablatura de bajo.
    lines = tab._layout.main_lines
    assert lines["G"] < lines["D"] < lines["A"] < lines["E"]
    ruler = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_withtag("ruler") if tab.canvas.type(i) == "text"]
    assert "0.0" in ruler and "1.0" in ruler


def test_algorithm_radio_and_algorithm_selected_event(app: Any, tab: Any) -> None:
    """Los botones de radio cambian el algoritmo de la app; ``algorithm_selected`` externo cambia la vista."""
    tab.algo_radios["softmax"].invoke()
    pump(app)
    assert app.state.selected_algorithm == "softmax"
    assert tab.algorithm == "softmax"
    assert labels_on_canvas(tab) == {n.position: str(n.fret) for n in app.state.transcriptions["softmax"].notes}

    app.state.select_algorithm("egreedy")
    pump(app)
    assert tab.algorithm == "egreedy" and tab.algo_var.get() == "egreedy"
    assert [n.label for n in tab.notes] == [n.label for n in app.state.transcriptions["egreedy"].notes]
    assert "ε-greedy" in tab.summary_label.cget("text")


def test_click_on_note_selects_segment(app: Any, tab: Any) -> None:
    """Un clic en una nota publica ``segment_selected`` con ``source="tablature"`` y rellena el panel inferior."""
    received: list[tuple[Any, str]] = []
    app.state.events.subscribe("segment_selected", lambda position=None, source="", **_k: received.append(
        (position, source)))
    click_note(tab, 5)
    assert app.state.selected_position == 5
    assert tab.selected == 5
    assert (5, "tablature") in received
    assert tab.canvas.find_withtag("selection")
    assert len(tab.tree.get_children()) == app.state.analysis.segment_data[5].n_arms


def test_external_selection_highlights_and_scrolls(app: Any, tab: Any) -> None:
    """Una selección desde otra pestaña resalta la nota y desplaza la vista hasta ella."""
    tab.set_zoom(600)
    tab.canvas.xview_moveto(0.0)
    pump(app)
    last = len(tab.notes) - 1
    app.state.select_segment(last, source="audio")
    pump(app)
    assert tab.selected == last
    left, right = tab.visible_range()
    x0, _y0, x1, _y1 = tab.note_bbox(last)
    assert left <= x0 and x1 <= right
    outline = [i for i in tab.canvas.find_withtag("selection") if tab.canvas.type(i) == "polygon"]
    assert outline, "la nota seleccionada debe tener contorno"
    # El eco de la propia selección no vuelve a desplazar la vista.
    tab.canvas.xview_moveto(0.0)
    app.state.events.emit("segment_selected", position=last, source="tablature")
    assert tab.visible_range()[0] == 0


def test_keyboard_navigation(app: Any, tab: Any) -> None:
    """← / → recorren las notas; Inicio / Fin van a la primera y a la última."""
    tab.canvas.focus_force()
    pump(app)
    tab.canvas.event_generate("<Right>")
    pump(app)
    assert app.state.selected_position == 0
    tab.canvas.event_generate("<Right>")
    tab.canvas.event_generate("<Right>")
    pump(app)
    assert app.state.selected_position == 2
    tab.canvas.event_generate("<Left>")
    pump(app)
    assert tab.selected == 1
    tab.canvas.event_generate("<End>")
    pump(app)
    assert tab.selected == len(tab.notes) - 1
    tab.canvas.event_generate("<Right>")  # no pasa de la última
    pump(app)
    assert tab.selected == len(tab.notes) - 1
    tab.canvas.event_generate("<Home>")
    pump(app)
    assert tab.selected == 0


def test_comparison_with_ground_truth(app: Any, tab: Any) -> None:
    """Con la comparación activa: estados iguales a los de src.experiments, símbolos y pauta del ground truth."""
    from gui.tabs.tablature_tab import OUTCOME_SYMBOLS

    gt = app.state.analysis.ground_truth
    assert gt and tab.comparing
    for alg in ("ucb1", "softmax"):
        app.state.select_algorithm(alg)
        pump(app)
        expected = expected_outcomes(app, alg)
        assert [tab.outcome(i) for i in range(len(tab.notes))] == expected
        badges = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_withtag("badge")
                  if tab.canvas.type(i) == "text"]
        assert sorted(badges) == sorted(OUTCOME_SYMBOLS[o] for o in expected)
    assert len({t for i in tab.canvas.find_withtag("gtnote") for t in tab.canvas.gettags(i) if t.startswith("gt:")}) \
        == len(gt)
    assert "pitch" in tab.summary_label.cget("text") and "posición" in tab.summary_label.cget("text")
    height_on = int(tab.canvas.cget("height"))

    tab.chk_compare.invoke()
    pump(app)
    assert not tab.comparing
    assert not tab.canvas.find_withtag("gtnote") and not tab.canvas.find_withtag("badge")
    assert int(tab.canvas.cget("height")) < height_on
    assert tab.outcome(0) is None


def test_zoom_controls(app: Any, tab: Any) -> None:
    """El zoom escala las distancias horizontales; −/+ multiplican por 1.25; «Ajustar» hace caber la pista."""
    from gui.tabs.tablature_tab import MAX_ZOOM, MIN_ZOOM

    notes = tab.notes
    tab.set_zoom(100)
    d100 = tab.note_bbox(1)[0] - tab.note_bbox(0)[0]
    tab.set_zoom(300)
    d300 = tab.note_bbox(1)[0] - tab.note_bbox(0)[0]
    assert d100 == pytest.approx((notes[1].start_s - notes[0].start_s) * 100)
    assert d300 == pytest.approx(3 * d100)

    tab.set_zoom(100)
    tab.btn_zoom_in.invoke()
    assert tab.zoom == pytest.approx(125)
    assert float(tab.zoom_scale.get()) == pytest.approx(math.log2(125))
    assert tab.zoom_label.cget("text") == "125 px/s"
    tab.btn_zoom_out.invoke()
    assert tab.zoom == pytest.approx(100)
    tab.zoom_scale.set(math.log2(200))  # el deslizador (escala log₂) también cambia el zoom
    pump(app)
    assert tab.zoom == pytest.approx(200, rel=1e-3)

    tab.btn_fit.invoke()
    pump(app)
    assert tab._content_width <= tab._visible_width() + 1
    tab.set_zoom(1e6)
    assert tab.zoom == MAX_ZOOM
    tab.set_zoom(1.0)
    assert tab.zoom == MIN_ZOOM


def test_ascii_view(app: Any, tab: Any) -> None:
    """La vista ASCII muestra render_ascii del algoritmo y, si se compara, la del ground truth."""
    from gui.tabs.tablature_tab import gt_tab_notes
    from src.tab import render_ascii

    tab.chk_ascii.invoke()
    pump(app, 0.2)
    assert tab.ascii_frame.winfo_ismapped() and not tab.graphic_frame.winfo_ismapped()
    assert tab.zoom_scale.instate(["disabled"])
    text = tab.ascii_text.get("1.0", "end-1c")
    assert text == tab.ascii_content()
    width = tab.ascii_line_width()
    assert render_ascii(app.state.transcriptions["ucb1"].notes, line_width=width) in text
    assert render_ascii(gt_tab_notes(app.state.analysis.ground_truth), line_width=width) in text
    assert text.splitlines()[0].startswith("UCB1 — linea_simple.mp3")
    assert max(len(line) for line in text.splitlines()) <= width

    app.state.select_algorithm("softmax")
    pump(app)
    assert render_ascii(app.state.transcriptions["softmax"].notes, line_width=width) in \
        tab.ascii_text.get("1.0", "end-1c")
    tab.chk_compare.invoke()
    pump(app)
    assert "Ground truth" not in tab.ascii_text.get("1.0", "end-1c")

    tab.chk_ascii.invoke()
    pump(app)
    assert tab.graphic_frame.winfo_ismapped() and not tab.ascii_frame.winfo_ismapped()
    assert tab.zoom_scale.instate(["!disabled"])


def test_export_buttons(app: Any, tab: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """[TXT] [JSON] [CSV] guardan la tablatura del algoritmo mostrado con src.tab.export_tab."""
    from tkinter import filedialog

    calls: list[dict[str, Any]] = []

    def fake_save(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / kwargs["initialfile"])

    monkeypatch.setattr(filedialog, "asksaveasfilename", fake_save)
    notes = app.state.transcriptions["ucb1"].notes
    for button in tab.export_buttons.values():
        button.invoke()
    assert [c["initialfile"] for c in calls] == [f"linea_simple_tab_ucb1.{ext}"
                                                 for ext in ("txt", "json", "csv", "mid")]

    txt = (tmp_path / "linea_simple_tab_ucb1.txt").read_text(encoding="utf-8")
    assert txt.startswith("Smartuner · UCB1 · linea_simple.mp3")
    assert "G|" in txt and "E|" in txt
    data = json.loads((tmp_path / "linea_simple_tab_ucb1.json").read_text(encoding="utf-8"))
    assert data["meta"]["algorithm"] == "ucb1"
    assert 0.0 <= data["meta"]["position_accuracy"] <= 1.0
    assert [n["label"] for n in data["notes"]] == [n.label for n in notes]
    with (tmp_path / "linea_simple_tab_ucb1.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["string"], int(r["fret"])) for r in rows] == [(n.string, n.fret) for n in notes]
    # MIDI: una nota por nota de la tablatura, con su pitch y su tiempo de inicio.
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(tmp_path / "linea_simple_tab_ucb1.mid"))
    played = midi.instruments[0].notes
    assert [n.pitch for n in played] == [n.midi for n in notes]
    assert [n.start for n in played] == pytest.approx([n.start_s for n in notes], abs=2e-3)  # resolución MIDI
    assert "guardada" in app.status.message.cget("text")

    # Cancelar el diálogo no escribe nada.
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_k: "")
    assert tab.export_current("txt") is None


def test_export_all_menu(app: Any, tab: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """«Exportar las 4…» escribe un archivo por algoritmo (y por formato elegido) en la carpeta."""
    from tkinter import filedialog

    from src.config import ALGORITHMS

    menu = tab.export_all_menu
    labels = {menu.entrycget(i, "label"): i for i in range(menu.index("end") + 1) if menu.type(i) == "command"}
    monkeypatch.setattr(filedialog, "askdirectory", lambda **_k: str(tmp_path / "csv"))
    menu.invoke(labels["Como CSV…"])
    assert sorted(p.name for p in (tmp_path / "csv").iterdir()) == sorted(
        f"linea_simple_tab_{alg}.csv" for alg in ALGORITHMS)

    monkeypatch.setattr(filedialog, "askdirectory", lambda **_k: str(tmp_path / "todo"))
    menu.invoke(labels["En todos los formatos…"])
    files = sorted(p.name for p in (tmp_path / "todo").iterdir())
    assert len(files) == 16 and "linea_simple_tab_egreedy.mid" in files
    data = json.loads((tmp_path / "todo" / "linea_simple_tab_softmax.json").read_text(encoding="utf-8"))
    assert [n["label"] for n in data["notes"]] == [n.label for n in app.state.transcriptions["softmax"].notes]


def test_detail_panel(app: Any, tab: Any) -> None:
    """El panel inferior resume la nota y lista sus brazos con μ, Q y pulls de la cadena; dibuja el espectro."""
    from gui.tabs.tablature_tab import detail_text

    pos = 3
    tab.select(pos)
    wait_until(app, lambda: not tab.detail_pending, timeout=10)
    pump(app, 0.2)
    analysis = app.state.analysis
    chain = app.state.transcriptions["ucb1"].chain
    data = analysis.segment_data[pos]
    gt_note = tab.gt_note_for(pos)
    assert gt_note is not None
    expected = detail_text(pos, data.segment.start_s, data.segment.end_s, data.segment.f0_hz,
                           chain.prev_frets[pos], chain.arms[pos].label, gt_note.label, tab.outcome(pos))
    assert tab.detail_label.cget("text") == expected
    assert expected.startswith(f"Segmento {pos} · ")

    rows = tab.tree.get_children()
    assert len(rows) == data.n_arms
    mus = [float(tab.tree.set(r, "mu")) for r in rows]
    assert mus == sorted(mus, reverse=True)  # ordenados por μ real
    chosen = [r for r in rows if "chosen" in tab.tree.item(r, "tags")]
    assert len(chosen) == 1
    i = int(chosen[0][3:])
    assert tab.tree.set(chosen[0], "arm") == chain.arms[pos].label == data.arms[i].label
    assert float(tab.tree.set(chosen[0], "mu")) == pytest.approx(chain.true_means[pos][i], abs=5e-4)
    assert float(tab.tree.set(chosen[0], "q")) == pytest.approx(chain.final_q[pos][i], abs=5e-4)
    assert int(tab.tree.set(chosen[0], "n")) == int(chain.final_counts[pos][i])
    assert "elegido" in tab.tree.set(chosen[0], "role")
    assert any("★" in tab.tree.set(r, "role") for r in rows)
    assert sum(int(tab.tree.set(r, "n")) for r in rows) == chain.rewards.shape[1]  # T pulls en total

    fig = tab.plot.figure
    assert fig.axes and fig.axes[0].get_xscale() == "log"
    assert fig.legends

    app.state.select_segment(None, source="audio")
    pump(app)
    assert tab.tree.get_children() == ()
    assert "Ninguna nota seleccionada" in tab.detail_label.cget("text")


def test_busy_disables_export(app: Any, tab: Any) -> None:
    """Mientras hay una tarea en segundo plano no se puede exportar."""
    app.state.events.emit("busy_changed", busy=True)
    try:
        assert all(b.instate(["disabled"]) for b in tab.export_buttons.values())
        assert tab.export_all_button.instate(["disabled"])
    finally:
        app.state.events.emit("busy_changed", busy=False)
    assert all(b.instate(["!disabled"]) for b in tab.export_buttons.values())
    assert tab.export_all_button.instate(["!disabled"])


def test_transcriptions_ready_redraws(app: Any, tab: Any) -> None:
    """Al llegar nuevas transcripciones la tablatura se redibuja (y se vuelve a puntuar)."""
    import copy

    from src.experiments import transcribe

    original = dict(app.state.transcriptions)
    try:
        cfg = copy.deepcopy(app.state.analysis.config)
        cfg.env.budget = 20  # otra corrida muy corta: posiciones distintas
        new = {alg: transcribe(app.state.analysis, alg, cfg, run_index=3) for alg in original}
        app.state.transcriptions = new
        app.state.events.emit("transcriptions_ready", transcriptions=new)
        pump(app)
        assert labels_on_canvas(tab) == {n.position: str(n.fret) for n in new["ucb1"].notes}
        assert [tab.outcome(i) for i in range(len(tab.notes))] == expected_outcomes(app, "ucb1")
    finally:
        app.state.transcriptions = original
        app.state.events.emit("transcriptions_ready", transcriptions=original)
        pump(app)


def test_missing_transcription_shows_message(app: Any, tab: Any) -> None:
    """Si falta la transcripción de un algoritmo, su botón se deshabilita y el lienzo lo explica."""
    original = dict(app.state.transcriptions)
    try:
        app.state.transcriptions = {k: v for k, v in original.items() if k != "optimistic"}
        app.state.events.emit("transcriptions_ready", transcriptions=app.state.transcriptions)
        pump(app)
        assert tab.algo_radios["optimistic"].instate(["disabled"])
        app.state.select_algorithm("optimistic")
        pump(app)
        assert tab.view_state == "message"
        assert all(b.instate(["disabled"]) for b in tab.export_buttons.values())
        assert "sin transcripción" in tab.summary_label.cget("text")
    finally:
        app.state.transcriptions = original
        app.state.events.emit("transcriptions_ready", transcriptions=original)
        pump(app)
    app.state.select_algorithm("ucb1")
    pump(app)
    assert tab.view_state == "tab"


def test_config_changes(app: Any, tab: Any) -> None:
    """λ distinto → aviso «re-transcribir»; tolerancia de onset → nuevo emparejamiento con el ground truth."""
    from gui.tabs.tablature_tab import OUTCOME_EXTRA

    cfg = app.state.config
    old_lam, old_tol = cfg.env.lam, cfg.experiment.onset_tolerance_s
    try:
        cfg.env.lam = old_lam + 0.2
        app.state.events.emit("config_changed", key="env.lam")
        pump(app)
        assert tab.banner.winfo_ismapped()
        cfg.env.lam = old_lam
        app.state.events.emit("config_changed", key="env.lam")
        pump(app)
        assert not tab.banner.winfo_ismapped()

        cfg.experiment.onset_tolerance_s = 1e-4  # ningún segmento empareja: todos «de más»
        app.state.events.emit("config_changed", key="experiment.onset_tolerance_s")
        pump(app)
        assert all(tab.outcome(i) == OUTCOME_EXTRA for i in range(len(tab.notes)))
        assert "sin detectar" in tab.summary_label.cget("text")
        assert len(tab.canvas.find_withtag("missed")) == len(app.state.analysis.ground_truth)
    finally:
        cfg.env.lam, cfg.experiment.onset_tolerance_s = old_lam, old_tol
        app.state.events.emit("config_changed", key=None)
        pump(app)
    assert tab.outcome(0) == expected_outcomes(app, "ucb1")[0]


def test_small_window_uses_compact_layout(app: Any, tab: Any) -> None:
    """A 1024×700 la pestaña usa la geometría compacta, oculta la columna «Nota» y sigue funcionando."""
    big_height = int(tab.canvas.cget("height"))
    app.root.geometry("1024x700+0+0")
    try:
        wait_until(app, lambda: tab.compact, timeout=5)
        pump(app, 0.3)
        assert int(tab.canvas.cget("height")) < big_height
        assert "note" not in tab.tree.cget("displaycolumns")
        # La pestaña no pide más alto que la ventana: la barra de estado sigue visible.
        assert app.status.winfo_ismapped() and app.status.winfo_height() > 10
        assert tab.winfo_reqheight() <= tab.winfo_height() + 2
        click_note(tab, 2)
        wait_until(app, lambda: not tab.detail_pending, timeout=10)
        assert tab.selected == 2 and tab.plot.figure.axes
    finally:
        app.root.geometry("1360x880+0+0")
        wait_until(app, lambda: not tab.compact, timeout=5)
        pump(app, 0.2)
    assert int(tab.canvas.cget("height")) == big_height


# ---------------------------------------------------------------------------
# Escuchar la tablatura (MIDI sintetizado con cursor)
# ---------------------------------------------------------------------------


@pytest.fixture()
def listen(app: Any, tab: Any) -> Iterator[tuple[Any, Any, Any]]:
    """Reproductor de la pestaña con sounddevice y reloj FALSOS (sin tarjeta de sonido, deterministas).

    Devuelve ``(reproductor, sounddevice_falso, reloj_falso)``. La síntesis
    (Karplus-Strong, rápida) corre de verdad como tarea de la aplicación.
    """
    from gui.playback import SoundDeviceOutput
    from tests.test_gui_playback import FakeClock, FakeSoundDevice

    player = tab.playback
    saved = (player._output, player._clock)
    fake, clock = FakeSoundDevice(), FakeClock()
    player.stop()
    player._output = SoundDeviceOutput(fake)
    player._clock = clock
    player.method = "karplus-strong"
    tab.play_mode_var.set("midi")
    player.mode = "midi"
    tab.follow_var.set(True)
    yield player, fake, clock
    player.stop()
    player._output, player._clock = saved


def wait_playing(app: Any, player: Any, timeout: float = 60.0) -> None:
    """Espera a que termine la síntesis en segundo plano y empiece a sonar."""
    wait_until(app, lambda: player.state == "playing" and not app.busy, timeout=timeout)
    pump(app, 0.05)


def cursor_x(tab: Any) -> float:
    """x (lienzo) de la línea del cursor de reproducción."""
    line = next(i for i in tab.canvas.find_withtag("playcursor") if tab.canvas.type(i) == "line")
    return tab.canvas.coords(line)[0]


def test_listen_row_controls(app: Any, tab: Any) -> None:
    """La fila «Escuchar» tiene ▶, ■, tiempo, los cuatro modos, el sintetizador y «Seguir»."""
    from src.tab import PLAYBACK_MODES

    assert set(tab.mode_radios) == set(PLAYBACK_MODES)
    assert tab.btn_play.instate(["!disabled"]) and tab.btn_play.cget("text") == "▶ Reproducir"
    assert tab.btn_stop.instate(["disabled"])  # parado: nada que detener
    assert tab.play_time_label.cget("text").startswith("0:00.0 / 0:1")  # pista de ~11 s
    assert "Automática" in tab.synth_combo.cget("values")
    assert tab.follow_var.get() is True
    assert "regla" in tab.hint_label.cget("text") and "espacio" in tab.hint_label.cget("text")
    # Mientras hay otra tarea en curso no se puede empezar a escuchar.
    app.state.events.emit("busy_changed", busy=True)
    try:
        assert tab.btn_play.instate(["disabled"])
    finally:
        app.state.events.emit("busy_changed", busy=False)
    assert tab.btn_play.instate(["!disabled"])


def test_play_cursor_moves_and_highlights_the_sounding_note(app: Any, tab: Any, listen: Any) -> None:
    """▶ con una nota seleccionada empieza 1 s antes; el cursor avanza y la nota que suena se rodea en verde."""
    from gui.tabs.tablature_tab import PLAY_PREROLL_S

    player, fake, clock = listen
    notes = tab.notes
    tab.select(4)
    tab.btn_play.invoke()
    assert player.state == "preparing" and tab.btn_play.cget("text") == "Sintetizando…"
    wait_playing(app, player)
    start = notes[4].start_s - PLAY_PREROLL_S
    assert player.position_s == pytest.approx(start)
    assert len(fake.plays) == 1
    data, rate = fake.plays[0]
    assert rate == app.state.analysis.sr and data.ndim == 1  # modo MIDI: mono
    assert tab.btn_play.cget("text") == "⏸ Pausa" and tab.btn_stop.instate(["!disabled"])
    assert cursor_x(tab) == pytest.approx(tab.x_of(start))

    # El reloj avanza hasta el centro de la nota 4: el cursor la alcanza y se resalta.
    target = (notes[4].start_s + notes[4].end_s) / 2
    clock.advance(target - start)
    pump(app, 0.1)
    assert cursor_x(tab) == pytest.approx(tab.x_of(target))
    assert player.note_at(target) == 4 and tab._playing_pos == 4
    (ring,) = tab.canvas.find_withtag("playing")
    x0, y0, x1, y1 = tab.canvas.bbox(ring)
    bx0, by0, bx1, by1 = tab.note_bbox(4)
    assert x0 < bx0 and y0 < by0 and x1 > bx1 and y1 > by1  # rodea el recuadro de la nota
    assert app.state.selected_position == 4  # sonar no cambia la selección (no redibuja el espectro)
    assert "0:0" in tab.play_time_label.cget("text")

    # Espacio = pausa / reanudar (desde el mismo instante).
    tab.canvas.focus_force()
    tab.canvas.event_generate("<space>")
    pump(app, 0.05)
    assert player.state == "paused" and tab.btn_play.cget("text") == "▶ Reanudar"
    assert tab.canvas.find_withtag("playcursor")  # en pausa el cursor se queda
    tab.canvas.event_generate("<space>")
    pump(app, 0.05)
    assert player.state == "playing" and player.position_s == pytest.approx(target)
    assert len(fake.plays) == 2

    tab.btn_stop.invoke()
    pump(app, 0.05)
    assert player.state == "stopped"
    assert not tab.canvas.find_withtag("playcursor") and not tab.canvas.find_withtag("playing")
    assert tab.btn_play.cget("text") == "▶ Reproducir" and tab.btn_stop.instate(["disabled"])


def test_change_mode_and_algorithm_while_playing(app: Any, tab: Any, listen: Any) -> None:
    """Cambiar de modo (A/B estéreo) o de algoritmo mientras suena continúa desde el mismo instante."""
    player, fake, clock = listen
    tab.play_from(2.0)
    wait_playing(app, player)
    clock.advance(1.5)
    pump(app, 0.05)
    tab.mode_radios["estereo"].invoke()
    pump(app, 0.05)
    assert player.state == "playing" and player.mode == "estereo"
    data, rate = fake.plays[-1]
    assert data.ndim == 2 and data.shape[1] == 2  # original a la izquierda, MIDI a la derecha
    assert player.position_s == pytest.approx(3.5)
    total = int(round(player.duration_s * rate))
    assert data.shape[0] == total - int(round(3.5 * rate))

    # Otro algoritmo: la tablatura cambia y la reproducción sigue en el mismo instante.
    clock.advance(0.5)
    tab.algo_radios["softmax"].invoke()
    wait_until(app, lambda: player.state == "playing" and not app.busy, timeout=60)
    pump(app, 0.05)
    assert player.position_s == pytest.approx(4.0, abs=0.05)
    assert [n.position for n in player.notes] == [n.position for n in app.state.transcriptions["softmax"].notes]
    assert tab.canvas.find_withtag("playcursor")


def test_click_on_ruler_plays_from_there(app: Any, tab: Any, listen: Any) -> None:
    """Un clic en la regla de tiempo escucha desde ese instante (y salta si ya sonaba)."""
    from gui.tabs.tablature_tab import X0

    player, _fake, _clock = listen
    tab.canvas.xview_moveto(0.0)
    pump(app, 0.05)
    x = int(X0 + 3.0 * tab.zoom)
    tab.canvas.event_generate("<Button-1>", x=x, y=5)
    tab.canvas.event_generate("<ButtonRelease-1>", x=x, y=5)
    wait_playing(app, player)
    assert player.position_s == pytest.approx(3.0, abs=1.0 / tab.zoom)
    # Ya sonando: otro clic salta al nuevo instante sin volver a sintetizar.
    x = int(X0 + 6.0 * tab.zoom - tab.canvas.canvasx(0))
    tab.canvas.event_generate("<Button-1>", x=x, y=5)
    pump(app, 0.05)
    assert player.state == "playing" and player.position_s == pytest.approx(6.0, abs=2.0 / tab.zoom)
    # Un clic fuera de la regla (sobre las cuerdas) no reproduce desde ahí.
    tab.stop_playback()
    tab.canvas.event_generate("<Button-1>", x=x, y=int(tab._layout.main_lines["E"]))
    pump(app, 0.05)
    assert player.state == "stopped"


def test_follow_pages_the_view_with_the_cursor(app: Any, tab: Any, listen: Any) -> None:
    """Con «Seguir», cuando el cursor pasa del 85 % del ancho visible la vista avanza una página."""
    from gui.tabs.tablature_tab import FOLLOW_MARGIN

    player, _fake, clock = listen
    tab.set_zoom(600)
    tab.canvas.xview_moveto(0.0)
    pump(app, 0.1)
    tab.play_from(0.0)
    wait_playing(app, player)
    left, right = tab.visible_range()
    t_edge = (left + FOLLOW_MARGIN * (right - left) + 20 - tab.x_of(0)) / tab.zoom
    clock.advance(t_edge)
    pump(app, 0.1)
    new_left, new_right = tab.visible_range()
    x = cursor_x(tab)
    assert new_left > left and new_left <= x <= new_left + 0.2 * (new_right - new_left)
    # Sin «Seguir» la vista no se mueve.
    tab.follow_var.set(False)
    clock.advance((new_right - new_left) / tab.zoom)
    pump(app, 0.1)
    assert tab.visible_range()[0] == pytest.approx(new_left)


def test_only_one_player_at_a_time(app: Any, tab: Any, listen: Any) -> None:
    """La pestaña Audio y la Tablatura comparten la salida: al reproducir una, la otra se detiene o pausa."""
    from gui.widgets import AudioPlayer

    player, fake, clock = listen
    audio_tab = app.tabs["audio"]
    saved = audio_tab.player
    audio_tab.player = AudioPlayer(fake)
    try:
        tab.play_from(1.0)
        wait_playing(app, player)
        clock.advance(0.7)
        audio_tab._on_play_all()  # «▶ Todo» de la pestaña Audio
        pump(app, 0.05)
        assert player.state == "paused" and player.position_s == pytest.approx(1.7)
        assert len(fake.plays) == 2 and audio_tab.player.is_active()  # suena el original completo
        stops = fake.stops
        tab.toggle_playback()  # reanudar la tablatura: la pestaña Audio se calla
        pump(app, 0.05)
        assert player.state == "playing" and fake.stops > stops and not audio_tab.player.is_active()
        assert player.position_s == pytest.approx(1.7)
    finally:
        audio_tab.player = saved


def test_new_analysis_stops_playback(app: Any, tab: Any, listen: Any) -> None:
    """Un análisis nuevo (o una re-transcripción) detiene la reproducción y borra el cursor."""
    player, _fake, _clock = listen
    tab.play_from(0.5)
    wait_playing(app, player)
    app.state.events.emit("analysis_ready", analysis=app.state.analysis)
    app.state.events.emit("transcriptions_ready", transcriptions=app.state.transcriptions)
    pump(app, 0.1)
    assert player.state == "stopped" and not tab.canvas.find_withtag("playcursor")


def test_play_without_sounddevice_opens_a_wav(app: Any, tab: Any, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin salida de audio, ▶ explica el motivo en su tooltip y abre un WAV con el reproductor del sistema."""
    import sys

    from gui.playback import SoundDeviceOutput

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    player = tab.playback
    saved = (player._output, player._opener, player.export_dir)
    opened: list[Path] = []
    player.stop()
    player._output, player._opener, player.export_dir = SoundDeviceOutput(), opened.append, tmp_path
    player.method = "karplus-strong"
    try:
        assert "reproductor del sistema" in tab._play_help()
        tab.play_from(0.0)
        wait_until(app, lambda: bool(opened) and not app.busy, timeout=60)
        assert opened[0].suffix == ".wav" and opened[0].parent == tmp_path
        assert opened[0].name.startswith("linea_simple_")
        assert (tmp_path / f"{opened[0].name.split('_midi')[0]}.mid").exists()
        assert player.state == "stopped" and not tab.canvas.find_withtag("playcursor")
        assert "no se sincroniza" in app.status.message.cget("text")
    finally:
        player._output, player._opener, player.export_dir = saved


def test_new_analysis_that_starts_late_opens_on_the_first_note(app: Any, tab: Any) -> None:
    """Regresión (Money: primera nota a 12.6 s): tras un análisis NUEVO la vista muestra la primera nota, no un
    pentagrama vacío; una re-transcripción (misma señal) no mueve la vista."""
    import dataclasses

    import numpy as np

    original_analysis, original_tr = app.state.analysis, dict(app.state.transcriptions)
    shift = 40.0
    sr = original_analysis.sr
    pad = np.zeros(int(shift * sr), dtype=original_analysis.y_raw.dtype)

    def moved(segment: Any) -> Any:
        return dataclasses.replace(segment, start_s=segment.start_s + shift, end_s=segment.end_s + shift,
                                   start_sample=segment.start_sample + pad.size,
                                   end_sample=segment.end_sample + pad.size)

    late = dataclasses.replace(
        original_analysis, y_raw=np.concatenate([pad, original_analysis.y_raw]),
        segments=[moved(s) for s in original_analysis.segments],
        segment_data=[dataclasses.replace(d, segment=moved(d.segment)) for d in original_analysis.segment_data],
        ground_truth=None)
    late_tr = {alg: dataclasses.replace(tr, notes=[dataclasses.replace(n, start_s=n.start_s + shift,
                                                                       end_s=n.end_s + shift) for n in tr.notes])
               for alg, tr in original_tr.items()}
    try:
        app.state.analysis, app.state.transcriptions = late, late_tr
        app.state.events.emit("analysis_ready", analysis=late)
        app.state.events.emit("transcriptions_ready", transcriptions=late_tr)
        pump(app, 0.3)
        left, right = tab.visible_range()
        first = min(tab._note_boxes.values(), key=lambda box: box[0])
        assert left > 0 and left <= first[0] <= right
        # Re-transcripción (misma señal): la vista se queda donde el usuario la dejó.
        tab.canvas.xview_moveto(0.0)
        pump(app, 0.1)
        retranscribed = dataclasses.replace(late)
        app.state.analysis = retranscribed
        app.state.events.emit("analysis_ready", analysis=retranscribed)
        app.state.events.emit("transcriptions_ready", transcriptions=late_tr)
        pump(app, 0.3)
        assert tab.visible_range()[0] == 0
    finally:
        app.state.analysis, app.state.transcriptions = original_analysis, original_tr
        app.state.events.emit("analysis_ready", analysis=original_analysis)
        app.state.events.emit("transcriptions_ready", transcriptions=original_tr)
        pump(app, 0.3)


def test_late_first_note_is_revealed_when_tab_was_hidden_during_analysis(app: Any, tab: Any) -> None:
    """Regresión: si el análisis termina con la pestaña Tablatura OCULTA (lo normal al abrir el MP3 desde la
    pestaña Audio), al mostrarla la vista salta igualmente a la primera nota (antes se quedaba en 0 s)."""
    import dataclasses

    import numpy as np

    original_analysis, original_tr = app.state.analysis, dict(app.state.transcriptions)
    shift = 40.0
    pad = np.zeros(int(shift * original_analysis.sr), dtype=original_analysis.y_raw.dtype)

    def moved(segment: Any) -> Any:
        return dataclasses.replace(segment, start_s=segment.start_s + shift, end_s=segment.end_s + shift,
                                   start_sample=segment.start_sample + pad.size,
                                   end_sample=segment.end_sample + pad.size)

    late = dataclasses.replace(
        original_analysis, y_raw=np.concatenate([pad, original_analysis.y_raw]),
        segments=[moved(s) for s in original_analysis.segments],
        segment_data=[dataclasses.replace(d, segment=moved(d.segment)) for d in original_analysis.segment_data],
        ground_truth=None)
    late_tr = {alg: dataclasses.replace(tr, notes=[dataclasses.replace(n, start_s=n.start_s + shift,
                                                                       end_s=n.end_s + shift) for n in tr.notes])
               for alg, tr in original_tr.items()}
    try:
        app.show_tab("audio")
        pump(app, 0.2)
        app.state.analysis, app.state.transcriptions = late, late_tr
        app.state.events.emit("analysis_ready", analysis=late)
        app.state.events.emit("transcriptions_ready", transcriptions=late_tr)
        pump(app, 0.3)
        app.show_tab("tab")
        pump(app, 0.5)
        left, right = tab.visible_range()
        first = min(tab._note_boxes.values(), key=lambda box: box[0])
        assert left > 0 and left <= first[0] <= right
    finally:
        app.show_tab("tab")
        app.state.analysis, app.state.transcriptions = original_analysis, original_tr
        app.state.events.emit("analysis_ready", analysis=original_analysis)
        app.state.events.emit("transcriptions_ready", transcriptions=original_tr)
        pump(app, 0.3)
