"""Pruebas de las piezas comunes de la GUI (:mod:`gui.widgets`, :mod:`gui.workers`).

* Sin pantalla: ejemplos ``>>>`` de :mod:`gui.widgets`, el reparto resumen /
  detalle de los mensajes de error y el intervalo del GIL de
  :class:`gui.workers.WorkerManager` (con un «root» falso).
* Con pantalla (``DISPLAY``; se omiten sin ella): :class:`ParamField` con
  varias ventanas ``tk.Tk``, la rueda del ratón sobre un control dentro de un
  :class:`ScrollableFrame`, ``set_enabled``/``show_error``, los tooltips
  (junto al puntero y según la zona), las ayudas de ``Treeview`` y
  :class:`AudioPlayer` con un ``sounddevice`` FALSO.

::

    xvfb-run -a -s "-screen 0 1400x900x24" .venv/bin/python -m pytest tests/test_gui_widgets.py -q
"""

from __future__ import annotations

import doctest
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

try:
    import tkinter as tk
    from tkinter import ttk

    HAS_TK = True
except ImportError:  # pragma: no cover - depende de la instalación de Python
    HAS_TK = False

pytestmark = pytest.mark.skipif(not HAS_TK, reason="tkinter no está disponible")

needs_display = [
    pytest.mark.gui,
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="sin servidor X (DISPLAY); usa xvfb-run"),
]


def gui_test(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Marca una prueba que abre ventanas."""
    for mark in reversed(needs_display):
        fn = mark(fn)
    return fn


# ---------------------------------------------------------------------------
# Sin pantalla
# ---------------------------------------------------------------------------


def test_docstring_examples() -> None:
    """Los ejemplos ``>>>`` de :mod:`gui.widgets` se cumplen."""
    import gui.widgets as module

    result = doctest.testmod(module)
    assert result.attempted > 0 and result.failed == 0


def test_split_error_message() -> None:
    """El primer párrafo (o línea) es el resumen; lo demás, el detalle; un resumen enorme se corta."""
    from gui.widgets import split_error_message

    assert split_error_message("Uno.") == ("Uno.", "")
    assert split_error_message("Resumen.\n\nDetalle 1\nDetalle 2") == ("Resumen.", "Detalle 1\nDetalle 2")
    long = "palabra " * 100
    summary, detail = split_error_message(long, max_chars=80)
    assert len(summary) <= 81 and summary.endswith("…") and detail.startswith("…")


class _FakeRoot:
    """«Root» mínimo para WorkerManager: ``after`` guarda los callbacks y la prueba los ejecuta."""

    def __init__(self) -> None:
        self.jobs: list[Callable[[], None]] = []

    def after(self, _ms: int, fn: Callable[[], None]) -> str:
        self.jobs.append(fn)
        return f"after#{len(self.jobs)}"

    def run(self) -> None:
        jobs, self.jobs = self.jobs, []
        for fn in jobs:
            fn()


def test_worker_manager_lowers_and_restores_the_gil_switch_interval() -> None:
    """Mientras hay tareas el intervalo del GIL baja (la GUI no frena al trabajador) y luego se restaura."""
    from gui.workers import WORKER_SWITCH_INTERVAL_S, WorkerManager

    before = sys.getswitchinterval()
    root = _FakeRoot()
    manager = WorkerManager(root)  # type: ignore[arg-type]
    release = threading.Event()
    done: list[Any] = []
    manager.start("prueba", lambda _p, _c: release.wait(5) and 42, on_done=done.append)
    try:
        assert sys.getswitchinterval() == pytest.approx(min(before, WORKER_SWITCH_INTERVAL_S))
        assert manager.busy
    finally:
        release.set()
    end = time.time() + 5
    while manager.busy and time.time() < end:
        time.sleep(0.01)
        root.run()
    assert done == [42] and not manager.busy
    assert sys.getswitchinterval() == pytest.approx(before)


# ---------------------------------------------------------------------------
# Con pantalla
# ---------------------------------------------------------------------------


@pytest.fixture()
def root() -> Any:
    """Ventana raíz temporal (se destruye al terminar)."""
    window = tk.Tk()
    window.geometry("500x300+0+0")
    yield window
    window.destroy()


def _update(widget: Any, seconds: float = 0.05) -> None:
    end = time.time() + seconds
    while time.time() < end:
        widget.update()
        time.sleep(0.005)


@gui_test
def test_param_field_variables_belong_to_their_own_window(root: Any) -> None:
    """Con dos ``tk.Tk`` (como en las pruebas) las variables del control son de SU ventana."""
    from gui.widgets import ParamField
    from src.config import PARAM_SPECS

    other = tk.Tk()  # la PRIMERA ventana creada sería el master por defecto de una variable sin master
    try:
        field = ParamField(root, "env.lam", PARAM_SPECS["env.lam"], 0.1, label_width=30, unit_width=8)
        assert str(field.var._root) == str(root)  # noqa: SLF001
        assert int(field.label.cget("width")) == 30 and int(field.unit.cget("width")) == 8
        flag = ParamField(root, "experiment.run_sweeps", PARAM_SPECS["experiment.run_sweeps"], True)
        assert str(flag.var._root) == str(root) and flag.parse() is True  # noqa: SLF001
    finally:
        other.destroy()


@gui_test
def test_param_field_errors_and_enabled_state(root: Any) -> None:
    """``show_error`` muestra un aviso externo; ``set_enabled`` deshabilita y atenúa; los valores se validan."""
    from gui.widgets import UI_TEXT_MUTED, ParamField
    from src.config import PARAM_SPECS

    changes: list[tuple[str, Any]] = []
    field = ParamField(root, "env.lam", PARAM_SPECS["env.lam"], 0.1, on_change=lambda k, v: changes.append((k, v)))
    field.pack()
    field.var.set("0.3")
    field._changed()
    assert changes == [("env.lam", 0.3)]
    field.var.set("abc")
    field._changed()
    assert changes == [("env.lam", 0.3)] and "no es un número" in field.error.cget("text")
    field.show_error("Fuera de la lista de barrido")
    assert field.error.winfo_manager() == "grid" and "barrido" in field.error.cget("text")
    field.show_error(None)
    assert field.error.winfo_manager() == ""
    field.set_enabled(False)
    assert not field.enabled and field.control.instate(["disabled"])
    assert str(field.label.cget("foreground")) == UI_TEXT_MUTED
    field.set_enabled(True)
    assert field.enabled and field.control.instate(["!disabled"])


@gui_test
def test_mouse_wheel_scrolls_the_form_instead_of_changing_the_value(root: Any) -> None:
    """La rueda sobre un Spinbox dentro de un ScrollableFrame desplaza la página y NO cambia el valor."""
    from gui.widgets import ParamField, ScrollableFrame
    from src.config import PARAM_SPECS

    scroller = ScrollableFrame(root)
    scroller.pack(fill="both", expand=True)
    fields = [ParamField(scroller.inner, "env.budget", PARAM_SPECS["env.budget"], 500) for _ in range(30)]
    for field in fields:
        field.pack(anchor="w")
    _update(root, 0.2)
    first = fields[0]
    assert scroller.canvas.yview()[0] == 0.0
    first.control.event_generate("<Button-5>")  # rueda hacia abajo en X11
    _update(root)
    assert first.var.get() == "500"  # el valor no cambió
    assert scroller.canvas.yview()[0] > 0.0  # la página sí se desplazó


@gui_test
def test_tooltip_appears_next_to_the_pointer_and_stays_on_screen(root: Any) -> None:
    """El tooltip se coloca junto al puntero (no debajo de todo el widget) y dentro de la pantalla."""
    from gui.widgets import Tooltip

    tall = tk.Canvas(root, width=400, height=280)
    tall.pack()
    _update(root, 0.2)
    tip = Tooltip(tall, "Ayuda", delay_ms=1)
    root.event_generate("<Motion>", warp=True, x=40, y=30)
    _update(root, 0.1)
    tip._show()
    assert tip._tip is not None
    _update(root, 0.1)
    x, y = tip._tip.winfo_rootx(), tip._tip.winfo_rooty()
    px, py = tall.winfo_pointerx(), tall.winfo_pointery()
    assert 0 < x - px <= 20 and 0 < y - py <= 30  # abajo a la derecha del puntero
    assert y < tall.winfo_rooty() + tall.winfo_height()  # no debajo de todo el lienzo
    tip.hide()
    assert tip._tip is None
    # Cerca del borde de la pantalla se coloca dentro.
    width, height = tip._position(300, 80)
    assert width >= 0 and height >= 0


@gui_test
def test_hover_tooltip_text_depends_on_the_zone(root: Any) -> None:
    """HoverTooltip: el texto depende de la zona bajo el puntero; al cambiar de zona se reinicia."""
    from gui.widgets import HoverTooltip

    canvas = tk.Canvas(root, width=400, height=200)
    canvas.pack()
    _update(root, 0.2)
    tip = HoverTooltip(canvas, lambda x, _y: "izq" if x < 200 else None, lambda key: f"zona {key}", delay_ms=1)
    root.event_generate("<Motion>", warp=True, x=50, y=50)
    _update(root, 0.1)
    assert tip.pointer_key() == "izq" and tip.current_text() == "zona izq"
    root.event_generate("<Motion>", warp=True, x=300, y=50)
    _update(root, 0.1)
    assert tip.pointer_key() is None and tip.current_text() == ""


@gui_test
def test_tree_helpers_resolve_visible_columns_and_headings(root: Any) -> None:
    """``tree_display_columns`` resuelve ``#all`` y ``tree_heading_at`` la columna bajo un punto del encabezado."""
    from gui.widgets import tree_display_columns, tree_heading_at

    tree = ttk.Treeview(root, columns=("a", "b", "c"), show="headings")
    for col in ("a", "b", "c"):
        tree.heading(col, text=col.upper())
        tree.column(col, width=100, stretch=False)
    tree.insert("", "end", values=(1, 2, 3))
    tree.pack(anchor="nw")
    _update(root, 0.2)
    assert tree_display_columns(tree) == ["a", "b", "c"]
    assert tree_heading_at(tree, 150, 8) == "b"
    assert tree_heading_at(tree, 150, 40) is None  # sobre una fila, no sobre el encabezado
    tree.configure(displaycolumns=("c", "a"))
    _update(root)
    assert tree_display_columns(tree) == ["c", "a"]
    assert tree_heading_at(tree, 50, 8) == "c"


@gui_test
def test_audio_player_uses_the_shared_output(root: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """AudioPlayer reproduce con sounddevice (remuestrea si el dispositivo lo exige) y explica por qué no puede."""
    from gui.widgets import AudioPlayer
    from tests.test_gui_playback import FakeSoundDevice

    fake = FakeSoundDevice(reject_rates=(8000,))
    player = AudioPlayer(fake)
    assert player.available and player.error == ""
    player.play(np.zeros(8000, dtype=np.float32), 8000)
    (data, rate), = fake.plays
    assert rate == 44100 and data.shape == (44100,)
    assert player.is_active() is True
    player.stop()
    assert fake.stops >= 2 and player.is_active() is None

    monkeypatch.setitem(sys.modules, "sounddevice", None)  # sin sounddevice (o sin PortAudio)
    missing = AudioPlayer()
    assert not missing.available and "sounddevice" in missing.error
    with pytest.raises(RuntimeError):
        missing.play(np.zeros(10, dtype=np.float32), 8000)
    missing.stop()  # no falla


@gui_test
def test_status_bar_long_message_keeps_cancel_visible(root: Any) -> None:
    """Un mensaje larguísimo (una ruta) se recorta: «Cancelar» y la barra de progreso siguen visibles."""
    from gui.widgets import StatusBar

    bar = StatusBar(root)
    bar.pack(side="bottom", fill="x")
    bar.set_message("Se guardaron 16 archivos en " + "/carpeta/muy/larga" * 30)
    _update(root, 0.2)
    width = bar.winfo_width()
    for widget in (bar.cancel_button, bar.progress):
        assert widget.winfo_ismapped()
        assert widget.winfo_x() + widget.winfo_width() <= width
    assert bar.winfo_reqwidth() < 600  # el texto no agranda la barra
