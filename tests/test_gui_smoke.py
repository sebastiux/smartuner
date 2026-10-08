"""Prueba de humo de la GUI completa: el flujo de un estudiante, de principio a fin, en < 1 min.

Abre la aplicación REAL (:class:`gui.app.SmartunerApp`, con las seis
pestañas) en una ventana pequeña (1024×700, la mínima admitida) y recorre
una versión rápida del flujo de trabajo:

1. estado vacío de las seis pestañas (sin errores y con la barra de estado visible);
2. abrir ``data/synthetic/linea_simple.mp3`` (análisis + cuatro transcripciones en
   el hilo trabajador);
3. selección de una nota sincronizada entre Audio, Ejecución en vivo y Tablatura;
4. Configuración: cambiar λ → «Aplicar y re-transcribir»;
5. Ejecución en vivo: pasos y corrida completa;
6. Tablatura: escuchar con un ``sounddevice`` FALSO (el cursor avanza y resalta la
   nota que suena) y exportar el MIDI;
7. Comparación: experimento corto (2 corridas, sin barridos) → tabla y gráficas;
8. archivo corrupto → diálogo de error y la aplicación sigue usable;
9. cerrar la ventana con una tarea en curso, sin excepciones.

Durante todo el recorrido se recogen las excepciones de los callbacks de
tkinter y los mensajes de ``logging`` de nivel ERROR: la prueba falla si hay
alguno que no sea el error esperado del archivo corrupto. Los diálogos
modales (``messagebox``/``filedialog``) se sustituyen por funciones que
responden solas.

Necesita tkinter y un servidor X (sin ``DISPLAY`` se omite)::

    xvfb-run -a -s "-screen 0 1400x900x24" .venv/bin/python -m pytest tests/test_gui_smoke.py -q
"""

from __future__ import annotations

import logging
import os
import time
import traceback
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
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_TK, reason="tkinter no está disponible"),
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="sin servidor X (DISPLAY); usa xvfb-run"),
    pytest.mark.skipif(not AUDIO.exists(), reason=f"falta {AUDIO.name} (genera el dataset sintético)"),
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def pump(app: Any, seconds: float = 0.1) -> None:
    """Procesa eventos de tkinter durante ``seconds`` (``after``, sondeo de tareas, redibujos)."""
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
    """Espera a que termine la tarea en segundo plano (y a que se procesen sus eventos)."""
    pump(app, 0.1)
    wait_until(app, lambda: not app.busy, timeout)
    pump(app, 0.2)


class _ErrorRecords(logging.Handler):
    """Guarda los mensajes de logging de nivel ERROR o superior."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


@pytest.fixture()
def dialogs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, list[Any]]]:
    """Sustituye los diálogos modales: registra los mensajes y contesta «sí» y rutas de ``tmp_path``."""
    from tkinter import filedialog, messagebox

    seen: dict[str, list[Any]] = {"error": [], "info": [], "ask": []}
    monkeypatch.setattr(messagebox, "showerror", lambda title, msg, **_k: seen["error"].append((title, str(msg))))
    monkeypatch.setattr(messagebox, "showinfo", lambda title, msg, **_k: seen["info"].append((title, str(msg))))
    monkeypatch.setattr(messagebox, "askyesno", lambda title, msg, **_k: seen["ask"].append(title) or True)
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(tmp_path / k.get("initialfile", "x.json")))
    monkeypatch.setattr(filedialog, "askdirectory", lambda **_k: str(tmp_path / "carpeta"))
    yield seen


# ---------------------------------------------------------------------------
# Prueba
# ---------------------------------------------------------------------------


def test_student_flow_end_to_end(dialogs: dict[str, list[Any]], tmp_path: Path) -> None:
    """Recorre el flujo completo del estudiante en una ventana de 1024×700 sin ninguna excepción."""
    from gui.app import SmartunerApp
    from gui.playback import SoundDeviceOutput
    from tests.test_gui_playback import FakeClock, FakeSoundDevice

    errors = _ErrorRecords()
    logging.getLogger().addHandler(errors)
    callback_errors: list[str] = []
    app = SmartunerApp()
    app.root.report_callback_exception = lambda exc, val, tb: callback_errors.append(
        "".join(traceback.format_exception(exc, val, tb)))
    closed = False
    try:
        app.root.geometry("1024x700+0+0")
        pump(app, 0.3)

        # 1. Estado vacío: cada pestaña se muestra y la barra de estado no desaparece.
        for key in app.tabs:
            app.show_tab(key)
            pump(app, 0.15)
            assert app.status.winfo_ismapped() and app.status.winfo_height() > 10, key
        assert app.tabs["tab"].view_state == "empty"

        # 2. Abrir y analizar un MP3 del dataset.
        app.open_audio(AUDIO)
        wait_idle(app)
        analysis = app.state.analysis
        assert analysis is not None and len(analysis.segment_data) == 16
        assert set(app.state.transcriptions) == {"egreedy", "optimistic", "ucb1", "softmax"}
        for key in app.tabs:
            app.show_tab(key)
            pump(app, 0.2)
        # Regresión: a 1024×700 la figura de la pestaña Audio no se queda por debajo de su alto
        # mínimo cuando el panel de información crece con los datos del análisis.
        from gui.tabs.audio_tab import MIN_PLOT_PX

        app.show_tab("audio")
        pump(app, 0.3)
        assert app.tabs["audio"].paned.sashpos(0) >= MIN_PLOT_PX - 2

        # 3. Selección sincronizada: fila de la pestaña Audio → En vivo y Tablatura.
        audio, live, tab = app.tabs["audio"], app.tabs["live"], app.tabs["tab"]
        app.show_tab("audio")
        audio.tree.selection_set("pos5")
        audio.tree.event_generate("<<TreeviewSelect>>")
        pump(app, 0.2)
        assert app.state.selected_position == 5 and tab.selected == 5
        app.show_tab("live")
        pump(app, 0.4)
        assert live.session is not None and live.session.data.position == 5
        live.next_button.invoke()
        pump(app, 0.1)
        assert app.state.selected_position == 6 and tab.selected == 6
        assert audio.tree.selection() == ("pos6",)

        # 4. Configuración: λ → «Aplicar y re-transcribir».
        config_tab = app.tabs["config"]
        app.show_tab("config")
        field = config_tab.fields["env.lam"]
        field.var.set("0.5")
        field._changed()  # lo que hace el control al perder el foco o con Intro
        pump(app, 0.1)
        assert app.state.config.env.lam == 0.5
        assert "re-transcribir" in config_tab.status_banner.title.cget("text").lower()
        config_tab.status_banner.button.invoke()
        wait_idle(app)
        assert app.state.analysis.config.env.lam == 0.5
        assert "re-transcribir" not in config_tab.status_banner.title.cget("text").lower()

        # 5. Ejecución en vivo: pasos y corrida completa.
        app.show_tab("live")
        pump(app, 0.3)
        live.step_button.invoke()
        live.step_button.invoke()
        assert live.session.t == 2
        live.run_to_end()
        pump(app, 0.2)
        assert live.session.done

        # 6. Tablatura: escuchar (sounddevice falso) y exportar el MIDI.
        app.show_tab("tab")
        pump(app, 0.3)
        fake, clock = FakeSoundDevice(), FakeClock()
        player = tab.playback
        player._output, player._clock = SoundDeviceOutput(fake), clock
        player.method = "karplus-strong"
        tab.select(3)
        tab.btn_play.invoke()
        wait_until(app, lambda: player.state == "playing" and not app.busy, timeout=60)
        x_start = tab._play_x
        note = tab.notes[3]
        clock.advance(1.0 + (note.end_s - note.start_s) / 2)
        pump(app, 0.15)
        assert tab._play_x > x_start and tab._playing_pos == 3 and tab.canvas.find_withtag("playing")
        tab.mode_radios["estereo"].invoke()
        pump(app, 0.1)
        assert player.state == "playing" and fake.plays[-1][0].ndim == 2
        tab.btn_stop.invoke()
        assert player.state == "stopped" and not tab.canvas.find_withtag("playcursor")
        tab.export_buttons["mid"].invoke()
        assert (tmp_path / "linea_simple_tab_ucb1.mid").exists()

        # 7. Comparación: experimento corto → tabla y gráficas.
        exp = app.state.config.experiment
        exp.n_runs, exp.run_sweeps, exp.run_lambda_sweep = 2, False, False
        app.state.events.emit("config_changed", key=None)
        compare = app.tabs["compare"]
        app.show_tab("compare")
        compare.run_button.invoke()
        wait_idle(app, timeout=180)
        assert app.state.experiment is not None and len(compare.table.get_children()) >= 4
        for i in range(3):
            compare.select_plot(i)
            pump(app, 0.15)
            assert compare.plot.figure.axes
        assert compare.export_table_csv(tmp_path / "tabla.csv") is not None

        # 8. Archivo corrupto → un diálogo de error; el análisis anterior sigue cargado y usable.
        bad = tmp_path / "corrupto.mp3"
        bad.write_bytes(b"ID3\x03\x00\x00\x00" + bytes(range(256)) * 20)
        previous = app.state.analysis
        app.open_audio(bad)
        wait_idle(app)
        assert len(dialogs["error"]) == 1 and "analizar" in dialogs["error"][0][0].lower()
        assert app.state.analysis is previous
        app.show_tab("tab")
        tab.select(1)
        pump(app, 0.2)
        assert app.state.selected_position == 1

        unexpected = [m for m in errors.messages if "corrupto" not in m and "analizar" not in m.lower()]
        assert not unexpected, unexpected
        assert not callback_errors, callback_errors

        # 9. Cerrar con una tarea en curso: sin excepciones.
        app.run_experiment()
        pump(app, 0.3)
        assert app.busy
        app.quit()
        closed = True
        assert not callback_errors, callback_errors
    finally:
        logging.getLogger().removeHandler(errors)
        if not closed:
            app.quit()
