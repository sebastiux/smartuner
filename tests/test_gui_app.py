"""Pruebas de :mod:`gui.app`: tamaño de la ventana según la pantalla y cancelación de las tareas.

Las pruebas con ventana necesitan tkinter y un servidor X (``xvfb-run``); sin
``DISPLAY`` se omiten. Las tareas de análisis se sustituyen por funciones
falsas e instantáneas: aquí se prueba la lógica de la aplicación (qué se
aplica y qué no al cancelar), no el pipeline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from gui.app import DEFAULT_WINDOW_SIZE, MIN_WINDOW_SIZE, window_layout

try:
    import tkinter as tk

    HAS_TK = True
except ImportError:  # pragma: no cover - depende de la instalación de Python
    HAS_TK = False

needs_display = [
    pytest.mark.gui,
    pytest.mark.skipif(not HAS_TK, reason="tkinter no está disponible"),
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="sin servidor X (DISPLAY); usa xvfb-run"),
]


def gui_test(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Marca una prueba que abre ventanas."""
    for mark in reversed(needs_display):
        fn = mark(fn)
    return fn


def pump(app: Any, seconds: float = 0.1) -> None:
    """Procesa eventos de tkinter durante ``seconds``."""
    end = time.time() + seconds
    while time.time() < end:
        app.root.update()
        time.sleep(0.005)


def wait_idle(app: Any, timeout: float = 30.0) -> None:
    """Espera a que termine la tarea en segundo plano."""
    end = time.time() + timeout
    pump(app, 0.05)
    while app.busy:
        if time.time() > end:
            raise TimeoutError("La tarea en segundo plano no terminó a tiempo")
        pump(app, 0.02)
    pump(app, 0.1)


# ---------------------------------------------------------------------------
# Tamaño de la ventana
# ---------------------------------------------------------------------------


def test_window_layout_keeps_the_default_on_large_screens() -> None:
    """En una pantalla 1080p la ventana mide 1360×880, centrada en horizontal, con el mínimo de diseño."""
    layout = window_layout(1920, 1080)
    assert (layout.width, layout.height) == DEFAULT_WINDOW_SIZE
    assert (layout.min_width, layout.min_height) == MIN_WINDOW_SIZE
    assert layout.x == (1920 - 1360) // 2 and layout.y == 0 and not layout.maximize


@pytest.mark.parametrize("screen", [(1280, 720), (1366, 768), (1024, 600), (800, 600)])
def test_window_layout_fits_small_screens(screen: tuple[int, int]) -> None:
    """Regresión (proyector 720p): ventana y mínimo caben en la pantalla, con margen para la barra de tareas."""
    width, height = screen
    layout = window_layout(width, height)
    assert layout.x + layout.width <= width and layout.y + layout.height <= height - 100
    assert layout.min_width <= layout.width and layout.min_height <= layout.height
    assert layout.maximize


@pytest.fixture()
def small_screen_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Aplicación REAL en una «pantalla» de 1280×720 (se simula el tamaño que informa tkinter)."""
    from gui.app import SmartunerApp

    root = tk.Tk()
    monkeypatch.setattr(root, "winfo_screenwidth", lambda: 1280)
    monkeypatch.setattr(root, "winfo_screenheight", lambda: 720)
    app = SmartunerApp(root)
    pump(app, 0.3)
    yield app
    app.quit()


@gui_test
def test_status_bar_is_on_screen_with_a_720p_display(small_screen_app: Any) -> None:
    """Con 1280×720 la barra de estado (progreso y «Cancelar») queda dentro de la pantalla."""
    app = small_screen_app
    assert app.root.winfo_width() <= 1280 and app.root.winfo_height() <= 720 - 100
    status = app.status
    assert status.winfo_ismapped()
    assert status.winfo_rooty() + status.winfo_height() <= 720
    assert app.root.winfo_rootx() + app.root.winfo_width() <= 1280


# ---------------------------------------------------------------------------
# Cancelación: lo cancelado no se aplica
# ---------------------------------------------------------------------------


@pytest.fixture()
def app() -> Iterator[Any]:
    """Aplicación real con su tamaño por defecto."""
    from gui.app import SmartunerApp

    application = SmartunerApp()
    pump(application, 0.2)
    yield application
    application.quit()


def _cancelling_transcribe(calls: list[str], cancel_at: int) -> Callable[..., Any]:
    """``transcribe`` falso: simula que el usuario pulsa «Cancelar» durante la transcripción n.º ``cancel_at``.

    Como el real (que consulta ``cancel`` antes de cada segmento), si ya está
    cancelado lanza CancelledError sin hacer nada.
    """
    from src.config import CancelledError

    def fake(analysis: Any, algorithm: str, cfg: Any, run_index: int = 0, *, cancel: Any = None) -> str:
        assert cancel is not None, "la tarea debe pasar su evento de cancelación a transcribe"
        if cancel.is_set():
            raise CancelledError("cancelada")
        calls.append(algorithm)
        if len(calls) == cancel_at:
            cancel.set()  # «Cancelar» mientras termina esta transcripción
        return f"transcripción {algorithm}"

    return fake


@gui_test
def test_cancel_during_retranscription_keeps_the_previous_state(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: «Cancelar» al re-transcribir ya no aplica las transcripciones nuevas (antes se ignoraba)."""
    from src import experiments, pipeline, plots

    old_analysis, old_transcriptions = object(), {"ucb1": "anterior"}
    app.state.analysis, app.state.transcriptions = old_analysis, dict(old_transcriptions)
    calls: list[str] = []
    events: list[str] = []
    monkeypatch.setattr(pipeline, "rebuild_segment_data", lambda analysis, cfg: object())
    monkeypatch.setattr(plots, "prepare_display_spectrum", lambda analysis: None)
    # Cancelar durante la ÚLTIMA transcripción: todas terminan, pero el resultado no se aplica.
    monkeypatch.setattr(experiments, "transcribe", _cancelling_transcribe(calls, cancel_at=4))
    app.state.events.subscribe("transcriptions_ready", lambda **_k: events.append("transcriptions_ready"))
    app.retranscribe()
    wait_idle(app)
    assert len(calls) == 4
    assert app.state.analysis is old_analysis and app.state.transcriptions == old_transcriptions
    assert events == []
    assert "cancelado" in str(app.status.message.cget("text"))


@gui_test
def test_cancel_during_analysis_transcriptions_publishes_nothing(app: Any, monkeypatch: pytest.MonkeyPatch,
                                                                 tmp_path: Any) -> None:
    """Regresión: «Cancelar» durante la fase de transcripción del análisis no publica el análisis nuevo."""
    from src import experiments, pipeline, plots

    calls: list[str] = []
    monkeypatch.setattr(pipeline, "analyze", lambda path, cfg, progress=None, cancel=None: object())
    monkeypatch.setattr(plots, "prepare_display_spectrum", lambda analysis: None)
    monkeypatch.setattr(experiments, "transcribe", _cancelling_transcribe(calls, cancel_at=2))
    app.analyze(tmp_path / "cancion.mp3")
    wait_idle(app)
    assert len(calls) == 2  # la tercera transcripción no llega a ejecutarse
    assert app.state.analysis is None and app.state.transcriptions == {}
    assert "cancelado" in str(app.status.message.cget("text"))
