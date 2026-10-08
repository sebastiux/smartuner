"""Pruebas de la pestaña 6 · Log (:mod:`gui.tabs.log_tab`).

Abren la aplicación real (:class:`gui.app.SmartunerApp`) y ejercitan los
controles de la pestaña: nivel mínimo, filtro de texto, Limpiar, Guardar…,
Copiar al portapapeles y Auto-scroll, además de los límites de memoria y del
análisis REAL de ``data/synthetic/linea_simple.mp3`` (cuyas etapas deben
aparecer en negrita). Verifican ESTADO —texto de la vista, tags, contadores,
niveles de :mod:`logging`, archivo guardado—, no píxeles.

Necesitan tkinter y un servidor X; sin ``DISPLAY`` se omiten. Para ejecutarlas::

    xvfb-run -a .venv/bin/python -m pytest tests/test_gui_log.py -q

La mayoría de las pruebas comparten una sola ventana (fixture de módulo); cada
prueba vacía la cola, limpia la vista y restablece los controles.
"""

from __future__ import annotations

import io
import logging
import os
import re
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
]

#: Logger de prueba dentro del paquete ``src`` (como los módulos reales).
TEST_LOGGER = "src.pruebas_log"

#: Formato esperado de la primera línea de cada entrada.
LINE_RE = re.compile(r"^\d\d:\d\d:\d\d  (DEBUG|INFO|WARNING|ERROR|CRITICAL) {1,5}\s?\S+: .*$")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def pump(app: Any, seconds: float = 0.15) -> None:
    """Procesa eventos de tkinter durante ``seconds`` (incluye ``after`` y el sondeo de la pestaña)."""
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


def new_app() -> Any:
    """Crea la aplicación con el tamaño de diseño (1360×880) y la pestaña Log visible."""
    from gui.app import SmartunerApp

    app = SmartunerApp()
    app.root.geometry("1360x880+0+0")
    app.show_tab("log")
    pump(app, 0.3)
    return app


def flush(tab: Any) -> None:
    """Procesa todos los registros pendientes de la cola (varios ciclos si hace falta)."""
    while tab.poll_now():
        pass
    tab.app.root.update_idletasks()


def view_lines(tab: Any) -> list[str]:
    """Líneas de texto de la vista."""
    text = tab.visible_text()
    return text.split("\n") if text else []


def line_of(tab: Any, needle: str) -> int:
    """Número de línea Tk (1 = primera) de la primera línea de la vista que contiene ``needle``."""
    for i, line in enumerate(view_lines(tab), start=1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} no aparece en la vista:\n{tab.visible_text()}")


def tags_at(tab: Any, needle: str) -> set[str]:
    """Tags del texto en la posición donde empieza ``needle`` dentro de la vista."""
    line = line_of(tab, needle)
    col = view_lines(tab)[line - 1].index(needle)
    return set(tab.text.tag_names(f"{line}.{col}"))


def emit(level: int, message: str, *args: Any) -> None:
    """Escribe un mensaje con el logger de prueba (``src.pruebas_log``)."""
    logging.getLogger(TEST_LOGGER).log(level, message, *args)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_app() -> Iterator[Any]:
    """Una ventana compartida por las pruebas del módulo (crearla cuesta ≈ 1 s)."""
    app = new_app()
    yield app
    app.quit()


@pytest.fixture
def tab(shared_app: Any) -> Iterator[Any]:
    """La pestaña Log de la ventana compartida, vacía y con los controles por defecto."""
    from gui.tabs import log_tab

    log = shared_app.tabs["log"]
    log.max_records = log_tab.MAX_BUFFER_RECORDS
    log.max_visible_lines = log_tab.MAX_VISIBLE_LINES
    log.max_per_tick = log_tab.MAX_RECORDS_PER_TICK
    log.set_level("INFO")
    log.set_filter("")
    log.autoscroll_var.set(True)
    flush(log)
    log.clear()
    yield log
    log.set_level("INFO")
    log.set_filter("")


# ---------------------------------------------------------------------------
# Funciones puras de formato
# ---------------------------------------------------------------------------


def test_format_helpers() -> None:
    """``module_label`` quita ``src.``; ``level_tag`` agrupa niveles; ``format_record`` no falla con argumentos rotos."""
    from gui.tabs.log_tab import format_record, level_tag, module_label, plural

    assert module_label("src.pipeline") == "pipeline"
    assert module_label("gui.tabs.log_tab") == "gui.tabs.log_tab"
    assert [level_tag(n) for n in (10, 20, 30, 40, 50)] == ["debug", "info", "warning", "error", "critical"]
    assert plural(1, "mensaje", "mensajes") == "1 mensaje" and plural(2, "mensaje", "mensajes") == "2 mensajes"

    bad = logging.LogRecord("src.x", logging.INFO, __file__, 1, "valor %d", ("no-número",), None)
    entry = format_record(bad)
    assert "valor %d" in entry.body and "no-número" in entry.body

    multi = logging.LogRecord("src.x", logging.WARNING, __file__, 1, "uno\ndos", None, None)
    entry = format_record(multi)
    assert entry.n_lines == 2
    assert entry.body.split("\n")[1] == " " * 19 + "dos"


# ---------------------------------------------------------------------------
# Pestaña
# ---------------------------------------------------------------------------


def test_welcome_message_on_startup() -> None:
    """Al abrir la app, la pestaña ya muestra el mensaje de bienvenida que registra ``app.py``."""
    app = new_app()
    try:
        tab = app.tabs["log"]
        assert type(tab).__name__ == "LogTab"
        wait_until(app, lambda: "Smartuner listo" in tab.visible_text(), timeout=5)
        line = view_lines(tab)[line_of(tab, "Smartuner listo") - 1]
        assert LINE_RE.match(line), line
        assert "  INFO     gui.app: Smartuner listo." in line
        assert tab.level_var.get() == "INFO"
        assert app.log_handler.level == logging.INFO
    finally:
        app.quit()


def test_format_and_level_tags(tab: Any) -> None:
    """Formato «HH:MM:SS  NIVEL    módulo: mensaje», colores por nivel y etapas en negrita."""
    emit(logging.INFO, "Etapa %d/6 — Carga: decodificando prueba.mp3", 1)
    emit(logging.INFO, "Segmento 3: f0=55.0 Hz, 8 brazos candidatos")
    emit(logging.WARNING, "Aviso de prueba")
    emit(logging.ERROR, "Error de prueba")
    emit(logging.DEBUG, "Detalle oculto con nivel INFO")
    flush(tab)

    lines = view_lines(tab)
    assert len(lines) == 4  # el DEBUG no pasa con nivel mínimo INFO
    assert all(LINE_RE.match(line) for line in lines), lines
    assert lines[0].endswith("  INFO     pruebas_log: Etapa 1/6 — Carga: decodificando prueba.mp3")
    assert "  WARNING  pruebas_log: Aviso de prueba" in lines[2]
    assert "  ERROR    pruebas_log: Error de prueba" in lines[3]

    assert {"stage", "info"} <= tags_at(tab, "Etapa 1/6")
    assert "time" in set(tab.text.tag_names("1.0"))
    assert "stage" not in tags_at(tab, "Segmento 3") and "info" in tags_at(tab, "Segmento 3")
    assert "warning" in tags_at(tab, "Aviso de prueba")
    assert "error" in tags_at(tab, "Error de prueba")
    assert "Detalle oculto" not in tab.visible_text()

    assert tab.count_label.cget("text") == "4 mensajes"
    assert tab.warnings_label.cget("text") == "⚠ 1 aviso"
    assert tab.errors_label.cget("text") == "✖ 1 error"
    assert tab.placeholder.winfo_manager() == ""  # hay mensajes: sin mensaje guía


def test_exception_traceback_is_indented(tab: Any) -> None:
    """``logger.exception`` muestra el traceback en líneas sangradas, en rojo."""
    try:
        raise ValueError("fallo simulado")
    except ValueError:
        logging.getLogger(TEST_LOGGER).exception("Algo salió mal")
    flush(tab)
    lines = view_lines(tab)
    assert "ERROR    pruebas_log: Algo salió mal" in lines[0]
    assert lines[1].startswith(" " * 19 + "Traceback")
    assert any("ValueError: fallo simulado" in line for line in lines)
    assert "error" in tags_at(tab, "ValueError: fallo simulado")
    assert tab.visible_line_count == len(lines)
    assert tab.count_label.cget("text") == "1 mensaje"


def test_level_combobox_debug_and_console(tab: Any) -> None:
    """DEBUG en el desplegable: ajusta el handler y deja pasar DEBUG de ``src.*`` sin inundar la consola.

    Reproduce la configuración de :func:`gui.app.main` (``logging.basicConfig``
    con nivel INFO): logger raíz en INFO y un ``StreamHandler`` de consola sin nivel.
    """
    console_stream = io.StringIO()
    console = logging.StreamHandler(console_stream)  # como el de logging.basicConfig (sin nivel)
    root_logger = logging.getLogger()
    root_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console)
    try:
        tab.set_level("INFO")  # re-evalúa los umbrales con el raíz en INFO: no hace falta tocar nada
        assert console.level == logging.NOTSET
        emit(logging.INFO, "Información visible también en la consola")
        assert "Información visible también en la consola" in console_stream.getvalue()

        tab.level_combo.set("DEBUG")
        tab.level_combo.event_generate("<<ComboboxSelected>>")
        assert tab.app.log_handler.level == logging.DEBUG
        assert logging.getLogger("src.environment").isEnabledFor(logging.DEBUG)
        assert logging.getLogger("gui.app").isEnabledFor(logging.DEBUG)

        emit(logging.DEBUG, "Entorno del segmento 0: K=8 (detalle)")
        flush(tab)
        assert "DEBUG    pruebas_log: Entorno del segmento 0" in tab.visible_text()
        assert "debug" in tags_at(tab, "Entorno del segmento 0")
        # La consola conserva su umbral: no recibe los DEBUG que sí muestra la pestaña.
        assert "Entorno del segmento 0" not in console_stream.getvalue()

        tab.level_combo.set("INFO")
        tab.level_combo.event_generate("<<ComboboxSelected>>")
        assert tab.app.log_handler.level == logging.INFO
        assert not logging.getLogger("src.environment").isEnabledFor(logging.DEBUG)
        assert console.level == logging.NOTSET  # la pestaña devolvió la consola a su estado original
        assert "Entorno del segmento 0" not in tab.visible_text()  # sigue en el búfer, pero oculto
        assert tab.buffer_size >= 1
        emit(logging.DEBUG, "Otro detalle que ya no se captura")
        flush(tab)
        assert all("Otro detalle" not in entry for entry in tab.matching_lines())
    finally:
        root_logger.removeHandler(console)
        root_logger.setLevel(root_level)
        tab.set_level("INFO")  # vuelve a ajustar los umbrales al raíz original


def test_level_warning_rerenders_from_buffer(tab: Any) -> None:
    """Subir el nivel oculta mensajes sin perderlos; bajarlo los vuelve a mostrar."""
    for i in range(5):
        emit(logging.INFO, "Información %d", i)
    emit(logging.WARNING, "Un aviso")
    emit(logging.ERROR, "Un error")
    flush(tab)
    assert len(view_lines(tab)) == 7

    tab.set_level("WARNING")
    assert tab.app.log_handler.level == logging.WARNING
    assert [line.split(": ", 1)[1] for line in view_lines(tab)] == ["Un aviso", "Un error"]
    assert tab.count_label.cget("text") == "2 de 7 mensajes"

    tab.set_level("ERROR")
    assert len(view_lines(tab)) == 1 and "Un error" in tab.visible_text()

    tab.set_level("INFO")
    assert len(view_lines(tab)) == 7
    assert tab.count_label.cget("text") == "7 mensajes"
    with pytest.raises(ValueError):
        tab.set_level("TRACE")


def test_text_filter(tab: Any) -> None:
    """El filtro muestra solo las líneas que contienen el texto (sin mayúsculas) y resalta coincidencias."""
    emit(logging.INFO, "Etapa 4/6 — Segmentación: detectando onsets")
    emit(logging.INFO, "Segmentación: 17 onsets → 16 segmentos")
    emit(logging.INFO, "Transcripción con UCB1: 16 notas")
    flush(tab)

    tab.filter_entry.focus_force()
    tab.filter_entry.insert(0, "SEGMENTACIÓN")  # se aplica tras la pausa anti-rebote
    wait_until(tab.app, lambda: len(view_lines(tab)) == 2, timeout=5)
    assert all("Segmentación" in line for line in view_lines(tab))
    assert tab.count_label.cget("text") == "2 de 3 mensajes"
    assert "match" in tags_at(tab, "Segmentación: detectando")
    assert "match" not in tags_at(tab, "Etapa 4/6")

    # Un mensaje nuevo que no coincide no aparece; uno que coincide, sí.
    emit(logging.INFO, "Otra cosa")
    emit(logging.INFO, "Fin de la segmentación")
    flush(tab)
    assert len(view_lines(tab)) == 3 and "Otra cosa" not in tab.visible_text()
    assert tab.count_label.cget("text") == "3 de 5 mensajes"

    # Sin coincidencias: mensaje guía en vez de una vista en blanco.
    tab.set_filter("no-aparece-en-ningún-mensaje")
    assert tab.visible_text() == ""
    assert tab.placeholder.winfo_manager() == "place"
    assert "no-aparece-en-ningún-mensaje" in tab.placeholder.cget("text")

    tab.filter_entry.event_generate("<Escape>")  # Esc borra el filtro
    assert tab.filter_var.get() == "" and len(view_lines(tab)) == 5
    tab.set_filter("ucb1")
    assert len(view_lines(tab)) == 1
    tab.clear_filter_button.invoke()  # botón ✕
    assert tab.filter_var.get() == "" and len(view_lines(tab)) == 5
    assert tab.placeholder.winfo_manager() == ""


def test_clear_button(tab: Any) -> None:
    """«Limpiar» vacía el búfer y la vista y muestra el mensaje guía."""
    emit(logging.INFO, "Mensaje que se borrará")
    emit(logging.WARNING, "Aviso que se borrará")
    flush(tab)
    assert tab.buffer_size == 2

    tab.clear_button.invoke()
    assert tab.buffer_size == 0 and tab.visible_text() == "" and tab.visible_line_count == 0
    assert tab.count_label.cget("text") == "0 mensajes"
    assert tab.warnings_label.cget("text") == ""
    assert tab.placeholder.winfo_manager() == "place"
    assert "Archivo → Abrir" in tab.placeholder.cget("text")

    emit(logging.INFO, "Después de limpiar")
    flush(tab)
    assert view_lines(tab)[0].endswith("pruebas_log: Después de limpiar")
    assert tab.placeholder.winfo_manager() == ""


def test_save_button(tab: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """«Guardar…» escribe en UTF-8 las líneas que pasan los filtros; cancelar no escribe nada."""
    from gui.tabs import log_tab

    emit(logging.INFO, "Etapa 1/6 — Carga: decodificando ñandú.mp3 → 22050 Hz")
    emit(logging.WARNING, "Segmento 7: f0 fuera de rango")
    flush(tab)

    target = tmp_path / "registro.txt"
    monkeypatch.setattr(log_tab.filedialog, "asksaveasfilename", lambda **_kw: str(target))
    tab.save_button.invoke()
    content = target.read_text(encoding="utf-8")
    lines = content.rstrip("\n").split("\n")
    assert lines == tab.matching_lines()
    assert "ñandú.mp3 → 22050 Hz" in content and "—" in content
    assert tab.app.status.message.cget("text") == "Log guardado en registro.txt."

    # Con filtro, se guarda solo lo filtrado (sin el límite de líneas de la vista).
    tab.set_filter("segmento 7")
    tab.save_button.invoke()
    saved = target.read_text(encoding="utf-8").rstrip("\n").split("\n")
    assert len(saved) == 1 and "Segmento 7" in saved[0]

    # Cancelar el diálogo no escribe nada.
    target.unlink()
    monkeypatch.setattr(log_tab.filedialog, "asksaveasfilename", lambda **_kw: "")
    tab.save_button.invoke()
    assert not target.exists()


def test_copy_button(tab: Any) -> None:
    """«Copiar al portapapeles» copia las líneas que pasan los filtros."""
    emit(logging.INFO, "Primera línea copiada")
    emit(logging.INFO, "Segunda línea copiada")
    flush(tab)
    tab.copy_button.invoke()
    clipboard = tab.app.root.clipboard_get()
    assert clipboard == "\n".join(tab.matching_lines())
    assert "Primera línea copiada" in clipboard and "Segunda línea copiada" in clipboard
    assert "2 mensajes" in tab.app.status.message.cget("text")


def test_autoscroll(tab: Any) -> None:
    """Con Auto-scroll la vista sigue al último mensaje; sin él, se queda donde estaba."""
    for i in range(300):
        emit(logging.INFO, "Mensaje número %d", i)
    flush(tab)
    pump(tab.app, 0.1)
    assert tab.text.yview()[1] == pytest.approx(1.0)

    tab.autoscroll_check.invoke()  # desactivar
    assert tab.autoscroll_var.get() is False
    tab.text.yview_moveto(0.0)
    for i in range(50):
        emit(logging.INFO, "Mensaje tardío %d", i)
    flush(tab)
    pump(tab.app, 0.1)
    assert tab.text.yview()[0] == pytest.approx(0.0)

    tab.autoscroll_check.invoke()  # reactivar: salta al final
    assert tab.autoscroll_var.get() is True
    pump(tab.app, 0.1)
    assert tab.text.yview()[1] == pytest.approx(1.0)


def test_memory_limits_and_tick_cap(tab: Any) -> None:
    """Búfer y vista acotados (se conservan los más recientes) y ≤ ``max_per_tick`` registros por ciclo."""
    tab.max_records = 300
    tab.max_visible_lines = 120
    tab.max_per_tick = 500
    for i in range(1000):
        emit(logging.INFO, "Corrida %04d", i)

    assert tab.poll_now() == 500  # un ciclo no procesa más de 500
    flush(tab)
    assert tab.buffer_size == 300
    assert tab.visible_line_count == 120
    text_lines = int(tab.text.index("end-1c").split(".")[0]) - 1  # la vista termina en "\n"
    assert text_lines == 120
    assert tab.matching_lines()[0].endswith("Corrida 0700")
    lines = view_lines(tab)
    assert lines[0].endswith("Corrida 0880") and lines[-1].endswith("Corrida 0999")
    assert "300 mensajes" in tab.count_label.cget("text")
    assert "se muestran los 120 más recientes" in tab.count_label.cget("text")

    # Re-dibujar (filtro) respeta también el límite de líneas.
    tab.set_filter("corrida 09")
    assert len(view_lines(tab)) == 100 and tab.matching_count == 100


def test_busy_changed_drains_queue(tab: Any) -> None:
    """``busy_changed`` vacía la cola en el acto (sin esperar al siguiente ciclo de sondeo)."""
    emit(logging.INFO, "Mensaje justo antes de terminar la tarea")
    tab.app.state.events.emit("busy_changed", busy=False)
    assert "Mensaje justo antes de terminar la tarea" in tab.visible_text()


def test_destroy_restores_logging_levels() -> None:
    """Al cerrar la ventana, los umbrales de ``src``/``gui`` vuelven a su valor original."""
    original = {name: logging.getLogger(name).level for name in ("src", "gui")}
    app = new_app()
    tab = app.tabs["log"]
    tab.set_level("DEBUG")
    assert logging.getLogger("src").getEffectiveLevel() == logging.DEBUG
    app.quit()
    assert {name: logging.getLogger(name).level for name in ("src", "gui")} == original


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado")
@pytest.mark.skipif(not AUDIO.exists(), reason=f"falta {AUDIO.name} (genera el dataset sintético)")
def test_real_analysis_shows_pipeline_stages(tab: Any) -> None:
    """Analizar ``linea_simple.mp3`` llena el log con las etapas del pipeline (en negrita) en orden."""
    app = tab.app
    app.analyze(AUDIO)
    wait_until(app, lambda: not app.busy, timeout=120)
    pump(app, 0.3)
    flush(tab)

    text = tab.visible_text()
    stages = [line for line in view_lines(tab) if "Etapa " in line]
    numbers = [int(re.search(r"Etapa (\d)/6", line).group(1)) for line in stages]
    assert numbers[0] == 1 and numbers == sorted(numbers) and max(numbers) == 6
    for line in stages:
        assert LINE_RE.match(line) and "  INFO     pipeline: Etapa" in line
    assert "stage" in tags_at(tab, "Etapa 1/6")
    assert "Segmento 0:" in text and "Transcripción con UCB1" in text
    assert tab.buffer_size >= 20

    tab.set_filter("etapa")
    assert len(view_lines(tab)) == len(stages)
