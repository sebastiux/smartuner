"""Pruebas de la pestaña 1 · Audio (:mod:`gui.tabs.audio_tab`).

Abren la aplicación real (:class:`gui.app.SmartunerApp`) y ejercitan los
controles de la pestaña con un análisis REAL pequeño
(``data/synthetic/linea_simple.mp3``, 11 s, 16 notas con ground truth).
Verifican ESTADO —filas de la tabla, segmento seleccionado en ``app.state``,
configuración, botones habilitados, qué muestra la figura—, no píxeles.

Necesitan tkinter y un servidor X; sin ``DISPLAY`` se omiten. Para ejecutarlas::

    xvfb-run -a .venv/bin/python -m pytest tests/test_gui_audio.py -q

Para que el archivo tarde poco, la mayoría de las pruebas comparten una sola
ventana (fixture de módulo) y un único análisis; cada prueba restablece la
configuración y la selección y publica de nuevo el análisis.
"""

from __future__ import annotations

import dataclasses
import os
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
        time.sleep(0.01)


def settle(tab: Any, timeout: float = 10.0) -> None:
    """Espera a que la pestaña termine sus redibujos diferidos (la figura refleja el estado)."""
    tab.app.root.update()  # entrega antes los eventos en cola (p. ej. <<TreeviewSelect>>)
    wait_until(tab.app, lambda: not tab.redraw_pending, timeout=timeout)
    tab.app.root.update()


def new_app() -> Any:
    """Crea la aplicación con el tamaño de diseño (1360×880) y la pestaña Audio visible."""
    from gui.app import SmartunerApp

    app = SmartunerApp()
    app.root.geometry("1360x880+0+0")
    app.show_tab("audio")
    pump(app, 0.3)
    return app


def publish(app: Any, analysis: Any) -> None:
    """Publica ``analysis`` como lo hace la aplicación al terminar de analizar."""
    app.state.analysis = analysis
    app.state.selected_position = None
    app.state.events.emit("analysis_ready", analysis=analysis)
    settle(app.tabs["audio"])


def figure_texts(tab: Any) -> str:
    """Todo el texto visible de la figura (textos de figura, títulos y ejes)."""
    fig = tab.plot.figure
    parts = [t.get_text() for t in fig.texts]
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        parts.append(suptitle.get_text())
    for ax in fig.axes:
        parts += [t.get_text() for t in ax.texts] + [ax.get_ylabel()]
    return "\n".join(parts)


def click_figure(tab: Any, t: float, axis: int = 1) -> None:
    """Simula un clic izquierdo sobre la figura en el instante ``t`` (s) del eje ``axis``."""
    from matplotlib.backend_bases import MouseEvent

    canvas = tab.plot.canvas
    canvas.draw()
    ax = tab._data_axes()[axis]
    y_mid = float(np.mean(ax.get_ylim())) if axis == 0 else float(np.sqrt(np.prod(ax.get_ylim())))
    x_px, y_px = ax.transData.transform((t, y_mid))
    event = MouseEvent("button_press_event", canvas, x_px, y_px, button=1)
    canvas.callbacks.process("button_press_event", event)


def with_discarded(analysis: Any) -> Any:
    """Copia de ``analysis`` con dos segmentos descartados añadidos (silencio inicial y uno corto).

    ``linea_simple`` no tiene descartados; se insertan en huecos sin nota para
    no alterar los conservados (sus posiciones siguen siendo 0…15).
    """
    from src.segmentation import Segment

    segments = list(analysis.segments)
    first = segments[0]
    silence = Segment(index=0, start_s=0.0, end_s=first.start_s, start_sample=0, end_sample=first.start_sample,
                      rms_db=-60.0, kept=False, reason="silencio")
    gap = next(i for i in range(len(segments) - 1) if segments[i + 1].start_s - segments[i].end_s > 0.02)
    a, b = segments[gap], segments[gap + 1]
    short = Segment(index=0, start_s=a.end_s, end_s=b.start_s, start_sample=a.end_sample, end_sample=b.start_sample,
                    rms_db=-20.0, kept=False, reason="corto")
    merged = [silence] + segments[:gap + 1] + [short] + segments[gap + 1:]
    merged = [dataclasses.replace(s, index=i) for i, s in enumerate(merged)]
    return dataclasses.replace(analysis, segments=merged)


class FakePlayer:
    """Reproductor de prueba: registra lo que se le pide reproducir."""

    available = True
    error = ""

    def __init__(self) -> None:
        """Empieza sin llamadas registradas."""
        self.calls: list[tuple[str, int, int]] = []

    def play(self, y: np.ndarray, sr: int) -> None:
        """Registra ``("play", n_muestras, sr)``."""
        self.calls.append(("play", len(y), int(sr)))

    def stop(self) -> None:
        """Registra ``("stop", 0, 0)``."""
        self.calls.append(("stop", 0, 0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis() -> Any:
    """Análisis real de ``linea_simple.mp3`` con la configuración por defecto (≈ 3 s)."""
    from src.config import Config
    from src.pipeline import analyze

    return analyze(AUDIO, Config())


@pytest.fixture(scope="module")
def shared_app() -> Iterator[Any]:
    """Una ventana compartida por las pruebas del módulo (crearla cuesta ≈ 1 s)."""
    app = new_app()
    yield app
    app.quit()


@pytest.fixture
def tab(shared_app: Any, analysis: Any) -> Iterator[Any]:
    """La pestaña Audio de la ventana compartida, con estado limpio y el análisis publicado."""
    from src.config import Config

    app = shared_app
    app.state.config = Config()
    app.state.events.emit("config_changed", key=None)
    app.state.events.emit("busy_changed", busy=False)
    audio_tab = app.tabs["audio"]
    toolbar = audio_tab.plot.toolbar
    if audio_tab._toolbar_active():  # una prueba fallida pudo dejar la lupa o la mano activas
        if "zoom" in str(toolbar.mode):
            toolbar.zoom()
        else:
            toolbar.pan()
    app.show_tab("audio")
    publish(app, analysis)
    yield audio_tab


# ---------------------------------------------------------------------------
# Pruebas con una ventana nueva (estado inicial y flujo completo)
# ---------------------------------------------------------------------------


def test_estado_vacio_muestra_mensaje_guia() -> None:
    """Antes de analizar: mensaje guía en la figura, tabla vacía y acciones coherentes."""
    from gui.tabs.audio_tab import AudioTab
    from src.separation import demucs_available

    app = new_app()
    try:
        tab = app.tabs["audio"]
        assert isinstance(tab, AudioTab)
        assert tab.plot_state == "empty"
        assert "Abre un MP3" in figure_texts(tab)
        assert tab.tree.get_children() == ()
        assert tab.empty_table.winfo_manager() == "place"
        assert tab.info_values["file"].cget("text") == "Ningún audio cargado"
        assert tab.btn_open.instate(["!disabled"])
        assert tab.btn_dataset.instate(["!disabled"])
        assert tab.btn_reanalyze.instate(["disabled"])
        assert tab.btn_play_segment.instate(["disabled"])
        assert tab.chk_separate.instate(["!disabled"] if demucs_available() else ["disabled"])
        if not demucs_available():
            assert "pip install" in tab._separate_help()
        if not tab.player.available:
            assert tab.btn_play_all.instate(["disabled"]) and tab.btn_stop.instate(["disabled"])
            assert tab._play_help("ayuda") == tab.player.error
        # La pestaña no debe pedir más alto que la ventana (si no, la barra de estado desaparece).
        assert app.status.winfo_ismapped()
    finally:
        app.quit()


@pytest.mark.slow
def test_boton_abrir_analiza_y_rellena_la_pestana(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Abrir MP3…» → diálogo (simulado) → análisis en segundo plano → figura, tabla e información."""
    import gui.app

    app = new_app()
    try:
        tab = app.tabs["audio"]
        monkeypatch.setattr(gui.app.filedialog, "askopenfilename", lambda **_kw: str(AUDIO))
        tab.btn_open.invoke()
        pump(app, 0.1)
        assert app.busy
        assert tab.btn_open.instate(["disabled"]) and tab.btn_dataset.instate(["disabled"])
        wait_until(app, lambda: not app.busy, timeout=120)
        pump(app, 0.4)

        analysis = app.state.analysis
        assert analysis is not None
        assert tab.analysis is analysis
        assert len(tab.tree.get_children()) == len(analysis.segments) == 16
        assert tab.plot_state == "analysis"
        assert "Forma de onda y espectrograma" in figure_texts(tab)
        assert tab.info_values["file"].cget("text") == "linea_simple.mp3"
        assert tab.info_values["gt"].cget("text").startswith("16 notas · 16 emparejadas")
        assert "STFT" in tab.info_values["spectrum"].cget("text")
        assert "gt" in tab.tree.cget("displaycolumns")
        assert tab.btn_open.instate(["!disabled"]) and tab.btn_reanalyze.instate(["!disabled"])
        assert tab.empty_table.winfo_manager() == ""
    finally:
        app.quit()


# ---------------------------------------------------------------------------
# Pruebas con la ventana compartida
# ---------------------------------------------------------------------------


def test_tabla_muestra_segmentos_y_nota_real(tab: Any, analysis: Any) -> None:
    """Cada fila conservada lleva posición, tiempos, f0, nota y la nota real emparejada."""
    from src.pitch import midi_to_name

    kept = analysis.kept
    assert list(tab.tree.get_children()) == [f"pos{i}" for i in range(len(kept))]
    values = tab.tree.item("pos0", "values")
    assert values[0] == "0"
    assert float(values[1]) == pytest.approx(kept[0].start_s, abs=1e-3)
    assert float(values[4]) == pytest.approx(kept[0].f0_hz, abs=0.05)
    assert values[5] == midi_to_name(int(round(kept[0].midi)))
    assert values[7] == "ok"
    assert values[8].endswith(analysis.ground_truth[0].label)  # «A1 · A-0»


def test_seleccionar_fila_publica_el_segmento(tab: Any) -> None:
    """Seleccionar una fila → ``app.state.select_segment(pos, "audio")`` y resaltado en la figura."""
    app = tab.app
    received: list[tuple[Any, str]] = []
    app.state.events.subscribe("segment_selected", lambda position, source: received.append((position, source)))
    tab.tree.selection_set("pos3")
    settle(tab)
    assert app.state.selected_position == 3
    assert received[-1] == (3, "audio")
    assert tab.drawn_selection == 3
    assert "Seg. 3" in figure_texts(tab)


def test_seleccion_externa_sincroniza_tabla_y_figura(tab: Any) -> None:
    """Un ``segment_selected`` de otra pestaña selecciona la fila y redibuja el resaltado."""
    app = tab.app
    app.state.select_segment(5, source="tab")
    settle(tab)
    assert tab.tree.selection() == ("pos5",)
    assert tab.drawn_selection == 5
    app.state.select_segment(None, source="tab")
    settle(tab)
    assert tab.tree.selection() == ()
    assert tab.drawn_selection is None


def test_clic_en_la_figura_selecciona_el_segmento(tab: Any, analysis: Any) -> None:
    """Clic → segmento conservado de ese instante; en modo zoom o fuera de las notas no se selecciona."""
    app = tab.app
    kept = analysis.kept
    click_figure(tab, (kept[7].start_s + kept[7].end_s) / 2)
    settle(tab)
    assert app.state.selected_position == 7
    assert tab.tree.selection() == ("pos7",)

    click_figure(tab, (kept[2].start_s + kept[2].end_s) / 2, axis=0)  # también sobre la forma de onda
    settle(tab)
    assert app.state.selected_position == 2

    toolbar = tab.plot.toolbar
    toolbar.zoom()  # modo zoom: el clic es para la lupa, no para seleccionar
    try:
        click_figure(tab, (kept[9].start_s + kept[9].end_s) / 2)
        pump(app, 0.2)
        assert app.state.selected_position == 2
    finally:
        toolbar.zoom()

    click_figure(tab, kept[0].start_s / 2)  # silencio inicial: no hay nota
    pump(app, 0.2)
    assert app.state.selected_position == 2
    assert "no hay ningún segmento" in app.status.message.cget("text")


def test_el_zoom_se_conserva_al_cambiar_de_segmento(tab: Any, analysis: Any) -> None:
    """La figura se reconstruye en cada selección, pero el zoom del usuario se mantiene."""
    app = tab.app
    settle(tab)
    wave = tab._data_axes()[0]
    wave.set_xlim(2.0, 6.0)
    tab.plot.canvas.draw()
    app.state.select_segment(5, source="live")
    settle(tab)
    assert tab.drawn_selection == 5
    assert tab._data_axes()[0].get_xlim() == pytest.approx((2.0, 6.0))
    tab.plot.toolbar.home()  # «Inicio» vuelve a la pista completa
    assert tab._data_axes()[0].get_xlim() == pytest.approx((0.0, analysis.duration_s))


def test_descartados_en_gris_y_no_seleccionables(tab: Any, analysis: Any) -> None:
    """Los segmentos descartados se listan en gris, explican su motivo y no se pueden seleccionar."""
    app = tab.app
    modified = with_discarded(analysis)
    publish(app, modified)
    rows = list(tab.tree.get_children())
    discarded = [r for r in rows if r.startswith("disc")]
    assert len(rows) == 18 and len(discarded) == 2
    assert all("discarded" in tab.tree.item(r, "tags") for r in discarded)
    assert tab.tree.item(discarded[0], "values")[0] == "–"
    assert tab.info_values["segments"].cget("text") == "16 conservados · 2 descartados (1 por silencio, 1 por corto)"
    assert "silencio" in tab._tree_help("row", discarded[0])
    assert "corto" in tab._tree_help("row", discarded[1])
    assert tab._tree_help("row", "pos0") == ""
    assert "pYIN" in tab._tree_help("heading", "f0")

    # Clic de ratón sobre una fila descartada: no cambia la selección.
    app.state.select_segment(4, source="test")
    settle(tab)
    tab.tree.see(discarded[0])
    pump(app, 0.1)
    x, y, _w, h = tab.tree.bbox(discarded[0])
    tab.tree.event_generate("<ButtonPress-1>", x=x + 10, y=y + h // 2)
    tab.tree.event_generate("<ButtonRelease-1>", x=x + 10, y=y + h // 2)
    settle(tab)
    assert app.state.selected_position == 4
    assert "descartado" in app.status.message.cget("text")

    # Con el teclado se salta a la siguiente fila conservada en la dirección del movimiento.
    index = rows.index(discarded[1])
    before = tab._row_position[rows[index - 1]]
    app.state.select_segment(before, source="test")
    settle(tab)
    tab.tree.selection_set(discarded[1])
    settle(tab)
    assert app.state.selected_position == before + 1

    # La figura tampoco selecciona dentro de un descartado.
    short = next(s for s in modified.segments if s.reason == "corto")
    assert tab.position_at((short.start_s + short.end_s) / 2) is None
    assert tab.position_at(0.1) is None


def test_sin_ground_truth_se_oculta_la_columna(tab: Any, analysis: Any) -> None:
    """Sin ``.gt.json`` no hay columna «Real (GT)» y el panel lo indica."""
    publish(tab.app, dataclasses.replace(analysis, ground_truth=None))
    assert "gt" not in tab.tree.cget("displaycolumns")
    assert tab.info_values["gt"].cget("text").startswith("no disponible")


def test_audio_sin_notas_y_stem_de_demucs(tab: Any, analysis: Any) -> None:
    """Pista sin notas: tabla vacía con explicación; con separación se indica el stem analizado."""
    app = tab.app
    silent = dataclasses.replace(analysis, segments=[], onsets_s=np.zeros(0), segment_data=[], ground_truth=None)
    publish(app, silent)
    assert tab.tree.get_children() == ()
    assert "No se detectó ninguna nota" in tab.empty_table.cget("text")
    assert tab.empty_table.winfo_manager() == "place"
    assert tab.position_at(1.0) is None
    assert tab.plot_state == "analysis"

    stem = Path("cache") / "separated" / "linea_simple_bass.wav"
    publish(app, dataclasses.replace(analysis, source_path=stem))
    assert tab.info_values["source"].cget("text") == "linea_simple_bass.wav (stem de bajo de Demucs)"
    assert tab._source_help() == str(stem)
    assert tab.empty_table.winfo_manager() == ""


def test_tolerancia_de_onset_cambia_el_emparejamiento(tab: Any) -> None:
    """``experiment.onset_tolerance_s`` (de la pestaña Configuración) se aplica al emparejar con el GT."""
    app = tab.app
    app.state.config.experiment.onset_tolerance_s = 0.001
    app.state.events.emit("config_changed", key="experiment.onset_tolerance_s")
    pump(app, 0.2)
    gt_texts = [tab.tree.item(r, "values")[8] for r in tab.tree.get_children()]
    assert "sin pareja" in gt_texts
    assert "(±1 ms)" in tab.info_values["gt"].cget("text")


def test_casilla_demucs_actualiza_configuracion_y_aviso(tab: Any) -> None:
    """La casilla escribe ``audio.separate_bass``, emite ``config_changed`` y muestra el aviso."""
    app = tab.app
    keys: list[Any] = []
    app.state.events.subscribe("config_changed", lambda key=None: keys.append(key))
    assert not tab.banner_visible
    tab.separate_var.set(True)
    tab._on_separate_toggled()  # lo que ejecuta el clic (la casilla puede estar deshabilitada sin Demucs)
    pump(app, 0.1)
    assert app.state.config.audio.separate_bass is True
    assert keys[-1] == "audio.separate_bass"
    assert tab.banner_visible

    # Cambio hecho desde otra pestaña (o al cargar un JSON): la casilla se sincroniza.
    app.state.config.audio.separate_bass = False
    app.state.events.emit("config_changed", key=None)
    pump(app, 0.1)
    assert tab.separate_var.get() is False
    assert not tab.banner_visible

    # Un parámetro de segmentación también exige volver a analizar.
    app.state.config.segmentation.onset_delta = 0.2
    app.state.events.emit("config_changed", key="segmentation.onset_delta")
    pump(app, 0.1)
    assert tab.banner_visible


def test_botones_de_accion_llaman_a_la_aplicacion(tab: Any, analysis: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """«Analizar de nuevo» y «Generar dataset» delegan en la app; se deshabilitan con una tarea en curso."""
    app = tab.app
    calls: list[Any] = []
    monkeypatch.setattr(app, "analyze", lambda path: calls.append(("analyze", Path(path))))
    monkeypatch.setattr(app, "generate_dataset", lambda: calls.append(("dataset", None)))
    tab.btn_reanalyze.invoke()
    tab.btn_dataset.invoke()
    assert calls == [("analyze", Path(analysis.path)), ("dataset", None)]

    app.state.events.emit("busy_changed", busy=True)
    assert all(b.instate(["disabled"]) for b in (tab.btn_open, tab.btn_reanalyze, tab.btn_dataset, tab.chk_separate))
    app.state.events.emit("busy_changed", busy=False)
    assert all(b.instate(["!disabled"]) for b in (tab.btn_open, tab.btn_reanalyze, tab.btn_dataset))


def test_reproduccion_de_pista_y_segmento(tab: Any, analysis: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con reproductor: ▶ Todo reproduce ``y_raw``; ▶ Segmento solo el seleccionado; ■ detiene."""
    app = tab.app
    fake = FakePlayer()
    monkeypatch.setattr(tab, "player", fake)
    tab._update_controls()
    assert tab.btn_play_all.instate(["!disabled"]) and tab.btn_stop.instate(["!disabled"])
    assert tab.btn_play_segment.instate(["disabled"])  # aún no hay segmento seleccionado

    tab.btn_play_all.invoke()
    assert fake.calls[-1] == ("play", len(analysis.y_raw), analysis.sr)

    app.state.select_segment(2, source="test")
    pump(app, 0.1)
    assert tab.btn_play_segment.instate(["!disabled"])
    tab.btn_play_segment.invoke()
    segment = analysis.kept[2]
    assert fake.calls[-1] == ("play", segment.end_sample - segment.start_sample, analysis.sr)
    tab.btn_stop.invoke()
    assert fake.calls[-1][0] == "stop"


def test_pestana_oculta_redibuja_al_mostrarse(tab: Any, analysis: Any) -> None:
    """Con la pestaña oculta no se redibuja (ahorra ≈ 0.3 s por selección); al mostrarla, sí."""
    app = tab.app
    app.show_tab("log")
    pump(app, 0.2)
    app.state.select_segment(6, source="live")
    pump(app, 0.2)
    assert tab.drawn_selection != 6 and tab.redraw_pending
    app.show_tab("audio")
    settle(tab)
    assert tab.drawn_selection == 6


def test_ventana_pequena_usa_modo_compacto(tab: Any) -> None:
    """A 1024×700 la figura se compacta (sin subtítulo, etiquetas cortas) y todo sigue visible."""
    from matplotlib.text import Annotation

    app = tab.app
    app.root.geometry("1024x700+0+0")
    try:
        pump(app, 0.3)
        settle(tab)
        assert app.status.winfo_ismapped()
        assert tab.plot.canvas.get_tk_widget().winfo_height() > 200
        tab.draw_analysis()
        if tab.compact:
            fig = tab.plot.figure
            assert not any(isinstance(a, Annotation) for a in fig.artists)
            assert tab._data_axes()[1].get_ylabel() == "Hz (log)"
        else:  # lienzo suficientemente alto en este servidor X: al menos debe poder compactarse
            tab._make_compact()
            assert tab._data_axes()[1].get_ylabel() == "Hz (log)"
    finally:
        app.root.geometry("1360x880+0+0")
        pump(app, 0.3)
        settle(tab)
