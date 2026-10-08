"""Pestaña 1 · Audio: qué «escucha» el sistema antes de que actúen los bandits.

Papel en la interfaz
--------------------
Es la puerta de entrada del flujo de trabajo y la pestaña con la que se
explican en clase las etapas 1–6 del pipeline (:mod:`src.pipeline`): carga,
separación opcional, preprocesamiento, segmentación por onsets, pitch con pYIN
y espectro de la recompensa. No calcula nada por su cuenta: llama a las
ACCIONES de :class:`gui.app.SmartunerApp` (``open_audio``, ``analyze``,
``generate_dataset``), dibuja con :func:`src.plots.plot_audio_overview` y
empareja con el ground truth mediante :func:`src.experiments.match_segments_to_gt`.

Contenido::

    ┌ Barra: [Abrir MP3…] [Analizar de nuevo] ☐ Demucs [Generar dataset…]   [▶ Todo] [▶ Segmento] [■ Detener]
    ├ Aviso (solo si cambiaron parámetros de análisis desde el último análisis)
    ├ Figura: forma de onda + espectrograma con onsets, descartados y f0 (clic = seleccionar)
    ├──────────────────────── separador arrastrable ────────────────────────
    └ Información del audio │ Tabla de segmentos (n.º, inicio, fin, duración, f0, nota, …)

En ventanas pequeñas (≈ 1024×700) la reproducción pasa a una segunda línea si no
cabe, la figura se dibuja en modo compacto y el panel de información se desplaza.

Sincronización con las demás pestañas
-------------------------------------
* ``analysis_ready`` → rellena la información, la tabla y la figura.
* ``segment_selected`` → selecciona la fila y redibuja el resaltado (venga de
  esta pestaña, de la tablatura o de la ejecución en vivo). La pestaña publica
  sus selecciones con ``source="audio"``; como :meth:`gui.app.AppState.select_segment`
  ignora la posición ya seleccionada y aquí solo se actúa si la vista no está
  sincronizada, no hay bucles de eventos.
* ``config_changed`` → casilla de Demucs, tolerancia del emparejamiento con el
  ground truth y aviso de «analiza de nuevo».
* ``busy_changed`` → deshabilita las acciones mientras hay una tarea en curso.
* ``audio_output_claimed`` → si otra pestaña va a reproducir (la tablatura
  sintetizada), se detiene la reproducción de aquí; antes de reproducir, esta
  pestaña reclama la salida (:meth:`gui.app.SmartunerApp.claim_audio_output`)
  y la tablatura se pausa. Nunca suenan dos cosas ni queda un botón incoherente.

Por rendimiento la figura (≈ 0.3 s por redibujo) se redibuja de forma diferida
(se agrupan selecciones rápidas, p. ej. al recorrer la tabla con las flechas) y
solo cuando la pestaña está visible; al volver a ella se redibuja si hizo falta.
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from collections.abc import Hashable, Sequence
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk
from typing import TYPE_CHECKING, Any

import numpy as np

from gui.widgets import AudioPlayer, HoverTooltip, PlotFrame, Tooltip, show_error, tree_heading_at
from src import plots
from src.experiments import match_segments_to_gt
from src.pitch import midi_to_name
from src.separation import demucs_available, resolve_method

if TYPE_CHECKING:  # solo para las anotaciones de tipo (evita importes circulares)
    from gui.app import SmartunerApp
    from src.pipeline import AnalysisResult
    from src.segmentation import Segment
    from src.synth_dataset import GTNote

logger = logging.getLogger(__name__)

#: Identificador de esta pestaña como origen de los eventos ``segment_selected``.
SOURCE = "audio"

#: Espera (ms) antes de redibujar la figura: agrupa selecciones muy seguidas.
REDRAW_DELAY_MS = 40
#: Espera máxima (ms) de un redibujo por cambio de selección (pistas largas; ver ``_request_redraw``).
SELECTION_REDRAW_MAX_MS = 250

#: Fracción inicial de la altura dedicada a la figura (el resto es la tabla).
PLOT_HEIGHT_FRACTION = 0.63

#: Altura mínima (px) de la zona inferior (información + tabla).
MIN_BOTTOM_PX = 190

#: Altura mínima (px, con su barra de herramientas) de la figura al colocar el separador.
MIN_PLOT_PX = 340

#: Por debajo de esta altura del lienzo (px) la figura se dibuja en modo compacto:
#: sin subtítulo (sus datos ya están en el panel de información) y con etiquetas
#: de eje cortas; si no, *constrained layout* no cabe y los paneles colapsan.
COMPACT_HEIGHT_PX = 360

#: Pistas más largas que esto (s): al seleccionar una nota desde otra pestaña (o con
#: «Acercar a la nota») la figura se acerca a ella; con la pista entera, una nota de
#: 0.2 s en 6 min sería una línea de un píxel.
ZOOM_LONG_TRACK_S: float = 20.0

#: Margen (s) a cada lado de la nota al acercarse a ella.
ZOOM_MARGIN_S: float = 2.0

#: Ancho (px) de la columna de valores del panel de información, normal y en ventanas
#: estrechas (zona inferior de menos de NARROW_WIDTH_PX), para dejar sitio a la tabla.
INFO_WRAP_PX = 290
INFO_WRAP_NARROW_PX = 200
NARROW_WIDTH_PX = 1150

#: Altura (px) del marco de la tabla por debajo de la cual se oculta su leyenda.
LEGEND_MIN_HEIGHT_PX = 200

#: Color de fondo del aviso «analiza de nuevo» (ámbar muy claro, tinta oscura).
BANNER_BG = "#fdf3d8"
BANNER_FG = "#5c4400"

#: Colores de las filas descartadas de la tabla (gris recesivo, como en la figura).
DISCARDED_FG = plots.TEXT_MUTED
DISCARDED_BG = plots.HIGHLIGHT

#: Columnas de la tabla: (id, encabezado, ancho px, alineación, ayuda del encabezado).
COLUMNS: tuple[tuple[str, str, int, str, str], ...] = (
    ("pos", "N.º", 40, "center",
     "Posición del segmento entre los CONSERVADOS (empieza en 0). Es el mismo número que aparece "
     "sobre la forma de onda y el que usan la tablatura y la ejecución en vivo. Los descartados no "
     "tienen número porque no se transcriben."),
    ("start", "Inicio (s)", 66, "center",
     "Instante del onset que abre el segmento, en segundos desde el comienzo de la pista."),
    ("end", "Fin (s)", 58, "center",
     "Fin del segmento (s): el siguiente onset o, si antes hay silencio, el último frame cuyo RMS "
     "supera el umbral de silencio."),
    ("dur", "Dur. (ms)", 66, "center", "Duración del segmento en milisegundos (fin − inicio)."),
    ("f0", "f0 (Hz)", 60, "center",
     "Frecuencia fundamental del segmento según pYIN (mediana de los frames con voz, ignorando el "
     "ataque). «—» si pYIN no encontró voz suficiente."),
    ("note", "Nota", 50, "center",
     "Nota más cercana a la f0: m = 69 + 12·log₂(f0 / 440 Hz), redondeado al semitono "
     "(notación anglosajona, C4 = MIDI 60)."),
    ("voiced", "% voz", 56, "center",
     "Porcentaje de frames del segmento que pYIN considera con voz (con tono definido). Si es menor "
     "que el mínimo configurado, la f0 se considera desconocida."),
    ("status", "Estado", 64, "center",
     "ok = se transcribe; silencio = ningún frame supera el umbral de RMS; corto = dura menos que la "
     "duración mínima. Los descartados no son problemas bandit."),
    ("gt", "Real (GT)", 84, "center",
     "Nota del ground truth emparejada con el segmento (inicio a ≤ tolerancia de onset), con su "
     "posición cuerda-traste real. «≠» marca que pYIN detectó otra nota; «sin pareja» que ninguna "
     "nota real empieza cerca."),
)

#: Ancho mínimo (px) de una columna de la tabla de segmentos.
COLUMN_MIN_WIDTH: int = 34

#: Fuente de los encabezados de la tabla de segmentos.
HEADING_FONT: tuple[str, int, str] = ("TkDefaultFont", 9, "bold")

#: Ayuda de cada dato del panel de información: (clave, etiqueta, ayuda).
INFO_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("file", "Archivo", "Archivo de audio elegido. Pasa el ratón sobre el valor para ver la ruta completa."),
    ("source", "Señal analizada",
     "Qué señal se analiza de verdad. Sin «Separar bajo de mezcla», el archivo original. Con la casilla "
     "marcada depende del método de separación (Configuración): Demucs escribe un stem de bajo en la caché "
     "y se analiza ese archivo; HPSS (si Demucs no está instalado, con el método «auto») separa la parte "
     "armónica y aplica un pasa-bajas sobre el original en memoria, sin escribir ningún archivo."),
    ("duration", "Duración", "Duración de la pista en segundos."),
    ("sr", "Frec. de muestreo",
     "El audio se decodifica con ffmpeg a mono y se remuestrea a esta frecuencia. La frecuencia "
     "más alta representable (Nyquist) es la mitad."),
    ("onsets", "Onsets detectados",
     "Inicios de nota: picos de la función de novedad (flujo espectral log-mel) calculada sobre la "
     "señal filtrada con el pasa-bajas de 400 Hz."),
    ("segments", "Segmentos",
     "Cada segmento CONSERVADO es un problema bandit independiente. Se descartan los silenciosos "
     "(ningún frame supera el umbral de RMS) y los demasiado cortos (menos que la duración mínima)."),
    ("gt", "Ground truth",
     "Notas reales del archivo <nombre>.gt.json junto al audio (lo crea el dataset sintético). Se "
     "emparejan uno a uno con los segmentos cuyo inicio está a ≤ la tolerancia de onset del experimento."),
    ("spectrum", "Espectro (recompensa)",
     "Representación tiempo-frecuencia de la señal SIN filtrar sobre la que se calcula la recompensa "
     "(saliencia armónica). La figura muestra siempre una CQT (eje de frecuencia logarítmico) porque "
     "en ella los armónicos de cualquier nota forman el mismo «peine»."),
)

#: Texto del aviso cuando cambian parámetros que exigen repetir el análisis.
STALE_TEXT = ("⚠  Cambiaste parámetros del análisis (carga, separación, preprocesamiento, segmentación o "
              "pitch) desde el último análisis. Pulsa «Analizar de nuevo» para aplicarlos.")

#: Secciones de :class:`src.config.Config` que solo se aplican volviendo a analizar el audio.
ANALYSIS_SECTIONS: tuple[str, ...] = ("audio", "preprocess", "segmentation", "pitch")


def _plural(n: int, word: str) -> str:
    """``n`` seguido de ``word`` en singular o plural (añadiendo «s»).

    Examples
    --------
    >>> _plural(1, "nota"), _plural(3, "nota")
    ('1 nota', '3 notas')
    """
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class AudioTab(ttk.Frame):
    """Pestaña 1 · Audio: abrir/analizar un MP3 y explorar sus segmentos de nota.

    Parameters
    ----------
    master : tk.Misc
        Contenedor (el ``ttk.Notebook`` de la ventana principal).
    app : SmartunerApp
        Aplicación: da acceso al estado compartido (``app.state``), a las
        acciones (``open_audio``, ``analyze``, ``generate_dataset``) y a la
        barra de estado.

    Attributes
    ----------
    plot : PlotFrame
        Figura principal (forma de onda + espectrograma).
    tree : ttk.Treeview
        Tabla de segmentos; las filas conservadas tienen id ``"pos<n>"`` y las
        descartadas ``"disc<índice>"``.
    player : AudioPlayer
        Reproductor (puede no estar disponible; ver ``player.error``).
    separate_var : tk.BooleanVar
        Estado de la casilla «Separar bajo de mezcla» (Demucs o HPSS según la configuración).
    info_values : dict[str, ttk.Label]
        Etiquetas con los valores del panel de información (claves de :data:`INFO_FIELDS`).
    drawn_selection : int | None
        Segmento resaltado en la figura dibujada (None = ninguno).
    plot_state : str
        Qué muestra la figura: ``"empty"`` (mensaje guía) o ``"analysis"``.
    compact : bool
        True si la figura se dibujó en modo compacto (lienzo de menos de
        :data:`COMPACT_HEIGHT_PX` píxeles de alto).
    """

    def __init__(self, master: tk.Misc, app: SmartunerApp) -> None:
        """Construye la pestaña, se suscribe a los eventos de ``app.state`` y muestra el estado vacío."""
        super().__init__(master, padding=(12, 10, 12, 8))
        self.app = app
        self.player = AudioPlayer()
        self.demucs_ok = demucs_available()
        self.analysis: AnalysisResult | None = None
        self.drawn_selection: int | None = None
        self.plot_state = "empty"
        self.compact = False
        self._empty_size: tuple[int, int] = (0, 0)
        self._busy = bool(app.busy)
        self._row_position: dict[str, int | None] = {}
        self._position_row: dict[int, str] = {}
        self._gt_by_position: dict[int, GTNote] = {}
        self._kept_starts = np.zeros(0)
        self._kept_ends = np.zeros(0)
        self._redraw_job: str | None = None
        self._last_draw_ms = 0.0  # lo que tardó en construirse la última figura (ms)
        self._dirty = True
        self._keep_view = False
        self._zoom_selection_pending = False   # acercar la figura a la selección en el próximo dibujo
        self._home_view: tuple[tuple[float, float], ...] | None = None
        self._toolbar_wrapped: bool | None = None
        self._sash_user = False
        self._sash_job: str | None = None

        self._setup_styles()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_toolbar()
        self._build_banner()
        self._build_body()

        events = app.state.events
        events.subscribe("analysis_ready", self._on_analysis_ready)
        events.subscribe("segment_selected", self._on_segment_selected)
        events.subscribe("config_changed", self._on_config_changed)
        events.subscribe("busy_changed", self._on_busy_changed)
        events.subscribe("audio_output_claimed", self._on_audio_output_claimed)
        self.bind("<Map>", self._on_map, add="+")

        self._refresh_info()
        self._fill_tree()
        self._update_controls()
        self._draw_empty_state()
        if app.state.analysis is not None:  # p. ej. si la pestaña se crea después de analizar
            self._on_analysis_ready(app.state.analysis)

    # ================================================================ construcción
    def _setup_styles(self) -> None:
        """Estilos propios de la pestaña (con prefijo ``Audio.`` para no afectar a las demás)."""
        style = ttk.Style(self)
        style.configure("Audio.TLabelframe", padding=(10, 6, 10, 8))
        style.configure("Audio.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"), foreground=plots.TEXT_PRIMARY)
        style.configure("AudioKey.TLabel", foreground=plots.TEXT_SECONDARY)
        style.configure("AudioValue.TLabel", foreground=plots.TEXT_PRIMARY)
        style.configure("AudioHint.TLabel", foreground=plots.TEXT_SECONDARY, font=("TkDefaultFont", 9))
        style.configure("AudioEmpty.TLabel", foreground=plots.TEXT_SECONDARY, background="white",
                        font=("TkDefaultFont", 10))
        style.configure("AudioBanner.TLabel", background=BANNER_BG, foreground=BANNER_FG, padding=(10, 6))
        # Botones de la barra con relleno horizontal moderado y sin ancho mínimo (el tema
        # clam impone 9 caracteres): así la barra cabe en una línea a 1024 px de ancho.
        style.configure("AudioBar.TButton", padding=(6, 4), width=0)
        style.configure("AudioBar.Accent.TButton", padding=(6, 4), width=0)
        style.configure("Audio.Treeview", rowheight=22)
        style.configure("Audio.Treeview.Heading", font=HEADING_FONT)

    def _build_toolbar(self) -> None:
        """Barra superior: acciones de análisis a la izquierda y reproducción a la derecha."""
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        bar.columnconfigure(0, weight=1)
        self.toolbar = bar

        actions = ttk.Frame(bar)
        actions.grid(row=0, column=0, sticky="w")
        self.toolbar_actions = actions
        self.btn_open = ttk.Button(actions, text="Abrir MP3…", style="AudioBar.Accent.TButton",
                                   command=self._on_open)
        self.btn_open.pack(side="left")
        Tooltip(self.btn_open, "Elige un archivo de audio (MP3, WAV, FLAC…) con un bajo aislado y lo analiza: "
                               "carga → separación (opcional) → preprocesamiento → onsets → pitch → espectro, "
                               "y después lo transcribe con los cuatro algoritmos. Atajo: Ctrl+O.")
        self.btn_reanalyze = ttk.Button(actions, text="Analizar de nuevo", style="AudioBar.TButton",
                                        command=self._on_reanalyze)
        self.btn_reanalyze.pack(side="left", padx=(6, 0))
        Tooltip(self.btn_reanalyze, self._reanalyze_help)
        ttk.Separator(actions, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)

        self.separate_var = tk.BooleanVar(master=self, value=bool(self.app.state.config.audio.separate_bass))
        self.chk_separate = ttk.Checkbutton(actions, text="Separar bajo de mezcla",
                                            variable=self.separate_var, command=self._on_separate_toggled)
        self.chk_separate.pack(side="left")
        Tooltip(self.chk_separate, self._separate_help)
        ttk.Separator(actions, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)

        self.btn_dataset = ttk.Button(actions, text="Generar dataset sintético…", style="AudioBar.TButton",
                                      command=self._on_generate_dataset)
        self.btn_dataset.pack(side="left")
        Tooltip(self.btn_dataset, "Crea piezas de bajo de ejemplo a partir de MIDI (sintetizadas con FluidSynth o "
                                  "Karplus-Strong), las guarda como MP3 y escribe junto a cada una su ground truth "
                                  "(.gt.json con las notas y posiciones reales) para poder medir la precisión.")

        playback = ttk.Frame(bar)
        self.toolbar_playback = playback
        self.btn_play_all = ttk.Button(playback, text="▶ Todo", style="AudioBar.TButton", command=self._on_play_all)
        self.btn_play_all.pack(side="left")
        self.btn_play_segment = ttk.Button(playback, text="▶ Segmento", style="AudioBar.TButton",
                                           command=self._on_play_segment)
        self.btn_play_segment.pack(side="left", padx=(4, 0))
        self.btn_stop = ttk.Button(playback, text="■ Detener", style="AudioBar.TButton", command=self._on_stop)
        self.btn_stop.pack(side="left", padx=(4, 0))
        ttk.Separator(playback, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)
        self.btn_zoom_note = ttk.Button(playback, text="Acercar a la nota", style="AudioBar.TButton",
                                        command=self.zoom_to_selection)
        self.btn_zoom_note.pack(side="left")
        Tooltip(self.btn_zoom_note, f"Acerca la figura al segmento seleccionado (±{ZOOM_MARGIN_S:g} s) para ver sus "
                                    "onsets, su f0 y su espectro. En pistas de más de "
                                    f"{ZOOM_LONG_TRACK_S:g} s se hace solo al elegir una nota en otra pestaña. "
                                    "«Inicio» (la casa de la barra de la figura) vuelve a la pista completa.")
        Tooltip(self.btn_play_all, lambda: self._play_help("Reproduce la pista completa tal como se analizó "
                                                           "(el stem de bajo si hubo separación)."))
        Tooltip(self.btn_play_segment, lambda: self._play_help("Reproduce solo el segmento seleccionado (clic en "
                                                               "la figura o en la tabla; doble clic en la tabla "
                                                               "también lo reproduce)."))
        Tooltip(self.btn_stop, lambda: self._play_help("Detiene la reproducción en curso."))
        self._layout_toolbar(wrapped=False)
        bar.bind("<Configure>", self._on_toolbar_configure, add="+")

    def _build_banner(self) -> None:
        """Aviso (oculto por defecto) de que hay que volver a analizar para aplicar cambios."""
        self.banner = ttk.Label(self, text=STALE_TEXT, style="AudioBanner.TLabel", anchor="w", wraplength=1200)
        self.banner.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.banner.grid_remove()
        self.banner.bind("<Configure>", lambda e: self.banner.configure(wraplength=max(200, e.width - 24)), add="+")

    def _build_body(self) -> None:
        """Figura arriba y, abajo, información del audio + tabla de segmentos (separador arrastrable)."""
        # Altura PEDIDA fija y modesta: al mover el separador, ttk fija el tamaño pedido de
        # cada panel al actual y la pestaña acabaría pidiendo más alto que la ventana.
        paned = ttk.PanedWindow(self, orient="vertical", height=440)
        paned.grid(row=2, column=0, sticky="nsew")
        self.paned = paned

        # La figura va dentro de un marco que NO propaga su tamaño pedido: el lienzo de
        # matplotlib puede pedir el tamaño que tuvo en su último redimensionado y, si la
        # pestaña pidiera más alto que la ventana, la barra de estado quedaría fuera.
        holder = ttk.Frame(paned, width=640, height=200)
        holder.pack_propagate(False)
        self.plot = PlotFrame(holder, figsize=(6.4, 2.0))
        self.plot.pack(fill="both", expand=True)
        self.plot.canvas.mpl_connect("button_press_event", self._on_figure_click)
        self.plot.canvas.get_tk_widget().bind("<Configure>", self._on_canvas_configure, add="+")
        if self.plot.toolbar is not None:
            hint = tk.Label(self.plot.toolbar, text="Clic en la figura: seleccionar el segmento de ese instante",
                            foreground=plots.TEXT_SECONDARY, background=self.plot.toolbar.cget("background"),
                            font=("TkDefaultFont", 9))
            hint.pack(side="left", padx=(12, 0))
        paned.add(holder, weight=3)

        bottom = ttk.Frame(paned, padding=(0, 8, 0, 0))
        self.bottom = bottom
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)
        paned.add(bottom, weight=1)
        self._build_info_panel(bottom)
        self._build_segment_table(bottom)
        bottom.bind("<Configure>", self._on_bottom_configure, add="+")
        paned.bind("<Configure>", self._on_paned_configure, add="+")
        paned.bind("<ButtonPress-1>", self._on_sash_drag, add="+")

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        """Panel «Información del audio» (pares etiqueta → valor con ayuda).

        El contenido va en un lienzo desplazable: en ventanas bajas (1024×700) la
        zona inferior puede ser más baja que el panel y aparece una barra de
        desplazamiento en lugar de cortar las últimas filas.
        """
        outer = ttk.LabelFrame(parent, text="Información del audio", style="Audio.TLabelframe")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        background = ttk.Style(self).lookup("TFrame", "background") or plots.HIGHLIGHT
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, background=background, height=60, width=200)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        self._info_canvas = canvas
        self._info_scroll = scroll
        self._info_inner = frame

        frame.columnconfigure(1, weight=1)
        self.info_values: dict[str, ttk.Label] = {}
        for row, (key, label, help_text) in enumerate(INFO_FIELDS):
            key_label = ttk.Label(frame, text=label, style="AudioKey.TLabel")
            key_label.grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=1)
            value = ttk.Label(frame, text="—", style="AudioValue.TLabel", wraplength=INFO_WRAP_PX, justify="left")
            value.grid(row=row, column=1, sticky="nw", pady=1)
            Tooltip(key_label, help_text)
            self.info_values[key] = value
            for widget in (key_label, value):
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(sequence, self._on_info_wheel, add="+")
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind(sequence, self._on_info_wheel, add="+")
        Tooltip(self.info_values["file"], self._file_help)
        Tooltip(self.info_values["source"], self._source_help)
        frame.bind("<Configure>", self._on_info_configure, add="+")
        canvas.bind("<Configure>", self._on_info_configure, add="+")

    def _on_info_configure(self, _event: tk.Event | None = None) -> None:
        """Ajusta el lienzo del panel a su contenido y muestra la barra solo si hace falta."""
        canvas, inner = self._info_canvas, self._info_inner
        width, height = inner.winfo_reqwidth(), inner.winfo_reqheight()
        canvas.configure(scrollregion=(0, 0, width, height))
        if int(canvas.cget("width")) != width or int(canvas.cget("height")) != height:
            canvas.configure(width=width, height=height)  # tamaño PEDIDO = contenido completo
        visible = canvas.winfo_height()
        if visible > 1 and height > visible + 1:
            self._info_scroll.grid(row=0, column=1, sticky="ns")
        else:
            self._info_scroll.grid_remove()
            canvas.yview_moveto(0.0)

    def _on_info_wheel(self, event: tk.Event) -> str | None:
        """Rueda del ratón sobre el panel de información (solo si hay barra de desplazamiento)."""
        if not self._info_scroll.winfo_ismapped():
            return None
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self._info_canvas.yview_scroll(-1, "units")
        else:
            self._info_canvas.yview_scroll(1, "units")
        return "break"

    def _build_segment_table(self, parent: ttk.Frame) -> None:
        """Tabla de segmentos con barra de desplazamiento, mensaje de tabla vacía y leyenda."""
        frame = ttk.LabelFrame(parent, text="Segmentos detectados", style="Audio.TLabelframe")
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.segments_frame = frame

        ids = [c[0] for c in COLUMNS]
        tree = ttk.Treeview(frame, columns=ids, show="headings", selectmode="browse", height=5,
                            style="Audio.Treeview")
        for cid, heading, width, anchor, _help in COLUMNS:
            tree.heading(cid, text=heading, anchor=anchor)
            tree.column(cid, width=width, minwidth=COLUMN_MIN_WIDTH, anchor=anchor, stretch=True)
        tree.tag_configure("discarded", foreground=DISCARDED_FG, background=DISCARDED_BG)
        self._shown_columns: tuple[str, ...] = tuple(ids)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        self.tree = tree

        tree.bind("<<TreeviewSelect>>", self._on_tree_select, add="+")
        tree.bind("<ButtonPress-1>", self._on_tree_press, add="+")
        tree.bind("<Double-Button-1>", self._on_tree_double_click, add="+")
        # Al cambiar el ancho de la tabla (ventana redimensionada, pestaña que se muestra por
        # primera vez) se vuelven a repartir las columnas: así «Real (GT)» nunca queda fuera.
        self._fitted_width = 0
        tree.bind("<Configure>", lambda _e: self._fit_columns(), add="+")
        self._tree_tooltip = HoverTooltip(tree, self._tree_zone, self._tree_help, delay_ms=500, wraplength=340)

        self.empty_table = ttk.Label(frame, text="", style="AudioEmpty.TLabel", anchor="center", justify="center")
        self.legend = ttk.Label(
            frame, style="AudioHint.TLabel",
            text="Gris = segmento descartado (no se transcribe) · Clic en una fila: seleccionarla en todas las "
                 "pestañas · Pasa el ratón por los encabezados para ver qué significa cada columna.",
            wraplength=600, justify="left")
        self.legend.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.legend.bind("<Configure>", lambda e: self.legend.configure(wraplength=max(200, e.width - 4)), add="+")
        frame.bind("<Configure>", self._on_segments_configure, add="+")

    # ============================================================== distribución
    def _layout_toolbar(self, wrapped: bool) -> None:
        """Coloca el bloque de reproducción a la derecha o, si no cabe, en una segunda línea."""
        if wrapped == self._toolbar_wrapped:
            return
        self._toolbar_wrapped = wrapped
        if wrapped:
            self.toolbar_playback.grid(row=1, column=0, sticky="w", pady=(8, 0))
        else:
            self.toolbar_playback.grid(row=0, column=1, sticky="e", padx=(10, 0))

    def _on_toolbar_configure(self, event: tk.Event) -> None:
        """Pasa la reproducción a una segunda línea cuando la ventana es estrecha."""
        needed = self.toolbar_actions.winfo_reqwidth() + self.toolbar_playback.winfo_reqwidth() + 10
        self._layout_toolbar(wrapped=event.width < needed)

    def _on_paned_configure(self, event: tk.Event) -> None:
        """Reparte la altura entre figura y zona inferior hasta que el usuario mueva el separador."""
        if self._sash_user or event.height < 200:
            return
        self.after_idle(lambda: self._place_sash(self._sash_target(event.height)))

    def _reapply_sash(self) -> None:
        """Recoloca el separador si cambió el alto que pide la zona inferior (p. ej. nuevo análisis)."""
        try:
            height = self.paned.winfo_height()
        except tk.TclError:  # la ventana se cerró antes
            return
        if not self._sash_user and height >= 200:
            self._place_sash(self._sash_target(height))

    def _on_bottom_configure(self, event: tk.Event) -> None:
        """En ventanas estrechas estrecha el panel de información; si cambió de alto, recoloca el separador.

        ttk.PanedWindow vuelve a repartir el alto cuando cambia el tamaño PEDIDO
        de un panel (p. ej. el panel de información con los textos de un análisis
        nuevo) y podía dejar la figura por debajo de :data:`MIN_PLOT_PX` a
        1024×700 (sus paneles colapsaban). Mientras el usuario no haya movido el
        separador, se vuelve a colocar donde corresponde.
        """
        wrap = INFO_WRAP_NARROW_PX if event.width < NARROW_WIDTH_PX else INFO_WRAP_PX
        for label in self.info_values.values():
            if int(label.cget("wraplength")) != wrap:
                label.configure(wraplength=wrap)
        if not self._sash_user and self._sash_job is None:
            self._sash_job = self.after_idle(self._reapply_sash_later)

    def _reapply_sash_later(self) -> None:
        """Callback diferido de :meth:`_on_bottom_configure` (agrupa varios ``<Configure>``)."""
        self._sash_job = None
        self._reapply_sash()

    def _on_segments_configure(self, event: tk.Event) -> None:
        """Oculta la leyenda de la tabla cuando la zona inferior es muy baja (más filas visibles)."""
        if event.height < LEGEND_MIN_HEIGHT_PX:
            self.legend.grid_remove()
        else:
            self.legend.grid()

    def _sash_target(self, height: int) -> int:
        """Posición del separador (px desde arriba) para una altura total ``height``.

        La figura recibe :data:`PLOT_HEIGHT_FRACTION` de la altura, o menos si la
        zona inferior no cabe entera, pero nunca menos de :data:`MIN_PLOT_PX`
        (por debajo, sus paneles serían ilegibles): entonces se desplaza el panel
        de información y la tabla muestra menos filas.
        """
        bottom = max(self.bottom.winfo_reqheight(), MIN_BOTTOM_PX)
        position = min(int(height * PLOT_HEIGHT_FRACTION), height - bottom)
        return max(position, min(MIN_PLOT_PX, height - MIN_BOTTOM_PX))

    def _place_sash(self, position: int) -> None:
        """Mueve el separador de la figura a ``position`` píxeles desde arriba."""
        try:
            if abs(self.paned.sashpos(0) - position) > 2:
                self.paned.sashpos(0, max(position, 120))
        except tk.TclError:  # la ventana se cerró antes
            pass

    def _on_sash_drag(self, event: tk.Event) -> None:
        """El usuario arrastró el separador: a partir de ahora se respeta su posición."""
        if self.paned.identify(event.x, event.y) != "":
            self._sash_user = True

    # ================================================================= ayudas
    def _reanalyze_help(self) -> str:
        """Tooltip de «Analizar de nuevo» (menciona el archivo actual)."""
        base = ("Repite el análisis completo (etapas 1–6) del archivo actual con la configuración vigente. "
                "Úsalo después de cambiar parámetros de preprocesamiento, segmentación o pitch en la pestaña "
                "Configuración (los del entorno bandit solo necesitan Ejecutar → Re-transcribir).")
        if self.analysis is None:
            return base + "\n\nPrimero abre un audio."
        return base + f"\n\nArchivo: {Path(self.analysis.path).name}"

    def _separate_help(self) -> str:
        """Tooltip de la casilla de separación: qué método se usará y cómo obtener Demucs."""
        base = ("Actívala solo si el MP3 es una canción completa (mezcla); si el audio ya es un bajo "
                "aislado, déjala desactivada.")
        method = self.app.state.config.audio.separation_method
        effective = resolve_method(method) if method == "auto" else method
        demucs_text = ("Demucs (red neuronal de separación de fuentes) aísla la pista de bajo y se analiza ese "
                       "«stem». Es el método de mayor calidad pero tarda varios minutos por canción en CPU; "
                       "el resultado se guarda en la caché.")
        hpss_text = ("HPSS (separación armónico-percusiva) + pasa-bajas: rápido y sin PyTorch. Quita la "
                     "percusión y los efectos, pero no separa guitarras o teclados graves.")
        if effective == "demucs":
            detail = "Método actual: Demucs.\n" + demucs_text
        else:
            detail = "Método actual: HPSS.\n" + hpss_text
            if not self.demucs_ok:
                detail += ("\n\nPara usar Demucs instala las dependencias opcionales (incluye PyTorch):\n"
                           "    pip install -r requirements-optional.txt\n"
                           "y reinicia Smartuner (la primera separación descarga ~80 MB de pesos).")
        return (base + "\n\n" + detail + "\n\nEl método se elige en Configuración (audio.separation_method). "
                "Al cambiar la casilla pulsa «Analizar de nuevo».")

    def _play_help(self, text: str) -> str:
        """Tooltip de un botón de reproducción (o el motivo por el que está deshabilitado)."""
        if not self.player.available:
            return self.player.error
        return text

    def _file_help(self) -> str:
        """Ruta completa del archivo elegido."""
        return str(self.analysis.path) if self.analysis is not None else ""

    def _source_help(self) -> str:
        """Ruta completa de la señal analizada."""
        return str(self.analysis.source_path) if self.analysis is not None else ""

    def _tree_zone(self, x: int, y: int) -> tuple[str, str] | None:
        """Zona de la tabla en ``(x, y)``: ``("heading", columna)``, ``("row", fila)`` o None."""
        column = tree_heading_at(self.tree, x, y)
        if column is not None:
            return ("heading", column)
        row = self.tree.identify_row(y)
        return ("row", row) if row else None

    def _tree_help(self, zone: Hashable) -> str:
        """Texto del tooltip de la tabla según la zona (encabezado: la columna; fila descartada: el motivo).

        Sobre las filas conservadas no se muestra nada (el tooltip no tapa la tabla).
        """
        region, ident = zone  # type: ignore[misc]
        if region == "heading":
            for cid, heading, _w, _a, help_text in COLUMNS:
                if cid == ident:
                    return f"{heading}: {help_text}"
            return ""
        segment = self._segment_for_row(ident)
        if segment is None or segment.kept or self.analysis is None:
            return ""
        seg_cfg = self.analysis.config.segmentation
        if segment.reason == "silencio":
            return (f"Descartado por silencio: ningún frame supera el umbral de {seg_cfg.rms_threshold_db:g} dB "
                    f"respecto al frame más fuerte de la pista (RMS medio {segment.rms_db:.1f} dB). "
                    "No se transcribe.")
        if segment.reason == "corto":
            return (f"Descartado por corto: dura {1000 * segment.duration_s:.0f} ms, menos que la duración "
                    f"mínima de {1000 * seg_cfg.min_duration_s:.0f} ms (no hay frames suficientes para pYIN "
                    "ni para muestrear recompensas). No se transcribe.")
        return f"Descartado ({segment.reason}). No se transcribe."

    # ============================================================ eventos de app
    def _on_analysis_ready(self, analysis: AnalysisResult, **_kwargs: Any) -> None:
        """Muestra un análisis nuevo (o el mismo recalculado tras re-transcribir).

        Parameters
        ----------
        analysis : AnalysisResult
            Resultado publicado por la aplicación.
        """
        previous = self.analysis
        same_track = (previous is not None and Path(previous.path) == Path(analysis.path)
                      and len(previous.y_raw) == len(analysis.y_raw))
        if not same_track:
            self.player.stop()
        self.analysis = analysis
        kept = analysis.kept
        self._kept_starts = np.array([s.start_s for s in kept], dtype=float)
        self._kept_ends = np.array([s.end_s for s in kept], dtype=float)
        self._compute_gt_matches()
        self._fill_tree()
        self._refresh_info()
        self._update_banner()
        self._update_controls()
        self._request_redraw(keep_view=same_track)

    def _on_segment_selected(self, position: int | None, source: str = "", **_kwargs: Any) -> None:
        """Sincroniza la fila seleccionada y el resaltado de la figura.

        Parameters
        ----------
        position : int | None
            Posición (entre los segmentos conservados) seleccionada, o None.
        source : str
            Pestaña que originó la selección (``"audio"`` si fue esta).
        """
        position = self._valid_position(position)
        self._sync_tree_selection(position)
        self._update_controls()
        if (source != SOURCE and position is not None and self.analysis is not None
                and self.analysis.duration_s > ZOOM_LONG_TRACK_S):
            # Elegida en otra pestaña (Tablatura, En vivo…): en una pista larga se acerca a ella.
            self._zoom_selection_pending = True
            if self.plot_state == "analysis" and self.drawn_selection == position:
                self._request_redraw(keep_view=True, trailing=True)
        if self.analysis is not None and (self.plot_state != "analysis" or self.drawn_selection != position):
            self._request_redraw(keep_view=True, trailing=True)

    def _on_config_changed(self, key: str | None = None, **_kwargs: Any) -> None:
        """Refleja cambios de configuración hechos en otras pestañas (o al cargar un JSON).

        Parameters
        ----------
        key : str | None
            Parámetro modificado (``"seccion.campo"``) o None si cambiaron varios.
        """
        cfg = self.app.state.config
        if key in (None, "audio.separate_bass") and self.separate_var.get() != bool(cfg.audio.separate_bass):
            self.separate_var.set(bool(cfg.audio.separate_bass))
        if self.analysis is not None and key in (None, "experiment.onset_tolerance_s"):
            self._compute_gt_matches()
            self._fill_tree()
            self._refresh_info()
        self._update_banner()

    def _on_busy_changed(self, busy: bool = False, **_kwargs: Any) -> None:
        """Deshabilita las acciones mientras hay una tarea en segundo plano.

        Parameters
        ----------
        busy : bool
            True si empezó una tarea; False si terminó.
        """
        self._busy = bool(busy)
        self._update_controls()
        if self.analysis is None:
            self._draw_empty_state()

    def _on_map(self, _event: tk.Event) -> None:
        """Al mostrarse la pestaña, redibuja la figura si quedó pendiente."""
        if self._dirty:
            self._request_redraw(keep_view=self._keep_view)

    # ======================================================== acciones de usuario
    def _on_open(self) -> None:
        """Botón «Abrir MP3…»: diálogo de archivo y análisis (acción de la aplicación)."""
        self.app.open_audio()

    def _on_reanalyze(self) -> None:
        """Botón «Analizar de nuevo»: repite el análisis del archivo actual con la configuración vigente."""
        if self.analysis is None:
            self.app.status.set_message("Primero abre un archivo de audio.")
            return
        self.app.analyze(Path(self.analysis.path))

    def _on_generate_dataset(self) -> None:
        """Botón «Generar dataset sintético…» (acción de la aplicación)."""
        self.app.generate_dataset()

    def _on_separate_toggled(self) -> None:
        """Casilla de separación: escribe ``config.audio.separate_bass`` y avisa con ``config_changed``."""
        value = bool(self.separate_var.get())
        cfg = self.app.state.config
        if value == bool(cfg.audio.separate_bass):
            return
        cfg.audio.separate_bass = value
        logger.info("Separación del bajo %s (método: %s; se aplicará al volver a analizar).",
                    "activada" if value else "desactivada", cfg.audio.separation_method)
        self.app.state.events.emit("config_changed", key="audio.separate_bass")

    def _on_play_all(self) -> None:
        """Reproduce la pista completa (``analysis.y_raw``)."""
        if self.analysis is None:
            return
        self._play(self.analysis.y_raw, f"Reproduciendo la pista completa ({self.analysis.duration_s:.1f} s)…")

    def _on_play_segment(self) -> None:
        """Reproduce el segmento seleccionado."""
        segment = self._selected_segment()
        if self.analysis is None or segment is None:
            self.app.status.set_message("Selecciona un segmento (clic en la figura o en la tabla) para escucharlo.")
            return
        y = self.analysis.y_raw[segment.start_sample:segment.end_sample]
        position = self._valid_position(self.app.state.selected_position)
        self._play(y, f"Reproduciendo el segmento {position} ({segment.start_s:.2f}–{segment.end_s:.2f} s)…")

    def _on_stop(self) -> None:
        """Detiene la reproducción."""
        self.player.stop()
        self.app.status.set_message("Reproducción detenida.")

    def _on_audio_output_claimed(self, owner: str = "", **_kwargs: Any) -> None:
        """Otra pestaña va a reproducir (o la ventana se cierra): se detiene la reproducción de aquí."""
        if owner != SOURCE:
            self.player.stop()

    def _play(self, y: np.ndarray, message: str) -> None:
        """Reproduce ``y`` a la frecuencia del análisis, mostrando los errores en un diálogo."""
        if not self.player.available or self.analysis is None:
            return
        # Una sola salida de audio: la pestaña Tablatura pausa su reproducción (si la había).
        self.app.claim_audio_output(SOURCE)
        try:
            self.player.play(y, self.analysis.sr)
        except Exception as exc:  # noqa: BLE001 - un fallo de audio no debe romper la GUI
            show_error(self.winfo_toplevel(), "No se pudo reproducir el audio", exc)
            return
        self.app.status.set_message(message)

    def _on_figure_click(self, event: Any) -> None:
        """Clic en la figura → selecciona el segmento conservado que contiene ese instante.

        Se ignora si la barra de matplotlib está en modo zoom o desplazamiento,
        si no es el botón izquierdo o si el clic cae fuera de los ejes de datos.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Evento ``button_press_event``.
        """
        if self.analysis is None or self.plot_state != "analysis" or getattr(event, "button", None) != 1:
            return
        if self._toolbar_active() or event.inaxes is None or event.xdata is None:
            return
        if getattr(event.inaxes, "_colorbar", None) is not None:
            return
        t = float(event.xdata)
        position = self.position_at(t)
        if position is None:
            self.app.status.set_message(f"En t = {t:.2f} s no hay ningún segmento conservado "
                                        "(silencio, segmento descartado o fuera de la pista).")
            return
        self.app.state.select_segment(position, source=SOURCE)

    def _on_tree_press(self, event: tk.Event) -> str | None:
        """Impide seleccionar con el ratón las filas descartadas (no son problemas bandit)."""
        self._tree_tooltip.hide()
        row = self.tree.identify_row(event.y)
        if row and self._row_position.get(row, -1) is None:
            self.app.status.set_message("Ese segmento está descartado (silencio o demasiado corto): no se transcribe.")
            return "break"
        return None

    def _on_tree_double_click(self, event: tk.Event) -> None:
        """Doble clic en una fila conservada: la reproduce si hay reproductor."""
        row = self.tree.identify_row(event.y)
        if row and self._row_position.get(row) is not None:
            if self.player.available:
                self._on_play_segment()
            else:
                self.zoom_to_selection()  # sin reproductor, el doble clic acerca la figura a la nota

    def _on_tree_select(self, _event: tk.Event | None = None) -> None:
        """Fila seleccionada → ``app.state.select_segment`` (las descartadas se saltan)."""
        selection = self.tree.selection()
        if not selection:
            return
        row = selection[0]
        position = self._row_position.get(row)
        if position is None:
            # Con el teclado se puede llegar a una fila descartada: se salta a la
            # siguiente conservada en la dirección del movimiento (o se vuelve atrás).
            target = self._nearest_kept_row(row)
            if target is None:
                self.tree.selection_remove(row)
            else:
                self.tree.selection_set(target)
                self.tree.focus(target)
                self.tree.see(target)
            return
        self.app.state.select_segment(position, source=SOURCE)

    # =========================================================== tabla de segmentos
    def _compute_gt_matches(self) -> None:
        """Empareja segmentos conservados ↔ notas del ground truth con la tolerancia vigente."""
        self._gt_by_position = {}
        analysis = self.analysis
        if analysis is None or not analysis.ground_truth:
            return
        tolerance = float(self.app.state.config.experiment.onset_tolerance_s)
        matches = match_segments_to_gt(analysis.kept, analysis.ground_truth, tolerance)
        self._gt_by_position = {pos: analysis.ground_truth[k] for k, pos in enumerate(matches) if pos is not None}

    def _fill_tree(self) -> None:
        """Rellena la tabla con todos los segmentos (los descartados en gris)."""
        tree = self.tree
        tree.delete(*tree.get_children())
        self._row_position = {}
        self._position_row = {}
        analysis = self.analysis
        has_gt = analysis is not None and bool(analysis.ground_truth)
        all_ids = [c[0] for c in COLUMNS]
        self._set_display_columns(tuple(all_ids if has_gt else [c for c in all_ids if c != "gt"]))
        if analysis is None:
            self._show_table_message("Todavía no hay segmentos.\nAbre un MP3 o genera el dataset sintético.")
            self.segments_frame.configure(text="Segmentos detectados")
            return
        position = 0
        for segment in analysis.segments:
            if segment.kept:
                row = f"pos{position}"
                self._row_position[row] = position
                self._position_row[position] = row
                values = self._row_values(segment, position, has_gt)
                tree.insert("", "end", iid=row, values=values, tags=("kept",))
                position += 1
            else:
                row = f"disc{segment.index}"
                self._row_position[row] = None
                tree.insert("", "end", iid=row, values=self._row_values(segment, None, has_gt), tags=("discarded",))
        n_kept = position
        n_discarded = len(analysis.segments) - n_kept
        self.segments_frame.configure(
            text=f"Segmentos detectados · {_plural(n_kept, 'conservado')}"
                 + (f", {_plural(n_discarded, 'descartado')}" if n_discarded else ""))
        if not analysis.segments:
            self._show_table_message("No se detectó ninguna nota en este audio.\n"
                                     "¿Es silencio o ruido? Revisa el log y los umbrales de segmentación.")
        else:
            self._show_table_message(None)
        self._sync_tree_selection(self._valid_position(self.app.state.selected_position))

    def _row_values(self, segment: Segment, position: int | None, has_gt: bool) -> tuple[str, ...]:
        """Valores de una fila de la tabla (ver :data:`COLUMNS`)."""
        f0 = segment.f0_hz
        midi = segment.midi
        note = midi_to_name(int(round(midi))) if midi is not None else "—"
        gt_text = ""
        if has_gt and position is not None:
            gt_note = self._gt_by_position.get(position)
            if gt_note is None:
                gt_text = "sin pareja"
            else:
                gt_text = f"{midi_to_name(gt_note.midi)} · {gt_note.label}"
                if midi is None or int(round(midi)) != int(gt_note.midi):
                    gt_text = "≠ " + gt_text
        elif has_gt:
            gt_text = "—"
        return (
            str(position) if position is not None else "–",
            f"{segment.start_s:.3f}",
            f"{segment.end_s:.3f}",
            f"{1000.0 * segment.duration_s:.0f}",
            f"{f0:.1f}" if f0 is not None else "—",
            note,
            f"{100.0 * segment.voiced_ratio:.0f} %",
            segment.reason,
            gt_text,
        )

    def _set_display_columns(self, columns: tuple[str, ...]) -> None:
        """Cambia las columnas visibles y reparte el ancho disponible entre ellas.

        ``ttk.Treeview`` solo reajusta los anchos al redimensionarse: si al
        mostrar una columna más no se reparten, las últimas quedan fuera de la vista.
        """
        if columns == self._shown_columns:
            return
        self._shown_columns = columns
        self.tree.configure(displaycolumns=list(columns))
        self._fitted_width = 0
        self._fit_columns()

    def _fit_columns(self) -> None:
        """Reparte el ancho ACTUAL de la tabla entre las columnas visibles, en proporción a su ancho nominal.

        Escala hacia arriba y hacia abajo (sin bajar de :data:`COLUMN_MIN_WIDTH`).
        Si la pestaña está oculta (``winfo_width() ≤ 1``, p. ej. el análisis
        terminó con otra pestaña visible) no hace nada: el ``<Configure>`` que
        llega al mostrarse la tabla vuelve a llamarla con el ancho real. Antes
        los anchos se calculaban con la pestaña oculta y la columna «Real (GT)»
        quedaba fuera de la vista.
        """
        available = self.tree.winfo_width() - 4  # bordes del Treeview
        columns = self._shown_columns
        if available <= COLUMN_MIN_WIDTH or not columns or available == self._fitted_width:
            return
        self._fitted_width = available
        nominal = {cid: width for cid, _h, width, _a, _help in COLUMNS}
        total = sum(nominal[c] for c in columns)
        # Cada columna recibe primero lo que necesita su encabezado y el resto del ancho se
        # reparte en proporción al ancho nominal; si ni los encabezados caben, se escala todo.
        floors = self._heading_widths()
        need = sum(floors[c] for c in columns)
        if available >= need:
            extra = available - need
            widths = {cid: floors[cid] + int(extra * nominal[cid] / total) for cid in columns}
        else:
            widths = {cid: max(COLUMN_MIN_WIDTH, int(nominal[cid] * available / total)) for cid in columns}
        # Redondeos y mínimos: el sobrante (o lo que falte) se ajusta en la columna más ancha.
        widest = max(columns, key=lambda c: widths[c])
        widths[widest] = max(COLUMN_MIN_WIDTH, widths[widest] + available - sum(widths.values()))
        for cid in columns:
            self.tree.column(cid, width=widths[cid])

    def _heading_widths(self) -> dict[str, int]:
        """Ancho mínimo (px) de cada columna: el de su encabezado en negrita más el relleno."""
        if not hasattr(self, "_heading_floor"):
            heading_font = tkfont.Font(root=self, font=HEADING_FONT)
            self._heading_floor = {cid: max(COLUMN_MIN_WIDTH, heading_font.measure(heading) + 14)
                                   for cid, heading, _w, _a, _help in COLUMNS}
        return self._heading_floor

    def _show_table_message(self, text: str | None) -> None:
        """Muestra (o quita) un mensaje centrado sobre la tabla vacía."""
        if text:
            self.empty_table.configure(text=text)
            self.empty_table.place(in_=self.tree, relx=0.5, rely=0.58, anchor="center")
            self.empty_table.lift()
        else:
            self.empty_table.place_forget()

    def _sync_tree_selection(self, position: int | None) -> None:
        """Selecciona en la tabla la fila de ``position`` (o ninguna) si no lo está ya."""
        target = self._position_row.get(position) if position is not None else None
        current = tuple(self.tree.selection())
        if target is None:
            if current:
                self.tree.selection_remove(*current)
            return
        if current != (target,):
            self.tree.selection_set(target)
        self.tree.focus(target)
        self.tree.see(target)

    def _nearest_kept_row(self, row: str) -> str | None:
        """Fila conservada más cercana a ``row`` en la dirección del último movimiento."""
        rows = list(self.tree.get_children())
        if row not in rows:
            return None
        index = rows.index(row)
        current = self._valid_position(self.app.state.selected_position)
        current_row = self._position_row.get(current) if current is not None else None
        previous = rows.index(current_row) if current_row in rows else -1
        step = -1 if previous > index else 1
        for direction in (step, -step):
            i = index + direction
            while 0 <= i < len(rows):
                if self._row_position.get(rows[i]) is not None:
                    return rows[i]
                i += direction
        return None

    def _segment_for_row(self, row: str) -> Segment | None:
        """Segmento mostrado en la fila ``row`` (None si no existe)."""
        if self.analysis is None or row not in self._row_position:
            return None
        position = self._row_position[row]
        if position is not None:
            kept = self.analysis.kept
            return kept[position] if position < len(kept) else None
        index = int(row.removeprefix("disc"))
        for segment in self.analysis.segments:
            if segment.index == index:
                return segment
        return None

    # ===================================================================== panel
    def _refresh_info(self) -> None:
        """Actualiza el panel «Información del audio»."""
        values = self.info_values
        analysis = self.analysis
        if analysis is None:
            for label in values.values():
                label.configure(text="—")
            values["file"].configure(text="Ningún audio cargado")
            values["gt"].configure(text="—")
            return
        path = Path(analysis.path)
        source = Path(analysis.source_path)
        values["file"].configure(text=path.name)
        values["source"].configure(text=self._source_summary(analysis))
        values["duration"].configure(text=f"{analysis.duration_s:.2f} s")
        values["sr"].configure(text=f"{analysis.sr:,} Hz".replace(",", " "))
        values["onsets"].configure(text=str(len(analysis.onsets_s)))
        values["segments"].configure(text=self._segments_summary(analysis))
        values["gt"].configure(text=self._gt_summary(analysis))
        values["spectrum"].configure(text=self._spectrum_summary(analysis))
        # Los textos nuevos pueden ocupar más líneas: se recoloca el separador cuando
        # tkinter haya recalculado los tamaños pedidos.
        self.after(60, self._reapply_sash)

    @staticmethod
    def _source_summary(analysis: AnalysisResult) -> str:
        """Texto de «Señal analizada» según la separación que se aplicó DE VERDAD.

        ``source_path`` no basta: HPSS no escribe ningún archivo, así que con
        HPSS ``source_path == path`` igual que sin separar. Se usa
        ``analysis.separation_method`` (``"demucs"``, ``"hpss"`` o None).

        Examples
        --------
        «bajo.wav (stem de bajo de Demucs)», «el archivo original, separado con
        HPSS + pasa-bajas (en memoria)», «el archivo original (sin separar)».
        """
        method = getattr(analysis, "separation_method", None)
        source = Path(analysis.source_path)
        if method == "demucs" or (method is None and source != Path(analysis.path)):
            return f"{source.name} (stem de bajo de Demucs)"
        if method == "hpss":
            return "el archivo original, separado con HPSS + pasa-bajas (en memoria)"
        return "el archivo original (sin separar)"

    @staticmethod
    def _segments_summary(analysis: AnalysisResult) -> str:
        """Texto «N conservados · M descartados (a por silencio, b por cortos)».

        Examples
        --------
        «16 conservados · 0 descartados», «21 conservados · 1 descartado (1 por corto)».
        """
        kept = sum(1 for s in analysis.segments if s.kept)
        silent = sum(1 for s in analysis.segments if not s.kept and s.reason == "silencio")
        short = sum(1 for s in analysis.segments if not s.kept and s.reason == "corto")
        discarded = len(analysis.segments) - kept
        text = f"{_plural(kept, 'conservado')} · {_plural(discarded, 'descartado')}"
        if discarded:
            parts = []
            if silent:
                parts.append(f"{silent} por silencio")
            if short:
                parts.append(f"{short} por {'corto' if short == 1 else 'cortos'}")
            other = discarded - silent - short
            if other:
                parts.append(f"{other} por otros motivos")
            text += f" ({', '.join(parts)})"
        return text

    def _gt_summary(self, analysis: AnalysisResult) -> str:
        """Texto del ground truth: número de notas y cuántas se emparejaron."""
        gt = analysis.ground_truth
        if not gt:
            return "no disponible (no hay .gt.json junto al audio)"
        tolerance_ms = 1000.0 * float(self.app.state.config.experiment.onset_tolerance_s)
        return (f"{_plural(len(gt), 'nota')} · {_plural(len(self._gt_by_position), 'emparejada')} "
                f"(±{tolerance_ms:.0f} ms)")

    @staticmethod
    def _spectrum_summary(analysis: AnalysisResult) -> str:
        """Tipo y resolución del espectro de la recompensa."""
        spec = analysis.spectrum
        env = analysis.config.env
        hop_ms = 1000.0 * spec.hop_length / float(spec.sr)
        if spec.kind == "stft":
            df = float(spec.sr) / float(env.n_fft)
            head = f"STFT · n_fft {env.n_fft} (Δf ≈ {df:.2f} Hz)"
        elif spec.kind == "cqt":
            head = f"CQT · {env.bins_per_octave} bins/octava desde {env.cqt_fmin_hz:.1f} Hz"
        else:
            head = spec.kind.upper()
        return f"{head}\nsalto entre frames {hop_ms:.1f} ms ({spec.hop_length} muestras)"

    def _update_banner(self) -> None:
        """Muestra el aviso si cambió algún parámetro que exige repetir el análisis."""
        stale = False
        if self.analysis is not None:
            current = self.app.state.config
            used = self.analysis.config
            stale = any(getattr(current, s) != getattr(used, s) for s in ANALYSIS_SECTIONS)
        if stale:
            self.banner.grid()
        else:
            self.banner.grid_remove()

    @property
    def redraw_pending(self) -> bool:
        """True si hay un redibujo de la figura programado (o pendiente de mostrar la pestaña)."""
        return self._redraw_job is not None or self._dirty

    @property
    def banner_visible(self) -> bool:
        """True si el aviso «Analizar de nuevo» está visible."""
        return bool(self.banner.winfo_manager())

    def _update_controls(self) -> None:
        """Habilita/deshabilita botones según haya análisis, selección, tarea en curso y reproductor."""
        busy = self._busy
        has_audio = self.analysis is not None
        has_segment = self._selected_segment() is not None

        def enable(widget: ttk.Widget, on: bool) -> None:
            """Habilita (``on``) o deshabilita un control ttk."""
            widget.state(["!disabled"] if on else ["disabled"])

        enable(self.btn_open, not busy)
        enable(self.btn_dataset, not busy)
        enable(self.btn_reanalyze, has_audio and not busy)
        enable(self.chk_separate, not busy)
        playable = self.player.available
        enable(self.btn_play_all, playable and has_audio)
        enable(self.btn_play_segment, playable and has_segment)
        enable(self.btn_stop, playable)
        enable(self.btn_zoom_note, has_segment)

    # =================================================================== figura
    def _request_redraw(self, keep_view: bool = True, trailing: bool = False) -> None:
        """Programa un redibujo de la figura (se agrupan peticiones seguidas).

        Parameters
        ----------
        keep_view : bool
            Si es True se conserva el zoom actual (misma pista).
        trailing : bool
            Si es True (cambios de selección) cada petición nueva POSPONE el
            redibujo: al recorrer la tabla con ↓ mantenida solo se redibuja al
            soltar. La espera crece con lo que tardó el último dibujo (pistas
            largas: hasta :data:`SELECTION_REDRAW_MAX_MS`).
        """
        # Mientras haya un redibujo pendiente solo se conserva el zoom si TODAS las
        # peticiones lo permiten (p. ej. un análisis nuevo con la pestaña oculta lo anula).
        self._keep_view = (self._keep_view and keep_view) if self._dirty else keep_view
        self._dirty = True
        delay = REDRAW_DELAY_MS
        if trailing:
            delay = int(min(SELECTION_REDRAW_MAX_MS, max(REDRAW_DELAY_MS, self._last_draw_ms)))
            if self._redraw_job is not None:
                self.after_cancel(self._redraw_job)
                self._redraw_job = None
        if self._redraw_job is None:
            self._redraw_job = self.after(delay, self._redraw)

    def _redraw(self) -> None:
        """Redibuja la figura si la pestaña está visible (si no, queda pendiente)."""
        self._redraw_job = None
        if not self.winfo_ismapped():
            return  # _dirty sigue activo: se redibuja al mostrar la pestaña (<Map>)
        self._dirty = False
        if self.analysis is None:
            self._draw_empty_state()
            return
        self.draw_analysis(keep_view=self._keep_view)

    def draw_analysis(self, keep_view: bool = True) -> None:
        """Dibuja :func:`src.plots.plot_audio_overview` con el segmento seleccionado resaltado.

        Parameters
        ----------
        keep_view : bool
            Si es True y el usuario había hecho zoom, se restaura ese zoom
            después de redibujar (la figura se reconstruye en cada selección).
        """
        analysis = self.analysis
        if analysis is None:
            self._draw_empty_state()
            return
        saved = self._current_view() if keep_view and self.plot_state == "analysis" else None
        if saved is not None and self._home_view is not None and self._same_view(saved, self._home_view):
            saved = None  # el usuario no había hecho zoom: no hay nada que restaurar
        selected = self._valid_position(self.app.state.selected_position)
        # En lienzos bajos (ventanas de 1024×700) la figura se dibuja compacta: sin subtítulo
        # (sus datos están en el panel de información) y con etiquetas cortas.
        self.compact = self._canvas_height() < COMPACT_HEIGHT_PX
        started = time.perf_counter()
        self.plot.show(plots.plot_audio_overview, analysis, selected=selected, compact=self.compact)
        self._last_draw_ms = 1000.0 * (time.perf_counter() - started)
        self.plot_state = "analysis"
        self.drawn_selection = selected
        self._home_view = self._current_view()
        toolbar = self.plot.toolbar
        if toolbar is not None:
            toolbar.update()        # olvida las vistas de los ejes anteriores
            toolbar.push_current()  # «Inicio» = pista completa
        if saved is not None:
            self._apply_view(saved)
            if toolbar is not None:
                toolbar.push_current()
        if self._zoom_selection_pending:
            self._zoom_selection_pending = False
            if selected is not None and not self._segment_in_view(selected):
                self._zoom_to(selected)
        self.plot.draw()

    def _segment_window(self, position: int) -> tuple[float, float] | None:
        """Intervalo (s) ``[inicio − margen, fin + margen]`` del segmento ``position``, dentro de la pista."""
        analysis = self.analysis
        if analysis is None or not 0 <= position < len(analysis.kept):
            return None
        segment = analysis.kept[position]
        return (max(0.0, segment.start_s - ZOOM_MARGIN_S), min(analysis.duration_s, segment.end_s + ZOOM_MARGIN_S))

    def _segment_in_view(self, position: int) -> bool:
        """True si la vista actual ya muestra el segmento con detalle (zoom de ≤ 3 veces su ventana)."""
        window, view = self._segment_window(position), self._current_view()
        if window is None or view is None:
            return False
        (left, right), (lo, hi) = view[0], window
        return left <= lo and hi <= right and (right - left) <= 3.0 * (hi - lo)

    def _zoom_to(self, position: int) -> None:
        """Acerca el eje de tiempo al segmento ``position`` (la barra de matplotlib lo guarda: «atrás» vuelve)."""
        window = self._segment_window(position)
        axes = self._data_axes()
        if window is None or not axes:
            return
        axes[0].set_xlim(*window)  # eje x compartido: también el espectrograma
        if self.plot.toolbar is not None:
            self.plot.toolbar.push_current()

    def zoom_to_selection(self) -> None:
        """Botón «Acercar a la nota»: acerca la figura al segmento seleccionado (±:data:`ZOOM_MARGIN_S` s)."""
        position = self._valid_position(self.app.state.selected_position)
        if position is None or self.plot_state != "analysis":
            self.app.status.set_message("Selecciona un segmento (clic en la figura o en la tabla) para acercarte a él.")
            return
        self._zoom_to(position)
        self.plot.draw()

    def _canvas_height(self) -> int:
        """Altura actual del lienzo de matplotlib en píxeles."""
        return int(self.plot.canvas.get_tk_widget().winfo_height())

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Al cambiar el tamaño del lienzo redibuja lo que dependa de él.

        El mensaje guía se recoloca siempre (es barato); la figura del análisis
        solo si hay que entrar o salir del modo compacto.
        """
        if event.width <= 1 or event.height <= 1:
            return
        if self.plot_state == "empty":
            if (event.width, event.height) != self._empty_size:
                self._request_redraw(keep_view=False)
        elif (event.height < COMPACT_HEIGHT_PX) != self.compact:
            self._request_redraw(keep_view=True)

    def _draw_empty_state(self) -> None:
        """Mensaje guía grande en la figura antes de analizar ningún audio.

        Muestra qué hacer (abrir un MP3 o generar el dataset), las seis etapas
        del análisis como diagrama y qué se verá después. Las posiciones se
        calculan en píxeles a partir del tamaño real del lienzo, de modo que
        nada se solape en ventanas pequeñas (se reduce la letra y, si no cabe,
        se omiten el diagrama o el pie).
        """
        self.plot_state = "empty"
        self.drawn_selection = None
        self._home_view = None
        fig = self.plot.figure
        fig.clear()
        fig.set_layout_engine("none")
        fig.set_facecolor(plots.SURFACE)
        width_px, height_px = (float(v) for v in fig.get_size_inches() * fig.dpi)
        narrow = width_px < 1150 or height_px < 380
        scale = 0.86 if narrow else 1.0
        px_per_pt = fig.dpi / 72.0

        if self._busy:
            body = "Procesando… sigue el avance en la barra de estado (abajo a la derecha)."
        else:
            body = ("Abre un MP3 (Archivo → Abrir, Ctrl+O o el botón «Abrir MP3…»)\n"
                    "o genera el dataset sintético con «Generar dataset sintético…».")
        footer = ("Aquí verás la forma de onda y el espectrograma con los onsets detectados, los segmentos "
                  "descartados (gris) y la f0 de pYIN.\nHaz clic en una nota de la figura o de la tabla para "
                  "seleccionarla en todas las pestañas.")
        if narrow:
            steps = ("1 · Carga", "2 · Separación\n(opcional)", "3 · Preproceso\n(pasa-bajas)",
                     "4 · Onsets y\nsegmentos", "5 · Pitch\n(pYIN)", "6 · Espectro\ny bandits")
        else:
            steps = ("1 · Carga\n(ffmpeg)", "2 · Separación\n(opcional)",
                     "3 · Preprocesado\n(pasa-bajas 400 Hz)", "4 · Onsets y\nsegmentos", "5 · Pitch\n(pYIN)",
                     "6 · Espectro y\nbandits")

        def lines_px(text: str, size: float, spacing: float) -> float:
            """Alto aproximado (px) de un texto de varias líneas."""
            return (text.count("\n") + 1) * size * spacing * px_per_pt

        title_size, body_size, step_size, foot_size = 19 * scale, 13 * scale, 10 * scale, 10.5 * scale
        blocks: list[tuple[str, float]] = [("title", lines_px("x", title_size, 1.3)),
                                           ("body", lines_px(body, body_size, 1.6))]
        boxes_h = 1.15 * (2 * step_size * 1.4 * px_per_pt + 2 * 0.7 * step_size * px_per_pt)
        foot_h = lines_px(footer, foot_size, 1.6)
        gap = 22 * scale
        total = sum(h for _k, h in blocks) + gap
        if total + boxes_h + gap + 24 <= height_px:
            blocks.append(("steps", boxes_h))
            total += boxes_h + gap
        if total + foot_h + 24 <= height_px:
            blocks.append(("footer", foot_h))
            total += foot_h + gap
        top = max((height_px - (total - gap)) / 2.0, 6.0)

        def y_of(center_px: float) -> float:
            """Píxeles desde arriba → fracción de la figura."""
            return 1.0 - center_px / height_px

        for kind, block_h in blocks:
            center = y_of(top + block_h / 2)
            if kind == "title":
                fig.text(0.5, center, "Todavía no hay ningún audio analizado", ha="center", va="center",
                         fontsize=title_size, fontweight="bold", color=plots.TEXT_PRIMARY)
            elif kind == "body":
                fig.text(0.5, center, body, ha="center", va="center", fontsize=body_size,
                         color=plots.TEXT_SECONDARY, linespacing=1.6, multialignment="center")
            elif kind == "steps":
                xs = np.linspace(0.08, 0.92, len(steps))
                for i, (x, step) in enumerate(zip(xs, steps, strict=True)):
                    fig.text(x, center, step, ha="center", va="center", fontsize=step_size,
                             color=plots.TEXT_PRIMARY, linespacing=1.4, multialignment="center",
                             bbox={"boxstyle": "round,pad=0.7", "facecolor": plots.HIGHLIGHT,
                                   "edgecolor": plots.AXIS, "linewidth": 0.9})
                    if i < len(steps) - 1:
                        fig.text((x + xs[i + 1]) / 2, center, "→", ha="center", va="center",
                                 fontsize=14 * scale, color=plots.TEXT_MUTED)
            else:
                fig.text(0.5, center, footer, ha="center", va="center", fontsize=foot_size,
                         color=plots.TEXT_SECONDARY, linespacing=1.6, multialignment="center", wrap=True)
            top += block_h + gap
        widget = self.plot.canvas.get_tk_widget()
        self._empty_size = (int(widget.winfo_width()), int(widget.winfo_height()))
        if self.plot.toolbar is not None:
            self.plot.toolbar.update()
        self.plot.draw()

    def _data_axes(self) -> list[Any]:
        """Ejes de datos de la figura (onda y espectrograma; sin la barra de color)."""
        return [ax for ax in self.plot.figure.axes if getattr(ax, "_colorbar", None) is None and ax.axison]

    def _current_view(self) -> tuple[tuple[float, float], ...] | None:
        """Límites actuales ``(xlim, ylim de la onda, ylim del espectrograma)`` o None."""
        axes = self._data_axes()
        if len(axes) < 2:
            return None
        wave, spec = axes[0], axes[1]
        return (tuple(wave.get_xlim()), tuple(wave.get_ylim()), tuple(spec.get_ylim()))

    def _apply_view(self, view: tuple[tuple[float, float], ...]) -> None:
        """Restaura unos límites guardados con :meth:`_current_view`."""
        axes = self._data_axes()
        if len(axes) < 2:
            return
        wave, spec = axes[0], axes[1]
        wave.set_xlim(*view[0])  # el eje x es compartido: también mueve el espectrograma
        wave.set_ylim(*view[1])
        spec.set_ylim(*view[2])

    @staticmethod
    def _same_view(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> bool:
        """True si dos vistas coinciden (tolerancia relativa pequeña)."""
        return all(np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rtol=1e-6, atol=1e-9)
                   for x, y in zip(a, b, strict=True))

    def _toolbar_active(self) -> bool:
        """True si la barra de matplotlib está en modo zoom o desplazamiento."""
        toolbar = self.plot.toolbar
        if toolbar is None:
            return False
        mode = getattr(toolbar, "mode", "")
        return bool(str(getattr(mode, "value", mode)))

    # ================================================================ utilidades
    def position_at(self, t: float) -> int | None:
        """Posición del segmento CONSERVADO que contiene el instante ``t``.

        Parameters
        ----------
        t : float
            Instante en segundos.

        Returns
        -------
        int | None
            Posición entre los conservados (``inicio ≤ t < fin``) o None si
            ``t`` cae en un silencio, en un segmento descartado o fuera de la pista.
        """
        if self._kept_starts.size == 0:
            return None
        # Último segmento que empieza en o antes de t (los conservados están ordenados).
        i = int(np.searchsorted(self._kept_starts, t, side="right")) - 1
        if 0 <= i < self._kept_ends.size and t < self._kept_ends[i]:
            return i
        return None

    def _valid_position(self, position: int | None) -> int | None:
        """``position`` si es una posición válida del análisis actual; si no, None."""
        if position is None or self.analysis is None:
            return None
        return position if 0 <= position < len(self._kept_starts) else None

    def _selected_segment(self) -> Segment | None:
        """Segmento conservado seleccionado (None si no hay)."""
        position = self._valid_position(self.app.state.selected_position)
        if position is None or self.analysis is None:
            return None
        return self.analysis.kept[position]

    def destroy(self) -> None:
        """Cancela el redibujo pendiente y la reproducción antes de destruir la pestaña."""
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except tk.TclError:
                pass
            self._redraw_job = None
        try:
            self.player.stop()
        except Exception:  # noqa: BLE001 - al cerrar no importa un fallo de audio
            pass
        super().destroy()
