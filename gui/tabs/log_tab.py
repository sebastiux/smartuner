"""Pestaña 6 · Log: el registro de actividad del sistema, en vivo.

Papel en la GUI
---------------
Los módulos de ``src/`` narran lo que hacen con :mod:`logging` (en español):
cada etapa del pipeline («Etapa 4/6 — Segmentación…»), cuántos segmentos se
conservaron, qué brazos tiene cada entorno bandit, el avance del experimento,
los avisos y los errores. En la CLI (``main.py --cli``) esos mensajes salen
por la consola; en la GUI llegan aquí. Así la pestaña sirve para explicar en
clase *qué está haciendo el sistema y en qué orden*, y para diagnosticar un
análisis que salió mal sin abrir una terminal.

Flujo de datos
--------------
::

    módulos src.* / gui.*        hilo trabajador o de la GUI
        │ logger.info(...)
        ▼
    QueueLogHandler (app.log_handler) ──put──▶ queue.Queue
                                                  │ get_nowait (cada 100 ms, ≤ 500 por ciclo)
                                                  ▼
    LogTab: LogEntry (texto ya formateado) ──▶ búfer interno (≤ 20 000 registros)
                                                  │ filtros: nivel mínimo + texto
                                                  ▼
                                   tk.Text de solo lectura (≤ 5 000 líneas)

Los registros pueden llegar desde cualquier hilo (el handler solo los encola);
la pestaña los saca de la cola con ``after`` en el hilo de la GUI, que es el
único que toca widgets. Como se guardan en un búfer, cambiar el filtro o el
nivel mínimo vuelve a dibujar la vista sin perder mensajes; el búfer y la
vista tienen un límite para que una sesión larga no agote la memoria.

Formato de cada línea::

    HH:MM:SS  NIVEL    módulo: mensaje
    14:03:21  INFO     pipeline: Etapa 4/6 — Segmentación: detectando onsets…

El módulo se muestra sin el prefijo ``src.``. Colores por nivel: DEBUG gris,
INFO tinta normal, WARNING ámbar oscuro, ERROR rojo; los mensajes que empiezan
por «Etapa» (encabezados del pipeline) van en negrita.

Controles
---------
* **Nivel mínimo** (DEBUG/INFO/WARNING/ERROR): filtra la vista y ajusta el
  nivel de ``app.log_handler``. Con DEBUG, además, baja el umbral de los
  loggers ``src`` y ``gui`` para que sus detalles lleguen al handler (la
  consola conserva su nivel: ver :meth:`LogTab._apply_logger_threshold`).
* **Filtrar**: muestra solo las líneas que contienen el texto (sin distinguir
  mayúsculas) y resalta las coincidencias.
* **Auto-scroll**, **Copiar al portapapeles**, **Guardar…** (TXT UTF-8) y
  **Limpiar**; un contador indica cuántos mensajes hay y cuántos son avisos o errores.

Eventos
-------
Solo escucha ``busy_changed``: al empezar o terminar una tarea vacía la cola en
el acto, para que el log quede sincronizado con la barra de estado. Los demás
eventos (análisis, transcripciones, experimento, segmento o algoritmo
seleccionado, configuración) no cambian lo que muestra esta pestaña: lo que
hacen los módulos ya llega como mensajes de log.
"""

from __future__ import annotations

import logging
import queue
import re
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from gui.widgets import UI_ERROR, UI_SURFACE, UI_TEXT, UI_TEXT_SECONDARY, Tooltip, show_error
from src.plots import GRID, TEXT_MUTED

logger = logging.getLogger(__name__)

#: Nombre de esta pestaña como origen de eventos (``source=``).
TAB_SOURCE = "log"

#: Cada cuánto se vacía la cola del handler (ms).
POLL_MS = 100
#: Máximo de registros que se procesan por ciclo de sondeo (para no congelar la GUI
#: si un módulo escribe miles de mensajes de golpe; el resto espera al siguiente ciclo).
MAX_RECORDS_PER_TICK = 500
#: Máximo de registros que se conservan en el búfer interno (los más antiguos se descartan).
MAX_BUFFER_RECORDS = 20_000
#: Máximo de líneas en el widget de texto (se muestran las más recientes).
MAX_VISIBLE_LINES = 5_000
#: Espera tras teclear en el filtro antes de volver a dibujar (ms).
FILTER_DEBOUNCE_MS = 150

#: Niveles que ofrece el selector de nivel mínimo, de más a menos detallado.
LEVEL_CHOICES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")

#: Loggers del proyecto cuyo umbral se ajusta para que el nivel elegido llegue al handler.
PROJECT_LOGGERS: tuple[str, ...] = ("src", "gui")

#: Ancho (caracteres) de ``"HH:MM:SS  NIVEL    "``: sangría de las líneas de continuación.
PREFIX_CHARS = 19

#: Colores del texto por nivel (sobre el fondo claro de la vista).
LEVEL_COLORS: dict[str, str] = {
    "debug": TEXT_MUTED,     # gris: detalles internos
    "info": UI_TEXT,         # tinta normal
    "warning": "#8f5b00",    # ámbar oscuro (contraste ≥ 4.5:1 sobre el fondo)
    "error": UI_ERROR,       # rojo
    "critical": UI_ERROR,    # rojo, en negrita y con fondo
}
#: Fondo de las coincidencias del filtro.
MATCH_BACKGROUND = "#ffe58a"
#: Fondo de los mensajes CRITICAL.
CRITICAL_BACKGROUND = "#fdecea"

EMPTY_MESSAGE = (
    "Aún no hay mensajes.\n\n"
    "Abre un MP3 (Archivo → Abrir) o genera el dataset sintético:\n"
    "aquí verás, en vivo, cada etapa del análisis y del experimento."
)


# ---------------------------------------------------------------------------
# Formato de los registros (funciones puras, sin widgets)
# ---------------------------------------------------------------------------


def module_label(logger_name: str) -> str:
    """Nombre corto del módulo que emitió un registro (sin el prefijo ``src.``).

    Parameters
    ----------
    logger_name : str
        ``record.name`` (p. ej. ``"src.pipeline"``).

    Returns
    -------
    str
        El nombre sin ``"src."``; los demás loggers se dejan igual.

    Examples
    --------
    >>> module_label("src.pipeline"), module_label("gui.app"), module_label("root")
    ('pipeline', 'gui.app', 'root')
    """
    return logger_name[4:] if logger_name.startswith("src.") else logger_name


def level_tag(levelno: int) -> str:
    """Nombre del tag de color de un nivel numérico de :mod:`logging`.

    Parameters
    ----------
    levelno : int
        Nivel del registro (``logging.DEBUG`` = 10 … ``logging.CRITICAL`` = 50).
        Los niveles intermedios o personalizados se asignan al nivel estándar
        inmediatamente inferior.

    Returns
    -------
    str
        ``"debug"``, ``"info"``, ``"warning"``, ``"error"`` o ``"critical"``.

    Examples
    --------
    >>> [level_tag(n) for n in (5, 10, 20, 25, 30, 40, 50)]
    ['debug', 'debug', 'info', 'info', 'warning', 'error', 'critical']
    """
    if levelno >= logging.CRITICAL:
        return "critical"
    if levelno >= logging.ERROR:
        return "error"
    if levelno >= logging.WARNING:
        return "warning"
    if levelno >= logging.INFO:
        return "info"
    return "debug"


def plural(n: int, singular: str, plural_form: str) -> str:
    """``"1 mensaje"`` / ``"3 mensajes"``: número con la forma gramatical correcta.

    Parameters
    ----------
    n : int
        Cantidad.
    singular, plural_form : str
        Formas del sustantivo.

    Returns
    -------
    str
        Texto con separador de miles (espacio fino) para cantidades grandes.

    Examples
    --------
    >>> plural(1, "aviso", "avisos"), plural(0, "error", "errores"), plural(20000, "mensaje", "mensajes")
    ('1 aviso', '0 errores', '20 000 mensajes')
    """
    number = f"{n:,}".replace(",", " ")
    return f"{number} {singular if n == 1 else plural_form}"


@dataclass(frozen=True)
class LogEntry:
    """Un registro de :mod:`logging` ya formateado para la vista.

    Attributes
    ----------
    levelno : int
        Nivel numérico (10 = DEBUG, 20 = INFO, 30 = WARNING, 40 = ERROR, 50 = CRITICAL).
    timestamp : str
        Hora local del registro, ``"HH:MM:SS"``.
    body : str
        Resto de la entrada: ``"NIVEL    módulo: mensaje"``; si el mensaje
        ocupa varias líneas (p. ej. un traceback) las de continuación van
        sangradas :data:`PREFIX_CHARS` espacios. Sin salto de línea final.
    is_stage : bool
        True si el mensaje empieza por «Etapa» (encabezado del pipeline).
    """

    levelno: int
    timestamp: str
    body: str
    is_stage: bool = False

    @property
    def text(self) -> str:
        """Entrada completa tal como se muestra/guarda (sin salto de línea final)."""
        return f"{self.timestamp}  {self.body}"

    @property
    def n_lines(self) -> int:
        """Número de líneas de texto que ocupa la entrada en la vista."""
        return self.body.count("\n") + 1

    @property
    def tag(self) -> str:
        """Tag de color de su nivel (ver :func:`level_tag`)."""
        return level_tag(self.levelno)


def format_record(record: logging.LogRecord) -> LogEntry:
    """Convierte un :class:`logging.LogRecord` en una :class:`LogEntry`.

    Parameters
    ----------
    record : logging.LogRecord
        Registro recibido del :class:`gui.workers.QueueLogHandler`.

    Returns
    -------
    LogEntry
        Entrada con el formato ``"HH:MM:SS  NIVEL    módulo: mensaje"``. Si el
        registro trae una excepción (``logger.exception``) o una pila
        (``stack_info=True``), se añaden como líneas de continuación sangradas.

    Notes
    -----
    Si los argumentos del mensaje no encajan con su plantilla
    (``logger.info("%d", "x")``), se muestra la plantilla y los argumentos en
    vez de lanzar una excepción: un mensaje mal escrito no debe romper el log.

    Examples
    --------
    >>> rec = logging.LogRecord("src.pipeline", logging.INFO, __file__, 1,
    ...                         "Etapa %d/6 — Carga", (1,), None)
    >>> entry = format_record(rec)
    >>> entry.body, entry.is_stage
    ('INFO     pipeline: Etapa 1/6 — Carga', True)
    """
    try:
        message = record.getMessage()
    except Exception:  # noqa: BLE001 - plantilla y argumentos incompatibles
        message = f"{record.msg} {record.args!r}"
    message = str(message).rstrip()
    extra: list[str] = []
    if record.exc_info:
        extra.append(logging.Formatter().formatException(record.exc_info))
    elif record.exc_text:
        extra.append(record.exc_text)
    if record.stack_info:
        extra.append(str(record.stack_info))
    if extra:
        message = message + "\n" + "\n".join(part.rstrip() for part in extra)
    indent = " " * PREFIX_CHARS
    message = message.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\n" + indent)
    timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
    body = f"{record.levelname:<8} {module_label(record.name)}: {message}"
    first_line = message.split("\n", 1)[0]
    return LogEntry(levelno=record.levelno, timestamp=timestamp, body=body, is_stage=first_line.lstrip().startswith("Etapa"))


# ---------------------------------------------------------------------------
# Pestaña
# ---------------------------------------------------------------------------


class LogTab(ttk.Frame):
    """Pestaña «6 · Log»: muestra en vivo los mensajes de :mod:`logging`.

    Parameters
    ----------
    master : tk.Misc
        El ``ttk.Notebook`` de la ventana principal.
    app : SmartunerApp
        Aplicación; se usan ``app.log_handler`` (cola de registros),
        ``app.state.events`` y ``app.status``.

    Attributes
    ----------
    max_records : int
        Tamaño máximo del búfer de registros (:data:`MAX_BUFFER_RECORDS`).
    max_visible_lines : int
        Líneas máximas en la vista (:data:`MAX_VISIBLE_LINES`).
    max_per_tick : int
        Registros procesados por ciclo de sondeo (:data:`MAX_RECORDS_PER_TICK`).
    level_var, filter_var : tk.StringVar
        Nivel mínimo elegido y texto del filtro.
    autoscroll_var : tk.BooleanVar
        Si es True la vista sigue al último mensaje.
    text : tk.Text
        Vista de solo lectura.

    Notes
    -----
    Los límites son atributos de instancia para que las pruebas puedan
    reducirlos; cambiarlos no re-dibuja la vista por sí solo.
    """

    def __init__(self, master: Any, app: Any) -> None:
        super().__init__(master, padding=(12, 10, 12, 8))
        self.app = app
        self.max_records = MAX_BUFFER_RECORDS
        self.max_visible_lines = MAX_VISIBLE_LINES
        self.max_per_tick = MAX_RECORDS_PER_TICK

        # Búfer de TODOS los registros recibidos (pasen o no los filtros) y
        # contadores incrementales para no recorrerlo en cada ciclo.
        self._buffer: deque[LogEntry] = deque()
        self._n_matching = 0          # entradas del búfer que pasan los filtros
        self._n_warnings = 0          # WARNING en el búfer
        self._n_errors = 0            # ERROR o CRITICAL en el búfer
        # Entradas visibles en el widget (líneas de cada una), para recortar por arriba.
        self._visible: deque[int] = deque()
        self._visible_lines = 0

        self._min_level = logging.INFO
        self._filter_re: re.Pattern[str] | None = None
        self._poll_id: str | None = None
        self._filter_after_id: str | None = None
        # Umbrales originales de loggers/handlers que la pestaña modificó (para restaurarlos).
        self._saved_logger_levels: dict[str, int] = {}
        self._saved_handler_levels: dict[logging.Handler, int] = {}

        handler = getattr(app, "log_handler", None)
        initial = handler.level if handler is not None and handler.level else logging.INFO
        initial_name = logging.getLevelName(initial)
        self.level_var = tk.StringVar(value=initial_name if initial_name in LEVEL_CHOICES else "INFO")
        self.filter_var = tk.StringVar(value="")
        self.autoscroll_var = tk.BooleanVar(value=True)

        self._build_fonts()
        self._build_header()
        self._build_toolbar()
        self._build_view()
        self._build_footer()
        # Filas: 0 título · 1 explicación · 2 controles · 3 vista (se estira) · 4 contador y leyenda.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self._set_min_level(logging.getLevelName(self.level_var.get()))
        self.filter_var.trace_add("write", self._on_filter_typed)
        app.state.events.subscribe("busy_changed", self._on_busy_changed)
        self.bind("<Destroy>", self._on_destroy, add="+")

        self._update_counter()
        self._update_placeholder()
        # Lee lo que ya estaba en la cola (mensajes emitidos antes de crear la pestaña)
        # y empieza el sondeo periódico.
        self._drain()
        self._poll_id = self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ construcción
    def _build_fonts(self) -> None:
        """Crea la fuente monoespaciada de la vista (normal y negrita)."""
        family = tkfont.nametofont("TkFixedFont").actual("family")
        self.font = tkfont.Font(self, family=family, size=10)
        self.font_bold = tkfont.Font(self, family=family, size=10, weight="bold")

    def _build_header(self) -> None:
        """Título y explicación breve de la pestaña."""
        ttk.Label(self, text="Registro de actividad", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.description = ttk.Label(
            self, style="Muted.TLabel", justify="left",
            text=(
                "Mensajes que escribe el sistema mientras trabaja: cada etapa del pipeline (en negrita), "
                "los segmentos y entornos bandit construidos, el avance de los experimentos, los avisos y "
                "los errores. Son los mismos que muestra la consola en modo CLI (main.py --cli)."
            ),
        )
        self.description.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.description.bind("<Configure>", lambda e: self.description.configure(wraplength=max(200, e.width - 8)))

    def _build_toolbar(self) -> None:
        """Fila de controles: nivel, filtro, auto-scroll y botones."""
        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._toolbar = bar

        ttk.Label(bar, text="Nivel mínimo:").grid(row=0, column=0, sticky="w")
        self.level_combo = ttk.Combobox(bar, textvariable=self.level_var, values=list(LEVEL_CHOICES), state="readonly", width=9)
        self.level_combo.grid(row=0, column=1, sticky="w", padx=(4, 14))
        self.level_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_level_selected())
        Tooltip(
            self.level_combo,
            "Muestra solo los mensajes de este nivel o más graves:\n"
            "• DEBUG: detalles internos (frames usados por la recompensa, agentes creados, oráculo…).\n"
            "• INFO: lo que hace el sistema paso a paso (etapas, segmentos, experimentos).\n"
            "• WARNING: avisos; algo inesperado, pero el sistema continúa.\n"
            "• ERROR: fallos.\n"
            "Los mensajes DEBUG solo se capturan desde que eliges DEBUG.",
        )

        ttk.Label(bar, text="Filtrar:").grid(row=0, column=2, sticky="w")
        self.filter_entry = ttk.Entry(bar, textvariable=self.filter_var, width=30)
        self.filter_entry.grid(row=0, column=3, sticky="ew", padx=(4, 2))
        self.filter_entry.bind("<Escape>", lambda _e: self.set_filter(""))
        self.filter_entry.bind("<Return>", lambda _e: self._apply_filter_now())
        Tooltip(
            self.filter_entry,
            "Muestra solo las líneas que contienen este texto (sin distinguir mayúsculas) y resalta "
            "las coincidencias. Ejemplos: «Etapa», «Segmento 12», «ucb1», «pipeline».\n"
            "Esc borra el filtro; Ctrl+F (desde la vista) salta aquí.",
        )
        self.clear_filter_button = ttk.Button(bar, text="✕", width=2, command=lambda: self.set_filter(""))
        self.clear_filter_button.grid(row=0, column=4, sticky="w", padx=(0, 14))
        Tooltip(self.clear_filter_button, "Borra el filtro de texto.")

        self.autoscroll_check = ttk.Checkbutton(bar, text="Auto-scroll", variable=self.autoscroll_var, command=self._on_autoscroll_toggled)
        self.autoscroll_check.grid(row=0, column=5, sticky="e", padx=(0, 10))
        Tooltip(self.autoscroll_check, "Si está marcado, la vista baja sola para mostrar cada mensaje nuevo. "
                                       "Desmárcalo para leer con calma mensajes anteriores mientras el sistema trabaja.")
        self.copy_button = ttk.Button(bar, text="Copiar al portapapeles", command=self.copy_to_clipboard)
        self.copy_button.grid(row=0, column=6, sticky="e", padx=(0, 6))
        Tooltip(self.copy_button, "Copia todas las líneas que pasan los filtros actuales (nivel y texto). "
                                  "Para copiar solo una parte, selecciónala con el ratón y pulsa Ctrl+C.")
        self.save_button = ttk.Button(bar, text="Guardar…", command=self._on_save)
        self.save_button.grid(row=0, column=7, sticky="e", padx=(0, 6))
        Tooltip(self.save_button, "Guarda en un archivo de texto (UTF-8) todas las líneas que pasan los filtros "
                                  "actuales, incluidas las que ya no caben en la vista.")
        self.clear_button = ttk.Button(bar, text="Limpiar", command=self.clear)
        self.clear_button.grid(row=0, column=8, sticky="e")
        Tooltip(self.clear_button, "Borra los mensajes de esta vista. No afecta al análisis, a los "
                                   "resultados ni a ningún archivo.")
        # La columna del filtro absorbe el espacio libre (y es la primera en encogerse).
        bar.columnconfigure(3, weight=1)

    def _build_view(self) -> None:
        """Widget de texto de solo lectura con barra de desplazamiento y mensaje de estado vacío."""
        frame = tk.Frame(self, background=UI_SURFACE, highlightthickness=1, highlightbackground=GRID, highlightcolor=GRID)
        frame.grid(row=3, column=0, sticky="nsew")
        self._view_frame = frame
        self.text = tk.Text(
            frame, wrap="word", font=self.font, background=UI_SURFACE, foreground=UI_TEXT,
            borderwidth=0, highlightthickness=0, padx=10, pady=8, undo=False,
            selectbackground="#cfe0f5", selectforeground=UI_TEXT, inactiveselectbackground="#dfe9f6",
            spacing1=1, spacing3=1,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        # Sangría de las líneas largas que se parten: quedan alineadas bajo el módulo.
        indent_px = self.font.measure(" " * PREFIX_CHARS)
        self.text.tag_configure("line", lmargin2=indent_px)
        self.text.tag_configure("time", foreground=TEXT_MUTED)
        for name in ("debug", "info", "warning", "error"):
            self.text.tag_configure(name, foreground=LEVEL_COLORS[name])
        self.text.tag_configure("critical", foreground=LEVEL_COLORS["critical"], background=CRITICAL_BACKGROUND, font=self.font_bold)
        self.text.tag_configure("stage", font=self.font_bold, spacing1=6)
        self.text.tag_configure("match", background=MATCH_BACKGROUND)
        self.text.tag_raise("match")
        self.text.tag_raise("sel")
        self.text.configure(state="disabled")
        self.text.bind("<Control-f>", lambda _e: (self.filter_entry.focus_set(), "break")[1])
        # Un clic en la vista le da el foco para que funcionen Ctrl+C y Ctrl+F.
        self.text.bind("<Button-1>", lambda _e: self.text.focus_set(), add="+")

        self.placeholder = tk.Label(
            frame, text=EMPTY_MESSAGE, justify="center", background=UI_SURFACE,
            foreground=UI_TEXT_SECONDARY, font=("TkDefaultFont", 11), wraplength=760,
        )
        # El mensaje guía se ajusta al ancho de la vista (sin pasar de ~760 px para que se lea bien).
        frame.bind("<Configure>", lambda e: self.placeholder.configure(wraplength=max(240, min(760, e.width - 80))), add="+")

    def _build_footer(self) -> None:
        """Contador de mensajes y leyenda de colores."""
        footer = ttk.Frame(self)
        footer.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.count_label = ttk.Label(footer, text="")
        self.count_label.pack(side="left")
        self.warnings_label = ttk.Label(footer, text="", foreground=LEVEL_COLORS["warning"])
        self.warnings_label.pack(side="left", padx=(12, 0))
        self.errors_label = ttk.Label(footer, text="", foreground=LEVEL_COLORS["error"])
        self.errors_label.pack(side="left", padx=(12, 0))
        Tooltip(self.count_label, "Mensajes recibidos que pasan los filtros, del total guardado. "
                                  f"Se guardan como mucho {plural(MAX_BUFFER_RECORDS, 'mensaje', 'mensajes')} y la vista "
                                  f"muestra las {plural(MAX_VISIBLE_LINES, 'línea', 'líneas')} más recientes.")

        legend = ttk.Frame(footer)
        legend.pack(side="right")
        ttk.Label(legend, text="Colores:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        for name in ("DEBUG", "INFO", "WARNING", "ERROR"):
            ttk.Label(legend, text=name, foreground=LEVEL_COLORS[name.lower()], font=self.font).pack(side="left", padx=(0, 8))
        stage = ttk.Label(legend, text="Etapa del pipeline", font=self.font_bold)
        stage.pack(side="left")
        Tooltip(stage, "Los mensajes que empiezan por «Etapa» marcan el inicio de cada etapa del "
                       "pipeline (carga, separación, preprocesamiento, segmentación, pitch, entornos bandit).")

    # ------------------------------------------------------------ API pública
    @property
    def buffer_size(self) -> int:
        """Número de registros guardados en el búfer (pasen o no los filtros)."""
        return len(self._buffer)

    @property
    def matching_count(self) -> int:
        """Número de registros del búfer que pasan el nivel mínimo y el filtro de texto."""
        return self._n_matching

    @property
    def visible_line_count(self) -> int:
        """Número de líneas de texto presentes en la vista."""
        return self._visible_lines

    def visible_text(self) -> str:
        """Texto que muestra la vista ahora mismo (sin el salto de línea final).

        Returns
        -------
        str
            Las líneas visibles separadas por ``"\\n"``.
        """
        return self.text.get("1.0", "end-1c").rstrip("\n")

    def matching_lines(self) -> list[str]:
        """Todas las entradas del búfer que pasan los filtros (sin el límite de la vista).

        Returns
        -------
        list[str]
            Texto de cada entrada (puede contener saltos de línea si trae traceback).
        """
        return [entry.text for entry in self._buffer if self._passes(entry)]

    def set_level(self, name: str) -> None:
        """Cambia el nivel mínimo como si el usuario lo eligiera en el desplegable.

        Parameters
        ----------
        name : str
            ``"DEBUG"``, ``"INFO"``, ``"WARNING"`` o ``"ERROR"``.

        Raises
        ------
        ValueError
            Si ``name`` no es uno de :data:`LEVEL_CHOICES`.
        """
        if name not in LEVEL_CHOICES:
            raise ValueError(f"Nivel desconocido: {name!r}; usa uno de {', '.join(LEVEL_CHOICES)}")
        self.level_var.set(name)
        self._on_level_selected()

    def set_filter(self, text: str) -> None:
        """Fija el texto del filtro y vuelve a dibujar la vista inmediatamente.

        Parameters
        ----------
        text : str
            Texto a buscar (vacío = sin filtro).
        """
        self.filter_var.set(text)
        self._apply_filter_now()

    def poll_now(self) -> int:
        """Procesa ya los registros pendientes de la cola (como un ciclo de sondeo).

        Returns
        -------
        int
            Registros procesados (como mucho :attr:`max_per_tick`).
        """
        return self._drain()

    def clear(self) -> None:
        """Vacía el búfer y la vista (botón «Limpiar»).

        Notes
        -----
        Solo afecta a esta pestaña: el análisis, los resultados y los archivos no cambian.
        Los registros que aún esperan en la cola se mostrarán en el siguiente ciclo.
        """
        self._buffer.clear()
        self._n_matching = self._n_warnings = self._n_errors = 0
        self._set_text_state("normal")
        self.text.delete("1.0", "end")
        self._set_text_state("disabled")
        self._visible.clear()
        self._visible_lines = 0
        self._update_counter()
        self._update_placeholder()

    def save_to(self, path: str | Path) -> int:
        """Guarda en ``path`` (texto UTF-8) las entradas que pasan los filtros.

        Parameters
        ----------
        path : str | Path
            Archivo de destino (se sobrescribe).

        Returns
        -------
        int
            Número de entradas guardadas.

        Raises
        ------
        OSError
            Si no se puede escribir el archivo.
        """
        lines = self.matching_lines()
        content = "\n".join(lines) + ("\n" if lines else "")
        Path(path).write_text(content, encoding="utf-8")
        return len(lines)

    def copy_to_clipboard(self) -> int:
        """Copia al portapapeles las entradas que pasan los filtros (botón «Copiar»).

        Returns
        -------
        int
            Número de entradas copiadas.
        """
        lines = self.matching_lines()
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self._status(f"Log copiado al portapapeles ({plural(len(lines), 'mensaje', 'mensajes')}).")
        return len(lines)

    # ------------------------------------------------------------ sondeo
    def _poll(self) -> None:
        """Ciclo periódico (cada :data:`POLL_MS` ms): vacía la cola y se reprograma."""
        self._poll_id = None
        try:
            self._drain()
        except Exception:  # noqa: BLE001 - un error aquí no debe detener el sondeo
            logger.exception("Error al mostrar mensajes del log")
        if self.winfo_exists():
            self._poll_id = self.after(POLL_MS, self._poll)

    def _drain(self) -> int:
        """Saca de la cola hasta :attr:`max_per_tick` registros y los muestra.

        Returns
        -------
        int
            Registros procesados en esta llamada.
        """
        handler = getattr(self.app, "log_handler", None)
        if handler is None:
            return 0
        new_entries: list[LogEntry] = []
        for _ in range(self.max_per_tick):
            try:
                record = handler.queue.get_nowait()
            except queue.Empty:
                break
            new_entries.append(format_record(record))
        if not new_entries:
            return 0
        shown = [entry for entry in new_entries if self._add_to_buffer(entry)]
        if len(shown) > self.max_visible_lines:  # ráfaga enorme: solo las más recientes
            shown = shown[-self.max_visible_lines:]
        if shown:
            self._append_to_view(shown)
        self._update_counter()
        self._update_placeholder()
        return len(new_entries)

    def _add_to_buffer(self, entry: LogEntry) -> bool:
        """Añade ``entry`` al búfer (descartando la más antigua si está lleno).

        Returns
        -------
        bool
            True si la entrada pasa los filtros actuales (hay que mostrarla).
        """
        while len(self._buffer) >= self.max_records > 0:
            old = self._buffer.popleft()
            self._count(old, -1)
        self._buffer.append(entry)
        self._count(entry, +1)
        return self._passes(entry)

    def _count(self, entry: LogEntry, delta: int) -> None:
        """Actualiza los contadores incrementales al añadir (+1) o descartar (−1) una entrada."""
        if entry.levelno >= logging.ERROR:
            self._n_errors += delta
        elif entry.levelno >= logging.WARNING:
            self._n_warnings += delta
        if self._passes(entry):
            self._n_matching += delta

    def _passes(self, entry: LogEntry) -> bool:
        """True si ``entry`` cumple el nivel mínimo y contiene el texto del filtro."""
        if entry.levelno < self._min_level:
            return False
        return self._filter_re is None or self._filter_re.search(entry.text) is not None

    # ------------------------------------------------------------ vista
    def _segments(self, entry: LogEntry) -> list[tuple[str, tuple[str, ...]]]:
        """Trozos de texto con sus tags para insertar ``entry`` (incluye el salto de línea final).

        La hora va en gris; el resto con el color de su nivel (y en negrita si
        es un encabezado de etapa); las coincidencias del filtro llevan además
        el tag ``match``.
        """
        full = entry.text + "\n"
        extra: tuple[str, ...] = ("stage",) if entry.is_stage else ()
        level = entry.tag
        body_tags = ("line", level, *extra)
        spans: list[tuple[int, int, tuple[str, ...]]] = [
            (0, len(entry.timestamp), ("line", "time", *extra)),
            (len(entry.timestamp), len(full), body_tags),
        ]
        if self._filter_re is None:
            return [(full[a:b], tags) for a, b, tags in spans]
        matches = [(m.start(), m.end()) for m in self._filter_re.finditer(full) if m.end() > m.start()]
        cuts = sorted({0, len(full), *(a for a, _, _ in spans), *(p for m in matches for p in m)})
        segments: list[tuple[str, tuple[str, ...]]] = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            tags = next(t for s, e, t in spans if s <= a < e)
            if any(ms <= a < me for ms, me in matches):
                tags = tags + ("match",)
            segments.append((full[a:b], tags))
        return segments

    def _insert_entries(self, entries: Sequence[LogEntry]) -> None:
        """Inserta ``entries`` al final del widget con una sola llamada a Tk."""
        args: list[Any] = []
        for entry in entries:
            for chunk, tags in self._segments(entry):
                args.extend((chunk, tags))
        if args:
            self.text.insert("end", *args)

    def _append_to_view(self, entries: Sequence[LogEntry]) -> None:
        """Añade entradas nuevas al final y recorta por arriba si se supera el límite de líneas."""
        self._set_text_state("normal")
        # La vista termina siempre en "\n": se inserta justo antes de la línea vacía final.
        self._insert_entries(entries)
        for entry in entries:
            self._visible.append(entry.n_lines)
            self._visible_lines += entry.n_lines
        remove = 0
        while self._visible_lines > self.max_visible_lines and len(self._visible) > 1:
            n = self._visible.popleft()
            remove += n
            self._visible_lines -= n
        if remove:
            self.text.delete("1.0", f"{remove + 1}.0")
        self._set_text_state("disabled")
        if self.autoscroll_var.get():
            self.text.see("end")

    def _rerender(self) -> None:
        """Vuelve a dibujar la vista desde el búfer con los filtros actuales."""
        matching = [entry for entry in self._buffer if self._passes(entry)]
        self._n_matching = len(matching)
        selected: list[LogEntry] = []
        lines = 0
        for entry in reversed(matching):  # las más recientes que quepan
            if selected and lines + entry.n_lines > self.max_visible_lines:
                break
            selected.append(entry)
            lines += entry.n_lines
        selected.reverse()
        self._set_text_state("normal")
        self.text.delete("1.0", "end")
        self._insert_entries(selected)
        self._set_text_state("disabled")
        self._visible = deque(entry.n_lines for entry in selected)
        self._visible_lines = lines
        if self.autoscroll_var.get():
            self.text.see("end")
        else:
            self.text.yview_moveto(0.0)
        self._update_counter()
        self._update_placeholder()

    def _set_text_state(self, state: str) -> None:
        """Habilita (``"normal"``) o bloquea (``"disabled"``) la edición del widget de texto."""
        self.text.configure(state=state)

    def _update_counter(self) -> None:
        """Actualiza «N mensajes» (y «k de N» si hay filtros) y los avisos/errores."""
        total = len(self._buffer)
        if self._n_matching == total:
            text = plural(total, "mensaje", "mensajes")
        else:
            text = f"{self._n_matching:,} de {plural(total, 'mensaje', 'mensajes')}".replace(",", " ")
        shown = len(self._visible)
        if shown < self._n_matching:
            text += f" (se muestran los {shown:,} más recientes)".replace(",", " ")
        self.count_label.configure(text=text)
        self.warnings_label.configure(text=f"⚠ {plural(self._n_warnings, 'aviso', 'avisos')}" if self._n_warnings else "")
        self.errors_label.configure(text=f"✖ {plural(self._n_errors, 'error', 'errores')}" if self._n_errors else "")

    def _update_placeholder(self) -> None:
        """Muestra el mensaje guía si la vista está vacía (sin mensajes o nada pasa los filtros)."""
        if self._visible_lines:
            self.placeholder.place_forget()
            return
        if not self._buffer:
            message = EMPTY_MESSAGE
        else:
            conditions = []
            if self._filter_re is not None:
                conditions.append(f"contengan «{self.filter_var.get().strip()}»")
            if self._min_level > logging.DEBUG:
                conditions.append(f"sean de nivel {self.level_var.get()} o más grave")
            message = (
                "No hay mensajes que " + " y ".join(conditions) + ".\n\n"
                f"Hay {plural(len(self._buffer), 'mensaje', 'mensajes')} ocultos por los filtros: "
                "borra el filtro (✕) o baja el nivel mínimo."
            )
        self.placeholder.configure(text=message)
        self.placeholder.place(relx=0.5, rely=0.4, anchor="center")

    # ------------------------------------------------------------ nivel mínimo
    def _on_level_selected(self) -> None:
        """El usuario eligió otro nivel mínimo: ajusta el handler y vuelve a dibujar."""
        self._set_min_level(logging.getLevelName(self.level_var.get()))
        self._rerender()
        if self._min_level <= logging.DEBUG:
            logger.debug("Nivel DEBUG activado: se muestran los detalles internos de los módulos src.*")

    def _set_min_level(self, level: int) -> None:
        """Fija el nivel mínimo de la vista, del handler de la GUI y de los loggers del proyecto.

        Parameters
        ----------
        level : int
            Nivel numérico de :mod:`logging`.
        """
        self._min_level = int(level)
        handler = getattr(self.app, "log_handler", None)
        if handler is not None:
            handler.setLevel(self._min_level)
        self._apply_logger_threshold(self._min_level)

    def _apply_logger_threshold(self, level: int) -> None:
        """Garantiza que los loggers ``src`` y ``gui`` dejen pasar ``level`` hacia el handler.

        Parameters
        ----------
        level : int
            Nivel mínimo elegido en la pestaña.

        Notes
        -----
        En :mod:`logging` un registro se descarta en el propio logger si su nivel
        es menor que el *nivel efectivo* del logger (el suyo o, si no tiene, el
        del primer ancestro que lo tenga; normalmente el raíz, INFO en la GUI).
        Por eso bajar solo el handler a DEBUG no basta: los ``logger.debug`` de
        ``src.*`` nunca llegarían a él. Esta función:

        1. Restaura los umbrales que la propia pestaña cambió antes.
        2. Si el nivel efectivo de ``src``/``gui`` es mayor que ``level``, lo baja
           a ``level`` (y recuerda el original).
        3. Para que la consola no se inunde de DEBUG, sube los ``StreamHandler``
           sin nivel del logger raíz al umbral que tenían de hecho (el nivel
           efectivo original de ``src``). Las bibliotecas externas (numba,
           matplotlib…) no se tocan: siguen filtradas por el logger raíz.

        Al destruir la pestaña todo vuelve a su estado original.
        """
        self._restore_logging()
        root = logging.getLogger()
        original_effective = logging.getLogger("src").getEffectiveLevel()
        for name in PROJECT_LOGGERS:
            project_logger = logging.getLogger(name)
            if project_logger.getEffectiveLevel() > level:
                self._saved_logger_levels[name] = project_logger.level
                project_logger.setLevel(level)
        if self._saved_logger_levels:
            own = getattr(self.app, "log_handler", None)
            for handler in root.handlers:
                if handler is own or type(handler) is not logging.StreamHandler:
                    continue
                if handler.level < original_effective:
                    self._saved_handler_levels[handler] = handler.level
                    handler.setLevel(original_effective)

    def _restore_logging(self) -> None:
        """Devuelve a su valor original los umbrales de loggers y handlers que la pestaña cambió."""
        for name, level in self._saved_logger_levels.items():
            logging.getLogger(name).setLevel(level)
        for handler, level in self._saved_handler_levels.items():
            handler.setLevel(level)
        self._saved_logger_levels.clear()
        self._saved_handler_levels.clear()

    # ------------------------------------------------------------ callbacks
    def _on_filter_typed(self, *_args: Any) -> None:
        """Cambió el texto del filtro: re-dibuja tras una breve pausa (no en cada tecla)."""
        if self._filter_after_id is not None:
            self.after_cancel(self._filter_after_id)
        self._filter_after_id = self.after(FILTER_DEBOUNCE_MS, self._apply_filter_now)

    def _apply_filter_now(self) -> None:
        """Compila el filtro de texto actual y vuelve a dibujar la vista."""
        if self._filter_after_id is not None:
            self.after_cancel(self._filter_after_id)
            self._filter_after_id = None
        needle = self.filter_var.get().strip()
        self._filter_re = re.compile(re.escape(needle), re.IGNORECASE) if needle else None
        self._rerender()

    def _on_autoscroll_toggled(self) -> None:
        """Al activar el auto-scroll, salta al último mensaje."""
        if self.autoscroll_var.get():
            self.text.see("end")

    def _on_save(self) -> None:
        """Botón «Guardar…»: pide un archivo y guarda las líneas filtradas en UTF-8."""
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(), title="Guardar el log",
            defaultextension=".txt", filetypes=[("Texto", "*.txt"), ("Todos los archivos", "*.*")],
            initialfile=time.strftime("smartuner_log_%Y%m%d_%H%M%S.txt"),
        )
        if not path:
            return
        try:
            n = self.save_to(path)
        except OSError as exc:
            show_error(self.winfo_toplevel(), "No se pudo guardar el log", exc)
            return
        logger.info("Log guardado en %s (%s)", path, plural(n, "mensaje", "mensajes"))
        self._status(f"Log guardado en {Path(path).name}.")

    def _on_busy_changed(self, busy: bool = False, **_kwargs: Any) -> None:
        """Al empezar o terminar una tarea, vacía la cola en el acto (log al día con la barra de estado)."""
        if self.winfo_exists():
            self._drain()

    def _status(self, message: str) -> None:
        """Muestra ``message`` en la barra de estado de la aplicación (si existe)."""
        status = getattr(self.app, "status", None)
        if status is not None:
            status.set_message(message)

    def _on_destroy(self, event: tk.Event) -> None:
        """Al cerrar: detiene el sondeo y restaura los umbrales de logging modificados."""
        if event.widget is not self:
            return
        for after_id in (self._poll_id, self._filter_after_id):
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
        self._poll_id = self._filter_after_id = None
        self._restore_logging()


__all__ = ["LogEntry", "LogTab", "format_record", "level_tag", "module_label", "plural"]
