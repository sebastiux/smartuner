"""Widgets reutilizables de la interfaz gráfica.

Papel en la arquitectura
------------------------
Piezas comunes a varias pestañas, para que cada pestaña solo contenga su
lógica de presentación:

* :class:`Tooltip` — texto de ayuda al pasar el ratón (explica hiperparámetros).
* :class:`ParamField` — control (spinbox/checkbox/combobox/entrada) construido a
  partir de un :class:`src.config.ParamSpec`, con etiqueta, unidad, tooltip y validación.
* :class:`PlotFrame` — figura de matplotlib embebida (``FigureCanvasTkAgg``) con
  su barra de herramientas (zoom, desplazamiento, guardar).
* :class:`ScrollableFrame` — marco con barra de desplazamiento vertical.
* :class:`StatusBar` — mensaje + barra de progreso + botón Cancelar.
* :class:`AudioPlayer` — reproducción opcional con ``sounddevice``.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.config import ParamSpec

logger = logging.getLogger(__name__)

#: Paleta de la interfaz (coherente con las gráficas).
UI_SURFACE = "#fcfcfb"
UI_TEXT = "#0b0b0b"
UI_TEXT_SECONDARY = "#52514e"
UI_ACCENT = "#2a78d6"
UI_ERROR = "#d03b3b"


class Tooltip:
    """Muestra un texto de ayuda cuando el ratón se detiene sobre un widget.

    Parameters
    ----------
    widget : tk.Widget
        Widget al que se asocia la ayuda.
    text : str | Callable[[], str]
        Texto (o función que lo genera, para ayudas dinámicas).
    delay_ms : int, optional
        Espera antes de mostrarse (500 ms).
    wraplength : int, optional
        Ancho máximo del texto en píxeles (360).
    """

    def __init__(self, widget: tk.Widget, text: str | Callable[[], str], delay_ms: int = 500, wraplength: int = 360) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        text = self.text() if callable(self.text) else self.text
        if not text or self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip, text=text, justify="left", wraplength=self.wraplength, background="#fffbe8",
            foreground=UI_TEXT, relief="solid", borderwidth=1, padx=8, pady=6, font=("TkDefaultFont", 9),
        )
        label.pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class ParamField(ttk.Frame):
    """Control de un hiperparámetro construido desde su :class:`ParamSpec`.

    Layout: ``[etiqueta] [control] [unidad]`` con tooltip en la etiqueta y en el control.

    Parameters
    ----------
    master : tk.Misc
        Contenedor.
    key : str
        Clave ``"seccion.campo"`` (p. ej. ``"env.lam"``).
    spec : ParamSpec
        Metadatos (tipo, rango, ayuda).
    value : Any
        Valor inicial.
    on_change : Callable[[str, Any], None] | None
        Se llama con ``(key, valor)`` cuando el usuario cambia un valor VÁLIDO.

    Notes
    -----
    Tipos soportados: ``int``, ``float`` (Spinbox), ``bool`` (Checkbutton),
    ``choice`` (Combobox), ``int_or_none`` (Spinbox; vacío = None) y
    ``float_list`` (Entry con valores separados por comas).
    """

    def __init__(self, master: tk.Misc, key: str, spec: ParamSpec, value: Any, on_change: Callable[[str, Any], None] | None = None) -> None:
        super().__init__(master)
        self.key = key
        self.spec = spec
        self.on_change = on_change
        self.var: tk.Variable
        self.label = ttk.Label(self, text=spec.label, width=26, anchor="w")
        self.label.grid(row=0, column=0, sticky="w", padx=(0, 6))
        if spec.kind == "bool":
            self.var = tk.BooleanVar(value=bool(value))
            self.control: tk.Widget = ttk.Checkbutton(self, variable=self.var, command=self._changed)
        elif spec.kind == "choice":
            self.var = tk.StringVar(value=str(value))
            self.control = ttk.Combobox(self, textvariable=self.var, values=list(spec.choices), state="readonly", width=14)
            self.control.bind("<<ComboboxSelected>>", lambda _e: self._changed())
        elif spec.kind == "float_list":
            self.var = tk.StringVar(value=", ".join(f"{v:g}" for v in (value or [])))
            self.control = ttk.Entry(self, textvariable=self.var, width=30)
            self.control.bind("<FocusOut>", lambda _e: self._changed())
            self.control.bind("<Return>", lambda _e: self._changed())
        else:
            self.var = tk.StringVar(value="" if value is None else f"{value:g}" if isinstance(value, float) else str(value))
            self.control = ttk.Spinbox(
                self, textvariable=self.var, width=12,
                from_=spec.minimum if spec.minimum is not None else -1e9,
                to=spec.maximum if spec.maximum is not None else 1e9,
                increment=spec.step or 1, command=self._changed,
            )
            self.control.bind("<FocusOut>", lambda _e: self._changed())
            self.control.bind("<Return>", lambda _e: self._changed())
        self.control.grid(row=0, column=1, sticky="w")
        self.unit = ttk.Label(self, text=spec.unit, foreground=UI_TEXT_SECONDARY, width=4)
        self.unit.grid(row=0, column=2, sticky="w", padx=(4, 0))
        self.error = ttk.Label(self, text="", foreground=UI_ERROR)
        self.error.grid(row=1, column=0, columnspan=3, sticky="w")
        self.error.grid_remove()
        help_text = spec.help + (f"\nRango: {spec.minimum:g} – {spec.maximum:g}" if spec.minimum is not None and spec.maximum is not None else "")
        Tooltip(self.label, help_text)
        Tooltip(self.control, help_text)

    def parse(self) -> Any:
        """Convierte el texto del control al tipo del parámetro.

        Returns
        -------
        Any
            Valor validado.

        Raises
        ------
        ValueError
            Con un mensaje en español si el valor es inválido o está fuera de rango.
        """
        spec = self.spec
        raw = self.var.get()
        if spec.kind == "bool":
            return bool(raw)
        if spec.kind == "choice":
            if raw not in spec.choices:
                raise ValueError(f"Opción inválida: {raw}")
            return raw
        if spec.kind == "float_list":
            parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
            if not parts:
                raise ValueError("Escribe al menos un valor (separados por comas).")
            try:
                return [float(p) for p in parts]
            except ValueError as exc:
                raise ValueError("Lista inválida: usa números separados por comas.") from exc
        text = str(raw).strip()
        if spec.kind == "int_or_none" and text in ("", "None", "none"):
            return None
        try:
            value: float = int(float(text)) if spec.kind in ("int", "int_or_none") else float(text)
        except ValueError as exc:
            raise ValueError(f"'{text}' no es un número válido.") from exc
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"Debe ser ≥ {spec.minimum:g}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"Debe ser ≤ {spec.maximum:g}.")
        return value

    def set(self, value: Any) -> None:
        """Muestra ``value`` en el control (sin disparar ``on_change``)."""
        if self.spec.kind == "bool":
            self.var.set(bool(value))
        elif self.spec.kind == "float_list":
            self.var.set(", ".join(f"{v:g}" for v in (value or [])))
        elif value is None:
            self.var.set("")
        else:
            self.var.set(f"{value:g}" if isinstance(value, float) else str(value))
        self._show_error(None)

    def _show_error(self, message: str | None) -> None:
        if message:
            self.error.configure(text="⚠ " + message)
            self.error.grid()
        else:
            self.error.grid_remove()

    def _changed(self) -> None:
        try:
            value = self.parse()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._show_error(None)
        if self.on_change:
            self.on_change(self.key, value)


class PlotFrame(ttk.Frame):
    """Figura de matplotlib embebida con barra de herramientas.

    Las funciones de :mod:`src.plots` aceptan ``fig=``: se les pasa
    ``plot_frame.figure`` para dibujar sobre la figura ya embebida y luego se
    llama a :meth:`draw`.

    Parameters
    ----------
    master : tk.Misc
        Contenedor.
    figsize : tuple[float, float]
        Tamaño inicial en pulgadas.
    toolbar : bool
        Si es True muestra la barra de navegación de matplotlib.
    """

    def __init__(self, master: tk.Misc, figsize: tuple[float, float] = (9.0, 5.5), toolbar: bool = True, dpi: int = 100) -> None:
        super().__init__(master)
        self.figure = Figure(figsize=figsize, dpi=dpi, facecolor=UI_SURFACE)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar: NavigationToolbar2Tk | None = None
        if toolbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw(self) -> None:
        """Redibuja el lienzo (después de modificar ``figure``)."""
        self.canvas.draw_idle()

    def show(self, plot_fn: Callable[..., Figure], *args: Any, **kwargs: Any) -> None:
        """Dibuja ``plot_fn(*args, fig=self.figure, **kwargs)`` y refresca.

        Si la función falla se muestra el error dentro de la figura en vez de romper la GUI.
        """
        try:
            plot_fn(*args, fig=self.figure, **kwargs)
        except Exception as exc:  # noqa: BLE001 - se muestra en la figura
            logger.exception("Error al dibujar %s", getattr(plot_fn, "__name__", plot_fn))
            self.figure.clear()
            self.figure.text(0.5, 0.5, f"No se pudo dibujar la gráfica:\n{exc}", ha="center", va="center", color=UI_ERROR, wrap=True)
        self.draw()

    def clear(self, message: str = "") -> None:
        """Limpia la figura y opcionalmente muestra un mensaje centrado."""
        self.figure.clear()
        if message:
            self.figure.text(0.5, 0.5, message, ha="center", va="center", color=UI_TEXT_SECONDARY, fontsize=11, wrap=True)
        self.draw()


class ScrollableFrame(ttk.Frame):
    """Marco con desplazamiento vertical; el contenido va en ``self.inner``."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.inner.bind_all(seq, self._on_wheel, add="+")

    def _on_wheel(self, event: tk.Event) -> None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None or not str(widget).startswith(str(self)):
            return
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")


class StatusBar(ttk.Frame):
    """Barra inferior: mensaje, barra de progreso y botón Cancelar."""

    def __init__(self, master: tk.Misc, on_cancel: Callable[[], None] | None = None) -> None:
        super().__init__(master, padding=(8, 2))
        self.message = ttk.Label(self, text="Listo.", anchor="w")
        self.message.pack(side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(self, text="Cancelar", command=on_cancel or (lambda: None), state="disabled")
        self.cancel_button.pack(side="right", padx=(6, 0))
        self.progress = ttk.Progressbar(self, length=220, mode="determinate", maximum=1.0)
        self.progress.pack(side="right")

    def set_message(self, text: str) -> None:
        """Muestra ``text`` en la barra de estado."""
        self.message.configure(text=text)

    def set_progress(self, fraction: float, text: str | None = None) -> None:
        """Actualiza la barra (fracción 0–1) y opcionalmente el mensaje."""
        self.progress.configure(value=max(0.0, min(1.0, float(fraction))))
        if text is not None:
            self.set_message(text)

    def set_busy(self, busy: bool) -> None:
        """Habilita Cancelar mientras hay una tarea; reinicia la barra al terminar."""
        self.cancel_button.configure(state="normal" if busy else "disabled")
        if not busy:
            self.progress.configure(value=0.0)


class AudioPlayer:
    """Reproducción opcional de audio con ``sounddevice``.

    Si ``sounddevice`` (o PortAudio) no está disponible, ``available`` es False y
    la GUI deshabilita los botones de reproducción con un tooltip explicativo.
    """

    def __init__(self) -> None:
        self._sd: Any = None
        self.error: str = ""
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            sd.query_devices()
            self._sd = sd
        except Exception as exc:  # noqa: BLE001 - dependencia opcional
            self.error = (
                "Reproducción no disponible: instala la dependencia opcional 'sounddevice' "
                f"(pip install sounddevice) y PortAudio. Detalle: {exc}"
            )

    @property
    def available(self) -> bool:
        """True si se puede reproducir audio."""
        return self._sd is not None

    def play(self, y: np.ndarray, sr: int) -> None:
        """Reproduce ``y`` (no bloqueante); detiene cualquier reproducción previa."""
        if self._sd is None:
            raise RuntimeError(self.error)
        self._sd.stop()
        self._sd.play(np.asarray(y, dtype=np.float32), int(sr))

    def stop(self) -> None:
        """Detiene la reproducción en curso."""
        if self._sd is not None:
            self._sd.stop()


def show_error(parent: tk.Misc, title: str, exc: BaseException | str, details: str | None = None) -> None:
    """Diálogo de error con el mensaje de la excepción (los detalles van al log).

    Parameters
    ----------
    parent : tk.Misc
        Ventana padre.
    title : str
        Título del diálogo.
    exc : BaseException | str
        Error a mostrar.
    details : str | None
        Traceback u otra información técnica (se registra con ``logging``).
    """
    if details:
        logger.error("%s\n%s", title, details)
    messagebox.showerror(title, str(exc), parent=parent)
