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
* :class:`AudioPlayer` — reproducción del audio original con ``sounddevice``.
* :class:`HoverTooltip`, :func:`tree_heading_at` — ayudas según lo que hay bajo el puntero.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable, Hashable
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
UI_TEXT_MUTED = "#898781"
UI_ACCENT = "#2a78d6"
UI_ERROR = "#d03b3b"


class Tooltip:
    """Muestra un texto de ayuda cuando el ratón se detiene sobre un widget.

    El recuadro aparece JUNTO AL PUNTERO (abajo a la derecha) y nunca se sale
    de la pantalla: en widgets altos (una tabla, un lienzo) un tooltip
    colocado debajo de todo el widget quedaría fuera de la vista.

    Parameters
    ----------
    widget : tk.Widget
        Widget al que se asocia la ayuda.
    text : str | Callable[[], str]
        Texto (o función que lo genera, para ayudas dinámicas). Una cadena
        vacía no muestra nada.
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
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:  # el widget ya se destruyó
                pass
            self._after_id = None

    def current_text(self) -> str:
        """Texto que se mostraría ahora (evalúa ``text`` si es una función)."""
        return self.text() if callable(self.text) else self.text

    def _show(self) -> None:
        self._after_id = None
        try:
            text = self.current_text()
        except tk.TclError:  # el widget se destruyó mientras se esperaba
            return
        if not text or self._tip is not None:
            return
        self._tip = tip = tk.Toplevel(self.widget)
        tip.withdraw()  # se coloca antes de mostrarse: sin parpadeo en una esquina
        tip.wm_overrideredirect(True)
        label = tk.Label(
            tip, text=text, justify="left", wraplength=self.wraplength, background="#fffbe8",
            foreground=UI_TEXT, relief="solid", borderwidth=1, padx=8, pady=6, font=("TkDefaultFont", 9),
        )
        label.pack()
        tip.update_idletasks()
        x, y = self._position(tip.winfo_reqwidth(), tip.winfo_reqheight())
        tip.wm_geometry(f"+{x}+{y}")
        tip.deiconify()

    def _position(self, width: int, height: int) -> tuple[int, int]:
        """Esquina superior izquierda (px de pantalla) del recuadro de ``width`` × ``height`` px.

        Abajo a la derecha del puntero; si no cabe debajo, encima; nunca fuera
        de la pantalla por la derecha. Si el puntero no está en esta pantalla,
        se usa la esquina inferior izquierda del widget.
        """
        w = self.widget
        px, py = w.winfo_pointerx(), w.winfo_pointery()
        if px < 0 or py < 0:
            px, py = w.winfo_rootx() + 2, w.winfo_rooty() + w.winfo_height() - 14
        screen_w, screen_h = w.winfo_screenwidth(), w.winfo_screenheight()
        x, y = px + 14, py + 18
        if y + height > screen_h:
            y = max(0, py - height - 10)
        x = max(0, min(x, screen_w - width - 4))
        return x, y

    def hide(self) -> None:
        """Oculta el tooltip (y cancela uno a punto de aparecer)."""
        self._hide()

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


class HoverTooltip(Tooltip):
    """Tooltip cuyo texto depende de la ZONA bajo el puntero (una nota de un lienzo, un encabezado...).

    Al pasar a otra zona se oculta y se vuelve a esperar ``delay_ms``; fuera
    de cualquier zona no se muestra nada (no tapa el contenido).

    Parameters
    ----------
    widget : tk.Widget
        Widget al que se asocia.
    key_for : Callable[[int, int], Hashable | None]
        ``(x, y)`` (px relativos al widget) → identificador de la zona
        (None = nada que explicar).
    text_for : Callable[[Hashable], str]
        Texto de una zona ("" = no mostrar).
    delay_ms : int, optional
        Espera antes de mostrarse (350 ms).
    wraplength : int, optional
        Ancho máximo del texto (px).

    Examples
    --------
    Ayuda por columna en una tabla::

        HoverTooltip(tree, lambda x, y: tree_heading_at(tree, x, y), help_for_column)
    """

    def __init__(self, widget: tk.Widget, key_for: Callable[[int, int], Hashable | None],
                 text_for: Callable[[Hashable], str], delay_ms: int = 350, wraplength: int = 360) -> None:
        self._key_for = key_for
        self._text_for = text_for
        self._hover: Hashable | None = None
        super().__init__(widget, self._text_under_pointer, delay_ms=delay_ms, wraplength=wraplength)
        widget.bind("<Motion>", self._on_motion, add="+")

    def hide(self) -> None:
        """Oculta el tooltip (p. ej. al redibujar el lienzo: lo que hay bajo el puntero cambió)."""
        self._hover = None
        self._hide()

    def pointer_key(self) -> Hashable | None:
        """Zona bajo el puntero (o None)."""
        w = self.widget
        try:
            return self._key_for(w.winfo_pointerx() - w.winfo_rootx(), w.winfo_pointery() - w.winfo_rooty())
        except tk.TclError:
            return None

    def _text_under_pointer(self) -> str:
        """Texto de la zona actual (cadena vacía = no mostrar nada)."""
        key = self.pointer_key()
        return self._text_for(key) if key is not None else ""

    def _on_motion(self, event: tk.Event) -> None:
        """Reinicia la espera cuando el ratón pasa a otra zona."""
        key = self._key_for(event.x, event.y)
        if key != self._hover:
            self._hover = key
            self._hide()
            if key is not None:
                self._schedule()


def tree_display_columns(tree: ttk.Treeview) -> list[str]:
    """Columnas visibles de ``tree`` en orden (resuelve ``displaycolumns = "#all"``).

    Parameters
    ----------
    tree : ttk.Treeview
        Tabla.

    Returns
    -------
    list[str]
        Identificadores de las columnas de datos visibles (sin ``"#0"``).
    """
    shown = tree.cget("displaycolumns")
    if isinstance(shown, str):
        shown = tree.tk.splitlist(shown)
    shown = [str(c) for c in shown]
    if not shown or shown == ["#all"]:
        columns = tree.cget("columns")
        return [str(c) for c in (tree.tk.splitlist(columns) if isinstance(columns, str) else columns)]
    return shown


def tree_heading_at(tree: ttk.Treeview, x: int, y: int) -> str | None:
    """Columna cuyo ENCABEZADO está en ``(x, y)`` (px relativos a la tabla), o None.

    Parameters
    ----------
    tree : ttk.Treeview
        Tabla.
    x, y : int
        Coordenadas dentro de la tabla (px).

    Returns
    -------
    str | None
        ``"#0"`` para la columna del árbol, el identificador de la columna en
        las demás, o None fuera de los encabezados.
    """
    if tree.identify_region(x, y) not in ("heading", "separator"):
        return None
    column = tree.identify_column(x)  # "#n": n-ésima columna VISIBLE (#0 = la del árbol)
    if column == "#0":
        return "#0"
    try:
        index = int(column.lstrip("#")) - 1
    except ValueError:
        return None
    shown = tree_display_columns(tree)
    return shown[index] if 0 <= index < len(shown) else None


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
    label_width : int, optional
        Ancho de la etiqueta en caracteres (26); una sección puede medir sus
        etiquetas y alinear todos sus campos.
    unit_width : int, optional
        Ancho de la unidad en caracteres (4; «muestras» necesita 8).

    Notes
    -----
    Tipos soportados: ``int``, ``float`` (Spinbox), ``bool`` (Checkbutton),
    ``choice`` (Combobox), ``int_or_none`` (Spinbox; vacío = None) y
    ``float_list`` (Entry con valores separados por comas).

    Los ``Spinbox`` y ``Combobox`` de ttk cambian su valor con la rueda del
    ratón: dentro de un formulario desplazable eso modificaría parámetros sin
    querer al desplazar la página. Aquí la rueda sobre el control desplaza el
    :class:`ScrollableFrame` que lo contiene (si lo hay) y nunca cambia el valor.
    """

    def __init__(self, master: tk.Misc, key: str, spec: ParamSpec, value: Any,
                 on_change: Callable[[str, Any], None] | None = None, label_width: int = 26,
                 unit_width: int = 4) -> None:
        super().__init__(master)
        self.key = key
        self.spec = spec
        self.on_change = on_change
        self.var: tk.Variable
        self.label = ttk.Label(self, text=spec.label, width=label_width, anchor="w")
        self.label.grid(row=0, column=0, sticky="w", padx=(0, 6))
        # master=self: las variables pertenecen al intérprete de ESTA ventana (con varias tk.Tk,
        # p. ej. en las pruebas, una variable sin master se crearía en la primera).
        if spec.kind == "bool":
            self.var = tk.BooleanVar(master=self, value=bool(value))
            self.control: tk.Widget = ttk.Checkbutton(self, variable=self.var, command=self._changed)
        elif spec.kind == "choice":
            # El Combobox muestra las etiquetas en español (ParamSpec.choice_labels); parse()
            # las traduce de vuelta al identificador que guarda la configuración.
            labels = [spec.choice_label(c) for c in spec.choices]
            self.var = tk.StringVar(master=self, value=spec.choice_label(str(value)))
            self.control = ttk.Combobox(self, textvariable=self.var, values=labels, state="readonly",
                                        width=max(14, max((len(t) for t in labels), default=0) + 2))
            self.control.bind("<<ComboboxSelected>>", lambda _e: self._changed())
        elif spec.kind == "float_list":
            self.var = tk.StringVar(master=self, value=", ".join(f"{v:g}" for v in (value or [])))
            self.control = ttk.Entry(self, textvariable=self.var, width=30)
            self.control.bind("<FocusOut>", lambda _e: self._changed())
            self.control.bind("<Return>", lambda _e: self._changed())
        else:
            self.var = tk.StringVar(master=self, value="" if value is None else f"{value:g}" if isinstance(value, float) else str(value))
            self.control = ttk.Spinbox(
                self, textvariable=self.var, width=12,
                from_=spec.minimum if spec.minimum is not None else -1e9,
                to=spec.maximum if spec.maximum is not None else 1e9,
                increment=spec.step or 1, command=self._changed,
            )
            self.control.bind("<FocusOut>", lambda _e: self._changed())
            self.control.bind("<Return>", lambda _e: self._changed())
        if spec.kind in ("choice", "int", "float", "int_or_none"):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.control.bind(sequence, self._on_wheel)
        self.control.grid(row=0, column=1, sticky="w")
        self.unit = ttk.Label(self, text=spec.unit, foreground=UI_TEXT_SECONDARY, width=unit_width)
        self.unit.grid(row=0, column=2, sticky="w", padx=(4, 0))
        self.error = ttk.Label(self, text="", foreground=UI_ERROR)
        self.error.grid(row=1, column=0, columnspan=3, sticky="w")
        self.error.grid_remove()
        self._enabled = True
        help_text = spec.help + (f"\nRango: {spec.minimum:g} – {spec.maximum:g}" if spec.minimum is not None and spec.maximum is not None else "")
        Tooltip(self.label, help_text)
        Tooltip(self.control, help_text)

    def _on_wheel(self, event: tk.Event) -> str:
        """Rueda sobre el control: desplaza el formulario que lo contiene y NO cambia el valor."""
        widget: Any = self.master
        while widget is not None:
            if isinstance(widget, ScrollableFrame):
                widget.scroll_wheel(event)
                break
            widget = getattr(widget, "master", None)
        return "break"

    @property
    def enabled(self) -> bool:
        """True si el control admite cambios (ver :meth:`set_enabled`)."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Habilita o deshabilita el control (la etiqueta se atenúa cuando no se usa).

        Parameters
        ----------
        enabled : bool
            False, p. ej., para ``env.n_fft`` cuando el espectro es CQT.
        """
        self._enabled = bool(enabled)
        self.control.state(["!disabled"] if enabled else ["disabled"])
        self.label.configure(foreground=UI_TEXT if enabled else UI_TEXT_MUTED)

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
            value = spec.choice_value(str(raw))  # acepta la etiqueta mostrada o el identificador
            if value is None:
                raise ValueError(f"Opción inválida: {raw}")
            return value
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
        elif self.spec.kind == "choice":
            self.var.set(self.spec.choice_label(str(value)))
        elif value is None:
            self.var.set("")
        else:
            self.var.set(f"{value:g}" if isinstance(value, float) else str(value))
        self.show_error(None)

    def show_error(self, message: str | None) -> None:
        """Muestra ``message`` en rojo bajo el control (None u "" lo oculta).

        Sirve también para errores que detecta otra capa (p. ej.
        :meth:`src.config.Config.validate` sobre una lista de barrido).

        Parameters
        ----------
        message : str | None
            Texto del error.
        """
        if message:
            self.error.configure(text="⚠ " + message)
            self.error.grid()
        else:
            self.error.grid_remove()

    def _changed(self) -> None:
        try:
            value = self.parse()
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.show_error(None)
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
    """Marco con desplazamiento vertical; el contenido va en ``self.inner``.

    La rueda del ratón desplaza el marco cuando el puntero está sobre él (en
    Windows/macOS llega ``<MouseWheel>``; en Linux/X11, ``<Button-4/5>``).
    """

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
        """Rueda en cualquier parte de la ventana: solo actúa si el puntero está sobre este marco."""
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (tk.TclError, KeyError):  # puntero sobre un menú o una ventana ya destruida
            return
        if widget is None or not str(widget).startswith(str(self)):
            return
        self.scroll_wheel(event)

    def scroll_wheel(self, event: tk.Event) -> None:
        """Desplaza tres líneas arriba o abajo según el evento de rueda (``delta`` o botón 4/5).

        Parameters
        ----------
        event : tk.Event
            ``<MouseWheel>`` (``delta`` > 0 = hacia arriba) o ``<Button-4>``/``<Button-5>``.
        """
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(delta, "units")


class StatusBar(ttk.Frame):
    """Barra inferior: mensaje, barra de progreso y botón Cancelar.

    La barra de progreso y «Cancelar» se colocan primero (a la derecha) y el
    mensaje ocupa el resto: un mensaje largo (p. ej. una ruta de exportación)
    se recorta en lugar de empujar fuera de la ventana el botón Cancelar; el
    texto completo se ve en el tooltip del mensaje.
    """

    def __init__(self, master: tk.Misc, on_cancel: Callable[[], None] | None = None) -> None:
        super().__init__(master, padding=(8, 2))
        self.cancel_button = ttk.Button(self, text="Cancelar", command=on_cancel or (lambda: None), state="disabled")
        self.cancel_button.pack(side="right", padx=(6, 0))
        self.progress = ttk.Progressbar(self, length=220, mode="determinate", maximum=1.0)
        self.progress.pack(side="right")
        # width=1: el tamaño PEDIDO no depende del texto (si no, un texto largo agrandaría la barra).
        self.message = ttk.Label(self, text="Listo.", anchor="w", width=1)
        self.message.pack(side="left", fill="x", expand=True, padx=(0, 8))
        Tooltip(self.message, lambda: str(self.message.cget("text")), delay_ms=700, wraplength=600)

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
    """Reproducción del audio original con ``sounddevice`` (pestaña Audio).

    ``sounddevice`` es una dependencia principal (``requirements.txt``), pero
    puede faltar PortAudio o una salida de audio (p. ej. un servidor sin
    tarjeta de sonido). La detección es la misma que la del reproductor de la
    tablatura (:func:`gui.playback.load_sounddevice`): exige una salida por
    defecto, no solo que el módulo se importe. Si no se puede reproducir,
    ``available`` es False y la GUI deshabilita los botones con ``error`` en
    su tooltip.

    Parameters
    ----------
    module : Any | None, optional
        Módulo ``sounddevice`` (o un sustituto con ``play``/``stop``, p. ej. en
        las pruebas). Si es None se detecta.

    Notes
    -----
    sounddevice tiene UNA reproducción «actual» por proceso: ``sd.play``
    detiene la anterior. La aplicación coordina los dos reproductores (este y
    el de la pestaña Tablatura) con :meth:`gui.app.SmartunerApp.claim_audio_output`
    para que nunca haya dos activos ni un estado incoherente.
    """

    def __init__(self, module: Any | None = None) -> None:
        from gui.playback import SoundDeviceOutput  # importe diferido: playback depende de src.tab

        self._output = SoundDeviceOutput(module)

    @property
    def available(self) -> bool:
        """True si se puede reproducir audio."""
        return self._output.available

    @property
    def error(self) -> str:
        """Motivo por el que no se puede reproducir ("" si se puede)."""
        return self._output.error

    def play(self, y: np.ndarray, sr: int) -> None:
        """Reproduce ``y`` (no bloqueante); detiene cualquier reproducción previa.

        Si la tarjeta no acepta ``sr`` (p. ej. WASAPI en Windows solo admite
        44 100/48 000 Hz) se remuestrea y se reintenta
        (:meth:`gui.playback.SoundDeviceOutput.play`).

        Parameters
        ----------
        y : np.ndarray
            Señal mono en [-1, 1].
        sr : int
            Frecuencia de muestreo (Hz).

        Raises
        ------
        RuntimeError
            Si no hay reproducción disponible o el dispositivo rechaza el audio.
        """
        self._output.play(y, int(sr))

    def stop(self) -> None:
        """Detiene la reproducción en curso (no falla si no había ninguna)."""
        self._output.stop()

    def is_active(self) -> bool | None:
        """True si lo último que se reprodujo aquí sigue sonando (None si no se sabe)."""
        return self._output.is_active()


def split_error_message(text: str, max_chars: int = 300) -> tuple[str, str]:
    r"""Separa un mensaje de error en resumen (lo que se lee primero) y detalle técnico.

    El resumen es el primer párrafo (hasta una línea en blanco) o, si no hay
    párrafos, la primera línea; si es muy largo se corta y el resto pasa al
    detalle (p. ej. la salida de ffmpeg).

    Parameters
    ----------
    text : str
        Mensaje completo de la excepción.
    max_chars : int, optional
        Longitud máxima del resumen (caracteres).

    Returns
    -------
    tuple[str, str]
        ``(resumen, detalle)``; el detalle puede ser "".

    Examples
    --------
    >>> split_error_message("No existe el archivo.\n\nRevisa la ruta.")
    ('No existe el archivo.', 'Revisa la ruta.')
    >>> split_error_message("ffmpeg falló: x\nlínea 2\nlínea 3")
    ('ffmpeg falló: x', 'línea 2\nlínea 3')
    """
    text = text.strip()
    if "\n\n" in text:
        summary, detail = text.split("\n\n", 1)
    else:
        summary, _, detail = text.partition("\n")
    if len(summary) > max_chars:
        cut = summary.rfind(" ", 0, max_chars)
        cut = cut if cut > max_chars // 2 else max_chars
        summary, detail = summary[:cut].rstrip() + "…", ("…" + summary[cut:].lstrip() + "\n" + detail).strip()
    return summary.strip(), detail.strip()


def show_error(parent: tk.Misc, title: str, exc: BaseException | str, details: str | None = None) -> None:
    """Diálogo de error con el mensaje de la excepción (el traceback va al log).

    El primer párrafo del mensaje se muestra destacado y el resto (p. ej. la
    salida técnica de ffmpeg) debajo, en letra más pequeña
    (:func:`split_error_message`).

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
    summary, extra = split_error_message(str(exc) or type(exc).__name__)
    if extra:
        messagebox.showerror(title, summary, detail=extra, parent=parent)
    else:
        messagebox.showerror(title, summary, parent=parent)
