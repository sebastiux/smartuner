"""Pestañas de la ventana principal (una clase ``ttk.Frame`` por pestaña).

Cada pestaña recibe ``(master, app)`` donde ``app`` es la
:class:`gui.app.SmartunerApp`; se suscribe a los eventos de
``app.state.events`` y llama a las acciones de ``app`` o a funciones de ``src/``.
"""
