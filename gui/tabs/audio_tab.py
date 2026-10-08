"""Pestaña provisional (AudioTab). [CONTRATO: implementar]"""

from __future__ import annotations

from tkinter import ttk
from typing import Any


class AudioTab(ttk.Frame):
    """Pestaña provisional."""

    def __init__(self, master: Any, app: Any) -> None:
        super().__init__(master, padding=12)
        self.app = app
        ttk.Label(self, text="AudioTab: en construcción").pack()
