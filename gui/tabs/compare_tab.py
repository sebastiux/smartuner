"""Pestaña provisional (CompareTab). [CONTRATO: implementar]"""

from __future__ import annotations

from tkinter import ttk
from typing import Any


class CompareTab(ttk.Frame):
    """Pestaña provisional."""

    def __init__(self, master: Any, app: Any) -> None:
        super().__init__(master, padding=12)
        self.app = app
        ttk.Label(self, text="CompareTab: en construcción").pack()
