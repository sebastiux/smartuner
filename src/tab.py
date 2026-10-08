"""Etapa 8 — Tablatura. [CONTRATO: implementar]

Convierte la posición elegida para cada segmento en una tablatura de bajo de
4 líneas (G, D, A, E de arriba abajo) con marcas de tiempo, y la exporta a
TXT, JSON y CSV.

Formato ASCII (una columna por nota, en orden temporal; los sistemas se
parten al llegar a ``line_width``)::

    t(s) 0.50   0.80   1.10
    G|-----------------------|
    D|-----------------2-----|
    A|--0------4-------------|
    E|-----------------------|
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.environment import Arm
from src.segmentation import Segment


@dataclass
class TabNote:
    """Una nota de la tablatura.

    Attributes
    ----------
    position : int
        Índice del segmento entre los conservados.
    start_s, end_s : float
        Tiempos del segmento (s).
    string : str
        Cuerda elegida.
    fret : int
        Traste elegido.
    midi : int
        Pitch de la posición elegida.
    f0_hz : float | None
        f0 estimada por pYIN (informativa).
    """

    position: int
    start_s: float
    end_s: float
    string: str
    fret: int
    midi: int
    f0_hz: float | None = None

    @property
    def label(self) -> str:
        """Etiqueta ``"A-0"``."""
        return f"{self.string}-{self.fret}"


def notes_from_choices(segments: list[Segment], arms: list[Arm]) -> list[TabNote]:
    """Empareja segmentos conservados (en orden) con el brazo elegido para cada uno."""
    raise NotImplementedError


def render_ascii(notes: list[TabNote], line_width: int = 80, show_times: bool = True, title: str | None = None) -> str:
    """Tablatura ASCII de 4 líneas con marcas de tiempo (ver encabezado)."""
    raise NotImplementedError


def export_txt(notes: list[TabNote], path: str | Path, title: str | None = None) -> Path:
    """Guarda :func:`render_ascii` en un archivo de texto UTF-8."""
    raise NotImplementedError


def export_json(notes: list[TabNote], path: str | Path, meta: dict | None = None) -> Path:
    """Guarda ``{"meta": {...}, "notes": [ ... ]}``."""
    raise NotImplementedError


def export_csv(notes: list[TabNote], path: str | Path) -> Path:
    """CSV con columnas: position,start_s,end_s,string,fret,midi,note,f0_hz."""
    raise NotImplementedError


def export_tab(notes: list[TabNote], path: str | Path, meta: dict | None = None) -> Path:
    """Exporta según la extensión de ``path`` (.txt, .json o .csv)."""
    raise NotImplementedError
