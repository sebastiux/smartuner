"""Etapa 8 — Tablatura.

Papel en el pipeline
--------------------
Es la salida final del sistema. Cada segmento conservado fue un problema
bandit independiente y su agente recomendó un brazo (cuerda, traste); este
módulo empareja cada segmento con su brazo (:func:`notes_from_choices`), lo
dibuja como una tablatura de bajo de 4 líneas (G, D, A, E de arriba abajo)
con marcas de tiempo (:func:`render_ascii`) y la exporta a TXT, JSON y CSV
(:func:`export_tab`).

Formato ASCII
-------------
Cada nota ocupa una columna propia, en orden temporal, de ancho
``len(traste) + 2`` (el traste rodeado de un guion a cada lado); en las demás
cuerdas esa columna se rellena con ``-``. Encima va la línea ``t(s)`` con el
inicio de cada nota alineado sobre su traste cuando cabe (si dos tiempos se
solaparían, se omite el del medio; el primero y el último de cada sistema se
muestran siempre que quepan). Los sistemas se parten al llegar a
``line_width`` caracteres y todas las líneas de un sistema tienen el mismo
largo::

    t(s)  0.50 1.10
       G|---------|
       D|-------2-|
       A|-0--4----|
       E|---------|

(notas A-0 en 0.50 s, A-4 en 0.80 s y D-2 en 1.10 s; el tiempo 0.80 se omite
porque se pegaría a los otros dos.)

La cuerda más aguda (G) va arriba, como en cualquier tablatura de bajo: la
línea de abajo es la cuerda más gruesa (E).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import STRING_ORDER
from src.environment import Arm
from src.pitch import midi_to_name
from src.segmentation import Segment

logger = logging.getLogger(__name__)

#: Cuerdas en el orden en que se imprimen (de arriba abajo): la más aguda arriba.
TAB_STRINGS: tuple[str, ...] = tuple(reversed(STRING_ORDER))  # ("G", "D", "A", "E")

#: Etiqueta de la línea de tiempos.
TIME_LABEL: str = "t(s)"

#: Decimales de las marcas de tiempo (s).
TIME_DECIMALS: int = 2

#: Columnas del CSV exportado (en este orden).
CSV_COLUMNS: tuple[str, ...] = ("position", "start_s", "end_s", "string", "fret", "midi", "note", "f0_hz")


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

    Examples
    --------
    >>> n = TabNote(position=0, start_s=0.5, end_s=0.8, string="A", fret=0, midi=33)
    >>> n.label, n.note_name
    ('A-0', 'A1')
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

    @property
    def note_name(self) -> str:
        """Nombre de la nota (``"A1"``) según :func:`src.pitch.midi_to_name`."""
        return midi_to_name(self.midi)

    def to_dict(self) -> dict[str, Any]:
        """Diccionario serializable (JSON/CSV) con los campos, el nombre de nota y la etiqueta.

        Returns
        -------
        dict[str, Any]
            Claves ``position``, ``start_s`` y ``end_s`` (s, 4 decimales),
            ``string``, ``fret``, ``midi``, ``note`` (p. ej. ``"A1"``),
            ``label`` (``"A-0"``) y ``f0_hz`` (Hz, 3 decimales, o None).
        """
        return {
            "position": int(self.position),
            "start_s": round(float(self.start_s), 4),
            "end_s": round(float(self.end_s), 4),
            "string": self.string,
            "fret": int(self.fret),
            "midi": int(self.midi),
            "note": self.note_name,
            "label": self.label,
            "f0_hz": None if self.f0_hz is None else round(float(self.f0_hz), 3),
        }


# ---------------------------------------------------------------------------
# Segmentos + brazos → notas
# ---------------------------------------------------------------------------


def notes_from_choices(segments: list[Segment], arms: list[Arm]) -> list[TabNote]:
    """Empareja segmentos conservados (en orden) con el brazo elegido para cada uno.

    Parameters
    ----------
    segments : list[Segment]
        Segmentos CONSERVADOS (``kept=True``), en orden temporal: el i-ésimo
        es el i-ésimo problema bandit.
    arms : list[Arm]
        Brazo recomendado por el agente para cada segmento (mismo orden).

    Returns
    -------
    list[TabNote]
        Una nota por segmento; ``position`` es su índice entre los
        conservados, ``midi`` el de la posición elegida (no el de pYIN) y
        ``f0_hz`` la f0 de pYIN como referencia.

    Raises
    ------
    ValueError
        Si las dos listas no tienen la misma longitud.

    Examples
    --------
    >>> seg = Segment(index=3, start_s=0.5, end_s=0.8, start_sample=0, end_sample=1, rms_db=-3.0, f0_hz=55.2)
    >>> notes_from_choices([seg], [Arm("A", 0)])[0]
    TabNote(position=0, start_s=0.5, end_s=0.8, string='A', fret=0, midi=33, f0_hz=55.2)
    """
    if len(segments) != len(arms):
        raise ValueError(
            f"Hay {len(segments)} segmentos pero {len(arms)} posiciones elegidas: "
            "debe haber exactamente un brazo por segmento conservado."
        )
    notes = [
        TabNote(
            position=i,
            start_s=float(seg.start_s),
            end_s=float(seg.end_s),
            string=arm.string,
            fret=int(arm.fret),
            midi=int(arm.midi),
            f0_hz=None if seg.f0_hz is None else float(seg.f0_hz),
        )
        for i, (seg, arm) in enumerate(zip(segments, arms))
    ]
    logger.debug("Tablatura: %d notas emparejadas con su posición", len(notes))
    return notes


# ---------------------------------------------------------------------------
# Tablatura ASCII
# ---------------------------------------------------------------------------


def _cell(note: TabNote) -> str:
    """Celda de una nota en su cuerda: el traste con un guion a cada lado (``"-12-"``)."""
    return f"-{note.fret}-"


def _split_systems(notes: list[TabNote], max_content: int) -> list[list[TabNote]]:
    """Reparte las notas en sistemas cuyo contenido no supere ``max_content`` caracteres.

    Cada sistema lleva al menos una nota (aunque no quepa), para que nunca se
    produzca un bucle infinito con ``line_width`` muy pequeño.
    """
    systems: list[list[TabNote]] = []
    current: list[TabNote] = []
    width = 0
    for note in notes:
        w = len(_cell(note))
        if current and width + w > max_content:
            systems.append(current)
            current, width = [], 0
        current.append(note)
        width += w
    if current:
        systems.append(current)
    return systems


def _time_row(system: list[TabNote], col_starts: list[int], gutter: int, total: int) -> str:
    """Línea ``t(s)`` del sistema: tiempo de inicio de cada nota sobre su traste.

    Cada etiqueta se coloca empezando en la columna del traste; si se saldría
    del sistema por la derecha se desplaza a la izquierda (sin salir de la
    zona de notas). Una etiqueta se omite si quedaría pegada (sin al menos un
    espacio) a otra ya colocada. Primero se colocan la del primer y la del
    último tiempo del sistema, y después las intermedias de izquierda a derecha.
    """
    chars = [" "] * total
    chars[:len(TIME_LABEL)] = TIME_LABEL
    first_free = gutter + 1  # las etiquetas empiezan después de la columna de la barra inicial
    placed: list[tuple[int, int]] = []  # intervalos [inicio, fin) ya ocupados

    order = [0] + ([len(system) - 1] if len(system) > 1 else []) + list(range(1, len(system) - 1))
    for i in order:
        text = f"{system[i].start_s:.{TIME_DECIMALS}f}"
        start = col_starts[i] + 1                      # alineada con el primer dígito del traste
        start = min(start, total - len(text))          # no sobrepasar el final del sistema
        if start < first_free:
            continue                                   # no cabe en el sistema
        end = start + len(text)
        if any(start <= b and a <= end for a, b in placed):  # exige ≥ 1 espacio de separación
            continue
        chars[start:end] = text
        placed.append((start, end))
    return "".join(chars)


def _render_system(system: list[TabNote], gutter: int, show_times: bool) -> list[str]:
    """Líneas (``t(s)``, G, D, A, E) de un sistema, todas del mismo largo."""
    col_starts: list[int] = []
    pos = gutter + 1  # nombre de la cuerda (gutter caracteres) + "|"
    for note in system:
        col_starts.append(pos)
        pos += len(_cell(note))
    total = pos + 1  # barra final

    lines: list[str] = []
    if show_times:
        lines.append(_time_row(system, col_starts, gutter, total))
    for string in TAB_STRINGS:
        content = "".join(_cell(n) if n.string == string else "-" * len(_cell(n)) for n in system)
        lines.append(f"{string:>{gutter}}|{content}|")
    return lines


def render_ascii(notes: list[TabNote], line_width: int = 80, show_times: bool = True, title: str | None = None) -> str:
    """Tablatura ASCII de 4 líneas con marcas de tiempo (ver encabezado).

    Parameters
    ----------
    notes : list[TabNote]
        Notas en orden temporal (una columna por nota).
    line_width : int, optional
        Ancho máximo de cada línea en caracteres; al llegar a él la
        tablatura continúa en un nuevo sistema (separado por una línea en
        blanco). Un sistema contiene al menos una nota aunque no quepa.
    show_times : bool, optional
        Si es True, encima de cada sistema se imprime la línea ``t(s)`` con el
        inicio (s) de cada nota alineado sobre su traste cuando cabe.
    title : str | None, optional
        Título impreso en la primera línea.

    Returns
    -------
    str
        Texto multilínea (sin salto de línea final).

    Raises
    ------
    ValueError
        Si alguna nota tiene una cuerda distinta de G, D, A o E.

    Examples
    --------
    >>> notes = [TabNote(0, 0.5, 0.8, "A", 0, 33), TabNote(1, 0.8, 1.1, "A", 4, 37),
    ...          TabNote(2, 1.1, 1.4, "D", 2, 40)]
    >>> print(render_ascii(notes))
    t(s)  0.50 1.10
       G|---------|
       D|-------2-|
       A|-0--4----|
       E|---------|
    >>> print(render_ascii(notes, show_times=False, title="Ejemplo"))
    Ejemplo
    <BLANKLINE>
    G|---------|
    D|-------2-|
    A|-0--4----|
    E|---------|
    """
    bad = sorted({n.string for n in notes} - set(TAB_STRINGS))
    if bad:
        raise ValueError(f"Cuerda(s) desconocida(s) en la tablatura: {', '.join(bad)} (se esperaba G, D, A o E).")

    # Ancho de la columna de nombres: "t(s)" si hay tiempos (los nombres de cuerda
    # se alinean a la derecha, junto a la barra), 1 carácter si no.
    gutter = len(TIME_LABEL) if show_times else 1
    max_content = line_width - gutter - 2  # descontando el nombre y las dos barras
    blocks: list[str] = []
    if title:
        blocks.append(title)
    if not notes:
        empty = [f"{s:>{gutter}}|----|" for s in TAB_STRINGS]
        if show_times:
            empty.insert(0, TIME_LABEL.ljust(len(empty[0])))
        blocks.append("\n".join(empty))
        return "\n\n".join(blocks)
    systems = _split_systems(notes, max_content)
    blocks.extend("\n".join(_render_system(system, gutter, show_times)) for system in systems)
    logger.debug("Tablatura ASCII: %d notas en %d sistemas (ancho %d)", len(notes), len(systems), line_width)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------


def _title_from_meta(meta: dict | None) -> str | None:
    """Título para el TXT: ``meta["title"]`` o un resumen ``clave: valor`` de los metadatos simples."""
    if not meta:
        return None
    if meta.get("title"):
        return str(meta["title"])
    parts = [f"{k}: {v}" for k, v in meta.items() if isinstance(v, (str, int, float, bool)) or v is None]
    return " · ".join(parts) or None


def export_txt(notes: list[TabNote], path: str | Path, title: str | None = None) -> Path:
    """Guarda :func:`render_ascii` en un archivo de texto UTF-8.

    Parameters
    ----------
    notes : list[TabNote]
        Notas de la tablatura.
    path : str | Path
        Archivo de salida (se crean las carpetas intermedias).
    title : str | None, optional
        Título de la primera línea.

    Returns
    -------
    Path
        Ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ascii(notes, title=title) + "\n", encoding="utf-8")
    logger.info("Tablatura TXT guardada en %s (%d notas)", path, len(notes))
    return path


def export_json(notes: list[TabNote], path: str | Path, meta: dict | None = None) -> Path:
    """Guarda ``{"meta": {...}, "notes": [ ... ]}``.

    Cada nota es ``{position, start_s, end_s, string, fret, midi, note, label,
    f0_hz}`` (``note`` = nombre como ``"A1"``; ``f0_hz`` = null si se desconoce).

    Parameters
    ----------
    notes : list[TabNote]
        Notas de la tablatura.
    path : str | Path
        Archivo de salida (se crean las carpetas intermedias).
    meta : dict | None, optional
        Metadatos serializables (algoritmo, archivo de audio, configuración...).

    Returns
    -------
    Path
        Ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": dict(meta or {}), "notes": [n.to_dict() for n in notes]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Tablatura JSON guardada en %s (%d notas)", path, len(notes))
    return path


def export_csv(notes: list[TabNote], path: str | Path) -> Path:
    """CSV con columnas: position,start_s,end_s,string,fret,midi,note,f0_hz.

    ``note`` es el nombre de la nota (``midi_to_name``, p. ej. ``"A1"``) y
    ``f0_hz`` queda vacío si pYIN no estimó f0.

    Parameters
    ----------
    notes : list[TabNote]
        Notas de la tablatura.
    path : str | Path
        Archivo de salida (se crean las carpetas intermedias).

    Returns
    -------
    Path
        Ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for note in notes:
            row = note.to_dict()
            if row["f0_hz"] is None:
                row["f0_hz"] = ""
            writer.writerow(row)
    logger.info("Tablatura CSV guardada en %s (%d notas)", path, len(notes))
    return path


def export_tab(notes: list[TabNote], path: str | Path, meta: dict | None = None) -> Path:
    """Exporta según la extensión de ``path`` (.txt, .json o .csv).

    Parameters
    ----------
    notes : list[TabNote]
        Notas de la tablatura.
    path : str | Path
        Archivo de salida; su extensión (sin distinguir mayúsculas) decide el formato.
    meta : dict | None, optional
        Metadatos: van al JSON completos; en el TXT se usa ``meta["title"]``
        o un resumen de una línea como título; el CSV los ignora.

    Returns
    -------
    Path
        Ruta escrita.

    Raises
    ------
    ValueError
        Si la extensión no es ``.txt``, ``.json`` ni ``.csv``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return export_txt(notes, path, title=_title_from_meta(meta))
    if suffix == ".json":
        return export_json(notes, path, meta=meta)
    if suffix == ".csv":
        return export_csv(notes, path)
    raise ValueError(f"Formato de tablatura no soportado: «{path.suffix}». Usa .txt, .json o .csv.")
