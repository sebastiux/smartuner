"""Etapa 8 — Tablatura.

Papel en el pipeline
--------------------
Es la salida final del sistema. Cada segmento conservado fue un problema
bandit independiente y su agente recomendó un brazo (cuerda, traste); este
módulo empareja cada segmento con su brazo (:func:`notes_from_choices`), lo
dibuja como una tablatura de bajo de 4 líneas (G, D, A, E de arriba abajo)
con marcas de tiempo (:func:`render_ascii`) y la exporta a TXT, JSON, CSV y
MIDI (:func:`export_tab`).

Escuchar la tablatura
---------------------
Para comprobar DE OÍDO si la transcripción es correcta, la tablatura se
convierte en un MIDI (:func:`notes_to_midi`, GM 33 = bajo eléctrico con
dedos, tiempos absolutos de cada segmento), se sintetiza
(:func:`render_notes`: fluidsynth + soundfont si están instalados, si no
Karplus-Strong en numpy) y se combina con el audio original
(:func:`playback_mix`) para compararlos::

    list[TabNote] ──► render_notes() ──► síntesis (mono, MISMOS tiempos que el original)
                                              │
    audio original (y_raw) ──────────────────┼──► playback_mix(modo)
                                              ▼
            "midi"      solo la síntesis           (¿suena como el bajo?)
            "original"  solo el original           (referencia)
            "mezcla"    ambos sumados, misma sonoridad (los errores de pitch «chocan»)
            "estereo"   original a la IZQUIERDA, síntesis a la DERECHA (A/B con auriculares)

Como la síntesis conserva los tiempos absolutos de los segmentos, en la
mezcla cada nota transcrita suena a la vez que la real: un pitch
equivocado se oye como una disonancia (batido) y un onset desplazado,
como un eco. La síntesis solo depende del pitch (``midi``) y de los
tiempos: dos posiciones con la misma nota (A-0 y E-5) suenan igual, así
que de oído se juzga el pitch y el ritmo, no la elección de cuerda.

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
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import STRING_ORDER
from src.environment import Arm
from src.pitch import midi_to_name
from src.segmentation import Segment
from src.synth_dataset import (
    BASS_PROGRAM,
    METHODS,
    OUTPUT_PEAK,
    TAIL_S,
    GTNote,
    _finish_audio,
    _load_wav_mono,
    _resolve_method,
    render_fluidsynth,
    render_karplus_strong,
)

if TYPE_CHECKING:  # solo para anotaciones: pretty_midi se importa de forma diferida
    import pretty_midi

logger = logging.getLogger(__name__)

#: Cuerdas en el orden en que se imprimen (de arriba abajo): la más aguda arriba.
TAB_STRINGS: tuple[str, ...] = tuple(reversed(STRING_ORDER))  # ("G", "D", "A", "E")

#: Etiqueta de la línea de tiempos.
TIME_LABEL: str = "t(s)"

#: Decimales de las marcas de tiempo (s).
TIME_DECIMALS: int = 2

#: Columnas del CSV exportado (en este orden).
CSV_COLUMNS: tuple[str, ...] = ("position", "start_s", "end_s", "string", "fret", "midi", "note", "f0_hz")

#: Extensiones que :func:`export_tab` guarda como archivo MIDI.
MIDI_SUFFIXES: tuple[str, ...] = (".mid", ".midi")

#: Programa General MIDI (numeración 0–127) del MIDI exportado: 33 = "Electric Bass (finger)".
MIDI_PROGRAM: int = BASS_PROGRAM

#: Ticks por negra del MIDI. Un archivo MIDI guarda los tiempos en ticks, no en
#: segundos: a 120 bpm, 960 ticks/negra dan 0.52 ms por tick, así que el
#: redondeo a ticks es inaudible y la síntesis no se desalinea del original
#: (con los 220 ticks por defecto de pretty_midi el paso sería de 2.3 ms).
MIDI_RESOLUTION: int = 960

#: Tempo (bpm) guardado en el MIDI. Es solo la escala de los ticks: los
#: tiempos de las notas ya están en segundos y no dependen de él.
MIDI_TEMPO_BPM: float = 120.0

#: Duración mínima (s) de una nota en el MIDI: una nota de duración 0 la
#: descartan algunos lectores y fluidsynth no la haría sonar.
MIN_MIDI_NOTE_S: float = 0.01

#: Métodos de síntesis de :func:`render_notes` (los mismos que el dataset sintético).
SYNTH_METHODS: tuple[str, ...] = METHODS

#: Modos de :func:`playback_mix`, en el orden en que se ofrecen en la GUI.
PLAYBACK_MODES: tuple[str, ...] = ("midi", "original", "mezcla", "estereo")

#: Nombre legible de cada modo de reproducción (botones y menús de la GUI).
PLAYBACK_MODE_LABELS: dict[str, str] = {
    "midi": "MIDI",
    "original": "Original",
    "mezcla": "Mezcla",
    "estereo": "A/B estéreo",
}

#: Pico (escala completa = 1) al que :func:`playback_mix` normaliza la salida:
#: deja ~1 dB de margen para que el conversor de la tarjeta no sature.
PLAYBACK_PEAK: float = OUTPUT_PEAK


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
    """Exporta según la extensión de ``path`` (.txt, .json, .csv o .mid).

    Parameters
    ----------
    notes : list[TabNote]
        Notas de la tablatura.
    path : str | Path
        Archivo de salida; su extensión (sin distinguir mayúsculas) decide el
        formato. ``.mid`` (o ``.midi``) guarda el MIDI de :func:`export_midi`.
    meta : dict | None, optional
        Metadatos: van al JSON completos; en el TXT se usa ``meta["title"]``
        o un resumen de una línea como título; el CSV y el MIDI los ignoran.

    Returns
    -------
    Path
        Ruta escrita.

    Raises
    ------
    ValueError
        Si la extensión no es ``.txt``, ``.json``, ``.csv`` ni ``.mid``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return export_txt(notes, path, title=_title_from_meta(meta))
    if suffix == ".json":
        return export_json(notes, path, meta=meta)
    if suffix == ".csv":
        return export_csv(notes, path)
    if suffix in MIDI_SUFFIXES:
        return export_midi(notes, path)
    raise ValueError(f"Formato de tablatura no soportado: «{path.suffix}». Usa .txt, .json, .csv o .mid.")


# ---------------------------------------------------------------------------
# MIDI: la tablatura como partitura que se puede escuchar
# ---------------------------------------------------------------------------


def notes_to_midi(notes: Sequence[TabNote], program: int = MIDI_PROGRAM, velocity: int = 100) -> pretty_midi.PrettyMIDI:
    """Convierte la tablatura en un MIDI de un solo instrumento (tiempos absolutos en s).

    Cada :class:`TabNote` se vuelve una nota MIDI con su pitch (``midi``) que
    empieza en ``start_s`` y termina en ``end_s``: los MISMOS instantes del
    segmento en el audio original, así que la síntesis queda alineada con él.
    El MIDI solo guarda pitch y tiempos; la posición (cuerda, traste) se
    añade como letra (*lyric*) en el inicio de cada nota (``"A-0"``), que
    muchos reproductores MIDI muestran sincronizada con la música.

    Parameters
    ----------
    notes : Sequence[TabNote]
        Notas de la tablatura (en cualquier orden; se escriben ordenadas por inicio).
    program : int, optional
        Programa General MIDI en numeración 0–127 (33 = Electric Bass (finger)).
    velocity : int, optional
        Intensidad 1–127 de todas las notas (la tablatura no estima dinámica).

    Returns
    -------
    pretty_midi.PrettyMIDI
        Objeto en memoria (guárdalo con ``pm.write(ruta)`` o :func:`export_midi`)
        con :data:`MIDI_RESOLUTION` ticks por negra y un instrumento.

    Raises
    ------
    ValueError
        Si ``program``, ``velocity`` o el ``midi`` de alguna nota están fuera de rango.

    Examples
    --------
    >>> pm = notes_to_midi([TabNote(0, 0.5, 0.8, "A", 0, 33), TabNote(1, 0.8, 1.1, "D", 2, 40)])
    >>> [(n.pitch, round(n.start, 3), round(n.end, 3)) for n in pm.instruments[0].notes]
    [(33, 0.5, 0.8), (40, 0.8, 1.1)]
    >>> pm.instruments[0].program, [lyric.text for lyric in pm.lyrics]
    (33, ['A-0', 'D-2'])
    """
    import pretty_midi  # import diferido: solo se necesita al exportar o escuchar

    if not 0 <= int(program) <= 127:
        raise ValueError(f"Programa MIDI fuera de rango: {program} (debe estar entre 0 y 127).")
    if not 1 <= int(velocity) <= 127:
        raise ValueError(f"Velocidad MIDI fuera de rango: {velocity} (debe estar entre 1 y 127).")
    pm = pretty_midi.PrettyMIDI(resolution=MIDI_RESOLUTION, initial_tempo=MIDI_TEMPO_BPM)
    instrument = pretty_midi.Instrument(program=int(program), name=pretty_midi.program_to_instrument_name(int(program)))
    for note in sorted(notes, key=lambda n: (n.start_s, n.position)):
        if not 0 <= int(note.midi) <= 127:
            raise ValueError(f"Nota {note.position}: MIDI {note.midi} fuera de rango (0–127).")
        start = max(0.0, float(note.start_s))
        end = max(float(note.end_s), start + MIN_MIDI_NOTE_S)
        instrument.notes.append(pretty_midi.Note(velocity=int(velocity), pitch=int(note.midi), start=start, end=end))
        pm.lyrics.append(pretty_midi.Lyric(text=note.label, time=start))
    pm.instruments.append(instrument)
    logger.debug("MIDI en memoria: %d notas, programa %d, %d ticks/negra", len(instrument.notes), program,
                 MIDI_RESOLUTION)
    return pm


def export_midi(notes: Sequence[TabNote], path: str | Path, program: int = MIDI_PROGRAM, velocity: int = 100) -> Path:
    """Guarda la tablatura como archivo MIDI estándar (``.mid``).

    El archivo se abre con cualquier reproductor o editor MIDI (MuseScore,
    Guitar Pro, un DAW...) para escuchar la transcripción o seguir editándola.

    Parameters
    ----------
    notes : Sequence[TabNote]
        Notas de la tablatura.
    path : str | Path
        Archivo de salida (se crean las carpetas intermedias).
    program : int, optional
        Programa General MIDI 0–127 (33 = Electric Bass (finger)).
    velocity : int, optional
        Intensidad 1–127 de todas las notas.

    Returns
    -------
    Path
        Ruta escrita.

    Examples
    --------
    >>> export_midi([TabNote(0, 0.5, 0.8, "A", 0, 33)], "tab.mid")  # doctest: +SKIP
    PosixPath('tab.mid')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notes_to_midi(notes, program=program, velocity=velocity).write(str(path))
    logger.info("Tablatura MIDI guardada en %s (%d notas, programa GM %d)", path, len(notes), program)
    return path


# ---------------------------------------------------------------------------
# Síntesis y mezcla para escuchar la tablatura
# ---------------------------------------------------------------------------


def resolve_synth_method(method: str = "auto") -> str:
    """Traduce el método de síntesis pedido al que se usará en esta máquina.

    Parameters
    ----------
    method : str, optional
        ``"auto"`` (fluidsynth si hay ejecutable y soundfont GM, si no
        Karplus-Strong), ``"fluidsynth"`` o ``"karplus-strong"``.

    Returns
    -------
    str
        ``"fluidsynth"`` o ``"karplus-strong"``.

    Raises
    ------
    ValueError
        Si el método no está en :data:`SYNTH_METHODS`.
    RuntimeError
        Si se pide ``"fluidsynth"`` y no está disponible.

    Examples
    --------
    >>> resolve_synth_method("karplus-strong")
    'karplus-strong'
    """
    return _resolve_method(method)


def _as_gt_notes(notes: Sequence[TabNote]) -> list[GTNote]:
    """Adapta las notas de la tablatura al tipo que usan los sintetizadores del dataset.

    Karplus-Strong (:func:`src.synth_dataset.render_karplus_strong`) solo
    necesita pitch y tiempos; :class:`GTNote` los guarda con otros nombres
    (``onset_s``/``offset_s``). Se fuerza una duración mínima para que ninguna
    nota quede sin sonar.
    """
    return [
        GTNote(onset_s=max(0.0, float(n.start_s)),
               offset_s=max(float(n.end_s), max(0.0, float(n.start_s)) + MIN_MIDI_NOTE_S),
               midi=int(n.midi), string=n.string, fret=int(n.fret))
        for n in notes
    ]


def _render_fluidsynth_notes(notes: Sequence[TabNote], sr: int, n_samples: int) -> np.ndarray:
    """MIDI temporal → fluidsynth → WAV temporal → mono recortado a ``n_samples`` muestras."""
    with tempfile.TemporaryDirectory(prefix="smartuner_tab_") as tmp:
        midi_path = Path(tmp) / "tablatura.mid"
        notes_to_midi(notes).write(str(midi_path))
        wav_path = render_fluidsynth(midi_path, Path(tmp) / "tablatura.wav", sr=sr)
        y = _load_wav_mono(wav_path, sr)
    # fluidsynth deja varios segundos de cola (release + silencio): se ajusta a la
    # duración esperada con un fundido de salida y se normaliza el pico.
    return _finish_audio(y, n_samples, sr)


def render_notes(
    notes: Sequence[TabNote],
    sr: int = 22050,
    method: str = "auto",
    seed: int = 0,
    tail_s: float = TAIL_S,
) -> np.ndarray:
    """Sintetiza la tablatura como audio de bajo, con los tiempos absolutos del original.

    La muestra ``k`` del resultado corresponde al instante ``k / sr`` segundos
    del audio original: la primera nota NO se desplaza al inicio, así que la
    síntesis y el original se pueden sumar o poner uno en cada canal sin
    alinearlos (ver :func:`playback_mix`).

    Métodos
    -------
    * ``"fluidsynth"``: el MIDI de :func:`notes_to_midi` se renderiza con un
      soundfont General MIDI (programa 33, bajo eléctrico con dedos). Suena
      como un bajo de verdad (muestras grabadas).
    * ``"karplus-strong"``: cada nota es una cuerda pulsada simulada con una
      línea de retardo realimentada (ver
      :func:`src.synth_dataset.render_karplus_strong`). No necesita nada
      instalado y es rápido (~1 s para 1000 notas / 6 min de audio).
    * ``"auto"``: fluidsynth si está disponible; si no (o si falla),
      Karplus-Strong.

    Parameters
    ----------
    notes : Sequence[TabNote]
        Notas de la tablatura (tiempos en s).
    sr : int, optional
        Frecuencia de muestreo (Hz); usa la del audio original para poder mezclarlos.
    method : str, optional
        ``"auto"``, ``"fluidsynth"`` o ``"karplus-strong"``.
    seed : int, optional
        Semilla del ruido de excitación de Karplus-Strong (resultado reproducible).
    tail_s : float, optional
        Cola después del final de la última nota (s), para que su release no se corte.

    Returns
    -------
    np.ndarray
        Señal mono ``float32`` de ``round((fin de la última nota + tail_s)·sr)``
        muestras con pico :data:`PLAYBACK_PEAK` (``round(tail_s·sr)`` ceros si
        no hay notas).

    Raises
    ------
    ValueError
        Método desconocido o ``sr`` no positiva.
    RuntimeError
        Si se pide ``"fluidsynth"`` y no está disponible o falla.

    Examples
    --------
    >>> y = render_notes([TabNote(0, 0.5, 0.8, "A", 0, 33)], sr=8000, method="karplus-strong")
    >>> y.shape, y.dtype, bool(abs(y[:4000]).max() == 0.0)
    ((10400,), dtype('float32'), True)
    """
    if int(sr) <= 0:
        raise ValueError(f"La frecuencia de muestreo debe ser positiva (recibido {sr}).")
    sr = int(sr)
    used = _resolve_method(method)
    end_s = max((max(float(n.end_s), float(n.start_s) + MIN_MIDI_NOTE_S) for n in notes), default=0.0)
    n_samples = int(round((end_s + tail_s) * sr))
    if not notes:
        return np.zeros(n_samples, dtype=np.float32)

    t0 = time.perf_counter()
    y: np.ndarray | None = None
    if used == "fluidsynth":
        try:
            y = _render_fluidsynth_notes(notes, sr, n_samples)
        except RuntimeError as exc:
            if method != "auto":
                raise
            # En modo automático un fallo de fluidsynth no impide escuchar: se usa el respaldo.
            logger.warning("fluidsynth falló (%s); se sintetiza con Karplus-Strong.", exc)
            used = "karplus-strong"
    if y is None:
        y = render_karplus_strong(_as_gt_notes(notes), sr=sr, seed=seed, tail_s=tail_s)
        y = _pad_to(np.asarray(y, dtype=np.float32), n_samples)  # misma longitud que con fluidsynth
    logger.info("Tablatura sintetizada con %s: %d notas, %.1f s de audio en %.2f s",
                used, len(notes), y.size / sr, time.perf_counter() - t0)
    return y.astype(np.float32, copy=False)


def _as_mono(y: np.ndarray) -> np.ndarray:
    """Señal mono ``float32``: promedia los canales si viene con forma ``(n, canales)``."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1, dtype=np.float32)
    elif y.ndim != 1:
        raise ValueError(f"Se esperaba una señal 1-D o (n, canales); recibida con forma {y.shape}.")
    return y


def _pad_to(y: np.ndarray, n: int) -> np.ndarray:
    """Rellena con ceros al final hasta ``n`` muestras (los tiempos absolutos no cambian)."""
    if y.size >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - y.size, dtype=y.dtype)])


def _rms(y: np.ndarray) -> float:
    """Valor eficaz (RMS) de la señal, en escala lineal (1 = escala completa)."""
    # La suma se acumula en float64 para no perder precisión con millones de muestras.
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64))) if y.size else 0.0


def _peak(y: np.ndarray) -> float:
    """Pico absoluto max|y| sin crear el array ``|y|`` (importa con 6 min de audio)."""
    return float(max(np.max(y), -np.min(y))) if y.size else 0.0


def _normalize_peak(y: np.ndarray, peak: float = PLAYBACK_PEAK) -> np.ndarray:
    """Escala EN SITIO la señal float32 (mono o estéreo) para que su pico sea ``peak`` (silencio: sin cambios)."""
    current = _peak(y)
    if current > 0.0:
        y *= np.float32(peak / current)
    return y


def playback_mix(
    original: np.ndarray | None,
    synth: np.ndarray,
    mode: str,
    synth_gain_db: float = 0.0,
) -> np.ndarray:
    """Combina el audio original y la síntesis de la tablatura para escucharlos.

    Ambas señales deben tener la misma frecuencia de muestreo y empezar en el
    mismo instante (como devuelve :func:`render_notes`). Si hay original, las
    dos se rellenan con ceros hasta la más larga, de modo que TODOS los modos
    duran lo mismo y un instante ``t`` es la muestra ``t·sr`` en cualquiera
    de ellos (la GUI puede cambiar de modo sin perder la posición).

    Modos
    -----
    * ``"midi"``: solo la síntesis.
    * ``"original"``: solo el original.
    * ``"mezcla"``: ambos sumados. Antes se **iguala la sonoridad**: cada
      señal se divide por su RMS (valor eficaz, √(media de x²)), de modo que
      el original y la síntesis pesen lo mismo aunque uno esté grabado más
      bajo. Si una nota transcrita es incorrecta, las dos frecuencias
      cercanas producen un batido (|f₁ − f₂| pulsos por segundo) muy audible.
    * ``"estereo"``: original en el canal izquierdo y síntesis en el derecho
      (salida ``(n, 2)``), también con la sonoridad igualada. Con auriculares
      se comparan A/B sin que una señal tape a la otra.

    Al final la salida se normaliza a un pico de :data:`PLAYBACK_PEAK`
    (en estéreo, el mismo factor para los dos canales, para no romper el
    equilibrio). Todos los modos quedan así a un nivel parecido y se puede
    cambiar de uno a otro sin saltos de volumen bruscos.

    Parameters
    ----------
    original : np.ndarray | None
        Audio original (mono ``(n,)`` o ``(n, canales)``, que se promedia a
        mono). Puede ser None solo en el modo ``"midi"``.
    synth : np.ndarray
        Síntesis de la tablatura (:func:`render_notes`).
    mode : str
        Uno de :data:`PLAYBACK_MODES`.
    synth_gain_db : float, optional
        Nivel de la síntesis respecto al original en ``"mezcla"`` y
        ``"estereo"`` (dB; 0 = misma sonoridad, −6 = la síntesis a la mitad).

    Returns
    -------
    np.ndarray
        ``float32``; forma ``(n,)`` o ``(n, 2)`` en ``"estereo"``, con
        ``n = max(len(original), len(synth))`` (``len(synth)`` sin original).

    Raises
    ------
    ValueError
        Modo desconocido, o modo que necesita el original sin él.

    Examples
    --------
    >>> orig, synth = np.ones(4, dtype=np.float32), np.full(6, 0.5, dtype=np.float32)
    >>> playback_mix(orig, synth, "estereo").shape   # original rellenado con ceros hasta 6 muestras
    (6, 2)
    >>> mix = playback_mix(orig, synth, "mezcla")
    >>> mix.shape, round(float(np.abs(mix).max()), 3)
    ((6,), 0.9)
    >>> playback_mix(None, synth, "midi").shape
    (6,)
    """
    if mode not in PLAYBACK_MODES:
        raise ValueError(f"Modo de reproducción desconocido: «{mode}». Opciones: {', '.join(PLAYBACK_MODES)}.")
    synth = _as_mono(synth)
    if original is None:
        if mode != "midi":
            raise ValueError(f"El modo «{mode}» necesita el audio original y no se recibió.")
        return _normalize_peak(synth.copy())
    original = _as_mono(original)
    n = max(original.size, synth.size)
    original, synth = _pad_to(original, n), _pad_to(synth, n)
    if mode == "midi":
        return _normalize_peak(synth.copy())
    if mode == "original":
        return _normalize_peak(original.copy())

    # Igualación de sonoridad: x · g con g = 1/RMS(x) deja las dos señales con RMS = 1
    # (la síntesis, además, con la ganancia relativa 10^(dB/20)).
    rms_o, rms_s = _rms(original), _rms(synth)
    g_o = 1.0 / rms_o if rms_o > 0 else 1.0
    g_s = 10.0 ** (synth_gain_db / 20.0) / rms_s if rms_s > 0 else 1.0
    logger.debug("Mezcla «%s»: %d muestras, RMS original %.4f, RMS síntesis %.4f", mode, n, rms_o, rms_s)
    if mode == "mezcla":
        mixed = original * np.float32(g_o)
        mixed += synth * np.float32(g_s)
        return _normalize_peak(mixed)
    # Estéreo: el pico conjunto es el mayor de los dos canales, así que se calcula
    # un único factor k y se escribe cada canal directamente en su columna (sin
    # copias intermedias: con 6 min de audio, np.stack + normalizar tardaba ~1 s).
    joint_peak = max(g_o * _peak(original), g_s * _peak(synth))
    k = PLAYBACK_PEAK / joint_peak if joint_peak > 0 else 1.0
    out = np.empty((n, 2), dtype=np.float32)
    np.multiply(original, np.float32(g_o * k), out=out[:, 0])
    np.multiply(synth, np.float32(g_s * k), out=out[:, 1])
    return out
