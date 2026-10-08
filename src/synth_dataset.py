"""Dataset sintético con ground truth. [CONTRATO: implementar]

Cada pieza se define a mano como una lista de posiciones (cuerda, traste,
duración en tiempos), tal como la tocaría un bajista; de ahí se derivan las
notas MIDI (el MIDI solo guarda el pitch, la posición es el ground truth extra).

Flujo de generación: posiciones → :class:`GTNote` → ``.mid`` (pretty_midi,
programa GM 33 "Electric Bass (finger)") → WAV con el CLI de ``fluidsynth`` y un
soundfont GM → MP3 con ffmpeg. Si fluidsynth o el soundfont no están
disponibles se usa un sintetizador Karplus-Strong (cuerda pulsada) en numpy.
Junto a cada ``<nombre>.mp3`` se escribe ``<nombre>.gt.json`` con el ground truth.

Piezas:

* ``cromatica``: escala cromática en primera posición E-0 … G-5.
* ``linea_simple``: línea I–IV–V en La con negras.
* ``riff_saltos``: riff con saltos de cuerda, notas repetidas y semicorcheas cortas.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import ProgressCallback

@dataclass
class GTNote:
    """Nota del ground truth.

    Attributes
    ----------
    onset_s, offset_s : float
        Inicio y fin de la nota en segundos (ya incluye el silencio inicial).
    midi : int
        Número MIDI.
    string : str
        Cuerda (``"E"``, ``"A"``, ``"D"``, ``"G"``).
    fret : int
        Traste.
    """

    onset_s: float
    offset_s: float
    midi: int
    string: str
    fret: int

    @property
    def label(self) -> str:
        """Etiqueta ``"A-0"``."""
        return f"{self.string}-{self.fret}"


@dataclass
class DatasetItem:
    """Archivos generados para una pieza."""

    name: str
    mp3_path: Path
    gt_path: Path
    midi_path: Path
    method: str  # "fluidsynth" o "karplus-strong"


#: Definición de las piezas: nombre → lista de (cuerda, traste, duración_en_tiempos).
PIECES: dict[str, list[tuple[str, int, float]]] = {}

#: Silencio inicial añadido antes de la primera nota (s), para que el primer onset sea detectable.
LEAD_IN_S: float = 0.5


def piece_notes(name: str, bpm: float = 100.0, legato: float = 0.9) -> list[GTNote]:
    """Convierte la pieza ``name`` en notas con tiempos absolutos.

    Cada nota dura ``legato`` × su duración nominal (deja un pequeño hueco
    entre notas para que el re-ataque de notas repetidas sea audible).
    """
    raise NotImplementedError


def write_midi(notes: list[GTNote], path: str | Path, program: int = 33, velocity: int = 100) -> Path:
    """Escribe un archivo MIDI de un solo instrumento con ``notes``."""
    raise NotImplementedError


def find_soundfont() -> Path | None:
    """Busca un soundfont GM (.sf2) en ubicaciones habituales y en la variable
    de entorno ``SMARTUNER_SOUNDFONT``. None si no hay."""
    raise NotImplementedError


def fluidsynth_available() -> bool:
    """True si el ejecutable ``fluidsynth`` y un soundfont están disponibles."""
    raise NotImplementedError


def render_fluidsynth(midi_path: str | Path, wav_path: str | Path, sr: int = 22050, soundfont: str | Path | None = None) -> Path:
    """Renderiza el MIDI a WAV con el CLI de fluidsynth. Lanza RuntimeError si falla."""
    raise NotImplementedError


def render_karplus_strong(notes: list[GTNote], sr: int = 22050, seed: int = 0) -> np.ndarray:
    """Sintetiza las notas con el algoritmo Karplus-Strong (respaldo sin fluidsynth).

    Devuelve la señal mono float32 normalizada.
    """
    raise NotImplementedError


def save_ground_truth(notes: list[GTNote], path: str | Path, meta: dict | None = None) -> Path:
    """Guarda el ground truth en JSON: ``{"meta": {...}, "notes": [{onset_s, offset_s, midi, string, fret}, ...]}``."""
    raise NotImplementedError


def load_ground_truth(path: str | Path) -> list[GTNote]:
    """Lee un archivo ``.gt.json``."""
    raise NotImplementedError


def gt_path_for(audio_path: str | Path) -> Path:
    """Ruta del ground truth asociado a un audio: ``<carpeta>/<nombre>.gt.json``."""
    p = Path(audio_path)
    return p.with_name(p.stem + ".gt.json")


def generate_piece(
    name: str,
    out_dir: str | Path,
    bpm: float = 100.0,
    sr: int = 22050,
    method: str = "auto",
) -> DatasetItem:
    """Genera MIDI, MP3 y ground truth de una pieza.

    ``method``: ``"auto"`` (fluidsynth si está disponible, si no Karplus-Strong),
    ``"fluidsynth"`` o ``"karplus-strong"``.
    """
    raise NotImplementedError


def generate_dataset(
    out_dir: str | Path | None = None,
    bpm: float = 100.0,
    sr: int = 22050,
    method: str = "auto",
    pieces: list[str] | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> list[DatasetItem]:
    """Genera todas las piezas (o las de ``pieces``) en ``out_dir`` (por defecto ``data/synthetic``)."""
    raise NotImplementedError
