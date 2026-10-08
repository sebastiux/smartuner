"""Dataset sintético con ground truth (cuerda, traste) conocido.

Papel en el pipeline
--------------------
Para evaluar a los agentes bandit hace falta saber, para cada nota, **qué
posición del diapasón es la correcta**. Una grabación real no trae esa
información, así que este módulo fabrica pistas de bajo cuya tablatura se
conoce de antemano. Con ellas :mod:`src.experiments` mide la precisión de
pitch y de posición de cada algoritmo, y :mod:`src.pipeline` carga
automáticamente el ``.gt.json`` que acompaña a cada MP3.

Cada pieza se define a mano como una lista de posiciones (cuerda, traste,
duración en tiempos), tal como la tocaría un bajista; de ahí se derivan las
notas MIDI (el MIDI solo guarda el pitch, la posición es el ground truth extra).

Flujo de generación
-------------------
::

    PIECES[nombre] ──► piece_notes() ──► list[GTNote] ──┬──► <nombre>.gt.json
    (cuerda, traste,     tiempos en s,                  │
     duración en tiempos)  MIDI = cuerda + traste       ├──► <nombre>.mid  (pretty_midi, GM 33)
                                                        │         │
                                                        │         ▼
                                                        │   fluidsynth + soundfont GM ─┐
                                                        │                              ├─► WAV temporal
                                                        └─► Karplus-Strong (numpy) ────┘   (fuera del dataset)
                                                                                          │
                                                                    mono, recorte, normalización
                                                                                          ▼
                                                                               <nombre>.mp3 (ffmpeg)

1. Las posiciones se convierten en :class:`GTNote` con tiempos absolutos.
2. Se escribe un ``.mid`` de un solo instrumento (programa GM 33 "Electric
   Bass (finger)").
3. El MIDI se renderiza a WAV con el CLI de ``fluidsynth`` y un soundfont GM
   (sin reverb ni chorus, para un bajo limpio). Si fluidsynth o el soundfont
   no están disponibles se usa un sintetizador **Karplus-Strong** (cuerda
   pulsada) escrito en numpy.
4. El audio se pasa a mono, se recorta a ``último offset + TAIL_S`` y se
   normaliza; después se codifica a MP3 con ffmpeg (:mod:`src.io_audio`).
   El WAV intermedio vive en una carpeta temporal: en ``data/synthetic`` solo
   quedan ``.mid``, ``.mp3`` y ``.gt.json``.

Piezas
------
* ``cromatica``: escala cromática en primera posición E-0 … G-5.
* ``linea_simple``: línea I–IV–V en La con negras.
* ``riff_saltos``: riff con saltos de cuerda, notas repetidas y semicorcheas cortas.

(Ver :data:`PIECES` para qué pone a prueba cada una.)

Ejemplo
-------
>>> notes = piece_notes("linea_simple", bpm=100.0)
>>> notes[0].label, notes[0].midi, notes[0].onset_s
('A-0', 33, 0.5)
>>> round(notes[1].onset_s, 3)  # una negra a 100 bpm dura 60/100 = 0.6 s
1.1
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DATA_DIR,
    OPEN_STRING_HZ,
    OPEN_STRING_MIDI,
    STRING_ORDER,
    CancelledError,
    ProgressCallback,
)
from src.pitch import midi_to_hz

logger = logging.getLogger(__name__)


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

    Examples
    --------
    >>> n = GTNote(onset_s=0.5, offset_s=1.04, midi=33, string="A", fret=0)
    >>> n.label
    'A-0'
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
    """Archivos generados para una pieza.

    Attributes
    ----------
    name : str
        Nombre de la pieza (clave de :data:`PIECES`).
    mp3_path : Path
        Audio codificado en MP3 (lo que analiza el pipeline).
    gt_path : Path
        Ground truth ``<nombre>.gt.json``.
    midi_path : Path
        Archivo MIDI del que se renderizó el audio.
    method : str
        Sintetizador usado: ``"fluidsynth"`` o ``"karplus-strong"``.
    """

    name: str
    mp3_path: Path
    gt_path: Path
    midi_path: Path
    method: str  # "fluidsynth" o "karplus-strong"


# ---------------------------------------------------------------------------
# Definición de las piezas
# ---------------------------------------------------------------------------

#: Definición de las piezas: nombre → lista de (cuerda, traste, duración_en_tiempos).
PIECES: dict[str, list[tuple[str, int, float]]] = {
    # Corcheas (0.5 tiempos) en primera posición: cada cuerda desde el aire
    # hasta el traste 4 (la G hasta el 5). Recorre E1 (41.2 Hz) … C3 (130.8 Hz).
    "cromatica": (
        [("E", fret, 0.5) for fret in range(0, 5)]
        + [("A", fret, 0.5) for fret in range(0, 5)]
        + [("D", fret, 0.5) for fret in range(0, 5)]
        + [("G", fret, 0.5) for fret in range(0, 6)]
    ),
    # I–IV–V en La con negras (1 tiempo): raíz, 3.ª mayor, 5.ª, 3.ª mayor.
    "linea_simple": [
        ("A", 0, 1.0), ("A", 4, 1.0), ("D", 2, 1.0), ("A", 4, 1.0),   # A  (I)
        ("D", 0, 1.0), ("D", 4, 1.0), ("G", 2, 1.0), ("D", 4, 1.0),   # D  (IV)
        ("E", 0, 1.0), ("E", 4, 1.0), ("A", 2, 1.0), ("E", 4, 1.0),   # E  (V)
        ("A", 0, 1.0), ("A", 4, 1.0), ("D", 2, 1.0), ("A", 0, 2.0),   # A  (I) y final largo
    ],
    # Riff con saltos de cuerda, notas repetidas y semicorcheas (0.25 tiempos).
    "riff_saltos": [
        ("E", 0, 0.5), ("E", 0, 0.5), ("D", 2, 0.5), ("E", 0, 0.5),
        ("E", 3, 0.25), ("E", 3, 0.25), ("A", 5, 0.5), ("G", 2, 0.5),
        ("A", 5, 0.5), ("E", 5, 0.25), ("E", 5, 0.25), ("E", 5, 0.5),
        ("D", 7, 0.5), ("E", 5, 0.5), ("A", 7, 0.25), ("A", 7, 0.25),
        ("G", 4, 0.5), ("D", 5, 0.5), ("A", 3, 0.5), ("E", 3, 1.0),
    ],
}
"""Piezas del dataset y qué pone a prueba cada una (tiempos a 100 bpm).

``cromatica`` — 21 corcheas (0.30 s) E-0 … G-5, todas las notas de E1 a C3.
    * **Pitch en todo el registro grave**: incluye E1 = 41.2 Hz, donde la
      fundamental es débil y pYIN/la recompensa armónica sufren más.
    * **Errores de semitono**: notas vecinas a un semitono, así que la poda
      ±k y la tolerancia de afinación deben ser finas.
    * **Ambigüedad de posición con cuerda al aire**: A-0 = E-5, D-0 = A-5,
      G-0 = D-5 (mismo pitch). El ground truth usa la primera posición; el
      bandit solo puede acertar gracias a la penalización λ·|Δtraste|/12.
    * **Onsets regulares**: ritmo constante, fácil de segmentar.

``linea_simple`` — 16 negras (0.60 s) I–IV–V en La + La final de 2 tiempos.
    * **Caso fácil de referencia**: notas largas (muchos frames por segmento
      para muestrear recompensas) y separadas.
    * **Tocabilidad**: la mano se queda en la zona de los trastes 0–4; cada
      nota tiene alternativas lejanas (A-4 = E-9, D-2 = A-7 = E-12, G-2 = D-7)
      que λ debe descartar.
    * **Saltos de cuerda moderados** (A → D → G) y una nota final larga.

``riff_saltos`` — 20 notas con corcheas (0.30 s) y semicorcheas (0.15 s).
    * **Notas repetidas** (E-0 E-0, E-3 E-3, E-5 E-5 E-5, A-7 A-7): el mismo
      pitch se re-ataca; el onset solo es detectable por el hueco que deja el
      ``legato`` < 1 y por el ruido del ataque.
    * **Semicorcheas cortas**: 0.15 s nominales (0.135 s sonando): pocos frames
      para pYIN y para la recompensa; ponen a prueba ``min_duration_s``.
    * **Saltos de cuerda** E → D, A → G, E → A y cambios de posición hasta el
      traste 7 (D-7, A-7), donde la posición previa de la mano importa.
    * **Posiciones ambiguas fuera de la primera posición**: A-5 (= D-0),
      A-7 (= D-2), D-5 (= G-0), D-7 (= G-2), E-5 (= A-0): el ground truth no
      siempre es la posición con el traste más bajo.
"""

#: Silencio inicial añadido antes de la primera nota (s), para que el primer onset sea detectable.
LEAD_IN_S: float = 0.5

#: Silencio/cola después del offset de la última nota (s): deja sonar el release.
TAIL_S: float = 0.5

#: Programa General MIDI del instrumento (numeración 0–127 de los archivos MIDI):
#: 33 = "Electric Bass (finger)" (34 en la numeración 1–128 de las tablas GM).
BASS_PROGRAM: int = 33

#: Ganancia de síntesis de fluidsynth (opción ``-g``). Por defecto fluidsynth usa 0.2.
FLUIDSYNTH_GAIN: float = 0.8

#: Tiempo máximo (s) que se deja correr a fluidsynth antes de abortar.
FLUIDSYNTH_TIMEOUT_S: float = 120.0

#: Pico al que se normaliza el audio antes de codificarlo (deja margen al MP3).
OUTPUT_PEAK: float = 0.9

#: Fundido de salida al recortar el render de fluidsynth (s), para evitar un clic.
FADE_OUT_S: float = 0.05

#: Factor de decaimiento ρ por vuelta del delay line de Karplus-Strong.
KS_DECAY: float = 0.996

#: Pendiente p del pasa-bajas aplicado al ruido inicial de Karplus-Strong: el
#: armónico h empieza con amplitud 1/hᵖ (p = 1 → −6 dB/octava, como un filtro
#: de primer orden; 0 = ruido blanco, sonido metálico).
KS_ROLLOFF: float = 1.0

#: Duración (s) del apagado de cada nota de Karplus-Strong después de su offset.
KS_RELEASE_S: float = 0.02

#: Variable de entorno con la ruta explícita a un soundfont ``.sf2``.
SOUNDFONT_ENV_VAR: str = "SMARTUNER_SOUNDFONT"

#: Variable de entorno opcional con la ruta explícita al ejecutable de fluidsynth.
FLUIDSYNTH_ENV_VAR: str = "SMARTUNER_FLUIDSYNTH"

#: Métodos de síntesis válidos para :func:`generate_piece`.
METHODS: tuple[str, ...] = ("auto", "fluidsynth", "karplus-strong")


# ---------------------------------------------------------------------------
# Piezas → notas con tiempos
# ---------------------------------------------------------------------------


def piece_notes(name: str, bpm: float = 100.0, legato: float = 0.9) -> list[GTNote]:
    """Convierte la pieza ``name`` en notas con tiempos absolutos.

    Cada nota dura ``legato`` × su duración nominal (deja un pequeño hueco
    entre notas para que el re-ataque de notas repetidas sea audible).

    Parameters
    ----------
    name : str
        Clave de :data:`PIECES`.
    bpm : float, optional
        Tempo en negras por minuto (100 por defecto). Una negra dura 60/bpm s.
    legato : float, optional
        Fracción (0, 1] de la duración nominal durante la que suena la nota.

    Returns
    -------
    list[GTNote]
        Notas en orden temporal. Para la nota i::

            onset_i    = LEAD_IN_S + Σ_{k<i} tiempos_k · 60/bpm
            duración_i = tiempos_i · 60/bpm                       (nominal)
            offset_i   = onset_i + legato · duración_i
            midi_i     = OPEN_STRING_MIDI[cuerda_i] + traste_i

    Raises
    ------
    ValueError
        Si la pieza no existe, ``bpm ≤ 0`` o ``legato`` no está en (0, 1].

    Examples
    --------
    >>> notes = piece_notes("cromatica", bpm=100.0)
    >>> len(notes), notes[0].label, notes[-1].label
    (21, 'E-0', 'G-5')
    >>> [n.midi for n in notes[:6]]  # E1 … A1: semitonos consecutivos
    [28, 29, 30, 31, 32, 33]
    >>> round(notes[1].onset_s - notes[0].onset_s, 6)  # corchea a 100 bpm
    0.3
    """
    if name not in PIECES:
        raise ValueError(f"Pieza desconocida: «{name}». Piezas disponibles: {', '.join(PIECES)}")
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"El tempo debe ser un número positivo y finito (bpm={bpm}).")
    if not 0.0 < legato <= 1.0:
        raise ValueError(f"legato debe estar en (0, 1] (legato={legato}).")

    beat_s = 60.0 / bpm  # duración de una negra en segundos
    notes: list[GTNote] = []
    beats_elapsed = 0.0
    for string, fret, beats in PIECES[name]:
        # Se acumulan TIEMPOS (no segundos) para no arrastrar errores de redondeo.
        onset = LEAD_IN_S + beats_elapsed * beat_s
        duration = beats * beat_s
        notes.append(GTNote(
            onset_s=onset,
            offset_s=onset + legato * duration,
            midi=OPEN_STRING_MIDI[string] + fret,
            string=string,
            fret=fret,
        ))
        beats_elapsed += beats
    return notes


def piece_duration_s(notes: list[GTNote], tail_s: float = TAIL_S) -> float:
    """Duración total de la pista generada: offset de la última nota + cola.

    Parameters
    ----------
    notes : list[GTNote]
        Notas de la pieza.
    tail_s : float, optional
        Cola después de la última nota (s).

    Returns
    -------
    float
        Duración en segundos (``tail_s`` si no hay notas).

    Examples
    --------
    >>> piece_duration_s([GTNote(0.5, 1.04, 33, "A", 0)], tail_s=0.5)
    1.54
    """
    last = max((n.offset_s for n in notes), default=0.0)
    return float(last + tail_s)


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------


def write_midi(
    notes: list[GTNote],
    path: str | Path,
    program: int = BASS_PROGRAM,
    velocity: int = 100,
    bpm: float = 100.0,
) -> Path:
    """Escribe un archivo MIDI de un solo instrumento con ``notes``.

    Parameters
    ----------
    notes : list[GTNote]
        Notas con tiempos absolutos (s). El MIDI solo guarda pitch y tiempos;
        la cuerda y el traste viven en el ``.gt.json``.
    path : str | Path
        Archivo ``.mid`` de salida (se crean las carpetas intermedias).
    program : int, optional
        Programa General MIDI en numeración 0–127 (33 = Electric Bass (finger)).
    velocity : int, optional
        Intensidad 1–127 de todas las notas.
    bpm : float, optional
        Tempo guardado en el archivo (solo metadato: los tiempos de las notas
        ya están en segundos y pretty_midi los convierte a ticks con él).

    Returns
    -------
    Path
        Ruta escrita.

    Examples
    --------
    >>> write_midi(piece_notes("cromatica"), "cromatica.mid")  # doctest: +SKIP
    PosixPath('cromatica.mid')
    """
    import pretty_midi  # import diferido: solo se necesita al generar el dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=int(program), name=pretty_midi.program_to_instrument_name(int(program)))
    for note in notes:
        instrument.notes.append(pretty_midi.Note(
            velocity=int(velocity), pitch=int(note.midi), start=float(note.onset_s), end=float(note.offset_s),
        ))
    pm.instruments.append(instrument)
    pm.write(str(path))
    logger.debug("MIDI escrito: %s (%d notas, programa %d)", path, len(notes), program)
    return path


# ---------------------------------------------------------------------------
# fluidsynth + soundfont
# ---------------------------------------------------------------------------


def _soundfont_candidates() -> list[str]:
    """Rutas o patrones glob donde se buscan soundfonts, en orden de preferencia."""
    home = Path.home()
    return [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",       # Debian/Ubuntu: paquete fluid-soundfont-gm
        "/usr/share/soundfonts/*.sf2",                 # Arch/Fedora
        "/usr/share/sounds/sf2/*.sf2",                 # otros soundfonts de Debian/Ubuntu
        str(DATA_DIR / "*.sf2"),                       # soundfont copiado en data/ del proyecto
        str(home / "Library/Audio/Sounds/Banks/*.sf2"),  # macOS (usuario)
        "/Library/Audio/Sounds/Banks/*.sf2",           # macOS (sistema)
        "/opt/homebrew/share/soundfonts/*.sf2",        # macOS con Homebrew (Apple Silicon)
        "/usr/local/share/soundfonts/*.sf2",           # macOS Homebrew (Intel) / Linux manual
        "C:/soundfonts/*.sf2",                         # Windows
        str(home / "soundfonts/*.sf2"),                # Windows/otros: carpeta del usuario
    ]


def find_soundfont() -> Path | None:
    """Busca un soundfont GM (.sf2) en ubicaciones habituales y en la variable
    de entorno ``SMARTUNER_SOUNDFONT``. None si no hay.

    Orden de búsqueda: la variable de entorno, ``FluidR3_GM.sf2`` de
    Debian/Ubuntu, ``/usr/share/soundfonts/*.sf2``, ``/usr/share/sounds/sf2/*.sf2``,
    ``data/*.sf2`` del proyecto y rutas típicas de macOS
    (``~/Library/Audio/Sounds/Banks``, Homebrew) y Windows (``C:/soundfonts``).

    Returns
    -------
    Path | None
        Primer ``.sf2`` existente, o ``None``.
    """
    explicit = os.environ.get(SOUNDFONT_ENV_VAR)
    if explicit:
        if Path(explicit).is_file():
            return Path(explicit)
        logger.warning("%s apunta a un archivo inexistente: %s (se buscará en otras rutas)", SOUNDFONT_ENV_VAR, explicit)
    for pattern in _soundfont_candidates():
        for match in sorted(glob.glob(pattern)):
            if Path(match).is_file():
                return Path(match)
    return None


def find_fluidsynth() -> str | None:
    """Localiza el ejecutable de fluidsynth (``SMARTUNER_FLUIDSYNTH`` o el PATH).

    Returns
    -------
    str | None
        Ruta al ejecutable, o ``None`` si no está instalado.
    """
    explicit = os.environ.get(FLUIDSYNTH_ENV_VAR)
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("fluidsynth")


def fluidsynth_available() -> bool:
    """Indica si se puede sintetizar con fluidsynth.

    Returns
    -------
    bool
        True si el ejecutable ``fluidsynth`` y un soundfont General MIDI
        están disponibles (:func:`find_fluidsynth`, :func:`find_soundfont`).
    """
    return find_fluidsynth() is not None and find_soundfont() is not None


def render_fluidsynth(
    midi_path: str | Path,
    wav_path: str | Path,
    sr: int = 22050,
    soundfont: str | Path | None = None,
    timeout_s: float = FLUIDSYNTH_TIMEOUT_S,
) -> Path:
    """Renderiza el MIDI a WAV con el CLI de fluidsynth. Lanza RuntimeError si falla.

    Ejecuta::

        fluidsynth -ni -R 0 -C 0 -g 0.8 -r <sr> -F <wav> <sf2> <mid>
                    ││  │    │    │      │       └─ escribe a archivo (render rápido, sin tarjeta de sonido)
                    ││  │    │    │      └─ frecuencia de muestreo en Hz
                    ││  │    │    └─ ganancia (por defecto 0.2 suena muy bajo)
                    ││  │    └─ sin chorus
                    ││  └─ sin reverb (bajo "seco": la cola de reverb ensuciaría los onsets)
                    │└─ sin shell interactivo
                    └─ sin entrada MIDI en vivo

    Parameters
    ----------
    midi_path : str | Path
        Archivo ``.mid`` de entrada.
    wav_path : str | Path
        WAV de salida (estéreo, 16 bits, a ``sr`` Hz).
    sr : int, optional
        Frecuencia de muestreo en Hz.
    soundfont : str | Path | None, optional
        Soundfont ``.sf2``; ``None`` usa :func:`find_soundfont`.
    timeout_s : float, optional
        Tiempo máximo de ejecución (s).

    Returns
    -------
    Path
        Ruta del WAV escrito.

    Raises
    ------
    RuntimeError
        Si falta fluidsynth, el soundfont o el MIDI, si se agota el tiempo o
        si fluidsynth termina con error o no produce audio.
    """
    midi_path, wav_path = Path(midi_path), Path(wav_path)
    exe = find_fluidsynth()
    if exe is None:
        raise RuntimeError(
            "No se encontró el ejecutable de fluidsynth. Instálalo (Linux: sudo apt install fluidsynth; "
            f"macOS: brew install fluid-synth; Windows: choco install fluidsynth) o indica su ruta en {FLUIDSYNTH_ENV_VAR}."
        )
    sf2 = Path(soundfont) if soundfont is not None else find_soundfont()
    if sf2 is None or not sf2.is_file():
        raise RuntimeError(
            f"No se encontró un soundfont General MIDI (.sf2){f' en {sf2}' if sf2 else ''}. "
            f"Instala uno (p. ej. FluidR3_GM.sf2), cópialo en data/ o indica su ruta en {SOUNDFONT_ENV_VAR}."
        )
    if not midi_path.is_file():
        raise RuntimeError(f"No existe el archivo MIDI: {midi_path}")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "-ni", "-R", "0", "-C", "0", "-g", str(FLUIDSYNTH_GAIN),
        "-r", str(int(sr)), "-F", str(wav_path), str(sf2), str(midi_path),
    ]
    logger.info("Renderizando %s con fluidsynth (%s, %d Hz)", midi_path.name, sf2.name, sr)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"fluidsynth no terminó en {timeout_s:.0f} s al renderizar {midi_path.name}.") from exc
    except OSError as exc:
        raise RuntimeError(f"No se pudo ejecutar fluidsynth ({exe}): {exc}") from exc
    if proc.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
        detail = (proc.stderr or proc.stdout).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"fluidsynth falló al renderizar {midi_path.name} (código {proc.returncode}): {detail}")
    return wav_path


def _finish_audio(y: np.ndarray, n_samples: int, sr: int, fade_s: float = FADE_OUT_S) -> np.ndarray:
    """Ajusta la señal a ``n_samples`` (recorta o rellena con ceros) y normaliza el pico.

    Si se recorta audio que todavía suena se aplica un fundido de salida de
    ``fade_s`` segundos para no introducir un clic (un escalón = banda ancha).

    Parameters
    ----------
    y : np.ndarray
        Señal mono.
    n_samples : int
        Longitud final en muestras.
    sr : int
        Frecuencia de muestreo en Hz.
    fade_s : float, optional
        Duración del fundido de salida (s).

    Returns
    -------
    np.ndarray
        Señal ``float32`` de longitud ``n_samples`` con pico :data:`OUTPUT_PEAK`.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size >= n_samples:
        y = y[:n_samples].copy()
        n_fade = min(int(round(fade_s * sr)), n_samples)
        if n_fade > 0:
            y[-n_fade:] *= np.linspace(1.0, 0.0, n_fade)
    else:
        y = np.concatenate([y, np.zeros(n_samples - y.size)])
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y *= OUTPUT_PEAK / peak
    return y.astype(np.float32)


def _load_wav_mono(path: Path, sr: int) -> np.ndarray:
    """Lee el WAV de fluidsynth y lo mezcla a mono (promedio de canales)."""
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float64", always_2d=True)
    if int(file_sr) != int(sr):
        raise RuntimeError(f"fluidsynth produjo audio a {file_sr} Hz en lugar de {sr} Hz.")
    return y.mean(axis=1)


# ---------------------------------------------------------------------------
# Karplus-Strong (respaldo sin fluidsynth)
# ---------------------------------------------------------------------------


def _karplus_strong_note(
    f0_hz: float,
    n_samples: int,
    sr: int,
    rng: np.random.Generator,
    decay: float = KS_DECAY,
    rolloff: float = KS_ROLLOFF,
) -> np.ndarray:
    """Una nota de cuerda pulsada con el algoritmo de Karplus-Strong (1983).

    Teoría
    ------
    Una cuerda ideal fijada en sus extremos es un medio en el que la onda va
    y vuelve: su forma se repite cada periodo T = 1/f0. Karplus-Strong lo
    imita con una **línea de retardo** (*delay line*) de N ≈ sr/f0 muestras
    que se realimenta:

    1. **Excitación (el dedo)**: el delay line se llena con N muestras de
       ruido. El ruido contiene todas las frecuencias, como el golpe del
       dedo, y su fase aleatoria hace que cada nota suene ligeramente
       distinta. Antes se **filtra con un leve pasa-bajas**: como esas N
       muestras serán UN periodo de la nota, el bin h de su FFT de N puntos
       es exactamente el armónico h·f0; se conserva la fase del ruido y se
       fija la amplitud de cada armónico a 1/hᵖ (p = 1: −6 dB/octava, la
       pendiente de un pasa-bajas de primer orden; bin 0 = 0, sin DC).
       Con ruido blanco puro todos los armónicos arrancarían con la misma
       energía media y, en el registro grave (el delay line da solo 40–130
       vueltas por segundo, así que el filtro del paso 2 actúa poco), el
       sonido sería metálico y a veces el 2.º armónico superaría a la
       fundamental. Con 1/h el ataque es redondo, de dedo, y la fundamental
       domina como en un bajo eléctrico.
    2. **Recirculación con promedio de dos muestras**::

           y[n] = ρ · ½ · (y[n−N] + y[n−N−1])      (n ≥ N)

       Al repetirse cada N muestras, la señal se vuelve periódica: el
       espectro se concentra en f0 y sus armónicos h·f0 (serie armónica de
       una cuerda). El promedio ½(y[n−N] + y[n−N−1]) es un pasa-bajas
       (|H(f)| = |cos(π f / sr)|): en cada vuelta los armónicos agudos se
       atenúan más que la fundamental, igual que en una cuerda real, donde
       los agudos se apagan primero. ρ < 1 (≈0.996) añade una pérdida
       uniforme por vuelta (amortiguamiento global).
    3. **Afinación**: el promedio de dos muestras retrasa la señal media
       muestra, así que el periodo efectivo es N + ½ y f0 = sr/(N + ½).
       Por eso se usa N = round(sr/f0 − ½). El error de redondeo es de
       como mucho ½ muestra: < 6 cents en el registro del bajo a 22 050 Hz.

    Implementación: como y[n] solo depende de muestras de al menos N pasos
    atrás, se calcula un bloque de N muestras a la vez con numpy (un bucle
    de ``n_samples/N`` iteraciones en lugar de una por muestra).

    Parameters
    ----------
    f0_hz : float
        Frecuencia fundamental (Hz).
    n_samples : int
        Longitud de la nota en muestras.
    sr : int
        Frecuencia de muestreo (Hz).
    rng : np.random.Generator
        Generador para el ruido de excitación.
    decay : float, optional
        Factor de decaimiento ρ por vuelta.
    rolloff : float, optional
        Pendiente p del pasa-bajas del ruido inicial (amplitud 1/hᵖ).

    Returns
    -------
    np.ndarray
        Señal ``float64`` de ``n_samples`` muestras (pico ≈ 1).
    """
    period = max(2, int(round(sr / f0_hz - 0.5)))  # N: longitud del delay line (muestras)
    # 1) Excitación: ruido uniforme filtrado en frecuencia. Bin h = armónico h.
    noise = rng.uniform(-1.0, 1.0, period)
    spectrum = np.fft.rfft(noise)
    harmonic = np.arange(spectrum.size, dtype=np.float64)
    amplitude = np.zeros(spectrum.size)
    amplitude[1:] = harmonic[1:] ** (-rolloff)              # 1/hᵖ; sin DC (h = 0)
    excitation = np.fft.irfft(amplitude * np.exp(1j * np.angle(spectrum)), n=period)
    excitation /= max(float(np.max(np.abs(excitation))), 1e-12)

    # 2) Recirculación. z[k] guarda y[k−1]; z[0] = y[−1] = 0, así y[n−N−1] = z[n−N].
    z = np.zeros(n_samples + 1)
    n_init = min(period, n_samples)
    z[1:n_init + 1] = excitation[:n_init]
    for start in range(period, n_samples, period):
        end = min(start + period, n_samples)
        # y[start:end] = ρ · ½ · (y[start−N:end−N] + y[start−N−1:end−N−1])
        z[start + 1:end + 1] = decay * 0.5 * (z[start - period + 1:end - period + 1] + z[start - period:end - period])
    return z[1:]


def render_karplus_strong(
    notes: list[GTNote],
    sr: int = 22050,
    seed: int = 0,
    tail_s: float = TAIL_S,
) -> np.ndarray:
    """Sintetiza las notas con el algoritmo Karplus-Strong (respaldo sin fluidsynth).

    Devuelve la señal mono float32 normalizada.

    Cada nota se sintetiza con :func:`_karplus_strong_note` desde su onset;
    al llegar a su offset se apaga con un fundido coseno de
    :data:`KS_RELEASE_S` (el dedo apaga la cuerda), lo que deja un breve
    silencio antes de la siguiente nota y hace audible el re-ataque de notas
    repetidas. Las notas se suman en sus tiempos y la mezcla se normaliza.

    Parameters
    ----------
    notes : list[GTNote]
        Notas con tiempos absolutos (s).
    sr : int, optional
        Frecuencia de muestreo (Hz).
    seed : int, optional
        Semilla del ruido de excitación (resultado reproducible).
    tail_s : float, optional
        Silencio después del offset de la última nota (s).

    Returns
    -------
    np.ndarray
        Señal mono ``float32`` de ``round((último offset + tail_s)·sr)``
        muestras, con pico :data:`OUTPUT_PEAK`.

    Examples
    --------
    >>> y = render_karplus_strong([GTNote(0.1, 0.5, 33, "A", 0)], sr=22050, tail_s=0.2)
    >>> y.shape, y.dtype
    ((15435,), dtype('float32'))
    """
    rng = np.random.default_rng(seed)
    n_total = int(round(piece_duration_s(notes, tail_s) * sr))
    out = np.zeros(n_total)
    n_release = max(1, int(round(KS_RELEASE_S * sr)))
    for note in notes:
        start = int(round(note.onset_s * sr))
        n_sustain = max(1, int(round(note.offset_s * sr)) - start)
        n_note = min(n_sustain + n_release, n_total - start)
        if n_note <= 0:
            continue
        tone = _karplus_strong_note(midi_to_hz(note.midi), n_note, sr, rng)
        # Envolvente: 1 mientras se sostiene la nota, después medio coseno de 1 → 0.
        envelope = np.ones(n_note)
        release = envelope[n_sustain:]
        release[:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(release.size) / n_release))
        out[start:start + n_note] += tone * envelope
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= OUTPUT_PEAK / peak
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def save_ground_truth(notes: list[GTNote], path: str | Path, meta: dict | None = None) -> Path:
    """Guarda el ground truth en JSON: ``{"meta": {...}, "notes": [{onset_s, offset_s, midi, string, fret}, ...]}``.

    Parameters
    ----------
    notes : list[GTNote]
        Notas del ground truth.
    path : str | Path
        Archivo ``.gt.json`` (se crean las carpetas intermedias).
    meta : dict | None, optional
        Metadatos serializables (pieza, bpm, método de síntesis...).

    Returns
    -------
    Path
        Ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": dict(meta or {}),
        "notes": [
            {
                "onset_s": round(float(n.onset_s), 6),
                "offset_s": round(float(n.offset_s), 6),
                "midi": int(n.midi),
                "string": str(n.string),
                "fret": int(n.fret),
            }
            for n in notes
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Ground truth escrito: %s (%d notas)", path, len(notes))
    return path


def load_ground_truth(path: str | Path) -> list[GTNote]:
    """Lee un archivo ``.gt.json``.

    Parameters
    ----------
    path : str | Path
        Archivo escrito por :func:`save_ground_truth`.

    Returns
    -------
    list[GTNote]
        Notas en el orden del archivo.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el JSON no tiene el formato esperado o alguna nota es incoherente:
        cuerda fuera de E, A, D, G; traste negativo; MIDI distinto de
        cuerda al aire + traste; tiempos no finitos, negativos o con
        ``offset_s < onset_s``.

    Notes
    -----
    La coherencia MIDI = MIDI(cuerda al aire) + traste importa: las
    precisiones de pitch (compara MIDI) y de posición (compara cuerda y
    traste) se contradirían sin aviso si el archivo mezcla ambas cosas.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} no es un JSON válido: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("notes"), list):
        raise ValueError(f"{path.name} no tiene la clave 'notes' con la lista de notas del ground truth.")
    notes: list[GTNote] = []
    for i, item in enumerate(data["notes"]):
        try:
            note = GTNote(
                onset_s=float(item["onset_s"]),
                offset_s=float(item["offset_s"]),
                midi=int(item["midi"]),
                string=str(item["string"]),
                fret=int(item["fret"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Nota {i} inválida en {path.name}: {exc!r}") from exc
        problem = _gt_note_problem(note)
        if problem:
            raise ValueError(f"Nota {i} inválida en {path.name}: {problem}")
        notes.append(note)
    return notes


def _gt_note_problem(note: GTNote) -> str | None:
    """Describe por qué ``note`` es incoherente (None si es válida).

    Parameters
    ----------
    note : GTNote
        Nota leída de un ``.gt.json``.

    Returns
    -------
    str | None
        Mensaje en español, o None.

    Examples
    --------
    >>> _gt_note_problem(GTNote(0.5, 0.9, 33, "A", 0)) is None
    True
    >>> _gt_note_problem(GTNote(0.5, 0.9, 50, "B", 10))
    "cuerda 'B' desconocida (use E, A, D o G)"
    >>> _gt_note_problem(GTNote(0.5, 0.9, 40, "A", 5))
    'MIDI 40 no coincide con A-5 (MIDI 38 = 33 de la cuerda al aire + 5 trastes)'
    """
    if note.string not in STRING_ORDER:
        return f"cuerda {note.string!r} desconocida (use {', '.join(STRING_ORDER[:-1])} o {STRING_ORDER[-1]})"
    if note.fret < 0:
        return f"traste negativo ({note.fret})"
    expected = OPEN_STRING_MIDI[note.string] + note.fret
    if note.midi != expected:
        return (f"MIDI {note.midi} no coincide con {note.label} (MIDI {expected} = "
                f"{OPEN_STRING_MIDI[note.string]} de la cuerda al aire + {note.fret} trastes)")
    if not (np.isfinite(note.onset_s) and np.isfinite(note.offset_s)) or note.onset_s < 0:
        return f"tiempos inválidos (onset {note.onset_s} s, offset {note.offset_s} s)"
    if note.offset_s < note.onset_s:
        return f"offset_s ({note.offset_s} s) anterior al onset_s ({note.onset_s} s)"
    return None


def load_ground_truth_meta(path: str | Path) -> dict[str, Any]:
    """Lee solo los metadatos (``"meta"``) de un ``.gt.json`` ({} si no hay).

    Parameters
    ----------
    path : str | Path
        Archivo ``.gt.json``.

    Returns
    -------
    dict
        Metadatos guardados por :func:`generate_piece` (pieza, bpm, sr, método...).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    return dict(meta) if isinstance(meta, dict) else {}


def gt_path_for(audio_path: str | Path) -> Path:
    """Ruta del ground truth asociado a un audio: ``<carpeta>/<nombre>.gt.json``.

    Parameters
    ----------
    audio_path : str | Path
        Archivo de audio (MP3, WAV...).

    Returns
    -------
    Path
        Misma carpeta y nombre base con la extensión ``.gt.json`` (exista o no).

    Examples
    --------
    >>> gt_path_for("data/synthetic/cromatica.mp3").name
    'cromatica.gt.json'
    """
    p = Path(audio_path)
    return p.with_name(p.stem + ".gt.json")


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------


def _resolve_method(method: str) -> str:
    """Traduce ``"auto"`` al sintetizador disponible y valida el resto."""
    if method not in METHODS:
        raise ValueError(f"Método de síntesis desconocido: «{method}». Opciones: {', '.join(METHODS)}")
    if method == "auto":
        return "fluidsynth" if fluidsynth_available() else "karplus-strong"
    if method == "fluidsynth" and not fluidsynth_available():
        raise RuntimeError(
            "Se pidió el método 'fluidsynth' pero no hay ejecutable de fluidsynth o soundfont GM disponible. "
            "Usa method='auto' o 'karplus-strong'."
        )
    return method


def generate_piece(
    name: str,
    out_dir: str | Path,
    bpm: float = 100.0,
    sr: int = 22050,
    method: str = "auto",
    legato: float = 0.9,
    seed: int = 0,
    tmp_dir: str | Path | None = None,
) -> DatasetItem:
    """Genera MIDI, MP3 y ground truth de una pieza.

    ``method``: ``"auto"`` (fluidsynth si está disponible, si no Karplus-Strong),
    ``"fluidsynth"`` o ``"karplus-strong"``.

    Parameters
    ----------
    name : str
        Clave de :data:`PIECES`.
    out_dir : str | Path
        Carpeta de salida; se escriben ``<name>.mid``, ``<name>.mp3`` y
        ``<name>.gt.json`` (nunca el WAV intermedio).
    bpm : float, optional
        Tempo en negras por minuto.
    sr : int, optional
        Frecuencia de muestreo del audio (Hz).
    method : str, optional
        Sintetizador (ver arriba).
    legato : float, optional
        Fracción de la duración nominal durante la que suena cada nota.
    seed : int, optional
        Semilla del ruido de Karplus-Strong.
    tmp_dir : str | Path | None, optional
        Carpeta donde crear el directorio temporal del WAV de fluidsynth
        (``None`` = carpeta temporal del sistema, respeta ``TMPDIR``).

    Returns
    -------
    DatasetItem
        Rutas generadas y método efectivamente usado.

    Raises
    ------
    ValueError
        Pieza o método desconocidos.
    RuntimeError
        Si se pide ``"fluidsynth"`` sin tenerlo, o si fluidsynth/ffmpeg fallan.
    """
    from src.io_audio import save_mp3  # import diferido: io_audio exige ffmpeg solo al usarse

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = piece_notes(name, bpm=bpm, legato=legato)
    used = _resolve_method(method)
    n_samples = int(round(piece_duration_s(notes) * sr))
    logger.info("Pieza «%s»: %d notas, %.0f bpm, %.2f s, síntesis con %s",
                name, len(notes), bpm, n_samples / sr, used)

    midi_path = write_midi(notes, out_dir / f"{name}.mid", program=BASS_PROGRAM, bpm=bpm)
    mp3_path = out_dir / f"{name}.mp3"
    soundfont: Path | None = None
    if used == "fluidsynth":
        soundfont = find_soundfont()
        if tmp_dir is not None:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        # El WAV intermedio vive en una carpeta temporal que se borra al salir del bloque.
        with tempfile.TemporaryDirectory(prefix="smartuner_", dir=None if tmp_dir is None else str(tmp_dir)) as tmp:
            wav_path = render_fluidsynth(midi_path, Path(tmp) / f"{name}.wav", sr=sr, soundfont=soundfont)
            y = _load_wav_mono(wav_path, sr)
        # fluidsynth deja varios segundos de cola: se recorta a la duración esperada.
        y = _finish_audio(y, n_samples, sr)
    else:
        # render_karplus_strong ya devuelve n_samples muestras normalizadas.
        y = render_karplus_strong(notes, sr=sr, seed=seed)
    save_mp3(mp3_path, y, sr)

    meta: dict[str, Any] = {
        "piece": name,
        "bpm": float(bpm),
        "sr": int(sr),
        "method": used,
        "tuning": dict(OPEN_STRING_HZ),
        "program": BASS_PROGRAM,
        "program_name": "Electric Bass (finger)",
        "soundfont": soundfont.name if soundfont is not None else None,
        "legato": float(legato),
        "lead_in_s": LEAD_IN_S,
        "duration_s": round(n_samples / sr, 6),
        "n_notes": len(notes),
    }
    gt_path = save_ground_truth(notes, gt_path_for(mp3_path), meta=meta)
    logger.info("Pieza «%s» lista: %s, %s, %s", name, midi_path.name, mp3_path.name, gt_path.name)
    return DatasetItem(name=name, mp3_path=mp3_path, gt_path=gt_path, midi_path=midi_path, method=used)


def generate_dataset(
    out_dir: str | Path | None = None,
    bpm: float = 100.0,
    sr: int = 22050,
    method: str = "auto",
    pieces: list[str] | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    tmp_dir: str | Path | None = None,
) -> list[DatasetItem]:
    """Genera todas las piezas (o las de ``pieces``) en ``out_dir`` (por defecto ``data/synthetic``).

    Parameters
    ----------
    out_dir : str | Path | None, optional
        Carpeta de salida (``data/synthetic`` del proyecto si es None).
    bpm : float, optional
        Tempo en negras por minuto.
    sr : int, optional
        Frecuencia de muestreo (Hz).
    method : str, optional
        ``"auto"``, ``"fluidsynth"`` o ``"karplus-strong"``.
    pieces : list[str] | None, optional
        Subconjunto de :data:`PIECES` (todas si es None).
    progress : ProgressCallback | None, optional
        ``progress(fracción, mensaje)`` antes de cada pieza y al terminar.
    cancel : threading.Event | None, optional
        Si se activa, la generación se detiene antes de la siguiente pieza.
    tmp_dir : str | Path | None, optional
        Carpeta para los WAV temporales de fluidsynth (ver :func:`generate_piece`).

    Returns
    -------
    list[DatasetItem]
        Un elemento por pieza generada, en orden.

    Raises
    ------
    ValueError
        Si alguna pieza no existe.
    CancelledError
        Si ``cancel`` se activa.
    """
    out = Path(out_dir) if out_dir is not None else DATA_DIR / "synthetic"
    names = list(PIECES) if pieces is None else list(pieces)
    unknown = [n for n in names if n not in PIECES]
    if unknown:
        raise ValueError(f"Piezas desconocidas: {', '.join(unknown)}. Disponibles: {', '.join(PIECES)}")
    logger.info("Generando dataset sintético en %s: %d piezas (%s)", out, len(names), ", ".join(names))
    items: list[DatasetItem] = []
    for i, name in enumerate(names):
        if cancel is not None and cancel.is_set():
            logger.info("Generación del dataset cancelada tras %d de %d piezas", i, len(names))
            raise CancelledError("Generación del dataset cancelada por el usuario.")
        if progress is not None:
            progress(i / max(len(names), 1), f"Generando «{name}» ({i + 1}/{len(names)})…")
        items.append(generate_piece(name, out, bpm=bpm, sr=sr, method=method, tmp_dir=tmp_dir))
    if progress is not None:
        progress(1.0, f"Dataset generado: {len(items)} piezas en {out}")
    logger.info("Dataset sintético generado: %d piezas en %s", len(items), out)
    return items
