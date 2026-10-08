"""Etapa 1 — Carga de audio (y escritura de WAV/MP3).

Papel en el pipeline
--------------------
Es la puerta de entrada: convierte el MP3 del usuario en un arreglo numpy
mono ``float32`` a la frecuencia de muestreo de trabajo (22 050 Hz), que es lo
que consumen todas las etapas siguientes.

¿Por qué ffmpeg directamente?
-----------------------------
librosa decodificaba MP3 a través de *audioread*, que a su vez llamaba a
ffmpeg. librosa ≥ 1.0 eliminó ese backend, así que aquí se invoca ffmpeg por
subproceso, pidiéndole que haga todo el trabajo de una vez::

    ffmpeg -i entrada.mp3 -f f32le -ac 1 -ar 22050 -
           │              │       │     │        └─ escribe a stdout
           │              │       │     └─ remuestrea a 22 050 Hz
           │              │       └─ mezcla a mono (promedio de canales)
           │              └─ muestras float32 little-endian "crudas"
           └─ cualquier formato que ffmpeg entienda (MP3, WAV, FLAC, OGG...)

y los bytes de stdout se reinterpretan como ``np.float32``. Si ffmpeg no está
instalado se lanza :class:`FFmpegNotFoundError` con instrucciones claras, que
la GUI muestra en un cuadro de diálogo.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

#: Variable de entorno opcional con la ruta explícita al ejecutable de ffmpeg.
FFMPEG_ENV_VAR = "SMARTUNER_FFMPEG"

_INSTALL_HELP = (
    "No se encontró ffmpeg, necesario para leer y escribir MP3.\n\n"
    "Instálalo y asegúrate de que esté en el PATH:\n"
    "  • Windows:  winget install Gyan.FFmpeg   (o: choco install ffmpeg)\n"
    "  • macOS:    brew install ffmpeg\n"
    "  • Linux:    sudo apt install ffmpeg      (o el gestor de tu distribución)\n\n"
    f"También puedes indicar la ruta con la variable de entorno {FFMPEG_ENV_VAR}."
)


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg no está instalado o no está en el PATH.

    El mensaje incluye instrucciones de instalación para Windows, macOS y Linux.
    """


class AudioLoadError(RuntimeError):
    """El archivo no existe, está vacío o ffmpeg no pudo decodificarlo."""


def find_ffmpeg() -> str | None:
    """Localiza el ejecutable de ffmpeg.

    Primero consulta la variable de entorno ``SMARTUNER_FFMPEG`` y después el PATH.

    Returns
    -------
    str | None
        Ruta al ejecutable, o ``None`` si no está disponible.

    Examples
    --------
    >>> path = find_ffmpeg()  # doctest: +SKIP
    >>> path                   # doctest: +SKIP
    '/usr/bin/ffmpeg'
    """
    explicit = os.environ.get(FFMPEG_ENV_VAR)
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    """Igual que :func:`find_ffmpeg` pero falla con un mensaje explicativo.

    Returns
    -------
    str
        Ruta al ejecutable de ffmpeg.

    Raises
    ------
    FFmpegNotFoundError
        Si ffmpeg no está disponible (mensaje en español con instrucciones).
    """
    path = find_ffmpeg()
    if path is None:
        raise FFmpegNotFoundError(_INSTALL_HELP)
    return path


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Decodifica un archivo de audio a mono ``float32`` en [-1, 1].

    Parameters
    ----------
    path : str | Path
        Archivo de entrada (MP3, WAV, FLAC, OGG... cualquier formato de ffmpeg).
    sr : int, optional
        Frecuencia de muestreo de salida en Hz (por defecto 22 050).

    Returns
    -------
    y : np.ndarray
        Señal mono, forma ``(n_muestras,)``, ``float32``.
    sr : int
        Frecuencia de muestreo de ``y`` en Hz.

    Raises
    ------
    FFmpegNotFoundError
        Si ffmpeg no está instalado.
    AudioLoadError
        Si el archivo no existe o no se puede decodificar.

    Examples
    --------
    >>> y, sr = load_audio("data/synthetic/linea_simple.mp3")  # doctest: +SKIP
    >>> sr, y.dtype                                             # doctest: +SKIP
    (22050, dtype('float32'))
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"No existe el archivo de audio: {path}")
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",  # float32 crudo
        "-ac", "1",                             # mono
        "-ar", str(int(sr)),                    # remuestreo
        "-",                                    # salida por stdout
    ]
    logger.info("Decodificando %s con ffmpeg (mono, %d Hz)", path.name, sr)
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLoadError(f"ffmpeg no pudo decodificar {path.name}: {detail}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size == 0:
        raise AudioLoadError(f"El archivo {path.name} no contiene audio decodificable.")
    logger.info("Audio cargado: %.2f s, %d muestras", y.size / sr, y.size)
    return y, int(sr)


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> Path:
    """Escribe ``y`` como WAV PCM de 16 bits.

    Parameters
    ----------
    path : str | Path
        Archivo de salida; se crean las carpetas intermedias.
    y : np.ndarray
        Señal mono en [-1, 1] (los valores fuera de rango se recortan).
    sr : int
        Frecuencia de muestreo en Hz.

    Returns
    -------
    Path
        Ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0), int(sr), subtype="PCM_16")
    return path


def save_mp3(path: str | Path, y: np.ndarray, sr: int, bitrate: str = "192k") -> Path:
    """Codifica ``y`` a MP3 con ffmpeg (libmp3lame), enviándole float32 por stdin.

    Parameters
    ----------
    path : str | Path
        Archivo MP3 de salida; se crean las carpetas intermedias.
    y : np.ndarray
        Señal mono en [-1, 1].
    sr : int
        Frecuencia de muestreo de ``y`` en Hz.
    bitrate : str, optional
        Tasa de bits de ffmpeg (``"192k"`` por defecto).

    Returns
    -------
    Path
        Ruta escrita.

    Raises
    ------
    FFmpegNotFoundError
        Si ffmpeg no está instalado.
    RuntimeError
        Si la codificación falla.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()
    data = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0).tobytes()
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "-",
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        str(path),
    ]
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo crear {path.name}: {proc.stderr.decode('utf-8', errors='replace')}")
    logger.info("MP3 escrito: %s", path)
    return path


def convert_to_mp3(src: str | Path, dst: str | Path, bitrate: str = "192k") -> Path:
    """Convierte un archivo de audio existente (p. ej. WAV) a MP3 con ffmpeg.

    Parameters
    ----------
    src : str | Path
        Archivo de entrada.
    dst : str | Path
        Archivo MP3 de salida.
    bitrate : str, optional
        Tasa de bits (``"192k"`` por defecto).

    Returns
    -------
    Path
        Ruta del MP3 escrito.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise AudioLoadError(f"No existe el archivo de audio: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo convertir {src.name}: {proc.stderr.decode('utf-8', errors='replace')}")
    logger.info("Convertido a MP3: %s", dst)
    return dst
