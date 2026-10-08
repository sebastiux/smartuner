"""Etapa 1 — Carga de audio. [CONTRATO: implementar]

Decodifica MP3 (o WAV/FLAC/OGG) a una señal mono float32 a la frecuencia de
muestreo de trabajo usando ffmpeg. librosa ≥ 1.0 eliminó el backend audioread
(que internamente llamaba a ffmpeg), así que aquí se invoca ffmpeg directamente
por subproceso: ``ffmpeg -i <archivo> -f f32le -ac 1 -ar <sr> -``.
También ofrece utilidades para escribir WAV y MP3 (usadas por el dataset sintético).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg no está instalado o no está en el PATH (mensaje con instrucciones de instalación)."""


class AudioLoadError(RuntimeError):
    """El archivo no existe o ffmpeg no pudo decodificarlo."""


def find_ffmpeg() -> str | None:
    """Ruta al ejecutable de ffmpeg o ``None`` si no está disponible."""
    raise NotImplementedError


def require_ffmpeg() -> str:
    """Igual que :func:`find_ffmpeg` pero lanza :class:`FFmpegNotFoundError` con un
    mensaje claro en español (cómo instalarlo en Windows/macOS/Linux)."""
    raise NotImplementedError


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Decodifica ``path`` a mono float32 en [-1, 1] a ``sr`` Hz.

    Returns ``(y, sr)``. Lanza :class:`FFmpegNotFoundError` o :class:`AudioLoadError`.
    """
    raise NotImplementedError


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> Path:
    """Escribe ``y`` como WAV PCM16 (crea carpetas intermedias). Devuelve la ruta."""
    raise NotImplementedError


def save_mp3(path: str | Path, y: np.ndarray, sr: int, bitrate: str = "192k") -> Path:
    """Codifica ``y`` a MP3 con ffmpeg (libmp3lame). Devuelve la ruta."""
    raise NotImplementedError


def convert_to_mp3(src: str | Path, dst: str | Path, bitrate: str = "192k") -> Path:
    """Convierte un archivo de audio existente (p. ej. WAV) a MP3 con ffmpeg."""
    raise NotImplementedError
