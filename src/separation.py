"""Etapa 2 (opcional) — Separación del bajo con Demucs. [CONTRATO: implementar]

Si el usuario marca "Separar bajo de mezcla", se ejecuta Demucs (modelo
``htdemucs``, modo ``--two-stems bass``) como subproceso
(``python -m demucs ...``) para poder cancelarlo y leer su progreso. El stem
resultante se guarda en ``cache/`` con una clave = SHA-1 del contenido del
archivo + modelo, de modo que la segunda vez es instantáneo.
"""

from __future__ import annotations

import threading
from pathlib import Path

from src.config import ProgressCallback

class SeparationError(RuntimeError):
    """Demucs no está instalado, falló o fue cancelado (mensaje en español)."""


def demucs_available() -> bool:
    """True si el paquete ``demucs`` se puede importar (sin importarlo por completo)."""
    raise NotImplementedError


def cache_key(path: str | Path, model: str = "htdemucs") -> str:
    """Clave de caché: SHA-1 (hex) del contenido del archivo concatenado con el modelo."""
    raise NotImplementedError


def cached_stem_path(path: str | Path, model: str = "htdemucs", cache_dir: str | Path | None = None) -> Path:
    """Ruta donde se guarda (o guardaría) el stem de bajo: ``<cache_dir>/<clave>_bass.wav``."""
    raise NotImplementedError


def separate_bass(
    path: str | Path,
    model: str = "htdemucs",
    cache_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Devuelve la ruta al stem de bajo aislado (WAV), usando la caché si existe.

    ``progress(fraccion_0_1, mensaje)`` se llama a medida que Demucs avanza.
    Si ``cancel`` se activa se mata el subproceso y se lanza :class:`SeparationError`.
    """
    raise NotImplementedError
