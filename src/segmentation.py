"""Etapa 4 — Segmentación por onsets. [CONTRATO: implementar]

Detecta inicios de nota (onsets) con librosa sobre la señal de análisis y
divide la pista en segmentos [onset_i, fin_i). El fin de cada segmento es el
siguiente onset, recortado al último frame cuyo RMS supera el umbral de
silencio (así el release/silencio no forma parte de la nota). Se marcan como
descartados (``kept=False``) los segmentos silenciosos y los demasiado cortos;
se conservan en la lista para poder dibujarlos en gris en la GUI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import SegmentationConfig


@dataclass
class Segment:
    """Un segmento de nota detectado.

    Attributes
    ----------
    index : int
        Posición en la lista completa de segmentos (incluye descartados).
    start_s, end_s : float
        Inicio y fin en segundos.
    start_sample, end_sample : int
        Inicio y fin en muestras (``end`` exclusivo).
    rms_db : float
        RMS medio del segmento en dB relativos al máximo de la pista.
    kept : bool
        False si se descartó (silencio o muy corto).
    reason : str
        ``"ok"``, ``"silencio"`` o ``"corto"``.
    f0_hz : float | None
        f0 estimada por pYIN (la llena :mod:`src.pitch`); None si es desconocida.
    voiced_ratio : float
        Fracción de frames del segmento con voz según pYIN.
    """

    index: int
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int
    rms_db: float
    kept: bool = True
    reason: str = "ok"
    f0_hz: float | None = None
    voiced_ratio: float = 0.0

    @property
    def duration_s(self) -> float:
        """Duración en segundos."""
        return self.end_s - self.start_s

    @property
    def midi(self) -> float | None:
        """f0 expresada como número MIDI fraccionario (None si f0 desconocida)."""
        if self.f0_hz is None or self.f0_hz <= 0:
            return None
        return float(69.0 + 12.0 * np.log2(self.f0_hz / 440.0))


def onset_envelope(y: np.ndarray, sr: int, cfg: SegmentationConfig) -> np.ndarray:
    """Función de novedad espectral (``librosa.onset.onset_strength``) con ``hop_length`` de ``cfg``."""
    raise NotImplementedError


def detect_onsets(y: np.ndarray, sr: int, cfg: SegmentationConfig) -> np.ndarray:
    """Tiempos de onset en segundos (array 1-D ordenado) detectados sobre ``y``."""
    raise NotImplementedError


def segment_audio(y: np.ndarray, sr: int, cfg: SegmentationConfig, onsets_s: np.ndarray | None = None) -> list[Segment]:
    """Divide ``y`` en segmentos usando ``onsets_s`` (o los detecta si es None).

    Devuelve TODOS los segmentos (conservados y descartados), ordenados, con
    ``index`` consecutivo desde 0.
    """
    raise NotImplementedError


def kept_segments(segments: list[Segment]) -> list[Segment]:
    """Filtra los segmentos con ``kept=True`` (conserva el orden)."""
    return [s for s in segments if s.kept]
