"""Orquestación de las etapas 1–6 (análisis de audio). [CONTRATO: implementar]

:func:`analyze` ejecuta, en orden y registrando cada paso con ``logging``:

1. Carga del audio (:mod:`src.io_audio`).
2. Separación opcional del bajo (:mod:`src.separation`).
3. Preprocesamiento (:mod:`src.preprocessing`).
4. Segmentación por onsets (:mod:`src.segmentation`).
5. Estimación de pitch con pYIN (:mod:`src.pitch`).
6. Espectro y datos bandit precalculados por segmento (:mod:`src.environment`).

Además carga automáticamente el ground truth si existe ``<nombre>.gt.json``
junto al audio. La GUI y la CLI solo llaman a :func:`analyze`; ninguna conoce
el orden interno de las etapas.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import Config, ProgressCallback
from src.environment import SegmentBanditData, Spectrum
from src.pitch import PitchTrack
from src.segmentation import Segment
from src.synth_dataset import GTNote


@dataclass
class AnalysisResult:
    """Resultado del análisis de una pista.

    Attributes
    ----------
    path : Path
        Archivo de entrada elegido por el usuario.
    source_path : Path
        Archivo efectivamente analizado (el stem de Demucs si hubo separación).
    sr : int
        Frecuencia de muestreo.
    y_raw : np.ndarray
        Señal decodificada (mono, sin procesar).
    y_analysis : np.ndarray
        Señal filtrada + normalizada (onsets, pYIN).
    y_spectral : np.ndarray
        Señal normalizada sin filtrar (recompensa, espectrograma).
    onsets_s : np.ndarray
        Onsets detectados (s).
    segments : list[Segment]
        Todos los segmentos (conservados y descartados).
    pitch_track : PitchTrack
        Trayectoria de f0 de toda la pista.
    spectrum : Spectrum
        Espectro usado por la recompensa.
    segment_data : list[SegmentBanditData]
        Datos bandit de cada segmento conservado (en orden).
    ground_truth : list[GTNote] | None
        Ground truth si existe ``.gt.json`` junto al audio.
    config : Config
        Copia de la configuración usada.
    """

    path: Path
    source_path: Path
    sr: int
    y_raw: np.ndarray
    y_analysis: np.ndarray
    y_spectral: np.ndarray
    onsets_s: np.ndarray
    segments: list[Segment]
    pitch_track: PitchTrack
    spectrum: Spectrum
    segment_data: list[SegmentBanditData]
    ground_truth: list[GTNote] | None
    config: Config = field(default_factory=Config)

    @property
    def kept(self) -> list[Segment]:
        """Segmentos conservados, en orden."""
        return [s for s in self.segments if s.kept]

    @property
    def duration_s(self) -> float:
        """Duración de la pista en segundos."""
        return len(self.y_raw) / float(self.sr)


def analyze(
    path: str | Path,
    cfg: Config,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> AnalysisResult:
    """Ejecuta las etapas 1–6 sobre ``path`` (ver encabezado del módulo).

    Lanza :class:`src.config.CancelledError` si ``cancel`` se activa entre etapas.
    """
    raise NotImplementedError


def rebuild_segment_data(analysis: AnalysisResult, cfg: Config) -> AnalysisResult:
    """Recalcula espectro (si cambió su tipo/parámetros) y ``segment_data`` cuando
    cambian parámetros del entorno (k, N, β, tolerancia...) sin volver a
    decodificar ni segmentar. Devuelve un NUEVO :class:`AnalysisResult`."""
    raise NotImplementedError
