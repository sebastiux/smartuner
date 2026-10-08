"""Etapa 3 — Preprocesamiento. [CONTRATO: implementar]

Produce dos versiones de la señal:

* ``y_analysis``: pasa-bajas Butterworth (``sosfiltfilt``, fase cero) +
  normalización de pico. Se usa para detectar onsets y estimar pitch (pYIN),
  donde los armónicos agudos y el ruido de trastes estorban.
* ``y_spectral``: solo normalizada. Se usa para la recompensa, que necesita
  los armónicos por encima del corte (N=5 armónicos de 196 Hz llegan a ~980 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import PreprocessConfig


@dataclass
class PreprocessResult:
    """Señales preprocesadas (misma longitud y frecuencia de muestreo que la entrada)."""

    y_analysis: np.ndarray
    y_spectral: np.ndarray
    sr: int


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Filtro pasa-bajas Butterworth de fase cero. Si ``cutoff_hz >= sr/2`` devuelve una copia."""
    raise NotImplementedError


def normalize_peak(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Escala ``y`` para que max|y| = ``peak`` (una señal nula se devuelve sin cambios)."""
    raise NotImplementedError


def preprocess(y: np.ndarray, sr: int, cfg: PreprocessConfig) -> PreprocessResult:
    """Aplica el preprocesamiento descrito en el encabezado del módulo."""
    raise NotImplementedError
