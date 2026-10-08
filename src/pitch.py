"""Etapa 5 — Estimación de pitch con pYIN. [CONTRATO: implementar]

Se ejecuta ``librosa.pyin`` UNA vez sobre la señal de análisis completa (más
eficiente y con mejor continuidad que por segmento) y luego se resume cada
segmento con la mediana de la f0 de sus frames con voz (ignorando el ataque).
La f0 solo se usa para PODAR brazos (±k semitonos); la decisión final de
cuerda/traste la toman los agentes bandit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import PitchConfig
from src.segmentation import Segment


@dataclass
class PitchTrack:
    """Trayectoria de f0 de toda la pista.

    Attributes
    ----------
    times_s : np.ndarray
        Centro de cada frame en segundos.
    f0_hz : np.ndarray
        f0 por frame en Hz (NaN donde no hay voz).
    voiced_prob : np.ndarray
        Probabilidad de voz por frame (0–1).
    voiced : np.ndarray
        Booleano por frame.
    """

    times_s: np.ndarray
    f0_hz: np.ndarray
    voiced_prob: np.ndarray
    voiced: np.ndarray


def hz_to_midi(f_hz: float) -> float:
    """Convierte Hz a número MIDI fraccionario (A4 = 440 Hz = 69)."""
    return float(69.0 + 12.0 * np.log2(f_hz / 440.0))


def midi_to_hz(midi: float) -> float:
    """Convierte número MIDI (posiblemente fraccionario) a Hz."""
    return float(440.0 * 2.0 ** ((midi - 69.0) / 12.0))


def midi_to_name(midi: int) -> str:
    """Nombre de nota en notación anglosajona con octava, p. ej. 33 → ``"A1"``."""
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    m = int(round(midi))
    return f"{names[m % 12]}{m // 12 - 1}"


def estimate_pitch_track(y: np.ndarray, sr: int, cfg: PitchConfig, hop_length: int = 256) -> PitchTrack:
    """Ejecuta pYIN sobre toda la señal y devuelve la trayectoria de f0."""
    raise NotImplementedError


def segment_f0(track: PitchTrack, segment: Segment, cfg: PitchConfig, attack_skip_s: float = 0.03) -> tuple[float | None, float]:
    """Resume la f0 de un segmento: ``(mediana de f0 con voz o None, fracción con voz)``.

    Devuelve ``None`` como f0 si la fracción con voz es menor que
    ``cfg.min_voiced_ratio``.
    """
    raise NotImplementedError


def annotate_segments(track: PitchTrack, segments: list[Segment], cfg: PitchConfig, attack_skip_s: float = 0.03) -> None:
    """Rellena ``f0_hz`` y ``voiced_ratio`` de cada segmento conservado (in place)."""
    raise NotImplementedError
