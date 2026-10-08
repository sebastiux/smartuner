"""Etapa 6 — Entorno bandit por segmento. [CONTRATO: implementar]

Cada segmento de nota es un problema Multi-Armed Bandit independiente:

* **Brazos**: posiciones (cuerda, traste) con f(cuerda, traste) = f_cuerda·2^(traste/12),
  podadas a ±k semitonos de la f0 estimada por pYIN (k=None → 52 brazos).
* **Recompensa** de un pull del brazo a: se elige un frame j al azar del
  segmento y se calcula

      X̃_j      = |X_j| / max|X_j|                          (espectro del frame normalizado)
      S⁺_j(f)  = Σ_h w_h · X̃_j(h·f) / Σ_h w_h             (energía en los armónicos)
      S⁻_j(f)  = Σ_h w_h · X̃_j((h−½)·f) / Σ_h w_h         (energía ENTRE armónicos)
      S_j(f)   = max(S⁺_j(f) − β·S⁻_j(f), 0)               (saliencia ∈ [0, 1])
      r        = S_j(f_a) − λ·|traste_a − traste_previo|/12 (+ ruido opcional)

  con h = 1..N y w_h = 1/h. X̃(x) se lee como el máximo de la magnitud
  interpolada dentro de ±``tolerance_semitones`` alrededor de x.
* **Valor real** μ_a = media sobre TODOS los frames del segmento de S_j(f_a)
  menos la penalización: se calcula exactamente y solo se usa para medir
  regret y % de brazo óptimo (el agente nunca lo ve).

Para que cada pull sea O(1) se precalcula, una sola vez por segmento, la
matriz ``salience[frame, brazo]`` (:class:`SegmentBanditData`); el entorno
(:class:`BanditEnvironment`) solo añade la penalización, que depende del
traste previo elegido por el propio agente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import OPEN_STRING_HZ, OPEN_STRING_MIDI, STRING_ORDER, EnvConfig
from src.segmentation import Segment


# ---------------------------------------------------------------------------
# Brazos
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Arm:
    """Una posición del diapasón: cuerda + traste.

    Attributes
    ----------
    string : str
        ``"E"``, ``"A"``, ``"D"`` o ``"G"``.
    fret : int
        Traste (0 = cuerda al aire).
    """

    string: str
    fret: int

    @property
    def freq_hz(self) -> float:
        """Frecuencia fundamental f = f_cuerda · 2^(traste/12) en Hz."""
        return fret_frequency(self.string, self.fret)

    @property
    def midi(self) -> int:
        """Número MIDI de la nota (E1 al aire = 28)."""
        return OPEN_STRING_MIDI[self.string] + self.fret

    @property
    def label(self) -> str:
        """Etiqueta corta ``"A-0"`` usada en GUI, logs y gráficas."""
        return f"{self.string}-{self.fret}"

    @property
    def string_index(self) -> int:
        """Índice de la cuerda de grave a aguda (E=0, A=1, D=2, G=3)."""
        return STRING_ORDER.index(self.string)


def fret_frequency(string: str, fret: int) -> float:
    """f(cuerda, traste) = f_cuerda · 2^(traste/12) en Hz."""
    return float(OPEN_STRING_HZ[string] * 2.0 ** (fret / 12.0))


def all_arms(n_frets: int = 12) -> list[Arm]:
    """Las 4·(n_frets+1) posiciones, ordenadas por cuerda (E, A, D, G) y traste."""
    raise NotImplementedError


def candidate_arms(f0_hz: float | None, k: int | None, n_frets: int = 12) -> list[Arm]:
    """Brazos con |midi(brazo) − round(midi(f0))| ≤ k.

    Si ``f0_hz`` es None o ``k`` es None devuelve :func:`all_arms`. Si la
    poda deja la lista vacía (f0 fuera del rango del bajo) también devuelve
    todos los brazos. Orden: por MIDI ascendente y luego por cuerda.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Espectro
# ---------------------------------------------------------------------------


@dataclass
class Spectrum:
    """Representación tiempo-frecuencia de magnitud de toda la pista.

    Attributes
    ----------
    mag : np.ndarray
        Magnitud, forma ``(n_bins, n_frames)``, float32.
    freqs_hz : np.ndarray
        Frecuencia central de cada bin (Hz), creciente.
    times_s : np.ndarray
        Centro temporal de cada frame (s).
    hop_length : int
        Salto entre frames en muestras.
    sr : int
        Frecuencia de muestreo.
    kind : str
        ``"cqt"`` o ``"stft"``.
    """

    mag: np.ndarray
    freqs_hz: np.ndarray
    times_s: np.ndarray
    hop_length: int
    sr: int
    kind: str


def compute_spectrum(y: np.ndarray, sr: int, cfg: EnvConfig, hop_length: int = 256) -> Spectrum:
    """Calcula la CQT (``cfg.spectrum == "cqt"``) o la STFT de ``y``."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Template armónico y saliencia
# ---------------------------------------------------------------------------


@dataclass
class HarmonicTemplate:
    """Posiciones y pesos del template armónico de una frecuencia f.

    Attributes
    ----------
    f0_hz : float
        Fundamental del template.
    harmonic_hz : np.ndarray
        h·f para h = 1..N (posiciones positivas).
    interharmonic_hz : np.ndarray
        (h−½)·f para h = 1..N (posiciones que se restan con peso β).
    weights : np.ndarray
        w_h = 1/h normalizados para que sumen 1.
    """

    f0_hz: float
    harmonic_hz: np.ndarray
    interharmonic_hz: np.ndarray
    weights: np.ndarray


def harmonic_template(f0_hz: float, n_harmonics: int) -> HarmonicTemplate:
    """Construye el template armónico de ``f0_hz`` (ver ecuaciones del encabezado)."""
    raise NotImplementedError


def salience(frames_mag: np.ndarray, freqs_hz: np.ndarray, arm_freqs_hz: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Saliencia armónica S_j(f) de cada frame para cada frecuencia candidata.

    Parameters
    ----------
    frames_mag : np.ndarray
        Magnitudes, forma ``(n_bins, n_frames)`` (sin normalizar; se normaliza
        cada frame por su máximo aquí).
    freqs_hz : np.ndarray
        Frecuencia de cada bin, forma ``(n_bins,)``.
    arm_freqs_hz : np.ndarray
        Frecuencias fundamentales candidatas, forma ``(n_arms,)``.
    cfg : EnvConfig
        Usa ``n_harmonics``, ``beta`` y ``tolerance_semitones``.

    Returns
    -------
    np.ndarray
        Forma ``(n_frames, n_arms)``, valores en [0, 1].
    """
    raise NotImplementedError


def playability_penalty(frets: np.ndarray, prev_fret: int | None, lam: float) -> np.ndarray:
    """λ·|traste − traste_previo|/12 por brazo (0 si ``prev_fret`` es None)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Datos precalculados por segmento y entorno
# ---------------------------------------------------------------------------


@dataclass
class SegmentBanditData:
    """Todo lo que el entorno necesita de un segmento, precalculado una vez.

    Attributes
    ----------
    segment : Segment
        Segmento de origen (tiempos, f0 estimada).
    position : int
        Posición del segmento dentro de la lista de segmentos CONSERVADOS
        (0..n-1); es el índice usado por experimentos y tablatura.
    arms : list[Arm]
        Brazos candidatos (tras la poda ±k).
    frame_indices : np.ndarray
        Índices de frames del :class:`Spectrum` usados (tras saltar el ataque;
        al menos 1).
    salience : np.ndarray
        Forma ``(n_frames_seg, n_arms)``: S_j(f_a) para cada frame y brazo.
    """

    segment: Segment
    position: int
    arms: list[Arm]
    frame_indices: np.ndarray
    salience: np.ndarray

    @property
    def n_arms(self) -> int:
        """Número de brazos candidatos K."""
        return len(self.arms)

    @property
    def mean_salience(self) -> np.ndarray:
        """Media exacta de la saliencia de cada brazo sobre todos los frames, forma ``(K,)``."""
        return self.salience.mean(axis=0)

    @property
    def frets(self) -> np.ndarray:
        """Traste de cada brazo, forma ``(K,)``."""
        return np.array([a.fret for a in self.arms], dtype=int)


def build_segment_data(spectrum: Spectrum, segment: Segment, position: int, cfg: EnvConfig) -> SegmentBanditData:
    """Precalcula brazos candidatos y matriz de saliencia de un segmento conservado."""
    raise NotImplementedError


def build_all_segment_data(spectrum: Spectrum, segments: list[Segment], cfg: EnvConfig) -> list[SegmentBanditData]:
    """:func:`build_segment_data` para cada segmento con ``kept=True``, en orden."""
    raise NotImplementedError


@dataclass
class PullResult:
    """Detalle de un pull (para la ejecución en vivo y el log).

    Attributes
    ----------
    arm_index : int
        Brazo jalado.
    reward : float
        Recompensa observada r.
    frame_index : int
        Índice (dentro de ``SegmentBanditData.frame_indices``) del frame muestreado.
    salience : float
        S_j(f_a) del frame muestreado.
    penalty : float
        Penalización de tocabilidad aplicada.
    noise : float
        Ruido gaussiano añadido (0 si ``noise_std == 0``).
    """

    arm_index: int
    reward: float
    frame_index: int
    salience: float
    penalty: float
    noise: float = 0.0


class BanditEnvironment:
    """Bandit estocástico de un segmento.

    Parameters
    ----------
    data : SegmentBanditData
        Datos precalculados del segmento.
    prev_fret : int | None
        Traste previo (posición de la mano) para la penalización.
    lam : float
        λ de la penalización de tocabilidad.
    rng : np.random.Generator
        Generador para muestrear frames (y ruido). Usar la MISMA semilla para
        todos los algoritmos de una corrida (números aleatorios comunes).
    noise_std : float
        Desviación del ruido gaussiano extra.
    open_string_free : bool
        Si es True, las cuerdas al aire (traste 0) no tienen penalización.

    Attributes
    ----------
    penalties : np.ndarray
        Penalización de cada brazo, forma ``(K,)``.
    true_means : np.ndarray
        μ_a = mean_salience − penalties, forma ``(K,)``.
    optimal_arms : np.ndarray
        Índices de los brazos con μ máximo (empates con tolerancia 1e-12).
    best_mean : float
        μ* = max μ_a.
    """

    def __init__(
        self,
        data: SegmentBanditData,
        prev_fret: int | None,
        lam: float,
        rng: np.random.Generator,
        noise_std: float = 0.0,
        open_string_free: bool = False,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, data: SegmentBanditData, prev_fret: int | None, cfg: EnvConfig, rng: np.random.Generator
    ) -> "BanditEnvironment":
        """Atajo que toma ``lam``, ``noise_std`` y ``open_string_free`` de ``cfg``."""
        return cls(data, prev_fret, cfg.lam, rng, noise_std=cfg.noise_std, open_string_free=cfg.open_string_free)

    @property
    def n_arms(self) -> int:
        """Número de brazos K."""
        raise NotImplementedError

    def pull(self, arm_index: int) -> float:
        """Jala el brazo y devuelve solo la recompensa (camino rápido para experimentos)."""
        raise NotImplementedError

    def pull_detailed(self, arm_index: int) -> PullResult:
        """Jala el brazo y devuelve el detalle completo (frame, saliencia, penalización)."""
        raise NotImplementedError

    def regret_of(self, arm_index: int) -> float:
        """Pseudo-regret instantáneo μ* − μ_a (≥ 0)."""
        raise NotImplementedError

    def is_optimal(self, arm_index: int) -> bool:
        """True si ``arm_index`` está entre los brazos óptimos."""
        raise NotImplementedError


def next_hand_fret(arm: Arm, prev_fret: int | None, cfg: EnvConfig) -> int | None:
    """Posición de la mano tras tocar ``arm``.

    Normalmente es ``arm.fret``; si ``cfg.open_string_free`` y el brazo es una
    cuerda al aire, la mano se queda en ``prev_fret``.
    """
    raise NotImplementedError


__all__ = [
    "Arm", "fret_frequency", "all_arms", "candidate_arms", "Spectrum", "compute_spectrum",
    "HarmonicTemplate", "harmonic_template", "salience", "playability_penalty",
    "SegmentBanditData", "build_segment_data", "build_all_segment_data", "PullResult",
    "BanditEnvironment", "next_hand_fret",
]
