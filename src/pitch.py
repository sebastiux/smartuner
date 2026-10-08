"""Etapa 5 — Estimación de pitch con pYIN.

Papel en el pipeline
--------------------
Recibe la señal de análisis (filtrada a 400 Hz, :mod:`src.preprocessing`) y
los segmentos de :mod:`src.segmentation`, y asigna a cada segmento conservado
una frecuencia fundamental f0 (Hz) y la fracción de sus frames con voz.

Se ejecuta ``librosa.pyin`` UNA vez sobre la señal completa (más eficiente y
con mejor continuidad que por segmento: el HMM de pYIN aprovecha el contexto
temporal) y luego se resume cada segmento con la **mediana** de la f0 de sus
frames con voz, ignorando el ataque.

La f0 solo se usa para PODAR brazos (±k semitonos alrededor de la nota
estimada, ver :mod:`src.environment`); la decisión final de cuerda/traste la
toman los agentes bandit. Si la f0 es desconocida (pocos frames con voz) no
se poda y el agente explora los 52 brazos.

Teoría: YIN y pYIN
------------------
**YIN** (de Cheveigné y Kawahara, 2002) busca el periodo τ que mejor repite
la señal dentro de una ventana. Para cada retardo τ calcula la función
diferencia::

    d(τ) = Σ_j (x_j − x_{j+τ})²

que es ≈ 0 cuando τ es un múltiplo del periodo. Para no favorecer τ
pequeños se normaliza por su media acumulada::

    d'(τ) = d(τ) / [(1/τ) · Σ_{k=1..τ} d(k)],   d'(0) = 1

y se toma el primer mínimo de d' por debajo de un umbral (p. ej. 0.1);
f0 = sr / τ.

**pYIN** (Mauch y Dixon, 2014) no fija un único umbral: lo recorre con una
distribución Beta y obtiene, por frame, varios candidatos de f0 con su
probabilidad. Después un **HMM** con estados (pitch × {con voz, sin voz})
y transiciones que favorecen cambios pequeños de pitch elige, con Viterbi,
la trayectoria más probable. Resultado por frame: f0, decisión con/sin voz y
la probabilidad de voz.

Ejemplo
-------
>>> import numpy as np
>>> from src.config import PitchConfig
>>> from src.segmentation import Segment
>>> sr = 22050
>>> t = np.arange(int(0.5 * sr)) / sr
>>> y = (0.5 * np.sin(2 * np.pi * 55.0 * t)).astype(np.float32)
>>> track = estimate_pitch_track(y, sr, PitchConfig())
>>> seg = Segment(index=0, start_s=0.0, end_s=0.5, start_sample=0, end_sample=len(y), rms_db=0.0)
>>> f0, ratio = segment_f0(track, seg, PitchConfig())
>>> round(f0), midi_to_name(hz_to_midi(f0)), ratio > 0.9
(55, 'A1', True)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import librosa
import numpy as np

from src.config import PitchConfig
from src.segmentation import Segment

logger = logging.getLogger(__name__)

#: Probabilidad de voz mínima (0–1) para aceptar un frame como "con voz".
#: El Viterbi de pYIN a veces marca como sonoros tramos de ruido blanco con
#: probabilidad de voz ≈ 0.01 (y f0 pegada a ``fmin``); en notas reales la
#: probabilidad de los frames sonoros es casi siempre > 0.1. Exigir ambas
#: cosas elimina esos falsos positivos sin perder frames útiles.
MIN_VOICED_PROB: float = 0.1

_NOTE_NAMES: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class PitchTrack:
    """Trayectoria de f0 de toda la pista.

    Attributes
    ----------
    times_s : np.ndarray
        Centro de cada frame en segundos, forma ``(n_frames,)``.
    f0_hz : np.ndarray
        f0 por frame en Hz (NaN donde no hay voz).
    voiced_prob : np.ndarray
        Probabilidad de voz por frame (0–1).
    voiced : np.ndarray
        Booleano por frame: pYIN lo marcó con voz **y** su probabilidad de
        voz es ≥ :data:`MIN_VOICED_PROB`.
    """

    times_s: np.ndarray
    f0_hz: np.ndarray
    voiced_prob: np.ndarray
    voiced: np.ndarray

    @property
    def n_frames(self) -> int:
        """Número de frames de la trayectoria."""
        return int(self.times_s.size)


# ---------------------------------------------------------------------------
# Conversiones Hz ↔ MIDI ↔ nombre de nota
# ---------------------------------------------------------------------------


def hz_to_midi(f_hz: float) -> float:
    """Convierte Hz a número MIDI fraccionario (A4 = 440 Hz = 69).

    Parameters
    ----------
    f_hz : float
        Frecuencia en Hz (> 0).

    Returns
    -------
    float
        m = 69 + 12·log₂(f / 440). Cada semitono multiplica la frecuencia por
        2^(1/12) ≈ 1.0595, así que la escala MIDI es logarítmica en Hz.

    Raises
    ------
    ValueError
        Si ``f_hz`` no es un número positivo y finito.

    Examples
    --------
    >>> hz_to_midi(440.0)
    69.0
    >>> hz_to_midi(55.0)
    33.0
    >>> round(hz_to_midi(41.2), 2)  # E1, cuerda al aire más grave del bajo
    28.0
    """
    if not np.isfinite(f_hz) or f_hz <= 0:
        raise ValueError(f"La frecuencia debe ser positiva y finita (se recibió {f_hz} Hz).")
    return float(69.0 + 12.0 * np.log2(f_hz / 440.0))


def midi_to_hz(midi: float) -> float:
    """Convierte número MIDI (posiblemente fraccionario) a Hz.

    Parameters
    ----------
    midi : float
        Número MIDI (69 = A4).

    Returns
    -------
    float
        f = 440 · 2^((m − 69) / 12) en Hz.

    Examples
    --------
    >>> midi_to_hz(69)
    440.0
    >>> midi_to_hz(33)  # A1, cuerda La al aire
    55.0
    >>> round(midi_to_hz(28), 2)  # E1
    41.2
    """
    return float(440.0 * 2.0 ** ((midi - 69.0) / 12.0))


def midi_to_name(midi: int) -> str:
    """Nombre de nota en notación anglosajona con octava, p. ej. 33 → ``"A1"``.

    Parameters
    ----------
    midi : int
        Número MIDI; si es fraccionario se redondea al semitono más cercano.

    Returns
    -------
    str
        Nombre con sostenidos y número de octava científica (C4 = 60).

    Examples
    --------
    >>> [midi_to_name(m) for m in (28, 33, 38, 43)]  # cuerdas al aire del bajo
    ['E1', 'A1', 'D2', 'G2']
    >>> midi_to_name(61), midi_to_name(32.6)
    ('C#4', 'A1')
    """
    m = int(round(midi))
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


# ---------------------------------------------------------------------------
# pYIN sobre toda la pista
# ---------------------------------------------------------------------------


def estimate_pitch_track(
    y: np.ndarray,
    sr: int,
    cfg: PitchConfig,
    hop_length: int = 256,
    min_voiced_prob: float = MIN_VOICED_PROB,
) -> PitchTrack:
    """Ejecuta pYIN sobre toda la señal y devuelve la trayectoria de f0.

    Parameters
    ----------
    y : np.ndarray
        Señal mono, forma ``(n_muestras,)`` (señal de análisis filtrada).
    sr : int
        Frecuencia de muestreo en Hz.
    cfg : PitchConfig
        Usa ``fmin_hz`` y ``fmax_hz`` (rango de búsqueda en Hz; 35–250 Hz
        cubre de E1 = 41.2 Hz a G2 traste 12 = 196 Hz con margen) y
        ``frame_length`` (ventana de pYIN en muestras; 2048 ≈ 93 ms abarca
        casi 4 periodos de E1), además de ``max_transition_rate``
        (octavas/s) y ``switch_prob``, que controlan las transiciones del HMM
        (sus valores por defecto evitan errores de octava tras saltos; ver
        :class:`src.config.PitchConfig`).
    hop_length : int, optional
        Salto entre frames en muestras (256 ≈ 11.6 ms; conviene usar el
        mismo que la segmentación).
    min_voiced_prob : float, optional
        Probabilidad de voz mínima para considerar un frame con voz (ver
        :data:`MIN_VOICED_PROB`).

    Returns
    -------
    PitchTrack
        Tiempos (centro de cada frame, ``librosa.times_like``), f0 en Hz
        (NaN sin voz), probabilidad de voz y máscara de frames con voz.

    Notes
    -----
    pYIN es la etapa más lenta del análisis (≈ 0.02–0.1 s por segundo de
    audio tras la primera llamada, que compila el código con numba). Por eso
    se llama una sola vez por pista y la GUI la ejecuta en un hilo aparte.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        empty = np.zeros(0, dtype=float)
        return PitchTrack(empty, empty.copy(), empty.copy(), np.zeros(0, dtype=bool))

    t0 = time.perf_counter()
    # max_transition_rate y switch_prob controlan las transiciones del HMM:
    # un bajo salta más de una octava entre notas cortas y el valor de librosa
    # (35.92 oct/s) hace que la f0 de la nota anterior "se arrastre" por la
    # suboctava (ver PitchConfig).
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=y,
        fmin=float(cfg.fmin_hz),
        fmax=float(cfg.fmax_hz),
        sr=sr,
        frame_length=int(cfg.frame_length),
        hop_length=int(hop_length),
        max_transition_rate=float(cfg.max_transition_rate),
        switch_prob=float(cfg.switch_prob),
    )
    elapsed = time.perf_counter() - t0
    times = librosa.times_like(X=f0, sr=sr, hop_length=hop_length)

    # Un frame cuenta como "con voz" si el Viterbi de pYIN lo decidió así Y
    # su probabilidad de voz no es despreciable (filtra ruido, ver arriba).
    voiced = np.asarray(voiced_flag, dtype=bool) & (voiced_prob >= min_voiced_prob) & np.isfinite(f0)
    f0 = np.where(voiced, f0, np.nan)

    logger.info(
        "pYIN: %d frames (%.1f s de audio, %.0f–%.0f Hz) calculados en %.2f s; %.0f %% de frames con voz.",
        f0.size, y.size / sr, cfg.fmin_hz, cfg.fmax_hz, elapsed, 100.0 * float(voiced.mean()) if voiced.size else 0.0,
    )
    return PitchTrack(
        times_s=np.asarray(times, dtype=float),
        f0_hz=np.asarray(f0, dtype=float),
        voiced_prob=np.asarray(voiced_prob, dtype=float),
        voiced=voiced,
    )


# ---------------------------------------------------------------------------
# Resumen por segmento
# ---------------------------------------------------------------------------


def _segment_frame_mask(times_s: np.ndarray, start_s: float, end_s: float, attack_skip_s: float) -> np.ndarray:
    """Máscara de los frames de pYIN que se usan para resumir un segmento.

    Frames con centro en ``[start + attack_skip, end)``; si no hay ninguno
    (segmento más corto que el ataque), todos los de ``[start, end)``; y si
    tampoco hay (segmento más corto que un hop), el frame más cercano a su
    punto medio.
    """
    mask = (times_s >= start_s + attack_skip_s) & (times_s < end_s)
    if not mask.any():
        mask = (times_s >= start_s) & (times_s < end_s)
    if not mask.any() and times_s.size:
        mask = np.zeros(times_s.shape, dtype=bool)
        mask[int(np.argmin(np.abs(times_s - 0.5 * (start_s + end_s))))] = True
    return mask


def segment_f0(
    track: PitchTrack,
    segment: Segment,
    cfg: PitchConfig,
    attack_skip_s: float = 0.03,
) -> tuple[float | None, float]:
    """Resume la f0 de un segmento: ``(mediana de f0 con voz o None, fracción con voz)``.

    Devuelve ``None`` como f0 si la fracción con voz es menor que
    ``cfg.min_voiced_ratio``.

    Parameters
    ----------
    track : PitchTrack
        Trayectoria de pYIN de toda la pista.
    segment : Segment
        Segmento a resumir (se usan ``start_s`` y ``end_s``).
    cfg : PitchConfig
        Usa ``min_voiced_ratio`` (fracción mínima de frames con voz, 0–1).
    attack_skip_s : float, optional
        Segundos iniciales ignorados (ataque de la púa/dedo), 0.03 s por
        defecto. Si el segmento es más corto, se usa completo.

    Returns
    -------
    f0_hz : float | None
        Mediana de la f0 (Hz) de los frames con voz, o None si es desconocida.
    voiced_ratio : float
        Fracción de los frames considerados que tienen voz (0–1).

    Notes
    -----
    **¿Por qué saltar el ataque?** Los primeros ~30 ms de una nota pulsada
    son un transitorio casi inarmónico (golpe de la púa/dedo, ruido de traste)
    donde pYIN puede dar f0 erráticas.

    **¿Por qué la mediana?** Es robusta a valores atípicos: si unos pocos
    frames tienen un error de octava (f0 al doble o a la mitad) o el final de
    la nota se desafina, la mediana sigue en el pitch de la mayoría de frames,
    mientras que la media se desplazaría hacia el error.

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import PitchConfig
    >>> from src.segmentation import Segment
    >>> t = np.arange(10) * 0.05
    >>> f0 = np.array([110.0, 55.0, 55.2, 54.9, 55.1, np.nan, np.nan, np.nan, np.nan, np.nan])
    >>> track = PitchTrack(t, f0, np.full(10, 0.9), np.isfinite(f0))
    >>> seg = Segment(index=0, start_s=0.0, end_s=0.5, start_sample=0, end_sample=11025, rms_db=0.0)
    >>> segment_f0(track, seg, PitchConfig())  # el ataque (110 Hz) se ignora
    (55.05, 0.4444444444444444)
    """
    mask = _segment_frame_mask(track.times_s, segment.start_s, segment.end_s, attack_skip_s)
    n_frames = int(mask.sum())
    if n_frames == 0:
        return None, 0.0

    f0_seg = track.f0_hz[mask]
    voiced_seg = track.voiced[mask] & np.isfinite(f0_seg)
    voiced_ratio = float(voiced_seg.sum()) / n_frames
    if not voiced_seg.any() or voiced_ratio < cfg.min_voiced_ratio:
        return None, voiced_ratio
    return float(np.median(f0_seg[voiced_seg])), voiced_ratio


def segment_log_name(position: int, index: int) -> str:
    """Nombre de un segmento conservado en el log: ``"Segmento 3"`` (+ su n.º de detección si difiere).

    Todo el sistema (entorno, tablatura, experimentos, GUI) numera los
    segmentos por su POSICIÓN entre los conservados. La segmentación, en
    cambio, numera todos los detectados (también los que luego descarta como
    silencio o demasiado cortos); si ambos números difieren se añade el de
    detección para poder seguir el log de principio a fin.

    Parameters
    ----------
    position : int
        Posición entre los segmentos conservados (0, 1, 2, ...).
    index : int
        Índice del segmento entre todos los detectados (``Segment.index``).

    Returns
    -------
    str
        Texto para el log.

    Examples
    --------
    >>> segment_log_name(3, 3), segment_log_name(3, 4)
    ('Segmento 3', 'Segmento 3 [detectado n.º 4]')
    """
    if int(position) == int(index):
        return f"Segmento {int(position)}"
    return f"Segmento {int(position)} [detectado n.º {int(index)}]"


def annotate_segments(
    track: PitchTrack,
    segments: list[Segment],
    cfg: PitchConfig,
    attack_skip_s: float = 0.03,
) -> None:
    """Rellena ``f0_hz`` y ``voiced_ratio`` de cada segmento conservado (in place).

    Los segmentos descartados (``kept=False``) no se modifican. El log
    numera cada segmento por su posición entre los conservados, igual que el
    entorno y la tablatura (:func:`segment_log_name`).

    Parameters
    ----------
    track : PitchTrack
        Trayectoria de pYIN de toda la pista (:func:`estimate_pitch_track`).
    segments : list[Segment]
        Lista completa de segmentos (:func:`src.segmentation.segment_audio`).
    cfg : PitchConfig
        Usa ``min_voiced_ratio``.
    attack_skip_s : float, optional
        Segundos iniciales de cada segmento que se ignoran (ver
        :func:`segment_f0`).

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import PitchConfig
    >>> from src.segmentation import Segment
    >>> t = np.arange(20) * 0.05
    >>> f0 = np.full(20, 98.0)
    >>> track = PitchTrack(t, f0, np.ones(20), np.ones(20, dtype=bool))
    >>> segs = [Segment(0, 0.0, 0.5, 0, 11025, -3.0),
    ...         Segment(1, 0.5, 0.52, 11025, 11466, -3.0, kept=False, reason="corto")]
    >>> annotate_segments(track, segs, PitchConfig())
    >>> segs[0].f0_hz, segs[1].f0_hz
    (98.0, None)
    """
    n_known = 0
    n_kept = 0
    for seg in segments:
        if not seg.kept:
            continue
        # Mismo número que en el entorno, la tablatura y los experimentos: la
        # posición entre los segmentos CONSERVADOS (ver segment_log_name).
        name = segment_log_name(n_kept, seg.index)
        n_kept += 1
        f0, ratio = segment_f0(track, seg, cfg, attack_skip_s=attack_skip_s)
        seg.f0_hz = f0
        seg.voiced_ratio = ratio
        if f0 is None:
            logger.info(
                "%s: f0 desconocida (%.0f %% de frames con voz < %.0f %% mínimo); no se podarán brazos.",
                name, 100.0 * ratio, 100.0 * cfg.min_voiced_ratio,
            )
        else:
            n_known += 1
            logger.info(
                "%s: f0=%.1f Hz (%s), %.0f %% de frames con voz",
                name, f0, midi_to_name(hz_to_midi(f0)), 100.0 * ratio,
            )
    logger.info("pYIN: f0 asignada a %d de %d segmentos conservados.", n_known, n_kept)
