"""Etapa 4 — Segmentación por onsets.

Papel en el pipeline
--------------------
Recibe la señal de análisis (``y_analysis`` de :mod:`src.preprocessing`:
pasa-bajas a 400 Hz + normalización) y la divide en **segmentos de nota**.
Cada segmento conservado será, más adelante, un problema bandit
independiente (:mod:`src.environment`), así que de esta etapa depende cuántos
problemas hay y qué frames del espectro ve cada uno.

El proceso tiene tres pasos:

1. **Función de novedad** (:func:`onset_envelope`): una curva que vale mucho
   cuando "aparece" energía espectral nueva, es decir, cuando se pulsa una
   cuerda.
2. **Selección de picos** (:func:`detect_onsets`): los máximos locales
   suficientemente altos de esa curva son los *onsets* (inicios de nota).
3. **Segmentación** (:func:`segment_audio`): el segmento *i* va del onset *i*
   al onset *i+1* (el último, hasta el final de la pista). El **fin** se
   recorta al último frame cuyo RMS supera el umbral de silencio, para que el
   silencio entre notas no forme parte de la nota. Se marcan como descartados
   (``kept=False``) los segmentos silenciosos y los demasiado cortos; se
   conservan en la lista para poder dibujarlos en gris en la GUI.

Ejemplo
-------
>>> import numpy as np
>>> from src.config import SegmentationConfig
>>> sr = 22050
>>> t = np.arange(int(0.5 * sr)) / sr
>>> nota = np.exp(-t / 0.3) * np.sin(2 * np.pi * 55.0 * t)
>>> y = np.concatenate([np.zeros(sr // 4), nota, np.zeros(sr // 4)]).astype(np.float32)
>>> segs = segment_audio(y, sr, SegmentationConfig(), onsets_s=np.array([0.25]))
>>> len(segs), segs[0].kept, segs[0].reason
(1, True, 'ok')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from src.config import SegmentationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros fijos de la función de novedad (justificados en onset_envelope)
# ---------------------------------------------------------------------------

#: Frecuencia máxima (Hz) del banco de filtros mel de la función de novedad.
#: La señal de análisis está filtrada a 400 Hz; por encima de ~1 kHz solo
#: quedan residuos del filtro, así que esas bandas solo aportarían ruido.
ONSET_FMAX_HZ: float = 1000.0

#: Número de bandas mel entre 0 y :data:`ONSET_FMAX_HZ`. Por debajo de 1 kHz
#: la escala mel es casi lineal: ≈ 25 Hz entre bandas, de modo que incluso
#: los armónicos de E1 (separados 41.2 Hz) caen en bandas distintas, y cada
#: banda abarca varios bins de la FFT (10.8 Hz con ``n_fft = 2048``).
ONSET_N_MELS: int = 40

#: Rango dinámico (dB) del espectrograma log-mel: todo lo que esté más de
#: este valor por debajo del máximo de la pista se recorta a ese "piso".
ONSET_FLOOR_DB: float = 40.0

#: Tamaño de la FFT (muestras) de la función de novedad. 2048 ≈ 93 ms a
#: 22.05 kHz: casi 4 periodos de E1 (41.2 Hz) y resolución de ~10.8 Hz.
ONSET_N_FFT: int = 2048

#: Margen (s) antes del siguiente onset en el que el RMS no se usa para
#: recortar el fin de un segmento: el onset detectado puede llegar ~10–20 ms
#: después del ataque real, y ese ataque no debe contar como "sonido" de la
#: nota anterior.
ONSET_GUARD_S: float = 0.025

#: Piso (dB relativos al máximo) usado al expresar en dB un RMS nulo.
_RMS_DB_FLOOR: float = -120.0


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
        RMS medio del segmento en dB relativos al máximo de la pista
        (0 dB = frame más fuerte de la pista; siempre ≤ 0).
    kept : bool
        False si se descartó (silencio o muy corto).
    reason : str
        ``"ok"``, ``"silencio"`` o ``"corto"``.
    f0_hz : float | None
        f0 estimada por pYIN (la llena :mod:`src.pitch`); None si es desconocida.
    voiced_ratio : float
        Fracción de frames del segmento con voz según pYIN.

    Examples
    --------
    >>> s = Segment(index=0, start_s=1.0, end_s=1.5, start_sample=22050,
    ...             end_sample=33075, rms_db=-6.0, f0_hz=55.0)
    >>> s.duration_s
    0.5
    >>> s.midi
    33.0
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
        """Duración en segundos (``end_s − start_s``)."""
        return self.end_s - self.start_s

    @property
    def midi(self) -> float | None:
        """f0 expresada como número MIDI fraccionario (None si f0 desconocida).

        Usa m = 69 + 12·log₂(f0 / 440 Hz): cada semitono es un factor 2^(1/12)
        en frecuencia y A4 = 440 Hz es el MIDI 69.
        """
        if self.f0_hz is None or self.f0_hz <= 0:
            return None
        return float(69.0 + 12.0 * np.log2(self.f0_hz / 440.0))


# ---------------------------------------------------------------------------
# 1. Función de novedad
# ---------------------------------------------------------------------------


def onset_envelope(
    y: np.ndarray,
    sr: int,
    cfg: SegmentationConfig,
    *,
    fmax_hz: float = ONSET_FMAX_HZ,
    n_mels: int = ONSET_N_MELS,
    floor_db: float = ONSET_FLOOR_DB,
    n_fft: int = ONSET_N_FFT,
) -> np.ndarray:
    """Función de novedad espectral (*spectral flux* log-mel) de ``y``.

    Se calcula con ``librosa.onset.onset_strength`` (``hop_length`` de
    ``cfg``) sobre un espectrograma log-mel adaptado al bajo: bandas solo
    hasta ``fmax_hz`` y rango dinámico limitado a ``floor_db``. Ver *Notes*
    para la teoría y la evaluación que justifica esta elección.

    Parameters
    ----------
    y : np.ndarray
        Señal mono, forma ``(n_muestras,)``; normalmente la señal de análisis
        (filtrada a 400 Hz y normalizada).
    sr : int
        Frecuencia de muestreo en Hz.
    cfg : SegmentationConfig
        Se usa ``cfg.hop_length`` (muestras entre frames).
    fmax_hz : float, optional
        Frecuencia máxima del banco mel en Hz (1000 por defecto; se limita a
        ``sr/2``).
    n_mels : int, optional
        Número de bandas mel (40 por defecto).
    floor_db : float, optional
        Rango dinámico en dB: los valores más de ``floor_db`` dB por debajo
        del máximo de la pista se recortan (40 por defecto).
    n_fft : int, optional
        Tamaño de la FFT en muestras (2048 por defecto).

    Returns
    -------
    np.ndarray
        Novedad por frame, forma ``(n_frames,)`` con
        ``n_frames = 1 + len(y) // hop_length``; valores ≥ 0 en dB/frame
        (promedio sobre bandas). Ceros si la señal es silencio digital.

    Notes
    -----
    **Spectral flux.** Sea L(t, b) = 10·log₁₀ P(t, b) la potencia (dB) de la
    banda mel *b* en el frame *t*. La novedad es la media, sobre las B
    bandas, de los *aumentos* de nivel respecto al frame anterior::

        SF(t) = (1/B) · Σ_b max(0, L(t, b) − L(t−1, b))

    Solo cuentan los aumentos (rectificación de media onda): al pulsar una
    cuerda aparece energía nueva en las bandas de sus armónicos, mientras que
    el decaimiento natural de la nota solo produce diferencias negativas. El
    logaritmo hace que un re-ataque de +4 dB sobre una nota que aún suena
    pese lo mismo en una nota fuerte que en una suave.

    **Por qué no la novedad mel por defecto.** El valor por defecto de
    librosa usa 128 bandas hasta sr/2 = 11 kHz y un piso de −80 dB. Con el
    bajo filtrado a 400 Hz, ~3/4 de esas bandas están casi vacías; en ellas
    el nivel está en el piso de −80 dB y cualquier fuga espectral lo dispara:
    al *terminar* una nota (apagado con la mano) el armónico se ensancha en
    frecuencia y las bandas vecinas "suben" decenas de dB, generando onsets
    falsos en los finales de nota. Por eso:

    * ``fmax ≈ 1000 Hz`` concentra las bandas donde hay bajo
      (fundamentales de 41–196 Hz y sus primeros armónicos);
    * el piso de ``−40 dB`` respecto al máximo de la pista ignora esos
      cambios en bandas casi vacías: que una banda pase de −70 a −50 dB no
      es un ataque, es fuga espectral.

    **Evaluación.** Se compararon, con ``delta = 0.07`` y ``backtrack``,
    sobre (a) 20 pistas sintéticas de 8 notas de bajo filtradas a 400 Hz
    (5 armónicos 1/h, ataque de 5 ms, decaimiento exponencial, silencios
    cortos y re-ataques del mismo pitch sin silencio, con y sin ruido) y
    (b) 12 pistas de 24 notas renderizadas con fluidsynth (bajo GM 33, notas
    repetidas, legato 90 % y 100 %). F-measure media (tolerancia 30 ms en
    sintéticas, 80 ms en fluidsynth por la latencia del soundfont):

    ==========================================  =========
    Función de novedad                          F media
    ==========================================  =========
    mel por defecto (128 bandas, 11 kHz)        0.91
    mel hasta 1 kHz (piso −80 dB)               0.90
    mel por defecto + derivada positiva RMS dB  0.77
    mel 1 kHz con compuerta "RMS sube"          0.90
    **mel 1 kHz, 40 bandas, piso −40 dB**       **0.98**
    ==========================================  =========

    Combinar con la derivada positiva del RMS en dB empeora: normalizada por
    su máximo, queda dominada por el salto silencio→nota (decenas de dB),
    así que un re-ataque de 3–4 dB casi no se ve, y su pico no coincide en el
    tiempo con el del flux (dobles detecciones). Usar el RMS como compuerta
    elimina los onsets falsos de final de nota pero pierde notas en legato
    cuyo ataque es más suave que la cola de la nota anterior (el RMS no
    sube). La opción elegida además fue la más estable al variar ``delta``
    entre 0.04 y 0.15 (F ≥ 0.95).

    **Compromiso del piso.** Un piso más alto (−30/−35 dB) separa aún mejor
    los re-ataques débiles, pero deja de "ver" las notas suaves: con −40 dB
    se detectan notas hasta ≈ 27 dB por debajo de la más fuerte; con −30 dB,
    solo hasta ≈ −18 dB. El piso por defecto de librosa (−80 dB) detecta
    notas aún más suaves a costa de muchos onsets falsos en los finales de
    nota. −40 dB cubre la dinámica habitual de una línea de bajo (rara vez
    hay más de 20–25 dB entre la nota más fuerte y la más suave).

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import SegmentationConfig
    >>> sr = 22050
    >>> y = np.zeros(sr, dtype=np.float32)
    >>> y[sr // 2:] = np.sin(2 * np.pi * 55.0 * np.arange(sr // 2) / sr)
    >>> env = onset_envelope(y, sr, SegmentationConfig())
    >>> env.shape, int(np.argmax(env)) in range(40, 50)  # pico cerca de 0.5 s
    ((87,), True)
    """
    y = np.asarray(y, dtype=np.float32)
    hop = int(cfg.hop_length)
    n_frames = 1 + len(y) // hop
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)

    # Espectrograma mel de potencia restringido a la zona útil del bajo.
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmax=min(float(fmax_hz), sr / 2.0),
        power=2.0,
    )
    if not np.any(mel > 0):
        return np.zeros(n_frames, dtype=np.float32)

    # Potencia → dB relativos al máximo de la pista, con piso en −floor_db.
    log_mel = librosa.power_to_db(S=mel, ref=np.max, top_db=floor_db)

    # Flux espectral: media sobre bandas de los aumentos positivos de nivel.
    # (librosa desplaza la curva n_fft/(2·hop) frames para alinear cada valor
    # con el instante del cambio a pesar de la ventana larga.)
    env = librosa.onset.onset_strength(S=log_mel, sr=sr, hop_length=hop, n_fft=n_fft)
    return env[:n_frames]


# ---------------------------------------------------------------------------
# 2. Detección de onsets (selección de picos)
# ---------------------------------------------------------------------------


def detect_onsets(
    y: np.ndarray,
    sr: int,
    cfg: SegmentationConfig,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Tiempos de onset en segundos (array 1-D ordenado) detectados sobre ``y``.

    Parameters
    ----------
    y : np.ndarray
        Señal mono, forma ``(n_muestras,)`` (señal de análisis).
    sr : int
        Frecuencia de muestreo en Hz.
    cfg : SegmentationConfig
        Usa ``hop_length`` (muestras), ``onset_delta`` (umbral δ sobre la
        novedad normalizada a [0, 1]), ``onset_wait_s`` (separación mínima en
        s) y ``backtrack``.
    envelope : np.ndarray, optional
        Función de novedad ya calculada con :func:`onset_envelope` (para no
        recalcularla si la GUI también la dibuja). Si es None se calcula.

    Returns
    -------
    np.ndarray
        Tiempos de onset en segundos, ``float``, ordenados y sin repetidos.
        Vacío si la señal es silencio.

    Notes
    -----
    ``librosa.onset.onset_detect`` normaliza la novedad a [0, 1] y acepta el
    frame *t* como onset si (1) es el máximo local en una ventana de ~30 ms
    antes y 1 frame después, (2) supera en ``δ`` la media local de la novedad
    (~100 ms antes y después) y (3) han pasado al menos
    ``wait = round(onset_wait_s · sr / hop)`` frames desde el onset anterior.

    El pico del flux llega cuando el ataque ya está dentro de la ventana de
    análisis; con ``backtrack=True`` cada onset se "retrocede" al mínimo
    local anterior de la novedad, que marca el comienzo real de la subida
    (error típico de ±5 ms en las señales de prueba).

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import SegmentationConfig
    >>> detect_onsets(np.zeros(22050, dtype=np.float32), 22050, SegmentationConfig()).size
    0
    """
    y = np.asarray(y, dtype=np.float32)
    hop = int(cfg.hop_length)
    if y.size == 0:
        return np.zeros(0, dtype=float)

    env = onset_envelope(y, sr, cfg) if envelope is None else np.asarray(envelope)
    if not np.any(env > 0):
        logger.info("Onsets: la función de novedad es nula (silencio); 0 onsets.")
        return np.zeros(0, dtype=float)

    wait_frames = int(round(cfg.onset_wait_s * sr / hop))
    onsets = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        hop_length=hop,
        units="time",
        backtrack=bool(cfg.backtrack),
        delta=float(cfg.onset_delta),
        wait=wait_frames,
    )
    # np.unique ordena y elimina duplicados (dos picos pueden retroceder al
    # mismo mínimo).
    onsets = np.unique(np.asarray(onsets, dtype=float))
    logger.info(
        "Onsets: %d detectados (δ=%.3f, separación mínima %d frames ≈ %.0f ms, backtrack=%s).",
        onsets.size, cfg.onset_delta, wait_frames, 1000.0 * wait_frames * hop / sr, cfg.backtrack,
    )
    return onsets


# ---------------------------------------------------------------------------
# 3. Segmentación
# ---------------------------------------------------------------------------


def _rms_db(y: np.ndarray, hop_length: int, frame_length: int) -> tuple[np.ndarray, float]:
    """RMS por frame en dB relativos al máximo de la pista.

    Parameters
    ----------
    y : np.ndarray
        Señal mono.
    hop_length : int
        Salto entre frames en muestras (frame *k* centrado en ``k·hop``).
    frame_length : int
        Longitud de la ventana de RMS en muestras.

    Returns
    -------
    rms_db : np.ndarray
        20·log₁₀(RMS_k / RMS_max) por frame (≤ 0 dB; −120 dB si RMS_k = 0).
    rms_max : float
        RMS lineal del frame más fuerte (0 si la señal es silencio digital).
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0].astype(float)
    rms_max = float(rms.max()) if rms.size else 0.0
    if rms_max <= 0.0:
        return np.full(rms.shape, _RMS_DB_FLOOR), 0.0
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / rms_max)
    return np.maximum(db, _RMS_DB_FLOOR), rms_max


def _frame_span(
    start_sample: int,
    end_sample: int,
    hop: int,
    n_frames: int,
    left_guard: int = 0,
    right_guard: int = 0,
    is_last: bool = False,
) -> tuple[int, int]:
    """Frames ``[k0, k1)`` de RMS que pertenecen al intervalo ``[start, end)``.

    Se eligen los frames cuyo centro ``k·hop`` cumple
    ``start + left_guard ≤ k·hop ≤ end − right_guard``. Con
    ``left_guard = L/2`` la ventana del frame no incluye la cola de la nota
    anterior; con ``right_guard = L/2 + margen`` tampoco alcanza el ataque de
    la siguiente aunque el onset detectado llegue unos ms tarde. En el último
    segmento no hay nota siguiente y se admiten todos los frames hasta el
    final de la señal.

    Si no hay ningún frame así (intervalo muy corto), se usan los frames
    cuyo centro cae en ``[start, end)``; y si tampoco hay, el frame más
    cercano al inicio. Siempre se devuelve al menos un frame.

    Parameters
    ----------
    start_sample, end_sample : int
        Intervalo en muestras (``end`` exclusivo).
    hop : int
        Salto entre frames en muestras.
    n_frames : int
        Número total de frames de RMS.
    left_guard, right_guard : int, optional
        Muestras excluidas al principio y al final del intervalo.
    is_last : bool, optional
        True para el último segmento (sin restricción por la derecha).

    Returns
    -------
    tuple[int, int]
        ``(k0, k1)`` con ``0 ≤ k0 < k1 ≤ n_frames``.
    """
    k0 = -(-(start_sample + left_guard) // hop)  # ceil((start + guarda) / hop)
    k1 = n_frames if is_last else (end_sample - right_guard) // hop + 1  # floor(...) + 1
    k0, k1 = min(k0, n_frames - 1), min(k1, n_frames)
    if k1 <= k0:  # intervalo más corto que una ventana: basta con el centro
        k0 = min(-(-start_sample // hop), n_frames - 1)
        k1 = min(-(-end_sample // hop), n_frames)
    if k1 <= k0:
        k0 = min(start_sample // hop, n_frames - 1)
        k1 = k0 + 1
    return k0, k1


def segment_audio(
    y: np.ndarray,
    sr: int,
    cfg: SegmentationConfig,
    onsets_s: np.ndarray | None = None,
    rms_frame_length: int | None = None,
) -> list[Segment]:
    """Divide ``y`` en segmentos usando ``onsets_s`` (o los detecta si es None).

    Devuelve TODOS los segmentos (conservados y descartados), ordenados, con
    ``index`` consecutivo desde 0.

    Parameters
    ----------
    y : np.ndarray
        Señal mono, forma ``(n_muestras,)`` (señal de análisis).
    sr : int
        Frecuencia de muestreo en Hz.
    cfg : SegmentationConfig
        Usa ``hop_length`` (muestras), ``rms_threshold_db`` (umbral de
        silencio en dB relativos al RMS máximo, p. ej. −35 dB) y
        ``min_duration_s`` (s); y los parámetros de onsets si hay que
        detectarlos.
    onsets_s : np.ndarray, optional
        Tiempos de onset en segundos. Si es None se llama a
        :func:`detect_onsets`. Se ordenan, se descartan los que caen fuera de
        la señal y los que coinciden en la misma muestra.
    rms_frame_length : int, optional
        Ventana del RMS en muestras. Por defecto ``2·hop_length`` (≈ 23 ms
        a 22.05 kHz): ver *Notes*.

    Returns
    -------
    list[Segment]
        Segmentos en orden temporal. Lista vacía si la señal está vacía o es
        silencio digital.

    Notes
    -----
    Para cada onset *i*:

    1. Intervalo bruto ``[onset_i, onset_{i+1})`` (el último llega hasta el
       final de la señal).
    2. Umbral de silencio: un frame "suena" si
       RMS_dB(k) ≥ max_k RMS_dB + ``rms_threshold_db``.
    3. Si ningún frame del intervalo suena → ``kept=False``,
       ``reason="silencio"`` (se conserva el intervalo bruto para la GUI).
    4. Si no, el **fin** se recorta al último frame que suena (+1 frame):
       así la cola silenciosa entre notas no se analiza como parte de la nota.
       Si el último frame del intervalo todavía suena, la nota dura hasta el
       siguiente onset y no se recorta.
    5. Si la duración resultante es menor que ``min_duration_s`` →
       ``kept=False``, ``reason="corto"`` (no hay frames suficientes para
       pYIN ni para muestrear recompensas).

    También se marca como silencio un segmento cuyo RMS medio (ya recortado)
    queda bajo el umbral, como indica :class:`SegmentationConfig`.

    Si no hay onsets pero hay audio, se crea un único segmento que empieza en
    el primer frame que suena.

    **Ventana del RMS.** Cada frame de RMS está centrado en ``k·hop`` y mide
    la energía de ±``frame_length/2`` muestras alrededor. Con la ventana de
    2048 muestras (93 ms) los frames de un silencio corto "ven" la cola de la
    nota anterior y el ataque de la siguiente, y no habría nada que recortar
    en huecos de menos de ~90 ms. Con ``2·hop`` (23 ms) la "fuga" es de
    ±12 ms; para E1 (41.2 Hz, periodo de 24 ms) la ondulación del RMS por no
    abarcar un número entero de periodos es < 0.5 dB, despreciable frente a
    un umbral de −35 dB. Además, de cada intervalo solo se miran los frames
    cuya ventana completa cabe entre sus dos onsets, dejando un margen extra
    de :data:`ONSET_GUARD_S` antes del siguiente onset (que puede detectarse
    unos ms tarde), de modo que el ataque de la nota siguiente nunca cuenta
    como "sonido" de la actual.

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import SegmentationConfig
    >>> sr = 22050
    >>> t = np.arange(int(0.4 * sr)) / sr
    >>> nota = np.sin(2 * np.pi * 73.42 * t)
    >>> y = np.concatenate([nota, np.zeros(sr // 2), nota]).astype(np.float32)
    >>> segs = segment_audio(y, sr, SegmentationConfig(), onsets_s=np.array([0.0, 0.6, 0.9]))
    >>> [(s.index, s.kept, s.reason) for s in segs]
    [(0, True, 'ok'), (1, False, 'silencio'), (2, True, 'ok')]
    >>> round(segs[0].end_s, 2)  # recortado al final real de la nota (0.4 s)
    0.42
    """
    y = np.asarray(y, dtype=np.float32)
    n_samples = int(y.size)
    hop = int(cfg.hop_length)
    if n_samples == 0:
        logger.warning("Segmentación: la señal está vacía; no hay segmentos.")
        return []

    frame_length = int(rms_frame_length) if rms_frame_length is not None else 2 * hop
    half_window = frame_length // 2
    onset_guard = int(round(ONSET_GUARD_S * sr))
    rms_db, rms_max = _rms_db(y, hop, frame_length)
    if rms_max <= 0.0:
        logger.warning("Segmentación: la señal es silencio digital; no hay segmentos.")
        return []
    n_frames = rms_db.size
    # rms_db ya es relativo al máximo (0 dB), así que el umbral
    # max_dB + rms_threshold_db es simplemente rms_threshold_db.
    sounding = rms_db >= cfg.rms_threshold_db

    if onsets_s is None:
        onsets_s = detect_onsets(y, sr, cfg)
    onsets_arr = np.sort(np.asarray(onsets_s, dtype=float).ravel())
    n_onsets = int(onsets_arr.size)

    # Onsets → muestras de inicio (dentro de la señal y sin repetidos).
    starts = np.unique(np.clip(np.round(onsets_arr * sr).astype(np.int64), 0, None))
    starts = starts[starts < n_samples]
    if starts.size == 0:
        if not sounding.any():  # pragma: no cover - el frame máximo siempre suena si umbral ≤ 0
            logger.info("Segmentación: 0 onsets y ningún frame supera el umbral; no hay segmentos.")
            return []
        first = int(np.argmax(sounding))
        starts = np.array([min(first * hop, n_samples - 1)], dtype=np.int64)
        logger.info(
            "Segmentación: sin onsets; se usa un único segmento desde el primer frame "
            "sobre el umbral (%.3f s).", starts[0] / sr,
        )
    ends = np.append(starts[1:], n_samples)

    segments: list[Segment] = []
    for index, (start, raw_end) in enumerate(zip(starts.tolist(), ends.tolist())):
        is_last = index == starts.size - 1
        k0, k1 = _frame_span(
            start, raw_end, hop, n_frames,
            left_guard=half_window, right_guard=half_window + onset_guard, is_last=is_last,
        )
        seg_sounding = sounding[k0:k1]

        reason = "ok"
        end = raw_end
        if not seg_sounding.any():
            reason = "silencio"
        else:
            k_last = k0 + int(np.flatnonzero(seg_sounding)[-1])
            if k_last < k1 - 1:
                # La nota se apaga antes del siguiente onset: el fin es el
                # borde derecho del último frame que suena (+1 frame).
                end = min(raw_end, (k_last + 1) * hop)
            # (si el último frame mirado aún suena, la nota llega hasta el
            # siguiente onset y el fin bruto se conserva)
            k1 = k_last + 1

        # RMS medio (lineal) del segmento, expresado en dB relativos al máximo.
        mean_rms = float(np.mean(10.0 ** (rms_db[k0:k1] / 20.0)))
        seg_rms_db = max(20.0 * np.log10(max(mean_rms, 1e-12)), _RMS_DB_FLOOR)
        if reason == "ok" and seg_rms_db < cfg.rms_threshold_db:
            reason = "silencio"
        if reason == "ok" and (end - start) / sr < cfg.min_duration_s:
            reason = "corto"

        segments.append(
            Segment(
                index=index,
                start_s=start / sr,
                end_s=end / sr,
                start_sample=int(start),
                end_sample=int(end),
                rms_db=float(seg_rms_db),
                kept=reason == "ok",
                reason=reason,
            )
        )
        logger.debug(
            "Segmento detectado n.º %d: %.3f–%.3f s (%.0f ms), RMS medio %.1f dB → %s",
            index, start / sr, end / sr, 1000.0 * (end - start) / sr, seg_rms_db, reason,
        )

    n_kept = sum(s.kept for s in segments)
    n_silence = sum(s.reason == "silencio" for s in segments)
    n_short = sum(s.reason == "corto" for s in segments)
    logger.info(
        "Segmentación: %d onsets, %d segmentos conservados, %d silencios, %d cortos "
        "(umbral %.0f dB, duración mínima %.0f ms).",
        n_onsets, n_kept, n_silence, n_short, cfg.rms_threshold_db, 1000.0 * cfg.min_duration_s,
    )
    return segments


def kept_segments(segments: list[Segment]) -> list[Segment]:
    """Filtra los segmentos con ``kept=True`` (conserva el orden).

    Parameters
    ----------
    segments : list[Segment]
        Lista completa devuelta por :func:`segment_audio`.

    Returns
    -------
    list[Segment]
        Solo los conservados, en el mismo orden (su posición en esta lista es
        la que usan el entorno bandit, los experimentos y la tablatura).
    """
    return [s for s in segments if s.kept]
