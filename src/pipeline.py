"""Orquestación de las etapas 1–6: del archivo de audio a los problemas bandit.

Papel en el pipeline
--------------------
:func:`analyze` es la ÚNICA puerta de entrada al análisis de una pista: la GUI
y la CLI la llaman y reciben un :class:`AnalysisResult` con todo lo que
necesitan los agentes, las gráficas y la evaluación. Ninguna de las dos
conoce el orden interno de las etapas, que es::

    archivo.mp3
       │ 1. Carga (src.io_audio): ffmpeg (o soundfile) → mono float32 a cfg.audio.sample_rate
       │ 2. Separación opcional (src.separation): Demucs o HPSS aíslan el bajo de una mezcla
       ▼
    y_raw ──3. Preprocesamiento (src.preprocessing)──┬──► y_analysis (pasa-bajas + normalizada)
                                                     └──► y_spectral (solo normalizada)
    y_analysis ──4. Segmentación (src.segmentation)──► onsets, segmentos (conservados/descartados)
    y_analysis ──5. Pitch pYIN (src.pitch)───────────► trayectoria f0 → f0 de cada segmento
    y_spectral ──6. Espectro + datos bandit (src.environment)──► SegmentBanditData por segmento
                                                     +
    <nombre>.gt.json junto al audio ──► ground truth (src.synth_dataset), si existe

¿Por qué dos señales? (decisión de diseño 1)
--------------------------------------------
El pasa-bajas de 400 Hz deja solo la fundamental y los primeros armónicos:
eso es lo que quieren la detección de onsets y pYIN (menos ruido de trastes y
de armónicos agudos que confunden la octava). La recompensa, en cambio, mide
la energía de hasta N = 5 armónicos (5 × 196 Hz = 980 Hz para G-12), así que el
espectro se calcula sobre la señal SIN filtrar.

Coherencia temporal
-------------------
Onsets, pYIN y espectro usan el mismo salto ``cfg.segmentation.hop_length``
(256 muestras ≈ 11.6 ms a 22 050 Hz): un frame de pYIN y un frame del espectro
corresponden al mismo instante, lo que simplifica las gráficas y la selección
de frames de cada segmento.

Separación del bajo (etapa 2)
-----------------------------
Solo si ``cfg.audio.separate_bass``. El método sale de
``cfg.audio.separation_method`` (:func:`src.separation.resolve_method`):
``"auto"`` usa Demucs si está instalado y, si no, HPSS; ``"demucs"`` exige
Demucs (si falta, :class:`src.separation.SeparationError` con instrucciones).

* **Demucs** escribe un stem ``<caché>/<clave>_bass.wav``: ``source_path`` es
  ese archivo y ``y_raw`` se vuelve a cargar desde él.
* **HPSS** trabaja sobre la señal ya cargada y NO escribe ningún archivo:
  ``y_raw`` pasa a ser la señal separada y ``source_path`` sigue siendo el
  archivo original (no hay otro archivo que señalar).

Por eso ``source_path`` no basta para saber si se separó: el método que se
usó de verdad queda en ``AnalysisResult.separation_method`` (``"demucs"``,
``"hpss"`` o None). Con ``"auto"``, si el proceso de Demucs no logra importar
PyTorch (p. ej. un torch roto en Windows: ``WinError 126``), se avisa y se
recurre a HPSS en lugar de fallar.

Progreso y cancelación
----------------------
Cada etapa informa ``progress(fracción, mensaje)`` con fracciones ponderadas
por su coste aproximado (pYIN y Demucs son las más lentas) y, ANTES de cada
etapa, se consulta ``cancel``: si está activo se lanza
:class:`src.config.CancelledError`. Las etapas no se interrumpen a la mitad
(salvo Demucs, que vigila ``cancel`` por su cuenta, y pYIN, que en pistas
largas va por bloques y lo consulta entre bloque y bloque).

:func:`rebuild_segment_data` permite cambiar los parámetros del entorno (k, N,
β, tolerancia, tipo de espectro...) y obtener nuevos problemas bandit sin
volver a decodificar, segmentar ni ejecutar pYIN.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import CancelledError, Config, ProgressCallback
from src.environment import SegmentBanditData, Spectrum
from src.pitch import PitchTrack
from src.segmentation import Segment
from src.synth_dataset import GTNote

logger = logging.getLogger(__name__)

#: Peso relativo (≈ coste de cómputo) de cada etapa, para repartir la barra de
#: progreso. Solo importan las proporciones; ``separation`` (Demucs: minutos en
#: CPU) o ``separation_hpss`` (HPSS: segundos, ~1/5 de lo que tarda pYIN) se
#: omiten si no se separa el bajo.
STAGE_WEIGHTS: dict[str, float] = {
    "load": 0.06,
    "separation": 0.60,
    "separation_hpss": 0.08,
    "preprocess": 0.03,
    "segmentation": 0.08,
    "pitch": 0.45,
    "spectrum": 0.15,
    "bandit": 0.10,
    "ground_truth": 0.02,
}

#: Nivel mínimo (dBFS) del frame más fuerte de la pista ORIGINAL (RMS en
#: ventanas de 2048 muestras ≈ 93 ms) para considerar que contiene música. El
#: preprocesamiento normaliza el pico y la segmentación mide el silencio en dB
#: RELATIVOS al máximo de la pista, así que, sin este umbral absoluto, una
#: pista de ruido a −90 dBFS (o un stem de Demucs de una canción sin bajo)
#: se convertiría en "música" a escala completa y en una tablatura inventada.
#: Un bajo grabado normalmente tiene frames de −30…−10 dBFS.
MIN_TRACK_RMS_DBFS: float = -60.0

#: Parámetros de :class:`src.config.EnvConfig` que determinan el ESPECTRO (si
#: cambian, :func:`rebuild_segment_data` lo recalcula). Los de la CQT solo
#: importan con ``spectrum == "cqt"`` y los de la STFT solo con ``"stft"``:
#: ``n_fft`` y, además, los que fijan la banda guardada
#: (:func:`src.environment.stft_max_frequency`).
_CQT_PARAMS: tuple[str, ...] = ("bins_per_octave", "cqt_fmin_hz", "n_octaves")
_STFT_PARAMS: tuple[str, ...] = ("n_fft", "n_harmonics", "tolerance_semitones", "n_frets")


@dataclass
class AnalysisResult:
    """Resultado del análisis de una pista.

    Attributes
    ----------
    path : Path
        Archivo de entrada elegido por el usuario.
    source_path : Path
        Archivo efectivamente analizado: el stem de Demucs si se separó con
        Demucs; con HPSS (que no escribe archivos) o sin separación, el
        archivo original.
    sr : int
        Frecuencia de muestreo (Hz).
    y_raw : np.ndarray
        Señal decodificada (mono, sin preprocesar), forma ``(n_muestras,)``.
        Con separación es el bajo aislado (el stem de Demucs o la salida de
        :func:`src.separation.hpss_bass`).
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
    separation_method : str | None
        Separación que se aplicó DE VERDAD: ``"demucs"``, ``"hpss"`` o None
        (sin separar). Con ``audio.separation_method = "auto"`` es el método
        resuelto (HPSS si Demucs no está instalado o no pudo importarse).

    Examples
    --------
    >>> from src.config import Config
    >>> res = analyze("data/synthetic/linea_simple.mp3", Config())   # doctest: +SKIP
    >>> len(res.kept), res.ground_truth is not None                  # doctest: +SKIP
    (16, True)
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
    separation_method: str | None = None

    @property
    def kept(self) -> list[Segment]:
        """Segmentos conservados, en orden (su posición es la de ``segment_data``)."""
        return [s for s in self.segments if s.kept]

    @property
    def duration_s(self) -> float:
        """Duración de la pista en segundos."""
        return len(self.y_raw) / float(self.sr)


# ---------------------------------------------------------------------------
# Progreso y cancelación
# ---------------------------------------------------------------------------


class _StageReporter:
    """Reparte la barra de progreso [0, 1] entre las etapas según su peso.

    Parameters
    ----------
    stages : list[str]
        Claves de :data:`STAGE_WEIGHTS` en el orden en que se ejecutarán.
    progress : ProgressCallback | None
        Callback del usuario (None = no informar).
    cancel : threading.Event | None
        Evento de cancelación que se consulta al iniciar cada etapa.
    """

    def __init__(self, stages: list[str], progress: ProgressCallback | None, cancel: threading.Event | None) -> None:
        self._progress = progress
        self._cancel = cancel
        total = sum(STAGE_WEIGHTS[s] for s in stages)
        self._start: dict[str, float] = {}
        self._span: dict[str, float] = {}
        acc = 0.0
        for s in stages:
            self._start[s] = acc / total
            self._span[s] = STAGE_WEIGHTS[s] / total
            acc += STAGE_WEIGHTS[s]

    def begin(self, stage: str, message: str) -> None:
        """Comprueba la cancelación y anuncia el inicio de ``stage``.

        Raises
        ------
        CancelledError
            Si el usuario activó ``cancel``.
        """
        if self._cancel is not None and self._cancel.is_set():
            logger.info("Análisis cancelado por el usuario antes de: %s", message)
            raise CancelledError(f"Análisis cancelado por el usuario (antes de: {message}).")
        logger.info("%s", message)
        self.report(stage, 0.0, message)

    def report(self, stage: str, fraction: float, message: str) -> None:
        """Informa el avance ``fraction`` ∈ [0, 1] DENTRO de la etapa ``stage``."""
        if self._progress is not None:
            f = self._start[stage] + self._span[stage] * min(max(float(fraction), 0.0), 1.0)
            self._progress(min(f, 1.0), message)

    def sub_progress(self, stage: str) -> ProgressCallback:
        """Callback para un proceso interno (Demucs) que informa su propio 0→1."""
        return lambda fraction, message: self.report(stage, fraction, message)

    def finish(self, message: str) -> None:
        """Informa el 100 %."""
        if self._progress is not None:
            self._progress(1.0, message)


# ---------------------------------------------------------------------------
# Etapas 1–6
# ---------------------------------------------------------------------------


def _load_ground_truth_for(path: Path) -> list[GTNote] | None:
    """Carga ``<carpeta>/<nombre>.gt.json`` si existe (None si no existe o es inválido)."""
    from src.synth_dataset import gt_path_for, load_ground_truth

    gt_file = gt_path_for(path)
    if not gt_file.is_file():
        logger.info("Sin ground truth: no existe %s (no se podrá medir la precisión)", gt_file.name)
        return None
    try:
        notes = load_ground_truth(gt_file)
    except ValueError as exc:
        logger.warning("Ground truth ignorado: %s", exc)
        return None
    logger.info("Ground truth cargado de %s: %d notas", gt_file.name, len(notes))
    return notes


def track_level_dbfs(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Nivel (dBFS) del frame más fuerte de ``y``: 20·log10(max RMS por frame).

    Parameters
    ----------
    y : np.ndarray
        Señal mono en [-1, 1] (sin normalizar).
    frame_length : int, optional
        Ventana del RMS en muestras (2048 ≈ 93 ms a 22 050 Hz).
    hop_length : int, optional
        Salto entre ventanas en muestras.

    Returns
    -------
    float
        Nivel en dB respecto a la escala completa (0 dBFS = RMS 1); −inf
        para una señal vacía o digitalmente nula.

    Examples
    --------
    >>> t = np.arange(22050) / 22050
    >>> round(track_level_dbfs(0.5 * np.sin(2 * np.pi * 55 * t)))   # seno de amplitud 0.5 → RMS ≈ 0.354
    -9
    >>> track_level_dbfs(np.zeros(1000))
    -inf
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return float("-inf")
    length = min(int(frame_length), y.size)
    # Suma acumulada de y²: la energía de cada ventana es una resta (O(n) en total).
    energy = np.concatenate(([0.0], np.cumsum(y * y)))
    starts = np.arange(0, y.size - length + 1, max(int(hop_length), 1))
    mean_square = (energy[starts + length] - energy[starts]) / length
    peak_rms = float(np.sqrt(max(float(mean_square.max()), 0.0)))
    return float(20.0 * np.log10(peak_rms)) if peak_rms > 0 else float("-inf")


def analyze(
    path: str | Path,
    cfg: Config,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> AnalysisResult:
    """Ejecuta las etapas 1–6 sobre ``path`` (ver encabezado del módulo).

    Parameters
    ----------
    path : str | Path
        Archivo de audio (MP3, WAV, ...). Debe contener un bajo monofónico
        (o una mezcla si ``cfg.audio.separate_bass``).
    cfg : Config
        Configuración completa. Se guarda una COPIA PROFUNDA en el resultado:
        modificar ``cfg`` después no altera el análisis.
    progress : ProgressCallback | None, optional
        ``progress(fracción_0_1, mensaje)`` al inicio de cada etapa (y durante
        Demucs).
    cancel : threading.Event | None, optional
        Si se activa, el análisis se detiene antes de la siguiente etapa.

    Returns
    -------
    AnalysisResult
        Señales, onsets, segmentos, trayectoria de pitch, espectro, datos
        bandit de cada segmento conservado y ground truth (si existe).

    Raises
    ------
    ValueError
        Si la configuración no es válida (:meth:`src.config.Config.validate`);
        se comprueba ANTES de decodificar el audio.
    src.config.CancelledError
        Si ``cancel`` se activa entre etapas (o durante Demucs).
    src.io_audio.AudioLoadError, src.io_audio.FFmpegNotFoundError
        Si el audio no se puede decodificar.
    src.separation.SeparationError
        Si se pidió separar el bajo con Demucs (``separation_method="demucs"``)
        y no está instalado, o si Demucs falla.

    Notes
    -----
    Si el frame más fuerte de la pista está por debajo de
    :data:`MIN_TRACK_RMS_DBFS` (ruido, silencio, stem vacío), se avisa en el
    log y el resultado no tiene segmentos: no se inventa una tablatura a
    partir del ruido amplificado por la normalización.

    Examples
    --------
    >>> res = analyze("data/synthetic/linea_simple.mp3", Config(),
    ...               progress=lambda f, m: print(f"{100*f:3.0f} % {m}"))  # doctest: +SKIP
      0 % Etapa 1/6 — Carga: decodificando linea_simple.mp3
    ...
    """
    # Importes diferidos: librosa y sus dependencias tardan en cargarse y la GUI
    # importa este módulo al arrancar.
    from src import io_audio, preprocessing, separation
    from src.environment import build_all_segment_data, compute_spectrum
    from src.pitch import annotate_segments, estimate_pitch_track
    from src.segmentation import detect_onsets, segment_audio

    t_start = time.perf_counter()
    path = Path(path)
    cfg.validate()            # un parámetro inválido falla aquí, no tras minutos de análisis
    cfg = copy.deepcopy(cfg)  # el resultado guarda exactamente la configuración usada
    sr = int(cfg.audio.sample_rate)
    hop = int(cfg.segmentation.hop_length)

    # El método de separación se resuelve ANTES de cargar: si se pidió Demucs
    # y no está instalado, se falla enseguida con las instrucciones.
    method = separation.resolve_method(cfg.audio.separation_method) if cfg.audio.separate_bass else None
    sep_stage = {"demucs": ["separation"], "hpss": ["separation_hpss"]}.get(method or "", [])
    stages = ["load"] + sep_stage + ["preprocess", "segmentation", "pitch", "spectrum", "bandit", "ground_truth"]
    rep = _StageReporter(stages, progress, cancel)
    logger.info("Análisis de %s (%d Hz, hop=%d muestras = %.1f ms)", path.name, sr, hop, 1000.0 * hop / sr)

    # 1. Carga: decodificar primero el archivo elegido valida enseguida que el
    #    audio es legible (antes de lanzar Demucs, que tarda minutos).
    rep.begin("load", f"Etapa 1/6 — Carga: decodificando {path.name}")
    y_raw, sr = io_audio.load_audio(path, sr=sr)
    source_path = path

    # 2. Separación opcional (ver "Separación del bajo" en el encabezado).
    if method == "demucs":
        how = "elegido" if cfg.audio.separation_method == "demucs" else "automático: Demucs está instalado"
        rep.begin("separation", f"Etapa 2/6 — Separación: aislando el bajo con Demucs "
                                f"({cfg.audio.demucs_model}; método {how})")
        try:
            source_path = separation.separate_bass(
                path, model=cfg.audio.demucs_model, progress=rep.sub_progress("separation"), cancel=cancel)
        except separation.SeparationDepsError as exc:
            if cfg.audio.separation_method != "auto":
                raise
            # «auto» eligió Demucs porque torch/demucs están instalados, pero no se
            # pueden importar (instalación rota): se usa HPSS en lugar de fallar.
            logger.warning("Demucs está instalado pero no se pudo usar (%s); se separa con HPSS.",
                           str(exc).split("\n", 1)[0])
            rep.report("separation", 1.0, "Etapa 2/6 — Separación: Demucs no se pudo importar; aproximando el bajo "
                                          "con HPSS (componente armónica + pasa-bajas)")
            method = "hpss"
            y_raw = separation.hpss_bass(y_raw, sr)
        else:
            logger.info("Separación: se analizará el stem de bajo %s", source_path.name)
            y_raw, sr = io_audio.load_audio(source_path, sr=sr)
    elif method == "hpss":
        how = ("elegido" if cfg.audio.separation_method == "hpss"
               else "automático: Demucs no está instalado (pip install demucs para mejor calidad)")
        rep.begin("separation_hpss", f"Etapa 2/6 — Separación: aproximando el bajo con HPSS (componente armónica "
                                     f"+ pasa-bajas; método {how})")
        y_raw = separation.hpss_bass(y_raw, sr)
        logger.info("Separación HPSS: se analizará la señal separada en memoria (no se escribe archivo; "
                    "source_path sigue siendo %s)", path.name)
    else:
        logger.info("Etapa 2/6 — Separación: omitida (se asume que el audio ya es un bajo aislado)")

    # Nivel absoluto ANTES de normalizar: las etapas siguientes solo miden niveles
    # relativos y convertirían el ruido de fondo en notas.
    level_db = track_level_dbfs(y_raw)
    silent_track = level_db < MIN_TRACK_RMS_DBFS
    if silent_track:
        logger.warning("La pista es prácticamente silencio: su frame más fuerte está a %.1f dBFS (umbral %.0f dBFS); "
                       "no se transcribirá ninguna nota.", level_db, MIN_TRACK_RMS_DBFS)
    else:
        logger.info("Nivel de la pista: frame más fuerte a %.1f dBFS", level_db)

    # 3. Preprocesamiento: dos señales (ver encabezado del módulo).
    rep.begin("preprocess", f"Etapa 3/6 — Preprocesamiento: pasa-bajas {cfg.preprocess.lowpass_hz:.0f} Hz "
                            "para onsets/pYIN y normalización")
    pre = preprocessing.preprocess(y_raw, sr, cfg.preprocess)

    # 4. Segmentación: onsets sobre la señal filtrada → segmentos de nota.
    rep.begin("segmentation", "Etapa 4/6 — Segmentación: detectando onsets y descartando silencios/notas cortas")
    if silent_track:
        onsets_s, segments = np.zeros(0), []
    else:
        onsets_s = detect_onsets(pre.y_analysis, sr, cfg.segmentation)
        segments = segment_audio(pre.y_analysis, sr, cfg.segmentation, onsets_s=onsets_s)
    n_kept = sum(s.kept for s in segments)
    logger.info("Segmentación: %d onsets → %d segmentos, %d conservados", len(onsets_s), len(segments), n_kept)

    # 5. Pitch: pYIN sobre toda la pista (mismo hop que la segmentación) y f0
    #    de cada segmento conservado (se ignora el ataque, igual que la recompensa).
    rep.begin("pitch", f"Etapa 5/6 — Pitch: pYIN entre {cfg.pitch.fmin_hz:.0f} y {cfg.pitch.fmax_hz:.0f} Hz")
    # En pistas largas pYIN va por bloques: informa el avance y consulta
    # ``cancel`` entre ellos (y libera el GIL para que la GUI siga viva).
    pitch_track = estimate_pitch_track(pre.y_analysis, sr, cfg.pitch, hop_length=hop,
                                       progress=rep.sub_progress("pitch"), cancel=cancel)
    annotate_segments(pitch_track, segments, cfg.pitch, attack_skip_s=cfg.env.attack_skip_s)

    # 6. Espectro (sin filtrar) y matriz de saliencia de cada segmento.
    rep.begin("spectrum", f"Etapa 6/6 — Espectro {cfg.env.spectrum.upper()} de la señal sin filtrar")
    spectrum = compute_spectrum(pre.y_spectral, sr, cfg.env, hop_length=hop)
    rep.begin("bandit", f"Etapa 6/6 — Entornos bandit: brazos candidatos (k=±{cfg.env.k_semitones}) "
                        f"y saliencia armónica (N={cfg.env.n_harmonics}, β={cfg.env.beta:g})")
    segment_data = build_all_segment_data(spectrum, segments, cfg.env)

    # Ground truth: se busca junto al archivo ELEGIDO (no junto al stem de la caché).
    rep.begin("ground_truth", "Buscando ground truth (.gt.json) junto al audio")
    ground_truth = _load_ground_truth_for(path)

    elapsed = time.perf_counter() - t_start
    summary = (f"Análisis terminado en {elapsed:.1f} s: {len(y_raw) / sr:.2f} s de audio, "
               f"{n_kept} notas, {sum(d.n_arms for d in segment_data)} brazos en total")
    logger.info("%s", summary)
    rep.finish(summary)
    return AnalysisResult(
        path=path,
        source_path=Path(source_path),
        sr=int(sr),
        y_raw=y_raw,
        y_analysis=pre.y_analysis,
        y_spectral=pre.y_spectral,
        onsets_s=np.asarray(onsets_s, dtype=float),
        segments=segments,
        pitch_track=pitch_track,
        spectrum=spectrum,
        segment_data=segment_data,
        ground_truth=ground_truth,
        config=cfg,
        separation_method=method,
    )


# ---------------------------------------------------------------------------
# Recalcular solo la etapa 6
# ---------------------------------------------------------------------------


def _spectrum_changed(old: Config, new: Config) -> bool:
    """True si los parámetros que determinan el espectro difieren entre ``old`` y ``new``."""
    kind_old, kind_new = str(old.env.spectrum).lower(), str(new.env.spectrum).lower()
    if kind_old != kind_new:
        return True
    names = _CQT_PARAMS if kind_new == "cqt" else _STFT_PARAMS
    return any(getattr(old.env, n) != getattr(new.env, n) for n in names)


def _ignored_changes(old: Config, new: Config) -> list[str]:
    """Parámetros de etapas 1–5 que cambiaron pero que :func:`rebuild_segment_data` no aplica."""
    changed: list[str] = []
    for section in ("audio", "preprocess", "segmentation", "pitch"):
        old_sec, new_sec = getattr(old, section), getattr(new, section)
        for f in dataclasses.fields(old_sec):
            if section == "pitch" and f.name == "min_voiced_ratio":
                continue  # este sí se aplica (re-anotación barata)
            if getattr(old_sec, f.name) != getattr(new_sec, f.name):
                changed.append(f"{section}.{f.name}")
    return changed


def rebuild_segment_data(analysis: AnalysisResult, cfg: Config) -> AnalysisResult:
    """Recalcula la etapa 6 (y la f0 por segmento si hace falta) con una nueva configuración.

    Sirve para explorar los parámetros del entorno (k, N, β, tolerancia,
    ataque ignorado, tipo de espectro...) sin volver a decodificar, segmentar
    ni ejecutar pYIN. Devuelve un NUEVO :class:`AnalysisResult`; ``analysis``
    no se modifica.

    Parameters
    ----------
    analysis : AnalysisResult
        Resultado previo de :func:`analyze`.
    cfg : Config
        Configuración nueva.

    Returns
    -------
    AnalysisResult
        Copia con ``segment_data`` reconstruido (y ``spectrum`` / ``segments``
        nuevos si cambiaron).

    Notes
    -----
    * El **espectro** solo se recalcula si cambió su tipo (``env.spectrum``) o
      sus parámetros (``bins_per_octave``, ``cqt_fmin_hz``, ``n_octaves`` para
      la CQT; ``n_fft``, ``n_harmonics``, ``tolerance_semitones`` y
      ``n_frets`` para la STFT, porque los tres últimos fijan la banda
      guardada). Usa el mismo ``hop_length`` que el
      análisis original, para seguir alineado con los segmentos.
    * La **f0 de cada segmento** se recalcula a partir de la trayectoria de
      pYIN ya guardada (barato) si cambió ``pitch.min_voiced_ratio`` o
      ``env.attack_skip_s``; los segmentos se copian antes de re-anotarlos.
    * Los cambios en audio, preprocesamiento, segmentación o en el rango de
      pYIN NO se aplican (requieren :func:`analyze`): se avisa en el log y la
      ``config`` del resultado conserva los valores realmente usados en esas
      etapas (y toma ``env``, ``agent`` y ``experiment`` de ``cfg``).
    """
    from src.environment import build_all_segment_data, compute_spectrum
    from src.pitch import annotate_segments

    old_cfg = analysis.config
    new_cfg = copy.deepcopy(analysis.config)
    new_cfg.env = copy.deepcopy(cfg.env)
    new_cfg.agent = copy.deepcopy(cfg.agent)
    new_cfg.experiment = copy.deepcopy(cfg.experiment)
    new_cfg.pitch.min_voiced_ratio = cfg.pitch.min_voiced_ratio

    ignored = _ignored_changes(old_cfg, cfg)
    if ignored:
        logger.warning("Estos cambios requieren volver a analizar el audio y no se aplican al recalcular "
                       "los entornos: %s", ", ".join(ignored))

    spectrum = analysis.spectrum
    if _spectrum_changed(old_cfg, new_cfg):
        logger.info("Recalculando el espectro (%s) porque cambiaron sus parámetros", new_cfg.env.spectrum.upper())
        spectrum = compute_spectrum(analysis.y_spectral, analysis.sr, new_cfg.env, hop_length=analysis.spectrum.hop_length)

    segments = analysis.segments
    if (new_cfg.pitch.min_voiced_ratio != old_cfg.pitch.min_voiced_ratio
            or new_cfg.env.attack_skip_s != old_cfg.env.attack_skip_s):
        logger.info("Recalculando la f0 de cada segmento (fracción con voz mínima %.2f, ataque ignorado %.3f s)",
                    new_cfg.pitch.min_voiced_ratio, new_cfg.env.attack_skip_s)
        segments = copy.deepcopy(analysis.segments)
        annotate_segments(analysis.pitch_track, segments, new_cfg.pitch, attack_skip_s=new_cfg.env.attack_skip_s)

    logger.info("Recalculando entornos bandit: k=±%s, N=%d, β=%g, tolerancia ±%.2f semitonos",
                new_cfg.env.k_semitones, new_cfg.env.n_harmonics, new_cfg.env.beta, new_cfg.env.tolerance_semitones)
    segment_data = build_all_segment_data(spectrum, segments, new_cfg.env)
    return dataclasses.replace(analysis, spectrum=spectrum, segments=segments,
                               segment_data=segment_data, config=new_cfg)


__all__ = ["AnalysisResult", "analyze", "rebuild_segment_data", "track_level_dbfs", "STAGE_WEIGHTS",
           "MIN_TRACK_RMS_DBFS"]
