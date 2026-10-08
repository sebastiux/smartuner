"""Configuración central del sistema (fuente única de verdad de hiperparámetros).

Papel en el pipeline
--------------------
Todas las etapas (carga, preprocesamiento, segmentación, pitch, entorno bandit,
agentes y experimentos) leen sus parámetros de las dataclasses de este módulo.
La GUI construye sus controles a partir de :data:`PARAM_SPECS` (etiqueta,
tooltip y rango de cada parámetro) y la CLI carga/guarda la misma
configuración en JSON, de modo que GUI, CLI y documentación nunca divergen.

Ejemplo
-------
>>> cfg = Config()
>>> cfg.env.lam
0.1
>>> cfg2 = Config.from_dict(cfg.to_dict())
>>> cfg2 == cfg
True
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tipos compartidos por todos los módulos
# ---------------------------------------------------------------------------

#: Callback de progreso: ``progress(fraccion_entre_0_y_1, mensaje)``. Lo usan los
#: procesos largos para informar a la barra de progreso de la GUI o a la consola.
ProgressCallback = Callable[[float, str], None]


class CancelledError(RuntimeError):
    """El usuario canceló un proceso largo (se comprueba un ``threading.Event``)."""


# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

#: Identificadores internos de los cuatro algoritmos, en el orden fijo en que
#: se muestran en tablas y gráficas (el color sigue al algoritmo, nunca al rango).
ALGORITHMS: tuple[str, ...] = ("egreedy", "optimistic", "ucb1", "softmax")

#: Nombres legibles (español) de cada algoritmo. ``"oracle"`` es el oráculo
#: MIOPE (argmax μ en cada segmento dado su traste previo) y
#: ``"oracle_viterbi"`` el óptimo de toda la cadena (programación dinámica).
ALGO_LABELS: dict[str, str] = {
    "egreedy": "ε-greedy",
    "optimistic": "ε-greedy optimista",
    "ucb1": "UCB1",
    "softmax": "Softmax (Boltzmann)",
    "oracle": "Oráculo miope (argmax μ)",
    "oracle_viterbi": "Oráculo de cadena (Viterbi)",
}

#: Color fijo por algoritmo (paleta categórica validada para daltonismo,
#: orden adyacente). Los oráculos son referencias neutras en gris.
ALGO_COLORS: dict[str, str] = {
    "egreedy": "#2a78d6",
    "optimistic": "#eb6834",
    "ucb1": "#1baf7a",
    "softmax": "#eda100",
    "oracle": "#52514e",
    "oracle_viterbi": "#8a8780",
}

#: Codificación secundaria (marcador y estilo de línea) para que la identidad
#: de cada algoritmo nunca dependa solo del color.
ALGO_MARKERS: dict[str, str] = {
    "egreedy": "o",
    "optimistic": "s",
    "ucb1": "^",
    "softmax": "D",
    "oracle": "x",
    "oracle_viterbi": "+",
}

#: Estilo de línea de matplotlib: un nombre (``"-"``, ``"--"``...) o un patrón
#: de trazos ``(desfase, (trazo, hueco, ...))`` en puntos.
LineStyle = str | tuple[int, tuple[int, ...]]

ALGO_LINESTYLES: dict[str, LineStyle] = {
    "egreedy": "-",
    "optimistic": "--",
    "ucb1": "-.",
    "softmax": ":",
    "oracle": (0, (1, 1)),
    "oracle_viterbi": (0, (6, 2, 1, 2)),
}

#: Frecuencias de las cuerdas al aire del bajo en afinación estándar (Hz),
#: de la más grave a la más aguda.
OPEN_STRING_HZ: dict[str, float] = {"E": 41.20, "A": 55.00, "D": 73.42, "G": 98.00}

#: Número MIDI de cada cuerda al aire (E1=28, A1=33, D2=38, G2=43).
OPEN_STRING_MIDI: dict[str, int] = {"E": 28, "A": 33, "D": 38, "G": 43}

#: Orden de cuerdas de grave a aguda. La tablatura se imprime al revés (G arriba).
STRING_ORDER: tuple[str, ...] = ("E", "A", "D", "G")

#: Frecuencia de muestreo de trabajo (Hz). Todos los tamaños en MUESTRAS de la
#: configuración (salto de 256, FFT de onsets de 2048, ventana de pYIN de 2048,
#: n_fft de 8192) están calibrados para ella; con otra frecuencia esas ventanas
#: durarían otra cosa (a 44.1 kHz la mitad: la E1 se parte en varios segmentos y
#: pYIN ya no abarca dos periodos de 41 Hz). :meth:`Config.validate` la exige.
SUPPORTED_SAMPLE_RATE: int = 22050

#: Métodos de separación del bajo (``audio.separation_method``): ``"auto"``
#: (Demucs si está instalado, si no HPSS), ``"demucs"`` y ``"hpss"``.
SEPARATION_METHODS: tuple[str, ...] = ("auto", "demucs", "hpss")

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Dataclasses de configuración (una por etapa del pipeline)
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    """Parámetros de carga del audio (etapa 1) y separación (etapa 2).

    Attributes
    ----------
    sample_rate : int
        Frecuencia de muestreo de trabajo en Hz. Debe ser 22 050 Hz
        (:data:`SUPPORTED_SAMPLE_RATE`): los tamaños de ventana en muestras de
        las demás etapas están calibrados para ella (ver :meth:`Config.validate`).
    separate_bass : bool
        Si es True se aísla el bajo de una mezcla (etapa 2) con el método
        ``separation_method``.
    demucs_model : str
        Nombre del modelo de Demucs (``"htdemucs"``).
    separation_method : str
        Cómo se separa el bajo cuando ``separate_bass`` es True (ver
        :data:`SEPARATION_METHODS`):

        * ``"auto"`` (por defecto): Demucs si está instalado; si no, HPSS.
        * ``"demucs"``: red neuronal Demucs (la mejor calidad; requiere
          PyTorch y tarda minutos en CPU). Si no está instalado, el análisis
          falla con instrucciones de instalación.
        * ``"hpss"``: separación armónico-percusiva + pasa-bajas con librosa
          (segundos, sin dependencias extra). Es una aproximación: quita
          batería, efectos percusivos y voces/platillos agudos, pero no las
          guitarras ni los teclados graves.
    """

    sample_rate: int = 22050
    separate_bass: bool = False
    demucs_model: str = "htdemucs"
    separation_method: str = "auto"


@dataclass
class PreprocessConfig:
    """Parámetros del preprocesamiento (etapa 3).

    Attributes
    ----------
    lowpass_hz : float
        Frecuencia de corte del filtro pasa-bajas Butterworth en Hz. Solo se
        aplica a la señal de análisis (onsets y pYIN); la recompensa usa la
        señal normalizada sin filtrar para conservar los armónicos.
    filter_order : int
        Orden del filtro Butterworth (se aplica ida y vuelta, fase cero).
    normalize : bool
        Si es True se normaliza el pico de la señal a 0.99.
    """

    lowpass_hz: float = 400.0
    filter_order: int = 4
    normalize: bool = True


@dataclass
class SegmentationConfig:
    """Parámetros de la segmentación por onsets (etapa 4).

    Attributes
    ----------
    hop_length : int
        Salto entre frames de análisis en muestras (256 ≈ 11.6 ms a 22.05 kHz).
    onset_delta : float
        Umbral de selección de picos de la función de novedad (``delta`` de
        ``librosa.util.peak_pick``). Más alto = menos onsets.
    onset_wait_s : float
        Separación mínima entre dos onsets consecutivos en segundos.
    backtrack : bool
        Si es True cada onset se retrasa al mínimo de energía previo.
    rms_threshold_db : float
        Umbral de silencio en dB relativos al RMS máximo de la pista. Los
        segmentos cuyo RMS medio queda por debajo se descartan.
    min_duration_s : float
        Duración mínima de un segmento en segundos; los más cortos se descartan.

    Notes
    -----
    **Calibración de los valores por defecto.** Se evaluaron sobre las 3
    piezas del dataset sintetizadas con fluidsynth y con Karplus-Strong
    (57 notas cada versión) y sobre 8 piezas de validación que NO forman
    parte del dataset (semicorcheas con cambio de nota, trastes 5–12 y
    ``linea_simple``/``riff_saltos`` a 140 bpm; 230 notas). Métrica:
    segmentos conservados emparejados uno a uno con el ground truth (±0.1 s).

    =====================================  ===========================  =================
    (δ, separación mínima)                 dataset (6 pistas)            validación (8)
    =====================================  ===========================  =================
    0.07, 0.05 s (valores anteriores)      1 nota perdida, 1 extra       10 perdidas, 2 extra
    **0.05, 0.07 s**                       **0 perdidas, 0 extra**       4 perdidas, 1 extra
    0.04, 0.07 s                           0 perdidas, 0 extra           2 perdidas, 1 extra
    0.06, 0.07 s                           0 perdidas, 0 extra           7 perdidas, 1 extra
    =====================================  ===========================  =================

    El dataset queda perfecto (1 segmento por nota, incluidas las notas
    repetidas y las semicorcheas de ``riff_saltos``) para δ ∈ [0.03, 0.06]
    con separación 0.06–0.08 s. δ = 0.05 está en el centro de esa región.
    Cuanto menor es δ, más onsets falsos aparecen en los apagados de nota de
    las señales de prueba de ``tests/test_segmentation.py`` (0 con δ = 0.07,
    1 con 0.05, 2 con 0.04); ``segment_audio`` los descarta como "corto"
    porque solo abarcan unos ms de cola, así que no llegan a la tablatura.
    Con δ = 0.06, en cambio, se pierden cambios de nota en semicorcheas de
    Karplus-Strong. Las 4 notas perdidas en
    validación con δ = 0.05 son re-ataques del MISMO pitch en semicorcheas a
    140 bpm (≈ 0.1 s), el caso más difícil para cualquier función de novedad.
    Una separación mínima de 0.07 s (hasta ≈ 14 notas/s) elimina los dobles
    onsets del apagado de Karplus-Strong sin limitar líneas de bajo reales.
    ``rms_threshold_db`` (−25…−50 dB) y ``min_duration_s`` (0.04–0.1 s) no
    cambian el resultado en este rango y conservan sus valores.
    """

    hop_length: int = 256
    onset_delta: float = 0.05
    onset_wait_s: float = 0.07
    backtrack: bool = True
    rms_threshold_db: float = -35.0
    min_duration_s: float = 0.06


@dataclass
class PitchConfig:
    """Parámetros de la estimación de pitch con pYIN (etapa 5).

    Attributes
    ----------
    fmin_hz : float
        Frecuencia mínima buscada por pYIN (Hz). E1 = 41.2 Hz.
    fmax_hz : float
        Frecuencia máxima buscada por pYIN (Hz). G2 traste 12 = 196 Hz.
    frame_length : int
        Longitud de ventana de pYIN en muestras (2048 ≈ 93 ms, más de dos
        periodos de E1).
    min_voiced_ratio : float
        Fracción mínima de frames con voz para aceptar la f0 de un segmento;
        por debajo se considera f0 desconocida y no se podan brazos.
    max_transition_rate : float
        Máximo cambio de pitch (octavas/s) que admite el HMM de pYIN entre
        frames consecutivos (librosa usa 35.92 por defecto). En una línea de
        bajo hay saltos de más de una octava entre notas de 0.1–0.3 s.
    switch_prob : float
        Probabilidad de pasar de "con voz" a "sin voz" (o al revés) entre
        frames en el HMM de pYIN (librosa usa 0.01).

    Notes
    -----
    Con los valores de librosa (35.92 oct/s, 0.01) el Viterbi de pYIN
    "arrastra" la f0 de la nota anterior por la suboctava y produce errores
    de octava hacia abajo en notas rápidas tras un salto (p. ej. D-7 = 98 Hz
    leído como 49 Hz). Medido sobre 299 notas (dataset fluidsynth y
    Karplus-Strong + piezas de validación con semicorcheas y a 140 bpm):
    librosa por defecto → 3 errores de octava; 150 oct/s y 0.1 → 0 errores
    (la región 100–200 oct/s × 0.05–0.2 da el mismo resultado). Subir solo
    ``max_transition_rate`` EMPEORA (18 errores): hace falta también
    ``switch_prob`` para que el HMM pueda "cortar" entre notas.
    """

    fmin_hz: float = 35.0
    fmax_hz: float = 250.0
    frame_length: int = 2048
    min_voiced_ratio: float = 0.2
    max_transition_rate: float = 150.0
    switch_prob: float = 0.1


@dataclass
class EnvConfig:
    """Parámetros del entorno bandit por segmento (etapa 6).

    Attributes
    ----------
    n_frets : int
        Traste máximo considerado (0..n_frets). 12 → 4 × 13 = 52 brazos.
    k_semitones : int | None
        Poda: solo se consideran brazos a ±k semitonos de la f0 estimada.
        ``None`` usa los 52 brazos.
    n_harmonics : int
        Número N de armónicos del template (incluye la fundamental).
    beta : float
        Peso β de la penalización inter-armónica (energía en (h−½)·f). Castiga
        los errores de octava. 0 = suma armónica pura.
    lam : float
        λ: peso de la penalización de tocabilidad λ·|traste − traste_previo|/12.
    budget : int
        Presupuesto T de pulls por segmento.
    spectrum : str
        Representación espectral para la recompensa: ``"cqt"`` o ``"stft"``.
    bins_per_octave : int
        Resolución de la CQT en bins por octava (36 = tercio de semitono).
    cqt_fmin_hz : float
        Frecuencia más baja de la CQT en Hz (C1 = 32.70 Hz).
    n_octaves : int
        Octavas cubiertas por la CQT (6 → hasta C7 ≈ 2093 Hz; debe quedar
        por debajo de la frecuencia de Nyquist sr/2).
    n_fft : int
        Tamaño de la FFT en muestras cuando ``spectrum == "stft"`` (8192 ≈
        0.37 s a 22 050 Hz; resolución sr/n_fft ≈ 2.7 Hz por bin).
    attack_skip_s : float
        Se ignoran los primeros segundos de cada segmento (ataque/ruido de
        púa) al muestrear frames para la recompensa.
    tolerance_semitones : float
        Tolerancia de afinación al leer la energía en h·f (± semitonos).
    initial_hand_fret : int
        Posición de la mano supuesta antes de la primera nota (traste_previo
        del primer segmento).
    open_string_free : bool
        Si es True, tocar una cuerda al aire no cuesta movimiento y no cambia
        la posición de la mano (extensión opcional; por defecto se usa la
        fórmula literal λ·|traste − traste_previo|).
    noise_std : float
        Desviación estándar de ruido gaussiano opcional añadido a cada
        recompensa (0 = sin ruido extra; el muestreo de frames ya es estocástico).

    Notes
    -----
    **CQT frente a STFT** (``spectrum``). Evaluación sobre el dataset (3 piezas
    × {fluidsynth, Karplus-Strong}, 114 notas) y 8 piezas de validación que no
    forman parte de él (semicorcheas con cambio de nota, trastes 5–12, piezas
    a 140 bpm; 230 notas). "Oráculo" = argmax μ por segmento (oráculo miope:
    techo de la recompensa, no de la precisión de posición; ver
    :mod:`src.experiments`); "sin poda" = los 52 brazos (poder
    discriminativo puro de la saliencia); "margen" = μ(mejor brazo con el
    pitch correcto) − μ(mejor brazo con otro pitch), medio y mínimo sobre
    todas las notas (negativo = el oráculo se equivoca); "bandits" =
    precisión de pitch media de los 4 algoritmos (10 corridas, T = 500). Las
    notas no detectadas cuentan como error.

    ============================  ===============  ==========  ========  ========  ===============
    Espectro                      Oráculo k=2      Sin poda    Margen    Mínimo    Bandits
                                  dataset / valid  dataset                         dataset / valid
    ============================  ===============  ==========  ========  ========  ===============
    CQT 36 bins/oct (antes)       99.2 / 92.0 %    96.7 %      0.40      −0.02     98.6 / 91.4 %
    CQT 24 bins/oct               100 / 93.1 %     99.2 %      0.29      0.00      98.8 / 92.1 %
    STFT n_fft = 4096             100 / 96.2 %     100 %       0.20      +0.003    97.3 / 94.1 %
    STFT n_fft = 8192 (defecto)   100 / 94.6 %     100 %       0.33      +0.05     99.4 / 94.2 %
    ============================  ===============  ==========  ========  ========  ===============

    La CQT de 36 bins/octava tiene el mayor margen medio pero ventanas de
    ≈ 1.2 s en E1: la nota vecina se "cuela" en el grave (E-0 seguido de E-1
    → el oráculo elige E-1) y falla en semicorcheas. La STFT de 4096 muestras
    resuelve mejor el tiempo pero su margen es menor (2 bins por semitono a
    5·f₀) y Softmax con τ = 0.1 cae al ≈ 93 %. n_fft = 8192 (0.37 s, 2.7 Hz por
    bin) es el mejor compromiso: 100 % con y sin poda en el dataset, margen
    positivo en TODAS sus notas y la mejor precisión de los bandits.
    (Cifras medidas con la saliencia calculada una vez por nota MIDI y el
    máximo exacto de la ventana de tolerancia; ver :mod:`src.environment`.)

    **Otros parámetros** (barridos sobre los mismos datos con la STFT 8192):
    β ∈ {0, 0.25, 0.5, 1}, N ∈ {3, 5, 8}, tolerancia ∈ {0, 0.2, 0.33, 0.5} y
    ataque ignorado ∈ {0, 0.03, 0.05, 0.08} s dan todos 100 % de pitch del
    oráculo en el dataset; se conservan β = 0.5 (protege de errores de octava,
    que este dataset sintético con fundamental fuerte no provoca), N = 5,
    ±0.33 semitonos (absorbe desafinación real) y 0.03 s. Con k = None (52
    brazos) la precisión de pitch del ε-greedy cae al ≈ 51 % con T = 300 y al
    ≈ 68 % con el T = 500 actual (3 piezas fluidsynth, 10 corridas): su
    exploración uniforme apenas prueba cada brazo; UCB1 y el optimista siguen
    en ≈ 100 % y Softmax en ≈ 96 %. k = 2 es el mejor valor para los bandits.
    λ entre 0.02 y 0.3 da la misma precisión de posición del oráculo
    (meseta); con λ ≥ 0.5 la penalización empieza a imponerse a la saliencia
    y el oráculo pierde pitch (99.2 % con 0.5, 97 % con 1.0, 89 % con 2.0).

    **Presupuesto** ``budget``: precisión de pitch media de ε-greedy / Softmax
    (UCB1 y el optimista ya están en el 100 % desde T = 100) en el dataset:
    T = 100 → 75 / 91 %, 200 → 89 / 96 %, 300 → 95 / 97 %, 500 → 99 / 99 %,
    1000 → 100 / 100 %. Con T = 500 todos quedan a ≤ 1.5 puntos del oráculo
    y el experimento completo (100 corridas + barridos) dura ≈ 2 min.
    """

    n_frets: int = 12
    k_semitones: int | None = 2
    n_harmonics: int = 5
    beta: float = 0.5
    lam: float = 0.1
    budget: int = 500
    spectrum: str = "stft"
    bins_per_octave: int = 36
    cqt_fmin_hz: float = 32.70
    n_octaves: int = 6
    n_fft: int = 8192
    attack_skip_s: float = 0.03
    tolerance_semitones: float = 0.33
    initial_hand_fret: int = 0
    open_string_free: bool = False
    noise_std: float = 0.0


@dataclass
class AgentConfig:
    """Hiperparámetros de los cuatro agentes (etapa 7).

    Attributes
    ----------
    epsilon : float
        Probabilidad de explorar de ε-greedy.
    epsilon_decay : float
        Factor multiplicativo por pull: ε_t = max(ε_min, ε₀·dᵗ). 1 = sin decaimiento.
    epsilon_min : float
        Cota inferior de ε cuando hay decaimiento.
    q0 : float
        Valor inicial optimista Q₀ del ε-greedy optimista (mayor que la
        recompensa máxima posible, que es 1).
    optimistic_epsilon : float
        ε del agente optimista (0 = totalmente greedy; explora solo por optimismo).
    optimistic_alpha : float
        Tamaño de paso constante α del agente optimista. Con α = 1/n el
        optimismo desaparecería tras el primer pull de cada brazo.
    ucb_c : float
        Constante c de exploración de UCB1 (√2 en la versión original).
    tau : float
        Temperatura τ de Softmax. Debe estar en la escala de las diferencias
        entre recompensas (≈0.01–1).
    tau_decay : float
        Annealing: τ_t = max(τ_min, τ₀·dᵗ). 1 = sin annealing.
    tau_min : float
        Cota inferior de τ con annealing.
    recommend : str
        Cómo se elige la posición final del segmento al agotar el presupuesto:
        ``"most_pulled"`` (brazo más jalado, desempate por Q) o ``"greedy"``
        (argmax Q).

    Notes
    -----
    Se conservan los valores "de libro" (ε = 0.1, Q₀ = 2 con α = 0.1,
    c = √2, τ = 0.1) para que la comparación sea la clásica; los barridos de
    sensibilidad del experimento muestran cuánto mejora cada algoritmo al
    ajustarlo. Con la recompensa por defecto (T = 500, dataset sintético) la
    recompensa media del último 10 % de pulls es máxima con ε ≈ 0.05,
    Q₀ ≈ 1–2, τ = 0.1 y c = 0.1. El caso de UCB1 es didáctico: c = √2 viene
    de la cota de Hoeffding para recompensas en [0, 1] con varianza máxima,
    pero aquí el ruido por frame es ≈ 0.05 y las brechas entre posiciones
    del mismo pitch son de λ·Δtraste/12 ≈ 0.01–0.03, así que el bono
    √(2·ln t / n) domina durante todo el presupuesto (≈ 48 % de pulls
    óptimos con c = √2). Aun así UCB1 acierta el 100 % de los pitches: la
    sobre-exploración solo le cuesta regret y precisión de posición.
    """

    epsilon: float = 0.1
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01
    q0: float = 2.0
    optimistic_epsilon: float = 0.0
    optimistic_alpha: float = 0.1
    ucb_c: float = 1.414
    tau: float = 0.1
    tau_decay: float = 1.0
    tau_min: float = 0.01
    recommend: str = "most_pulled"


@dataclass
class ExperimentConfig:
    """Parámetros de los experimentos comparativos.

    Attributes
    ----------
    algorithms : list[str]
        Algoritmos a comparar (subconjunto de :data:`ALGORITHMS`).
    n_runs : int
        Número de corridas independientes (semillas distintas), ≥ 100 para
        el experimento principal.
    seed : int
        Semilla maestra; de ella se derivan todas las demás con
        ``numpy.random.SeedSequence`` (resultados reproducibles).
    example_segment : int | None
        Índice (entre los segmentos conservados) del segmento usado en las
        gráficas de un solo segmento. ``None`` elige el de más brazos.
    run_sweeps : bool
        Si es True se ejecutan los barridos de sensibilidad (ε, Q₀, c, τ).
    run_lambda_sweep : bool
        Si es True se ejecuta el barrido de λ (requiere ground truth para la
        precisión de posición).
    sweep_runs : int
        Corridas por punto en los barridos.
    sweep_epsilon, sweep_q0, sweep_c, sweep_tau, sweep_lambda : list[float]
        Valores evaluados en cada barrido.
    onset_tolerance_s : float
        Tolerancia (s) para emparejar un segmento detectado con una nota del
        ground truth.

    Notes
    -----
    ``onset_tolerance_s = 0.08``: los onsets detectados en el audio de
    fluidsynth llegan de media +27 ms (máximo +66 ms) después del note-on del
    MIDI, porque el sample de bajo de FluidR3 tarda ≈ 20 ms en alcanzar su
    primer pico y la ventana de 2048 muestras de la función de novedad suaviza
    el ataque. Con la tolerancia anterior (0.05 s) 2 notas de ``cromatica`` y
    3 de ``riff_saltos`` quedaban "no detectadas" aunque el segmento existía
    y su pitch era correcto. 80 ms es algo MÁS que la mitad del intervalo
    entre onsets más corto del dataset (semicorcheas de ``riff_saltos``,
    0.15 s → 75 ms), así que un segmento podría quedar a ≤ 80 ms de dos
    notas; en el dataset no ocurre (ningún segmento tiene dos notas GT a
    ±80 ms) y, para cuando ocurra (p. ej. semicorcheas a 140 bpm, 0.107 s),
    el emparejamiento es bipartito MÁXIMO y uno a uno
    (:func:`src.experiments.match_segments_to_gt`): cada segmento cuenta
    para una sola nota y no se pierden notas por emparejar primero el par
    más cercano.
    """

    algorithms: list[str] = field(default_factory=lambda: list(ALGORITHMS))
    n_runs: int = 100
    seed: int = 42
    example_segment: int | None = None
    run_sweeps: bool = True
    run_lambda_sweep: bool = True
    sweep_runs: int = 30
    sweep_epsilon: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    sweep_q0: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    sweep_c: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 1.414, 2.0, 4.0])
    sweep_tau: list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1, 0.3, 1.0])
    sweep_lambda: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.5, 1.0])
    onset_tolerance_s: float = 0.08


_SECTIONS: dict[str, type] = {
    "audio": AudioConfig,
    "preprocess": PreprocessConfig,
    "segmentation": SegmentationConfig,
    "pitch": PitchConfig,
    "env": EnvConfig,
    "agent": AgentConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class Config:
    """Configuración completa: una sección por etapa del pipeline.

    Se serializa a/desde JSON con :meth:`save_json` / :meth:`load_json`.
    Las claves desconocidas al cargar se ignoran (compatibilidad hacia atrás);
    los valores conocidos se convierten al tipo del campo y se validan
    (:meth:`validate`) para que un JSON erróneo falle ANTES del análisis, con
    un mensaje en español, y no con una traza a mitad del proceso.

    Examples
    --------
    >>> Config.from_dict({"env": {"n_harmonics": 5.0}}).env.n_harmonics   # float entero → int
    5
    >>> Config.from_dict({"env": {"lam": "0.1"}})
    Traceback (most recent call last):
        ...
    ValueError: env.lam debe ser un número (recibido '0.1', de tipo str).
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Devuelve la configuración como diccionario anidado (serializable a JSON).

        Returns
        -------
        dict[str, dict[str, Any]]
            Una entrada por sección (``"audio"``, ``"env"``...) con sus campos.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Construye y valida una configuración a partir de un diccionario anidado.

        Parameters
        ----------
        data : dict
            Diccionario con secciones como las de :meth:`to_dict`. Las
            secciones o claves ausentes toman su valor por defecto y las
            desconocidas se ignoran.

        Returns
        -------
        Config
            Configuración con cada valor convertido al tipo de su campo (un
            float entero como ``5.0`` en un campo ``int`` pasa a ``5``).

        Raises
        ------
        ValueError
            Si ``data`` o alguna sección no es un diccionario, si un valor no
            tiene el tipo de su campo o si la configuración no pasa
            :meth:`validate`.
        """
        if not isinstance(data, dict):
            raise ValueError(f"La configuración debe ser un objeto JSON con las secciones {', '.join(_SECTIONS)}; "
                             f"se recibió {type(data).__name__}.")
        kwargs: dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            section_data = data.get(name)
            if section_data is None:
                section_data = {}
            if not isinstance(section_data, dict):
                raise ValueError(f"La sección '{name}' de la configuración debe ser un objeto JSON (diccionario); "
                                 f"se recibió {type(section_data).__name__}.")
            values = {f.name: _coerce_value(f"{name}.{f.name}", section_data[f.name], str(f.type))
                      for f in fields(section_cls) if f.name in section_data}
            kwargs[name] = section_cls(**values)
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        """Comprueba tipos, rangos y restricciones entre parámetros.

        Se comprueban: el tipo de cada campo; el rango (o las opciones) de
        cada parámetro de :data:`PARAM_SPECS` (p. ej. ``audio.separation_method``
        debe ser uno de :data:`SEPARATION_METHODS`); que las listas de los barridos
        no estén vacías y queden en el rango del parámetro barrido; y las
        restricciones que no caben en un rango: frecuencia de muestreo
        soportada, salto entre frames, 0 < fmin < fmax < sr/2 de pYIN con al
        menos dos periodos de fmin en su ventana, CQT por debajo de Nyquist,
        mano inicial dentro del diapasón y algoritmos conocidos.

        Raises
        ------
        ValueError
            Con la lista de TODOS los problemas encontrados (en español).

        Examples
        --------
        >>> Config().validate()            # los valores por defecto son válidos
        >>> cfg = Config(); cfg.segmentation.hop_length = 0; cfg.experiment.sweep_lambda = []
        >>> cfg.validate()
        Traceback (most recent call last):
            ...
        ValueError: Configuración inválida:
          • experiment.sweep_lambda no puede estar vacía (para omitir el barrido usa run_lambda_sweep = false).
          • segmentation.hop_length debe estar entre 16 y 4096 muestras (recibido 0).
        """
        problems: list[str] = []
        for name, section_cls in _SECTIONS.items():
            section = getattr(self, name)
            for f in fields(section_cls):
                try:
                    _coerce_value(f"{name}.{f.name}", getattr(section, f.name), str(f.type))
                except ValueError as exc:
                    problems.append(str(exc))
        if not problems:  # los rangos solo tienen sentido con los tipos correctos
            for key, spec in PARAM_SPECS.items():
                problems += _spec_problems(key, self.get(key), spec)
            problems += self._sweep_problems()
            problems += self._cross_problems()
        if problems:
            raise ValueError("Configuración inválida:\n  • " + "\n  • ".join(problems))

    def _sweep_problems(self) -> list[str]:
        """Listas de barrido vacías o con valores fuera del rango del parámetro barrido."""
        problems: list[str] = []
        for list_name, param in _SWEEP_LISTS.items():
            values = getattr(self.experiment, list_name)
            if not values:
                what = "el barrido usa run_lambda_sweep" if list_name == "sweep_lambda" else "los barridos usa run_sweeps"
                problems.append(f"experiment.{list_name} no puede estar vacía (para omitir {what} = false).")
                continue
            spec = PARAM_SPECS[param]
            for v in values:
                if (spec.minimum is not None and v < spec.minimum) or (spec.maximum is not None and v > spec.maximum):
                    problems.append(f"experiment.{list_name}: el valor {v:g} está fuera del rango de {param} "
                                    f"[{spec.minimum:g}, {spec.maximum:g}].")
        return problems

    def _cross_problems(self) -> list[str]:
        """Restricciones de campos sin :class:`ParamSpec` y entre parámetros."""
        problems: list[str] = []
        sr = self.audio.sample_rate
        if sr != SUPPORTED_SAMPLE_RATE:
            problems.append(f"audio.sample_rate debe ser {SUPPORTED_SAMPLE_RATE} Hz (recibido {sr}): los tamaños de "
                            "ventana en muestras (salto, FFT de onsets, ventana de pYIN, n_fft) están calibrados "
                            "para esa frecuencia; el audio se remuestrea al cargarlo.")
        if not self.audio.demucs_model.strip():
            problems.append("audio.demucs_model no puede estar vacío.")
        if not 16 <= self.segmentation.hop_length <= 4096:
            problems.append(f"segmentation.hop_length debe estar entre 16 y 4096 muestras "
                            f"(recibido {self.segmentation.hop_length}).")
        if not 0 < self.preprocess.lowpass_hz < sr / 2:
            problems.append(f"preprocess.lowpass_hz debe estar entre 0 y sr/2 = {sr / 2:g} Hz.")
        fmin, fmax, frame = self.pitch.fmin_hz, self.pitch.fmax_hz, self.pitch.frame_length
        if not 0 < fmin < fmax:
            problems.append(f"pitch: se necesita 0 < fmin_hz < fmax_hz (recibido fmin={fmin:g}, fmax={fmax:g}).")
        elif fmax > sr / 2:
            problems.append(f"pitch.fmax_hz={fmax:g} supera la frecuencia de Nyquist sr/2 = {sr / 2:g} Hz.")
        elif not 64 <= frame or sr / fmin >= frame // 2:
            problems.append(f"pitch.frame_length={frame} muestras es demasiado corto: pYIN necesita al menos dos "
                            f"periodos de fmin_hz={fmin:g} Hz ({2 * sr / fmin:.0f} muestras).")
        env = self.env
        if not 0 <= env.n_frets <= 24:
            problems.append(f"env.n_frets debe estar entre 0 y 24 (recibido {env.n_frets}).")
        elif env.initial_hand_fret > env.n_frets:
            problems.append(f"env.initial_hand_fret={env.initial_hand_fret} está fuera del diapasón (0–{env.n_frets}).")
        if env.cqt_fmin_hz <= 0 or env.n_octaves < 1:
            problems.append("env.cqt_fmin_hz debe ser > 0 Hz y env.n_octaves ≥ 1.")
        elif env.spectrum == "cqt" and env.cqt_fmin_hz * 2.0 ** env.n_octaves >= sr / 2:
            problems.append(f"La CQT llegaría a {env.cqt_fmin_hz * 2.0 ** env.n_octaves:.0f} Hz, por encima de la "
                            f"frecuencia de Nyquist ({sr / 2:g} Hz): reduce env.n_octaves.")
        unknown = [a for a in self.experiment.algorithms if a not in ALGORITHMS]
        if unknown or not self.experiment.algorithms:
            problems.append(f"experiment.algorithms debe ser una lista no vacía de {', '.join(ALGORITHMS)}"
                            + (f" (desconocidos: {', '.join(unknown)})." if unknown else "."))
        if self.experiment.example_segment is not None and self.experiment.example_segment < 0:
            problems.append("experiment.example_segment debe ser ≥ 0 o null.")
        return problems

    def save_json(self, path: str | Path) -> None:
        """Guarda la configuración en un archivo JSON legible (UTF-8, indentado).

        Parameters
        ----------
        path : str | Path
            Archivo de salida (se sobrescribe si existe).
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        """Carga y valida una configuración desde un archivo JSON.

        Parameters
        ----------
        path : str | Path
            Archivo JSON (por ejemplo, el guardado por :meth:`save_json`).

        Returns
        -------
        Config
            Configuración validada (ver :meth:`from_dict`).

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        ValueError
            Si la ruta es una carpeta, el archivo no es JSON válido o la
            configuración no es válida.
        """
        path = Path(path)
        if path.is_dir():
            raise ValueError(f"{path} es una carpeta, no un archivo de configuración JSON.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} no es un JSON válido: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path.name} no es un archivo de texto UTF-8: {exc}") from exc
        return cls.from_dict(data)

    def get(self, key: str) -> Any:
        """Lee un parámetro con notación ``"seccion.campo"``.

        Parameters
        ----------
        key : str
            Clave como ``"env.lam"``.

        Returns
        -------
        Any
            Valor actual del parámetro.

        Examples
        --------
        >>> Config().get("env.budget")
        500
        """
        section, name = key.split(".", 1)
        return getattr(getattr(self, section), name)

    def set(self, key: str, value: Any) -> None:
        """Escribe un parámetro con notación ``"seccion.campo"`` (sin validar; ver :meth:`validate`).

        Parameters
        ----------
        key : str
            Clave como ``"env.lam"``.
        value : Any
            Nuevo valor.
        """
        section, name = key.split(".", 1)
        setattr(getattr(self, section), name, value)


#: Listas de los barridos y el parámetro cuyo rango deben respetar.
_SWEEP_LISTS: dict[str, str] = {
    "sweep_epsilon": "agent.epsilon",
    "sweep_q0": "agent.q0",
    "sweep_c": "agent.ucb_c",
    "sweep_tau": "agent.tau",
    "sweep_lambda": "env.lam",
}


def _coerce_value(key: str, value: Any, annotation: str) -> Any:
    """Convierte ``value`` al tipo del campo (anotación en texto) o lanza ``ValueError``.

    Solo se admiten conversiones sin pérdida: un ``int`` en un campo ``float``
    y un float ENTERO (``5.0``) en un campo ``int``. Un texto como ``"0.1"``
    o un booleano en un campo numérico es un error.

    Parameters
    ----------
    key : str
        Clave ``"seccion.campo"`` (para el mensaje de error).
    value : Any
        Valor leído.
    annotation : str
        Anotación del campo: ``"int"``, ``"float"``, ``"bool"``, ``"str"``,
        ``"int | None"``, ``"list[float]"`` o ``"list[str]"``.

    Returns
    -------
    Any
        Valor convertido.

    Raises
    ------
    ValueError
        Si el valor no es del tipo esperado.

    Examples
    --------
    >>> _coerce_value("env.n_harmonics", 5.0, "int"), _coerce_value("env.lam", 1, "float")
    (5, 1.0)
    """

    def bad(expected: str) -> ValueError:
        """Error en español: qué se esperaba y qué se recibió."""
        return ValueError(f"{key} debe ser {expected} (recibido {value!r}, de tipo {type(value).__name__}).")

    if annotation == "bool":
        if isinstance(value, bool):
            return value
        raise bad("true o false")
    if annotation in ("int", "int | None"):
        if value is None and annotation == "int | None":
            return None
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
            return int(value)
        raise bad("un entero" + (" o null" if annotation == "int | None" else ""))
    if annotation == "float":
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise bad("un número")
        if not math.isfinite(float(value)):
            raise bad("un número finito")
        return float(value)
    if annotation == "str":
        if isinstance(value, str):
            return value
        raise bad("un texto")
    if annotation in ("list[float]", "list[str]"):
        item = "float" if annotation == "list[float]" else "str"
        if not isinstance(value, (list, tuple)):
            raise bad("una lista de " + ("números" if item == "float" else "textos"))
        return [_coerce_value(f"{key}[{i}]", v, item) for i, v in enumerate(value)]
    return value


def _spec_problems(key: str, value: Any, spec: "ParamSpec") -> list[str]:
    """Problemas de ``value`` respecto al rango u opciones de su :class:`ParamSpec`.

    Parameters
    ----------
    key : str
        Clave ``"seccion.campo"``.
    value : Any
        Valor actual (ya con el tipo correcto).
    spec : ParamSpec
        Especificación del parámetro.

    Returns
    -------
    list[str]
        Mensajes en español (vacía si el valor es válido).
    """
    if spec.kind == "choice":
        return [] if value in spec.choices else [f"{key} debe ser una de: {', '.join(spec.choices)} (recibido {value!r})."]
    if spec.kind in ("int", "float", "int_or_none"):
        if value is None:
            return [] if spec.kind == "int_or_none" else [f"{key} no puede ser null."]
        low = -math.inf if spec.minimum is None else spec.minimum
        high = math.inf if spec.maximum is None else spec.maximum
        if not low <= value <= high:
            unit = f" {spec.unit}" if spec.unit else ""
            return [f"{key} debe estar entre {low:g} y {high:g}{unit} (recibido {value!r})."]
    return []


# ---------------------------------------------------------------------------
# Metadatos de parámetros para la GUI (etiquetas, tooltips y rangos)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Descripción de un hiperparámetro para construir su control en la GUI.

    Attributes
    ----------
    label : str
        Etiqueta corta en español.
    help : str
        Texto del tooltip: qué hace el parámetro y cómo afecta al resultado.
    kind : str
        ``"int"``, ``"float"``, ``"bool"``, ``"choice"``, ``"int_or_none"`` o
        ``"float_list"``.
    minimum, maximum : float | None
        Rango válido (inclusive) para tipos numéricos.
    step : float | None
        Incremento sugerido para el spinbox.
    choices : tuple[str, ...]
        Opciones válidas cuando ``kind == "choice"``.
    unit : str
        Unidad mostrada junto al control (``"Hz"``, ``"s"``, ``"dB"``...).
    choice_labels : tuple[tuple[str, str], ...]
        Texto en español que la GUI muestra para cada opción, como pares
        ``(valor, etiqueta)`` (p. ej. ``("most_pulled", "más jalado (robusto)")``);
        el valor guardado en la configuración sigue siendo el identificador.
        Las opciones sin etiqueta se muestran tal cual.
    """

    label: str
    help: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    choice_labels: tuple[tuple[str, str], ...] = ()

    def choice_label(self, value: str) -> str:
        """Etiqueta en español de la opción ``value`` (o ``value`` si no tiene).

        Examples
        --------
        >>> PARAM_SPECS["agent.recommend"].choice_label("greedy")
        'mayor Q (greedy)'
        >>> PARAM_SPECS["agent.recommend"].choice_label("otra")
        'otra'
        """
        return dict(self.choice_labels).get(value, value)

    def choice_value(self, text: str) -> str | None:
        """Valor de la opción cuya etiqueta (o cuyo identificador) es ``text``; None si no existe.

        Examples
        --------
        >>> PARAM_SPECS["agent.recommend"].choice_value("más jalado (robusto)")
        'most_pulled'
        >>> PARAM_SPECS["agent.recommend"].choice_value("greedy"), PARAM_SPECS["agent.recommend"].choice_value("x")
        ('greedy', None)
        """
        for value in self.choices:
            if text in (value, self.choice_label(value)):
                return value
        return None


#: Especificación de cada parámetro editable, indexada por ``"seccion.campo"``.
PARAM_SPECS: dict[str, ParamSpec] = {
    # --- Audio (etapa 2: separación) ---
    "audio.separation_method": ParamSpec(
        "Método de separación del bajo", "Cómo se aísla el bajo de una mezcla cuando está marcada «Separar "
        "bajo de mezcla». «auto»: Demucs si está instalado y, si no, HPSS. «demucs»: red neuronal Demucs, "
        "la mejor calidad (quita voz, batería, guitarras y teclados), pero requiere PyTorch (pip install "
        "demucs), descarga ~80 MB de pesos la primera vez y tarda varios minutos por canción en CPU. "
        "«hpss»: separación armónico-percusiva de librosa + pasa-bajas de 1.2 kHz; tarda segundos y no "
        "necesita nada más, pero es una aproximación: quita batería, efectos percusivos y voces/platillos "
        "agudos, no las guitarras ni los teclados graves. Solo actúa si está marcada «Separar bajo de mezcla» "
        "en la pestaña Audio.", "choice", choices=SEPARATION_METHODS,
        choice_labels=(("auto", "automático"), ("demucs", "Demucs"), ("hpss", "HPSS"))),
    # --- Preprocesamiento / segmentación / pitch ---
    "preprocess.lowpass_hz": ParamSpec(
        "Corte pasa-bajas", "Frecuencia de corte del filtro Butterworth aplicado a la señal de "
        "análisis (onsets y pYIN). Elimina ruido de trastes y armónicos agudos que confunden a pYIN. "
        "La recompensa usa la señal sin filtrar para conservar los armónicos.",
        "float", 100.0, 4000.0, 50.0, unit="Hz"),
    "preprocess.filter_order": ParamSpec(
        "Orden del filtro", "Orden del Butterworth. Mayor orden = caída más abrupta después del corte.",
        "int", 1, 10, 1),
    "segmentation.onset_delta": ParamSpec(
        "Umbral de onset (δ)", "Umbral sobre la función de novedad espectral para aceptar un pico como "
        "inicio de nota. Más alto = menos onsets (se pierden notas suaves); más bajo = onsets falsos.",
        "float", 0.0, 1.0, 0.01),
    "segmentation.onset_wait_s": ParamSpec(
        "Separación mínima", "Tiempo mínimo entre dos onsets consecutivos. Evita detectar dos veces el "
        "mismo ataque.", "float", 0.0, 0.5, 0.01, unit="s"),
    "segmentation.rms_threshold_db": ParamSpec(
        "Umbral de silencio", "Los segmentos cuyo RMS medio está por debajo de este nivel (dB relativos al "
        "máximo de la pista) se consideran silencio y se descartan.", "float", -80.0, 0.0, 1.0, unit="dB"),
    "segmentation.min_duration_s": ParamSpec(
        "Duración mínima", "Segmentos más cortos que esto se descartan (no hay suficientes frames para "
        "estimar pitch ni para muestrear recompensas).", "float", 0.01, 1.0, 0.01, unit="s"),
    "pitch.min_voiced_ratio": ParamSpec(
        "Fracción con voz mínima", "Si pYIN detecta voz en menos de esta fracción de frames del segmento, "
        "la f0 se considera desconocida y se usan todos los brazos.", "float", 0.0, 1.0, 0.05),
    "pitch.max_transition_rate": ParamSpec(
        "Salto máximo de pitch (pYIN)", "Máximo cambio de pitch entre frames que admite el HMM de pYIN. "
        "Valores bajos (librosa: 35.92) suavizan la trayectoria pero arrastran la nota anterior por la "
        "suboctava en notas rápidas (errores de octava).", "float", 10.0, 500.0, 10.0, unit="oct/s"),
    "pitch.switch_prob": ParamSpec(
        "Prob. de cambio voz/sin voz (pYIN)", "Probabilidad de transición entre 'con voz' y 'sin voz' en el "
        "HMM de pYIN (librosa: 0.01). Más alta = el HMM corta antes entre notas.", "float", 0.001, 0.5, 0.01),
    # --- Entorno ---
    "env.k_semitones": ParamSpec(
        "k (poda ± semitonos)", "Solo se crean brazos (cuerda, traste) a ±k semitonos de la f0 de pYIN. "
        "k pequeño = problema bandit fácil pero sensible a errores de pYIN; vacío (None) = 52 brazos.",
        "int_or_none", 0, 24, 1),
    "env.n_harmonics": ParamSpec(
        "N armónicos", "Número de armónicos (incluida la fundamental) del template con que se mide la "
        "energía de cada posición candidata.", "int", 1, 12, 1),
    "env.beta": ParamSpec(
        "β inter-armónico", "Peso de la energía medida ENTRE armónicos ((h−½)·f), que se resta. Castiga "
        "los errores de octava: un candidato una octava arriba tiene sus medios-armónicos sobre los "
        "armónicos impares reales. 0 = suma armónica pura.", "float", 0.0, 2.0, 0.05),
    "env.lam": ParamSpec(
        "λ tocabilidad", "Peso de la penalización λ·|traste − traste_previo|/12. Es lo único que distingue "
        "posiciones con el mismo pitch (p. ej. A-0 y E-5). 0 = ignorar la tocabilidad.",
        "float", 0.0, 2.0, 0.01),
    "env.budget": ParamSpec(
        "T pulls por segmento", "Presupuesto de interacciones del agente con cada segmento. Más pulls = "
        "estimaciones Q más precisas pero más cómputo.", "int", 10, 5000, 10),
    "env.n_fft": ParamSpec(
        "Tamaño FFT (STFT)", "Muestras de la ventana de la STFT: más grande = mejor resolución en frecuencia "
        "(sr/n_fft Hz por bin; 8192 → 2.7 Hz) pero peor en tiempo (8192 ≈ 0.37 s): las notas vecinas se mezclan.",
        "int", 1024, 16384, 1024, unit="muestras"),
    "env.bins_per_octave": ParamSpec(
        "Bins por octava (CQT)", "Resolución de la CQT: 36 = tercio de semitono. Más bins = picos más finos pero "
        "ventanas más largas en el grave (más de 1 s en E1 con 36 bins/oct).", "int", 12, 48, 12, unit="bins/oct"),
    "env.spectrum": ParamSpec(
        "Espectro", "Representación tiempo-frecuencia usada para la recompensa: STFT (ventana fija de "
        "n_fft muestras; por defecto 8192 ≈ 0.37 s) o CQT (resolución logarítmica, 1/3 de semitono, pero "
        "ventanas de más de 1 s en el grave que mezclan notas vecinas).", "choice", choices=("cqt", "stft"),
        choice_labels=(("cqt", "CQT"), ("stft", "STFT"))),
    "env.tolerance_semitones": ParamSpec(
        "Tolerancia de afinación", "Al leer la energía en h·f se toma el máximo dentro de ± esta "
        "cantidad de semitonos (absorbe desafinación e inarmonicidad de la cuerda).",
        "float", 0.0, 1.0, 0.05, unit="st"),
    "env.attack_skip_s": ParamSpec(
        "Ignorar ataque", "Los primeros segundos de cada nota (ataque percusivo) no se muestrean como "
        "frames de recompensa.", "float", 0.0, 0.2, 0.005, unit="s"),
    "env.initial_hand_fret": ParamSpec(
        "Posición inicial de la mano", "traste_previo supuesto para el primer segmento.",
        "int", 0, 12, 1),
    "env.open_string_free": ParamSpec(
        "Cuerdas al aire gratis", "Extensión opcional: tocar una cuerda al aire no cuesta movimiento y la "
        "mano se queda donde estaba.", "bool"),
    "env.noise_std": ParamSpec(
        "Ruido extra (σ)", "Ruido gaussiano añadido a cada recompensa para hacer el problema más difícil "
        "(0 = solo la estocasticidad del muestreo de frames).", "float", 0.0, 1.0, 0.01),
    # --- Agentes ---
    "agent.epsilon": ParamSpec(
        "ε", "Probabilidad de explorar (elegir un brazo al azar) en cada pull. Con 1−ε se explota el "
        "brazo de mayor Q.", "float", 0.0, 1.0, 0.01),
    "agent.epsilon_decay": ParamSpec(
        "Decaimiento de ε", "Factor por pull: ε_t = max(ε_min, ε₀·dᵗ). 1 = ε constante; 0.99 = explora "
        "mucho al inicio y poco al final.", "float", 0.9, 1.0, 0.001),
    "agent.epsilon_min": ParamSpec(
        "ε mínimo", "Cota inferior de ε cuando hay decaimiento.", "float", 0.0, 1.0, 0.01),
    "agent.q0": ParamSpec(
        "Q₀ optimista", "Valor inicial de todas las estimaciones del agente optimista. Si Q₀ supera la "
        "recompensa máxima (1), cada brazo 'decepciona' al probarlo y el agente greedy se ve forzado a "
        "explorar los demás.", "float", 0.0, 20.0, 0.5),
    "agent.optimistic_epsilon": ParamSpec(
        "ε del optimista", "Exploración aleatoria adicional del agente optimista (normalmente 0).",
        "float", 0.0, 1.0, 0.01),
    "agent.optimistic_alpha": ParamSpec(
        "α del optimista", "Tamaño de paso constante: Q ← Q + α(r − Q). Con α constante el optimismo se "
        "desvanece gradualmente; con 1/n desaparecería tras un pull.", "float", 0.001, 1.0, 0.01),
    "agent.ucb_c": ParamSpec(
        "c de UCB1", "Peso del bono de incertidumbre c·√(ln t / n_a). Mayor c = más exploración de brazos "
        "poco probados. √2 ≈ 1.414 es el valor del UCB1 original.", "float", 0.0, 10.0, 0.1),
    "agent.tau": ParamSpec(
        "τ (temperatura)", "Temperatura de Boltzmann: π(a) ∝ exp(Q_a/τ). τ alta = elección casi uniforme; "
        "τ→0 = greedy. Depende de la escala de las recompensas.", "float", 0.001, 10.0, 0.01),
    "agent.tau_decay": ParamSpec(
        "Annealing de τ", "Factor por pull: τ_t = max(τ_min, τ₀·dᵗ). 1 = temperatura constante.",
        "float", 0.9, 1.0, 0.001),
    "agent.tau_min": ParamSpec(
        "τ mínima", "Cota inferior de τ con annealing.", "float", 0.0001, 1.0, 0.001),
    "agent.recommend": ParamSpec(
        "Recomendación final", "Posición reportada al agotar el presupuesto: el brazo más jalado "
        "(robusto) o el de mayor Q (greedy).", "choice", choices=("most_pulled", "greedy"),
        choice_labels=(("most_pulled", "más jalado (robusto)"), ("greedy", "mayor Q (greedy)"))),
    # --- Experimento ---
    "experiment.n_runs": ParamSpec(
        "Corridas", "Número de corridas independientes con semillas distintas para promediar curvas.",
        "int", 1, 2000, 10),
    "experiment.seed": ParamSpec(
        "Semilla", "Semilla maestra: con la misma semilla y configuración los resultados son idénticos.",
        "int", 0, 2**31 - 1, 1),
    "experiment.sweep_runs": ParamSpec(
        "Corridas por barrido", "Corridas por cada valor en los barridos de sensibilidad (ε, Q₀, c, τ y λ). "
        "En pistas de más de 50 notas se reducen automáticamente para que cada barrido cueste como el de "
        "una pista de 50 notas (una canción de 320 notas pasa de 30 a 5 corridas por valor); el pie de esta "
        "sección y la pestaña Comparación dicen cuántas se usarán.",
        "int", 1, 500, 5),
    "experiment.run_sweeps": ParamSpec(
        "Barridos de sensibilidad", "Ejecutar los barridos de ε, Q₀, c y τ.", "bool"),
    "experiment.run_lambda_sweep": ParamSpec(
        "Barrido de λ", "Ejecutar el barrido de λ (efecto de la tocabilidad en la precisión de posición).",
        "bool"),
    "experiment.sweep_epsilon": ParamSpec("Valores de ε", "Lista separada por comas.", "float_list"),
    "experiment.sweep_q0": ParamSpec("Valores de Q₀", "Lista separada por comas.", "float_list"),
    "experiment.sweep_c": ParamSpec("Valores de c", "Lista separada por comas.", "float_list"),
    "experiment.sweep_tau": ParamSpec("Valores de τ", "Lista separada por comas.", "float_list"),
    "experiment.sweep_lambda": ParamSpec("Valores de λ", "Lista separada por comas.", "float_list"),
    "experiment.onset_tolerance_s": ParamSpec(
        "Tolerancia de onset", "Distancia máxima entre el onset detectado y el del ground truth para "
        "considerarlos la misma nota.", "float", 0.005, 0.5, 0.005, unit="s"),
}

#: Ayuda de las listas de barrido: qué se barre, con qué algoritmo y qué se mide.
#: El rango válido se toma del parámetro barrido (mismo criterio que Config.validate).
_SWEEP_HELP: dict[str, tuple[str, str]] = {
    "experiment.sweep_epsilon": (
        "agent.epsilon",
        "Valores de ε que se prueban con ε-greedy (probabilidad de elegir un brazo al azar; 0 = greedy "
        "puro, 1 = siempre al azar). Cada valor se ejecuta «Corridas por barrido» veces sobre todas las "
        "notas y se mide la recompensa media del último 10 % de los pulls (gráfica «Sensibilidad»)."),
    "experiment.sweep_q0": (
        "agent.q0",
        "Valores iniciales Q₀ que se prueban con ε-greedy optimista (α constante). Con Q₀ por encima de "
        "la recompensa máxima (≈ 1) todos los brazos «decepcionan» al jalarlos y el agente los prueba "
        "todos. Cada valor se ejecuta «Corridas por barrido» veces y se mide la recompensa media del "
        "último 10 % de los pulls (gráfica «Sensibilidad»)."),
    "experiment.sweep_c": (
        "agent.ucb_c",
        "Constantes de exploración c que se prueban con UCB1 (índice Q + c·√(ln t / n); c = 0 es greedy, "
        "√2 ≈ 1.414 el UCB1 original). Cada valor se ejecuta «Corridas por barrido» veces y se mide la "
        "recompensa media del último 10 % de los pulls (gráfica «Sensibilidad»)."),
    "experiment.sweep_tau": (
        "agent.tau",
        "Temperaturas τ que se prueban con Softmax (Boltzmann): τ → 0 elige casi siempre el mayor Q; τ "
        "grande reparte los pulls casi al azar. Cada valor se ejecuta «Corridas por barrido» veces y se "
        "mide la recompensa media del último 10 % de los pulls (gráfica «Sensibilidad»)."),
    "experiment.sweep_lambda": (
        "env.lam",
        "Pesos λ de la penalización de tocabilidad que se prueban con los cuatro algoritmos y los dos "
        "oráculos; se mide la precisión de posición y de pitch frente al ground truth (gráfica «Efecto "
        "de λ»; sin .gt.json se omite)."),
}
for _key, (_param, _text) in _SWEEP_HELP.items():
    _base = PARAM_SPECS[_param]
    _range = f" Rango válido: {_base.minimum:g}–{_base.maximum:g}." if _base.minimum is not None else ""
    PARAM_SPECS[_key] = ParamSpec(PARAM_SPECS[_key].label, _text + _range + " Separa los valores con comas.",
                                  "float_list")
del _key, _param, _text, _base, _range
