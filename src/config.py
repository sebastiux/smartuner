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

#: Nombres legibles (español) de cada algoritmo.
ALGO_LABELS: dict[str, str] = {
    "egreedy": "ε-greedy",
    "optimistic": "ε-greedy optimista",
    "ucb1": "UCB1",
    "softmax": "Softmax (Boltzmann)",
    "oracle": "Oráculo (argmax μ)",
}

#: Color fijo por algoritmo (paleta categórica validada para daltonismo,
#: orden adyacente). El oráculo es una referencia neutra en gris.
ALGO_COLORS: dict[str, str] = {
    "egreedy": "#2a78d6",
    "optimistic": "#eb6834",
    "ucb1": "#1baf7a",
    "softmax": "#eda100",
    "oracle": "#52514e",
}

#: Codificación secundaria (marcador y estilo de línea) para que la identidad
#: de cada algoritmo nunca dependa solo del color.
ALGO_MARKERS: dict[str, str] = {
    "egreedy": "o",
    "optimistic": "s",
    "ucb1": "^",
    "softmax": "D",
    "oracle": "x",
}
ALGO_LINESTYLES: dict[str, str] = {
    "egreedy": "-",
    "optimistic": "--",
    "ucb1": "-.",
    "softmax": ":",
    "oracle": (0, (1, 1)),  # type: ignore[dict-item]
}

#: Frecuencias de las cuerdas al aire del bajo en afinación estándar (Hz),
#: de la más grave a la más aguda.
OPEN_STRING_HZ: dict[str, float] = {"E": 41.20, "A": 55.00, "D": 73.42, "G": 98.00}

#: Número MIDI de cada cuerda al aire (E1=28, A1=33, D2=38, G2=43).
OPEN_STRING_MIDI: dict[str, int] = {"E": 28, "A": 33, "D": 38, "G": 43}

#: Orden de cuerdas de grave a aguda. La tablatura se imprime al revés (G arriba).
STRING_ORDER: tuple[str, ...] = ("E", "A", "D", "G")

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
        Frecuencia de muestreo de trabajo en Hz (22050 por defecto).
    separate_bass : bool
        Si es True se ejecuta Demucs para aislar el bajo de una mezcla.
    demucs_model : str
        Nombre del modelo de Demucs (``"htdemucs"``).
    """

    sample_rate: int = 22050
    separate_bass: bool = False
    demucs_model: str = "htdemucs"


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
    """

    hop_length: int = 256
    onset_delta: float = 0.07
    onset_wait_s: float = 0.05
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
    """

    fmin_hz: float = 35.0
    fmax_hz: float = 250.0
    frame_length: int = 2048
    min_voiced_ratio: float = 0.2


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
        Resolución de la CQT (36 = tercio de semitono).
    cqt_fmin_hz : float
        Frecuencia más baja de la CQT (C1 = 32.70 Hz).
    n_octaves : int
        Octavas cubiertas por la CQT (6 → hasta C7 ≈ 2093 Hz).
    n_fft : int
        Tamaño de la FFT cuando ``spectrum == "stft"``.
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
    """

    n_frets: int = 12
    k_semitones: int | None = 2
    n_harmonics: int = 5
    beta: float = 0.5
    lam: float = 0.1
    budget: int = 300
    spectrum: str = "cqt"
    bins_per_octave: int = 36
    cqt_fmin_hz: float = 32.70
    n_octaves: int = 6
    n_fft: int = 4096
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
    onset_tolerance_s: float = 0.05


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
    Las claves desconocidas al cargar se ignoran (compatibilidad hacia atrás).
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Devuelve la configuración como diccionario anidado (serializable a JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Construye una configuración a partir de un diccionario anidado.

        Parameters
        ----------
        data : dict
            Diccionario con secciones como las de :meth:`to_dict`. Las
            secciones o claves ausentes toman su valor por defecto.
        """
        kwargs: dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            section_data = data.get(name, {}) or {}
            valid = {f.name for f in fields(section_cls)}
            kwargs[name] = section_cls(**{k: v for k, v in section_data.items() if k in valid})
        return cls(**kwargs)

    def save_json(self, path: str | Path) -> None:
        """Guarda la configuración en un archivo JSON legible."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        """Carga una configuración desde un archivo JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, key: str) -> Any:
        """Lee un parámetro con notación ``"seccion.campo"`` (p. ej. ``"env.lam"``)."""
        section, name = key.split(".", 1)
        return getattr(getattr(self, section), name)

    def set(self, key: str, value: Any) -> None:
        """Escribe un parámetro con notación ``"seccion.campo"``."""
        section, name = key.split(".", 1)
        setattr(getattr(self, section), name, value)


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
    """

    label: str
    help: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""


#: Especificación de cada parámetro editable, indexada por ``"seccion.campo"``.
PARAM_SPECS: dict[str, ParamSpec] = {
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
    "env.spectrum": ParamSpec(
        "Espectro", "Representación tiempo-frecuencia usada para la recompensa: CQT (resolución "
        "logarítmica, 1/3 de semitono) o STFT (ventana fija).", "choice", choices=("cqt", "stft")),
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
        "(robusto) o el de mayor Q (greedy).", "choice", choices=("most_pulled", "greedy")),
    # --- Experimento ---
    "experiment.n_runs": ParamSpec(
        "Corridas", "Número de corridas independientes con semillas distintas para promediar curvas.",
        "int", 1, 2000, 10),
    "experiment.seed": ParamSpec(
        "Semilla", "Semilla maestra: con la misma semilla y configuración los resultados son idénticos.",
        "int", 0, 2**31 - 1, 1),
    "experiment.sweep_runs": ParamSpec(
        "Corridas por barrido", "Corridas por cada valor en los barridos de sensibilidad.",
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
