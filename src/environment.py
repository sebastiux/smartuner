"""Etapa 6 — Entorno bandit por segmento: brazos, recompensa armónica y regret.

Papel en el pipeline
--------------------
Las etapas 1–5 entregan, para cada nota detectada, un :class:`Segment` con
sus tiempos y una f0 aproximada (pYIN). Este módulo convierte cada segmento
en un problema **Multi-Armed Bandit** independiente que los agentes de
:mod:`src.agents` resuelven pull a pull:

    espectro de la pista ─┐
    segmento + f0 ────────┼─► brazos candidatos ─► matriz salience[frame, brazo]
    configuración (k,N,β) ┘                                   │
                                                              ▼
                       BanditEnvironment.pull(a)  ─►  r = S_j(f_a) − penalización

Brazos
------
Cada brazo es una posición (cuerda, traste) del diapasón, con frecuencia

    f(cuerda, traste) = f_cuerda · 2^(traste/12)

(cada traste sube un semitono = multiplica la frecuencia por 2^(1/12)). En
afinación estándar hay 4 cuerdas × 13 trastes (0–12) = 52 brazos. Se podan a
los que están a ±k semitonos de la f0 estimada por pYIN (k=None → 52 brazos):
la f0 solo sirve para que el problema sea más pequeño; la DECISIÓN la toman
los agentes.

Recompensa de un pull
---------------------
Al jalar el brazo a se elige un frame j al azar del segmento y se calcula::

    X̃_j      = |X_j| / max|X_j|                          (espectro del frame normalizado)
    S⁺_j(f)  = Σ_h w_h · X̃_j(h·f) / Σ_h w_h             (energía en los armónicos)
    S⁻_j(f)  = Σ_h w_h · X̃_j((h−½)·f) / Σ_h w_h         (energía ENTRE armónicos)
    S_j(f)   = max(S⁺_j(f) − β·S⁻_j(f), 0)               (saliencia ∈ [0, 1])
    r        = S_j(f_a) − λ·|traste_a − traste_previo|/12 (+ ruido opcional)

con h = 1..N y w_h = 1/h. X̃(x) se lee como el máximo de la magnitud
interpolada dentro de ±``tolerance_semitones`` alrededor de x (máximo exacto
de la interpolación lineal en la ventana, ver :func:`salience_components`).
La saliencia depende solo del PITCH del brazo: se calcula una vez por nota
MIDI m con la frecuencia temperada f(m) = 440·2^((m − 69)/12) Hz y se copia
a todas las posiciones (cuerda, traste) que tocan esa nota.

**Template armónico.** Una cuerda pulsada vibra a la vez en su fundamental f
y en sus múltiplos 2f, 3f, 4f... (serie armónica). La "huella" de una nota en
el espectro es, por tanto, un peine de picos en h·f. La saliencia S⁺ mide
cuánto encaja el espectro observado con el peine de cada candidato
(*suma armónica*, Klapuri 2006). Los pesos w_h = 1/h dan más importancia a
los armónicos graves, que en el bajo son los más fuertes y fiables, y
favorecen al candidato más grave que explica el peine. Como se divide entre
Σ_h w_h (los pesos normalizados w_h/Σ_k w_k suman 1) y 0 ≤ X̃ ≤ 1, S⁺ es una
media ponderada y queda en [0, 1].

**¿Por qué β castiga los errores de octava?** El candidato una octava arriba,
2f₀, tiene armónicos {2f₀, 4f₀, 6f₀, ...}: todos son armónicos REALES de la
nota f₀ (los pares). La suma armónica pura (β=0) no puede descartarlo y, si la
fundamental es débil (algo muy habitual en el bajo: altavoces pequeños,
micrófonos, fundamental ausente), puede incluso preferirlo. Pero sus
posiciones inter-armónicas (h−½)·2f₀ = (2h−1)·f₀ = {f₀, 3f₀, 5f₀, ...} caen
exactamente sobre los armónicos IMPARES de la nota real, que tienen mucha
energía → S⁻ grande → β·S⁻ lo hunde. Para el candidato correcto f₀, las
posiciones (h−½)·f₀ = {½f₀, 1½f₀, 2½f₀, ...} están entre picos y no tienen
energía → S⁻ ≈ 0. El error contrario (una octava abajo, f₀/2) ya lo castiga
S⁺: la mitad de su peine (los múltiplos impares de f₀/2) cae en zonas vacías.
β = 0 recupera la suma armónica pura (ver ``tests/test_reward.py``).

**¿Por qué el valor real μ se puede calcular exactamente?** El segmento tiene
un número FINITO de frames y el entorno elige uno de forma uniforme. La
recompensa del brazo a es una variable aleatoria discreta que toma el valor
S_j(f_a) − pen_a con probabilidad 1/n_frames (más ruido de media cero), así
que su esperanza es exactamente::

    μ_a = E[r | a] = (1/n_frames) · Σ_j S_j(f_a) − pen_a

Basta precalcular la matriz ``salience[frame, brazo]`` para obtener μ sin
simulación. El agente NUNCA ve μ (solo muestras r); μ se usa como "oráculo"
para medir el regret μ* − μ_{a_t} y el % de veces que se eligió el óptimo.

**¿Qué distingue posiciones con el mismo pitch?** A-0 y E-5 suenan a la misma
nota (A1, MIDI 33): el espectro no "sabe" en qué cuerda se tocó. Por eso su
saliencia se calcula UNA vez, con la frecuencia temperada de la nota
(55.00 Hz), y es idéntica en cada frame. Calcularla con la frecuencia de cada
cuerda (55.00 Hz frente a 41.20·2^(5/12) ≈ 54.995 Hz, por el redondeo de
:data:`src.config.OPEN_STRING_HZ`) daría diferencias de ~10⁻⁴ que no son
música sino artefactos numéricos, y decidirían el desempate. Así, la única
diferencia entre sus μ es la penalización de tocabilidad λ·|Δtraste|/12, que
favorece la posición que exige MENOS movimiento de la mano respecto al traste
previo; con λ = 0 esas posiciones quedan EXACTAMENTE empatadas (todas son
óptimas y el oráculo aplica su regla de desempate).

Eficiencia y números aleatorios comunes
---------------------------------------
Para que cada pull sea O(1) se precalcula, una sola vez por segmento, la
matriz ``salience[frame, brazo]`` (:class:`SegmentBanditData`); el entorno
(:class:`BanditEnvironment`) solo añade la penalización, que depende del
traste previo elegido por el propio agente. Cada pull consume exactamente UNA
llamada al generador para elegir el frame (y otra para el ruido solo si
``noise_std > 0``), de modo que con la misma semilla todos los algoritmos ven
la MISMA secuencia de frames aunque jalen brazos distintos (números
aleatorios comunes): las diferencias entre algoritmos no se deben a la suerte.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from src.config import OPEN_STRING_HZ, OPEN_STRING_MIDI, STRING_ORDER, EnvConfig
from src.pitch import hz_to_midi, midi_to_hz, segment_log_name
from src.segmentation import Segment

logger = logging.getLogger(__name__)

#: Un frame cuyo máximo de magnitud no supera este valor se considera
#: silencioso: no se normaliza (evita dividir entre ~0) y su saliencia es 0.
SILENCE_FLOOR: float = 1e-10

#: Tolerancia absoluta para considerar empatados dos valores reales μ.
OPTIMAL_TOL: float = 1e-12

#: Máximo de elementos de la matriz temporal (frames × consultas) que se
#: materializa a la vez en :func:`salience_components` (acota la memoria).
_MAX_GATHER_ELEMENTS: int = 2_000_000


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

    Examples
    --------
    >>> arm = Arm("A", 0)
    >>> arm.label, arm.midi, arm.freq_hz
    ('A-0', 33, 55.0)
    >>> Arm("G", 12).freq_hz
    196.0
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
    """f(cuerda, traste) = f_cuerda · 2^(traste/12) en Hz.

    Cada traste acorta la cuerda de modo que la frecuencia sube un semitono
    temperado, es decir, se multiplica por 2^(1/12) ≈ 1.0595; 12 trastes
    duplican la frecuencia (una octava).

    Parameters
    ----------
    string : str
        Cuerda (``"E"``, ``"A"``, ``"D"`` o ``"G"``).
    fret : int
        Traste (0 = cuerda al aire).

    Returns
    -------
    float
        Frecuencia fundamental en Hz.

    Examples
    --------
    >>> fret_frequency("A", 0)
    55.0
    >>> round(fret_frequency("E", 5), 3)
    54.995
    >>> fret_frequency("A", 12)
    110.0
    """
    return float(OPEN_STRING_HZ[string] * 2.0 ** (fret / 12.0))


def all_arms(n_frets: int = 12) -> list[Arm]:
    """Las 4·(n_frets+1) posiciones, ordenadas por cuerda (E, A, D, G) y traste.

    Parameters
    ----------
    n_frets : int
        Traste máximo considerado (incluido). 12 → 52 brazos.

    Returns
    -------
    list[Arm]
        ``[E-0, E-1, ..., E-n, A-0, ..., G-n]``.

    Raises
    ------
    ValueError
        Si ``n_frets`` es negativo.

    Examples
    --------
    >>> arms = all_arms()
    >>> len(arms), arms[0].label, arms[13].label, arms[-1].label
    (52, 'E-0', 'A-0', 'G-12')
    """
    if n_frets < 0:
        raise ValueError(f"n_frets debe ser ≥ 0 (recibido {n_frets})")
    return [Arm(string, fret) for string in STRING_ORDER for fret in range(n_frets + 1)]


def candidate_arms(f0_hz: float | None, k: int | None, n_frets: int = 12) -> list[Arm]:
    """Brazos con |midi(brazo) − round(midi(f0))| ≤ k.

    Si ``f0_hz`` es None o ``k`` es None devuelve :func:`all_arms`. Si la
    poda deja la lista vacía (f0 fuera del rango del bajo) también devuelve
    todos los brazos. Orden: por MIDI ascendente y luego por cuerda.

    Parameters
    ----------
    f0_hz : float | None
        f0 estimada del segmento en Hz (None = desconocida).
    k : int | None
        Radio de la poda en semitonos (None = sin poda).
    n_frets : int
        Traste máximo considerado.

    Returns
    -------
    list[Arm]
        Brazos candidatos, ordenados por (MIDI, cuerda de grave a aguda).

    Raises
    ------
    ValueError
        Si ``k`` es negativo.

    Notes
    -----
    El objetivo es la nota temperada más cercana a la f0, ``round(midi(f0))``,
    por lo que una f0 algo desafinada (±50 cents) sigue apuntando a su nota.
    Con k = 0 quedan solo las posiciones de ese pitch (p. ej. A-0 y E-5 para
    55 Hz); con k = 2 se admite que pYIN se equivoque hasta en un tono.

    Examples
    --------
    >>> [a.label for a in candidate_arms(55.0, 0)]
    ['E-5', 'A-0']
    >>> len(candidate_arms(None, 2)), len(candidate_arms(55.0, None))
    (52, 52)
    """
    arms = all_arms(n_frets)
    if f0_hz is None or k is None:
        return arms
    if k < 0:
        raise ValueError(f"k_semitones debe ser ≥ 0 o None (recibido {k})")
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        logger.debug("f0=%s no es válida; se usan los %d brazos", f0_hz, len(arms))
        return arms
    target = int(round(hz_to_midi(f0_hz)))
    pruned = [a for a in arms if abs(a.midi - target) <= k]
    if not pruned:
        logger.debug("f0=%.1f Hz (MIDI %d) está fuera del rango del bajo; se usan los %d brazos",
                     f0_hz, target, len(arms))
        return arms
    return sorted(pruned, key=lambda a: (a.midi, a.string_index))


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
        Frecuencia de muestreo en Hz.
    kind : str
        ``"cqt"`` o ``"stft"``.
    """

    mag: np.ndarray
    freqs_hz: np.ndarray
    times_s: np.ndarray
    hop_length: int
    sr: int
    kind: str


def stft_max_frequency(cfg: EnvConfig) -> float:
    """Frecuencia más alta (Hz) de la STFT que se guarda para la recompensa.

    f_lím = (N + 1) · f(G, n_frets) · 2^(tol/12): el armónico N de la nota
    más aguda del diapasón, más el extremo superior de su ventana de
    tolerancia y un armónico de holgura (la gráfica del template muestra
    hasta (N + 0.6)·f).

    Parameters
    ----------
    cfg : EnvConfig
        Usa ``n_harmonics``, ``n_frets`` y ``tolerance_semitones`` (semitonos).

    Returns
    -------
    float
        Frecuencia límite en Hz.

    Examples
    --------
    >>> round(stft_max_frequency(EnvConfig()), 1)      # (5 + 1) · 196 Hz · 2^(0.33/12)
    1198.6
    """
    f_top = fret_frequency(STRING_ORDER[-1], int(cfg.n_frets))
    return float((int(cfg.n_harmonics) + 1) * f_top * 2.0 ** (max(float(cfg.tolerance_semitones), 0.0) / 12.0))


def compute_spectrum(y: np.ndarray, sr: int, cfg: EnvConfig, hop_length: int = 256) -> Spectrum:
    """Calcula la CQT (``cfg.spectrum == "cqt"``) o la STFT de ``y``.

    Parameters
    ----------
    y : np.ndarray
        Señal mono (normalizada, SIN el pasa-bajas: la recompensa necesita los
        armónicos por encima de 400 Hz).
    sr : int
        Frecuencia de muestreo en Hz.
    cfg : EnvConfig
        Usa ``spectrum``, ``cqt_fmin_hz``, ``n_octaves``, ``bins_per_octave`` y
        ``n_fft`` (y, con la STFT, ``n_harmonics``, ``n_frets`` y
        ``tolerance_semitones`` para recortar la banda guardada).
    hop_length : int
        Salto entre frames en muestras (256 ≈ 11.6 ms a 22 050 Hz).

    Returns
    -------
    Spectrum
        Magnitud ``(n_bins, n_frames)`` en float32, frecuencias de cada bin y
        tiempos (centro) de cada frame.

    Raises
    ------
    ValueError
        Si ``cfg.spectrum`` no es ``"cqt"`` ni ``"stft"``.

    Notes
    -----
    * **CQT** (transformada de Q constante): bins espaciados
      logarítmicamente, ``bins_per_octave`` por octava (36 = un tercio de
      semitono), igual que las notas musicales. Cada armónico h·f cae siempre
      en la misma posición relativa del eje, sea cual sea la nota.
    * **STFT**: bins espaciados linealmente cada sr/n_fft Hz (≈ 2.7 Hz con
      n_fft = 8192, el valor por defecto); en el registro grave del bajo
      (41–200 Hz) un semitono ocupa solo 1–4 bins, pero los armónicos
      superiores sí se separan bien (en 5·f un semitono es 5 veces más
      ancho en Hz). Su ventana es la MISMA para todas las frecuencias
      (8192 muestras ≈ 0.37 s, ancho efectivo de la Hann ≈ 0.19 s).

    **¿Por qué STFT por defecto?** La CQT con 36 bins/octava necesita
    ventanas de ≈ 1.2 s en 41 Hz (Q ≈ 51): en el grave mezcla la nota con
    sus vecinas y, en notas rápidas, el pitch de la nota siguiente "contamina"
    la saliencia (p. ej. E-0 seguido de E-1 en ``cromatica``: el oráculo elegía
    E-1). Ver la comparación numérica en :class:`src.config.EnvConfig`.

    La saliencia lee el espectro interpolando sobre log2(frecuencia), así
    que sirve para ambas representaciones sin cambios.

    **Banda guardada (STFT).** La STFT completa llega hasta sr/2 = 11 025 Hz,
    pero la recompensa solo lee hasta N·f_máx·2^(tol/12) ≈ 1 kHz (ver
    :func:`stft_max_frequency`); se guardan solo esos bins. Consecuencia: la
    normalización por frame X̃ = |X|/max|X| usa el máximo DENTRO de esa
    banda, que en el bajo es el de la fundamental o los primeros armónicos
    (ruido o clics por encima de ≈ 1 kHz ya no cambian la escala).
    """
    import librosa  # import diferido: librosa tarda en cargarse

    kind = str(cfg.spectrum).lower()
    if kind not in ("cqt", "stft"):
        raise ValueError(f"Espectro desconocido: {cfg.spectrum!r} (use 'cqt' o 'stft')")
    y = np.asarray(y, dtype=np.float32)
    if kind == "cqt":
        n_bins = cfg.n_octaves * cfg.bins_per_octave
        with warnings.catch_warnings():
            # La CQT de librosa submuestrea la señal una vez por octava; en clips
            # de pocos segundos la versión más submuestreada es más corta que su
            # FFT interna y librosa avisa "n_fft=... is too large". Es inocuo
            # (se rellena con ceros), así que se silencia solo ese aviso.
            warnings.filterwarnings("ignore", message=r"n_fft=\d+ is too large", category=UserWarning)
            mag = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=cfg.cqt_fmin_hz,
                                     n_bins=n_bins, bins_per_octave=cfg.bins_per_octave))
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=cfg.cqt_fmin_hz,
                                        bins_per_octave=cfg.bins_per_octave)
    else:
        stft = librosa.stft(y=y, n_fft=cfg.n_fft, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
        # Solo se guardan los bins que puede leer la recompensa (hasta f_lím, más
        # uno para interpolar en el borde): con n_fft = 8192 son ≈ 450 de 4097
        # bins, ≈ 9 veces menos memoria en pistas largas.
        n_keep = min(freqs.size, max(3, int(np.searchsorted(freqs, stft_max_frequency(cfg), side="right")) + 1))
        mag = np.abs(stft[:n_keep])
        freqs = freqs[:n_keep]
        del stft
    mag = mag.astype(np.float32, copy=False)
    times = librosa.times_like(X=mag, sr=sr, hop_length=hop_length)
    logger.info("Espectro %s: %d bins (%.1f–%.1f Hz) × %d frames (hop=%d muestras = %.1f ms)",
                kind.upper(), mag.shape[0], float(freqs[0]), float(freqs[-1]), mag.shape[1],
                hop_length, 1000.0 * hop_length / sr)
    return Spectrum(mag=mag, freqs_hz=np.asarray(freqs, dtype=float), times_s=np.asarray(times, dtype=float),
                    hop_length=int(hop_length), sr=int(sr), kind=kind)


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
    """Construye el template armónico de ``f0_hz`` (ver ecuaciones del encabezado).

    Parameters
    ----------
    f0_hz : float
        Fundamental candidata f en Hz (> 0).
    n_harmonics : int
        Número N de armónicos, incluida la fundamental (≥ 1).

    Returns
    -------
    HarmonicTemplate
        Posiciones h·f (dientes del peine), (h−½)·f (huecos del peine) y
        pesos w_h = (1/h) / Σ_k (1/k).

    Raises
    ------
    ValueError
        Si ``f0_hz`` ≤ 0 o ``n_harmonics`` < 1.

    Notes
    -----
    Los huecos (h−½)·f son los puntos medios entre armónicos consecutivos
    (h = 1 da ½·f, la "suboctava"). En un sonido armónico de fundamental f
    allí no hay energía; si la hay, f probablemente no es la fundamental real
    (ver "¿Por qué β castiga los errores de octava?" en el encabezado).

    Examples
    --------
    >>> t = harmonic_template(55.0, 3)
    >>> t.harmonic_hz.tolist(), t.interharmonic_hz.tolist()
    ([55.0, 110.0, 165.0], [27.5, 82.5, 137.5])
    >>> np.round(t.weights, 3).tolist()
    [0.545, 0.273, 0.182]
    """
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        raise ValueError(f"La fundamental debe ser > 0 Hz (recibido {f0_hz})")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics debe ser ≥ 1 (recibido {n_harmonics})")
    h = np.arange(1, n_harmonics + 1, dtype=float)
    raw = 1.0 / h  # armónicos graves pesan más: son los más fuertes y fiables en el bajo
    return HarmonicTemplate(
        f0_hz=float(f0_hz),
        harmonic_hz=h * f0_hz,
        interharmonic_hz=(h - 0.5) * f0_hz,
        weights=raw / raw.sum(),
    )


def _log_interp_plan(freqs_hz: np.ndarray, query_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Índices y pesos de interpolación lineal sobre el eje log2(frecuencia).

    Para cada frecuencia consultada x se buscan los bins vecinos i0 < x ≤ i1
    en log2(f) y el valor interpolado es ``w0·X[i0] + w1·X[i1]``. Las
    consultas fuera del rango del espectro reciben w0 = w1 = 0 (valor 0).
    Se ignoran los bins con frecuencia ≤ 0 (el bin DC de la STFT).

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frecuencia de cada bin (Hz), creciente, forma ``(n_bins,)``.
    query_hz : np.ndarray
        Frecuencias a leer (Hz), cualquier forma.

    Returns
    -------
    tuple of np.ndarray
        ``(i0, i1, w0, w1)`` con la forma de ``query_hz``.

    Raises
    ------
    ValueError
        Si el espectro tiene menos de 2 bins con frecuencia > 0 Hz.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    valid = np.flatnonzero(freqs_hz > 0)
    if valid.size < 2:
        raise ValueError("El espectro necesita al menos 2 bins con frecuencia > 0 Hz")
    log_f = np.log2(freqs_hz[valid])
    query = np.asarray(query_hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.where(query > 0, np.log2(np.where(query > 0, query, 1.0)), -np.inf)
    in_range = (log_q >= log_f[0]) & (log_q <= log_f[-1])
    pos = np.clip(np.searchsorted(log_f, log_q, side="right") - 1, 0, log_f.size - 2)
    t = (log_q - log_f[pos]) / (log_f[pos + 1] - log_f[pos])
    t = np.where(in_range, np.clip(t, 0.0, 1.0), 0.0)
    w0 = np.where(in_range, 1.0 - t, 0.0)
    w1 = t
    return valid[pos], valid[pos + 1], w0, w1


@dataclass
class _WindowPlan:
    """Plan para leer max X̃ dentro de la ventana de tolerancia de cada consulta.

    Para cada frecuencia consultada x la ventana es
    [x·2^(−tol/12), x·2^(+tol/12)]. Se guardan la interpolación en sus dos
    extremos y los bins que caen ESTRICTAMENTE dentro (concatenados para
    todas las consultas, para usar ``np.maximum.reduceat``).

    Attributes
    ----------
    lo, hi : tuple of np.ndarray
        Plan de :func:`_log_interp_plan` en el extremo inferior y superior.
    inner_bins : np.ndarray
        Índices de los bins interiores de todas las consultas, concatenados.
    inner_starts : np.ndarray
        Posición en ``inner_bins`` donde empieza el grupo de cada consulta con
        bins interiores.
    inner_queries : np.ndarray
        Índice de la consulta a la que pertenece cada grupo.
    """

    lo: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    hi: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    inner_bins: np.ndarray
    inner_starts: np.ndarray
    inner_queries: np.ndarray

    @property
    def n_columns(self) -> int:
        """Columnas que se leen por frame (2 extremos × 2 vecinos + bins interiores)."""
        return 4 * int(self.lo[0].size) + int(self.inner_bins.size)


def _window_plan(freqs_hz: np.ndarray, query_hz: np.ndarray, tolerance_semitones: float) -> _WindowPlan:
    """Prepara la lectura de max X̃ en [x·2^(−tol/12), x·2^(+tol/12)] para cada consulta x.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frecuencia de cada bin (Hz), creciente, forma ``(n_bins,)``.
    query_hz : np.ndarray
        Frecuencias centrales x (Hz), forma ``(Q,)``.
    tolerance_semitones : float
        Semiancho de la ventana en semitonos (≤ 0 → sin ventana: se lee x).

    Returns
    -------
    _WindowPlan
        Plan de lectura (independiente de los frames).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    query = np.asarray(query_hz, dtype=float).ravel()
    ratio = 2.0 ** (max(float(tolerance_semitones), 0.0) / 12.0)
    lo_hz, hi_hz = query / ratio, query * ratio
    valid = np.flatnonzero(freqs_hz > 0)
    f_valid = freqs_hz[valid]
    first = np.searchsorted(f_valid, lo_hz, side="right")   # primer bin con f > extremo inferior
    stop = np.searchsorted(f_valid, hi_hz, side="left")     # primer bin con f ≥ extremo superior
    counts = np.maximum(stop - first, 0)
    with_inner = np.flatnonzero(counts > 0)
    if with_inner.size:
        inner_bins = np.concatenate([valid[first[q]:stop[q]] for q in with_inner])
        inner_starts = np.concatenate(([0], np.cumsum(counts[with_inner])[:-1]))
    else:
        inner_bins = np.zeros(0, dtype=int)
        inner_starts = np.zeros(0, dtype=int)
    return _WindowPlan(lo=_log_interp_plan(freqs_hz, lo_hz), hi=_log_interp_plan(freqs_hz, hi_hz),
                       inner_bins=inner_bins, inner_starts=inner_starts.astype(int), inner_queries=with_inner)


def _read_window_max(block: np.ndarray, plan: _WindowPlan) -> np.ndarray:
    """max X̃ en la ventana de cada consulta, para un bloque de frames.

    Parameters
    ----------
    block : np.ndarray
        Espectro normalizado X̃, forma ``(n_frames_bloque, n_bins)``.
    plan : _WindowPlan
        Plan de :func:`_window_plan`.

    Returns
    -------
    np.ndarray
        Forma ``(n_frames_bloque, Q)``.
    """
    i0, i1, w0, w1 = plan.lo
    best = block[:, i0] * w0 + block[:, i1] * w1                 # X̃ interpolado en x·2^(−tol/12)
    i0, i1, w0, w1 = plan.hi
    np.maximum(best, block[:, i0] * w0 + block[:, i1] * w1, out=best)   # ... y en x·2^(+tol/12)
    if plan.inner_bins.size:
        # Máximo de los bins interiores de cada ventana (grupos contiguos de columnas).
        inner = np.maximum.reduceat(block[:, plan.inner_bins], plan.inner_starts, axis=1)
        best[:, plan.inner_queries] = np.maximum(best[:, plan.inner_queries], inner)
    return best


def salience_components(
    frames_mag: np.ndarray,
    freqs_hz: np.ndarray,
    arm_freqs_hz: np.ndarray,
    cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Energía armónica S⁺ y energía inter-armónica S⁻ de cada frame y candidato.

    Es el núcleo de :func:`salience`, expuesto por separado para poder
    graficar/explicar ambos términos.

    Parameters
    ----------
    frames_mag : np.ndarray
        Magnitudes sin normalizar, forma ``(n_bins, n_frames)`` (o ``(n_bins,)``
        para un único frame).
    freqs_hz : np.ndarray
        Frecuencia de cada bin (Hz), creciente, forma ``(n_bins,)``.
    arm_freqs_hz : np.ndarray
        Fundamentales candidatas (Hz), forma ``(n_arms,)``.
    cfg : EnvConfig
        Usa ``n_harmonics`` y ``tolerance_semitones`` (semitonos).

    Returns
    -------
    s_plus, s_minus : np.ndarray
        Ambas de forma ``(n_frames, n_arms)`` con valores en [0, 1].

    Raises
    ------
    ValueError
        Si las formas no son coherentes o alguna frecuencia candidata es ≤ 0.

    Notes
    -----
    Implementación vectorizada en tres pasos:

    1. **Normalización por frame**: X̃_j = |X_j| / max|X_j|. Así la
       saliencia no depende del volumen de la nota (un frame suave y uno
       fuerte con el mismo timbre dan la misma saliencia). Los frames con
       máximo ≤ :data:`SILENCE_FLOOR` se dejan en 0 (saliencia 0).
    2. **Plan de lectura** (independiente de los frames): para cada
       (brazo, armónico o hueco) con frecuencia central x se toma la
       ventana [x·2^(−tol/12), x·2^(+tol/12)]. Leer "el máximo dentro de
       ±tol" absorbe la desafinación y la inarmonicidad de las cuerdas
       reales (los armónicos de una cuerda gruesa salen ligeramente agudos).
    3. **Gather**: una sola indexación avanzada lee todas las ventanas de
       todos los frames a la vez; luego máximo dentro de cada ventana y suma
       ponderada sobre h con los pesos w_h.

    **Máximo EXACTO de la ventana.** X̃ entre bins es la interpolación
    lineal sobre log2(f), una función lineal a trozos; su máximo en un
    intervalo se alcanza en uno de los dos extremos o en un bin interior. Por
    eso basta leer X̃ interpolado en los dos extremos y el máximo de los bins
    que caen dentro: el resultado no depende de cuántos puntos se muestreen
    (una rejilla fija de puntos, más separados que el ancho de un pico en los
    armónicos altos, podía caer a ambos lados del pico y leer su falda) y con
    tolerancia 0 se reduce a la interpolación en x.
    """
    mag = np.asarray(frames_mag, dtype=np.float64)
    if mag.ndim == 1:
        mag = mag[:, None]
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if mag.ndim != 2 or mag.shape[0] != freqs_hz.shape[0]:
        raise ValueError(f"frames_mag debe tener forma (n_bins={freqs_hz.shape[0]}, n_frames); "
                         f"recibida {mag.shape}")
    arm_freqs = np.atleast_1d(np.asarray(arm_freqs_hz, dtype=float))
    n_frames, n_arms = mag.shape[1], arm_freqs.shape[0]
    if n_frames == 0 or n_arms == 0:
        empty = np.zeros((n_frames, n_arms))
        return empty, empty.copy()

    # 1) Normalización por el máximo de cada frame (frames silenciosos → 0).
    peak = mag.max(axis=0)
    silent = peak <= SILENCE_FLOOR
    norm = mag / np.where(silent, 1.0, peak)
    norm[:, silent] = 0.0

    # 2) Plan de lectura: una ventana por (brazo, {armónico, hueco}, h).
    templates = [harmonic_template(float(f), cfg.n_harmonics) for f in arm_freqs]
    weights = templates[0].weights  # w_h depende solo de h y N: igual para todos los brazos
    centers = np.stack([np.stack([t.harmonic_hz, t.interharmonic_hz]) for t in templates])  # (A, 2, H) en Hz
    plan = _window_plan(freqs_hz, centers.ravel(), cfg.tolerance_semitones)
    shape = (n_arms, 2, cfg.n_harmonics)

    # 3) Gather por bloques de frames (acota la memoria en pistas largas).
    frames_first = np.ascontiguousarray(norm.T)  # (n_frames, n_bins)
    energy = np.empty((n_frames, n_arms, 2))
    chunk = max(1, _MAX_GATHER_ELEMENTS // max(plan.n_columns, 1))
    for start in range(0, n_frames, chunk):
        block = frames_first[start:start + chunk]
        best = _read_window_max(block, plan).reshape((block.shape[0],) + shape)  # max X̃ en ±tolerancia
        energy[start:start + chunk] = best @ weights                              # Σ_h w_h · max X̃(...)
    s_plus = np.clip(energy[..., 0], 0.0, 1.0)
    s_minus = np.clip(energy[..., 1], 0.0, 1.0)
    return s_plus, s_minus


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

    Notes
    -----
    S = max(S⁺ − β·S⁻, 0): energía en los armónicos h·f menos β veces la
    energía en los huecos (h−½)·f (ver :func:`salience_components`). El
    recorte en 0 mantiene la recompensa en [0, 1] aunque un candidato muy
    malo tenga más energía en los huecos que en los dientes de su peine.

    Examples
    --------
    Espectro "de juguete" con picos en 55, 110 y 165 Hz (nota A1):

    >>> freqs = np.arange(1.0, 1001.0)                # 1 Hz por bin
    >>> mag = np.zeros((freqs.size, 1))
    >>> mag[[54, 109, 164], 0] = [1.0, 0.5, 0.25]     # bins de 55, 110, 165 Hz
    >>> cfg = EnvConfig(n_harmonics=3, beta=0.5, tolerance_semitones=0.0)
    >>> s = salience(mag, freqs, np.array([55.0, 110.0]), cfg)
    >>> s.shape, round(float(s[0, 0]), 3), round(float(s[0, 1]), 3)
    ((1, 2), 0.727, 0.0)
    """
    s_plus, s_minus = salience_components(frames_mag, freqs_hz, arm_freqs_hz, cfg)
    return np.maximum(s_plus - cfg.beta * s_minus, 0.0)


def playability_penalty(frets: np.ndarray, prev_fret: int | None, lam: float) -> np.ndarray:
    """λ·|traste − traste_previo|/12 por brazo (0 si ``prev_fret`` es None).

    Parameters
    ----------
    frets : np.ndarray
        Traste de cada brazo, forma ``(K,)``.
    prev_fret : int | None
        Posición previa de la mano (traste); None = sin información.
    lam : float
        Peso λ (adimensional; un desplazamiento de 12 trastes cuesta λ).

    Returns
    -------
    np.ndarray
        Penalización de cada brazo, forma ``(K,)``, float.

    Notes
    -----
    Modela la tocabilidad: saltar muchos trastes entre notas consecutivas es
    incómodo. Dividir por 12 expresa el salto en octavas del mástil, de modo
    que λ está en la misma escala que la saliencia (∈ [0, 1]).

    Examples
    --------
    >>> np.round(playability_penalty(np.array([0, 5, 7]), 5, 0.12), 3).tolist()
    [0.05, 0.0, 0.02]
    >>> playability_penalty(np.array([0, 5]), None, 0.12).tolist()
    [0.0, 0.0]
    """
    frets = np.asarray(frets, dtype=float)
    if prev_fret is None:
        return np.zeros(frets.shape, dtype=float)
    return lam * np.abs(frets - float(prev_fret)) / 12.0


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
        Las posiciones con el mismo pitch (A-0 y E-5) tienen columnas
        idénticas (ver :func:`build_segment_data`).
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


def _segment_frame_indices(times_s: np.ndarray, segment: Segment, attack_skip_s: float) -> np.ndarray:
    """Frames del espectro que representan la nota (siempre al menos uno).

    1. Frames con tiempo en [inicio + attack_skip, fin): la nota ya estable,
       sin el ataque percusivo de la púa/dedo.
    2. Si no hay ninguno (nota muy corta): frames en [inicio, fin).
    3. Si tampoco: el frame más cercano al centro del segmento.
    """
    times_s = np.asarray(times_s, dtype=float)
    idx = np.flatnonzero((times_s >= segment.start_s + attack_skip_s) & (times_s < segment.end_s))
    if idx.size == 0:
        idx = np.flatnonzero((times_s >= segment.start_s) & (times_s < segment.end_s))
    if idx.size == 0:
        center = 0.5 * (segment.start_s + segment.end_s)
        idx = np.array([int(np.argmin(np.abs(times_s - center)))])
    return idx.astype(int)


def _format_arm_list(arms: list[Arm], max_shown: int = 8) -> str:
    """``"A-0, E-5, …"``: etiquetas de los primeros ``max_shown`` brazos."""
    labels = [a.label for a in arms[:max_shown]]
    if len(arms) > max_shown:
        labels.append("…")
    return ", ".join(labels)


def build_segment_data(spectrum: Spectrum, segment: Segment, position: int, cfg: EnvConfig) -> SegmentBanditData:
    """Precalcula brazos candidatos y matriz de saliencia de un segmento conservado.

    Parameters
    ----------
    spectrum : Spectrum
        Espectro de toda la pista.
    segment : Segment
        Segmento conservado (usa ``start_s``, ``end_s`` y ``f0_hz``).
    position : int
        Posición del segmento entre los conservados.
    cfg : EnvConfig
        Usa ``k_semitones``, ``n_frets``, ``attack_skip_s`` y los parámetros
        de la saliencia (``n_harmonics``, ``beta``, ``tolerance_semitones``).

    Returns
    -------
    SegmentBanditData
        Brazos candidatos, frames usados y matriz ``salience[frame, brazo]``.

    Notes
    -----
    Es el ÚNICO paso costoso por segmento: después, cada pull del bandit
    es una simple lectura de esta matriz. La saliencia se calcula una sola
    vez por nota MIDI candidata (con su frecuencia temperada) y se copia a
    las columnas de todas las posiciones que tocan esa nota: las posiciones
    gemelas (A-0 / E-5, ...) tienen exactamente la misma columna.
    """
    frame_idx = _segment_frame_indices(spectrum.times_s, segment, cfg.attack_skip_s)
    arms = candidate_arms(segment.f0_hz, cfg.k_semitones, cfg.n_frets)
    # La saliencia depende solo del PITCH: se calcula una vez por nota MIDI
    # (frecuencia temperada 440·2^((m−69)/12)) y se copia a todas sus posiciones.
    # Así A-0 y E-5 tienen columnas idénticas y solo la penalización los separa.
    midis = np.array([a.midi for a in arms])
    unique_midis, column = np.unique(midis, return_inverse=True)
    pitch_freqs = np.array([midi_to_hz(float(m)) for m in unique_midis])
    sal_per_pitch = salience(spectrum.mag[:, frame_idx], spectrum.freqs_hz, pitch_freqs, cfg)
    sal = sal_per_pitch[:, column.ravel()]
    data = SegmentBanditData(segment=segment, position=position, arms=arms,
                             frame_indices=frame_idx, salience=sal)
    means = data.mean_salience
    best = int(np.argmax(means))
    f0_text = "desconocida" if segment.f0_hz is None else f"{segment.f0_hz:.1f} Hz"
    name = segment_log_name(position, segment.index)
    logger.info("%s: f0=%s, %d brazos candidatos (%s), mejor saliencia media: %s (%.2f)",
                name, f0_text, len(arms), _format_arm_list(arms), arms[best].label, float(means[best]))
    logger.debug("%s: %d frames (%.3f–%.3f s) usados para la recompensa",
                 name, frame_idx.size, float(spectrum.times_s[frame_idx[0]]),
                 float(spectrum.times_s[frame_idx[-1]]))
    return data


def build_all_segment_data(spectrum: Spectrum, segments: list[Segment], cfg: EnvConfig) -> list[SegmentBanditData]:
    """:func:`build_segment_data` para cada segmento con ``kept=True``, en orden.

    Parameters
    ----------
    spectrum : Spectrum
        Espectro de toda la pista.
    segments : list[Segment]
        Todos los segmentos (los descartados se omiten).
    cfg : EnvConfig
        Configuración del entorno.

    Returns
    -------
    list[SegmentBanditData]
        Uno por segmento conservado, con ``position`` = 0, 1, 2, ...
    """
    kept = [s for s in segments if s.kept]
    data = [build_segment_data(spectrum, seg, pos, cfg) for pos, seg in enumerate(kept)]
    logger.info("Datos bandit precalculados para %d segmentos conservados (de %d detectados)",
                len(kept), len(segments))
    return data


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
    gaps : np.ndarray
        Brecha de cada brazo Δ_a = μ* − μ_a ≥ 0, forma ``(K,)`` (regret por pull).

    Notes
    -----
    La distribución de recompensas del brazo a es la distribución empírica
    de {S_j(f_a) − pen_a} sobre los frames j del segmento (cada uno con
    probabilidad 1/n_frames), más ruido N(0, σ²) opcional. Por eso
    μ_a = media_j S_j(f_a) − pen_a es EXACTO (ver encabezado del módulo).

    Examples
    --------
    >>> seg = Segment(index=0, start_s=0.0, end_s=1.0, start_sample=0, end_sample=22050, rms_db=0.0)
    >>> data = SegmentBanditData(seg, 0, [Arm("E", 5), Arm("A", 0)], np.arange(2),
    ...                          np.array([[0.8, 0.8], [0.6, 0.6]]))
    >>> env = BanditEnvironment(data, prev_fret=5, lam=0.12, rng=np.random.default_rng(0))
    >>> env.true_means.round(3).tolist(), env.optimal_arms.tolist()
    ([0.7, 0.65], [0])
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
        if data.salience.ndim != 2 or data.salience.shape[0] < 1 or data.salience.shape[1] != data.n_arms:
            raise ValueError(f"La matriz de saliencia debe tener forma (n_frames ≥ 1, K={data.n_arms}); "
                             f"recibida {data.salience.shape}")
        if noise_std < 0:
            raise ValueError(f"noise_std debe ser ≥ 0 (recibido {noise_std})")
        self.data = data
        self.prev_fret = prev_fret
        self.lam = float(lam)
        self.rng = rng
        self.noise_std = float(noise_std)
        self.open_string_free = bool(open_string_free)

        frets = data.frets
        penalties = playability_penalty(frets, prev_fret, self.lam)
        if self.open_string_free:
            # Extensión opcional: tocar al aire no obliga a mover la mano.
            penalties[frets == 0] = 0.0
        self.penalties: np.ndarray = penalties

        # Valor real de cada brazo (lo conoce el entorno, NUNCA el agente):
        #   μ_a = (1/n_frames)·Σ_j S_j(f_a) − pen_a
        self.true_means: np.ndarray = data.mean_salience - penalties
        self.best_mean: float = float(self.true_means.max())
        self.optimal_arms: np.ndarray = np.flatnonzero(self.true_means >= self.best_mean - OPTIMAL_TOL)
        # Brecha Δ_a = μ* − μ_a: el pseudo-regret que cuesta jalar a (0 para los óptimos).
        self.gaps: np.ndarray = np.maximum(self.best_mean - self.true_means, 0.0)
        self.gaps[self.optimal_arms] = 0.0
        self._optimal_mask = np.zeros(data.n_arms, dtype=bool)
        self._optimal_mask[self.optimal_arms] = True

        # Atajos para que pull() sea O(1) sin indirecciones.
        self._salience = data.salience
        self._n_frames = int(data.salience.shape[0])
        self._n_arms = int(data.n_arms)
        logger.debug("Entorno del segmento %d: K=%d, %d frames, traste previo=%s, μ*=%.3f, óptimos=%s",
                     data.position, self._n_arms, self._n_frames, prev_fret, self.best_mean,
                     [data.arms[i].label for i in self.optimal_arms])

    @classmethod
    def from_config(
        cls, data: SegmentBanditData, prev_fret: int | None, cfg: EnvConfig, rng: np.random.Generator
    ) -> "BanditEnvironment":
        """Atajo que toma ``lam``, ``noise_std`` y ``open_string_free`` de ``cfg``.

        Parameters
        ----------
        data : SegmentBanditData
            Datos precalculados del segmento.
        prev_fret : int | None
            Traste previo (posición de la mano); None = sin penalización.
        cfg : EnvConfig
            Configuración del entorno (λ, ruido σ, cuerdas al aire gratis).
        rng : np.random.Generator
            Generador para muestrear frames (y ruido).

        Returns
        -------
        BanditEnvironment
            Entorno equivalente a ``BanditEnvironment(data, prev_fret, cfg.lam, rng, ...)``.
        """
        return cls(data, prev_fret, cfg.lam, rng, noise_std=cfg.noise_std, open_string_free=cfg.open_string_free)

    @property
    def n_arms(self) -> int:
        """Número de brazos K."""
        return self._n_arms

    @property
    def arms(self) -> list[Arm]:
        """Brazos candidatos del segmento (atajo a ``data.arms``)."""
        return self.data.arms

    def _check_arm(self, arm_index: int) -> int:
        """Valida el índice de brazo (0..K−1) y lo devuelve como ``int``."""
        a = int(arm_index)
        if not 0 <= a < self._n_arms:
            raise IndexError(f"Brazo {arm_index} fuera de rango (K={self._n_arms})")
        return a

    def _sample(self, a: int) -> tuple[int, float, float]:
        """Muestrea (frame j, S_j(f_a), ruido) consumiendo el rng siempre igual.

        La primera llamada al generador elige el frame, sea cual sea el brazo:
        así dos algoritmos con la misma semilla ven la misma secuencia de
        frames (números aleatorios comunes). El ruido, si existe, es la
        segunda llamada.
        """
        j = int(self.rng.integers(self._n_frames))
        noise = float(self.rng.normal(0.0, self.noise_std)) if self.noise_std > 0 else 0.0
        return j, float(self._salience[j, a]), noise

    def pull(self, arm_index: int) -> float:
        """Jala el brazo y devuelve solo la recompensa (camino rápido para experimentos).

        Parameters
        ----------
        arm_index : int
            Índice del brazo (0..K−1).

        Returns
        -------
        float
            r = S_j(f_a) − pen_a (+ ruido), con j uniforme entre los frames.

        Raises
        ------
        IndexError
            Si el índice está fuera de rango.
        """
        a = self._check_arm(arm_index)
        _, s, noise = self._sample(a)
        return s - float(self.penalties[a]) + noise

    def pull_detailed(self, arm_index: int) -> PullResult:
        """Jala el brazo y devuelve el detalle completo (frame, saliencia, penalización).

        Consume el generador exactamente igual que :meth:`pull`, así que
        ambos producen la misma secuencia de recompensas con la misma semilla.

        Parameters
        ----------
        arm_index : int
            Índice del brazo (0..K−1).

        Returns
        -------
        PullResult
            Recompensa y sus componentes.
        """
        a = self._check_arm(arm_index)
        j, s, noise = self._sample(a)
        pen = float(self.penalties[a])
        return PullResult(arm_index=a, reward=s - pen + noise, frame_index=j,
                          salience=s, penalty=pen, noise=noise)

    def regret_of(self, arm_index: int) -> float:
        """Pseudo-regret instantáneo μ* − μ_a (≥ 0).

        Parameters
        ----------
        arm_index : int
            Índice del brazo.

        Returns
        -------
        float
            0 para los brazos óptimos; Δ_a > 0 para los demás.
        """
        return float(self.gaps[self._check_arm(arm_index)])

    def is_optimal(self, arm_index: int) -> bool:
        """Indica si un brazo es óptimo (μ_a = μ* salvo :data:`OPTIMAL_TOL`).

        Parameters
        ----------
        arm_index : int
            Índice del brazo (0..K−1).

        Returns
        -------
        bool
            True si ``arm_index`` está entre los brazos óptimos.

        Raises
        ------
        IndexError
            Si el índice está fuera de rango.
        """
        return bool(self._optimal_mask[self._check_arm(arm_index)])


def next_hand_fret(arm: Arm, prev_fret: int | None, cfg: EnvConfig) -> int | None:
    """Posición de la mano tras tocar ``arm``.

    Normalmente es ``arm.fret``; si ``cfg.open_string_free`` y el brazo es una
    cuerda al aire, la mano se queda en ``prev_fret``.

    Parameters
    ----------
    arm : Arm
        Posición recién tocada.
    prev_fret : int | None
        Posición de la mano antes de tocarla.
    cfg : EnvConfig
        Usa ``open_string_free``.

    Returns
    -------
    int | None
        Traste previo para el siguiente segmento.

    Examples
    --------
    >>> next_hand_fret(Arm("A", 0), 5, EnvConfig())
    0
    >>> next_hand_fret(Arm("A", 0), 5, EnvConfig(open_string_free=True))
    5
    >>> next_hand_fret(Arm("D", 3), 5, EnvConfig(open_string_free=True))
    3
    """
    if cfg.open_string_free and arm.fret == 0:
        return prev_fret
    return arm.fret


__all__ = [
    "Arm", "fret_frequency", "all_arms", "candidate_arms", "Spectrum", "compute_spectrum", "stft_max_frequency",
    "HarmonicTemplate", "harmonic_template", "salience", "salience_components", "playability_penalty",
    "SegmentBanditData", "build_segment_data", "build_all_segment_data", "PullResult",
    "BanditEnvironment", "next_hand_fret", "SILENCE_FLOOR", "OPTIMAL_TOL",
]
