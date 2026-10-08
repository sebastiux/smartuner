"""Etapa 3 — Preprocesamiento: filtro pasa-bajas de fase cero y normalización.

Papel en el pipeline
--------------------
Recibe la señal mono decodificada por :mod:`src.io_audio` (o el stem de bajo
de :mod:`src.separation`) y produce **dos** versiones de ella, con la misma
longitud y frecuencia de muestreo que la entrada:

* ``y_analysis``: pasa-bajas Butterworth (``sosfiltfilt``, fase cero) +
  normalización de pico. Se usa para detectar onsets (:mod:`src.segmentation`)
  y estimar pitch con pYIN (:mod:`src.pitch`), donde los armónicos agudos, el
  ruido de trastes y el "clic" de la púa estorban.
* ``y_spectral``: solo normalizada. Se usa para la recompensa del bandit
  (:mod:`src.environment`), que necesita los armónicos por encima del corte
  (N=5 armónicos de 196 Hz llegan a ~980 Hz).

¿Por qué dos señales? (decisión de diseño 1)
--------------------------------------------
Las dos etapas que consumen la señal quieren cosas opuestas:

* **pYIN y los onsets** funcionan mejor con una señal "limpia" donde domine la
  fundamental. La nota más aguda que buscamos es G2 traste 12 = 196 Hz, así que
  un corte en 400 Hz deja pasar todas las fundamentales (y el 2.º armónico de
  la mayoría) pero elimina el ruido percusivo y los armónicos altos que
  provocan errores de octava en pYIN y onsets falsos.
* **La recompensa** mide la *saliencia armónica*: suma la energía en
  f, 2f, 3f, ..., N·f y resta la energía entre armónicos ((h−½)·f). Con N = 5
  y f = 196 Hz necesita ver hasta 5·196 = 980 Hz; si usara la señal filtrada a
  400 Hz, los armónicos 3.º–5.º habrían desaparecido y la recompensa no podría
  distinguir, por ejemplo, un candidato de su octava.

Por eso se filtra **solo** la copia de análisis y la recompensa usa la señal
normalizada sin filtrar.

¿Por qué fase cero?
-------------------
Un filtro causal (``sosfilt``) retrasa la señal (retardo de grupo de unas
decenas de muestras cerca del corte) y ese retraso no es igual en todas las
frecuencias. Los onsets detectados en la señal filtrada quedarían desplazados
respecto a la señal sin filtrar que usa la recompensa. ``sosfiltfilt`` aplica
el filtro hacia adelante y luego hacia atrás: los desfases se cancelan
(respuesta de fase φ(ω) = 0) y la magnitud queda al cuadrado, |H(ω)|². Ninguna
muestra se desplaza en el tiempo, así que ambas señales siguen alineadas.

Ejemplo
-------
>>> import numpy as np
>>> from src.config import PreprocessConfig
>>> sr = 22050
>>> t = np.arange(sr) / sr
>>> y = 0.3 * np.sin(2 * np.pi * 55.0 * t)
>>> res = preprocess(y, sr, PreprocessConfig())
>>> len(res.y_analysis) == len(res.y_spectral) == len(y)
True
>>> round(float(np.max(np.abs(res.y_spectral))), 2)
0.99
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.config import PreprocessConfig

logger = logging.getLogger(__name__)

#: Por debajo de este pico (≈ −200 dBFS) la señal se considera nula: no se
#: normaliza, porque solo se amplificaría ruido numérico.
_SILENCE_PEAK: float = 1e-10


@dataclass
class PreprocessResult:
    """Señales preprocesadas (misma longitud y frecuencia de muestreo que la entrada).

    Attributes
    ----------
    y_analysis : np.ndarray
        Señal filtrada pasa-bajas (fase cero) y normalizada. Para onsets y pYIN.
    y_spectral : np.ndarray
        Señal solo normalizada (conserva todos los armónicos). Para la
        recompensa y el espectrograma.
    sr : int
        Frecuencia de muestreo de ambas señales en Hz.
    """

    y_analysis: np.ndarray
    y_spectral: np.ndarray
    sr: int


def _as_float(y: np.ndarray) -> np.ndarray:
    """Devuelve ``y`` como arreglo de punto flotante (float32 si era entero)."""
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)
    return y


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Filtro pasa-bajas Butterworth de fase cero. Si ``cutoff_hz >= sr/2`` devuelve una copia.

    Parameters
    ----------
    y : np.ndarray
        Señal de entrada, forma ``(n_muestras,)``. Se filtra a lo largo del
        último eje.
    sr : int
        Frecuencia de muestreo en Hz.
    cutoff_hz : float
        Frecuencia de corte en Hz (punto de −3 dB del Butterworth de una
        pasada; tras ida y vuelta ese punto queda en −6 dB).
    order : int, optional
        Orden del Butterworth (4 por defecto). Al aplicarse dos veces, la
        caída efectiva es de 2·order·6 dB/octava (48 dB/octava con orden 4).

    Returns
    -------
    np.ndarray
        Señal filtrada, misma forma y tipo de punto flotante que ``y``.
        Si ``cutoff_hz >= sr/2`` (no hay nada que quitar por encima de
        Nyquist) se devuelve una copia sin filtrar.

    Raises
    ------
    ValueError
        Si ``cutoff_hz <= 0`` o ``order < 1``.

    Notes
    -----
    Respuesta en magnitud del Butterworth de orden n:

        |H(f)| = 1 / √(1 + (f / f_c)^(2n))

    Es máximamente plana en la banda de paso (no hay rizado que altere la
    amplitud de las fundamentales del bajo). Con ``sosfiltfilt`` la magnitud
    efectiva es |H(f)|² y la fase es exactamente cero.

    Se usa la forma SOS (secciones de segundo orden en cascada) en lugar de
    los coeficientes (b, a) de un único polinomio porque esta última es
    numéricamente inestable cuando el corte es bajo respecto a ``sr``
    (400 Hz frente a 22 050 Hz): los polos quedan muy cerca del círculo unidad.

    Examples
    --------
    >>> import numpy as np
    >>> sr = 22050
    >>> t = np.arange(sr) / sr
    >>> agudo = np.sin(2 * np.pi * 3000.0 * t)
    >>> filtrado = lowpass(agudo, sr, 400.0)
    >>> bool(np.max(np.abs(filtrado[1000:-1000])) < 1e-3)
    True
    >>> bool(np.array_equal(lowpass(agudo, sr, 20000.0), agudo))
    True
    """
    y = _as_float(y)
    if cutoff_hz <= 0:
        raise ValueError(f"La frecuencia de corte debe ser positiva (se recibió {cutoff_hz} Hz).")
    if order < 1:
        raise ValueError(f"El orden del filtro debe ser ≥ 1 (se recibió {order}).")
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist or y.shape[-1] == 0:
        # Por encima de Nyquist no hay contenido que eliminar: el filtro sería la identidad.
        return y.copy()

    sos = butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    # sosfiltfilt rellena los extremos (reflexión impar) para que el transitorio
    # inicial del filtro no ensucie el comienzo y el final de la pista. El
    # relleno por defecto es 3·(2·n_secciones + 1) muestras; se recorta para
    # señales más cortas que eso (scipy fallaría con ValueError).
    padlen = min(3 * (2 * sos.shape[0] + 1), y.shape[-1] - 1)
    filtered = sosfiltfilt(sos, y, axis=-1, padlen=padlen)
    logger.debug("Pasa-bajas Butterworth orden %d a %.1f Hz aplicado (fase cero)", order, cutoff_hz)
    return filtered.astype(y.dtype, copy=False)


def normalize_peak(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Escala ``y`` para que max|y| = ``peak`` (una señal nula se devuelve sin cambios).

    Parameters
    ----------
    y : np.ndarray
        Señal de entrada (no se modifica).
    peak : float, optional
        Amplitud de pico deseada (lineal, 0.99 ≈ −0.09 dBFS por defecto).
        Se deja un pequeño margen bajo 1.0 para que escribir la señal a WAV
        o MP3 no recorte.

    Returns
    -------
    np.ndarray
        Copia escalada de ``y`` (mismo tipo de punto flotante). Si la señal es
        nula (pico < 1e-10) se devuelve una copia sin cambios.

    Notes
    -----
    La normalización es una ganancia g = peak / max|y| que no altera la forma
    de onda ni el espectro relativo; sirve para que los umbrales absolutos de
    las etapas siguientes (onsets, RMS en dB, recompensa) se comporten igual
    sin importar si el MP3 se grabó fuerte o suave.

    Examples
    --------
    >>> import numpy as np
    >>> normalize_peak(np.array([0.1, -0.5, 0.25]))
    array([ 0.198, -0.99 ,  0.495])
    >>> normalize_peak(np.zeros(3))
    array([0., 0., 0.])
    """
    y = _as_float(y)
    current = float(np.max(np.abs(y))) if y.size else 0.0
    if not np.isfinite(current) or current < _SILENCE_PEAK:
        if y.size:
            logger.debug("Normalización omitida: la señal es nula o no finita (pico=%g)", current)
        return y.copy()
    gain = peak / current
    logger.debug("Normalización de pico: ganancia ×%.3f (%.1f dB)", gain, 20.0 * np.log10(gain))
    return (y * gain).astype(y.dtype, copy=False)


def preprocess(y: np.ndarray, sr: int, cfg: PreprocessConfig) -> PreprocessResult:
    """Aplica el preprocesamiento descrito en el encabezado del módulo.

    Parameters
    ----------
    y : np.ndarray
        Señal mono, forma ``(n_muestras,)``, valores en [-1, 1].
    sr : int
        Frecuencia de muestreo en Hz.
    cfg : PreprocessConfig
        Corte del pasa-bajas (Hz), orden del filtro y si se normaliza.

    Returns
    -------
    PreprocessResult
        ``y_analysis`` (filtrada y, si ``cfg.normalize``, normalizada),
        ``y_spectral`` (solo normalizada si ``cfg.normalize``; si no, una copia)
        y ``sr``. Ambas señales tienen la longitud de ``y``.

    Notes
    -----
    Cada señal se normaliza por separado: el pasa-bajas reduce el pico de la
    señal de análisis (le quita energía aguda), y normalizarla después
    garantiza que los umbrales de onset trabajen siempre en la misma escala.

    Examples
    --------
    >>> import numpy as np
    >>> from src.config import PreprocessConfig
    >>> sr = 22050
    >>> t = np.arange(sr) / sr
    >>> y = 0.5 * np.sin(2 * np.pi * 55.0 * t) + 0.5 * np.sin(2 * np.pi * 2000.0 * t)
    >>> res = preprocess(y, sr, PreprocessConfig(lowpass_hz=400.0))
    >>> espectro = np.abs(np.fft.rfft(res.y_analysis))  # 1 bin = 1 Hz (1 s de señal)
    >>> bool(espectro[2000] < 1e-3 * espectro[55])
    True
    """
    y = _as_float(y)

    # 1) Señal de ANÁLISIS (onsets + pYIN): se quita todo lo que está por encima
    #    del corte (ruido de trastes, ataque de la púa, armónicos altos que
    #    inducen errores de octava en pYIN). Fase cero para no desplazar onsets.
    y_analysis = lowpass(y, sr, cfg.lowpass_hz, order=cfg.filter_order)

    # 2) Señal ESPECTRAL (recompensa): NO se filtra, porque la saliencia armónica
    #    necesita ver N·f (hasta ~980 Hz para G2 traste 12 con N = 5).
    y_spectral = y.copy()

    if cfg.normalize:
        y_analysis = normalize_peak(y_analysis)
        y_spectral = normalize_peak(y_spectral)

    logger.info(
        "Preprocesamiento: pasa-bajas Butterworth orden %d a %.0f Hz (fase cero) para onsets/pYIN; "
        "señal espectral sin filtrar para la recompensa%s",
        cfg.filter_order,
        cfg.lowpass_hz,
        "; ambas normalizadas a pico 0.99" if cfg.normalize else "",
    )
    return PreprocessResult(y_analysis=y_analysis, y_spectral=y_spectral, sr=int(sr))
