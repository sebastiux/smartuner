"""Escuchar la tablatura: reproducción del MIDI sintetizado con cursor sincronizado.

Papel en la interfaz
--------------------
Componente reutilizable e independiente de las pestañas. La pestaña
Tablatura le entrega las notas transcritas y el audio original; él prepara el
audio en segundo plano, lo reproduce y le avisa cada ~30 ms del instante que
suena para que mueva un cursor sobre la tablatura y resalte la nota::

    pestaña ──set_source(notas, y_raw, sr)──► TabPlayback
                                                 │ play()
                                                 ▼
                         run_in_background(«Sintetizando…»)   (hilo trabajador, sin congelar la GUI)
                            src.tab.render_notes ──► caché (huella de las notas, sr, método, semilla)
                            src.tab.playback_mix(modo)
                                                 │ on_done (hilo de la GUI)
                                                 ▼
                         sounddevice.play (no bloqueante) ──► altavoces
                                                 │
    pestaña ◄── on_position(t) cada POLL_MS ms ──┘   (widget.after, hilo de la GUI)
            ◄── on_state("playing" | "paused" | "stopped" | "preparing"), on_finished()

Reloj de reproducción
---------------------
``sounddevice.play`` no informa de qué muestra está sonando, así que la
posición se mide con un reloj monotónico (:func:`time.perf_counter`)::

    t = inicio + max(0, (ahora − t₀) − latencia)

donde ``inicio`` es el instante (s) desde el que se lanzó la reproducción,
``t₀`` el momento en que se lanzó y ``latencia`` la latencia de salida que
informa PortAudio (lo que tarda una muestra entregada en llegar al altavoz;
sin restarla, el cursor iría por delante del sonido). **Pausar** = detener la
salida y recordar ``t``; **reanudar** = volver a reproducir desde la muestra
``round(t · sr)`` con ``inicio = t``. Así la posición es monotónica y
coherente tras pausar y reanudar.

Como la síntesis conserva los tiempos absolutos de los segmentos
(:func:`src.tab.render_notes`), ``t`` es directamente comparable con
``TabNote.start_s``/``end_s``: :meth:`TabPlayback.note_at` devuelve la nota
que suena.

Sin sounddevice
---------------
Si ``sounddevice`` (o PortAudio, o un dispositivo de salida) no está
disponible, :meth:`TabPlayback.play` exporta el audio preparado a un WAV (y
la tablatura a ``.mid``) en ``cache/playback/`` y los abre con el
reproductor del sistema (``os.startfile`` en Windows, ``open`` en macOS,
``xdg-open`` en Linux). En ese modo el cursor NO se sincroniza (no se sabe
qué está sonando en otro programa) y se avisa con ``on_message``. Cada
escucha desde otro instante escribe un WAV nuevo; los anteriores se borran
(y todos al cerrar), para que la carpeta no crezca sin límite.

Tarjetas que no aceptan 22 050 Hz
---------------------------------
Algunos controladores (WASAPI en Windows) solo aceptan la frecuencia nativa
del dispositivo. La primera reproducción lo descubre (se remuestrea esa vez)
y :class:`SoundDeviceOutput` lo recuerda en ``forced_sr``; desde entonces el
audio se prepara directamente a esa frecuencia EN SEGUNDO PLANO, de modo que
reanudar, saltar o cambiar de modo no congela la ventana.

Hilos
-----
Todos los métodos públicos se llaman desde el hilo de la GUI. Solo la
síntesis (y el remuestreo, si la tarjeta lo exige) corre en otro hilo (la
función que se pasa a ``run_in_background``)
y no toca el estado del objeto: devuelve su resultado y ``on_done`` lo aplica
en el hilo de la GUI.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.config import CACHE_DIR, CancelledError, ProgressCallback
from src.synth_dataset import TAIL_S, fluidsynth_available
from src.tab import (
    PLAYBACK_MODES,
    SYNTH_METHODS,
    TabNote,
    export_midi,
    playback_mix,
    render_notes,
    resolve_synth_method,
)

logger = logging.getLogger(__name__)

#: Intervalo (ms) entre avisos de posición: ~33 por segundo, suficiente para que
#: el cursor se mueva con fluidez sin cargar el bucle de eventos de tkinter.
POLL_MS: int = 30

#: Síntesis guardadas en la caché (LRU): una por algoritmo. Cada render de 6 min
#: a 22 050 Hz ocupa ~34 MB (float32).
CACHE_SIZE: int = 4

#: Carpeta del modo sin sounddevice (WAV y MIDI que se abren con el reproductor del sistema).
FALLBACK_DIR: Path = CACHE_DIR / "playback"

#: Margen (s) antes del final dentro del cual una salida que ya no está activa se
#: considera terminada con normalidad (y no interrumpida): cubre la latencia y la cola.
END_SLACK_S: float = 0.5

#: Estados de :class:`TabPlayback`.
STATES: tuple[str, ...] = ("stopped", "preparing", "playing", "paused")

#: Firma de ``run_in_background``: compatible con ``SmartunerApp.run_task(title, fn, on_done)``;
#: ``fn(progress, cancel)`` corre en otro hilo y ``on_done(resultado)`` en el de la GUI.
#: Puede devolver False si no lanzó la tarea (p. ej. ya había otra en curso).
RunInBackground = Callable[
    [str, Callable[[ProgressCallback, threading.Event], Any], Callable[[Any], None]], Any
]


# ---------------------------------------------------------------------------
# Salida de audio
# ---------------------------------------------------------------------------


class AudioOutput(Protocol):
    """Interfaz mínima de una salida de audio no bloqueante (ver :class:`SoundDeviceOutput`)."""

    error: str

    @property
    def available(self) -> bool:
        """True si se puede reproducir."""
        ...

    def play(self, y: np.ndarray, sr: int) -> float:
        """Empieza a reproducir ``y`` sin bloquear; devuelve la latencia de salida (s)."""
        ...

    def stop(self) -> None:
        """Detiene la reproducción en curso."""
        ...

    def is_active(self) -> bool | None:
        """True si lo último que se reprodujo sigue sonando; None si no se sabe."""
        ...


def load_sounddevice() -> tuple[Any | None, str]:
    """Importa ``sounddevice`` y comprueba que haya un dispositivo de salida.

    Es la misma detección que :class:`gui.widgets.AudioPlayer`, más estricta:
    ``sd.query_devices()`` funciona aunque no haya tarjeta de sonido (lista
    vacía), pero ``sd.query_devices(kind="output")`` falla si no hay salida
    por defecto, que es lo que usaría ``sd.play``.

    Returns
    -------
    tuple[Any | None, str]
        ``(módulo, "")`` si se puede reproducir; ``(None, motivo)`` si no.
    """
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - ImportError, u OSError si falta la biblioteca PortAudio
        return None, ("Reproducción no disponible: falta el paquete «sounddevice» o la biblioteca PortAudio "
                      "(pip install sounddevice; en Linux además: sudo apt install libportaudio2). "
                      f"Detalle: {exc}")
    try:
        sd.query_devices(kind="output")
    except Exception as exc:  # noqa: BLE001 - sin dispositivo de salida por defecto
        return None, f"No hay ningún dispositivo de salida de audio disponible (sounddevice: {exc})."
    return sd, ""


def _resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Remuestrea ``y`` de ``sr_from`` a ``sr_to`` Hz con un filtro polifásico (eje 0)."""
    from scipy.signal import resample_poly

    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(y, int(sr_to) // g, int(sr_from) // g, axis=0).astype(np.float32)


class SoundDeviceOutput:
    """Salida de audio con ``sounddevice`` (PortAudio): ``sd.play`` no bloqueante y ``sd.stop``.

    Parameters
    ----------
    module : Any | None, optional
        Módulo ``sounddevice`` (o un sustituto con ``play``, ``stop``,
        ``query_devices`` y ``get_stream``, p. ej. en las pruebas). Si es
        None se detecta con :func:`load_sounddevice`.

    Attributes
    ----------
    error : str
        Motivo por el que no se puede reproducir ("" si se puede).
    forced_sr : int | None
        Frecuencia (Hz) a la que hay que reproducir porque el dispositivo
        rechazó otra (p. ej. WASAPI en Windows solo acepta su frecuencia
        nativa, 44 100/48 000 Hz). None mientras no haya habido rechazos.

    Notes
    -----
    Remuestrear una canción de 6 min tarda ~0.5–1 s y :meth:`play` se llama
    en el hilo de la GUI. Por eso, tras el PRIMER rechazo se recuerda la
    frecuencia aceptada (:attr:`forced_sr`): las siguientes llamadas van
    directas a ella, sin el intento fallido; :class:`TabPlayback` prepara su
    audio ya a esa frecuencia en segundo plano, y aquí se guarda el último
    remuestreo de una misma señal (la pista completa de la pestaña Audio).
    """

    def __init__(self, module: Any | None = None) -> None:
        if module is None:
            self._sd, self.error = load_sounddevice()
        else:
            self._sd, self.error = module, ""
        self._stream: Any = None
        self.forced_sr: int | None = None
        # Último remuestreo: (señal original, sr de origen, sr de destino, señal remuestreada).
        self._resampled: tuple[np.ndarray, int, int, np.ndarray] | None = None

    @property
    def available(self) -> bool:
        """True si se puede reproducir audio."""
        return self._sd is not None

    def _device_samplerate(self) -> int | None:
        """Frecuencia de muestreo nativa (Hz) del dispositivo de salida, si se conoce."""
        try:
            info = self._sd.query_devices(kind="output")
            return int(round(float(info["default_samplerate"])))
        except Exception:  # noqa: BLE001 - información opcional
            return None

    def _current_stream(self) -> Any:
        """Stream que creó el último ``sd.play`` de CUALQUIER parte de la aplicación (None si no hay)."""
        try:
            return self._sd.get_stream()
        except Exception:  # noqa: BLE001 - get_stream falla si nunca se llamó a play
            return None

    @staticmethod
    def _latency_of(stream: Any) -> float:
        """Latencia de salida (s) del stream; 0 si no se conoce."""
        latency = getattr(stream, "latency", 0.0)
        if isinstance(latency, (tuple, list)):  # streams de entrada/salida: (entrada, salida)
            latency = latency[-1]
        try:
            value = float(latency)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value > 0 else 0.0

    def _resampled_copy(self, y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """``y`` remuestreada de ``sr_from`` a ``sr_to`` Hz y recortada a [-1, 1].

        Reutiliza el último resultado si ``y`` es el MISMO objeto (p. ej. «▶ Todo»
        de la pestaña Audio pasa siempre ``analysis.y_raw``).
        """
        cached = self._resampled
        if cached is not None and cached[0] is y and cached[1:3] == (sr_from, sr_to):
            return cached[3]
        out = np.clip(_resample(np.asarray(y, dtype=np.float32), sr_from, sr_to), -1.0, 1.0)
        self._resampled = (y, sr_from, sr_to, out)
        return out

    def play(self, y: np.ndarray, sr: int) -> float:
        """Reproduce ``y`` sin bloquear (detiene antes cualquier reproducción previa).

        Las muestras se recortan a [-1, 1] (un valor fuera de rango saturaría
        PortAudio o el mezclador del sistema con un chasquido).

        Parameters
        ----------
        y : np.ndarray
            Señal ``(n,)`` o ``(n, 2)`` en [-1, 1].
        sr : int
            Frecuencia de muestreo (Hz).

        Returns
        -------
        float
            Latencia de salida informada por PortAudio (s), para compensar el cursor.

        Raises
        ------
        RuntimeError
            Si sounddevice no está disponible o el dispositivo rechaza el audio.
        """
        if self._sd is None:
            raise RuntimeError(self.error)
        sr = int(sr)
        self._sd.stop()
        if self.forced_sr is not None and self.forced_sr != sr:
            # Ya se sabe que el dispositivo no acepta ``sr``: directo a la frecuencia que sí acepta.
            data = self._resampled_copy(y, sr, self.forced_sr)
            try:
                self._sd.play(data, self.forced_sr)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"No se pudo reproducir el audio: {exc}") from exc
        else:
            data = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)  # copia contigua: no toca ``y``
            try:
                self._sd.play(data, sr)
            except Exception as exc:  # noqa: BLE001 - se reintenta a la frecuencia del dispositivo
                # Algunos controladores (p. ej. WASAPI en Windows) solo aceptan la frecuencia
                # nativa del dispositivo (44 100 / 48 000 Hz): se remuestrea y se reintenta.
                device_sr = self._device_samplerate()
                if device_sr is None or device_sr == sr:
                    raise RuntimeError(f"No se pudo reproducir el audio: {exc}") from exc
                logger.info("La salida de audio no acepta %d Hz (%s); se remuestrea a %d Hz y se recordará "
                            "para las próximas reproducciones.", sr, exc, device_sr)
                try:
                    self._sd.play(self._resampled_copy(y, sr, device_sr), device_sr)
                except Exception as exc2:  # noqa: BLE001
                    raise RuntimeError(f"No se pudo reproducir el audio: {exc2}") from exc2
                self.forced_sr = device_sr
        self._stream = self._current_stream()
        return self._latency_of(self._stream)

    def stop(self) -> None:
        """Detiene la reproducción en curso (no falla si no había ninguna)."""
        self._stream = None
        if self._sd is None:
            return
        try:
            self._sd.stop()
        except Exception as exc:  # noqa: BLE001 - detener nunca debe romper la GUI
            logger.debug("sd.stop() falló: %s", exc)

    def is_active(self) -> bool | None:
        """Indica si lo que lanzó el último :meth:`play` sigue sonando.

        Returns
        -------
        bool | None
            False si terminó o si otra parte de la aplicación lanzó su propio
            ``sd.play`` (sounddevice solo mantiene un stream «actual»); None
            si no se puede saber.
        """
        if self._stream is None:
            return None
        current = self._current_stream()
        if current is not self._stream:
            return False
        try:
            return bool(current.active)
        except Exception:  # noqa: BLE001
            return None


def open_with_system_player(path: str | Path, platform: str | None = None) -> None:
    """Abre ``path`` con la aplicación predeterminada del sistema (sin esperar a que termine).

    Parameters
    ----------
    path : str | Path
        Archivo a abrir (WAV, MIDI...).
    platform : str | None, optional
        Valor tipo ``sys.platform`` (por defecto el actual): ``"win32"`` →
        ``os.startfile``; ``"darwin"`` → ``open``; otro → ``xdg-open``.

    Raises
    ------
    OSError
        Si no hay programa asociado o no existe ``open``/``xdg-open``.
    """
    platform = sys.platform if platform is None else platform
    target = str(path)
    if platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]  # solo existe en Windows
    elif platform == "darwin":
        subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def notes_fingerprint(notes: Sequence[TabNote]) -> str:
    """Huella (SHA-1) de lo que se oye de la tablatura: inicio, fin y pitch de cada nota.

    La cuerda y el traste no cambian el sonido sintetizado (A-0 y E-5 son la
    misma nota), así que dos algoritmos que solo difieren en la posición
    comparten la síntesis de la caché.

    Parameters
    ----------
    notes : Sequence[TabNote]
        Notas de la tablatura.

    Returns
    -------
    str
        40 caracteres hexadecimales.

    Examples
    --------
    >>> a = [TabNote(0, 0.5, 0.8, "A", 0, 33)]
    >>> notes_fingerprint(a) == notes_fingerprint([TabNote(0, 0.5, 0.8, "E", 5, 33)])
    True
    """
    data = np.array([(float(n.start_s), float(n.end_s), float(n.midi)) for n in notes], dtype=np.float64)
    return hashlib.sha1(data.tobytes()).hexdigest()


def _safe_stem(name: str) -> str:
    """Nombre de archivo seguro en cualquier sistema (sin espacios ni caracteres reservados)."""
    stem = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return stem or "tablatura"


@dataclass
class _Prepared:
    """Resultado de la tarea en segundo plano (se aplica en el hilo de la GUI).

    Attributes
    ----------
    synth : np.ndarray | None
        Síntesis a la frecuencia del análisis (para la caché).
    audio : np.ndarray | None
        Audio listo para el modo pedido, a ``sr`` Hz.
    error : str
        Motivo del fallo ("" si fue bien).
    sr : int
        Frecuencia (Hz) de ``audio``: la del análisis o la que exige la tarjeta.
    synth_out, original_out : np.ndarray | None
        Síntesis y original remuestreados a ``sr`` (si hubo que remuestrear),
        para no repetirlo al cambiar de modo.
    """

    synth: np.ndarray | None
    audio: np.ndarray | None
    error: str = ""
    sr: int = 0
    synth_out: np.ndarray | None = None
    original_out: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Reproductor de la tablatura
# ---------------------------------------------------------------------------


class TabPlayback:
    """Reproduce la tablatura sintetizada (sola, con el original o en A/B estéreo) con cursor.

    Parameters
    ----------
    widget : tkinter.Misc
        Cualquier widget vivo: se usa su ``after`` para avisar de la posición
        desde el hilo de la GUI (en las pruebas basta un objeto con ``after``
        y ``after_cancel``).
    run_in_background : RunInBackground | None, optional
        Lanza ``fn(progress, cancel)`` en otro hilo y llama a
        ``on_done(resultado)`` en el de la GUI; p. ej. ``app.run_task``. Si es
        None la síntesis corre en el hilo que llama (CLI y pruebas).
    on_position : Callable[[float], None] | None, optional
        Recibe el instante que suena (s) cada ``poll_ms`` ms mientras se
        reproduce (y al empezar, pausar y terminar).
    on_finished : Callable[[], None] | None, optional
        Se llama cuando la reproducción llega al final por sí sola.
    on_state : Callable[[str], None] | None, optional
        Recibe el nuevo estado (:data:`STATES`) cada vez que cambia (para
        habilitar botones).
    on_message : Callable[[str], None] | None, optional
        Avisos para el usuario (p. ej. a la barra de estado). Siempre se
        registran también con ``logging``.
    output : AudioOutput | None, optional
        Salida de audio; por defecto :class:`SoundDeviceOutput`.
    opener : Callable[[Path], None], optional
        Abre un archivo con el reproductor del sistema (modo sin sounddevice).
    export_dir : str | Path, optional
        Carpeta de los WAV/MIDI del modo sin sounddevice.
    clock : Callable[[], float], optional
        Reloj monotónico en segundos (inyectable en las pruebas).
    poll_ms : int, optional
        Intervalo de los avisos de posición (ms).
    cache_size : int, optional
        Síntesis guardadas en la caché.
    mode : str, optional
        Modo inicial (:data:`src.tab.PLAYBACK_MODES`).
    method : str, optional
        Sintetizador inicial (:data:`src.tab.SYNTH_METHODS`).
    seed : int, optional
        Semilla de Karplus-Strong.

    Attributes
    ----------
    notes : list[TabNote]
        Notas de la tablatura (ordenadas por inicio).
    original : np.ndarray | None
        Audio original (mono) para los modos «original», «mezcla» y «estéreo».
    sr : int
        Frecuencia de muestreo (Hz) del original y de la síntesis.
    last_export : Path | None
        Último WAV escrito en el modo sin sounddevice.

    Examples
    --------
    Uso típico desde una pestaña (``app`` es :class:`gui.app.SmartunerApp`)::

        self.playback = TabPlayback(self, app.run_task, on_position=self._on_play_position,
                                    on_state=self._on_play_state, on_message=app.status.set_message)
        self.playback.set_source(tr.notes, original=analysis.y_raw, sr=analysis.sr, name="money_ucb1")
        self.playback.play(start_s=12.0)        # sintetiza (1.ª vez, en segundo plano) y reproduce
        self.playback.mode = "estereo"          # sigue sonando desde el mismo instante, ahora en A/B
    """

    def __init__(
        self,
        widget: Any,
        run_in_background: RunInBackground | None = None,
        *,
        on_position: Callable[[float], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        on_message: Callable[[str], None] | None = None,
        output: AudioOutput | None = None,
        opener: Callable[[Path], None] = open_with_system_player,
        export_dir: str | Path = FALLBACK_DIR,
        clock: Callable[[], float] = time.perf_counter,
        poll_ms: int = POLL_MS,
        cache_size: int = CACHE_SIZE,
        mode: str = "midi",
        method: str = "auto",
        seed: int = 0,
    ) -> None:
        if mode not in PLAYBACK_MODES:
            raise ValueError(f"Modo de reproducción desconocido: «{mode}». Opciones: {', '.join(PLAYBACK_MODES)}.")
        if method not in SYNTH_METHODS:
            raise ValueError(f"Método de síntesis desconocido: «{method}». Opciones: {', '.join(SYNTH_METHODS)}.")
        self.widget = widget
        self.on_position = on_position
        self.on_finished = on_finished
        self.on_state = on_state
        self.on_message = on_message
        self._run_in_background = run_in_background
        self._output: AudioOutput = output if output is not None else SoundDeviceOutput()
        self._opener = opener
        self.export_dir = Path(export_dir)
        self._clock = clock
        self.poll_ms = int(poll_ms)
        self._cache_size = max(1, int(cache_size))
        self._mode = mode
        self._method = method
        self.seed = int(seed)

        self.notes: list[TabNote] = []
        self.original: np.ndarray | None = None
        self._original_ref: Any = None
        self.sr: int = 22050
        self.name: str = "tablatura"
        self.last_export: Path | None = None
        self._fingerprint = notes_fingerprint([])
        self._starts = np.zeros(0)
        self._ends = np.zeros(0)
        self._positions: list[int] = []

        self._cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._audio: np.ndarray | None = None   # audio listo para el modo actual
        self._audio_sr: int = self.sr           # frecuencia (Hz) de self._audio
        # Remuestreos para una tarjeta que no acepta self.sr (ver _output_sr):
        # (clave de la síntesis, sr, señal) y (original, sr, señal).
        self._synth_out: tuple[tuple[Any, ...], int, np.ndarray] | None = None
        self._original_out: tuple[np.ndarray, int, np.ndarray] | None = None
        self._exported: list[Path] = []         # WAV del modo sin sounddevice (se borran los viejos)
        self._state = "stopped"
        self._generation = 0                    # cambia con la fuente, el modo o el método
        self._in_flight = False                 # hay una síntesis en segundo plano
        self._pending_start: float | None = None  # reproducir desde aquí cuando esté listo
        self._t0 = 0.0
        self._offset = 0.0
        self._latency = 0.0
        self._paused_at = 0.0
        self._tick_job: Any = None

    # ------------------------------------------------------------ propiedades
    @property
    def available(self) -> bool:
        """True si se puede reproducir dentro de la aplicación (con cursor sincronizado)."""
        return self._output.available

    @property
    def unavailable_reason(self) -> str:
        """Motivo por el que no hay reproducción integrada ("" si la hay)."""
        return self._output.error

    @property
    def state(self) -> str:
        """Estado actual: ``"stopped"``, ``"preparing"``, ``"playing"`` o ``"paused"``."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """True mientras suena."""
        return self._state == "playing"

    @property
    def is_paused(self) -> bool:
        """True si está en pausa (``resume`` continúa desde :attr:`position_s`)."""
        return self._state == "paused"

    @property
    def mode(self) -> str:
        """Modo de reproducción (:data:`src.tab.PLAYBACK_MODES`); ver :meth:`set_mode`."""
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        self.set_mode(mode)

    @property
    def method(self) -> str:
        """Sintetizador pedido (``"auto"``, ``"fluidsynth"`` o ``"karplus-strong"``); ver :meth:`set_method`."""
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        self.set_method(method)

    @property
    def resolved_method(self) -> str:
        """Sintetizador que se usará de verdad en esta máquina (resuelve ``"auto"``)."""
        try:
            return resolve_synth_method(self._method)
        except RuntimeError:
            return "karplus-strong"

    @staticmethod
    def available_methods() -> tuple[str, ...]:
        """Métodos de síntesis utilizables aquí (``"fluidsynth"`` solo si está instalado con un soundfont).

        Returns
        -------
        tuple[str, ...]
            Subconjunto ordenado de :data:`src.tab.SYNTH_METHODS`.
        """
        usable = fluidsynth_available()
        return tuple(m for m in SYNTH_METHODS if m != "fluidsynth" or usable)

    @property
    def duration_s(self) -> float:
        """Duración (s) del audio que se reproduce (estimada si aún no se sintetizó)."""
        if self._audio is not None:
            return self._audio.shape[0] / float(self._audio_sr)
        if not self.notes:
            return 0.0
        # Misma regla que render_notes + playback_mix: fin de la última nota + cola,
        # o la duración del original si es mayor (la mezcla rellena con ceros).
        duration = float(self._ends.max()) + TAIL_S
        if self.original is not None:
            duration = max(duration, self.original.shape[0] / float(self.sr))
        return duration

    @property
    def position_s(self) -> float:
        """Instante (s, en el tiempo del audio original) que está sonando.

        Mientras suena: ``inicio + max(0, (reloj − t₀) − latencia)``, sin pasar
        de :attr:`duration_s`; en pausa, el instante en que se pausó; parado, 0.
        """
        if self._state == "playing":
            elapsed = self._clock() - self._t0 - self._latency
            return min(self._offset + max(0.0, elapsed), self.duration_s)
        if self._state == "paused":
            return self._paused_at
        if self._state == "preparing" and self._pending_start is not None:
            return self._pending_start
        return 0.0

    # ------------------------------------------------------------- fuente
    def set_source(
        self,
        notes: Sequence[TabNote],
        original: np.ndarray | None = None,
        sr: int = 22050,
        name: str | None = None,
    ) -> None:
        """Define qué se escucha: la tablatura y (opcional) el audio original.

        Si algo cambió y estaba sonando, sigue sonando desde el mismo instante
        con el audio nuevo (p. ej. al cambiar de algoritmo en la pestaña: se
        compara de oído sin volver al principio). Si las notas (pitch y
        tiempos) y el original son los mismos, no hace nada.

        Parameters
        ----------
        notes : Sequence[TabNote]
            Notas de la tablatura.
        original : np.ndarray | None, optional
            Audio original mono (``analysis.y_raw``), a ``sr`` Hz.
        sr : int, optional
            Frecuencia de muestreo (Hz) del original; la síntesis se hace a la misma.
        name : str | None, optional
            Nombre base de los archivos del modo sin sounddevice (p. ej. ``"money_ucb1"``).
        """
        if name is not None:
            self.name = _safe_stem(name)
        ordered = sorted(notes, key=lambda n: (n.start_s, n.position))
        fingerprint = notes_fingerprint(ordered)
        if fingerprint == self._fingerprint and original is self._original_ref and int(sr) == self.sr:
            self.notes = ordered  # mismas notas audibles; se guardan por si cambió la posición (cuerda/traste)
            return
        self.notes = ordered
        self._original_ref = original  # objeto del llamador: para reconocerlo en la próxima llamada
        self.original = None if original is None else np.asarray(original, dtype=np.float32)
        self.sr = int(sr)
        self._fingerprint = fingerprint
        self._starts = np.array([n.start_s for n in ordered], dtype=np.float64)
        self._ends = np.array([n.end_s for n in ordered], dtype=np.float64)
        self._positions = [int(n.position) for n in ordered]
        logger.debug("Reproductor: nueva fuente con %d notas (%s)", len(ordered),
                     "con original" if original is not None else "sin original")
        self._reload()

    def set_mode(self, mode: str) -> None:
        """Cambia el modo (MIDI / original / mezcla / estéreo) sin perder la posición.

        La síntesis está en la caché, así que solo se rehace la mezcla
        (décimas de segundo para 6 min) y, si estaba sonando, continúa en el
        mismo instante: ideal para comparar A/B en clase.

        Parameters
        ----------
        mode : str
            Uno de :data:`src.tab.PLAYBACK_MODES`.

        Raises
        ------
        ValueError
            Si el modo no existe.
        """
        if mode not in PLAYBACK_MODES:
            raise ValueError(f"Modo de reproducción desconocido: «{mode}». Opciones: {', '.join(PLAYBACK_MODES)}.")
        if mode != self._mode:
            self._mode = mode
            self._reload()

    def set_method(self, method: str) -> None:
        """Cambia el sintetizador (re-sintetiza si no está en la caché).

        Parameters
        ----------
        method : str
            ``"auto"``, ``"fluidsynth"`` o ``"karplus-strong"``.

        Raises
        ------
        ValueError
            Si el método no existe.
        """
        if method not in SYNTH_METHODS:
            raise ValueError(f"Método de síntesis desconocido: «{method}». Opciones: {', '.join(SYNTH_METHODS)}.")
        if method != self._method:
            self._method = method
            self._reload()

    def note_at(self, t: float) -> int | None:
        """Posición (``TabNote.position``) de la nota que suena en ``t`` segundos, o None en un silencio.

        Parameters
        ----------
        t : float
            Instante (s) en el tiempo del audio original (p. ej. :attr:`position_s`).

        Returns
        -------
        int | None
            Nota con ``start_s ≤ t < end_s`` (búsqueda binaria), o None.
        """
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        if 0 <= i < len(self._positions) and t < self._ends[i]:
            return self._positions[i]
        return None

    # ------------------------------------------------------------ transporte
    def play(self, start_s: float = 0.0) -> None:
        """Reproduce desde ``start_s`` segundos (también sirve para saltar a otro instante).

        Si el audio aún no está preparado, lo sintetiza (en segundo plano si
        hay ``run_in_background``; el estado pasa a ``"preparing"``) y
        empieza al terminar. Sin sounddevice, abre un WAV con el reproductor
        del sistema (ver encabezado del módulo).

        Parameters
        ----------
        start_s : float, optional
            Instante de inicio (s) en el tiempo del audio original; si está
            más allá del final se empieza desde 0.
        """
        if not self.notes:
            self._message("No hay notas que reproducir: primero analiza y transcribe un audio.")
            return
        start_s = max(0.0, float(start_s))
        self._cancel_tick()
        if self._state == "playing":
            self._output.stop()
        if self._audio is None:
            self._pending_start = start_s
            self._prepare()
            return
        self._start(start_s)

    def pause(self) -> None:
        """Pausa (recuerda la posición; :meth:`resume` continúa desde ahí)."""
        if self._state == "playing":
            position = self.position_s
            self._cancel_tick()
            self._output.stop()
            self._paused_at = position
            self._emit_position(position)
            self._set_state("paused")
        elif self._state == "preparing":
            # Pausa antes de empezar: cuando termine la síntesis no arrancará sola.
            self._paused_at = self._pending_start or 0.0
            self._pending_start = None
            self._set_state("paused")

    def resume(self) -> None:
        """Reanuda desde el instante en que se pausó (no hace nada si no está en pausa)."""
        if self._state == "paused":
            self.play(self._paused_at)

    def toggle(self) -> None:
        """Reproducir / pausar / reanudar según el estado (p. ej. para la barra espaciadora)."""
        if self._state in ("playing", "preparing"):
            self.pause()
        elif self._state == "paused":
            self.resume()
        else:
            self.play(0.0)

    def stop(self) -> None:
        """Detiene la reproducción y vuelve al principio (también cancela un inicio pendiente)."""
        self._cancel_tick()
        if self._state == "playing":
            self._output.stop()
        self._pending_start = None
        self._paused_at = 0.0
        self._set_state("stopped")

    def close(self) -> None:
        """Detiene todo, libera la caché y borra los WAV exportados (llamar al destruir la pestaña)."""
        self.stop()
        self._cache.clear()
        self._audio = None
        self._synth_out = self._original_out = None
        self._remove_old_exports(keep=None)

    # ----------------------------------------------------------- preparación
    def _render_key(self) -> tuple[Any, ...]:
        """Clave de la caché: la síntesis solo depende de las notas audibles, sr, método y semilla."""
        return (self._fingerprint, self.sr, self._method, self.seed)

    def _effective_mode(self) -> str:
        """Modo que se puede usar: sin original solo existe el MIDI."""
        return self._mode if self.original is not None else "midi"

    def _reload(self) -> None:
        """La fuente, el modo o el método cambiaron: descarta el audio preparado.

        Si estaba sonando (o preparándose), continúa desde el mismo instante con
        el audio nuevo; si estaba en pausa, sigue en pausa en el mismo instante.
        """
        resume_at = self.position_s if self._state in ("playing", "preparing") else None
        self._generation += 1
        self._audio = None
        if resume_at is None:
            return
        if not self.notes:
            self.stop()
            return
        self.play(resume_at)

    def _cache_get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        """Síntesis guardada (y la marca como la más reciente), o None."""
        synth = self._cache.get(key)
        if synth is not None:
            self._cache.move_to_end(key)
        return synth

    def _cache_put(self, key: tuple[Any, ...], synth: np.ndarray) -> None:
        """Guarda una síntesis y descarta la menos usada si se supera :data:`CACHE_SIZE`."""
        self._cache[key] = synth
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _output_sr(self) -> int:
        """Frecuencia (Hz) a la que se debe entregar el audio a la salida.

        Es la del análisis salvo que la tarjeta ya haya rechazado esa
        frecuencia (:attr:`SoundDeviceOutput.forced_sr`): entonces el audio se
        prepara directamente a la que acepta, en segundo plano, y reproducir
        (o reanudar, o saltar) es solo recortar el arreglo.
        """
        forced = getattr(self._output, "forced_sr", None)
        return int(forced) if forced else self.sr

    def _prepare(self) -> None:
        """Deja listo ``self._audio`` (síntesis de la caché o en segundo plano) y arranca si hay inicio pendiente."""
        mode = self._effective_mode()
        if mode != self._mode:
            self._message("No hay audio original: se reproduce solo el MIDI sintetizado.")
        key = self._render_key()
        synth = self._cache_get(key)
        out_sr = self._output_sr()
        if synth is not None and out_sr == self.sr:
            # Rápido (décimas de segundo para 6 min): se mezcla en el hilo de la GUI.
            self._audio = playback_mix(self.original, synth, mode)
            self._audio_sr = self.sr
            self._on_ready()
            return
        self._set_state("preparing")
        if self._in_flight:
            return  # al terminar la síntesis en curso se vuelve a llamar a _prepare
        self._in_flight = True
        generation = self._generation
        notes, original, sr, method, seed = list(self.notes), self.original, self.sr, self._method, self.seed
        # Remuestreos ya hechos para esta misma síntesis / este mismo original (cambio de modo).
        synth_ready = self._synth_out[2] if self._synth_out is not None and self._synth_out[:2] == (key, out_sr) \
            else None
        original_ready = (self._original_out[2] if self._original_out is not None
                          and self._original_out[0] is original and self._original_out[1] == out_sr else None)

        def task(progress: ProgressCallback, cancel: threading.Event) -> _Prepared:
            """Hilo trabajador: sintetiza, remuestrea si hace falta y mezcla, sin tocar el estado del reproductor."""
            synth_out: np.ndarray | None = synth
            try:
                if synth_out is None:
                    progress(0.05, f"Sintetizando la tablatura ({len(notes)} notas)…")
                    synth_out = render_notes(notes, sr=sr, method=method, seed=seed)
                if cancel.is_set():
                    raise CancelledError("Síntesis cancelada")
                if out_sr == sr:
                    progress(0.9, "Preparando la mezcla…")
                    return _Prepared(synth_out, playback_mix(original, synth_out, mode), sr=sr)
                # La tarjeta no acepta sr: se remuestrea AQUÍ (no en el hilo de la GUI) y una sola vez.
                progress(0.6, f"Adaptando el audio a {out_sr} Hz (la frecuencia que acepta la tarjeta de sonido)…")
                synth_rs = synth_ready if synth_ready is not None else _resample(synth_out, sr, out_sr)
                original_rs = original_ready
                if original_rs is None and original is not None:
                    original_rs = _resample(original, sr, out_sr)
                if cancel.is_set():
                    raise CancelledError("Síntesis cancelada")
                progress(0.9, "Preparando la mezcla…")
                return _Prepared(synth_out, playback_mix(original_rs, synth_rs, mode), sr=out_sr,
                                 synth_out=synth_rs, original_out=original_rs)
            except CancelledError:
                return _Prepared(synth_out, None, error="se canceló la síntesis")
            except Exception as exc:  # noqa: BLE001 - se informa en el hilo de la GUI
                logger.exception("No se pudo sintetizar la tablatura")
                return _Prepared(synth_out, None, error=str(exc) or type(exc).__name__)

        def done(result: _Prepared) -> None:
            """Hilo de la GUI: guarda la síntesis en la caché y arranca si sigue vigente."""
            self._in_flight = False
            if result.synth is not None:
                self._cache_put(key, result.synth)  # sirve aunque la petición ya no esté vigente
            if result.synth_out is not None:
                self._synth_out = (key, result.sr, result.synth_out)
            if result.original_out is not None:
                self._original_out = (original, result.sr, result.original_out)
            if generation != self._generation:
                # La fuente o el modo cambiaron mientras se sintetizaba: se atiende la petición actual.
                if self._state == "preparing":
                    self._prepare()
                return
            if result.error:
                self._pending_start = None
                if self._state == "preparing":
                    self._set_state("stopped")
                self._message(f"No se pudo preparar el audio de la tablatura: {result.error}")
                return
            self._audio = result.audio
            self._audio_sr = result.sr
            self._on_ready()

        if self._run_in_background is None:
            done(task(lambda _f, _m="": None, threading.Event()))
            return
        title = "Sintetizando la tablatura" if synth is None else f"Adaptando el audio de la tablatura a {out_sr} Hz"
        started = self._run_in_background(title, task, done)
        if started is False:
            self._in_flight = False
            self._pending_start = None
            self._set_state("stopped")
            self._message("Hay otra tarea en curso: espera a que termine para escuchar la tablatura.")

    def _on_ready(self) -> None:
        """El audio está listo: arranca si había un inicio pendiente."""
        if self._pending_start is not None:
            start, self._pending_start = self._pending_start, None
            self._start(start)
        elif self._state == "preparing":
            self._set_state("stopped")

    # ---------------------------------------------------------- reproducción
    def _start(self, start_s: float) -> None:
        """Lanza la salida desde ``start_s`` (s) con el audio preparado y arranca los avisos de posición."""
        assert self._audio is not None
        if start_s >= self.duration_s - 1e-3:
            start_s = 0.0
        if not self._output.available:
            self._play_external(start_s)
            return
        if self._audio_sr != self._output_sr():
            # La tarjeta rechazó esta frecuencia en la reproducción anterior: se prepara el
            # audio a la que acepta en segundo plano (antes se remuestreaba aquí, en el hilo
            # de la GUI, en cada ▶, reanudación o salto: hasta 1 s congelada).
            self._audio = None
            self._pending_start = start_s
            self._prepare()
            return
        first = int(round(start_s * self._audio_sr))
        try:
            latency = self._output.play(self._audio[first:], self._audio_sr)
        except Exception as exc:  # noqa: BLE001 - se informa al usuario
            self._set_state("stopped")
            self._message(f"No se pudo reproducir la tablatura: {exc}")
            return
        self._offset, self._t0, self._latency = start_s, self._clock(), float(latency)
        self._set_state("playing")
        logger.info("Reproduciendo la tablatura (%s, %s) desde %.2f s de %.1f s", self._effective_mode(),
                    self.resolved_method, start_s, self.duration_s)
        self._schedule_tick()
        self._emit_position(start_s)

    def _tick(self) -> None:
        """Aviso periódico (hilo de la GUI): posición, fin de la pista o interrupción externa."""
        # Se cancela el aviso pendiente (si esta llamada no vino de él): nunca hay dos cadenas
        # de avisos a la vez, aunque alguien llame a _tick directamente.
        self._cancel_tick()
        if self._state != "playing":
            return
        position, duration = self.position_s, self.duration_s
        if position >= duration:
            self._finish()
            return
        active = self._output.is_active()
        if active is False:
            if duration - position <= END_SLACK_S:
                self._finish()  # el stream ya entregó todo: terminó con normalidad
                return
            # Otra parte de la aplicación (p. ej. «▶ Todo» de la pestaña Audio) tomó la salida.
            self._paused_at = position
            self._emit_position(position)
            self._set_state("paused")
            self._message(f"La reproducción de la tablatura se interrumpió en {position:.1f} s (otra "
                          "reproducción ocupó la salida de audio). Pulsa reanudar para continuar.")
            return
        self._schedule_tick()
        self._emit_position(position)

    def _finish(self) -> None:
        """Fin natural de la pista: detiene, avisa la posición final y llama a ``on_finished``."""
        self._cancel_tick()
        if self._output.is_active() is not False:  # no cortar una reproducción ajena
            self._output.stop()
        duration = self.duration_s
        self._paused_at = 0.0
        self._emit_position(duration)  # el cursor llega al final antes de que la pestaña lo oculte
        self._set_state("stopped")
        logger.info("Fin de la reproducción de la tablatura (%.1f s)", duration)
        if self.on_finished is not None:
            try:
                self.on_finished()
            except Exception:  # noqa: BLE001 - un callback roto no debe romper el reproductor
                logger.exception("Error en on_finished del reproductor")

    def _play_external(self, start_s: float) -> None:
        """Modo sin sounddevice: WAV + MIDI en ``export_dir`` y reproductor del sistema (sin cursor)."""
        import soundfile as sf

        assert self._audio is not None
        self._set_state("stopped")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        first = int(round(start_s * self._audio_sr))
        audio = np.clip(self._audio[first:], -1.0, 1.0)
        mode = self._effective_mode()
        suffix = "" if first == 0 else f"_desde_{start_s:.1f}s"
        try:
            wav = self._write_unique(self.export_dir / f"{self.name}_{mode}{suffix}.wav",
                                     lambda p: sf.write(str(p), audio, self._audio_sr, subtype="PCM_16"))
            midi = self._write_unique(self.export_dir / f"{self.name}.mid", lambda p: export_midi(self.notes, p))
        except Exception as exc:  # noqa: BLE001 - se informa al usuario
            self._message(f"No se pudo guardar el audio de la tablatura en {self.export_dir}: {exc}")
            return
        self.last_export = wav
        # Cada ▶ desde otro instante escribe un WAV nuevo (hasta ~30 MB con 6 min en estéreo):
        # se borran los anteriores para que la carpeta no crezca sin límite en una clase.
        self._remove_old_exports(keep=wav)
        reason = self._output.error or "sounddevice no está disponible"
        try:
            self._opener(wav)
        except Exception as exc:  # noqa: BLE001
            self._message(f"{reason} No se pudo abrir el reproductor del sistema ({exc}); el audio está en {wav} "
                          f"y el MIDI en {midi}.")
            return
        start_text = f" desde {start_s:.1f} s" if first else ""
        self._message(f"Reproducción integrada no disponible: se abrió {wav.name}{start_text} con el reproductor "
                      f"del sistema; el cursor de la tablatura no se sincroniza en este modo. MIDI: {midi}.")
        logger.info("Modo sin sounddevice: %s abierto con el reproductor del sistema (%s)", wav, reason)

    @staticmethod
    def _write_unique(path: Path, writer: Callable[[Path], Any]) -> Path:
        """Escribe con ``writer``; si el archivo está bloqueado (Windows: abierto en otro programa), usa otro nombre.

        soundfile informa de un archivo que no puede abrir con
        ``soundfile.LibsndfileError``, que hereda de ``RuntimeError`` (no de
        ``OSError``, que es lo que lanza ``open`` en mido): se capturan ambos.
        """
        try:
            writer(path)
            return path
        except (OSError, RuntimeError) as exc:
            alternative = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
            logger.info("No se pudo escribir %s (%s); se usa %s.", path.name, exc, alternative.name)
            writer(alternative)
            return alternative

    def _remove_old_exports(self, keep: Path | None) -> None:
        """Borra los WAV exportados antes (salvo ``keep``); los bloqueados se reintentan la próxima vez.

        Parameters
        ----------
        keep : Path | None
            WAV recién escrito (el que está abriendo el reproductor del sistema),
            o None para borrarlos todos (al cerrar).
        """
        remaining: list[Path] = []
        for path in self._exported:
            if keep is not None and path == keep:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:  # Windows: el reproductor del sistema aún lo tiene abierto
                remaining.append(path)
        self._exported = remaining + ([keep] if keep is not None else [])

    # ------------------------------------------------------------- avisos
    def _set_state(self, state: str) -> None:
        """Cambia el estado y avisa con ``on_state`` (solo si cambió)."""
        if state == self._state:
            return
        self._state = state
        if self.on_state is not None:
            try:
                self.on_state(state)
            except Exception:  # noqa: BLE001
                logger.exception("Error en on_state del reproductor")

    def _emit_position(self, position: float) -> None:
        """Llama a ``on_position`` protegiendo el bucle de avisos de un callback roto."""
        if self.on_position is not None:
            try:
                self.on_position(position)
            except Exception:  # noqa: BLE001
                logger.exception("Error en on_position del reproductor")

    def _message(self, text: str) -> None:
        """Aviso para el usuario: al log y a ``on_message``."""
        logger.info("%s", text)
        if self.on_message is not None:
            try:
                self.on_message(text)
            except Exception:  # noqa: BLE001
                logger.exception("Error en on_message del reproductor")

    def _schedule_tick(self) -> None:
        """Programa el próximo aviso de posición con ``widget.after``."""
        self._cancel_tick()
        try:
            self._tick_job = self.widget.after(self.poll_ms, self._tick)
        except Exception:  # noqa: BLE001 - la ventana pudo cerrarse
            self._tick_job = None

    def _cancel_tick(self) -> None:
        """Cancela el aviso de posición programado, si lo hay."""
        if self._tick_job is not None:
            try:
                self.widget.after_cancel(self._tick_job)
            except Exception:  # noqa: BLE001 - la ventana pudo cerrarse
                pass
            self._tick_job = None
