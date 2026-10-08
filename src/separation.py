"""Etapa 2 (opcional) — Separación del bajo de una mezcla: Demucs o HPSS.

Papel en el pipeline
--------------------
Smartuner espera un MP3 con el bajo aislado. Si el usuario solo tiene la
mezcla completa (bajo + batería + voz + guitarras...), puede marcar "Separar
bajo de mezcla" y esta etapa aísla el bajo antes del preprocesamiento. Hay
dos métodos (``cfg.audio.separation_method``; ``"auto"`` elige Demucs si está
instalado y, si no, HPSS — ver :func:`resolve_method`):

* **Demucs** (:func:`separate_bass`): red neuronal de separación de fuentes
  (modelo ``htdemucs``) que estima por separado batería, bajo, voz y "otros".
  Es la mejor opción (quita también guitarras y teclados), pero necesita
  PyTorch, descarga ~80 MB de pesos la primera vez y tarda minutos en CPU.
  Produce un archivo ``<caché>/<clave>_bass.wav`` que el resto del pipeline
  analiza como si fuera la entrada original.
* **HPSS** (:func:`hpss_bass`): separación armónico-percusiva con librosa +
  pasa-bajas. Tarda segundos y no necesita nada más; es una aproximación
  (ver su docstring). Trabaja sobre la señal ya cargada y no escribe archivos.

Diseño de la ejecución de Demucs
--------------------------------
* **Subproceso** (``python -m src.demucs_runner ...``, ver
  :mod:`src.demucs_runner`) en lugar de llamar a Demucs dentro del proceso:
  así se puede *cancelar* (matando el proceso) sin dejar la GUI colgada, se lee
  su barra de progreso y PyTorch no se carga en memoria de la aplicación.
* **Sin ffmpeg ni torchaudio**: este proceso decodifica la mezcla con
  :func:`src.io_audio.load_audio` (estéreo, 44 100 Hz; funciona con
  imageio-ffmpeg o soundfile) y se la pasa al runner como WAV float en una
  carpeta temporal; el runner escribe el bajo con soundfile. La CLI oficial
  ``python -m demucs`` necesitaría ffmpeg en el PATH para leer MP3 y
  ``torchcodec`` para guardar con torchaudio ≥ 2.9.
* **Progreso**: el runner escribe líneas ``PROGRESO: 45% mensaje`` en sus
  fases (cargar PyTorch y el modelo, escribir) y, mientras separa, la barra
  ``tqdm`` de ``apply_model`` en stderr (``" 45%|████▌     | 9.0/20.0 [...]"``,
  reescrita con ``\\r``). Un hilo lector consume la salida por bloques, extrae
  los porcentajes y el hilo principal los convierte en llamadas
  ``progress(fracción, "Demucs: ...")``.
* **Caché**: el stem se guarda en ``cache/`` con una clave = SHA-1 del
  *contenido* del archivo + nombre del modelo. Si el usuario vuelve a abrir el
  mismo MP3 (aunque lo haya renombrado o movido), la separación es instantánea.
  Cambiar de modelo produce otra clave.
* **Opcional**: Demucs requiere PyTorch (pesado), así que no es dependencia
  obligatoria. :func:`demucs_available` lo comprueba sin importarlo y, si falta,
  :func:`separate_bass` lanza :class:`SeparationError` con instrucciones.

Ejemplo
-------
>>> stem = separate_bass("data/mezcla.mp3", progress=lambda f, m: None)  # doctest: +SKIP
>>> stem.name                                                            # doctest: +SKIP
'3f1c...e9_bass.wav'
"""

from __future__ import annotations

import codecs
import hashlib
import importlib.util
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import IO

import numpy as np

from src import config as _config
from src import demucs_runner as _runner
from src.config import CancelledError, ProgressCallback

logger = logging.getLogger(__name__)

#: Tamaño de bloque (bytes) al leer el archivo para calcular su SHA-1.
_HASH_BLOCK_SIZE: int = 1 << 20  # 1 MiB

#: Cada cuánto (s) el hilo principal revisa ``cancel`` mientras Demucs trabaja.
_POLL_INTERVAL_S: float = 0.1

#: Número de líneas finales de la salida de Demucs que se incluyen en los errores.
_TAIL_LINES: int = 25

#: Porcentaje de una barra tqdm: "  45%|" o " 7.5%|" (número justo antes de "%|").
_TQDM_PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)%\|")

#: Líneas del protocolo del runner (ver :mod:`src.demucs_runner`).
_PROGRESS_LINE = re.compile(re.escape(_runner.PROGRESS_PREFIX) + r"\s*(\d{1,3}(?:\.\d+)?)%\s*(.*)")
_SEPARATING_LINE = re.compile(re.escape(_runner.SEPARATING_PREFIX) + r"\s*(\d+)")
_ERROR_LINE = re.compile(re.escape(_runner.ERROR_PREFIX) + r"\s*(.*)")

#: Módulo que se ejecuta con ``python -m`` para separar (las pruebas inyectan uno falso).
DEFAULT_RUNNER_MODULE: str = "src.demucs_runner"

#: Frecuencia (Hz) y canales con que se entrega la mezcla a Demucs (los de
#: todos sus modelos preentrenados).
DEMUCS_SAMPLE_RATE: int = 44100
DEMUCS_CHANNELS: int = 2

#: Fracción de la barra de progreso al terminar de decodificar la mezcla (antes de lanzar el runner).
_FRACTION_DECODED: float = 0.02

#: Margen de HPSS: un bin se considera armónico solo si su energía armónica supera
#: ``margen`` veces la percusiva (con 1 la separación es "completa"; con > 1 lo dudoso se descarta).
HPSS_MARGIN: float = 2.5
#: Corte (Hz) del pasa-bajas tras HPSS: conserva hasta el 5.º armónico de las notas
#: más usadas (5 × 196 Hz = 980 Hz para G-12) y quita voces/platillos agudos.
HPSS_CUTOFF_HZ: float = 1200.0
#: Tamaño de la FFT de HPSS (muestras): 4096 a 22 050 Hz = 186 ms y 5.4 Hz por bin.
HPSS_N_FFT: int = 4096
#: Salto de la STFT de HPSS (muestras): 512 a 22 050 Hz ≈ 23 ms.
HPSS_HOP_LENGTH: int = 512
#: Duración (s) del filtro de mediana TEMPORAL (componente armónica).
HPSS_HARMONIC_KERNEL_S: float = 0.2
#: Ancho (Hz) del filtro de mediana en FRECUENCIA (componente percusiva).
HPSS_PERCUSSIVE_KERNEL_HZ: float = 150.0
#: Ventana (s) del máximo temporal aplicado a la máscara armónica: un golpe de
#: batería más corto que esto no puede "morder" una nota sostenida del bajo.
HPSS_MASK_SMOOTH_S: float = 0.2

_INSTALL_HELP = (
    "La separación del bajo con Demucs requiere el paquete demucs, que no está instalado.\n\n"
    "Instálalo (junto con las demás dependencias opcionales) con:\n"
    "  pip install -r requirements-optional.txt\n"
    "(o solo Demucs: pip install demucs).\n\n"
    "Demucs requiere PyTorch, que se instala automáticamente pero es pesado "
    "(~1–2 GB); funciona sin GPU, aunque tarda varios minutos por canción. La primera "
    "separación descarga además los pesos del modelo (~80 MB), por lo que necesita "
    "conexión a Internet.\n\n"
    "Sin instalar nada puedes elegir el método «hpss» (separación armónico-percusiva, "
    "más rápida pero aproximada). Si tu MP3 ya contiene solo el bajo, desmarca "
    "«Separar bajo de mezcla»."
)


class SeparationError(RuntimeError):
    """Demucs no está instalado, falló o fue cancelado (mensaje en español).

    Notes
    -----
    La cancelación se señala con :class:`SeparationCancelledError`, que hereda
    de esta clase *y* de :class:`src.config.CancelledError`.
    """


class SeparationDepsError(SeparationError):
    """El proceso de Demucs no pudo importar PyTorch/Demucs (instalación incompleta o rota).

    Se lanza cuando el runner termina con
    :data:`src.demucs_runner.EXIT_MISSING_DEPS`: p. ej. en Windows, un torch
    sin el Redistributable de Visual C++ (``WinError 126``). Con el método
    ``"auto"`` el pipeline recurre entonces a HPSS en lugar de fallar
    (:func:`src.pipeline.analyze`).
    """


class SeparationCancelledError(SeparationError, CancelledError):
    """El usuario canceló la separación (se mató el subproceso de Demucs).

    Hereda de :class:`SeparationError` (contrato de este módulo) y de
    :class:`src.config.CancelledError` (convención común a todos los procesos
    largos), de modo que ``except CancelledError`` en el pipeline o en la GUI
    la trata como una cancelación y no como un fallo.
    """


# ---------------------------------------------------------------------------
# Disponibilidad, elección del método y caché
# ---------------------------------------------------------------------------


def demucs_available() -> bool:
    """True si los paquetes ``demucs`` y ``torch`` se pueden importar (sin importarlos).

    Returns
    -------
    bool
        ``True`` si ``import demucs`` e ``import torch`` encontrarían sus paquetes.

    Notes
    -----
    Se usa :func:`importlib.util.find_spec`, que solo *localiza* el paquete en
    ``sys.path`` sin ejecutar su ``__init__``; así no se paga el coste de
    importar PyTorch (varios segundos) solo para habilitar una casilla de la GUI.

    Examples
    --------
    >>> isinstance(demucs_available(), bool)
    True
    """
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("demucs", "torch"))
    except (ImportError, ValueError):
        return False


def resolve_method(method: str) -> str:
    """Traduce ``audio.separation_method`` al método que realmente se ejecutará.

    Parameters
    ----------
    method : str
        ``"auto"``, ``"demucs"`` o ``"hpss"`` (ver
        :data:`src.config.SEPARATION_METHODS`).

    Returns
    -------
    str
        ``"demucs"`` o ``"hpss"``. ``"auto"`` da ``"demucs"`` si
        :func:`demucs_available` y ``"hpss"`` en caso contrario.

    Raises
    ------
    SeparationError
        Si se pidió ``"demucs"`` y no está instalado (con instrucciones de
        instalación y la alternativa «hpss»).
    ValueError
        Si ``method`` no es uno de los tres válidos.

    Examples
    --------
    >>> resolve_method("hpss")
    'hpss'
    >>> resolve_method("auto") in ("demucs", "hpss")
    True
    """
    method = str(method).strip().lower()
    if method not in _config.SEPARATION_METHODS:
        raise ValueError(f"Método de separación desconocido {method!r}; usa uno de: "
                         f"{', '.join(_config.SEPARATION_METHODS)}.")
    if method == "auto":
        return "demucs" if demucs_available() else "hpss"
    if method == "demucs" and not demucs_available():
        raise SeparationError(_INSTALL_HELP)
    return method


def cache_key(path: str | Path, model: str = "htdemucs") -> str:
    """Clave de caché: SHA-1 (hex) del contenido del archivo concatenado con el modelo.

    Parameters
    ----------
    path : str | Path
        Archivo de audio de entrada.
    model : str, optional
        Nombre del modelo de Demucs (``"htdemucs"`` por defecto).

    Returns
    -------
    str
        40 dígitos hexadecimales. Depende solo de los bytes del archivo y del
        modelo: renombrar o mover el archivo no cambia la clave; cambiar un
        solo byte o el modelo, sí.

    Raises
    ------
    FileNotFoundError
        Si ``path`` no existe.

    Notes
    -----
    El archivo se lee por bloques de 1 MiB para no cargar en memoria un audio
    largo completo. Entre el contenido y el modelo se inserta un byte nulo
    como separador.

    Examples
    --------
    >>> cache_key("data/mezcla.mp3")  # doctest: +SKIP
    '9a0364b9e99bb480dd25e1f0284c8555e1c2b0f6'
    """
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    digest.update(b"\x00")
    digest.update(model.encode("utf-8"))
    return digest.hexdigest()


def cached_stem_path(path: str | Path, model: str = "htdemucs", cache_dir: str | Path | None = None) -> Path:
    """Ruta donde se guarda (o guardaría) el stem de bajo: ``<cache_dir>/<clave>_bass.wav``.

    Parameters
    ----------
    path : str | Path
        Archivo de audio de entrada.
    model : str, optional
        Nombre del modelo de Demucs.
    cache_dir : str | Path | None, optional
        Carpeta de caché. ``None`` usa :data:`src.config.CACHE_DIR`.

    Returns
    -------
    Path
        Ruta del stem en caché (puede no existir todavía; no se crea nada).

    Examples
    --------
    >>> cached_stem_path("data/mezcla.mp3", cache_dir="cache").name  # doctest: +SKIP
    '9a0364b9e99bb480dd25e1f0284c8555e1c2b0f6_bass.wav'
    """
    folder = Path(cache_dir) if cache_dir is not None else Path(_config.CACHE_DIR)
    return folder / f"{cache_key(path, model)}_bass.wav"


# ---------------------------------------------------------------------------
# Ejecución de Demucs (subproceso src.demucs_runner)
# ---------------------------------------------------------------------------


def _demucs_command(
    input_wav: str | Path,
    output_wav: str | Path,
    model: str,
    device: str = "cpu",
    runner_module: str = DEFAULT_RUNNER_MODULE,
) -> list[str]:
    """Construye la línea de comandos del runner de Demucs.

    Parameters
    ----------
    input_wav : str | Path
        WAV de entrada (la mezcla, estéreo a 44 100 Hz).
    output_wav : str | Path
        WAV de salida con el bajo.
    model : str
        Nombre del modelo (``--model``).
    device : str, optional
        Dispositivo de PyTorch (``"cpu"`` por defecto).
    runner_module : str, optional
        Módulo que se ejecuta con ``python -m`` (por defecto
        :mod:`src.demucs_runner`; las pruebas pasan uno falso).

    Returns
    -------
    list[str]
        Argumentos para :class:`subprocess.Popen` (una lista, sin shell: las
        rutas con espacios o tildes llegan intactas también en Windows). Se usa
        ``sys.executable`` para ejecutar el runner con el mismo intérprete (y
        entorno virtual) que la aplicación.

    Examples
    --------
    >>> cmd = _demucs_command("in.wav", "bajo.wav", "htdemucs")
    >>> cmd[1:]
    ['-m', 'src.demucs_runner', '--input', 'in.wav', '--output', 'bajo.wav', '--model', 'htdemucs', '--device', 'cpu']
    """
    return [
        sys.executable, "-m", str(runner_module),
        "--input", str(input_wav),
        "--output", str(output_wav),
        "--model", str(model),
        "--device", str(device),
    ]


def _parse_percent(text: str) -> float | None:
    """Devuelve el último porcentaje tqdm (0–100) que aparezca en ``text``, o ``None``.

    Examples
    --------
    >>> _parse_percent(" 45%|████▌     | 9.0/20.0 [00:03<00:04, 2.6seconds/s]")
    45.0
    >>> _parse_percent("Separating track mezcla.mp3") is None
    True
    """
    matches = _TQDM_PERCENT.findall(text)
    if not matches:
        return None
    return min(100.0, max(0.0, float(matches[-1])))


class _OutputReader(threading.Thread):
    """Hilo que lee la salida del runner por bloques y la traduce a progreso.

    Se usa un hilo porque leer de una tubería bloquea: si el hilo principal
    leyera directamente, no podría reaccionar a ``cancel`` mientras el runner
    no escriba nada. tqdm reescribe su barra con ``\\r`` (sin salto de línea),
    por eso no se lee línea a línea sino por bloques, partiendo en ``\\r`` y
    ``\\n``.

    Protocolo (ver :mod:`src.demucs_runner`):

    * ``PROGRESO: N% mensaje`` → fracción N/100 con ese mensaje.
    * ``SEPARANDO: n`` → empiezan ``n`` barras tqdm que cubren el tramo
      :data:`src.demucs_runner.PCT_SEPARATION_START` –
      :data:`src.demucs_runner.PCT_SEPARATION_END`; cuando un porcentaje baja
      (empieza la barra siguiente) se pasa a la siguiente "red".
    * Barras tqdm ANTES de ``SEPARANDO`` (p. ej. la descarga de los pesos) se
      ignoran: no son el avance de la separación.
    * ``ERROR: mensaje`` → se guarda para el mensaje de :class:`SeparationError`.

    Attributes
    ----------
    updates : queue.Queue[tuple[float, str]]
        ``(fracción 0–1, mensaje)`` en orden, con fracción no decreciente y
        sin repetir actualizaciones idénticas consecutivas.
    tail : collections.deque[str]
        Últimas líneas no vacías de la salida (para mensajes de error).
    errors : list[str]
        Mensajes de las líneas ``ERROR:``.
    """

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(name="demucs-output-reader", daemon=True)
        self._stream = stream
        # Decodificador incremental: un carácter UTF-8 multibyte (p. ej. "█")
        # puede quedar partido entre dos bloques.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._last: tuple[float, str] = (0.0, "")
        self._passes = 0              # 0 = aún no empezó apply_model
        self._pass_idx = 0
        self._last_tqdm = 0.0
        self.updates: queue.Queue[tuple[float, str]] = queue.Queue()
        self.tail: deque[str] = deque(maxlen=_TAIL_LINES)
        self.errors: list[str] = []

    def _put(self, fraction: float, message: str) -> None:
        """Encola ``(fracción, mensaje)`` garantizando que la fracción nunca retroceda."""
        fraction = max(min(fraction, 1.0), self._last[0])
        if (fraction, message) != self._last:
            self._last = (fraction, message)
            self.updates.put(self._last)

    def _tqdm(self, text: str) -> None:
        """Convierte un porcentaje de tqdm (si lo hay) en avance global de la separación."""
        percent = _parse_percent(text)
        if percent is None or self._passes == 0:
            return
        if percent < self._last_tqdm - 5.0 and self._pass_idx < self._passes - 1:
            self._pass_idx += 1  # el porcentaje bajó: empezó la barra de la red siguiente
        self._last_tqdm = percent
        lo, hi = _runner.PCT_SEPARATION_START / 100.0, _runner.PCT_SEPARATION_END / 100.0
        fraction = lo + (hi - lo) * (self._pass_idx + percent / 100.0) / self._passes
        which = f" (red {self._pass_idx + 1} de {self._passes})" if self._passes > 1 else ""
        self._put(fraction, f"Demucs: separando… {percent:.0f} %{which}")

    def _line(self, line: str) -> None:
        """Procesa una línea COMPLETA de la salida."""
        if m := _PROGRESS_LINE.search(line):
            message = m.group(2).strip() or "trabajando..."
            logger.info("Demucs: %s", message)
            self._put(float(m.group(1)) / 100.0, f"Demucs: {message}")
        elif m := _SEPARATING_LINE.search(line):
            self._passes, self._pass_idx, self._last_tqdm = max(1, int(m.group(1))), 0, 0.0
        elif m := _ERROR_LINE.search(line):
            self.errors.append(m.group(1).strip())
        else:
            self._tqdm(line)

    def _feed(self, text: str) -> None:
        self._pending += text
        pieces = re.split(r"[\r\n]", self._pending)
        self._pending = pieces.pop()  # fragmento aún incompleto
        for piece in pieces:
            self._line(piece)
            if piece.strip():
                self.tail.append(piece.rstrip())
                logger.debug("demucs: %s", piece.rstrip())
        # La barra de tqdm no termina en salto de línea hasta que se redibuja:
        # se analiza también el fragmento incompleto para no ir un paso atrasado
        # (solo como tqdm: las líneas del protocolo se procesan completas).
        self._tqdm(self._pending)

    def run(self) -> None:
        """Lee la salida del subproceso por bloques hasta que se cierra (cuerpo del hilo)."""
        read = getattr(self._stream, "read1", self._stream.read)
        while True:
            try:
                chunk = read(4096)
            except (OSError, ValueError):  # tubería cerrada al matar el proceso
                break
            if not chunk:
                break
            self._feed(self._decoder.decode(chunk))
        self._feed(self._decoder.decode(b"", final=True) + "\n")

    def tail_text(self) -> str:
        """Últimas líneas de la salida como un solo texto.

        Returns
        -------
        str
            Las líneas guardadas en ``tail``, unidas con saltos de línea.
        """
        return "\n".join(self.tail)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Mata el subproceso de Demucs y espera a que termine (sin dejar zombis)."""
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - muy improbable tras kill()
        logger.warning("Demucs no terminó tras kill(); se abandona el proceso %d", proc.pid)


def _subprocess_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Entorno del runner: el dado (o el actual) + raíz del proyecto en PYTHONPATH + UTF-8.

    La raíz se ANTEPONE a ``PYTHONPATH`` para que ``python -m src.demucs_runner``
    encuentre el paquete ``src`` aunque la aplicación se haya lanzado desde otra
    carpeta; ``PYTHONIOENCODING`` evita que en Windows los mensajes con tildes
    lleguen en cp1252.
    """
    sub_env = dict(os.environ if env is None else env)
    root = str(_config.PROJECT_ROOT)
    existing = sub_env.get("PYTHONPATH", "")
    sub_env["PYTHONPATH"] = os.pathsep.join(p for p in (root, existing) if p)
    sub_env.setdefault("PYTHONUNBUFFERED", "1")  # que los mensajes lleguen en cuanto se escriben
    sub_env["PYTHONIOENCODING"] = "utf-8"
    return sub_env


def _prepare_input(path: Path, wav: Path) -> float:
    """Decodifica la mezcla a estéreo 44.1 kHz y la escribe como WAV float en ``wav``.

    Returns
    -------
    float
        Duración de la mezcla en segundos.

    Raises
    ------
    SeparationError
        Si el audio no se puede decodificar (sin ffmpeg ni soundfile capaz,
        archivo dañado...) o escribir.
    """
    import soundfile as sf  # noqa: PLC0415

    from src import io_audio  # noqa: PLC0415 - import diferido (evita ciclos y coste al importar)

    try:
        y, sr = io_audio.load_audio(path, sr=DEMUCS_SAMPLE_RATE, channels=DEMUCS_CHANNELS)
        sf.write(str(wav), np.ascontiguousarray(y.T), sr, subtype="FLOAT")
    except (io_audio.AudioLoadError, io_audio.FFmpegNotFoundError, OSError, RuntimeError) as exc:
        raise SeparationError(f"No se pudo preparar el audio para Demucs: {exc}") from exc
    return y.shape[-1] / float(sr)


def separate_bass(
    path: str | Path,
    model: str = "htdemucs",
    cache_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    env: Mapping[str, str] | None = None,
    runner_module: str = DEFAULT_RUNNER_MODULE,
    device: str = "cpu",
) -> Path:
    """Devuelve la ruta al stem de bajo aislado por Demucs (WAV), usando la caché si existe.

    ``progress(fraccion_0_1, mensaje)`` se llama a medida que Demucs avanza.
    Si ``cancel`` se activa se mata el subproceso y se lanza
    :class:`SeparationCancelledError` (subclase de :class:`SeparationError` y
    de :class:`src.config.CancelledError`).

    Parameters
    ----------
    path : str | Path
        Archivo de audio con la mezcla (MP3, WAV, ...).
    model : str, optional
        Modelo de Demucs (``"htdemucs"`` por defecto).
    cache_dir : str | Path | None, optional
        Carpeta de caché; ``None`` usa :data:`src.config.CACHE_DIR`.
    progress : ProgressCallback | None, optional
        Callback ``progress(fraccion, mensaje)`` con fracción en [0, 1] no
        decreciente y mensajes del tipo ``"Demucs: separando… 45 %"``.
    cancel : threading.Event | None, optional
        Evento de cancelación; se revisa cada 0.1 s mientras Demucs trabaja.
    env : Mapping[str, str] | None, optional
        Variables de entorno del subproceso (``None`` hereda las del proceso
        actual). La raíz del proyecto se antepone siempre a ``PYTHONPATH``.
    runner_module : str, optional
        Módulo ejecutado con ``python -m`` (por defecto
        :mod:`src.demucs_runner`). Las pruebas inyectan un runner FALSO (vía
        ``PYTHONPATH`` en ``env``) que sigue el mismo protocolo de argumentos
        y salida.
    device : str, optional
        Dispositivo de PyTorch para el runner (``"cpu"`` por defecto).

    Returns
    -------
    Path
        ``<cache_dir>/<clave>_bass.wav`` (ver :func:`cached_stem_path`):
        WAV PCM de 16 bits, estéreo, 44 100 Hz.

    Raises
    ------
    SeparationError
        Si el archivo no existe o no se puede decodificar, si Demucs no está
        instalado (mensaje con instrucciones de instalación), si el runner
        termina con código de salida ≠ 0 (mensaje con su explicación y el
        final de su salida) o si no genera el WAV del bajo.
    SeparationCancelledError
        Si ``cancel`` se activa (también es :class:`src.config.CancelledError`).

    Notes
    -----
    La mezcla decodificada y el bajo se escriben en una carpeta temporal
    *dentro* de la caché y al terminar el stem se mueve con :func:`os.replace`
    (atómico en el mismo sistema de archivos): una separación cancelada o
    fallida nunca deja en la caché un archivo a medias que después se tomaría
    por bueno, y la carpeta temporal se borra siempre.

    Con modelos que son "bolsas" de varias redes (p. ej. ``htdemucs_ft``)
    tqdm recorre 0→100 % una vez por red; el lector lo detecta y reparte el
    tramo de separación entre ellas, así que el progreso nunca retrocede.

    Examples
    --------
    >>> stem = separate_bass("data/mezcla.mp3",
    ...                      progress=lambda f, m: print(m))  # doctest: +SKIP
    Demucs: decodificando mezcla.mp3 (estéreo, 44100 Hz)...
    Demucs: cargando PyTorch y Demucs...
    ...
    Demucs: separación terminada
    """
    path = Path(path)
    if not path.is_file():
        raise SeparationError(f"No existe el archivo de audio a separar: {path}")

    def report(fraction: float, message: str) -> None:
        """Reenvía ``(fracción 0–1, mensaje)`` al callback del usuario, si lo hay."""
        if progress is not None:
            progress(fraction, message)

    def check_cancel(when: str) -> None:
        """Lanza :class:`SeparationCancelledError` si el usuario pidió cancelar."""
        if cancel is not None and cancel.is_set():
            raise SeparationCancelledError(f"Separación cancelada por el usuario {when}.")

    target = cached_stem_path(path, model, cache_dir)
    if target.is_file():
        logger.info("Separación: usando caché %s (no se ejecuta Demucs)", target)
        report(1.0, "Demucs: usando caché")
        return target

    if not demucs_available():
        raise SeparationError(_INSTALL_HELP)
    check_cancel("antes de iniciar Demucs")

    target.parent.mkdir(parents=True, exist_ok=True)
    root = Path(_config.PROJECT_ROOT)
    # En Windows, evita que aparezca una ventana de consola al lanzar desde la GUI.
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix=".demucs_tmp_", dir=target.parent, ignore_cleanup_errors=True
    ) as tmp:
        mix_wav, bass_wav = Path(tmp) / "mezcla.wav", Path(tmp) / "bajo.wav"
        report(0.0, f"Demucs: decodificando {path.name} (estéreo, {DEMUCS_SAMPLE_RATE} Hz)...")
        duration = _prepare_input(path, mix_wav)
        check_cancel("antes de iniciar Demucs")
        cmd = _demucs_command(mix_wav, bass_wav, model, device=device, runner_module=runner_module)
        logger.info("Separando el bajo con Demucs (modelo %s, %s): %s (%.1f s de audio)",
                    model, device, path.name, duration)
        logger.debug("Comando: %s", " ".join(cmd))
        report(_FRACTION_DECODED, f"Demucs: iniciando el proceso de separación (modelo {model})...")

        # stdout se redirige al mismo canal que stderr: tqdm escribe el progreso
        # en stderr y el runner sus líneas de protocolo en stdout; así se lee
        # todo con un solo hilo y el final de la salida sirve para diagnosticar.
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(root),
                env=_subprocess_env(env),
                creationflags=creationflags,
            )
        except OSError as exc:
            raise SeparationError(f"No se pudo lanzar el proceso de Demucs ({cmd[0]}): {exc}") from exc
        assert proc.stdout is not None
        reader = _OutputReader(proc.stdout)
        reader.start()

        def forward(update: tuple[float, str]) -> None:
            """Reenvía al usuario el avance del runner (0–1), detrás de la fase de decodificación."""
            fraction, message = update
            report(_FRACTION_DECODED + (1.0 - _FRACTION_DECODED) * fraction, message)

        try:
            while True:
                if cancel is not None and cancel.is_set():
                    logger.info("Separación cancelada: se detiene Demucs (pid %d)", proc.pid)
                    _kill(proc)
                    raise SeparationCancelledError("Separación cancelada por el usuario.")
                try:
                    forward(reader.updates.get(timeout=_POLL_INTERVAL_S))
                except queue.Empty:
                    if proc.poll() is not None:
                        break
        finally:
            # Ante cualquier salida (cancelación o excepción en ``progress``) el
            # proceso no debe quedar vivo, y el hilo lector termina al ver EOF.
            _kill(proc)
            reader.join(timeout=5)
            proc.stdout.close()

        while not reader.updates.empty():  # avances leídos al final
            forward(reader.updates.get_nowait())

        tail = reader.tail_text() or "(sin salida)"
        if proc.returncode != 0:
            explanation = "\n".join(reader.errors) or "terminó sin explicar el motivo."
            error_class = SeparationDepsError if proc.returncode == _runner.EXIT_MISSING_DEPS else SeparationError
            raise error_class(
                f"Demucs falló (código de salida {proc.returncode}): {explanation}\n\n"
                f"Últimas líneas de su salida:\n{tail}"
            )
        if not bass_wav.is_file():
            raise SeparationError(
                f"Demucs terminó sin generar el stem de bajo esperado ({bass_wav.name}).\n"
                f"Últimas líneas de su salida:\n{tail}"
            )
        os.replace(bass_wav, target)

    logger.info("Stem de bajo guardado en caché: %s (separación en %.1f s)", target, time.perf_counter() - t_start)
    report(1.0, "Demucs: separación terminada")
    return target


# ---------------------------------------------------------------------------
# HPSS: separación armónico-percusiva + pasa-bajas (sin PyTorch)
# ---------------------------------------------------------------------------


def hpss_bass(
    y: np.ndarray,
    sr: int,
    margin: float = HPSS_MARGIN,
    cutoff_hz: float = HPSS_CUTOFF_HZ,
    n_fft: int = HPSS_N_FFT,
    hop_length: int = HPSS_HOP_LENGTH,
    harmonic_kernel_s: float = HPSS_HARMONIC_KERNEL_S,
    percussive_kernel_hz: float = HPSS_PERCUSSIVE_KERNEL_HZ,
    mask_smooth_s: float = HPSS_MASK_SMOOTH_S,
) -> np.ndarray:
    """Aproxima el bajo de una mezcla: componente armónica (HPSS) + pasa-bajas.

    Parameters
    ----------
    y : np.ndarray
        Mezcla mono, forma ``(n_muestras,)``.
    sr : int
        Frecuencia de muestreo en Hz.
    margin : float, optional
        Margen de HPSS (2.5 por defecto; ~2–3 es razonable). Un bin del
        espectrograma va a la parte armónica solo si su energía "armónica"
        supera ``margin`` veces la "percusiva".
    cutoff_hz : float, optional
        Corte del pasa-bajas Butterworth de fase cero en Hz (1200 por defecto).
    n_fft : int, optional
        Tamaño de la FFT en muestras (4096 ≈ 186 ms a 22 050 Hz).
    hop_length : int, optional
        Salto de la STFT en muestras (512 ≈ 23 ms a 22 050 Hz).
    harmonic_kernel_s : float, optional
        Longitud (s) de la mediana temporal (0.2 s por defecto).
    percussive_kernel_hz : float, optional
        Ancho (Hz) de la mediana en frecuencia (150 Hz por defecto).
    mask_smooth_s : float, optional
        Ventana (s) del máximo temporal de la máscara armónica (0.2 s por
        defecto; 0 = sin suavizar). Ver Notes.

    Returns
    -------
    np.ndarray
        Señal "solo bajo" aproximada, ``float32``, misma longitud que ``y``.

    Raises
    ------
    ValueError
        Si ``y`` no es 1-D o algún parámetro no es positivo.

    Notes
    -----
    **Idea de HPSS** (Fitzgerald, 2010; ``librosa.decompose.hpss``). En el
    espectrograma de magnitud S(f, t):

    * un sonido ARMÓNICO sostenido (una nota de bajo) es una línea
      HORIZONTAL: energía estable en unas pocas frecuencias durante un rato;
    * un sonido PERCUSIVO (golpe de batería, "clic" de caja registradora,
      ataque de una cuerda) es una línea VERTICAL: energía en todas las
      frecuencias durante un instante.

    Una mediana a lo largo del tiempo, H = mediana_t(S) (ventana de
    ``harmonic_kernel_s``), conserva las líneas horizontales y borra las
    verticales; una mediana a lo largo de la frecuencia, P = mediana_f(S)
    (ventana de ``percussive_kernel_hz``), hace lo contrario. La máscara
    suave armónica es

        M_H = H² / (H² + (margen · P)²)

    y la señal armónica es ISTFT(M_H · STFT(y)). Con margen > 1 los bins
    dudosos (ni claramente armónicos ni percusivos) se descartan: es
    preferible perder algo de bajo a dejar pasar golpes que la segmentación
    tomaría por notas.

    **Suavizado de la máscara**: en el instante de un golpe de batería, P
    crece en TODAS las frecuencias, también en las del bajo, y M_H cae
    aunque la nota siga sonando: el bajo queda "mordido" y, al recuperarse,
    la detección de onsets ve una nota nueva (con «Money» de Pink Floyd, sin
    suavizar, ~20 % de los segmentos eran notas partidas en dos). Por eso se toma el
    máximo de M_H en una ventana de ``mask_smooth_s`` (0.2 s): un golpe más
    corto no puede apagar una nota sostenida. El precio es que en los bins
    donde suena el bajo se cuela hasta ~0.1 s del golpe (con 0.2 s, las notas
    partidas bajan al ~6 %, como sin separar).

    Después, el **pasa-bajas** de ``cutoff_hz`` quita
    voces, platillos y el brillo de guitarras, y conserva los armónicos que
    usa la recompensa (hasta 5 × 196 Hz = 980 Hz para G-12).

    **Es una aproximación**: no "sabe" qué instrumento suena, solo separa
    sonidos sostenidos de golpes y graves de agudos. Guitarras, teclados o
    voces graves sostenidas por debajo de ~1.2 kHz se quedan. Para una
    separación real del bajo usa Demucs.

    **Coste**: la mediana sobre todo el espectrograma de una canción de 6 min
    tarda decenas de segundos, así que se calcula solo en los bins por debajo
    de 2 × ``cutoff_hz`` (todo lo de arriba lo eliminaría el pasa-bajas de
    todos modos; el margen de una octava evita un escalón audible en el
    corte): 6 min de audio a 22 050 Hz se procesan en ~5 s.

    Examples
    --------
    >>> sr = 22050
    >>> t = np.arange(2 * sr) / sr
    >>> bajo = 0.5 * np.sin(2 * np.pi * 55.0 * t)                      # A1 sostenido
    >>> clics = np.zeros_like(t); clics[::sr // 4] = 1.0              # golpes secos cada 0.25 s
    >>> agudo = 0.3 * np.sin(2 * np.pi * 3000.0 * t)                    # "platillo/voz" agudo
    >>> sep = hpss_bass(bajo + clics + agudo, sr)
    >>> mid = slice(sr // 2, -sr // 2)                                 # sin los bordes
    >>> err = np.sqrt(np.mean((sep[mid] - bajo[mid]) ** 2)) / np.sqrt(np.mean(bajo[mid] ** 2))
    >>> bool(err < 0.2)                                                # queda (casi) solo el bajo
    True
    """
    import librosa  # noqa: PLC0415 - import diferido: librosa tarda en cargarse
    from scipy.ndimage import maximum_filter1d  # noqa: PLC0415

    from src.preprocessing import lowpass  # noqa: PLC0415

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"hpss_bass espera una señal mono 1-D; se recibió forma {y.shape}.")
    if min(margin, cutoff_hz, n_fft, hop_length, harmonic_kernel_s, percussive_kernel_hz) <= 0 or mask_smooth_s < 0:
        raise ValueError("Todos los parámetros de hpss_bass deben ser positivos (mask_smooth_s puede ser 0).")
    if y.size == 0:
        return y.copy()
    t0 = time.perf_counter()
    D = librosa.stft(y, n_fft=int(n_fft), hop_length=int(hop_length))
    bin_hz = sr / float(n_fft)
    k_max = min(D.shape[0], int(np.ceil(2.0 * cutoff_hz / bin_hz)) + 1)
    # Kernels en unidades físicas → frames / bins (impares, para que la mediana esté centrada).
    k_harm = max(3, int(round(harmonic_kernel_s * sr / hop_length)) | 1)
    k_perc = max(3, int(round(percussive_kernel_hz / bin_hz)) | 1)
    # mask=True devuelve las máscaras suaves (M_H, M_P) en vez de los espectrogramas separados.
    mask_h, _mask_p = librosa.decompose.hpss(D[:k_max], kernel_size=(k_harm, k_perc), margin=float(margin),
                                             mask=True)
    k_smooth = int(round(mask_smooth_s * sr / hop_length))
    if k_smooth > 1:
        mask_h = maximum_filter1d(mask_h, size=k_smooth, axis=1)  # máximo a lo largo del TIEMPO
    D_harm = np.zeros_like(D)
    D_harm[:k_max] = D[:k_max] * mask_h
    y_harm = librosa.istft(D_harm, hop_length=int(hop_length), n_fft=int(n_fft), length=y.size)
    out = lowpass(y_harm, sr, cutoff_hz, order=4).astype(np.float32, copy=False)
    logger.info("HPSS: componente armónica (margen %.1f, medianas de %d frames ≈ %.2f s y %d bins ≈ %.0f Hz, "
                "máscara suavizada %.2f s) + pasa-bajas %.0f Hz en %.1f s para %.1f s de audio", margin, k_harm,
                k_harm * hop_length / sr, k_perc, k_perc * bin_hz, max(k_smooth, 1) * hop_length / sr, cutoff_hz,
                time.perf_counter() - t0, y.size / sr)
    return out


__all__ = [
    "SeparationError", "SeparationCancelledError", "demucs_available", "resolve_method", "cache_key",
    "cached_stem_path", "separate_bass", "hpss_bass", "DEMUCS_SAMPLE_RATE", "DEFAULT_RUNNER_MODULE",
]
