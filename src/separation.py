"""Etapa 2 (opcional) — Separación del bajo con Demucs.

Papel en el pipeline
--------------------
Smartuner espera un MP3 con el bajo aislado. Si el usuario solo tiene la
mezcla completa (bajo + batería + voz + ...), puede marcar "Separar bajo de
mezcla" y esta etapa ejecuta Demucs (modelo ``htdemucs``, modo
``--two-stems bass``), una red neuronal de separación de fuentes que produce
dos pistas: ``bass.wav`` (solo bajo) y ``no_bass.wav`` (todo lo demás). El
resto del pipeline (:mod:`src.preprocessing`, segmentación, pitch, bandits)
trabaja después sobre ``bass.wav`` como si fuera la entrada original.

Diseño
------
* **Subproceso** (``python -m demucs ...``) en lugar de llamar a Demucs dentro
  del proceso: así se puede *cancelar* (matando el proceso) sin dejar la GUI
  colgada, se lee su barra de progreso y PyTorch no se carga en memoria de la
  aplicación.
* **Progreso**: Demucs dibuja una barra ``tqdm`` en stderr
  (``" 45%|████▌     | 9.0/20.0 [...]"``) reescribiéndola con ``\\r``. Un hilo
  lector consume la salida por bloques, extrae los porcentajes y el hilo
  principal los convierte en llamadas ``progress(0.45, "Demucs: 45 %")``.
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
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from src import config as _config
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

_INSTALL_HELP = (
    "La separación del bajo requiere Demucs, que no está instalado.\n\n"
    "Instálalo (junto con las demás dependencias opcionales) con:\n"
    "  pip install -r requirements-optional.txt\n"
    "(o solo Demucs: pip install demucs).\n\n"
    "Demucs requiere PyTorch, que se instala automáticamente pero es pesado "
    "(~1–2 GB); funciona sin GPU, aunque más lento. La primera ejecución descarga "
    "además los pesos del modelo (~80 MB), por lo que necesita conexión a Internet.\n\n"
    "Si tu MP3 ya contiene solo el bajo, desmarca «Separar bajo de mezcla»."
)


class SeparationError(RuntimeError):
    """Demucs no está instalado, falló o fue cancelado (mensaje en español).

    Notes
    -----
    La cancelación se señala con :class:`SeparationCancelledError`, que hereda
    de esta clase *y* de :class:`src.config.CancelledError`.
    """


class SeparationCancelledError(SeparationError, CancelledError):
    """El usuario canceló la separación (se mató el subproceso de Demucs).

    Hereda de :class:`SeparationError` (contrato de este módulo) y de
    :class:`src.config.CancelledError` (convención común a todos los procesos
    largos), de modo que ``except CancelledError`` en el pipeline o en la GUI
    la trata como una cancelación y no como un fallo.
    """


# ---------------------------------------------------------------------------
# Disponibilidad y caché
# ---------------------------------------------------------------------------


def demucs_available() -> bool:
    """True si el paquete ``demucs`` se puede importar (sin importarlo por completo).

    Returns
    -------
    bool
        ``True`` si ``import demucs`` encontraría el paquete.

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
        return importlib.util.find_spec("demucs") is not None
    except (ImportError, ValueError):
        return False


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
# Ejecución de Demucs
# ---------------------------------------------------------------------------


def _demucs_command(path: str | Path, model: str, out_dir: str | Path) -> list[str]:
    """Construye la línea de comandos de Demucs.

    Parameters
    ----------
    path : str | Path
        Archivo de audio de entrada.
    model : str
        Nombre del modelo (``-n``).
    out_dir : str | Path
        Carpeta de salida (``-o``). Demucs escribe en
        ``<out_dir>/<model>/<nombre_sin_extension>/bass.wav``.

    Returns
    -------
    list[str]
        Argumentos para :class:`subprocess.Popen`. Se usa ``sys.executable``
        para ejecutar Demucs con el mismo intérprete (y entorno virtual) que
        la aplicación.
    """
    return [
        sys.executable, "-m", "demucs",
        "--two-stems", "bass",  # solo dos pistas: bajo y "todo lo demás" (más rápido)
        "-n", str(model),
        "-o", str(out_dir),
        str(path),
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


def _expected_stem(out_dir: Path, model: str, path: Path) -> Path:
    """Ruta donde Demucs deja el stem de bajo dentro de ``out_dir``.

    Demucs nombra la carpeta de la pista con el nombre del archivo sin la
    última extensión (``mezcla.v2.mp3`` → ``mezcla.v2``).
    """
    track = path.name.rsplit(".", 1)[0]
    return out_dir / model / track / "bass.wav"


class _OutputReader(threading.Thread):
    """Hilo que lee la salida de Demucs por bloques y extrae el progreso.

    Se usa un hilo porque leer de una tubería bloquea: si el hilo principal
    leyera directamente, no podría reaccionar a ``cancel`` mientras Demucs no
    escriba nada. tqdm reescribe su barra con ``\\r`` (sin salto de línea), por
    eso no se lee línea a línea sino por bloques, partiendo en ``\\r`` y ``\\n``.

    Attributes
    ----------
    updates : queue.Queue[float]
        Porcentajes (0–100) en el orden en que aparecen, sin repetir valores
        consecutivos iguales.
    tail : collections.deque[str]
        Últimas líneas no vacías de la salida (para mensajes de error).
    """

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(name="demucs-output-reader", daemon=True)
        self._stream = stream
        # Decodificador incremental: un carácter UTF-8 multibyte (p. ej. "█")
        # puede quedar partido entre dos bloques.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._last_percent: float | None = None
        self.updates: queue.Queue[float] = queue.Queue()
        self.tail: deque[str] = deque(maxlen=_TAIL_LINES)

    def _emit(self, text: str) -> None:
        percent = _parse_percent(text)
        if percent is not None and percent != self._last_percent:
            self._last_percent = percent
            self.updates.put(percent)

    def _feed(self, text: str) -> None:
        self._pending += text
        pieces = re.split(r"[\r\n]", self._pending)
        self._pending = pieces.pop()  # fragmento aún incompleto
        for piece in pieces:
            self._emit(piece)
            if piece.strip():
                self.tail.append(piece.rstrip())
                logger.debug("demucs: %s", piece.rstrip())
        # La barra de tqdm no termina en salto de línea hasta que se redibuja:
        # se analiza también el fragmento incompleto para no ir un paso atrasado.
        self._emit(self._pending)

    def run(self) -> None:
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
        """Últimas líneas de la salida como un solo texto."""
        return "\n".join(self.tail)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Mata el subproceso de Demucs y espera a que termine (sin dejar zombis)."""
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - muy improbable tras kill()
        logger.warning("Demucs no terminó tras kill(); se abandona el proceso %d", proc.pid)


def separate_bass(
    path: str | Path,
    model: str = "htdemucs",
    cache_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Devuelve la ruta al stem de bajo aislado (WAV), usando la caché si existe.

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
        Callback ``progress(fraccion, mensaje)`` con fracción en [0, 1] y
        mensajes del tipo ``"Demucs: 45 %"``.
    cancel : threading.Event | None, optional
        Evento de cancelación; se revisa cada 0.1 s mientras Demucs trabaja.
    env : Mapping[str, str] | None, optional
        Variables de entorno del subproceso (``None`` hereda las del proceso
        actual). Útil en pruebas para inyectar un paquete ``demucs`` falso vía
        ``PYTHONPATH``.

    Returns
    -------
    Path
        ``<cache_dir>/<clave>_bass.wav`` (ver :func:`cached_stem_path`).

    Raises
    ------
    SeparationError
        Si el archivo no existe, si Demucs no está instalado (mensaje con
        instrucciones de instalación), si termina con código de salida ≠ 0
        (mensaje con el final de su salida) o si no genera ``bass.wav``.
    SeparationCancelledError
        Si ``cancel`` se activa (también es :class:`src.config.CancelledError`).

    Notes
    -----
    Demucs escribe en una carpeta temporal *dentro* de la caché y al terminar
    el stem se mueve con :func:`os.replace` (atómico en el mismo sistema de
    archivos): una separación cancelada o fallida nunca deja en la caché un
    archivo a medias que después se tomaría por bueno.

    Con modelos que son "bolsas" de varias redes (p. ej. ``htdemucs_ft``)
    tqdm recorre 0→100 % una vez por red, así que el progreso puede reiniciarse.

    Examples
    --------
    >>> stem = separate_bass("data/mezcla.mp3",
    ...                      progress=lambda f, m: print(m))  # doctest: +SKIP
    Demucs: iniciando (modelo htdemucs)...
    Demucs: 3 %
    ...
    Demucs: separación terminada
    """
    path = Path(path)
    if not path.is_file():
        raise SeparationError(f"No existe el archivo de audio a separar: {path}")

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    target = cached_stem_path(path, model, cache_dir)
    if target.is_file():
        logger.info("Separación: usando caché %s (no se ejecuta Demucs)", target)
        report(1.0, "Demucs: usando caché")
        return target

    if not demucs_available():
        raise SeparationError(_INSTALL_HELP)
    if cancel is not None and cancel.is_set():
        raise SeparationCancelledError("Separación cancelada por el usuario antes de iniciar Demucs.")

    target.parent.mkdir(parents=True, exist_ok=True)
    sub_env = dict(os.environ if env is None else env)
    sub_env.setdefault("PYTHONUNBUFFERED", "1")  # que los mensajes lleguen en cuanto se escriben
    # En Windows, evita que aparezca una ventana de consola al lanzar desde la GUI.
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

    with tempfile.TemporaryDirectory(
        prefix=".demucs_tmp_", dir=target.parent, ignore_cleanup_errors=True
    ) as tmp:
        out_dir = Path(tmp)
        cmd = _demucs_command(path, model, out_dir)
        logger.info("Separando el bajo con Demucs (modelo %s): %s", model, path.name)
        logger.debug("Comando: %s", " ".join(cmd))
        report(0.0, f"Demucs: iniciando (modelo {model})...")

        # stdout se redirige al mismo canal que stderr: tqdm escribe el progreso
        # en stderr y Demucs algunos mensajes (y errores) en stdout; así se lee
        # todo con un solo hilo y el final de la salida sirve para diagnosticar.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=sub_env,
            creationflags=creationflags,
        )
        assert proc.stdout is not None
        reader = _OutputReader(proc.stdout)
        reader.start()

        def forward(percent: float) -> None:
            logger.debug("Demucs: %.0f %%", percent)
            report(percent / 100.0, f"Demucs: {percent:.0f} %")

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

        while not reader.updates.empty():  # porcentajes leídos al final
            forward(reader.updates.get_nowait())

        if proc.returncode != 0:
            raise SeparationError(
                f"Demucs falló (código de salida {proc.returncode}).\n"
                f"Últimas líneas de su salida:\n{reader.tail_text() or '(sin salida)'}"
            )

        stem = _expected_stem(out_dir, model, path)
        if not stem.is_file():
            # Respaldo por si una versión de Demucs organiza las carpetas distinto.
            found = sorted(out_dir.rglob("bass.wav"))
            if len(found) != 1:
                raise SeparationError(
                    f"Demucs terminó sin generar el stem de bajo esperado ({stem.relative_to(out_dir)}).\n"
                    f"Últimas líneas de su salida:\n{reader.tail_text() or '(sin salida)'}"
                )
            stem = found[0]
        os.replace(stem, target)

    logger.info("Stem de bajo guardado en caché: %s", target)
    report(1.0, "Demucs: separación terminada")
    return target
