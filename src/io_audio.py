"""Etapa 1 — Carga de audio (y escritura de WAV/MP3).

Papel en el pipeline
--------------------
Es la puerta de entrada: convierte el MP3 del usuario en un arreglo numpy
``float32`` a la frecuencia de muestreo de trabajo (22 050 Hz, mono), que es lo
que consumen todas las etapas siguientes. La separación con Demucs (etapa 2)
la usa también para obtener la mezcla en ESTÉREO a 44 100 Hz
(``load_audio(path, sr=44100, channels=2)``), que es lo que espera la red.

¿Por qué ffmpeg directamente?
-----------------------------
librosa decodificaba MP3 a través de *audioread*, que a su vez llamaba a
ffmpeg. librosa ≥ 1.0 eliminó ese backend, así que aquí se invoca ffmpeg por
subproceso, pidiéndole que haga todo el trabajo de una vez::

    ffmpeg -i entrada.mp3 -f f32le -ac 1 -ar 22050 -
           │              │       │     │        └─ escribe a stdout
           │              │       │     └─ remuestrea a 22 050 Hz
           │              │       └─ mezcla a mono (promedio de canales); -ac 2 = estéreo
           │              └─ muestras float32 little-endian "crudas"
           └─ cualquier formato que ffmpeg entienda (MP3, WAV, FLAC, OGG, M4A...)

y los bytes de stdout se reinterpretan como ``np.float32``.

¿De dónde sale ffmpeg? (sin instalación manual)
-----------------------------------------------
:func:`find_ffmpeg` lo busca, en este orden:

1. La variable de entorno ``SMARTUNER_FFMPEG`` (ruta explícita al ejecutable).
2. El ``PATH`` del proceso (ffmpeg instalado con winget/brew/apt...).
3. Solo en Windows: el ``PATH`` de usuario y de sistema guardado en el
   REGISTRO y las carpetas habituales de instalación (WinGet, ``C:\\ffmpeg``,
   ``Program Files``, Chocolatey). Una terminal (o la GUI lanzada desde ella)
   abierta ANTES de ``winget install Gyan.FFmpeg`` conserva el PATH viejo y no
   ve el ffmpeg recién instalado; el registro y esas carpetas sí.
4. El paquete pip **imageio-ffmpeg** (dependencia de ``requirements.txt``),
   que trae un binario de ffmpeg para Windows, macOS y Linux: con
   ``pip install -r requirements.txt`` basta, sin tocar el PATH.

Respaldo sin ffmpeg: soundfile
------------------------------
Si aun así no hay ffmpeg, se decodifica con **soundfile** (libsndfile ≥ 1.1
lee MP3, además de WAV/FLAC/OGG): se lee el archivo, se mezcla a mono
(promedio de canales) y se remuestrea con ``librosa.resample`` (``soxr_hq``,
un remuestreador de alta calidad). Solo si ambos caminos fallan se lanza
:class:`FFmpegNotFoundError` con instrucciones claras, que la GUI muestra en un
cuadro de diálogo. La escritura de MP3 (:func:`save_mp3`,
:func:`convert_to_mp3`) tiene el mismo respaldo.
"""

from __future__ import annotations

import logging
import ntpath
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

#: Variable de entorno opcional con la ruta explícita al ejecutable de ffmpeg.
FFMPEG_ENV_VAR = "SMARTUNER_FFMPEG"

_INSTALL_HELP = (
    "No se encontró ffmpeg, necesario para leer este archivo (y para escribir MP3).\n\n"
    "La forma más sencilla, sin instalar nada en el sistema ni tocar el PATH:\n"
    "  pip install imageio-ffmpeg        (trae un ffmpeg listo para usar)\n\n"
    "O instálalo en el sistema y asegúrate de que esté en el PATH:\n"
    "  • Windows:  winget install Gyan.FFmpeg   (o: choco install ffmpeg)\n"
    "  • macOS:    brew install ffmpeg\n"
    "  • Linux:    sudo apt install ffmpeg      (o el gestor de tu distribución)\n\n"
    "Si acabas de instalarlo y sigue sin encontrarse, cierra y vuelve a abrir la terminal "
    "(y la aplicación): una terminal abierta antes de la instalación no ve el PATH actualizado.\n\n"
    f"También puedes indicar la ruta con la variable de entorno {FFMPEG_ENV_VAR}."
)

#: Clave del registro de Windows con el PATH de SISTEMA (el de usuario está en HKCU\Environment).
_MACHINE_ENV_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"

#: Fracción máxima de muestras no finitas (NaN/Inf) que se reparan con 0; por
#: encima, el archivo se considera corrupto.
MAX_NONFINITE_FRACTION: float = 0.01

#: Último ffmpeg anunciado en el log (para decir UNA vez cuál se usa, y otra
#: vez solo si cambia).
_announced_ffmpeg: str | None = None


class FFmpegNotFoundError(RuntimeError):
    """No hay ffmpeg utilizable y soundfile tampoco pudo leer/escribir el archivo.

    El mensaje incluye instrucciones de instalación (``pip install
    imageio-ffmpeg`` primero; después winget, brew o apt).
    """


class AudioLoadError(RuntimeError):
    """El archivo no existe, está vacío o no se pudo decodificar."""


# ---------------------------------------------------------------------------
# Localización de ffmpeg
# ---------------------------------------------------------------------------


def _imageio_ffmpeg_exe() -> str | None:
    """Ruta del ffmpeg que trae el paquete pip ``imageio-ffmpeg`` (o ``None``).

    Returns
    -------
    str | None
        Ruta al binario, o ``None`` si el paquete no está instalado o no
        encontró un binario válido.

    Notes
    -----
    El import es perezoso (solo se paga si no hay ffmpeg en el sistema) y
    CUALQUIER excepción se interpreta como "no disponible": un paquete roto
    no debe impedir el respaldo con soundfile.
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - import perezoso a propósito

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - ImportError, RuntimeError, OSError...
        logger.debug("imageio-ffmpeg no disponible: %s", exc)
        return None
    return str(exe) if exe and Path(exe).is_file() else None


def _announce(path: str, source: str) -> None:
    """Anota en el log (INFO) qué ffmpeg se usa, solo la primera vez o si cambia."""
    global _announced_ffmpeg
    if path != _announced_ffmpeg:
        _announced_ffmpeg = path
        logger.info("Se usará ffmpeg (%s): %s", source, path)


def _is_executable(path: Path) -> bool:
    """True si ``path`` es un archivo ejecutable (en Windows basta con que exista)."""
    try:
        return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))
    except OSError:  # ruta inválida, unidad desconectada, sin permisos...
        return False


def _windows_registry_path_dirs() -> list[str]:
    """Carpetas del PATH de usuario y de sistema guardadas en el registro de Windows.

    Returns
    -------
    list[str]
        Carpetas en el orden de Windows (primero las del sistema, luego las
        del usuario), con las variables ``%VAR%`` ya expandidas. Lista vacía
        fuera de Windows o si el registro no se puede leer.

    Notes
    -----
    El PATH de un proceso se copia del de su padre al arrancar: una terminal
    abierta antes de ``winget install`` no ve el cambio, pero el registro
    (``HKCU\\Environment`` y ``HKLM\\...\\Session Manager\\Environment``,
    valor ``Path``) siempre tiene el PATH vigente. Suele ser ``REG_EXPAND_SZ``
    con variables como ``%USERPROFILE%``, por eso se expande con
    :func:`ntpath.expandvars` (que entiende la sintaxis ``%VAR%``).
    """
    try:
        import winreg  # noqa: PLC0415 - solo existe en Windows
    except ImportError:
        return []
    dirs: list[str] = []
    for root_name, subkey in (("HKEY_LOCAL_MACHINE", _MACHINE_ENV_KEY), ("HKEY_CURRENT_USER", "Environment")):
        try:
            with winreg.OpenKey(getattr(winreg, root_name), subkey) as key:
                value, _kind = winreg.QueryValueEx(key, "Path")
        except (OSError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("No se pudo leer el PATH de %s\\%s del registro: %s", root_name, subkey, exc)
            continue
        expanded = ntpath.expandvars(str(value))
        dirs += [d.strip().strip('"') for d in expanded.split(";") if d.strip()]
    return dirs


def _windows_known_locations() -> list[Path]:
    """Ubicaciones habituales de ffmpeg.exe en Windows, en orden de preferencia.

    Returns
    -------
    list[Path]
        Candidatos (pueden no existir): enlace de WinGet
        (``%LOCALAPPDATA%\\Microsoft\\WinGet\\Links\\ffmpeg.exe``), paquetes
        de WinGet ``Gyan.FFmpeg*`` (el más reciente primero),
        ``C:\\ffmpeg\\bin``, ``%ProgramFiles%\\ffmpeg\\bin`` y Chocolatey.
    """
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        winget = Path(local) / "Microsoft" / "WinGet"
        candidates.append(winget / "Links" / "ffmpeg.exe")
        try:
            found = list((winget / "Packages").glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
            # Varias versiones instaladas → la más reciente (fecha de modificación).
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates += found
        except OSError as exc:
            logger.debug("No se pudieron recorrer los paquetes de WinGet: %s", exc)
    candidates.append(Path("C:\\ffmpeg\\bin\\ffmpeg.exe"))
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "ffmpeg" / "bin" / "ffmpeg.exe")
    program_data = os.environ.get("ProgramData") or "C:\\ProgramData"
    candidates.append(Path(program_data) / "chocolatey" / "bin" / "ffmpeg.exe")
    return candidates


def _find_ffmpeg_windows() -> tuple[str, str] | None:
    """Busca ffmpeg.exe en el PATH del registro y en las ubicaciones habituales de Windows.

    Returns
    -------
    tuple[str, str] | None
        ``(ruta, origen)`` o ``None`` si no se encontró.
    """
    for folder in _windows_registry_path_dirs():
        candidate = Path(folder) / "ffmpeg.exe"
        if _is_executable(candidate):
            return str(candidate), "PATH del registro de Windows: instalado después de abrir la terminal"
    for candidate in _windows_known_locations():
        if _is_executable(candidate):
            return str(candidate), "carpeta de instalación habitual de Windows"
    return None


def find_ffmpeg() -> str | None:
    """Localiza el ejecutable de ffmpeg.

    Orden de búsqueda (ver el encabezado del módulo): variable de entorno
    ``SMARTUNER_FFMPEG`` → ``PATH`` del proceso → (solo Windows) PATH del
    registro y carpetas habituales (WinGet, ``C:\\ffmpeg``, Program Files,
    Chocolatey) → binario del paquete ``imageio-ffmpeg``. La variable solo se
    usa si apunta a un archivo EJECUTABLE; si no, se avisa en el log y se
    sigue buscando. La primera vez (y cada vez que cambie) se anota en el log
    cuál se usa.

    Returns
    -------
    str | None
        Ruta al ejecutable, o ``None`` si no está disponible de ninguna forma
        (en ese caso :func:`load_audio` recurre a soundfile).

    Notes
    -----
    Todas las búsquedas específicas de Windows están protegidas con
    ``try/except``: un registro ilegible o una carpeta sin permisos solo hacen
    que se pase al siguiente candidato.

    Examples
    --------
    >>> path = find_ffmpeg()  # doctest: +SKIP
    >>> path                   # doctest: +SKIP
    '/usr/bin/ffmpeg'
    """
    explicit = os.environ.get(FFMPEG_ENV_VAR)
    if explicit:
        if _is_executable(Path(explicit)):
            _announce(explicit, f"variable de entorno {FFMPEG_ENV_VAR}")
            return explicit
        logger.warning("%s=%s no es un archivo ejecutable; se busca ffmpeg en el PATH", FFMPEG_ENV_VAR, explicit)
    on_path = shutil.which("ffmpeg")
    if on_path:
        _announce(on_path, "PATH del sistema")
        return on_path
    if sys.platform == "win32":
        try:
            found = _find_ffmpeg_windows()
        except Exception as exc:  # noqa: BLE001 - nunca debe impedir los demás respaldos
            logger.debug("Búsqueda de ffmpeg específica de Windows fallida: %s", exc)
            found = None
        if found is not None:
            _announce(*found)
            return found[0]
    bundled = _imageio_ffmpeg_exe()
    if bundled:
        _announce(bundled, "paquete pip imageio-ffmpeg")
        return bundled
    return None


def require_ffmpeg() -> str:
    """Igual que :func:`find_ffmpeg` pero falla con un mensaje explicativo.

    Returns
    -------
    str
        Ruta al ejecutable de ffmpeg.

    Raises
    ------
    FFmpegNotFoundError
        Si ffmpeg no está disponible (mensaje en español con instrucciones).
    """
    path = find_ffmpeg()
    if path is None:
        raise FFmpegNotFoundError(_INSTALL_HELP)
    return path


def _run_ffmpeg(cmd: list[str], data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    """Ejecuta ffmpeg y traduce los fallos al LANZARLO (no los de decodificación).

    Parameters
    ----------
    cmd : list[str]
        Comando completo (el primer elemento es la ruta de ffmpeg).
    data : bytes | None, optional
        Datos para la entrada estándar.

    Returns
    -------
    subprocess.CompletedProcess[bytes]
        Resultado con ``returncode``, ``stdout`` y ``stderr``.

    Raises
    ------
    FFmpegNotFoundError
        Si el sistema no pudo ejecutar el programa (sin permiso de ejecución,
        borrado entre la búsqueda y la llamada, formato no ejecutable...).
    """
    # En Windows, evita que se abra una ventana de consola al llamar desde la GUI.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        return subprocess.run(cmd, input=data, capture_output=True, check=False, creationflags=flags)
    except OSError as exc:
        raise FFmpegNotFoundError(f"No se pudo ejecutar ffmpeg ({cmd[0]}): {exc}.\n\n{_INSTALL_HELP}") from exc


def _first_line(exc: BaseException) -> str:
    """Primera línea del mensaje de una excepción (sin las instrucciones de instalación)."""
    return str(exc).split("\n", 1)[0]


# ---------------------------------------------------------------------------
# Decodificación
# ---------------------------------------------------------------------------


def _decode_with_ffmpeg(ffmpeg: str, path: Path, sr: int, channels: int) -> np.ndarray:
    """Decodifica con ffmpeg a float32 ``(n,)`` (mono) o ``(2, n)`` (estéreo).

    Raises
    ------
    FFmpegNotFoundError
        Si ffmpeg no se puede ejecutar.
    AudioLoadError
        Si ffmpeg no pudo decodificar el archivo.
    """
    cmd = [
        ffmpeg, "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",  # float32 crudo
        "-ac", str(int(channels)),              # 1 = mezcla a mono; 2 = estéreo
        "-ar", str(int(sr)),                    # remuestreo
        "-",                                    # salida por stdout
    ]
    logger.info("Decodificando %s con ffmpeg (%s, %d Hz)", path.name, "mono" if channels == 1 else "estéreo", sr)
    proc = _run_ffmpeg(cmd)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLoadError(f"ffmpeg no pudo decodificar {path.name}: {detail}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if channels == 2:
        # f32le estéreo viene ENTRELAZADO: L0 R0 L1 R1 ... → filas = canales.
        y = np.ascontiguousarray(y[: y.size - y.size % 2].reshape(-1, 2).T)
    return y


def _decode_with_soundfile(path: Path, sr: int, channels: int) -> np.ndarray:
    """Respaldo sin ffmpeg: soundfile (libsndfile) + mezcla de canales + ``librosa.resample``.

    Parameters
    ----------
    path : Path
        Archivo de audio (MP3 requiere libsndfile ≥ 1.1; también WAV/FLAC/OGG).
    sr : int
        Frecuencia de salida en Hz.
    channels : int
        1 (promedio de canales) o 2 (estéreo; un archivo mono se duplica y de
        uno multicanal se toman los dos primeros canales, izquierdo y derecho
        frontales en la disposición estándar de WAV).

    Returns
    -------
    np.ndarray
        float32 ``(n,)`` o ``(2, n)``.

    Raises
    ------
    FFmpegNotFoundError
        Si soundfile no puede leer el archivo (formato no soportado por
        libsndfile, como M4A/AAC, o archivo dañado): sin ffmpeg no hay otra vía.
    """
    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, canales)
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        mp3_note = ""
        if path.suffix.lower() == ".mp3" and "MP3" not in sf.available_formats():
            mp3_note = f" (esta versión de libsndfile, {sf.__libsndfile_version__}, no lee MP3: se necesita ≥ 1.1)"
        raise FFmpegNotFoundError(
            f"No se pudo decodificar {path.name} sin ffmpeg: soundfile respondió «{exc}»{mp3_note}.\n\n"
            f"{_INSTALL_HELP}") from exc
    logger.info("Decodificando %s con soundfile/libsndfile %s (sin ffmpeg): %d canal(es) a %d Hz",
                path.name, sf.__libsndfile_version__, data.shape[1], file_sr)
    if channels == 1:
        y = data.mean(axis=1)                        # mezcla a mono: promedio de canales (como ffmpeg -ac 1)
    elif data.shape[1] == 1:
        y = np.repeat(data.T, 2, axis=0)             # mono → estéreo duplicando el canal
    else:
        y = data[:, :2].T                            # (2, n): izquierdo y derecho
    if int(file_sr) != int(sr) and y.shape[-1] > 0:
        import librosa  # noqa: PLC0415 - import diferido: librosa tarda en cargarse

        # soxr_hq: remuestreador de banda limitada de alta calidad (el mismo
        # criterio que usa ffmpeg con -ar, sin aliasing audible).
        y = librosa.resample(y, orig_sr=int(file_sr), target_sr=int(sr), res_type="soxr_hq", axis=-1)
    return np.ascontiguousarray(y, dtype=np.float32)


def load_audio(path: str | Path, sr: int = 22050, channels: int = 1) -> tuple[np.ndarray, int]:
    """Decodifica un archivo de audio a ``float32`` en [-1, 1].

    Usa ffmpeg (variable ``SMARTUNER_FFMPEG``, PATH o imageio-ffmpeg; ver
    :func:`find_ffmpeg`) y, si no hay ninguno o no se puede ejecutar,
    soundfile + ``librosa.resample``.

    Parameters
    ----------
    path : str | Path
        Archivo de entrada (MP3, WAV, FLAC, OGG... cualquier formato de ffmpeg;
        sin ffmpeg, los que lea libsndfile).
    sr : int, optional
        Frecuencia de muestreo de salida en Hz (por defecto 22 050).
    channels : int, optional
        1 = mono (promedio de los canales; por defecto) o 2 = estéreo (lo que
        necesita Demucs). Un archivo mono pedido en estéreo se duplica.

    Returns
    -------
    y : np.ndarray
        Señal ``float32``: forma ``(n_muestras,)`` si ``channels == 1`` o
        ``(2, n_muestras)`` si ``channels == 2``.
    sr : int
        Frecuencia de muestreo de ``y`` en Hz.

    Raises
    ------
    ValueError
        Si ``channels`` no es 1 ni 2.
    FFmpegNotFoundError
        Si no hay ffmpeg utilizable y soundfile tampoco pudo leer el archivo.
    AudioLoadError
        Si el archivo no existe, no se puede decodificar o más del 1 % de sus
        muestras no son finitas (NaN/Inf).

    Notes
    -----
    Un WAV en coma flotante puede contener muestras NaN o ±Inf (archivo
    dañado o mal exportado); librosa las rechaza más adelante con un error
    críptico. Si son pocas (≤ :data:`MAX_NONFINITE_FRACTION`) se reemplazan
    por 0 (silencio de una muestra) con un aviso en el log.

    Examples
    --------
    >>> y, sr = load_audio("data/synthetic/linea_simple.mp3")  # doctest: +SKIP
    >>> sr, y.dtype, y.ndim                                     # doctest: +SKIP
    (22050, dtype('float32'), 1)
    >>> y2, _ = load_audio("data/mezcla.mp3", sr=44100, channels=2)  # doctest: +SKIP
    >>> y2.shape[0]                                                   # doctest: +SKIP
    2
    """
    if channels not in (1, 2):
        raise ValueError(f"channels debe ser 1 (mono) o 2 (estéreo); se recibió {channels!r}.")
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"No existe el archivo de audio: {path}")
    ffmpeg = find_ffmpeg()
    if ffmpeg is not None:
        try:
            y = _decode_with_ffmpeg(ffmpeg, path, int(sr), channels)
        except FFmpegNotFoundError as exc:
            # ffmpeg existe pero el sistema no puede ejecutarlo (permisos,
            # antivirus, binario de otra arquitectura...): se intenta soundfile.
            logger.warning("%s Se intenta decodificar con soundfile.", _first_line(exc))
            try:
                y = _decode_with_soundfile(path, int(sr), channels)
            except FFmpegNotFoundError as sf_exc:
                raise FFmpegNotFoundError(f"{_first_line(exc)}\n{sf_exc}") from sf_exc
    else:
        logger.info("No hay ffmpeg disponible: se usa soundfile como respaldo")
        y = _decode_with_soundfile(path, int(sr), channels)
    if y.shape[-1] == 0:
        raise AudioLoadError(f"El archivo {path.name} no contiene audio decodificable.")
    bad = ~np.isfinite(y)
    if bad.any():
        n_bad = int(bad.sum())
        if n_bad > MAX_NONFINITE_FRACTION * y.size:
            raise AudioLoadError(f"El archivo {path.name} está dañado: {n_bad} de {y.size} muestras no son "
                                 "números finitos (NaN/Inf).")
        logger.warning("%s contiene %d muestras no finitas (NaN/Inf); se reemplazan por 0", path.name, n_bad)
        y[bad] = 0.0
    n = y.shape[-1]
    logger.info("Audio cargado: %.2f s, %d muestras%s", n / sr, n, "" if channels == 1 else " por canal (estéreo)")
    return y, int(sr)


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> Path:
    """Escribe ``y`` como WAV PCM de 16 bits.

    Parameters
    ----------
    path : str | Path
        Archivo de salida; se crean las carpetas intermedias.
    y : np.ndarray
        Señal mono en [-1, 1] (los valores fuera de rango se recortan).
    sr : int
        Frecuencia de muestreo en Hz.

    Returns
    -------
    Path
        Ruta escrita.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0), int(sr), subtype="PCM_16")
    return path


def _mp3_compression_level(bitrate: str, sr: int) -> float:
    """Traduce una tasa de bits de ffmpeg (``"192k"``) al ``compression_level`` de libsndfile.

    libsndfile no acepta una tasa de bits: recibe un nivel en [0, 1) que
    interpola linealmente entre la tasa máxima (0) y la mínima de la versión
    de MPEG que corresponde a ``sr``: MPEG-1 (≥ 32 kHz) 32–320 kbps, MPEG-2
    (16–24 kHz) 8–160 kbps y MPEG-2.5 (< 16 kHz) 8–64 kbps. Una tasa por
    encima del máximo da la máxima calidad.

    Parameters
    ----------
    bitrate : str
        Tasa en el formato de ffmpeg (``"192k"``, ``"128k"`` o en bit/s).
    sr : int
        Frecuencia de muestreo en Hz.

    Returns
    -------
    float
        Nivel de compresión en [0, 0.99].

    Examples
    --------
    >>> _mp3_compression_level("192k", 22050)   # por encima de 160 kbps → máxima calidad
    0.0
    >>> round(_mp3_compression_level("176k", 44100), 2)
    0.5
    """
    text = str(bitrate).strip().lower()
    kbps = float(text[:-1]) if text.endswith("k") else float(text) / 1000.0
    lo, hi = (32.0, 320.0) if sr >= 32000 else (8.0, 160.0) if sr >= 16000 else (8.0, 64.0)
    return float(min(max((hi - min(kbps, hi)) / (hi - lo), 0.0), 0.99))


def _write_mp3_soundfile(path: Path, y: np.ndarray, sr: int, bitrate: str, reason: str) -> Path:
    """Escribe un MP3 con soundfile/libsndfile (respaldo cuando ffmpeg no está o no funciona).

    Raises
    ------
    FFmpegNotFoundError
        Si libsndfile no sabe escribir MP3 (versión < 1.1) o falla.
    """
    if "MP3" not in sf.available_formats():
        raise FFmpegNotFoundError(f"{reason} y libsndfile {sf.__libsndfile_version__} no sabe escribir MP3 "
                                  f"(se necesita ≥ 1.1).\n\n{_INSTALL_HELP}")
    try:
        sf.write(str(path), y, int(sr), format="MP3", bitrate_mode="CONSTANT",
                 compression_level=_mp3_compression_level(bitrate, int(sr)))
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        raise FFmpegNotFoundError(f"{reason} y soundfile no pudo escribir {path.name}: {exc}.\n\n"
                                  f"{_INSTALL_HELP}") from exc
    logger.info("MP3 escrito con soundfile (sin ffmpeg): %s", path)
    return path


def save_mp3(path: str | Path, y: np.ndarray, sr: int, bitrate: str = "192k") -> Path:
    """Codifica ``y`` a MP3 con ffmpeg (libmp3lame), enviándole float32 por stdin.

    Sin ffmpeg utilizable se escribe con soundfile (libsndfile ≥ 1.1, que
    también usa LAME); la tasa de bits se aproxima con
    :func:`_mp3_compression_level`.

    Parameters
    ----------
    path : str | Path
        Archivo MP3 de salida; se crean las carpetas intermedias.
    y : np.ndarray
        Señal mono en [-1, 1].
    sr : int
        Frecuencia de muestreo de ``y`` en Hz.
    bitrate : str, optional
        Tasa de bits de ffmpeg (``"192k"`` por defecto).

    Returns
    -------
    Path
        Ruta escrita.

    Raises
    ------
    FFmpegNotFoundError
        Si no hay ffmpeg utilizable y soundfile tampoco puede escribir MP3.
    RuntimeError
        Si la codificación con ffmpeg falla.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return _write_mp3_soundfile(path, clipped, sr, bitrate, "No hay ffmpeg")
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "-",
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        str(path),
    ]
    try:
        proc = _run_ffmpeg(cmd, data=clipped.tobytes())
    except FFmpegNotFoundError as exc:
        logger.warning("%s Se escribe el MP3 con soundfile.", _first_line(exc))
        return _write_mp3_soundfile(path, clipped, sr, bitrate, "ffmpeg no se pudo ejecutar")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo crear {path.name}: {proc.stderr.decode('utf-8', errors='replace')}")
    logger.info("MP3 escrito: %s", path)
    return path


def convert_to_mp3(src: str | Path, dst: str | Path, bitrate: str = "192k") -> Path:
    """Convierte un archivo de audio existente (p. ej. WAV) a MP3 con ffmpeg.

    Sin ffmpeg utilizable se lee ``src`` con soundfile y se escribe el MP3
    con soundfile (conserva canales y frecuencia de muestreo).

    Parameters
    ----------
    src : str | Path
        Archivo de entrada.
    dst : str | Path
        Archivo MP3 de salida.
    bitrate : str, optional
        Tasa de bits (``"192k"`` por defecto).

    Returns
    -------
    Path
        Ruta del MP3 escrito.

    Raises
    ------
    AudioLoadError
        Si ``src`` no existe.
    FFmpegNotFoundError
        Si no hay ffmpeg utilizable y soundfile no puede leer ``src`` o
        escribir MP3.
    RuntimeError
        Si la conversión con ffmpeg falla.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise AudioLoadError(f"No existe el archivo de audio: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    def fallback(reason: str) -> Path:
        """Conversión con soundfile: lee ``src`` tal cual y lo escribe como MP3."""
        try:
            data, sr = sf.read(str(src), dtype="float32")
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            raise FFmpegNotFoundError(f"{reason} y soundfile no pudo leer {src.name}: {exc}.\n\n"
                                      f"{_INSTALL_HELP}") from exc
        return _write_mp3_soundfile(dst, np.clip(data, -1.0, 1.0), int(sr), bitrate, reason)

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return fallback("No hay ffmpeg")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    try:
        proc = _run_ffmpeg(cmd)
    except FFmpegNotFoundError as exc:
        logger.warning("%s Se convierte con soundfile.", _first_line(exc))
        return fallback("ffmpeg no se pudo ejecutar")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg no pudo convertir {src.name}: {proc.stderr.decode('utf-8', errors='replace')}")
    logger.info("Convertido a MP3: %s", dst)
    return dst
