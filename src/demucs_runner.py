"""Subproceso de Demucs: separa el bajo de una mezcla en WAV (etapa 2, método «demucs»).

Papel en el pipeline
--------------------
:func:`src.separation.separate_bass` NO importa PyTorch en el proceso de la
aplicación: decodifica la mezcla (con :func:`src.io_audio.load_audio`, que
funciona con o sin ffmpeg en el PATH) a un WAV estéreo de 44 100 Hz en una
carpeta temporal y lanza este módulo en un proceso aparte::

    python -m src.demucs_runner --input mezcla.wav --output bajo.wav \\
                                --model htdemucs --device cpu

Así la separación se puede *cancelar* matando el proceso, la GUI no se
congela y PyTorch (cientos de MB) no se carga en la aplicación.

¿Por qué no ``python -m demucs`` (la CLI oficial)?
--------------------------------------------------
La CLI de Demucs lee el MP3 con ffmpeg *del PATH* (falla en un Windows sin
ffmpeg instalado) y guarda los stems con ``torchaudio.save``, que en
torchaudio ≥ 2.9 exige el paquete ``torchcodec``. Este módulo solo usa la
API estable de Demucs (``demucs.pretrained.get_model`` y
``demucs.apply.apply_model``) y lee/escribe WAV con soundfile: no depende de
ffmpeg, torchaudio ni torchcodec.

Algoritmo (igual que ``demucs.separate``)
-----------------------------------------
1. Leer el WAV ``(n, canales)`` → tensor ``mix`` de forma ``(canales, n)``.
2. Normalizar con la referencia mono ``ref = mix.mean(0)``:
   ``mix ← (mix − media(ref)) / (desv(ref) + 1e-8)``. La red se entrenó con
   audio de varianza unitaria, así que el volumen de la grabación no influye.
3. ``out = apply_model(model, mix[None], shifts=1, split=True, overlap=0.25)``:
   la pista se trocea en segmentos de ~8 s con 25 % de solapamiento (memoria
   acotada), se desplaza un instante aleatorio (``shifts``: mejora la
   equivarianza temporal) y se recompone con ventanas triangulares.
   ``out`` tiene forma ``(1, fuentes, canales, n)``.
4. Desnormalizar (``out · desv + media``) y quedarse con la fuente ``"bass"``.
5. Escribir el bajo como WAV PCM de 16 bits a la frecuencia del modelo.

Protocolo con el proceso padre
------------------------------
* **stdout**, líneas con prefijo fijo (UTF-8):

  - ``PROGRESO: 45% mensaje`` — avance global (0–100) en las fases propias
    (carga de PyTorch y del modelo, escritura).
  - ``SEPARANDO: n`` — empieza ``apply_model``; vendrán ``n`` barras tqdm
    (una por red del modelo: ``htdemucs`` = 1, ``htdemucs_ft`` = 4), que
    cubren el tramo :data:`PCT_SEPARATION_START`–:data:`PCT_SEPARATION_END`.
  - ``ERROR: mensaje`` — explicación en español antes de salir con código ≠ 0.

* **stderr**: barras tqdm de ``apply_model`` (``" 45%|████▌ | 9.0/20.0 [...]"``)
  y avisos de PyTorch/Demucs.
* **Código de salida**: :data:`EXIT_OK` (0) o uno de los ``EXIT_*``.

La opción ``--untrained`` (SOLO para pruebas sin Internet) construye un
``HTDemucs`` con pesos ALEATORIOS en lugar de descargar los entrenados: el
"bajo" resultante es ruido sin ningún valor musical, pero ejercita la misma
mecánica (tensores, ``apply_model``, escritura, progreso, cancelación) con
las versiones instaladas de torch/demucs.

Este módulo no importa torch ni demucs al cargarse (solo dentro de
:func:`main`), para que :mod:`src.separation` pueda leer sus constantes sin
coste.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

#: Prefijo de las líneas de avance global (stdout).
PROGRESS_PREFIX: str = "PROGRESO:"
#: Prefijo de la línea que anuncia ``apply_model`` y su número de barras tqdm.
SEPARATING_PREFIX: str = "SEPARANDO:"
#: Prefijo de la línea con el mensaje de error (stdout) antes de salir con código ≠ 0.
ERROR_PREFIX: str = "ERROR:"

#: Avance global (%) al empezar a importar PyTorch/Demucs.
PCT_IMPORT: int = 3
#: Avance global (%) al empezar a cargar (o descargar) el modelo.
PCT_MODEL: int = 6
#: Avance global (%) al empezar ``apply_model`` (0 % de la barra tqdm).
PCT_SEPARATION_START: int = 10
#: Avance global (%) al terminar ``apply_model`` (100 % de la última barra tqdm).
PCT_SEPARATION_END: int = 95
#: Avance global (%) al empezar a escribir el stem.
PCT_WRITE: int = 96

#: Fuentes del HTDemucs de 4 pistas (el orden de ``model.sources`` de ``htdemucs``).
HTDEMUCS_SOURCES: tuple[str, ...] = ("drums", "bass", "other", "vocals")

EXIT_OK: int = 0
#: Faltan PyTorch o Demucs (o están rotos).
EXIT_MISSING_DEPS: int = 3
#: No se pudo cargar/descargar el modelo (p. ej. sin Internet la primera vez).
EXIT_MODEL: int = 4
#: No se pudo leer el WAV de entrada.
EXIT_INPUT: int = 5
#: Falló ``apply_model`` (memoria insuficiente, dispositivo inválido...).
EXIT_SEPARATION: int = 6
#: No se pudo escribir el WAV de salida.
EXIT_OUTPUT: int = 7

_INTERNET_HELP = (
    "Se necesita conexión a Internet la primera vez (~80 MB de pesos para htdemucs): Demucs descarga el "
    "modelo al usarlo por primera vez y lo guarda en la caché de tu carpeta de usuario (Hugging Face o "
    "PyTorch); después ya no hace falta conexión. Comprueba también que el nombre del modelo exista "
    "(htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra...)."
)


class RunnerError(Exception):
    """Fallo controlado del runner: mensaje en español y código de salida.

    Parameters
    ----------
    message : str
        Explicación para el usuario (se emite como ``ERROR: ...``).
    code : int
        Código de salida del proceso (uno de los ``EXIT_*``).
    """

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = int(code)


def _emit(line: str) -> None:
    """Escribe una línea del protocolo en stdout y la vacía enseguida.

    No es un mensaje de log sino la *interfaz* con el proceso padre (que la
    analiza), por eso va a stdout sin formato de logging; ``flush`` hace que
    el padre la reciba en el momento y la barra de progreso no se quede atrás.
    """
    sys.stdout.write(line.rstrip("\n") + "\n")
    sys.stdout.flush()


def progress_line(percent: float, message: str) -> str:
    """Formatea una línea ``PROGRESO: N% mensaje``.

    Parameters
    ----------
    percent : float
        Avance global en % (0–100).
    message : str
        Descripción de la fase.

    Returns
    -------
    str
        La línea (sin salto de línea final).

    Examples
    --------
    >>> progress_line(45, "separando")
    'PROGRESO: 45% separando'
    """
    return f"{PROGRESS_PREFIX} {int(round(percent))}% {message}".rstrip()


def build_parser() -> argparse.ArgumentParser:
    """Parser de argumentos del runner.

    Returns
    -------
    argparse.ArgumentParser
        Acepta ``--input``, ``--output``, ``--model``, ``--device``,
        ``--shifts``, ``--overlap``, ``--seed`` y ``--untrained``.

    Examples
    --------
    >>> a = build_parser().parse_args(["--input", "in.wav", "--output", "bajo.wav"])
    >>> a.model, a.device, a.shifts, a.overlap, a.untrained
    ('htdemucs', 'cpu', 1, 0.25, False)
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.demucs_runner",
        description="Separa el bajo de una mezcla (WAV) con Demucs y lo escribe como WAV.")
    parser.add_argument("--input", required=True, help="WAV de entrada (la mezcla; idealmente 44 100 Hz estéreo)")
    parser.add_argument("--output", required=True, help="WAV de salida con el bajo aislado")
    parser.add_argument("--model", default="htdemucs", help="modelo de Demucs (por defecto htdemucs)")
    parser.add_argument("--device", default="cpu", help="dispositivo de PyTorch: cpu (por defecto), cuda, mps...")
    parser.add_argument("--shifts", type=int, default=1,
                        help="desplazamientos aleatorios promediados (1 = como demucs; más = mejor y más lento)")
    parser.add_argument("--overlap", type=float, default=0.25, help="solapamiento entre segmentos (0–1)")
    parser.add_argument("--seed", type=int, default=0, help="semilla de los desplazamientos (reproducibilidad)")
    parser.add_argument("--untrained", action="store_true",
                        help="SOLO PRUEBAS: modelo HTDemucs con pesos aleatorios (sin descargar nada); "
                             "el resultado no sirve musicalmente")
    return parser


def _import_backend() -> tuple[Any, Any, Any]:
    """Importa torch, ``demucs.apply`` y ``demucs.pretrained``.

    Raises
    ------
    RunnerError
        Con :data:`EXIT_MISSING_DEPS` si falta alguno (o está roto).
    """
    try:
        import torch  # noqa: PLC0415 - import pesado, solo en el subproceso
        from demucs import apply as demucs_apply  # noqa: PLC0415
        from demucs import pretrained as demucs_pretrained  # noqa: PLC0415
    except ImportError as exc:
        raise RunnerError(
            f"No se pudo importar PyTorch/Demucs ({exc}). Instálalo con: pip install demucs "
            "(instala también PyTorch para CPU).", EXIT_MISSING_DEPS) from exc
    return torch, demucs_apply, demucs_pretrained


def _load_model(name: str, untrained: bool, torch: Any, demucs_pretrained: Any) -> Any:
    """Carga el modelo entrenado (descargándolo la primera vez) o uno sin entrenar.

    Raises
    ------
    RunnerError
        Con :data:`EXIT_MODEL` si no se puede cargar (sin Internet la primera
        vez, nombre inexistente, caché dañada...).
    """
    if untrained:
        from demucs.htdemucs import HTDemucs  # noqa: PLC0415

        # Pesos aleatorios pero reproducibles (misma semilla → mismo "modelo").
        torch.manual_seed(0)
        model = HTDemucs(sources=list(HTDEMUCS_SOURCES))
    else:
        try:
            model = demucs_pretrained.get_model(name)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - descarga, YAML, nombre: todo es "no se pudo cargar"
            # SystemExit: algunas rutas de Demucs llaman a sys.exit() (utils.fatal).
            raise RunnerError(f"No se pudo cargar el modelo «{name}» de Demucs ({type(exc).__name__}: {exc}). "
                              f"{_INTERNET_HELP}", EXIT_MODEL) from exc
    model.eval()
    return model


def _read_mix(path: Path, model: Any, torch: Any) -> tuple[Any, int]:
    """Lee el WAV y lo adapta a los canales y la frecuencia del modelo.

    Returns
    -------
    mix : torch.Tensor
        Forma ``(canales_del_modelo, n)``, float32.
    sr : int
        Frecuencia de ``mix`` (la del modelo) en Hz.

    Raises
    ------
    RunnerError
        Con :data:`EXIT_INPUT` si el archivo no se puede leer o está vacío.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, canales)
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        raise RunnerError(f"No se pudo leer el WAV de entrada {path}: {exc}", EXIT_INPUT) from exc
    if data.shape[0] == 0:
        raise RunnerError(f"El WAV de entrada {path} está vacío.", EXIT_INPUT)
    mix = torch.from_numpy(np.ascontiguousarray(data.T))  # (canales, n)
    want = int(getattr(model, "audio_channels", 2))
    # Igual que demucs.audio.convert_audio_channels: modelo mono → promedio;
    # archivo mono → duplicar el canal; más canales de los necesarios → los primeros.
    if want == 1:
        mix = mix.mean(0, keepdim=True)
    elif mix.shape[0] == 1:
        mix = mix.expand(want, -1).contiguous()
    elif mix.shape[0] > want:
        mix = mix[:want]
    elif mix.shape[0] < want:
        raise RunnerError(f"El WAV tiene {mix.shape[0]} canales y el modelo necesita {want}.", EXIT_INPUT)
    model_sr = int(getattr(model, "samplerate", sr))
    if int(sr) != model_sr:
        import julius  # noqa: PLC0415 - dependencia de demucs

        mix = julius.resample_frac(mix, int(sr), model_sr)
    return mix, model_sr


def _n_passes(model: Any, shifts: int) -> int:
    """Número de barras tqdm que imprimirá ``apply_model`` (una por red y desplazamiento)."""
    n_models = len(getattr(model, "models", [model]))
    return max(1, n_models) * max(1, int(shifts))


def separate(args: argparse.Namespace) -> Path:
    """Ejecuta la separación completa descrita en el encabezado del módulo.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de :func:`build_parser`.

    Returns
    -------
    Path
        Ruta del WAV escrito con el bajo.

    Raises
    ------
    RunnerError
        Ante cualquier fallo previsto (con su código de salida).
    """
    _emit(progress_line(PCT_IMPORT, "cargando PyTorch y Demucs..."))
    torch, demucs_apply, demucs_pretrained = _import_backend()

    if args.untrained:
        _emit(progress_line(PCT_MODEL, "creando un modelo SIN ENTRENAR (pesos aleatorios, solo pruebas)..."))
    else:
        _emit(progress_line(PCT_MODEL, f"cargando el modelo {args.model} (la primera vez se descargan ~80 MB)..."))
    model = _load_model(args.model, args.untrained, torch, demucs_pretrained)
    sources = list(model.sources)
    if "bass" not in sources:
        raise RunnerError(f"El modelo «{args.model}» no separa el bajo (sus fuentes son: {', '.join(sources)}).",
                          EXIT_MODEL)

    mix, sr = _read_mix(Path(args.input), model, torch)

    # Normalización de demucs.separate: la red espera audio de media 0 y varianza 1.
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    random.seed(int(args.seed))  # apply_model elige el desplazamiento con `random`
    torch.manual_seed(int(args.seed))

    passes = _n_passes(model, args.shifts)
    _emit(progress_line(PCT_SEPARATION_START, f"separando {mix.shape[-1] / sr:.1f} s de audio..."))
    _emit(f"{SEPARATING_PREFIX} {passes}")
    try:
        with torch.no_grad():
            out = demucs_apply.apply_model(model, ((mix - mean) / std)[None], device=args.device,
                                           shifts=int(args.shifts), split=True, overlap=float(args.overlap),
                                           progress=True)[0]  # (fuentes, canales, n)
    except (RuntimeError, ValueError, AssertionError, MemoryError) as exc:
        raise RunnerError(f"Demucs falló al separar ({type(exc).__name__}: {exc}). Si es falta de memoria, "
                          "cierra otras aplicaciones o prueba con una canción más corta.", EXIT_SEPARATION) from exc
    out = out * std + mean
    bass = out[sources.index("bass")].cpu().numpy().T  # (n, canales)

    _emit(progress_line(PCT_WRITE, "escribiendo el stem de bajo..."))
    peak = float(np.max(np.abs(bass))) if bass.size else 0.0
    if peak > 0.999:
        # Igual que Demucs (clip="rescale"): se baja el volumen en vez de recortar picos.
        bass = bass * (0.999 / peak)
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), bass.astype(np.float32), sr, subtype="PCM_16")
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        raise RunnerError(f"No se pudo escribir {output}: {exc}", EXIT_OUTPUT) from exc
    _emit(progress_line(100, "listo"))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del subproceso.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Argumentos (sin el nombre del programa); ``None`` usa ``sys.argv[1:]``.

    Returns
    -------
    int
        :data:`EXIT_OK` o el código ``EXIT_*`` del fallo (argumentos inválidos:
        2, el código estándar de argparse).
    """
    for stream in (sys.stdout, sys.stderr):
        # El padre decodifica en UTF-8; en Windows la consola/tubería usaría cp1252
        # y las tildes del protocolo llegarían rotas.
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # pragma: no cover - flujo no reconfigurable
                pass
    args = build_parser().parse_args(argv)
    try:
        separate(args)
    except RunnerError as exc:
        _emit(f"{ERROR_PREFIX} {exc}")
        return exc.code
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
