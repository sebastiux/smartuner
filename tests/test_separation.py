"""Pruebas de la etapa 2 (separación del bajo: Demucs vía runner y HPSS).

Demucs/PyTorch no suelen estar instalados en el entorno de pruebas, así que
las pruebas de extremo a extremo de :func:`separate_bass` crean un **runner
falso** (módulo ``fake_demucs_runner`` en ``tmp_path``) que sigue el mismo
protocolo que :mod:`src.demucs_runner`: acepta ``--input/--output/--model/
--device``, escribe líneas ``PROGRESO: N% ...``, ``SEPARANDO: n`` y
``ERROR: ...`` en stdout, dibuja barras estilo tqdm en stderr (reescritas con
``\\r``) y escribe el WAV de salida. Se inyecta con ``runner_module`` y el
parámetro ``env`` (``PYTHONPATH``).

Las pruebas marcadas ``slow`` usan el runner REAL con un modelo sin entrenar
(``--untrained``) y se omiten si torch/demucs no están instalados.
"""

from __future__ import annotations

import doctest
import importlib.util
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import config, demucs_runner, separation
from src.config import CancelledError
from src.separation import (
    SeparationCancelledError,
    SeparationError,
    _demucs_command,
    _parse_percent,
    cache_key,
    cached_stem_path,
    demucs_available,
    hpss_bass,
    resolve_method,
    separate_bass,
)

FAKE_RUNNER = textwrap.dedent(
    '''
    """Runner de Demucs FALSO para pruebas: mismo protocolo de argumentos y salida que src.demucs_runner."""
    import argparse
    import os
    import shutil
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    log = os.environ.get("FAKE_RUNNER_LOG")
    if log:  # registra cada invocación (para comprobar que la caché evita ejecutarlo)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}|{' '.join(sys.argv[1:])}|{os.getcwd()}|{os.environ.get('PYTHONPATH', '')}\\n")

    mode = os.environ.get("FAKE_RUNNER_MODE", "ok")
    delay = float(os.environ.get("FAKE_RUNNER_DELAY", "0"))

    def bar(pct):
        sys.stderr.write(f"\\r{pct:3d}%|{chr(0x2588) * (pct // 10):<10}| {pct / 10:.1f}/10.0 [00:01<00:01, 9.9s/s]")
        sys.stderr.flush()

    print("PROGRESO: 3% cargando PyTorch y Demucs...", flush=True)
    print(f"PROGRESO: 6% cargando el modelo {args.model}...", flush=True)
    bar(50)  # barra de DESCARGA de pesos: no es avance de la separación
    sys.stderr.write("\\n")
    if mode == "fail":
        sys.stderr.write("Traceback (most recent call last):\\nRuntimeError: modelo roto\\n")
        print("ERROR: No se pudo cargar el modelo: se necesita conexión a Internet la primera vez (~80 MB).",
              flush=True)
        sys.exit(4)
    if mode == "nodeps":  # torch instalado pero roto (p. ej. WinError 126 en Windows)
        print("ERROR: No se pudo importar PyTorch/Demucs ([WinError 126] c10.dll).", flush=True)
        sys.exit(3)
    passes = 2 if mode == "bag" else 1
    print("PROGRESO: 10% separando 1.0 s de audio...", flush=True)
    print(f"SEPARANDO: {passes}", flush=True)
    for _ in range(passes):
        for pct in (0, 25, 50, 75, 100):
            bar(pct)
            if mode == "hang" and pct == 25:
                time.sleep(60)  # se queda "pensando" hasta que lo maten
            time.sleep(delay)
        sys.stderr.write("\\n")
    if mode == "nooutput":
        sys.exit(0)
    print("PROGRESO: 96% escribiendo el stem de bajo...", flush=True)
    shutil.copyfile(args.input, args.output)  # el "bajo" es una copia de la mezcla
    print("PROGRESO: 100% listo", flush=True)
    '''
)

RUNNER_NAME = "fake_demucs_runner"


@dataclass
class FakeRunner:
    """Runner falso instalado en ``root``: llamarlo construye el ``env`` del subproceso.

    Attributes
    ----------
    root : Path
        Carpeta que contiene ``fake_demucs_runner.py`` (se añade a PYTHONPATH).
    log : Path
        Archivo donde el runner falso registra cada invocación.
    """

    root: Path
    log: Path

    def __call__(self, mode: str = "ok", delay: float = 0.0) -> dict[str, str]:
        """Variables de entorno para lanzar el runner falso en modo ``mode`` (ok, bag, fail, nodeps, hang, nooutput)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(self.root), env.get("PYTHONPATH", "")) if p)
        env["FAKE_RUNNER_MODE"] = mode
        env["FAKE_RUNNER_DELAY"] = str(delay)
        env["FAKE_RUNNER_LOG"] = str(self.log)
        return env

    def invocations(self) -> list[list[str]]:
        """Invocaciones registradas: ``[pid, argumentos, cwd, PYTHONPATH]`` (vacío si nunca se ejecutó)."""
        if not self.log.exists():
            return []
        return [line.split("|") for line in self.log.read_text(encoding="utf-8").splitlines()]


@pytest.fixture()
def fake_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    """Crea el runner falso y hace creer a separation que Demucs está instalado."""
    root = tmp_path / "fake_site"
    root.mkdir()
    (root / f"{RUNNER_NAME}.py").write_text(FAKE_RUNNER, encoding="utf-8")
    monkeypatch.setattr(separation, "demucs_available", lambda: True)
    return FakeRunner(root=root, log=tmp_path / "invocaciones.log")


@pytest.fixture()
def song(tmp_path: Path) -> Path:
    """Audio real (1 s, 110 Hz, mono, 22 050 Hz) en una ruta con espacios, tildes y un punto en el nombre."""
    folder = tmp_path / "Mis canciones ñandú"
    folder.mkdir()
    path = folder / "mi canción.v2.wav"
    t = np.arange(22050) / 22050
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32), 22050, subtype="PCM_16")
    return path


def _run(song: Path, cache: Path, env: dict[str, str], **kwargs: object) -> Path:
    """separate_bass con el runner falso."""
    return separate_bass(song, cache_dir=cache, env=env, runner_module=RUNNER_NAME, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Clave y ruta de caché
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic_and_content_based(tmp_path: Path, song: Path) -> None:
    """La clave de caché es un SHA-1 del CONTENIDO: no depende del nombre y cambia con un solo byte."""
    key = cache_key(song)
    assert key == cache_key(song)
    assert len(key) == 40 and all(c in "0123456789abcdef" for c in key)

    # Mismo contenido con otro nombre → misma clave.
    copy = tmp_path / "otro_nombre.mp3"
    copy.write_bytes(song.read_bytes())
    assert cache_key(copy) == key

    # Un byte distinto → otra clave.
    changed = tmp_path / "cambiado.mp3"
    data = bytearray(song.read_bytes())
    data[-1] ^= 0xFF
    changed.write_bytes(bytes(data))
    assert cache_key(changed) != key


def test_cache_key_depends_on_model(song: Path) -> None:
    """La clave de caché incluye el modelo de Demucs (htdemucs ≠ htdemucs_ft)."""
    assert cache_key(song, "htdemucs") != cache_key(song, "htdemucs_ft")
    assert cache_key(song) == cache_key(song, "htdemucs")


def test_cache_key_reads_large_files_in_blocks(tmp_path: Path) -> None:
    """Archivos grandes se leen por bloques y dan el mismo SHA-1 que el contenido completo."""
    import hashlib

    big = tmp_path / "grande.wav"
    payload = os.urandom(3 * (1 << 20) + 123)  # > varios bloques de 1 MiB
    big.write_bytes(payload)
    assert cache_key(big, "m") == hashlib.sha1(payload + b"\x00m").hexdigest()


def test_cached_stem_path(tmp_path: Path, song: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cached_stem_path solo calcula la ruta <caché>/<clave>_bass.wav (no crea carpetas) y usa CACHE_DIR por defecto."""
    explicit = cached_stem_path(song, cache_dir=tmp_path / "c")
    assert explicit == tmp_path / "c" / f"{cache_key(song)}_bass.wav"
    assert not explicit.parent.exists()  # solo calcula la ruta

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "por_defecto")
    assert cached_stem_path(song).parent == tmp_path / "por_defecto"


# ---------------------------------------------------------------------------
# Disponibilidad, método, comando y progreso
# ---------------------------------------------------------------------------


def test_demucs_available_does_not_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """demucs_available() busca demucs Y torch sin importarlos (importar PyTorch tardaría segundos)."""
    site = tmp_path / "site"
    for name in ("demucs", "torch"):
        (site / name).mkdir(parents=True)
        (site / name / "__init__.py").write_text("raise RuntimeError('no debe importarse')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    assert demucs_available() is True
    assert "demucs" not in sys.modules and "torch" not in sys.modules


def test_demucs_without_torch_is_not_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Un demucs sin PyTorch (instalación rota) no cuenta como disponible: «auto» usará HPSS."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object) -> object:
        """find_spec que finge que torch no está instalado y demucs sí."""
        if name == "torch":
            return None
        if name == "demucs":
            return object()
        return real_find_spec(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(separation.importlib.util, "find_spec", find_spec)
    assert demucs_available() is False
    assert resolve_method("auto") == "hpss"


def test_resolve_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto → demucs si está instalado (si no, hpss); demucs sin instalar → SeparationError con instrucciones."""
    assert resolve_method("hpss") == "hpss"
    monkeypatch.setattr(separation, "demucs_available", lambda: True)
    assert resolve_method("auto") == "demucs"
    assert resolve_method("demucs") == "demucs"
    monkeypatch.setattr(separation, "demucs_available", lambda: False)
    assert resolve_method("auto") == "hpss"
    assert resolve_method("HPSS") == "hpss"  # insensible a mayúsculas
    with pytest.raises(SeparationError) as info:
        resolve_method("demucs")
    assert "pip install demucs" in str(info.value) and "«hpss»" in str(info.value)
    with pytest.raises(ValueError, match="desconocido"):
        resolve_method("spleeter")


def test_demucs_command() -> None:
    """El comando lanza ``python -m src.demucs_runner --input … --output … --model … --device cpu`` como lista."""
    cmd = _demucs_command("C:/Mis canciones/mezcla ñ.wav", "/tmp/salida/bajo.wav", "htdemucs")
    assert cmd[:3] == [sys.executable, "-m", "src.demucs_runner"]
    assert cmd[cmd.index("--input") + 1] == str(Path("C:/Mis canciones/mezcla ñ.wav"))  # un solo argumento
    assert cmd[cmd.index("--output") + 1] == str(Path("/tmp/salida/bajo.wav"))
    assert cmd[cmd.index("--model") + 1] == "htdemucs"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert all(isinstance(arg, str) for arg in cmd)
    assert _demucs_command("a.wav", "b.wav", "m", device="cuda", runner_module="otro")[2] == "otro"


def test_runner_and_parent_agree_on_the_protocol() -> None:
    """Las líneas que escribe el runner real son las que el lector del padre entiende."""
    line = demucs_runner.progress_line(45, "cargando el modelo")
    assert line.startswith(demucs_runner.PROGRESS_PREFIX)
    m = separation._PROGRESS_LINE.search(line)
    assert m is not None and float(m.group(1)) == 45.0 and m.group(2) == "cargando el modelo"
    assert separation._SEPARATING_LINE.search(f"{demucs_runner.SEPARATING_PREFIX} 4").group(1) == "4"
    assert demucs_runner.PCT_IMPORT < demucs_runner.PCT_MODEL < demucs_runner.PCT_SEPARATION_START
    assert demucs_runner.PCT_SEPARATION_START < demucs_runner.PCT_SEPARATION_END < demucs_runner.PCT_WRITE < 100


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  45%|████▌     | 9.0/20.0 [00:03<00:04, 2.6seconds/s]", 45.0),
        ("100%|██████████| 20.0/20.0 [00:07<00:00, 2.7seconds/s]", 100.0),
        ("  0%|          | 0.0/20.0 [00:00<?, ?seconds/s]", 0.0),
        (" 12%|█▏ \r 13%|█▎ ", 13.0),
        ("Separating track mezcla.mp3", None),
        ("PROGRESO: 45% cargando", None),
        ("", None),
    ],
)
def test_parse_percent(text: str, expected: float | None) -> None:
    """_parse_percent extrae el último porcentaje de una línea de tqdm (o None)."""
    assert _parse_percent(text) == expected


# ---------------------------------------------------------------------------
# separate_bass con el runner falso
# ---------------------------------------------------------------------------


def test_returns_cache_without_running_anything(
    tmp_path: Path, song: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Si el stem ya está en caché se devuelve sin lanzar Demucs (progreso 100 % y aviso en el log)."""
    cache = tmp_path / "cache"
    stem = cached_stem_path(song, cache_dir=cache)
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"stem guardado")

    def forbidden(*args: object, **kwargs: object) -> None:
        """Sustituto de Popen que falla si se intenta lanzar Demucs."""
        raise AssertionError("no debe ejecutarse Demucs si hay caché")

    monkeypatch.setattr(separation.subprocess, "Popen", forbidden)
    monkeypatch.setattr(separation, "demucs_available", lambda: False)  # ni siquiera hace falta
    calls: list[tuple[float, str]] = []
    with caplog.at_level(logging.INFO, logger="src.separation"):
        result = separate_bass(song, cache_dir=cache, progress=lambda f, m: calls.append((f, m)))
    assert result == stem
    assert result.read_bytes() == b"stem guardado"
    assert "usando caché" in caplog.text
    assert calls and calls[-1][0] == 1.0


def test_missing_demucs_raises_helpful_error(tmp_path: Path, song: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin Demucs instalado, SeparationError explica cómo instalarlo (requirements-optional.txt, PyTorch)."""
    monkeypatch.setattr(separation, "demucs_available", lambda: False)
    with pytest.raises(SeparationError) as info:
        separate_bass(song, cache_dir=tmp_path / "cache")
    message = str(info.value)
    assert "pip install -r requirements-optional.txt" in message
    assert "PyTorch" in message and "80 MB" in message
    assert not (tmp_path / "cache").exists()


def test_missing_input_file_raises(tmp_path: Path) -> None:
    """Un archivo de entrada inexistente es SeparationError con mensaje en español."""
    with pytest.raises(SeparationError, match="No existe"):
        separate_bass(tmp_path / "no_existe.mp3", cache_dir=tmp_path / "cache")


def test_undecodable_input_raises_before_launching(tmp_path: Path, fake_runner: FakeRunner) -> None:
    """Un archivo que no es audio da SeparationError («No se pudo preparar el audio») sin lanzar el runner."""
    bad = tmp_path / "corrupto.mp3"
    bad.write_bytes(b"esto no es audio" * 10)
    cache = tmp_path / "cache"
    with pytest.raises(SeparationError, match="No se pudo preparar el audio"):
        _run(bad, cache, fake_runner())
    assert fake_runner.invocations() == []
    assert list(cache.iterdir()) == []


def test_end_to_end_with_fake_runner(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """De extremo a extremo: WAV estéreo 44.1 kHz al runner, stem en caché, temporales borrados, progreso y caché reutilizada."""
    cache = tmp_path / "cache"
    calls: list[tuple[float, str]] = []
    result = _run(song, cache, fake_runner(), model="htdemucs", progress=lambda f, m: calls.append((f, m)))

    # El stem quedó en la caché: es la copia de la mezcla que preparó el padre
    # (estéreo, 44 100 Hz, misma duración que el original).
    assert result == cached_stem_path(song, "htdemucs", cache)
    info = sf.info(str(result))
    assert (info.samplerate, info.channels) == (44100, 2)
    assert abs(info.frames - 44100) < 50
    # La carpeta temporal se borró: en la caché solo queda el stem.
    assert sorted(p.name for p in cache.iterdir()) == [result.name]

    # Progreso: fases del runner y barra tqdm, en orden y sin retroceder.
    messages = [m for _, m in calls]
    assert messages[0].startswith("Demucs: decodificando mi canción.v2.wav")
    assert "Demucs: cargando PyTorch y Demucs..." in messages
    for pct in (0, 25, 50, 75, 100):
        assert messages.count(f"Demucs: separando… {pct} %") == 1  # la barra de DESCARGA (50 %) se ignoró
    assert messages.index("Demucs: separando 1.0 s de audio...") < messages.index("Demucs: separando… 0 %")
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    # 50 % de la separación → 10 % + 85 % · 0.5 del runner, detrás del 2 % de la decodificación.
    half = dict((m, f) for f, m in calls)["Demucs: separando… 50 %"]
    assert half == pytest.approx(0.02 + 0.98 * (0.10 + 0.85 * 0.5))
    assert calls[-1] == (1.0, "Demucs: separación terminada")

    # Se invocó con los argumentos esperados, desde la raíz del proyecto y con
    # la raíz al frente de PYTHONPATH (para que ``-m src.demucs_runner`` funcione).
    runs = fake_runner.invocations()
    assert len(runs) == 1
    _pid, argv, cwd, pythonpath = runs[0]
    assert "--model htdemucs --device cpu" in argv
    assert Path(cwd) == Path(config.PROJECT_ROOT)
    assert pythonpath.split(os.pathsep)[0] == str(config.PROJECT_ROOT)
    assert str(fake_runner.root) in pythonpath.split(os.pathsep)
    again = _run(song, cache, fake_runner())
    assert again == result
    assert len(fake_runner.invocations()) == 1

    # Otro modelo → otra clave → se vuelve a ejecutar.
    other = _run(song, cache, fake_runner(), model="htdemucs_ft")
    assert other != result and other.is_file()
    assert len(fake_runner.invocations()) == 2


def test_bag_of_models_progress_never_goes_back(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Con una "bolsa" de 2 redes (tqdm 0→100 % dos veces) el progreso se reparte entre ambas y nunca retrocede."""
    calls: list[tuple[float, str]] = []
    _run(song, tmp_path / "cache", fake_runner(mode="bag"), progress=lambda f, m: calls.append((f, m)))
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions)
    messages = [m for _, m in calls]
    assert "Demucs: separando… 100 % (red 1 de 2)" in messages
    assert "Demucs: separando… 50 % (red 2 de 2)" in messages
    first_done = dict((m, f) for f, m in calls)["Demucs: separando… 100 % (red 1 de 2)"]
    assert first_done == pytest.approx(0.02 + 0.98 * (0.10 + 0.85 * 0.5))


def test_progress_is_incremental(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Cada porcentaje llega mientras Demucs sigue trabajando, no todos al final."""
    arrivals: list[tuple[str, float]] = []
    start = time.monotonic()
    _run(song, tmp_path / "cache", fake_runner(delay=0.3),
         progress=lambda f, m: arrivals.append((m, time.monotonic() - start)))
    t25 = next(t for m, t in arrivals if m == "Demucs: separando… 25 %")
    t_end = next(t for m, t in arrivals if m == "Demucs: separación terminada")
    assert t_end - t25 > 0.5  # 25 % se notificó bastante antes de terminar


def test_cancellation_kills_process(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Cancelar durante Demucs mata el proceso enseguida y lanza SeparationCancelledError (también CancelledError)."""
    cache = tmp_path / "cache"
    cancel = threading.Event()

    def on_progress(fraction: float, message: str) -> None:
        """Callback que simula al usuario pulsando «Cancelar» en cuanto ve la separación avanzar."""
        if "25 %" in message:
            cancel.set()

    start = time.monotonic()
    with pytest.raises(CancelledError) as info:
        _run(song, cache, fake_runner(mode="hang"), progress=on_progress, cancel=cancel)
    assert time.monotonic() - start < 15  # no esperó los 60 s del runner "colgado"
    assert isinstance(info.value, SeparationError)
    assert isinstance(info.value, SeparationCancelledError)

    # No queda nada en la caché (ni stem a medias ni carpeta temporal con la mezcla).
    assert not cached_stem_path(song, cache_dir=cache).exists()
    assert list(cache.iterdir()) == []

    # El proceso del runner ya no existe.
    pid = int(fake_runner.invocations()[0][0])
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_cancel_before_start_does_not_launch(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Con ``cancel`` ya activo no se decodifica ni se lanza Demucs."""
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        _run(song, tmp_path / "cache", fake_runner(), cancel=cancel)
    assert fake_runner.invocations() == []


def test_nonzero_exit_raises_with_runner_explanation(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Si el runner termina con error, el mensaje incluye el código, su explicación (ERROR:) y el final de la salida."""
    cache = tmp_path / "cache"
    with pytest.raises(SeparationError) as info:
        _run(song, cache, fake_runner(mode="fail"))
    assert not isinstance(info.value, CancelledError)
    message = str(info.value)
    assert "código de salida 4" in message
    assert "se necesita conexión a Internet la primera vez (~80 MB)" in message
    assert "modelo roto" in message
    assert list(cache.iterdir()) == []


def test_runner_missing_deps_raises_deps_error(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Si el runner sale con EXIT_MISSING_DEPS, separate_bass lanza SeparationDepsError (el pipeline recurre a HPSS)."""
    with pytest.raises(separation.SeparationDepsError, match="WinError 126"):
        _run(song, tmp_path / "cache", fake_runner(mode="nodeps"))


def test_runner_with_broken_torch_dll_exits_with_missing_deps(tmp_path: Path, song: Path,
                                                             monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    """Un torch que falla al importarse con OSError (DLL de Windows) da EXIT_MISSING_DEPS y una línea ERROR: útil."""
    import builtins

    real_import = builtins.__import__

    def broken_import(name: str, *args: object, **kwargs: object) -> object:
        """``import torch`` falla como en Windows sin el Redistributable de Visual C++."""
        if name == "torch":
            raise OSError('[WinError 126] No se puede encontrar el módulo especificado. Error loading "c10.dll"')
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)
    code = demucs_runner.main(["--input", str(song), "--output", str(tmp_path / "b.wav")])
    out = capsys.readouterr().out
    assert code == demucs_runner.EXIT_MISSING_DEPS
    error = next(line for line in out.splitlines() if line.startswith(demucs_runner.ERROR_PREFIX))
    assert "WinError 126" in error and "Visual C++" in error


def test_missing_stem_raises(tmp_path: Path, song: Path, fake_runner: FakeRunner) -> None:
    """Si el runner termina sin escribir el WAV del bajo se lanza SeparationError y la caché queda limpia."""
    cache = tmp_path / "cache"
    with pytest.raises(SeparationError, match="sin generar"):
        _run(song, cache, fake_runner(mode="nooutput"))
    assert list(cache.iterdir()) == []


# ---------------------------------------------------------------------------
# Runner (sin PyTorch: errores controlados)
# ---------------------------------------------------------------------------


def test_runner_without_torch_explains_how_to_install(tmp_path: Path, song: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    """Sin torch/demucs el runner sale con EXIT_MISSING_DEPS y una línea ERROR: con «pip install demucs»."""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch → ImportError
    code = demucs_runner.main(["--input", str(song), "--output", str(tmp_path / "b.wav")])
    out = capsys.readouterr().out
    assert code == demucs_runner.EXIT_MISSING_DEPS != 0
    assert f"{demucs_runner.ERROR_PREFIX} No se pudo importar PyTorch/Demucs" in out
    assert "pip install demucs" in out
    assert out.splitlines()[0] == demucs_runner.progress_line(demucs_runner.PCT_IMPORT, "cargando PyTorch y Demucs...")


def test_runner_download_failure_says_internet_is_needed(tmp_path: Path, song: Path,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """Si get_model falla (p. ej. sin Internet la primera vez) el runner sale con EXIT_MODEL y lo explica."""
    fake_torch = types.ModuleType("torch")
    fake_demucs = types.ModuleType("demucs")
    fake_apply = types.ModuleType("demucs.apply")
    fake_pretrained = types.ModuleType("demucs.pretrained")

    def get_model(name: str) -> None:
        """get_model de un Demucs sin conexión."""
        raise OSError("<urlopen error [Errno -3] Temporary failure in name resolution>")

    fake_pretrained.get_model = get_model  # type: ignore[attr-defined]
    fake_demucs.apply, fake_demucs.pretrained = fake_apply, fake_pretrained  # type: ignore[attr-defined]
    for name, module in (("torch", fake_torch), ("demucs", fake_demucs), ("demucs.apply", fake_apply),
                         ("demucs.pretrained", fake_pretrained)):
        monkeypatch.setitem(sys.modules, name, module)
    code = demucs_runner.main(["--input", str(song), "--output", str(tmp_path / "b.wav"), "--model", "htdemucs"])
    out = capsys.readouterr().out
    assert code == demucs_runner.EXIT_MODEL
    error = next(line for line in out.splitlines() if line.startswith(demucs_runner.ERROR_PREFIX))
    assert "«htdemucs»" in error and "Temporary failure" in error
    assert "Se necesita conexión a Internet la primera vez (~80 MB" in error
    assert not (tmp_path / "b.wav").exists()


def test_runner_rejects_bad_arguments() -> None:
    """Sin --input/--output el runner termina con el código 2 de argparse (sin importar torch)."""
    proc = subprocess.run([sys.executable, "-m", "src.demucs_runner"], cwd=str(config.PROJECT_ROOT),
                          capture_output=True, text=True, encoding="utf-8", check=False)
    assert proc.returncode == 2
    assert "--input" in proc.stderr


# ---------------------------------------------------------------------------
# HPSS
# ---------------------------------------------------------------------------


def _mix(sr: int = 22050, dur: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Mezcla sintética: bajo (A1 sostenido + 2.º armónico) + golpes secos + "voz" aguda; devuelve (mezcla, bajo)."""
    t = np.arange(int(sr * dur)) / sr
    bass = 0.4 * np.sin(2 * np.pi * 55.0 * t) + 0.15 * np.sin(2 * np.pi * 110.0 * t)
    rng = np.random.default_rng(0)
    clicks = np.zeros_like(t)
    for start in range(0, t.size, sr // 4):  # "batería": ráfagas de ruido de 10 ms cada 0.25 s
        clicks[start:start + sr // 100] = rng.uniform(-0.8, 0.8, size=min(sr // 100, t.size - start))
    voice = 0.3 * np.sin(2 * np.pi * 2500.0 * t)
    return (bass + clicks + voice).astype(np.float32), bass.astype(np.float32)


def test_hpss_bass_keeps_bass_and_removes_percussion_and_highs() -> None:
    """HPSS + pasa-bajas: conserva el bajo (fundamental y 2.º armónico) y quita golpes y agudos."""
    sr = 22050
    mix, bass = _mix(sr)
    out = hpss_bass(mix, sr)
    assert out.shape == mix.shape and out.dtype == np.float32
    mid = slice(sr // 2, -sr // 2)  # sin los bordes de la STFT
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))  # noqa: E731
    # Mucho más parecida al bajo que la mezcla original.
    assert rms(out[mid] - bass[mid]) < 0.25 * rms(bass[mid])
    assert rms(out[mid] - bass[mid]) < 0.35 * rms(mix[mid] - bass[mid])
    # Espectro: los agudos (2.5 kHz) desaparecen y los graves se conservan.
    spec = np.abs(np.fft.rfft(out[mid]))
    freqs = np.fft.rfftfreq(out[mid].size, 1.0 / sr)
    band = lambda f: spec[np.abs(freqs - f) < 3].max()  # noqa: E731
    assert band(2500.0) < 0.01 * band(55.0)
    assert band(110.0) > 0.2 * band(55.0)
    # Los golpes (energía entre notas) se atenúan mucho: medido en los 10 ms de cada golpe.
    hits = np.concatenate([np.arange(s, s + sr // 100) for s in range(sr, 2 * sr, sr // 4)])
    assert rms(out[hits] - bass[hits]) < 0.3 * rms(mix[hits] - bass[hits])


def test_hpss_bass_edge_cases() -> None:
    """Señal vacía → vacía; señal 2-D o parámetros no positivos → ValueError."""
    assert hpss_bass(np.zeros(0, dtype=np.float32), 22050).size == 0
    with pytest.raises(ValueError, match="1-D"):
        hpss_bass(np.zeros((2, 100), dtype=np.float32), 22050)
    with pytest.raises(ValueError, match="positivos"):
        hpss_bass(np.zeros(1000, dtype=np.float32), 22050, margin=0.0)


def test_hpss_bass_is_fast_enough() -> None:
    """60 s de audio a 22 050 Hz se procesan en pocos segundos (la canción completa de 6 min, en ~10 s)."""
    sr = 22050
    rng = np.random.default_rng(1)
    y = (0.1 * rng.standard_normal(60 * sr)).astype(np.float32)
    start = time.perf_counter()
    hpss_bass(y, sr)
    assert time.perf_counter() - start < 10.0


def test_doctests_pass() -> None:
    """Los ejemplos de los docstrings de src.separation y src.demucs_runner se ejecutan sin fallos."""
    assert doctest.testmod(separation).failed == 0
    assert doctest.testmod(demucs_runner).failed == 0


# ---------------------------------------------------------------------------
# Demucs REAL (torch + demucs instalados): modelo sin entrenar, sin Internet
# ---------------------------------------------------------------------------

needs_real_demucs = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("demucs") is None,
    reason="torch/demucs no instalados (pip install demucs)")

#: Runner que añade ``--untrained`` (modelo con pesos aleatorios: no descarga nada).
UNTRAINED_RUNNER = textwrap.dedent(
    '''
    """Runner real con --untrained: SOLO para pruebas sin Internet."""
    import sys

    from src.demucs_runner import main

    sys.exit(main(sys.argv[1:] + ["--untrained"]))
    '''
)


@pytest.fixture()
def untrained_runner(tmp_path: Path) -> dict[str, str]:
    """Instala el envoltorio ``untrained_runner`` y devuelve el ``env`` para lanzarlo."""
    site = tmp_path / "untrained_site"
    site.mkdir()
    (site / "untrained_runner.py").write_text(UNTRAINED_RUNNER, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(site), env.get("PYTHONPATH", "")) if p)
    return env


def _synthetic_song(path: Path, dur: float, sr: int = 44100) -> Path:
    """Mezcla estéreo sintética (bajo + golpes) de ``dur`` s."""
    mix, _ = _mix(sr, dur)
    sf.write(str(path), np.stack([mix, 0.8 * mix], axis=1), sr, subtype="PCM_16")
    return path


@pytest.mark.slow
@needs_real_demucs
def test_real_runner_untrained_mechanics(tmp_path: Path) -> None:
    """Runner REAL (torch + demucs instalados) con pesos aleatorios: protocolo, apply_model y escritura del WAV."""
    song = _synthetic_song(tmp_path / "mezcla real.wav", 6.0)
    out = tmp_path / "salida" / "bajo.wav"
    proc = subprocess.run(
        [sys.executable, "-m", "src.demucs_runner", "--input", str(song), "--output", str(out), "--untrained"],
        cwd=str(config.PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", check=False, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0].startswith(demucs_runner.PROGRESS_PREFIX)
    assert f"{demucs_runner.SEPARATING_PREFIX} 1" in lines
    assert lines[-1] == demucs_runner.progress_line(100, "listo")
    assert "100%|" in proc.stderr  # barra tqdm de apply_model
    info = sf.info(str(out))
    assert (info.samplerate, info.channels, info.frames) == (44100, 2, 6 * 44100)
    data, _ = sf.read(str(out), dtype="float32")
    assert np.all(np.isfinite(data)) and float(np.max(np.abs(data))) <= 1.0


@pytest.mark.slow
@needs_real_demucs
def test_real_separate_bass_progress_and_cancel(tmp_path: Path, untrained_runner: dict[str, str],
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """separate_bass con el runner REAL (sin entrenar): progreso monótono con la barra tqdm y cancelación inmediata."""
    song = _synthetic_song(tmp_path / "mezcla.wav", 30.0)
    calls: list[tuple[float, str]] = []
    stem = separate_bass(song, cache_dir=tmp_path / "cache", env=untrained_runner, runner_module="untrained_runner",
                         progress=lambda f, m: calls.append((f, m)))
    assert sf.info(str(stem)).frames == 30 * 44100
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions)
    assert sum("Demucs: separando…" in m for _, m in calls) >= 3
    assert calls[-1] == (1.0, "Demucs: separación terminada")

    # Cancelación en cuanto empieza la barra de separación de OTRA canción.
    other = _synthetic_song(tmp_path / "otra.wav", 60.0)
    cancel = threading.Event()

    def on_progress(fraction: float, message: str) -> None:
        """El usuario cancela al ver el primer avance de la separación."""
        if "separando…" in message:
            cancel.set()

    start = time.monotonic()
    with pytest.raises(SeparationCancelledError):
        separate_bass(other, cache_dir=tmp_path / "cache", env=untrained_runner, runner_module="untrained_runner",
                      progress=on_progress, cancel=cancel)
    assert time.monotonic() - start < 120
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [stem.name]
