"""Pruebas de la etapa 2 (separación del bajo con Demucs).

Demucs/PyTorch no están instalados en el entorno de pruebas, así que las
pruebas de extremo a extremo crean un paquete **falso** ``demucs`` en
``tmp_path`` (``demucs/__main__.py``) que imita lo esencial del real:
acepta los mismos argumentos, dibuja una barra de progreso estilo tqdm en
stderr (reescrita con ``\\r``) y escribe ``<out>/<modelo>/<pista>/bass.wav``.
Se inyecta en el subproceso vía ``PYTHONPATH`` con el parámetro ``env``.
"""

from __future__ import annotations

import doctest
import importlib.util
import logging
import os
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from src import config, separation
from src.config import CancelledError
from src.separation import (
    SeparationCancelledError,
    SeparationError,
    _demucs_command,
    _parse_percent,
    cache_key,
    cached_stem_path,
    demucs_available,
    separate_bass,
)

FAKE_MAIN = textwrap.dedent(
    '''
    """Demucs FALSO para pruebas: imita argumentos, progreso tqdm y archivos de salida."""
    import argparse
    import os
    import shutil
    import sys
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--two-stems")
    parser.add_argument("-n", dest="name")
    parser.add_argument("-o", dest="out")
    parser.add_argument("track")
    args = parser.parse_args()

    log = os.environ.get("FAKE_DEMUCS_LOG")
    if log:  # registra cada invocación (para comprobar que la caché evita ejecutarlo)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {' '.join(sys.argv[1:])}\\n")

    mode = os.environ.get("FAKE_DEMUCS_MODE", "ok")
    delay = float(os.environ.get("FAKE_DEMUCS_DELAY", "0"))
    print(f"Selected model is a bag of 1 models. You will see that many progress bars per track.")
    print(f"Separating track {args.track}", flush=True)
    if mode == "fail":
        sys.stderr.write("Traceback (most recent call last):\\nRuntimeError: modelo roto\\n")
        sys.exit(3)

    for pct in (0, 25, 50, 75, 100):
        bar = "\\u2588" * (pct // 10)
        sys.stderr.write(f"\\r{pct:3d}%|{bar:<10}| {pct / 10:.1f}/10.0 [00:01<00:01, 9.9seconds/s]")
        sys.stderr.flush()
        if mode == "hang" and pct == 25:
            time.sleep(60)  # se queda "pensando" hasta que lo maten
        time.sleep(delay)
    sys.stderr.write("\\n")

    if mode == "nooutput":
        sys.exit(0)
    track = Path(args.track)
    dest = Path(args.out) / args.name / track.name.rsplit(".", 1)[0]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(track, dest / "bass.wav")       # el "bajo" es una copia de la entrada
    (dest / "no_bass.wav").write_bytes(b"resto")
    '''
)


@dataclass
class FakeDemucs:
    """Demucs falso instalado en ``root``: llamarlo construye el ``env`` del subproceso.

    Attributes
    ----------
    root : Path
        Carpeta que contiene el paquete ``demucs`` falso (se añade a PYTHONPATH).
    log : Path
        Archivo donde el Demucs falso registra cada invocación.
    """

    root: Path
    log: Path

    def __call__(self, mode: str = "ok", delay: float = 0.0) -> dict[str, str]:
        """Variables de entorno para lanzar el Demucs falso en modo ``mode`` (ok, fail, hang, nooutput)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(self.root), env.get("PYTHONPATH", "")) if p)
        env["FAKE_DEMUCS_MODE"] = mode
        env["FAKE_DEMUCS_DELAY"] = str(delay)
        env["FAKE_DEMUCS_LOG"] = str(self.log)
        return env


@pytest.fixture()
def fake_demucs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDemucs:
    """Crea el paquete falso y devuelve el objeto que construye el ``env`` del subproceso."""
    root = tmp_path / "fake_site"
    pkg = root / "demucs"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Paquete demucs falso."""\n', encoding="utf-8")
    (pkg / "__main__.py").write_text(FAKE_MAIN, encoding="utf-8")
    monkeypatch.setattr(separation, "demucs_available", lambda: True)
    return FakeDemucs(root=root, log=tmp_path / "invocaciones.log")


@pytest.fixture()
def song(tmp_path: Path) -> Path:
    """Archivo 'MP3' falso (bytes deterministas) con un punto en el nombre."""
    path = tmp_path / "mi.cancion.mp3"
    path.write_bytes(b"ID3" + bytes(range(256)) * 50)
    return path


def _invocations(log: Path) -> list[str]:
    """Líneas del registro de invocaciones del Demucs falso (vacío si nunca se ejecutó)."""
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


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
# Disponibilidad, comando y progreso
# ---------------------------------------------------------------------------


def test_demucs_available_does_not_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """demucs_available() busca el paquete sin importarlo (importar PyTorch tardaría segundos)."""
    if importlib.util.find_spec("demucs") is None:
        assert demucs_available() is False
    pkg = tmp_path / "site" / "demucs"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('no debe importarse')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    assert demucs_available() is True
    assert "demucs" not in sys.modules


def test_demucs_command() -> None:
    """El comando lanza ``python -m demucs --two-stems bass -n <modelo> -o <salida> <pista>`` con argumentos de texto."""
    cmd = _demucs_command("musica/mezcla.mp3", "htdemucs", "/tmp/salida")
    assert cmd[:3] == [sys.executable, "-m", "demucs"]
    assert cmd[cmd.index("--two-stems") + 1] == "bass"
    assert cmd[cmd.index("-n") + 1] == "htdemucs"
    assert cmd[cmd.index("-o") + 1] == "/tmp/salida"
    assert cmd[-1] == str(Path("musica/mezcla.mp3"))
    assert all(isinstance(arg, str) for arg in cmd)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  45%|████▌     | 9.0/20.0 [00:03<00:04, 2.6seconds/s]", 45.0),
        ("100%|██████████| 20.0/20.0 [00:07<00:00, 2.7seconds/s]", 100.0),
        ("  0%|          | 0.0/20.0 [00:00<?, ?seconds/s]", 0.0),
        (" 12%|█▏ \r 13%|█▎ ", 13.0),
        ("Separating track mezcla.mp3", None),
        ("", None),
    ],
)
def test_parse_percent(text: str, expected: float | None) -> None:
    """_parse_percent extrae el último porcentaje de una línea de tqdm (o None)."""
    assert _parse_percent(text) == expected


# ---------------------------------------------------------------------------
# separate_bass
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
    assert "PyTorch" in message
    assert not (tmp_path / "cache").exists()


def test_missing_input_file_raises(tmp_path: Path) -> None:
    """Un archivo de entrada inexistente es SeparationError con mensaje en español."""
    with pytest.raises(SeparationError, match="No existe"):
        separate_bass(tmp_path / "no_existe.mp3", cache_dir=tmp_path / "cache")


def test_end_to_end_with_fake_demucs(tmp_path: Path, song: Path, fake_demucs: FakeDemucs) -> None:
    """De extremo a extremo con el Demucs falso: stem en caché, carpeta temporal borrada, progreso leído de tqdm y caché reutilizada."""
    cache = tmp_path / "cache"
    calls: list[tuple[float, str]] = []
    result = separate_bass(
        song, model="htdemucs", cache_dir=cache, progress=lambda f, m: calls.append((f, m)), env=fake_demucs()
    )

    # El stem quedó en la caché, con el contenido que escribió "Demucs".
    assert result == cached_stem_path(song, "htdemucs", cache)
    assert result.read_bytes() == song.read_bytes()
    # La carpeta temporal se borró: en la caché solo queda el stem.
    assert sorted(p.name for p in cache.iterdir()) == [result.name]

    # Progreso: se leyeron los porcentajes de tqdm y llegan en orden.
    messages = [m for _, m in calls]
    for pct in (25, 50, 75, 100):
        assert f"Demucs: {pct} %" in messages
    assert (0.5, "Demucs: 50 %") in calls
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert calls[-1] == (1.0, "Demucs: separación terminada")

    # Se invocó con los argumentos esperados, y una segunda llamada usa la caché.
    runs = _invocations(fake_demucs.log)
    assert len(runs) == 1
    assert "--two-stems bass -n htdemucs -o" in runs[0]
    again = separate_bass(song, cache_dir=cache, env=fake_demucs())
    assert again == result
    assert len(_invocations(fake_demucs.log)) == 1

    # Otro modelo → otra clave → se vuelve a ejecutar.
    other = separate_bass(song, model="htdemucs_ft", cache_dir=cache, env=fake_demucs())
    assert other != result and other.is_file()
    assert len(_invocations(fake_demucs.log)) == 2


def test_progress_is_incremental(tmp_path: Path, song: Path, fake_demucs: FakeDemucs) -> None:
    """Cada porcentaje llega mientras Demucs sigue trabajando, no todos al final."""
    arrivals: list[tuple[float, float]] = []
    start = time.monotonic()
    separate_bass(
        song,
        cache_dir=tmp_path / "cache",
        progress=lambda f, m: arrivals.append((f, time.monotonic() - start)),
        env=fake_demucs(delay=0.3),
    )
    t25 = next(t for f, t in arrivals if f == 0.25)
    t100 = next(t for f, t in arrivals if f == 1.0)
    assert t100 - t25 > 0.5  # 25 % se notificó bastante antes de terminar


def test_cancellation_kills_process(tmp_path: Path, song: Path, fake_demucs: FakeDemucs) -> None:
    """Cancelar durante Demucs mata el proceso enseguida y lanza SeparationCancelledError (también CancelledError)."""
    cache = tmp_path / "cache"
    cancel = threading.Event()

    def on_progress(fraction: float, message: str) -> None:
        """Callback que simula al usuario pulsando «Cancelar» en cuanto ve progreso."""
        if fraction >= 0.25:  # el usuario pulsa "Cancelar" en cuanto ve progreso
            cancel.set()

    start = time.monotonic()
    with pytest.raises(CancelledError) as info:
        separate_bass(song, cache_dir=cache, progress=on_progress, cancel=cancel, env=fake_demucs(mode="hang"))
    assert time.monotonic() - start < 15  # no esperó los 60 s del Demucs "colgado"
    assert isinstance(info.value, SeparationError)
    assert isinstance(info.value, SeparationCancelledError)

    # No queda nada en la caché (ni stem a medias ni carpeta temporal).
    assert not cached_stem_path(song, cache_dir=cache).exists()
    assert list(cache.iterdir()) == []

    # El proceso de Demucs ya no existe.
    pid = int(_invocations(fake_demucs.log)[0].split()[0])
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_cancel_before_start_does_not_launch(tmp_path: Path, song: Path, fake_demucs: FakeDemucs) -> None:
    """Con ``cancel`` ya activo no se lanza Demucs."""
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        separate_bass(song, cache_dir=tmp_path / "cache", cancel=cancel, env=fake_demucs())
    assert _invocations(fake_demucs.log) == []


def test_nonzero_exit_raises_with_output_tail(tmp_path: Path, song: Path, fake_demucs: FakeDemucs) -> None:
    """Si Demucs termina con error, el mensaje incluye el código de salida y el final de su salida; la caché queda limpia."""
    cache = tmp_path / "cache"
    with pytest.raises(SeparationError) as info:
        separate_bass(song, cache_dir=cache, env=fake_demucs(mode="fail"))
    assert not isinstance(info.value, CancelledError)
    message = str(info.value)
    assert "código de salida 3" in message
    assert "modelo roto" in message
    assert list(cache.iterdir()) == []


def test_missing_stem_raises(tmp_path: Path, song: Path, fake_demucs: FakeDemucs) -> None:
    """Si Demucs termina sin escribir bass.wav se lanza SeparationError y la caché queda limpia."""
    cache = tmp_path / "cache"
    with pytest.raises(SeparationError, match="sin generar"):
        separate_bass(song, cache_dir=cache, env=fake_demucs(mode="nooutput"))
    assert list(cache.iterdir()) == []


def test_doctests_pass() -> None:
    """Los ejemplos de los docstrings de src.separation se ejecutan sin fallos."""
    assert doctest.testmod(separation).failed == 0
