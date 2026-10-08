"""Pruebas de la etapa 1 (carga de audio con ffmpeg, imageio-ffmpeg o soundfile)."""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import io_audio
from src.io_audio import AudioLoadError, FFmpegNotFoundError, convert_to_mp3, load_audio, save_mp3, save_wav

needs_ffmpeg = pytest.mark.skipif(io_audio.find_ffmpeg() is None, reason="ffmpeg no instalado")
needs_sf_mp3 = pytest.mark.skipif("MP3" not in sf.available_formats(),
                                  reason="libsndfile < 1.1: soundfile no lee/escribe MP3")

#: La función real que lee el PATH del registro (el fixture ``fake_windows`` la sustituye en el módulo).
_REAL_REGISTRY_DIRS = io_audio._windows_registry_path_dirs


def _tone(freq: float = 110.0, sr: int = 22050, dur: float = 1.0) -> np.ndarray:
    """Seno de amplitud 0.5 (float32) de ``freq`` Hz y ``dur`` s."""
    t = np.arange(int(sr * dur)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _peak_hz(y: np.ndarray, sr: int) -> float:
    """Frecuencia (Hz) del pico del espectro de magnitud de ``y``."""
    spectrum = np.abs(np.fft.rfft(y))
    return float(np.argmax(spectrum) * sr / len(y))


@pytest.fixture()
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simula un sistema sin ffmpeg de ninguna forma (ni variable, ni PATH, ni imageio-ffmpeg)."""
    monkeypatch.setattr(io_audio, "find_ffmpeg", lambda: None)


@pytest.fixture()
def stereo_mp3(tmp_path: Path) -> Path:
    """MP3 estéreo real de 2 s a 44.1 kHz (izquierdo 110 Hz, derecho 220 Hz) escrito con soundfile."""
    if "MP3" not in sf.available_formats():
        pytest.skip("libsndfile < 1.1: soundfile no escribe MP3")
    sr = 44100
    data = np.stack([_tone(110.0, sr, 2.0), _tone(220.0, sr, 2.0)], axis=1)  # (n, 2)
    path = tmp_path / "estereo.mp3"
    sf.write(str(path), data, sr, format="MP3")
    return path


# ---------------------------------------------------------------------------
# Decodificación con ffmpeg
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_mp3_round_trip_preserves_pitch_and_length(tmp_path: Path) -> None:
    """Codificar a MP3 y decodificar conserva la frecuencia del tono y (casi) la duración."""
    sr = 22050
    y = _tone(110.0, sr, 1.0)
    mp3 = save_mp3(tmp_path / "tono.mp3", y, sr)
    y2, sr2 = load_audio(mp3, sr=sr)
    assert sr2 == sr
    assert y2.dtype == np.float32
    # El MP3 añade algo de relleno (encoder delay): la duración debe ser parecida.
    assert abs(len(y2) - len(y)) < 0.1 * sr
    assert abs(_peak_hz(y2, sr) - 110.0) < 2.0


@needs_ffmpeg
def test_load_resamples_and_mixes_to_mono(tmp_path: Path) -> None:
    """Un WAV a 44.1 kHz se remuestrea a la frecuencia pedida (22 050 Hz) sin cambiar su duración."""
    wav = save_wav(tmp_path / "tono.wav", _tone(220.0, 44100, 0.5), 44100)
    y, sr = load_audio(wav, sr=22050)
    assert sr == 22050
    assert abs(len(y) - 0.5 * 22050) < 50


@needs_ffmpeg
def test_ffmpeg_mono_downmix_is_channel_average(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Estéreo → mono con ffmpeg es el PROMEDIO (L + R)/2, no (L + R)/√2, e igual que con soundfile.

    Regresión: sin ``-rematrix_maxval 1.0`` un bajo centrado (L = R) a pico 0.999
    llegaba a 1.41 y la pestaña Audio lo reproducía saturado.
    """
    sr = 22050
    t = np.arange(sr) / sr
    left = (0.999 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    right = (0.5 * np.sin(2 * np.pi * 55.0 * t)).astype(np.float32)
    centered = tmp_path / "centrado.wav"
    sf.write(str(centered), np.stack([left, left], axis=1), sr, subtype="FLOAT")
    y, _ = load_audio(centered, sr=sr)
    assert float(np.max(np.abs(y))) <= 1.0
    np.testing.assert_allclose(y[:sr], left, atol=1e-5)
    mixed = tmp_path / "mezcla.wav"
    sf.write(str(mixed), np.stack([left, right], axis=1), sr, subtype="FLOAT")
    y_ffmpeg, _ = load_audio(mixed, sr=sr)
    np.testing.assert_allclose(y_ffmpeg[:sr], (left + right) / 2, atol=1e-5)
    monkeypatch.setattr(io_audio, "find_ffmpeg", lambda: None)
    y_soundfile, _ = load_audio(mixed, sr=sr)
    np.testing.assert_allclose(y_ffmpeg[:sr], y_soundfile[:sr], atol=1e-5)


@needs_ffmpeg
def test_stereo_with_ffmpeg_keeps_channels(stereo_mp3: Path) -> None:
    """``channels=2`` con ffmpeg: forma (2, n), cada canal con su tono y la duración correcta."""
    y, sr = load_audio(stereo_mp3, sr=44100, channels=2)
    assert y.shape[0] == 2 and y.dtype == np.float32 and y.flags["C_CONTIGUOUS"]
    assert abs(y.shape[1] - 2.0 * sr) < 0.05 * sr
    assert abs(_peak_hz(y[0], sr) - 110.0) < 2.0
    assert abs(_peak_hz(y[1], sr) - 220.0) < 2.0
    mono, _ = load_audio(stereo_mp3, sr=22050)
    assert mono.ndim == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    """Un archivo inexistente da AudioLoadError (no una traza de ffmpeg)."""
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "no_existe.mp3")


def test_invalid_channels_raise(tmp_path: Path) -> None:
    """Solo se admite channels = 1 (mono) o 2 (estéreo)."""
    wav = save_wav(tmp_path / "tono.wav", _tone(), 22050)
    with pytest.raises(ValueError, match="channels"):
        load_audio(wav, channels=3)


@needs_ffmpeg
def test_corrupt_file_raises_audio_load_error(tmp_path: Path) -> None:
    """Bytes que no son audio dan AudioLoadError con el detalle de ffmpeg."""
    bad = tmp_path / "corrupto.mp3"
    bad.write_bytes(b"esto no es un mp3")
    with pytest.raises(AudioLoadError):
        load_audio(bad)


@needs_ffmpeg
def test_non_finite_samples_are_repaired_or_rejected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Regresión: un WAV float con alguna muestra NaN/Inf se repara con 0 (aviso); si hay muchas, se rechaza."""
    y = _tone(110.0, 22050, 0.5)
    y[100], y[200] = np.nan, np.inf
    few = tmp_path / "con_nan.wav"
    sf.write(str(few), y, 22050, subtype="FLOAT")
    with caplog.at_level(logging.WARNING, logger="src.io_audio"):
        loaded, _ = load_audio(few)
    assert np.all(np.isfinite(loaded))
    assert "no finitas" in caplog.text
    y[: y.size // 2] = np.nan
    many = tmp_path / "muchos_nan.wav"
    sf.write(str(many), y, 22050, subtype="FLOAT")
    with pytest.raises(AudioLoadError, match="NaN"):
        load_audio(many)


# ---------------------------------------------------------------------------
# Localización de ffmpeg: variable, PATH, imageio-ffmpeg
# ---------------------------------------------------------------------------


def test_non_executable_env_var_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           caplog: pytest.LogCaptureFixture) -> None:
    """Regresión: SMARTUNER_FFMPEG apuntando a un archivo NO ejecutable se ignora (con aviso)."""
    fake = tmp_path / "ffmpeg_falso"
    fake.write_text("no soy un ejecutable")
    os.chmod(fake, 0o644)
    monkeypatch.setenv(io_audio.FFMPEG_ENV_VAR, str(fake))
    monkeypatch.setattr(io_audio.shutil, "which", lambda name: "/ruta/del/path/ffmpeg")
    with caplog.at_level(logging.WARNING, logger="src.io_audio"):
        assert io_audio.find_ffmpeg() == "/ruta/del/path/ffmpeg"
    assert "no es un archivo ejecutable" in caplog.text


def test_imageio_ffmpeg_is_used_when_not_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    """Sin ffmpeg en el PATH ni SMARTUNER_FFMPEG se usa el binario de imageio-ffmpeg (anunciado UNA vez) y decodifica."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    monkeypatch.delenv(io_audio.FFMPEG_ENV_VAR, raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    monkeypatch.setattr(io_audio.shutil, "which", lambda name: None)
    monkeypatch.setattr(io_audio.sys, "platform", "linux")  # sin la búsqueda específica de Windows
    monkeypatch.setattr(io_audio, "_announced_ffmpeg", None)
    with caplog.at_level(logging.INFO, logger="src.io_audio"):
        first = io_audio.find_ffmpeg()
        second = io_audio.find_ffmpeg()
    assert first == second == imageio_ffmpeg.get_ffmpeg_exe()
    assert caplog.text.count("imageio-ffmpeg") == 1  # se anuncia una sola vez
    # Y ese ffmpeg decodifica de verdad (WAV a 44.1 kHz → 22 050 Hz).
    wav = save_wav(tmp_path / "tono.wav", _tone(220.0, 44100, 0.5), 44100)
    y, sr = load_audio(wav, sr=22050)
    assert abs(len(y) - 0.5 * sr) < 50
    assert abs(_peak_hz(y, sr) - 220.0) < 3.0


def test_broken_imageio_ffmpeg_means_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cualquier excepción de imageio-ffmpeg se interpreta como "no disponible" (no rompe la búsqueda)."""
    broken = types.ModuleType("imageio_ffmpeg")

    def explode() -> str:
        """get_ffmpeg_exe() de un paquete roto."""
        raise RuntimeError("binario ausente")

    broken.get_ffmpeg_exe = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", broken)
    monkeypatch.delenv(io_audio.FFMPEG_ENV_VAR, raising=False)
    monkeypatch.setattr(io_audio.shutil, "which", lambda name: None)
    monkeypatch.setattr(io_audio.sys, "platform", "linux")
    assert io_audio._imageio_ffmpeg_exe() is None
    assert io_audio.find_ffmpeg() is None


# ---------------------------------------------------------------------------
# Búsqueda específica de Windows (simulada en cualquier sistema)
# ---------------------------------------------------------------------------


def _fake_exe(path: Path, mtime: float | None = None) -> Path:
    """Crea un ``ffmpeg.exe`` falso ejecutable (con fecha de modificación opcional)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ffmpeg falso")
    os.chmod(path, 0o755)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture()
def fake_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Simula Windows sin ffmpeg en el PATH del proceso; devuelve la carpeta %LOCALAPPDATA% falsa."""
    local = tmp_path / "AppData" / "Local"
    local.mkdir(parents=True)
    monkeypatch.setattr(io_audio.sys, "platform", "win32")
    monkeypatch.setattr(io_audio.shutil, "which", lambda name: None)
    monkeypatch.delenv(io_audio.FFMPEG_ENV_VAR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.setenv("ProgramData", str(tmp_path / "ProgramData"))
    monkeypatch.setattr(io_audio, "_windows_registry_path_dirs", lambda: [])

    def imageio_must_not_run() -> str | None:
        """Si se llega a imageio-ffmpeg es que la búsqueda de Windows no encontró el ffmpeg."""
        return None

    monkeypatch.setattr(io_audio, "_imageio_ffmpeg_exe", imageio_must_not_run)
    return local


def test_windows_finds_newest_winget_package(fake_windows: Path, tmp_path: Path) -> None:
    """En Windows, un ffmpeg de WinGet (Gyan.FFmpeg) se encuentra aunque el PATH de la terminal sea viejo; gana el más reciente."""
    packages = fake_windows / "Microsoft" / "WinGet" / "Packages"
    _fake_exe(packages / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" / "ffmpeg-6.0-full_build" / "bin"
              / "ffmpeg.exe", mtime=1_000_000)
    newest = _fake_exe(packages / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" / "ffmpeg-7.1-full_build"
                       / "bin" / "ffmpeg.exe", mtime=2_000_000)
    assert io_audio.find_ffmpeg() == str(newest)
    # El enlace de WinGet\Links tiene prioridad sobre las carpetas de paquetes.
    link = _fake_exe(fake_windows / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
    assert io_audio.find_ffmpeg() == str(link)


@pytest.mark.parametrize("where", [("Program Files", "ffmpeg", "bin"), ("ProgramData", "chocolatey", "bin")])
def test_windows_finds_program_files_and_chocolatey(fake_windows: Path, tmp_path: Path, where: tuple[str, ...]) -> None:
    """En Windows también se buscan %ProgramFiles%\\ffmpeg\\bin y la carpeta de Chocolatey."""
    exe = _fake_exe(tmp_path.joinpath(*where) / "ffmpeg.exe")
    assert io_audio.find_ffmpeg() == str(exe)


def test_windows_reads_path_from_registry(fake_windows: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          caplog: pytest.LogCaptureFixture) -> None:
    """El PATH de usuario/sistema del registro (con %VARIABLES%) se usa para hallar un ffmpeg recién instalado."""
    exe = _fake_exe(tmp_path / "herramientas" / "ffmpeg" / "bin" / "ffmpeg.exe")
    monkeypatch.setenv("MIS_HERRAMIENTAS", str(tmp_path / "herramientas"))

    class FakeKey:
        """Clave de registro falsa (gestor de contexto, como la de winreg)."""

        def __init__(self, value: str) -> None:
            self.value = value

        def __enter__(self) -> "FakeKey":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def open_key(root: str, subkey: str) -> FakeKey:
        """HKLM ilegible (OSError, como sin permisos); HKCU con un PATH que usa %VARIABLES%."""
        if root == "HKLM":
            raise OSError("acceso denegado")
        assert subkey == "Environment"
        return FakeKey("C:/no/existe;%MIS_HERRAMIENTAS%/ffmpeg/bin;")

    fake_winreg = types.SimpleNamespace(HKEY_LOCAL_MACHINE="HKLM", HKEY_CURRENT_USER="HKCU", OpenKey=open_key,
                                        QueryValueEx=lambda key, name: (key.value, 2))
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)
    dirs = _REAL_REGISTRY_DIRS()  # la función real (el fixture la había sustituido por [])
    assert dirs == ["C:/no/existe", str(tmp_path / "herramientas") + "/ffmpeg/bin"]
    monkeypatch.setattr(io_audio, "_windows_registry_path_dirs", _REAL_REGISTRY_DIRS)
    monkeypatch.setattr(io_audio, "_announced_ffmpeg", None)
    with caplog.at_level(logging.INFO, logger="src.io_audio"):
        assert io_audio.find_ffmpeg() == str(Path(dirs[1]) / "ffmpeg.exe") == str(exe)
    assert "registro de Windows" in caplog.text


def test_windows_search_failures_are_harmless(fake_windows: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la búsqueda de Windows lanza cualquier excepción, se sigue con imageio-ffmpeg / soundfile sin romper."""

    def explode() -> list[str]:
        """Registro que falla de forma inesperada."""
        raise RuntimeError("registro corrupto")

    monkeypatch.setattr(io_audio, "_windows_registry_path_dirs", explode)
    assert io_audio.find_ffmpeg() is None


def test_registry_lookup_is_empty_outside_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin el módulo winreg (Linux/macOS) el PATH del registro es simplemente una lista vacía."""
    monkeypatch.setitem(sys.modules, "winreg", None)  # import winreg → ImportError
    assert _REAL_REGISTRY_DIRS() == []


def test_windows_search_is_skipped_on_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """En Linux/macOS no se ejecuta la búsqueda específica de Windows."""

    def forbidden() -> None:
        """No debe llamarse fuera de Windows."""
        raise AssertionError("búsqueda de Windows fuera de Windows")

    monkeypatch.setattr(io_audio.sys, "platform", "darwin")
    monkeypatch.setattr(io_audio.shutil, "which", lambda name: None)
    monkeypatch.delenv(io_audio.FFMPEG_ENV_VAR, raising=False)
    monkeypatch.setattr(io_audio, "_find_ffmpeg_windows", forbidden)
    monkeypatch.setattr(io_audio, "_imageio_ffmpeg_exe", lambda: None)
    assert io_audio.find_ffmpeg() is None


# ---------------------------------------------------------------------------
# Respaldo con soundfile (sin ffmpeg)
# ---------------------------------------------------------------------------


@needs_sf_mp3
def test_soundfile_fallback_decodes_real_mp3(stereo_mp3: Path, no_ffmpeg: None,
                                             caplog: pytest.LogCaptureFixture) -> None:
    """Sin ffmpeg, un MP3 real (estéreo, 44.1 kHz) se decodifica con soundfile: mono a 22 050 Hz con pitch y duración correctos."""
    with caplog.at_level(logging.INFO, logger="src.io_audio"):
        y, sr = load_audio(stereo_mp3, sr=22050)
    assert "soundfile" in caplog.text
    assert sr == 22050 and y.ndim == 1 and y.dtype == np.float32
    assert abs(len(y) - 2.0 * sr) < 0.02 * sr
    # Mezcla a mono = promedio: quedan los dos tonos con la mitad de amplitud.
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    top2 = sorted(freqs[np.argsort(spectrum)[-2:]])
    assert abs(top2[0] - 110.0) < 2.0 and abs(top2[1] - 220.0) < 2.0
    assert 0.2 < float(np.max(np.abs(y))) < 0.6


@needs_sf_mp3
def test_soundfile_fallback_stereo_and_mono_duplication(stereo_mp3: Path, tmp_path: Path, no_ffmpeg: None) -> None:
    """Sin ffmpeg, ``channels=2`` devuelve (2, n) con cada canal intacto, y un archivo mono se duplica."""
    y, sr = load_audio(stereo_mp3, sr=44100, channels=2)
    assert y.shape[0] == 2 and abs(y.shape[1] - 2.0 * sr) < 0.02 * sr
    assert abs(_peak_hz(y[0], sr) - 110.0) < 2.0
    assert abs(_peak_hz(y[1], sr) - 220.0) < 2.0
    mono = save_wav(tmp_path / "mono.wav", _tone(55.0, 22050, 1.0), 22050)
    y2, sr2 = load_audio(mono, sr=44100, channels=2)  # mono + remuestreo 22 050 → 44 100 Hz
    assert y2.shape == (2, 44100) and sr2 == 44100
    np.testing.assert_array_equal(y2[0], y2[1])
    assert abs(_peak_hz(y2[0], sr2) - 55.0) < 2.0


def test_no_ffmpeg_and_unreadable_file_gives_clear_error(tmp_path: Path, no_ffmpeg: None) -> None:
    """Solo si ffmpeg y soundfile fallan se lanza FFmpegNotFoundError, sugiriendo primero pip install imageio-ffmpeg."""
    bad = tmp_path / "cancion.m4a"
    bad.write_bytes(b"\x00\x00\x00\x20ftypM4A " + bytes(200))
    with pytest.raises(FFmpegNotFoundError) as info:
        load_audio(bad)
    message = str(info.value)
    assert message.index("pip install imageio-ffmpeg") < message.index("winget install Gyan.FFmpeg")
    assert "brew install ffmpeg" in message and "apt install ffmpeg" in message
    assert "cierra y vuelve a abrir la terminal" in message
    assert "cancion.m4a" in message


def test_ffmpeg_that_cannot_run_falls_back_to_soundfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         caplog: pytest.LogCaptureFixture) -> None:
    """Regresión: si el sistema no puede ejecutar ffmpeg (OSError) se avisa y se usa soundfile; si tampoco puede, FFmpegNotFoundError."""
    wav = save_wav(tmp_path / "tono.wav", _tone(), 22050)
    fake = tmp_path / "ffmpeg_sin_permiso"
    fake.write_text("#!/bin/sh\n")
    os.chmod(fake, 0o644)
    monkeypatch.setattr(io_audio, "find_ffmpeg", lambda: str(fake))
    with caplog.at_level(logging.WARNING, logger="src.io_audio"):
        y, _ = load_audio(wav)
    assert "No se pudo ejecutar ffmpeg" in caplog.text
    assert abs(_peak_hz(y, 22050) - 110.0) < 2.0
    bad = tmp_path / "corrupto.mp3"
    bad.write_bytes(b"esto no es un mp3")
    with pytest.raises(FFmpegNotFoundError, match="No se pudo ejecutar ffmpeg"):
        load_audio(bad)


@needs_sf_mp3
def test_save_and_convert_mp3_without_ffmpeg(tmp_path: Path, no_ffmpeg: None) -> None:
    """Sin ffmpeg, save_mp3 y convert_to_mp3 escriben MP3 válidos con soundfile."""
    sr = 22050
    mp3 = save_mp3(tmp_path / "tono.mp3", _tone(110.0, sr, 1.0), sr)
    assert sf.info(str(mp3)).format == "MP3"
    y, _ = load_audio(mp3, sr=sr)
    assert abs(len(y) - sr) < 0.05 * sr
    assert abs(_peak_hz(y, sr) - 110.0) < 2.0
    wav = save_wav(tmp_path / "tono.wav", _tone(220.0, sr, 1.0), sr)
    converted = convert_to_mp3(wav, tmp_path / "convertido.mp3")
    assert sf.info(str(converted)).format == "MP3"


def test_mp3_writing_without_any_backend_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_ffmpeg: None) -> None:
    """Sin ffmpeg y con un libsndfile que no escribe MP3, save_mp3 lanza FFmpegNotFoundError con instrucciones."""
    monkeypatch.setattr(io_audio.sf, "available_formats", lambda: {"WAV": "WAV (Microsoft)"})
    with pytest.raises(FFmpegNotFoundError, match="pip install imageio-ffmpeg"):
        save_mp3(tmp_path / "x.mp3", _tone(), 22050)


def test_ffmpeg_that_cannot_run_when_writing_mp3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: ffmpeg no ejecutable al escribir → respaldo soundfile o, si no escribe MP3, FFmpegNotFoundError."""
    fake = tmp_path / "ffmpeg_sin_permiso"
    fake.write_text("#!/bin/sh\n")
    os.chmod(fake, 0o644)
    monkeypatch.setattr(io_audio, "find_ffmpeg", lambda: str(fake))
    if "MP3" in sf.available_formats():
        assert save_mp3(tmp_path / "salida.mp3", _tone(), 22050).is_file()
    monkeypatch.setattr(io_audio.sf, "available_formats", lambda: {})
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg no se pudo ejecutar"):
        save_mp3(tmp_path / "otra.mp3", _tone(), 22050)
