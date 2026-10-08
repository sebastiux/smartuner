"""Pruebas de la etapa 1 (carga de audio con ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import io_audio
from src.io_audio import AudioLoadError, FFmpegNotFoundError, load_audio, save_mp3, save_wav

needs_ffmpeg = pytest.mark.skipif(io_audio.find_ffmpeg() is None, reason="ffmpeg no instalado")


def _tone(freq: float = 110.0, sr: int = 22050, dur: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@needs_ffmpeg
def test_mp3_round_trip_preserves_pitch_and_length(tmp_path: Path) -> None:
    sr = 22050
    y = _tone(110.0, sr, 1.0)
    mp3 = save_mp3(tmp_path / "tono.mp3", y, sr)
    y2, sr2 = load_audio(mp3, sr=sr)
    assert sr2 == sr
    assert y2.dtype == np.float32
    # El MP3 añade algo de relleno (encoder delay): la duración debe ser parecida.
    assert abs(len(y2) - len(y)) < 0.1 * sr
    spectrum = np.abs(np.fft.rfft(y2))
    peak_hz = np.argmax(spectrum) * sr / len(y2)
    assert abs(peak_hz - 110.0) < 2.0


@needs_ffmpeg
def test_load_resamples_and_mixes_to_mono(tmp_path: Path) -> None:
    wav = save_wav(tmp_path / "tono.wav", _tone(220.0, 44100, 0.5), 44100)
    y, sr = load_audio(wav, sr=22050)
    assert sr == 22050
    assert abs(len(y) - 0.5 * 22050) < 50


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "no_existe.mp3")


def test_missing_ffmpeg_gives_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = save_wav(tmp_path / "tono.wav", _tone(), 22050)
    monkeypatch.setattr(io_audio.shutil, "which", lambda name: None)
    monkeypatch.delenv(io_audio.FFMPEG_ENV_VAR, raising=False)
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg"):
        load_audio(wav)


@needs_ffmpeg
def test_corrupt_file_raises_audio_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "corrupto.mp3"
    bad.write_bytes(b"esto no es un mp3")
    with pytest.raises(AudioLoadError):
        load_audio(bad)
