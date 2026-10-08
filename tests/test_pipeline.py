"""Pruebas de la orquestación de las etapas 1–6 (:mod:`src.pipeline`).

Se genera en ``tmp_path`` un MP3 sintético con Karplus-Strong (pieza
``cromatica``: 21 notas de E1 a C3, con su ``.gt.json``) y se comprueba que
:func:`analyze` encuentra una nota por nota del ground truth con la f0
correcta, construye los entornos bandit y carga el ground truth. También se
prueban el progreso, la cancelación, la separación (simulada) y
:func:`rebuild_segment_data`. Requieren ffmpeg (se omiten si falta).
"""

from __future__ import annotations

import copy
import logging
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from src import io_audio
from src.config import CancelledError, Config, ProgressCallback
from src.experiments import match_segments_to_gt
from src.pipeline import AnalysisResult, analyze, rebuild_segment_data
from src.pitch import hz_to_midi
from src.synth_dataset import TAIL_S, DatasetItem, generate_piece

pytestmark = pytest.mark.skipif(io_audio.find_ffmpeg() is None, reason="ffmpeg no instalado")

#: Tolerancia de onset para emparejar segmentos con el ground truth en estas pruebas (s).
ONSET_TOL_S = 0.05


@pytest.fixture(scope="module")
def ks_item(tmp_path_factory: pytest.TempPathFactory) -> DatasetItem:
    """Pieza ``cromatica`` sintetizada con Karplus-Strong (determinista: semilla fija)."""
    out = tmp_path_factory.mktemp("ks_dataset")
    return generate_piece("cromatica", out, method="karplus-strong", seed=0)


@pytest.fixture(scope="module")
def analysis(ks_item: DatasetItem) -> AnalysisResult:
    """Análisis completo de la pieza Karplus-Strong con la configuración por defecto."""
    return analyze(ks_item.mp3_path, Config())


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_finds_the_ground_truth_notes(analysis: AnalysisResult, ks_item: DatasetItem) -> None:
    """analyze encuentra ≈ una nota por nota del GT (≥ 90 % emparejadas) y pYIN da el MIDI correcto en ≥ 90 % de ellas."""
    gt = analysis.ground_truth
    assert gt is not None and len(gt) == 21                       # GT cargado del .gt.json
    kept = analysis.kept
    assert abs(len(kept) - len(gt)) <= 1                          # ≈ una nota por segmento
    matches = match_segments_to_gt(kept, gt, ONSET_TOL_S)
    matched = [(k, j) for k, j in enumerate(matches) if j is not None]
    assert len(matched) >= 0.9 * len(gt)
    # f0 de pYIN → MIDI correcto en (casi) todas las notas emparejadas.
    correct = sum(1 for k, j in matched
                  if kept[j].f0_hz is not None and round(hz_to_midi(kept[j].f0_hz)) == gt[k].midi)
    assert correct >= 0.9 * len(matched)


def test_analyze_result_fields(analysis: AnalysisResult, ks_item: DatasetItem) -> None:
    """Campos del resultado: rutas, sr, señales de igual longitud normalizadas a 0.99, onsets crecientes y frames alineados."""
    cfg = Config()
    assert analysis.path == ks_item.mp3_path and analysis.source_path == ks_item.mp3_path
    assert analysis.sr == cfg.audio.sample_rate
    n = len(analysis.y_raw)
    assert len(analysis.y_analysis) == len(analysis.y_spectral) == n
    assert analysis.duration_s == pytest.approx(n / analysis.sr)
    # Duración = último offset del GT + cola (TAIL_S), como la escribe synth_dataset.
    assert analysis.duration_s == pytest.approx(analysis.ground_truth[-1].offset_s + TAIL_S, abs=0.05)
    # Ambas señales normalizadas a pico 0.99.
    assert np.max(np.abs(analysis.y_analysis)) == pytest.approx(0.99, abs=1e-3)
    assert np.max(np.abs(analysis.y_spectral)) == pytest.approx(0.99, abs=1e-3)
    assert np.all(np.diff(analysis.onsets_s) > 0)
    assert [s.index for s in analysis.segments] == list(range(len(analysis.segments)))
    # Mismo hop en pitch y espectro (frames alineados).
    hop = cfg.segmentation.hop_length
    assert analysis.spectrum.hop_length == hop and analysis.spectrum.kind == cfg.env.spectrum
    np.testing.assert_allclose(analysis.pitch_track.times_s[:5], np.arange(5) * hop / analysis.sr)


def test_analyze_builds_bandit_data(analysis: AnalysisResult) -> None:
    """Un SegmentBanditData por segmento conservado, con la poda ±k alrededor de la f0 y saliencia en [0, 1]."""
    data = analysis.segment_data
    assert data and len(data) == len(analysis.kept)
    for pos, (d, seg) in enumerate(zip(data, analysis.kept)):
        assert d.position == pos and d.segment is seg
        assert d.n_arms >= 1 and d.salience.shape == (d.frame_indices.size, d.n_arms)
        assert np.all((d.salience >= 0) & (d.salience <= 1))
        if seg.f0_hz is not None:
            # Poda ±k semitonos alrededor de la f0 de pYIN (k=2 por defecto).
            target = round(hz_to_midi(seg.f0_hz))
            assert all(abs(a.midi - target) <= 2 for a in d.arms)
    # El brazo de mayor saliencia media tiene el pitch correcto en la mayoría de las notas.
    gt = analysis.ground_truth
    matches = match_segments_to_gt(analysis.kept, gt, ONSET_TOL_S)
    good = sum(1 for k, j in enumerate(matches)
               if j is not None and data[j].arms[int(np.argmax(data[j].mean_salience))].midi == gt[k].midi)
    assert good >= 0.8 * len(gt)


def test_analyze_keeps_deep_copy_of_config(ks_item: DatasetItem) -> None:
    """El resultado guarda una COPIA de la configuración: modificarla después no altera el análisis."""
    cfg = Config()
    cfg.env.k_semitones = 1
    res = analyze(ks_item.mp3_path, cfg)
    assert res.config == cfg and res.config is not cfg
    cfg.env.k_semitones = 5            # modificar después no altera el análisis
    assert res.config.env.k_semitones == 1
    assert all(d.n_arms <= 4 * 3 for d in res.segment_data)


def test_analyze_progress_and_logging(ks_item: DatasetItem, caplog: pytest.LogCaptureFixture) -> None:
    """El progreso va de 0 a 1 de forma monótona y el log narra las etapas en español."""
    calls: list[tuple[float, str]] = []
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        analyze(ks_item.mp3_path, Config(), progress=lambda f, m: calls.append((f, m)))
    fractions = [f for f, _ in calls]
    assert fractions[0] == 0.0 and fractions[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    messages = " ".join(m for _, m in calls)
    for stage in ("Etapa 1/6", "Etapa 3/6", "Etapa 4/6", "Etapa 5/6", "Etapa 6/6"):
        assert stage in messages
    logged = caplog.text
    assert "Separación: omitida" in logged and "Ground truth cargado" in logged


def test_analyze_cancel_between_stages(ks_item: DatasetItem) -> None:
    """Activar ``cancel`` detiene el análisis antes de la siguiente etapa (pYIN nunca empieza)."""
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        analyze(ks_item.mp3_path, Config(), cancel=cancel)

    cancel2 = threading.Event()
    seen: list[str] = []

    def progress(fraction: float, message: str) -> None:
        """Callback que simula al usuario cancelando durante la segmentación."""
        seen.append(message)
        if "Etapa 4/6" in message:
            cancel2.set()               # el usuario cancela durante la segmentación

    with pytest.raises(CancelledError):
        analyze(ks_item.mp3_path, Config(), progress=progress, cancel=cancel2)
    assert not any("Etapa 5/6" in m for m in seen)   # pYIN nunca empezó


def test_analyze_without_ground_truth(ks_item: DatasetItem, tmp_path: Path) -> None:
    """Sin .gt.json junto al audio el análisis funciona y ground_truth es None."""
    lonely = tmp_path / "sin_gt.mp3"
    shutil.copy(ks_item.mp3_path, lonely)
    res = analyze(lonely, Config())
    assert res.ground_truth is None
    assert res.segment_data


def test_analyze_quiet_noise_gives_no_notes(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Regresión: ruido a −90 dBFS (sin ninguna nota) ya no se convierte en una tablatura inventada.

    La normalización de pico y los umbrales RELATIVOS de la segmentación
    convertían el ruido en "música" a escala completa (25 notas falsas).
    """
    rng = np.random.default_rng(0)
    noise = io_audio.save_wav(tmp_path / "ruido.wav", 10 ** (-90 / 20) * rng.standard_normal(3 * 22050), 22050)
    with caplog.at_level(logging.WARNING, logger="src.pipeline"):
        res = analyze(noise, Config())
    assert res.segments == [] and res.segment_data == [] and res.onsets_s.size == 0
    assert "prácticamente silencio" in caplog.text


def test_track_level_dbfs_of_real_piece_is_far_above_threshold(ks_item: DatasetItem) -> None:
    """Una pieza normal está decenas de dB por encima del umbral de silencio (no se descarta)."""
    from src.pipeline import MIN_TRACK_RMS_DBFS, track_level_dbfs

    y, _ = io_audio.load_audio(ks_item.mp3_path, sr=22050)
    assert track_level_dbfs(y) > MIN_TRACK_RMS_DBFS + 30


def test_analyze_validates_config_before_decoding(tmp_path: Path) -> None:
    """Regresión: una configuración inválida falla ANTES de la etapa 1 (ni siquiera se busca el audio)."""
    cfg = Config()
    cfg.pitch.fmin_hz, cfg.pitch.fmax_hz = 300.0, 100.0
    with pytest.raises(ValueError, match="fmin_hz < fmax_hz"):
        analyze(tmp_path / "no_existe.mp3", cfg)


def test_analyze_missing_file_raises(tmp_path: Path) -> None:
    """Un archivo inexistente da AudioLoadError."""
    with pytest.raises(io_audio.AudioLoadError):
        analyze(tmp_path / "no_existe.mp3", Config())


def test_analyze_with_separation_uses_stem(ks_item: DatasetItem, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con ``separate_bass`` se analiza el stem que devuelve Demucs (aquí simulado) y el GT
    se sigue buscando junto al archivo ORIGINAL."""
    from src import separation

    y, sr = io_audio.load_audio(ks_item.mp3_path, sr=22050)
    stem = io_audio.save_wav(tmp_path / "stem_bass.wav", y, sr)
    calls: dict[str, object] = {}

    def fake_separate(path: str | Path, model: str = "htdemucs", cache_dir: Path | None = None,
                      progress: ProgressCallback | None = None, cancel: threading.Event | None = None,
                      env: dict[str, str] | None = None) -> Path:
        """Demucs simulado: informa progreso y devuelve el stem ya preparado."""
        calls["path"], calls["model"] = Path(path), model
        if progress is not None:
            progress(0.5, "Demucs: 50 %")
            progress(1.0, "Demucs: separación terminada")
        return stem

    monkeypatch.setattr(separation, "separate_bass", fake_separate)
    cfg = Config()
    cfg.audio.separate_bass = True
    fractions: list[float] = []
    res = analyze(ks_item.mp3_path, cfg, progress=lambda f, m: fractions.append(f))
    assert calls == {"path": ks_item.mp3_path, "model": cfg.audio.demucs_model}
    assert res.path == ks_item.mp3_path and res.source_path == stem
    assert res.ground_truth is not None and len(res.ground_truth) == 21
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))


def test_analyze_separation_cancel_propagates(ks_item: DatasetItem, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelar durante Demucs se traduce en CancelledError del análisis."""
    from src import separation

    def cancelled(*args: object, **kwargs: object) -> Path:
        """Demucs simulado que el usuario cancela."""
        raise separation.SeparationCancelledError("cancelado")

    monkeypatch.setattr(separation, "separate_bass", cancelled)
    cfg = Config()
    cfg.audio.separate_bass = True
    with pytest.raises(CancelledError):
        analyze(ks_item.mp3_path, cfg)


# ---------------------------------------------------------------------------
# rebuild_segment_data
# ---------------------------------------------------------------------------


def test_rebuild_with_new_env_params_reuses_spectrum(analysis: AnalysisResult) -> None:
    """Cambiar k, β o el agente reconstruye los entornos sin recalcular el espectro ni re-anotar la f0; el original no cambia."""
    before = copy.deepcopy([d.arms for d in analysis.segment_data])
    cfg = copy.deepcopy(analysis.config)
    cfg.env.k_semitones = 0
    cfg.env.beta = 0.0
    cfg.agent.epsilon = 0.3
    new = rebuild_segment_data(analysis, cfg)
    assert new is not analysis
    assert new.spectrum is analysis.spectrum                 # mismo espectro: no se recalcula
    assert new.segments is analysis.segments                 # f0 sin cambios: no se re-anota
    assert len(new.segment_data) == len(analysis.segment_data)
    for d in new.segment_data:
        if d.segment.f0_hz is not None:
            assert len({a.midi for a in d.arms}) == 1        # k=0: solo posiciones del mismo pitch
    assert new.config.env.k_semitones == 0 and new.config.agent.epsilon == 0.3
    # El análisis original no cambia.
    assert [d.arms for d in analysis.segment_data] == before
    assert analysis.config.env.k_semitones == 2


def test_rebuild_recomputes_spectrum_when_needed(analysis: AnalysisResult) -> None:
    """El espectro se recalcula solo si cambian su tipo o sus parámetros (n_fft con STFT, bins/oct con CQT)."""
    assert analysis.spectrum.kind == "stft"                  # valor por defecto (n_fft = 8192)
    cfg = copy.deepcopy(analysis.config)
    cfg.env.spectrum = "cqt"
    cqt = rebuild_segment_data(analysis, cfg)
    assert cqt.spectrum is not analysis.spectrum and cqt.spectrum.kind == "cqt"
    assert cqt.spectrum.hop_length == analysis.spectrum.hop_length
    assert analysis.spectrum.kind == "stft"                  # el original no cambia
    # n_fft solo importa con STFT: con CQT no se recalcula...
    cfg2 = copy.deepcopy(cqt.config)
    cfg2.env.n_fft = 2048
    assert rebuild_segment_data(cqt, cfg2).spectrum is cqt.spectrum
    # ...y con STFT sí.
    cfg3 = copy.deepcopy(analysis.config)
    cfg3.env.n_fft = 4096
    stft = rebuild_segment_data(analysis, cfg3)
    assert stft.spectrum is not analysis.spectrum and stft.spectrum.freqs_hz[1] == pytest.approx(22050 / 4096)
    # Los parámetros de la CQT no afectan a la STFT.
    cfg4 = copy.deepcopy(analysis.config)
    cfg4.env.bins_per_octave = 24
    assert rebuild_segment_data(analysis, cfg4).spectrum is analysis.spectrum


def test_rebuild_reannotates_f0_without_touching_original(analysis: AnalysisResult) -> None:
    """Cambiar min_voiced_ratio re-anota la f0 sobre una copia de los segmentos (sin f0 → 52 brazos)."""
    cfg = copy.deepcopy(analysis.config)
    cfg.pitch.min_voiced_ratio = 1.01         # imposible → todas las f0 desconocidas
    new = rebuild_segment_data(analysis, cfg)
    assert new.segments is not analysis.segments
    assert all(s.f0_hz is None for s in new.kept)
    assert all(d.n_arms == 52 for d in new.segment_data)     # sin f0 no se poda
    assert any(s.f0_hz is not None for s in analysis.kept)   # el original conserva sus f0


def test_rebuild_warns_about_changes_that_need_reanalysis(analysis: AnalysisResult, caplog: pytest.LogCaptureFixture) -> None:
    """Los cambios de etapas 1–5 (p. ej. δ de onsets) no se aplican al reconstruir y se avisa en el log."""
    cfg = copy.deepcopy(analysis.config)
    cfg.segmentation.onset_delta = 0.3
    with caplog.at_level(logging.WARNING, logger="src.pipeline"):
        new = rebuild_segment_data(analysis, cfg)
    assert "segmentation.onset_delta" in caplog.text
    assert new.config.segmentation.onset_delta == analysis.config.segmentation.onset_delta
    assert len(new.segments) == len(analysis.segments)
