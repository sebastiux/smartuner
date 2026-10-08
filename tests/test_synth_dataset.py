"""Pruebas del dataset sintético: piezas, MIDI, Karplus-Strong, fluidsynth y ground truth.

Las pruebas que codifican MP3 requieren ffmpeg y las de fluidsynth requieren
el ejecutable y un soundfont GM; si faltan se omiten (``skipif``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from src import io_audio
from src import synth_dataset as sd
from src.config import DATA_DIR, OPEN_STRING_MIDI, CancelledError
from src.pitch import midi_to_hz
from src.synth_dataset import (
    LEAD_IN_S,
    PIECES,
    TAIL_S,
    GTNote,
    generate_dataset,
    generate_piece,
    gt_path_for,
    load_ground_truth,
    load_ground_truth_meta,
    piece_notes,
    render_fluidsynth,
    render_karplus_strong,
    save_ground_truth,
    write_midi,
)

SR = 22050

needs_ffmpeg = pytest.mark.skipif(io_audio.find_ffmpeg() is None, reason="ffmpeg no instalado")
needs_fluidsynth = pytest.mark.skipif(not sd.fluidsynth_available(), reason="fluidsynth o soundfont GM no disponibles")


def _expected_duration(notes: list[GTNote]) -> float:
    """Duración esperada del audio: último offset del GT + la cola TAIL_S."""
    return notes[-1].offset_s + TAIL_S


def _fft_peak_hz(y: np.ndarray, sr: int, pad: int = 8) -> float:
    """Frecuencia del pico del espectro (ventana de Hann, zero-padding ×pad)."""
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size), n=pad * y.size))
    return float(np.argmax(spectrum) * sr / (pad * y.size))


def _cents(f: float, ref: float) -> float:
    """Diferencia en cents entre ``f`` y ``ref`` (1200·log2(f/ref))."""
    return float(1200.0 * np.log2(f / ref))


# ---------------------------------------------------------------------------
# Definición de las piezas y tiempos
# ---------------------------------------------------------------------------


def test_pieces_are_valid_bass_positions() -> None:
    """Las 3 piezas usan cuerdas reales, trastes 0–12 y duraciones positivas."""
    assert set(PIECES) == {"cromatica", "linea_simple", "riff_saltos"}
    for name, positions in PIECES.items():
        assert positions, name
        for string, fret, beats in positions:
            assert string in OPEN_STRING_MIDI
            assert 0 <= fret <= 12
            assert beats > 0


def test_cromatica_is_chromatic_from_e1_to_c3() -> None:
    """'cromatica' sube semitono a semitono de E1 a C3 en corcheas (0.3 s a 100 bpm) con legato 0.9."""
    notes = piece_notes("cromatica", bpm=100.0, legato=0.9)
    assert [n.midi for n in notes] == list(range(28, 49))  # E1 … C3, semitono a semitono
    assert notes[0].label == "E-0" and notes[5].label == "A-0" and notes[-1].label == "G-5"
    onsets = np.array([n.onset_s for n in notes])
    assert onsets[0] == pytest.approx(LEAD_IN_S)
    np.testing.assert_allclose(np.diff(onsets), 0.3)  # corchea a 100 bpm = 0.5 · 60/100 s
    for n in notes:
        assert n.offset_s - n.onset_s == pytest.approx(0.9 * 0.3)


def test_linea_simple_timing_and_pitches() -> None:
    """'linea_simple' tiene 16 negras con los pitches y tiempos esperados (la última dura 2 tiempos)."""
    notes = piece_notes("linea_simple", bpm=100.0)
    assert len(notes) == 16
    assert [n.label for n in notes[:4]] == ["A-0", "A-4", "D-2", "A-4"]
    assert [n.midi for n in notes[:4]] == [33, 37, 40, 37]  # A1, C#2, E2, C#2
    np.testing.assert_allclose(np.diff([n.onset_s for n in notes]), 0.6)  # negras
    last = notes[-1]
    assert last.label == "A-0"
    assert last.onset_s == pytest.approx(LEAD_IN_S + 15 * 0.6)
    assert last.offset_s - last.onset_s == pytest.approx(0.9 * 1.2)  # 2 tiempos


def test_riff_saltos_sixteenths_and_midi() -> None:
    """'riff_saltos' incluye semicorcheas de 0.15 s y saltos de cuerda con el mismo pitch (D-7 y G-2 = A2)."""
    notes = piece_notes("riff_saltos", bpm=100.0)
    assert len(notes) == 20
    for note, (string, fret, beats) in zip(notes, PIECES["riff_saltos"]):
        assert note.midi == OPEN_STRING_MIDI[string] + fret
        assert note.offset_s - note.onset_s == pytest.approx(0.9 * beats * 0.6)
    # Las semicorcheas duran 0.15 s nominales a 100 bpm.
    assert notes[4].label == "E-3" and notes[4].onset_s == pytest.approx(LEAD_IN_S + 4 * 0.3)
    assert notes[5].onset_s - notes[4].onset_s == pytest.approx(0.15)
    # Saltos de cuerda con el mismo pitch en otra cuerda: D-7 y G-2 son ambos A2 (45).
    assert notes[12].label == "D-7" and notes[12].midi == notes[7].midi == 45


def test_piece_notes_scales_with_tempo_and_legato() -> None:
    """El tempo escala los tiempos (120 bpm → corcheas de 0.25 s) y legato 1 elimina el hueco entre notas."""
    fast = piece_notes("cromatica", bpm=120.0, legato=1.0)
    np.testing.assert_allclose(np.diff([n.onset_s for n in fast]), 0.25)
    assert fast[0].offset_s == pytest.approx(fast[1].onset_s)  # legato 1: sin hueco


@pytest.mark.parametrize("kwargs", [{"name": "no_existe"}, {"name": "cromatica", "bpm": 0.0},
                                    {"name": "cromatica", "bpm": float("inf")},
                                    {"name": "cromatica", "bpm": float("nan")},
                                    {"name": "cromatica", "legato": 0.0}, {"name": "cromatica", "legato": 1.5}])
def test_piece_notes_rejects_invalid_arguments(kwargs: dict) -> None:
    """Pieza desconocida, tempo no positivo o no finito (inf/nan) y legato fuera de (0, 1] → ValueError."""
    with pytest.raises(ValueError):
        piece_notes(**kwargs)


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------


def test_write_midi_round_trip_with_pretty_midi(tmp_path: Path) -> None:
    """El MIDI escrito se relee con pretty_midi: bajo eléctrico (programa 33), mismos pitches y tiempos."""
    import pretty_midi

    notes = piece_notes("riff_saltos")
    path = write_midi(notes, tmp_path / "sub" / "riff.mid")
    assert path.is_file()
    pm = pretty_midi.PrettyMIDI(str(path))
    assert len(pm.instruments) == 1
    inst = pm.instruments[0]
    assert inst.program == 33
    assert pretty_midi.program_to_instrument_name(inst.program) == "Electric Bass (finger)"
    read = sorted(inst.notes, key=lambda n: n.start)
    assert [n.pitch for n in read] == [n.midi for n in notes]
    np.testing.assert_allclose([n.start for n in read], [n.onset_s for n in notes], atol=2e-3)
    np.testing.assert_allclose([n.end for n in read], [n.offset_s for n in notes], atol=2e-3)
    assert all(n.velocity == 100 for n in read)


# ---------------------------------------------------------------------------
# Karplus-Strong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("midi", [28, 33, 43, 48])  # E1, A1, G2, C3
def test_karplus_strong_pitch_and_length(midi: int) -> None:
    """Karplus-Strong genera la nota afinada (< 15 cents), con silencio antes del onset y la duración pedida."""
    note = GTNote(onset_s=0.1, offset_s=1.1, midi=midi, string="E", fret=midi - 28)
    y = render_karplus_strong([note], sr=SR, seed=0, tail_s=0.2)
    assert y.dtype == np.float32
    assert y.size == int(round(1.3 * SR))
    assert np.max(np.abs(y)) == pytest.approx(sd.OUTPUT_PEAK, rel=1e-5)
    assert np.all(y[: int(0.1 * SR) - 1] == 0.0)  # silencio antes del onset
    sounding = y[int(0.1 * SR):int(1.1 * SR)]
    f0 = midi_to_hz(midi)
    # La fundamental domina (excitación con pendiente 1/h) y está afinada a < 15 cents.
    assert abs(_cents(_fft_peak_hz(sounding, SR), f0)) < 15.0


def test_karplus_strong_is_reproducible_and_seed_dependent() -> None:
    """Misma semilla → mismo audio; otra semilla → otro audio."""
    notes = piece_notes("linea_simple")[:3]
    a = render_karplus_strong(notes, sr=SR, seed=3)
    b = render_karplus_strong(notes, sr=SR, seed=3)
    c = render_karplus_strong(notes, sr=SR, seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.size == int(round(_expected_duration(notes) * SR))


def test_karplus_strong_notes_start_at_their_onsets() -> None:
    """Cada nota empieza a sonar en su onset (energía en los primeros 20 ms) y antes hay silencio."""
    notes = piece_notes("riff_saltos")
    y = render_karplus_strong(notes, sr=SR)
    assert np.max(np.abs(y[: int(LEAD_IN_S * SR)])) == 0.0
    for n in notes:
        start = int(round(n.onset_s * SR))
        assert np.sqrt(np.mean(y[start:start + int(0.02 * SR)] ** 2)) > 0.05


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def test_ground_truth_round_trip(tmp_path: Path) -> None:
    """save_ground_truth / load_ground_truth conservan notas y metadatos con el formato JSON documentado."""
    notes = piece_notes("riff_saltos")
    meta = {"piece": "riff_saltos", "bpm": 100.0}
    path = save_ground_truth(notes, tmp_path / "riff.gt.json", meta=meta)
    loaded = load_ground_truth(path)
    assert [(n.midi, n.string, n.fret) for n in loaded] == [(n.midi, n.string, n.fret) for n in notes]
    np.testing.assert_allclose([n.onset_s for n in loaded], [n.onset_s for n in notes], atol=1e-6)
    np.testing.assert_allclose([n.offset_s for n in loaded], [n.offset_s for n in notes], atol=1e-6)
    assert load_ground_truth_meta(path) == meta
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"meta", "notes"}
    assert set(raw["notes"][0]) == {"onset_s", "offset_s", "midi", "string", "fret"}


@pytest.mark.parametrize("content", ["no es json", '{"meta": {}}', '{"notes": [{"onset_s": 1.0}]}'])
def test_load_ground_truth_rejects_malformed_files(tmp_path: Path, content: str) -> None:
    """JSON roto, sin la lista ``notes`` o con notas incompletas → ValueError (no KeyError)."""
    path = tmp_path / "malo.gt.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(path)


@pytest.mark.parametrize("note, message", [
    ({"onset_s": 0.5, "offset_s": 0.9, "midi": 50, "string": "B", "fret": 10}, "cuerda 'B' desconocida"),
    ({"onset_s": 0.5, "offset_s": 0.9, "midi": 31, "string": "A", "fret": -2}, "traste negativo"),
    ({"onset_s": 0.5, "offset_s": 0.9, "midi": 40, "string": "A", "fret": 5}, "no coincide con A-5"),
    ({"onset_s": 0.9, "offset_s": 0.5, "midi": 33, "string": "A", "fret": 0}, "anterior al onset"),
    ({"onset_s": float("nan"), "offset_s": 0.5, "midi": 33, "string": "A", "fret": 0}, "tiempos inválidos"),
])
def test_load_ground_truth_rejects_incoherent_notes(tmp_path: Path, note: dict, message: str) -> None:
    """Regresión: una cuerda desconocida, un traste negativo o MIDI ≠ cuerda + traste se rechazan al cargar.

    Antes una cuerda "B" llegaba hasta las gráficas (``STRING_ORDER.index``) y
    un MIDI incoherente hacía que las precisiones de pitch y posición se
    contradijeran sin aviso.
    """
    path = tmp_path / "incoherente.gt.json"
    path.write_text(json.dumps({"notes": [{"onset_s": 0.1, "offset_s": 0.4, "midi": 28, "string": "E", "fret": 0},
                                          note]}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Nota 1 inválida.*{message}"):
        load_ground_truth(path)


def test_gt_path_for() -> None:
    """El ground truth de 'x.mp3' es 'x.gt.json' en la misma carpeta."""
    assert gt_path_for("data/synthetic/cromatica.mp3") == Path("data/synthetic/cromatica.gt.json")


# ---------------------------------------------------------------------------
# fluidsynth / soundfont
# ---------------------------------------------------------------------------


def test_find_soundfont_honours_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """find_soundfont respeta la variable de entorno y, si apunta a un archivo inexistente, busca en las rutas habituales."""
    fake = tmp_path / "mi_banco.sf2"
    fake.write_bytes(b"RIFF")
    monkeypatch.setenv(sd.SOUNDFONT_ENV_VAR, str(fake))
    assert sd.find_soundfont() == fake
    # Si la variable apunta a un archivo inexistente se sigue buscando en las rutas habituales.
    monkeypatch.setenv(sd.SOUNDFONT_ENV_VAR, str(tmp_path / "no_existe.sf2"))
    found = sd.find_soundfont()
    assert found is None or found.is_file()


def test_fluidsynth_available_is_bool() -> None:
    """fluidsynth_available() devuelve un booleano."""
    assert isinstance(sd.fluidsynth_available(), bool)


def test_render_fluidsynth_raises_runtime_error_on_bad_inputs(tmp_path: Path) -> None:
    """fluidsynth con soundfont o MIDI inexistentes lanza RuntimeError en español."""
    midi = write_midi(piece_notes("cromatica")[:2], tmp_path / "x.mid")
    with pytest.raises(RuntimeError):
        render_fluidsynth(midi, tmp_path / "x.wav", soundfont=tmp_path / "no_existe.sf2")
    with pytest.raises(RuntimeError):
        render_fluidsynth(tmp_path / "no_existe.mid", tmp_path / "y.wav")


def test_method_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """'auto' elige fluidsynth si está disponible y Karplus-Strong si no; pedir fluidsynth sin tenerlo es un error."""
    monkeypatch.setattr(sd, "fluidsynth_available", lambda: False)
    assert sd._resolve_method("auto") == "karplus-strong"
    with pytest.raises(RuntimeError):
        sd._resolve_method("fluidsynth")
    monkeypatch.setattr(sd, "fluidsynth_available", lambda: True)
    assert sd._resolve_method("auto") == "fluidsynth"
    with pytest.raises(ValueError):
        sd._resolve_method("sinte")


# ---------------------------------------------------------------------------
# Generación completa
# ---------------------------------------------------------------------------


def _check_generated_item(item: sd.DatasetItem, out_dir: Path, method: str) -> tuple[np.ndarray, list[GTNote]]:
    """Comprobaciones comunes: archivos, ausencia de WAV, MP3 decodificable y ground truth."""
    assert item.method == method
    for p in (item.mp3_path, item.gt_path, item.midi_path):
        assert p.is_file() and p.parent == out_dir
    assert item.gt_path == gt_path_for(item.mp3_path)
    assert not list(out_dir.glob("*.wav"))
    notes = piece_notes(item.name)
    gt = load_ground_truth(item.gt_path)
    assert [(n.string, n.fret, n.midi) for n in gt] == [(n.string, n.fret, n.midi) for n in notes]
    np.testing.assert_allclose([n.onset_s for n in gt], [n.onset_s for n in notes], atol=1e-6)
    meta = load_ground_truth_meta(item.gt_path)
    assert {"piece", "bpm", "sr", "method", "tuning", "program"} <= set(meta)
    assert meta["piece"] == item.name and meta["method"] == method and meta["program"] == 33
    y, sr = io_audio.load_audio(item.mp3_path, sr=SR)
    assert sr == SR
    assert abs(y.size / SR - _expected_duration(notes)) < 0.05
    return y, notes


@needs_ffmpeg
def test_generate_piece_karplus_strong(tmp_path: Path) -> None:
    """generate_piece con Karplus-Strong escribe MIDI, MP3 y GT coherentes (silencio inicial y nota afinada tras el MP3)."""
    item = generate_piece("riff_saltos", tmp_path, method="karplus-strong")
    y, notes = _check_generated_item(item, tmp_path, "karplus-strong")
    lead = y[: int((LEAD_IN_S - 0.05) * SR)]
    assert np.max(np.abs(lead)) < 0.01  # el silencio inicial sobrevive al MP3
    first = y[int(notes[0].onset_s * SR):int(notes[0].offset_s * SR)]
    assert abs(_cents(_fft_peak_hz(first, SR), midi_to_hz(notes[0].midi))) < 30.0


@needs_ffmpeg
@needs_fluidsynth
def test_generate_piece_fluidsynth(tmp_path: Path) -> None:
    """generate_piece con fluidsynth: el WAV temporal se borra, el ataque cae en el onset del GT y pYIN lee la nota correcta."""
    import librosa

    item = generate_piece("linea_simple", tmp_path, method="fluidsynth", tmp_dir=tmp_path / "tmp")
    y, notes = _check_generated_item(item, tmp_path, "fluidsynth")
    assert not list((tmp_path / "tmp").iterdir())  # el WAV temporal se borró
    # El primer ataque aparece en el onset del ground truth (± 50 ms).
    hop = 128
    rms = np.sqrt(np.convolve(y ** 2, np.ones(hop) / hop, mode="same"))
    first_sound = np.nonzero(rms > 0.05 * rms.max())[0][0] / SR
    assert abs(first_sound - notes[0].onset_s) < 0.05
    # El soundfont toca la nota correcta (A1 = 55 Hz) según pYIN.
    seg = y[int((notes[0].onset_s + 0.05) * SR):int(notes[0].offset_s * SR)]
    f0, voiced, _ = librosa.pyin(y=seg, sr=SR, fmin=35.0, fmax=250.0, frame_length=2048, hop_length=256)
    assert np.any(voiced)
    assert abs(librosa.hz_to_midi(np.nanmedian(f0[voiced])) - notes[0].midi) < 0.5


def test_generate_piece_rejects_unknown_piece_or_method(tmp_path: Path) -> None:
    """Una pieza o un método de síntesis desconocidos son un error."""
    with pytest.raises(ValueError):
        generate_piece("no_existe", tmp_path)
    with pytest.raises(ValueError):
        generate_piece("cromatica", tmp_path, method="sinte")


@needs_ffmpeg
def test_generate_dataset_reports_progress(tmp_path: Path) -> None:
    """generate_dataset informa progreso monótono de 0 a 1 y escribe .mid, .mp3 y .gt.json de cada pieza."""
    calls: list[tuple[float, str]] = []
    items = generate_dataset(tmp_path, method="karplus-strong", pieces=["linea_simple", "cromatica"],
                             progress=lambda f, msg: calls.append((f, msg)))
    assert [it.name for it in items] == ["linea_simple", "cromatica"]
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{n}{ext}" for n in ("linea_simple", "cromatica") for ext in (".mid", ".mp3", ".gt.json")
    )


@needs_ffmpeg
def test_generate_dataset_can_be_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Evento ya activo: no se genera nada.
    """Cancelar antes de empezar no genera nada; a mitad, la pieza en curso se completa y la siguiente no empieza."""
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        generate_dataset(tmp_path / "a", method="karplus-strong", cancel=cancel)
    assert not (tmp_path / "a").exists()

    # Cancelación a mitad (el usuario pulsa «Cancelar» mientras se genera la
    # primera pieza): esa pieza queda completa y la segunda no se empieza.
    cancel = threading.Event()
    real_generate_piece = sd.generate_piece

    def generate_then_cancel(*args: object, **kwargs: object) -> sd.DatasetItem:
        """generate_piece real seguido de la cancelación (el usuario pulsa «Cancelar» durante la primera pieza)."""
        item = real_generate_piece(*args, **kwargs)  # type: ignore[arg-type]
        cancel.set()
        return item

    monkeypatch.setattr(sd, "generate_piece", generate_then_cancel)
    with pytest.raises(CancelledError):
        generate_dataset(tmp_path / "b", method="karplus-strong", pieces=["cromatica", "riff_saltos"], cancel=cancel)
    assert (tmp_path / "b" / "cromatica.mp3").is_file()
    assert not (tmp_path / "b" / "riff_saltos.mp3").exists()


def test_generate_dataset_rejects_unknown_pieces(tmp_path: Path) -> None:
    """Una pieza desconocida en la lista es un error."""
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, pieces=["cromatica", "no_existe"])


# ---------------------------------------------------------------------------
# Dataset incluido en el repositorio
# ---------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("name", sorted(PIECES))
def test_committed_dataset_matches_definitions(name: str) -> None:
    """El dataset de data/synthetic coincide con las definiciones de las piezas (notas, MIDI y duración)."""
    folder = DATA_DIR / "synthetic"
    mp3 = folder / f"{name}.mp3"
    if not mp3.is_file():
        pytest.skip("dataset no generado todavía (python -c 'from src.synth_dataset import generate_dataset; generate_dataset()')")
    meta = load_ground_truth_meta(gt_path_for(mp3))
    notes = piece_notes(name, bpm=meta.get("bpm", 100.0), legato=meta.get("legato", 0.9))
    gt = load_ground_truth(gt_path_for(mp3))
    assert [(n.string, n.fret, n.midi) for n in gt] == [(n.string, n.fret, n.midi) for n in notes]
    assert (folder / f"{name}.mid").is_file()
    y, _ = io_audio.load_audio(mp3, sr=SR)
    assert abs(y.size / SR - _expected_duration(notes)) < 0.05
