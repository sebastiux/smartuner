"""Pruebas de :mod:`src.config`: serialización, conversión de tipos y validación.

La configuración llega de un JSON escrito a mano o guardado por la GUI; un
valor erróneo debe fallar al CARGARLA, con un mensaje en español, y no con
una traza a mitad del análisis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    ALGO_COLORS,
    ALGO_LABELS,
    ALGO_LINESTYLES,
    ALGO_MARKERS,
    ALGORITHMS,
    PARAM_SPECS,
    SUPPORTED_SAMPLE_RATE,
    Config,
)


def test_defaults_are_valid_and_round_trip(tmp_path: Path) -> None:
    """Los valores por defecto pasan la validación y sobreviven a JSON ida y vuelta."""
    cfg = Config()
    cfg.validate()
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    assert Config.load_json(path) == cfg
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_missing_sections_and_unknown_keys_use_defaults() -> None:
    """Secciones ausentes → valores por defecto; claves desconocidas → se ignoran (compatibilidad)."""
    cfg = Config.from_dict({"env": {"lam": 0.2, "clave_vieja": 1}, "seccion_rara": {}})
    assert cfg.env.lam == 0.2 and cfg.env.budget == Config().env.budget


def test_integral_floats_become_ints() -> None:
    """Un float entero de JSON (``5.0``) en un campo int se convierte a int (antes rompía un reshape)."""
    cfg = Config.from_dict({"env": {"n_harmonics": 5.0, "n_fft": 8192.0}, "segmentation": {"hop_length": 256.0}})
    assert cfg.env.n_harmonics == 5 and isinstance(cfg.env.n_harmonics, int)
    assert isinstance(cfg.env.n_fft, int) and isinstance(cfg.segmentation.hop_length, int)
    assert Config.from_dict({"env": {"lam": 1}}).env.lam == 1.0


@pytest.mark.parametrize("data, message", [
    ([1, 2], "objeto JSON con las secciones"),
    ({"env": 5}, "La sección 'env'"),
    ({"env": {"k_semitones": "2"}}, "env.k_semitones debe ser un entero o null"),
    ({"env": {"lam": "0.1"}}, "env.lam debe ser un número"),
    ({"env": {"n_harmonics": 5.5}}, "env.n_harmonics debe ser un entero"),
    ({"env": {"open_string_free": 1}}, "env.open_string_free debe ser true o false"),
    ({"experiment": {"algorithms": "ucb1"}}, "experiment.algorithms debe ser una lista"),
    ({"experiment": {"sweep_tau": [0.1, "x"]}}, r"experiment.sweep_tau\[1\] debe ser un número"),
    ({"env": {"beta": float("nan")}}, "env.beta debe ser un número finito"),
])
def test_wrong_types_are_rejected(data: object, message: str) -> None:
    """Regresión: tipos erróneos dan ValueError en español (antes AttributeError/TypeError en mitad del análisis)."""
    with pytest.raises(ValueError, match=message):
        Config.from_dict(data)  # type: ignore[arg-type]


@pytest.mark.parametrize("section, values, message", [
    ("segmentation", {"hop_length": 0}, "segmentation.hop_length"),
    ("pitch", {"fmin_hz": 300.0, "fmax_hz": 100.0}, "fmin_hz < fmax_hz"),
    ("pitch", {"fmin_hz": 10.0}, "dos periodos"),
    ("pitch", {"fmax_hz": 20000.0}, "Nyquist"),
    ("experiment", {"sweep_lambda": []}, "sweep_lambda no puede estar vacía"),
    ("experiment", {"sweep_runs": 0}, "experiment.sweep_runs"),
    ("experiment", {"n_runs": 0}, "experiment.n_runs"),
    ("experiment", {"sweep_epsilon": [0.1, 1.5]}, "fuera del rango de agent.epsilon"),
    ("experiment", {"algorithms": ["ucb1", "thompson"]}, "desconocidos: thompson"),
    ("experiment", {"algorithms": []}, "lista no vacía"),
    ("env", {"k_semitones": -1}, "env.k_semitones"),
    ("env", {"spectrum": "mel"}, "env.spectrum debe ser una de: cqt, stft"),
    ("env", {"n_harmonics": 0}, "env.n_harmonics"),
    ("env", {"budget": 0}, "env.budget"),
    ("env", {"n_frets": -1}, "env.n_frets"),
    ("env", {"n_fft": 100}, "env.n_fft"),
    ("env", {"initial_hand_fret": 12, "n_frets": 5}, "fuera del diapasón"),
    ("env", {"spectrum": "cqt", "n_octaves": 9}, "Nyquist"),
    ("agent", {"recommend": "random"}, "agent.recommend"),
])
def test_out_of_range_values_are_rejected(section: str, values: dict, message: str) -> None:
    """Regresión: rangos de PARAM_SPECS y restricciones cruzadas se comprueban al cargar."""
    with pytest.raises(ValueError, match=message):
        Config.from_dict({section: values})


def test_sample_rate_must_be_the_calibrated_one() -> None:
    """Regresión: a 44.1 kHz las ventanas en muestras durarían la mitad (E1 partida en 4 segmentos); se rechaza."""
    assert Config().audio.sample_rate == SUPPORTED_SAMPLE_RATE == 22050
    with pytest.raises(ValueError, match="audio.sample_rate debe ser 22050 Hz"):
        Config.from_dict({"audio": {"sample_rate": 44100}})


def test_validate_lists_all_problems() -> None:
    """validate() informa TODOS los problemas a la vez, no solo el primero."""
    cfg = Config()
    cfg.env.budget = 0
    cfg.segmentation.hop_length = 0
    with pytest.raises(ValueError) as info:
        cfg.validate()
    assert "env.budget" in str(info.value) and "segmentation.hop_length" in str(info.value)


def test_validate_accepts_numpy_numbers() -> None:
    """Valores numpy (p. ej. de un barrido) son válidos: np.int64 cuenta como entero."""
    cfg = Config()
    cfg.env.budget = np.int64(300)  # type: ignore[assignment]
    cfg.env.lam = np.float64(0.2)   # type: ignore[assignment]
    cfg.validate()


def test_load_json_errors_are_value_errors(tmp_path: Path) -> None:
    """Regresión: --config carpeta o JSON roto → ValueError (antes IsADirectoryError / JSONDecodeError)."""
    with pytest.raises(ValueError, match="es una carpeta"):
        Config.load_json(tmp_path)
    bad = tmp_path / "roto.json"
    bad.write_text('{"env": {lam: 0.1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="no es un JSON válido"):
        Config.load_json(bad)
    with pytest.raises(FileNotFoundError):
        Config.load_json(tmp_path / "no_existe.json")
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"env": {"lam": 0.3}}), encoding="utf-8")
    assert Config.load_json(ok).env.lam == 0.3


def test_param_specs_cover_existing_fields_and_defaults_fit() -> None:
    """Cada ParamSpec apunta a un campo real y su valor por defecto está en rango (la GUI los usa)."""
    cfg = Config()
    for key, spec in PARAM_SPECS.items():
        value = cfg.get(key)
        if spec.kind in ("int", "float") and spec.minimum is not None:
            assert spec.minimum <= value <= spec.maximum, key
        if spec.kind == "choice":
            assert value in spec.choices, key
    # La comparación STFT/CQT documentada en EnvConfig se puede reproducir desde la GUI.
    assert {"env.n_fft", "env.bins_per_octave"} <= set(PARAM_SPECS)
    assert PARAM_SPECS["env.n_fft"].unit == "muestras"


def test_get_and_set_with_dotted_keys() -> None:
    """get/set con notación "sección.campo"."""
    cfg = Config()
    cfg.set("env.lam", 0.25)
    assert cfg.get("env.lam") == 0.25 == cfg.env.lam


def test_algorithm_style_tables_are_complete() -> None:
    """Cada algoritmo y los dos oráculos tienen etiqueta, color, marcador y estilo de línea."""
    keys = set(ALGORITHMS) | {"oracle", "oracle_viterbi"}
    for table in (ALGO_LABELS, ALGO_COLORS, ALGO_MARKERS, ALGO_LINESTYLES):
        assert keys <= set(table)
