"""Pruebas de la pestaña «2 · Configuración» (:mod:`gui.tabs.config_tab`).

Las funciones puras (diferencias entre configuraciones, textos de los avisos,
renderizado de fórmulas) se prueban sin pantalla. Las pruebas marcadas con
``gui`` crean la aplicación completa (necesitan ``DISPLAY``; en CI se ejecutan
con ``xvfb-run``), ejercitan los controles de la pestaña invocando sus
callbacks y comprueban el ESTADO (configuración, eventos, avisos), no píxeles.

Ejecutar::

    xvfb-run -a .venv/bin/python -m pytest tests/test_gui_config.py -q
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

tk = pytest.importorskip("tkinter")

from src.config import ALGORITHMS, DATA_DIR, PARAM_SPECS, Config  # noqa: E402

AUDIO = DATA_DIR / "synthetic" / "linea_simple.mp3"

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="requiere una pantalla (DISPLAY); usa xvfb-run")


# ---------------------------------------------------------------------------
# Funciones puras (sin pantalla)
# ---------------------------------------------------------------------------


def test_config_differences_classifies_stages() -> None:
    """Etapas 1–5 → re-analizar; env/agent y la fracción con voz → re-transcribir; experiment → nada."""
    from gui.tabs.config_tab import config_differences

    ref, cur = Config(), Config()
    assert config_differences(ref, cur) == ([], [])
    cur.segmentation.onset_delta = 0.07
    cur.segmentation.hop_length = 512           # sin control en la GUI, pero cuenta (JSON cargado)
    cur.pitch.min_voiced_ratio = 0.4            # se re-anota barato: basta re-transcribir
    cur.env.k_semitones = None
    cur.agent.tau = 0.3
    cur.experiment.n_runs = 7                   # solo afecta al experimento
    reanalyze, retranscribe = config_differences(ref, cur)
    assert reanalyze == ["segmentation.hop_length", "segmentation.onset_delta"]
    assert retranscribe == ["pitch.min_voiced_ratio", "env.k_semitones", "agent.tau"]


def test_change_descriptions_and_values() -> None:
    """Los avisos muestran etiqueta, valor anterior → nuevo y unidad."""
    from gui.tabs.config_tab import describe_changes, format_seconds, format_value

    cur = Config()
    cur.preprocess.lowpass_hz = 300.0
    cur.env.k_semitones = None
    text = describe_changes(["preprocess.lowpass_hz", "env.k_semitones"], Config(), cur)
    assert text == "Corte pasa-bajas (400 Hz → 300 Hz), k (poda ± semitonos) (2 → vacío)"
    many = describe_changes(["env.lam"] * 6, Config(), Config(), limit=4)
    assert many.endswith("y 2 más")
    assert format_value("env.open_string_free", True) == "sí"
    assert format_value("experiment.sweep_c", [0.1, 1.414]) == "0.1, 1.414"
    assert format_seconds(30.0) == "30 s" and format_seconds(150.0) == "2.5 min"


def test_validation_problems_lists_each_problem() -> None:
    """``validation_problems`` devuelve un mensaje por problema de ``Config.validate``."""
    from gui.tabs.config_tab import validation_problems

    cfg = Config()
    assert validation_problems(cfg) == []
    cfg.experiment.sweep_epsilon = [0.1, 1.5]
    cfg.experiment.sweep_lambda = []
    problems = validation_problems(cfg)
    assert len(problems) == 2
    assert any("sweep_epsilon" in p and "1.5" in p for p in problems)
    assert all(not p.startswith("•") for p in problems)


def test_render_math_png_and_formula_blocks() -> None:
    """Todas las fórmulas del panel se renderizan a PNG con mathtext."""
    from gui.tabs.config_tab import FORMULA_BLOCKS, render_math_png

    for block in FORMULA_BLOCKS:
        for tex in block.formulas:
            png = render_math_png(tex, size=13, dpi=96)
            assert png[:8] == b"\x89PNG\r\n\x1a\n", tex
        if block.values is not None:
            assert block.values(Config()).startswith("Ahora:")


# ---------------------------------------------------------------------------
# Pestaña dentro de la aplicación (requiere pantalla)
# ---------------------------------------------------------------------------


def _pump(app: Any, seconds: float = 0.05) -> None:
    """Procesa eventos de Tk durante ``seconds`` segundos."""
    end = time.time() + seconds
    while True:
        app.root.update()
        if time.time() >= end:
            break
        time.sleep(0.01)


def _wait_idle(app: Any, timeout: float = 120.0) -> None:
    """Espera a que termine la tarea en segundo plano (hilo trabajador)."""
    end = time.time() + timeout
    _pump(app, 0.1)
    while app.busy and time.time() < end:
        _pump(app, 0.05)
    _pump(app, 0.1)
    assert not app.busy, "la tarea en segundo plano no terminó a tiempo"


def _edit(tab: Any, key: str, text: str) -> None:
    """Escribe ``text`` en el control de ``key`` y lo confirma (como pulsar Enter)."""
    field = tab.fields[key]
    field.var.set(text)
    field._changed()


@pytest.fixture(scope="module")
def app() -> Any:
    """Aplicación completa con la pestaña Configuración visible (una por módulo: crearla cuesta ≈ 1.5 s)."""
    if not os.environ.get("DISPLAY"):
        pytest.skip("requiere una pantalla (DISPLAY); usa xvfb-run")
    from gui.app import SmartunerApp

    try:
        application = SmartunerApp()
    except tk.TclError as exc:  # pragma: no cover - pantalla no disponible
        pytest.skip(f"no se pudo abrir tkinter: {exc}")
    application.root.geometry("1360x880+0+0")
    application.show_tab("config")
    _pump(application, 0.3)
    yield application
    application.quit()


@pytest.fixture()
def dialogs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...]]]:
    """Sustituye los diálogos modales (bloquearían la prueba) y registra sus llamadas."""
    from tkinter import messagebox

    calls: list[tuple[str, tuple[Any, ...]]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(messagebox, name, lambda *a, _n=name, **k: calls.append((_n, a)))
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: calls.append(("askyesno", a)) or True)
    return calls


@pytest.fixture()
def tab(app: Any, dialogs: list[Any]) -> Any:
    """La pestaña Configuración con estado limpio (configuración por defecto y sin análisis)."""
    app.state.config = Config()
    app.state.analysis = None
    app.state.transcriptions = {}
    app.state.experiment = None
    app.state.events.emit("config_changed", key=None)
    config_tab = app.tabs["config"]
    config_tab.set_formulas_visible(True)
    _pump(app)
    return config_tab


@pytest.fixture()
def events(app: Any) -> list[dict[str, Any]]:
    """Registra los ``config_changed`` emitidos durante la prueba."""
    received: list[dict[str, Any]] = []
    app.state.events.subscribe("config_changed", lambda **kw: received.append(kw))
    return received


@pytest.mark.gui
@needs_display
def test_tab_has_one_field_per_param_spec(tab: Any) -> None:
    """Todos los parámetros de PARAM_SPECS tienen su control, con el valor vigente."""
    from gui.tabs.config_tab import FORMULA_BLOCKS, ConfigTab

    assert isinstance(tab, ConfigTab)
    assert set(tab.fields) == set(PARAM_SPECS)
    cfg = Config()
    for key, field in tab.fields.items():
        assert field.parse() == cfg.get(key), key
    assert all(var.get() for var in tab.algo_vars.values())
    assert list(tab.algo_vars) == list(ALGORITHMS)
    # Estado vacío: aviso guía para abrir un audio y estimación por nota.
    assert tab.status_kind == "empty"
    assert "Abre un MP3" in tab.status_banner.detail.cget("text")
    assert "por nota" in tab.estimate_label.cget("text")
    assert str(tab.run_experiment_button.cget("state")) == "disabled"
    assert len(tab.formula_images) == sum(len(b.formulas) for b in FORMULA_BLOCKS)


@pytest.mark.gui
@needs_display
def test_valid_change_updates_config_and_emits(tab: Any, app: Any, events: list[dict[str, Any]]) -> None:
    """Un valor válido se escribe en la configuración y se anuncia con su clave."""
    _edit(tab, "env.lam", "0.3")
    assert app.state.config.env.lam == pytest.approx(0.3)
    assert events == [{"key": "env.lam"}]
    _edit(tab, "env.lam", "0.3")                      # mismo valor (p. ej. perder el foco): sin evento
    assert len(events) == 1
    _edit(tab, "env.k_semitones", "")                 # int_or_none vacío → None (52 brazos)
    assert app.state.config.env.k_semitones is None
    _edit(tab, "env.budget", "99999")                 # fuera de rango: ParamField lo rechaza
    assert app.state.config.env.budget == 500
    assert tab.fields["env.budget"].error.winfo_manager() == "grid"
    assert "λ = 0.3" in tab._formula_values[1][0].cget("text")   # línea «Ahora» de la recompensa


@pytest.mark.gui
@needs_display
def test_sweep_values_outside_parameter_range_are_rejected(tab: Any, app: Any, events: list[dict[str, Any]]) -> None:
    """Una lista de barrido con valores fuera del rango del parámetro no llega a la configuración."""
    _edit(tab, "experiment.sweep_epsilon", "0.1, 1.5")
    assert app.state.config.experiment.sweep_epsilon == Config().experiment.sweep_epsilon
    field = tab.fields["experiment.sweep_epsilon"]
    assert field.error.winfo_manager() == "grid"
    assert "1.5" in field.error.cget("text")
    assert events == [] and tab.problems == []
    _edit(tab, "experiment.sweep_epsilon", "0.1, 0.5")
    assert app.state.config.experiment.sweep_epsilon == [0.1, 0.5]
    assert field.error.winfo_manager() == ""


@pytest.mark.gui
@needs_display
def test_algorithm_checkboxes_edit_experiment_algorithms(tab: Any, app: Any, events: list[dict[str, Any]]) -> None:
    """Las casillas editan experiment.algorithms en el orden de ALGORITHMS y nunca lo dejan vacío."""
    tab.algo_vars["egreedy"].set(False)
    tab._on_algorithm_toggle("egreedy")
    assert app.state.config.experiment.algorithms == ["optimistic", "ucb1", "softmax"]
    assert events[-1] == {"key": "experiment.algorithms"}
    # El barrido de ε se desactiva: su algoritmo ya no participa.
    assert tab.fields["experiment.sweep_epsilon"].control.instate(["disabled"])
    for algo in ("optimistic", "ucb1", "softmax"):
        tab.algo_vars[algo].set(False)
        tab._on_algorithm_toggle(algo)
    assert app.state.config.experiment.algorithms == ["softmax"]
    assert tab.algo_vars["softmax"].get()
    assert "al menos un algoritmo" in tab.algo_message.cget("text")
    tab.algo_vars["egreedy"].set(True)
    tab._on_algorithm_toggle("egreedy")
    assert app.state.config.experiment.algorithms == ["egreedy", "softmax"]
    assert tab.algo_message.cget("text") == ""


@pytest.mark.gui
@needs_display
def test_fields_not_used_by_the_config_are_disabled(tab: Any) -> None:
    """n_fft solo con STFT, bins por octava solo con CQT, mínimos solo con decaimiento, listas con barridos."""
    def disabled(key: str) -> bool:
        return tab.fields[key].control.instate(["disabled"])

    assert not disabled("env.n_fft") and disabled("env.bins_per_octave")
    tab.fields["env.spectrum"].var.set("cqt")
    tab.fields["env.spectrum"]._changed()
    assert disabled("env.n_fft") and not disabled("env.bins_per_octave")
    assert disabled("agent.epsilon_min")
    _edit(tab, "agent.epsilon_decay", "0.99")
    assert not disabled("agent.epsilon_min")
    tab.fields["experiment.run_sweeps"].var.set(False)
    tab.fields["experiment.run_sweeps"]._changed()
    assert disabled("experiment.sweep_q0") and not disabled("experiment.sweep_lambda")
    assert not disabled("experiment.sweep_runs")      # lo sigue usando el barrido de λ


@pytest.mark.gui
@needs_display
def test_external_changes_refresh_the_fields(tab: Any, app: Any) -> None:
    """Un config_changed de otra fuente (JSON, otra pestaña) refresca los controles."""
    new = Config()
    new.env.lam = 0.7
    new.experiment.algorithms = ["ucb1"]
    app.state.config = new
    app.state.events.emit("config_changed", key=None)
    assert tab.fields["env.lam"].var.get() == "0.7"
    assert [a for a, v in tab.algo_vars.items() if v.get()] == ["ucb1"]
    app.state.config.set("env.beta", 1.25)                   # otra pestaña cambia un solo valor
    app.state.events.emit("config_changed", key="env.beta")
    assert tab.fields["env.beta"].var.get() == "1.25"


@pytest.mark.gui
@needs_display
def test_restore_defaults(tab: Any, app: Any, dialogs: list[Any], monkeypatch: pytest.MonkeyPatch,
                          events: list[dict[str, Any]]) -> None:
    """Restaurar pide confirmación, crea un Config() nuevo y emite config_changed(key=None)."""
    from tkinter import messagebox

    _edit(tab, "agent.ucb_c", "3")
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: False)
    tab.restore_button.invoke()                              # el usuario responde «No»
    assert app.state.config.agent.ucb_c == pytest.approx(3.0)
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    tab.restore_button.invoke()
    assert app.state.config == Config()
    assert events[-1] == {"key": None}
    assert tab.fields["agent.ucb_c"].var.get() == "1.414"


@pytest.mark.gui
@needs_display
def test_save_and_load_json(tab: Any, app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                            dialogs: list[Any]) -> None:
    """[Guardar JSON…] y [Cargar JSON…] usan las acciones de la app y la vista se sincroniza."""
    from tkinter import filedialog

    path = tmp_path / "cfg.json"
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(path))
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **k: str(path))
    _edit(tab, "env.n_harmonics", "7")
    tab.save_button.invoke()
    assert json.loads(path.read_text(encoding="utf-8"))["env"]["n_harmonics"] == 7
    _edit(tab, "env.n_harmonics", "3")
    tab.load_button.invoke()
    assert app.state.config.env.n_harmonics == 7
    assert tab.fields["env.n_harmonics"].var.get() == "7"
    # Un JSON inválido no cambia nada (la app muestra el error).
    path.write_text(json.dumps({"env": {"lam": 9.0}}), encoding="utf-8")
    tab.load_button.invoke()
    assert app.state.config.env.lam == pytest.approx(0.1)
    assert dialogs and dialogs[-1][0] == "showerror"


@pytest.mark.gui
@needs_display
def test_mouse_wheel_scrolls_instead_of_changing_values(tab: Any, app: Any) -> None:
    """La rueda sobre un Spinbox/Combobox desplaza la página y no cambia el parámetro."""
    for key in ("env.budget", "env.spectrum"):
        field = tab.fields[key]
        before = field.var.get()
        for sequence in ("<Button-4>", "<Button-5>", "<Button-5>"):
            field.control.event_generate(sequence)
        _pump(app)
        assert field.var.get() == before, key
    assert app.state.config == Config()


@pytest.mark.gui
@needs_display
def test_formulas_panel_toggle_and_selected_algorithm(tab: Any, app: Any) -> None:
    """El panel de fórmulas se oculta/muestra y la tarjeta del algoritmo seleccionado se marca."""
    panes = lambda: [str(p) for p in tab.paned.panes()]  # noqa: E731
    assert str(tab.formula_panel) in panes()
    tab.formulas_visible.set(False)
    tab.formulas_check.invoke()                     # el clic vuelve a marcarla → visible
    assert tab.formulas_visible.get() and str(tab.formula_panel) in panes()
    tab.formulas_check.invoke()
    assert str(tab.formula_panel) not in panes()
    tab.set_formulas_visible(True)
    app.state.select_algorithm("softmax")
    assert tab._algo_tags["softmax"].winfo_manager() == "pack"
    assert tab._algo_tags["ucb1"].winfo_manager() == ""
    app.state.select_algorithm("ucb1")
    assert tab._algo_tags["ucb1"].winfo_manager() == "pack"


@pytest.mark.gui
@needs_display
def test_invalid_external_config_shows_validation_banner(tab: Any, app: Any) -> None:
    """Una configuración inválida (p. ej. de otra pestaña) se señala y bloquea las acciones."""
    app.state.config.experiment.sweep_lambda = [5.0]          # λ ∈ [0, 2]
    app.state.events.emit("config_changed", key="experiment.sweep_lambda")
    assert tab.problems and "sweep_lambda" in tab.problems[0]
    assert tab.validation_banner.winfo_manager() == "pack"
    assert tab.validation_banner.kind == "error"
    app.state.config.experiment.sweep_lambda = [0.5]
    app.state.events.emit("config_changed", key="experiment.sweep_lambda")
    assert tab.problems == [] and tab.validation_banner.winfo_manager() == ""


@pytest.mark.gui
@pytest.mark.slow
@needs_display
@pytest.mark.skipif(not AUDIO.exists(), reason="falta data/synthetic/linea_simple.mp3")
def test_pending_changes_with_a_real_analysis(tab: Any, app: Any) -> None:
    """Con un audio analizado: avisos «re-transcribir» / «re-analizar» y sus botones."""
    app.open_audio(AUDIO)
    _wait_idle(app)
    analysis = app.state.analysis
    assert analysis is not None and tab.status_kind == "synced"
    assert f"{len(analysis.segment_data)} notas" in tab.estimate_label.cget("text")
    assert str(tab.run_experiment_button.cget("state")) == "normal"

    # Entorno/agentes → re-transcribir (con el cambio resaltado).
    _edit(tab, "env.lam", "0.25")
    assert tab.status_kind == "retranscribe"
    assert tab.pending_keys == {"env.lam"}
    assert tab.status_banner.button.cget("text") == "Aplicar y re-transcribir"
    assert "0.1 → 0.25" in tab.status_banner.detail.cget("text")
    # Etapas 1–5 → re-analizar; volver al valor original quita el aviso.
    _edit(tab, "preprocess.lowpass_hz", "300")
    assert tab.status_kind == "reanalyze"
    assert tab.status_banner.button.cget("text") == "Analizar de nuevo"
    _edit(tab, "preprocess.lowpass_hz", "400")
    assert tab.status_kind == "retranscribe"

    # [Aplicar y re-transcribir]: botón deshabilitado mientras corre la tarea.
    tab.status_banner.button.invoke()
    assert app.busy and tab.status_banner.button.instate(["disabled"])
    _wait_idle(app)
    assert tab.status_kind == "synced" and tab.pending_keys == set()
    assert app.state.analysis.config.env.lam == pytest.approx(0.25)

    # [Analizar de nuevo] con un parámetro de segmentación.
    _edit(tab, "segmentation.onset_delta", "0.06")
    assert tab.status_kind == "reanalyze"
    tab.status_banner.button.invoke()
    _wait_idle(app)
    assert tab.status_kind == "synced"
    assert app.state.analysis.config.segmentation.onset_delta == pytest.approx(0.06)
    assert "Aún no hay resultados" in tab.experiment_state_label.cget("text")


def _label_texts(widget: Any) -> list[str]:
    """Textos de todas las etiquetas (ttk/tk Label y LabelFrame) bajo ``widget``."""
    texts: list[str] = []
    for child in widget.winfo_children():
        try:
            text = str(child.cget("text"))
        except tk.TclError:
            text = ""
        if text:
            texts.append(text)
        texts += _label_texts(child)
    return texts


@pytest.mark.gui
@needs_display
def test_separation_method_has_its_own_stage_2_group(tab: Any) -> None:
    """Regresión: «Método de separación del bajo» ya no cae en «Otros parámetros»; la sección dice «Etapas 2–5»."""
    from gui.tabs.config_tab import SECTIONS

    analysis_section = next(s for s in SECTIONS if s.key == "analysis")
    assert analysis_section.groups[0].keys == ("audio.separation_method",)
    assert "Etapas 2–5" in analysis_section.note
    texts = _label_texts(tab._sections["analysis"])
    assert "Otros parámetros" not in texts
    assert any(t.startswith("Separación del bajo (etapa 2") for t in texts)


@pytest.mark.gui
@needs_display
def test_choice_fields_show_spanish_labels_and_store_identifiers(tab: Any, app: Any) -> None:
    """Regresión: «Recomendación final» muestra «más jalado (robusto)»/«mayor Q (greedy)», no most_pulled/greedy."""
    field = tab.fields["agent.recommend"]
    assert field.var.get() == "más jalado (robusto)"
    assert list(field.control.cget("values")) == ["más jalado (robusto)", "mayor Q (greedy)"]
    _edit(tab, "agent.recommend", "mayor Q (greedy)")
    assert app.state.config.agent.recommend == "greedy"
    app.state.config.agent.recommend = "most_pulled"
    app.state.events.emit("config_changed", key="agent.recommend")
    _pump(app)
    assert field.var.get() == "más jalado (robusto)"
    assert tab.fields["audio.separation_method"].var.get() == "automático"


@pytest.mark.gui
@needs_display
def test_footer_explains_reduced_sweep_runs_on_long_tracks(tab: Any, app: Any) -> None:
    """Regresión: con una canción de 993 notas el pie dice cuántas corridas por valor usarán los barridos."""
    from types import SimpleNamespace

    from src.experiments import effective_sweep_runs

    cfg = app.state.config
    runs = effective_sweep_runs(cfg, 993)
    assert runs < cfg.experiment.sweep_runs
    app.state.analysis = SimpleNamespace(segment_data=[None] * 993, ground_truth=None)
    try:
        tab._update_experiment_info()
        text = tab.estimate_label.cget("text")
        assert f"se usarán {runs} corridas por valor (no {cfg.experiment.sweep_runs})" in text
        assert "993 notas" in text
        assert "50 notas" in PARAM_SPECS["experiment.sweep_runs"].help
    finally:
        app.state.analysis = None
        tab._update_experiment_info()
