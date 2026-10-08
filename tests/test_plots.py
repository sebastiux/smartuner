"""Pruebas de :mod:`src.plots` con resultados FALSOS construidos con numpy.

No dependen de que ``run_experiment`` ni ``analyze`` estén implementados: el
``ExperimentResult`` se arma a mano con curvas sintéticas y el
``AnalysisResult`` con tonos armónicos sintéticos, segmentos y una trayectoria
de f0 escritos a mano (solo se calcula la CQT con :mod:`src.environment`).

Cada figura se RENDERIZA (``savefig`` a memoria) para que los errores de
dibujo o de *layout* aparezcan aquí y no en la GUI.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from src import plots
from src.config import ALGO_COLORS, ALGO_MARKERS, ALGORITHMS, STRING_ORDER, Config
from src.environment import Arm, build_all_segment_data, compute_spectrum
from src.experiments import (
    OUTCOME_EXACT,
    OUTCOME_LABELS,
    OUTCOME_MISSED,
    OUTCOME_WRONG_PITCH,
    OUTCOME_WRONG_POSITION,
    AccuracyResult,
    ExampleSegmentResult,
    ExperimentResult,
    LambdaSweepResult,
    SweepResult,
)
from src.pipeline import AnalysisResult
from src.pitch import PitchTrack
from src.segmentation import Segment
from src.synth_dataset import GTNote

ROOT = Path(__file__).resolve().parent.parent
SR = 22050
HOP = 256

# ---------------------------------------------------------------------------
# Datos falsos
# ---------------------------------------------------------------------------

#: Notas GT del resultado falso: (cuerda, traste); la nota 4 no tiene segmento.
GT_POSITIONS = [("E", 0), ("A", 0), ("E", 5), ("A", 2), ("D", 0), ("D", 2), ("G", 0), ("A", 7), ("G", 2)]
#: Segmento emparejado con cada nota GT (None = no detectada). El segmento 5 es un onset falso.
TRUE_MATCHES: list[int | None] = [0, 1, 2, 3, None, 4, 6, 7, 8]
N_SEGMENTS = 9


def _score(arms: list[Arm], matches: list[int | None], gt: list[GTNote]) -> AccuracyResult:
    """Puntuación de referencia (misma definición que el contrato de experiments.score_arms)."""
    out = np.zeros(len(gt), dtype=int)
    for g, note in enumerate(gt):
        s = matches[g]
        if s is None:
            out[g] = OUTCOME_MISSED
        elif arms[s].midi != note.midi:
            out[g] = OUTCOME_WRONG_PITCH
        elif (arms[s].string, arms[s].fret) != (note.string, note.fret):
            out[g] = OUTCOME_WRONG_POSITION
        else:
            out[g] = OUTCOME_EXACT
    return AccuracyResult(pitch=float(np.mean(out >= OUTCOME_WRONG_POSITION)),
                          position=float(np.mean(out == OUTCOME_EXACT)), outcomes=out)


def _gt_notes() -> list[GTNote]:
    """Ground truth sintético: una nota cada 0.3 s en las posiciones de GT_POSITIONS."""
    notes = []
    for i, (string, fret) in enumerate(GT_POSITIONS):
        arm = Arm(string, fret)
        notes.append(GTNote(onset_s=0.5 + 0.3 * i, offset_s=0.75 + 0.3 * i, midi=arm.midi, string=string, fret=fret))
    return notes


def _segment_truth(gt: list[GTNote]) -> list[Arm]:
    """Posición "real" de cada segmento (el falso, entre dos notas, es un D-2)."""
    arms: list[Arm] = [Arm("D", 2)] * N_SEGMENTS
    for g, s in enumerate(TRUE_MATCHES):
        if s is not None:
            arms[s] = Arm(gt[g].string, gt[g].fret)
    return arms


def make_fake_result(algorithms: tuple[str, ...] = ALGORITHMS, n_runs: int = 6, budget: int = 60,
                     seed: int = 0) -> ExperimentResult:
    """``ExperimentResult`` realista y completo (con GT, ejemplo, barridos y λ)."""
    rng = np.random.default_rng(seed)
    cfg = Config()
    cfg.env.budget = budget
    cfg.experiment.n_runs = n_runs
    gt = _gt_notes()
    truth = _segment_truth(gt)
    t = np.arange(1, budget + 1)
    reward, regret, optimal, choices, elapsed, accuracy = {}, {}, {}, {}, {}, {}
    for k, alg in enumerate(algorithms):
        speed = 8.0 + 6.0 * k
        base = 0.62 - 0.3 * np.exp(-t / speed)
        reward[alg] = base + rng.normal(0.0, 0.02, (n_runs, budget))
        inst = np.clip(0.3 * np.exp(-t / speed) + rng.normal(0.0, 0.01, (n_runs, budget)), 0.0, None)
        regret[alg] = np.cumsum(inst, axis=1)
        optimal[alg] = np.clip(1.0 - np.exp(-t / speed) + rng.normal(0.0, 0.05, (n_runs, budget)), 0.0, 1.0)
        ch = np.zeros((n_runs, N_SEGMENTS, 2), dtype=int)
        runs_acc = []
        for r in range(n_runs):
            arms = []
            for s, true_arm in enumerate(truth):
                u = rng.random()
                if u < 0.6:
                    arm = true_arm
                elif u < 0.85 and true_arm.fret <= 7 and true_arm.string != "E":
                    # misma altura en la cuerda de abajo (+5 trastes)
                    arm = Arm(STRING_ORDER[true_arm.string_index - 1], true_arm.fret + 5)
                else:
                    arm = Arm(true_arm.string, min(true_arm.fret + 1, 12))
                arms.append(arm)
                ch[r, s] = (arm.string_index, arm.fret)
            runs_acc.append(_score(arms, TRUE_MATCHES, gt))
        choices[alg] = ch
        accuracy[alg] = runs_acc
        elapsed[alg] = 0.01 * (1 + k) + rng.normal(0.0, 0.001, n_runs).clip(-0.005, None)

    oracle = list(truth)
    oracle[2] = Arm("A", 0)  # el oráculo prefiere A-0 a E-5 (misma altura)

    k_arms = 6
    labels = ["A-0", "E-5", "A-1", "E-6", "D-0", "G-0"]
    mu = np.array([0.82, 0.78, 0.40, 0.35, 0.20, -0.001])
    counts, q_mean, q_std = {}, {}, {}
    for k, alg in enumerate(algorithms):
        p = np.exp(mu / (0.05 + 0.05 * k))
        p /= p.sum()
        counts[alg] = rng.multinomial(budget, p, size=n_runs)
        q0 = 2.0 if alg == "optimistic" else 0.0
        q = q0 + (mu - q0)[None, :] * (1.0 - np.exp(-t / (5.0 + 3 * k)))[:, None]
        q_mean[alg] = q
        q_std[alg] = np.full_like(q, 0.05)
    example = ExampleSegmentResult(position=2, arm_labels=labels, true_means=mu, optimal_arms=np.array([0]),
                                   prev_fret=0, counts=counts, q_mean=q_mean, q_std=q_std)

    sweeps = []
    for param, alg, values, best in [("agent.epsilon", "egreedy", [0.01, 0.05, 0.1, 0.2, 0.5], 0.1),
                                     ("agent.q0", "optimistic", [0.5, 1.0, 2.0, 5.0, 10.0], 2.0),
                                     ("agent.ucb_c", "ucb1", [0.1, 0.5, 1.0, 1.414, 4.0], 0.5),
                                     ("agent.tau", "softmax", [0.01, 0.03, 0.1, 0.3, 1.0], 0.03)]:
        if alg not in algorithms:
            continue
        v = np.array(values)
        m = 0.6 - 0.05 * np.abs(np.log10(v / best))
        sweeps.append(SweepResult(param=param, algorithm=alg, values=values,
                                  final_reward=m[:, None] + rng.normal(0, 0.01, (v.size, 4)),
                                  mean_reward=m[:, None] - 0.03 + rng.normal(0, 0.01, (v.size, 4)),
                                  final_regret=5 + rng.normal(0, 0.5, (v.size, 4))))

    lam_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    lam_sweep = LambdaSweepResult(
        values=lam_values,
        position_acc={a: np.clip(0.4 + 0.3 * np.tanh(np.array(lam_values) * 5)[:, None]
                                 + rng.normal(0, 0.05, (6, 4)), 0, 1) for a in algorithms},
        pitch_acc={a: np.clip(0.9 + rng.normal(0, 0.02, (6, 4)), 0, 1) for a in algorithms},
        oracle_position_acc=np.clip(0.5 + 0.35 * np.tanh(np.array(lam_values) * 5), 0, 1),
        oracle_pitch_acc=np.full(6, 0.95),
    )
    return ExperimentResult(
        config=cfg, algorithms=list(algorithms), n_runs=n_runs, budget=budget, n_segments=N_SEGMENTS,
        reward_curves=reward, regret_curves=regret, optimal_curves=optimal, choices=choices, elapsed_s=elapsed,
        accuracy=accuracy, oracle_arms=oracle, oracle_accuracy=_score(oracle, TRUE_MATCHES, gt), gt_notes=gt,
        example=example, sweeps=sweeps, lambda_sweep=lam_sweep,
    )


def _tone(f0: float, dur_s: float, n_harm: int = 5) -> np.ndarray:
    """Tono armónico (pesos 1/h) con decaimiento exponencial, de ``dur_s`` segundos."""
    t = np.arange(int(dur_s * SR)) / SR
    y = sum((1.0 / h) * np.sin(2 * np.pi * h * f0 * t) for h in range(1, n_harm + 1))
    return y * np.exp(-3.0 * t)


def make_fake_analysis(spectrum_kind: str = "cqt") -> AnalysisResult:
    """``AnalysisResult`` sintético: 4 notas, un golpe corto descartado y silencios."""
    cfg = Config()
    cfg.env.spectrum = spectrum_kind
    notes = [(55.0, 0.5, 0.4), (61.74, 0.95, 0.4), (98.0, 1.4, 0.4), (41.2, 2.2, 0.45)]  # (Hz, inicio, dur)
    dur_total = 2.9
    y = np.zeros(int(dur_total * SR))
    for f0, start, dur in notes:
        i0 = int(start * SR)
        tone = _tone(f0, dur)
        y[i0:i0 + tone.size] += tone
    blip0 = int(1.95 * SR)
    y[blip0:blip0 + 400] += 0.3 * np.random.default_rng(1).normal(size=400)  # golpe corto
    y = 0.9 * y / np.max(np.abs(y))

    segments = []
    bounds = [(0.5, 0.92, True, "ok"), (0.95, 1.38, True, "ok"), (1.4, 1.93, True, "ok"),
              (1.95, 1.99, False, "corto"), (2.2, 2.75, True, "ok")]
    f0s = iter([55.0, 61.74, 98.0, 41.2])
    for i, (a, b, kept, reason) in enumerate(bounds):
        segments.append(Segment(index=i, start_s=a, end_s=b, start_sample=int(a * SR), end_sample=int(b * SR),
                                rms_db=-6.0 if kept else -40.0, kept=kept, reason=reason,
                                f0_hz=next(f0s) if kept else None, voiced_ratio=0.9 if kept else 0.0))
    spectrum = compute_spectrum(y, SR, cfg.env, hop_length=HOP)
    times = spectrum.times_s
    f0_track = np.full(times.size, np.nan)
    for seg in segments:
        if seg.kept:
            f0_track[(times >= seg.start_s + 0.02) & (times < seg.end_s - 0.05)] = seg.f0_hz
    voiced = np.isfinite(f0_track)
    track = PitchTrack(times_s=times, f0_hz=f0_track, voiced_prob=voiced.astype(float), voiced=voiced)
    data = build_all_segment_data(spectrum, segments, cfg.env)
    return AnalysisResult(path=Path("falso.mp3"), source_path=Path("falso.mp3"), sr=SR, y_raw=y, y_analysis=y,
                          y_spectral=y, onsets_s=np.array([b[0] for b in bounds]), segments=segments,
                          pitch_track=track, spectrum=spectrum, segment_data=data, ground_truth=None, config=cfg)


@pytest.fixture(scope="module")
def res() -> ExperimentResult:
    """Resultado de experimento falso compartido por las pruebas del módulo."""
    return make_fake_result()


@pytest.fixture(scope="module")
def analysis() -> AnalysisResult:
    """Análisis falso (CQT) compartido por las pruebas del módulo."""
    return make_fake_analysis()


# ---------------------------------------------------------------------------
# Utilidades de prueba
# ---------------------------------------------------------------------------


def _render(fig: Figure) -> None:
    """Dibuja la figura completa (layout incluido) y falla si matplotlib avisa de algo."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(io.BytesIO(), format="png", dpi=40)
    problems = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not problems, problems


def _title(fig: Figure) -> str:
    """Título (suptitle) de la figura, o cadena vacía."""
    sup = fig._suptitle  # noqa: SLF001 - el título de todas las gráficas es el suptitle
    return sup.get_text().strip() if sup is not None else ""


def _all_texts(fig: Figure) -> str:
    """Todos los textos de la figura unidos por saltos de línea."""
    texts = [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text"))]
    return "\n".join(t for t in texts if isinstance(t, str))


def _assert_plot(fig: Figure, min_axes: int = 1) -> None:
    """Comprueba que la figura tiene al menos ``min_axes`` ejes visibles y título, y que se puede dibujar."""
    assert isinstance(fig, Figure)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.axison]
    assert len(visible) >= min_axes
    assert _title(fig), "la figura no tiene título"
    _render(fig)


def _is_message(fig: Figure) -> bool:
    """True si la figura es un mensaje (un solo eje sin marco con un texto)."""
    return len(fig.axes) == 1 and not fig.axes[0].axison and len(fig.axes[0].texts) == 1


# ---------------------------------------------------------------------------
# Módulo y catálogo
# ---------------------------------------------------------------------------


def test_plots_does_not_import_pyplot() -> None:
    """src.plots no importa pyplot (la GUI embebe Figures en tkinter sin el backend global)."""
    source = (ROOT / "src" / "plots.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            assert all("pyplot" not in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "pyplot" not in (node.module or "")
            assert all(alias.name != "pyplot" for alias in node.names)
    code = "import sys; import src.plots; print('matplotlib.pyplot' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_catalog_is_complete_and_documented() -> None:
    """El catálogo de gráficas comparativas está completo, en orden, y cada una tiene título y una descripción de 2–3 frases."""
    keys = [spec.key for spec in plots.COMPARISON_PLOTS]
    assert keys == ["average_reward", "cumulative_regret", "optimal_action", "arm_distribution", "q_evolution",
                    "sensitivity", "lambda_effect", "accuracy", "runtime", "tab_heatmap"]
    for spec in plots.COMPARISON_PLOTS:
        assert callable(spec.func)
        assert spec.title and spec.title[0].isupper()
        assert spec.description.count(".") >= 2, f"{spec.key}: la descripción debe tener 2–3 frases"


@pytest.mark.parametrize("spec", plots.COMPARISON_PLOTS, ids=lambda s: s.key)
def test_comparison_plot_returns_figure_with_axes_and_title(spec: plots.PlotSpec, res: ExperimentResult) -> None:
    """Cada gráfica comparativa devuelve una Figure con ejes, título y leyenda."""
    assert plots.missing_data_message(spec.key, res) is None
    fig = spec.func(res)
    assert not _is_message(fig)
    _assert_plot(fig)
    # Leyenda siempre presente (en el eje o en la figura), salvo en el tiempo, donde
    # las etiquetas del eje Y ya nombran cada barra.
    if spec.key != "runtime":
        assert fig.legends or any(ax.get_legend() is not None for ax in fig.axes)


@pytest.mark.parametrize("spec", plots.COMPARISON_PLOTS, ids=lambda s: s.key)
def test_comparison_plot_reuses_existing_figure(spec: plots.PlotSpec, res: ExperimentResult) -> None:
    """Dibujar sobre la Figure de la GUI la reutiliza (mismo tamaño) sin acumular ejes."""
    fig = Figure(figsize=(9.0, 5.5), dpi=100)  # el tamaño de la figura embebida en la GUI
    fig.add_subplot().plot([0, 1], [0, 1])
    fig.suptitle("vieja")
    out = spec.func(res, fig=fig)
    assert out is fig
    assert _title(fig) != "vieja"
    assert tuple(fig.get_size_inches()) == (9.0, 5.5), "no debe cambiar el tamaño de la figura de la GUI"
    _render(fig)
    # Volver a dibujar sobre la misma figura no acumula ejes.
    n_axes = len(fig.axes)
    spec.func(res, fig=fig)
    assert len(fig.axes) == n_axes


# ---------------------------------------------------------------------------
# Estilo y codificación por algoritmo
# ---------------------------------------------------------------------------


def test_color_marker_follow_algorithm_not_rank() -> None:
    """Color, marcador y estilo siguen al ALGORITMO (no al orden), con banda ±1 std translúcida y leyenda en tinta neutra."""
    subset = make_fake_result(algorithms=("ucb1", "softmax"), n_runs=3, budget=30)
    fig = plots.plot_average_reward(subset)
    ax = fig.axes[0]
    lines = [ln for ln in ax.get_lines() if ln.get_xdata().size > 1]
    assert [ln.get_color() for ln in lines] == [ALGO_COLORS["ucb1"], ALGO_COLORS["softmax"]]
    assert [ln.get_marker() for ln in lines] == [ALGO_MARKERS["ucb1"], ALGO_MARKERS["softmax"]]
    assert all(ln.get_linewidth() == pytest.approx(2.0) for ln in lines)
    # Banda ±1 std con el color del algoritmo al 15 %.
    bands = ax.collections
    assert bands and all(b.get_alpha() == pytest.approx(plots.BAND_ALPHA) for b in bands)
    # El texto de la leyenda usa tinta, nunca el color de la serie.
    legend_colors = {t.get_color() for t in ax.get_legend().get_texts()}
    assert legend_colors == {plots.TEXT_PRIMARY}


def test_apply_style_sets_surface_spines_and_grid(res: ExperimentResult) -> None:
    """El estilo común fija el fondo, oculta los bordes superior y derecho y dibuja la rejilla detrás de los datos."""
    fig = plots.plot_cumulative_regret(res)
    ax = fig.axes[0]
    assert fig.get_facecolor()[:3] == pytest.approx((0xfc / 255, 0xfc / 255, 0xfb / 255))
    assert ax.get_facecolor()[:3] == pytest.approx((0xfc / 255, 0xfc / 255, 0xfb / 255))
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_edgecolor()[:3] == pytest.approx((0xc3 / 255, 0xc2 / 255, 0xb7 / 255))
    grid = [ln for ln in ax.get_ygridlines() if ln.get_visible()]
    assert grid and all(ln.get_linestyle() == "-" for ln in grid)
    assert ax.get_axisbelow() is True
    assert "Regret acumulado" in ax.get_ylabel()
    assert "Pull t" in ax.get_xlabel()


def test_axis_labels_have_units(res: ExperimentResult) -> None:
    """Los ejes llevan unidades (ms por segmento, %, recompensa r)."""
    assert plots.plot_runtime(res).axes[0].get_xlabel() == "Tiempo (ms por segmento)"
    assert "%" in plots.plot_optimal_action(res).axes[0].get_ylabel()
    assert plots.plot_average_reward(res).axes[0].get_ylabel() == "Recompensa media r"


def test_ink_on_picks_contrasting_text() -> None:
    """El texto sobre una celda de color elige blanco o tinta oscura según el contraste."""
    assert plots._ink_on(plots.OUTCOME_COLORS[OUTCOME_WRONG_PITCH]) == "#ffffff"
    assert plots._ink_on(plots.OUTCOME_COLORS[OUTCOME_WRONG_POSITION]) == plots.TEXT_PRIMARY
    assert plots._ink_on(plots.OUTCOME_COLORS[OUTCOME_MISSED]) == plots.TEXT_PRIMARY


# ---------------------------------------------------------------------------
# Gráficas concretas
# ---------------------------------------------------------------------------


def test_arm_distribution_marks_optimal_and_orders_by_mu(res: ExperimentResult) -> None:
    """El reparto de pulls ordena los brazos por μ real, marca el óptimo y usa barras delgadas."""
    fig = plots.plot_arm_distribution(res)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "A-0\nμ=0.82" and labels[1] == "E-5\nμ=0.78"
    assert labels[-1] == "G-0\nμ=0.00"  # sin «-0.00»
    assert any(t.get_text() == "óptimo" for t in ax.texts)
    # Barras delgadas: ninguna supera BAR_MAX_IN pulgadas en la figura nueva.
    bars = [p for c in ax.containers if isinstance(c, BarContainer) for p in c]
    assert len(bars) == len(res.algorithms) * len(labels)
    width_in = fig.get_size_inches()[0] * 0.8 * max(p.get_width() for p in bars) / (len(labels) + 0.2)
    assert width_in <= plots.BAR_MAX_IN + 1e-9


def test_arm_distribution_folds_many_arms_into_others() -> None:
    """Con muchos brazos, los peores se agrupan en 'Otros (n brazos)'."""
    res = make_fake_result(n_runs=3, budget=30)
    k = 15
    rng = np.random.default_rng(3)
    ex = res.example
    res.example = ExampleSegmentResult(
        position=0, arm_labels=[f"A-{i}" for i in range(k)], true_means=np.linspace(0.9, 0.0, k),
        optimal_arms=np.array([0]), prev_fret=None,
        counts={a: rng.multinomial(30, np.full(k, 1 / k), size=3) for a in ex.counts},
        q_mean={a: np.zeros((30, k)) for a in ex.counts}, q_std={a: np.zeros((30, k)) for a in ex.counts})
    fig = plots.plot_arm_distribution(res)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert len(labels) == plots.MAX_ARM_GROUPS + 1
    assert labels[-1].startswith("Otros") and f"({k - plots.MAX_ARM_GROUPS} brazos)" in labels[-1]
    _render(fig)
    fig_q = plots.plot_q_evolution(res)
    assert "se muestran los" in _all_texts(fig_q)
    _render(fig_q)


def test_q_evolution_has_one_panel_per_algorithm_and_arm_legend(res: ExperimentResult) -> None:
    """La evolución de Q tiene un panel por algoritmo, μ real como líneas horizontales y leyenda de brazos."""
    fig = plots.plot_q_evolution(res)
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == len(res.algorithms)
    assert [ax.get_title(loc="left").split(" · ")[0] for ax in panels] == \
        ["ε-greedy", "ε-greedy optimista", "UCB1", "Softmax (Boltzmann)"]
    legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    for label in res.example.arm_labels[:5]:
        assert label in legend_text
    assert "óptimo" in legend_text
    # μ real como líneas horizontales en cada panel.
    for ax in panels:
        horizontal = [ln for ln in ax.get_lines() if np.ptp(np.asarray(ln.get_ydata(), dtype=float)) == 0]
        assert len(horizontal) >= 1


def test_sensitivity_log_axes_and_current_value(res: ExperimentResult) -> None:
    """Los barridos usan eje logarítmico donde los valores cubren órdenes de magnitud y marcan el valor actual."""
    fig = plots.plot_sensitivity(res)
    panels = {ax.get_title(loc="left").split(" — ")[0]: ax for ax in fig.axes if ax.get_visible()}
    assert set(panels) == {"ε", "Q₀", "c", "τ"}
    assert panels["c"].get_xscale() == "log" and panels["τ"].get_xscale() == "log"
    assert panels["Q₀"].get_xscale() == "log" and panels["ε"].get_xscale() == "linear"
    assert any("actual: 0.1" in t.get_text() for t in panels["ε"].texts)


def test_sensitivity_with_partial_sweeps() -> None:
    """Si solo se barrieron algunos parámetros, solo se dibujan sus paneles."""
    res = make_fake_result(n_runs=3, budget=30)
    res.sweeps = res.sweeps[:2]
    fig = plots.plot_sensitivity(res)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2
    _render(fig)


def test_lambda_effect_uses_symlog_with_zero_and_percent_axis(res: ExperimentResult) -> None:
    """El efecto de λ usa eje symlog (incluye λ = 0), eje Y de 0 a 100 % y el oráculo en la leyenda."""
    fig = plots.plot_lambda_effect(res)
    ax = fig.axes[0]
    assert ax.get_xscale() == "symlog"  # el barrido incluye λ = 0
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] >= 100
    legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "Oráculo" in legend_text


def test_accuracy_values_and_oracle_reference(res: ExperimentResult) -> None:
    """La gráfica de precisión rotula el valor de cada barra y la línea del oráculo en cada grupo."""
    fig = plots.plot_accuracy(res)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    pitch_egreedy = 100 * np.mean([a.pitch for a in res.accuracy["egreedy"]])
    assert f"{pitch_egreedy:.0f} %" in texts
    assert sum(t.startswith("Oráculo") for t in texts) == 2
    assert ax.get_ylim()[0] == 0


def test_accuracy_and_lambda_show_chain_oracle_when_available() -> None:
    """Con el oráculo de cadena (Viterbi) hay dos líneas de referencia y su fila en el heatmap."""
    res = make_fake_result(n_runs=3, budget=30)
    res.viterbi_accuracy = AccuracyResult(1.0, 0.75, res.oracle_accuracy.outcomes, n_segments=N_SEGMENTS)
    res.viterbi_arms = list(res.oracle_arms)
    ls = res.lambda_sweep
    res.lambda_sweep = type(ls)(ls.values, ls.position_acc, ls.pitch_acc, ls.oracle_position_acc,
                                ls.oracle_pitch_acc, viterbi_position_acc=np.full(len(ls.values), 0.8),
                                viterbi_pitch_acc=np.full(len(ls.values), 0.97))
    fig = plots.plot_accuracy(res)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert sum(t.startswith("Oráculo miope") for t in texts) == 2
    assert sum(t.startswith("Oráculo cadena") for t in texts) == 2
    _render(fig)
    fig = plots.plot_lambda_effect(res)
    legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "Viterbi" in legend_text and "miope" in legend_text
    _render(fig)
    fig = plots.plot_tab_heatmap(res, matches=TRUE_MATCHES)
    assert len(fig.axes[0].get_yticklabels()) == len(res.algorithms) + 2
    _render(fig)


def test_infer_gt_matches_recovers_true_matching(res: ExperimentResult) -> None:
    """El emparejamiento nota GT ↔ segmento se reconstruye a partir de los códigos guardados en el resultado."""
    assert plots._infer_gt_matches(res) == TRUE_MATCHES


def test_modal_arms_is_deterministic_with_ties() -> None:
    """La posición modal entre corridas es determinista en caso de empate."""
    choices = np.array([[[1, 0], [0, 5]], [[0, 5], [0, 5]]])  # segmento 0: empate A-0 / E-5
    arms = plots._modal_arms(choices)
    assert [a.label for a in arms] == ["E-5", "E-5"]


def test_tab_heatmap_cells_texts_and_legend(res: ExperimentResult) -> None:
    """El heatmap por nota GT colorea cada celda según el resultado (fila del oráculo incluida) y escribe la posición predicha."""
    fig = plots.plot_tab_heatmap(res)
    ax = fig.axes[0]
    mesh = ax.collections[0]
    codes = np.asarray(mesh.get_array()).reshape(len(res.algorithms) + 1, len(res.gt_notes))
    assert np.all(codes[:, 4] == OUTCOME_MISSED)  # la nota sin segmento
    oracle_row = codes[-1]
    expected = res.oracle_accuracy.outcomes
    assert np.array_equal(oracle_row, expected)
    cell_texts = [t.get_text() for t in ax.texts]
    assert "—" in cell_texts and "A-0" in cell_texts
    assert [t.get_text() for t in ax.get_xticklabels()] == [n.label for n in res.gt_notes]
    legend_text = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend_text == [OUTCOME_LABELS[k] for k in sorted(OUTCOME_LABELS)]
    # Matches explícitos dan el mismo resultado que los reconstruidos.
    fig2 = plots.plot_tab_heatmap(res, matches=TRUE_MATCHES)
    assert np.array_equal(np.asarray(fig2.axes[0].collections[0].get_array()), np.asarray(mesh.get_array()))


def test_tab_heatmap_many_notes_uses_short_text() -> None:
    """Con muchas notas el heatmap abrevia el texto de las celdas para que quepa."""
    res = make_fake_result(n_runs=2, budget=20)
    gt = res.gt_notes * 5  # 45 notas: no caben etiquetas completas
    res.gt_notes = gt
    for alg in res.accuracy:
        res.accuracy[alg] = [AccuracyResult(a.pitch, a.position, np.tile(a.outcomes, 5)) for a in res.accuracy[alg]]
    res.oracle_accuracy = AccuracyResult(0.5, 0.5, np.tile(res.oracle_accuracy.outcomes, 5))
    fig = plots.plot_tab_heatmap(res, matches=TRUE_MATCHES * 5)
    texts = [t.get_text() for t in fig.axes[0].texts if t.get_text() not in ("—",)]
    assert all("-" not in t for t in texts if not t.endswith("%") and "\n" not in t)
    _render(fig)


# ---------------------------------------------------------------------------
# Datos faltantes
# ---------------------------------------------------------------------------


def test_missing_ground_truth_shows_message() -> None:
    """Sin ground truth, las gráficas que lo necesitan muestran un mensaje explicativo en lugar de fallar."""
    res = make_fake_result(n_runs=2, budget=20)
    res.gt_notes = None
    res.accuracy = {}
    res.oracle_accuracy = None
    res.lambda_sweep = None
    for key, func in [("accuracy", plots.plot_accuracy), ("tab_heatmap", plots.plot_tab_heatmap),
                      ("lambda_effect", plots.plot_lambda_effect)]:
        assert plots.missing_data_message(key, res) == plots.MSG_NO_GT
        fig = func(res)
        assert _is_message(fig)
        assert "Requiere ground truth" in fig.axes[0].texts[0].get_text()
        _render(fig)


def test_missing_sweeps_example_and_lambda_messages() -> None:
    """Sin barridos, segmento ejemplo, barrido de λ o curvas, cada gráfica muestra su mensaje."""
    res = make_fake_result(n_runs=2, budget=20)
    res.sweeps = []
    res.example = None
    res.lambda_sweep = None
    assert _is_message(plots.plot_sensitivity(res))
    assert plots.missing_data_message("sensitivity", res) == plots.MSG_NO_SWEEPS
    assert _is_message(plots.plot_arm_distribution(res))
    assert _is_message(plots.plot_q_evolution(res))
    assert plots.missing_data_message("lambda_effect", res) == plots.MSG_NO_LAMBDA  # hay GT, falta el barrido
    empty = replace(res, reward_curves={}, regret_curves={}, optimal_curves={}, elapsed_s={})
    assert _is_message(plots.plot_average_reward(empty))
    assert plots.missing_data_message("runtime", empty) == plots.MSG_NO_RESULTS


def test_message_figure_reuses_figure() -> None:
    """message_figure dibuja el mensaje en la Figure recibida."""
    fig = Figure()
    out = plots.message_figure("Requiere ground truth: genera un MP3 sintético.", fig=fig)
    assert out is fig and _is_message(fig)
    assert fig.axes[0].texts[0].get_text().startswith("Requiere ground truth")
    _render(fig)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def test_spectrogram_has_log_frequency_note_ticks_and_overlays(analysis: AnalysisResult) -> None:
    """El espectrograma usa eje de frecuencia logarítmico con nombres de nota, onsets, segmentos descartados y la f0."""
    fig = plots.plot_spectrogram(analysis)
    _assert_plot(fig)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any(t.startswith("E1") for t in tick_labels) and any(t.startswith("A1") for t in tick_labels)
    assert ax.get_xlabel() == "Tiempo (s)" and "Hz" in ax.get_ylabel()
    f0_lines = [ln for ln in ax.get_lines() if ln.get_color() == plots.F0_COLOR]
    assert len(f0_lines) == 1
    legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "Onset" in legend_text and "descartado" in legend_text and "f0" in legend_text
    assert _title(fig).startswith("Espectrograma con onsets y f0")


def test_spectrogram_from_stft_analysis_computes_cqt() -> None:
    """Con un análisis STFT el espectrograma calcula su propia CQT y el espectro de nota maneja el bin de 0 Hz."""
    stft = make_fake_analysis("stft")
    assert stft.spectrum.kind == "stft"
    fig = plots.plot_spectrogram(stft, selected=1)
    _assert_plot(fig)
    assert "Seg. 1" in _title(fig)
    fig2 = plots.plot_note_spectrum(stft, 0, Arm("A", 0))  # la STFT incluye 0 Hz
    _assert_plot(fig2)


def test_audio_overview_shares_time_axis_and_highlights_selection(analysis: AnalysisResult) -> None:
    """La vista de audio comparte el eje temporal entre onda y espectrograma, numera los segmentos y resalta el elegido."""
    fig = plots.plot_audio_overview(analysis, selected=2)
    _assert_plot(fig, min_axes=2)
    ax_wave, ax_spec = fig.axes[0], fig.axes[1]
    assert ax_wave.get_shared_x_axes().joined(ax_wave, ax_spec)
    assert ax_spec.get_yscale() == "log"
    assert "Seg. 2" in _title(fig) and "G2" in _title(fig)
    # Números de los segmentos conservados sobre la onda.
    numbers = {t.get_text() for t in ax_wave.texts}
    assert {"0", "1", "2", "3"} <= numbers
    # Reutilización con la figura de la GUI y sin selección.
    gui = Figure(figsize=(9.0, 5.5))
    assert plots.plot_audio_overview(analysis, fig=gui) is gui
    assert "Seg." not in _title(gui)
    _render(gui)


def test_note_spectrum_combs_and_title(analysis: AnalysisResult) -> None:
    """El espectro de una nota dibuja los peines h·f y (h−½)·f del brazo elegido y del GT."""
    fig = plots.plot_note_spectrum(analysis, 0, Arm("A", 0), gt_arm=Arm("A", 12))
    _assert_plot(fig)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    title = _title(fig)
    assert "segmento 0" in title and "A1" in title and "55.0 Hz" in title
    n = analysis.config.env.n_harmonics
    arm_lines = [ln for ln in ax.get_lines() if ln.get_color() == plots.ARM_COLOR]
    gt_lines = [ln for ln in ax.get_lines() if ln.get_color() == plots.GT_COLOR]
    assert len(arm_lines) >= n and len(gt_lines) >= n  # h·f (+ huecos (h−½)·f visibles)
    legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "A-0" in legend_text and "A-12" in legend_text
    # Mismo pitch, otra posición: un solo peine y aviso en el subtítulo.
    same = plots.plot_note_spectrum(analysis, 0, Arm("A", 0), gt_arm=Arm("E", 5))
    assert not [ln for ln in same.axes[0].get_lines() if ln.get_color() == plots.GT_COLOR]
    assert "misma altura" in _all_texts(same)
    with pytest.raises(IndexError):
        plots.plot_note_spectrum(analysis, 99, Arm("A", 0))


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------


def test_save_figure_creates_folder_and_writes_png_pdf(tmp_path: Path, res: ExperimentResult) -> None:
    """save_figure crea la carpeta y escribe PNG y PDF válidos."""
    out_dir = tmp_path / "a" / "b"
    paths = plots.save_figure(plots.plot_runtime(res), out_dir, "runtime", dpi=50)
    assert paths == [out_dir / "runtime.png", out_dir / "runtime.pdf"]
    assert paths[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths[1].read_bytes()[:5] == b"%PDF-"


def test_save_all_plots_writes_every_plot_and_spectrogram(tmp_path: Path, res: ExperimentResult,
                                                          analysis: AnalysisResult) -> None:
    """save_all_plots escribe todas las gráficas del catálogo y el espectrograma."""
    paths = plots.save_all_plots(res, analysis, tmp_path / "figs", dpi=40)
    names = {p.name for p in paths}
    for spec in plots.COMPARISON_PLOTS:
        assert {f"{spec.key}.png", f"{spec.key}.pdf"} <= names
    assert {"spectrogram.png", "spectrogram.pdf"} <= names
    assert all(p.stat().st_size > 1000 for p in paths)
    assert all(p.read_bytes()[:4] == (b"%PDF" if p.suffix == ".pdf" else b"\x89PNG") for p in paths)


def test_save_all_plots_skips_plots_without_data(tmp_path: Path) -> None:
    """save_all_plots omite (sin fallar) las gráficas que no tienen datos."""
    res = make_fake_result(n_runs=2, budget=20)
    res.gt_notes = None
    res.accuracy = {}
    res.oracle_accuracy = None
    res.lambda_sweep = None
    paths = plots.save_all_plots(res, None, tmp_path, formats=("png",), dpi=40)
    names = {p.stem for p in paths}
    assert "accuracy" not in names and "tab_heatmap" not in names and "lambda_effect" not in names
    assert {"average_reward", "cumulative_regret", "sensitivity"} <= names
    assert "spectrogram" not in names


@pytest.mark.parametrize("func", [plots.plot_average_reward, plots.plot_cumulative_regret, plots.plot_optimal_action,
                                  plots.plot_accuracy, plots.plot_runtime, plots.plot_tab_heatmap])
def test_single_run_result_has_no_band_errors(func: Callable[..., Figure]) -> None:
    """Con una sola corrida la std es 0: no hay banda, pero la gráfica se dibuja igual."""
    _render(func(make_fake_result(n_runs=1, budget=15)))
