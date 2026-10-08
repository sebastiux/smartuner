"""Smartuner — transcripción de bajo a tablatura con Multi-Armed Bandits.

Punto de entrada único del proyecto: abre la interfaz gráfica (tkinter) o,
con ``--cli``, ejecuta el sistema desde la línea de comandos. Toda la lógica
vive en ``src/``; este archivo solo traduce argumentos a llamadas.

Uso
---
Interfaz gráfica (por defecto)::

    python main.py                    # abre la GUI
    python main.py pista.mp3          # abre la GUI y analiza pista.mp3

Línea de comandos (``--cli`` + subcomando)::

    # 1. Dataset sintético con ground truth (MIDI → audio → MP3 + .gt.json)
    python main.py --cli dataset [--out data/synthetic] [--bpm 100]
                                 [--method auto|fluidsynth|karplus-strong]

    # 2. Transcribir una pista con un algoritmo: imprime la tablatura ASCII y la
    #    exporta a TXT/JSON/CSV; si hay ground truth imprime la precisión.
    python main.py --cli analyze AUDIO [--algo ucb1] [--separate] [--config cfg.json]
                                       [--seed N] [--out results/<nombre>]

    # 3. Experimento comparativo de los 4 algoritmos: tabla, CSV y gráficas.
    python main.py --cli experiment AUDIO [--runs 100] [--budget T] [--no-sweeps]
                                          [--no-lambda] [--config cfg.json] [--seed N]
                                          [--out results/<nombre>_<fecha>]

Opciones comunes: ``--quiet`` (solo avisos y errores) y ``--verbose``
(detalle DEBUG). El log va a la consola con el formato
``HH:MM:SS NIVEL módulo: mensaje``. Si se escribe un subcomando sin ``--cli``
se entiende igualmente que se quiere la línea de comandos.

Ejemplos
--------
::

    python main.py --cli dataset
    python main.py --cli analyze data/synthetic/linea_simple.mp3 --algo softmax
    python main.py --cli experiment data/synthetic/cromatica.mp3 --runs 100 --no-sweeps

Códigos de salida: 0 = correcto, 1 = error (archivo ilegible, Demucs ausente,
gráficas fallidas...), 2 = argumentos inválidos, 130 = cancelado con Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import ALGO_LABELS, ALGORITHMS, DATA_DIR, RESULTS_DIR, Config

logger = logging.getLogger("smartuner")

#: Subcomandos de la línea de comandos.
COMMANDS: tuple[str, ...] = ("dataset", "analyze", "experiment")

#: Formato del log en consola: "HH:MM:SS NIVEL módulo: mensaje".
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Utilidades de consola
# ---------------------------------------------------------------------------


def setup_logging(quiet: bool = False, verbose: bool = False) -> None:
    """Configura el log de consola (INFO por defecto; WARNING con ``quiet``; DEBUG con ``verbose``).

    Parameters
    ----------
    quiet : bool
        Solo avisos y errores.
    verbose : bool
        Todos los detalles (DEBUG). Tiene prioridad sobre ``quiet``.
    """
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    # Bibliotecas muy habladoras en DEBUG: solo sus avisos.
    for noisy in ("numba", "matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class ConsoleProgress:
    """Callback de progreso para la consola: informa por el log cada ``step`` (fracción).

    Parameters
    ----------
    label : str
        Prefijo del mensaje (p. ej. ``"Experimento"``).
    step : float
        Cada cuánto (fracción del total) se escribe una línea.
    """

    def __init__(self, label: str, step: float = 0.1) -> None:
        self.label = label
        self.step = step
        self._next = step
        self._t0 = time.perf_counter()

    def __call__(self, fraction: float, message: str) -> None:
        """Informa el avance; solo escribe una línea al cruzar cada múltiplo de ``step``.

        Parameters
        ----------
        fraction : float
            Avance del proceso en [0, 1].
        message : str
            Qué se está haciendo (se copia en la línea del log).
        """
        if fraction + 1e-9 < self._next:
            return
        while self._next <= fraction + 1e-9:  # salta los umbrales ya superados
            self._next += self.step
        logger.info("%s: %3.0f %% (%.0f s) — %s", self.label, 100 * fraction,
                    time.perf_counter() - self._t0, message)


def load_config(path: str | None) -> Config:
    """Configuración desde JSON (``--config``) o la de por defecto.

    Parameters
    ----------
    path : str | None
        Archivo JSON guardado desde la GUI (o escrito a mano); None = valores
        por defecto.

    Returns
    -------
    Config
        Configuración validada (:meth:`src.config.Config.validate`).

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si no es un JSON válido o algún parámetro es inválido: se detecta
        aquí, ANTES de decodificar y analizar el audio.
    """
    if path is None:
        return Config()
    cfg = Config.load_json(path)
    logger.info("Configuración cargada de %s", path)
    return cfg


def _check_out_dir(out_dir: Path) -> Path:
    """Comprueba que ``out_dir`` se puede usar como carpeta de salida (sin crearla).

    Parameters
    ----------
    out_dir : Path
        Carpeta pedida con ``--out`` (o la de por defecto).

    Returns
    -------
    Path
        La misma ruta.

    Raises
    ------
    ValueError
        Si ya existe y no es una carpeta (se avisa ANTES del análisis, no al exportar).
    """
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out apunta a un archivo existente, no a una carpeta: {out_dir}")
    return out_dir


def _fmt_mean_std(mean: Any, std: Any, scale: float = 1.0, decimals: int = 3, suffix: str = "") -> str:
    """``"0.612 ± 0.004"``; ``"—"`` si no hay valor.

    Parameters
    ----------
    mean, std : Any
        Media y desviación (None = sin dato; una std None se escribe como 0).
    scale : float, optional
        Factor aplicado a ambos (100 para porcentajes).
    decimals : int, optional
        Decimales mostrados.
    suffix : str, optional
        Texto final (p. ej. ``" %"``).

    Returns
    -------
    str
        Texto formateado.

    Examples
    --------
    >>> _fmt_mean_std(0.75, 0.05, scale=100, decimals=1, suffix=" %"), _fmt_mean_std(None, None)
    ('75.0 ± 5.0 %', '—')
    """
    if mean is None:
        return "—"
    return f"{scale * float(mean):.{decimals}f} ± {scale * float(std or 0.0):.{decimals}f}{suffix}"


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    """Tabla comparativa en texto (una fila por algoritmo y por cada oráculo).

    Parameters
    ----------
    rows : list[dict]
        Salida de :meth:`src.experiments.ExperimentResult.summary_rows`.

    Returns
    -------
    str
        Tabla alineada con columnas: recompensa media, regret final por
        segmento, % de pulls óptimos (último 10 %), precisiones (notas del GT
        acertadas), F1 de posición (penaliza también los segmentos de más) y
        ms/segmento. Los valores ausentes se muestran como «—».
    """
    headers = ["Algoritmo", "Recompensa media", "Regret final", "% óptimo (últ. 10 %)",
               "Precisión pitch", "Precisión posición", "F1 posición", "ms/segmento"]
    table: list[list[str]] = []
    for r in rows:
        table.append([
            str(r["label"]),
            _fmt_mean_std(r.get("mean_reward"), r.get("mean_reward_std")),
            _fmt_mean_std(r.get("final_regret"), r.get("final_regret_std"), decimals=2),
            _fmt_mean_std(r.get("optimal_pct"), r.get("optimal_pct_std"), decimals=1),
            _fmt_mean_std(r.get("pitch_acc"), r.get("pitch_acc_std"), scale=100, decimals=1, suffix=" %"),
            _fmt_mean_std(r.get("position_acc"), r.get("position_acc_std"), scale=100, decimals=1, suffix=" %"),
            _fmt_mean_std(r.get("position_f1"), r.get("position_f1_std"), scale=100, decimals=1, suffix=" %"),
            _fmt_mean_std(r.get("ms_per_segment"), r.get("ms_per_segment_std"), decimals=2),
        ])
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths)),
             "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(c.ljust(w) for c, w in zip(row, widths)) for row in table]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------


def cmd_dataset(args: argparse.Namespace) -> int:
    """``dataset``: genera las piezas sintéticas (MIDI, MP3 y ground truth).

    Parameters
    ----------
    args : argparse.Namespace
        Usa ``out`` (carpeta), ``bpm`` (negras por minuto) y ``method``.

    Returns
    -------
    int
        Código de salida (0 = correcto).
    """
    from src.synth_dataset import generate_dataset

    cfg = Config()
    items = generate_dataset(args.out, bpm=args.bpm, sr=cfg.audio.sample_rate, method=args.method,
                             progress=ConsoleProgress("Dataset", step=0.25))
    print(f"\nDataset generado en {Path(args.out).resolve()}:")
    for it in items:
        print(f"  • {it.mp3_path.name:<20} ({it.method}) + {it.gt_path.name} + {it.midi_path.name}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """``analyze``: análisis + transcripción con un algoritmo + exportación (+ precisión si hay GT).

    Parameters
    ----------
    args : argparse.Namespace
        Usa ``audio``, ``algo``, ``separate``, ``config``, ``seed`` y ``out``.

    Returns
    -------
    int
        Código de salida: 0 = tablatura exportada; 1 = no se detectó ninguna nota.
    """
    from src.experiments import match_segments_to_gt, oracle_arms, oracle_viterbi_arms, score_arms, transcribe
    from src.pipeline import analyze
    from src.tab import export_csv, export_json, export_txt, render_ascii

    audio = Path(args.audio)
    cfg = load_config(args.config)
    if args.separate:
        cfg.audio.separate_bass = True
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    cfg.validate()
    out_dir = _check_out_dir(Path(args.out) if args.out else RESULTS_DIR / audio.stem)

    analysis = analyze(audio, cfg)
    if not analysis.segment_data:
        print("No se detectó ninguna nota (todos los segmentos se descartaron): no hay tablatura.")
        return 1
    tr = transcribe(analysis, args.algo, cfg)
    title = f"{audio.name} — {ALGO_LABELS[args.algo]} (T={cfg.env.budget}, λ={cfg.env.lam:g}, semilla {cfg.experiment.seed})"

    print()
    print(render_ascii(tr.notes, title=title))
    print()

    meta: dict[str, Any] = {
        "title": title,
        "audio": str(audio),
        "analyzed_audio": str(analysis.source_path),
        "algorithm": args.algo,
        "algorithm_label": ALGO_LABELS[args.algo],
        "seed": cfg.experiment.seed,
        "n_notes": len(tr.notes),
        "config": cfg.to_dict(),
    }
    gt = analysis.ground_truth
    if gt:
        matches = match_segments_to_gt([d.segment for d in analysis.segment_data], gt, cfg.experiment.onset_tolerance_s)
        acc = score_arms(tr.chain.arms, matches, gt)
        oracle = score_arms(oracle_arms(analysis.segment_data, cfg), matches, gt)
        chain = score_arms(oracle_viterbi_arms(analysis.segment_data, cfg), matches, gt)
        print(f"Ground truth: {len(gt)} notas ({acc.n_missed} sin detectar, {acc.n_extra} segmentos de más; "
              f"tolerancia ±{1000 * cfg.experiment.onset_tolerance_s:.0f} ms)")
        print(f"Precisión de {ALGO_LABELS[args.algo]}: pitch {100 * acc.pitch:.1f} % · posición {100 * acc.position:.1f} % "
              f"(F1 {100 * acc.pitch_f1:.1f} % / {100 * acc.position_f1:.1f} %)")
        for key, ref in (("oracle", oracle), ("oracle_viterbi", chain)):
            print(f"Referencia, {ALGO_LABELS[key]}: pitch {100 * ref.pitch:.1f} % · posición {100 * ref.position:.1f} %")
        print("(precisión = notas del GT acertadas / notas del GT; el F1 también penaliza los segmentos de más)")
        print()
        meta["accuracy"] = {"pitch": acc.pitch, "position": acc.position,
                            "pitch_f1": acc.pitch_f1, "position_f1": acc.position_f1,
                            "n_missed": acc.n_missed, "n_extra": acc.n_extra,
                            "oracle_pitch": oracle.pitch, "oracle_position": oracle.position,
                            "oracle_chain_pitch": chain.pitch, "oracle_chain_position": chain.position}

    base = out_dir / f"{audio.stem}_{args.algo}"
    paths = [export_txt(tr.notes, base.with_suffix(".txt"), title=title),
             export_json(tr.notes, base.with_suffix(".json"), meta=meta),
             export_csv(tr.notes, base.with_suffix(".csv"))]
    print("Tablatura exportada:")
    for p in paths:
        print(f"  {p}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """``experiment``: experimento comparativo completo + CSV + gráficas + tabla.

    La carpeta de salida se crea solo cuando el análisis y el experimento
    terminaron bien (un audio ilegible no deja carpetas vacías en ``results/``).

    Parameters
    ----------
    args : argparse.Namespace
        Usa ``audio``, ``runs``, ``budget``, ``no_sweeps``, ``no_lambda``,
        ``config``, ``seed`` y ``out``.

    Returns
    -------
    int
        Código de salida: 0 = correcto; 1 = las gráficas fallaron (los CSV sí
        se guardaron).
    """
    from src.experiments import run_experiment
    from src.pipeline import analyze

    audio = Path(args.audio)
    cfg = load_config(args.config)
    if args.runs is not None:
        cfg.experiment.n_runs = args.runs
    if args.budget is not None:
        cfg.env.budget = args.budget
    if args.no_sweeps:
        cfg.experiment.run_sweeps = False
    if args.no_lambda:
        cfg.experiment.run_lambda_sweep = False
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    cfg.validate()   # p. ej. --runs 0 o --budget 0: error claro antes de analizar
    out_dir = _check_out_dir(Path(args.out) if args.out else
                             RESULTS_DIR / f"{audio.stem}_{datetime.now():%Y%m%d-%H%M%S}")

    analysis = analyze(audio, cfg)
    t0 = time.perf_counter()
    result = run_experiment(analysis, cfg, progress=ConsoleProgress("Experimento", step=0.1))
    logger.info("Experimento completo en %.1f s", time.perf_counter() - t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "config.json")
    summary_csv = result.to_csv(str(out_dir / "summary.csv"))       # reproducible con la misma semilla
    curves_csv = result.curves_to_csv(str(out_dir / "curves.csv"))    # reproducible con la misma semilla
    timing_csv = result.timing_to_csv(str(out_dir / "timing.csv"))    # tiempos de pared (varían)
    saved = [Path(summary_csv), Path(curves_csv), Path(timing_csv), out_dir / "config.json"]

    exit_code = 0
    try:
        from src.plots import save_all_plots

        figures = save_all_plots(result, analysis, out_dir)
        saved += list(figures)
        logger.info("%d archivos de gráficas guardados en %s", len(figures), out_dir)
    except Exception:  # noqa: BLE001 — los resultados ya están guardados; se informa y se sigue
        logger.exception("No se pudieron generar las gráficas (los CSV sí se guardaron)")
        exit_code = 1

    print()
    print(f"Experimento: {audio.name} — {result.n_runs} corridas × {result.n_segments} segmentos × T={result.budget}")
    print(format_summary_table(result.summary_rows()))
    if result.gt_notes:
        acc = next((runs[0] for runs in result.accuracy.values() if runs), None)
        if acc is not None:
            print(f"Ground truth: {len(result.gt_notes)} notas · {acc.n_missed} sin detectar · {acc.n_extra} segmentos "
                  "de más. Precisión = notas del GT acertadas / notas del GT; el F1 también penaliza los segmentos "
                  "de más.")
    print()
    print(f"Resultados en {out_dir}:")
    for p in saved:
        print(f"  {p}")
    return exit_code


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Parser de argumentos: ``--cli`` + subcomandos ``dataset``, ``analyze`` y ``experiment``.

    Returns
    -------
    argparse.ArgumentParser
        Parser con un subparser por comando; cada uno guarda su función en
        ``func`` (``args.func(args)`` ejecuta el comando).
    """
    defaults = Config()
    # --quiet/--verbose se aceptan antes o después del subcomando.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="solo mostrar avisos y errores en el log")
    common.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="mostrar el log detallado (DEBUG)")

    parser = argparse.ArgumentParser(
        prog="main.py", parents=[common],
        description="Smartuner: tablatura de bajo con Multi-Armed Bandits. Sin argumentos abre la GUI.",
        epilog="Ejemplo: python main.py --cli analyze data/synthetic/linea_simple.mp3 --algo ucb1",
    )
    parser.add_argument("--cli", action="store_true", help="usar la línea de comandos en lugar de la GUI")
    sub = parser.add_subparsers(dest="command", metavar="{dataset,analyze,experiment}")

    p_ds = sub.add_parser("dataset", parents=[common], help="generar el dataset sintético con ground truth")
    p_ds.add_argument("--out", default=str(DATA_DIR / "synthetic"), help="carpeta de salida (por defecto data/synthetic)")
    p_ds.add_argument("--bpm", type=float, default=100.0, help="tempo en negras por minuto (por defecto 100)")
    p_ds.add_argument("--method", choices=("auto", "fluidsynth", "karplus-strong"), default="auto",
                      help="sintetizador: auto (fluidsynth si está disponible), fluidsynth o karplus-strong")
    p_ds.set_defaults(func=cmd_dataset)

    p_an = sub.add_parser("analyze", parents=[common], help="transcribir un audio con un algoritmo")
    p_an.add_argument("audio", help="archivo de audio (MP3, WAV, ...)")
    p_an.add_argument("--algo", choices=ALGORITHMS, default="ucb1", help="algoritmo bandit (por defecto ucb1)")
    p_an.add_argument("--separate", action="store_true", help="aislar el bajo de una mezcla con Demucs")
    p_an.add_argument("--config", help="configuración JSON (guardada desde la GUI)")
    p_an.add_argument("--seed", type=int, help="semilla maestra (por defecto la de la configuración)")
    p_an.add_argument("--out", help="carpeta de salida (por defecto results/<nombre>)")
    p_an.set_defaults(func=cmd_analyze)

    p_ex = sub.add_parser("experiment", parents=[common], help="experimento comparativo de los 4 algoritmos")
    p_ex.add_argument("audio", help="archivo de audio (idealmente con .gt.json para medir la precisión)")
    p_ex.add_argument("--runs", type=int,
                      help=f"corridas por algoritmo (por defecto las de la configuración: {defaults.experiment.n_runs})")
    p_ex.add_argument("--budget", type=int,
                      help=f"presupuesto T de pulls por segmento (por defecto el de la configuración: {defaults.env.budget})")
    p_ex.add_argument("--no-sweeps", action="store_true", help="omitir los barridos de ε, Q₀, c y τ")
    p_ex.add_argument("--no-lambda", action="store_true", help="omitir el barrido de λ")
    p_ex.add_argument("--config", help="configuración JSON (guardada desde la GUI)")
    p_ex.add_argument("--seed", type=int, help="semilla maestra (por defecto la de la configuración)")
    p_ex.add_argument("--out", help="carpeta de salida (por defecto results/<nombre>_<fecha>)")
    p_ex.set_defaults(func=cmd_experiment)
    return parser


def launch_gui(argv: list[str]) -> int:
    """Abre la GUI (import diferido: la CLI no necesita tkinter).

    Parameters
    ----------
    argv : list[str]
        Argumentos para la GUI (p. ej. la ruta de un audio para analizarlo al abrir).

    Returns
    -------
    int
        Código de salida de la GUI (1 si tkinter o la GUI no están disponibles).
    """
    try:
        from gui.app import main as gui_main
    except ImportError as exc:
        print(f"No se pudo cargar la interfaz gráfica ({exc}).\n"
              "Usa la línea de comandos: python main.py --cli --help", file=sys.stderr)
        return 1
    try:
        return int(gui_main(argv) or 0)
    except NotImplementedError as exc:
        print(f"La interfaz gráfica aún no está completa ({exc or 'NotImplementedError'}).\n"
              "Usa la línea de comandos: python main.py --cli --help", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: GUI sin ``--cli`` (ni subcomando); CLI en caso contrario.

    Parameters
    ----------
    argv : list[str] | None
        Argumentos (sin el nombre del programa); None usa ``sys.argv[1:]``.

    Returns
    -------
    int
        Código de salida.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    _ensure_utf8_output()
    wants_cli = "--cli" in argv or any(a in COMMANDS for a in argv)
    if not wants_cli and not any(a in ("-h", "--help") for a in argv):
        return launch_gui(argv)

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    setup_logging(quiet=getattr(args, "quiet", False), verbose=getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.", file=sys.stderr)
        return 130
    except _expected_errors() as exc:
        # AudioLoadError, FFmpegNotFoundError, SeparationError y CancelledError son RuntimeError;
        # OSError cubre carpetas/archivos inaccesibles (FileExistsError, IsADirectoryError...).
        logger.debug("Detalle del error", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _ensure_utf8_output() -> None:
    """Fuerza UTF-8 en stdout/stderr cuando la codificación no lo es.

    En Windows, si la salida se redirige a un archivo o tubería, Python usa la
    codificación local (cp1252), que no tiene letras griegas (ε, τ, λ, μ) y
    haría fallar los ``print`` de la tabla y la tablatura. Con UTF-8 (y
    ``errors="replace"`` como red de seguridad) la CLI nunca se cae por eso.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # pragma: no cover - flujo no reconfigurable
                pass


def _expected_errors() -> tuple[type[BaseException], ...]:
    """Excepciones que la CLI muestra como ``Error: …`` (código 1) en lugar de una traza.

    Returns
    -------
    tuple[type[BaseException], ...]
        ``ValueError``, ``RuntimeError``, ``OSError`` (incluye
        ``FileNotFoundError``) y ``librosa.util.exceptions.ParameterError``
        (parámetros que librosa rechaza) si librosa está instalado.
    """
    errors: list[type[BaseException]] = [ValueError, RuntimeError, OSError]
    try:
        from librosa.util.exceptions import ParameterError
    except ImportError:  # pragma: no cover - librosa es una dependencia obligatoria
        pass
    else:
        errors.append(ParameterError)
    return tuple(errors)


if __name__ == "__main__":
    raise SystemExit(main())
