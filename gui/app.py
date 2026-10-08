"""Ventana principal de Smartuner (tkinter) y controlador de la aplicación.

Papel en la arquitectura
------------------------
La GUI NO contiene lógica del sistema: solo llama a los módulos de ``src/``
(que funcionan igual desde la CLI). Este módulo define:

* :class:`EventBus` — publicación/suscripción simple para que las pestañas se
  enteren de los cambios sin conocerse entre sí.
* :class:`AppState` — el estado compartido (configuración, análisis,
  transcripciones, resultados del experimento, segmento seleccionado).
* :class:`SmartunerApp` — crea la ventana, el menú, las seis pestañas
  (``ttk.Notebook``) y la barra de estado, y ofrece las ACCIONES de alto nivel
  (abrir/analizar audio, re-transcribir, ejecutar el experimento, generar el
  dataset, guardar/cargar configuración) que corren en hilos con
  :class:`gui.workers.WorkerManager`.

Eventos publicados en ``app.state.events`` (argumentos por palabra clave):

==========================  =========================================================
``config_changed``           ``key`` (str | None): un parámetro cambió (None = varios).
``analysis_ready``           ``analysis``: nuevo :class:`src.pipeline.AnalysisResult`.
``transcriptions_ready``     ``transcriptions``: dict algoritmo → TranscriptionResult.
``experiment_ready``         ``result``: :class:`src.experiments.ExperimentResult`.
``segment_selected``         ``position`` (int | None), ``source`` (str): pestaña origen.
``algorithm_selected``       ``algorithm`` (str).
``busy_changed``             ``busy`` (bool): hay/no hay una tarea en segundo plano.
==========================  =========================================================
"""

from __future__ import annotations

import copy
import logging
import sys
import threading
import tkinter as tk
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from gui.widgets import StatusBar, show_error
from gui.workers import QueueLogHandler, WorkerManager
from src.config import ALGORITHMS, DATA_DIR, Config, ProgressCallback

logger = logging.getLogger(__name__)

APP_TITLE = "Smartuner — Tablatura de bajo con Multi-Armed Bandits"


class EventBus:
    """Publicación/suscripción mínima (todo ocurre en el hilo de la GUI)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable[..., None]) -> None:
        """Registra ``callback`` para ``event``."""
        self._subscribers[event].append(callback)

    def emit(self, event: str, **kwargs: Any) -> None:
        """Llama a todos los suscriptores de ``event``; un suscriptor roto no afecta a los demás."""
        for callback in list(self._subscribers[event]):
            try:
                callback(**kwargs)
            except Exception:  # noqa: BLE001
                logger.exception("Error en un suscriptor del evento '%s'", event)


@dataclass
class AppState:
    """Estado compartido entre pestañas.

    Attributes
    ----------
    config : Config
        Configuración vigente (la edita la pestaña Configuración).
    analysis : AnalysisResult | None
        Último análisis de audio.
    transcriptions : dict[str, TranscriptionResult]
        Tablatura de una corrida por algoritmo.
    experiment : ExperimentResult | None
        Resultados del experimento comparativo.
    selected_position : int | None
        Segmento seleccionado (índice entre los conservados).
    selected_algorithm : str
        Algoritmo mostrado en las vistas de un solo algoritmo.
    events : EventBus
        Bus de eventos.
    """

    config: Config = field(default_factory=Config)
    analysis: Any = None
    transcriptions: dict[str, Any] = field(default_factory=dict)
    experiment: Any = None
    selected_position: int | None = None
    selected_algorithm: str = "ucb1"
    events: EventBus = field(default_factory=EventBus)

    def select_segment(self, position: int | None, source: str = "") -> None:
        """Cambia el segmento seleccionado y avisa a las pestañas."""
        if position == self.selected_position:
            return
        self.selected_position = position
        self.events.emit("segment_selected", position=position, source=source)

    def select_algorithm(self, algorithm: str) -> None:
        """Cambia el algoritmo mostrado y avisa a las pestañas."""
        if algorithm == self.selected_algorithm:
            return
        self.selected_algorithm = algorithm
        self.events.emit("algorithm_selected", algorithm=algorithm)


class SmartunerApp:
    """Aplicación tkinter: ventana, pestañas y acciones de alto nivel.

    Parameters
    ----------
    root : tk.Tk | None
        Ventana raíz (se crea si es None).
    config : Config | None
        Configuración inicial.
    """

    def __init__(self, root: tk.Tk | None = None, config: Config | None = None) -> None:
        self.root = root or tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1360x880")
        self.root.minsize(1024, 700)
        self.state = AppState(config=config or Config())
        self.workers = WorkerManager(self.root)
        self.log_handler = QueueLogHandler(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        self._setup_style()
        self._build_menu()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        logger.info("Smartuner listo. Abre un MP3 (Archivo → Abrir) o genera el dataset sintético.")

    # ------------------------------------------------------------------ UI
    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Accent.TButton", foreground="white", background="#2a78d6")
        style.map("Accent.TButton", background=[("active", "#256abf"), ("disabled", "#9ec5f4")])
        style.configure("Header.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#52514e")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Abrir audio (MP3)…", command=self.open_audio, accelerator="Ctrl+O")
        file_menu.add_command(label="Generar dataset sintético…", command=self.generate_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar configuración…", command=self.save_config)
        file_menu.add_command(label="Cargar configuración…", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.quit)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        run_menu = tk.Menu(menubar, tearoff=False)
        run_menu.add_command(label="Re-transcribir con la configuración actual", command=self.retranscribe)
        run_menu.add_command(label="Ejecutar experimento completo", command=self.run_experiment)
        menubar.add_cascade(label="Ejecutar", menu=run_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Acerca de", command=self._about)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        self.root.config(menu=menubar)
        self.root.bind_all("<Control-o>", lambda _e: self.open_audio())

    def _build_layout(self) -> None:
        from gui.tabs.audio_tab import AudioTab
        from gui.tabs.compare_tab import CompareTab
        from gui.tabs.config_tab import ConfigTab
        from gui.tabs.live_tab import LiveTab
        from gui.tabs.log_tab import LogTab
        from gui.tabs.tablature_tab import TablatureTab

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="top", fill="both", expand=True)
        self.status = StatusBar(self.root, on_cancel=self.workers.cancel_all)
        self.status.pack(side="bottom", fill="x")
        self.tabs: dict[str, ttk.Frame] = {}
        for key, title, cls in (
            ("audio", "1 · Audio", AudioTab),
            ("config", "2 · Configuración", ConfigTab),
            ("live", "3 · Ejecución en vivo", LiveTab),
            ("tab", "4 · Tablatura", TablatureTab),
            ("compare", "5 · Comparación", CompareTab),
            ("log", "6 · Log", LogTab),
        ):
            frame = cls(self.notebook, self)
            self.notebook.add(frame, text=title)
            self.tabs[key] = frame

    def show_tab(self, key: str) -> None:
        """Muestra la pestaña ``key`` (``"audio"``, ``"config"``, ``"live"``, ``"tab"``, ``"compare"``, ``"log"``)."""
        self.notebook.select(self.tabs[key])

    def _about(self) -> None:
        messagebox.showinfo(
            "Acerca de Smartuner",
            "Transcripción de tablatura de bajo desde MP3 comparando cuatro algoritmos de "
            "Multi-Armed Bandits: ε-greedy, ε-greedy optimista, UCB1 y Softmax.\n\n"
            "Cada nota es un problema bandit: los brazos son posiciones (cuerda, traste) y la "
            "recompensa mide la energía armónica del audio en esa frecuencia menos una "
            "penalización por mover la mano.",
            parent=self.root,
        )

    # ------------------------------------------------------------ tareas
    @property
    def busy(self) -> bool:
        """True si hay una tarea en segundo plano."""
        return self.workers.busy

    def run_task(
        self,
        title: str,
        fn: Callable[[ProgressCallback, threading.Event], Any],
        on_done: Callable[[Any], None] | None = None,
        error_title: str | None = None,
    ) -> bool:
        """Ejecuta ``fn(progress, cancel)`` en un hilo, con barra de progreso y manejo de errores.

        Returns
        -------
        bool
            False si ya había una tarea en curso (no se lanza otra).
        """
        if self.workers.busy:
            messagebox.showinfo("Tarea en curso", "Espera a que termine la tarea actual (o cancélala).", parent=self.root)
            return False

        def finish() -> None:
            self.status.set_busy(False)
            self.state.events.emit("busy_changed", busy=False)

        def done(result: Any) -> None:
            finish()
            self.status.set_message(f"{title}: terminado.")
            if on_done:
                on_done(result)

        def error(exc: BaseException, tb: str) -> None:
            finish()
            self.status.set_message(f"{title}: error.")
            show_error(self.root, error_title or f"Error: {title}", exc, tb)

        def cancelled() -> None:
            finish()
            self.status.set_message(f"{title}: cancelado.")

        self.status.set_busy(True)
        self.status.set_progress(0.0, f"{title}…")
        self.state.events.emit("busy_changed", busy=True)
        self.workers.start(title, fn, on_done=done, on_error=error, on_progress=lambda f, m: self.status.set_progress(f, m or None), on_cancelled=cancelled)
        return True

    # ----------------------------------------------------------- acciones
    def open_audio(self, path: str | Path | None = None) -> None:
        """Pide un archivo de audio (si ``path`` es None) y lo analiza."""
        if path is None:
            initial = DATA_DIR / "synthetic"
            chosen = filedialog.askopenfilename(
                parent=self.root, title="Abrir audio de bajo",
                initialdir=str(initial if initial.exists() else DATA_DIR),
                filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("Todos los archivos", "*.*")],
            )
            if not chosen:
                return
            path = chosen
        self.analyze(Path(path))

    def analyze(self, path: Path) -> None:
        """Analiza ``path`` (etapas 1–6) y transcribe con los cuatro algoritmos, en segundo plano."""
        from src.experiments import transcribe
        from src.pipeline import analyze as run_analysis

        cfg = copy.deepcopy(self.state.config)

        def task(progress: ProgressCallback, cancel: threading.Event) -> tuple[Any, dict[str, Any]]:
            analysis = run_analysis(path, cfg, progress=lambda f, m: progress(0.85 * f, m), cancel=cancel)
            transcriptions: dict[str, Any] = {}
            for i, algo in enumerate(ALGORITHMS):
                progress(0.85 + 0.15 * i / len(ALGORITHMS), f"Transcribiendo con {algo}…")
                transcriptions[algo] = transcribe(analysis, algo, cfg)
            return analysis, transcriptions

        def done(result: tuple[Any, dict[str, Any]]) -> None:
            analysis, transcriptions = result
            self.state.analysis = analysis
            self.state.transcriptions = transcriptions
            self.state.experiment = None
            self.state.selected_position = None
            self.root.title(f"{APP_TITLE} — {Path(path).name}")
            self.state.events.emit("analysis_ready", analysis=analysis)
            self.state.events.emit("transcriptions_ready", transcriptions=transcriptions)

        self.run_task(f"Analizando {Path(path).name}", task, done, error_title="No se pudo analizar el audio")

    def retranscribe(self) -> None:
        """Recalcula los entornos con la configuración actual y vuelve a transcribir (sin re-decodificar)."""
        if self.state.analysis is None:
            messagebox.showinfo("Sin audio", "Primero abre y analiza un archivo de audio.", parent=self.root)
            return
        from src.experiments import transcribe
        from src.pipeline import rebuild_segment_data

        cfg = copy.deepcopy(self.state.config)
        old = self.state.analysis

        def task(progress: ProgressCallback, cancel: threading.Event) -> tuple[Any, dict[str, Any]]:
            progress(0.1, "Recalculando entornos bandit…")
            analysis = rebuild_segment_data(old, cfg)
            transcriptions: dict[str, Any] = {}
            for i, algo in enumerate(ALGORITHMS):
                progress(0.3 + 0.7 * i / len(ALGORITHMS), f"Transcribiendo con {algo}…")
                transcriptions[algo] = transcribe(analysis, algo, cfg)
            return analysis, transcriptions

        def done(result: tuple[Any, dict[str, Any]]) -> None:
            self.state.analysis, self.state.transcriptions = result
            self.state.events.emit("analysis_ready", analysis=self.state.analysis)
            self.state.events.emit("transcriptions_ready", transcriptions=self.state.transcriptions)

        self.run_task("Re-transcribiendo", task, done)

    def run_experiment(self) -> None:
        """Ejecuta el experimento comparativo completo en segundo plano."""
        if self.state.analysis is None:
            messagebox.showinfo("Sin audio", "Primero abre y analiza un archivo de audio.", parent=self.root)
            return
        from src.experiments import run_experiment
        from src.pipeline import rebuild_segment_data

        cfg = copy.deepcopy(self.state.config)
        analysis = self.state.analysis

        def task(progress: ProgressCallback, cancel: threading.Event) -> Any:
            current = rebuild_segment_data(analysis, cfg)
            return run_experiment(current, cfg, progress=progress, cancel=cancel)

        def done(result: Any) -> None:
            self.state.experiment = result
            self.state.events.emit("experiment_ready", result=result)
            self.show_tab("compare")

        self.run_task("Experimento comparativo", task, done)

    def generate_dataset(self) -> None:
        """Genera el dataset sintético (MIDI → audio → MP3 + ground truth) y ofrece abrir una pieza."""
        from src.synth_dataset import generate_dataset

        out_dir = filedialog.askdirectory(parent=self.root, title="Carpeta de salida del dataset", initialdir=str(DATA_DIR), mustexist=False)
        if not out_dir:
            return
        sr = self.state.config.audio.sample_rate

        def task(progress: ProgressCallback, cancel: threading.Event) -> Any:
            return generate_dataset(out_dir, sr=sr, progress=progress, cancel=cancel)

        def done(items: Any) -> None:
            names = "\n".join(f"• {it.mp3_path.name} ({it.method})" for it in items)
            if items and messagebox.askyesno("Dataset generado", f"Se generaron:\n{names}\n\n¿Abrir «{items[0].mp3_path.name}» ahora?", parent=self.root):
                self.analyze(items[0].mp3_path)

        self.run_task("Generando dataset sintético", task, done)

    def save_config(self) -> None:
        """Guarda la configuración vigente en JSON."""
        path = filedialog.asksaveasfilename(parent=self.root, title="Guardar configuración", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.state.config.save_json(path)
            logger.info("Configuración guardada en %s", path)

    def load_config(self) -> None:
        """Carga una configuración desde JSON y avisa a las pestañas."""
        path = filedialog.askopenfilename(parent=self.root, title="Cargar configuración", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.state.config = Config.load_json(path)
        except Exception as exc:  # noqa: BLE001
            show_error(self.root, "Configuración inválida", exc)
            return
        logger.info("Configuración cargada desde %s", path)
        self.state.events.emit("config_changed", key=None)

    def quit(self) -> None:
        """Cancela tareas en curso y cierra la ventana."""
        self.workers.cancel_all()
        logging.getLogger().removeHandler(self.log_handler)
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        """Entra en el bucle principal de tkinter."""
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la GUI.

    Parameters
    ----------
    argv : list[str] | None
        Argumentos opcionales: una ruta de audio para abrirla al iniciar.

    Returns
    -------
    int
        Código de salida (0 = correcto).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        app = SmartunerApp()
    except tk.TclError as exc:
        print(f"No se pudo abrir la interfaz gráfica (¿hay pantalla disponible?): {exc}", file=sys.stderr)
        print("Usa el modo sin interfaz: python main.py --cli --help", file=sys.stderr)
        return 1
    if argv:
        app.root.after(200, lambda: app.open_audio(argv[0]))
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
