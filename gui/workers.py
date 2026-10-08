"""Ejecución de tareas pesadas en hilos sin congelar la GUI.

Papel en la arquitectura
------------------------
tkinter no es seguro entre hilos: solo el hilo principal puede tocar widgets.
Por eso las tareas largas (análisis, Demucs, experimentos de ≥100 corridas,
generación del dataset) se ejecutan en un ``threading.Thread`` que NUNCA toca la
GUI; en su lugar deposita mensajes en una ``queue.Queue``. El hilo principal
revisa esa cola cada 100 ms con ``root.after`` y actualiza la barra de progreso,
el log y, al terminar, entrega el resultado a un callback.

::

    hilo trabajador ──put──▶ queue.Queue ──get (after 100 ms)─▶ hilo de la GUI
       fn(progress, cancel)     ("progress", 0.4, "Segmentando…")    barra de progreso
                                ("done", resultado)                  on_done(resultado)
                                ("error", excepción)                 diálogo de error

La cancelación es cooperativa: la GUI activa un ``threading.Event`` y las
funciones de ``src/`` lo consultan entre etapas (lanzan ``CancelledError``).

Hilo de la GUI y GIL
--------------------
Solo un hilo de Python ejecuta código a la vez (el GIL). El trabajador suelta
el GIL muchas veces por segundo (cada operación de numpy sobre un array
mediano lo hace) y, si el hilo de la GUI lo pide en ese momento, el
trabajador tiene que esperar a que se lo devuelvan: con el intervalo de
cambio por defecto de CPython (5 ms) cada despertar del bucle de tkinter
(sondeo, cursor de reproducción, registro) le costaba hasta 5 ms, y un
experimento tardaba 2–3 veces más en la GUI que en la CLI. Mientras hay
tareas en curso se reduce ese intervalo a :data:`WORKER_SWITCH_INTERVAL_S`
(y se restaura al terminar): el coste por despertar baja a décimas de
milisegundo y el experimento tarda casi lo mismo que en la CLI.

Excepción importante: el código compilado con numba en modo ``nopython`` sin
``nogil`` (p. ej. el Viterbi de ``librosa.pyin``) NO suelta el GIL mientras
corre, y ningún intervalo de cambio lo evita: durante esa llamada la ventana
no se repinta ni responde. Por eso :func:`src.pitch.estimate_pitch_track`
procesa las pistas largas por bloques de 15 s (< 1 s de GIL retenido cada
uno, y ``cancel`` se consulta entre bloques). Cualquier otra llamada larga de
ese tipo debe trocearse igual.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import traceback
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.config import CancelledError, ProgressCallback

logger = logging.getLogger(__name__)

#: Firma de una tarea: recibe el callback de progreso y el evento de cancelación.
TaskFn = Callable[[ProgressCallback, threading.Event], Any]

#: Intervalo (ms) entre lecturas de la cola de mensajes: 10 por segundo bastan para una barra
#: de progreso fluida y molestan poco al hilo trabajador.
POLL_MS = 100

#: Intervalo de cambio del GIL (s) mientras hay tareas en segundo plano (ver el encabezado).
WORKER_SWITCH_INTERVAL_S = 0.0005


@dataclass
class Worker:
    """Una tarea en segundo plano.

    Attributes
    ----------
    title : str
        Descripción corta mostrada en la barra de estado ("Analizando audio").
    fn : TaskFn
        Función a ejecutar en el hilo trabajador.
    on_done : Callable[[Any], None] | None
        Se llama EN EL HILO DE LA GUI con el valor devuelto por ``fn``.
    on_error : Callable[[BaseException, str], None] | None
        Se llama en el hilo de la GUI con la excepción y su traceback.
    on_progress : Callable[[float, str], None] | None
        Se llama en el hilo de la GUI con (fracción 0–1, mensaje).
    on_cancelled : Callable[[], None] | None
        Se llama en el hilo de la GUI si la tarea fue cancelada.
    """

    title: str
    fn: TaskFn
    on_done: Callable[[Any], None] | None = None
    on_error: Callable[[BaseException, str], None] | None = None
    on_progress: Callable[[float, str], None] | None = None
    on_cancelled: Callable[[], None] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    messages: "queue.Queue[tuple[Any, ...]]" = field(default_factory=queue.Queue)
    thread: threading.Thread | None = None
    finished: bool = False

    def _run(self) -> None:
        """Cuerpo del hilo: ejecuta ``fn`` y traduce su resultado a mensajes."""

        def progress(fraction: float, message: str = "") -> None:
            self.messages.put(("progress", float(fraction), str(message)))

        try:
            result = self.fn(progress, self.cancel_event)
        except CancelledError:
            self.messages.put(("cancelled",))
        except BaseException as exc:  # noqa: BLE001 - se reporta a la GUI
            self.messages.put(("error", exc, traceback.format_exc()))
        else:
            self.messages.put(("done", result))

    def start(self) -> None:
        """Lanza el hilo trabajador (``daemon`` para no bloquear el cierre de la app)."""
        self.thread = threading.Thread(target=self._run, name=f"worker:{self.title}", daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        """Pide la cancelación cooperativa de la tarea."""
        self.cancel_event.set()

    @property
    def alive(self) -> bool:
        """True mientras la tarea no ha entregado su resultado."""
        return not self.finished


class WorkerManager:
    """Lanza :class:`Worker` y despacha sus mensajes en el hilo de la GUI.

    Parameters
    ----------
    root : tk.Misc
        Widget raíz usado para programar el sondeo con ``after``.
    """

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.active: list[Worker] = []
        self._polling = False
        self._saved_switch_interval: float | None = None

    def start(
        self,
        title: str,
        fn: TaskFn,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException, str], None] | None = None,
        on_progress: Callable[[float, str], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
    ) -> Worker:
        """Crea, registra y lanza una tarea en segundo plano.

        Returns
        -------
        Worker
            El trabajador (se puede cancelar con ``worker.cancel()``).
        """
        worker = Worker(title, fn, on_done, on_error, on_progress, on_cancelled)
        self.active.append(worker)
        logger.debug("Iniciando tarea en segundo plano: %s", title)
        self._lower_switch_interval()
        worker.start()
        if not self._polling:
            self._polling = True
            self.root.after(POLL_MS, self._poll)
        return worker

    @property
    def busy(self) -> bool:
        """True si hay alguna tarea en curso."""
        return any(w.alive for w in self.active)

    def cancel_all(self) -> None:
        """Pide cancelar todas las tareas en curso."""
        for worker in self.active:
            worker.cancel()

    def _poll(self) -> None:
        """Vacía las colas de mensajes (hilo de la GUI) y se reprograma si hace falta."""
        for worker in list(self.active):
            while True:
                try:
                    msg = worker.messages.get_nowait()
                except queue.Empty:
                    break
                self._dispatch(worker, msg)
        self.active = [w for w in self.active if w.alive]
        if self.active:
            try:
                self.root.after(POLL_MS, self._poll)
            except tk.TclError:  # la ventana se cerró con la tarea en curso
                self._polling = False
                self._restore_switch_interval()
        else:
            self._polling = False
            self._restore_switch_interval()

    def _lower_switch_interval(self) -> None:
        """Reduce el intervalo de cambio del GIL mientras haya tareas (ver el encabezado del módulo)."""
        if self._saved_switch_interval is None:
            self._saved_switch_interval = sys.getswitchinterval()
            sys.setswitchinterval(min(self._saved_switch_interval, WORKER_SWITCH_INTERVAL_S))

    def _restore_switch_interval(self) -> None:
        """Restaura el intervalo de cambio del GIL que había antes de la primera tarea."""
        if self._saved_switch_interval is not None:
            sys.setswitchinterval(self._saved_switch_interval)
            self._saved_switch_interval = None

    def shutdown(self) -> None:
        """Cancela las tareas en curso y restaura el intervalo del GIL (al cerrar la ventana)."""
        self.cancel_all()
        self._restore_switch_interval()

    def _dispatch(self, worker: Worker, msg: tuple[Any, ...]) -> None:
        """Entrega un mensaje al callback correspondiente."""
        kind = msg[0]
        try:
            if kind == "progress" and worker.on_progress:
                worker.on_progress(msg[1], msg[2])
            elif kind == "done":
                worker.finished = True
                if worker.on_done:
                    worker.on_done(msg[1])
            elif kind == "error":
                worker.finished = True
                logger.error("La tarea '%s' falló: %s", worker.title, msg[1])
                if worker.on_error:
                    worker.on_error(msg[1], msg[2])
            elif kind == "cancelled":
                worker.finished = True
                logger.warning("Tarea cancelada: %s", worker.title)
                if worker.on_cancelled:
                    worker.on_cancelled()
        except Exception:  # noqa: BLE001 - un callback roto no debe matar el sondeo
            logger.exception("Error en el callback de la tarea '%s'", worker.title)


class QueueLogHandler(logging.Handler):
    """Handler de ``logging`` que deposita los registros en una cola.

    Los módulos de ``src/`` escriben con ``logging`` (sirve igual en la CLI);
    en la GUI este handler los envía a la pestaña Log, que vacía la cola desde
    el hilo principal. Es seguro llamarlo desde cualquier hilo.
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.queue: "queue.Queue[logging.LogRecord]" = queue.Queue()

    def emit(self, record: logging.LogRecord) -> None:
        """Encola el registro (sin formatear; la pestaña Log decide el formato)."""
        self.queue.put(record)
