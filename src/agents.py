"""Etapa 7 — Agentes bandit implementados desde cero con numpy.

Papel en el pipeline
--------------------
Cada segmento de nota es un problema **Multi-Armed Bandit** independiente
(:mod:`src.environment`): los brazos son las posiciones (cuerda, traste)
candidatas y cada pull devuelve una recompensa ruidosa r ∈ ≈[0, 1]. Los
agentes de este módulo deciden, pull a pull, qué brazo probar y, al agotar el
presupuesto T, qué posición recomendar. :mod:`src.experiments` los ejecuta
en lote y la GUI los ejecuta paso a paso (pestaña "Ejecución en vivo").

Los agentes **no conocen** las etiquetas de los brazos ("A-0", "E-5"...):
trabajan con índices 0..K−1; la traducción a posiciones la hace el entorno.

Interfaz común (:class:`BanditAgent`)
-------------------------------------
* ``select_arm()`` → índice del brazo a jalar (y deja en ``last_decision`` una
  :class:`Decision` con la explicación en español que muestra la GUI).
* ``update(arm, reward)`` → aprende de la recompensa observada.
* ``reset()`` → olvida todo (Q = Q₀, conteos a 0, t = 0).
* ``decision_scores()`` → el criterio que usa para decidir (Q, índice UCB o
  probabilidades softmax), para la gráfica en vivo.
* ``recommend(mode)`` → posición final del segmento.

Regla de actualización incremental (todos los agentes)
------------------------------------------------------
Sutton & Barto (2018, §2.4) muestran que la media muestral de las n
recompensas de un brazo se puede actualizar sin guardar el historial::

    Q_{n+1} = (1/n)·Σ_{i=1..n} R_i
            = (1/n)·(R_n + (n−1)·Q_n)
            = Q_n + (1/n)·(R_n − Q_n)

Es la forma general "NuevaEstimación ← Vieja + TamañoPaso·[Objetivo − Vieja]"::

    Q_{n+1} = Q_n + α_n · (r_n − Q_n)

* α_n = 1/n  → media muestral exacta. El primer pull hace Q = R_1 (paso 1),
  así que el valor inicial Q₀ se olvida por completo tras un pull. Cumple las
  condiciones de Robbins–Monro (Σα = ∞, Σα² < ∞) y converge a la media real μ.
* α constante → promedio exponencial ("recency-weighted average")::

      Q_{n+1} = (1−α)ⁿ·Q₁ + Σ_{i=1..n} α·(1−α)^{n−i}·R_i

  Las recompensas recientes pesan más y el sesgo de Q₁ decae como (1−α)ⁿ
  (nunca llega a cero). No converge (Σα² = ∞): sigue "rastreando", útil en
  problemas no estacionarios y necesario para el agente optimista.

El dilema exploración/explotación
---------------------------------
**Explotar** (elegir el mejor brazo según Q) maximiza la recompensa inmediata;
**explorar** (probar otros) mejora las estimaciones y puede descubrir un brazo
mejor. Un agente puramente greedy puede quedarse para siempre en un brazo
subóptimo cuya Q se sobreestimó por azar. Cada agente resuelve el dilema de
una forma distinta:

==================  ============================================================
Agente              Cómo explora
==================  ============================================================
ε-greedy            Al azar (uniforme) con probabilidad ε; "ciega" al valor.
ε-greedy optimista  Q₀ alto: cada brazo "decepciona" al probarlo → recorre todos.
UCB1                Optimismo ante la incertidumbre: bono c·√(ln t / n_a).
Softmax             Muestrea con π(a) ∝ e^(Q_a/τ): explora más los brazos buenos.
==================  ============================================================

Los empates en argmax se rompen al azar con el ``rng`` del agente (nunca con
el índice más bajo, que sesgaría la exploración hacia los primeros brazos).

Referencias
-----------
* R. S. Sutton y A. G. Barto, *Reinforcement Learning: An Introduction*, 2.ª
  ed., MIT Press, 2018, cap. 2 (bandits de K brazos).
* P. Auer, N. Cesa-Bianchi y P. Fischer, "Finite-time Analysis of the
  Multiarmed Bandit Problem", *Machine Learning* 47, 235–256, 2002 (UCB1).

Examples
--------
Bucle bandit mínimo (el que ejecuta :func:`src.experiments.run_segment`):

>>> import numpy as np
>>> from src.config import AgentConfig
>>> rng_env = np.random.default_rng(0)
>>> mu = np.array([0.2, 0.8, 0.5])                 # medias reales (ocultas)
>>> agent = make_agent("ucb1", n_arms=3, cfg=AgentConfig(), rng=np.random.default_rng(1))
>>> for _ in range(500):
...     a = agent.select_arm()
...     agent.update(a, rng_env.normal(mu[a], 0.1))
>>> agent.recommend("most_pulled")
1
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config import AgentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explicación de cada decisión (lo que la GUI muestra paso a paso)
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """Explicación de una elección del agente.

    Attributes
    ----------
    arm : int
        Brazo elegido.
    kind : str
        ``"init"`` (pull inicial obligatorio, UCB1), ``"explore"``,
        ``"exploit"`` o ``"sample"`` (muestreo de la distribución softmax).
    scores : np.ndarray | None
        Criterio usado para decidir (forma ``(K,)``): Q para ε-greedy, índice
        UCB para UCB1, probabilidades π para Softmax. Es una COPIA tomada en
        el momento de decidir (las actualizaciones posteriores no la cambian).
    score_name : str
        Nombre legible del criterio (p. ej. ``"Q + c·√(ln t / n)"``).
    text : str
        Frase en español para el log/GUI, p. ej.
        ``"u=0.03 < ε=0.10 → explora: brazo 2 al azar"``.
    """

    arm: int
    kind: str
    scores: np.ndarray | None
    score_name: str
    text: str


def _fmt(x: float) -> str:
    """Formatea un número para los textos de :class:`Decision`.

    Usa 2 decimales (como en clase) salvo para magnitudes muy pequeñas o
    infinitas, que se muestran en notación científica o como ``+∞``.

    Examples
    --------
    >>> _fmt(0.7231), _fmt(0.004), _fmt(float("inf")), _fmt(0.0)
    ('0.72', '4.0e-03', '+∞', '0.00')
    """
    if math.isinf(x):
        return "+∞" if x > 0 else "−∞"
    if x == 0.0 or abs(x) >= 0.01:
        return f"{x:.2f}"
    return f"{x:.1e}"


def _fmt_pair(u: float, threshold: float) -> tuple[str, str]:
    """Formatea u y un umbral (ε) de modo que su comparación se vea correcta.

    Si con 2 decimales ambos números se verían iguales (p. ej. u=0.099 y
    ε=0.10 → "0.10 < 0.10"), se usan 4 decimales.
    """
    a, b = _fmt(u), _fmt(threshold)
    if a == b:
        a, b = f"{u:.4f}", f"{threshold:.4f}"
    return a, b


# ---------------------------------------------------------------------------
# Clase base
# ---------------------------------------------------------------------------


class BanditAgent(ABC):
    """Clase base de los agentes: estado (Q, N, t) y regla incremental.

    Las subclases solo definen CÓMO ELEGIR el brazo (:meth:`select_arm`); el
    aprendizaje (:meth:`update`) es común a todas y sigue la regla
    Q ← Q + α·(r − Q) explicada en el encabezado del módulo.

    Parameters
    ----------
    n_arms : int
        Número de brazos K (≥ 1).
    rng : np.random.Generator | None
        Generador propio del agente (exploración y desempates). ``None`` crea
        uno nuevo con ``np.random.default_rng()`` (no reproducible): para
        experimentos pasa siempre un generador sembrado.
    q0 : float
        Valor inicial de las estimaciones Q (misma unidad que la recompensa).
    alpha : float | None
        Tamaño de paso constante en (0, 1]; ``None`` = 1/n (media muestral).

    Attributes
    ----------
    q : np.ndarray
        Estimaciones Q_a, forma ``(K,)``, ``float64``.
    counts : np.ndarray
        Número de pulls de cada brazo n_a (int), forma ``(K,)``.
    t : int
        Pulls totales realizados.
    last_decision : Decision | None
        Explicación de la última llamada a :meth:`select_arm` (None si
        ``explain`` es False).
    explain : bool
        Si se construye la explicación de cada decisión (ver atributo de clase).

    Raises
    ------
    ValueError
        Si ``n_arms < 1`` o ``alpha`` está fuera de (0, 1].
    """

    #: Identificador interno (clave de :data:`src.config.ALGORITHMS`).
    name: str = "base"

    #: Nombre del criterio que devuelve :meth:`decision_scores` por defecto.
    SCORE_NAME: str = "Q estimado"

    #: Si es True (por defecto), :meth:`select_arm` deja en ``last_decision``
    #: la explicación en español de cada elección (la usa la ejecución en vivo
    #: de la GUI). Los experimentos lo ponen en False: la ELECCIÓN y el consumo
    #: del generador son idénticos, pero no se construye el texto, que es lo
    #: más caro de un pull (≈ 2–3× más rápido) y ``last_decision`` queda en None.
    explain: bool = True

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None, q0: float = 0.0, alpha: float | None = None) -> None:
        if int(n_arms) < 1:
            raise ValueError(f"El número de brazos debe ser ≥ 1 (se recibió {n_arms}).")
        if alpha is not None and not 0.0 < alpha <= 1.0:
            raise ValueError(f"α debe estar en (0, 1] o ser None para usar 1/n (se recibió {alpha}).")
        self.n_arms: int = int(n_arms)
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self.q0: float = float(q0)
        self.alpha: float | None = None if alpha is None else float(alpha)
        # Estado de aprendizaje; reset() lo crea (y lo vuelve a crear al reiniciar).
        self.q: np.ndarray
        self.counts: np.ndarray
        self.t: int
        self.last_decision: Decision | None
        self.reset()

    # ------------------------------------------------------------------ API

    @abstractmethod
    def select_arm(self) -> int:
        """Elige el siguiente brazo a jalar (y actualiza ``last_decision``).

        Returns
        -------
        int
            Índice del brazo, en ``0..K−1``.
        """

    def update(self, arm: int, reward: float) -> None:
        """Aprende de una recompensa: actualiza n_a, Q_a y t.

        Implementa literalmente (Sutton & Barto, 2018, §2.4)::

            N(A) ← N(A) + 1
            Q(A) ← Q(A) + α · [R − Q(A)]      con α = 1/N(A) si alpha es None

        El término ``R − Q(A)`` es el **error de predicción**: si la
        recompensa supera lo esperado, Q sube; si no, baja. α decide cuánto
        se corrige: con 1/N(A) el resultado es exactamente la media de todas
        las recompensas de ese brazo; con α constante, un promedio que pesa
        más lo reciente.

        Parameters
        ----------
        arm : int
            Brazo jalado, en ``0..K−1``.
        reward : float
            Recompensa observada (finita).

        Raises
        ------
        IndexError
            Si ``arm`` no es un brazo válido (los índices negativos de numpy
            actualizarían en silencio el brazo equivocado).
        ValueError
            Si ``reward`` es NaN o infinita (envenenaría Q para siempre).

        Examples
        --------
        >>> agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, rng=np.random.default_rng(0))
        >>> for r in (1.0, 0.0, 0.5):
        ...     agent.update(1, r)
        >>> float(agent.q[1]), int(agent.counts[1]), agent.t     # media de 1, 0, 0.5
        (0.5, 3, 3)
        """
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"Brazo {arm} fuera de rango: el agente tiene {self.n_arms} brazos (0..{self.n_arms - 1}).")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"Recompensa no finita ({reward}) para el brazo {arm}.")
        # N(A) ← N(A) + 1
        n = int(self.counts[arm]) + 1
        self.counts[arm] = n
        # α_n: constante, o 1/n (media muestral incremental).
        step = self.alpha if self.alpha is not None else 1.0 / n
        # Q(A) ← Q(A) + α·(R − Q(A))   (aritmética con floats de Python: más rápida
        # que operar con escalares de numpy y numéricamente idéntica).
        q_old = float(self.q[arm])
        self.q[arm] = q_old + step * (reward - q_old)
        self.t += 1

    def reset(self) -> None:
        """Vuelve al estado inicial.

        Q = q0, conteos a 0, t = 0 y ``last_decision = None``. Los
        hiperparámetros con decaimiento (ε, τ) se calculan a partir de t, así
        que también vuelven a su valor inicial.
        """
        self.q = np.full(self.n_arms, self.q0, dtype=np.float64)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.t = 0
        self.last_decision = None

    def recommend(self, mode: str = "most_pulled") -> int:
        """Brazo recomendado al final del presupuesto.

        * ``"most_pulled"``: argmax de ``counts`` (desempate por mayor Q). Es
          la opción robusta: un agente que funciona bien concentra sus pulls
          en el mejor brazo, mientras que un Q alto con pocos pulls puede ser
          suerte.
        * ``"greedy"``: argmax de ``q`` (desempate por más pulls).

        Si persiste el empate se elige el índice más bajo: este método es
        determinista y **no consume** el ``rng`` del agente, para que
        consultarlo (p. ej. desde la GUI en cada paso) no altere la secuencia
        aleatoria ni, por tanto, los resultados.

        Parameters
        ----------
        mode : str
            ``"most_pulled"`` o ``"greedy"``.

        Returns
        -------
        int
            Índice del brazo recomendado.

        Raises
        ------
        ValueError
            Si ``mode`` no es una de las dos opciones.

        Examples
        --------
        >>> agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, rng=np.random.default_rng(0))
        >>> for arm, r in [(0, 0.5), (0, 0.5), (0, 0.5), (2, 0.9)]:
        ...     agent.update(arm, r)
        >>> agent.recommend("most_pulled"), agent.recommend("greedy")
        (0, 2)
        """
        if mode == "most_pulled":
            primary, secondary = self.counts, self.q
        elif mode == "greedy":
            primary, secondary = self.q, self.counts
        else:
            raise ValueError(f"Modo de recomendación desconocido: '{mode}'. Usa 'most_pulled' o 'greedy'.")
        best = np.flatnonzero(primary == primary.max())
        if best.size > 1:  # desempate por el segundo criterio
            sub = secondary[best]
            best = best[sub == sub.max()]
        return int(best[0])

    def decision_scores(self) -> tuple[np.ndarray, str]:
        """Criterio de decisión ACTUAL y su nombre, para la gráfica en vivo.

        En la clase base (ε-greedy y optimista) el criterio es Q.

        Returns
        -------
        scores : np.ndarray
            Copia del criterio, forma ``(K,)``.
        name : str
            Nombre legible (``"Q estimado"``).
        """
        return self.q.copy(), self.SCORE_NAME

    # -------------------------------------------------------------- auxiliares

    def _argmax_random_tie(self, values: np.ndarray) -> int:
        """argmax rompiendo empates uniformemente al azar.

        ``np.argmax`` devuelve siempre el PRIMER máximo; si varios brazos
        empatan (p. ej. al inicio, todos con Q = Q₀) eso sesgaría la elección
        hacia los índices bajos. Aquí se elige uno al azar entre los empatados.

        Parameters
        ----------
        values : np.ndarray
            Valores, forma ``(K,)``. ``+inf`` se trata como un valor más (UCB1
            usa +∞ para los brazos no jalados); los NaN se ignoran.

        Returns
        -------
        int
            Índice elegido.
        """
        return self._argmax_with_ties(values)[0]

    def _argmax_with_ties(self, values: np.ndarray) -> tuple[int, int]:
        """Como :meth:`_argmax_random_tie`, pero devuelve también cuántos empataron.

        Returns
        -------
        arm : int
            Índice elegido.
        n_tied : int
            Número de brazos empatados en el máximo (1 = sin empate).
        """
        first = int(values.argmax())  # primer máximo (argmax es más rápido que max + ==)
        best = values[first]
        if best != best:  # NaN ≠ NaN: argmax se detuvo en un NaN → se ignoran los NaN
            valid = ~np.isnan(values)
            if not valid.any():
                return int(self.rng.integers(values.size)), int(values.size)
            best = values[valid].max()
            first = int(np.flatnonzero(values == best)[0])
        n_tied = int(np.count_nonzero(values == best))
        if n_tied == 1:
            # Caso habitual: un único máximo → no hace falta consumir el rng.
            return first, 1
        candidates = np.flatnonzero(values == best)
        return int(self.rng.choice(candidates)), n_tied

    def _params_text(self) -> str:
        """Hiperparámetros en texto, para ``repr`` y los logs."""
        alpha = "1/n" if self.alpha is None else f"{self.alpha:g}"
        return f"Q₀={self.q0:g}, α={alpha}"

    def __repr__(self) -> str:
        """Representación compacta para logs y depuración.

        Returns
        -------
        str
            Clase, número de brazos K, hiperparámetros y pulls realizados,
            p. ej. ``"UCB1Agent(K=5, Q₀=0, α=1/n, c=1.414, t=37)"``.
        """
        return f"{type(self).__name__}(K={self.n_arms}, {self._params_text()}, t={self.t})"


# ---------------------------------------------------------------------------
# ε-greedy (y su variante optimista)
# ---------------------------------------------------------------------------


class EpsilonGreedyAgent(BanditAgent):
    """ε-greedy: con probabilidad ε explora un brazo uniforme, si no explota argmax Q.

    Pseudocódigo (Sutton & Barto, 2018, §2.4, "A simple bandit algorithm")::

        Inicializar, para a = 0..K−1:  Q(a) ← Q₀,  N(a) ← 0
        Repetir en cada pull:
            u ~ Uniforme[0, 1)
            si u < ε:  A ← brazo uniforme al azar            (explorar)
            si no:     A ← argmax_a Q(a), empates al azar    (explotar)
            R ← bandit(A)
            N(A) ← N(A) + 1
            Q(A) ← Q(A) + α·[R − Q(A)]

    **Teoría.** Con ε > 0 cada brazo se sigue probando indefinidamente, de
    modo que (con α = 1/n) todas las Q convergen a sus medias reales μ_a. El
    precio es que, incluso tras aprender, se explora una fracción ε de las
    veces: la tasa de acierto asintótica es 1 − ε + ε/K. La exploración es
    "ciega": un brazo pésimo se prueba tanto como uno casi óptimo.

    **Decaimiento.** ε puede decaer: ε_t = max(ε_min, ε₀·dᵗ), con t = pulls
    realizados. Así se explora mucho al principio (estimaciones pobres) y poco
    al final (estimaciones buenas). Con d = 1 se usa ε₀ exacto (sin aplicar
    ε_min). La teoría (GLIE: "greedy in the limit with infinite exploration")
    pide que ε → 0 lo bastante despacio para que cada brazo se siga probando;
    la cota ε_min es la versión práctica de esa idea.

    Parameters
    ----------
    n_arms : int
        Número de brazos K.
    epsilon : float
        ε₀, probabilidad de explorar en [0, 1].
    decay : float
        Factor d por pull en (0, 1]; 1 = ε constante.
    epsilon_min : float
        Cota inferior de ε cuando hay decaimiento, en [0, 1].
    rng : np.random.Generator | None
        Generador del agente.
    q0 : float
        Valor inicial de Q.
    alpha : float | None
        Tamaño de paso constante o ``None`` (1/n).

    Attributes
    ----------
    epsilon0 : float
        ε inicial ε₀.
    decay : float
        Factor de decaimiento d.
    epsilon_min : float
        Cota inferior ε_min.

    Raises
    ------
    ValueError
        Si ε₀ o ε_min están fuera de [0, 1] o d fuera de (0, 1].

    Examples
    --------
    >>> agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, rng=np.random.default_rng(0))
    >>> agent.update(2, 1.0)
    >>> agent.select_arm()
    2
    >>> agent.last_decision.text
    'ε=0 → explota: brazo 2 tiene el mayor Q (1.00)'
    """

    name = "egreedy"

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        decay: float = 1.0,
        epsilon_min: float = 0.0,
        rng: np.random.Generator | None = None,
        q0: float = 0.0,
        alpha: float | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"ε debe estar en [0, 1] (se recibió {epsilon}).")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"El decaimiento de ε debe estar en (0, 1] (se recibió {decay}).")
        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError(f"ε_min debe estar en [0, 1] (se recibió {epsilon_min}).")
        self.epsilon0: float = float(epsilon)
        self.decay: float = float(decay)
        self.epsilon_min: float = float(epsilon_min)
        super().__init__(n_arms, rng=rng, q0=q0, alpha=alpha)

    @property
    def epsilon(self) -> float:
        """ε vigente en el pull actual: ε_t = max(ε_min, ε₀·dᵗ) (ε₀ si d = 1).

        Examples
        --------
        >>> agent = EpsilonGreedyAgent(n_arms=2, epsilon=0.5, decay=0.5, epsilon_min=0.1)
        >>> agent.epsilon
        0.5
        >>> agent.t = 2          # tras 2 pulls: max(0.1, 0.5·0.5²) = 0.125
        >>> agent.epsilon
        0.125
        >>> agent.t = 10         # 0.5·0.5¹⁰ ≈ 0.0005 < ε_min
        >>> agent.epsilon
        0.1
        """
        if self.decay == 1.0:
            return self.epsilon0
        return max(self.epsilon_min, self.epsilon0 * self.decay ** self.t)

    def select_arm(self) -> int:
        """Elige con la regla ε-greedy y (si ``explain``) explica la decisión.

        Returns
        -------
        int
            Brazo elegido. ``last_decision.kind`` es ``"explore"`` o
            ``"exploit"`` y ``last_decision.scores`` una copia de Q.
        """
        eps = self.epsilon
        u: float | None = None
        if eps > 0.0:
            # Moneda de exploración: u ~ U[0, 1); se explora si u < ε.
            u = self.rng.random()
        if u is not None and u < eps:
            # EXPLORAR: brazo uniforme entre TODOS (puede salir el greedy).
            arm, n_tied = int(self.rng.integers(self.n_arms)), 1
        else:
            # EXPLOTAR: argmax Q con desempate al azar (ε = 0: greedy puro,
            # ni siquiera se tira la moneda).
            arm, n_tied = self._argmax_with_ties(self.q)
        self.last_decision = self._explain_choice(arm, u, eps, n_tied) if self.explain else None
        return arm

    def _explain_choice(self, arm: int, u: float | None, eps: float, n_tied: int) -> Decision:
        """Construye la :class:`Decision` (texto en español) de una elección ε-greedy.

        Parameters
        ----------
        arm : int
            Brazo elegido.
        u : float | None
            Valor de la moneda de exploración (None si ε = 0 y no se tiró).
        eps : float
            ε vigente.
        n_tied : int
            Brazos empatados en el máximo de Q (solo al explotar).

        Returns
        -------
        Decision
            Explicación con ``kind`` ``"explore"`` o ``"exploit"`` y una copia de Q.
        """
        scores = self.q.copy()  # Q en el momento de decidir (para la GUI)
        if u is not None and u < eps:
            u_txt, eps_txt = _fmt_pair(u, eps)
            text = f"u={u_txt} < ε={eps_txt} → explora: brazo {arm} al azar"
            return Decision(arm, "explore", scores, self.SCORE_NAME, text)
        if u is not None:
            u_txt, eps_txt = _fmt_pair(u, eps)
            prefix = f"u={u_txt} ≥ ε={eps_txt} → "
        else:
            prefix = "ε=0 → "  # greedy puro: no hace falta tirar la moneda
        q_txt = _fmt(float(self.q[arm]))
        if n_tied > 1:
            text = f"{prefix}explota: empate en Q={q_txt} entre {n_tied} brazos → brazo {arm} al azar"
        else:
            text = f"{prefix}explota: brazo {arm} tiene el mayor Q ({q_txt})"
        if self.counts[arm] == 0:
            # Típico del agente optimista: el brazo "gana" solo porque aún vale Q₀.
            text += " — aún sin probar (Q = Q₀)"
        return Decision(arm, "exploit", scores, self.SCORE_NAME, text)

    def _params_text(self) -> str:
        decay = "" if self.decay == 1.0 else f", d={self.decay:g}, ε_min={self.epsilon_min:g}"
        return f"ε₀={self.epsilon0:g}{decay}, {super()._params_text()}"


class OptimisticGreedyAgent(EpsilonGreedyAgent):
    """ε-greedy optimista: Q₀ alto (> recompensa máxima) y α constante.

    Con Q₀ = 2 y recompensas ≤ 1, cada brazo probado baja su Q por debajo de
    los no probados, así que un agente greedy (ε=0) recorre todos los brazos
    antes de estabilizarse: exploración dirigida "gratis" al inicio.

    **Teoría** (Sutton & Barto, 2018, §2.6). Un Q₀ por encima de cualquier
    recompensa posible es "optimista": el agente cree que todo brazo no
    probado es excelente. Al probarlo, la recompensa real r < Q₀ lo
    "decepciona" y su Q baja, de modo que el argmax pasa a otro brazo aún
    optimista. Resultado: incluso con ε = 0 todos los brazos se prueban al
    principio, sin ninguna moneda al azar.

    **¿Por qué α constante?** Con α = 1/n el primer pull usa paso 1/1 = 1, es
    decir Q ← r: el optimismo desaparece de golpe tras UN pull por brazo y el
    agente queda greedy puro con estimaciones de una sola muestra (frágil).
    Con α constante (p. ej. 0.1) Q₀ se olvida gradualmente (sesgo ∝ (1−α)ⁿ;
    0.9¹⁰ ≈ 0.35): cada brazo se prueba varias veces antes de descartarlo, lo
    que da estimaciones más fiables (decisión de diseño 3 del proyecto).

    **Limitación.** La exploración es solo temporal (ocurre al inicio): si el
    problema cambiara con el tiempo, el optimismo inicial no ayudaría.

    Parameters
    ----------
    n_arms : int
        Número de brazos K.
    q0 : float
        Valor inicial optimista Q₀ (mayor que la recompensa máxima, ≈1).
    epsilon : float
        Exploración aleatoria adicional (normalmente 0), constante.
    alpha : float
        Tamaño de paso constante α en (0, 1].
    rng : np.random.Generator | None
        Generador del agente (desempates y ε).

    Examples
    --------
    Con ε = 0 y recompensas < Q₀, los K primeros pulls recorren los K brazos:

    >>> agent = OptimisticGreedyAgent(n_arms=4, q0=2.0, alpha=0.1, rng=np.random.default_rng(0))
    >>> seen = set()
    >>> for _ in range(4):
    ...     a = agent.select_arm()
    ...     seen.add(a)
    ...     agent.update(a, 1.0)          # Q: 2.0 → 1.9 (decepción)
    >>> sorted(seen)
    [0, 1, 2, 3]
    """

    name = "optimistic"

    def __init__(
        self,
        n_arms: int,
        q0: float = 2.0,
        epsilon: float = 0.0,
        alpha: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms, epsilon=epsilon, decay=1.0, epsilon_min=0.0, rng=rng, q0=q0, alpha=alpha)


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------


class UCB1Agent(BanditAgent):
    """UCB1: primero jala cada brazo una vez; luego argmax Q_a + c·√(ln t / n_a).

    Pseudocódigo (Auer et al., 2002; Sutton & Barto, 2018, §2.7)::

        Inicializar, para a = 0..K−1:  Q(a) ← 0,  N(a) ← 0
        Repetir en cada pull (t = pulls ya realizados):
            si algún N(a) = 0:  A ← uno de esos brazos (al azar)          (init)
            si no:              A ← argmax_a [ Q(a) + c·√(ln t / N(a)) ]
            R ← bandit(A)
            N(A) ← N(A) + 1
            Q(A) ← Q(A) + [R − Q(A)] / N(A)

    **Teoría: optimismo ante la incertidumbre.** En vez de explorar al azar,
    UCB1 actúa como si cada brazo fuera tan bueno como *plausiblemente*
    podría ser. El índice ``Q(a) + c·√(ln t / N(a))`` es una **cota superior
    de confianza** de la media real μ_a: por la desigualdad de Hoeffding (para
    recompensas en [0, 1]), P(μ_a > Q(a) + √(2·ln t / N(a))) ≤ t⁻⁴, así que con
    c = √2 (UCB1 original) la cota casi nunca subestima μ_a.

    * El **bono** c·√(ln t / N(a)) mide la incertidumbre: decrece como 1/√N(a)
      al probar el brazo y crece (muy despacio, como √ln t) mientras no se
      prueba. Por eso todo brazo acaba revisándose, pero cada vez con menos
      frecuencia.
    * Un brazo se elige si su Q es alto (**explotar**) o si su bono es alto
      (**explorar**): ambos motivos quedan unidos en un único índice.
    * Los brazos no jalados tienen N = 0 → bono infinito: por eso UCB1 empieza
      jalando cada brazo una vez.
    * Garantía: el regret crece solo logarítmicamente, O(Σ_a ln T / Δ_a), el
      orden óptimo (Lai & Robbins). Es una cota ASINTÓTICA: con brechas Δ_a
      pequeñas (aquí, posiciones del mismo pitch casi empatadas) y T de unos
      cientos de pulls, el término ln T / Δ_a es grande y UCB1 explora casi
      todo el presupuesto, así que su regret puede verse casi lineal y su
      recompensa media quedar por debajo de ε-greedy (con c menor que √2 se
      aplana antes; ver el barrido de c en la pestaña Comparación).

    Aquí t es el número de pulls YA realizados (el ``n`` de Auer et al.). c
    regula el peso de la exploración: c = 0 es greedy puro.

    Parameters
    ----------
    n_arms : int
        Número de brazos K.
    c : float
        Constante de exploración c ≥ 0 (√2 ≈ 1.414 en UCB1 original).
    rng : np.random.Generator | None
        Generador del agente (orden de los pulls iniciales y desempates).

    Raises
    ------
    ValueError
        Si ``c < 0``.

    Examples
    --------
    >>> agent = UCB1Agent(n_arms=2, c=1.0, rng=np.random.default_rng(0))
    >>> agent.update(0, 1.0)
    >>> agent.update(1, 0.0)
    >>> agent.ucb_values().round(3)        # t=2: bono = √(ln 2 / 1) = 0.833
    array([1.833, 0.833])
    """

    name = "ucb1"

    SCORE_NAME = "Q + c·√(ln t / n)"

    def __init__(self, n_arms: int, c: float = 1.414, rng: np.random.Generator | None = None) -> None:
        if c < 0:
            raise ValueError(f"La constante c de UCB1 debe ser ≥ 0 (se recibió {c}).")
        self.c: float = float(c)
        super().__init__(n_arms, rng=rng, q0=0.0, alpha=None)

    def _bonus(self) -> np.ndarray:
        """Bono de exploración c·√(ln t / n_a) por brazo (``inf`` si n_a = 0)."""
        bonus = np.full(self.n_arms, np.inf)
        pulled = self.counts > 0
        if self.t > 0:
            bonus[pulled] = self.c * np.sqrt(math.log(self.t) / self.counts[pulled])
        return bonus

    def ucb_values(self) -> np.ndarray:
        """Índice UCB de cada brazo (``inf`` para brazos no jalados).

        Returns
        -------
        np.ndarray
            Q_a + c·√(ln t / n_a), forma ``(K,)``; ``+inf`` donde n_a = 0.
        """
        return self.q + self._bonus()

    def decision_scores(self) -> tuple[np.ndarray, str]:
        """Índice UCB actual de cada brazo y su nombre (para la GUI en vivo).

        Returns
        -------
        tuple[np.ndarray, str]
            ``(Q_a + c·√(ln t / n_a), "Q + c·√(ln t / n)")``; la primera
            componente tiene forma ``(K,)`` y vale ``+inf`` donde n_a = 0.
        """
        return self.ucb_values(), self.SCORE_NAME

    def select_arm(self) -> int:
        """Elige con UCB1 y (si ``explain``) explica la decisión.

        Returns
        -------
        int
            Brazo elegido. ``last_decision.kind`` es ``"init"`` durante los
            pulls iniciales obligatorios; después ``"exploit"`` si el brazo
            elegido es también el de mayor Q, o ``"explore"`` si gana gracias
            al bono de incertidumbre.
        """
        if self.counts.min() == 0:
            # Fase inicial: N(a) = 0 → índice +∞; se elige uno de ellos al azar.
            unpulled = np.flatnonzero(self.counts == 0)
            arm = int(self.rng.choice(unpulled)) if unpulled.size > 1 else int(unpulled[0])
            self.last_decision = None
            if self.explain:
                text = f"UCB1: pull inicial obligatorio del brazo {arm} (n=0 → índice +∞)"
                self.last_decision = Decision(arm, "init", self.ucb_values(), self.SCORE_NAME, text)
            return arm
        # Fase normal: argmax de la cota superior de confianza.
        bonus = self.c * np.sqrt(math.log(self.t) / self.counts)
        index = self.q + bonus
        arm = self._argmax_random_tie(index)
        self.last_decision = self._explain_choice(arm, index, bonus) if self.explain else None
        return arm

    def _explain_choice(self, arm: int, index: np.ndarray, bonus: np.ndarray) -> Decision:
        """Construye la :class:`Decision` de una elección UCB1 (fase normal).

        Parameters
        ----------
        arm : int
            Brazo elegido.
        index : np.ndarray
            Índice UCB Q + bono de todos los brazos, forma ``(K,)``.
        bonus : np.ndarray
            Bono c·√(ln t / n_a) de todos los brazos, forma ``(K,)``.

        Returns
        -------
        Decision
            ``kind`` ``"exploit"`` si el brazo es también el de mayor Q,
            ``"explore"`` si gana gracias al bono.
        """
        q_arm = float(self.q[arm])
        if q_arm >= self.q.max():
            kind, why = "exploit", "es el de mayor Q → explota"
        else:
            kind, why = "explore", "gana por el bono de incertidumbre → explora"
        text = (
            f"UCB1: brazo {arm} con índice {_fmt(float(index[arm]))} = "
            f"Q {_fmt(q_arm)} + bono {_fmt(float(bonus[arm]))} ({why})"
        )
        return Decision(arm, kind, index, self.SCORE_NAME, text)

    def _params_text(self) -> str:
        return f"c={self.c:g}"


# ---------------------------------------------------------------------------
# Softmax (Boltzmann)
# ---------------------------------------------------------------------------


class SoftmaxAgent(BanditAgent):
    """Softmax/Boltzmann: muestrea a con π(a) = exp(Q_a/τ) / Σ_b exp(Q_b/τ).

    Se resta max(Q) antes de exponenciar (estabilidad numérica). τ puede
    decaer (annealing): τ_t = max(τ_min, τ₀·dᵗ).

    Pseudocódigo (Sutton & Barto, 1998, §2.3 «Softmax Action Selection»; en
    la 2.ª ed., 2018, §2.8 «Gradient Bandit Algorithms» usa la misma
    distribución softmax pero sobre PREFERENCIAS H_t(a) aprendidas por ascenso
    de gradiente con línea base R̄, sin Q ni temperatura τ)::

        Inicializar, para a = 0..K−1:  Q(a) ← 0,  N(a) ← 0
        Repetir en cada pull:
            π(a) ← e^(Q(a)/τ) / Σ_b e^(Q(b)/τ)      para todo a
            A ~ π                                    (muestrear)
            R ← bandit(A)
            N(A) ← N(A) + 1
            Q(A) ← Q(A) + [R − Q(A)] / N(A)

    **Teoría.** La distribución de Boltzmann (de la física estadística, donde
    τ es la temperatura) convierte las estimaciones Q en probabilidades:

    * τ → ∞: todos los e^(Q/τ) → 1 → elección uniforme (exploración pura).
    * τ → 0: el brazo de mayor Q acapara toda la probabilidad → greedy.
    * τ intermedio: la exploración es **graduada por el valor**: un brazo
      casi tan bueno como el mejor se prueba a menudo y uno pésimo casi nunca
      (a diferencia de ε-greedy, que explora todos por igual).

    τ tiene las unidades de la recompensa: lo que importa es la diferencia
    ΔQ/τ (π_a/π_b = e^(ΔQ/τ)). Con τ = 0.1 una ventaja de 0.2 en Q equivale
    a un factor e² ≈ 7.4 en probabilidad.

    **Truco numérico.** π no cambia si se resta la misma constante m a todas
    las Q (numerador y denominador se multiplican por e^(−m/τ)). Restando
    m = max(Q) el mayor exponente es 0: nunca hay overflow (con Q = 10⁴ y
    τ = 0.1, e^(10⁵) = inf) y el denominador es ≥ 1 (nunca 0/0).

    **Annealing.** τ_t = max(τ_min, τ₀·dᵗ) (t = pulls realizados; con d = 1
    se usa τ₀ exacto): temperatura alta al inicio (explorar) que se "enfría"
    hacia greedy, como en el recocido simulado.

    Parameters
    ----------
    n_arms : int
        Número de brazos K.
    tau : float
        Temperatura inicial τ₀ > 0 (en unidades de recompensa).
    decay : float
        Factor d por pull en (0, 1]; 1 = temperatura constante.
    tau_min : float
        Cota inferior de τ con annealing (≥ 0).
    rng : np.random.Generator | None
        Generador del agente (muestreo de π).

    Attributes
    ----------
    tau0 : float
        Temperatura inicial τ₀.
    decay : float
        Factor de annealing d.
    tau_min : float
        Cota inferior τ_min.

    Raises
    ------
    ValueError
        Si τ₀ ≤ 0, τ_min < 0 o d fuera de (0, 1].

    Examples
    --------
    >>> agent = SoftmaxAgent(n_arms=2, tau=0.5, rng=np.random.default_rng(0))
    >>> agent.update(0, 1.0)                # Q = [1, 0]
    >>> agent.probabilities().round(3)      # e² / (e² + 1) = 0.881
    array([0.881, 0.119])
    """

    name = "softmax"

    SCORE_NAME = "π(a) = e^(Q/τ) / Σ e^(Q/τ)"

    def __init__(
        self,
        n_arms: int,
        tau: float = 0.1,
        decay: float = 1.0,
        tau_min: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not tau > 0.0:
            raise ValueError(f"La temperatura τ debe ser > 0 (se recibió {tau}).")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"El annealing de τ debe estar en (0, 1] (se recibió {decay}).")
        if not tau_min >= 0.0:
            raise ValueError(f"τ_min debe ser ≥ 0 (se recibió {tau_min}).")
        self.tau0: float = float(tau)
        self.decay: float = float(decay)
        self.tau_min: float = float(tau_min)
        super().__init__(n_arms, rng=rng, q0=0.0, alpha=None)

    @property
    def tau(self) -> float:
        """τ vigente en el pull actual: τ_t = max(τ_min, τ₀·dᵗ) (τ₀ si d = 1)."""
        if self.decay == 1.0:
            return self.tau0
        return max(self.tau_min, self.tau0 * self.decay ** self.t)

    def probabilities(self) -> np.ndarray:
        """Distribución de Boltzmann actual π, forma ``(K,)``, suma 1.

        Returns
        -------
        np.ndarray
            π(a) = e^((Q_a − max Q)/τ) / Σ_b e^((Q_b − max Q)/τ).

        Notes
        -----
        Si τ llega a 0 (annealing con τ_min = 0 y subdesbordamiento) o algún Q
        es infinito, π es el límite τ → 0: uniforme entre los argmax de Q.
        """
        tau = self.tau
        q_max = self.q.max()
        if tau <= 0.0 or not math.isfinite(q_max):
            greedy = (self.q == q_max).astype(np.float64)
            return greedy / greedy.sum()
        # Restar max(Q): exponentes ≤ 0 → sin overflow; el máximo aporta e⁰ = 1.
        weights = np.exp((self.q - q_max) / tau)
        return weights / weights.sum()

    def decision_scores(self) -> tuple[np.ndarray, str]:
        """Probabilidades de Boltzmann actuales y su nombre (para la GUI en vivo).

        Returns
        -------
        tuple[np.ndarray, str]
            ``(π, "π(a) = e^(Q/τ) / Σ e^(Q/τ)")`` con π de forma ``(K,)``,
            no negativa y que suma 1.
        """
        return self.probabilities(), self.SCORE_NAME

    def select_arm(self) -> int:
        """Muestrea un brazo de π y (si ``explain``) explica la decisión.

        El muestreo usa el método de la transformada inversa: con
        u ~ U[0, 1) se elige el primer brazo cuya probabilidad acumulada
        supera u (equivale a ``rng.choice(K, p=π)``, pero sin su costosa
        validación de π en cada pull).

        Returns
        -------
        int
            Brazo elegido; ``last_decision.kind == "sample"``.
        """
        tau = self.tau
        probs = self.probabilities()
        cdf = np.cumsum(probs)
        u = self.rng.random()
        # u·cdf[-1] < cdf[-1] siempre: el índice queda en 0..K−1 y nunca cae en
        # un brazo con π = 0 (aunque el redondeo haga que Σπ ≠ 1 exactamente).
        arm = min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), self.n_arms - 1)
        self.last_decision = self._explain_choice(arm, tau, probs) if self.explain else None
        return arm

    def _explain_choice(self, arm: int, tau: float, probs: np.ndarray) -> Decision:
        """Construye la :class:`Decision` de un muestreo softmax.

        Parameters
        ----------
        arm : int
            Brazo muestreado.
        tau : float
            Temperatura vigente.
        probs : np.ndarray
            Distribución π usada, forma ``(K,)``.

        Returns
        -------
        Decision
            ``kind == "sample"`` con las probabilidades π como criterio.
        """
        if self.q[arm] >= self.q.max():
            why = "es el de mayor Q"
        else:
            why = "no es el de mayor Q → explora"
        text = f"Softmax (τ={_fmt(tau)}): muestrea brazo {arm} con π={_fmt(float(probs[arm]))} ({why})"
        return Decision(arm, "sample", probs, self.SCORE_NAME, text)

    def _params_text(self) -> str:
        decay = "" if self.decay == 1.0 else f", d={self.decay:g}, τ_min={self.tau_min:g}"
        return f"τ₀={self.tau0:g}{decay}"


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------

#: Clase de cada algoritmo (las claves coinciden con :data:`src.config.ALGORITHMS`).
AGENT_CLASSES: dict[str, type[BanditAgent]] = {
    "egreedy": EpsilonGreedyAgent,
    "optimistic": OptimisticGreedyAgent,
    "ucb1": UCB1Agent,
    "softmax": SoftmaxAgent,
}


def make_agent(name: str, n_arms: int, cfg: AgentConfig, rng: np.random.Generator | None = None,
               explain: bool = True) -> BanditAgent:
    """Fábrica: crea el agente ``name`` con los hiperparámetros de ``cfg``.

    Parameters
    ----------
    name : str
        ``"egreedy"``, ``"optimistic"``, ``"ucb1"`` o ``"softmax"``.
    n_arms : int
        Número de brazos K del segmento.
    cfg : AgentConfig
        Hiperparámetros: ``epsilon``, ``epsilon_decay``, ``epsilon_min``
        (ε-greedy); ``q0``, ``optimistic_epsilon``, ``optimistic_alpha``
        (optimista); ``ucb_c`` (UCB1); ``tau``, ``tau_decay``, ``tau_min``
        (Softmax).
    rng : np.random.Generator | None
        Generador del agente.
    explain : bool, optional
        Si es False el agente no construye la explicación en texto de cada
        decisión (modo rápido de los experimentos; mismas elecciones).

    Returns
    -------
    BanditAgent
        Agente nuevo (estado inicial).

    Raises
    ------
    ValueError
        Si ``name`` no es un algoritmo conocido o algún hiperparámetro es inválido.

    Examples
    --------
    >>> agent = make_agent("softmax", n_arms=5, cfg=AgentConfig(tau=0.2))
    >>> agent.name, agent.n_arms, agent.tau
    ('softmax', 5, 0.2)
    """
    if name == "egreedy":
        agent: BanditAgent = EpsilonGreedyAgent(
            n_arms, epsilon=cfg.epsilon, decay=cfg.epsilon_decay, epsilon_min=cfg.epsilon_min, rng=rng
        )
    elif name == "optimistic":
        agent = OptimisticGreedyAgent(
            n_arms, q0=cfg.q0, epsilon=cfg.optimistic_epsilon, alpha=cfg.optimistic_alpha, rng=rng
        )
    elif name == "ucb1":
        agent = UCB1Agent(n_arms, c=cfg.ucb_c, rng=rng)
    elif name == "softmax":
        agent = SoftmaxAgent(n_arms, tau=cfg.tau, decay=cfg.tau_decay, tau_min=cfg.tau_min, rng=rng)
    else:
        valid = ", ".join(f"'{k}'" for k in AGENT_CLASSES)
        raise ValueError(f"Algoritmo desconocido: '{name}'. Opciones válidas: {valid}.")
    agent.explain = bool(explain)
    logger.debug("Agente creado: %r", agent)
    return agent
