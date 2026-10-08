"""Smartuner: transcripción de tablatura de bajo con Multi-Armed Bandits.

Paquete con toda la lógica del sistema, independiente de la interfaz gráfica.
Cada módulo corresponde a una etapa del pipeline:

``io_audio`` → ``separation`` → ``preprocessing`` → ``segmentation`` → ``pitch``
→ ``environment`` (bandit por segmento) ↔ ``agents`` → ``tab``;
``pipeline`` orquesta las etapas 1–6 (del archivo de audio hasta los datos
bandit de cada segmento), ``experiments`` compara algoritmos, ``plots``
genera las gráficas y ``synth_dataset`` crea audio con ground truth.
``config`` reúne todos los hiperparámetros.
"""

from __future__ import annotations
