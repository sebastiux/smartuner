"""Smartuner: transcripción de tablatura de bajo con Multi-Armed Bandits.

Paquete con toda la lógica del sistema, independiente de la interfaz gráfica.
Cada módulo corresponde a una etapa del pipeline:

``io_audio`` → ``separation`` → ``preprocessing`` → ``segmentation`` → ``pitch``
→ ``environment`` (bandit por segmento) ↔ ``agents`` → ``tab``;
``pipeline`` orquesta las etapas 1–5, ``experiments`` compara algoritmos,
``plots`` genera las gráficas y ``synth_dataset`` crea audio con ground truth.
"""
