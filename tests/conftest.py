"""Configuración común de pytest: añade la raíz del proyecto al ``sys.path``
para que los tests importen ``src`` y ``gui`` igual que ``main.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
