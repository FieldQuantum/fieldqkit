"""Automation agent entrypoint.

Base imports and token only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quantum_hw import QuantumHardwareClient
