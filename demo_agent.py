"""Automation agent entrypoint.

Base imports and token only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quantum_hw import QuantumHardwareClient

token = "5gjq36bZsMvqFoSNomvnfPy4y[iDJWe[tBx9fIndISQ/:m{O5FEPyRkM4B{N{RkNyd{OypkJxiY[jxjJ4RkPyBkPxJEJ4FUMyBUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"
