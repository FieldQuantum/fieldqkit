"""Simulation utilities facade.

This subpackage requires PyTorch.  Install with::

    pip install quantum-hw[sim]
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError as _exc:
    raise ModuleNotFoundError(
        "quantum_hw.sim requires PyTorch. "
        "Install it with:  pip install quantum-hw[sim]"
    ) from _exc

from .common import auto_sim_device
from .interface import get_sim_config
from .interface import sample_probabilities
from .interface import energy_and_expectations
from .interface import expectation_pauli
from .interface import set_sim_config
from .interface import simulate_counts
from .statevector import simulate_statevector
from .mpo import simulate_mpo_process
from .mps import simulate_mps


__all__ = [
    "auto_sim_device",
    "get_sim_config",
    "sample_probabilities",
    "energy_and_expectations",
    "expectation_pauli",
    "set_sim_config",
    "simulate_counts",
    "simulate_mpo_process",
    "simulate_mps",
    "simulate_statevector",
]
