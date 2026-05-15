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
from .interface import build_state_from_symbolic
from .interface import sample_probabilities
from .interface import energy_and_expectations
from .interface import expectation_pauli
from .interface import set_sim_config
from .interface import simulate_counts
from .statevector import simulate_statevector
from .mpo import simulate_mpo_process
from .mps import simulate_mps
from .clifford import (
    CliffordError,
    is_clifford_circuit,
    simulate_clifford_expectation,
    simulate_clifford_expectations,
)
from .clifford_t import (
    count_non_clifford_gates,
    count_t_gates,
    simulate_clifford_t_expectation,
    simulate_clifford_t_expectations,
)


__all__ = [
    "auto_sim_device",
    "CliffordError",
    "count_non_clifford_gates",
    "count_t_gates",
    "get_sim_config",
    "build_state_from_symbolic",
    "is_clifford_circuit",
    "sample_probabilities",
    "energy_and_expectations",
    "expectation_pauli",
    "set_sim_config",
    "simulate_clifford_expectation",
    "simulate_clifford_expectations",
    "simulate_clifford_t_expectation",
    "simulate_clifford_t_expectations",
    "simulate_counts",
    "simulate_mpo_process",
    "simulate_mps",
    "simulate_statevector",
]
