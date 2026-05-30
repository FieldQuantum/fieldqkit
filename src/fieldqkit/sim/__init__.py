"""Simulation utilities facade.

This subpackage requires PyTorch.  Install with::

    pip install fieldqkit[sim]
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError as _exc:
    raise ModuleNotFoundError(
        "fieldqkit.sim requires PyTorch. "
        "Install it with:  pip install fieldqkit[sim]"
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
from .density_matrix import simulate_density_matrix
from .density_matrix import simulate_noisy_counts
from .density_matrix import expectation_pauli_dm
from .density_matrix import sample_probabilities_dm
from .density_matrix import energy_and_expectations as energy_and_expectations_dm
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
    "energy_and_expectations",
    "energy_and_expectations_dm",
    "expectation_pauli",
    "expectation_pauli_dm",
    "get_sim_config",
    "build_state_from_symbolic",
    "is_clifford_circuit",
    "sample_probabilities",
    "sample_probabilities_dm",
    "set_sim_config",
    "simulate_clifford_expectation",
    "simulate_clifford_expectations",
    "simulate_clifford_t_expectation",
    "simulate_clifford_t_expectations",
    "simulate_counts",
    "simulate_density_matrix",
    "simulate_mpo_process",
    "simulate_mps",
    "simulate_noisy_counts",
    "simulate_statevector",
]
