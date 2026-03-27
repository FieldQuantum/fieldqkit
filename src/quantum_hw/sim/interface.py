"""Simulator interface and threshold-based dispatch.

This module is the single place that decides whether to use statevector or MPS.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..circuit import QuantumCircuit
from .mps import energy_and_expectations as _energy_and_expectations_mps
from .mps import expectation_pauli as _expectation_pauli_mps
from .mps import simulate_counts as _simulate_counts_mps
from .statevector import energy_and_expectations as _energy_and_expectations_statevector
from .statevector import expectation_pauli as _expectation_pauli_statevector
from .statevector import simulate_counts as _simulate_counts_statevector


MPS_THRESHOLD_QUBITS: int = 16

def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    param_values: Dict[str, object] | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Simulate counts with threshold-based backend selection."""

    nqubits = int(getattr(qc, "nqubits", 0) or 0)
    if nqubits > MPS_THRESHOLD_QUBITS:
        return _simulate_counts_mps(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
            device=device,
        )
    else:
        return _simulate_counts_statevector(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
            device=device,
        )


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> using threshold-based backend selection."""
    if int(num_qubits) > MPS_THRESHOLD_QUBITS:
        return _expectation_pauli_mps(state, pauli, num_qubits=num_qubits)
    return _expectation_pauli_statevector(state, pauli, num_qubits=num_qubits)


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names,
    hamiltonian,
    device: torch.device | str | None = None,
):
    """Evaluate Hamiltonian energy via threshold-based backend selection."""
    nqubits = int(getattr(symbolic_qc, "nqubits", 0) or 0)
    if nqubits > MPS_THRESHOLD_QUBITS:
        return _energy_and_expectations_mps(
            symbolic_qc,
            params=params,
            param_names=param_names,
            hamiltonian=hamiltonian,
            device=device,
        )
    return _energy_and_expectations_statevector(
        symbolic_qc,
        params=params,
        param_names=param_names,
        hamiltonian=hamiltonian,
        device=device,
    )
