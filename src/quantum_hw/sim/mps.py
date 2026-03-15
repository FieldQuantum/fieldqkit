"""MPS simulator scaffold.

This file only exposes a function-level interface. The numerical MPS kernel can
be implemented here without changing upper layers.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..circuit import QuantumCircuit
from .statevector import energy_and_expectations as _energy_and_expectations_statevector
from .statevector import expectation_pauli as _expectation_pauli_statevector
from .statevector import simulate_counts as _simulate_counts_statevector

ENABLE_STATEVECTOR_FALLBACK: bool = True


def set_statevector_fallback(enabled: bool) -> None:
    """Enable or disable temporary fallback to statevector."""
    global ENABLE_STATEVECTOR_FALLBACK
    ENABLE_STATEVECTOR_FALLBACK = bool(enabled)


def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    param_values: Dict[str, object] | None = None,
) -> Dict[str, int]:
    """Simulate counts with MPS backend (placeholder implementation)."""
    if ENABLE_STATEVECTOR_FALLBACK:
        return _simulate_counts_statevector(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
        )
    raise NotImplementedError("MPS simulate_counts kernel is not implemented yet")


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Dummy MPS expectation interface (currently fallback-based)."""
    if ENABLE_STATEVECTOR_FALLBACK:
        return _expectation_pauli_statevector(state, pauli, num_qubits=num_qubits)
    raise NotImplementedError("MPS expectation_pauli kernel is not implemented yet")


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names,
    hamiltonian,
):
    """Dummy MPS energy interface (currently fallback-based)."""
    if ENABLE_STATEVECTOR_FALLBACK:
        return _energy_and_expectations_statevector(
            symbolic_qc,
            params=params,
            param_names=param_names,
            hamiltonian=hamiltonian,
        )
    raise NotImplementedError("MPS energy_and_expectations kernel is not implemented yet")
