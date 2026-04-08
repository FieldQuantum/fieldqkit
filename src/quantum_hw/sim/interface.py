"""Simulator interface and threshold-based dispatch.

This module is the single place that decides whether to use statevector or MPS.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from ..circuit import QuantumCircuit

from .mps import energy_and_expectations as _energy_and_expectations_mps
from .mps import expectation_pauli as _expectation_pauli_mps
from .mps import sample_probabilities as _sample_probabilities_mps
from .mps import simulate_counts as _simulate_counts_mps
from .statevector import energy_and_expectations as _energy_and_expectations_statevector
from .statevector import expectation_pauli as _expectation_pauli_statevector
from .statevector import sample_probabilities as _sample_probabilities_statevector
from .statevector import simulate_counts as _simulate_counts_statevector


MPS_THRESHOLD_QUBITS: int = 16


def _extract_measurements(qc: QuantumCircuit):
    """Extract qubit-to-cbit mapping from measure gates in a circuit.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.

    Returns:
        Tuple of ``(measured_qubits, measured_cbits, num_cbits)`` or ``None``
        if the circuit contains no measurement gates.
    """
    measured_qubits: list[int] = []
    measured_cbits: list[int] = []
    for gate in getattr(qc, "gates", []):
        if gate[0] == "measure":
            measured_qubits.extend(gate[1])
            measured_cbits.extend(gate[2])
    if not measured_qubits:
        return None
    num_cbits = max(measured_cbits) + 1
    return measured_qubits, measured_cbits, num_cbits


def _project_counts_to_cbits(
    counts: Dict[str, int],
    measured_qubits: Sequence[int],
    measured_cbits: Sequence[int],
    num_cbits: int,
) -> Dict[str, int]:
    """Project full-qubit simulator counts to the classical-bit subspace.

    Each qubit in *measured_qubits* is mapped to the corresponding classical
    bit in *measured_cbits*.  Unmeasured qubits are marginalized out.

    Args:
        counts (*Dict[str, int]*): Full-qubit counts from the simulator.
        measured_qubits (*Sequence[int]*): Qubit indices that are measured.
        measured_cbits (*Sequence[int]*): Classical bit index each qubit maps to.
        num_cbits (*int*): Width of the output bitstring (``max(cbits) + 1``).

    Returns:
        Counts dictionary with ``num_cbits``-wide bitstrings.
    """
    projected: Dict[str, int] = {}
    for bitstring, count in counts.items():
        cbits = [0] * num_cbits
        for q, c in zip(measured_qubits, measured_cbits):
            cbits[c] = int(bitstring[q])
        key = "".join(str(b) for b in cbits)
        projected[key] = projected.get(key, 0) + count
    return projected


def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    param_values: Dict[str, object] | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Simulate counts with threshold-based backend selection.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        shots (*int*): Number of measurement shots.
        seed (*Optional[int]*): Random seed for reproducibility. Defaults to ``None``.
        param_values (*Dict[str, object] | None*): Parameter name to value mapping. Defaults to ``None``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``Dict[str, int]`` mapping bitstrings to their occurrence counts.
        When the circuit contains explicit measure gates with a qubit-to-cbit
        mapping, the returned bitstrings are projected to the classical-bit
        subspace (width = ``max(cbit) + 1``).
    """

    nqubits = int(getattr(qc, "nqubits", 0) or 0)
    if nqubits > MPS_THRESHOLD_QUBITS:
        raw = _simulate_counts_mps(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
            device=device,
        )
    else:
        raw = _simulate_counts_statevector(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
            device=device,
        )

    meas = _extract_measurements(qc)
    if meas is not None:
        return _project_counts_to_cbits(raw, *meas)
    return raw


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> using threshold-based backend selection.

    Args:
        state: Flat statevector tensor (≤ threshold qubits) or MPS tensor list (> threshold qubits).
        pauli (*str*): Pauli string (e.g. ``'XZI'``).
        num_qubits (*int*): Number of qubits.

    Returns:
        Expectation value as a scalar.
    """
    if int(num_qubits) > MPS_THRESHOLD_QUBITS:
        return _expectation_pauli_mps(state, pauli, num_qubits=num_qubits)
    return _expectation_pauli_statevector(state, pauli, num_qubits=num_qubits)


def sample_probabilities(
    state,
    samples,
    *,
    num_qubits: int,
):
    """Return probabilities for sample vectors via threshold-based dispatch.

    Args:
        state: Statevector or MPS.
        samples: ``(N, n_qubits)`` integer tensor/array with entries 0/1.
        num_qubits (*int*): Number of qubits, used to select backend.

    Returns:
        Probability tensor for the given samples.
    """
    if int(num_qubits) > MPS_THRESHOLD_QUBITS:
        return _sample_probabilities_mps(state, samples)
    return _sample_probabilities_statevector(state, samples)


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names,
    hamiltonian,
    device: torch.device | str | None = None,
):
    """Evaluate Hamiltonian energy via threshold-based backend selection.

    Args:
        symbolic_qc (*QuantumCircuit*): Symbolic (unbound) quantum circuit.
        params (*torch.Tensor*): 1-D tensor of variational parameter values.
        param_names (*List[str]*): Names of variational parameters, matching ``params`` element-wise.
        hamiltonian (*List[Tuple[float, str]]*): Target Hamiltonian as coefficient–Pauli-string pairs.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``(energy, expectations)`` tuple from the selected backend.
    """
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
