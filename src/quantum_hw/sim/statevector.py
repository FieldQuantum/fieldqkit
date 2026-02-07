"""Simple statevector simulator (ketn0 axis order, counts in little-endian)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from ..circuit import QuantumCircuit
from ..circuit.matrix import gate_matrix_dict, ketn0
from ..circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)


def _resolve_param(qc: QuantumCircuit, param):
    if isinstance(param, (float, int)):
        return float(param)
    if isinstance(param, str):
        if param not in qc.params_value:
            raise ValueError(f"missing parameter value for {param}")
        value = qc.params_value[param]
        if isinstance(value, (float, int)):
            return float(value)
        raise ValueError(f"invalid parameter value for {param}")
    raise TypeError(f"unsupported parameter type: {type(param)}")


def _apply_k_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubits: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    k = len(qubits)
    if k == 0:
        return state
    # State tensor follows the same qubit order as ketn0 (q0 is the first axis).
    # Move target axes to the front so the gate acts on the correct subspace.
    axes = list(qubits)
    tensor = state.reshape([2] * num_qubits)
    tensor = np.moveaxis(tensor, axes, range(k))
    tensor = tensor.reshape(2**k, -1)
    tensor = gate @ tensor
    tensor = tensor.reshape([2] * num_qubits)
    tensor = np.moveaxis(tensor, range(k), axes)
    return tensor.reshape(-1)


def _apply_reset(state: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    axis = qubit
    tensor = state.reshape([2] * num_qubits)
    slicer = [slice(None)] * num_qubits
    slicer[axis] = 1
    tensor[tuple(slicer)] = 0.0
    state = tensor.reshape(-1)
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    return state


def simulate_statevector(qc: QuantumCircuit) -> np.ndarray:
    num_qubits = qc.nqubits
    if num_qubits <= 0:
        return np.array([1.0 + 0.0j])
    # Start from |0...0> and apply gates in circuit order.
    state = ketn0(num_qubits).reshape(-1)

    for gate_info in qc.gates:
        gate = gate_info[0]
        if gate in functional_gates_available:
            if gate == "reset":
                state = _apply_reset(state, gate_info[1], num_qubits)
            continue

        if gate in one_qubit_gates_available:
            qubit = gate_info[1]
            mat = gate_matrix_dict[gate]
            # Single-qubit gate acts on the specified qubit axis.
            state = _apply_k_qubit_gate(state, mat, [qubit], num_qubits)
            continue

        if gate in two_qubit_gates_available:
            qubits = gate_info[1:3]
            mat = gate_matrix_dict[gate]
            # Two-qubit gates follow the circuit's (control, target) order.
            state = _apply_k_qubit_gate(state, mat, qubits, num_qubits)
            continue

        if gate in three_qubit_gates_available:
            qubits = gate_info[1:4]
            mat = gate_matrix_dict[gate]
            # Three-qubit gate order matches the stored gate tuple.
            state = _apply_k_qubit_gate(state, mat, qubits, num_qubits)
            continue

        if gate in one_qubit_parameter_gates_available:
            qubit = gate_info[-1]
            params = [_resolve_param(qc, p) for p in gate_info[1:-1]]
            mat_fn = gate_matrix_dict[gate]
            mat = mat_fn(*params)
            state = _apply_k_qubit_gate(state, mat, [qubit], num_qubits)
            continue

        if gate in two_qubit_parameter_gates_available:
            qubits = gate_info[-2:]
            params = [_resolve_param(qc, p) for p in gate_info[1:-2]]
            mat_fn = gate_matrix_dict[gate]
            mat = mat_fn(*params)
            state = _apply_k_qubit_gate(state, mat, qubits, num_qubits)
            continue

        raise ValueError(f"unsupported gate for simulator: {gate}")

    return state


def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: int | None = None,
) -> Dict[str, int]:
    # Sample from the statevector distribution, then emit little-endian bitstrings.
    state = simulate_statevector(qc)
    num_qubits = qc.nqubits
    probs = np.abs(state) ** 2
    total = probs.sum()
    if total > 0:
        probs = probs / total
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=shots, p=probs)
    counts = np.bincount(samples, minlength=len(probs))
    out: Dict[str, int] = {}
    for idx, count in enumerate(counts):
        if count == 0:
            continue
        bits = format(idx, f"0{num_qubits}b")[::-1]
        out[bits] = int(count)
    return out
