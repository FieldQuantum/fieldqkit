"""Torch-based statevector simulator (standard axis order, counts in little-endian)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from ..circuit import QuantumCircuit
from .matrix import gate_matrix_dict, ketn0
from ..core.observables import pauli_basis_pattern
from ..circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)


def _resolve_param(qc: QuantumCircuit, param, param_values: Dict[str, object] | None = None):
    if isinstance(param, (float, int)):
        return float(param)
    if isinstance(param, str):
        if param_values is not None and param in param_values:
            return param_values[param]
        if param in qc.params_value:
            value = qc.params_value[param]
            if isinstance(value, (float, int)):
                return float(value)
        def _symbol_resolver(name: str):
            if name == "pi":
                return float(torch.pi)
            if param_values is not None and name in param_values:
                return param_values[name]
            if name in qc.params_value and isinstance(qc.params_value[name], (float, int)):
                return float(qc.params_value[name])
            raise ValueError(f"missing parameter value for {name}")

        return qc._eval_param_expression(param, symbol_resolver=_symbol_resolver)
    raise TypeError(f"unsupported parameter type: {type(param)}")


def _apply_k_qubit_gate_torch(
    state,
    gate,
    qubits: Sequence[int],
    num_qubits: int,
):
    k = len(qubits)
    if k == 0:
        return state
    axes = list(qubits)
    tensor = state.reshape([2] * num_qubits)
    tensor = torch.moveaxis(tensor, axes, list(range(k)))
    tensor = tensor.reshape(2**k, -1)
    tensor = gate @ tensor
    tensor = tensor.reshape([2] * num_qubits)
    tensor = torch.moveaxis(tensor, list(range(k)), axes)
    return tensor.reshape(-1)


def _apply_reset_torch(state, qubit: int, num_qubits: int):
    axis = qubit
    tensor = state.reshape([2] * num_qubits)
    slicer = [slice(None)] * num_qubits
    slicer[axis] = 1
    tensor[tuple(slicer)] = 0.0
    state = tensor.reshape(-1)
    norm = torch.linalg.norm(state)
    if float(norm.detach().cpu().item()) > 0.0:
        state = state / norm
    return state


def _materialize_gate_matrix(gate: str, params, *, dtype: torch.dtype, device: torch.device):
    mat_or_fn = gate_matrix_dict[gate]
    if callable(mat_or_fn):
        return mat_or_fn(*params, dtype=dtype, device=device)
    return mat_or_fn.to(device=device, dtype=dtype)


def simulate_statevector(
    qc: QuantumCircuit,
    *,
    param_values: Dict[str, object] | None = None,
):
    num_qubits = int(qc.nqubits)
    if num_qubits <= 0:
        return torch.tensor([1.0 + 0.0j], dtype=torch.complex128)

    # Start from |0...0> and apply gates in circuit order.
    state = ketn0(num_qubits).reshape(-1)
    dtype = state.dtype
    device = state.device

    for gate_info in qc.gates:
        gate = gate_info[0]
        if gate in functional_gates_available:
            if gate == "reset":
                state = _apply_reset_torch(state, gate_info[1], num_qubits)
            continue

        if gate in one_qubit_gates_available:
            qubit = gate_info[1]
            mat = _materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, [qubit], num_qubits)
            continue

        if gate in two_qubit_gates_available:
            qubits = gate_info[1:3]
            mat = _materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, qubits, num_qubits)
            continue

        if gate in three_qubit_gates_available:
            qubits = gate_info[1:4]
            mat = _materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, qubits, num_qubits)
            continue

        if gate in one_qubit_parameter_gates_available:
            qubit = gate_info[-1]
            params = [_resolve_param(qc, p, param_values) for p in gate_info[1:-1]]
            mat = _materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, [qubit], num_qubits)
            continue

        if gate in two_qubit_parameter_gates_available:
            qubits = gate_info[-2:]
            params = [_resolve_param(qc, p, param_values) for p in gate_info[1:-2]]
            mat = _materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, qubits, num_qubits)
            continue

        raise ValueError(f"unsupported gate for simulator: {gate}")

    return state


def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: int | None = None,
    param_values: Dict[str, object] | None = None,
) -> Dict[str, int]:
    # Sample from the statevector distribution, then emit little-endian bitstrings.
    state = simulate_statevector(qc, param_values=param_values)
    num_qubits = int(qc.nqubits)
    probs = (state.abs() ** 2).real
    total = probs.sum()
    if float(total.detach().cpu().item()) > 0.0:
        probs = probs / total

    generator = torch.Generator(device=probs.device)
    if seed is not None:
        generator.manual_seed(int(seed))
    samples = torch.multinomial(probs, num_samples=shots, replacement=True, generator=generator)
    counts = torch.bincount(samples, minlength=probs.numel())

    out: Dict[str, int] = {}
    for idx, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        bits = format(idx, f"0{num_qubits}b")[::-1]
        out[bits] = int(count)
    return out


def build_state_from_symbolic(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names: Sequence[str],
):
    """Build statevector from a symbolic circuit and differentiable param tensor."""
    expected = len(param_names)
    if params.numel() != expected:
        raise ValueError(f"params length must be {expected}")

    flat_params = params.reshape(-1)
    param_values = {name: flat_params[i] for i, name in enumerate(param_names)}
    return simulate_statevector(symbolic_qc, param_values=param_values)


def _single_pauli(op: str, *, dtype, device):
    if op == "X":
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    if op == "Y":
        return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
    if op == "Z":
        return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    raise ValueError(f"unsupported Pauli: {op}")


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> for a Pauli string using local operator application."""
    pattern = pauli_basis_pattern(pauli, num_qubits=num_qubits)
    acted = state
    for idx, op in enumerate(pattern):
        if op == "I":
            continue
        acted = _apply_k_qubit_gate_torch(
            acted,
            _single_pauli(op, dtype=state.dtype, device=state.device),
            [idx],
            num_qubits,
        )
    return torch.vdot(state, acted)


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names: Sequence[str],
    hamiltonian: List[Tuple[float, str]],
) -> Tuple[object, dict[str, float]]:
    """Evaluate Hamiltonian energy from a symbolic circuit template in a differentiable way."""
    num_qubits = int(symbolic_qc.nqubits)
    state = build_state_from_symbolic(symbolic_qc, params=params, param_names=param_names)
    energy = torch.zeros((), dtype=params.dtype, device=params.device)
    expectations: dict[str, float] = {}
    for coeff, obs in hamiltonian:
        exp_val = expectation_pauli(state, obs, num_qubits=num_qubits).real
        energy = energy + float(coeff) * exp_val
        expectations[obs] = float(exp_val.detach().cpu().item())
    return energy, expectations
