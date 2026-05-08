"""Torch-based statevector simulator (standard axis order)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from ..circuit import QuantumCircuit
from .matrix import ketn0
from .common import (
    auto_sim_device,
    build_param_values_from_tensor,
    materialize_gate_matrix,
    resolve_param,
    single_pauli,
)
from ..core.observables import pauli_basis_pattern
from ..circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)


def _apply_k_qubit_gate_torch(
    state,
    gate,
    qubits: Sequence[int],
    num_qubits: int,
):
    """Apply a k-qubit gate to a statevector tensor via axis permutation and matrix multiply.

    Reshapes the flat state into an n-qubit tensor, moves target qubit axes to
    the front, multiplies the ``(2^k, 2^k)`` gate matrix, and restores axis order.

    Args:
        state: Flat statevector tensor of length ``2**num_qubits``.
        gate: Unitary gate matrix of shape ``(2**k, 2**k)``.
        qubits (*Sequence[int]*): Target qubit indices for the gate.
        num_qubits (*int*): Total number of qubits in the system.

    Returns:
        Updated flat statevector tensor of length ``2**num_qubits``.
    """
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
    """Apply a reset operation on a single qubit, projecting it to |0⟩ and renormalizing.

    Zeros all amplitudes where the target qubit is |1⟩, then renormalizes
    the resulting state to unit norm.

    Args:
        state: Flat statevector tensor of length ``2**num_qubits``.
        qubit (*int*): Target qubit index to reset.
        num_qubits (*int*): Total number of qubits in the system.

    Returns:
        Renormalized statevector with the target qubit projected to |0⟩.
    """
    axis = qubit
    tensor = state.reshape([2] * num_qubits)
    slicer0 = [slice(None)] * num_qubits
    slicer1 = [slice(None)] * num_qubits
    slicer0[axis] = 0
    slicer1[axis] = 1
    # Move |1⟩ amplitudes into |0⟩ before clearing, so no amplitude is lost.
    tensor[tuple(slicer0)] = tensor[tuple(slicer0)] + tensor[tuple(slicer1)]
    tensor[tuple(slicer1)] = 0.0
    state = tensor.reshape(-1)
    norm = torch.linalg.norm(state)
    if float(norm.detach().cpu().item()) > 0.0:
        state = state / norm
    return state


def simulate_statevector(
    qc: QuantumCircuit,
    *,
    param_values: Dict[str, object] | None = None,
    device: torch.device | str | None = None,
):
    """Simulate a quantum circuit and return the full statevector.

    Starts from the |0...0⟩ state and applies each gate in circuit order using
    tensor-axis permutation and matrix multiplication.

    Args:
        qc (*QuantumCircuit*): Quantum circuit to simulate.
        param_values (*Dict[str, object] | None*): Symbolic parameter name-to-value map. Defaults to ``None``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Complex statevector tensor of length ``2**nqubits``.

    Raises:
        ValueError: If a gate in the circuit is not supported by the simulator.
    """
    num_qubits = int(qc.nqubits)
    sim_device = auto_sim_device(device)
    if num_qubits <= 0:
        return torch.tensor([1.0 + 0.0j], dtype=torch.complex128, device=sim_device)

    # Start from |0...0> and apply gates in circuit order.
    state = ketn0(num_qubits, device=sim_device).reshape(-1)
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
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, [qubit], num_qubits)
            continue

        if gate in two_qubit_gates_available:
            qubits = gate_info[1:3]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, qubits, num_qubits)
            continue

        if gate in three_qubit_gates_available:
            qubits = gate_info[1:4]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, qubits, num_qubits)
            continue

        if gate in one_qubit_parameter_gates_available:
            qubit = gate_info[-1]
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-1]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            state = _apply_k_qubit_gate_torch(state, mat, [qubit], num_qubits)
            continue

        if gate in two_qubit_parameter_gates_available:
            qubits = gate_info[-2:]
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-2]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
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
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Sample measurement outcomes from a statevector simulation.

    Simulates the circuit to obtain the statevector, computes Born-rule
    probabilities, and draws *shots* samples via multinomial sampling.

    Args:
        qc (*QuantumCircuit*): Quantum circuit to simulate.
        shots (*int*): Number of measurement shots.
        seed (*int | None*): Random seed for reproducibility. Defaults to ``None``.
        param_values (*Dict[str, object] | None*): Symbolic parameter name-to-value map. Defaults to ``None``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Dictionary mapping bitstrings to their observed counts.
    """
    # Sample from the statevector distribution.
    state = simulate_statevector(qc, param_values=param_values, device=device)
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
        bits = format(idx, f"0{num_qubits}b")
        out[bits] = int(count)
    return out


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> for a Pauli string using local operator application.

    Args:
        state: Quantum state vector or tensor.
        pauli (*str*): Pauli string (e.g. ``'XZI'``).
        num_qubits (*int*): Number of qubits.

    Returns:
        ``torch.Tensor`` scalar (complex) expectation value ``<psi|P|psi>``.
        For real Hamiltonians take ``.real``.
    """
    pattern = pauli_basis_pattern(pauli, num_qubits=num_qubits)
    acted = state
    for idx, op in enumerate(pattern):
        if op == "I":
            continue
        acted = _apply_k_qubit_gate_torch(
            acted,
            single_pauli(op, dtype=state.dtype, device=state.device),
            [idx],
            num_qubits,
        )
    return torch.vdot(state, acted)


def sample_probabilities(
    state,
    samples,
):
    """Return probabilities for given sample vectors.

    Args:
        state: Statevector tensor of length ``2**n``.
        samples: ``(N, n_qubits)`` integer tensor or array with entries 0/1,
            big-endian (column 0 = qubit 0).

    Returns:
        1-D tensor of length *N* with ``P(sample_i) = |⟨b_i|ψ⟩|²``.
        Fully differentiable.
    """
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.long, device=state.device)
    else:
        samples = samples.to(device=state.device, dtype=torch.long)
    n_qubits = samples.shape[1]
    weights = (2 ** torch.arange(n_qubits - 1, -1, -1, device=samples.device)).unsqueeze(0)
    indices = (samples * weights).sum(dim=1)
    amplitudes = state[indices]
    return amplitudes.real ** 2 + amplitudes.imag ** 2


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names: Sequence[str],
    hamiltonian: List[Tuple[float, str]],
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Evaluate Hamiltonian energy from a symbolic circuit template in a differentiable way.

    Args:
        symbolic_qc (*QuantumCircuit*): Symbolic (unbound) quantum circuit.
        params (*torch.Tensor | Sequence[float]*): Parameter values.
        param_names (*Sequence[str]*): Names of variational parameters.
        hamiltonian (*List[Tuple[float, str]]*): Target Hamiltonian.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Tuple of ``(energy, expectations)`` where *energy* is a differentiable
        scalar tensor and *expectations* is a ``dict[str, float]`` mapping each
        Pauli string to its expectation value.
    """
    num_qubits = int(symbolic_qc.nqubits)
    sim_device = auto_sim_device(device)
    if params.device != sim_device:
        params = params.to(sim_device)
    param_values = build_param_values_from_tensor(params=params, param_names=param_names)
    state = simulate_statevector(symbolic_qc, param_values=param_values, device=sim_device)
    energy = torch.zeros((), dtype=params.dtype, device=params.device)
    expectations: dict[str, float] = {}
    for coeff, obs in hamiltonian:
        exp_val = expectation_pauli(state, obs, num_qubits=num_qubits).real
        energy = energy + float(coeff) * exp_val
        expectations[obs] = float(exp_val.detach().cpu().item())
    return energy, expectations
