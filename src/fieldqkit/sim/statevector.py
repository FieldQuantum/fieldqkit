"""Torch-based statevector simulator (standard axis order)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from ..circuit import QuantumCircuit
from .matrix import ketn0
from .common import (
    auto_sim_device,
    build_param_values_from_tensor,
    grad_checkpoint_enabled,
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
    gates = list(qc.gates)

    # For large differentiable runs, segment the gate sequence and recompute
    # intermediates in backward (gradient checkpointing): O(sqrt(n_gates))
    # saved states instead of O(n_gates).  Sampling / inference is untouched.
    if grad_checkpoint_enabled(num_qubits, param_values) and len(gates) > 1:
        chunk = max(1, int(len(gates) ** 0.5))
        for i in range(0, len(gates), chunk):
            state = checkpoint(
                _apply_gate_block,
                state, gates[i:i + chunk], qc, param_values,
                num_qubits, dtype, device,
                use_reentrant=False,
            )
    else:
        state = _apply_gate_block(
            state, gates, qc, param_values, num_qubits, dtype, device,
        )

    return state


def _apply_gate_block(state, gates, qc, param_values, num_qubits, dtype, device):
    """Apply a contiguous block of circuit gates to ``state`` (in order).

    Extracted from :func:`simulate_statevector` so a block can be wrapped in
    :func:`torch.utils.checkpoint.checkpoint`.
    """
    for gate_info in gates:
        gate = gate_info[0]
        if gate in functional_gates_available:
            if gate == "reset":
                raise NotImplementedError(
                    "The statevector simulator does not support the 'reset' operation."
                )
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
    else:
        generator.seed()  # a fresh Generator is otherwise deterministic

    # Inverse-CDF sampling via searchsorted. Unlike torch.multinomial, this has
    # no 2**24-category limit, so it scales past 24 qubits. The CDF is built in
    # float64 to preserve resolution between tiny probabilities at large N.
    cdf = torch.cumsum(probs.double(), dim=0)
    cdf[-1] = 1.0  # guard against float rounding leaving the last edge < 1
    u = torch.rand(shots, device=probs.device, dtype=torch.float64, generator=generator)
    samples = torch.searchsorted(cdf, u, right=True)
    samples = samples.clamp_(max=probs.numel() - 1)

    # Tally only the outcomes that were actually drawn (<= shots distinct), which
    # avoids materialising a length-2**num_qubits bincount and looping over it.
    idxs, counts = torch.unique(samples, return_counts=True)
    out: Dict[str, int] = {}
    for idx, count in zip(idxs.tolist(), counts.tolist()):
        bits = format(idx, f"0{num_qubits}b")
        out[bits] = int(count)
    return out


def apply_pauli_string(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return ``P|psi>`` for a Pauli string via local operator application.
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
    return acted


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
    return torch.vdot(state, apply_pauli_string(state, pauli, num_qubits=num_qubits))


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

    h_psi = torch.zeros_like(state)
    expectations: dict[str, float] = {}
    for coeff, obs in hamiltonian:
        acted = apply_pauli_string(state, obs, num_qubits=num_qubits)
        h_psi = h_psi + float(coeff) * acted
        expectations[obs] = float(torch.vdot(state, acted).real.detach().cpu().item())
    energy = torch.vdot(state, h_psi).real
    return energy, expectations
