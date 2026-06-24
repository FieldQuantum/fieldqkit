"""Density matrix simulator for noisy quantum circuits."""

from __future__ import annotations
from typing import Dict, Optional, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ..circuit import QuantumCircuit
from .common import (
    auto_sim_device,
    grad_checkpoint_enabled,
    materialize_gate_matrix,
    resolve_param,
    single_pauli,
)
from ..circuit.quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    three_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    single_qubit_noise_channel_gates_available,
    two_qubit_noise_channel_gates_available,
)
from .noise_kraus import get_kraus_ops


def _apply_mat_left(tensor: torch.Tensor, mat: torch.Tensor, axes: Sequence[int], total_dims: int) -> torch.Tensor:
    """Apply a matrix to specified axes of a tensor via left-multiplication.

    For a 2n-qubit density matrix reshaped as (2,)*2n, applies:
    - When axis indices < n: (mat ⊗ I) ρ  (acts on rows)
    - When axis indices >= n: ρ (I ⊗ mat†)  (acts on columns via conjugate)

    Args:
        tensor: Tensor of shape (2,)*total_dims
        mat: Matrix of shape (2^k, 2^k)
        axes: Qubit axes to apply the matrix to
        total_dims: Total number of qubit axes (usually 2n for n-qubit density matrix)

    Returns:
        Updated tensor after applying the matrix.
    """
    k = len(axes)
    if k == 0:
        return tensor

    tensor_moved = torch.moveaxis(tensor, axes, list(range(k)))
    tensor_reshaped = tensor_moved.reshape(2**k, -1)
    tensor_applied = mat @ tensor_reshaped
    tensor_reshaped = tensor_applied.reshape([2] * total_dims)
    tensor_final = torch.moveaxis(tensor_reshaped, list(range(k)), axes)
    return tensor_final


def _apply_unitary_to_dm(rho: torch.Tensor, U: torch.Tensor, qubits: Sequence[int], n: int) -> torch.Tensor:
    """Apply a unitary gate U as: ρ' = U ρ U†.

    Args:
        rho: Density matrix as a (2,)*2n tensor
        U: Unitary matrix of shape (2^k, 2^k)
        qubits: Qubit indices to apply the unitary
        n: Total number of qubits

    Returns:
        Updated density matrix.
    """
    rho = _apply_mat_left(rho, U, list(qubits), 2 * n)
    rho = _apply_mat_left(rho, U.conj(), [n + q for q in qubits], 2 * n)
    return rho


def _apply_kraus_to_dm(rho: torch.Tensor, kraus_ops: list, qubits: Sequence[int], n: int) -> torch.Tensor:
    """Apply Kraus operators as: ρ' = Σ_k K_k ρ K_k†.

    Args:
        rho: Density matrix as a (2,)*2n tensor
        kraus_ops: List of Kraus operator matrices
        qubits: Qubit indices to apply the channel
        n: Total number of qubits

    Returns:
        Updated density matrix.
    """
    rho_new = torch.zeros_like(rho)
    for K in kraus_ops:
        rho_temp = _apply_mat_left(rho, K, list(qubits), 2 * n)
        rho_temp = _apply_mat_left(rho_temp, K.conj(), [n + q for q in qubits], 2 * n)
        rho_new = rho_new + rho_temp
    return rho_new


def simulate_density_matrix(
    qc: QuantumCircuit,
    *,
    param_values: Dict[str, object] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Simulate a quantum circuit and return the density matrix.

    Handles both unitary gates and noise channels (via Kraus operators).
    Preserves torch autograd for differentiable parameters.

    Args:
        qc (QuantumCircuit): Quantum circuit.
        param_values (*Dict[str, object] | None*): Parameter name to value mapping. Defaults to ``None``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Density matrix as a torch tensor of shape (2^n, 2^n), with requires_grad=True if needed.
    """
    device = auto_sim_device(device)
    nqubits = int(getattr(qc, "nqubits", 0) or 0)

    dtype = torch.complex128

    rho = torch.zeros(2**nqubits, 2**nqubits, dtype=dtype, device=device, requires_grad=False)
    rho[0, 0] = 1.0
    rho = rho.reshape([2] * (2 * nqubits))

    gates = list(getattr(qc, "gates", []))

    # Large differentiable runs: segment the gate sequence and recompute in
    # backward (gradient checkpointing) -> O(sqrt(n_gates)) saved density
    # matrices instead of O(n_gates).  Sampling / inference is untouched.
    if grad_checkpoint_enabled(nqubits, param_values) and len(gates) > 1:
        chunk = max(1, int(len(gates) ** 0.5))
        for i in range(0, len(gates), chunk):
            rho = checkpoint(
                _apply_dm_gate_block,
                rho, gates[i:i + chunk], qc, param_values, nqubits, dtype, device,
                use_reentrant=False,
            )
    else:
        rho = _apply_dm_gate_block(
            rho, gates, qc, param_values, nqubits, dtype, device,
        )

    rho = rho.reshape(2**nqubits, 2**nqubits)
    return rho


def _apply_dm_gate_block(rho, gates, qc, param_values, nqubits, dtype, device):
    """Apply a contiguous block of gates / channels to the DM tensor ``rho``.

    Operates on the ``(2,)*2n`` reshaped density matrix and returns it in the
    same shape.  Extracted from :func:`simulate_density_matrix` so a block can
    be wrapped in :func:`torch.utils.checkpoint.checkpoint`.
    """
    for gate_info in gates:
        gate = gate_info[0]

        if gate in one_qubit_gates_available:
            qubit = gate_info[1]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            rho = _apply_unitary_to_dm(rho, mat, [qubit], nqubits)

        elif gate in two_qubit_gates_available:
            q0, q1 = gate_info[1], gate_info[2]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            rho = _apply_unitary_to_dm(rho, mat, [q0, q1], nqubits)

        elif gate in three_qubit_gates_available:
            q0, q1, q2 = gate_info[1], gate_info[2], gate_info[3]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            rho = _apply_unitary_to_dm(rho, mat, [q0, q1, q2], nqubits)

        elif gate in one_qubit_parameter_gates_available:
            qubit = gate_info[-1]
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-1]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            rho = _apply_unitary_to_dm(rho, mat, [qubit], nqubits)

        elif gate in two_qubit_parameter_gates_available:
            q0, q1 = gate_info[-2], gate_info[-1]
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-2]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            rho = _apply_unitary_to_dm(rho, mat, [q0, q1], nqubits)

        elif gate in single_qubit_noise_channel_gates_available:
            qubit = gate_info[2]
            kraus_ops = get_kraus_ops(gate, float(gate_info[1]), dtype=dtype, device=device)
            rho = _apply_kraus_to_dm(rho, kraus_ops, [qubit], nqubits)

        elif gate in two_qubit_noise_channel_gates_available:
            q0, q1 = gate_info[2], gate_info[3]
            kraus_ops = get_kraus_ops(gate, float(gate_info[1]), dtype=dtype, device=device)
            rho = _apply_kraus_to_dm(rho, kraus_ops, [q0, q1], nqubits)

        elif gate in ['reset']:
            raise NotImplementedError(
                "The density-matrix simulator does not support the 'reset' operation."
            )

        elif gate in ['barrier', 'delay', 'measure']:
            continue

        else:
            raise ValueError(f"Unsupported gate for DM simulator: {gate}")

    return rho


def simulate_noisy_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    param_values: Dict[str, object] | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Simulate a noisy quantum circuit and return measurement counts.

    Samples from the diagonal of the final density matrix.

    Args:
        qc (QuantumCircuit): Quantum circuit.
        shots (int): Number of measurement shots.
        seed (*Optional[int]*): Random seed for reproducibility. Defaults to ``None``.
        param_values (*Dict[str, object] | None*): Parameter name to value mapping. Defaults to ``None``.
        device (*torch.device | str | None*): Torch device. Defaults to ``None``.

    Returns:
        ``Dict[str, int]`` mapping bitstrings to occurrence counts.
    """
    rho = simulate_density_matrix(qc, param_values=param_values, device=device)
    num_qubits = int(qc.nqubits)

    probs = torch.diag(rho).real.clamp(min=0.0)
    probs = probs / probs.sum()

    # Sample using torch.multinomial
    generator = torch.Generator(device=rho.device)
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.seed()  # a fresh Generator is otherwise deterministic
    samples = torch.multinomial(probs, num_samples=shots, replacement=True, generator=generator)
    counts = torch.bincount(samples, minlength=probs.numel())

    out: Dict[str, int] = {}
    for idx, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        bits = format(idx, f"0{num_qubits}b")
        out[bits] = int(count)
    return out


def apply_pauli_left_dm(
    state,
    pauli: str,
    *,
    num_qubits: int,
) -> torch.Tensor:
    """Return ``P rho`` (as a ``(2^n, 2^n)`` matrix) for a Pauli string.
    """
    from ..core.observables import pauli_basis_pattern

    rho = state.reshape(2**num_qubits, 2**num_qubits)
    dtype = rho.dtype
    device = rho.device

    rho_tensor = rho.reshape([2] * (2 * num_qubits))
    pattern = pauli_basis_pattern(pauli, num_qubits=num_qubits)
    for qubit, op in enumerate(pattern):
        if op == 'I':
            continue
        P_qubit = single_pauli(op, dtype=dtype, device=device)
        # Apply P_qubit to row-axis (qubit) of the DM tensor
        rho_tensor = _apply_mat_left(rho_tensor, P_qubit, [qubit], 2 * num_qubits)

    return rho_tensor.reshape(2**num_qubits, 2**num_qubits)


def expectation_pauli_dm(
    state,
    pauli: str,
    *,
    num_qubits: int,
) -> torch.Tensor:
    """Return trace(P ρ) for a Pauli string on density matrix state.

    Args:
        state: Density matrix tensor of shape (2^n, 2^n) or reshaped to (2,)*2n.
        pauli (*str*): Pauli string (e.g., ``'XZI'`` or ``'Z0Z1'``).
        num_qubits (*int*): Number of qubits.

    Returns:
        ``torch.Tensor`` scalar expectation value (real for Hermitian Pauli).
    """
    return torch.trace(
        apply_pauli_left_dm(state, pauli, num_qubits=num_qubits)
    ).real


def sample_probabilities_dm(
    state,
    samples,
):
    """Return probabilities for given sample vectors from a density matrix.

    For each computational basis state |i⟩, computes P(i) = ⟨i|ρ|i⟩.

    Args:
        state: Density matrix tensor of shape (2^n, 2^n).
        samples: ``(N, n_qubits)`` integer tensor or array with entries 0/1,
            big-endian (column 0 = qubit 0).

    Returns:
        1-D tensor of length *N* with ``P(sample_i) = ⟨sample_i|ρ|sample_i⟩.real``.
    """
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.long, device=state.device)
    else:
        samples = samples.to(device=state.device, dtype=torch.long)

    n_qubits = samples.shape[1]

    # Convert samples to basis state indices (big-endian)
    weights = (2 ** torch.arange(n_qubits - 1, -1, -1, device=samples.device)).unsqueeze(0)
    indices = (samples * weights).sum(dim=1)

    # Extract diagonal elements (probabilities)
    probs = torch.diag(state)[indices]
    return probs.real


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names: Sequence[str],
    hamiltonian,
    device: torch.device | str | None = None,
):
    """Evaluate Hamiltonian energy from a noisy symbolic circuit (DM backend).

    Args:
        symbolic_qc (*QuantumCircuit*): Symbolic (unbound) quantum circuit with noise.
        params (*torch.Tensor*): 1-D differentiable parameter tensor.
        param_names (*Sequence[str]*): Names of variational parameters.
        hamiltonian (*Sequence[Tuple[float, str]]*): Hamiltonian as coefficient–Pauli pairs.
        device (*torch.device | str | None*): Torch device. Defaults to ``None``.

    Returns:
        Tuple of ``(energy, expectations)`` where energy is differentiable scalar.
    """
    from .common import build_param_values_from_tensor

    nqubits = int(getattr(symbolic_qc, "nqubits", 0) or 0)
    sim_device = auto_sim_device(device)
    if hasattr(params, 'device') and params.device != sim_device:
        params = params.to(sim_device)

    param_values = build_param_values_from_tensor(params=params, param_names=param_names)
    state = simulate_density_matrix(symbolic_qc, param_values=param_values, device=sim_device)

    dim = 2**nqubits
    h_rho = torch.zeros((dim, dim), dtype=state.dtype, device=sim_device)
    expectations: Dict[str, float] = {}
    for coeff, obs in hamiltonian:
        p_rho = apply_pauli_left_dm(state, obs, num_qubits=nqubits)
        h_rho = h_rho + float(coeff) * p_rho
        expectations[obs] = float(torch.trace(p_rho).real.detach().cpu().item())
    energy = torch.trace(h_rho).real
    return energy, expectations
