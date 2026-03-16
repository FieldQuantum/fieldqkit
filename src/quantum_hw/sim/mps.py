"""
Torch-based MPS simulator aligned with statevector interfaces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
torch.backends.opt_einsum.enabled = True
torch.backends.opt_einsum.strategy = "auto"

from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from ..core.observables import pauli_basis_pattern
from .common import (
    build_param_values_from_tensor,
    materialize_gate_matrix,
    resolve_param,
    single_pauli,
)

MAX_BOND_DIM: int | None = 256


class ComplexSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        u, s, vh = torch.linalg.svd(input_tensor, full_matrices=False)
        ctx.save_for_backward(input_tensor, u, s, vh)
        return u, s, vh

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_vh):
        input_tensor, u, s, vh = ctx.saved_tensors
        m, n = input_tensor.shape
        k = s.shape[-1]

        complex_dtype = input_tensor.dtype

        # Build singular value matrices and safe inverse.
        s_mat = torch.diag_embed(s.to(dtype=complex_dtype))
        s_inv = torch.where(torch.abs(s_mat) > 1e-12, 1.0 / s_mat, torch.zeros_like(s_mat))

        # Denominator for anti-Hermitian projection terms.
        f = s.unsqueeze(-2) ** 2 - s.unsqueeze(-1) ** 2
        f_inv = torch.where(torch.abs(f) > 1e-12, 1.0 / f, torch.zeros_like(f))

        ut_du = u.transpose(-2, -1).conj() @ grad_u
        vt_dv = vh @ grad_vh.transpose(-2, -1).conj()

        su = f_inv * (ut_du - ut_du.transpose(-2, -1).conj())
        sv = f_inv * (vt_dv - vt_dv.transpose(-2, -1).conj())
        l_term = torch.eye(k, dtype=complex_dtype, device=input_tensor.device) * (
            vt_dv.transpose(-2, -1).conj() - vt_dv
        )

        eye_m = torch.eye(m, dtype=complex_dtype, device=input_tensor.device)
        eye_n = torch.eye(n, dtype=complex_dtype, device=input_tensor.device)
        d_input = (
            u @ (torch.diag_embed(grad_s).to(dtype=complex_dtype) + su @ s_mat + s_mat @ sv + 0.5 * s_inv @ l_term) @ vh
            + (eye_m - u @ u.transpose(-2, -1).conj()) @ grad_u @ s_inv @ vh
            + u @ s_inv @ grad_vh @ (eye_n - vh.transpose(-2, -1).conj() @ vh)
        )
        return d_input


def complex_svd(x: torch.Tensor):
    return ComplexSVD.apply(x)


def _auto_sim_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)

    # Respect torch global default device when explicitly configured.
    if hasattr(torch, "get_default_device"):
        default = torch.device(torch.get_default_device())
        return default

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _mps_zero_state(num_qubits: int, *, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    if num_qubits <= 0:
        return []
    out: List[torch.Tensor] = []
    t0 = torch.zeros((1, 2, 1), dtype=dtype, device=device)
    t0[0, 0, 0] = 1.0 + 0.0j
    for _ in range(num_qubits):
        out.append(t0.clone())
    return out


def _identity2(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(2, dtype=dtype, device=device)


def _projector(bit: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    p = torch.zeros((2, 2), dtype=dtype, device=device)
    p[bit, bit] = 1.0
    return p


def _apply_one_qubit_gate(mps: List[torch.Tensor], qubit: int, gate: torch.Tensor) -> None:
    mps[qubit] = torch.einsum("lpr,sp->lsr", mps[qubit], gate)


def _split_two_site_theta(
    mps: List[torch.Tensor],
    left: int,
    *,
    max_bond_dim: int | None,
    direction: str = 'left',
) -> None:
    a = mps[left]
    b = mps[left + 1]

    theta = torch.einsum("lpa,aqr->lpqr", a, b)

    dl, _, _, dr = theta.shape
    mat = theta.reshape(dl * 2, 2 * dr)
    u, s, vh = complex_svd(mat)

    chi = s.shape[0] if max_bond_dim is None else min(int(max_bond_dim), int(s.shape[0]))
    u = u[:, :chi]
    s = s[:chi]
    vh = vh[:chi, :]

    if direction == 'left':
        left_t = u.reshape(dl, 2, chi)
        right_t = (torch.diag(s.to(dtype=vh.dtype)) @ vh).reshape(chi, 2, dr)
    elif direction == 'right':
        left_t = (u @ torch.diag(s.to(dtype=u.dtype))).reshape(dl, 2, chi)
        right_t = vh.reshape(chi, 2, dr)
    else:
        raise ValueError("direction must be 'left' or 'right'")
    mps[left] = left_t
    mps[left + 1] = right_t


def _identity_bridge_mpo_tensor(
    bond_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Shape [Dl, pout, Dr, pin], with W[a,p,a,p] = 1.
    t = torch.zeros((bond_dim, 2, bond_dim, 2), dtype=dtype, device=device)
    idx = torch.arange(bond_dim, device=device)
    t[idx, 0, idx, 0] = 1.0
    t[idx, 1, idx, 1] = 1.0
    return t


def _two_qubit_unitary_to_mpo(
    gate_2q: torch.Tensor,
    *,
    max_bond_dim: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # gate_2q rows/cols are ordered as |out0 out1><in0 in1|.
    g = gate_2q.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = complex_svd(g)
    chi = s.shape[0] if max_bond_dim is None else min(int(max_bond_dim), int(s.shape[0]))
    u = u[:, :chi]
    s = s[:chi]
    vh = vh[:chi, :]

    left = u.reshape(2, 2, chi).permute(0, 2, 1).unsqueeze(0)
    right = (torch.diag(s.to(dtype=vh.dtype)) @ vh).reshape(chi, 2, 2).unsqueeze(2)
    return left, right


def _unitary_to_mpo(
    unitary: torch.Tensor,
    num_sites: int,
    *,
    max_bond_dim: int | None,
) -> List[torch.Tensor]:
    if num_sites <= 0:
        return []
    if num_sites == 1:
        # [1, pout, 1, pin]
        return [unitary.reshape(1, 2, 1, 2)]

    cur = unitary.reshape(1, 2**num_sites, 1, 2**num_sites)
    mpo: List[torch.Tensor] = [None] * num_sites  # type: ignore[list-item]
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=unitary.real.dtype, device=unitary.device))

    for i in range(0, num_sites - 1):
        rest = 2 ** (num_sites - i - 1)
        cur = cur.reshape(-1, 2, rest, 1, 2, rest)
        cur = cur.permute(0, 1, 4, 2, 3, 5).reshape(-1, rest * rest)
        u, s, vh = complex_svd(cur)
        chi = s.shape[0] if max_bond_dim is None else min(int(max_bond_dim), int(s.shape[0]))
        u = u[:, :chi]
        s = s[:chi]
        vh = vh[:chi, :]
        mpo[i] = u.reshape(-1, 2, 2, chi).permute(0, 1, 3, 2) * sqrt2
        cur = (torch.diag(s.to(dtype=vh.dtype)) @ vh).reshape(chi, rest, 1, rest) / sqrt2

    mpo[-1] = cur
    return mpo


def _expand_sparse_gate_mpo_to_span(
    acted_mpo: Sequence[torch.Tensor],
    acted_qubits_sorted: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[torch.Tensor], int, int]:
    if not acted_qubits_sorted:
        raise ValueError("acted_qubits_sorted cannot be empty")
    qmin = int(acted_qubits_sorted[0])
    qmax = int(acted_qubits_sorted[-1])
    span_len = qmax - qmin + 1

    pos_to_mpo_index = {int(q): i for i, q in enumerate(acted_qubits_sorted)}
    full: List[torch.Tensor] = [None] * span_len  # type: ignore[list-item]

    for offset in range(span_len):
        q = qmin + offset
        if q in pos_to_mpo_index:
            full[offset] = acted_mpo[pos_to_mpo_index[q]]
            continue

        # Bridge tensor uses the bond that connects two acted MPO endpoints.
        left_acted = None
        right_acted = None
        for aq in acted_qubits_sorted:
            if aq < q:
                left_acted = aq
            elif aq > q:
                right_acted = aq
                break
        if left_acted is None or right_acted is None:
            raise ValueError("invalid sparse MPO span construction")

        left_idx = pos_to_mpo_index[int(left_acted)]
        bridge_dim = int(acted_mpo[left_idx].shape[2])
        full[offset] = _identity_bridge_mpo_tensor(bridge_dim, dtype=dtype, device=device)

    return full, qmin, qmax


def _apply_mpo_to_segment(
    mps: List[torch.Tensor],
    mpo_span: Sequence[torch.Tensor],
    *,
    start: int,
) -> None:
    # Apply local MPO tensor W[a,p_out,b,p_in] to A[l,p_in,r] -> B[(l,a),p_out,(r,b)].
    for i, w in enumerate(mpo_span):
        site = start + i
        a = mps[site]
        merged = torch.einsum("lpr,aqbp->laqrb", a, w)
        dl, da, pout, dr, db = merged.shape
        mps[site] = merged.reshape(dl * da, pout, dr * db)


def _canonicalize_segment(
    mps: List[torch.Tensor],
    *,
    start: int,
    end: int,
    direction: str,
    max_bond_dim: int | None = None,
) -> None:
    if end <= start:
        return

    if direction == "left":
        for site in range(start, end):
            _split_two_site_theta(mps, site, max_bond_dim=max_bond_dim, direction="left")            
    elif direction == "right":
        for site in range(end, start, -1):
            _split_two_site_theta(mps, site - 1, max_bond_dim=max_bond_dim, direction="right")
    else:
        raise ValueError("direction must be 'left' or 'right'")


def _max_bond_in_span(
    mps: Sequence[torch.Tensor],
    *,
    start: int,
    end: int,
) -> int:
    if end <= start:
        return 1
    return max(int(mps[bond].shape[2]) for bond in range(start, end))


def _move_canonical_center_to_span_left(
    mps: List[torch.Tensor],
    *,
    center: int,
    start: int,
    end: int,
) -> int:
    """Relocate canonical center to the left edge of a dirty span."""
    if end <= start:
        return int(start)

    c = int(center)
    l = int(start)
    r = int(end)

    if l <= c <= r:
        _canonicalize_segment(mps, start=l, end=r, direction="right")
        return l

    if c < l:
        _canonicalize_segment(mps, start=c, end=l, direction="left")
        _canonicalize_segment(mps, start=l, end=r, direction="right")
        return l

    # c > r
    _canonicalize_segment(mps, start=l, end=c, direction="right")
    return l


def _apply_k_qubit_gate_with_mpo(
    mps: List[torch.Tensor],
    qubits: Sequence[int],
    gate_matrix: torch.Tensor,
    *,
    max_bond_dim: int | None,
) -> Tuple[int, int] | None:
    acted = sorted(int(q) for q in qubits)
    if len(set(acted)) != len(acted):
        raise ValueError("gate qubits must be distinct")
    if not acted:
        return None
    if len(acted) == 1:
        _apply_one_qubit_gate(mps, acted[0], gate_matrix.reshape(2, 2))
        return None

    if len(acted) == 2:
        left_mpo, right_mpo = _two_qubit_unitary_to_mpo(gate_matrix, max_bond_dim=max_bond_dim)
        acted_mpo = [left_mpo, right_mpo]
    else:
        acted_mpo = _unitary_to_mpo(gate_matrix, len(acted), max_bond_dim=max_bond_dim)

    mpo_span, qmin, _ = _expand_sparse_gate_mpo_to_span(
        acted_mpo,
        acted,
        dtype=gate_matrix.dtype,
        device=gate_matrix.device,
    )
    _apply_mpo_to_segment(
        mps,
        mpo_span,
        start=qmin,
    )
    return qmin, int(acted[-1])


def _expectation_with_local_ops(mps: Sequence[torch.Tensor], ops: Sequence[torch.Tensor]) -> torch.Tensor:
    env = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)
    for t, op in zip(mps, ops):
        env = torch.einsum("ab,api,bqj,pq->ij", env, torch.conj(t), t, op)
    return env.squeeze()


def _norm2_mps(mps: Sequence[torch.Tensor]) -> torch.Tensor:
    dtype = mps[0].dtype
    device = mps[0].device
    ops = [_identity2(dtype=dtype, device=device) for _ in range(len(mps))]
    return _expectation_with_local_ops(mps, ops).real


def _apply_reset_mps(mps: List[torch.Tensor], qubit: int) -> None:
    t = mps[qubit].clone()
    t[:, 1, :] = 0.0
    mps[qubit] = t

    n2 = _norm2_mps(mps)
    if float(n2.detach().cpu().item()) > 0.0:
        # Normalization can be absorbed into one site tensor.
        mps[qubit] = mps[qubit] / torch.sqrt(n2)


def _compute_right_envs(mps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    n = len(mps)
    right: List[torch.Tensor] = [None] * (n + 1)  # type: ignore[list-item]
    right[n] = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)
    for i in range(n - 1, -1, -1):
        t = mps[i]
        right[i] = torch.einsum("lpr,mps,rs->lm", t, torch.conj(t), right[i + 1])
    return right


def _sample_bits_from_mps(
    mps: Sequence[torch.Tensor],
    shots: int,
    *,
    seed: int | None,
) -> List[List[int]]:
    n = len(mps)
    if n == 0:
        return [[] for _ in range(shots)]

    device = mps[0].device
    dtype = mps[0].dtype
    right = _compute_right_envs(mps)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))

    proj0 = _projector(0, dtype=dtype, device=device)
    proj1 = _projector(1, dtype=dtype, device=device)

    left = torch.ones((int(shots), 1, 1), dtype=dtype, device=device)
    bits_all = torch.empty((int(shots), n), dtype=torch.int64, device=device)
    default_probs = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)

    for i in range(n):
        t = mps[i]
        probs = torch.real(torch.einsum('iab,ajc,bjd,cd->ij', left, t, torch.conj(t), right[i + 1])).to(dtype=torch.float64)
        probs = torch.clamp(probs, min=0.0)

        sums = probs.sum(dim=1, keepdim=True)
        normalized = probs / torch.where(sums > 0.0, sums, torch.ones_like(sums))
        probs = torch.where(sums > 0.0, normalized, default_probs.expand_as(probs))

        bits = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(1)
        bits_all[:, i] = bits

        left0 = torch.einsum("sab,api,bqj,pq->sij", left, torch.conj(t), t, proj0)
        left1 = torch.einsum("sab,api,bqj,pq->sij", left, torch.conj(t), t, proj1)
        selector = bits.view(-1, 1, 1) == 0
        left = torch.where(selector, left0, left1)

    return bits_all.detach().cpu().tolist()


def _simulate_mps(
    qc: QuantumCircuit,
    *,
    param_values: Dict[str, object] | None = None,
    max_bond_dim: int | None = None,
    device: torch.device | str | None = None,
) -> List[torch.Tensor]:
    num_qubits = int(qc.nqubits)
    if num_qubits <= 0:
        return []

    dtype = torch.complex128
    sim_device = _auto_sim_device(device)
    mps = _mps_zero_state(num_qubits, dtype=dtype, device=sim_device)
    dirty_start: int | None = None
    dirty_end: int | None = None
    canon_center: int = 0

    def _mark_dirty(start: int, end: int) -> None:
        nonlocal dirty_start, dirty_end
        if dirty_start is None:
            dirty_start = int(start)
            dirty_end = int(end)
            return
        dirty_start = min(dirty_start, int(start))
        dirty_end = max(dirty_end, int(end))

    def _maybe_compress_dirty_span() -> None:
        nonlocal dirty_start, dirty_end, canon_center
        if dirty_start is None or dirty_end is None:
            return

        # Move canonical center to dirty_start
        canon_center = _move_canonical_center_to_span_left(
            mps,
            center=canon_center,
            start=dirty_start,
            end=dirty_end,
        )

        if max_bond_dim is None:
            dirty_start = None
            dirty_end = None
            return

        if _max_bond_in_span(mps, start=dirty_start, end=dirty_end) <= int(max_bond_dim):
            dirty_start = None
            dirty_end = None
            return

        # perform a left-to-right sweep over the dirty span to compress bonds.
        _canonicalize_segment(
            mps,
            start=dirty_start,
            end=dirty_end,
            direction="left",
            max_bond_dim=int(max_bond_dim),
        )
        canon_center = dirty_end
        dirty_start = None
        dirty_end = None

    for gate_info in qc.gates:
        gate = gate_info[0]
        if gate in functional_gates_available:
            if gate == "reset":
                _apply_reset_mps(mps, int(gate_info[1]))
            # barrier/measure/delay do not change state simulation here.
            continue

        if gate in one_qubit_gates_available:
            qubit = int(gate_info[1])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=sim_device)
            _apply_one_qubit_gate(mps, qubit, mat)
            continue

        if gate in one_qubit_parameter_gates_available:
            qubit = int(gate_info[-1])
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-1]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=sim_device)
            _apply_one_qubit_gate(mps, qubit, mat)
            continue

        if gate in two_qubit_gates_available:
            q0 = int(gate_info[1])
            q1 = int(gate_info[2])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=sim_device)
            span = _apply_k_qubit_gate_with_mpo(mps, [q0, q1], mat, max_bond_dim=max_bond_dim)
            if span is not None:
                _mark_dirty(*span)
                _maybe_compress_dirty_span()
            continue

        if gate in two_qubit_parameter_gates_available:
            q0 = int(gate_info[-2])
            q1 = int(gate_info[-1])
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-2]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=sim_device)
            span = _apply_k_qubit_gate_with_mpo(mps, [q0, q1], mat, max_bond_dim=max_bond_dim)
            if span is not None:
                _mark_dirty(*span)
                _maybe_compress_dirty_span()
            continue

        if gate in three_qubit_gates_available:
            qubits = [int(gate_info[1]), int(gate_info[2]), int(gate_info[3])]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=sim_device)
            span = _apply_k_qubit_gate_with_mpo(mps, qubits, mat, max_bond_dim=max_bond_dim)
            if span is not None:
                _mark_dirty(*span)
                _maybe_compress_dirty_span()
            continue

        raise ValueError(f"unsupported gate for simulator: {gate}")

    return mps


def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    param_values: Dict[str, object] | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Simulate counts with MPS backend, matching statevector bitstring order."""
    mps = _simulate_mps(qc, param_values=param_values, device=device)
    samples = _sample_bits_from_mps(mps, int(shots), seed=seed)

    out: Dict[str, int] = {}
    for bits in samples:
        bitstr = "".join(str(int(b)) for b in bits)[::-1]
        out[bitstr] = out.get(bitstr, 0) + 1
    return out


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> for a Pauli string, consistent with statevector API."""
    if not isinstance(state, list) or len(state) == 0:
        raise TypeError("MPS expectation_pauli expects a non-empty MPS tensor list when fallback is disabled")

    pattern = pauli_basis_pattern(pauli, num_qubits=num_qubits)
    dtype = state[0].dtype
    device = state[0].device
    ops = []
    for op in pattern:
        if op == "I":
            ops.append(_identity2(dtype=dtype, device=device))
        else:
            ops.append(single_pauli(op, dtype=dtype, device=device))
    return _expectation_with_local_ops(state, ops)


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names,
    hamiltonian,
    device: torch.device | str | None = None,
):
    """Evaluate Hamiltonian energy from symbolic circuit using MPS contraction."""
    if not isinstance(params, torch.Tensor):
        raise TypeError("params must be a torch.Tensor")

    selected_device = _auto_sim_device(device)
    if params.device != selected_device:
        params = params.to(selected_device)

    param_values = build_param_values_from_tensor(params=params, param_names=param_names)
    mps = _simulate_mps(symbolic_qc, param_values=param_values, device=selected_device)

    energy = torch.zeros((), dtype=params.dtype, device=params.device)
    expectations: dict[str, float] = {}
    n = int(symbolic_qc.nqubits)
    for coeff, obs in hamiltonian:
        exp_val = expectation_pauli(mps, obs, num_qubits=n).real
        energy = energy + float(coeff) * exp_val
        expectations[obs] = float(exp_val.detach().cpu().item())
    return energy, expectations
