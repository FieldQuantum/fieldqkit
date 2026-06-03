"""
Torch-based MPS simulator aligned with statevector interfaces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
if hasattr(torch.backends, "opt_einsum"):
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
    auto_sim_device,
    build_param_values_from_tensor,
    materialize_gate_matrix,
    resolve_param,
    single_pauli,
)

MAX_BOND_DIM: int | None = 256
SVD_EPS: float = 1e-12  # Singular values below this are truncated


class ComplexSVD(torch.autograd.Function):
    """Custom autograd function for differentiable SVD of complex-valued tensors.

    Provides numerically stable forward and backward passes for ``torch.linalg.svd``
    on complex matrices, enabling gradient-based optimization through MPS bond truncation.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        """Compute SVD and save tensors for backward pass.

        Args:
            ctx: Autograd context for saving tensors.
            input_tensor: Complex-valued input matrix.

        Returns:
            Tuple of (U, S, Vh) singular value decomposition.
        """
        u, s, vh = torch.linalg.svd(input_tensor, full_matrices=False)
        ctx.save_for_backward(input_tensor, u, s, vh)
        return u, s, vh

    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_vh):
        """Compute gradient of the SVD with respect to the input matrix.

        Uses the method from Wan & Zhang (2019) for complex-valued SVD differentiation
        with safe handling of degenerate singular values.

        Args:
            ctx: Autograd context with saved tensors.
            grad_u: Gradient w.r.t. U factor.
            grad_s: Gradient w.r.t. singular values.
            grad_vh: Gradient w.r.t. Vh factor.

        Returns:
            Gradient w.r.t. the input matrix.
        """
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
    """Differentiable SVD for complex tensors via ``ComplexSVD`` custom autograd.

    Args:
        x (*torch.Tensor*): Complex-valued input matrix.

    Returns:
        Tuple of (U, S, Vh).
    """
    return ComplexSVD.apply(x)


def _mps_zero_state(num_qubits: int, *, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """Initialize an MPS in the computational |0...0⟩ product state.

    Each site tensor has shape ``(1, 2, 1)`` with amplitude 1 on the |0⟩ component,
    representing a bond-dimension-1 product state.

    Args:
        num_qubits (*int*): Number of qubits.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        List of ``num_qubits`` site tensors, each of shape ``(1, 2, 1)``.
    """
    if num_qubits <= 0:
        return []
    out: List[torch.Tensor] = []
    t0 = torch.zeros((1, 2, 1), dtype=dtype, device=device)
    t0[0, 0, 0] = 1.0 + 0.0j
    for _ in range(num_qubits):
        out.append(t0.clone())
    return out


def _identity2(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the 2×2 identity matrix as a torch tensor.

    Args:
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        Identity matrix of shape ``(2, 2)``.
    """
    return torch.eye(2, dtype=dtype, device=device)


def _projector(bit: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the single-qubit projector |bit⟩⟨bit| as a 2×2 matrix.

    Args:
        bit (*int*): Computational basis index (0 or 1) to project onto.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        Projector matrix of shape ``(2, 2)`` with a single 1 at ``[bit, bit]``.
    """
    p = torch.zeros((2, 2), dtype=dtype, device=device)
    p[bit, bit] = 1.0
    return p


def _apply_one_qubit_gate(mps: List[torch.Tensor], qubit: int, gate: torch.Tensor) -> None:
    """Apply a single-qubit gate to an MPS by contracting it into the site tensor.

    Replaces ``mps[qubit]`` in-place via ``einsum('lpr,sp->lsr', A, gate)``.

    Args:
        mps (*List[torch.Tensor]*): MPS site tensor list, each of shape ``(bond_l, 2, bond_r)``.
        qubit (*int*): Target qubit index.
        gate (*torch.Tensor*): Unitary gate matrix of shape ``(2, 2)``.
    """
    mps[qubit] = torch.einsum("lpr,sp->lsr", mps[qubit], gate)


def _split_two_site_theta(
    mps: List[torch.Tensor],
    left: int,
    *,
    max_bond_dim: int | None,
    direction: str = 'left',
) -> None:
    """Contract two adjacent MPS sites and re-split via truncated SVD.

    Merges ``mps[left]`` and ``mps[left+1]`` into a two-site tensor, performs SVD
    with optional bond-dimension truncation, and writes the factors back. When
    ``direction='left'`` the singular values are absorbed into the right tensor;
    when ``'right'`` they are absorbed into the left tensor.

    Args:
        mps (*List[torch.Tensor]*): MPS site tensor list, each of shape ``(bond_l, 2, bond_r)``.
        left (*int*): Index of the left site in the pair to split.
        max_bond_dim (*int | None*): Maximum bond dimension to keep after truncation.
            ``None`` means no truncation.
        direction (*str*): Where to absorb singular values: ``'left'`` (into right tensor)
            or ``'right'`` (into left tensor). Defaults to ``'left'``.

    Raises:
        ValueError: If *direction* is not ``'left'`` or ``'right'``.
    """
    a = mps[left]
    b = mps[left + 1]

    theta = torch.einsum("lpa,aqr->lpqr", a, b)

    dl, _, _, dr = theta.shape
    mat = theta.reshape(dl * 2, 2 * dr)
    u, s, vh = complex_svd(mat)

    chi = s.shape[0] if max_bond_dim is None else min(int(max_bond_dim), int(s.shape[0]))
    # Precision-based truncation: drop singular values below SVD_EPS
    s_real = s.detach().real
    n_significant = int((s_real > SVD_EPS).sum())
    if n_significant > 0:
        chi = min(chi, n_significant)
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
    """Build an identity MPO tensor for bridging non-adjacent gate qubits.

    Returns a tensor of shape ``(bond_dim, 2, bond_dim, 2)`` that acts as the
    identity on the physical leg while passing the virtual bond through unchanged.

    Args:
        bond_dim (*int*): Virtual bond dimension of adjacent MPO tensors.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        Identity bridge tensor of shape ``(bond_dim, 2, bond_dim, 2)``.
    """
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
    """Decompose a two-qubit unitary into a two-site MPO via SVD.

    Reshapes the 4×4 gate into ``(out0, in0) × (out1, in1)`` and performs a
    (possibly truncated) SVD to obtain left and right MPO tensors with shapes
    ``(1, 2, chi, 2)`` and ``(chi, 2, 1, 2)`` respectively.

    Args:
        gate_2q (*torch.Tensor*): Two-qubit unitary matrix of shape ``(4, 4)``.
        max_bond_dim (*int | None*): Maximum bond dimension to keep. ``None`` keeps all.

    Returns:
        Tuple ``(left, right)`` of MPO site tensors with shape
        ``[Dl, pout, Dr, pin]``.
    """
    # gate_2q rows/cols are ordered as |out0 out1><in0 in1|.
    g = gate_2q.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = complex_svd(g)
    chi = s.shape[0] if max_bond_dim is None else min(int(max_bond_dim), int(s.shape[0]))
    s_real = s.detach().real
    n_significant = int((s_real > SVD_EPS).sum())
    if n_significant > 0:
        chi = min(chi, n_significant)
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
    """Decompose a multi-qubit unitary into an MPO chain via sequential SVD.

    Splits the ``2^n × 2^n`` unitary into ``num_sites`` local tensors each of
    shape ``[Dl, pout, Dr, pin]`` by peeling off one site at a time from the left
    using truncated SVD.

    Args:
        unitary (*torch.Tensor*): Unitary matrix of shape ``(2**num_sites, 2**num_sites)``.
        num_sites (*int*): Number of qubit sites the gate acts on.
        max_bond_dim (*int | None*): Maximum bond dimension to keep. ``None`` keeps all.

    Returns:
        List of ``num_sites`` MPO tensors with shape ``[Dl, pout, Dr, pin]``.
    """
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
        s_real = s.detach().real
        n_significant = int((s_real > SVD_EPS).sum())
        if n_significant > 0:
            chi = min(chi, n_significant)
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
    """Expand a sparse gate MPO on non-contiguous qubits into a contiguous span.

    Inserts identity bridge tensors at qubit positions between the acted qubits
    so the resulting MPO covers the full range ``[qmin, qmax]`` contiguously.

    Args:
        acted_mpo (*Sequence[torch.Tensor]*): MPO tensors for the acted qubits only.
        acted_qubits_sorted (*Sequence[int]*): Sorted list of qubit indices the gate acts on.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        Tuple of ``(full_mpo_span, qmin, qmax)`` where *full_mpo_span* covers
        all sites from ``qmin`` to ``qmax`` inclusive.

    Raises:
        ValueError: If *acted_qubits_sorted* is empty.
    """
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
    """Apply a contiguous MPO span to the corresponding MPS segment in-place.

    For each site, contracts the MPO tensor ``W[a, p_out, b, p_in]`` with the
    MPS tensor ``A[l, p_in, r]``, merging virtual bond dimensions to produce
    ``B[(l*a), p_out, (r*b)]``.

    Args:
        mps (*List[torch.Tensor]*): MPS site tensor list (modified in-place).
        mpo_span (*Sequence[torch.Tensor]*): Contiguous MPO tensors to apply.
        start (*int*): Starting qubit index in the MPS for the first MPO tensor.
    """
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
    """Sweep-canonicalize an MPS segment with optional bond truncation.

    Performs sequential two-site SVD splits across ``[start, end)`` to bring
    the segment into left- or right-canonical form. When ``direction='left'``,
    sweeps left-to-right; when ``'right'``, sweeps right-to-left.

    Args:
        mps (*List[torch.Tensor]*): MPS site tensor list (modified in-place).
        start (*int*): First site index of the segment (inclusive).
        end (*int*): Last site index of the segment (exclusive for the sweep).
        direction (*str*): Sweep direction: ``'left'`` or ``'right'``.
        max_bond_dim (*int | None*): Maximum bond dimension per split. Defaults to ``None``.

    Raises:
        ValueError: If *direction* is not ``'left'`` or ``'right'``.
    """
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
    """Return the maximum right-bond dimension among MPS tensors in ``[start, end)``.

    Args:
        mps (*Sequence[torch.Tensor]*): MPS or MPO site tensor list.
        start (*int*): First site index (inclusive).
        end (*int*): Last site index (exclusive).

    Returns:
        Maximum bond dimension found in the span, or 1 if the span is empty.
    """
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
    """Relocate canonical center to the left edge of a dirty span.

    Args:
        mps (*List[torch.Tensor]*): MPS site tensor list (modified in-place).
        center (*int*): Current canonical center position.
        start (*int*): Left edge of the target span.
        end (*int*): Right edge of the target span.

    Returns:
        New canonical center position (always *start*).
    """
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


def _permute_gate_to_sorted_qubits(
    gate_matrix: torch.Tensor,
    qubits: Sequence[int],
) -> torch.Tensor:
    """Permute a k-qubit unitary so axis ``i`` corresponds to the i-th smallest qubit.

    The input ``gate_matrix`` is in user order: matrix axis ``i`` corresponds to
    ``qubits[i]``. The MPO decomposition assigns site ``i`` to the i-th smallest
    qubit, so when ``qubits`` is not already sorted ascending, the unitary needs
    to be re-indexed accordingly to preserve gate semantics.

    Args:
        gate_matrix (*torch.Tensor*): Unitary of shape ``(2**k, 2**k)`` in user
            qubit order.
        qubits (*Sequence[int]*): Target qubit indices in user order.

    Returns:
        Unitary of shape ``(2**k, 2**k)`` whose axes are in ascending qubit
        order. If ``qubits`` is already ascending, returns ``gate_matrix``
        unchanged.
    """
    k = len(qubits)
    if k <= 1:
        return gate_matrix
    order = sorted(range(k), key=lambda i: int(qubits[i]))
    if order == list(range(k)):
        return gate_matrix
    perm = list(order) + [k + o for o in order]
    return (
        gate_matrix.reshape([2] * (2 * k))
        .permute(*perm)
        .reshape(2**k, 2**k)
        .contiguous()
    )


def _apply_k_qubit_gate_with_mpo(
    mps: List[torch.Tensor],
    qubits: Sequence[int],
    gate_matrix: torch.Tensor,
    *,
    max_bond_dim: int | None,
) -> Tuple[int, int] | None:
    """Apply a k-qubit gate to the MPS by decomposing it into an MPO and contracting.

    For single-qubit gates, directly updates the site tensor. For multi-qubit
    gates, decomposes the unitary into an MPO (filling identity bridges for
    non-contiguous qubits) and contracts it into the MPS segment. When
    ``qubits`` is not already in ascending order, the gate matrix is permuted
    to match the sorted qubit layout used by the MPO so the applied unitary
    matches the user-specified qubit-to-axis mapping.

    Args:
        mps (*List[torch.Tensor]*): MPS site tensor list (modified in-place).
        qubits (*Sequence[int]*): Target qubit indices for the gate, in the
            order matching the gate matrix axes.
        gate_matrix (*torch.Tensor*): Unitary gate matrix of shape ``(2**k, 2**k)``.
        max_bond_dim (*int | None*): Maximum bond dimension for MPO decomposition.

    Returns:
        Tuple ``(qmin, qmax)`` of the affected qubit span, or ``None`` for
        single-qubit gates.

    Raises:
        ValueError: If gate qubits are not distinct.
    """

    qubits_int = [int(q) for q in qubits]
    if len(set(qubits_int)) != len(qubits_int):
        raise ValueError("gate qubits must be distinct")
    if not qubits_int:
        return None
    acted = sorted(qubits_int)
    if len(acted) == 1:
        _apply_one_qubit_gate(mps, acted[0], gate_matrix.reshape(2, 2))
        return None

    sorted_gate_matrix = _permute_gate_to_sorted_qubits(gate_matrix, qubits_int)

    if len(acted) == 2:
        left_mpo, right_mpo = _two_qubit_unitary_to_mpo(sorted_gate_matrix, max_bond_dim=max_bond_dim)
        acted_mpo = [left_mpo, right_mpo]
    else:
        acted_mpo = _unitary_to_mpo(sorted_gate_matrix, len(acted), max_bond_dim=max_bond_dim)

    mpo_span, qmin, _ = _expand_sparse_gate_mpo_to_span(
        acted_mpo,
        acted,
        dtype=sorted_gate_matrix.dtype,
        device=sorted_gate_matrix.device,
    )
    _apply_mpo_to_segment(
        mps,
        mpo_span,
        start=qmin,
    )
    return qmin, int(acted[-1])


def _expectation_with_local_ops(mps: Sequence[torch.Tensor], ops: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute ⟨ψ|O₁⊗O₂⊗…⊗Oₙ|ψ⟩ for an MPS state and per-site operators.

    Contracts the MPS with its conjugate, inserting local operators ``ops[i]``
    on each physical leg, sweeping left to right via transfer matrices.

    Args:
        mps (*Sequence[torch.Tensor]*): MPS site tensor list.
        ops (*Sequence[torch.Tensor]*): Per-site 2×2 operator matrices (one per qubit).

    Returns:
        Scalar tensor containing the expectation value.
    """
    env = torch.ones((1, 1), dtype=mps[0].dtype, device=mps[0].device)
    for t, op in zip(mps, ops):
        env = torch.einsum("ab,api,bqj,pq->ij", env, torch.conj(t), t, op)
    return env.squeeze()


def _norm2_mps(mps: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute the squared norm ⟨ψ|ψ⟩ of an MPS state.

    Uses identity operators on every site to evaluate the full overlap.

    Args:
        mps (*Sequence[torch.Tensor]*): MPS site tensor list.

    Returns:
        Real scalar tensor with the squared norm.
    """
    dtype = mps[0].dtype
    device = mps[0].device
    ops = [_identity2(dtype=dtype, device=device) for _ in range(len(mps))]
    return _expectation_with_local_ops(mps, ops).real


def _compute_right_envs(mps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Precompute right environment (partial overlap) tensors for MPS sampling.

    Sweeps from right to left, contracting each site tensor with its conjugate
    to build the partial norms ``right[i]`` = ⟨ψ_{i..n-1}|ψ_{i..n-1}⟩ boundary.

    Args:
        mps (*Sequence[torch.Tensor]*): MPS site tensor list.

    Returns:
        List of ``n+1`` right-environment matrices, where ``right[n]`` is ``[[1]]``.
    """
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
    """Draw computational-basis measurement samples from an MPS state.

    Uses a sequential site-by-site sampling strategy: at each qubit, computes
    the marginal probability for |0⟩ and |1⟩ from left/right environments,
    samples a bit, and updates the left environment conditioned on the outcome.

    Args:
        mps (*Sequence[torch.Tensor]*): MPS site tensor list.
        shots (*int*): Number of independent measurement samples to draw.
        seed (*int | None*): Random seed for reproducibility.

    Returns:
        List of ``shots`` bitstrings, each a list of ``n`` integers (0 or 1).
    """
    n = len(mps)
    if n == 0:
        return [[] for _ in range(shots)]

    device = mps[0].device
    dtype = mps[0].dtype
    right = _compute_right_envs(mps)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.seed()  # a fresh Generator is otherwise deterministic

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

        left0 = torch.einsum("sab,api,bqj,pq->sij", left, t, torch.conj(t), proj0)
        left1 = torch.einsum("sab,api,bqj,pq->sij", left, t, torch.conj(t), proj1)
        selector = bits.view(-1, 1, 1) == 0
        left = torch.where(selector, left0, left1)

    return bits_all.detach().cpu().tolist()


def sample_probabilities(
    mps: Sequence[torch.Tensor],
    samples,
) -> torch.Tensor:
    """Compute probabilities for given sample vectors from an MPS state.

    Args:
        mps: List of MPS site tensors.
        samples: ``(N, n_qubits)`` integer tensor or array with entries 0/1,
            big-endian (column 0 = qubit 0).

    Returns:
        1-D tensor of length *N* with probabilities (differentiable).
    """
    n = len(mps)
    device = mps[0].device
    dtype = mps[0].dtype

    if not isinstance(samples, torch.Tensor):
        bits = torch.tensor(samples, dtype=torch.long, device=device)
    else:
        bits = samples.to(device=device, dtype=torch.long)
    num_bs = bits.shape[0]

    # Batched MPS amplitude contraction:
    env = torch.ones((num_bs, 1), dtype=dtype, device=device)
    for i in range(n):
        t = mps[i]  # (bond_l, 2, bond_r)
        t_selected = t[:, bits[:, i], :]  # (bond_l, num_bs, bond_r)
        t_selected = t_selected.permute(1, 0, 2)  # (num_bs, bond_l, bond_r)
        env = torch.einsum("ia,iab->ib", env, t_selected)

    amplitudes = env.squeeze(-1).squeeze(-1)  # (num_bs,)
    return amplitudes.real ** 2 + amplitudes.imag ** 2


def simulate_mps(
    qc: QuantumCircuit,
    *,
    param_values: Dict[str, object] | None = None,
    max_bond_dim: int | None = MAX_BOND_DIM,
    device: torch.device | str | None = None,
) -> List[torch.Tensor]:
    """Simulate a quantum circuit using the Matrix Product State (MPS) method.

    Initializes an MPS in the |0...0⟩ state and applies each gate in circuit
    order. Multi-qubit gates are decomposed into MPO form and contracted into
    the MPS; bond dimensions are compressed on-the-fly when they exceed
    *max_bond_dim*.

    Args:
        qc (*QuantumCircuit*): Quantum circuit to simulate.
        param_values (*Dict[str, object] | None*): Symbolic parameter name-to-value map. Defaults to ``None``.
        max_bond_dim (*int | None*): Maximum MPS bond dimension for truncation. ``None`` keeps all. Defaults to ``MAX_BOND_DIM``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        List of MPS site tensors, each of shape ``(bond_l, 2, bond_r)``.

    Raises:
        ValueError: If a gate in the circuit is not supported by the simulator.
    """
    num_qubits = int(qc.nqubits)
    if num_qubits <= 0:
        return []

    dtype = torch.complex128
    sim_device = auto_sim_device(device)
    mps = _mps_zero_state(num_qubits, dtype=dtype, device=sim_device)
    dirty_start: int | None = None
    dirty_end: int | None = None
    canon_center: int = 0

    def _mark_dirty(start: int, end: int) -> None:
        """Expand the dirty (uncompressed) qubit span to include ``[start, end]``.

        Args:
            start (*int*): Left-most qubit index of the newly dirtied region.
            end (*int*): Right-most qubit index of the newly dirtied region.
        """
        nonlocal dirty_start, dirty_end
        if dirty_start is None:
            dirty_start = int(start)
            dirty_end = int(end)
            return
        dirty_start = min(dirty_start, int(start))
        dirty_end = max(dirty_end, int(end))

    def _maybe_compress_dirty_span() -> None:
        """Compress the dirty MPS segment if its bond dimension exceeds the limit.

        Moves the canonical center to the left edge of the dirty span, then
        performs a truncating left-to-right canonicalization sweep if any bond
        exceeds *max_bond_dim*. Resets the dirty span afterward.
        """
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
                raise NotImplementedError(
                    "The MPS simulator does not support the 'reset' operation."
                )
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
    max_bond_dim: int | None = MAX_BOND_DIM,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Simulate counts with MPS backend, matching statevector bitstring order.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        shots (*int*): Number of measurement shots.
        seed (*Optional[int]*): Random seed for reproducibility. Defaults to ``None``.
        param_values (*Dict[str, object] | None*): Param values (``Dict[str, object] | None``). Defaults to ``None``.
        max_bond_dim (*int | None*): Maximum MPS bond dimension. Defaults to ``MAX_BOND_DIM``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``Dict[str, int]`` mapping bitstrings to their occurrence counts.
    """
    with torch.no_grad():
        mps = simulate_mps(qc, param_values=param_values, max_bond_dim=max_bond_dim, device=device)
        samples = _sample_bits_from_mps(mps, int(shots), seed=seed)

    out: Dict[str, int] = {}
    for bits in samples:
        bitstr = "".join(str(int(b)) for b in bits)
        out[bitstr] = out.get(bitstr, 0) + 1
    return out


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> for a Pauli string, consistent with statevector API.

    Args:
        state: MPS site tensor list (``List[torch.Tensor]``).
        pauli (*str*): Pauli string (e.g. ``'XZI'``).
        num_qubits (*int*): Number of qubits.

    Returns:
        Scalar expectation value ``<psi|P|psi> / <psi|psi>``.

    Raises:
        TypeError: If *state* is not a non-empty list of MPS tensors.
    """
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
    raw = _expectation_with_local_ops(state, ops)
    norm2 = _norm2_mps(state)
    return raw / norm2


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names,
    hamiltonian,
    max_bond_dim: int | None = MAX_BOND_DIM,
    device: torch.device | str | None = None,
):
    """Evaluate Hamiltonian energy from symbolic circuit using MPS contraction.

    Args:
        symbolic_qc (*QuantumCircuit*): Symbolic (unbound) quantum circuit.
        params: Parameter values.
        param_names: Names of variational parameters.
        hamiltonian: Target Hamiltonian as ``List[Tuple[float, str]]``.
        max_bond_dim (*int | None*): Maximum MPS bond dimension. Defaults to ``MAX_BOND_DIM``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``(energy, expectations)`` tuple.

    Raises:
        TypeError: params must be a torch.Tensor
    """
    if not isinstance(params, torch.Tensor):
        raise TypeError("params must be a torch.Tensor")

    selected_device = auto_sim_device(device)
    if params.device != selected_device:
        params = params.to(selected_device)

    param_values = build_param_values_from_tensor(params=params, param_names=param_names)
    mps = simulate_mps(symbolic_qc, param_values=param_values, max_bond_dim=max_bond_dim, device=selected_device)

    energy = torch.zeros((), dtype=params.dtype, device=params.device)
    expectations: dict[str, float] = {}
    n = int(symbolic_qc.nqubits)
    for coeff, obs in hamiltonian:
        exp_val = expectation_pauli(mps, obs, num_qubits=n).real
        energy = energy + float(coeff) * exp_val
        expectations[obs] = float(exp_val.detach().cpu().item())
    return energy, expectations
