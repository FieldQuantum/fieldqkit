"""Torch-based MPO process simulator.

This module simulates the circuit unitary as an MPO list with local tensor shape
[Dl, pout, Dr, pin]. It focuses on process evolution only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from .common import auto_sim_device, materialize_gate_matrix, resolve_param
from .mps import (
    _expand_sparse_gate_mpo_to_span,
    _max_bond_in_span,
    _two_qubit_unitary_to_mpo,
    _unitary_to_mpo,
    complex_svd,
)


def _identity_mpo(num_qubits: int, *, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """Create an identity MPO (trivial process) for the given number of qubits.

    Each site tensor is a reshaped 2×2 identity with shape ``(1, 2, 1, 2)``,
    representing ``[Dl, pout, Dr, pin]``.

    Args:
        num_qubits (*int*): Number of qubits.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        List of ``num_qubits`` identity MPO tensors.
    """
    if num_qubits <= 0:
        return []
    i2 = torch.eye(2, dtype=dtype, device=device).reshape(1, 2, 1, 2)
    return [i2.clone() for _ in range(num_qubits)]


def _apply_one_qubit_gate_to_mpo_tensor(t: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Left-multiply a single-qubit gate onto the output leg of an MPO tensor.

    Computes ``T'[l,s,r,q] = gate[s,p] · T[l,p,r,q]`` via einsum.

    Args:
        t (*torch.Tensor*): MPO site tensor of shape ``(Dl, pout, Dr, pin)``.
        gate (*torch.Tensor*): Single-qubit gate matrix of shape ``(2, 2)``.

    Returns:
        Updated MPO site tensor of shape ``(Dl, pout, Dr, pin)``.
    """
    # Left-multiply local output leg: T[l,p,r,q] <- gate[s,p] @ T[l,p,r,q]
    return torch.einsum("lprq,sp->lsrq", t, gate)


def _apply_gate_mpo_to_process_segment(
    mpo: List[torch.Tensor],
    gate_mpo_span: Sequence[torch.Tensor],
    *,
    start: int,
) -> None:
    """Compose a gate MPO into the process MPO on a contiguous segment.

    For each site, contracts gate tensor ``W[a,s,b,p]`` with process tensor
    ``T[l,p,r,q]`` and merges virtual bonds to yield ``T'[(l·a), s, (r·b), q]``.

    Args:
        mpo (*List[torch.Tensor]*): Process MPO tensor list (modified in-place).
        gate_mpo_span (*Sequence[torch.Tensor]*): Gate MPO tensors covering the segment.
        start (*int*): Starting qubit index in the process MPO.
    """
    # Compose gate MPO W[a,s,b,p] with process MPO T[l,p,r,q], yielding
    # T'[(l,a),s,(r,b),q].
    for i, w in enumerate(gate_mpo_span):
        site = start + i
        t = mpo[site]
        merged = torch.einsum("lprq,asbp->lasrbq", t, w)
        dl, da, pout, dr, db, pin = merged.shape
        mpo[site] = merged.reshape(dl * da, pout, dr * db, pin)


def _split_two_site_theta_mpo(
    mpo: List[torch.Tensor],
    left: int,
    *,
    max_bond_dim: int | None,
    direction: str,
) -> None:
    """Contract two adjacent MPO sites and re-split via truncated SVD.

    Merges ``mpo[left]`` and ``mpo[left+1]`` (each with shape ``[Dl, pout, Dr, pin]``)
    into a two-site tensor, performs SVD with optional bond truncation, and
    writes the factors back. Singular values are absorbed left or right depending
    on *direction*.

    Args:
        mpo (*List[torch.Tensor]*): Process MPO tensor list (modified in-place).
        left (*int*): Index of the left site in the pair.
        max_bond_dim (*int | None*): Maximum bond dimension after truncation.
        direction (*str*): ``'left'`` absorbs singular values into the right tensor;
            ``'right'`` absorbs them into the left tensor.

    Raises:
        ValueError: If *direction* is not ``'left'`` or ``'right'``.
    """
    a = mpo[left]
    b = mpo[left + 1]

    # a: [Dl, pout_l, Dm, pin_l], b: [Dm, pout_r, Dr, pin_r]
    theta = torch.einsum("lpmi,mqrj->lpiqrj", a, b)
    dl, _, _, _, dr, _ = theta.shape
    mat = theta.reshape(dl * 4, 4 * dr)
    u, s, vh = complex_svd(mat)

    chi = s.shape[0] if max_bond_dim is None else min(int(max_bond_dim), int(s.shape[0]))
    u = u[:, :chi]
    s = s[:chi]
    vh = vh[:chi, :]

    if direction == "left":
        left_t = u.reshape(dl, 2, 2, chi).permute(0, 1, 3, 2)
        right_t = (torch.diag(s.to(dtype=vh.dtype)) @ vh).reshape(chi, 2, dr, 2)
    elif direction == "right":
        left_t = (u @ torch.diag(s.to(dtype=u.dtype))).reshape(dl, 2, 2, chi).permute(0, 1, 3, 2)
        right_t = vh.reshape(chi, 2, dr, 2)
    else:
        raise ValueError("direction must be 'left' or 'right'")

    mpo[left] = left_t
    mpo[left + 1] = right_t


def _canonicalize_mpo_segment(
    mpo: List[torch.Tensor],
    *,
    start: int,
    end: int,
    direction: str,
    max_bond_dim: int | None = None,
) -> None:
    """Sweep-canonicalize an MPO segment with optional bond truncation.

    Performs sequential two-site SVD splits across ``[start, end)`` to bring
    the process MPO segment into left- or right-canonical form.

    Args:
        mpo (*List[torch.Tensor]*): Process MPO tensor list (modified in-place).
        start (*int*): First site index (inclusive).
        end (*int*): Last site index (exclusive for the sweep).
        direction (*str*): ``'left'`` sweeps left-to-right; ``'right'`` sweeps right-to-left.
        max_bond_dim (*int | None*): Maximum bond dimension per split. Defaults to ``None``.

    Raises:
        ValueError: If *direction* is not ``'left'`` or ``'right'``.
    """
    if end <= start:
        return

    if direction == "left":
        for site in range(start, end):
            _split_two_site_theta_mpo(mpo, site, max_bond_dim=max_bond_dim, direction="left")
        return

    if direction == "right":
        for site in range(end, start, -1):
            _split_two_site_theta_mpo(mpo, site - 1, max_bond_dim=max_bond_dim, direction="right")
        return

    raise ValueError("direction must be 'left' or 'right'")


def _move_mpo_canonical_center_to_span_left(
    mpo: List[torch.Tensor],
    *,
    center: int,
    start: int,
    end: int,
) -> int:
    """Relocate the MPO canonical center to the left edge of a target span.

    Performs canonicalization sweeps as needed so that the orthogonality center
    ends up at site *start*, regardless of its current position relative to the
    ``[start, end]`` range.

    Args:
        mpo (*List[torch.Tensor]*): Process MPO tensor list (modified in-place).
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
        _canonicalize_mpo_segment(mpo, start=l, end=r, direction="right")
        return l

    if c < l:
        _canonicalize_mpo_segment(mpo, start=c, end=l, direction="left")
        _canonicalize_mpo_segment(mpo, start=l, end=r, direction="right")
        return l

    _canonicalize_mpo_segment(mpo, start=l, end=c, direction="right")
    return l


def _apply_k_qubit_gate_to_process_with_mpo(
    mpo: List[torch.Tensor],
    qubits: Sequence[int],
    gate_matrix: torch.Tensor,
) -> Tuple[int, int] | None:
    """Apply a k-qubit gate to the process MPO by composing its MPO decomposition.

    For single-qubit gates, left-multiplies onto the site tensor directly.
    For multi-qubit gates, decomposes the unitary into an MPO, expands to a
    contiguous span with identity bridges, and composes it into the process.

    Args:
        mpo (*List[torch.Tensor]*): Process MPO tensor list (modified in-place).
        qubits (*Sequence[int]*): Target qubit indices for the gate.
        gate_matrix (*torch.Tensor*): Unitary gate matrix of shape ``(2**k, 2**k)``.

    Returns:
        Tuple ``(qmin, qmax)`` of the affected qubit span, or ``None`` for
        single-qubit gates.

    Raises:
        ValueError: If gate qubits are not distinct.
    """
    acted = sorted(int(q) for q in qubits)
    if len(set(acted)) != len(acted):
        raise ValueError("gate qubits must be distinct")
    if not acted:
        return None

    if len(acted) == 1:
        q = acted[0]
        mpo[q] = _apply_one_qubit_gate_to_mpo_tensor(mpo[q], gate_matrix.reshape(2, 2))
        return None

    if len(acted) == 2:
        left_mpo, right_mpo = _two_qubit_unitary_to_mpo(gate_matrix, max_bond_dim=None)
        acted_mpo = [left_mpo, right_mpo]
    else:
        acted_mpo = _unitary_to_mpo(gate_matrix, len(acted), max_bond_dim=None)

    mpo_span, qmin, _ = _expand_sparse_gate_mpo_to_span(
        acted_mpo,
        acted,
        dtype=gate_matrix.dtype,
        device=gate_matrix.device,
    )
    _apply_gate_mpo_to_process_segment(mpo, mpo_span, start=qmin)
    return qmin, int(acted[-1])


def simulate_mpo_process(
    qc: QuantumCircuit,
    *,
    param_values: Dict[str, object] | None = None,
    max_bond_dim: int | None = None,
    device: torch.device | str | None = None,
) -> List[torch.Tensor]:
    """Simulate circuit process and return an MPO list.

    The returned tensors use shape [Dl, pout, Dr, pin] at each site.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        param_values (*Dict[str, object] | None*): Symbol-to-value mapping for parameterised gates. Defaults to ``None``.
        max_bond_dim (*int | None*): Maximum bond dimension for SVD truncation. Defaults to ``None``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        List of ``torch.Tensor`` site tensors with shape ``[Dl, pout, Dr, pin]``.

    Raises:
        ValueError: unsupported gate for simulator: {gate}
    """
    num_qubits = int(qc.nqubits)
    if num_qubits <= 0:
        return []

    dtype = torch.complex128
    sim_device = auto_sim_device(device)
    mpo = _identity_mpo(num_qubits, dtype=dtype, device=sim_device)

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
        """Compress the dirty MPO segment if its bond dimension exceeds the limit.

        Moves the canonical center to the left edge of the dirty span, then
        performs a truncating left-to-right canonicalization sweep if any bond
        exceeds *max_bond_dim*. Resets the dirty span afterward.
        """
        nonlocal dirty_start, dirty_end, canon_center
        if dirty_start is None or dirty_end is None:
            return

        canon_center = _move_mpo_canonical_center_to_span_left(
            mpo,
            center=canon_center,
            start=dirty_start,
            end=dirty_end,
        )

        if max_bond_dim is None:
            dirty_start = None
            dirty_end = None
            return

        if _max_bond_in_span(mpo, start=dirty_start, end=dirty_end) <= int(max_bond_dim):
            dirty_start = None
            dirty_end = None
            return

        _canonicalize_mpo_segment(
            mpo,
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
                    "The MPO simulator does not support the 'reset' operation."
                )
            continue

        if gate in one_qubit_gates_available:
            qubit = int(gate_info[1])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=sim_device)
            mpo[qubit] = _apply_one_qubit_gate_to_mpo_tensor(mpo[qubit], mat)
            continue

        if gate in one_qubit_parameter_gates_available:
            qubit = int(gate_info[-1])
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-1]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=sim_device)
            mpo[qubit] = _apply_one_qubit_gate_to_mpo_tensor(mpo[qubit], mat)
            continue

        if gate in two_qubit_gates_available:
            q0 = int(gate_info[1])
            q1 = int(gate_info[2])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=sim_device)
            span = _apply_k_qubit_gate_to_process_with_mpo(mpo, [q0, q1], mat)
            if span is not None:
                _mark_dirty(*span)
                _maybe_compress_dirty_span()
            continue

        if gate in two_qubit_parameter_gates_available:
            q0 = int(gate_info[-2])
            q1 = int(gate_info[-1])
            params = [resolve_param(qc, p, param_values) for p in gate_info[1:-2]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=sim_device)
            span = _apply_k_qubit_gate_to_process_with_mpo(mpo, [q0, q1], mat)
            if span is not None:
                _mark_dirty(*span)
                _maybe_compress_dirty_span()
            continue

        if gate in three_qubit_gates_available:
            qubits = [int(gate_info[1]), int(gate_info[2]), int(gate_info[3])]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=sim_device)
            span = _apply_k_qubit_gate_to_process_with_mpo(mpo, qubits, mat)
            if span is not None:
                _mark_dirty(*span)
                _maybe_compress_dirty_span()
            continue

        raise ValueError(f"unsupported gate for simulator: {gate}")

    return mpo
