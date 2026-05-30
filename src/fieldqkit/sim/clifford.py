"""Stabilizer-style Clifford simulator via Heisenberg Pauli conjugation.

For a Clifford circuit ``U`` acting on the all-zero state ``|0...0⟩``, this
module computes Pauli observable expectations

    ⟨0|U† P U|0⟩

by evolving each observable backwards through the gate sequence using
Clifford conjugation rules.  Each Clifford gate maps a Pauli operator to
another Pauli operator (with a ``±1`` sign), so the cost is ``O(g · n)``
per observable where ``g`` is the gate count and ``n`` the qubit count.
This is exponentially cheaper than statevector simulation (``O(2^n)``)
and scales to large qubit counts.

Supported Clifford gates: ``{h, s, sdg, x, y, z, sx, sxdg, id, cx/cnot,
cz, swap}``, plus parameterised rotations ``{rx, ry, rz, p}`` whose angle
is an integer multiple of ``π/2``, plus ``u``/``u3`` whose
``(theta, phi, lambda)`` corresponds to one of the 24 single-qubit
Cliffords.

Non-Clifford gates raise :class:`CliffordError`; use
:mod:`fieldqkit.sim.clifford_t` for circuits with ``t``/``tdg`` or
non-Clifford rotations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..circuit.quantumcircuit import QuantumCircuit
from ..core.observables import pauli_basis_pattern
from ..circuit.quantumcircuit_helpers import functional_gates_available


# Fixed single-qubit Pauli matrices used for U3 decomposition checks.
_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_PAULI_CHAR_MAT = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


class CliffordError(ValueError):
    """Raised when a non-Clifford gate is encountered by the stabilizer simulator."""


# ---------------------------------------------------------------------------
# Single-qubit Heisenberg rules: σ_in → (σ_out, sign)
# ---------------------------------------------------------------------------

# H† P H = H P H. Y → -Y (sign flip), X ↔ Z.
_H_TABLE = {"I": ("I", 1), "X": ("Z", 1), "Y": ("Y", -1), "Z": ("X", 1)}
# S† P S: X → -Y, Y → X, Z → Z, I → I (S = diag(1, i)).
_S_TABLE = {"I": ("I", 1), "X": ("Y", -1), "Y": ("X", 1), "Z": ("Z", 1)}
# Sdg† P Sdg: X → Y, Y → -X, Z → Z.
_SDG_TABLE = {"I": ("I", 1), "X": ("Y", 1), "Y": ("X", -1), "Z": ("Z", 1)}
# X P X: X → X, Y → -Y, Z → -Z.
_X_TABLE = {"I": ("I", 1), "X": ("X", 1), "Y": ("Y", -1), "Z": ("Z", -1)}
# Y P Y: X → -X, Y → Y, Z → -Z.
_Y_TABLE = {"I": ("I", 1), "X": ("X", -1), "Y": ("Y", 1), "Z": ("Z", -1)}
# Z P Z: X → -X, Y → -Y, Z → Z.
_Z_TABLE = {"I": ("I", 1), "X": ("X", -1), "Y": ("Y", -1), "Z": ("Z", 1)}
# SX = sqrt(X) (up to phase): SX = e^{iπ/4}·(I − iX)/√2.
# Verified: SX† X SX = X, SX† Y SX = −Z, SX† Z SX = +Y.
_SX_TABLE = {"I": ("I", 1), "X": ("X", 1), "Y": ("Z", -1), "Z": ("Y", 1)}
# SXdg = SX†: SX σ SX† has the inverse signs.
_SXDG_TABLE = {"I": ("I", 1), "X": ("X", 1), "Y": ("Z", 1), "Z": ("Y", -1)}

_SINGLE_QUBIT_TABLES: Dict[str, Dict[str, Tuple[str, int]]] = {
    "h": _H_TABLE,
    "s": _S_TABLE,
    "sdg": _SDG_TABLE,
    "x": _X_TABLE,
    "y": _Y_TABLE,
    "z": _Z_TABLE,
    "sx": _SX_TABLE,
    "sxdg": _SXDG_TABLE,
    "id": {"I": ("I", 1), "X": ("X", 1), "Y": ("Y", 1), "Z": ("Z", 1)},
}


# ---------------------------------------------------------------------------
# Two-qubit Heisenberg rules: (σ_c, σ_t) → (σ_c', σ_t', sign)
# Derived from CNOT/CZ propagation rules X_c → X_c X_t, etc.
# ---------------------------------------------------------------------------

_CX_TABLE: Dict[Tuple[str, str], Tuple[str, str, int]] = {
    ("I", "I"): ("I", "I", 1), ("I", "X"): ("I", "X", 1),
    ("I", "Y"): ("Z", "Y", 1), ("I", "Z"): ("Z", "Z", 1),
    ("X", "I"): ("X", "X", 1), ("X", "X"): ("X", "I", 1),
    ("X", "Y"): ("Y", "Z", 1), ("X", "Z"): ("Y", "Y", -1),
    ("Y", "I"): ("Y", "X", 1), ("Y", "X"): ("Y", "I", 1),
    ("Y", "Y"): ("X", "Z", -1), ("Y", "Z"): ("X", "Y", 1),
    ("Z", "I"): ("Z", "I", 1), ("Z", "X"): ("Z", "X", 1),
    ("Z", "Y"): ("I", "Y", 1), ("Z", "Z"): ("I", "Z", 1),
}

_CZ_TABLE: Dict[Tuple[str, str], Tuple[str, str, int]] = {
    ("I", "I"): ("I", "I", 1), ("I", "X"): ("Z", "X", 1),
    ("I", "Y"): ("Z", "Y", 1), ("I", "Z"): ("I", "Z", 1),
    ("X", "I"): ("X", "Z", 1), ("X", "X"): ("Y", "Y", 1),
    ("X", "Y"): ("Y", "X", -1), ("X", "Z"): ("X", "I", 1),
    ("Y", "I"): ("Y", "Z", 1), ("Y", "X"): ("X", "Y", -1),
    ("Y", "Y"): ("X", "X", 1), ("Y", "Z"): ("Y", "I", 1),
    ("Z", "I"): ("Z", "I", 1), ("Z", "X"): ("I", "X", 1),
    ("Z", "Y"): ("I", "Y", 1), ("Z", "Z"): ("Z", "Z", 1),
}

_SWAP_TABLE: Dict[Tuple[str, str], Tuple[str, str, int]] = {
    (a, b): (b, a, 1) for a in "IXYZ" for b in "IXYZ"
}

_TWO_QUBIT_TABLES: Dict[str, Dict[Tuple[str, str], Tuple[str, str, int]]] = {
    "cx": _CX_TABLE,
    "cnot": _CX_TABLE,
    "cz": _CZ_TABLE,
    "swap": _SWAP_TABLE,
}


# ---------------------------------------------------------------------------
# Clifford rotation helpers: π/2 multiples only.
# ---------------------------------------------------------------------------


def _angle_mod_4_halfpi(angle: float, atol: float = 1e-8) -> Optional[int]:
    """Return ``k`` in ``{0,1,2,3}`` if ``angle ≈ k·π/2`` (mod 2π), else ``None``."""
    k = angle / (math.pi / 2.0)
    rk = round(k)
    if abs(k - rk) < atol or abs(((k - rk) % 4) - 4) < atol:
        return int(rk) % 4
    return None


def _rz_clifford_table(k: int) -> Dict[str, Tuple[str, int]]:
    """Rotation Rz(k·π/2) Heisenberg table for ``k ∈ {0,1,2,3}``."""
    if k == 0:
        return {"I": ("I", 1), "X": ("X", 1), "Y": ("Y", 1), "Z": ("Z", 1)}
    if k == 1:
        return _S_TABLE
    if k == 2:
        return _Z_TABLE
    return _SDG_TABLE


def _rx_clifford_table(k: int) -> Dict[str, Tuple[str, int]]:
    if k == 0:
        return {"I": ("I", 1), "X": ("X", 1), "Y": ("Y", 1), "Z": ("Z", 1)}
    if k == 1:
        return _SX_TABLE
    if k == 2:
        return _X_TABLE
    return _SXDG_TABLE


def _ry_clifford_table(k: int) -> Dict[str, Tuple[str, int]]:
    # Ry(π/2) Heisenberg: X→Z, Z→-X, Y→Y. Ry(3π/2) is the inverse.
    if k == 0:
        return {"I": ("I", 1), "X": ("X", 1), "Y": ("Y", 1), "Z": ("Z", 1)}
    if k == 1:
        return {"I": ("I", 1), "X": ("Z", 1), "Y": ("Y", 1), "Z": ("X", -1)}
    if k == 2:
        return _Y_TABLE
    return {"I": ("I", 1), "X": ("Z", -1), "Y": ("Y", 1), "Z": ("X", 1)}


# ---------------------------------------------------------------------------
# Clifford detection for arbitrary U3(θ, φ, λ) at runtime.
# ---------------------------------------------------------------------------


def _u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """OpenQASM-3 ``U(theta, phi, lambda)`` 2×2 matrix."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array(
        [
            [c, -np.exp(1.0j * lam) * s],
            [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


def u3_clifford_table(theta: float, phi: float, lam: float, atol: float = 1e-7) -> Optional[Dict[str, Tuple[str, int]]]:
    """Return Heisenberg table for ``U3(θ, φ, λ)`` if it is Clifford, else ``None``.

    Computes ``U† σ U`` for ``σ ∈ {X, Y, Z}`` and checks whether each
    result is a single signed Pauli (the defining property of a Clifford
    operator).

    Args:
        theta (*float*): First Euler angle.
        phi (*float*): Second Euler angle.
        lam (*float*): Third Euler angle.
        atol (*float*): Numeric tolerance for "single Pauli" detection.

    Returns:
        Dict mapping each input Pauli char (``'I'/'X'/'Y'/'Z'``) to its
        ``(output_char, sign)`` image, or ``None`` if not Clifford.
    """
    u = _u3_matrix(theta, phi, lam)
    ud = u.conj().T
    table: Dict[str, Tuple[str, int]] = {"I": ("I", 1)}
    for char in ("X", "Y", "Z"):
        sigma = _PAULI_CHAR_MAT[char]
        conj = ud @ sigma @ u
        found: Optional[Tuple[str, int]] = None
        for p_char, p_mat in _PAULI_CHAR_MAT.items():
            coeff = 0.5 * np.trace(conj @ p_mat)
            magnitude = abs(coeff)
            if magnitude < atol:
                continue
            if abs(magnitude - 1.0) > atol:
                return None  # mixed Pauli content → not Clifford
            if abs(coeff.imag) > atol:
                return None  # imaginary component → not Clifford
            sign = 1 if coeff.real > 0 else -1
            if found is not None:
                return None  # already saw one Pauli → non-Clifford
            found = (p_char, sign)
        if found is None:
            return None
        table[char] = found
    return table


# ---------------------------------------------------------------------------
# Gate dispatcher
# ---------------------------------------------------------------------------


def _conjugate_single(p: List[str], q: int, table: Dict[str, Tuple[str, int]]) -> int:
    """Apply a single-qubit Heisenberg table to Pauli list ``p`` at qubit ``q``."""
    in_char = p[q]
    out_char, sign = table[in_char]
    p[q] = out_char
    return sign


def _conjugate_two(p: List[str], c: int, t: int, table: Dict[Tuple[str, str], Tuple[str, str, int]]) -> int:
    in_pair = (p[c], p[t])
    new_c, new_t, sign = table[in_pair]
    p[c] = new_c
    p[t] = new_t
    return sign


def conjugate_clifford_gate(p: List[str], gate_info: tuple) -> int:
    """Apply ``G† P G`` for one Clifford gate.

    Mutates ``p`` (a list of ``'I'/'X'/'Y'/'Z'`` chars indexed by qubit)
    in place and returns the accumulated sign factor (``±1``).

    Args:
        p (*List[str]*): Pauli string as a mutable list of chars; one entry per logical qubit.
        gate_info (*tuple*): A circuit gate tuple, e.g. ``('h', q)`` or ``('cx', c, t)``.

    Returns:
        ``+1`` or ``-1``.

    Raises:
        CliffordError: If the gate is non-Clifford.
    """
    gate = gate_info[0]
    if gate in functional_gates_available:
        return 1
    if gate in _SINGLE_QUBIT_TABLES:
        return _conjugate_single(p, int(gate_info[1]), _SINGLE_QUBIT_TABLES[gate])
    if gate in _TWO_QUBIT_TABLES:
        return _conjugate_two(p, int(gate_info[1]), int(gate_info[2]), _TWO_QUBIT_TABLES[gate])
    if gate in {"rz", "rx", "ry", "p"}:
        theta = float(gate_info[1])
        q = int(gate_info[-1])
        k = _angle_mod_4_halfpi(theta)
        if k is None:
            raise CliffordError(f"non-Clifford rotation '{gate}({theta})'")
        if gate == "rz" or gate == "p":
            return _conjugate_single(p, q, _rz_clifford_table(k))
        if gate == "rx":
            return _conjugate_single(p, q, _rx_clifford_table(k))
        return _conjugate_single(p, q, _ry_clifford_table(k))
    if gate in {"u", "u3"}:
        theta, phi, lam = float(gate_info[1]), float(gate_info[2]), float(gate_info[3])
        q = int(gate_info[-1])
        table = u3_clifford_table(theta, phi, lam)
        if table is None:
            raise CliffordError(f"non-Clifford U3({theta},{phi},{lam})")
        return _conjugate_single(p, q, table)
    if gate in {"t", "tdg"}:
        raise CliffordError(f"non-Clifford gate '{gate}' (use clifford_t simulator)")
    raise CliffordError(f"unsupported gate '{gate}' for Clifford simulator")


def is_clifford_circuit(qc: QuantumCircuit) -> bool:
    """Return ``True`` if every gate in ``qc`` is Clifford.

    Args:
        qc (*QuantumCircuit*): Circuit to inspect.

    Returns:
        ``bool`` — whether all gates are recognised as Clifford.
    """
    probe = ["I"] * int(qc.nqubits) if int(qc.nqubits) > 0 else ["I"]
    for gate_info in qc.gates:
        try:
            conjugate_clifford_gate(probe, gate_info)
        except CliffordError:
            return False
    return True


# ---------------------------------------------------------------------------
# Public API: expectation values
# ---------------------------------------------------------------------------


def pauli_pattern_to_list(pauli: str, num_qubits: int) -> List[str]:
    """Build a length-``num_qubits`` Pauli char list from a Pauli string spec.

    Reuses :func:`fieldqkit.core.observables.pauli_basis_pattern`, which
    accepts either dense (e.g. ``'XYIZ'``) or sparse (e.g. ``'X0 Z2'``)
    notation.

    Args:
        pauli (*str*): Observable specification.
        num_qubits (*int*): Number of qubits.

    Returns:
        Mutable list of length ``num_qubits``.
    """
    return list(pauli_basis_pattern(pauli, num_qubits=num_qubits))


def _pauli_expectation_on_zero(p: Sequence[str]) -> int:
    """Return ``⟨0...0|P|0...0⟩``: ``+1`` if ``P`` is diagonal (only I/Z), else ``0``."""
    for c in p:
        if c == "X" or c == "Y":
            return 0
    return 1


def simulate_clifford_expectation(
    qc: QuantumCircuit,
    pauli: str,
    *,
    num_qubits: Optional[int] = None,
) -> float:
    """Compute ``⟨0|U†_qc · P · U_qc|0⟩`` for a single Pauli ``P``.

    Args:
        qc (*QuantumCircuit*): Pure Clifford circuit.
        pauli (*str*): Pauli string spec (dense or sparse notation).
        num_qubits (*Optional[int]*): Override the qubit count. Defaults to ``qc.nqubits``.

    Returns:
        ``float`` — the expectation value, in ``{-1.0, 0.0, +1.0}``.

    Raises:
        CliffordError: If ``qc`` contains non-Clifford gates.
    """
    n = int(qc.nqubits) if num_qubits is None else int(num_qubits)
    p = pauli_pattern_to_list(pauli, n)
    sign = 1
    for gate_info in reversed(qc.gates):
        sign *= conjugate_clifford_gate(p, gate_info)
    if _pauli_expectation_on_zero(p):
        return float(sign)
    return 0.0


def simulate_clifford_expectations(
    qc: QuantumCircuit,
    observables: Sequence[str],
    *,
    num_qubits: Optional[int] = None,
) -> Dict[str, float]:
    """Compute expectations of multiple Pauli observables on ``U|0⟩``.

    Args:
        qc (*QuantumCircuit*): Pure Clifford circuit.
        observables (*Sequence[str]*): Pauli string specifications.
        num_qubits (*Optional[int]*): Override the qubit count. Defaults to ``qc.nqubits``.

    Returns:
        Dict mapping each observable to its expectation in ``{-1.0, 0.0, +1.0}``.

    Raises:
        CliffordError: If ``qc`` contains non-Clifford gates.
    """
    return {obs: simulate_clifford_expectation(qc, obs, num_qubits=num_qubits) for obs in observables}
