"""Clifford + arbitrary non-Clifford single-qubit gate simulator.

This module extends :mod:`quantum_hw.sim.clifford` to handle any non-Clifford
single-qubit gate by branching the observable's Heisenberg evolution along
the four-term Pauli basis expansion of ``U† σ U``.

Algorithm sketch
----------------

For a circuit ``U = G_k · ... · G_1`` applied to ``|0...0⟩``, the
expectation of a Pauli observable ``P`` factors as

    ⟨0|U† P U|0⟩

We evolve ``P`` backwards.  Each Clifford gate ``G`` produces a single
Pauli image ``G† P G`` with a ``±1`` sign.  Each non-Clifford
single-qubit gate ``V`` at qubit ``q`` produces up to four images,
because

    V† σ V = Σ_{P' ∈ {I,X,Y,Z}} α_{P'} · P'

with complex coefficients ``α_{P'}``.  We maintain a dictionary
``{pauli_string → coefficient}`` and at every gate either rewrite each
key in place (Clifford) or expand it into multiple keys (non-Clifford),
summing coefficients for identical strings.

Final expectation is ``Σ_P α_P · ⟨0|P|0⟩``, with
``⟨0|P|0⟩ = 1`` iff ``P`` is diagonal (only ``I``/``Z`` components).

Supported non-Clifford gates: ``{t, tdg}`` (2 branches each),
``{rx, ry, rz, p}`` with arbitrary angle (2 branches), ``{u, u3, r}``
(up to 4 branches), and ``{rxx, ryy, rzz, cp}`` with arbitrary angle
(2 branches).

The worst-case term count grows multiplicatively at each non-Clifford
gate.  Negligible coefficients are pruned with a configurable threshold
to keep the dictionary compact.
"""

from __future__ import annotations

import cmath
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..circuit.quantumcircuit import QuantumCircuit
from .clifford import (
    CliffordError,
    _pauli_expectation_on_zero,
    _u3_matrix,
    conjugate_clifford_gate,
    pauli_pattern_to_list,
    u3_clifford_table,
)


# Small numerical thresholds.
_DEFAULT_PRUNE_TOL = 1e-12
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Pauli term collection
# ---------------------------------------------------------------------------


PauliTerm = Tuple[str, ...]  # tuple of 'I'/'X'/'Y'/'Z' chars indexed by qubit


def _add_term(state: Dict[PauliTerm, complex], term: PauliTerm, coeff: complex, prune_tol: float) -> None:
    """Accumulate ``term -> coeff`` into the running dict with pruning."""
    cur = state.get(term, 0.0 + 0.0j) + coeff
    if abs(cur) <= prune_tol:
        if term in state:
            del state[term]
    else:
        state[term] = cur


# ---------------------------------------------------------------------------
# Heisenberg branching rules
# ---------------------------------------------------------------------------


def _branch_t(p_tuple: PauliTerm, q: int, dagger: bool) -> List[Tuple[PauliTerm, complex]]:
    """Branch list for ``T† σ T`` (or ``Tdg† σ Tdg`` if ``dagger``).

    ``T† X T = (X − Y)/√2``, ``T† Y T = (X + Y)/√2``, ``T† Z T = Z``.
    Tdg gives the conjugate signs: ``Tdg† X Tdg = (X + Y)/√2``,
    ``Tdg† Y Tdg = (−X + Y)/√2``.
    """
    char = p_tuple[q]
    if char == "I" or char == "Z":
        return [(p_tuple, 1.0 + 0.0j)]
    p_x = list(p_tuple); p_x[q] = "X"
    p_y = list(p_tuple); p_y[q] = "Y"
    if char == "X":
        # T:  X → (X − Y)/√2  ;  Tdg:  X → (X + Y)/√2
        sign_y = -1.0 if not dagger else 1.0
        return [(tuple(p_x), _INV_SQRT2 + 0j), (tuple(p_y), sign_y * _INV_SQRT2 + 0j)]
    # char == 'Y'
    # T:  Y → (X + Y)/√2  ;  Tdg:  Y → (−X + Y)/√2
    sign_x = 1.0 if not dagger else -1.0
    return [(tuple(p_x), sign_x * _INV_SQRT2 + 0j), (tuple(p_y), _INV_SQRT2 + 0j)]


def _branch_axis_rotation(p_tuple: PauliTerm, q: int, axis: str, theta: float) -> List[Tuple[PauliTerm, complex]]:
    """Branch list for ``R_axis(θ)† σ R_axis(θ)``.

    For ``R_z(θ) = exp(-iθZ/2)``:
        - ``X → cos(θ)X − sin(θ)Y``
        - ``Y → cos(θ)Y + sin(θ)X``
        - ``Z, I → unchanged``
    For ``R_x(θ)``:
        - ``Y → cos(θ)Y − sin(θ)Z``
        - ``Z → cos(θ)Z + sin(θ)Y``
        - ``X, I → unchanged``
    For ``R_y(θ)``:
        - ``X → cos(θ)X + sin(θ)Z``
        - ``Z → cos(θ)Z − sin(θ)X``
        - ``Y, I → unchanged``
    """
    char = p_tuple[q]
    if char == "I":
        return [(p_tuple, 1.0 + 0.0j)]
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    if axis == "z":
        if char == "Z":
            return [(p_tuple, 1.0 + 0.0j)]
        if char == "X":
            p_x = list(p_tuple); p_x[q] = "X"
            p_y = list(p_tuple); p_y[q] = "Y"
            return [(tuple(p_x), cos_t + 0j), (tuple(p_y), -sin_t + 0j)]
        # char == 'Y'
        p_x = list(p_tuple); p_x[q] = "X"
        p_y = list(p_tuple); p_y[q] = "Y"
        return [(tuple(p_y), cos_t + 0j), (tuple(p_x), sin_t + 0j)]
    if axis == "x":
        if char == "X":
            return [(p_tuple, 1.0 + 0.0j)]
        if char == "Y":
            # Rx†(θ) Y Rx(θ) = cos(θ) Y − sin(θ) Z
            p_y = list(p_tuple); p_y[q] = "Y"
            p_z = list(p_tuple); p_z[q] = "Z"
            return [(tuple(p_y), cos_t + 0j), (tuple(p_z), -sin_t + 0j)]
        # char == 'Z':  Rx†(θ) Z Rx(θ) = cos(θ) Z + sin(θ) Y
        p_y = list(p_tuple); p_y[q] = "Y"
        p_z = list(p_tuple); p_z[q] = "Z"
        return [(tuple(p_z), cos_t + 0j), (tuple(p_y), sin_t + 0j)]
    # axis == 'y'
    if char == "Y":
        return [(p_tuple, 1.0 + 0.0j)]
    if char == "X":
        p_x = list(p_tuple); p_x[q] = "X"
        p_z = list(p_tuple); p_z[q] = "Z"
        return [(tuple(p_x), cos_t + 0j), (tuple(p_z), sin_t + 0j)]
    # char == 'Z'
    p_x = list(p_tuple); p_x[q] = "X"
    p_z = list(p_tuple); p_z[q] = "Z"
    return [(tuple(p_z), cos_t + 0j), (tuple(p_x), -sin_t + 0j)]


def _branch_two_qubit_rotation(
    p_tuple: PauliTerm,
    q0: int,
    q1: int,
    pauli_kind: str,
    theta: float,
) -> List[Tuple[PauliTerm, complex]]:
    """Branch list for ``R_{PP}(θ)† σ R_{PP}(θ)`` (``RXX``/``RYY``/``RZZ``).

    Uses ``R_{PP}(θ) = exp(-iθ P⊗P/2)``.  Heisenberg conjugation of a
    Pauli term commuting with ``P⊗P`` is unchanged; anti-commuting terms
    rotate as ``A → cos(θ)A + i·sin(θ)[P⊗P, A]/2`` which evaluates to a
    real combination of two Paulis.
    """
    char0 = p_tuple[q0]
    char1 = p_tuple[q1]
    p_char = pauli_kind.upper()
    # Operator A=σ_{q0}⊗σ_{q1} commutes with P⊗P iff A and P⊗P share parity on
    # qubits where they both differ from I.  Standard rule: A commutes iff
    # #{anticommuting positions} is even.
    anti = 0
    for c, q_p in ((char0, p_char), (char1, p_char)):
        if c == "I" or c == q_p:
            continue
        anti += 1
    if anti % 2 == 0:
        return [(p_tuple, 1.0 + 0.0j)]
    # Anti-commuting: rotates into A' where A' = i (P⊗P)·A up to sign.
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # Compute the rotated Pauli image: apply P_q0 then P_q1 to A.
    new_chars = [char0, char1]
    sign = 1
    for idx, (orig_char, p_q) in enumerate(((char0, p_char), (char1, p_char))):
        if orig_char == "I":
            new_chars[idx] = p_q
        elif orig_char == p_q:
            new_chars[idx] = "I"
        else:
            # σ_a · σ_b for distinct nontrivial Pauli ⇒ i·ε_{abc}·σ_c.
            mapping = {("X", "Y"): ("Z", 1), ("Y", "X"): ("Z", -1),
                       ("X", "Z"): ("Y", -1), ("Z", "X"): ("Y", 1),
                       ("Y", "Z"): ("X", 1), ("Z", "Y"): ("X", -1)}
            out, eps = mapping[(p_q, orig_char)]
            new_chars[idx] = out
            sign *= eps
    # i · sign already gives the imaginary unit factor in [P⊗P, A]/(2i)
    # For exp(-i θ A/2) conjugation: A → cos(θ) A + sin(θ) · ([A, P⊗P]/(2i))
    # the imaginary factor cancels and yields a real coefficient.  We
    # have set new_chars to the Pauli image of i·(P⊗P)·A divided by i,
    # so the rotated term picks up factor ``sin(θ) * sign``.
    new_tuple = list(p_tuple)
    new_tuple[q0] = new_chars[0]
    new_tuple[q1] = new_chars[1]
    # For V = exp(-iθH/2) with {A,H}=0:  V†AV = cos(θ)A + i·sin(θ)·HA.
    # Collecting per-qubit products gives HA = i^anti · sign · σ_new (here
    # anti=1 for the two-qubit anti-commuting case), so the resulting
    # real coefficient is  i·sin(θ)·i·sign = −sin(θ)·sign.
    out_pair = (tuple(new_tuple), -float(sign) * sin_t + 0.0j)
    same_pair = (p_tuple, cos_t + 0.0j)
    return [same_pair, out_pair]


def _branch_u3(p_tuple: PauliTerm, q: int, theta: float, phi: float, lam: float, atol: float = 1e-12) -> List[Tuple[PauliTerm, complex]]:
    """Branch list for an arbitrary ``U(θ, φ, λ)`` gate.

    Computes ``U† σ U`` as a 2×2 matrix and decomposes it on the Pauli
    basis.  Returns up to four ``(new_term, coefficient)`` pairs.
    """
    char = p_tuple[q]
    if char == "I":
        return [(p_tuple, 1.0 + 0.0j)]
    sigma = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }[char]
    u = _u3_matrix(theta, phi, lam)
    conj = u.conj().T @ sigma @ u
    out: List[Tuple[PauliTerm, complex]] = []
    for new_char, p_mat in (
        ("I", np.eye(2, dtype=complex)),
        ("X", np.array([[0, 1], [1, 0]], dtype=complex)),
        ("Y", np.array([[0, -1j], [1j, 0]], dtype=complex)),
        ("Z", np.array([[1, 0], [0, -1]], dtype=complex)),
    ):
        coeff = 0.5 * complex(np.trace(conj @ p_mat))
        if abs(coeff) <= atol:
            continue
        new_term = list(p_tuple); new_term[q] = new_char
        out.append((tuple(new_term), coeff))
    return out


# ---------------------------------------------------------------------------
# Gate dispatcher with branching
# ---------------------------------------------------------------------------


def _apply_branching_gate(
    state: Dict[PauliTerm, complex],
    gate_info: tuple,
    prune_tol: float,
) -> Dict[PauliTerm, complex]:
    """Apply ``G† · state · G`` for one gate, branching as needed.

    Returns a new ``state`` dictionary.
    """
    gate = gate_info[0]
    # Try the pure-Clifford fast path first (no branching).
    try:
        new_state: Dict[PauliTerm, complex] = {}
        for term, coeff in state.items():
            p_list = list(term)
            sign = conjugate_clifford_gate(p_list, gate_info)
            new_term = tuple(p_list)
            new_state[new_term] = new_state.get(new_term, 0.0 + 0.0j) + coeff * sign
        return new_state
    except CliffordError:
        pass

    # Non-Clifford path: enumerate branches per term.
    new_state = {}
    for term, coeff in state.items():
        if gate in {"t", "tdg"}:
            branches = _branch_t(term, int(gate_info[1]), gate == "tdg")
        elif gate in {"rz", "p"}:
            branches = _branch_axis_rotation(term, int(gate_info[-1]), "z", float(gate_info[1]))
        elif gate == "rx":
            branches = _branch_axis_rotation(term, int(gate_info[-1]), "x", float(gate_info[1]))
        elif gate == "ry":
            branches = _branch_axis_rotation(term, int(gate_info[-1]), "y", float(gate_info[1]))
        elif gate in {"u", "u3"}:
            theta, phi, lam = float(gate_info[1]), float(gate_info[2]), float(gate_info[3])
            q = int(gate_info[-1])
            # Check Clifford fast path one more time at the U3 level: many
            # U3 entries are Clifford but landed here only because *another*
            # gate in the same call triggered the CliffordError.
            tbl = u3_clifford_table(theta, phi, lam)
            if tbl is not None:
                p_list = list(term)
                in_char = p_list[q]
                out_char, sign = tbl[in_char]
                p_list[q] = out_char
                branches = [(tuple(p_list), float(sign) + 0.0j)]
            else:
                branches = _branch_u3(term, q, theta, phi, lam)
        elif gate == "r":
            # R(theta, phi) = U(theta, phi - pi/2, pi/2 - phi).
            theta_r = float(gate_info[1])
            phi_r = float(gate_info[2])
            q = int(gate_info[-1])
            branches = _branch_u3(term, q, theta_r, phi_r - math.pi / 2.0, math.pi / 2.0 - phi_r)
        elif gate == "rxx":
            branches = _branch_two_qubit_rotation(term, int(gate_info[-2]), int(gate_info[-1]), "X", float(gate_info[1]))
        elif gate == "ryy":
            branches = _branch_two_qubit_rotation(term, int(gate_info[-2]), int(gate_info[-1]), "Y", float(gate_info[1]))
        elif gate == "rzz":
            branches = _branch_two_qubit_rotation(term, int(gate_info[-2]), int(gate_info[-1]), "Z", float(gate_info[1]))
        elif gate == "cp":
            # CP(theta) = diag(1, 1, 1, e^{i theta}).  Equivalent (up to
            # global phase) to RZZ(-theta/2) · RZ(theta/2)_c · RZ(theta/2)_t.
            theta_cp = float(gate_info[1])
            c, t = int(gate_info[-2]), int(gate_info[-1])
            inter = _branch_two_qubit_rotation(term, c, t, "Z", -theta_cp / 2.0)
            tmp_state: Dict[PauliTerm, complex] = {}
            for inter_term, inter_coeff in inter:
                for b_term, b_coeff in _branch_axis_rotation(inter_term, c, "z", theta_cp / 2.0):
                    for c_term, c_coeff in _branch_axis_rotation(b_term, t, "z", theta_cp / 2.0):
                        tmp_state[c_term] = tmp_state.get(c_term, 0.0 + 0.0j) + inter_coeff * b_coeff * c_coeff
            branches = list(tmp_state.items())
        else:
            raise CliffordError(f"clifford_t simulator: unsupported non-Clifford gate '{gate}'")
        for new_term, b_coeff in branches:
            _add_term(new_state, new_term, coeff * b_coeff, prune_tol)
    return new_state


def count_t_gates(qc: QuantumCircuit) -> int:
    """Return the number of ``t`` and ``tdg`` gates in a circuit.

    Args:
        qc (*QuantumCircuit*): Circuit to inspect.

    Returns:
        ``int`` — total ``t``/``tdg`` count (useful for budget checks).
    """
    return sum(1 for g in qc.gates if g[0] in {"t", "tdg"})


def count_non_clifford_gates(qc: QuantumCircuit) -> int:
    """Return the number of non-Clifford gates in a circuit.

    Counts gates that are *not* recognised as Clifford by
    :func:`quantum_hw.sim.clifford.conjugate_clifford_gate`.  Useful for
    estimating worst-case branching factor (≤4 per single-qubit
    non-Clifford gate, ≤2 per axis rotation, ≤2 per ``t``/``tdg``).

    Args:
        qc (*QuantumCircuit*): Circuit to inspect.

    Returns:
        ``int`` — non-Clifford gate count.
    """
    probe = ["I"] * max(int(qc.nqubits), 1)
    count = 0
    for gate_info in qc.gates:
        try:
            conjugate_clifford_gate(list(probe), gate_info)
        except CliffordError:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_clifford_t_expectation(
    qc: QuantumCircuit,
    pauli: str,
    *,
    num_qubits: Optional[int] = None,
    prune_tol: float = _DEFAULT_PRUNE_TOL,
    max_terms: Optional[int] = None,
) -> float:
    """Compute ``⟨0|U† P U|0⟩`` for a Clifford + non-Clifford circuit.

    Evolves ``P`` backwards through the gate sequence, branching at every
    non-Clifford single-qubit gate.  Returns the real expectation value.

    Args:
        qc (*QuantumCircuit*): Circuit (may contain ``t``/``tdg``, arbitrary rotations, or generic ``u``/``u3`` gates).
        pauli (*str*): Pauli string (dense or sparse notation).
        num_qubits (*Optional[int]*): Override the qubit count. Defaults to ``qc.nqubits``.
        prune_tol (*float*): Absolute threshold below which Pauli-term coefficients are dropped. Defaults to ``1e-12``.
        max_terms (*Optional[int]*): Maximum allowed number of Pauli terms during evolution.  ``None`` means unbounded. Defaults to ``None``.

    Returns:
        Real-valued expectation.

    Raises:
        RuntimeError: If the term count exceeds ``max_terms``.
        CliffordError: If the circuit contains an unsupported non-Clifford 2-qubit gate.
    """
    n = int(qc.nqubits) if num_qubits is None else int(num_qubits)
    init_term = tuple(pauli_pattern_to_list(pauli, n))
    state: Dict[PauliTerm, complex] = {init_term: 1.0 + 0.0j}
    for gate_info in reversed(qc.gates):
        state = _apply_branching_gate(state, gate_info, prune_tol)
        if max_terms is not None and len(state) > int(max_terms):
            raise RuntimeError(
                f"clifford_t expansion exceeded max_terms={max_terms} "
                f"(current={len(state)}); reduce non-Clifford gates or relax the bound."
            )
    total = 0.0 + 0.0j
    for term, coeff in state.items():
        if _pauli_expectation_on_zero(term):
            total += coeff
    if abs(total.imag) > 1e-9:
        # Real-valued Hamiltonian expectations should always be real for a
        # unitary evolution; flag if floating-point error grows.
        raise RuntimeError(f"clifford_t expectation has unexpected imaginary part {total.imag}")
    return float(total.real)


def simulate_clifford_t_expectations(
    qc: QuantumCircuit,
    observables: Sequence[str],
    *,
    num_qubits: Optional[int] = None,
    prune_tol: float = _DEFAULT_PRUNE_TOL,
    max_terms: Optional[int] = None,
) -> Dict[str, float]:
    """Compute expectations of multiple Pauli observables on ``U|0⟩``.

    Args:
        qc (*QuantumCircuit*): Circuit (may contain non-Clifford gates).
        observables (*Sequence[str]*): Pauli string specifications.
        num_qubits (*Optional[int]*): Override the qubit count. Defaults to ``qc.nqubits``.
        prune_tol (*float*): Absolute prune threshold. Defaults to ``1e-12``.
        max_terms (*Optional[int]*): Maximum allowed Pauli-term count. Defaults to ``None``.

    Returns:
        Dict mapping each observable to its expectation.
    """
    return {
        obs: simulate_clifford_t_expectation(
            qc, obs, num_qubits=num_qubits, prune_tol=prune_tol, max_terms=max_terms,
        )
        for obs in observables
    }
