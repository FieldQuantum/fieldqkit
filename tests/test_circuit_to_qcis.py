"""Tests for circuit_to_qcis: QuantumCircuit → QCIS direct conversion.

Verifies that circuit_to_qcis produces correct QCIS output for all
supported gate types, covering all registered internal gates and functional
instructions.
"""

import math
import numpy as np
import pytest

from fieldqkit.circuit import QuantumCircuit
from fieldqkit.circuit.qcis import circuit_to_qcis
from fieldqkit.compile.translate import TranslateToBasisGates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _translate(qc: QuantumCircuit, two_qubit_basis: str = "cz") -> QuantumCircuit:
    """Run TranslateToBasisGates so the circuit is in hardware submission state."""
    return TranslateToBasisGates(
        convert_single_qubit_gate_to_u=True,
        two_qubit_gate_basis=two_qubit_basis,
    ).run(qc)


def _lines(qcis: str) -> list[str]:
    return [line.strip().upper() for line in qcis.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Unitary-reconstruction oracle
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)


def _rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


# QCIS single-qubit native gates → 2x2 matrices.
_NATIVE_1Q = {
    "X2P": _rx(math.pi / 2),
    "X2M": _rx(-math.pi / 2),
    "Y2P": _ry(math.pi / 2),
    "Y2M": _ry(-math.pi / 2),
}


def _expand_1q(g: np.ndarray, k: int, n: int) -> np.ndarray:
    """Embed a 1-qubit gate on qubit ``k`` into an ``n``-qubit operator (q0 = MSB)."""
    op = np.array([[1.0 + 0j]])
    for q in range(n):
        op = np.kron(op, g if q == k else _I2)
    return op


def _cz(a: int, b: int, n: int) -> np.ndarray:
    """CZ between qubits ``a`` and ``b`` (symmetric) on an ``n``-qubit register."""
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]  # q0 = MSB
        if bits[a] and bits[b]:
            diag[idx] = -1.0
    return np.diag(diag)


def _native_unitary(qcis: str, n: int) -> np.ndarray:
    """Reconstruct the n-qubit unitary realized by an emitted QCIS instruction string."""
    U = np.eye(2 ** n, dtype=complex)
    for line in _lines(qcis):
        tok = line.split()
        name = tok[0]
        qubits = [int(t[1:]) for t in tok if t.startswith("Q")]
        args = [float(t) for t in tok[1:] if not t.startswith("Q")]
        if name in _NATIVE_1Q:
            M = _expand_1q(_NATIVE_1Q[name], qubits[0], n)
        elif name == "RZ":
            M = _expand_1q(_rz(args[0]), qubits[0], n)
        elif name == "CZ":
            M = _cz(qubits[0], qubits[1], n)
        elif name == "I":
            continue  # idle = identity
        elif name in ("M", "B", "RST"):
            continue  # non-unitary / structural — ignore for unitary comparison
        else:
            raise AssertionError(f"Unhandled QCIS native '{name}' in line: {line}")
        U = M @ U  # instructions apply left-to-right → left-multiply
    return U


def _equal_up_to_phase(A: np.ndarray, B: np.ndarray, atol: float = 1e-4) -> bool:
    """True if A == e^{iφ} B for some global phase φ. (RZ uses pi≈round(pi,6), hence loose atol.)"""
    idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
    if abs(B[idx]) < 1e-9:
        return False
    phase = A[idx] / B[idx]
    if abs(abs(phase) - 1.0) > 1e-3:
        return False
    return np.allclose(A, phase * B, atol=atol)


# Reference (intended) unitaries, defined independently with plain numpy.
_S2 = 1 / math.sqrt(2)
_REF_1Q = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    "sxdg": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex),
}


def _ref_u(theta, phi, lam):
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
    ], dtype=complex)


# ---------------------------------------------------------------------------
# Unitary-reconstruction tests (the real correctness guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate", list(_REF_1Q))
def test_single_qubit_rule_reconstructs_unitary(gate):
    """Each 1-qubit native rule must rebuild the gate's unitary (own rule, no translate)."""
    qc = QuantumCircuit(1)
    getattr(qc, gate)(0)
    U = _native_unitary(circuit_to_qcis(qc), 1)
    assert _equal_up_to_phase(U, _REF_1Q[gate]), f"{gate} QCIS decomposition is not equivalent to {gate}"


@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 4, math.pi / 2, 2.0, -1.1])
def test_rx_rule_reconstructs_unitary(theta):
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    assert _equal_up_to_phase(_native_unitary(circuit_to_qcis(qc), 1), _rx(theta))


@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 4, math.pi / 2, 2.0, -1.1])
def test_ry_rule_reconstructs_unitary(theta):
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    assert _equal_up_to_phase(_native_unitary(circuit_to_qcis(qc), 1), _ry(theta))


@pytest.mark.parametrize("theta", [0.3, math.pi / 4, math.pi / 2, 2.0, -1.1])
def test_rz_rule_reconstructs_unitary(theta):
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)
    assert _equal_up_to_phase(_native_unitary(circuit_to_qcis(qc), 1), _rz(theta))


@pytest.mark.parametrize("theta,phi,lam", [
    (0.7, 1.3, 2.1),
    (1.1, 0.3, -0.9),
    (2.4, -1.7, 0.5),
    (math.pi / 3, math.pi / 4, math.pi / 6),
    (0.0, 0.0, 0.0),
    (math.pi / 2, 0.0, math.pi),
])
def test_u_rule_reconstructs_unitary(theta, phi, lam):
    """Regression guard for the historical u arg-swap bug: U(θ,φ,λ) must round-trip."""
    qc = QuantumCircuit(1)
    qc.u(theta, phi, lam, 0)
    U = _native_unitary(circuit_to_qcis(qc), 1)
    assert _equal_up_to_phase(U, _ref_u(theta, phi, lam)), (
        f"u({theta},{phi},{lam}) QCIS decomposition is not U(θ,φ,λ)"
    )


@pytest.mark.parametrize("gate,ref", [
    ("cz", np.diag([1, 1, 1, -1]).astype(complex)),
    ("cx", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)),
    ("cy", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex)),
])
def test_two_qubit_rule_reconstructs_unitary(gate, ref):
    """cx/cy/cz native rules must rebuild the intended 2-qubit unitary (control q0, target q1)."""
    qc = QuantumCircuit(2)
    getattr(qc, gate)(0, 1)
    U = _native_unitary(circuit_to_qcis(qc), 2)
    assert _equal_up_to_phase(U, ref), f"{gate} QCIS decomposition is not equivalent"


def test_swap_rule_reconstructs_unitary():
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    ref = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    assert _equal_up_to_phase(_native_unitary(circuit_to_qcis(qc), 2), ref)


def test_ccx_rule_reconstructs_unitary():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    ref = np.eye(8, dtype=complex)
    ref[[6, 7]] = ref[[7, 6]]  # Toffoli: flip target when both controls are 1
    assert _equal_up_to_phase(_native_unitary(circuit_to_qcis(qc), 3), ref)


@pytest.mark.parametrize("gate", ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg"])
def test_translate_to_u_path_reconstructs_unitary(gate):
    """End-to-end hardware path: gate → TranslateToBasisGates (→ u) → QCIS must stay equivalent.

    This is the real GuoDun/TianYan submission path (convert_single_qubit_gate_to_u=True),
    and the one that exercises the fixed `u` rule for every single-qubit gate.
    """
    qc = QuantumCircuit(1)
    getattr(qc, gate)(0)
    U = _native_unitary(circuit_to_qcis(_translate(qc)), 1)
    assert _equal_up_to_phase(U, _REF_1Q[gate]), f"translate→u→QCIS path corrupts {gate}"


# ---------------------------------------------------------------------------
# 1-qubit non-parametric gates (all registered)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate", ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "id"])
def test_single_qubit_gate_produces_qcis(gate):
    qc = QuantumCircuit(1)
    getattr(qc, gate)(0)
    qcis = circuit_to_qcis(_translate(qc))
    assert qcis.strip(), f"Expected non-empty QCIS for gate={gate}"
    assert "Q0" in qcis.upper()


# ---------------------------------------------------------------------------
# 1-qubit parametric gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3])
def test_rx_produces_rz_sandwich(theta):
    # Skip translate so rx stays as rx (not converted to u)
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    from fieldqkit.circuit.quantumcircuit_helpers import one_qubit_parameter_gates_available
    assert "rx" in one_qubit_parameter_gates_available
    qcis = circuit_to_qcis(qc)  # rx is already an internal gate
    lines = _lines(qcis)
    # RX decomposes to Y2M, RZ, Y2P
    assert any("Y2M" in l for l in lines)
    assert any("RZ" in l for l in lines)
    assert any("Y2P" in l for l in lines)


@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3])
def test_ry_produces_rz_sandwich(theta):
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qcis = circuit_to_qcis(qc)
    lines = _lines(qcis)
    # RY decomposes to X2P, RZ, X2M
    assert any("X2P" in l for l in lines)
    assert any("RZ" in l for l in lines)
    assert any("X2M" in l for l in lines)


@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, -math.pi / 3])
def test_rz_produces_single_rz(theta):
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)
    qcis = circuit_to_qcis(qc)
    lines = _lines(qcis)
    rz_lines = [l for l in lines if l.startswith("RZ")]
    assert len(rz_lines) == 1


def test_u_gate_produces_five_instructions():
    qc = QuantumCircuit(1)
    qc.u(math.pi / 3, math.pi / 4, math.pi / 6, 0)
    qcis = circuit_to_qcis(_translate(qc))
    # U decomposes to RZ, X2P, RZ, X2M, RZ
    lines = _lines(qcis)
    assert len(lines) == 5


# ---------------------------------------------------------------------------
# 2-qubit gates — all basis choices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate,basis,expected_native", [
    ("cx", "cx", "Y2M"),   # CX stays as CX, which decomposes to Y2M+CZ+Y2P
    ("cz", "cz", "CZ"),
    ("cy", "cz", "CZ"),
    ("swap", "cx", "CZ"),
    ("iswap", "cz", "CZ"),
    ("ecr", "cx", "CZ"),
])
def test_two_qubit_gate_contains_expected_native(gate, basis, expected_native):
    qc = QuantumCircuit(2)
    getattr(qc, gate)(0, 1)
    translated = TranslateToBasisGates(
        convert_single_qubit_gate_to_u=True,
        two_qubit_gate_basis=basis,
    ).run(qc)
    qcis = circuit_to_qcis(translated)
    assert expected_native in qcis.upper(), (
        f"Expected '{expected_native}' in QCIS for gate={gate} basis={basis}:\n{qcis}"
    )


# ---------------------------------------------------------------------------
# 3-qubit gate: CCX (Toffoli)
# ---------------------------------------------------------------------------

def test_ccx_decomposes_to_cz():
    # CCX is handled directly by NativeQcisRules without the translate pass
    # (TranslateToBasisGates doesn't support CCX; a higher-level 3q decompose runs first).
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qcis = circuit_to_qcis(qc)
    assert "CZ" in qcis.upper()
    assert "Q0" in qcis.upper()
    assert "Q2" in qcis.upper()


# ---------------------------------------------------------------------------
# Functional gates
# ---------------------------------------------------------------------------

def test_measure_all_produces_m_per_qubit():
    qc = QuantumCircuit(3)
    # Need at least one gate so qubits are registered before measure_all()
    qc.h(0).h(1).h(2).measure_all()
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    lines = _lines(qcis)
    m_lines = [l for l in lines if l.startswith("M ")]
    assert len(m_lines) == 3


def test_reset_produces_rst():
    qc = QuantumCircuit(2)
    qc.reset(0)
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    assert "RST Q0" in qcis.upper()


def test_barrier_produces_b_instruction():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.barrier(0, 1, 2)
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    lines = _lines(qcis)
    assert any(l.startswith("B ") for l in lines)


def test_delay_single_qubit_produces_i():
    qc = QuantumCircuit(1)
    qc.delay(60, 0, unit="ns")
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    lines = _lines(qcis)
    # duration stored in seconds → circuit_to_qcis converts to nanoseconds
    assert any(l == "I Q0 60" for l in lines)


def test_delay_multi_qubit_produces_i_per_qubit():
    qc = QuantumCircuit(3)
    qc.delay(100, 0, 1, 2, unit="ns")
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    lines = _lines(qcis)
    i_lines = [l for l in lines if l.startswith("I ")]
    assert len(i_lines) == 3
    assert all(l.endswith(" 100") for l in i_lines)


# ---------------------------------------------------------------------------
# RZ ±π clamping (GuoDun rejects exact ±π)
# ---------------------------------------------------------------------------

def test_rz_at_pi_is_clamped():
    qc = QuantumCircuit(1)
    qc.rz(math.pi, 0)
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    for line in qcis.strip().splitlines():
        if "RZ" in line.upper():
            val = float(line.strip().split()[-1])
            assert -math.pi < val < math.pi


def test_rz_at_negative_pi_is_clamped():
    qc = QuantumCircuit(1)
    qc.rz(-math.pi, 0)
    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    for line in qcis.strip().splitlines():
        if "RZ" in line.upper():
            val = float(line.strip().split()[-1])
            assert -math.pi < val < math.pi


# ---------------------------------------------------------------------------
# Mixed-gate circuits
# ---------------------------------------------------------------------------

def test_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).measure_all()
    qcis = circuit_to_qcis(_translate(qc))
    lines = _lines(qcis)
    assert any("CZ" in l for l in lines)
    assert any(l == "M Q0" for l in lines)
    assert any(l == "M Q1" for l in lines)


def test_ghz_3():
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(0, 2).measure_all()
    qcis = circuit_to_qcis(_translate(qc))
    lines = _lines(qcis)
    cz_count = sum(1 for l in lines if "CZ" in l)
    assert cz_count >= 2
    assert any(l == "M Q2" for l in lines)


def test_parametric_circuit():
    qc = QuantumCircuit(3)
    qc.rx(math.pi / 7, 0)
    qc.ry(math.pi / 5, 1)
    qc.rz(math.pi / 3, 2)
    qc.cx(0, 1)
    qc.u(math.pi / 4, math.pi / 6, math.pi / 8, 0)
    qc.cz(1, 2)
    qc.measure_all()
    qcis = circuit_to_qcis(_translate(qc))
    assert qcis.strip()
    assert "CZ" in qcis.upper()


def test_circuit_with_barrier_and_reset():
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).barrier().reset(2).h(2).cx(1, 2).measure_all()
    qcis = circuit_to_qcis(_translate(qc))
    lines = _lines(qcis)
    assert any(l.startswith("B ") for l in lines)
    assert any(l.startswith("RST ") for l in lines)


# ---------------------------------------------------------------------------
# Unsupported gate raises (feature parity with old QASM path)
# ---------------------------------------------------------------------------

def test_unsupported_gate_raises():
    """Gates without a NativeQcisRules decomposition must raise NotImplementedError."""
    qc = QuantumCircuit(2)
    qc.iswap(0, 1)
    translated = TranslateToBasisGates(
        convert_single_qubit_gate_to_u=True,
        two_qubit_gate_basis="iswap",
    ).run(qc)
    with pytest.raises(NotImplementedError):
        circuit_to_qcis(translated)


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------

def test_empty_circuit_produces_empty_qcis():
    qc = QuantumCircuit(0)
    assert circuit_to_qcis(qc) == ""


def test_single_qubit_no_gates_produces_empty_qcis():
    qc = QuantumCircuit(1)
    assert circuit_to_qcis(qc) == ""


def test_identity_gate_produces_idle_instruction():
    qc = QuantumCircuit(1)
    qc.id(0)
    qcis = circuit_to_qcis(qc)
    lines = _lines(qcis)
    assert any(l.startswith("I Q0") for l in lines)


def test_instruction_str_formatting():
    from fieldqkit.circuit.qcis import Instruction
    assert str(Instruction("cz", [0, 1])) == "CZ Q0 Q1"
    assert str(Instruction("m", [3])) == "M Q3"
    assert str(Instruction("i", [2], [60])) == "I Q2 60"


def test_instruction_rz_clamps_exact_pi():
    from fieldqkit.circuit.qcis import Instruction
    s = str(Instruction("rz", [0], [math.pi]))
    val = float(s.split()[-1])
    assert -math.pi < val < math.pi


# ---------------------------------------------------------------------------
# Large-scale conversion
# ---------------------------------------------------------------------------

def test_wide_circuit_conversion_20_qubits():
    n = 20
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    translated = _translate(qc)
    qcis = circuit_to_qcis(translated)
    lines = _lines(qcis)

    m_lines = [l for l in lines if l.startswith("M ")]
    assert len(m_lines) == n
    # n-1 CX gates each decompose through one CZ
    cz_lines = [l for l in lines if l.startswith("CZ ")]
    assert len(cz_lines) == n - 1


def test_deep_circuit_conversion_many_rz():
    n = 4
    layers = 50
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for i in range(n):
            qc.rz(0.1, i)

    qcis = circuit_to_qcis(qc)
    lines = _lines(qcis)
    rz_lines = [l for l in lines if l.startswith("RZ ")]
    assert len(rz_lines) == layers * n


def test_measure_all_on_wide_circuit_emits_one_m_per_qubit():
    n = 16
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    qc.measure_all()

    qcis = circuit_to_qcis(_translate(qc))
    lines = _lines(qcis)
    m_lines = [l for l in lines if l.startswith("M ")]
    assert len(m_lines) == n


def test_deep_multi_qubit_delay_conversion():
    n = 10
    qc = QuantumCircuit(n)
    qc.delay(100, *range(n), unit="ns")
    qcis = circuit_to_qcis(_translate(qc))
    lines = _lines(qcis)
    i_lines = [l for l in lines if l.startswith("I ")]
    assert len(i_lines) == n
    assert all(l.endswith(" 100") for l in i_lines)
