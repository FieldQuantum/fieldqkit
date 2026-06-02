"""Tests for circuit_to_qcis: QuantumCircuit → QCIS direct conversion.

Verifies that circuit_to_qcis produces correct QCIS output for all
supported gate types, covering all registered internal gates and functional
instructions.
"""

import math
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
