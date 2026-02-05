import numpy as np
import pytest

from quantum_hw.circuit.qasm2 import parse_openqasm2_to_gates
from quantum_hw.circuit.quantumcircuit_helpers import parse_expression


def test_openqasm_custom_gate_and_multi_registers():
    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[1];
creg c[3];

gate mygate(theta) q0,q1 { cx q0,q1; rz(theta) q0; }

mygate(pi/2) a[0], b[0];
rx(pi) a[1];
measure a[1] -> c[2];
""".strip()

    gates, qubits, cbits = parse_openqasm2_to_gates(qasm)
    assert qubits == {0, 1, 2}
    assert cbits == {2}

    assert gates[0] == ("cx", 0, 2)
    assert gates[1][0] == "rz"
    assert gates[1][2] == 0
    assert gates[1][1] == pytest.approx(np.pi / 2)
    assert gates[2] == ("rx", pytest.approx(np.pi), 1)
    assert gates[3] == ("measure", [1], [2])


def test_parse_expression_complex_math():
    assert parse_expression("pi/2") == pytest.approx(np.pi / 2)
    assert parse_expression("2*pi**2") == pytest.approx(2 * np.pi**2)
    assert parse_expression("-3*pi/4") == pytest.approx(-3 * np.pi / 4)
    with pytest.raises(ValueError):
        parse_expression("os.system('echo bad')")
