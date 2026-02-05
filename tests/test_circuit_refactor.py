import pytest

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.qasm2 import parse_openqasm2_to_gates
from quantum_hw.circuit.qasm3 import parse_openqasm3_to_gates
from quantum_hw.circuit.render import draw_circuit, draw_circuit_simply


def test_openqasm2_export_contains_expected_lines():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 1)
    qc.measure([0, 1], [0, 1])

    qasm = qc.to_openqasm2
    assert qasm.startswith("OPENQASM 2.0;")
    assert "qreg q[2];" in qasm
    assert "creg c[2];" in qasm
    assert "measure q[0] -> c[0];" in qasm
    assert "measure q[1] -> c[1];" in qasm


def test_openqasm3_export_contains_expected_lines():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 1)
    qc.delay(1e-6, 0)
    qc.barrier(0, 1)
    qc.measure([0, 1], [0, 1])

    qasm = qc.to_openqasm3
    assert qasm.startswith("OPENQASM 3.0;")
    assert "qubit[2] q;" in qasm
    assert "bit[2] c;" in qasm
    assert "c[0] = measure q[0];" in qasm
    assert "c[1] = measure q[1];" in qasm
    assert "delay[" in qasm


def test_render_helpers_smoke():
    lines = ["q[0]──H──", "q[1]────", "c:  2/ "]
    draw_circuit(lines)
    draw_circuit_simply(lines, [0, 1], 1)


def test_parse_openqasm2_and_qasm3_modules():
    qasm2 = """
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
h q[0];
rx(pi/2) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
""".strip()
    gates2, qubits2, cbits2 = parse_openqasm2_to_gates(qasm2)
    assert qubits2 == {0, 1}
    assert cbits2 == {0, 1}
    assert any(g[0] == "rx" for g in gates2)

    qasm3 = """
OPENQASM 3.0;
include \"stdgates.inc\";
qubit[2] q;
bit[2] c;
h q[0];
rx(pi/2) q[1];
barrier q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
""".strip()
    gates3, qubits3, cbits3 = parse_openqasm3_to_gates(qasm3)
    assert qubits3 == {0, 1}
    assert cbits3 == {0, 1}
    assert any(g[0] == "barrier" for g in gates3)
