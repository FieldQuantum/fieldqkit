import numpy as np

from quantum_hw.circuit.qasm2 import (
    parse_openqasm2_regs,
    parse_openqasm2_to_gates,
    parse_openqasm2_custom_gates,
)
from quantum_hw.circuit.qasm3 import parse_openqasm3_to_gates


def _find_gate(gates, name):
    return [g for g in gates if g[0] == name]


def test_qasm2_regs_and_custom_gate_strip():
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];
    creg c[2];
    gate mygate a,b { cx a,b; }
    mygate q[0],q[1];
    """
    qregs, cregs, cleaned = parse_openqasm2_regs(qasm)
    assert qregs == [("q", 2)]
    assert cregs == [("c", 2)]
    custom, stripped = parse_openqasm2_custom_gates(cleaned)
    assert "mygate" in custom
    assert "gate mygate" not in stripped


def test_qasm2_basic_parse():
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q[0] -> c[0];
    """
    gates, qubits, cbits = parse_openqasm2_to_gates(qasm)
    assert ("h", 0) in gates
    assert ("cx", 0, 1) in gates
    assert ("measure", [0], [0]) in gates
    assert qubits == {0, 1}
    assert cbits == {0}


def test_qasm2_multi_register_mapping():
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg a[1];
    qreg b[1];
    creg c[2];
    x a[0];
    x b[0];
    measure a[0] -> c[0];
    measure b[0] -> c[1];
    """
    gates, qubits, cbits = parse_openqasm2_to_gates(qasm)
    assert ("x", 0) in gates
    assert ("x", 1) in gates
    assert ("measure", [0], [0]) in gates
    assert ("measure", [1], [1]) in gates
    assert qubits == {0, 1}
    assert cbits == {0, 1}


def test_qasm2_params_delay_reset_barrier():
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];
    creg c[2];
    rx(pi/2) q[0];
    delay(1.5) q[1];
    reset q[1];
    barrier q[0],q[1];
    """
    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    rx_gate = _find_gate(gates, "rx")[0]
    assert np.isclose(rx_gate[1], np.pi / 2)
    assert ("delay", 1.5, (1,)) in gates
    assert ("reset", 1) in gates
    assert ("barrier", (0, 1)) in gates
    assert qubits == {0, 1}


def test_qasm3_basic_parse():
    qasm = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    """
    gates, qubits, cbits = parse_openqasm3_to_gates(qasm)
    assert ("h", 0) in gates
    assert ("cx", 0, 1) in gates
    assert ("measure", [0], [0]) in gates
    assert qubits == {0, 1}
    assert cbits == {0}


def test_qasm3_delay_reset_barrier():
    qasm = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] c;
    delay[5ns] q[0];
    reset q[1];
    barrier q[0], q[1];
    c[1] = measure q[1];
    """
    gates, qubits, cbits = parse_openqasm3_to_gates(qasm)
    assert _find_gate(gates, "delay")
    assert ("reset", 1) in gates
    assert ("barrier", (0, 1)) in gates
    assert ("measure", [1], [1]) in gates
    assert qubits == {0, 1}
    assert cbits == {1}


def test_qasm2_custom_gate_expansion():
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[1];
    gate g(theta) a { rz(theta) a; }
    g(pi/4) q[0];
    """
    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    rz_gate = _find_gate(gates, "rz")[0]
    assert np.isclose(rz_gate[1], np.pi / 4)
    assert qubits == {0}


def test_qasm3_custom_gate_and_param():
    qasm = """
    OPENQASM 3.0;
    qubit[1] q;
    gate g a { x a; }
    g q[0];
    rx(pi/4) q[0];
    """
    gates, qubits, _ = parse_openqasm3_to_gates(qasm)
    assert _find_gate(gates, "x")
    rx_gate = _find_gate(gates, "rx")[0]
    assert np.isclose(rx_gate[1], np.pi / 4)
    assert 0 in qubits


def test_qasm3_measurement_statement_variant():
    qasm = """
    OPENQASM 3.0;
    qubit[1] q;
    bit[1] c;
    c[0] = measure q[0];
    """
    gates, qubits, cbits = parse_openqasm3_to_gates(qasm)
    assert ("measure", [0], [0]) in gates
    assert qubits == {0}
    assert cbits == {0}
