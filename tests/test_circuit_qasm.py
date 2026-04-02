"""Tests for OpenQASM 2/3 parsing, custom gates, and circuit rendering helpers."""

import numpy as np

from quantum_hw.circuit.qasm2 import (
    parse_openqasm2_regs,
    parse_openqasm2_to_gates,
    parse_openqasm2_custom_gates,
)
from quantum_hw.circuit.qasm3 import parse_openqasm3_to_gates
from quantum_hw.circuit.quantumcircuit_helpers import (
    add_gates_to_lines,
    format_gates_layerd,
    parse_expression,
)
import pytest


def _find_gate(gates, name):
    return [g for g in gates if g[0] == name]


# ═══════════════════════════════════════════════════════════
#  QASM2 parsing
# ═══════════════════════════════════════════════════════════


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


def test_qasm2_custom_gate_parse_has_no_debug_stdout(capsys):
    qasm = """
    OPENQASM 2.0;
    include \"qelib1.inc\";
    qreg q[2];
    gate mygate a,b { cx a,b; }
    mygate q[0],q[1];
    """
    _, _, cleaned = parse_openqasm2_regs(qasm)
    parse_openqasm2_custom_gates(cleaned)
    captured = capsys.readouterr()
    assert captured.out == ""


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


# ═══════════════════════════════════════════════════════════
#  QASM3 parsing
# ═══════════════════════════════════════════════════════════


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
    delay_gate = _find_gate(gates, "delay")[0]
    assert np.isclose(delay_gate[1], 5e-9)
    assert ("reset", 1) in gates
    assert ("barrier", (0, 1)) in gates
    assert ("measure", [1], [1]) in gates
    assert qubits == {0, 1}
    assert cbits == {1}


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


# ═══════════════════════════════════════════════════════════
#  Rendering helpers
# ═══════════════════════════════════════════════════════════


def test_add_gates_to_lines_formats_mixed_parameter_tokens():
    gates = [
        ("u", "theta", "phi", "lam", 0),
        ("cp", "alpha", 0, 1),
        ("rz", "beta", 1),
    ]
    params_value = {
        "theta": np.pi / 2,
        "phi": "phi_expr",
        "alpha": np.pi,
        "beta": np.pi / 4,
    }

    lines, _ = add_gates_to_lines(2, 1, gates, params_value, width=1)
    rendered = "\n".join(lines)

    assert "U(" in rendered
    assert "phi_expr" in rendered
    assert "lam" in rendered
    assert "0.5" in rendered
    assert "0.25" in rendered
    assert "1.0" in rendered


def test_format_gates_layerd_normalizes_row_cell_widths():
    gates = [
        ("h", 0),
        ("cp", np.pi / 2, 0, 1),
        ("measure", [0], [0]),
    ]

    layers, _ = format_gates_layerd(2, 1, gates, params_value={})

    for row in layers[1:]:
        row_width = max(len(cell) for cell in row)
        assert row_width % 2 == 1
        assert all(len(cell) == row_width for cell in row)


def test_add_gates_to_lines_width_parameter_changes_output_spacing():
    gates = [
        ("h", 0),
        ("cx", 0, 1),
        ("rz", np.pi / 4, 1),
    ]

    lines_w1, _ = add_gates_to_lines(2, 1, gates, params_value={}, width=1)
    lines_w4, _ = add_gates_to_lines(2, 1, gates, params_value={}, width=4)

    assert len(lines_w1) == len(lines_w4)
    assert all(len(line4) > len(line1) for line1, line4 in zip(lines_w1, lines_w4))


def test_add_gates_to_lines_reports_only_used_qubit_rows():
    gates = [("h", 2)]

    lines, lines_use = add_gates_to_lines(3, 1, gates, params_value={}, width=1)

    assert set(lines_use) == {4, 5}
    assert "H" in lines[4]


def test_add_gates_to_lines_snapshot_small_circuit():
    gates = [
        ("h", 0),
        ("cx", 0, 1),
        ("rz", 0.5, 1),
        ("measure", [0], [0]),
    ]

    lines, _ = add_gates_to_lines(2, 1, gates, params_value={}, width=1)

    expected = [
        "q[0]  \u2500H\u2500\u25cf\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500M\u2500",
        "         \u2502         \u2502 ",
        "q[1]  \u2500\u2500\u2500X\u2500Rz(0.5)\u2500\u2502\u2500",
        "                   \u2502 ",
        "c:  1/\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550",
        "                   0 ",
    ]

    assert lines == expected
