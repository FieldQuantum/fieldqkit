"""Tests for OpenQASM 2 parsing, custom gates, and circuit rendering helpers."""

import numpy as np

from fieldqkit.circuit.qasm2 import (
    parse_openqasm2_regs,
    parse_openqasm2_to_gates,
    parse_openqasm2_custom_gates,
    generate_reg_map,
)
from fieldqkit.circuit.quantumcircuit_helpers import (
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
#  Rendering helpers
# ═══════════════════════════════════════════════════════════


def test_add_gates_to_lines_formats_mixed_parameter_tokens():
    gates = [
        ("u", "theta", "phi", "lam", 0),
        ("rz", "alpha", 0),
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
        ("rzz", np.pi / 2, 0, 1),
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


# ═══════════════════════════════════════════════════════════
#  QASM2 boundary cases
# ═══════════════════════════════════════════════════════════


def test_qasm2_program_with_no_instructions_yields_no_gates():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    """
    gates, qubits, cbits = parse_openqasm2_to_gates(qasm)
    assert gates == []
    assert qubits == set()
    assert cbits == set()


def test_qasm2_single_gate_single_qubit():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    h q[0];
    """
    gates, qubits, cbits = parse_openqasm2_to_gates(qasm)
    assert gates == [("h", 0)]
    assert qubits == {0}


def test_qasm2_cnot_alias_maps_to_cx():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    cnot q[0],q[1];
    """
    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    assert gates == [("cx", 0, 1)]
    assert qubits == {0, 1}


def test_qasm2_u1_u2_p_phase_aliases_expand_to_u():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u1(0.5) q[0];
    p(0.3) q[0];
    u2(0.1,0.2) q[0];
    """
    gates, _, _ = parse_openqasm2_to_gates(qasm)
    assert gates[0] == ("u", 0, 0, 0.5, 0)
    assert gates[1] == ("u", 0, 0, 0.3, 0)
    assert gates[2][0] == "u"
    assert gates[2][1] == pytest.approx(np.pi / 2)


def test_qasm2_comment_line_is_ignored():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    h q[0]; // a trailing comment
    x q[0];
    """
    gates, _, _ = parse_openqasm2_to_gates(qasm)
    assert gates == [("h", 0), ("x", 0)]


# ═══════════════════════════════════════════════════════════
#  QASM2 custom-gate handling
# ═══════════════════════════════════════════════════════════


def test_qasm2_multiple_custom_gates_parsed_and_expanded():
    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
gate g1 a { h a; }
gate g2 a,b { cx a,b; }
g1 q[0];
g2 q[0],q[1];
""".strip()
    _, _, cleaned = parse_openqasm2_regs(qasm)
    custom, stripped = parse_openqasm2_custom_gates(cleaned)
    assert sorted(custom.keys()) == ["g1", "g2"]
    assert "gate g1" not in stripped
    assert "gate g2" not in stripped

    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    assert gates == [("h", 0), ("cx", 0, 1)]
    assert qubits == {0, 1}


def test_qasm2_unknown_gate_raises_value_error():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    notagate q[0];
    """
    with pytest.raises(ValueError):
        parse_openqasm2_to_gates(qasm)


# ═══════════════════════════════════════════════════════════
#  Large-scale QASM parsing & round-trips
# ═══════════════════════════════════════════════════════════


def test_qasm2_wide_register_parse_30_qubits():
    n = 30
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    for i in range(n):
        lines.append(f"h q[{i}];")
    qasm = "\n".join(lines)

    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    assert len(gates) == n
    assert qubits == set(range(n))
    assert all(g[0] == "h" for g in gates)


def test_qasm2_large_register_map_is_dense():
    qasm = "OPENQASM 2.0;\nqreg q[40];\ncreg c[40];"
    qregs, cregs, _ = parse_openqasm2_regs(qasm)
    assert qregs == [("q", 40)]
    assert cregs == [("c", 40)]
    reg_map = generate_reg_map(qregs)
    assert len(reg_map["q"]) == 40
    assert reg_map["q"][0] == 0
    assert reg_map["q"][39] == 39


def test_qasm2_deep_circuit_parse_many_gates():
    n = 5
    layers = 60
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    for _ in range(layers):
        for i in range(n):
            lines.append(f"rz(0.25) q[{i}];")
        for i in range(n - 1):
            lines.append(f"cx q[{i}],q[{i + 1}];")
    qasm = "\n".join(lines)

    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    rz_gates = _find_gate(gates, "rz")
    cx_gates = _find_gate(gates, "cx")
    assert len(rz_gates) == layers * n
    assert len(cx_gates) == layers * (n - 1)
    assert qubits == set(range(n))


def test_qasm2_multi_register_dense_remap_three_registers():
    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[2];
qreg d[1];
creg c[5];
x a[1];
x b[0];
x d[0];
""".strip()
    gates, qubits, _ = parse_openqasm2_to_gates(qasm)
    # a -> 0,1 ; b -> 2,3 ; d -> 4
    assert ("x", 1) in gates
    assert ("x", 2) in gates
    assert ("x", 4) in gates
    assert qubits == {1, 2, 4}


# ═══════════════════════════════════════════════════════════
#  Rendering helpers — large/boundary
# ═══════════════════════════════════════════════════════════


def test_add_gates_to_lines_wide_circuit_has_one_h_row_per_qubit():
    n = 20
    gates = [("h", i) for i in range(n)]

    lines, _ = add_gates_to_lines(n, n, gates, params_value={}, width=1)

    # Each qubit occupies its own (even-indexed) wire row carrying its H gate.
    h_rows = [ln for ln in lines if "H" in ln and "q[" in ln]
    assert len(h_rows) == n
    assert all("H" in lines[2 * i] for i in range(n))


def test_format_gates_layerd_handles_empty_gate_list():
    layers, _ = format_gates_layerd(2, 1, [], params_value={})
    assert isinstance(layers, list)