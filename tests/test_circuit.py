"""Tests for the circuit module: QuantumCircuit core, parameter binding, rendering, and safety checks."""

import pytest
import numpy as np

from fieldqkit.circuit import QuantumCircuit
from fieldqkit.circuit.qasm2 import parse_openqasm2_to_gates
from fieldqkit.circuit.quantumcircuit_helpers import parse_expression
from fieldqkit.circuit.render import draw_circuit, draw_circuit_simply


# ═══════════════════════════════════════════════════════════
#  Expression parsing
# ═══════════════════════════════════════════════════════════


def test_parse_expression_safe_math():
    assert parse_expression("pi/2") == pytest.approx(np.pi / 2)
    assert parse_expression("2*pi") == pytest.approx(2 * np.pi)
    assert parse_expression("np.pi/4") == pytest.approx(np.pi / 4)
    assert parse_expression("3") == pytest.approx(3.0)


def test_parse_expression_rejects_unsafe():
    with pytest.raises(ValueError):
        parse_expression("__import__('os').system('echo 1')")


def test_parse_expression_complex_math():
    assert parse_expression("pi/2") == pytest.approx(np.pi / 2)
    assert parse_expression("2*pi**2") == pytest.approx(2 * np.pi**2)
    assert parse_expression("-3*pi/4") == pytest.approx(-3 * np.pi / 4)
    with pytest.raises(ValueError):
        parse_expression("os.system('echo bad')")


# ═══════════════════════════════════════════════════════════
#  Parameter binding
# ═══════════════════════════════════════════════════════════


def test_parameter_binding_behaviour():
    qc = QuantumCircuit(2, 2)
    qc.rx("theta", 0)
    qc.rz("phi", 1)
    qc.rzz("gamma", 0, 1)

    qc.apply_value({"theta": 0.1, "phi": -0.2})
    assert qc.params_value["theta"] == 0.1
    assert qc.params_value["phi"] == -0.2
    assert qc.params_value["gamma"] == "gamma"

    qc.apply_value({"theta": 0.3, "phi": -0.4, "gamma": 0.5}, deep=True)
    assert qc.params_value["theta"] == 0.3
    assert qc.params_value["phi"] == -0.4
    assert qc.params_value["gamma"] == 0.5

    for gate in qc.gates:
        if gate[0] in {"rx", "rz", "rzz"}:
            assert isinstance(gate[1], (float, int))

    qasm = qc.to_openqasm2()
    assert "theta" not in qasm
    assert "phi" not in qasm
    assert "gamma" not in qasm


def test_parameter_expression_binding_with_negative_symbol():
    qc = QuantumCircuit(1, 1)
    qc.ry("theta", 0)
    qc.ry("-theta", 0)

    qc.apply_value({"theta": 0.25})
    qasm = qc.to_openqasm2()

    assert "ry(0.25) q[0];" in qasm
    assert "ry(-0.25) q[0];" in qasm


def test_parameter_expression_binding_with_division_symbol():
    qc = QuantumCircuit(1, 1)
    qc.rx("theta/2", 0)

    qc.apply_value({"theta": np.pi})
    qasm = qc.to_openqasm2()

    assert f"rx({np.pi / 2}) q[0];" in qasm


def test_apply_value_deep_materializes_general_expression_params():
    qc = QuantumCircuit(1, 1)
    qc.ry("-theta", 0)
    qc.rx("theta/2", 0)

    qc.apply_value({"theta": np.pi}, deep=True)

    assert qc.gates[0] == ("ry", -np.pi, 0)
    assert qc.gates[1] == ("rx", np.pi / 2, 0)


# ═══════════════════════════════════════════════════════════
#  Gate registration & chaining
# ═══════════════════════════════════════════════════════════


def test_u_gate_parameter_registration_behaviour():
    qc = QuantumCircuit(2, 2)
    ret = qc.u("theta", "phi", "lam", 1)

    assert ret is qc
    assert qc.gates == [("u", "theta", "phi", "lam", 1)]
    assert qc.params_value == {"theta": "theta", "phi": "phi", "lam": "lam"}


def test_mutating_methods_return_self_for_chaining():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.x(2)

    ret_delay = qc.delay(1e-6, 0, 2)
    ret_barrier = qc.barrier(0, 2)
    ret_measure = qc.measure([0, 2], [0, 1])
    ret_measure_all = qc.measure_all()

    assert ret_delay is qc
    assert ret_barrier is qc
    assert ret_measure is qc
    assert ret_measure_all is qc


def test_adjust_index_preserves_all_gate_kinds_and_offsets():
    qc = QuantumCircuit(4, 4)
    qc.ccz(0, 1, 2)
    qc.rz("theta", 1)
    qc.delay(2e-6, 0, 3)
    qc.barrier(0, 1, 2, 3)
    qc.measure([0, 3], [1, 2])

    returned = qc.adjust_index(2, cbit_offset=5)

    assert returned is qc
    assert qc.nqubits == 6
    assert qc.ncbits == 9
    assert qc.qubits == [2, 3, 4, 5]
    assert qc.gates == [
        ("ccz", 2, 3, 4),
        ("rz", "theta", 3),
        ("delay", 2e-6, (2, 5)),
        ("barrier", (2, 3, 4, 5)),
        ("measure", [2, 5], [6, 7]),
    ]


# ═══════════════════════════════════════════════════════════
#  OpenQASM export
# ═══════════════════════════════════════════════════════════


def test_openqasm2_export_contains_expected_lines():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 1)
    qc.measure([0, 1], [0, 1])

    qasm = qc.to_openqasm2()
    assert qasm.startswith("OPENQASM 2.0;")
    assert "qreg q[2];" in qasm
    assert "creg c[2];" in qasm
    assert "measure q[0] -> c[0];" in qasm
    assert "measure q[1] -> c[1];" in qasm


# ═══════════════════════════════════════════════════════════
#  OpenQASM import
# ═══════════════════════════════════════════════════════════


def test_from_openqasm_qubits_are_sorted_and_deterministic():
    qasm2 = """
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[3];
creg c[3];
h q[2];
x q[0];
measure q[2] -> c[2];
measure q[0] -> c[0];
""".strip()
    qc2 = QuantumCircuit().from_openqasm2(qasm2)
    assert qc2.qubits == [0, 2]


def test_openqasm_header_validation_raises_value_error():
    with pytest.raises(ValueError, match="OpenQASM 2.0"):
        QuantumCircuit().from_openqasm2("OPENQASM 3.0;\nqubit[1] q;")


# ═══════════════════════════════════════════════════════════
#  Mapping & rendering
# ═══════════════════════════════════════════════════════════


def test_mapping_to_others_validates_mapping_size():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    with pytest.raises(ValueError, match="Mapping size"):
        qc.mapping_to_others({0: 2})


def test_to_latex_is_explicitly_not_implemented():
    qc = QuantumCircuit(1, 1)
    with pytest.raises(NotImplementedError, match="not implemented"):
        _ = qc.to_latex


def test_render_helpers_smoke():
    lines = ["q[0]──H──", "q[1]────", "c:  2/ "]
    draw_circuit(lines)
    draw_circuit_simply(lines, [0, 1], 1)


# ═══════════════════════════════════════════════════════════
#  Stdout / side-effect checks
# ═══════════════════════════════════════════════════════════


def test_parse_openqasm2_to_gates_has_no_stdout_for_multi_registers(capsys):
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
""".strip()
    parse_openqasm2_to_gates(qasm)
    captured = capsys.readouterr()
    assert captured.out == ""


# ═══════════════════════════════════════════════════════════
#  Boundary cases
# ═══════════════════════════════════════════════════════════


def test_empty_circuit_exports_zero_registers():
    qc = QuantumCircuit()
    assert qc.nqubits == 0
    assert qc.ncbits == 0
    assert qc.gates == []
    assert qc.qubits == []

    qasm = qc.to_openqasm2()
    assert qasm.startswith("OPENQASM 2.0;")
    assert "qreg q[0];" in qasm
    assert "creg c[0];" in qasm


def test_single_qubit_default_ncbits_matches_nqubits():
    qc = QuantumCircuit(1)
    assert qc.nqubits == 1
    assert qc.ncbits == 1


def test_single_gate_circuit_records_one_gate():
    qc = QuantumCircuit(1, 1)
    ret = qc.h(0)
    assert ret is qc
    assert qc.gates == [("h", 0)]
    assert qc.qubits == [0]


def test_constructor_rejects_more_than_two_args():
    with pytest.raises(ValueError):
        QuantumCircuit(1, 2, 3)


def test_out_of_range_qubit_index_raises():
    qc = QuantumCircuit(2, 2)
    with pytest.raises(ValueError, match="out of range"):
        qc.h(2)
    with pytest.raises(ValueError, match="out of range"):
        qc.cx(0, 5)


def test_duplicate_qubit_args_raise_conflict():
    qc = QuantumCircuit(2, 2)
    with pytest.raises(ValueError, match="conflict"):
        qc.cx(1, 1)
    with pytest.raises(ValueError, match="conflict"):
        qc.swap(0, 0)


def test_ccx_duplicate_indices_raise():
    qc = QuantumCircuit(3, 3)
    with pytest.raises(ValueError, match="conflict"):
        qc.ccx(0, 1, 0)


def test_measure_mismatched_lengths_raise():
    qc = QuantumCircuit(2, 2)
    with pytest.raises(ValueError):
        qc.measure([0, 1], [0])


def test_barrier_duplicate_qubits_raise():
    qc = QuantumCircuit(3, 3)
    with pytest.raises(ValueError):
        qc.barrier(0, 0, 1)


def test_remove_gate_and_remove_barrier_are_noop_on_empty():
    qc = QuantumCircuit(1, 1)
    assert qc.remove_barrier() is qc
    assert qc.gates == []
    assert qc.remove_gate("h") is qc
    assert qc.gates == []


# ═══════════════════════════════════════════════════════════
#  Counting / introspection helpers
# ═══════════════════════════════════════════════════════════


def test_count_gate_ncz_and_qubits_in_use():
    qc = QuantumCircuit(3, 3)
    qc.h(0).h(1).cx(0, 1).rzz(0.1, 1, 2)

    assert qc.count_gate("h") == 2
    assert qc.count_gate("cx") == 1
    assert qc.count_gate("nonexistent") == 0
    # cx + rzz both count as two-qubit gates
    assert qc.ncz == 2
    assert qc.qubits_in_use == [0, 1, 2]


def test_remove_barrier_and_remove_gate_filter_named_gates():
    qc = QuantumCircuit(2, 2)
    qc.h(0).barrier(0, 1).x(1).barrier(0, 1)
    assert qc.count_gate("barrier") == 2

    qc.remove_barrier()
    assert qc.count_gate("barrier") == 0
    assert qc.gates == [("h", 0), ("x", 1)]

    qc.remove_gate("h")
    assert qc.gates == [("x", 1)]


# ═══════════════════════════════════════════════════════════
#  Large-scale construction
# ═══════════════════════════════════════════════════════════


def test_wide_circuit_construction_50_qubits():
    n = 50
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    assert qc.nqubits == n
    assert len(qc.qubits) == n
    assert qc.count_gate("h") == n
    assert qc.count_gate("cx") == n - 1


def test_deep_circuit_construction_hundreds_of_gates():
    n = 6
    layers = 100
    qc = QuantumCircuit(n, n)
    for d in range(layers):
        for i in range(n):
            qc.rz(0.01 * d, i)
        for i in range(n - 1):
            qc.cx(i, i + 1)

    assert len(qc.gates) == layers * (n + (n - 1))
    assert qc.count_gate("rz") == layers * n
    assert qc.count_gate("cx") == layers * (n - 1)


def test_wide_circuit_openqasm_export_has_one_line_per_gate():
    n = 30
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)

    qasm = qc.to_openqasm2()
    h_lines = [ln for ln in qasm.splitlines() if ln.startswith("h q[")]
    assert len(h_lines) == n
    assert f"qreg q[{n}];" in qasm


def test_adjust_index_large_offset_shifts_every_index():
    n = 10
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)

    qc.adjust_index(100)

    assert qc.nqubits == n + 100
    assert qc.qubits == list(range(100, 100 + n))
    assert qc.gates[0] == ("h", 100)
    assert qc.gates[-1] == ("h", 100 + n - 1)


def test_mapping_to_others_identity_preserves_gates():
    qc = QuantumCircuit(3, 3)
    qc.h(0).cx(0, 1).cx(1, 2)
    original = list(qc.gates)

    qc.mapping_to_others({0: 0, 1: 1, 2: 2})

    assert qc.gates == original


def test_deepcopy_is_independent():
    a = QuantumCircuit(2, 2)
    a.h(0)
    b = a.deepcopy()
    b.x(1)

    assert len(a.gates) == 1
    assert len(b.gates) == 2
    assert a.gates == [("h", 0)]


# ═══════════════════════════════════════════════════════════
#  OpenQASM round-trip invariants (large)
# ═══════════════════════════════════════════════════════════


def test_openqasm_roundtrip_idempotent_when_all_qubits_used():
    n = 8
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.rz(0.1 * i, i)
    qc.measure(list(range(n)), list(range(n)))

    s1 = qc.to_openqasm2()
    qc2 = QuantumCircuit().from_openqasm2(s1)
    s2 = qc2.to_openqasm2()

    assert s1 == s2


def test_openqasm_roundtrip_preserves_gate_semantics_large():
    n = 12
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)

    s1 = qc.to_openqasm2()
    qc2 = QuantumCircuit().from_openqasm2(s1)

    assert qc2.count_gate("h") == n
    assert qc2.count_gate("cx") == n // 2
    assert qc2.to_openqasm2() == s1


def test_symbolic_openqasm_keeps_unbound_parameters():
    qc = QuantumCircuit(1, 1)
    qc.rx("theta", 0)
    qasm = qc.to_openqasm2(symbolic=True)
    assert "rx(theta) q[0];" in qasm
