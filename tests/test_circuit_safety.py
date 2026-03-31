import pytest
import numpy as np

from quantum_hw.circuit.quantumcircuit_helpers import parse_expression
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.qasm2 import parse_openqasm2_to_gates


def test_parse_expression_safe_math():
    assert parse_expression("pi/2") == pytest.approx(np.pi / 2)
    assert parse_expression("2*pi") == pytest.approx(2 * np.pi)
    assert parse_expression("np.pi/4") == pytest.approx(np.pi / 4)
    assert parse_expression("3") == pytest.approx(3.0)


def test_parse_expression_rejects_unsafe():
    with pytest.raises(ValueError):
        parse_expression("__import__('os').system('echo 1')")


def test_parameter_binding_behaviour():
    qc = QuantumCircuit(2, 2)
    qc.rx("theta", 0)
    qc.rz("phi", 1)
    qc.cp("gamma", 0, 1)

    qc.apply_value({"theta": 0.1, "phi": -0.2})
    assert qc.params_value["theta"] == 0.1
    assert qc.params_value["phi"] == -0.2
    assert qc.params_value["gamma"] == "gamma"

    qc.apply_value({"theta": 0.3, "phi": -0.4, "gamma": 0.5}, deep=True)
    assert qc.params_value["theta"] == 0.3
    assert qc.params_value["phi"] == -0.4
    assert qc.params_value["gamma"] == 0.5

    for gate in qc.gates:
        if gate[0] in {"rx", "rz"}:
            assert isinstance(gate[1], (float, int))
        if gate[0] == "cp":
            assert isinstance(gate[1], (float, int))

    qasm = qc.to_openqasm2
    assert "theta" not in qasm
    assert "phi" not in qasm
    assert "gamma" not in qasm


def test_parameter_expression_binding_with_negative_symbol():
    qc = QuantumCircuit(1, 1)
    qc.ry("theta", 0)
    qc.ry("-theta", 0)

    qc.apply_value({"theta": 0.25})
    qasm = qc.to_openqasm2

    assert "ry(0.25) q[0];" in qasm
    assert "ry(-0.25) q[0];" in qasm


def test_parameter_expression_binding_with_division_symbol():
    qc = QuantumCircuit(1, 1)
    qc.rx("theta/2", 0)

    qc.apply_value({"theta": np.pi})
    qasm = qc.to_openqasm2

    assert f"rx({np.pi / 2}) q[0];" in qasm


def test_apply_value_deep_materializes_general_expression_params():
    qc = QuantumCircuit(1, 1)
    qc.ry("-theta", 0)
    qc.rx("theta/2", 0)

    qc.apply_value({"theta": np.pi}, deep=True)

    assert qc.gates[0] == ("ry", -np.pi, 0)
    assert qc.gates[1] == ("rx", np.pi / 2, 0)


def test_adjust_index_preserves_all_gate_kinds_and_offsets():
    qc = QuantumCircuit(4, 4)
    qc.ccz(0, 1, 2)
    qc.cp("theta", 1, 3)
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
        ("cp", "theta", 3, 5),
        ("delay", 2e-6, (2, 5)),
        ("barrier", (2, 3, 4, 5)),
        ("measure", [2, 5], [6, 7]),
    ]


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

    qasm3 = """
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
h q[2];
x q[0];
c[2] = measure q[2];
c[0] = measure q[0];
""".strip()
    qc3 = QuantumCircuit().from_openqasm3(qasm3)
    assert qc3.qubits == [0, 2]


def test_openqasm_header_validation_raises_value_error():
    with pytest.raises(ValueError, match="OpenQASM 2.0"):
        QuantumCircuit().from_openqasm2("OPENQASM 3.0;\nqubit[1] q;")
    with pytest.raises(ValueError, match="OpenQASM 3.0"):
        QuantumCircuit().from_openqasm3("OPENQASM 2.0;\nqreg q[1];")


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
