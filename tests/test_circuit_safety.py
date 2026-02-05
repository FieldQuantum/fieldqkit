import pytest
import numpy as np

from quantum_hw.circuit.quantumcircuit_helpers import parse_expression
from quantum_hw.circuit import QuantumCircuit


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

    qc.shallow_apply_value({"theta": 0.1, "phi": -0.2})
    assert qc.params_value["theta"] == 0.1
    assert qc.params_value["phi"] == -0.2
    assert qc.params_value["gamma"] == "gamma"

    qc.deep_apply_value({"theta": 0.3, "phi": -0.4, "gamma": 0.5})
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
