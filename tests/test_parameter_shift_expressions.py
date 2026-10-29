"""Classical angle-expression Jacobians used by hardware parameter shift."""

import copy

import numpy as np
import pytest

from fieldqkit.algorithms._parameter_shift import parameter_shift_terms
from fieldqkit.circuit import QuantumCircuit


def _angle_term(expression, names, values):
    qc = QuantumCircuit(1)
    qc.rx(expression, 0)
    return parameter_shift_terms(qc, names, np.asarray(values, dtype=float))


@pytest.mark.parametrize(
    "expression,values,expected_value,expected_derivative",
    [
        ("theta", [0.4, 0.7], 0.4, [1.0, 0.0]),
        ("-theta + +alpha / 2", [0.4, 0.7], -0.05, [-1.0, 0.5]),
        ("2*theta + alpha*theta", [0.4, 0.7], 1.08, [2.7, 0.4]),
        ("theta / alpha", [0.4, 0.5], 0.8, [2.0, -1.6]),
        ("2 / theta", [0.5, 0.7], 4.0, [-8.0, 0.0]),
        ("theta - 2", [0.4, 0.7], -1.6, [1.0, 0.0]),
        ("2 - theta", [0.4, 0.7], 1.6, [-1.0, 0.0]),
        ("theta**3", [-2.0, 0.7], -8.0, [12.0, 0.0]),
        ("theta**1", [0.0, 0.7], 0.0, [1.0, 0.0]),
        ("theta**alpha", [2.0, 3.0], 8.0, [12.0, 8.0 * np.log(2.0)]),
        ("2**theta", [3.0, 0.7], 8.0, [8.0 * np.log(2.0), 0.0]),
        ("theta + pi + np.pi + π", [0.4, 0.7], 0.4 + 3*np.pi, [1.0, 0.0]),
    ],
)
def test_expression_chain_rule(expression, values, expected_value, expected_derivative):
    [(gate_index, angle_index, value, derivative)] = _angle_term(
        expression, ["theta", "alpha"], values,
    )
    assert (gate_index, angle_index) == (0, 1)
    assert value == pytest.approx(expected_value)
    assert derivative == pytest.approx(expected_derivative)


@pytest.mark.parametrize("expression", ["theta**2", "theta**0", "0*theta", "theta-theta"])
def test_zero_derivatives_are_omitted(expression):
    assert _angle_term(expression, ["theta"], [0.0]) == []


def test_gate_occurrences_u_slots_and_bound_constants_preserve_template():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx("theta", 0)
    qc.ry("theta", 1)
    qc.rz("fixed", 0)
    qc.rxx("2*theta", 0, 1)
    qc.ryy("theta-offset", 0, 1)
    qc.rzz("offset*theta", 0, 1)
    qc.u("theta", "alpha", "theta+alpha+fixed", 1)
    qc.params_value.update({"fixed": 0.25, "offset": 2.0, "theta": 100.0})
    before = copy.deepcopy(qc.__dict__)

    terms = parameter_shift_terms(qc, ["theta", "alpha"], np.array([0.4, 0.7]))

    assert [(g, a) for g, a, _, _ in terms] == [
        (1, 1), (2, 1), (4, 1), (5, 1), (6, 1), (7, 1), (7, 2), (7, 3),
    ]
    assert [value for _, _, value, _ in terms] == pytest.approx([
        0.4, 0.4, 0.8, -1.6, 0.8, 0.4, 0.7, 1.35,
    ])
    assert np.array([d for _, _, _, d in terms]) == pytest.approx(np.array([
        [1, 0], [1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [0, 1], [1, 1],
    ]))
    assert qc.__dict__ == before


def test_deep_binding_precedes_stale_full_expression_constant():
    qc = QuantumCircuit(1)
    qc.rx("2*theta", 0)
    qc.params_value["2*theta"] = 0.2
    [(gate_index, angle_index, value, derivative)] = parameter_shift_terms(
        qc, ["theta"], np.array([0.4]),
    )
    assert (gate_index, angle_index) == (0, 1)
    assert value == pytest.approx(0.8)
    assert derivative == pytest.approx([2.0])


def test_unresolved_expression_uses_full_expression_constant_before_symbols():
    qc = QuantumCircuit(1)
    qc.rx("theta+fixed", 0)
    qc.params_value.update({"theta+fixed": 0.2, "fixed": 0.3})
    assert parameter_shift_terms(qc, ["theta"], np.array([0.4])) == []


def test_unresolved_full_expression_override_precedes_subexpression_derivative():
    qc = QuantumCircuit(1)
    qc.ry("theta**0.5+fixed", 0)
    qc.params_value["theta**0.5+fixed"] = 0.2
    assert parameter_shift_terms(qc, ["theta"], np.array([0.0])) == []


def test_raw_fixed_parameter_label_uses_whole_expression_lookup():
    qc = QuantumCircuit(1)
    qc.ry("fixed angle", 0)
    qc.params_value["fixed angle"] = 0.2
    assert parameter_shift_terms(qc, ["theta"], np.array([0.4])) == []


def test_unresolved_expression_falls_back_to_fixed_symbols():
    qc = QuantumCircuit(1)
    qc.rx("theta+fixed", 0)
    qc.params_value["fixed"] = 0.3
    [(gate_index, angle_index, value, derivative)] = parameter_shift_terms(
        qc, ["theta"], np.array([0.4]),
    )
    assert (gate_index, angle_index) == (0, 1)
    assert value == pytest.approx(0.7)
    assert derivative == pytest.approx([1.0])


def test_undefined_derivative_does_not_fall_back_to_full_expression_constant():
    qc = QuantumCircuit(1)
    qc.rx("theta**0.5", 0)
    qc.params_value["theta**0.5"] = 0.2
    with pytest.raises(ValueError, match="angle 1 of gate 0"):
        parameter_shift_terms(qc, ["theta"], np.array([0.0]))


@pytest.mark.parametrize("name", ["angle 0", "pi", "π"])
def test_exact_raw_parameter_labels_take_precedence(name):
    [(gate_index, angle_index, value, derivative)] = _angle_term(name, [name], [0.4])
    assert (gate_index, angle_index) == (0, 1)
    assert value == pytest.approx(0.4)
    assert derivative == pytest.approx([1.0])


def test_pi_remains_a_constant_inside_composite_expressions():
    [(gate_index, angle_index, value, derivative)] = _angle_term(
        "theta+pi+π", ["theta", "pi", "π"], [0.4, 0.1, 0.2],
    )
    assert (gate_index, angle_index) == (0, 1)
    assert value == pytest.approx(0.4 + 2*np.pi)
    assert derivative == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "expression,values",
    [
        ("missing", [0.0]),
        ("theta / 0", [0.0]),
        ("1/theta", [0.0]),
        ("theta**-1", [0.0]),
        ("theta**0.5", [0.0]),
        ("theta**0.5", [-1.0]),
        ("(-2)**theta", [2.0]),
        ("theta*1e309", [1.0]),
        ("theta + 1j", [1.0]),
    ],
)
def test_invalid_or_undefined_expression_has_gate_context(expression, values):
    with pytest.raises(ValueError, match="angle 1 of gate 0"):
        _angle_term(expression, ["theta"], values)


@pytest.mark.parametrize(
    "names,values,message",
    [
        (["theta"], [], "matching param_names"),
        (["theta"], [[0.0]], "one-dimensional"),
        (["theta", "theta"], [0.0, 0.0], "unique"),
        ([0], [0.0], "strings"),
        ([None], [0.0], "strings"),
        (["theta"], [float("inf")], "finite"),
        (["theta"], [float("nan")], "finite"),
    ],
)
def test_parameter_mapping_validation(names, values, message):
    with pytest.raises(ValueError, match=message):
        _angle_term("theta", names, values)


def test_empty_parameter_vector_has_no_terms():
    qc = QuantumCircuit(1)
    qc.rx(0.4, 0)
    assert parameter_shift_terms(qc, [], np.array([])) == []
