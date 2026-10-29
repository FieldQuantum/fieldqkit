"""Exact regressions for hardware gradients with shared circuit parameters.

The client keeps the real transpilation and energy-evaluation paths; only shot
sampling is replaced by exact statevector expectations, so gradient failures
cannot be hidden by statistical tolerances.
"""

from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fieldqkit.algorithms.optimizer_utils import (
    instantiate_transpiled_template,
    parameter_shift_gradient,
)
from fieldqkit.algorithms.qaoa import (
    build_maxcut_hamiltonian,
    build_qaoa_ansatz_symbolic,
    run_qaoa_with_backend,
)
from fieldqkit.algorithms.vqe import run_vqe_with_backend
from fieldqkit.api.backend import Backend
from fieldqkit.api.client import QuantumHardwareClient
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.core.types import RunResult
from fieldqkit.sim.statevector import (
    energy_and_expectations,
    expectation_pauli,
    simulate_statevector,
)


class _ExactClient(QuantumHardwareClient):
    def __init__(self):
        super().__init__()
        self.evaluations = []

    def _run_with_backend(self, qc, name, num_qubits, **kwargs):
        self.evaluations.append((qc.deepcopy(), dict(kwargs)))
        targets = kwargs.get("target_qubits")
        if targets is not None:
            qc, _ = self._compact_for_sim(qc, targets)
        state = simulate_statevector(qc, device="cpu")
        values = {
            obs: float(expectation_pauli(state, obs, num_qubits=num_qubits).real)
            for obs in kwargs["observables"]
        }
        return RunResult(
            task_ids=None, samples=None, samples_zne=None, probabilities=None,
            probabilities_raw=None, observable_values=values,
            observable_values_raw=values,
        )


def _autograd(template, names, params, hamiltonian):
    values = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    energy, _ = energy_and_expectations(
        template, params=values, param_names=names,
        hamiltonian=hamiltonian, device="cpu",
    )
    energy.backward()
    return values.grad.detach().numpy()


def _gradient(client, template, names, params, hamiltonian, **kwargs):
    options = dict(
        name="exact_gradient", num_qubits=template.nqubits,
        backend=Backend("Simulator"), chip_name="Simulator", shots=128,
        hamiltonian=hamiltonian, shift=np.pi / 2,
        zne=False, readout_mitigation=False,
        param_template=template, param_names=names,
        convert_single_qubit_gate_to_u=False,
    )
    options.update(kwargs)
    return parameter_shift_gradient(client, np.asarray(params, dtype=float), **options)


@pytest.mark.parametrize("beta", [0.0, 0.7])
@pytest.mark.parametrize("transpile", [False, True])
def test_one_edge_qaoa_gradient_matches_closed_form(beta, transpile):
    client = _ExactClient()
    backend = Backend("Simulator")
    names, symbolic = build_qaoa_ansatz_symbolic(2, [(0, 1)], p=1)
    original = deepcopy(symbolic.__dict__)
    template = symbolic
    targets = [0, 1]
    if transpile:
        # Exercise actual translation of RZZ and RX into the backend basis.
        template = client._transpile_with_backend(
            symbolic, backend, use_dd=False, use_gate_compressor=False,
            convert_single_qubit_gate_to_u=True,
        )
        targets = client._ordered_target_qubits_from_layout(
            compiled_qc=template, original_qc=symbolic, num_qubits=2,
        )
    before = deepcopy(template.__dict__)
    gamma = 0.4
    actual = _gradient(
        client, template, names, [gamma, beta], [(0.5, "ZZ")],
        target_qubits=targets,
    )
    # E(gamma,beta) = sin(gamma) sin(2 beta) / 2 for this ansatz.
    expected = [
        0.5 * np.cos(gamma) * np.sin(2 * beta),
        np.sin(gamma) * np.cos(2 * beta),
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert abs(actual[1]) > 0.05
    assert symbolic.__dict__ == original
    assert template.__dict__ == before


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("transpile", [False, True])
def test_triangle_qaoa_shared_parameters_match_autograd(p, transpile):
    edges = [(0, 1), (1, 2), (0, 2)]
    names, symbolic = build_qaoa_ansatz_symbolic(3, edges, p)
    params = [0.31, -0.29, 0.57, 0.43][:2 * p]
    hamiltonian = build_maxcut_hamiltonian(edges, 3)
    expected = _autograd(symbolic, names, params, hamiltonian)
    client = _ExactClient()
    template = symbolic
    targets = [0, 1, 2]
    if transpile:
        template = client._transpile_with_backend(
            symbolic, Backend("Simulator"), use_dd=False,
            use_gate_compressor=False, convert_single_qubit_gate_to_u=True,
        )
        targets = client._ordered_target_qubits_from_layout(
            compiled_qc=template, original_qc=symbolic, num_qubits=3,
        )
    actual = _gradient(
        client, template, names, params, hamiltonian,
        target_qubits=targets,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("theta", [0.27, -0.63])
def test_scaled_shared_ry_gradient_obeys_chain_rule(theta):
    template = QuantumCircuit(1).ry("2*theta", 0).ry("-theta/2", 0)
    before = deepcopy(template.__dict__)
    actual = _gradient(_ExactClient(), template, ["theta"], [theta], [(1.0, "Z")])
    np.testing.assert_allclose(actual, [-1.5 * np.sin(1.5 * theta)], atol=1e-10)
    assert template.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
def test_custom_vqe_shared_u_slots_match_autograd(transpile):
    template = QuantumCircuit(2)
    template.h(0).u("2*theta", "-phi/2", "theta+phi", 0)
    template.u("phi", "theta/3", "-2*phi", 1).cx(0, 1).ry("theta", 1)
    hamiltonian = [(0.7, "ZX"), (-0.4, "YY"), (0.3, "IZ")]
    params = [0.37, -0.51]
    expected = _autograd(template, ["theta", "phi"], params, hamiltonian)
    before = deepcopy(template.__dict__)
    result = run_vqe_with_backend(
        _ExactClient(), name="shared_u_vqe", num_qubits=2,
        backend=Backend("Simulator"), chip_name="Simulator",
        hamiltonian=hamiltonian, ansatz="custom", custom_ansatz_circuit=template,
        init_params=params, max_iters=1, shots=128, use_dd=False,
        layers=1, learning_rate=0.1, beta1=0.9, beta2=0.98,
        eps=1e-8, shift=np.pi / 2, zne=False, readout_mitigation=False,
        gradient_method="parameter-shift", transpile=transpile,
    )
    np.testing.assert_allclose(result.grad_history[0], expected, atol=1e-9)
    assert template.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
def test_qaoa_parameter_shift_escapes_zero_cost_initial_mixer(transpile):
    result = run_qaoa_with_backend(
        _ExactClient(), name="one_edge_optimization", num_qubits=2,
        backend=Backend("Simulator"), chip_name="Simulator",
        hamiltonian=[(0.5, "ZZ")], edges=[(0, 1)],
        init_params=[0.4, 0.0], max_iters=15, learning_rate=0.15,
        shots=128, use_dd=False, transpile=transpile,
        gradient_method="parameter-shift",
    )
    assert result.cost_history[0] == pytest.approx(0.0, abs=1e-12)
    assert result.grad_history[0][1] == pytest.approx(np.sin(0.4), abs=1e-10)
    assert result.best_cost < -0.4


@pytest.mark.parametrize("shift", [np.pi / 4, -np.pi / 3, 2.3])
def test_nondefault_gate_shift_has_correct_normalization(shift):
    template = QuantumCircuit(1).ry("theta", 0)
    actual = _gradient(
        _ExactClient(), template, ["theta"], [0.43], [(1.0, "Z")], shift=shift,
    )
    np.testing.assert_allclose(actual, [-np.sin(0.43)], atol=1e-10)


@pytest.mark.parametrize("shift", [0.0, np.pi, np.nan, np.inf, -np.inf])
def test_invalid_gate_shift_fails_before_backend_evaluation(shift):
    client = _ExactClient()
    template = QuantumCircuit(1).ry("theta", 0)
    before = deepcopy(template.__dict__)
    with pytest.raises(ValueError, match="shift"):
        _gradient(client, template, ["theta"], [0.43], [(1.0, "Z")], shift=shift)
    assert client.evaluations == []
    assert template.__dict__ == before


def test_gradient_keeps_backend_options_and_circuit_transform():
    template = QuantumCircuit(1).ry("2*theta", 0).ry("theta", 0)
    client = _ExactClient()
    transformed_parameters = []

    def transform(qc, param_index):
        transformed_parameters.append(param_index)
        return qc.barrier()

    actual = _gradient(
        client, template, ["theta"], [0.31], [(1.0, "Z")],
        zne=True, readout_mitigation=True, target_qubits=[0],
        clifford_fit_map={"Z": (0.7, 0.1)},
        circuit_transform=transform, submit_options={"sleep_time": 2},
    )
    np.testing.assert_allclose(actual, [-0.7 * 3 * np.sin(3 * 0.31)], atol=1e-9)
    assert transformed_parameters and set(transformed_parameters) == {0}
    for _, options in client.evaluations:
        assert options["zne"] is True
        assert options["readout_mitigation"] is True
        assert options["target_qubits"] == [0]
        assert options["submit_options"] == {"sleep_time": 2}
        assert options["qasm_version"] == "2.0"
        assert options["transpile"] is False


def test_single_use_parameter_keeps_standard_parameter_shift():
    template = QuantumCircuit(1).ry("theta", 0)
    actual = _gradient(_ExactClient(), template, ["theta"], [0.43], [(1.0, "Z")])
    np.testing.assert_allclose(actual, [-np.sin(0.43)], atol=1e-10)


def test_multivariate_angle_transform_receives_each_original_parameter_index():
    template = QuantumCircuit(1).ry("theta-2*phi", 0)
    transformed_parameters = []

    def transform(qc, param_index):
        transformed_parameters.append(param_index)
        return qc

    params = [0.31, -0.23]
    actual = _gradient(
        _ExactClient(), template, ["theta", "phi"], params, [(1.0, "Z")],
        circuit_transform=transform,
    )
    angle = params[0] - 2 * params[1]
    np.testing.assert_allclose(actual, [-np.sin(angle), 2 * np.sin(angle)], atol=1e-10)
    assert set(transformed_parameters) == {0, 1}


@pytest.mark.parametrize(
    "expression,stored_values,angle,expected_gradient",
    [
        ("2*theta", {"2*theta": 0.2}, 0.8, -2 * np.sin(0.8)),
        ("theta+fixed", {"fixed": 0.3}, 0.7, -np.sin(0.7)),
        ("theta+fixed", {"theta+fixed": 0.2}, 0.2, 0.0),
    ],
)
def test_gradient_matches_existing_deep_binding_precedence(
    expression, stored_values, angle, expected_gradient,
):
    template = QuantumCircuit(1).ry(expression, 0)
    template.params_value.update(stored_values)
    before = deepcopy(template.__dict__)

    def bound_energy(theta):
        bound = instantiate_transpiled_template(template, ["theta"], np.array([theta]))
        state = simulate_statevector(bound, device="cpu")
        return float(expectation_pauli(state, "Z", num_qubits=1).real)

    assert bound_energy(0.4) == pytest.approx(np.cos(angle), abs=1e-12)
    # This checks the actual pre-existing binder, independently of the gradient
    # helper's expression-resolution logic.
    step = 1e-5
    finite_difference = (bound_energy(0.4 + step) - bound_energy(0.4 - step)) / (2 * step)
    assert finite_difference == pytest.approx(expected_gradient, abs=1e-9)
    client = _ExactClient()
    actual = _gradient(client, template, ["theta"], [0.4], [(1.0, "Z")])
    np.testing.assert_allclose(actual, [expected_gradient], atol=1e-10)
    if expected_gradient == 0.0:
        assert client.evaluations == []
    assert template.__dict__ == before


@pytest.mark.parametrize("expression,theta", [(0.2, 0.4), ("theta**2", 0.0)])
def test_unused_or_locally_stationary_parameter_needs_no_backend_jobs(expression, theta):
    client = _ExactClient()
    template = QuantumCircuit(1).ry(expression, 0)
    actual = _gradient(client, template, ["theta"], [theta], [(1.0, "Z")])
    np.testing.assert_array_equal(actual, [0.0])
    assert client.evaluations == []


def test_one_multivariate_angle_reuses_backend_evaluations():
    client = _ExactClient()
    template = QuantumCircuit(1).ry("theta-2*phi", 0)
    params = [0.31, -0.23]
    actual = _gradient(client, template, ["theta", "phi"], params, [(1.0, "Z")])
    angle = params[0] - 2 * params[1]
    np.testing.assert_allclose(actual, [-np.sin(angle), 2 * np.sin(angle)], atol=1e-10)
    assert len(client.evaluations) == 2


def test_legacy_fixed_angle_label_survives_trainable_parameter_binding():
    template = QuantumCircuit(1).ry("fixed angle", 0).ry("theta", 0)
    template.params_value["fixed angle"] = 0.2
    before = deepcopy(template.__dict__)
    bound = instantiate_transpiled_template(template, ["theta"], np.array([0.4]))
    state = simulate_statevector(bound, device="cpu")
    energy = float(expectation_pauli(state, "Z", num_qubits=1).real)
    assert energy == pytest.approx(np.cos(0.6), abs=1e-12)
    actual = _gradient(_ExactClient(), template, ["theta"], [0.4], [(1.0, "Z")])
    np.testing.assert_allclose(actual, [-np.sin(0.6)], atol=1e-10)
    assert template.__dict__ == before


def test_numeric_expression_fallback_precedes_undefined_symbolic_derivative():
    expression = "theta**0.5+fixed"
    template = QuantumCircuit(1).ry(expression, 0)
    template.params_value[expression] = 0.2
    before = deepcopy(template.__dict__)
    bound = instantiate_transpiled_template(template, ["theta"], np.array([0.0]))
    state = simulate_statevector(bound, device="cpu")
    energy = float(expectation_pauli(state, "Z", num_qubits=1).real)
    assert energy == pytest.approx(np.cos(0.2), abs=1e-12)
    client = _ExactClient()
    actual = _gradient(client, template, ["theta"], [0.0], [(1.0, "Z")])
    np.testing.assert_array_equal(actual, [0.0])
    assert client.evaluations == []
    assert template.__dict__ == before
