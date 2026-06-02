"""Tests for the algorithms module: QAOA, VQE, QML, and circuit compression.

Happy-path optimization tests are covered by examples/ demos.
This file focuses on edge cases and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fieldqkit.algorithms.qaoa import (
    build_maxcut_hamiltonian,
    build_qaoa_ansatz_symbolic,
    run_qaoa_with_backend,
)
from fieldqkit.algorithms.vqe import build_ising_hamiltonian, run_vqe_with_backend
from fieldqkit.algorithms.qml_encoding import (
    angle_encoding_circuit,
    angle_encoding_circuit_symbolic,
    iqp_encoding_circuit,
    iqp_encoding_circuit_symbolic,
)
from fieldqkit.algorithms.qml import (
    run_pqc_classifier,
    run_qnn_conditional,
    run_qnn_unsupervised,
)
from fieldqkit.algorithms.qml_runner import QMLRunner
from fieldqkit.algorithms.circuit_compression import (
    compress_circuit_with_hybrid_objective,
    plan_hybrid_suffix_blocks,
)
from fieldqkit.api.backend import Backend
from fieldqkit.circuit import QuantumCircuit
import fieldqkit.sim as sim_pkg


# ═══════════════════════════════════════════════════════════
#  QAOA
# ═══════════════════════════════════════════════════════════


class TestQAOAHamiltonian:
    def test_edge_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            build_maxcut_hamiltonian([(0, 3)], num_qubits=3)

    def test_self_loop_raises(self):
        with pytest.raises(ValueError, match="self-loop"):
            build_maxcut_hamiltonian([(1, 1)], num_qubits=2)

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_maxcut_hamiltonian([], num_qubits=0)


class TestQAOAAnsatz:
    def test_p_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_qaoa_ansatz_symbolic(2, [(0, 1)], p=0)


class TestQAOAAutograd:
    def test_rejects_non_simulator(self):
        edges = [(0, 1)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=2)

        with pytest.raises(ValueError, match="only supported on Simulator"):
            run_qaoa_with_backend(
                object(),
                name="test_qaoa_bad_chip",
                num_qubits=2,
                backend=Backend("Simulator"),
                chip_name="Baihua",
                hamiltonian=hamiltonian,
                edges=edges,
                p=1,
                shots=512,
                max_iters=1,
                learning_rate=0.2,
                beta1=0.9,
                beta2=0.98,
                eps=1e-8,
                shift=np.pi / 2,
                zne=False,
                readout_mitigation=False,
                gradient_method="autograd",
            )

    def test_wrong_init_params_length_raises(self):
        edges = [(0, 1)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=2)

        with pytest.raises(ValueError, match="init_params length"):
            run_qaoa_with_backend(
                object(),
                name="test_qaoa_bad_init",
                num_qubits=2,
                backend=Backend("Simulator"),
                chip_name="Simulator",
                hamiltonian=hamiltonian,
                edges=edges,
                p=1,
                shots=512,
                max_iters=1,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.98,
                eps=1e-8,
                shift=np.pi / 2,
                zne=False,
                readout_mitigation=False,
                init_params=[0.5, 1.0, 2.0],  # p=1 expects 2 params
                gradient_method="autograd",
            )


# ═══════════════════════════════════════════════════════════
#  VQE
# ═══════════════════════════════════════════════════════════


def test_vqe_autograd_rejects_non_simulator():
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    with pytest.raises(ValueError, match="only supported on Simulator"):
        run_vqe_with_backend(
            object(),
            name="test_vqe_autograd_bad_chip",
            num_qubits=2,
            backend=backend,
            chip_name="Baihua",
            hamiltonian=hamiltonian,
            layers=1,
            shots=512,
            max_iters=1,
            learning_rate=0.2,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
            shift=np.pi / 2,
            zne=False,
            readout_mitigation=False,
            gradient_method="autograd",
        )


def test_vqe_autograd_custom_ansatz_supports_negative_expression_param():
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    qc = QuantumCircuit(2)
    qc.ry("theta", 0)
    qc.ry("-theta", 1)
    qc.cx(0, 1)

    result = run_vqe_with_backend(
        object(),
        name="test_vqe_autograd_custom_negative",
        num_qubits=2,
        backend=backend,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        layers=1,
        shots=256,
        max_iters=2,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=7,
        gradient_method="autograd",
        ansatz="custom",
        custom_ansatz_circuit=qc,
    )

    assert len(result.energy_history) == 2
    assert np.isfinite(result.best_energy)


def test_vqe_autograd_custom_ansatz_supports_division_expression_param():
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    qc = QuantumCircuit(2)
    qc.ry("theta", 0)
    qc.rx("theta/2", 1)
    qc.cx(0, 1)

    result = run_vqe_with_backend(
        object(),
        name="test_vqe_autograd_custom_division",
        num_qubits=2,
        backend=backend,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        layers=1,
        shots=256,
        max_iters=2,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=7,
        gradient_method="autograd",
        ansatz="custom",
        custom_ansatz_circuit=qc,
    )

    assert len(result.energy_history) == 2
    assert np.isfinite(result.best_energy)


# ═══════════════════════════════════════════════════════════
#  QML encoding edge cases
# ═══════════════════════════════════════════════════════════


class TestAngleEncoding:
    def test_fewer_features_than_qubits(self):
        qc = angle_encoding_circuit([0.5], num_qubits=4)
        assert len(qc.gates) == 1


class TestAngleEncodingSymbolic:
    def test_fewer_features(self):
        qc, names = angle_encoding_circuit_symbolic(4, 1)
        assert len(names) == 1
        assert len(qc.gates) == 1

    def test_features_capped_at_num_qubits(self):
        qc, names = angle_encoding_circuit_symbolic(2, 5)
        assert len(names) == 2


class TestIQPEncodingSymbolic:
    def test_apply_value_resolves_product(self):
        qc, names = iqp_encoding_circuit_symbolic(2, 2)
        qc_copy = qc.deepcopy()
        qc_copy.apply_value({"x_0": 0.5, "x_1": 0.3}, deep=True)
        rz_params = [g[1] for g in qc_copy.gates if g[0] == "rz"]
        assert any(abs(p - 0.15) < 1e-10 for p in rz_params if isinstance(p, float))


class TestPQCClassifierErrors:
    def test_unknown_encoding(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            run_pqc_classifier(
                num_qubits=2,
                train_data=[([0.0, 0.0], 0)],
                encoding="nonexistent",
                layers=1,
                max_iters=1,
            )

    def test_bad_gradient_method(self):
        with pytest.raises(ValueError, match="gradient_method"):
            run_pqc_classifier(
                num_qubits=2,
                train_data=[([0.0], 0)],
                encoding="angle",
                layers=1,
                max_iters=1,
                gradient_method="invalid",
            )

    def test_parameter_shift_requires_backend(self):
        with pytest.raises(ValueError, match="parameter-shift requires"):
            run_pqc_classifier(
                num_qubits=2,
                train_data=[([0.0], 0)],
                encoding="angle",
                layers=1,
                max_iters=1,
                gradient_method="parameter-shift",
            )


# ═══════════════════════════════════════════════════════════
#  Circuit compression
# ═══════════════════════════════════════════════════════════


def test_hybrid_suffix_planner_blocks_cover_suffix_contiguously():
    qc = QuantumCircuit(4)
    qc.ry(0.11, 0)
    qc.ry(0.22, 1)
    qc.cx(0, 1)
    qc.ry(0.33, 2)
    qc.cx(2, 3)
    qc.cz(1, 2)
    qc.rx(0.44, 3)

    plan = plan_hybrid_suffix_blocks(
        qc,
        bond_cap=8,
        trunc_tol=1e-8,
        max_layers_per_block=3,
    )

    assert 0 <= plan.split_layer <= plan.total_layers

    if plan.split_layer == plan.total_layers:
        assert plan.blocks == []
        return

    assert len(plan.blocks) > 0
    assert plan.blocks[0].start_layer == plan.split_layer
    assert plan.blocks[-1].end_layer == plan.total_layers - 1

    for i in range(1, len(plan.blocks)):
        assert plan.blocks[i - 1].end_layer + 1 == plan.blocks[i].start_layer


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bond_cap": 0},
        {"trunc_tol": -1e-9},
        {"max_layers_per_block": 0},
    ],
)
def test_hybrid_suffix_planner_rejects_invalid_thresholds(kwargs):
    qc = QuantumCircuit(2)
    qc.ry(0.1, 0)

    with pytest.raises(ValueError):
        plan_hybrid_suffix_blocks(qc, **kwargs)


def test_compose_stage_circuits_rejects_inconsistent_qubits_layout():
    """Stages with mismatched qubits ordering must raise to preserve transpiler layout."""
    from fieldqkit.algorithms.circuit_compression import _compose_stage_circuits

    qc1 = QuantumCircuit(3)
    qc1.ry(0.1, 0)
    qc1.cx(0, 1)
    qc1.qubits = [0, 1, 2]

    qc2 = QuantumCircuit(3)
    qc2.rx(0.2, 0)
    qc2.qubits = [2, 0, 1]  # different physical-qubit layout

    with pytest.raises(ValueError, match="qubits layout"):
        _compose_stage_circuits([qc1, qc2], num_qubits=3)


def test_compose_stage_circuits_preserves_qubits_layout():
    """When all stages share the same qubits list, composition keeps it intact."""
    from fieldqkit.algorithms.circuit_compression import _compose_stage_circuits

    qc1 = QuantumCircuit(3)
    qc1.ry(0.1, 0)
    qc1.qubits = [2, 0, 1]

    qc2 = QuantumCircuit(3)
    qc2.rx(0.2, 1)
    qc2.qubits = [2, 0, 1]

    out = _compose_stage_circuits([qc1, qc2], num_qubits=3)
    assert out.qubits == [2, 0, 1]
    assert len(out.gates) == 2


def test_compress_circuit_returns_shallow_hardware_efficient_circuit():
    qc = QuantumCircuit(3)
    qc.ry(0.15, 0)
    qc.ry(-0.22, 1)
    qc.cx(0, 1)
    qc.ry(0.31, 1)
    qc.cx(1, 2)

    compressed_qc, warm_start, summary = compress_circuit_with_hybrid_objective(
        qc,
        num_qubits=3,
        approx_layers=1,
        optimizer_steps=2,
        optimizer_lr=0.05,
        objective_mode="mps",
        bond_cap=16,
        warm_start_params=None,
    )

    assert isinstance(compressed_qc, QuantumCircuit)
    assert int(compressed_qc.nqubits) == 3
    assert warm_start.shape[0] == 2 * 3 * (1 + 1)
    assert summary["best_loss"] >= 0.0


def test_compress_circuit_accepts_warm_start_shape():
    qc = QuantumCircuit(3)
    qc.ry(0.2, 0)
    qc.cx(0, 1)
    qc.ry(-0.1, 1)

    param_count = 2 * 3 * (1 + 1)
    warm = [0.0] * param_count

    _, params_out, _ = compress_circuit_with_hybrid_objective(
        qc,
        num_qubits=3,
        approx_layers=1,
        optimizer_steps=2,
        optimizer_lr=0.05,
        objective_mode="mps",
        bond_cap=16,
        warm_start_params=warm,
    )

    assert params_out.shape[0] == param_count


# ═══════════════════════════════════════════════════════════
#  QNN unsupervised
# ═══════════════════════════════════════════════════════════


class TestQNNUnsupervisedErrors:
    def test_bad_gradient_method(self):
        with pytest.raises(ValueError, match="gradient_method"):
            run_qnn_unsupervised(
                num_qubits=2,
                train_samples=np.array([[0, 1], [1, 0]]),
                gradient_method="invalid",
                max_iters=1,
            )

    def test_parameter_shift_requires_backend(self):
        with pytest.raises(ValueError, match="parameter-shift requires"):
            run_qnn_unsupervised(
                num_qubits=2,
                train_samples=np.array([[0, 1]]),
                gradient_method="parameter-shift",
                max_iters=1,
            )


class TestQNNUnsupervisedAutograd:
    def test_basic_training_returns_qbm_result(self):
        train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        result = run_qnn_unsupervised(
            num_qubits=2,
            train_samples=train,
            layers=1,
            max_iters=3,
            learning_rate=0.05,
            seed=42,
            gradient_method="autograd",
            gen_shots=64,
        )
        assert len(result.loss_history) == 3
        assert np.isfinite(result.best_loss)
        assert len(result.best_params) > 0
        assert result.generated_samples is not None
        assert len(result.generated_samples) == 64

    def test_with_test_samples(self):
        train = np.array([[0, 0], [1, 1]])
        test = np.array([[0, 1], [1, 0]])
        result = run_qnn_unsupervised(
            num_qubits=2,
            train_samples=train,
            test_samples=test,
            layers=1,
            max_iters=2,
            seed=7,
            gradient_method="autograd",
            gen_shots=32,
        )
        assert result.test_loss_history is not None
        assert len(result.test_loss_history) == 2

    def test_loss_decreases_over_iterations(self):
        # A simple distribution — all-zeros should be easy to learn
        train = np.zeros((10, 2), dtype=int)
        result = run_qnn_unsupervised(
            num_qubits=2,
            train_samples=train,
            layers=2,
            max_iters=30,
            learning_rate=0.1,
            seed=123,
            gradient_method="autograd",
            gen_shots=32,
        )
        # Best loss should be lower than initial loss
        assert result.best_loss <= result.loss_history[0] + 1e-6

    def test_generated_samples_are_binary(self):
        train = np.array([[0, 1], [1, 0]])
        result = run_qnn_unsupervised(
            num_qubits=2,
            train_samples=train,
            layers=1,
            max_iters=2,
            seed=0,
            gradient_method="autograd",
            gen_shots=16,
        )
        samples = np.array(result.generated_samples)
        assert samples.shape == (16, 2)
        assert set(samples.flatten()).issubset({0, 1})


# ═══════════════════════════════════════════════════════════
#  QNN conditional
# ═══════════════════════════════════════════════════════════


class TestQNNConditionalErrors:
    def test_bad_gradient_method(self):
        with pytest.raises(ValueError, match="gradient_method"):
            run_qnn_conditional(
                num_qubits=2,
                train_pairs=[([0, 1], [1, 0])],
                gradient_method="invalid",
                max_iters=1,
            )

    def test_parameter_shift_requires_backend(self):
        with pytest.raises(ValueError, match="parameter-shift requires"):
            run_qnn_conditional(
                num_qubits=2,
                train_pairs=[([0, 1], [1, 0])],
                gradient_method="parameter-shift",
                max_iters=1,
            )


class TestQNNConditionalAutograd:
    def test_basic_training_returns_qbm_result(self):
        pairs = [
            ([0, 0], [1, 1]),
            ([1, 0], [0, 1]),
            ([0, 1], [1, 0]),
        ]
        result = run_qnn_conditional(
            num_qubits=2,
            train_pairs=pairs,
            layers=1,
            max_iters=3,
            learning_rate=0.05,
            seed=42,
            gradient_method="autograd",
            gen_shots=64,
        )
        assert len(result.loss_history) == 3
        assert np.isfinite(result.best_loss)
        assert len(result.best_params) > 0
        assert result.generated_samples is not None
        assert len(result.generated_samples) == 64

    def test_with_test_pairs(self):
        train = [([0, 0], [1, 1]), ([1, 1], [0, 0])]
        test = [([0, 1], [1, 0])]
        result = run_qnn_conditional(
            num_qubits=2,
            train_pairs=train,
            test_pairs=test,
            layers=1,
            max_iters=2,
            seed=7,
            gradient_method="autograd",
            gen_shots=32,
        )
        assert result.test_loss_history is not None
        assert len(result.test_loss_history) == 2

    def test_different_inputs_give_different_loss(self):
        """Verify the model sees different inputs (basis prep matters)."""
        # Identity mapping: x -> x
        pairs_identity = [([0, 0], [0, 0]), ([1, 1], [1, 1])]
        # Flip mapping: x -> NOT(x)
        pairs_flip = [([0, 0], [1, 1]), ([1, 1], [0, 0])]
        r1 = run_qnn_conditional(
            num_qubits=2, train_pairs=pairs_identity,
            layers=1, max_iters=5, seed=42,
            gradient_method="autograd", gen_shots=8,
        )
        r2 = run_qnn_conditional(
            num_qubits=2, train_pairs=pairs_flip,
            layers=1, max_iters=5, seed=42,
            gradient_method="autograd", gen_shots=8,
        )
        # With the same seed and different data, losses should differ
        assert r1.loss_history != r2.loss_history

    def test_generated_samples_are_binary(self):
        pairs = [([0, 0], [0, 0]), ([1, 1], [1, 1])]
        result = run_qnn_conditional(
            num_qubits=2,
            train_pairs=pairs,
            layers=1,
            max_iters=2,
            seed=0,
            gradient_method="autograd",
            gen_shots=16,
        )
        samples = np.array(result.generated_samples)
        assert samples.shape == (16, 2)
        assert set(samples.flatten()).issubset({0, 1})

    def test_loss_is_finite_across_iterations(self):
        pairs = [([0, 1], [1, 0])] * 4
        result = run_qnn_conditional(
            num_qubits=2,
            train_pairs=pairs,
            layers=1,
            max_iters=5,
            seed=99,
            gradient_method="autograd",
            gen_shots=8,
        )
        for loss in result.loss_history:
            assert np.isfinite(loss)

    def test_callback_invoked(self):
        calls = []
        pairs = [([0, 0], [1, 1])]
        run_qnn_conditional(
            num_qubits=2,
            train_pairs=pairs,
            layers=1,
            max_iters=3,
            seed=0,
            gradient_method="autograd",
            gen_shots=8,
            callback=lambda it, loss: calls.append((it, loss)),
        )
        assert len(calls) == 3
        assert all(isinstance(l, float) for _, l in calls)


# ═══════════════════════════════════════════════════════════
#  QMLRunner (import & construction only — hardware not mocked)
# ═══════════════════════════════════════════════════════════


class TestQMLRunnerConstruction:
    def test_default_construction(self):
        runner = QMLRunner(client=object())
        assert runner.layers == 2
        assert runner.shots == 4096
        assert runner.max_iters == 100
        assert runner.gradient_method == "parameter-shift"
        assert runner.gen_shots == 1024
        assert runner.mmd_sigma == 1.0

    def test_custom_parameters(self):
        runner = QMLRunner(
            client=object(),
            layers=3,
            shots=2048,
            max_iters=50,
            learning_rate=0.1,
            seed=42,
            gradient_method="autograd",
            gen_shots=512,
            mmd_sigma=2.0,
        )
        assert runner.layers == 3
        assert runner.shots == 2048
        assert runner.max_iters == 50
        assert runner.learning_rate == 0.1
        assert runner.seed == 42
        assert runner.gradient_method == "autograd"
        assert runner.gen_shots == 512
        assert runner.mmd_sigma == 2.0

    def test_has_all_public_methods(self):
        runner = QMLRunner(client=object())
        assert callable(getattr(runner, "run_classifier", None))
        assert callable(getattr(runner, "run_unsupervised", None))
        assert callable(getattr(runner, "run_conditional", None))


# ═══════════════════════════════════════════════════════════
#  Additional large-scale & boundary-case tests (appended)
# ═══════════════════════════════════════════════════════════

from fieldqkit.algorithms.vqe import _resolve_ansatz_layout
from fieldqkit.algorithms.ansatz_templates import (
    build_hardware_efficient_ansatz,
    build_hardware_efficient_ansatz_symbolic,
)
from fieldqkit.algorithms.shadow import estimate_observables, _median_of_means


_SIM_BACKEND = Backend("Simulator")


# -----------------------------------------------------------------
#  VQE invariants & boundary cases (autograd / Simulator path)
# -----------------------------------------------------------------


def _run_vqe(num_qubits, hamiltonian, *, layers=1, max_iters=2, seed=0,
             shots=128, learning_rate=0.2, **kwargs):
    """Helper: run a small VQE autograd job on the local Simulator.

    ``gradient_method`` defaults to ``"autograd"`` but may be overridden via
    *kwargs* (used by the error-path tests).
    """
    kwargs.setdefault("gradient_method", "autograd")
    return run_vqe_with_backend(
        object(),
        name="vqe_extra",
        num_qubits=num_qubits,
        backend=_SIM_BACKEND,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        layers=layers,
        shots=shots,
        max_iters=max_iters,
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=seed,
        **kwargs,
    )


class TestVQEInvariants:
    def test_single_qubit_z_reaches_ground_state(self):
        """Ground state of H=Z is -1; VQE should approach it and never go below."""
        result = _run_vqe(1, [(1.0, "Z")], layers=1, max_iters=40, seed=1)
        # Energy is an expectation of Z, so it is bounded in [-1, 1].
        assert result.best_energy >= -1.0 - 1e-6
        assert result.best_energy <= 1.0 + 1e-6
        # Optimization should reach close to the true ground state.
        assert result.best_energy < -0.9
        # Best energy is the minimum seen across the history.
        assert result.best_energy <= min(result.energy_history) + 1e-9

    def test_energy_improves_over_optimization(self):
        """Best energy should be no worse than the very first iteration's energy."""
        result = _run_vqe(1, [(1.0, "Z")], layers=1, max_iters=30, seed=5,
                          learning_rate=0.25)
        assert result.best_energy <= result.energy_history[0] + 1e-9

    def test_identity_hamiltonian_is_constant_one(self):
        """H = I is a constant; ⟨H⟩ must be 1.0 regardless of parameters."""
        result = _run_vqe(2, [(1.0, "II")], layers=1, max_iters=2, seed=0)
        for e in result.energy_history:
            assert abs(e - 1.0) < 1e-6
        assert abs(result.best_energy - 1.0) < 1e-6

    def test_empty_hamiltonian_autograd_raises(self):
        """An empty Hamiltonian has no differentiable energy in autograd mode.

        ⟨H⟩ for an empty term list is the constant 0, which carries no grad_fn,
        so the autograd path raises a RuntimeError when calling ``.backward()``.
        This documents the current (boundary) behavior.
        """
        with pytest.raises(RuntimeError):
            _run_vqe(2, [], layers=1, max_iters=2, seed=0)

    def test_history_lengths_match_iterations(self):
        result = _run_vqe(2, build_ising_hamiltonian(2, 1.0, 1.0),
                          layers=1, max_iters=4, seed=3)
        assert len(result.energy_history) == 4
        assert len(result.params_history) == 4
        assert len(result.grad_history) == 4
        # Hardware-efficient ansatz param count: 2 * n * (layers + 1).
        assert len(result.best_params) == 2 * 2 * (1 + 1)

    def test_ising_energy_respects_spectral_lower_bound(self):
        """⟨H⟩ for a real Hamiltonian must lie above its ground-state energy.

        A loose but valid bound is -(sum of |coeff|): any expectation value of a
        sum of Paulis is at least the negative L1 norm of the coefficients.
        """
        ham = build_ising_hamiltonian(3, 1.0, 1.0)
        lower_bound = -sum(abs(c) for c, _ in ham)
        result = _run_vqe(3, ham, layers=2, max_iters=10, seed=7,
                          learning_rate=0.15)
        assert result.best_energy >= lower_bound - 1e-6
        assert np.isfinite(result.best_energy)


class TestVQEModerateScale:
    def test_six_qubit_ansatz_builds_and_evaluates(self):
        """A 6-qubit, 1-layer ansatz should build and run a couple of steps."""
        ham = build_ising_hamiltonian(6, 1.0, 0.5)
        result = _run_vqe(6, ham, layers=1, max_iters=2, seed=11, shots=64)
        assert len(result.energy_history) == 2
        assert len(result.best_params) == 2 * 6 * (1 + 1)
        assert np.isfinite(result.best_energy)
        # Energy must respect the spectral L1 lower bound.
        assert result.best_energy >= -sum(abs(c) for c, _ in ham) - 1e-6


class TestVQEAnsatzLayout:
    def test_hardwareefficient_param_count(self):
        for layers in (0, 1, 2):
            names, qc = _resolve_ansatz_layout(
                ansatz="hardwareefficient", num_qubits=3, layers=layers,
            )
            assert len(names) == 2 * 3 * (layers + 1)
            assert int(qc.nqubits) == 3

    def test_unknown_ansatz_raises(self):
        with pytest.raises(ValueError, match="hardwareefficient"):
            _resolve_ansatz_layout(ansatz="bogus", num_qubits=2, layers=1)

    def test_custom_ansatz_requires_circuit(self):
        with pytest.raises(ValueError, match="custom_ansatz_circuit"):
            _resolve_ansatz_layout(ansatz="custom", num_qubits=2, layers=1)

    def test_custom_ansatz_without_symbolic_params_raises(self):
        qc = QuantumCircuit(2)
        qc.ry(0.3, 0)  # concrete value, no symbols
        qc.cx(0, 1)
        with pytest.raises(ValueError, match="no unresolved symbolic"):
            _resolve_ansatz_layout(
                ansatz="custom", num_qubits=2, layers=1, custom_ansatz_circuit=qc,
            )

    def test_custom_ansatz_qubit_mismatch_raises(self):
        qc = QuantumCircuit(3)
        qc.ry("theta", 0)
        with pytest.raises(ValueError, match="nqubits"):
            _resolve_ansatz_layout(
                ansatz="custom", num_qubits=2, layers=1, custom_ansatz_circuit=qc,
            )


class TestVQEArgErrors:
    def test_bad_gradient_method_raises(self):
        with pytest.raises(ValueError, match="gradient_method"):
            _run_vqe(2, build_ising_hamiltonian(2), gradient_method="bogus",
                     max_iters=1)

    def test_init_params_wrong_length_raises(self):
        with pytest.raises(ValueError, match="init_params length"):
            _run_vqe(2, build_ising_hamiltonian(2), max_iters=1,
                     init_params=[0.1, 0.2])  # he ansatz expects 8

    @pytest.mark.parametrize("kwargs", [
        {"planner_bond_cap": 0},
        {"planner_trunc_tol": -1.0},
        {"planner_max_layers_per_block": 0},
        {"compression_optimizer_steps": 0},
        {"compression_optimizer_lr": 0.0},
        {"clifford_fitting_num_non_clifford_gates": -1},
    ])
    def test_invalid_numeric_hyperparams_raise(self, kwargs):
        with pytest.raises(ValueError):
            _run_vqe(2, build_ising_hamiltonian(2), max_iters=1, **kwargs)

    def test_compression_without_block_layers_raises(self):
        with pytest.raises(ValueError, match="compression_block_layers"):
            _run_vqe(2, build_ising_hamiltonian(2), max_iters=1,
                     gradient_method="parameter-shift",
                     enable_circuit_compression=True)


# -----------------------------------------------------------------
#  QAOA invariants & boundary cases
# -----------------------------------------------------------------


def _run_qaoa(num_qubits, edges, *, p=1, max_iters=2, seed=0, shots=128):
    hamiltonian = build_maxcut_hamiltonian(edges, num_qubits)
    return run_qaoa_with_backend(
        object(),
        name="qaoa_extra",
        num_qubits=num_qubits,
        backend=_SIM_BACKEND,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        edges=edges,
        p=p,
        shots=shots,
        max_iters=max_iters,
        learning_rate=0.25,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=seed,
        gradient_method="autograd",
    )


class TestQAOAInvariants:
    def test_triangle_maxcut_cut_value_in_valid_range(self):
        """For a triangle (3 edges) the max cut is 2.

        Cut value = num_edges/2 - ⟨H⟩  (since H = Σ 0.5 Z_iZ_j and the dropped
        constant is num_edges/2).  The cut must lie in [0, num_edges].
        """
        edges = [(0, 1), (1, 2), (2, 0)]
        result = _run_qaoa(3, edges, p=2, max_iters=25, seed=2)
        num_edges = len(edges)
        cut_value = num_edges / 2.0 - result.best_cost
        assert 0.0 - 1e-6 <= cut_value <= num_edges + 1e-6
        # Each ZZ term has expectation in [-1, 1], so ⟨H⟩ ∈ [-1.5, 1.5].
        assert -1.5 - 1e-6 <= result.best_cost <= 1.5 + 1e-6

    def test_single_edge_two_qubit_finds_full_cut(self):
        """A single edge can always be fully cut → cost should approach -0.5."""
        edges = [(0, 1)]
        result = _run_qaoa(2, edges, p=1, max_iters=40, seed=4)
        # Minimum possible cost is -0.5 (state |01>/|10>), giving a cut of 1.
        assert result.best_cost >= -0.5 - 1e-6
        assert result.best_cost < 0.0  # should find a cut better than random
        cut_value = len(edges) / 2.0 - result.best_cost
        assert cut_value <= len(edges) + 1e-6

    def test_history_lengths_and_param_count(self):
        edges = [(0, 1), (1, 2)]
        result = _run_qaoa(3, edges, p=2, max_iters=3, seed=1)
        assert len(result.cost_history) == 3
        assert len(result.best_params) == 2 * 2  # 2 params per layer * p
        assert result.best_cost <= min(result.cost_history) + 1e-9


class TestQAOAHamiltonianScale:
    def test_larger_ring_graph_hamiltonian_structure(self):
        """Build (not optimize) a Hamiltonian for an 8-node ring graph."""
        n = 8
        edges = [(i, (i + 1) % n) for i in range(n)]
        ham = build_maxcut_hamiltonian(edges, n)
        assert len(ham) == len(edges)
        for coeff, pauli in ham:
            assert coeff == 0.5
            assert len(pauli) == n
            assert pauli.count("Z") == 2  # exactly two qubits per edge term

    def test_qaoa_ansatz_symbolic_param_and_qubit_counts(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        for p in (1, 3):
            names, qc = build_qaoa_ansatz_symbolic(4, edges, p=p)
            assert len(names) == 2 * p
            assert int(qc.nqubits) == 4

    def test_num_qubits_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_qaoa_ansatz_symbolic(0, [(0, 1)], p=1)


class TestQAOAArgErrors:
    def test_bad_gradient_method_raises(self):
        edges = [(0, 1)]
        ham = build_maxcut_hamiltonian(edges, 2)
        with pytest.raises(ValueError, match="gradient_method"):
            run_qaoa_with_backend(
                object(), name="t", num_qubits=2, backend=_SIM_BACKEND,
                chip_name="Simulator", hamiltonian=ham, edges=edges, p=1,
                shots=16, max_iters=1, learning_rate=0.1, beta1=0.9, beta2=0.98,
                eps=1e-8, shift=np.pi / 2, zne=False, readout_mitigation=False,
                gradient_method="bogus",
            )


# -----------------------------------------------------------------
#  Classical-shadow estimator: invariants & convergence
# -----------------------------------------------------------------


def _single_qubit_zero_shadow(nshots, seed):
    """Synthesize classical-shadow data for the single-qubit state |0>.

    In the Z basis the outcome is always 0; in X/Y bases it is uniformly random.
    Returns ``(samples, basis_patterns)`` ready for ``estimate_observables``.
    """
    rng = np.random.default_rng(seed)
    bases = rng.choice(["X", "Y", "Z"], size=nshots)
    samples = np.zeros((nshots, 1), dtype=int)
    for i, b in enumerate(bases):
        samples[i, 0] = 0 if b == "Z" else int(rng.integers(0, 2))
    return samples, [[b] for b in bases]


class TestShadowEstimator:
    def test_z_expectation_converges_to_true_value(self):
        """Many snapshots → ⟨Z⟩ estimate for |0> approaches +1 (generous tol)."""
        samples, bases = _single_qubit_zero_shadow(5000, seed=0)
        est, err = estimate_observables(samples, bases, ["Z"], num_qubits=1)
        assert abs(est["Z"] - 1.0) < 0.1
        assert err["Z"] >= 0.0

    def test_x_expectation_near_zero_for_z_eigenstate(self):
        samples, bases = _single_qubit_zero_shadow(5000, seed=1)
        est, _ = estimate_observables(samples, bases, ["X"], num_qubits=1)
        assert abs(est["X"]) < 0.1

    def test_identity_observable_is_exactly_one(self):
        samples, bases = _single_qubit_zero_shadow(200, seed=2)
        est, _ = estimate_observables(samples, bases, ["I"], num_qubits=1)
        assert abs(est["I"] - 1.0) < 1e-12

    def test_mom_estimator_matches_mean_for_clean_data(self):
        samples, bases = _single_qubit_zero_shadow(3000, seed=3)
        est, _ = estimate_observables(
            samples, bases, ["Z"], num_qubits=1,
            estimator="mom", rng=np.random.default_rng(99),
        )
        assert abs(est["Z"] - 1.0) < 0.15

    def test_empty_samples_returns_empty_dicts(self):
        est, err = estimate_observables(
            np.zeros((0, 2), dtype=int), [], ["ZZ"], num_qubits=2,
        )
        assert est == {}
        assert err == {}

    def test_single_observable_single_snapshot(self):
        samples = np.array([[0]], dtype=int)
        est, err = estimate_observables(samples, [["Z"]], ["Z"], num_qubits=1)
        assert "Z" in est
        assert err["Z"] == 0.0  # stderr is 0 for a single shot

    def test_non_2d_samples_raises(self):
        with pytest.raises(ValueError, match="2D array"):
            estimate_observables(
                np.zeros(4, dtype=int), [], ["Z"], num_qubits=1,
            )

    def test_basis_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="basis_patterns length"):
            estimate_observables(
                np.zeros((3, 1), dtype=int), [["Z"]], ["Z"], num_qubits=1,
            )


class TestMedianOfMeans:
    def test_single_group_returns_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        median, stderr = _median_of_means(values, groups=1)
        assert abs(median - 2.5) < 1e-12
        assert stderr >= 0.0

    def test_multiple_groups_close_to_mean_for_uniform_data(self):
        values = np.full(100, 0.7)
        median, stderr = _median_of_means(
            values, groups=10, rng=np.random.default_rng(0),
        )
        assert abs(median - 0.7) < 1e-9
        assert abs(stderr) < 1e-9


# -----------------------------------------------------------------
#  Ansatz templates: shape / param-count invariants
# -----------------------------------------------------------------


class TestAnsatzTemplates:
    @pytest.mark.parametrize("num_qubits,layers", [(1, 1), (3, 0), (4, 2), (6, 1)])
    def test_symbolic_and_concrete_have_same_gate_count(self, num_qubits, layers):
        n_params = 2 * num_qubits * (layers + 1)
        names = [f"t_{i}" for i in range(n_params)]
        sym = build_hardware_efficient_ansatz_symbolic(num_qubits, names, layers=layers)
        conc = build_hardware_efficient_ansatz(num_qubits, np.zeros(n_params), layers=layers)
        assert int(sym.nqubits) == num_qubits
        assert int(conc.nqubits) == num_qubits
        assert len(sym.gates) == len(conc.gates)
        # Expected gates: per layer 2n rotations + (n-1) CZ, plus final 2n rotations.
        expected_gates = layers * (2 * num_qubits + (num_qubits - 1)) + 2 * num_qubits
        assert len(sym.gates) == expected_gates

    def test_symbolic_wrong_param_length_raises(self):
        with pytest.raises(ValueError, match="param_names length"):
            build_hardware_efficient_ansatz_symbolic(2, ["only_one"], layers=1)

    def test_concrete_wrong_param_length_raises(self):
        with pytest.raises(ValueError, match="params length"):
            build_hardware_efficient_ansatz(2, [0.1, 0.2], layers=1)

    def test_single_qubit_minimal_ansatz_has_no_entanglers(self):
        names = [f"t_{i}" for i in range(2 * 1 * (1 + 1))]
        qc = build_hardware_efficient_ansatz_symbolic(1, names, layers=1)
        gate_names = [str(g[0]).lower() for g in qc.gates]
        assert "cz" not in gate_names  # no two-qubit gates on a single qubit


# -----------------------------------------------------------------
#  QML encoding: parameter / gate-count invariants
# -----------------------------------------------------------------


class TestEncodingParamCounts:
    @pytest.mark.parametrize("num_qubits,num_features", [(2, 2), (4, 3), (3, 5)])
    def test_angle_symbolic_param_count_capped(self, num_qubits, num_features):
        qc, names = angle_encoding_circuit_symbolic(num_qubits, num_features)
        expected = min(num_features, num_qubits)
        assert len(names) == expected
        assert len(qc.gates) == expected
        assert int(qc.nqubits) == num_qubits

    def test_angle_concrete_uses_min_of_features_and_qubits(self):
        qc = angle_encoding_circuit([0.1, 0.2, 0.3, 0.4, 0.5], num_qubits=3)
        assert len(qc.gates) == 3  # capped at num_qubits

    def test_iqp_symbolic_param_count(self):
        for reps in (1, 2):
            qc, names = iqp_encoding_circuit_symbolic(3, 3, reps=reps)
            assert len(names) == 3
            assert int(qc.nqubits) == 3
            # Each rep: H on n qubits, RZ on n features, and per adjacent pair a
            # CX-RZ-CX block (3 gates). Final H layer on n qubits.
            n = 3
            per_rep = n + n + (n - 1) * 3
            assert len(qc.gates) == reps * per_rep + n

    def test_iqp_concrete_product_angle(self):
        qc = iqp_encoding_circuit([0.5, 0.3], num_qubits=2, reps=1)
        rz_angles = [g[1] for g in qc.gates if str(g[0]).lower() == "rz"]
        # The ZZ-coupling RZ angle is the product of the two features.
        assert any(abs(a - 0.15) < 1e-9 for a in rz_angles if isinstance(a, float))


# -----------------------------------------------------------------
#  PQC classifier: end-to-end autograd invariants (tiny problem)
# -----------------------------------------------------------------


class TestPQCClassifierAutograd:
    def test_linearly_separable_trains_and_reports_metrics(self):
        train = [
            ([0.1, 0.2], 0), ([0.2, 0.1], 0),
            ([1.5, 1.4], 1), ([1.4, 1.5], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2, train_data=train, encoding="angle", layers=1,
            max_iters=12, learning_rate=0.15, seed=1, gradient_method="autograd",
        )
        assert result.task == "supervised"
        assert len(result.loss_history) == 12
        assert np.isfinite(result.best_loss)
        assert result.best_loss <= result.loss_history[0] + 1e-6
        assert 0.0 <= result.accuracy <= 1.0

    def test_callback_receives_each_iteration(self):
        calls = []
        train = [([0.1, 0.2], 0), ([1.4, 1.5], 1)]
        run_pqc_classifier(
            num_qubits=2, train_data=train, encoding="angle", layers=1,
            max_iters=4, seed=0, gradient_method="autograd",
            callback=lambda it, loss: calls.append((it, loss)),
        )
        # callback fires once per iteration with (iter, train_loss), matching
        # run_qnn_unsupervised / run_qnn_conditional.
        assert [it for it, _ in calls] == [0, 1, 2, 3]
        assert all(np.isfinite(loss) for _, loss in calls)
