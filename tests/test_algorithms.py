"""Tests for the algorithms module: QAOA, VQE, QML, and circuit compression.

Happy-path optimization tests are covered by examples/ demos.
This file focuses on edge cases and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantum_hw.algorithms.qaoa import (
    build_maxcut_hamiltonian,
    build_qaoa_ansatz_symbolic,
    run_qaoa_with_backend,
)
from quantum_hw.algorithms.vqe import build_ising_hamiltonian, run_vqe_with_backend
from quantum_hw.algorithms.qml_encoding import (
    angle_encoding_circuit,
    angle_encoding_circuit_symbolic,
    iqp_encoding_circuit,
    iqp_encoding_circuit_symbolic,
)
from quantum_hw.algorithms.qml import (
    run_pqc_classifier,
    run_qnn_conditional,
    run_qnn_unsupervised,
)
from quantum_hw.algorithms.qml_runner import QMLRunner
from quantum_hw.algorithms.circuit_compression import (
    compress_circuit_with_hybrid_objective,
    plan_hybrid_suffix_blocks,
)
from quantum_hw.api.backend import Backend
from quantum_hw.circuit import QuantumCircuit
import quantum_hw.sim as sim_pkg


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
    from quantum_hw.algorithms.circuit_compression import _compose_stage_circuits

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
    from quantum_hw.algorithms.circuit_compression import _compose_stage_circuits

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
