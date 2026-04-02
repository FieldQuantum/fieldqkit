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
from quantum_hw.algorithms.qml import run_pqc_classifier
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
