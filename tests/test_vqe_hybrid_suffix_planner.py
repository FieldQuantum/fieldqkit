import pytest


torch = pytest.importorskip("torch")

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.algorithms.circuit_compression import compress_circuit_with_hybrid_objective
from quantum_hw.algorithms.circuit_compression import plan_hybrid_suffix_blocks


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
