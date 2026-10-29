"""Compilation keeps declared registers, including qubits without gates."""

from itertools import combinations

import pytest

from fieldqkit.api.backend import Backend, _build_simulator_chip_info
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.compile import Transpiler
from fieldqkit.compile.schedule import DynamicalDecoupling


def _sparse_backend():
    """Three connected physical qubits with deliberately non-dense indices."""
    info = _build_simulator_chip_info(10)
    info["chip_name"] = "idle-qubit-test-backend"
    targets = [2, 5, 9]
    info["qubits_info"] = {f"Q{q}": {"fidelity": 1.0} for q in targets}
    info["couplers_info"] = {
        f"C{index}": {"qubits_index": list(pair), "fidelity": 1.0}
        for index, pair in enumerate(combinations(targets, 2))
    }
    info["priority_qubits"] = [targets]
    return Backend(info)


@pytest.mark.parametrize("circuit_kind", ["empty", "x0", "cx02"])
def test_full_pipeline_preserves_idle_qubits_and_classical_register(circuit_kind):
    qc = QuantumCircuit(3, 7)
    if circuit_kind != "empty":
        qc.x(0)
    if circuit_kind == "cx02":
        qc.cx(0, 2)
    # The API's logical-register preparation includes idle qubits in the
    # compiler metadata, while retaining the requested classical width.
    qc.qubits = list(range(3))

    compiled = Transpiler(_sparse_backend()).run(
        qc, target_qubits=[2, 5, 9], use_dd=True, seed=0,
    )

    assert compiled.nqubits == 10
    assert compiled.ncbits == 7
    assert compiled.qubits == [2, 5, 9]
    layout = compiled.logical_to_physical
    assert set(layout) == {0, 1, 2}
    assert set(layout.values()) == {2, 5, 9}
    assert compiled.physical_to_logical == {physical: logical for logical, physical in layout.items()}

    # Measurement order follows logical qubits, even if routing grouped the
    # interacting pair (0, 2) before the idle qubit 1.
    measured_qubits = [layout[q] for q in range(3)]
    compiled.measure(measured_qubits, [4, 0, 6])
    assert compiled.gates[-1] == ("measure", measured_qubits, [4, 0, 6])
    assert (compiled.nqubits, compiled.ncbits) == (10, 7)


@pytest.mark.parametrize(
    "circuit_kind,expected_bitstring",
    [("empty", "0000000"), ("x0", "0000100"), ("cx02", "0000101")],
)
def test_idle_qubit_pipeline_measurements_reach_requested_classical_bits(circuit_kind, expected_bitstring):
    pytest.importorskip("torch")
    from fieldqkit.sim import simulate_counts

    qc = QuantumCircuit(3, 7)
    if circuit_kind != "empty":
        qc.x(0)
    if circuit_kind == "cx02":
        qc.cx(0, 2)
    qc.qubits = list(range(3))
    compiled = Transpiler(_sparse_backend()).run(
        qc, target_qubits=[2, 5, 9], use_dd=True, seed=0,
    )
    compiled.measure([compiled.logical_to_physical[q] for q in range(3)], [4, 0, 6])

    assert simulate_counts(compiled, shots=16, seed=0, device="cpu") == {expected_bitstring: 16}


def test_dynamical_decoupling_preserves_fully_idle_declared_registers():
    qc = QuantumCircuit(3, 7)
    qc.qubits = list(range(3))

    compiled = DynamicalDecoupling(1e-8, 5e-8).run(qc)

    assert (compiled.nqubits, compiled.ncbits) == (3, 7)
    assert compiled.qubits == [0, 1, 2]
    assert compiled.gates == []
    compiled.measure([0, 1, 2], [4, 0, 6])
    assert compiled.gates == [("measure", [0, 1, 2], [4, 0, 6])]
