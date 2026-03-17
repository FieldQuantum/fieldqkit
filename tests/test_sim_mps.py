import pytest


torch = pytest.importorskip("torch")

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim.mps import simulate_mps
from quantum_hw.sim.mps import simulate_counts as simulate_counts_mps
from quantum_hw.sim.statevector import simulate_counts as simulate_counts_statevector
from quantum_hw.sim.statevector import simulate_statevector


def _mps_to_statevector(mps):
    if not mps:
        return torch.tensor([1.0 + 0.0j], dtype=torch.complex128)

    current = mps[0]
    for next_tensor in mps[1:]:
        current = torch.einsum("lpr,rsq->lpsq", current, next_tensor)
        left_dim, phys_left, phys_right, right_dim = current.shape
        current = current.reshape(left_dim, phys_left * phys_right, right_dim)
    return current.reshape(-1)


def _align_global_phase(reference, candidate):
    pivot = int(torch.argmax(reference.abs()).item())
    if float(reference.abs()[pivot].item()) < 1e-12:
        return candidate

    phase = candidate[pivot] / reference[pivot]
    if float(torch.abs(phase).item()) < 1e-12:
        return candidate
    return candidate / phase


def _build_reference_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.ry(0.37, 1)
    qc.rx(-0.41, 2)
    qc.rz(0.23, 3)
    return qc


@pytest.mark.parametrize(
    ("gate_name", "gate_args"),
    [
        ("swap", (0, 2)),
        ("iswap", (0, 2)),
        ("ecr", (0, 2)),
        ("cx", (0, 2)),
        ("cy", (0, 2)),
        ("cz", (0, 2)),
        ("rxx", (0.31, 0, 2)),
        ("ryy", (-0.52, 0, 2)),
        ("rzz", (0.77, 0, 2)),
        ("cp", (0.19, 0, 2)),
        ("swap", (1, 3)),
        ("cx", (1, 3)),
        ("rzz", (-0.45, 1, 3)),
    ],
)
def test_nonadjacent_two_qubit_mps_matches_statevector(gate_name, gate_args):
    qc = _build_reference_circuit(4)
    getattr(qc, gate_name)(*gate_args)

    expected = simulate_statevector(qc)
    actual = _mps_to_statevector(simulate_mps(qc))
    actual = _align_global_phase(expected, actual)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("gate_name", "num_qubits", "gate_args"),
    [
        ("ccx", 3, (0, 1, 2)),
        ("ccz", 3, (0, 1, 2)),
        ("cswap", 3, (0, 1, 2)),
        ("ccx", 4, (0, 2, 3)),
        ("ccz", 4, (0, 2, 3)),
        ("cswap", 4, (0, 2, 3)),
    ],
)
def test_three_qubit_mps_matches_statevector(gate_name, num_qubits, gate_args):
    qc = _build_reference_circuit(4)
    if num_qubits == 3:
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.ry(0.37, 1)
        qc.rx(-0.41, 2)

    getattr(qc, gate_name)(*gate_args)

    expected = simulate_statevector(qc)
    actual = _mps_to_statevector(simulate_mps(qc))
    actual = _align_global_phase(expected, actual)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("qubit", "expected_bitstring"),
    [
        (0, "001"),
        (1, "010"),
        (2, "100"),
    ],
)
def test_mps_counts_endianness_basis_flip(qubit, expected_bitstring):
    qc = QuantumCircuit(3)
    qc.x(qubit)

    shots = 128
    counts_mps = simulate_counts_mps(qc, shots=shots, seed=7)
    counts_sv = simulate_counts_statevector(qc, shots=shots, seed=7)

    assert counts_mps == {expected_bitstring: shots}
    assert counts_mps == counts_sv


def test_mps_counts_matches_statevector_for_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    shots = 4096
    counts_mps = simulate_counts_mps(qc, shots=shots, seed=123)
    counts_sv = simulate_counts_statevector(qc, shots=shots, seed=123)

    # Exact per-sample equality is not required across backends; compare support and frequencies.
    assert set(counts_mps.keys()) == {"00", "11"}
    assert set(counts_sv.keys()) == {"00", "11"}
    assert abs(counts_mps["00"] - counts_sv["00"]) <= 0.08 * shots
    assert abs(counts_mps["11"] - counts_sv["11"]) <= 0.08 * shots