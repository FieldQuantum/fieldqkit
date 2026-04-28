"""Tests for the simulator module: MPS and MPO tensor methods."""

import pytest

torch = pytest.importorskip("torch")

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from quantum_hw.sim.common import materialize_gate_matrix, resolve_param
from quantum_hw.sim.mpo import simulate_mpo_process
from quantum_hw.sim.mps import simulate_mps
from quantum_hw.sim.mps import simulate_counts as simulate_counts_mps
from quantum_hw.sim.statevector import simulate_counts as simulate_counts_statevector
from quantum_hw.sim.statevector import simulate_statevector


# ═══════════════════════════════════════════════════════════
#  MPS helpers
# ═══════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════
#  MPS tests
# ═══════════════════════════════════════════════════════════


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
    ("gate_name", "gate_args"),
    [
        # Non-ascending qubit orders for 2-qubit and 3-qubit gates exercise
        # the gate-matrix permutation in _apply_k_qubit_gate_with_mpo.
        ("cx", (3, 0)),
        ("cx", (2, 1)),
        ("cz", (3, 1)),
        ("rzz", (0.61, 3, 0)),
        ("ccx", (3, 0, 1)),
        ("ccx", (2, 0, 3)),
        ("ccz", (3, 1, 0)),
        ("cswap", (3, 0, 2)),
        ("cswap", (2, 1, 3)),
    ],
)
def test_unsorted_qubits_mps_matches_statevector(gate_name, gate_args):
    qc = _build_reference_circuit(4)
    getattr(qc, gate_name)(*gate_args)

    expected = simulate_statevector(qc)
    actual = _mps_to_statevector(simulate_mps(qc))
    actual = _align_global_phase(expected, actual)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    ("qubit", "expected_bitstring"),
    [
        (0, "100"),
        (1, "010"),
        (2, "001"),
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

    assert set(counts_mps.keys()) == {"00", "11"}
    assert set(counts_sv.keys()) == {"00", "11"}
    assert abs(counts_mps["00"] - counts_sv["00"]) <= 0.08 * shots
    assert abs(counts_mps["11"] - counts_sv["11"]) <= 0.08 * shots


# ═══════════════════════════════════════════════════════════
#  MPO helpers
# ═══════════════════════════════════════════════════════════


def _mpo_to_matrix(mpo):
    if not mpo:
        return torch.eye(1, dtype=torch.complex128)

    current = mpo[0]
    for next_tensor in mpo[1:]:
        current = torch.einsum("lprq,rstu->lpstqu", current, next_tensor)
        dl, p0, p1, dr, q0, q1 = current.shape
        current = current.reshape(dl, p0 * p1, dr, q0 * q1)
    return current.reshape(current.shape[1], current.shape[3])


def _left_apply_gate_to_matrix(unitary, gate, qubits, num_qubits):
    k = len(qubits)
    if k == 0:
        return unitary

    out_shape = [2] * num_qubits + [2**num_qubits]
    tensor = unitary.reshape(out_shape)
    tensor = torch.moveaxis(tensor, list(qubits), list(range(k)))
    tensor = tensor.reshape(2**k, -1)
    tensor = gate @ tensor
    tensor = tensor.reshape(out_shape)
    tensor = torch.moveaxis(tensor, list(range(k)), list(qubits))
    return tensor.reshape(2**num_qubits, 2**num_qubits)


def _reference_unitary_from_circuit(qc):
    n = int(qc.nqubits)
    dtype = torch.complex128
    device = torch.device("cpu")
    unitary = torch.eye(2**n, dtype=dtype, device=device)

    for gate_info in qc.gates:
        gate = gate_info[0]

        if gate in functional_gates_available:
            if gate == "reset":
                raise ValueError("reset is not unitary and cannot be mapped to dense unitary reference")
            continue

        if gate in one_qubit_gates_available:
            q = int(gate_info[1])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q], n)
            continue

        if gate in one_qubit_parameter_gates_available:
            q = int(gate_info[-1])
            params = [resolve_param(qc, p, None) for p in gate_info[1:-1]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q], n)
            continue

        if gate in two_qubit_gates_available:
            q0 = int(gate_info[1])
            q1 = int(gate_info[2])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q0, q1], n)
            continue

        if gate in two_qubit_parameter_gates_available:
            q0 = int(gate_info[-2])
            q1 = int(gate_info[-1])
            params = [resolve_param(qc, p, None) for p in gate_info[1:-2]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q0, q1], n)
            continue

        if gate in three_qubit_gates_available:
            qubits = [int(gate_info[1]), int(gate_info[2]), int(gate_info[3])]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, qubits, n)
            continue

        raise ValueError(f"unsupported gate for reference builder: {gate}")

    return unitary


# ═══════════════════════════════════════════════════════════
#  MPO tests
# ═══════════════════════════════════════════════════════════


def test_simulate_mpo_process_matches_dense_reference_on_nonadjacent_and_three_qubit_gates():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.ry(0.37, 2)
    qc.cx(0, 3)
    qc.rzz(-0.52, 1, 3)
    qc.ccx(0, 2, 3)

    mpo = simulate_mpo_process(qc)
    actual = _mpo_to_matrix(mpo).cpu()
    expected = _reference_unitary_from_circuit(qc)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_simulate_mpo_process_respects_max_bond_dim_cap():
    qc = QuantumCircuit(6)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.cx(0, 5)
    qc.cx(1, 4)
    qc.cx(2, 3)
    qc.rxx(0.31, 0, 5)
    qc.rzz(-0.29, 1, 4)

    mpo = simulate_mpo_process(qc, max_bond_dim=4)
    max_bond = max(int(t.shape[2]) for t in mpo)

    assert max_bond <= 4


# ═══════════════════════════════════════════════════════════
#  Statevector simulator tests
# ═══════════════════════════════════════════════════════════

from quantum_hw.sim.statevector import (
    expectation_pauli as sv_expectation_pauli,
    sample_probabilities as sv_sample_probabilities,
)
from quantum_hw.sim.common import (
    auto_sim_device,
    single_pauli,
    build_param_values_from_tensor,
)


class TestSimulateStatevector:
    def test_zero_qubit_circuit(self):
        qc = QuantumCircuit(0)
        state = simulate_statevector(qc)
        assert state.numel() == 1
        assert abs(state[0].item() - 1.0) < 1e-12

    def test_identity_circuit(self):
        """Circuit with no gates should give |000...0>."""
        qc = QuantumCircuit(3)
        state = simulate_statevector(qc)
        assert state.numel() == 8
        assert abs(state[0].item() - 1.0) < 1e-12
        assert float(state[1:].abs().sum().item()) < 1e-12

    def test_x_gate_flips_state(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        state = simulate_statevector(qc)
        assert abs(state[1].item() - 1.0) < 1e-12

    def test_h_gate_superposition(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        state = simulate_statevector(qc)
        expected = 1.0 / (2 ** 0.5)
        assert abs(state[0].item() - expected) < 1e-12
        assert abs(state[1].item() - expected) < 1e-12

    def test_bell_state(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        state = simulate_statevector(qc)
        expected = 1.0 / (2 ** 0.5)
        assert abs(state[0].item() - expected) < 1e-12
        assert abs(state[3].item() - expected) < 1e-12
        assert abs(state[1].item()) < 1e-12
        assert abs(state[2].item()) < 1e-12

    def test_parameterized_circuit(self):
        qc = QuantumCircuit(1)
        qc.rx("theta", 0)
        state = simulate_statevector(qc, param_values={"theta": 0.0})
        # rx(0) = identity
        assert abs(state[0].item() - 1.0) < 1e-12

    def test_unsupported_gate_raises(self):
        qc = QuantumCircuit(1)
        qc.gates.append(("unsupported_gate_xyz", 0))
        with pytest.raises(ValueError, match="unsupported"):
            simulate_statevector(qc)

    def test_reset_gate_from_superposition(self):
        """Reset from a superposition state projects onto |0> and renormalizes."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.reset(0)
        state = simulate_statevector(qc)
        # After H(0) the state is (|0>+|1>)/sqrt(2) on q0.
        # Reset collapses q0 to |0>: result is |00> (renormalized).
        assert float(state[0].abs().item()) == pytest.approx(1.0, abs=1e-12)

    def test_reset_gate_from_pure_one(self):
        """Reset on a pure |1> state should produce |0>."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.reset(0)
        state = simulate_statevector(qc)
        # |1> amplitude moved to |0>, then renormalized → |0>
        assert float(state[0].abs().item()) == pytest.approx(1.0, abs=1e-12)
        assert float(state[1].abs().item()) == pytest.approx(0.0, abs=1e-12)

    def test_normalization(self):
        """Statevector should always have unit norm."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(1.23, 2)
        qc.rz(0.45, 0)
        state = simulate_statevector(qc)
        norm = float((state.abs() ** 2).sum().item())
        assert norm == pytest.approx(1.0, abs=1e-12)


class TestSimulateCounts:
    def test_deterministic_state(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        counts = simulate_counts_statevector(qc, shots=100, seed=42)
        assert counts == {"10": 100}

    def test_seed_reproducibility(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        c1 = simulate_counts_statevector(qc, shots=1000, seed=42)
        c2 = simulate_counts_statevector(qc, shots=1000, seed=42)
        assert c1 == c2

    def test_different_seeds_differ(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        c1 = simulate_counts_statevector(qc, shots=10000, seed=1)
        c2 = simulate_counts_statevector(qc, shots=10000, seed=2)
        # With different seeds, at least shot counts should differ
        assert c1 != c2

    def test_total_shots(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        counts = simulate_counts_statevector(qc, shots=500, seed=99)
        assert sum(counts.values()) == 500


class TestSVExpectationPauli:
    def test_z_on_zero_state(self):
        qc = QuantumCircuit(1)
        state = simulate_statevector(qc)
        exp = sv_expectation_pauli(state, "Z", num_qubits=1)
        assert float(exp.real) == pytest.approx(1.0)

    def test_z_on_one_state(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        state = simulate_statevector(qc)
        exp = sv_expectation_pauli(state, "Z", num_qubits=1)
        assert float(exp.real) == pytest.approx(-1.0)

    def test_x_on_plus_state(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        state = simulate_statevector(qc)
        exp = sv_expectation_pauli(state, "X", num_qubits=1)
        assert float(exp.real) == pytest.approx(1.0)

    def test_identity_always_one(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        state = simulate_statevector(qc)
        exp = sv_expectation_pauli(state, "II", num_qubits=2)
        assert float(exp.real) == pytest.approx(1.0)

    def test_zz_on_bell_state(self):
        """Bell state |00> + |11> should have <ZZ> = +1."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        state = simulate_statevector(qc)
        exp = sv_expectation_pauli(state, "ZZ", num_qubits=2)
        assert float(exp.real) == pytest.approx(1.0)

    def test_zi_on_bell_state(self):
        """Bell state should have <ZI> = 0 (qubit 0 is maximally mixed)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        state = simulate_statevector(qc)
        exp = sv_expectation_pauli(state, "ZI", num_qubits=2)
        assert float(exp.real) == pytest.approx(0.0, abs=1e-12)


class TestSVSampleProbabilities:
    def test_deterministic_state(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        state = simulate_statevector(qc)
        samples = torch.tensor([[1, 0], [0, 0]], dtype=torch.long)
        probs = sv_sample_probabilities(state, samples)
        assert float(probs[0].item()) == pytest.approx(1.0)
        assert float(probs[1].item()) == pytest.approx(0.0)


class TestAutoSimDevice:
    def test_explicit_cpu(self):
        d = auto_sim_device("cpu")
        assert d == torch.device("cpu")

    def test_none_returns_device(self):
        d = auto_sim_device(None)
        assert isinstance(d, torch.device)


class TestSinglePauli:
    def test_pauli_x(self):
        mat = single_pauli("X", dtype=torch.complex128, device=torch.device("cpu"))
        assert mat.shape == (2, 2)
        assert float(mat[0, 1].real) == pytest.approx(1.0)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            single_pauli("W", dtype=torch.complex128, device=torch.device("cpu"))


class TestBuildParamValues:
    def test_basic(self):
        params = torch.tensor([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]
        pv = build_param_values_from_tensor(params=params, param_names=names)
        assert len(pv) == 3
        assert float(pv["a"]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        params = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="length"):
            build_param_values_from_tensor(params=params, param_names=["a"])


class TestResolveParam:
    def test_float_passthrough(self):
        qc = QuantumCircuit(1)
        assert resolve_param(qc, 3.14) == pytest.approx(3.14)

    def test_int_passthrough(self):
        qc = QuantumCircuit(1)
        assert resolve_param(qc, 2) == pytest.approx(2.0)

    def test_unsupported_type_raises(self):
        qc = QuantumCircuit(1)
        with pytest.raises(TypeError, match="unsupported"):
            resolve_param(qc, [1, 2, 3])
