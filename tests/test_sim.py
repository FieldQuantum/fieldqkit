"""Tests for the simulator module: MPS and MPO tensor methods."""

import pytest

torch = pytest.importorskip("torch")

from fieldqkit.circuit import QuantumCircuit
from fieldqkit.circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from fieldqkit.sim.common import materialize_gate_matrix, resolve_param
import fieldqkit.sim.common as sim_common
from fieldqkit.sim.mpo import simulate_mpo_process
from fieldqkit.sim.mps import simulate_mps
from fieldqkit.sim.mps import simulate_counts as simulate_counts_mps
from fieldqkit.sim.statevector import simulate_counts as simulate_counts_statevector
from fieldqkit.sim.statevector import simulate_statevector
from fieldqkit.sim.density_matrix import simulate_density_matrix


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
        ("ccx", 4, (0, 2, 3)),
        ("ccz", 4, (0, 2, 3)),
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

from fieldqkit.sim.statevector import (
    expectation_pauli as sv_expectation_pauli,
    sample_probabilities as sv_sample_probabilities,
)
from fieldqkit.sim.common import (
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

    def test_reset_from_out_of_phase_superposition(self):
        """Reset on (|0>-|1>)/sqrt(2) must give |0>, not cancel to a zero state.

        Regression test: summing the |0> and |1> amplitudes (the previous
        implementation) cancels for this out-of-phase state and wrongly leaves
        it unchanged. Projection onto |0> is the correct behaviour.
        """
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.h(0)  # H|1> = (|0> - |1>)/sqrt(2)
        qc.reset(0)
        state = simulate_statevector(qc)
        assert float(state[0].abs().item()) == pytest.approx(1.0, abs=1e-12)
        assert float(state[1].abs().item()) == pytest.approx(0.0, abs=1e-12)

    def test_reset_unentangled_consistency_across_backends(self):
        """Statevector, MPS and density-matrix backends agree for an
        unentangled reset (where a pure-state result is exact)."""
        def build():
            qc = QuantumCircuit(2)
            qc.x(0)
            qc.h(0)        # q0 = (|0> - |1>)/sqrt(2), unentangled
            qc.ry(0.7, 1)  # q1 in an arbitrary unentangled state
            qc.reset(0)    # force q0 -> |0>, q1 preserved
            return qc

        sv = simulate_statevector(build())
        sv_probs = (sv.abs() ** 2).detach().cpu().numpy()

        mps = _mps_to_statevector(simulate_mps(build()))
        mps_probs = (mps.abs() ** 2).detach().cpu().numpy()

        rho = simulate_density_matrix(build()).cpu()
        dm_probs = torch.diag(rho).real.numpy()

        assert np.allclose(sv_probs, dm_probs, atol=1e-5)
        assert np.allclose(mps_probs, dm_probs, atol=1e-5)
        # q0 reset to |0>: its marginal P(q0=1) must vanish.
        assert sv_probs.reshape(2, 2)[1, :].sum() == pytest.approx(0.0, abs=1e-6)

    def test_reset_pure_one_on_mps(self):
        """MPS reset on a pure |1> must yield |0> (not a zeroed/invalid state)."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.reset(0)
        state = _mps_to_statevector(simulate_mps(qc))
        assert float(state[0].abs().item()) == pytest.approx(1.0, abs=1e-9)
        assert float(state[1].abs().item()) == pytest.approx(0.0, abs=1e-9)

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

    def test_prefers_mps_when_available(self, monkeypatch):
        monkeypatch.setattr(sim_common, "_mps_is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        d = auto_sim_device(None)

        assert d == torch.device("mps")

    def test_picks_least_used_cuda_device(self, monkeypatch):
        monkeypatch.setattr(sim_common, "_mps_is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

        utilization_by_device = {0: 30, 1: 80, 2: 10}
        free_by_device = {0: 50, 1: 10, 2: 50}

        monkeypatch.setattr(
            sim_common,
            "_cuda_utilization_percent",
            lambda device_index: utilization_by_device[device_index],
        )
        monkeypatch.setattr(
            sim_common,
            "_cuda_free_memory_bytes",
            lambda device_index: free_by_device[device_index],
        )

        d = auto_sim_device(None)

        assert d == torch.device("cuda:2")


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


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
#  Clifford & Clifford+T Heisenberg-picture simulators
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

import numpy as np

from fieldqkit.core.observables import pauli_basis_pattern
from fieldqkit.sim.clifford import (
    CliffordError,
    is_clifford_circuit,
    simulate_clifford_expectation,
    simulate_clifford_expectations,
)
from fieldqkit.sim.clifford_t import (
    count_non_clifford_gates,
    count_t_gates,
    simulate_clifford_t_expectation,
    simulate_clifford_t_expectations,
)


def _statevector_expectations(qc: QuantumCircuit, observables):
    """Reference statevector expectation values for a list of Pauli strings."""
    psi = simulate_statevector(qc).reshape(-1).cpu().numpy().astype(np.complex128)
    pauli_mats = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.diag([1.0, -1.0]).astype(np.complex128),
    }
    out = {}
    for obs in observables:
        pat = pauli_basis_pattern(obs, qc.nqubits)
        op = np.array([[1.0 + 0j]], dtype=np.complex128)
        for char in pat:
            op = np.kron(op, pauli_mats[char])
        out[obs] = float(np.real(psi.conj() @ op @ psi))
    return out


class TestStabilizerClifford:
    def test_h_cx_bell_pair(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert is_clifford_circuit(qc)
        expectations = simulate_clifford_expectations(qc, ["ZZ", "XX", "YY", "IZ", "ZI"])
        assert expectations["ZZ"] == pytest.approx(1.0)
        assert expectations["XX"] == pytest.approx(1.0)
        assert expectations["YY"] == pytest.approx(-1.0)
        assert expectations["IZ"] == pytest.approx(0.0)
        assert expectations["ZI"] == pytest.approx(0.0)

    def test_single_qubit_paulis(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        qc.h(0)
        # H S H |0? = (|0? + i|1?)/��2  �� ?X?=0, ?Y?=1, ?Z?=0.
        assert simulate_clifford_expectation(qc, "X") == pytest.approx(0.0)
        assert simulate_clifford_expectation(qc, "Y") == pytest.approx(-1.0)
        assert simulate_clifford_expectation(qc, "Z") == pytest.approx(0.0)

    def test_rotation_pi_over_two_is_clifford(self):
        qc = QuantumCircuit(1)
        qc.rz(np.pi / 2, 0)  # equivalent to S up to phase
        qc.h(0)
        qc.rx(np.pi / 2, 0)
        assert is_clifford_circuit(qc)
        # Sanity-cross-check with statevector.
        ref = _statevector_expectations(qc, ["X", "Y", "Z"])
        got = simulate_clifford_expectations(qc, ["X", "Y", "Z"])
        for key in ["X", "Y", "Z"]:
            assert got[key] == pytest.approx(ref[key], abs=1e-9)

    def test_non_clifford_t_raises(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.t(0)
        assert not is_clifford_circuit(qc)
        with pytest.raises(CliffordError):
            simulate_clifford_expectation(qc, "Z")

    def test_arbitrary_rotation_raises(self):
        qc = QuantumCircuit(1)
        qc.rz(0.3, 0)
        with pytest.raises(CliffordError):
            simulate_clifford_expectation(qc, "Z")

    def test_matches_statevector_random_clifford(self):
        rng = np.random.default_rng(0)
        qc = QuantumCircuit(3)
        clifford_gates = ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg"]
        for _ in range(40):
            kind = rng.integers(0, 3)
            if kind == 0:
                getattr(qc, clifford_gates[rng.integers(0, len(clifford_gates))])(int(rng.integers(0, 3)))
            elif kind == 1:
                a, b = rng.choice(3, size=2, replace=False)
                getattr(qc, ["cx", "cz", "swap"][rng.integers(0, 3)])(int(a), int(b))
            else:
                axis = ["rx", "ry", "rz"][rng.integers(0, 3)]
                k = int(rng.integers(0, 4))
                getattr(qc, axis)(k * np.pi / 2.0, int(rng.integers(0, 3)))
        observables = ["ZZZ", "XXX", "XYZ", "YZX", "IZI", "IZX"]
        got = simulate_clifford_expectations(qc, observables)
        ref = _statevector_expectations(qc, observables)
        for obs in observables:
            assert got[obs] == pytest.approx(ref[obs], abs=1e-9)


class TestCliffordTBranching:
    def test_pure_clifford_matches_stabilizer(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cliff = simulate_clifford_expectations(qc, ["ZZ", "XX", "ZI"])
        branch = simulate_clifford_t_expectations(qc, ["ZZ", "XX", "ZI"])
        for key in cliff:
            assert branch[key] == pytest.approx(cliff[key], abs=1e-12)

    def test_single_t_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.t(0)
        # State (|0? + e^{i��/4}|1?)/��2 �� ?X? = cos(��/4), ?Y? = sin(��/4), ?Z? = 0.
        assert simulate_clifford_t_expectation(qc, "X") == pytest.approx(np.cos(np.pi / 4), abs=1e-9)
        assert simulate_clifford_t_expectation(qc, "Y") == pytest.approx(np.sin(np.pi / 4), abs=1e-9)
        assert simulate_clifford_t_expectation(qc, "Z") == pytest.approx(0.0, abs=1e-9)

    def test_count_helpers(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)
        qc.cx(0, 1)
        qc.tdg(1)
        qc.rz(0.3, 0)
        assert count_t_gates(qc) == 2
        assert count_non_clifford_gates(qc) == 3

    def test_matches_statevector_with_rotations(self):
        qc = QuantumCircuit(3)
        qc.h(0); qc.h(1); qc.h(2)
        qc.rx(0.3, 0)
        qc.ry(0.7, 1)
        qc.rz(1.1, 2)
        qc.cx(0, 1)
        qc.rzz(0.5, 1, 2)
        qc.t(0)
        qc.u(0.4, -0.6, 0.9, 1)
        observables = ["ZZZ", "XXX", "YYY", "XYZ", "IZX", "IIY"]
        got = simulate_clifford_t_expectations(qc, observables)
        ref = _statevector_expectations(qc, observables)
        for obs in observables:
            assert got[obs] == pytest.approx(ref[obs], abs=1e-7)

    def test_random_circuit_matches_statevector(self):
        rng = np.random.default_rng(123)
        qc = QuantumCircuit(3)
        choices = ["h", "s", "sdg", "x", "y", "z", "t", "tdg", "cx", "cz", "rx", "ry", "rz"]
        for _ in range(15):
            g = choices[int(rng.integers(0, len(choices)))]
            if g in {"rx", "ry", "rz"}:
                getattr(qc, g)(float(rng.uniform(-np.pi, np.pi)), int(rng.integers(0, 3)))
            elif g in {"cx", "cz"}:
                a, b = rng.choice(3, size=2, replace=False)
                getattr(qc, g)(int(a), int(b))
            else:
                getattr(qc, g)(int(rng.integers(0, 3)))
        observables = ["ZZZ", "XYZ", "IZI", "YXI", "ZIX"]
        got = simulate_clifford_t_expectations(qc, observables)
        ref = _statevector_expectations(qc, observables)
        for obs in observables:
            assert got[obs] == pytest.approx(ref[obs], abs=1e-7)

    def test_max_terms_guard(self):
        # Several T gates spread across qubits with entangling layers prevent
        # dedup from collapsing the Pauli sum, blowing past max_terms.
        qc = QuantumCircuit(3)
        qc.h(0); qc.h(1); qc.h(2)
        for _ in range(3):
            qc.t(0); qc.t(1); qc.t(2)
            qc.cx(0, 1); qc.cx(1, 2)
        with pytest.raises(RuntimeError, match="max_terms"):
            simulate_clifford_t_expectation(qc, "XXX", max_terms=4)

    def test_qubit_mapping_via_compact(self):
        """Sparse physical layout reproduces the dense statevector result."""
        from fieldqkit.api.client import QuantumHardwareClient

        client = QuantumHardwareClient()
        # Logical 0,1,2 mapped to physical 3,5,7.
        qc_phys = QuantumCircuit(8)
        qc_phys.h(3); qc_phys.cx(3, 5); qc_phys.t(5)
        qc_phys.cz(5, 7); qc_phys.rx(0.7, 7); qc_phys.cx(3, 7)

        qc_sim, _mapping = client._compact_for_sim(qc_phys, target_qubits=[3, 5, 7])
        got = simulate_clifford_t_expectations(qc_sim, ["XYZ", "ZZZ"], num_qubits=3)

        qc_ref = QuantumCircuit(3)
        qc_ref.h(0); qc_ref.cx(0, 1); qc_ref.t(1)
        qc_ref.cz(1, 2); qc_ref.rx(0.7, 2); qc_ref.cx(0, 2)
        ref = _statevector_expectations(qc_ref, ["XYZ", "ZZZ"])
        for obs in ["XYZ", "ZZZ"]:
            assert got[obs] == pytest.approx(ref[obs], abs=1e-7)


# ═══════════════════════════════════════════════════════════
#  Partial-measurement projection (interface dispatch)
#
#  fieldqkit.sim.simulate_counts (the interface-level entry that
#  client.run_auto uses) samples the full statevector from the chosen
#  engine (SV/MPS/DM) and then projects counts onto the classical-bit
#  subspace defined by the circuit's measure gates: each measured qubit
#  maps to its cbit, unmeasured qubits are marginalized out.  These tests
#  lock that behavior across all three engines.
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def restore_sim_config():
    """Snapshot and restore the global simulator config around a test."""
    from fieldqkit.sim import get_sim_config, set_sim_config

    cfg = get_sim_config()
    yield
    set_sim_config(
        mps_threshold_qubits=cfg["mps_threshold_qubits"],
        max_bond_dim=cfg["max_bond_dim"],
    )


def _bell_with_spectator(meas_q, meas_c):
    """Bell pair on q0,q1; q2 forced to |1>, q3 stays |0>; partial measure."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.measure(meas_q, meas_c)
    return qc


class TestPartialMeasurementProjection:
    def test_statevector_subset_marginalizes_unmeasured(self):
        from fieldqkit.sim import simulate_counts

        counts = simulate_counts(_bell_with_spectator([0, 1], [0, 1]), 4000, seed=1)
        assert set(counts) <= {"00", "11"}          # q2/q3 marginalized away
        assert set(counts) == {"00", "11"}          # both Bell branches present
        assert all(len(k) == 2 for k in counts)     # width = #cbits, not #qubits
        assert sum(counts.values()) == 4000

    def test_qubit_to_cbit_mapping_is_honored(self):
        from fieldqkit.sim import simulate_counts

        # cbit0 <- q1 (Bell, 0/1), cbit1 <- q2 (always 1).
        counts = simulate_counts(_bell_with_spectator([1, 2], [0, 1]), 4000, seed=1)
        assert set(counts) == {"01", "11"}
        assert all(k[1] == "1" for k in counts)      # cbit1 (q2) pinned to 1

    def test_full_measure_preserves_cbit_positions(self):
        from fieldqkit.sim import simulate_counts

        counts = simulate_counts(_bell_with_spectator([0, 1, 2, 3], [0, 1, 2, 3]), 4000, seed=1)
        assert set(counts) == {"0010", "1110"}       # cbit2=q2=1, cbit3=q3=0
        assert all(len(k) == 4 for k in counts)

    def test_sparse_cbit_indices_zero_fill_gaps(self):
        from fieldqkit.sim import simulate_counts

        # q0->cbit0, q1->cbit2; cbit1 is never written -> stays 0; width=3.
        counts = simulate_counts(_bell_with_spectator([0, 1], [0, 2]), 4000, seed=1)
        assert set(counts) == {"000", "101"}
        assert all(len(k) == 3 and k[1] == "0" for k in counts)

    def test_projection_equals_manual_marginalization(self):
        from fieldqkit.sim import simulate_counts

        partial = simulate_counts(_bell_with_spectator([0, 1], [0, 1]), 6000, seed=7)
        full = simulate_counts(_bell_with_spectator([0, 1, 2, 3], [0, 1, 2, 3]), 6000, seed=7)
        marginal = {}
        for bits, c in full.items():
            marginal[bits[:2]] = marginal.get(bits[:2], 0) + c   # keep cbits 0,1
        assert partial == marginal

    def test_density_matrix_engine_projects(self):
        from fieldqkit.sim import simulate_counts

        qc = QuantumCircuit(4)
        qc.h(0); qc.cx(0, 1); qc.x(2)
        qc.depolarize1(0.0, 0)                        # routes to DM backend, no-op strength
        qc.measure([0, 1], [0, 1])
        counts = simulate_counts(qc, 4000, seed=1)
        assert set(counts) == {"00", "11"}
        assert all(len(k) == 2 for k in counts)

    def test_mps_engine_projects(self, restore_sim_config):
        from fieldqkit.sim import set_sim_config, simulate_counts

        set_sim_config(mps_threshold_qubits=1)        # force MPS for a small circuit
        counts = simulate_counts(_bell_with_spectator([1, 2], [0, 1]), 4000, seed=1)
        assert set(counts) == {"01", "11"}

    def test_no_measure_gate_returns_full_width(self):
        from fieldqkit.sim import simulate_counts

        qc = QuantumCircuit(3)
        qc.h(0); qc.cx(0, 1); qc.x(2)
        counts = simulate_counts(qc, 2000, seed=1)
        assert all(len(k) == 3 for k in counts)       # no projection without measure
        assert set(counts) == {"001", "111"}

    def test_compaction_then_projection_uses_dense_indices(self):
        """Sparse physical qubits + partial measure: compaction remaps measure
        qubits into the dense space while preserving the cbit assignment."""
        from fieldqkit.api.client import QuantumHardwareClient
        from fieldqkit.sim import simulate_counts

        client = QuantumHardwareClient()
        qc_phys = QuantumCircuit(8)
        qc_phys.h(3); qc_phys.cx(3, 5); qc_phys.x(7)
        qc_phys.measure([3, 5], [0, 1])               # measure two of the three used qubits
        qc_sim, _ = client._compact_for_sim(qc_phys, target_qubits=[3, 5, 7])
        counts = simulate_counts(qc_sim, 4000, seed=1)
        assert set(counts) == {"00", "11"}            # Bell on the two measured qubits
        assert all(len(k) == 2 for k in counts)


# ═══════════════════════════════════════════════════════════
#  Large-scale / boundary simulation
# ═══════════════════════════════════════════════════════════


def _ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


class TestSimulationScaleAndBoundaries:
    def test_large_statevector_ghz_is_normalized(self):
        state = simulate_statevector(_ghz(12))
        assert state.numel() == 2 ** 12
        norm = float((state.abs() ** 2).sum().item())
        assert norm == pytest.approx(1.0, abs=1e-9)
        # Only the all-0 and all-1 amplitudes are populated.
        probs = (state.abs() ** 2).real
        assert float(probs[0].item()) == pytest.approx(0.5, abs=1e-9)
        assert float(probs[-1].item()) == pytest.approx(0.5, abs=1e-9)

    def test_large_statevector_counts_only_ghz_branches(self):
        from fieldqkit.sim import simulate_counts

        counts = simulate_counts(_ghz(12), 5000, seed=3)
        assert set(counts) == {"0" * 12, "1" * 12}
        assert sum(counts.values()) == 5000

    def test_mps_large_ghz_matches_expected_branches(self, restore_sim_config):
        from fieldqkit.sim import set_sim_config, simulate_counts

        set_sim_config(mps_threshold_qubits=8)        # 18 > 8 -> MPS engine
        counts = simulate_counts(_ghz(18), 3000, seed=5)
        assert set(counts) <= {"0" * 18, "1" * 18}
        assert set(counts) == {"0" * 18, "1" * 18}

    def test_single_qubit_circuit(self):
        from fieldqkit.sim import simulate_counts

        qc = QuantumCircuit(1)
        qc.x(0)
        counts = simulate_counts(qc, 256, seed=0)
        assert counts == {"1": 256}

    def test_shots_one_returns_single_outcome(self):
        from fieldqkit.sim import simulate_counts

        counts = simulate_counts(_ghz(4), 1, seed=0)
        assert sum(counts.values()) == 1
        assert len(counts) == 1

    def test_identity_observable_is_one_large(self):
        from fieldqkit.sim.statevector import expectation_pauli

        state = simulate_statevector(_ghz(10))
        val = expectation_pauli(state, "I" * 10, num_qubits=10)
        assert complex(val).real == pytest.approx(1.0, abs=1e-9)

    def test_seeded_counts_are_reproducible(self):
        from fieldqkit.sim import simulate_counts

        a = simulate_counts(_ghz(6), 2000, seed=42)
        b = simulate_counts(_ghz(6), 2000, seed=42)
        assert a == b