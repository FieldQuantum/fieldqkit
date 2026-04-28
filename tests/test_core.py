"""Tests for the core module: circuits, observables, readout, zne, utils."""

import pytest
import numpy as np

from quantum_hw.core.circuits import (
    build_ghz,
    build_cluster,
    build_qft,
    build_ising_time_evolution,
)
from quantum_hw.core.observables import (
    _parse_pauli_string,
    pauli_support,
    shift_pauli_string,
    pauli_basis_pattern,
    pauli_expectation,
    group_observables,
    apply_measurement_basis_rotations,
    _compatible_with_basis,
    _merge_basis,
)
from quantum_hw.core.readout import (
    build_local_confusion_matrix,
    mitigate_readout,
    expectation_from_samples_unbiased,
    mitigate_observable_from_samples,
)
from quantum_hw.core.zne import apply_zne_cz_tripling, zne_linear_extrapolate
from quantum_hw.core.utils import (
    get_probabilities,
    get_samples,
    get_probabilities_from_samples,
    marginal_samples,
    get_local_probabilities_from_samples,
    expectation_from_probabilities,
)
from quantum_hw.core.types import (
    RunResult,
    CalibrationResult,
    ShadowResult,
    VQEResult,
    QAOAResult,
    QMLResult,
    QBMResult,
)


# ═══════════════════════════════════════════════════════════
#  Circuit builders
# ═══════════════════════════════════════════════════════════


class TestBuildGHZ:
    def test_basic_3_qubit(self):
        qc = build_ghz(3)
        assert qc.nqubits == 3
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 1
        assert gate_names.count("cx") == 2

    def test_single_qubit(self):
        qc = build_ghz(1)
        assert qc.nqubits == 1
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 1
        assert gate_names.count("cx") == 0

    def test_with_measure(self):
        qc = build_ghz(3, measure=True)
        gate_names = [g[0] for g in qc.gates]
        assert "barrier" in gate_names
        assert "measure" in gate_names

    def test_without_measure(self):
        qc = build_ghz(3, measure=False)
        gate_names = [g[0] for g in qc.gates]
        assert "measure" not in gate_names


class TestBuildCluster:
    def test_basic_4_qubit(self):
        qc = build_cluster(4)
        assert qc.nqubits == 4
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 4
        assert gate_names.count("cz") >= 1

    def test_single_qubit(self):
        qc = build_cluster(1)
        assert qc.nqubits == 1
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 1
        assert gate_names.count("cz") == 0

    def test_two_qubits(self):
        qc = build_cluster(2)
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("cz") == 1

    def test_three_qubits_odd(self):
        """Odd qubit count exercises both CZ layers."""
        qc = build_cluster(3)
        gate_names = [g[0] for g in qc.gates]
        # Layer 1: (0,1), Layer 2: (1,2) -> 2 CZ gates
        assert gate_names.count("cz") == 2


class TestBuildQFT:
    def test_basic_3_qubit(self):
        qc = build_qft(3)
        assert qc.nqubits == 3
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 3
        assert "swap" in gate_names

    def test_without_swaps(self):
        qc = build_qft(3, with_swaps=False)
        gate_names = [g[0] for g in qc.gates]
        assert "swap" not in gate_names

    def test_single_qubit_no_controlled_phase(self):
        qc = build_qft(1)
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 1
        # No controlled phase or swap needed for 1 qubit
        assert "swap" not in gate_names

    def test_two_qubits_has_swap(self):
        qc = build_qft(2, with_swaps=True)
        gate_names = [g[0] for g in qc.gates]
        assert "swap" in gate_names

    def test_with_measure(self):
        qc = build_qft(2, measure=True)
        gate_names = [g[0] for g in qc.gates]
        assert "measure" in gate_names


class TestBuildIsingTimeEvolution:
    def test_basic_circuit(self):
        qc = build_ising_time_evolution(3, j=1.0, h=0.5, t=1.0)
        assert qc.nqubits == 3
        gate_names = [g[0] for g in qc.gates]
        assert "cx" in gate_names
        assert "rz" in gate_names
        assert "rx" in gate_names

    def test_multiple_steps(self):
        qc1 = build_ising_time_evolution(2, j=1.0, h=1.0, t=1.0, steps=1)
        qc2 = build_ising_time_evolution(2, j=1.0, h=1.0, t=1.0, steps=3)
        assert len(qc2.gates) > len(qc1.gates)

    def test_single_qubit_no_zz_interaction(self):
        qc = build_ising_time_evolution(1, j=1.0, h=1.0, t=1.0)
        gate_names = [g[0] for g in qc.gates]
        assert "cx" not in gate_names
        assert "rx" in gate_names

    def test_zero_coupling_still_has_rx(self):
        qc = build_ising_time_evolution(2, j=0.0, h=1.0, t=1.0)
        gate_names = [g[0] for g in qc.gates]
        assert "rx" in gate_names


# ═══════════════════════════════════════════════════════════
#  Observables
# ═══════════════════════════════════════════════════════════


class TestParsePauliString:
    def test_compact_form(self):
        terms = _parse_pauli_string("ZZIX")
        assert (0, "Z") in terms
        assert (1, "Z") in terms
        assert (3, "X") in terms
        # Identity at index 2 should be omitted
        assert all(idx != 2 for idx, _ in terms)

    def test_indexed_form(self):
        terms = _parse_pauli_string("Z0 X2")
        assert (0, "Z") in terms
        assert (2, "X") in terms

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_pauli_string("")

    def test_compact_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            _parse_pauli_string("ZZ", num_qubits=3)

    def test_indexed_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _parse_pauli_string("Z5", num_qubits=3)

    def test_negative_index_raises(self):
        # Indexed form with a negative index (parsed as int, should fail bounds check)
        with pytest.raises((ValueError, Exception)):
            _parse_pauli_string("Z-1", num_qubits=3)

    def test_unsupported_pauli_in_indexed_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            _parse_pauli_string("A0 B1")

    def test_all_identity_returns_empty(self):
        terms = _parse_pauli_string("III")
        assert terms == []

    def test_indexed_identity_omitted(self):
        terms = _parse_pauli_string("I0 Z1")
        assert len(terms) == 1
        assert terms[0] == (1, "Z")


class TestPauliSupport:
    def test_basic(self):
        support = pauli_support("Z0 X2")
        assert support == [0, 2]

    def test_all_identity(self):
        support = pauli_support("III")
        assert support == []

    def test_compact_form(self):
        support = pauli_support("XIZI")
        assert support == [0, 2]


class TestShiftPauliString:
    def test_basic_shift(self):
        result = shift_pauli_string("Z0 X2", 3)
        assert "Z3" in result
        assert "X5" in result

    def test_empty_terms_returns_empty(self):
        result = shift_pauli_string("I0", 3)
        assert result == ""


class TestPauliBasisPattern:
    def test_basic(self):
        pattern = pauli_basis_pattern("Z0 X2", num_qubits=4)
        assert pattern == ["Z", "I", "X", "I"]

    def test_compact(self):
        pattern = pauli_basis_pattern("XYIZ", num_qubits=4)
        assert pattern == ["X", "Y", "I", "Z"]

    def test_num_qubits_must_be_int(self):
        with pytest.raises(TypeError):
            pauli_basis_pattern("XYIZ", num_qubits=None)  # type: ignore[arg-type]

    def test_num_qubits_negative_raises(self):
        with pytest.raises(ValueError):
            pauli_basis_pattern("X0", num_qubits=-1)


class TestGroupObservables:
    def test_compatible_grouped_together(self):
        groups = group_observables(["ZZII", "ZIZI"], num_qubits=4)
        # Both are Z-basis compatible, should be in one group
        assert len(groups) == 1
        assert len(groups[0]["observables"]) == 2

    def test_incompatible_separated(self):
        groups = group_observables(["ZZII", "XXII"], num_qubits=4)
        assert len(groups) == 2

    def test_empty_input(self):
        groups = group_observables([], num_qubits=4)
        assert groups == []

    def test_single_observable(self):
        groups = group_observables(["XIZI"], num_qubits=4)
        assert len(groups) == 1


class TestCompatibleWithBasis:
    def test_both_identity(self):
        assert _compatible_with_basis(["I", "I"], ["I", "I"])

    def test_compatible(self):
        assert _compatible_with_basis(["Z", "I"], ["I", "X"])

    def test_incompatible(self):
        assert not _compatible_with_basis(["Z", "I"], ["X", "I"])


class TestMergeBasis:
    def test_merge(self):
        merged = _merge_basis(["Z", "I"], ["I", "X"])
        assert merged == ["Z", "X"]

    def test_no_overwrite(self):
        merged = _merge_basis(["I", "I"], ["Z", "X"])
        assert merged == ["Z", "X"]


class TestPauliExpectation:
    def test_all_zeros_z_basis(self):
        # All measurements are 0 -> eigenvalue +1 for Z
        samples = np.array([[0, 0], [0, 0], [0, 0]])
        exp = pauli_expectation(samples, "ZZ")
        assert exp == pytest.approx(1.0)

    def test_all_ones_z_basis(self):
        # All measurements are 1 -> eigenvalue (-1)^2 = +1 for ZZ
        samples = np.array([[1, 1], [1, 1]])
        exp = pauli_expectation(samples, "ZZ")
        assert exp == pytest.approx(1.0)

    def test_mixed_z_basis(self):
        # 01 -> Z eigenvalues: +1, -1 -> parity -1
        samples = np.array([[0, 1], [0, 1]])
        exp = pauli_expectation(samples, "ZZ")
        assert exp == pytest.approx(-1.0)

    def test_identity_observable(self):
        samples = np.array([[0, 1], [1, 0]])
        exp = pauli_expectation(samples, "II")
        assert exp == pytest.approx(1.0)

    def test_1d_samples_raises(self):
        with pytest.raises(ValueError, match="2D"):
            pauli_expectation(np.array([0, 1, 0]), "ZZZ")

    def test_single_qubit(self):
        samples = np.array([[0], [1], [0], [1]])
        exp = pauli_expectation(samples, "Z")
        assert exp == pytest.approx(0.0)


class TestApplyMeasurementBasisRotations:
    def test_unsupported_basis_raises(self):
        from quantum_hw.circuit import QuantumCircuit
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="unsupported"):
            apply_measurement_basis_rotations(qc, ["Q", "Z"], target_qubits=[0, 1])


# ═══════════════════════════════════════════════════════════
#  Readout mitigation
# ═══════════════════════════════════════════════════════════


class TestBuildLocalConfusionMatrix:
    def test_single_qubit(self):
        per_qubit = {0: np.eye(2)}
        cm = build_local_confusion_matrix(per_qubit, [0])
        np.testing.assert_array_almost_equal(cm, np.eye(2))

    def test_two_qubits_identity(self):
        per_qubit = {0: np.eye(2), 1: np.eye(2)}
        cm = build_local_confusion_matrix(per_qubit, [0, 1])
        np.testing.assert_array_almost_equal(cm, np.eye(4))

    def test_empty_target_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_local_confusion_matrix({}, [])

    def test_kronecker_order(self):
        m0 = np.array([[0.9, 0.1], [0.05, 0.95]])
        m1 = np.array([[0.8, 0.2], [0.1, 0.9]])
        per_qubit = {0: m0, 1: m1}
        cm = build_local_confusion_matrix(per_qubit, [0, 1])
        expected = np.kron(m0, m1)
        np.testing.assert_array_almost_equal(cm, expected)


class TestMitigateReadout:
    def test_identity_confusion_no_change(self):
        probs = np.array([0.5, 0.5])
        cm = np.eye(2)
        result = mitigate_readout(probs, cm)
        np.testing.assert_array_almost_equal(result, probs)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            mitigate_readout(np.array([0.5, 0.5]), np.array([[1, 0, 0], [0, 1, 0]]))

    def test_clipping_and_renormalization(self):
        """Pseudo-inverse can produce negative values; they should be clipped."""
        probs = np.array([0.3, 0.7])
        # Heavily biased confusion matrix
        cm = np.array([[0.5, 0.5], [0.4, 0.6]])
        result = mitigate_readout(probs, cm)
        assert np.all(result >= 0)
        assert result.sum() == pytest.approx(1.0)

    def test_zero_sum_probabilities(self):
        probs = np.array([0.0, 0.0])
        cm = np.eye(2)
        result = mitigate_readout(probs, cm)
        assert result.sum() == pytest.approx(0.0)


class TestExpectationFromSamplesUnbiased:
    def test_perfect_readout_all_zero(self):
        samples = np.array([[0, 0], [0, 0], [0, 0]])
        cms = [np.eye(2), np.eye(2)]
        exp = expectation_from_samples_unbiased(samples, cms)
        assert exp == pytest.approx(1.0)

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="2D"):
            expectation_from_samples_unbiased(np.array([0, 1]), [np.eye(2)])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            expectation_from_samples_unbiased(np.array([[0, 1]]), [np.eye(2)])

    def test_non_binary_raises(self):
        with pytest.raises(ValueError, match="0/1"):
            expectation_from_samples_unbiased(np.array([[0, 2]]), [np.eye(2), np.eye(2)])

    def test_empty_qubits_returns_one(self):
        samples = np.zeros((5, 0), dtype=int)
        exp = expectation_from_samples_unbiased(samples, [])
        assert exp == pytest.approx(1.0)

    def test_no_shots_returns_zero(self):
        samples = np.zeros((0, 2), dtype=int)
        cms = [np.eye(2), np.eye(2)]
        exp = expectation_from_samples_unbiased(samples, cms)
        assert exp == pytest.approx(0.0)

    def test_invalid_cm_shape_raises(self):
        with pytest.raises(ValueError, match="shape \\(2, 2\\)"):
            expectation_from_samples_unbiased(np.array([[0]]), [np.eye(3)])


# ═══════════════════════════════════════════════════════════
#  ZNE
# ═══════════════════════════════════════════════════════════


class TestZNE:
    def test_cz_tripling_triples_cz_gates(self):
        class MockCircuit:
            def __init__(self):
                self.gates = [("h", 0), ("cz", 0, 1), ("h", 1)]

        qc = MockCircuit()
        result = apply_zne_cz_tripling(qc)
        cz_count = sum(1 for g in result.gates if g[0] == "cz")
        assert cz_count == 3

    def test_cz_tripling_preserves_non_cz(self):
        class MockCircuit:
            def __init__(self):
                self.gates = [("h", 0), ("rx", 0.5, 0)]

        qc = MockCircuit()
        result = apply_zne_cz_tripling(qc)
        assert len(result.gates) == 2

    def test_cz_tripling_does_not_mutate_original(self):
        class MockCircuit:
            def __init__(self):
                self.gates = [("cz", 0, 1)]

        qc = MockCircuit()
        apply_zne_cz_tripling(qc)
        assert len(qc.gates) == 1

    def test_linear_extrapolate_scalar(self):
        # f(0) = (3*f(1) - f(3))/2
        result = zne_linear_extrapolate(1.0, 0.5)
        assert result == pytest.approx(1.25)

    def test_linear_extrapolate_vector(self):
        p1 = np.array([0.9, 0.1])
        p3 = np.array([0.7, 0.3])
        result = zne_linear_extrapolate(p1, p3)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 0.0]))

    def test_linear_extrapolate_exact_same(self):
        """If noise scale 1 and 3 give same result, extrapolation = same."""
        result = zne_linear_extrapolate(0.5, 0.5)
        assert result == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════
#  Utils
# ═══════════════════════════════════════════════════════════


class TestGetSamples:
    def test_basic(self):
        result = {"00": 2, "11": 1}
        samples = get_samples(result, 2)
        assert samples.shape == (3, 2)
        assert samples.dtype == int

    def test_single_bitstring(self):
        result = {"101": 4}
        samples = get_samples(result, 3)
        assert samples.shape == (4, 3)
        np.testing.assert_array_equal(samples[0], [1, 0, 1])


class TestGetProbabilities:
    def test_uniform(self):
        result = {"00": 50, "01": 50, "10": 50, "11": 50}
        probs = get_probabilities(result, 2)
        np.testing.assert_array_almost_equal(probs, [0.25, 0.25, 0.25, 0.25])

    def test_single_outcome(self):
        result = {"10": 100}
        probs = get_probabilities(result, 2)
        assert probs[2] == pytest.approx(1.0)  # "10" = index 2


class TestGetProbabilitiesFromSamples:
    def test_empty_samples(self):
        probs = get_probabilities_from_samples(np.zeros((0, 2), dtype=int), 2)
        np.testing.assert_array_equal(probs, [0.0, 0.0, 0.0, 0.0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="2D"):
            get_probabilities_from_samples(np.array([0, 1, 0]), 3)

    def test_wrong_num_qubits_raises(self):
        with pytest.raises(ValueError, match="2D"):
            get_probabilities_from_samples(np.array([[0, 1]]), 3)


class TestMarginalSamples:
    def test_basic(self):
        samples = np.array([[0, 1, 0], [1, 0, 1]])
        marginal = marginal_samples(samples, [0, 2])
        np.testing.assert_array_equal(marginal, [[0, 0], [1, 1]])

    def test_empty_support(self):
        samples = np.array([[0, 1], [1, 0]])
        marginal = marginal_samples(samples, [])
        assert marginal.shape == (2, 0)


class TestExpectationFromProbabilities:
    def test_all_zero_state(self):
        """Probability 1 on |00> -> Z-parity +1."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        exp = expectation_from_probabilities(probs, [0, 1])
        assert exp == pytest.approx(1.0)

    def test_bell_state(self):
        """Equal prob on |00> and |11> -> Z-parity +1 (both even parity)."""
        probs = np.array([0.5, 0.0, 0.0, 0.5])
        exp = expectation_from_probabilities(probs, [0, 1])
        assert exp == pytest.approx(1.0)

    def test_anti_correlated(self):
        """Equal prob on |01> and |10> -> Z-parity -1 (both odd parity)."""
        probs = np.array([0.0, 0.5, 0.5, 0.0])
        exp = expectation_from_probabilities(probs, [0, 1])
        assert exp == pytest.approx(-1.0)

    def test_empty_support(self):
        probs = np.array([0.5, 0.5])
        exp = expectation_from_probabilities(probs, [])
        assert exp == pytest.approx(1.0)

    def test_single_qubit(self):
        probs = np.array([0.7, 0.3])
        exp = expectation_from_probabilities(probs, [0])
        assert exp == pytest.approx(0.4)  # 0.7 - 0.3


class TestGetLocalProbabilitiesFromSamples:
    def test_basic(self):
        samples = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
        probs = get_local_probabilities_from_samples(samples, [1, 2])
        # marginal on qubits 1,2: 01, 10, 01 -> p(01)=2/3, p(10)=1/3
        assert probs[1] == pytest.approx(2 / 3)
        assert probs[2] == pytest.approx(1 / 3)

    def test_empty_support(self):
        samples = np.array([[0, 1], [1, 0]])
        probs = get_local_probabilities_from_samples(samples, [])
        np.testing.assert_array_almost_equal(probs, [1.0])


# ═══════════════════════════════════════════════════════════
#  Result types (dataclass smoke tests)
# ═══════════════════════════════════════════════════════════


class TestResultTypes:
    def test_run_result_fields(self):
        r = RunResult(
            task_ids=["t1"],
            samples=[[[0, 1]]],
            samples_zne=None,
            probabilities=[[0.5, 0.5]],
            probabilities_raw=[[0.5, 0.5]],
            observable_values={"ZZ": 0.9},
            observable_values_raw={"ZZ": 0.8},
        )
        assert r.task_ids == ["t1"]
        assert r.observable_values["ZZ"] == 0.9

    def test_vqe_result_defaults(self):
        r = VQEResult(best_energy=-1.0, best_params=[0.1], energy_history=[-0.5, -1.0])
        assert r.params_history is None
        assert r.best_energy == -1.0

    def test_shadow_result_defaults(self):
        r = ShadowResult()
        assert r.task_ids is None
        assert r.num_samples is None

    def test_qml_result_fields(self):
        r = QMLResult(task="classification", best_loss=0.1, best_params=[0.5], loss_history=[0.5, 0.1])
        assert r.task == "classification"
        assert r.accuracy is None
