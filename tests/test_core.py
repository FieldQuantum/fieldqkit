"""Tests for the core module: circuits, observables, readout, zne, utils."""

import pytest
import numpy as np

from fieldqkit.core.circuits import (
    build_ghz,
    build_cluster,
    build_qft,
    build_ising_time_evolution,
    build_heisenberg_time_evolution,
    build_xxz_time_evolution,
    build_xy_time_evolution,
)
from fieldqkit.core.observables import (
    _parse_pauli_string,
    pauli_support,
    shift_pauli_string,
    pauli_basis_pattern,
    pauli_expectation,
    group_observables,
    apply_measurement_basis_rotations,
    append_measurement_basis,
    _compatible_with_basis,
    _merge_basis,
)
from fieldqkit.core.readout import (
    build_local_confusion_matrix,
    mitigate_readout,
    expectation_from_samples_unbiased,
    mitigate_observable_from_samples,
)
from fieldqkit.core.zne import apply_zne_cz_tripling, zne_linear_extrapolate
from fieldqkit.core.utils import (
    get_probabilities,
    get_samples,
    get_probabilities_from_samples,
    marginal_samples,
    get_local_probabilities_from_samples,
    expectation_from_probabilities,
)
from fieldqkit.core.types import (
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
        assert "rzz" in gate_names
        assert "rx" in gate_names

    def test_multiple_steps(self):
        qc1 = build_ising_time_evolution(2, j=1.0, h=1.0, t=1.0, steps=1)
        qc2 = build_ising_time_evolution(2, j=1.0, h=1.0, t=1.0, steps=3)
        assert len(qc2.gates) > len(qc1.gates)

    def test_single_qubit_no_zz_interaction(self):
        qc = build_ising_time_evolution(1, j=1.0, h=1.0, t=1.0)
        gate_names = [g[0] for g in qc.gates]
        assert "rzz" not in gate_names
        assert "rx" in gate_names

    def test_zero_coupling_still_has_rx(self):
        qc = build_ising_time_evolution(2, j=0.0, h=1.0, t=1.0)
        gate_names = [g[0] for g in qc.gates]
        assert "rx" in gate_names


class TestBuildHeisenbergTimeEvolution:
    def test_basic_circuit(self):
        qc = build_heisenberg_time_evolution(3, t=1.0, jx=1.0, jy=1.0, jz=0.5, hz=0.1)
        assert qc.nqubits == 3
        gate_names = [g[0] for g in qc.gates]
        assert "rxx" in gate_names
        assert "ryy" in gate_names
        assert "rzz" in gate_names
        assert "rz" in gate_names  # from hz term

    def test_zero_couplings_skip_blocks(self):
        qc = build_heisenberg_time_evolution(2, t=1.0, jx=0.0, jy=0.0, jz=0.0, hz=0.0)
        # All couplings zero => no two-qubit gates and no rotations.
        gate_names = [g[0] for g in qc.gates]
        assert "rxx" not in gate_names
        assert "ryy" not in gate_names
        assert "rzz" not in gate_names
        assert "rz" not in gate_names

    def test_more_steps_more_gates(self):
        qc1 = build_heisenberg_time_evolution(2, t=1.0, steps=1)
        qc2 = build_heisenberg_time_evolution(2, t=1.0, steps=3)
        assert len(qc2.gates) > len(qc1.gates)


class TestBuildXxzTimeEvolution:
    def test_basic_circuit(self):
        qc = build_xxz_time_evolution(3, t=1.0, jxy=1.0, jz=0.5, hz=0.0)
        assert qc.nqubits == 3
        gate_names = [g[0] for g in qc.gates]
        assert "rxx" in gate_names
        assert "ryy" in gate_names
        assert "rzz" in gate_names


class TestBuildXyTimeEvolution:
    def test_basic_circuit(self):
        qc = build_xy_time_evolution(3, t=1.0, jx=1.0, jy=1.0, hz=0.0)
        assert qc.nqubits == 3
        gate_names = [g[0] for g in qc.gates]
        assert "rxx" in gate_names
        assert "ryy" in gate_names
        assert "rzz" not in gate_names  # XY has no ZZ coupling

    def test_no_zz_block(self):
        qc = build_xy_time_evolution(2, t=1.0, jx=1.0, jy=1.0, hz=0.0)
        gate_names = [g[0] for g in qc.gates]
        # 1 step * 1 pair * (rxx + ryy) = 2 two-qubit gates per step.
        assert gate_names.count("rxx") == 1
        assert gate_names.count("ryy") == 1
        assert "rzz" not in gate_names


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
        from fieldqkit.circuit import QuantumCircuit
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="unsupported"):
            apply_measurement_basis_rotations(qc, ["Q", "Z"], target_qubits=[0, 1])

    def test_length_mismatch_raises(self):
        # Guards against zip() silently truncating and applying rotations/measurements
        # to the wrong qubits when target_qubits and basis_pattern differ in length.
        from fieldqkit.circuit import QuantumCircuit
        qc = QuantumCircuit(3)
        with pytest.raises(ValueError, match="does not match"):
            apply_measurement_basis_rotations(qc, ["X", "Y", "Z"], target_qubits=[0, 1])
        with pytest.raises(ValueError, match="does not match"):
            append_measurement_basis(qc, ["X", "Y"], target_qubits=[0, 1, 2])


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
        from fieldqkit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cz(0, 1)
        qc.h(1)
        result = apply_zne_cz_tripling(qc)
        cz_count = sum(1 for g in result.gates if g[0] == "cz")
        assert cz_count == 3

    def test_cz_tripling_preserves_non_cz(self):
        from fieldqkit.circuit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rx(0.5, 0)
        result = apply_zne_cz_tripling(qc)
        assert len(result.gates) == 2

    def test_cz_tripling_does_not_mutate_original(self):
        from fieldqkit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        result = apply_zne_cz_tripling(qc)
        # Original gate list is untouched.
        assert len(qc.gates) == 1
        # The result is a deep copy: mutating its state must not leak back.
        result.qubits.append(999)
        result.params_value["leaked"] = 1.0
        assert 999 not in qc.qubits
        assert "leaked" not in qc.params_value

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


# ═══════════════════════════════════════════════════════════
#  Large-scale and boundary cases (appended)
# ═══════════════════════════════════════════════════════════


class TestCircuitBuildersLargeScale:
    def test_ghz_wide_gate_counts(self):
        n = 64
        qc = build_ghz(n)
        assert qc.nqubits == n
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == 1
        assert gate_names.count("cx") == n - 1

    def test_cluster_wide_cz_count(self):
        n = 32
        qc = build_cluster(n)
        assert qc.nqubits == n
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("h") == n
        # Layer 1: n//2 CZ, Layer 2: (n-1)//2 CZ.
        assert gate_names.count("cz") == n // 2 + (n - 1) // 2

    def test_qft_wide_gate_counts(self):
        n = 8
        qc = build_qft(n)
        assert qc.nqubits == n
        gate_names = [g[0] for g in qc.gates]
        pairs = n * (n - 1) // 2
        # Each controlled phase decomposes into 2 cx + 3 rz.
        assert gate_names.count("h") == n
        assert gate_names.count("cx") == 2 * pairs
        assert gate_names.count("rz") == 3 * pairs
        assert gate_names.count("swap") == n // 2

    def test_qft_no_swaps_no_swap_gates(self):
        qc = build_qft(7, with_swaps=False)
        gate_names = [g[0] for g in qc.gates]
        assert "swap" not in gate_names

    def test_ising_trotter_steps_scale_gate_count(self):
        n = 6
        steps = 5
        qc = build_ising_time_evolution(n, j=1.0, h=0.5, t=2.0, steps=steps)
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("rzz") == steps * (n - 1)
        assert gate_names.count("rx") == steps * n

    def test_heisenberg_full_couplings_counts(self):
        n = 5
        steps = 3
        qc = build_heisenberg_time_evolution(
            n, t=1.0, jx=1.0, jy=1.0, jz=1.0, hz=0.3, steps=steps
        )
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("rxx") == steps * (n - 1)
        assert gate_names.count("ryy") == steps * (n - 1)
        assert gate_names.count("rzz") == steps * (n - 1)
        assert gate_names.count("rz") == steps * n

    def test_xxz_counts(self):
        n = 5
        steps = 2
        qc = build_xxz_time_evolution(n, t=1.0, jxy=1.0, jz=0.5, hz=0.0, steps=steps)
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("rxx") == steps * (n - 1)
        assert gate_names.count("ryy") == steps * (n - 1)
        assert gate_names.count("rzz") == steps * (n - 1)
        # hz == 0 -> no longitudinal rz rotations.
        assert "rz" not in gate_names

    def test_xy_counts_no_zz(self):
        n = 6
        steps = 2
        qc = build_xy_time_evolution(n, t=1.0, jx=1.0, jy=1.0, hz=0.0, steps=steps)
        gate_names = [g[0] for g in qc.gates]
        assert gate_names.count("rxx") == steps * (n - 1)
        assert gate_names.count("ryy") == steps * (n - 1)
        assert "rzz" not in gate_names


class TestObservablesBoundaryAndLargeScale:
    def test_pauli_expectation_identity_invariant_wide(self):
        rng = np.random.default_rng(0)
        samples = rng.integers(0, 2, size=(5000, 12))
        # Identity on every qubit always yields +1 regardless of samples.
        assert pauli_expectation(samples, "I" * 12) == pytest.approx(1.0)

    def test_pauli_expectation_full_weight_z_all_zero(self):
        samples = np.zeros((200, 10), dtype=int)
        # All-zero outcomes -> every Z eigenvalue +1 -> parity +1.
        assert pauli_expectation(samples, "Z" * 10) == pytest.approx(1.0)

    def test_pauli_expectation_single_one_flips_parity(self):
        samples = np.zeros((50, 8), dtype=int)
        samples[:, 3] = 1  # one qubit always reads 1
        # Full-weight Z parity with an odd number of -1 eigenvalues -> -1.
        assert pauli_expectation(samples, "Z" * 8) == pytest.approx(-1.0)

    def test_pauli_expectation_x_basis_bits(self):
        # After basis rotation, X is read like Z from the 0/1 bits.
        samples = np.array([[0], [1], [0], [1]])
        assert pauli_expectation(samples, "X") == pytest.approx(0.0)

    def test_group_observables_all_z_single_group(self):
        obs = ["Z" + str(i) for i in range(10)]
        groups = group_observables(obs, num_qubits=10)
        assert len(groups) == 1
        assert len(groups[0]["observables"]) == 10
        assert groups[0]["basis"] == ["Z"] * 10

    def test_group_observables_disjoint_qubits_merge(self):
        # Different Paulis on disjoint qubits are mutually compatible.
        groups = group_observables(["X0", "Y1", "Z2", "X3"], num_qubits=4)
        assert len(groups) == 1
        assert groups[0]["basis"] == ["X", "Y", "Z", "X"]

    def test_group_observables_conflict_on_same_qubit(self):
        # X0 and Z0 conflict on qubit 0 -> two groups.
        groups = group_observables(["X0", "Z0"], num_qubits=2)
        assert len(groups) == 2

    def test_group_observables_many_returns_all(self):
        obs = ["XX", "YY", "ZZ", "XX", "ZZ"]
        groups = group_observables(obs, num_qubits=2)
        total = sum(len(g["observables"]) for g in groups)
        assert total == len(obs)

    def test_pauli_basis_pattern_all_identity_wide(self):
        pattern = pauli_basis_pattern("I" * 16, num_qubits=16)
        assert pattern == ["I"] * 16

    def test_shift_pauli_string_roundtrip_support(self):
        shifted = shift_pauli_string("X0 Z2 Y5", 10)
        assert pauli_support(shifted) == [10, 12, 15]

    def test_apply_basis_rotations_full_pattern(self):
        from fieldqkit.circuit import QuantumCircuit

        qc = QuantumCircuit(3)
        apply_measurement_basis_rotations(qc, ["X", "Y", "Z"], target_qubits=[0, 1, 2])
        gate_names = [g[0] for g in qc.gates]
        # X -> h ; Y -> sdg, h ; Z -> nothing.
        assert gate_names == ["h", "sdg", "h"]

    def test_apply_basis_rotations_identity_is_noop(self):
        from fieldqkit.circuit import QuantumCircuit

        qc = QuantumCircuit(4)
        apply_measurement_basis_rotations(qc, ["I", "I", "I", "I"], target_qubits=[0, 1, 2, 3])
        assert qc.gates == []


class TestReadoutLargeScaleAndBoundary:
    def test_identity_confusion_matrix_no_op_8q(self):
        n = 8
        cm = np.eye(2**n)
        rng = np.random.default_rng(1)
        probs = rng.dirichlet(np.ones(2**n))
        result = mitigate_readout(probs, cm)
        np.testing.assert_array_almost_equal(result, probs)

    def test_build_local_confusion_matrix_8q_identity(self):
        n = 8
        per_qubit = {i: np.eye(2) for i in range(n)}
        cm = build_local_confusion_matrix(per_qubit, list(range(n)))
        assert cm.shape == (2**n, 2**n)
        np.testing.assert_array_almost_equal(cm, np.eye(2**n))

    def test_build_local_confusion_matrix_kron_chain(self):
        rng = np.random.default_rng(2)
        per_qubit = {}
        for q in range(4):
            row0 = rng.random(2)
            row1 = rng.random(2)
            per_qubit[q] = np.array([row0 / row0.sum(), row1 / row1.sum()])
        cm = build_local_confusion_matrix(per_qubit, [0, 1, 2, 3])
        expected = per_qubit[0]
        for q in [1, 2, 3]:
            expected = np.kron(expected, per_qubit[q])
        np.testing.assert_array_almost_equal(cm, expected)

    def test_unbiased_estimator_identity_cm_recovers_parity(self):
        # With ideal (identity) readout the unbiased estimator equals the raw parity.
        rng = np.random.default_rng(3)
        k = 9
        samples = rng.integers(0, 2, size=(4000, k))
        cms = [np.eye(2)] * k
        unbiased = expectation_from_samples_unbiased(samples, cms)
        parity = pauli_expectation(samples, "Z" * k)
        assert unbiased == pytest.approx(parity, abs=1e-9)

    def test_unbiased_estimator_all_zero_returns_one(self):
        samples = np.zeros((100, 10), dtype=int)
        cms = [np.eye(2)] * 10
        assert expectation_from_samples_unbiased(samples, cms) == pytest.approx(1.0)

    def test_mitigate_observable_identity_cm_marginal_path(self):
        from fieldqkit.core.readout import mitigate_observable_from_samples

        rng = np.random.default_rng(4)
        samples = rng.integers(0, 2, size=(500, 6))
        per_qubit = {i: np.eye(2) for i in range(6)}
        support = [0, 2, 4]
        value = mitigate_observable_from_samples(
            samples, support, per_qubit, [0, 1, 2, 3, 4, 5], marginal_max_support=10
        )
        expected = pauli_expectation(samples, "Z0 Z2 Z4")
        assert value == pytest.approx(expected)

    def test_mitigate_observable_identity_cm_unbiased_path(self):
        from fieldqkit.core.readout import mitigate_observable_from_samples

        rng = np.random.default_rng(5)
        samples = rng.integers(0, 2, size=(500, 6))
        per_qubit = {i: np.eye(2) for i in range(6)}
        support = [0, 2, 4]
        # Force the unbiased branch by lowering the marginal threshold.
        value = mitigate_observable_from_samples(
            samples, support, per_qubit, [0, 1, 2, 3, 4, 5], marginal_max_support=1
        )
        expected = pauli_expectation(samples, "Z0 Z2 Z4")
        assert value == pytest.approx(expected)

    def test_mitigate_observable_empty_support_returns_one(self):
        from fieldqkit.core.readout import mitigate_observable_from_samples

        samples = np.zeros((10, 4), dtype=int)
        per_qubit = {i: np.eye(2) for i in range(4)}
        assert mitigate_observable_from_samples(samples, [], per_qubit, [0, 1, 2, 3]) == pytest.approx(1.0)

    def test_mitigate_readout_zero_probs_no_op(self):
        probs = np.zeros(8)
        cm = np.eye(8)
        result = mitigate_readout(probs, cm)
        assert result.sum() == pytest.approx(0.0)


class TestZNELargeScale:
    def test_cz_tripling_on_wide_cluster(self):
        qc = build_cluster(16)
        original_cz = sum(1 for g in qc.gates if g[0] == "cz")
        result = apply_zne_cz_tripling(qc)
        tripled_cz = sum(1 for g in result.gates if g[0] == "cz")
        assert tripled_cz == 3 * original_cz
        # Original is not mutated.
        assert sum(1 for g in qc.gates if g[0] == "cz") == original_cz

    def test_linear_extrapolate_equal_vectors(self):
        rng = np.random.default_rng(6)
        v = rng.random(64)
        result = zne_linear_extrapolate(v, v)
        np.testing.assert_array_almost_equal(result, v)

    def test_linear_extrapolate_scalar_formula(self):
        # (3*f1 - f3)/2
        assert zne_linear_extrapolate(2.0, 1.0) == pytest.approx(2.5)


class TestUtilsLargeScaleAndBoundary:
    def test_get_samples_probabilities_consistency(self):
        result = {"000": 10, "111": 30}
        samples = get_samples(result, 3)
        assert samples.shape == (40, 3)
        probs = get_probabilities(result, 3)
        assert probs[0] == pytest.approx(0.25)  # "000"
        assert probs[7] == pytest.approx(0.75)  # "111"
        assert probs.sum() == pytest.approx(1.0)

    def test_get_probabilities_from_samples_empty_wide(self):
        probs = get_probabilities_from_samples(np.zeros((0, 8), dtype=int), 8)
        assert probs.shape == (2**8,)
        assert probs.sum() == pytest.approx(0.0)

    def test_get_probabilities_from_samples_normalized_large(self):
        rng = np.random.default_rng(7)
        samples = rng.integers(0, 2, size=(20000, 6))
        probs = get_probabilities_from_samples(samples, 6)
        assert probs.shape == (2**6,)
        assert probs.sum() == pytest.approx(1.0)

    def test_marginal_samples_large(self):
        rng = np.random.default_rng(8)
        samples = rng.integers(0, 2, size=(1000, 12))
        marginal = marginal_samples(samples, [3, 7, 11])
        assert marginal.shape == (1000, 3)
        np.testing.assert_array_equal(marginal[:, 0], samples[:, 3])
        np.testing.assert_array_equal(marginal[:, 2], samples[:, 11])

    def test_marginal_samples_empty_support_zero_columns(self):
        samples = np.zeros((25, 9), dtype=int)
        marginal = marginal_samples(samples, [])
        assert marginal.shape == (25, 0)

    def test_local_probabilities_length(self):
        rng = np.random.default_rng(9)
        samples = rng.integers(0, 2, size=(1000, 12))
        probs = get_local_probabilities_from_samples(samples, [3, 7, 11])
        assert probs.shape == (2**3,)
        assert probs.sum() == pytest.approx(1.0)

    def test_local_probabilities_empty_support(self):
        samples = np.zeros((5, 4), dtype=int)
        probs = get_local_probabilities_from_samples(samples, [])
        np.testing.assert_array_almost_equal(probs, [1.0])

    def test_expectation_from_probabilities_all_zero_state_wide(self):
        n = 8
        probs = np.zeros(2**n)
        probs[0] = 1.0  # |00..0>
        exp = expectation_from_probabilities(probs, list(range(n)))
        assert exp == pytest.approx(1.0)

    def test_expectation_from_probabilities_empty_support(self):
        probs = np.array([0.3, 0.7])
        assert expectation_from_probabilities(probs, []) == pytest.approx(1.0)


class TestResultTypesAppended:
    def test_calibration_result_fields(self):
        r = CalibrationResult(
            target_qubits=[0, 1],
            per_qubit_confusion={0: [[1.0, 0.0], [0.0, 1.0]], 1: [[0.9, 0.1], [0.1, 0.9]]},
        )
        assert r.target_qubits == [0, 1]
        assert r.per_qubit_confusion[1][0][0] == 0.9

    def test_qaoa_result_defaults(self):
        r = QAOAResult(best_cost=-2.0, best_params=[0.1, 0.2], cost_history=[-1.0, -2.0])
        assert r.params_history is None
        assert r.best_cost == -2.0

    def test_qbm_result_defaults(self):
        r = QBMResult(best_loss=0.5, best_params=[0.0], loss_history=[0.9, 0.5])
        assert r.generated_samples is None
        assert r.test_loss_history is None
