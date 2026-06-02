"""Tests for the calibration subsystem: readout, native two-qubit RB and
process tomography, plus the shared cache / coupler utilities.

The three managers take injected callbacks (submit/wait/result/compact/simulate),
so the ``chip_name="simulator"`` paths can be driven end-to-end by the real
statevector sampler. For ideal (noiseless) circuits the sampled outcomes are
deterministic (exactly 0/1), so these tests are not flaky.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fieldqkit import QuantumHardwareClient
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.sim.statevector import simulate_counts as statevector_simulate_counts
from fieldqkit.sim.statevector import simulate_statevector
from fieldqkit.calibration import (
    ReadoutCalibrationManager,
    NativeTwoQubitRBManager,
    NativeTwoQubitTomographyManager,
)
from fieldqkit.calibration.readout import build_confusion_matrix
from fieldqkit.calibration import _cache, _coupler_utils


# ─────────────────────────────────────────────────────────────
#  Test doubles
# ─────────────────────────────────────────────────────────────

# Reuse the production circuit-compaction (sparse physical qubits -> dense 0..n-1
# range) so the injected callbacks match what QuantumHardwareClient wires up.
_CLIENT = QuantumHardwareClient()
_compact_for_sim = _CLIENT._compact_for_sim


def _sim_counts(qc, shots):
    """Injected simulate_counts callback backed by the real statevector sampler."""
    return statevector_simulate_counts(qc, shots, seed=7)


def _unused(*args, **kwargs):  # submit/wait/result must never be called on the simulator path
    raise AssertionError("hardware callback should not be invoked in simulator mode")


class _FakeBackend:
    """Minimal backend stub exposing only the attributes the managers read."""

    def __init__(self, *, basis="cz", qubits_with_attributes=None,
                 couplers_with_attributes=None, chip_info=None):
        self.two_qubit_gate_basis = basis
        if qubits_with_attributes is not None:
            self.qubits_with_attributes = qubits_with_attributes
        if couplers_with_attributes is not None:
            self.couplers_with_attributes = couplers_with_attributes
        if chip_info is not None:
            self.chip_info = chip_info


def _make_readout_manager(tmp_path, simulate_counts=_sim_counts):
    return ReadoutCalibrationManager(
        cache_dir=tmp_path,
        submit_circuit_async=_unused,
        wait_task=_unused,
        get_task_result=_unused,
        compact_for_sim=_compact_for_sim,
        simulate_counts=simulate_counts,
    )


def _make_rb_manager(tmp_path):
    return NativeTwoQubitRBManager(
        cache_dir=tmp_path,
        submit_circuit_async=_unused,
        wait_task=_unused,
        get_task_result=_unused,
        compact_for_sim=_compact_for_sim,
        simulate_counts=_sim_counts,
    )


def _make_tomo_manager(tmp_path, simulate_counts=_sim_counts):
    return NativeTwoQubitTomographyManager(
        cache_dir=tmp_path,
        submit_circuit_async=_unused,
        wait_task=_unused,
        get_task_result=_unused,
        compact_for_sim=_compact_for_sim,
        simulate_counts=simulate_counts,
    )


# ─────────────────────────────────────────────────────────────
#  _cache helpers
# ─────────────────────────────────────────────────────────────

class TestCacheHelpers:
    def test_cache_file_naming(self, tmp_path):
        assert _cache.cache_file(tmp_path, stem="readout", chip_name="abc").name == "readout_abc.json"
        # None chip_name falls back to "unknown".
        assert _cache.cache_file(tmp_path, stem="rb", chip_name=None).name == "rb_unknown.json"

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "c.json"
        ts = {"0": "2026-01-01T00:00:00+00:00"}
        payload = {"0": [[1.0, 0.0], [0.0, 1.0]]}
        _cache.save_timestamped_payload(path, payload_key="per_qubit", timestamps=ts, payload=payload)
        loaded_ts, loaded_payload = _cache.load_timestamped_payload(path, payload_key="per_qubit")
        assert loaded_ts == ts
        assert loaded_payload == payload

    def test_load_missing_file(self, tmp_path):
        ts, payload = _cache.load_timestamped_payload(tmp_path / "nope.json", payload_key="per_qubit")
        assert ts == {} and payload == {}

    def test_load_malformed_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all", encoding="utf-8")
        ts, payload = _cache.load_timestamped_payload(path, payload_key="per_qubit")
        assert ts == {} and payload == {}

    def test_load_non_dict_json(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        ts, payload = _cache.load_timestamped_payload(path, payload_key="per_qubit")
        assert ts == {} and payload == {}

    def test_cache_is_fresh(self):
        now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        fresh = (now - timedelta(hours=1)).isoformat()
        stale = (now - timedelta(hours=13)).isoformat()
        assert _cache.cache_is_fresh(fresh, now=now) is True
        assert _cache.cache_is_fresh(stale, now=now) is False
        assert _cache.cache_is_fresh(None, now=now) is False
        assert _cache.cache_is_fresh("not-a-timestamp", now=now) is False
        # Boundary: exactly at the TTL edge is still considered fresh (<=).
        edge = (now - timedelta(hours=12)).isoformat()
        assert _cache.cache_is_fresh(edge, now=now) is True
        # Custom TTL.
        assert _cache.cache_is_fresh(stale, now=now, ttl_hours=24) is True


# ─────────────────────────────────────────────────────────────
#  _coupler_utils helpers
# ─────────────────────────────────────────────────────────────

class TestCouplerUtils:
    def test_coupler_key_normalized(self):
        assert _coupler_utils.coupler_key(5, 2) == "2-5"
        assert _coupler_utils.coupler_key(2, 5) == "2-5"
        assert _coupler_utils.coupler_key(3, 3) == "3-3"

    def test_resolve_explicit_passthrough(self):
        backend = _FakeBackend()  # no couplers attribute needed
        out = _coupler_utils.resolve_positive_fidelity_couplers([(0, 1), (1, 2)], backend)
        assert out == [(0, 1), (1, 2)]

    def test_resolve_filters_positive_fidelity(self):
        backend = _FakeBackend(couplers_with_attributes=[
            (0, 1, {"fidelity": 0.98}),
            (1, 2, {"fidelity": 0.0}),     # filtered out
            (2, 3, {"fidelity": 0.95}),
            (3, 4, {}),                    # missing fidelity -> filtered out
        ])
        out = _coupler_utils.resolve_positive_fidelity_couplers(None, backend)
        assert out == [(0, 1), (2, 3)]

    def test_resolve_no_positive_couplers_raises(self):
        backend = _FakeBackend(couplers_with_attributes=[(0, 1, {"fidelity": 0.0})])
        with pytest.raises(RuntimeError, match="no available couplers"):
            _coupler_utils.resolve_positive_fidelity_couplers(None, backend)


# ─────────────────────────────────────────────────────────────
#  build_confusion_matrix
# ─────────────────────────────────────────────────────────────

class TestBuildConfusionMatrix:
    def test_perfect_readout_is_identity(self):
        # Row i = probabilities measured when preparing basis state i.
        res_list = [{"0": 1000}, {"1": 1000}]
        mat = build_confusion_matrix(res_list, num_qubits=1)
        assert np.allclose(mat, np.eye(2))

    def test_noisy_readout_rows_are_distributions(self):
        res_list = [{"0": 900, "1": 100}, {"0": 50, "1": 950}]
        mat = build_confusion_matrix(res_list, num_qubits=1)
        assert mat.shape == (2, 2)
        assert np.allclose(mat.sum(axis=1), 1.0)
        assert mat[0, 0] == pytest.approx(0.9)
        assert mat[1, 1] == pytest.approx(0.95)


# ─────────────────────────────────────────────────────────────
#  ReadoutCalibrationManager
# ─────────────────────────────────────────────────────────────

class TestReadoutCalibrationManager:
    def test_requires_backend(self, tmp_path):
        mgr = _make_readout_manager(tmp_path)
        with pytest.raises(RuntimeError, match="backend is not set"):
            mgr.calibrate_readout([0], chip_name="simulator", backend=None)

    def test_requires_chip_name(self, tmp_path):
        mgr = _make_readout_manager(tmp_path)
        with pytest.raises(RuntimeError, match="chip_name is not set"):
            mgr.calibrate_readout([0], chip_name=None, backend=_FakeBackend())

    def test_simulator_perfect_confusion_matrix(self, tmp_path):
        mgr = _make_readout_manager(tmp_path)
        result = mgr.calibrate_readout([0, 1], shots=512, chip_name="simulator", backend=_FakeBackend())
        assert sorted(result.target_qubits) == [0, 1]
        for q in (0, 1):
            mat = np.asarray(result.per_qubit_confusion[q])
            assert np.allclose(mat, np.eye(2)), f"qubit {q} confusion matrix not identity"

    def test_cache_avoids_recomputation(self, tmp_path):
        calls = {"n": 0}

        def counting_sim(qc, shots):
            calls["n"] += 1
            return _sim_counts(qc, shots)

        mgr = _make_readout_manager(tmp_path, simulate_counts=counting_sim)
        mgr.calibrate_readout([0], shots=256, chip_name="simulator", backend=_FakeBackend())
        first = calls["n"]
        assert first > 0
        # A cache file should now exist.
        assert (tmp_path / "readout_simulator.json").exists()
        # Second call for the same qubit must hit the cache (no further simulation).
        mgr2 = _make_readout_manager(tmp_path, simulate_counts=counting_sim)
        mgr2.calibrate_readout([0], shots=256, chip_name="simulator", backend=_FakeBackend())
        assert calls["n"] == first

    def test_resolve_target_qubits_from_backend(self, tmp_path):
        mgr = _make_readout_manager(tmp_path)
        backend = _FakeBackend(
            qubits_with_attributes=[(0, {}), (1, {}), (2, {})],
            chip_info={"qubits_info": {
                "Q0": {"fidelity": 0.9},
                "Q1": {"fidelity": 0.0},   # excluded (fidelity 0)
                "Q2": {"fidelity": 0.95},
            }},
        )
        result = mgr.calibrate_readout(None, shots=128, chip_name="simulator", backend=backend)
        assert result.target_qubits == [0, 2]


# ─────────────────────────────────────────────────────────────
#  NativeTwoQubitRBManager
# ─────────────────────────────────────────────────────────────

class TestNativeTwoQubitRB:
    def test_fit_decay_recovers_known_curve(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        dim = 4
        b = 1.0 / dim
        p_true, a_true = 0.95, 0.7
        lengths = [1, 2, 4, 8, 16, 32]
        survival = [a_true * p_true ** x + b for x in lengths]
        fit = mgr._fit_decay(lengths, survival)
        assert fit["p"] == pytest.approx(p_true, abs=1e-6)
        f_avg = ((dim - 1) * p_true + 1) / dim
        assert fit["fidelity"] == pytest.approx(f_avg, abs=1e-6)
        assert fit["epc"] == pytest.approx(1.0 - f_avg, abs=1e-6)

    def test_fit_decay_underdetermined(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        # Survival sitting at the asymptote 1/dim leaves <2 usable points.
        fit = mgr._fit_decay([1, 2, 4], [0.25, 0.25, 0.25])
        assert fit["p"] is None and fit["fidelity"] is None
        assert fit["B"] == pytest.approx(0.25)

    def test_random_sequence_is_identity(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        rng = np.random.default_rng(0)
        qc, total_length = mgr._build_random_sequence([0, 1], length=5, basis_gate="cz", rng=rng)
        # forward + inverse must compose to identity -> stays in |00>.
        state = simulate_statevector(qc)
        assert float(state[0].abs().item()) == pytest.approx(1.0, abs=1e-9)
        # total_length scaling: cz -> 2 per layer.
        assert total_length == 2 * 5

    def test_random_sequence_unsupported_basis(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="unsupported two-qubit"):
            mgr._build_random_sequence([0, 1], length=2, basis_gate="bogus", rng=rng)

    def test_apply_two_qubit_gate_unsupported(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="unsupported two-qubit gate"):
            mgr._apply_two_qubit_gate(qc, "swap", 0, 1)

    def test_requires_backend_and_chip(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        with pytest.raises(RuntimeError, match="backend is not set"):
            mgr.calibrate_native_two_qubit_rb([(0, 1)], chip_name="simulator", backend=None)
        with pytest.raises(RuntimeError, match="chip_name is not set"):
            mgr.calibrate_native_two_qubit_rb([(0, 1)], chip_name=None, backend=_FakeBackend())

    def test_simulator_noiseless_fidelity_is_one(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        results = mgr.calibrate_native_two_qubit_rb(
            [(0, 1)],
            lengths=[1, 2, 4],
            num_sequences=3,
            shots=512,
            chip_name="simulator",
            backend=_FakeBackend(basis="cz"),
            readout_mitigation=False,
            seed=0,
        )
        assert "0-1" in results
        fit = results["0-1"]["fit"]
        # Noiseless identity sequences -> survival 1.0 at every length -> fidelity ~ 1.
        assert all(v == pytest.approx(1.0, abs=1e-9) for v in results["0-1"]["survival_avg"].values())
        assert fit["fidelity"] == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────
#  NativeTwoQubitTomographyManager  (pure-math building blocks)
# ─────────────────────────────────────────────────────────────

class TestTomographyMath:
    def test_ptm_from_cz_is_orthogonal(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        ptm = mgr._ptm_from_unitary("cz")
        assert ptm.shape == (16, 16)
        assert np.allclose(ptm.imag if np.iscomplexobj(ptm) else 0.0, 0.0)
        # Unitary channel -> orthogonal PTM.
        assert np.allclose(ptm @ ptm.T, np.eye(16), atol=1e-9)
        # Trace- and identity-preserving: first row/column is e0.
        e0 = np.zeros(16); e0[0] = 1.0
        assert np.allclose(ptm[0, :], e0, atol=1e-9)
        assert np.allclose(ptm[:, 0], e0, atol=1e-9)

    def test_ideal_error_channel_is_identity(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        ideal = mgr._ptm_from_unitary("cz")
        # error PTM of a perfect gate against its own ideal == identity.
        ptm_error = ideal @ np.linalg.pinv(ideal)
        assert np.allclose(ptm_error, np.eye(16), atol=1e-9)
        choi = mgr._ptm_to_choi(ptm_error)
        # Choi matrix of any physical map is Hermitian.
        assert np.allclose(choi, choi.conj().T, atol=1e-9)

    def test_ptm_from_unknown_name_raises(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        with pytest.raises(ValueError, match="unknown gate name"):
            mgr._ptm_from_unitary("nope")

    def test_choi_payload_roundtrip(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        choi = np.arange(16 * 16).reshape(16, 16).astype(complex) + 1j * np.eye(16)
        payload = mgr._encode_choi_payload(choi)
        restored = mgr._decode_choi_payload(payload)["choi_error"]
        assert np.allclose(restored, choi)

    def test_expectations_from_probs(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        # |00> -> Z0=+1, Z1=+1, Z0Z1=+1
        assert mgr._expectations_from_probs(np.array([1.0, 0, 0, 0])) == pytest.approx((1.0, 1.0, 1.0))
        # |11> -> Z0=-1, Z1=-1, Z0Z1=+1
        assert mgr._expectations_from_probs(np.array([0, 0, 0, 1.0])) == pytest.approx((-1.0, -1.0, 1.0))
        # Maximally mixed -> all zero.
        assert mgr._expectations_from_probs(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx((0.0, 0.0, 0.0))

    def test_expectations_from_probs_wrong_length(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        with pytest.raises(ValueError, match="length 4"):
            mgr._expectations_from_probs(np.array([0.5, 0.5]))

    def test_state_density_unknown_label(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        with pytest.raises(ValueError, match="unsupported state label"):
            mgr._state_density("Q")

    def test_apply_two_qubit_gate_unsupported(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="unsupported two-qubit gate"):
            mgr._apply_two_qubit_gate(qc, "swap", 0, 1)

    def test_requires_backend_and_chip(self, tmp_path):
        mgr = _make_tomo_manager(tmp_path)
        with pytest.raises(RuntimeError, match="backend is not set"):
            mgr.calibrate_native_two_qubit_tomography([(0, 1)], chip_name="simulator", backend=None)
        with pytest.raises(RuntimeError, match="chip_name is not set"):
            mgr.calibrate_native_two_qubit_tomography([(0, 1)], chip_name=None, backend=_FakeBackend())


class TestTomographyEndToEnd:
    def test_simulator_recovers_near_identity_error(self, tmp_path):
        """End-to-end simulator tomography of an ideal CZ: the recovered error
        channel should be close to the identity channel, and the result must be
        cached (16x16 Choi)."""
        mgr = _make_tomo_manager(tmp_path)
        results = mgr.calibrate_native_two_qubit_tomography(
            [(0, 1)],
            shots=4096,
            chip_name="simulator",
            backend=_FakeBackend(basis="cz"),
            readout_mitigation=False,
        )
        choi = results["0-1"]["choi_error"]
        assert choi.shape == (16, 16)
        # Choi of the (near-)identity error channel is dominated by its trace;
        # sampling noise keeps it close to the ideal identity-channel Choi.
        ideal_choi = mgr._ptm_to_choi(np.eye(16))
        assert np.allclose(choi, ideal_choi, atol=0.15)
        assert (tmp_path / "tomo_two_qubit_simulator.json").exists()


# ─────────────────────────────────────────────────────────────
#  build_confusion_matrix — larger scale & invariants
# ─────────────────────────────────────────────────────────────

class TestConfusionMatrixLargeScale:
    @pytest.mark.parametrize("n", [1, 2, 3, 6, 8])
    def test_perfect_readout_identity_for_n_qubits(self, n):
        """Preparing each basis state perfectly yields the identity confusion
        matrix for up to 8 qubits (256x256)."""
        dim = 2 ** n
        res_list = [{format(i, f"0{n}b"): 1000} for i in range(dim)]
        mat = build_confusion_matrix(res_list, num_qubits=n)
        assert mat.shape == (dim, dim)
        assert np.allclose(mat, np.eye(dim))

    @pytest.mark.parametrize("n", [2, 3, 6])
    def test_rows_are_probability_distributions(self, n):
        """Every row of an arbitrary noisy confusion matrix is a valid
        probability distribution (non-negative, sums to 1)."""
        rng = np.random.default_rng(11)
        dim = 2 ** n
        res_list = []
        for _ in range(dim):
            counts = rng.integers(1, 50, size=dim)
            res_list.append({format(j, f"0{n}b"): int(counts[j]) for j in range(dim)})
        mat = build_confusion_matrix(res_list, num_qubits=n)
        assert mat.shape == (dim, dim)
        assert (mat >= 0).all()
        assert np.allclose(mat.sum(axis=1), 1.0)

    def test_noisy_two_qubit_rows_sum_to_one(self):
        res_list = [
            {"00": 800, "01": 100, "10": 50, "11": 50},
            {"01": 900, "00": 100},
            {"10": 950, "00": 50},
            {"11": 990, "10": 10},
        ]
        mat = build_confusion_matrix(res_list, num_qubits=2)
        assert np.allclose(mat.sum(axis=1), 1.0)
        # Diagonal entries are the "prepared == measured" probabilities.
        assert np.allclose(np.diag(mat), [0.8, 0.9, 0.95, 0.99])


# ─────────────────────────────────────────────────────────────
#  Readout mitigation invariants (core.readout building blocks)
# ─────────────────────────────────────────────────────────────

class TestReadoutMitigationInvariants:
    def test_mitigation_of_perfect_calibration_is_identity(self):
        from fieldqkit.core.readout import mitigate_readout
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        out = mitigate_readout(probs, np.eye(4))
        assert np.allclose(out, probs)

    def test_mitigation_recovers_true_distribution(self):
        """With the exact confusion matrix, mitigation inverts the readout
        noise back to the true distribution."""
        from fieldqkit.core.readout import mitigate_readout
        # Row i = measured-given-prepared, so measured = C^T @ p_true.
        cm = np.array([[0.9, 0.1], [0.05, 0.95]])
        true = np.array([0.7, 0.3])
        measured = cm.T @ true
        recovered = mitigate_readout(measured, cm.T)
        assert np.allclose(recovered, true, atol=1e-9)

    def test_mitigation_output_is_a_distribution(self):
        from fieldqkit.core.readout import mitigate_readout
        cm = np.array([[0.85, 0.15], [0.2, 0.8]])
        out = mitigate_readout(np.array([0.6, 0.4]), cm)
        assert (out >= 0).all()
        assert out.sum() == pytest.approx(1.0)

    def test_mitigation_rejects_non_square(self):
        from fieldqkit.core.readout import mitigate_readout
        with pytest.raises(ValueError, match="must be square"):
            mitigate_readout(np.array([0.5, 0.5]), np.ones((2, 3)))

    @pytest.mark.parametrize("n", [1, 4, 8])
    def test_local_confusion_kron_identity(self, n):
        """Tensoring n identity per-qubit matrices yields the 2^n identity."""
        from fieldqkit.core.readout import build_local_confusion_matrix
        per = {q: np.eye(2) for q in range(n)}
        out = build_local_confusion_matrix(per, list(range(n)))
        dim = 2 ** n
        assert out.shape == (dim, dim)
        assert np.allclose(out, np.eye(dim))

    def test_local_confusion_empty_raises(self):
        from fieldqkit.core.readout import build_local_confusion_matrix
        with pytest.raises(ValueError, match="empty"):
            build_local_confusion_matrix({}, [])


# ─────────────────────────────────────────────────────────────
#  ReadoutCalibrationManager — boundary & larger scale
# ─────────────────────────────────────────────────────────────

class TestReadoutCalibrationBoundary:
    def test_single_qubit_readout_is_identity(self, tmp_path):
        mgr = _make_readout_manager(tmp_path)
        result = mgr.calibrate_readout([3], shots=256, chip_name="simulator", backend=_FakeBackend())
        assert result.target_qubits == [3]
        mat = np.asarray(result.per_qubit_confusion[3])
        assert mat.shape == (2, 2)
        assert np.allclose(mat, np.eye(2))
        # Each per-qubit confusion matrix row is a valid distribution.
        assert np.allclose(mat.sum(axis=1), 1.0)

    def test_eight_qubit_readout_all_identity(self, tmp_path):
        """Larger-scale: calibrate 8 qubits at once; each is an identity 2x2."""
        mgr = _make_readout_manager(tmp_path)
        qubits = list(range(8))
        result = mgr.calibrate_readout(qubits, shots=128, chip_name="simulator", backend=_FakeBackend())
        assert sorted(result.target_qubits) == qubits
        for q in qubits:
            mat = np.asarray(result.per_qubit_confusion[q])
            assert np.allclose(mat, np.eye(2)), f"qubit {q} not identity"
            assert np.allclose(mat.sum(axis=1), 1.0)

    def test_empty_target_qubits_returns_empty(self, tmp_path):
        """An explicit empty qubit list calibrates nothing (no error)."""
        mgr = _make_readout_manager(tmp_path)
        result = mgr.calibrate_readout([], shots=128, chip_name="simulator", backend=_FakeBackend())
        assert list(result.target_qubits) == []
        assert result.per_qubit_confusion == {}

    def test_resolve_target_qubits_missing_attribute_raises(self, tmp_path):
        """No explicit qubits and no backend metadata -> RuntimeError."""
        mgr = _make_readout_manager(tmp_path)
        with pytest.raises(RuntimeError, match="qubits_with_attributes is missing"):
            mgr.calibrate_readout(None, shots=64, chip_name="simulator", backend=_FakeBackend())


# ─────────────────────────────────────────────────────────────
#  NativeTwoQubitRBManager — boundary & larger scale
# ─────────────────────────────────────────────────────────────

class TestNativeTwoQubitRBExtra:
    def test_fit_decay_empty_is_underdetermined(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        fit = mgr._fit_decay([], [])
        assert fit["p"] is None and fit["fidelity"] is None and fit["epc"] is None
        assert fit["B"] == pytest.approx(0.25)

    def test_fit_decay_single_point_is_underdetermined(self, tmp_path):
        mgr = _make_rb_manager(tmp_path)
        fit = mgr._fit_decay([1], [0.9])
        assert fit["p"] is None and fit["fidelity"] is None

    def test_fit_decay_parameters_in_sane_ranges(self, tmp_path):
        """A clean decaying curve yields p in (0,1], fidelity in [0,1],
        epc == 1 - fidelity."""
        mgr = _make_rb_manager(tmp_path)
        dim = 4
        b = 1.0 / dim
        p_true, a_true = 0.9, 0.8
        lengths = [1, 2, 4, 8, 16, 32, 64]
        survival = [a_true * p_true ** x + b for x in lengths]
        fit = mgr._fit_decay(lengths, survival)
        assert 0.0 < fit["p"] <= 1.0
        assert 0.0 <= fit["fidelity"] <= 1.0
        assert fit["epc"] == pytest.approx(1.0 - fit["fidelity"], abs=1e-12)
        assert fit["A"] == pytest.approx(a_true, abs=1e-6)

    @pytest.mark.parametrize("basis,scale", [("cz", 2), ("cx", 2), ("iswap", 4), ("ecr", 2)])
    def test_long_sequence_stays_identity_per_basis(self, tmp_path, basis, scale):
        """A length-10 forward+inverse sequence composes to identity for every
        supported native basis, and total_length scales correctly."""
        mgr = _make_rb_manager(tmp_path)
        rng = np.random.default_rng(3)
        qc, total_length = mgr._build_random_sequence([0, 1], length=10, basis_gate=basis, rng=rng)
        assert total_length == scale * 10
        state = simulate_statevector(qc)
        assert float(state[0].abs().item()) == pytest.approx(1.0, abs=1e-9)

    def test_simulator_rb_longer_lengths_fidelity_one(self, tmp_path):
        """Larger-scale: longer length list still yields perfect survival and
        unit fidelity on the noiseless simulator."""
        mgr = _make_rb_manager(tmp_path)
        results = mgr.calibrate_native_two_qubit_rb(
            [(0, 1)],
            lengths=[1, 2, 4, 8, 16],
            num_sequences=2,
            shots=256,
            chip_name="simulator",
            backend=_FakeBackend(basis="cz"),
            readout_mitigation=False,
            seed=0,
        )
        fit = results["0-1"]["fit"]
        assert all(v == pytest.approx(1.0, abs=1e-9) for v in results["0-1"]["survival_avg"].values())
        assert fit["fidelity"] == pytest.approx(1.0, abs=1e-6)
        # total_lengths follow the cz scale (x2) of the requested lengths.
        assert results["0-1"]["total_lengths"] == [2, 4, 8, 16, 32]

    def test_simulator_rb_multiple_couplers_aggregated(self, tmp_path):
        """Larger-scale: aggregate RB across several couplers in one call."""
        mgr = _make_rb_manager(tmp_path)
        couplers = [(0, 1), (2, 3), (4, 5)]
        results = mgr.calibrate_native_two_qubit_rb(
            couplers,
            lengths=[1, 2, 4],
            num_sequences=2,
            shots=256,
            chip_name="simulator",
            backend=_FakeBackend(basis="cz"),
            readout_mitigation=False,
            seed=1,
        )
        assert set(results) == {"0-1", "2-3", "4-5"}
        for key in results:
            assert results[key]["fit"]["fidelity"] == pytest.approx(1.0, abs=1e-6)
        # Every coupler's fidelity is cached for reuse.
        assert (tmp_path / "rb_two_qubit_simulator.json").exists()

    def test_rb_no_couplers_raises(self, tmp_path):
        """No explicit couplers and no positive-fidelity couplers -> RuntimeError."""
        mgr = _make_rb_manager(tmp_path)
        backend = _FakeBackend(
            basis="cz",
            couplers_with_attributes=[(0, 1, {"fidelity": 0.0})],
        )
        with pytest.raises(RuntimeError, match="no available couplers"):
            mgr.calibrate_native_two_qubit_rb(
                None, chip_name="simulator", backend=backend, readout_mitigation=False,
            )


# ─────────────────────────────────────────────────────────────
#  NativeTwoQubitTomographyManager — extra math boundaries
# ─────────────────────────────────────────────────────────────

class TestTomographyMathExtra:
    @pytest.mark.parametrize("label", ["0", "1", "+", "-", "+i", "-i"])
    def test_state_density_is_valid_pure_state(self, tmp_path, label):
        """Each supported single-qubit prep label yields a trace-1, rank-1,
        Hermitian density matrix."""
        mgr = _make_tomo_manager(tmp_path)
        rho = mgr._state_density(label)
        assert rho.shape == (2, 2)
        assert np.allclose(rho, rho.conj().T, atol=1e-12)  # Hermitian
        assert np.trace(rho) == pytest.approx(1.0)         # trace 1
        # Pure state: rho^2 == rho.
        assert np.allclose(rho @ rho, rho, atol=1e-12)

    @pytest.mark.parametrize("gate", ["cz", "cx", "iswap", "ecr"])
    def test_ptm_is_orthogonal_for_each_basis(self, tmp_path, gate):
        mgr = _make_tomo_manager(tmp_path)
        ptm = mgr._ptm_from_unitary(gate)
        assert ptm.shape == (16, 16)
        assert np.allclose(ptm @ ptm.T, np.eye(16), atol=1e-9)

    def test_expectations_from_probs_bell_like(self, tmp_path):
        """Equal weight on |00> and |11> -> Z0=Z1=0 but Z0Z1=+1."""
        mgr = _make_tomo_manager(tmp_path)
        z0, z1, z0z1 = mgr._expectations_from_probs(np.array([0.5, 0.0, 0.0, 0.5]))
        assert z0 == pytest.approx(0.0)
        assert z1 == pytest.approx(0.0)
        assert z0z1 == pytest.approx(1.0)

    def test_choi_of_identity_ptm_is_hermitian_psd(self, tmp_path):
        """The Choi matrix of the identity channel is Hermitian and PSD."""
        mgr = _make_tomo_manager(tmp_path)
        choi = mgr._ptm_to_choi(np.eye(16))
        assert np.allclose(choi, choi.conj().T, atol=1e-9)
        eigvals = np.linalg.eigvalsh(choi)
        assert eigvals.min() > -1e-9
