import numpy as np

import quantum_hw.api.client as client_module
from quantum_hw.api.client import QuantumHardwareClient, READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT


def test_observable_mitigation_uses_marginal_path_for_small_support(monkeypatch) -> None:
    samples = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)
    support = [0, 1]
    per_qubit = {0: np.eye(2), 1: np.eye(2), 2: np.eye(2)}
    target_qubits = [0, 1, 2]

    called = {"marginal": False}

    def fake_get_local_probabilities_from_samples(samples_arg, support_arg):
        called["marginal"] = True
        assert support_arg == support
        return np.array([0.2, 0.3, 0.1, 0.4], dtype=float)

    def fake_expectation_from_samples_unbiased(local_samples, local_cm_list):
        raise AssertionError("unbiased estimator should not be used for small support")

    monkeypatch.setattr(client_module, "get_local_probabilities_from_samples", fake_get_local_probabilities_from_samples)
    monkeypatch.setattr(client_module, "mitigate_readout", lambda probs, cm: probs)
    monkeypatch.setattr(client_module, "expectation_from_probabilities", lambda probs, support_arg: 0.123)
    monkeypatch.setattr(client_module, "expectation_from_samples_unbiased", fake_expectation_from_samples_unbiased)

    val = QuantumHardwareClient._mitigate_observable_from_samples(
        samples,
        support,
        per_qubit,
        target_qubits,
        marginal_max_support=READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT,
    )

    assert called["marginal"]
    assert np.isclose(val, 0.123)


def test_observable_mitigation_uses_unbiased_path_for_large_support(monkeypatch) -> None:
    k = READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT + 1
    samples = np.zeros((4, k), dtype=int)
    support = list(range(k))
    per_qubit = {i: np.eye(2) for i in range(k)}
    target_qubits = list(range(k))

    def fake_get_local_probabilities_from_samples(samples_arg, support_arg):
        raise AssertionError("marginal probability path should not be used for large support")

    monkeypatch.setattr(client_module, "get_local_probabilities_from_samples", fake_get_local_probabilities_from_samples)
    monkeypatch.setattr(client_module, "expectation_from_samples_unbiased", lambda local_samples, local_cm_list: 0.456)

    val = QuantumHardwareClient._mitigate_observable_from_samples(
        samples,
        support,
        per_qubit,
        target_qubits,
        marginal_max_support=READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT,
    )

    assert np.isclose(val, 0.456)
