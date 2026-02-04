import numpy as np

from quantum_hw.observables import pauli_expectation


def test_pauli_expectation_z():
    """Verify Z-basis expectation for a simple sample set."""
    samples = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
    ])
    val = pauli_expectation(samples, "Z0 Z2")
    assert np.isclose(val, 0.0)
