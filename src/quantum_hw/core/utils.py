"""Shared helpers for probability vectors and samples."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def get_probabilities(result: Dict[str, int], num_qubits: int) -> np.ndarray:
	"""Normalize counts into probabilities via sample expansion.

	Args:
		result (*Dict[str, int]*): Counts dictionary mapping bitstrings to occurrence counts.
		num_qubits (*int*): Number of qubits.

	Returns:
		Probability vector of length ``2**num_qubits``.
	"""
	samples = get_samples(result, num_qubits)
	return get_probabilities_from_samples(samples, num_qubits)


def get_samples(result: Dict[str, int], num_qubits: int) -> np.ndarray:
	"""Expand counts into a sample array aligned with get_probabilities bit order.

	Args:
		result (*Dict[str, int]*): Counts dictionary mapping bitstrings to their occurrence count.
		num_qubits (*int*): Number of qubits.

	Returns:
		2-D array of shape ``(total_shots, num_qubits)`` with 0/1 entries.
	"""
	samples = []
	for key, count in result.items():
		bits = [int(b) for b in key]
		# Expand to per-shot rows for downstream estimators.
		samples.extend([bits] * count)
	return np.asarray(samples, dtype=int).reshape(-1, num_qubits)


def get_probabilities_from_samples(samples: np.ndarray, num_qubits: int) -> np.ndarray:
	"""Compute global basis probabilities from sample rows.

	Args:
		samples (*np.ndarray*): 2-D array of shape ``(nshots, num_qubits)`` with 0/1 entries.
		num_qubits (*int*): Number of qubits.

	Returns:
		Probability vector of length ``2**num_qubits``.

	Raises:
		ValueError: samples must be a 2D array with shape (nshots, num_qubits)
	"""
	if samples.size == 0:
		return np.zeros(2**num_qubits, dtype=float)
	if samples.ndim != 2 or samples.shape[1] != num_qubits:
		raise ValueError("samples must be a 2D array with shape (nshots, num_qubits)")
	weights = 2 ** np.arange(num_qubits - 1, -1, -1)
	indices = (samples * weights).sum(axis=1)
	counts = np.bincount(indices, minlength=2**num_qubits).astype(float)
	total = counts.sum()
	if total == 0:
		return counts
	return counts / total


def marginal_samples(samples: np.ndarray, support: Sequence[int]) -> np.ndarray:
	"""Extract marginal samples on a subset of qubits.

	Args:
		samples (*np.ndarray*): 2-D array of shape ``(nshots, num_qubits)`` with 0/1 entries.
		support (*Sequence[int]*): Qubit column indices to extract.

	Returns:
		2-D NumPy array of shape ``(nshots, len(support))``.
	"""
	if not support:
		return np.zeros((samples.shape[0], 0), dtype=int)
	return samples[:, support]


def get_local_probabilities_from_samples(samples: np.ndarray, support: Sequence[int]) -> np.ndarray:
	"""Compute local probabilities on a subset of qubits from samples.

	Args:
		samples (*np.ndarray*): 2-D array of shape ``(nshots, num_qubits)`` with 0/1 entries.
		support (*Sequence[int]*): Qubit column indices to marginalise over.

	Returns:
		1-D probability vector of length ``2**len(support)``.
	"""
	support = list(support)
	if not support:
		return np.array([1.0])
	local_samples = marginal_samples(samples, support)
	return get_probabilities_from_samples(local_samples, len(support))


def expectation_from_probabilities(probabilities: np.ndarray, support: Sequence[int]) -> float:
	"""Compute Z-basis expectation value from probabilities.

	Args:
		probabilities (*np.ndarray*): 1-D probability vector of length ``2**num_qubits``.
		support (*Sequence[int]*): Qubit dimension indices over which to compute Z-parity expectation.

	Returns:
		Z-parity expectation value in ``[-1, 1]``.
	"""
	if not support:
		return 1.0
	num = len(support)
	probs = probabilities.reshape([2] * num)
	parity = np.zeros([2] * num, dtype=int)
	for i in range(num):
		shape = [1] * num
		shape[i] = 2
		parity += np.arange(2).reshape(shape)
	sign = 1.0 - 2.0 * (parity % 2)
	return float((probs * sign).sum())