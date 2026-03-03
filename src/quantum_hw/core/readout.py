"""Readout calibration and mitigation utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def build_local_confusion_matrix(per_qubit_confusion: Dict[int, np.ndarray], target_qubits: Sequence[int]) -> np.ndarray:
	"""Tensor product local per-qubit confusion matrices."""
	if not target_qubits:
		raise ValueError("target_qubits is empty")
	mats = [per_qubit_confusion[q] for q in target_qubits]
	out = mats[0]
	for m in mats[1:]:
		out = np.kron(out, m)
	return out


def mitigate_readout(probabilities: np.ndarray, confusion_matrix: np.ndarray) -> np.ndarray:
	"""Apply readout mitigation using a pseudo-inverse."""
	if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
		raise ValueError("confusion_matrix must be square")
	pinv = np.linalg.pinv(confusion_matrix)
	mitigated = pinv @ probabilities
	mitigated = np.clip(mitigated, 0.0, 1.0)
	s = mitigated.sum()
	if s == 0:
		return mitigated
	return mitigated / s


def expectation_from_probabilities(probabilities: np.ndarray, support: Sequence[int]) -> float:
	"""Compute Z-basis expectation value from probabilities."""
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


def expectation_from_samples_unbiased(local_samples: np.ndarray, local_confusion_matrices: Sequence[np.ndarray]) -> float:
	"""Unbiased readout-mitigated parity estimator from local samples.

	This estimator avoids building a full $2^k$ marginal when support size $k$ is large,
	at the cost of higher variance.
	"""
	if local_samples.ndim != 2:
		raise ValueError("local_samples must be a 2D array with shape (nshots, k)")
	k = local_samples.shape[1]
	if k == 0:
		return 1.0
	if len(local_confusion_matrices) != k:
		raise ValueError("local_confusion_matrices length must equal local_samples.shape[1]")
	if local_samples.shape[0] == 0:
		return 0.0

	scores = np.ones(local_samples.shape[0], dtype=float)
	norms = np.ones(local_samples.shape[0], dtype=float)
	for i, cm in enumerate(local_confusion_matrices):
		cm_arr = np.asarray(cm, dtype=float)
		if cm_arr.shape != (2, 2):
			raise ValueError("each local confusion matrix must have shape (2, 2)")
		inv = np.linalg.pinv(cm_arr)
		bits = local_samples[:, i]
		if np.any((bits != 0) & (bits != 1)):
			raise ValueError("local_samples must contain only 0/1 outcomes")

		w = inv[0] - inv[1]
		n = inv[0] + inv[1]
		scores *= np.where(bits == 0, w[0], w[1])
		norms *= np.where(bits == 0, n[0], n[1])

	mask = norms != 0
	if not np.any(mask):
		return 0.0

	return float((scores[mask] / norms[mask]).mean())
