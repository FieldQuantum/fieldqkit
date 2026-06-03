"""Readout calibration and mitigation utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from .utils import expectation_from_probabilities, get_local_probabilities_from_samples, marginal_samples



def build_local_confusion_matrix(per_qubit_confusion: Dict[int, np.ndarray], target_qubits: Sequence[int]) -> np.ndarray:
	"""Tensor product local per-qubit confusion matrices.

	Args:
		per_qubit_confusion (*Dict[int, np.ndarray]*): Mapping from qubit index to its 2×2 confusion matrix.
		target_qubits (*Sequence[int]*): Qubit indices whose per-qubit confusion matrices will be tensored together.

	Returns:
		Kronecker product confusion matrix of shape ``(2^k, 2^k)``.

	Raises:
		ValueError: target_qubits is empty
	"""
	if not target_qubits:
		raise ValueError("target_qubits is empty")
	mats = [per_qubit_confusion[q] for q in target_qubits]
	out = mats[0]
	for m in mats[1:]:
		out = np.kron(out, m)
	return out


def mitigate_readout(probabilities: np.ndarray, confusion_matrix: np.ndarray) -> np.ndarray:
	"""Apply readout mitigation using a pseudo-inverse.

	The result is clipped to ``[0, 1]`` and renormalized to sum to 1.

	Args:
		probabilities (*np.ndarray*): Raw probability vector.
		confusion_matrix (*np.ndarray*): Readout confusion matrix (``[measure, prepare]``).

	Returns:
		Mitigated probability vector (clipped and renormalized).

	Raises:
		ValueError: confusion_matrix must be square
	"""
	if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
		raise ValueError("confusion_matrix must be square")
	pinv = np.linalg.pinv(confusion_matrix)
	mitigated = pinv @ probabilities
	mitigated = np.clip(mitigated, 0.0, 1.0)
	s = mitigated.sum()
	if s == 0:
		return mitigated
	return mitigated / s


def expectation_from_samples_unbiased(local_samples: np.ndarray, local_confusion_matrices: Sequence[np.ndarray]) -> float:
	"""Unbiased readout-mitigated parity estimator from local samples.

	This estimator avoids building a full $2^k$ marginal when support size $k$ is large,
	at the cost of higher variance.

	Args:
		local_samples (*np.ndarray*): 2-D array of shape ``(nshots, k)`` with 0/1 outcomes.
		local_confusion_matrices (*Sequence[np.ndarray]*): Sequence of ``k`` 2×2 confusion matrices, one per qubit.

	Returns:
		Unbiased readout-mitigated parity expectation value.

	Raises:
		ValueError: local_samples must be a 2D array with shape (nshots, k)
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


def mitigate_observable_from_samples(
	samples: np.ndarray,
	support: Sequence[int],
	per_qubit: Dict[int, np.ndarray],
	target_qubits_group: Sequence[int],
	marginal_max_support: int = 10,
) -> float:
	"""Compute readout-mitigated observable value from samples with adaptive strategy.

	Args:
		samples (*np.ndarray*): Measurement samples.
		support (*Sequence[int]*): Logical qubit indices (relative to measurement group) of non-identity Pauli terms.
		per_qubit (*Dict[int, np.ndarray]*): Per-qubit 2×2 confusion matrices.
		target_qubits_group (*Sequence[int]*): Physical qubit indices corresponding to the measurement group.
		marginal_max_support (*int*): Maximum support size for exact marginal mitigation; larger supports use the unbiased estimator. Defaults to ``10``.

	Returns:
		Readout-mitigated expectation value for the observable.
	"""
	if not support:
		return 1.0
	support_phys = [target_qubits_group[i] for i in support]
	if len(support) <= marginal_max_support:
		local_cm = build_local_confusion_matrix(per_qubit, support_phys)
		local_probs = get_local_probabilities_from_samples(samples, support)
		local_probs_rem = mitigate_readout(local_probs, local_cm)
		return expectation_from_probabilities(local_probs_rem, support)
	local_samples = marginal_samples(samples, support)
	local_cm_list = [per_qubit[q] for q in support_phys]
	return expectation_from_samples_unbiased(local_samples, local_cm_list)
