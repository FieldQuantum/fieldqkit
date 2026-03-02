"""Shared helpers for probability vectors and samples."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np


QASM_QUBIT_PATTERN = re.compile(r"q\[(\d+)\]")


def counts_dict_to_vector(result: Dict[str, int], num_qubits: int, reverse_bits: bool = False) -> np.ndarray:
	"""Convert a counts dict to a dense vector ordered by basis index."""
	bases = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]
	counts = np.zeros(len(bases), dtype=int)
	for i, base in enumerate(bases):
		key = base[::-1] if reverse_bits else base
		counts[i] = int(result.get(key, 0))
	return counts


def get_probabilities(result: Dict[str, int], num_qubits: int) -> np.ndarray:
	"""Normalize counts into a probability vector (hardware bit order handled)."""
	# Hardware results typically report bitstrings in little-endian order.
	# We reverse bits to keep a consistent logical ordering across the stack.
	counts = counts_dict_to_vector(result, num_qubits, reverse_bits=True)
	total = counts.sum()
	if total == 0:
		return np.zeros_like(counts, dtype=float)
	return counts.astype(float) / float(total)


def get_samples(result: Dict[str, int], num_qubits: int) -> np.ndarray:
	"""Expand counts into a sample array aligned with get_probabilities bit order."""
	# Match the same logical ordering as get_probabilities by reversing bitstrings.
	samples = []
	for key, count in result.items():
		bits = [int(b) for b in key[::-1]]
		# Expand to per-shot rows for downstream estimators.
		samples.extend([bits] * count)
	return np.array(samples, dtype=int)


def get_local_probabilities_from_samples(samples: np.ndarray, support: Sequence[int]) -> np.ndarray:
	"""Compute local probabilities on a subset of qubits from samples."""
	support = list(support)
	if not support:
		return np.array([1.0])
	if samples.size == 0:
		return np.zeros(2 ** len(support), dtype=float)
	# Respect support order; last index is least significant.
	local = samples[:, support]
	weights = 2 ** np.arange(len(support) - 1, -1, -1)
	indices = (local * weights).sum(axis=1)
	counts = np.bincount(indices, minlength=2 ** len(support)).astype(float)
	total = counts.sum()
	if total == 0:
		return counts
	return counts / total
