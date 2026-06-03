"""Pauli observable parsing, grouping, and measurement helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_BASIS_ROTATION_OPS = {
	"I": (),
	"Z": (),
	"X": ("h",),
	"Y": ("sdg", "h"),
}


def _parse_pauli_string(pauli: str, num_qubits: int | None = None) -> List[Tuple[int, str]]:
	"""Parse Pauli strings in either compact or indexed form.

	Args:
		pauli (*str*): Pauli string in compact (``"ZZIX"``) or indexed (``"Z0 X2"``) form.
		num_qubits (*int | None*): Number of qubits. Defaults to ``None``.

	Returns:
		List of ``(qubit_index, pauli_operator)`` tuples for non-identity terms.

	Raises:
		ValueError: pauli string is empty
	"""
	pauli = pauli.strip()
	if not pauli:
		raise ValueError("pauli string is empty")

	tokens = pauli.split()
	# Compact form: "ZZIX" (no indices, fixed length).
	if len(tokens) == 1 and tokens[0].isalpha():
		if num_qubits is None:
			num_qubits = len(tokens[0])
		if len(tokens[0]) != num_qubits:
			raise ValueError("pauli length mismatch with num_qubits")
		return [(i, p.upper()) for i, p in enumerate(tokens[0]) if p.upper() != "I"]

	# Indexed form: "Z0 X2 Y3" (order doesn't matter).
	parsed = []
	for tok in tokens:
		op = tok[0].upper()
		idx = int(tok[1:])
		if op not in {"I", "X", "Y", "Z"}:
			raise ValueError(f"unsupported Pauli: {op}")
		if op != "I":
			parsed.append((idx, op))
	if num_qubits is not None:
		for idx, _ in parsed:
			if idx < 0 or idx >= num_qubits:
				raise ValueError("pauli index out of range")
	return parsed


def pauli_support(pauli: str, num_qubits: int | None = None) -> List[int]:
	"""Return the sorted support indices of non-identity Pauli terms.

	Args:
		pauli (*str*): Pauli string in compact or indexed form.
		num_qubits (*int | None*): Number of qubits. Defaults to ``None``.

	Returns:
		Sorted list of qubit indices with non-identity Pauli terms.
	"""
	terms = _parse_pauli_string(pauli, num_qubits=num_qubits)
	return sorted({idx for idx, _ in terms})


def shift_pauli_string(pauli: str, offset: int) -> str:
	"""Shift all qubit indices in a Pauli string by an offset.

	Args:
		pauli (*str*): Pauli string in indexed form.
		offset (*int*): Integer offset applied to each qubit index.

	Returns:
		Pauli string in indexed form with shifted qubit indices.
	"""
	terms = _parse_pauli_string(pauli)
	if not terms:
		return ""
	return " ".join(f"{op}{idx + offset}" for idx, op in terms)


def pauli_basis_pattern(pauli: str, num_qubits: int) -> List[str]:
	"""Return basis pattern (I/X/Y/Z) per qubit for a Pauli string.

	Args:
		pauli (*str*): Pauli string in compact or indexed form.
		num_qubits (*int*): Number of qubits. Must be a non-negative integer.

	Returns:
		List of ``'I'``/``'X'``/``'Y'``/``'Z'`` strings, one per qubit.

	Raises:
		TypeError: If ``num_qubits`` is not an ``int``.
		ValueError: If ``num_qubits`` is negative.
	"""
	if not isinstance(num_qubits, int) or isinstance(num_qubits, bool):
		raise TypeError("num_qubits must be an int")
	if num_qubits < 0:
		raise ValueError("num_qubits must be non-negative")
	pattern = ["I"] * num_qubits
	terms = _parse_pauli_string(pauli, num_qubits=num_qubits)
	for idx, op in terms:
		pattern[idx] = op
	return pattern

def apply_measurement_basis_rotations(qc, basis_pattern: Sequence[str], target_qubits: Optional[Sequence[int]] = None) -> None:
	"""Apply only basis rotations for a full I/X/Y/Z pattern.

	Args:
		qc: Quantum circuit.
		basis_pattern (*Sequence[str]*): Per-qubit measurement basis labels, e.g. ``['X', 'Z', 'Y']``.
		target_qubits (*Sequence[int]*): Qubit indices for partial measurement. Defaults to ``None``.

	Raises:
		ValueError: If ``target_qubits`` and ``basis_pattern`` have different lengths,
			or if a basis label is not one of ``I``/``X``/``Y``/``Z``.
	"""
	if target_qubits is None:
		target_qubits = qc.qubits if hasattr(qc, "qubits") else list(range(len(basis_pattern)))
	if len(target_qubits) != len(basis_pattern):
		raise ValueError(
			f"target_qubits length ({len(target_qubits)}) does not match "
			f"basis_pattern length ({len(basis_pattern)})"
		)
	for idx, op in zip(target_qubits, basis_pattern):
		ops = _BASIS_ROTATION_OPS.get(op)
		if ops is None:
			raise ValueError(f"unsupported basis op: {op}")
		for gate_op in ops:
			getattr(qc, gate_op)(idx)


def append_measurement_basis(qc, basis_pattern: Sequence[str], target_qubits: Optional[Sequence[int]] = None) -> None:
	"""Apply basis rotations for a full pattern and append measurements.

	Args:
		qc: Quantum circuit.
		basis_pattern (*Sequence[str]*): Per-qubit measurement basis labels, e.g. ``['X', 'Z', 'Y']``.
		target_qubits (*Sequence[int]*): Qubit indices for partial measurement. Defaults to ``None``.
	"""
	if target_qubits is None:
		target_qubits = qc.qubits if hasattr(qc, "qubits") else list(range(len(basis_pattern)))
	apply_measurement_basis_rotations(qc, basis_pattern, target_qubits=target_qubits)
	qc.barrier()
	# Map measured qubits onto a dense classical register order.
	qc.measure(target_qubits, list(range(len(target_qubits))))


def _compatible_with_basis(pattern: Sequence[str], basis: Sequence[str]) -> bool:
	"""Check whether two basis patterns are compatible for grouping.

	Args:
		pattern (*Sequence[str]*): Per-qubit Pauli pattern to check.
		basis (*Sequence[str]*): Per-qubit measurement basis.

	Returns:
		``True`` if the condition is satisfied.
	"""
	for p, b in zip(pattern, basis):
		if p != "I" and b != "I" and p != b:
			return False
	return True


def _merge_basis(pattern: Sequence[str], basis: Sequence[str]) -> List[str]:
	"""Merge two compatible basis patterns into a single pattern.

	Args:
		pattern (*Sequence[str]*): Per-qubit Pauli pattern to merge.
		basis (*Sequence[str]*): Per-qubit measurement basis.

	Returns:
		Merged basis list with non-identity entries from *pattern* filled in.
	"""
	merged = list(basis)
	for i, p in enumerate(pattern):
		if merged[i] == "I" and p != "I":
			merged[i] = p
	return merged


def group_observables(observables: Sequence[str], num_qubits: int) -> List[Dict[str, object]]:
	"""Group observables that can share a single measurement basis.

	Args:
		observables (*Sequence[str]*): Observable operators to measure.
		num_qubits (*int*): Number of qubits.

	Returns:
		List of group dicts, each with keys ``"basis"`` and ``"observables"``.
	"""
	groups: List[Dict[str, object]] = []
	for obs in observables:
		pattern = pauli_basis_pattern(obs, num_qubits=num_qubits)
		# Greedy grouping: merge into the first compatible basis.
		placed = False
		for group in groups:
			basis = group["basis"]
			if _compatible_with_basis(pattern, basis):
				group["basis"] = _merge_basis(pattern, basis)
				group["observables"].append(obs)
				placed = True
				break
		if not placed:
			groups.append({"basis": pattern, "observables": [obs]})
	return groups


def pauli_expectation(samples: np.ndarray, pauli: str) -> float:
	"""Compute expectation value from measurement samples in a Pauli basis.

	Args:
		samples (*np.ndarray*): Measurement samples.
		pauli (*str*): Pauli string specifying the observable.

	Returns:
		Expectation value in ``[-1, 1]``.
	Raises:
		ValueError: samples must be 2D
	"""
	if samples.ndim != 2:
		raise ValueError("samples must be 2D")
	num_qubits = samples.shape[1]
	terms = _parse_pauli_string(pauli, num_qubits=num_qubits)

	if not terms:
		return 1.0

	# After basis rotations, measurement bits encode eigenvalues: 0 -> +1, 1 -> -1.
	eigenvalues = np.ones(samples.shape[0], dtype=float)
	for idx, op in terms:
		if op in {"Z", "X", "Y"}:
			eigenvalues *= 1.0 - 2.0 * samples[:, idx]
		else:
			raise ValueError(f"unsupported Pauli: {op}")
	return float(eigenvalues.mean())
