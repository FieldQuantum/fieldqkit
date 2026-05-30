"""Shared helpers for coupler selection and identifiers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..api.backend import Backend


def coupler_key(q1: int, q2: int) -> str:
	"""Build normalized coupler key as `min-max`.

	Args:
		q1 (*int*): First qubit index in the coupler pair.
		q2 (*int*): Second qubit index in the coupler pair.

	Returns:
		Normalised coupler key string ``"min-max"`` (e.g. ``"2-5"``).
	"""
	return f"{min(q1, q2)}-{max(q1, q2)}"


def resolve_positive_fidelity_couplers(
	couplers: Optional[Sequence[Tuple[int, int]]],
	backend: Backend,
) -> List[Tuple[int, int]]:
	"""Resolve couplers from explicit input or from backend metadata.

	When *couplers* is provided, return them directly without filtering.
	When *couplers* is ``None``, extract couplers from the backend and keep
	only those with fidelity > 0.

	Args:
		couplers (*Optional[Sequence[Tuple[int, int]]]*): List of qubit coupler pairs.
		backend (*Backend*): Hardware backend descriptor.

	Returns:
		List of ``(q1, q2)`` coupler pairs with positive fidelity.

	Raises:
		RuntimeError: no available couplers with fidelity > 0
	"""
	if couplers is not None:
		return [tuple(c) for c in couplers]

	selected: List[Tuple[int, int]] = []
	for q1, q2, attrs in getattr(backend, "couplers_with_attributes", []):
		fidelity = attrs.get("fidelity", 0.0) if isinstance(attrs, dict) else 0.0
		if fidelity and fidelity > 0:
			selected.append((int(q1), int(q2)))
	if not selected:
		raise RuntimeError("no available couplers with fidelity > 0")
	return selected
