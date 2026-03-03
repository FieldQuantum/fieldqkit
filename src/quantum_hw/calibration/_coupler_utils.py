"""Shared helpers for coupler selection and identifiers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..api.backend import Backend


def coupler_key(q1: int, q2: int) -> str:
	"""Build normalized coupler key as `min-max`."""
	return f"{min(q1, q2)}-{max(q1, q2)}"


def resolve_positive_fidelity_couplers(
	couplers: Optional[Sequence[Tuple[int, int]]],
	backend: Backend,
) -> List[Tuple[int, int]]:
	"""Resolve couplers from input or backend metadata filtered by fidelity > 0."""
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
