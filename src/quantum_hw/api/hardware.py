"""Hardware selection and ranking helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

from .backend import Backend


def get_available_chip_status(tmgr) -> Dict[str, int]:
	"""Fetch chip queue status from task manager."""
	status = tmgr.status()
	if not isinstance(status, dict):
		raise RuntimeError("tmgr.status() must return a dict of chip -> queue length")
	return {k: v for k, v in status.items() if isinstance(v, int)}


def get_chip_info(chip_name: str) -> Dict[str, Union[int, float]]:
	"""Get chip metadata and optionally cache a topology drawing."""
	try:
		backend = Backend(chip_name)
		info = backend.chip_info
		try:
			cache_dir = Path(__file__).resolve().parent / ".cache"
			cache_dir.mkdir(parents=True, exist_ok=True)
			fig_path = cache_dir / f"{chip_name}_chip"
			backend.draw(
				show_couplers_fidelity=True,
				show_qubits_attributes="fidelity",
				save_svg_fname=str(fig_path),
				show_qubits_index=True,
				edge_fidelity_thres=0.9,
			)
		except Exception:
			pass
		if isinstance(info, dict):
			return info
	except Exception:
		pass
	return {}


def rank_chips(
	tmgr,
	*,
	num_qubits: int,
	prefer_chips: Optional[Sequence[str] | str] = None,
	weights: Optional[Dict[str, float]] = None,
) -> List[str]:
	"""Rank chips by queue length, size, and error rate with weights."""
	status = get_available_chip_status(tmgr)
	if isinstance(prefer_chips, str):
		prefer_chips = [prefer_chips]
	if prefer_chips is not None:
		prefer_set = {c for c in prefer_chips}
		status = {k: v for k, v in status.items() if k in prefer_set}

	ranked: List[Tuple[str, int, int, float]] = []
	for chip_name, queue_len in status.items():
		info = get_chip_info(chip_name)
		global_info = info.get("global_info", {}) if isinstance(info, dict) else {}
		nqubits = int(global_info.get("nqubits_available", info.get("nqubits_available", 0)) or 0)
		if nqubits < num_qubits:
			continue
		error_rate_2q = float(global_info.get("error_rate_2q", info.get("error_rate_2q", float("inf"))))
		ranked.append((chip_name, queue_len, nqubits, error_rate_2q))

	if not ranked:
		return []

	if weights is None:
		weights = {"queue": 0.2, "nqubits": 0.3, "error": 0.5}
	w_queue = float(weights.get("queue", 0.2))
	w_nqubits = float(weights.get("nqubits", 0.3))
	w_error = float(weights.get("error", 0.5))

	queues = [r[1] for r in ranked]
	nqubits_list = [r[2] for r in ranked]
	errors = [r[3] for r in ranked]

	def _normalize(values: List[float]) -> List[float]:
		vmin = min(values)
		vmax = max(values)
		if vmax == vmin:
			return [0.0 for _ in values]
		return [(v - vmin) / (vmax - vmin) for v in values]

	q_norm = _normalize([float(v) for v in queues])
	n_norm = _normalize([float(v) for v in nqubits_list])
	e_norm = _normalize([float(v) for v in errors])

	scored: List[Tuple[str, float]] = []
	for (chip_name, _, _, _), qn, nn, en in zip(ranked, q_norm, n_norm, e_norm):
		score = w_queue * qn + w_nqubits * (1.0 - nn) + w_error * en
		scored.append((chip_name, score))

	scored.sort(key=lambda x: x[1])
	return [name for name, _ in scored]
