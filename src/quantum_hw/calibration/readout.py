"""Readout calibration workflows and caching."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..circuit import QuantumCircuit
from ..core.types import CalibrationResult
from ..core.utils import get_probabilities
from ..api.backend import Backend
from ._cache import cache_file, cache_is_fresh, load_timestamped_payload, save_timestamped_payload


def build_confusion_matrix(res_list: Sequence[Dict[str, int]], num_qubits: int) -> np.ndarray:
	"""Build a confusion matrix from calibration results.

	Args:
		res_list (*Sequence[Dict[str, int]]*): List of measurement count dictionaries, one per prepared basis state.
		num_qubits (*int*): Number of qubits.

	Returns:
		NumPy array of shape ``(2**num_qubits, 2**num_qubits)`` representing the confusion matrix.
	"""
	dim = 2**num_qubits
	mat = np.zeros((dim, dim), dtype=float)
	for i, res in enumerate(res_list):
		probs = get_probabilities(res, num_qubits)
		mat[i, :] = probs
	return mat


class ReadoutCalibrationManager:
	"""Handle readout calibration runs with caching."""

	def __init__(
		self,
		*,
		cache_dir: Path,
		submit_openqasm_async: Callable[[str, str, int, Optional[str]], object],
		wait_task: Callable[[object], str],
		get_task_result: Callable[[object], Dict[str, object]],
		compact_for_sim: Callable[[QuantumCircuit], object],
		simulate_counts: Callable[[QuantumCircuit, int], Dict[str, int]],
	) -> None:
		"""Initialize readout calibration manager with backend submission and caching support.

		Args:
			cache_dir (*Path*): Directory for cache files.
			submit_openqasm_async (*Callable[[str, str, int, Optional[str]], object]*): Callback to submit an OpenQASM circuit asynchronously and return a task handle.
			wait_task (*Callable[[object], str]*): Callback to block until a task completes and return its status.
			get_task_result (*Callable[[object], Dict[str, object]]*): Callback to retrieve measurement results from a completed task.
			compact_for_sim (*Callable[[QuantumCircuit], object]*): Callback to prepare a circuit for local simulation.
			simulate_counts (*Callable[[QuantumCircuit, int], Dict[str, int]]*): Callback to simulate a circuit locally and return bitstring counts.
		"""
		self._cache_dir = cache_dir
		self._cache_dir.mkdir(parents=True, exist_ok=True)
		self._submit_openqasm_async = submit_openqasm_async
		self._wait_task = wait_task
		self._get_task_result = get_task_result
		self._compact_for_sim = compact_for_sim
		self._simulate_counts = simulate_counts

	def calibrate_readout(
		self,
		target_qubits: Optional[Sequence[int]],
		shots: Optional[int] = None,
		*,
		chip_name: Optional[str] = None,
		backend: Optional[Backend] = None,
		qasm_version: str = "2.0",
		print_true: bool = False,
	) -> CalibrationResult:
		"""Calibrate readout error for selected qubits with caching.

		Args:
			target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement.
			shots (*Optional[int]*): Number of measurement shots. Defaults to ``1024`` if not provided.
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.
			backend (*Optional[Backend]*): Hardware backend descriptor. Defaults to ``None``.
			qasm_version (*str*): OpenQASM version (``'2.0'`` or ``'3.0'``). Defaults to ``'2.0'``.
			print_true (*bool*): Whether to print progress information. Defaults to ``False``.

		Returns:
			``CalibrationResult`` containing per-qubit confusion matrices and metadata.

		Raises:
			RuntimeError: backend is not set; use run_auto or provide backend
		"""
		if backend is None:
			raise RuntimeError("backend is not set; use run_auto or provide backend")
		target_qubits = self._resolve_target_qubits(target_qubits, backend)
		if shots is None:
			shots = 1024
		if chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")

		use_simulator = str(chip_name).lower() == "simulator"

		# Cache is per-chip; only stale/missing qubits are recalibrated.
		raw = self._load_readout_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_qubit_raw = raw.get("per_qubit_confusion", {})
		now = datetime.now(timezone.utc)
		cached_confusion: Dict[int, List[List[float]]] = {}
		missing: List[int] = []
		for q in target_qubits:
			# Skip qubits with zero fidelity (unavailable for calibration).
			ts_str = timestamps.get(str(q))
			mat = per_qubit_raw.get(str(q))
			if ts_str is None or mat is None:
				missing.append(q)
				continue
			if not cache_is_fresh(ts_str, now=now):
				missing.append(q)
				continue
			cached_confusion[int(q)] = mat
		# Cache TTL is 12 hours.
		if not missing:
			if print_true:
				print("[readout] using cached readout calibration")
			return CalibrationResult(
				target_qubits=target_qubits,
				per_qubit_confusion=cached_confusion,
			)

		per_qubit_confusion: Dict[int, np.ndarray] = {}
		if print_true:
			print("[readout] run readout calibration on hardware")
		pending: List[Tuple[object, int, str]] = []
		res_map: Dict[int, Dict[str, Dict[str, int]]] = {q: {} for q in missing}
		for q in missing:
			for bits, qc in self._readout_calibration_circuits(q):
				qct = qc
				if use_simulator:
					# Simulator uses the same calibration flow but via statevector sampling.
					qct_sim = self._compact_for_sim(qct)
					if isinstance(qct_sim, tuple):
						qct_sim = qct_sim[0]
					res_map[q][bits] = self._simulate_counts(qct_sim, shots)
				else:
					task_id = self._submit_openqasm_async(
						name=f"readout_cal_q{q}_{bits}",
						qasm=qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3,
						shots=shots,
						chip_name=chip_name,
					)
					pending.append((task_id, q, bits))

		if not use_simulator:
			for task_id, q, bits in pending:
				status = self._wait_task(task_id)
				if status != "Finished":
					raise RuntimeError(f"readout calibration task {task_id} ended with status {status}")
				res = self._get_task_result(task_id)
				counts = res["count"]
				res_map[q][bits] = counts
		for q in missing:
			counts_list = [res_map[q]["0"], res_map[q]["1"]]
			per_qubit_confusion[q] = build_confusion_matrix(counts_list, num_qubits=1)
		result = CalibrationResult(
			target_qubits=target_qubits,
			per_qubit_confusion={
				**cached_confusion,
				**{k: v.tolist() for k, v in per_qubit_confusion.items()},
			},
		)
		self._save_readout_cache(result, chip_name=chip_name)
		return result

	def _resolve_target_qubits(
		self,
		target_qubits: Optional[Sequence[int]],
		backend: Backend,
	) -> List[int]:
		"""Resolve target qubits from backend metadata when not provided.

		Args:
			target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement.
			backend (*Backend*): Hardware backend descriptor.

		Returns:
			List of resolved qubit indices.

		Raises:
			RuntimeError: If target_qubits is not set and backend.qubits_with_attributes is missing or empty.
		"""
		if target_qubits is not None:
			return list(target_qubits)

		qubits: List[int] = []
		qubits_with_attributes = getattr(backend, "qubits_with_attributes", None)
		if isinstance(qubits_with_attributes, list) and qubits_with_attributes:
			qubits = sorted(int(q) for q, _ in qubits_with_attributes if backend.chip_info["qubits_info"][f"Q{q}"]["fidelity"] > 0)
			return qubits

		raise RuntimeError("target_qubits is not set and backend.qubits_with_attributes is missing")

	def _readout_cache_path(self, *, chip_name: Optional[str]) -> Path:
		"""Resolve the on-disk cache path for readout calibration.

		Args:
			chip_name (*Optional[str]*): Name of the target chip.

		Returns:
			``Path`` to the readout calibration cache file.
		"""
		return cache_file(self._cache_dir, stem="readout", chip_name=chip_name)

	def _load_readout_cache_raw(self, *, chip_name: Optional[str]) -> Dict[str, object]:
		"""Load cached readout data from disk (raw dictionary).

		Args:
			chip_name (*Optional[str]*): Name of the target chip.

		Returns:
			Raw cache dictionary with ``"timestamps"`` and ``"per_qubit_confusion"`` keys.
		"""
		path = self._readout_cache_path(chip_name=chip_name)
		timestamps, per_qubit = load_timestamped_payload(path, payload_key="per_qubit_confusion")
		return {"timestamps": timestamps, "per_qubit_confusion": per_qubit}

	# def _load_readout_cache(
	# 	self,
	# 	target_qubits: Sequence[int],
	# 	*,
	# 	chip_name: Optional[str],
	# ) -> Optional[CalibrationResult]:
	# 	"""Load cached readout data and validate freshness."""
	# 	raw = self._load_readout_cache_raw(chip_name=chip_name)
	# 	timestamps = raw.get("timestamps", {})
	# 	per_qubit = raw.get("per_qubit_confusion", {})
	# 	if not isinstance(timestamps, dict) or not isinstance(per_qubit, dict):
	# 		return None
	# 	now = datetime.now(timezone.utc)
	# 	selected_confusion: Dict[int, List[List[float]]] = {}
	# 	for q in target_qubits:
	# 		ts_str = timestamps.get(str(q))
	# 		if not cache_is_fresh(ts_str, now=now):
	# 			return None
	# 		mat = per_qubit.get(str(q))
	# 		if mat is None:
	# 			return None
	# 		selected_confusion[int(q)] = mat
	# 	return CalibrationResult(
	# 		target_qubits=list(target_qubits),
	# 		per_qubit_confusion=selected_confusion,
	# 	)

	def _save_readout_cache(self, result: CalibrationResult, *, chip_name: Optional[str]) -> None:
		"""Persist readout calibration data to cache.

		Args:
			result (*CalibrationResult*): Calibration result containing per-qubit confusion matrices.
			chip_name (*Optional[str]*): Name of the target chip.
		"""
		path = self._readout_cache_path(chip_name=chip_name)
		raw = self._load_readout_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_qubit = raw.get("per_qubit_confusion", {})
		now = datetime.now(timezone.utc).isoformat()
		for q in result.target_qubits:
			timestamps[str(q)] = now
			per_qubit[str(q)] = result.per_qubit_confusion[q]
		save_timestamped_payload(
			path,
			payload_key="per_qubit_confusion",
			timestamps=timestamps,
			payload=per_qubit,
		)

	def _readout_calibration_circuits(self, q: int) -> List[Tuple[str, QuantumCircuit]]:
		"""Build minimal calibration circuits for a single qubit.

		Args:
			q (*int*): Target qubit index.

		Returns:
			List of ``(bitstring_label, circuit)`` pairs for each basis state.
		"""
		circuits: List[Tuple[str, QuantumCircuit]] = []
		for i in range(2):
			bits = format(i, "01b")
			qc = QuantumCircuit(q+1)
			if bits == "1":
				qc.x(q)
			qc.measure([q], [0])
			circuits.append((bits, qc))
		return circuits

