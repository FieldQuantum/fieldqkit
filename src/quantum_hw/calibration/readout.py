"""Readout calibration workflows and caching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..circuit import QuantumCircuit
from ..core.types import CalibrationResult
from ..core.utils import get_probabilities
from ..api.backend import Backend


class ReadoutCalibrationManager:
	"""Handle readout calibration runs with caching."""

	def __init__(
		self,
		*,
		cache_dir: Path,
		transpile_with_backend: Callable[[QuantumCircuit, object, Optional[Sequence[int]]], QuantumCircuit],
		submit_openqasm_async: Callable[[str, str, int, Optional[str]], object],
		wait_task: Callable[[object], str],
		get_task_result: Callable[[object], Dict[str, object]],
		compact_for_sim: Callable[[QuantumCircuit], object],
		simulate_counts: Callable[[QuantumCircuit, int], Dict[str, int]],
	) -> None:
		self._cache_dir = cache_dir
		self._cache_dir.mkdir(parents=True, exist_ok=True)
		self._transpile_with_backend = transpile_with_backend
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
		"""Calibrate readout error for selected qubits with caching."""
		if backend is None:
			raise RuntimeError("backend is not set; use run_auto or provide backend")
		target_qubits = self._resolve_target_qubits(target_qubits, backend)
		print(target_qubits)
		if shots is None:
			shots = 1024
		if chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")

		use_simulator = str(chip_name).lower() == "simulator"

		# Cache is per-chip; only stale/missing qubits are recalibrated.
		raw = self._load_readout_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {}) if isinstance(raw, dict) else {}
		per_qubit_raw = raw.get("per_qubit_confusion", {}) if isinstance(raw, dict) else {}
		now = datetime.now(timezone.utc)
		cached_confusion: Dict[int, List[List[float]]] = {}
		missing: List[int] = []
		for q in target_qubits:
			ts_str = timestamps.get(str(q)) if isinstance(timestamps, dict) else None
			mat = per_qubit_raw.get(str(q)) if isinstance(per_qubit_raw, dict) else None
			if ts_str is None or mat is None:
				missing.append(q)
				continue
			ts = datetime.fromisoformat(ts_str)
			if now - ts > timedelta(hours=1):
				missing.append(q)
				continue
			cached_confusion[int(q)] = mat
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
			for bits, qc in self._readout_calibration_circuits():
				qct = self._transpile_with_backend(qc, backend, target_qubits=[q])
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
			per_qubit_confusion[q] = self._build_confusion_matrix(counts_list)
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
		"""Resolve target qubits from backend metadata when not provided."""
		if target_qubits is not None:
			return list(target_qubits)

		qubits: List[int] = []
		qubits_with_attributes = getattr(backend, "qubits_with_attributes", None)
		if isinstance(qubits_with_attributes, list) and qubits_with_attributes:
			qubits = sorted(int(q) for q, _ in qubits_with_attributes if backend.chip_info["qubits_info"][f"Q{q}"]["fidelity"] > 0)
			return qubits

		raise RuntimeError("target_qubits is not set and backend.qubits_with_attributes is missing")

	def _readout_cache_path(self, *, chip_name: Optional[str]) -> Path:
		"""Resolve the on-disk cache path for readout calibration."""
		name = chip_name if chip_name is not None else "unknown"
		return self._cache_dir / f"readout_{name}.json"

	def _load_readout_cache_raw(self, *, chip_name: Optional[str]) -> Dict[str, object]:
		"""Load cached readout data from disk (raw dictionary)."""
		path = self._readout_cache_path(chip_name=chip_name)
		if not path.exists():
			return {"timestamps": {}, "per_qubit_confusion": {}}
		try:
			data = json.loads(path.read_text(encoding="utf-8"))
			if not isinstance(data, dict):
				return {"timestamps": {}, "per_qubit_confusion": {}}
			timestamps = data.get("timestamps", {})
			per_qubit = data.get("per_qubit_confusion", {})
			if not isinstance(timestamps, dict) or not isinstance(per_qubit, dict):
				return {"timestamps": {}, "per_qubit_confusion": {}}
			return {"timestamps": timestamps, "per_qubit_confusion": per_qubit}
		except Exception:
			return {"timestamps": {}, "per_qubit_confusion": {}}

	def _load_readout_cache(
		self,
		target_qubits: Sequence[int],
		*,
		chip_name: Optional[str],
	) -> Optional[CalibrationResult]:
		"""Load cached readout data and validate freshness."""
		raw = self._load_readout_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_qubit = raw.get("per_qubit_confusion", {})
		if not isinstance(timestamps, dict) or not isinstance(per_qubit, dict):
			return None
		now = datetime.now(timezone.utc)
		selected_confusion: Dict[int, List[List[float]]] = {}
		for q in target_qubits:
			ts_str = timestamps.get(str(q))
			if ts_str is None:
				return None
			ts = datetime.fromisoformat(ts_str)
			if now - ts > timedelta(hours=1):
				return None
			mat = per_qubit.get(str(q))
			if mat is None:
				return None
			selected_confusion[int(q)] = mat
		return CalibrationResult(
			target_qubits=list(target_qubits),
			per_qubit_confusion=selected_confusion,
		)

	def _save_readout_cache(self, result: CalibrationResult, *, chip_name: Optional[str]) -> None:
		"""Persist readout calibration data to cache."""
		path = self._readout_cache_path(chip_name=chip_name)
		raw = self._load_readout_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {}) if isinstance(raw, dict) else {}
		per_qubit = raw.get("per_qubit_confusion", {}) if isinstance(raw, dict) else {}
		now = datetime.now(timezone.utc).isoformat()
		for q in result.target_qubits:
			timestamps[str(q)] = now
			per_qubit[str(q)] = result.per_qubit_confusion[q]
		payload = {
			"timestamps": timestamps,
			"per_qubit_confusion": per_qubit,
		}
		path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

	def _readout_calibration_circuits(self) -> List[Tuple[str, QuantumCircuit]]:
		"""Build minimal calibration circuits for a single qubit."""
		circuits: List[Tuple[str, QuantumCircuit]] = []
		for i in range(2):
			bits = format(i, "01b")
			qc = QuantumCircuit(1)
			if bits == "1":
				qc.x(0)
			qc.measure([0], [0])
			circuits.append((bits, qc))
		return circuits

	def _build_confusion_matrix(self, res_list: Sequence[Dict[str, int]]) -> np.ndarray:
		"""Build a 2x2 confusion matrix from two calibration results."""
		mat = np.zeros((2, 2), dtype=float)
		for i, res in enumerate(res_list):
			probs = get_probabilities(res, 1)
			mat[i, :] = probs
		return mat
