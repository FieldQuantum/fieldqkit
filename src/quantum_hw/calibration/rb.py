"""Randomized benchmarking utilities for native two-qubit gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..circuit import QuantumCircuit
from ..circuit.matrix import gate_matrix_dict, id_mat
from ..core.readout import build_local_confusion_matrix, mitigate_readout
from ..core.utils import get_probabilities
from ..api.backend import Backend
from .readout import ReadoutCalibrationManager


class NativeTwoQubitRBManager:
	"""Run native two-qubit gate randomized benchmarking with caching."""

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

	def calibrate_native_two_qubit_rb(
		self,
		couplers: Optional[Sequence[Tuple[int, int]]] = None,
		*,
		lengths: Optional[Sequence[int]] = None,
		num_sequences: int = 20,
		shots: int = 1024,
		chip_name: Optional[str] = None,
		backend: Optional[Backend] = None,
		qasm_version: str = "2.0",
		readout_mitigation: bool = True,
		readout_shots: Optional[int] = None,
		seed: Optional[int] = None,
		print_true: bool = False,
	) -> Dict[str, Dict[str, object]]:
		"""Run native two-qubit RB and return per-coupler results.

		Returns:
			dict: keyed by "q1-q2" with averaged survival probabilities and fit.
		"""
		if backend is None:
			raise RuntimeError("backend is not set; use run_auto or provide backend")
		if chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")
		if lengths is None:
			lengths = [1, 2, 4, 8, 16, 32]

		couplers = self._resolve_couplers(couplers, backend)
		use_simulator = str(chip_name).lower() == "simulator"
		rng = np.random.default_rng(seed)

		raw = self._load_rb_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {}) if isinstance(raw, dict) else {}
		per_coupler = raw.get("per_coupler", {}) if isinstance(raw, dict) else {}
		now = datetime.now(timezone.utc)

		per_qubit_confusion: Optional[Dict[int, np.ndarray]] = None
		if readout_mitigation:
			# Reuse readout cache to mitigate measured probabilities for RB survival.
			readout_manager = ReadoutCalibrationManager(
				cache_dir=self._cache_dir,
				transpile_with_backend=self._transpile_with_backend,
				submit_openqasm_async=self._submit_openqasm_async,
				wait_task=self._wait_task,
				get_task_result=self._get_task_result,
				compact_for_sim=self._compact_for_sim,
				simulate_counts=self._simulate_counts,
			)
			qubits = sorted({q for pair in couplers for q in pair})
			cal = readout_manager.calibrate_readout(
				target_qubits=qubits,
				shots=readout_shots,
				chip_name=chip_name,
				backend=backend,
				qasm_version=qasm_version,
				print_true=print_true,
			)
			per_qubit_confusion = {k: np.asarray(v) for k, v in cal.per_qubit_confusion.items()}

		results: Dict[str, Dict[str, object]] = {}

		for q1, q2 in couplers:
			pending: List[Tuple[int, object]] = []
			local_cm = None
			if readout_mitigation:
				if per_qubit_confusion is None:
					raise RuntimeError("readout mitigation requested but calibration is missing")
				# Two-qubit local confusion matrix for the coupler.
				local_cm = build_local_confusion_matrix(per_qubit_confusion, [q1, q2])
			key = self._coupler_key(q1, q2)
			ts_str = timestamps.get(key) if isinstance(timestamps, dict) else None
			cached = per_coupler.get(key) if isinstance(per_coupler, dict) else None
			if ts_str and cached is not None:
				ts = datetime.fromisoformat(ts_str)
				if now - ts <= timedelta(hours=1):
					results[key] = {"fit": {"fidelity": cached}}
					continue

			if print_true:
				print(f"[rb] run native two-qubit RB on coupler {key}")

			survival_samples: Dict[int, List[float]] = {length: [] for length in lengths}
			# Total gate count includes forward sequence plus explicit inverse sequence.
			total_length_by_length: Dict[int, int] = {}
			for length in lengths:
				for m in range(num_sequences):
					qc, total_length = self._build_random_sequence(
						length,
						backend.two_qubit_gate_basis,
						rng,
					)
					total_length_by_length[length] = total_length
					qc.measure([0, 1], [0, 1])
					qct = self._transpile_with_backend(qc, backend, target_qubits=[q1, q2], optimize_level=0)

					if use_simulator:
						qct_sim = self._compact_for_sim(qct)
						if isinstance(qct_sim, tuple):
							qct_sim = qct_sim[0]
						counts = self._simulate_counts(qct_sim, shots)
						probs = get_probabilities(counts, 2)
						if local_cm is not None:
							probs = mitigate_readout(probs, local_cm)
						survival_samples[length].append(float(probs[0]))
					else:
						qasm = qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3
						task_id = self._submit_openqasm_async(
							name=f"rb_2q_{key}_L{length}_batch{m}",
							qasm=qasm,
							shots=shots,
							chip_name=chip_name,
						)
						pending.append((length, task_id))
	
			if use_simulator:
				avg_survival = {
					length: float(np.mean(survival_samples[length])) if survival_samples[length] else 0.0
					for length in lengths
				}
				fit = self._fit_decay(
					[total_length_by_length[length] for length in lengths],
					[avg_survival[length] for length in lengths],
				)
				results[key] = {
					"lengths": list(lengths),
					"total_lengths": [total_length_by_length[length] for length in lengths],
					"num_sequences": num_sequences,
					"shots": shots,
					"survival_samples": {str(k): v for k, v in survival_samples.items()},
					"survival_avg": avg_survival,
					"fit": fit,
				}

			if not use_simulator:
				per_coupler_survival: Dict[int, List[float]] = {}
				for length, task_id in pending:
					status = self._wait_task(task_id)
					if status != "Finished":
						raise RuntimeError(f"rb task {task_id} ended with status {status}")
					res = self._get_task_result(task_id)
					counts = res["count"]
					probs = get_probabilities(counts, 2)
					if local_cm is not None:
						probs = mitigate_readout(probs, local_cm)
					per_coupler_survival.setdefault(length, []).append(float(probs[0]))

				avg_survival = {
					length: float(np.mean(per_coupler_survival.get(length, []))) if per_coupler_survival.get(length) else 0.0
					for length in lengths
				}
				fit = self._fit_decay(
					[total_length_by_length[length] for length in lengths],
					[avg_survival[length] for length in lengths],
				)
				results[key] = {
					"lengths": list(lengths),
					"total_lengths": [total_length_by_length[length] for length in lengths],
					"num_sequences": num_sequences,
					"shots": shots,
					"survival_samples": {str(k): v for k, v in per_coupler_survival.items()},
					"survival_avg": avg_survival,
					"fit": fit,
				}
			print(f"Coupler {key}: fidelity={results[key]['fit']['fidelity']}")

			# Cache stores fidelity only to keep payload minimal.
			self._save_rb_cache(results, chip_name=chip_name)
		return results

	def _resolve_couplers(
		self,
		couplers: Optional[Sequence[Tuple[int, int]]],
		backend: Backend,
	) -> List[Tuple[int, int]]:
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

	def _build_random_sequence(
		self,
		length: int,
		basis_gate: str,
		rng: np.random.Generator,
	) -> Tuple[QuantumCircuit, np.ndarray]:
		qc = QuantumCircuit(2)
		total = np.eye(4, dtype=complex)
		# Pauli-only single-qubit twirl for native two-qubit RB.
		single_gates = ["id", "x", "y", "z"]

		basis_gate = "cx" if basis_gate in {"cnot", "cx"} else basis_gate
		basis_mat = gate_matrix_dict.get(basis_gate)
		if basis_mat is None:
			raise ValueError(f"unsupported two-qubit basis gate: {basis_gate}")
		gates_list = []
		# Build forward sequence, then apply explicit inverse sequence.
		for l in range(length):
			g1 = rng.choice(single_gates)
			g2 = rng.choice(single_gates)
			self._apply_single_gate(qc, g1, 0)
			self._apply_single_gate(qc, g2, 1)
			self._apply_two_qubit_gate(qc, basis_gate, 0, 1)
			gates_list.append([g1, g2, basis_gate])
		for l in range(length):
			self._apply_two_qubit_gate_dg(qc, gates_list[length - 1 - l][2], 0, 1)	
			self._apply_single_gate_dg(qc, gates_list[length - 1 - l][0], 0)
			self._apply_single_gate_dg(qc, gates_list[length - 1 - l][1], 1)

		if basis_gate == "iswap":
			total_length = 4 * length
		elif basis_gate in ["ecr", "cz", "cx", "cnot"]:
			total_length = 2 * length
		return qc, total_length

	def _apply_single_gate(self, qc: QuantumCircuit, gate: str, qubit: int) -> None:
		if gate == "id":
			return
		if gate == "x":
			qc.x(qubit)
			return
		if gate == "y":
			qc.y(qubit)
			return
		if gate == "z":
			qc.z(qubit)
			return
		if gate == "h":
			qc.h(qubit)
			return
		if gate == "s":
			qc.s(qubit)
			return
		if gate == "sdg":
			qc.sdg(qubit)
			return
		if gate == "sx":
			qc.sx(qubit)
			return
		if gate == "sxdg":
			qc.sxdg(qubit)
			return
		raise ValueError(f"unsupported single-qubit gate: {gate}")

	def _apply_single_gate_dg(self, qc: QuantumCircuit, gate: str, qubit: int) -> None:
		if gate == "id":
			return
		if gate == "x":
			qc.x(qubit)
			return
		if gate == "y":
			qc.y(qubit)
			return
		if gate == "z":
			qc.z(qubit)
			return
		if gate == "h":
			qc.h(qubit)
			return
		if gate == "s":
			qc.sdg(qubit)
			return
		if gate == "sdg":
			qc.s(qubit)
			return
		if gate == "sx":
			qc.sxdg(qubit)
			return
		if gate == "sxdg":
			qc.sx(qubit)
			return
		raise ValueError(f"unsupported single-qubit gate: {gate}")

	def _apply_two_qubit_gate(self, qc: QuantumCircuit, gate: str, q1: int, q2: int) -> None:
		if gate == "cz":
			qc.cz(q1, q2)
			return
		if gate in {"cx", "cnot"}:
			qc.cx(q1, q2)
			return
		if gate == "iswap":
			qc.iswap(q1, q2)
			return
		if gate == "ecr":
			qc.ecr(q1, q2)
			return
		raise ValueError(f"unsupported two-qubit gate: {gate}")

	def _apply_two_qubit_gate_dg(self, qc: QuantumCircuit, gate: str, q1: int, q2: int) -> None:
		if gate == "cz":
			qc.cz(q1, q2)
			return
		if gate in {"cx", "cnot"}:
			qc.cx(q1, q2)
			return
		if gate == "iswap":
			qc.iswap(q1, q2)
			qc.iswap(q1, q2)
			qc.iswap(q1, q2)
			return
		if gate == "ecr":
			qc.ecr(q1, q2)
			return
		raise ValueError(f"unsupported two-qubit gate: {gate}")

	def _fit_decay(self, lengths: List[int], survival: List[float]) -> Dict[str, float | None]:
		dim = 4
		b = 1.0 / dim
		x = np.asarray(lengths, dtype=float)
		y = np.asarray(survival, dtype=float) - b
		mask = y > 0
		if mask.sum() < 2:
			return {"p": None, "epc": None, "fidelity": None, "A": None, "B": b}
		logy = np.log(y[mask])
		coeff = np.polyfit(x[mask], logy, 1)
		p = float(np.exp(coeff[0]))
		a = float(np.exp(coeff[1]))
		f_avg = float(((dim - 1) * p + 1) / dim)
		epc = float(1.0 - f_avg)
		return {"p": p, "epc": epc, "fidelity": f_avg, "A": a, "B": b}

	def _coupler_key(self, q1: int, q2: int) -> str:
		return f"{min(q1, q2)}-{max(q1, q2)}"

	def _rb_cache_path(self, *, chip_name: Optional[str]) -> Path:
		name = chip_name if chip_name is not None else "unknown"
		return self._cache_dir / f"rb_two_qubit_{name}.json"

	def _load_rb_cache_raw(self, *, chip_name: Optional[str]) -> Dict[str, object]:
		path = self._rb_cache_path(chip_name=chip_name)
		if not path.exists():
			return {"timestamps": {}, "per_coupler": {}}
		try:
			data = json.loads(path.read_text(encoding="utf-8"))
			if not isinstance(data, dict):
				return {"timestamps": {}, "per_coupler": {}}
			timestamps = data.get("timestamps", {})
			per_coupler = data.get("per_coupler", {})
			if not isinstance(timestamps, dict) or not isinstance(per_coupler, dict):
				return {"timestamps": {}, "per_coupler": {}}
			return {"timestamps": timestamps, "per_coupler": per_coupler}
		except Exception:
			return {"timestamps": {}, "per_coupler": {}}

	def _save_rb_cache(self, results: Dict[str, Dict[str, object]], *, chip_name: Optional[str]) -> None:
		path = self._rb_cache_path(chip_name=chip_name)
		raw = self._load_rb_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {}) if isinstance(raw, dict) else {}
		per_coupler = raw.get("per_coupler", {}) if isinstance(raw, dict) else {}
		now = datetime.now(timezone.utc).isoformat()
		for key, payload in results.items():
			fit = payload.get("fit", {}) if isinstance(payload, dict) else {}
			fidelity = fit.get("fidelity") if isinstance(fit, dict) else None
			timestamps[key] = now
			per_coupler[key] = fidelity
		payload = {"timestamps": timestamps, "per_coupler": per_coupler}
		path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
