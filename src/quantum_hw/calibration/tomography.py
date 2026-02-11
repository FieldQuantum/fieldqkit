"""Process tomography for native two-qubit gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..api.backend import Backend
from ..circuit import QuantumCircuit
from ..circuit.matrix import gate_matrix_dict
from ..core.readout import build_local_confusion_matrix, mitigate_readout
from ..core.utils import get_probabilities
from .readout import ReadoutCalibrationManager


class NativeTwoQubitTomographyManager:
	"""Run process tomography for native two-qubit gates with caching."""

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
		self._cache_dir = cache_dir
		self._cache_dir.mkdir(parents=True, exist_ok=True)
		self._submit_openqasm_async = submit_openqasm_async
		self._wait_task = wait_task
		self._get_task_result = get_task_result
		self._compact_for_sim = compact_for_sim
		self._simulate_counts = simulate_counts

	def calibrate_native_two_qubit_tomography(
		self,
		couplers: Optional[Sequence[Tuple[int, int]]] = None,
		*,
		shots: int = 1024,
		chip_name: Optional[str] = None,
		backend: Optional[Backend] = None,
		qasm_version: str = "2.0",
		readout_mitigation: bool = True,
		readout_shots: Optional[int] = None,
		print_true: bool = False,
	) -> Dict[str, Dict[str, object]]:
		"""Run two-qubit process tomography and return error channels.

		Returns:
			dict: keyed by "q1-q2" with Choi matrix for the error channel.
		"""
		if backend is None:
			raise RuntimeError("backend is not set; use run_auto or provide backend")
		if chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")

		couplers = self._resolve_couplers(couplers, backend)
		use_simulator = str(chip_name).lower() == "simulator"

		raw = self._load_tomo_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {}) if isinstance(raw, dict) else {}
		per_coupler = raw.get("per_coupler", {}) if isinstance(raw, dict) else {}
		now = datetime.now(timezone.utc)

		per_qubit_confusion: Optional[Dict[int, np.ndarray]] = None
		if readout_mitigation:
			# Reuse readout cache to mitigate tomography measurements.
			readout_manager = ReadoutCalibrationManager(
				cache_dir=self._cache_dir,
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
		basis_gate = backend.two_qubit_gate_basis
		ideal_ptm = self._ptm_from_unitary(self._native_gate_matrix(basis_gate))

		for q1, q2 in couplers:
			key = self._coupler_key(q1, q2)
			ts_str = timestamps.get(key) if isinstance(timestamps, dict) else None
			cached = per_coupler.get(key) if isinstance(per_coupler, dict) else None
			if ts_str and cached is not None:
				ts = datetime.fromisoformat(ts_str)
				if now - ts <= timedelta(hours=1):
					results[key] = self._decode_choi_payload(cached)
					continue

			if print_true:
				print(f"[tomo] run two-qubit process tomography on coupler {key}")

			local_cm = None
			if readout_mitigation:
				if per_qubit_confusion is None:
					raise RuntimeError("readout mitigation requested but calibration is missing")
				local_cm = build_local_confusion_matrix(per_qubit_confusion, [q1, q2])

			input_states = self._input_states()
			basis_pairs = self._measurement_bases()

			input_vectors: List[np.ndarray] = []
			output_vectors: List[np.ndarray] = []
			pending: List[Tuple[int, Tuple[str, str], object]] = []
			meas_cache: Dict[int, Dict[Tuple[str, str], np.ndarray]] = {i: {} for i in range(len(input_states))}

			for idx, (prep_a, prep_b, rho_in) in enumerate(input_states):
				input_vectors.append(self._rho_to_pauli_vector(rho_in))
				for basis_a, basis_b in basis_pairs:
					qc = QuantumCircuit(max(q1, q2) + 1)
					self._apply_state_prep(qc, prep_a, q1)
					self._apply_state_prep(qc, prep_b, q2)
					self._apply_two_qubit_gate(qc, basis_gate, q1, q2)
					self._apply_measurement_basis(qc, basis_a, q1)
					self._apply_measurement_basis(qc, basis_b, q2)
					qc.measure([q1, q2], [0, 1])
					qct = qc

					if use_simulator:
						qct_sim = self._compact_for_sim(qct)
						if isinstance(qct_sim, tuple):
							qct_sim = qct_sim[0]
						counts = self._simulate_counts(qct_sim, shots)
						probs = get_probabilities(counts, 2)
						if local_cm is not None:
							probs = mitigate_readout(probs, local_cm)
						meas_cache[idx][(basis_a, basis_b)] = probs
					else:
						qasm = qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3
						task_id = self._submit_openqasm_async(
							name=f"tomo_2q_{key}_{prep_a}{prep_b}_{basis_a}{basis_b}",
							qasm=qasm,
							shots=shots,
							chip_name=chip_name,
						)
						pending.append((idx, (basis_a, basis_b), task_id))

			if not use_simulator:
				for idx, basis_pair, task_id in pending:
					status = self._wait_task(task_id)
					if status != "Finished":
						raise RuntimeError(f"tomography task {task_id} ended with status {status}")
					res = self._get_task_result(task_id)
					counts = res["count"]
					probs = get_probabilities(counts, 2)
					if local_cm is not None:
						probs = mitigate_readout(probs, local_cm)
					meas_cache[idx][basis_pair] = probs

			for idx in range(len(input_states)):
				exp_vals = self._expectations_from_measurements(meas_cache[idx])
				output_vectors.append(self._expectations_to_pauli_vector(exp_vals))

			ptm_actual = self._fit_ptm(np.column_stack(output_vectors), np.column_stack(input_vectors))
			ptm_error = ptm_actual @ np.linalg.pinv(ideal_ptm)
			choi_error = self._ptm_to_choi(ptm_error)

			payload = self._encode_choi_payload(choi_error)
			results[key] = {"choi_error": choi_error}
			self._save_tomo_cache({key: payload}, chip_name=chip_name)

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

	def _input_states(self) -> List[Tuple[str, str, np.ndarray]]:
		states = ["0", "1", "+", "-", "+i", "-i"]
		out: List[Tuple[str, str, np.ndarray]] = []
		for a in states:
			for b in states:
				rho = np.kron(self._state_density(a), self._state_density(b))
				out.append((a, b, rho))
		return out

	def _measurement_bases(self) -> List[Tuple[str, str]]:
		axes = ["X", "Y", "Z"]
		return [(a, b) for a in axes for b in axes]

	def _state_density(self, label: str) -> np.ndarray:
		if label == "0":
			vec = np.array([1.0, 0.0], dtype=complex)
		elif label == "1":
			vec = np.array([0.0, 1.0], dtype=complex)
		elif label == "+":
			vec = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
		elif label == "-":
			vec = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
		elif label == "+i":
			vec = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)
		elif label == "-i":
			vec = np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2)
		else:
			raise ValueError(f"unsupported state label: {label}")
		return np.outer(vec, vec.conj())

	def _apply_state_prep(self, qc: QuantumCircuit, label: str, qubit: int) -> None:
		if label == "0":
			return
		if label == "1":
			qc.x(qubit)
			return
		if label == "+":
			qc.h(qubit)
			return
		if label == "-":
			qc.x(qubit)
			qc.h(qubit)
			return
		if label == "+i":
			qc.h(qubit)
			qc.s(qubit)
			return
		if label == "-i":
			qc.h(qubit)
			qc.sdg(qubit)
			return
		raise ValueError(f"unsupported state label: {label}")

	def _apply_measurement_basis(self, qc: QuantumCircuit, basis: str, qubit: int) -> None:
		if basis == "X":
			qc.h(qubit)
			return
		if basis == "Y":
			qc.sdg(qubit)
			qc.h(qubit)
			return
		if basis == "Z":
			return
		raise ValueError(f"unsupported measurement basis: {basis}")

	def _apply_two_qubit_gate(self, qc: QuantumCircuit, gate: str, q1: int, q2: int) -> None:
		gate = "cx" if gate in {"cnot", "cx"} else gate
		if gate == "cz":
			qc.cz(q1, q2)
			return
		if gate == "cx":
			qc.cx(q1, q2)
			return
		if gate == "iswap":
			qc.iswap(q1, q2)
			return
		if gate == "ecr":
			qc.ecr(q1, q2)
			return
		raise ValueError(f"unsupported two-qubit gate: {gate}")

	def _native_gate_matrix(self, gate: str) -> np.ndarray:
		gate = "cx" if gate in {"cnot", "cx"} else gate
		mat = gate_matrix_dict.get(gate)
		if mat is None:
			raise ValueError(f"unsupported two-qubit basis gate: {gate}")
		return mat

	def _expectations_from_measurements(
		self,
		meas_probs: Dict[Tuple[str, str], np.ndarray],
	) -> Dict[Tuple[str, str], float]:
		# Expectation values of Pauli operators from basis measurements.
		expectations: Dict[Tuple[str, str], float] = {("I", "I"): 1.0}
		for (basis_a, basis_b), probs in meas_probs.items():
			ex1, ex2, ex12 = self._expectations_from_probs(probs)
			expectations[(basis_a, "I")] = ex1
			expectations[("I", basis_b)] = ex2
			expectations[(basis_a, basis_b)] = ex12
		return expectations

	def _expectations_from_probs(self, probs: np.ndarray) -> Tuple[float, float, float]:
		# probs order is 00,01,10,11 with q0 as the first bit.
		ex1 = 0.0
		ex2 = 0.0
		ex12 = 0.0
		for idx, p in enumerate(probs):
			bits = format(idx, "02b")
			b0 = int(bits[0])
			b1 = int(bits[1])
			s1 = 1.0 if b0 == 0 else -1.0
			s2 = 1.0 if b1 == 0 else -1.0
			ex1 += s1 * float(p)
			ex2 += s2 * float(p)
			ex12 += s1 * s2 * float(p)
		return ex1, ex2, ex12

	def _expectations_to_pauli_vector(self, expectations: Dict[Tuple[str, str], float]) -> np.ndarray:
		basis = ["I", "X", "Y", "Z"]
		vec = np.zeros(16, dtype=float)
		for i, a in enumerate(basis):
			for j, b in enumerate(basis):
				val = expectations.get((a, b))
				if val is None:
					val = 0.0
				vec[i * 4 + j] = float(val) / 4.0
		return vec

	def _rho_to_pauli_vector(self, rho: np.ndarray) -> np.ndarray:
		basis = self._pauli_basis()
		vec = np.zeros(16, dtype=float)
		for idx, p in enumerate(basis):
			vec[idx] = float(np.trace(p @ rho).real) / 4.0
		return vec

	def _fit_ptm(self, out_mat: np.ndarray, in_mat: np.ndarray) -> np.ndarray:
		return out_mat @ np.linalg.pinv(in_mat)

	def _ptm_from_unitary(self, unitary: np.ndarray) -> np.ndarray:
		basis = self._pauli_basis()
		ptm = np.zeros((16, 16), dtype=float)
		for i, p_i in enumerate(basis):
			for j, p_j in enumerate(basis):
				rot = unitary @ p_j @ unitary.conj().T
				ptm[i, j] = float(np.trace(p_i @ rot).real) / 4.0
		return ptm

	def _ptm_to_choi(self, ptm: np.ndarray) -> np.ndarray:
		basis = self._pauli_basis()
		choi = np.zeros((16, 16), dtype=complex)
		for i, p_i in enumerate(basis):
			for j, p_j in enumerate(basis):
				choi += ptm[i, j] * np.kron(p_i, p_j.T)
		return choi / 4.0

	def _pauli_basis(self) -> List[np.ndarray]:
		I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
		X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
		Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
		Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
		single = [I, X, Y, Z]
		return [np.kron(a, b) for a in single for b in single]

	def _coupler_key(self, q1: int, q2: int) -> str:
		return f"{min(q1, q2)}-{max(q1, q2)}"

	def _tomo_cache_path(self, *, chip_name: Optional[str]) -> Path:
		name = chip_name if chip_name is not None else "unknown"
		return self._cache_dir / f"tomo_two_qubit_{name}.json"

	def _load_tomo_cache_raw(self, *, chip_name: Optional[str]) -> Dict[str, object]:
		path = self._tomo_cache_path(chip_name=chip_name)
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

	def _save_tomo_cache(self, results: Dict[str, Dict[str, object]], *, chip_name: Optional[str]) -> None:
		path = self._tomo_cache_path(chip_name=chip_name)
		raw = self._load_tomo_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {}) if isinstance(raw, dict) else {}
		per_coupler = raw.get("per_coupler", {}) if isinstance(raw, dict) else {}
		now = datetime.now(timezone.utc).isoformat()
		for key, payload in results.items():
			timestamps[key] = now
			per_coupler[key] = payload
		payload = {"timestamps": timestamps, "per_coupler": per_coupler}
		path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

	def _encode_choi_payload(self, choi: np.ndarray) -> Dict[str, object]:
		return {
			"real": choi.real.tolist(),
			"imag": choi.imag.tolist(),
		}

	def _decode_choi_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
		real = np.array(payload.get("real", []), dtype=float)
		imag = np.array(payload.get("imag", []), dtype=float)
		choi = real + 1j * imag
		return {"choi_error": choi}
