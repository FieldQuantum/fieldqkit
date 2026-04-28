"""Process tomography for native two-qubit gates."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from ..api.backend import Backend
from ..circuit import QuantumCircuit
from ..core.observables import apply_measurement_basis_rotations
from ..core.readout import build_local_confusion_matrix, mitigate_readout
from ..core.utils import get_probabilities
from ._cache import cache_file, cache_is_fresh, load_timestamped_payload, save_timestamped_payload
from ._coupler_utils import coupler_key, resolve_positive_fidelity_couplers
from .readout import ReadoutCalibrationManager


class NativeTwoQubitTomographyManager:
	"""Run process tomography for native two-qubit gates with caching."""
	_STATE_PREP_OPS = {
		"0": (),
		"1": ("x",),
		"+": ("h",),
		"-": ("x", "h"),
		"+i": ("h", "s"),
		"-i": ("h", "sdg"),
	}
	_MEASUREMENT_AXES = ("X", "Y", "Z")
	_TWO_QUBIT_GATE_METHODS = {
		"cz": "cz",
		"cx": "cx",
		"iswap": "iswap",
		"ecr": "ecr",
	}
	_STATE_LABELS = ("0", "1", "+", "-", "+i", "-i")
	_STATE_VECTORS = {
		"0": (1.0 + 0.0j, 0.0 + 0.0j),
		"1": (0.0 + 0.0j, 1.0 + 0.0j),
		"+": (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
		"-": (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
		"+i": (1.0 / np.sqrt(2), 1.0j / np.sqrt(2)),
		"-i": (1.0 / np.sqrt(2), -1.0j / np.sqrt(2)),
	}
	_GATE_UNITARIES = {
		"cz": np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, -1],
		], dtype=complex),
		"cx": np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 0, 1],
			[0, 0, 1, 0],
		], dtype=complex),
		"iswap": np.array([
			[1, 0, 0, 0],
			[0, 0, 1j, 0],
			[0, 1j, 0, 0],
			[0, 0, 0, 1],
		], dtype=complex),
		"ecr": (1.0 / np.sqrt(2)) * np.array([
			[0, 0, 1, 1j],
			[0, 0, 1j, 1],
			[1, -1j, 0, 0],
			[-1j, 1, 0, 0],
		], dtype=complex),
	}
	_PAULI_LABELS = ("I", "X", "Y", "Z")
	_PAULI_SINGLE = (
		np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
		np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
		np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
		np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
	)

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
		"""Initialize a tomography manager with task submission and simulation callbacks.

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

		Args:
			couplers (*Optional[Sequence[Tuple[int, int]]]*): List of qubit coupler pairs. Defaults to ``None``.
			shots (*int*): Number of measurement shots. Defaults to ``1024``.
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.
			backend (*Optional[Backend]*): Hardware backend descriptor. Defaults to ``None``.
			qasm_version (*str*): OpenQASM version (``'2.0'`` or ``'3.0'``). Defaults to ``'2.0'``.
			readout_mitigation (*bool*): Whether to apply readout error mitigation. Defaults to ``True``.
			readout_shots (*Optional[int]*): Number of shots for readout calibration. Defaults to ``None``.
			print_true (*bool*): Whether to print progress information. Defaults to ``False``.

		Returns:
			dict: keyed by "q1-q2" with Choi matrix for the error channel.

		Raises:
			RuntimeError: backend is not set; use run_auto or provide backend.
			RuntimeError: chip_name is not set; use run_auto or provide chip_name.
		"""
		if backend is None:
			raise RuntimeError("backend is not set; use run_auto or provide backend")
		if chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")

		couplers = resolve_positive_fidelity_couplers(couplers, backend)
		use_simulator = str(chip_name).lower() == "simulator"

		raw = self._load_tomo_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_coupler = raw.get("per_coupler", {})
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
		basis_gate = "cx" if basis_gate in {"cnot", "cx"} else basis_gate
		ideal_ptm = self._ptm_from_unitary(basis_gate)

		for q1, q2 in couplers:
			key = coupler_key(q1, q2)
			ts_str = timestamps.get(key)
			cached = per_coupler.get(key)
			if cache_is_fresh(ts_str, now=now) and cached is not None:
				results[key] = self._decode_choi_payload(cached)
				continue

			if print_true:
				logger.info("run two-qubit process tomography on coupler %s", key)

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
						qasm = qct.to_openqasm2() if qasm_version == "2.0" else qct.to_openqasm3
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

	def _input_states(self) -> List[Tuple[str, str, np.ndarray]]:
		"""Generate all two-qubit input state preparations for process tomography.

		Returns:
			List of ``(label_a, label_b, rho)`` tuples where *rho* is the 4×4
			density matrix for the tensor-product input state.
		"""
		out: List[Tuple[str, str, np.ndarray]] = []
		for a in self._STATE_LABELS:
			for b in self._STATE_LABELS:
				rho = np.kron(self._state_density(a), self._state_density(b))
				out.append((a, b, rho))
		return out

	def _measurement_bases(self) -> List[Tuple[str, str]]:
		"""Generate all two-qubit measurement basis combinations for tomography.

		Returns:
			List of ``(axis_a, axis_b)`` label pairs over ``_MEASUREMENT_AXES``.
		"""
		return [(a, b) for a in self._MEASUREMENT_AXES for b in self._MEASUREMENT_AXES]

	def _state_density(self, label: str) -> np.ndarray:
		"""Compute the density matrix |psi><psi| for a single-qubit state specified by *label*.

		Args:
			label (*str*): State label (e.g. ``'0'``, ``'1'``, ``'+'``, ``'-'``, ``'+i'``, ``'-i'``).

		Returns:
			2×2 density matrix as ``np.ndarray``.

		Raises:
			ValueError: f'unsupported state label: {label}'
		"""
		entries = self._STATE_VECTORS.get(label)
		if entries is None:
			raise ValueError(f"unsupported state label: {label}")
		vec = np.asarray(entries, dtype=complex)
		return np.outer(vec, vec.conj())

	def _apply_state_prep(self, qc: QuantumCircuit, label: str, qubit: int) -> None:
		"""Prepare a single-qubit state by applying the gate sequence from ``_STATE_PREP_OPS[label]``.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			label (*str*): State label (e.g. ``'0'``, ``'1'``, ``'+'``).
			qubit (*int*): Target qubit index.

		Raises:
			ValueError: f'unsupported state label: {label}'
		"""
		ops = self._STATE_PREP_OPS.get(label)
		if ops is None:
			raise ValueError(f"unsupported state label: {label}")
		for op in ops:
			getattr(qc, op)(qubit)

	def _apply_measurement_basis(self, qc: QuantumCircuit, basis: str, qubit: int) -> None:
		"""Apply measurement basis rotation gates to the circuit.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			basis (*str*): Measurement basis (``'X'``, ``'Y'``, or ``'Z'``).
			qubit (*int*): Target qubit index.
		"""
		apply_measurement_basis_rotations(qc, [basis], target_qubits=[qubit])

	def _apply_two_qubit_gate(self, qc: QuantumCircuit, gate: str, q1: int, q2: int) -> None:
		"""Apply a two-qubit gate to the circuit.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			gate (*str*): Gate name (e.g. ``'cz'``, ``'cx'``, ``'iswap'``).
			q1 (*int*): First qubit index.
			q2 (*int*): Second qubit index.

		Raises:
			ValueError: f'unsupported two-qubit gate: {gate}'
		"""
		canonical_gate = "cx" if gate in {"cnot", "cx"} else gate
		method_name = self._TWO_QUBIT_GATE_METHODS.get(canonical_gate)
		if method_name is None:
			raise ValueError(f"unsupported two-qubit gate: {gate}")
		getattr(qc, method_name)(q1, q2)

	def _expectations_from_measurements(
		self,
		meas_probs: Dict[Tuple[str, str], np.ndarray],
	) -> Dict[Tuple[str, str], float]:
		# Expectation values of Pauli operators from basis measurements.
		"""Compute Pauli expectation values from basis measurement probability distributions.

		Args:
			meas_probs (*Dict[Tuple[str, str], np.ndarray]*): Mapping of ``(basis_a, basis_b)`` to probability arrays of length 4.

		Returns:
			Dict mapping ``(pauli_a, pauli_b)`` tuples to expectation values.
		"""
		expectations: Dict[Tuple[str, str], float] = {("I", "I"): 1.0}
		for (basis_a, basis_b), probs in meas_probs.items():
			ex1, ex2, ex12 = self._expectations_from_probs(probs)
			expectations[(basis_a, "I")] = ex1
			expectations[("I", basis_b)] = ex2
			expectations[(basis_a, basis_b)] = ex12
		return expectations

	def _expectations_from_probs(self, probs: np.ndarray) -> Tuple[float, float, float]:
		"""Extract single-qubit Pauli expectation values from a two-qubit probability distribution.

		Args:
			probs (*np.ndarray*): Probability distribution of length 4.

		Returns:
			Tuple of ``(ex_qubit0, ex_qubit1, ex_correlated)`` expectation values.

		Raises:
			ValueError: probabilities must have length 4 for two-qubit expectations
		"""
		if probs.shape[0] != 4:
			raise ValueError("probabilities must have length 4 for two-qubit expectations")
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
		"""Convert Pauli expectation values to a 16-element decomposition vector.

		Args:
			expectations (*Dict[Tuple[str, str], float]*): Mapping of ``(pauli_a, pauli_b)`` to expectation values.

		Returns:
			16-element ``np.ndarray`` in two-qubit Pauli basis order.
		"""
		vec = np.zeros(16, dtype=float)
		for i, a in enumerate(self._PAULI_LABELS):
			for j, b in enumerate(self._PAULI_LABELS):
				val = expectations.get((a, b))
				if val is None:
					val = 0.0
				vec[i * 4 + j] = float(val) / 4.0
		return vec

	def _rho_to_pauli_vector(self, rho: np.ndarray) -> np.ndarray:
		"""Convert a density matrix to its 16-element Pauli decomposition vector.

		Args:
			rho (*np.ndarray*): 4×4 density matrix.

		Returns:
			16-element ``np.ndarray`` with Pauli coefficients.
		"""
		basis = self._pauli_basis()
		vec = np.zeros(16, dtype=float)
		for idx, p in enumerate(basis):
			vec[idx] = float(np.trace(p @ rho).real) / 4.0
		return vec

	def _fit_ptm(self, out_mat: np.ndarray, in_mat: np.ndarray) -> np.ndarray:
		"""Fit the Pauli transfer matrix via pseudo-inverse: ``R = out_mat @ pinv(in_mat)``.

		Args:
			out_mat (*np.ndarray*): Output expectation matrix of shape ``(16, N)``.
			in_mat (*np.ndarray*): Input state Pauli expansion matrix of shape ``(16, N)``.

		Returns:
			16×16 Pauli transfer matrix as ``np.ndarray``.
		"""
		return out_mat @ np.linalg.pinv(in_mat)

	def _ptm_from_unitary(self, unitary) -> np.ndarray:
		"""Compute the Pauli transfer matrix from a unitary matrix or gate name.

		Args:
			unitary: A ``np.ndarray`` unitary matrix, or a gate name string
				(``'cz'``, ``'cx'``, ``'iswap'``, ``'ecr'``) that will be
				resolved to its standard unitary matrix.

		Returns:
			16×16 real Pauli transfer matrix as ``np.ndarray``.

		Raises:
			ValueError: If a string gate name is not recognised.
		"""
		if isinstance(unitary, str):
			mat = self._GATE_UNITARIES.get(unitary)
			if mat is None:
				raise ValueError(f"unknown gate name for PTM computation: {unitary}")
			unitary = mat
		basis = self._pauli_basis()
		ptm = np.zeros((16, 16), dtype=float)
		for i, p_i in enumerate(basis):
			for j, p_j in enumerate(basis):
				rot = unitary @ p_j @ unitary.conj().T
				ptm[i, j] = float(np.trace(p_i @ rot).real) / 4.0
		return ptm

	def _ptm_to_choi(self, ptm: np.ndarray) -> np.ndarray:
		"""Convert a 16×16 Pauli transfer matrix to its Choi representation.

		Computes ``C = (1/4) ∑_{ij} R_{ij} (P_i ⊗P_j^T)`` using the two-qubit
		Pauli basis.

		Args:
			ptm (*np.ndarray*): 16×16 Pauli transfer matrix.

		Returns:
			16×16 complex Choi matrix as ``np.ndarray``.
		"""
		basis = self._pauli_basis()
		choi = np.zeros((16, 16), dtype=complex)
		for i, p_i in enumerate(basis):
			for j, p_j in enumerate(basis):
				choi += ptm[i, j] * np.kron(p_i, p_j.T)
		return choi / 4.0

	def _pauli_basis(self) -> List[np.ndarray]:
		"""Return the list of all 16 two-qubit Pauli matrices in tensor-product order.

		Returns:
			List of 16 ``np.ndarray`` matrices of shape ``(4, 4)``.
		"""
		return [np.kron(a, b) for a in self._PAULI_SINGLE for b in self._PAULI_SINGLE]

	def _tomo_cache_path(self, *, chip_name: Optional[str]) -> Path:
		"""Resolve the on-disk cache file path for process tomography results.

		Args:
			chip_name (*Optional[str]*): Name of the target chip.

		Returns:
			``Path`` to the cache file.
		"""
		return cache_file(self._cache_dir, stem="tomo_two_qubit", chip_name=chip_name)

	def _load_tomo_cache_raw(self, *, chip_name: Optional[str]) -> Dict[str, object]:
		"""Load raw process tomography cache data from disk.

		Args:
			chip_name (*Optional[str]*): Name of the target chip.

		Returns:
			Dict with ``'timestamps'`` and ``'per_coupler'`` entries.
		"""
		path = self._tomo_cache_path(chip_name=chip_name)
		timestamps, per_coupler = load_timestamped_payload(path, payload_key="per_coupler")
		return {"timestamps": timestamps, "per_coupler": per_coupler}

	def _save_tomo_cache(self, results: Dict[str, Dict[str, object]], *, chip_name: Optional[str]) -> None:
		"""Persist process tomography results to the timestamped cache file.

		Args:
			results (*Dict[str, Dict[str, object]]*): Collection of result objects.
			chip_name (*Optional[str]*): Name of the target chip.
		"""
		path = self._tomo_cache_path(chip_name=chip_name)
		raw = self._load_tomo_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_coupler = raw.get("per_coupler", {})
		now = datetime.now(timezone.utc).isoformat()
		for key, payload in results.items():
			timestamps[key] = now
			per_coupler[key] = payload
		save_timestamped_payload(
			path,
			payload_key="per_coupler",
			timestamps=timestamps,
			payload=per_coupler,
		)

	def _encode_choi_payload(self, choi: np.ndarray) -> Dict[str, object]:
		"""Serialize a complex Choi matrix into separate real and imaginary lists for JSON storage.

		Args:
			choi (*np.ndarray*): Complex Choi matrix.

		Returns:
			Dict with ``'real'`` and ``'imag'`` list entries.
		"""
		return {
			"real": choi.real.tolist(),
			"imag": choi.imag.tolist(),
		}

	def _decode_choi_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
		"""Reconstruct a complex Choi matrix from its stored real/imaginary components.

		Args:
			payload (*Dict[str, object]*): Dict with ``'real'`` and ``'imag'`` list entries.

		Returns:
			Dict with ``'choi_error'`` key containing the reconstructed complex ``np.ndarray``.
		"""
		real = np.array(payload.get("real", []), dtype=float)
		imag = np.array(payload.get("imag", []), dtype=float)
		choi = real + 1j * imag
		return {"choi_error": choi}
