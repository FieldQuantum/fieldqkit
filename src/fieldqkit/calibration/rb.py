"""Randomized benchmarking utilities for native two-qubit gates."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


import numpy as np

from ..circuit import QuantumCircuit
from ..core.readout import build_local_confusion_matrix, mitigate_readout
from ..core.utils import get_probabilities
from ..api.backend import Backend
from ._cache import cache_file, cache_is_fresh, load_timestamped_payload, save_timestamped_payload
from ._coupler_utils import coupler_key, resolve_positive_fidelity_couplers
from .readout import ReadoutCalibrationManager

logger = logging.getLogger(__name__)


class NativeTwoQubitRBManager:
	"""Run native two-qubit gate randomized benchmarking with caching."""
	_SINGLE_GATE_DAGGER = {
		"id": "id",
		"x": "x",
		"y": "y",
		"z": "z",
		"h": "h",
		"s": "sdg",
		"sdg": "s",
		"sx": "sxdg",
		"sxdg": "sx",
	}

	def __init__(
		self,
		*,
		cache_dir: Path,
		submit_circuit_async: Callable,
		wait_task: Callable[[object], str],
		get_task_result: Callable[[object], Dict[str, object]],
		compact_for_sim: Callable[[QuantumCircuit], object],
		simulate_counts: Callable[[QuantumCircuit, int], Dict[str, int]],
	) -> None:
		"""Initialize randomized benchmarking manager with task submission and simulation capabilities.

		Args:
			cache_dir (*Path*): Directory for cache files.
			submit_circuit_async (*Callable[[str, QuantumCircuit, int, Optional[str], Optional[Dict]], object]*): Callback ``(name, circuit, shots, chip_name, submit_options)`` that submits a circuit and returns a task handle.
			wait_task (*Callable[[object], str]*): Callback to block until a task completes and return its status.
			get_task_result (*Callable[[object], Dict[str, object]]*): Callback to retrieve measurement results from a completed task.
			compact_for_sim (*Callable[[QuantumCircuit], object]*): Callback to prepare a circuit for local simulation.
			simulate_counts (*Callable[[QuantumCircuit, int], Dict[str, int]]*): Callback to simulate a circuit locally and return bitstring counts.
		"""
		self._cache_dir = cache_dir
		self._cache_dir.mkdir(parents=True, exist_ok=True)
		self._submit_circuit_async = submit_circuit_async
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
		readout_mitigation: bool = True,
		readout_shots: Optional[int] = None,
		seed: Optional[int] = None,
		print_true: bool = False,
	) -> Dict[str, Dict[str, object]]:
		"""Run native two-qubit RB and return per-coupler results.

		Args:
			couplers (*Optional[Sequence[Tuple[int, int]]]*): List of qubit coupler pairs. Defaults to ``None``.
			lengths (*Optional[Sequence[int]]*): Sequence lengths for RB decay curve. Defaults to ``None`` (auto-generated).
			num_sequences (*int*): Number of random Clifford sequences per length. Defaults to ``20``.
			shots (*int*): Number of measurement shots. Defaults to ``1024``.
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.
			backend (*Optional[Backend]*): Hardware backend descriptor. Defaults to ``None``.
			readout_mitigation (*bool*): Whether to apply readout error mitigation. Defaults to ``True``.
			readout_shots (*Optional[int]*): Number of shots for readout calibration. Defaults to ``None``.
			seed (*Optional[int]*): Random seed for reproducibility. Defaults to ``None``.
			print_true (*bool*): Whether to print progress information. Defaults to ``False``.

		Returns:
			dict: keyed by "q1-q2" with averaged survival probabilities and fit.

		Raises:
			RuntimeError: backend is not set; use run_auto or provide backend.
			RuntimeError: chip_name is not set; use run_auto or provide chip_name.
		"""
		if backend is None:
			raise RuntimeError("backend is not set; use run_auto or provide backend")
		if chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")
		if lengths is None:
			lengths = [1, 2, 4, 8, 16, 32]

		couplers = resolve_positive_fidelity_couplers(couplers, backend)
		use_simulator = str(chip_name).lower() == "simulator"
		rng = np.random.default_rng(seed)

		raw = self._load_rb_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_coupler = raw.get("per_coupler", {})
		now = datetime.now(timezone.utc)

		per_qubit_confusion: Optional[Dict[int, np.ndarray]] = None
		if readout_mitigation:
			# Reuse readout cache to mitigate measured probabilities for RB survival.
			readout_manager = ReadoutCalibrationManager(
				cache_dir=self._cache_dir,
				submit_circuit_async=self._submit_circuit_async,
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
			key = coupler_key(q1, q2)
			ts_str = timestamps.get(key)
			cached = per_coupler.get(key)
			if cache_is_fresh(ts_str, now=now) and cached is not None:
				results[key] = {"fit": {"fidelity": cached}}
				continue

			if print_true:
				logger.info("run native two-qubit RB on coupler %s", key)

			survival_samples: Dict[int, List[float]] = {length: [] for length in lengths}
			# Total gate count includes forward sequence plus explicit inverse sequence.
			total_length_by_length: Dict[int, int] = {}
			for length in lengths:
				for m in range(num_sequences):
					qc, total_length = self._build_random_sequence(
						[q1, q2],
						length,
						backend.two_qubit_gate_basis,
						rng,
					)
					total_length_by_length[length] = total_length
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
						survival_samples[length].append(float(probs[0]))
					else:
						task_id = self._submit_circuit_async(
							name=f"rb_2q_{key}_L{length}_batch{m}",
							circuit=qct,
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
			logger.info("Coupler %s: fidelity=%s", key, results[key]['fit']['fidelity'])

			# Cache stores fidelity only to keep payload minimal.
			self._save_rb_cache(results, chip_name=chip_name)
		return results

	def _build_random_sequence(
		self,
		qubits: List[int],
		length: int,
		basis_gate: str,
		rng: np.random.Generator,
	) -> Tuple[QuantumCircuit, int]:
		"""Build a random Clifford gate sequence with its inverse appended.

		Args:
			qubits (*List[int]*): Target qubit indices.
			length (*int*): Number of random Clifford layers.
			basis_gate (*str*): Native two-qubit gate name (e.g. ``'cz'``, ``'cx'``, ``'iswap'``).
			rng (*np.random.Generator*): NumPy random generator for sequence sampling.

		Returns:
			Tuple of ``(circuit, total_length)`` where *total_length* is the effective gate count.

		Raises:
			ValueError: unsupported two-qubit basis gate: {basis_gate}
		"""
		qc = QuantumCircuit(max(qubits) + 1)
		# Pauli-only single-qubit twirl for native two-qubit RB.
		single_gates = ["id", "x", "y", "z"]

		basis_gate = "cx" if basis_gate in {"cnot", "cx"} else basis_gate
		gates_list = []
		# Build forward sequence, then apply explicit inverse sequence.
		for l in range(length):
			g1 = rng.choice(single_gates)
			g2 = rng.choice(single_gates)
			self._apply_single_gate(qc, g1, qubits[0])
			self._apply_single_gate(qc, g2, qubits[1])
			self._apply_two_qubit_gate(qc, basis_gate, qubits[0], qubits[1])
			gates_list.append([g1, g2, basis_gate])
		for l in range(length):
			self._apply_two_qubit_gate_dg(qc, gates_list[length - 1 - l][2], qubits[0], qubits[1])	
			self._apply_single_gate_dg(qc, gates_list[length - 1 - l][0], qubits[0])
			self._apply_single_gate_dg(qc, gates_list[length - 1 - l][1], qubits[1])

		total_length_scale = {
			"iswap": 4,
			"ecr": 2,
			"cz": 2,
			"cx": 2,
			"cnot": 2,
		}.get(basis_gate)
		if total_length_scale is None:
			raise ValueError(f"unsupported two-qubit basis gate: {basis_gate}")
		total_length = total_length_scale * length
		return qc, total_length

	def _apply_single_gate_name(self, qc: QuantumCircuit, gate_name: str, qubit: int) -> None:
		"""Append a named single-qubit gate to the circuit, raising on unsupported names.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			gate_name (*str*): Name of the quantum gate.
			qubit (*int*): Target qubit index.

		Raises:
			ValueError: unsupported single-qubit gate: {gate_name}
		"""
		if gate_name == "id":
			return
		gate_method = getattr(qc, gate_name, None)
		if gate_method is None:
			raise ValueError(f"unsupported single-qubit gate: {gate_name}")
		gate_method(qubit)

	def _apply_single_gate(self, qc: QuantumCircuit, gate: str, qubit: int) -> None:
		"""Delegate a single-qubit gate specification to :meth:`_apply_single_gate_name`.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			gate (*str*): Gate specification or name.
			qubit (*int*): Target qubit index.
		"""
		self._apply_single_gate_name(qc, gate, qubit)

	def _apply_single_gate_dg(self, qc: QuantumCircuit, gate: str, qubit: int) -> None:
		"""Apply the dagger (inverse) of a single-qubit gate to the circuit.

		Looks up the inverse gate via ``_SINGLE_GATE_DAGGER`` and appends it.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			gate (*str*): Gate name whose dagger to apply.
			qubit (*int*): Target qubit index.

		Raises:
			ValueError: If *gate* has no known dagger mapping.
		"""
		dagger_gate = self._SINGLE_GATE_DAGGER.get(gate)
		if dagger_gate is None:
			raise ValueError(f"unsupported single-qubit gate: {gate}")
		self._apply_single_gate_name(qc, dagger_gate, qubit)

	def _canonical_two_qubit_gate(self, gate: str) -> str:
		"""Canonicalize a two-qubit gate name (e.g. ``'cnot'`` →``'cx'``).

		Args:
			gate (*str*): Gate name to canonicalize.

		Returns:
			Canonical gate name string.
		"""
		return "cx" if gate == "cnot" else gate

	def _apply_two_qubit_gate(self, qc: QuantumCircuit, gate: str, q1: int, q2: int) -> None:
		"""Append a supported two-qubit gate (CZ, CX, iSWAP, ECR) to the circuit.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			gate (*str*): Gate specification or name.
			q1 (*int*): First qubit index in the coupler pair.
			q2 (*int*): Second qubit index in the coupler pair.

		Raises:
			ValueError: unsupported two-qubit gate: {gate}
		"""
		canonical = self._canonical_two_qubit_gate(gate)
		if canonical not in {"cz", "cx", "iswap", "ecr"}:
			raise ValueError(f"unsupported two-qubit gate: {gate}")
		getattr(qc, canonical)(q1, q2)

	def _apply_two_qubit_gate_dg(self, qc: QuantumCircuit, gate: str, q1: int, q2: int) -> None:
		"""Apply the inverse (dagger) of a two-qubit gate.

		For self-inverse gates (CZ, CX, ECR) this is the gate itself.
		For iSWAP the inverse is implemented as iSWAP³ (since iSWAP⁴= I).

		Args:
			qc (*QuantumCircuit*): Quantum circuit to append the gate to.
			gate (*str*): Gate specification or name.
			q1 (*int*): First qubit index.
			q2 (*int*): Second qubit index.

		Raises:
			ValueError: unsupported two-qubit gate: {gate}
		"""
		canonical = self._canonical_two_qubit_gate(gate)
		if canonical == "iswap":
			qc.iswap(q1, q2)
			qc.iswap(q1, q2)
			qc.iswap(q1, q2)
			return
		if canonical in {"cz", "cx", "ecr"}:
			self._apply_two_qubit_gate(qc, canonical, q1, q2)
			return
		raise ValueError(f"unsupported two-qubit gate: {gate}")

	def _fit_decay(self, lengths: List[int], survival: List[float]) -> Dict[str, float | None]:
		"""Fit an exponential decay model y = A * p^x + B to RB survival data.

		Uses linear regression on log(survival - 1/dim) to extract the
		depolarizing parameter *p*, error per Clifford *epc*, and average
		fidelity.

		Args:
			lengths (*List[int]*): Sequence lengths (number of Cliffords).
			survival (*List[float]*): Measured survival probabilities at each length.

		Returns:
			Dict with keys ``'p'``, ``'epc'``, ``'fidelity'``, ``'A'``, ``'B'``.
			Values are ``None`` if the fit is under-determined.
		"""
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

	def _rb_cache_path(self, *, chip_name: Optional[str]) -> Path:
		"""Construct the filesystem path for two-qubit RB calibration cache.

		Args:
			chip_name (*Optional[str]*): Name of the target chip.

		Returns:
			``Path`` to the RB cache file.
		"""
		return cache_file(self._cache_dir, stem="rb_two_qubit", chip_name=chip_name)

	def _load_rb_cache_raw(self, *, chip_name: Optional[str]) -> Dict[str, object]:
		"""Load raw RB cache containing timestamps and per-coupler fidelity data.

		Args:
			chip_name (*Optional[str]*): Name of the target chip.

		Returns:
			Raw cache dictionary with ``"timestamps"`` and ``"per_coupler"`` keys.
		"""
		path = self._rb_cache_path(chip_name=chip_name)
		timestamps, per_coupler = load_timestamped_payload(path, payload_key="per_coupler")
		return {"timestamps": timestamps, "per_coupler": per_coupler}

	def _save_rb_cache(self, results: Dict[str, Dict[str, object]], *, chip_name: Optional[str]) -> None:
		"""Persist RB calibration results to cache with updated timestamps.

		Args:
			results (*Dict[str, Dict[str, object]]*): Collection of result objects.
			chip_name (*Optional[str]*): Name of the target chip.
		"""
		path = self._rb_cache_path(chip_name=chip_name)
		raw = self._load_rb_cache_raw(chip_name=chip_name)
		timestamps = raw.get("timestamps", {})
		per_coupler = raw.get("per_coupler", {})
		now = datetime.now(timezone.utc).isoformat()
		for key, payload in results.items():
			fit = payload.get("fit", {}) if isinstance(payload, dict) else {}
			fidelity = fit.get("fidelity") if isinstance(fit, dict) else None
			timestamps[key] = now
			per_coupler[key] = fidelity
		save_timestamped_payload(
			path,
			payload_key="per_coupler",
			timestamps=timestamps,
			payload=per_coupler,
		)
