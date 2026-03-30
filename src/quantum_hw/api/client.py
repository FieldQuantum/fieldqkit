"""High-level hardware client for circuit execution and algorithms."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy
from pathlib import Path

import numpy as np
from .quantum_platform import create_provider_runtime
from .backend import Backend
from .task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter

from ..circuit import QuantumCircuit

from ..compile import Transpiler
from ..compile.translate import TranslateToBasisGates
from ..core.circuits import build_cluster, build_ghz, build_ising_time_evolution, build_qft
from ..core.observables import (
	append_measurement_basis,
	group_observables,
	pauli_expectation,
	pauli_basis_pattern,
	pauli_support,
)
from ..core.readout import (
	build_local_confusion_matrix,
	mitigate_observable_from_samples,
	mitigate_readout,
)
from ..calibration.readout import ReadoutCalibrationManager
from ..core.types import RunResult
from ..core.utils import get_probabilities_from_samples, get_samples
from ..core.zne import apply_zne_cz_tripling, zne_linear_extrapolate
from ..sim import simulate_counts

READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT = 10

# NOTE: API layer client. Keeps hardware selection + algorithm orchestration in one place.
class QuantumHardwareClient:
	def __init__(self):
		"""Create a hardware client."""
		self.chip_name = None
		self.chip_backend = None
		self._active_task_adapter: Optional[TaskAdapter] = None
		self._active_resolved_backend = None
		self._active_num_qubits: Optional[int] = None

	@staticmethod
	def _is_openqasm2(source: str) -> bool:
		"""Return True when the string looks like an OpenQASM2 program."""
		return source.strip().upper().startswith("OPENQASM 2.0")
	
	@staticmethod
	def _is_openqasm3(source: str) -> bool:
		"""Return True when the string looks like an OpenQASM3 program."""
		return source.strip().upper().startswith("OPENQASM 3.0")

	@staticmethod
	def _has_measurements(qc: QuantumCircuit) -> bool:
		"""Check whether the circuit already contains measurement operations."""
		return any(gate[0] == "measure" for gate in getattr(qc, "gates", []))

	@staticmethod
	def _infer_circuit_qubits(qc: QuantumCircuit) -> int:
		qubits = getattr(qc, "qubits", None) or []
		if qubits:
			return max(qubits) + 1
		return int(getattr(qc, "nqubits", 0) or 0)

	def _normalize_input_circuit(self, circuit: Union[str, QuantumCircuit], num_qubits: int) -> QuantumCircuit:
		"""Convert input into a QuantumCircuit and sanitize measurements."""
		if isinstance(circuit, QuantumCircuit):
			qc = circuit.deepcopy()
			# Measurements are appended later based on basis/targets.
			if self._has_measurements(qc):
				qc.remove_gate("measure")
			qc_qubits = self._infer_circuit_qubits(qc)
			if qc_qubits and qc_qubits != num_qubits:
				raise ValueError("num_qubits mismatch with QuantumCircuit")
			if qc_qubits == 0 and num_qubits > 0:
				qc.nqubits = num_qubits
				qc.ncbits = max(int(getattr(qc, "ncbits", 0) or 0), num_qubits)
			return qc
		if self._is_openqasm2(circuit):
			return QuantumCircuit().from_openqasm2(openqasm2_str=circuit)
		if self._is_openqasm3(circuit):
			return QuantumCircuit().from_openqasm3(openqasm3_str=circuit)
		return self.build_circuit(circuit, num_qubits=num_qubits)

	def build_circuit(self, kind: str, **kwargs) -> QuantumCircuit:
		"""Build a predefined circuit by name."""
		kind = kind.lower()
		if kind == "ghz":
			return build_ghz(**kwargs)
		if kind == "cluster":
			return build_cluster(**kwargs)
		if kind == "qft":
			return build_qft(**kwargs)
		if kind in {"ising", "ising_time_evolution", "ising_time"}:
			return build_ising_time_evolution(**kwargs)
		raise ValueError(f"unsupported circuit kind: {kind}")

	def _transpile_with_backend(
		self,
		qc: QuantumCircuit,
		backend: Backend,
		target_qubits: Optional[Sequence[int]] = None,
		use_dd: bool = True,
		use_three_qubit_decompose: bool = True,
		use_sabre_routing: bool = True,
		use_translate_to_basis: bool = True,
		use_gate_compressor: bool = True,
		noise_aware: bool | None = None,
		routing_n_trials: int = 1,
	) -> QuantumCircuit:
		"""Transpile with a specific backend and optional target qubits."""

		return Transpiler(backend).run(qc, target_qubits=list(target_qubits) if target_qubits is not None else None, use_dd=use_dd, use_three_qubit_decompose=use_three_qubit_decompose, use_sabre_routing=use_sabre_routing, use_translate_to_basis=use_translate_to_basis, use_gate_compressor=use_gate_compressor, noise_aware=noise_aware, routing_n_trials=routing_n_trials)

	def _submit_openqasm_async(
		self,
		name: str,
		qasm: str,
		shots: int,
		chip_name: Optional[str] = None,
		submit_options: Optional[Dict[str, object]] = None,
	):
		"""Submit an asynchronous OpenQASM task and return its task id."""
		resolved_chip_name = self._resolve_chip_name(chip_name)
		options = dict(submit_options or {})
		if "num_qubits" not in options and self._active_num_qubits is not None:
			options["num_qubits"] = self._active_num_qubits

		adapter = self._active_task_adapter
		backend = self._active_resolved_backend
		if adapter is None or backend is None:
			raise RuntimeError("active task adapter is required before submitting OpenQASM")

		handle = adapter.submit_openqasm(
			OpenQasmSubmitRequest(
				name=name,
				qasm=qasm,
				shots=shots,
				chip_name=resolved_chip_name,
				submit_options=options,
			),
			backend,
		)
		return handle

	def _resolve_chip_name(self, chip_name: Optional[str] = None) -> str:
		"""Resolve effective chip name from argument or client state."""
		if chip_name is not None:
			return chip_name
		if self.chip_name is not None:
			return self.chip_name
		raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")

	def _wait_task(self, task_id):
		"""Wait for a task to finish and return its final status."""
		adapter = self._active_task_adapter
		if adapter is None or not isinstance(task_id, ProviderTaskHandle):
			raise RuntimeError("active task adapter and ProviderTaskHandle are required when waiting task status")
		while True:
			status = adapter.query_status(task_id)
			if status in {"Finished", "Failed", "Canceled"}:
				return status

	def _get_task_result(self, task_id):
		"""Fetch normalized task result for current active adapter."""
		adapter = self._active_task_adapter
		if adapter is None or not isinstance(task_id, ProviderTaskHandle):
			raise RuntimeError("active task adapter and ProviderTaskHandle are required when fetching task result")
		return adapter.fetch_result(task_id)

	def _compact_for_sim(self, qct: QuantumCircuit, target_qubits: Optional[Sequence[int]] = None) -> Tuple[QuantumCircuit, Dict[int, int]]:
		qct_sim = qct.deepcopy()
		used = target_qubits if target_qubits is not None else qct_sim.qubits_in_use
		if not used:
			return qct_sim, {}
		# Remap sparse physical qubits to a dense 0..n-1 range for simulation.
		qct_sim.qubits = used
		mapping = {q: i for i, q in enumerate(used)}
		qct_sim.mapping_to_others(mapping)
		return qct_sim, mapping

	def _ordered_target_qubits_from_layout(
		self,
		*,
		compiled_qc: QuantumCircuit,
		original_qc: QuantumCircuit,
		num_qubits: int,
	) -> Optional[List[int]]:
		"""Recover measurement qubit order from transpiler layout mapping when available."""
		layout = getattr(compiled_qc, "logical_to_physical", None)
		if not isinstance(layout, dict) or not layout:
			return None

		logical_qubits = original_qc.qubits_in_use
		if not logical_qubits:
			logical_qubits = list(range(num_qubits))

		ordered: List[int] = []
		for lq in logical_qubits:
			pq = layout.get(lq)
			if not isinstance(pq, int):
				return None
			ordered.append(pq)

		if len(set(ordered)) != len(ordered):
			return None
		return ordered

	@staticmethod
	def _default_qasm_version_for_provider(provider: str) -> str:
		provider_name = str(provider).lower()
		if provider_name in {"tianyan", "guodun"}:
			return "3.0"
		return "2.0"
	
	def _run_with_backend(
		self,
		qc: QuantumCircuit,
		name: str,
		num_qubits: int,
		*,
		backend: Backend,
		chip_name: str,
		shots: int = 1024,
		zne: bool = False,
		readout_mitigation: bool = False,
		readout_shots: Optional[int] = None,
		observables: Optional[Sequence[str] | str] = None,
		return_probabilities: bool = False,
		target_qubits: Optional[Sequence[int]] = None,
		merge_groups: bool = True,
		qasm_version: str = "2.0",
		use_dd: bool = True,
		print_true: bool = False,
		transpile: bool = True,
		submit_options: Optional[Dict[str, object]] = None,
	) -> RunResult:
		"""Run a circuit on a specific backend with optional mitigation."""
		if isinstance(observables, str):
			observables = [observables]
		if observables is None:
			observables = []
		observables = list(observables)

		if print_true:
			print("[hardware] which hardware:", chip_name)

		use_simulator = str(chip_name).lower() == "simulator"

		# Precompute observable support for local expectations and mitigation.
		supports_by_obs = {obs: pauli_support(obs, num_qubits=num_qubits) for obs in observables}

		# Group observables by compatible measurement bases to reduce task count.
		if observables:
			if merge_groups:
				groups = group_observables(observables, num_qubits=num_qubits)
			else:
				groups = [
					{"basis": pauli_basis_pattern(obs, num_qubits=num_qubits), "observables": [obs]}
					for obs in observables
				]
		else:
			groups = [{"basis": None, "observables": []}]

		def _translate_to_basis(qct: QuantumCircuit) -> QuantumCircuit:
			translator = TranslateToBasisGates(
				convert_single_qubit_gate_to_u=True,
				two_qubit_gate_basis=backend.two_qubit_gate_basis,
			)
			return translator.run(qct)

		def _prepare_circuit(
			basis_pattern: Optional[Sequence[str]],
			scale_zne: bool,
			base_qct: QuantumCircuit,
			target_qubits_in_use: Optional[Sequence[int]] = None,
		) -> QuantumCircuit:
			"""Prepare transpiled circuit with optional basis rotation and ZNE scaling."""
			qct = base_qct.deepcopy()
			if basis_pattern is not None:
				append_measurement_basis(qct, basis_pattern, target_qubits=target_qubits_in_use)
			elif not self._has_measurements(qct):
				qct.measure(target_qubits_in_use, list(range(len(target_qubits_in_use))))
			if basis_pattern is not None or not self._has_measurements(qct):
				qct = _translate_to_basis(qct)
			if scale_zne:
				# Insert CZ tripling after transpilation for ZNE.
				qct = apply_zne_cz_tripling(qct)
			return qct

		pending: List[Tuple[int, str, object]] = []
		group_meta: List[Dict[str, object]] = []
		task_ids: List[object] = []
		group_counts: Dict[int, Dict[str, Dict[str, int]]] = {i: {} for i in range(len(groups))}
		# Transpile once and reuse across measurement groups.
		if transpile:
			base_qct = self._transpile_with_backend(
				deepcopy(qc),
				backend,
				target_qubits=target_qubits,
				use_dd=use_dd,
			)
			# Fall back to transpiler layout mapping first, then transpiled qubits/logical range.
			target_qubits_in_use = self._ordered_target_qubits_from_layout(
				compiled_qc=base_qct,
				original_qc=qc,
				num_qubits=num_qubits,
			)
		else:
			base_qct = deepcopy(qc)
			if target_qubits is not None:
				if len(target_qubits) != num_qubits:
					raise ValueError("target_qubits length mismatch with num_qubits")
				if not (set(base_qct.qubits_in_use) <= set(target_qubits)):
					raise ValueError("target_qubits must cover all qubits used in the circuit")
				target_qubits_in_use = list(target_qubits)
			else:
				used = list(base_qct.qubits)
				target_qubits_in_use = used if used else list(range(num_qubits))

		for gi, group in enumerate(groups):
			basis_pattern = group["basis"]
			qct = _prepare_circuit(
				basis_pattern,
				scale_zne=False,
				base_qct=base_qct,
				target_qubits_in_use=target_qubits_in_use,
			)
			if use_simulator:
				qct_sim, _ = self._compact_for_sim(qct, target_qubits_in_use)
				group_counts[gi]["1"] = simulate_counts(qct_sim, shots)
			else:
				# Hardware: submit async task and collect later.
				qasm_1 = qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3
				task_id_1 = self._submit_openqasm_async(
					name=f"{name}_g{gi}",
					qasm=qasm_1,
					shots=shots,
					chip_name=chip_name,
					submit_options=submit_options,
				)
				if print_true:
					print("[run] compile and run circuit:", f"{name}_g{gi}")
				pending.append((gi, "1", task_id_1))
				task_ids.append(task_id_1)

			if zne:
				qct = _prepare_circuit(
					basis_pattern,
					scale_zne=True,
					base_qct=base_qct,
					target_qubits_in_use=target_qubits_in_use,
				)
				if use_simulator:
					qct_sim, _ = self._compact_for_sim(qct, target_qubits_in_use)
					group_counts[gi]["3"] = simulate_counts(qct_sim, shots)
				else:
					# ZNE scale=3 path runs as an extra hardware task.
					task_id_3 = self._submit_openqasm_async(
						name=f"{name}_g{gi}_zne3",
						qasm=qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3,
						shots=shots,
						chip_name=chip_name,
						submit_options=submit_options,
					)
					if print_true:
						print("[run] run circuit:", "zero-noise extrapolation")
					pending.append((gi, "3", task_id_3))
					task_ids.append(task_id_3)

			meta = {
				"basis": basis_pattern,
				"observables": group["observables"],
				"qasm_1": qasm_1 if not use_simulator else None,
			}
			group_meta.append(meta)

		target_qubits_group = target_qubits_in_use.copy()
		
		per_qubit: Optional[Dict[int, np.ndarray]] = None
		if readout_mitigation:
			if len(target_qubits_group) != num_qubits:
				raise ValueError(
					f"num_qubits ({num_qubits}) must match len(target_qubits) ({len(target_qubits_group)}) for readout mitigation"
				)
			calibration_manager = ReadoutCalibrationManager(
				cache_dir=Path(__file__).resolve().parent / ".cache",
				submit_openqasm_async=self._submit_openqasm_async,
				wait_task=self._wait_task,
				get_task_result=self._get_task_result,
				compact_for_sim=self._compact_for_sim,
				simulate_counts=simulate_counts,
			)
			cal = calibration_manager.calibrate_readout(
				target_qubits=target_qubits_group,
				shots=readout_shots,
				chip_name=chip_name,
				backend=backend,
				qasm_version=qasm_version,
				print_true=print_true,
			)
			per_qubit = {k: np.asarray(v) for k, v in cal.per_qubit_confusion.items()}

		if print_true:
			print("[run] which qubits:", list(target_qubits_group) if target_qubits_group is not None else "auto")

		# Collect counts for each group (and ZNE scale if enabled).
		if not use_simulator:
			for gi, scale, task_id in pending:
				status = self._wait_task(task_id)
				if status != "Finished":
					raise RuntimeError(f"task {task_id} ended with status {status}")
				counts = self._get_task_result(task_id)["count"]
				group_counts[gi][scale] = counts

		samples_list: List[List[List[int]] | None] = []
		samples_zne_list: List[List[List[int]] | None] = []
		probabilities_list: List[List[float] | None] = []
		probabilities_raw_list: List[List[float] | None] = []
		observable_values: Dict[str, float] = {}
		observable_values_raw: Dict[str, float] = {}

		for gi, meta in enumerate(group_meta):
			counts_1 = group_counts[gi]["1"]
			samples_1 = get_samples(counts_1, num_qubits)
			samples_list.append(samples_1.tolist())
			if zne:
				counts_3 = group_counts[gi]["3"]
				samples_3 = get_samples(counts_3, num_qubits)
				samples_zne_list.append(samples_3.tolist())

			if return_probabilities:
				probs_1 = get_probabilities_from_samples(samples_1, num_qubits)
				probabilities_raw_list.append(probs_1.tolist())
				probabilities_list.append(probs_1.tolist())
				if zne:
					probs_3 = get_probabilities_from_samples(samples_3, num_qubits)
					probs_zne = zne_linear_extrapolate(probs_1, probs_3)
					probs_zne = np.clip(probs_zne, 0.0, 1.0)
					s = probs_zne.sum()
					probs_zne = probs_zne / s
					probabilities_list[-1] = probs_zne.tolist()

			for obs in meta["observables"]:
				val_1 = pauli_expectation(samples_1, obs)
				observable_values_raw[obs] = val_1
				observable_values[obs] = val_1
				if zne:
					val_3 = pauli_expectation(samples_3, obs)
					val_zne = zne_linear_extrapolate(val_1, val_3)
					val_zne = float(np.clip(val_zne, -1.0, 1.0))
					observable_values[obs] = val_zne

			if readout_mitigation and per_qubit is not None:
				if return_probabilities:
					full_cm = build_local_confusion_matrix(per_qubit, target_qubits_group)
					probs_1_rem = mitigate_readout(probs_1, full_cm)
					probabilities_list[-1] = probs_1_rem.tolist()
					if zne:
						probs_3_rem = mitigate_readout(probs_3, full_cm)
						probs_rem_zne = zne_linear_extrapolate(probs_1_rem, probs_3_rem) # REM first, then ZNE
						probs_rem_zne = np.clip(probs_rem_zne, 0.0, 1.0)
						s = probs_rem_zne.sum()
						probs_rem_zne = probs_rem_zne / s
						probabilities_list[-1] = probs_rem_zne.tolist()
				for obs in meta["observables"]:
					support = supports_by_obs[obs]
					if support:
						val_1_rem = mitigate_observable_from_samples(
							samples_1,
							support,
							per_qubit,
							target_qubits_group,
							marginal_max_support=READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT,
						)
						observable_values[obs] = val_1_rem
						if zne:
							val_3_rem = mitigate_observable_from_samples(
								samples_3,
								support,
								per_qubit,
								target_qubits_group,
								marginal_max_support=READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT,
							)
							val_rem_zne = float(zne_linear_extrapolate(val_1_rem, val_3_rem))
							observable_values[obs] = val_rem_zne

		if print_true:
			print("[finish] returning results")

		return RunResult(
			task_ids=[str(t) for t in task_ids] if task_ids else None,
			samples=samples_list,
			samples_zne=samples_zne_list if zne else None,
			probabilities=probabilities_list,
			probabilities_raw=probabilities_raw_list,
			observable_values=observable_values,
			observable_values_raw=observable_values_raw,
		)

	def run_auto(
		self,
		circuit: Union[str, QuantumCircuit],
		name: str,
		num_qubits: int,
		*,
		provider: str = "quafu",
		shots: int = 8192,
		zne: bool = False,
		readout_mitigation: bool = False,
		readout_shots: Optional[int] = None,
		observables: Optional[Sequence[str] | str] = None,
		return_probabilities: bool = False,
		target_qubits: Optional[Sequence[int]] = None,
		prefer_chips: Optional[Sequence[str] | str] = None,
		transpile_on_client: bool = True,
		max_wait_time: int = 3600,
		sleep_time: int = 5,
		print_true: bool = True,
	) -> RunResult:
		"""Automatically select hardware, run, and return results."""
		# Normalize input circuit and strip measurements if present.
		qc = self._normalize_input_circuit(circuit, num_qubits)
		provider = str(provider).lower()
		qasm_version = self._default_qasm_version_for_provider(provider)
		use_dd = provider not in {"tianyan", "guodun"}

		runtime = create_provider_runtime(provider=provider, client=self)

		resolved_backend = runtime.backend_adapter.resolve_backend(
			num_qubits=num_qubits,
			prefer_hardware=prefer_chips,
		)

		self.chip_name = resolved_backend.hardware_name
		self.chip_backend = resolved_backend.backend

		def _as_int(value, default):
			try:
				return int(value)
			except Exception:
				return int(default)

		submit_options = {
			"transpile_on_client": bool(transpile_on_client),
			"max_wait_time": _as_int(max_wait_time, 3600),
			"sleep_time": _as_int(sleep_time, 5),
		}
		self._active_task_adapter = runtime.task_adapter
		self._active_resolved_backend = resolved_backend
		self._active_num_qubits = num_qubits
		try:
			return self._run_with_backend(
				qc,
				name,
				num_qubits,
				backend=resolved_backend.backend,
				chip_name=resolved_backend.hardware_name,
				shots=shots,
				zne=zne,
				readout_mitigation=readout_mitigation,
				readout_shots=readout_shots,
				observables=observables,
				return_probabilities=return_probabilities,
				target_qubits=target_qubits,
				qasm_version=qasm_version,
				use_dd=use_dd,
				print_true=print_true,
				transpile=bool(submit_options["transpile_on_client"]),
				submit_options=submit_options,
			)
		finally:
			self._active_task_adapter = None
			self._active_resolved_backend = None
			self._active_num_qubits = None

