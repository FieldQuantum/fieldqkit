"""High-level hardware client for circuit execution and algorithms."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy
from pathlib import Path

import numpy as np
from .task import Task
from .backend import Backend

from ..circuit import QuantumCircuit

from ..compile import Transpiler
from ..compile.translate import TranslateToBasisGates
from ..core.circuits import build_cluster, build_ghz, build_ising_time_evolution, build_qft
from ..algorithms.shadow import run_shadow_with_backend
from ..algorithms.vqe import (
	build_custom_hamiltonian,
	build_heisenberg_hamiltonian,
	build_ising_hamiltonian,
	build_xy_hamiltonian,
	build_xxz_hamiltonian,
	run_vqe_with_backend,
)
from ..algorithms.qaoa import build_custom_cost_hamiltonian, build_maxcut_hamiltonian, run_qaoa_with_backend
from ..core.observables import (
	append_measurement_basis,
	group_observables,
	pauli_expectation,
	pauli_support,
)
from .hardware import rank_chips
from ..core.readout import (
	apply_readout_mitigation_multi,
	expectation_from_probabilities,
	marginal_probabilities,
)
from ..calibration.readout import ReadoutCalibrationManager
from ..core.types import QAOAResult, RunResult, ShadowResult, VQEResult
from ..core.utils import get_probabilities, get_samples
from ..core.zne import apply_zne_cz_tripling, zne_linear_extrapolate
from ..sim.statevector import simulate_counts


# NOTE: API layer client. Keeps hardware selection + algorithm orchestration in one place.
class QuantumHardwareClient:
	def __init__(self):
		"""Create a hardware client."""
		self.chip_name = None
		# Task manager is the single entry point for submitting hardware jobs.
		self.tmgr = Task()
		self.chip_backend = None

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

	def transpile(self, qc: QuantumCircuit, target_qubits: Optional[Sequence[int]] = None):
		"""Transpile a circuit using the selected backend."""
		if self.chip_backend is None:
			raise RuntimeError("chip_backend is not set; use run_auto or provide chip_name")
		return self._transpile_with_backend(qc, self.chip_backend, target_qubits=target_qubits)

	def _transpile_with_backend(
		self,
		qc: QuantumCircuit,
		backend: Backend,
		target_qubits: Optional[Sequence[int]] = None,
	):
		"""Transpile with a specific backend and optional target qubits."""
		if target_qubits is None:
			return Transpiler(backend).run(qc)
		# When target qubits are provided, enable DD to reduce idle errors.
		return Transpiler(backend).run(qc, target_qubits=list(target_qubits), use_dd=True)

	def _submit_openqasm(
		self,
		name: str,
		qasm: str,
		shots: int,
		chip_name: Optional[str] = None,
	):
		"""Submit a blocking OpenQASM task and return its counts."""
		if chip_name is None and self.chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")
		# Submit tasks in raw QASM mode to keep transpilation under our control.
		task = {
			"chip": self.chip_name if chip_name is None else chip_name,
			"name": name,
			"circuit": qasm,
			"shots": shots,
			"compile": False,
		}
		task_id = self.tmgr.run(task)
		while True:
			status = self.tmgr.status(task_id)
			if status in {"Finished", "Failed", "Canceled"}:
				break
		res = self.tmgr.result(task_id)["count"]

		return task_id, res

	def _submit_openqasm_async(
		self,
		name: str,
		qasm: str,
		shots: int,
		chip_name: Optional[str] = None,
	):
		"""Submit an asynchronous OpenQASM task and return its task id."""
		if chip_name is None and self.chip_name is None:
			raise RuntimeError("chip_name is not set; use run_auto or provide chip_name")
		task = {
			"chip": self.chip_name if chip_name is None else chip_name,
			"name": name,
			"circuit": qasm,
			"shots": shots,
			"compile": False,
		}
		return self.tmgr.run(task)

	def _wait_task(self, task_id):
		"""Wait for a task to finish and return its final status."""
		while True:
			status = self.tmgr.status(task_id)
			if status in {"Finished", "Failed", "Canceled"}:
				return status

	def _compact_for_sim(self, qct: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[int, int]]:
		qct_sim = qct.deepcopy()
		used = qct_sim.qubits_in_use
		if not used:
			return qct_sim, {}
		# Remap sparse physical qubits to a dense 0..n-1 range for simulation.
		qct_sim.qubits = used
		mapping = {q: i for i, q in enumerate(used)}
		qct_sim.mapping_to_others(mapping)
		return qct_sim, mapping
	
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
		qasm_version: str = "2.0",
		print_true: bool = False,
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

		# Precompute which qubits each observable touches for fast marginalization.
		supports_by_obs = {obs: pauli_support(obs, num_qubits=num_qubits) for obs in observables}

		# Group observables by compatible measurement bases to reduce task count.
		if observables:
			groups = group_observables(observables, num_qubits=num_qubits)
		else:
			groups = [{"basis": None, "observables": []}]

		def _prepare_circuit(qc, basis_pattern: Optional[Sequence[str]], scale_zne: bool) -> QuantumCircuit:
			"""Prepare transpiled circuit with optional basis rotation and ZNE scaling."""
			qc = deepcopy(qc)
			if basis_pattern is not None:
				append_measurement_basis(qc, basis_pattern)
			elif not self._has_measurements(qc):
				# Ensure we can always collect counts/samples even without observables.
				qc.measure_all()
			# Transpile to the target backend/topology before execution.
			qct = self._transpile_with_backend(qc, backend, target_qubits=target_qubits)
			if scale_zne:
				# Insert CZ tripling after transpilation for ZNE.
				qct = apply_zne_cz_tripling(qct)
			return qct

		pending: List[Tuple[int, str, object]] = []
		group_meta: List[Dict[str, object]] = []
		task_ids: List[object] = []
		group_counts: Dict[int, Dict[str, Dict[str, int]]] = {i: {} for i in range(len(groups))}

		for gi, group in enumerate(groups):
			basis_pattern = group["basis"]
			qct = _prepare_circuit(qc, basis_pattern, scale_zne=False)
			if use_simulator:
				qct_sim, _ = self._compact_for_sim(qct)
				group_counts[gi]["1"] = simulate_counts(qct_sim, shots)
			else:
				# Hardware: submit async task and collect later.
				qasm_1 = qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3
				task_id_1 = self._submit_openqasm_async(
					name=f"{name}_g{gi}",
					qasm=qasm_1,
					shots=shots,
					chip_name=chip_name,
				)
				if print_true:
					print("[run] compile and run circuit:", f"{name}_g{gi}")
				pending.append((gi, "1", task_id_1))
				task_ids.append(task_id_1)

			if zne:
				qct = _prepare_circuit(qc, basis_pattern, scale_zne=True)
				if use_simulator:
					qct_sim, _ = self._compact_for_sim(qct)
					group_counts[gi]["3"] = simulate_counts(qct_sim, shots)
				else:
					# ZNE scale=3 path runs as an extra hardware task.
					task_id_3 = self._submit_openqasm_async(
						name=f"{name}_g{gi}_zne3",
						qasm=qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3,
						shots=shots,
						chip_name=chip_name,
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

		if target_qubits is None:
			target_qubits_group = qct.qubits_in_use
		else:
			target_qubits_group = target_qubits

		if print_true:
			print("[run] which qubits:", list(target_qubits_group) if target_qubits_group is not None else "auto")

		# Collect counts for each group (and ZNE scale if enabled).
		if not use_simulator:
			for gi, scale, task_id in pending:
				status = self._wait_task(task_id)
				if status != "Finished":
					raise RuntimeError(f"task {task_id} ended with status {status}")
				counts = self.tmgr.result(task_id)["count"]
				group_counts[gi][scale] = counts

		samples_list: List[List[List[int]] | None] = []
		probabilities_list: List[List[float] | None] = []
		probabilities_raw_list: List[List[float] | None] = []
		observable_values: Dict[str, float] = {}
		observable_values_raw: Dict[str, float] = {}

		for gi, meta in enumerate(group_meta):
			counts_1 = group_counts[gi]["1"]
			probs_1 = get_probabilities(counts_1, num_qubits)
			if zne:
				counts_3 = group_counts[gi]["3"]
				probs_3 = get_probabilities(counts_3, num_qubits)
				probs = zne_linear_extrapolate(probs_1, probs_3)
			else:
				probs = probs_1

			probs = np.clip(probs, 0.0, 1.0)
			s = probs.sum()
			if s > 0:
				probs = probs / s

			samples = get_samples(counts_1, num_qubits)
			raw_probabilities = probs if return_probabilities or readout_mitigation or meta["observables"] else None
			if raw_probabilities is not None:
				raw_probabilities = raw_probabilities.copy()
			probabilities = raw_probabilities.copy() if raw_probabilities is not None else None

			for obs in meta["observables"]:
				if obs in observable_values_raw:
					continue
				support = supports_by_obs.get(obs)
				if raw_probabilities is not None and support:
					# Marginalize to the observable support when probabilities are available.
					local_probs = marginal_probabilities(raw_probabilities, num_qubits, support)
					val = expectation_from_probabilities(local_probs, support)
				else:
					val = pauli_expectation(samples, obs)
				observable_values_raw[obs] = float(np.clip(val, -1.0, 1.0))

			if readout_mitigation:
				if len(target_qubits_group) != num_qubits:
					raise ValueError(
						f"num_qubits ({num_qubits}) must match len(target_qubits) ({len(target_qubits_group)}) for readout mitigation"
					)
				# Readout mitigation uses per-qubit confusion matrices.
				calibration_manager = ReadoutCalibrationManager(
					cache_dir=Path(__file__).resolve().parent / ".cache",
					submit_openqasm_async=self._submit_openqasm_async,
					wait_task=self._wait_task,
					get_task_result=self.tmgr.result,
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
				supports_subset = {obs: supports_by_obs[obs] for obs in meta["observables"]}
				probabilities, mitigated_obs_values = apply_readout_mitigation_multi(
					probabilities,
					probs,
					num_qubits,
					supports_subset,
					target_qubits_group,
					per_qubit,
				)
				for obs, val in mitigated_obs_values.items():
					observable_values[obs] = float(np.clip(val, -1.0, 1.0))

			for obs in meta["observables"]:
				if obs in observable_values:
					continue
				support = supports_by_obs[obs]
				if probabilities is not None and support:
					local_probs = marginal_probabilities(probabilities, num_qubits, support)
					val = expectation_from_probabilities(local_probs, support)
				else:
					val = pauli_expectation(samples, obs)
				observable_values[obs] = float(np.clip(val, -1.0, 1.0))

			if probabilities is not None:
				probabilities = np.clip(probabilities, 0.0, 1.0)
				s = probabilities.sum()
				if s > 0:
					probabilities = probabilities / s

			samples_list.append(samples.tolist() if isinstance(samples, np.ndarray) else samples)
			probabilities_list.append(probabilities.tolist() if probabilities is not None else None)
			probabilities_raw_list.append(raw_probabilities.tolist() if raw_probabilities is not None else None)

		observable_values_out = observable_values if observables else None
		if len(observables) == 1 and observable_values_out is not None:
			observable_values_out = observable_values_out.get(observables[0])

		observable_values_raw_out = None
		if readout_mitigation and observables:
			observable_values_raw_out = observable_values_raw
			if len(observables) == 1 and observable_values_raw_out is not None:
				observable_values_raw_out = observable_values_raw_out.get(observables[0])

		samples_out = samples_list[0] if len(samples_list) == 1 else samples_list
		probabilities_out = probabilities_list[0] if len(probabilities_list) == 1 else probabilities_list
		probabilities_raw_out = None
		if readout_mitigation:
			probabilities_raw_out = probabilities_raw_list[0] if len(probabilities_raw_list) == 1 else probabilities_raw_list

		if print_true:
			print("[finish] returning results")

		return RunResult(
			task_ids=[str(t) for t in task_ids] if task_ids else None,
			samples=samples_out,
			probabilities=probabilities_out,
			probabilities_raw=probabilities_raw_out,
			observable_values=observable_values_out,
			observable_values_raw=observable_values_raw_out,
		)

	def _run_with_backend_batch(
		self,
		circuits: Sequence[QuantumCircuit],
		name: str,
		num_qubits: int,
		*,
		backend: Backend,
		chip_name: str,
		shots: int | Sequence[int] = 1024,
		zne: bool = False,
		readout_mitigation: bool = False,
		readout_shots: Optional[int] = None,
		observables: Optional[Sequence[str] | Sequence[Sequence[str]] | str] = None,
		return_probabilities: bool = False,
		target_qubits: Optional[Sequence[int]] = None,
		qasm_version: str = "2.0",
		print_true: bool = False,
		post_transpile_transform: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
		reuse_transpiled_base: bool = False,
		basis_patterns: Optional[Sequence[Sequence[str]]] = None,
	) -> List[RunResult]:
		"""Run a batch of circuits, submitting all tasks before waiting.
		TODO: maybe merge with _run_with_backend to reduce code duplication?"""
		if not circuits:
			raise ValueError("circuits must be non-empty")
		for qc in circuits:
			if not isinstance(qc, QuantumCircuit):
				raise TypeError("circuits must contain QuantumCircuit instances")
		if basis_patterns is not None and len(basis_patterns) != len(circuits):
			raise ValueError("basis_patterns length must match circuits length")

		if isinstance(shots, (list, tuple, np.ndarray)):
			shots_list = list(shots)
			if len(shots_list) != len(circuits):
				raise ValueError("shots length must match circuits length")
		else:
			shots_list = [shots] * len(circuits)

		per_circuit_obs = None
		if observables is not None and not isinstance(observables, str):
			obs_seq = list(observables)
			if obs_seq and isinstance(obs_seq[0], (list, tuple)):
				if len(obs_seq) != len(circuits):
					raise ValueError("observables length must match circuits length")
				per_circuit_obs = obs_seq

		def _normalize_observables(local_observables):
			if isinstance(local_observables, str):
				return [local_observables]
			if local_observables is None:
				return []
			return list(local_observables)

		base_transpiled_cache: Dict[int, QuantumCircuit] = {}

		def _translate_to_basis(qct: QuantumCircuit) -> QuantumCircuit:
			translator = TranslateToBasisGates(
				convert_single_qubit_gate_to_u=True,
				two_qubit_gate_basis=backend.two_qubit_gate_basis,
			)
			return translator.run(qct)

		def _prepare_circuit(
			qc,
			basis_pattern: Optional[Sequence[str]],
			scale_zne: bool,
			base_qct: Optional[QuantumCircuit],
			target_qubits_in_use: Optional[Sequence[int]] = None,
		) -> QuantumCircuit:
			if base_qct is None:
				qc = deepcopy(qc)
				if basis_pattern is not None:
					append_measurement_basis(qc, basis_pattern)
				elif not self._has_measurements(qc):
					qc.measure_all()
				qct = self._transpile_with_backend(qc, backend, target_qubits=target_qubits)
			else:
				qct = base_qct.deepcopy()
				if basis_pattern is not None:
					append_measurement_basis(qct, basis_pattern, target_qubits=target_qubits_in_use)
				elif not self._has_measurements(qct):
					qct.measure(target_qubits_in_use, list(range(len(target_qubits_in_use))))
				if basis_pattern is not None or not self._has_measurements(qct):
					qct = _translate_to_basis(qct)
			if scale_zne:
				qct = apply_zne_cz_tripling(qct)
			if post_transpile_transform is not None:
				qct = post_transpile_transform(qct)
			return qct

		def _submit_one(qc, suffix, local_observables, local_shots, basis_pattern_override):
			observables_list = _normalize_observables(local_observables)
			if basis_pattern_override is not None and observables_list:
				raise ValueError("basis_patterns cannot be combined with observables")
			if print_true:
				print("[hardware] which hardware:", chip_name)
			use_simulator = str(chip_name).lower() == "simulator"
			supports_by_obs = {obs: pauli_support(obs, num_qubits=num_qubits) for obs in observables_list}
			if basis_pattern_override is not None:
				groups = [{"basis": basis_pattern_override, "observables": observables_list}]
			elif observables_list:
				groups = group_observables(observables_list, num_qubits=num_qubits)
			else:
				groups = [{"basis": None, "observables": []}]

			pending: List[Tuple[int, str, object]] = []
			group_meta: List[Dict[str, object]] = []
			task_ids: List[object] = []
			group_counts: Dict[int, Dict[str, Dict[str, int]]] = {i: {} for i in range(len(groups))}
			base_qct = None
			target_qubits_in_use = None
			if reuse_transpiled_base:
				cache_key = id(qc)
				base_qct = base_transpiled_cache.get(cache_key)
				if base_qct is None:
					base_qct = self._transpile_with_backend(deepcopy(qc), backend, target_qubits=target_qubits)
					base_transpiled_cache[cache_key] = base_qct
				target_qubits_in_use = base_qct.qubits_in_use

			for gi, group in enumerate(groups):
				basis_pattern = group["basis"]
				qct = _prepare_circuit(qc, basis_pattern, scale_zne=False, base_qct=base_qct, target_qubits_in_use=target_qubits_in_use)
				if use_simulator:
					qct_sim, _ = self._compact_for_sim(qct)
					group_counts[gi]["1"] = simulate_counts(qct_sim, local_shots)
				else:
					qasm_1 = qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3
					task_id_1 = self._submit_openqasm_async(
						name=f"{name}{suffix}_g{gi}",
						qasm=qasm_1,
						shots=local_shots,
						chip_name=chip_name,
					)
					if print_true:
						print("[run] compile and run circuit:", f"{name}{suffix}_g{gi}")
					pending.append((gi, "1", task_id_1))
					task_ids.append(task_id_1)

				if zne:
					qct = _prepare_circuit(qc, basis_pattern, scale_zne=True, base_qct=base_qct, target_qubits_in_use=target_qubits_in_use)
					if use_simulator:
						qct_sim, _ = self._compact_for_sim(qct)
						group_counts[gi]["3"] = simulate_counts(qct_sim, local_shots)
					else:
						task_id_3 = self._submit_openqasm_async(
							name=f"{name}{suffix}_g{gi}_zne3",
							qasm=qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3,
							shots=local_shots,
							chip_name=chip_name,
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

			if target_qubits is None:
				target_qubits_group = qct.qubits_in_use
			else:
				target_qubits_group = target_qubits

			if print_true:
				print("[run] which qubits:", list(target_qubits_group) if target_qubits_group is not None else "auto")

			return {
				"use_simulator": use_simulator,
				"pending": pending,
				"group_meta": group_meta,
				"task_ids": task_ids,
				"group_counts": group_counts,
				"supports_by_obs": supports_by_obs,
				"target_qubits_group": target_qubits_group,
				"observables": observables_list,
			}

		def _finalize_one(ctx):
			use_simulator = ctx["use_simulator"]
			pending = ctx["pending"]
			group_meta = ctx["group_meta"]
			task_ids = ctx["task_ids"]
			group_counts = ctx["group_counts"]
			supports_by_obs = ctx["supports_by_obs"]
			target_qubits_group = ctx["target_qubits_group"]
			observables_list = ctx["observables"]

			if not use_simulator:
				for gi, scale, task_id in pending:
					status = self._wait_task(task_id)
					if status != "Finished":
						raise RuntimeError(f"task {task_id} ended with status {status}")
					counts = self.tmgr.result(task_id)["count"]
					group_counts[gi][scale] = counts

			samples_list: List[List[List[int]] | None] = []
			probabilities_list: List[List[float] | None] = []
			probabilities_raw_list: List[List[float] | None] = []
			observable_values: Dict[str, float] = {}
			observable_values_raw: Dict[str, float] = {}

			for gi, meta in enumerate(group_meta):
				counts_1 = group_counts[gi]["1"]
				probs_1 = get_probabilities(counts_1, num_qubits)
				if zne:
					counts_3 = group_counts[gi]["3"]
					probs_3 = get_probabilities(counts_3, num_qubits)
					probs = zne_linear_extrapolate(probs_1, probs_3)
				else:
					probs = probs_1

				probs = np.clip(probs, 0.0, 1.0)
				s = probs.sum()
				if s > 0:
					probs = probs / s

				samples = get_samples(counts_1, num_qubits)
				raw_probabilities = probs if return_probabilities or readout_mitigation or meta["observables"] else None
				if raw_probabilities is not None:
					raw_probabilities = raw_probabilities.copy()
				probabilities = raw_probabilities.copy() if raw_probabilities is not None else None

				for obs in meta["observables"]:
					if obs in observable_values_raw:
						continue
					support = supports_by_obs.get(obs)
					if raw_probabilities is not None and support:
						local_probs = marginal_probabilities(raw_probabilities, num_qubits, support)
						val = expectation_from_probabilities(local_probs, support)
					else:
						val = pauli_expectation(samples, obs)
					observable_values_raw[obs] = float(np.clip(val, -1.0, 1.0))

				if readout_mitigation:
					if len(target_qubits_group) != num_qubits:
						raise ValueError(
							f"num_qubits ({num_qubits}) must match len(target_qubits) ({len(target_qubits_group)}) for readout mitigation"
						)
					calibration_manager = ReadoutCalibrationManager(
						cache_dir=Path(__file__).resolve().parent / ".cache",
						submit_openqasm_async=self._submit_openqasm_async,
						wait_task=self._wait_task,
						get_task_result=self.tmgr.result,
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
					supports_subset = {obs: supports_by_obs[obs] for obs in meta["observables"]}
					probabilities, mitigated_obs_values = apply_readout_mitigation_multi(
						probabilities,
						probs,
						num_qubits,
						supports_subset,
						target_qubits_group,
						per_qubit,
					)
					for obs, val in mitigated_obs_values.items():
						observable_values[obs] = float(np.clip(val, -1.0, 1.0))

				for obs in meta["observables"]:
					if obs in observable_values:
						continue
					support = supports_by_obs[obs]
					if probabilities is not None and support:
						local_probs = marginal_probabilities(probabilities, num_qubits, support)
						val = expectation_from_probabilities(local_probs, support)
					else:
						val = pauli_expectation(samples, obs)
					observable_values[obs] = float(np.clip(val, -1.0, 1.0))

				if probabilities is not None:
					probabilities = np.clip(probabilities, 0.0, 1.0)
					s = probabilities.sum()
					if s > 0:
						probabilities = probabilities / s

				samples_list.append(samples.tolist() if isinstance(samples, np.ndarray) else samples)
				probabilities_list.append(probabilities.tolist() if probabilities is not None else None)
				probabilities_raw_list.append(raw_probabilities.tolist() if raw_probabilities is not None else None)

			observable_values_out = observable_values if observables_list else None
			if len(observables_list) == 1 and observable_values_out is not None:
				observable_values_out = observable_values_out.get(observables_list[0])

			observable_values_raw_out = None
			if readout_mitigation and observables_list:
				observable_values_raw_out = observable_values_raw
				if len(observables_list) == 1 and observable_values_raw_out is not None:
					observable_values_raw_out = observable_values_raw_out.get(observables_list[0])

			samples_out = samples_list[0] if len(samples_list) == 1 else samples_list
			probabilities_out = probabilities_list[0] if len(probabilities_list) == 1 else probabilities_list
			probabilities_raw_out = None
			if readout_mitigation:
				probabilities_raw_out = probabilities_raw_list[0] if len(probabilities_raw_list) == 1 else probabilities_raw_list

			if print_true:
				print("[finish] returning results")

			return RunResult(
				task_ids=[str(t) for t in task_ids] if task_ids else None,
				samples=samples_out,
				probabilities=probabilities_out,
				probabilities_raw=probabilities_raw_out,
				observable_values=observable_values_out,
				observable_values_raw=observable_values_raw_out,
			)

		contexts = []
		for idx, qc in enumerate(circuits):
			obs = per_circuit_obs[idx] if per_circuit_obs is not None else observables
			basis_pattern_override = basis_patterns[idx] if basis_patterns is not None else None
			contexts.append(_submit_one(qc, f"_b{idx}", obs, shots_list[idx], basis_pattern_override))

		return [_finalize_one(ctx) for ctx in contexts]

	def run_auto(
		self,
		circuit: str,
		name: str,
		num_qubits: int,
		*,
		shots: int = 8192,
		zne: bool = False,
		readout_mitigation: bool = False,
		readout_shots: Optional[int] = None,
		observables: Optional[Sequence[str] | str] = None,
		return_probabilities: bool = False,
		target_qubits: Optional[Sequence[int]] = None,
		prefer_chips: Optional[Sequence[str] | str] = None,
		rank_weights: Optional[Dict[str, float]] = None,
		print_true: bool = True,
	) -> RunResult:
		"""Automatically select hardware, run, and return results."""
		print("[hardware] read hardware information and select")
		if self._is_openqasm2(circuit):
			qc = QuantumCircuit().from_openqasm2(openqasm2_str=circuit)
		elif self._is_openqasm3(circuit):
			qc = QuantumCircuit().from_openqasm3(openqasm3_str=circuit)
		else:
			qc = self.build_circuit(circuit, num_qubits=num_qubits)
		ranked_chips = rank_chips(
			self.tmgr,
			num_qubits=num_qubits,
			prefer_chips=prefer_chips,
			weights=rank_weights,
		)
		if not ranked_chips:
			raise RuntimeError("no available chips satisfy num_qubits requirement")

		last_error: Optional[Exception] = None
		for chip_name in ranked_chips:
			backend = Backend(chip_name)
			self.chip_name = chip_name
			self.chip_backend = backend
			return self._run_with_backend(
				qc,
				name,
				num_qubits,
				backend=backend,
				chip_name=chip_name,
				shots=shots,
				zne=zne,
				readout_mitigation=readout_mitigation,
				readout_shots=readout_shots,
				observables=observables,
				return_probabilities=return_probabilities,
				target_qubits=target_qubits,
				print_true=print_true,
			)

		raise RuntimeError("all candidate chips failed to transpile or run") from last_error

	def run_shadow(
		self,
		circuit: str,
		name: str,
		num_qubits: int,
		*,
		shots: int = 8192,
		observables: Optional[Sequence[str] | str] = None,
		zne: bool = False,
		estimator: str = "mean",
		mom_groups: Optional[int] = None,
		target_qubits: Optional[Sequence[int]] = None,
		prefer_chips: Optional[Sequence[str] | str] = None,
		rank_weights: Optional[Dict[str, float]] = None,
		seed: Optional[int] = None,
		batch_size: int = 1,
	) -> ShadowResult:
		"""Run classical shadow tomography on selected hardware."""
		print("[shadow] read hardware information and select")
		if self._is_openqasm2(circuit):
			qc = QuantumCircuit().from_openqasm2(openqasm2_str=circuit)
		elif self._is_openqasm3(circuit):
			qc = QuantumCircuit().from_openqasm3(openqasm3_str=circuit)
		else:
			qc = self.build_circuit(circuit, num_qubits=num_qubits)

		ranked_chips = rank_chips(
			self.tmgr,
			num_qubits=num_qubits,
			prefer_chips=prefer_chips,
			weights=rank_weights,
		)
		if not ranked_chips:
			raise RuntimeError("no available chips satisfy num_qubits requirement")

		if isinstance(observables, str):
			observables = [observables]

		last_error: Optional[Exception] = None
		for chip_name in ranked_chips:
			backend = Backend(chip_name)
			self.chip_name = chip_name
			self.chip_backend = backend
			try:
				return run_shadow_with_backend(
					self,
					qc,
					name=name,
					num_qubits=num_qubits,
					backend=backend,
					chip_name=chip_name,
					shots=shots,
					batch_size=batch_size,
					observables=observables,
					target_qubits=target_qubits,
					zne=zne,
					estimator=estimator,
					mom_groups=mom_groups,
					seed=seed,
				)
			except Exception as exc:
				last_error = exc
				continue

		raise RuntimeError("all candidate chips failed to run shadow tomography") from last_error

	def run_vqe(
		self,
		*,
		name: str,
		num_qubits: int,
		model: str = "ising",
		model_params: Optional[Dict[str, float]] = None,
		hamiltonian: Optional[Sequence[Tuple[float, str]]] = None,
		layers: int = 1,
		shots: int = 1024,
		max_iters: int = 20,
		learning_rate: float = 0.1,
		beta1: float = 0.9,
		beta2: float = 0.999,
		eps: float = 1e-8,
		shift: float = np.pi / 2.0,
		zne: bool = False,
		readout_mitigation: bool = False,
		target_qubits: Optional[Sequence[int]] = None,
		seed: Optional[int] = None,
		init_params: Optional[Sequence[float]] = None,
		callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
		prefer_chips: Optional[Sequence[str] | str] = None,
		rank_weights: Optional[Dict[str, float]] = None,
	) -> VQEResult:
		"""Run VQE using the selected hardware backend."""
		print(
			"[vqe] prepare run:",
			f"name={name}",
			f"num_qubits={num_qubits}",
			f"model={model}",
			f"layers={layers}",
			f"shots={shots}",
			f"max_iters={max_iters}",
		)
		model = model.lower()
		params = model_params or {}
		if model == "ising":
			hamiltonian = build_ising_hamiltonian(num_qubits, **params)
		elif model == "heisenberg":
			hamiltonian = build_heisenberg_hamiltonian(num_qubits, **params)
		elif model == "xxz":
			hamiltonian = build_xxz_hamiltonian(num_qubits, **params)
		elif model == "xy":
			hamiltonian = build_xy_hamiltonian(num_qubits, **params)
		elif model == "custom":
			if hamiltonian is None:
				raise ValueError("custom model requires hamiltonian")
			hamiltonian = build_custom_hamiltonian(hamiltonian, num_qubits)
		else:
			raise ValueError(f"unsupported model: {model}")

		# Select a chip by queue length, size, and error rate.
		ranked_chips = rank_chips(
			self.tmgr,
			num_qubits=num_qubits,
			prefer_chips=prefer_chips,
			weights=rank_weights,
		)
		print("[vqe] candidate chips:", ranked_chips)
		if not ranked_chips:
			raise RuntimeError("no available chips satisfy num_qubits requirement")

		last_error: Optional[Exception] = None
		for chip_name in ranked_chips:
			backend = Backend(chip_name)
			self.chip_name = chip_name
			self.chip_backend = backend
			try:
				print("[vqe] running on chip:", chip_name)
				return run_vqe_with_backend(
					self,
					name=name,
					num_qubits=num_qubits,
					backend=backend,
					chip_name=chip_name,
					hamiltonian=hamiltonian,
					layers=layers,
					shots=shots,
					max_iters=max_iters,
					learning_rate=learning_rate,
					beta1=beta1,
					beta2=beta2,
					eps=eps,
					shift=shift,
					zne=zne,
					readout_mitigation=readout_mitigation,
					target_qubits=target_qubits,
					seed=seed,
					init_params=init_params,
					callback=callback,
				)
			except Exception as exc:
				last_error = exc
				continue

		raise RuntimeError("all candidate chips failed to run VQE") from last_error

	def run_qaoa(
		self,
		*,
		name: str,
		num_qubits: int,
		problem: str = "maxcut",
		edges: Optional[Sequence[Tuple[int, int]]] = None,
		weights: Optional[Sequence[float]] = None,
		terms: Optional[Sequence[Tuple[float, str]]] = None,
		constant: float = 0.0,
		p: int = 1,
		shots: int = 1024,
		max_iters: int = 20,
		learning_rate: float = 0.1,
		beta1: float = 0.9,
		beta2: float = 0.999,
		eps: float = 1e-8,
		shift: float = np.pi / 2.0,
		zne: bool = False,
		readout_mitigation: bool = False,
		target_qubits: Optional[Sequence[int]] = None,
		seed: Optional[int] = None,
		init_params: Optional[Sequence[float]] = None,
		callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
		prefer_chips: Optional[Sequence[str] | str] = None,
		rank_weights: Optional[Dict[str, float]] = None,
	) -> QAOAResult:
		"""Run QAOA using the selected hardware backend."""
		problem = problem.lower()
		if problem == "maxcut":
			if edges is None:
				raise ValueError("maxcut problem requires edges")
			terms = None
			constant = 0.0
		elif problem == "custom":
			if terms is None:
				raise ValueError("custom problem requires terms")
			terms, constant = build_custom_cost_hamiltonian(terms, num_qubits, constant=constant)
		else:
			raise ValueError(f"unsupported problem: {problem}")

		ranked_chips = rank_chips(
			self.tmgr,
			num_qubits=num_qubits,
			prefer_chips=prefer_chips,
			weights=rank_weights,
		)
		if not ranked_chips:
			raise RuntimeError("no available chips satisfy num_qubits requirement")

		last_error: Optional[Exception] = None
		for chip_name in ranked_chips:
			backend = Backend(chip_name)
			self.chip_name = chip_name
			self.chip_backend = backend
			try:
				return run_qaoa_with_backend(
					self,
					name=name,
					num_qubits=num_qubits,
					backend=backend,
					chip_name=chip_name,
					edges=edges or [],
					weights=weights,
					terms=terms,
					constant=constant,
					p=p,
					shots=shots,
					max_iters=max_iters,
					learning_rate=learning_rate,
					beta1=beta1,
					beta2=beta2,
					eps=eps,
					shift=shift,
					zne=zne,
					readout_mitigation=readout_mitigation,
					target_qubits=target_qubits,
					seed=seed,
					init_params=init_params,
					callback=callback,
				)
			except Exception as exc:
				last_error = exc
				continue

		raise RuntimeError("all candidate chips failed to run QAOA") from last_error
