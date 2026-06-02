"""High-level hardware client for circuit execution and algorithms."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)
from copy import deepcopy
from pathlib import Path

import numpy as np
from .quantum_platform import create_provider_runtime
from .backend import Backend, resolve_provider
from .task import OpenQasmSubmitRequest, QcisSubmitRequest, ProviderTaskHandle, TaskAdapter

from ..circuit import QuantumCircuit

from ..compile import Transpiler
from ..compile.translate import TranslateToBasisGates
from ..core.circuits import (
	build_cluster,
	build_ghz,
	build_heisenberg_time_evolution,
	build_ising_time_evolution,
	build_qft,
	build_xxz_time_evolution,
	build_xy_time_evolution,
)
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


def simulate_counts(*args, **kwargs):
	"""Lazy proxy to :func:`fieldqkit.sim.simulate_counts`.

	The local simulator requires PyTorch (the optional ``[sim]`` extra), so the
	import is deferred to call time.  This keeps ``import fieldqkit`` and
	hardware-only workflows working without PyTorch installed.
	"""
	from ..sim import simulate_counts as _simulate_counts
	return _simulate_counts(*args, **kwargs)


READOUT_OBSERVABLE_MARGINAL_MAX_SUPPORT = 10

# NOTE: API layer client. Keeps hardware selection + algorithm orchestration in one place.
class QuantumHardwareClient:
	def __init__(self):
		"""Create a hardware client.
		"""
		self.chip_name = None
		self.chip_backend = None
		self._active_task_adapter: Optional[TaskAdapter] = None
		self._active_resolved_backend = None
		# For sequential task submission (tianyan provider does not support batch submission)
		self._last_pending_task_id: Optional[object] = None

	@staticmethod
	def _is_openqasm2(source: str) -> bool:
		"""Return True when the string looks like an OpenQASM2 program.

		Args:
			source (*str*): OpenQASM source string.

		Returns:
			``True`` if the condition is satisfied.
		"""
		return source.strip().upper().startswith("OPENQASM 2.0")
	
	@staticmethod
	def _has_measurements(qc: QuantumCircuit) -> bool:
		"""Check whether the circuit already contains measurement operations.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.

		Returns:
			``True`` if the condition is satisfied.
		"""
		return any(gate[0] == "measure" for gate in getattr(qc, "gates", []))

	def _normalize_input_circuit(
		self,
		circuit: Union[str, QuantumCircuit],
		num_qubits: int,
		observables: Optional[Sequence[str]] = None,
	) -> QuantumCircuit:
		"""Convert input into a QuantumCircuit and sanitize measurements.

		When *observables* are provided and the circuit already contains
		measurements, a warning is emitted and the existing measurements are
		removed so that the correct measurement bases can be appended later.
		When no observables are given, existing measurements are preserved.

		Args:
			circuit (*Union[str, QuantumCircuit]*): Quantum circuit to execute.
			num_qubits (*int*): Number of qubits.
			observables (*Optional[Sequence[str]]*): Observable operators. Defaults to ``None``.

		Returns:
			Constructed ``QuantumCircuit``.

		Raises:
			ValueError: num_qubits mismatch with QuantumCircuit
		"""
		if isinstance(circuit, QuantumCircuit):
			qc = circuit.deepcopy()
		elif self._is_openqasm2(circuit):
			qc = QuantumCircuit().from_openqasm2(openqasm2_str=circuit)
		else:
			qc = self.build_circuit(circuit, num_qubits=num_qubits)
		has_obs = observables is not None and len(observables) > 0
		if self._has_measurements(qc):
			if has_obs:
				logger.warning(
					"Circuit contains measurement gates that conflict with the "
					"provided observables. The existing measurements will be "
					"removed and replaced by the observable-derived bases."
				)
				qc.remove_gate("measure")
			# else: keep user-specified measurements as-is
		qc_qubits = int(getattr(qc, "nqubits", 0) or 0)
		if qc_qubits and qc_qubits != num_qubits:
			raise ValueError("num_qubits mismatch with QuantumCircuit")
		if qc_qubits == 0 and num_qubits > 0:
			qc.nqubits = num_qubits
			qc.ncbits = max(int(getattr(qc, "ncbits", 0) or 0), num_qubits)
		return qc

	def build_circuit(self, kind: str, **kwargs) -> QuantumCircuit:
		"""Build a predefined circuit by name.

		Args:
			kind (*str*): Circuit type. One of ``"ghz"``, ``"cluster"``, ``"qft"``,
				``"ising"`` (transverse-field Ising Trotter evolution),
				``"heisenberg"``, ``"xxz"``, ``"xy"``.
			**kwargs: Circuit-specific keyword arguments.

		Returns:
			Constructed ``QuantumCircuit``.

		Raises:
			ValueError: f'unsupported circuit kind: {kind}'
		"""
		kind = kind.lower()
		if kind == "ghz":
			return build_ghz(**kwargs)
		if kind == "cluster":
			return build_cluster(**kwargs)
		if kind == "qft":
			return build_qft(**kwargs)
		if kind in {"ising", "ising_time_evolution", "ising_time"}:
			return build_ising_time_evolution(**kwargs)
		if kind in {"heisenberg", "heisenberg_time_evolution"}:
			return build_heisenberg_time_evolution(**kwargs)
		if kind in {"xxz", "xxz_time_evolution"}:
			return build_xxz_time_evolution(**kwargs)
		if kind in {"xy", "xy_time_evolution"}:
			return build_xy_time_evolution(**kwargs)
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
		convert_single_qubit_gate_to_u: bool | None = None,
	) -> QuantumCircuit:
		"""Transpile with a specific backend and optional target qubits.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			backend (*Backend*): Hardware backend descriptor.
			target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement. Defaults to ``None``.
			use_dd (*bool*): Whether to insert dynamical decoupling pulses. Defaults to ``True``.
			use_three_qubit_decompose (*bool*): Whether to decompose three-qubit gates into native gates. Defaults to ``True``.
			use_sabre_routing (*bool*): Whether to use SABRE for qubit mapping and routing. Defaults to ``True``.
			use_translate_to_basis (*bool*): Whether to convert gates to the hardware's native basis set. Defaults to ``True``.
			use_gate_compressor (*bool*): Whether to merge consecutive single-qubit gates. Defaults to ``True``.
			noise_aware (*bool | None*): Whether to use noise-aware strategies. Defaults to ``None``.
			routing_n_trials (*int*): Number of independent routing attempts for optimal layout. Defaults to ``1``.
			convert_single_qubit_gate_to_u (*bool | None*): Whether to convert single-qubit gates to U gates. Defaults to ``None``.

		Returns:
			Constructed ``QuantumCircuit``.
		"""
		return Transpiler(backend, convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u).run(qc, target_qubits=list(target_qubits) if target_qubits is not None else None, use_dd=use_dd, use_three_qubit_decompose=use_three_qubit_decompose, use_sabre_routing=use_sabre_routing, use_translate_to_basis=use_translate_to_basis, use_gate_compressor=use_gate_compressor, noise_aware=noise_aware, routing_n_trials=routing_n_trials)

	def _submit_openqasm_async(
		self,
		name: str,
		qasm: str,
		shots: int,
		chip_name: Optional[str] = None,
		submit_options: Optional[Dict[str, object]] = None,
	):
		"""Submit an asynchronous OpenQASM task and return its task id.

		Args:
			name (*str*): Experiment name or job label.
			qasm (*str*): OpenQASM circuit string.
			shots (*int*): Number of measurement shots.
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.
			submit_options (*Optional[Dict[str, object]]*): Extra provider-specific submission options. Defaults to ``None``.

		Returns:
			``ProviderTaskHandle`` for tracking the submitted task.

		Raises:
			RuntimeError: active task adapter is required before submitting OpenQASM
		"""
		resolved_chip_name = self._resolve_chip_name(chip_name)
		options = dict(submit_options or {})

		# Add timestamp suffix to avoid task name collision
		timestamp = int(time.time() * 1000)
		task_name = f"{name}_{timestamp}"

		adapter = self._active_task_adapter
		backend = self._active_resolved_backend
		if adapter is None or backend is None:
			raise RuntimeError(
				"active task adapter is required before submitting OpenQASM; "
				"call run_auto() or _run_with_backend() first to provision a runtime"
			)

		handle = adapter.submit_openqasm(
			OpenQasmSubmitRequest(
				name=task_name,
				qasm=qasm,
				shots=shots,
				chip_name=resolved_chip_name,
				submit_options=options,
			),
			backend,
		)
		return handle

	def _resolve_chip_name(self, chip_name: Optional[str] = None) -> str:
		"""Resolve effective chip name from argument or client state.

		Args:
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.

		Returns:
			Resolved chip name string.

		Raises:
			RuntimeError: chip_name is not set; call run_auto() or _run_with_backend() first, or pass chip_name explicitly
		"""
		if chip_name is not None:
			return chip_name
		if self.chip_name is not None:
			return self.chip_name
		raise RuntimeError(
			"chip_name is not set; call run_auto() or _run_with_backend() first, "
			"or pass chip_name explicitly"
		)

	def _submit_circuit_async(
		self,
		name: str,
		circuit,
		shots: int,
		chip_name: Optional[str] = None,
		submit_options: Optional[Dict[str, object]] = None,
	):
		"""Submit a circuit and return its task handle, dispatching to QCIS or OpenQASM 2.0.

		Args:
			name (*str*): Experiment name or job label.
			circuit (*QuantumCircuit*): Quantum circuit to submit.
			shots (*int*): Number of measurement shots.
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.
			submit_options (*Optional[Dict[str, object]]*): Extra provider-specific submission options. Defaults to ``None``.

		Returns:
			``ProviderTaskHandle`` for tracking the submitted task.
		"""
		adapter = self._active_task_adapter
		if adapter is not None and bool(getattr(adapter, "qcis_native", False)):
			from ..circuit.qcis import circuit_to_qcis
			return self._submit_qcis_async(
				name=name,
				qcis=circuit_to_qcis(circuit),
				shots=shots,
				chip_name=chip_name,
				submit_options=submit_options,
			)
		return self._submit_openqasm_async(
			name=name,
			qasm=circuit.to_openqasm2(),
			shots=shots,
			chip_name=chip_name,
			submit_options=submit_options,
		)

	def _needs_sequential_submission(self) -> bool:
		"""Check if the current provider requires sequential task submission.

		TianYan provider does not support batch submission, so tasks must be
		submitted one at a time, waiting for each to complete before submitting the next.

		Returns:
			``True`` if sequential submission is required.
		"""
		resolved = self._active_resolved_backend
		if resolved is None:
			return False
		provider = getattr(resolved, "provider", None)
		return provider is not None and provider.lower() == "tianyan"

	def _submit_qcis_async(
		self,
		name: str,
		qcis: str,
		shots: int,
		chip_name: Optional[str] = None,
		submit_options: Optional[Dict[str, object]] = None,
	):
		"""Submit an asynchronous QCIS task and return its task handle.

		Args:
			name (*str*): Experiment name or job label.
			qcis (*str*): QCIS instruction string.
			shots (*int*): Number of measurement shots.
			chip_name (*Optional[str]*): Name of the target chip. Defaults to ``None``.
			submit_options (*Optional[Dict[str, object]]*): Extra provider-specific submission options. Defaults to ``None``.

		Returns:
			``ProviderTaskHandle`` for tracking the submitted task.

		Raises:
			RuntimeError: active task adapter is required before submitting QCIS
		"""
		resolved_chip_name = self._resolve_chip_name(chip_name)
		options = dict(submit_options or {})

		timestamp = int(time.time() * 1000)
		task_name = f"{name}_{timestamp}"

		adapter = self._active_task_adapter
		backend = self._active_resolved_backend
		if adapter is None or backend is None:
			raise RuntimeError(
				"active task adapter is required before submitting QCIS; "
				"call run_auto() or _run_with_backend() first to provision a runtime"
			)

		handle = adapter.submit_qcis(
			QcisSubmitRequest(
				name=task_name,
				qcis=qcis,
				shots=shots,
				chip_name=resolved_chip_name,
				submit_options=options,
			),
			backend,
		)
		return handle

	def _wait_for_last_task(self) -> None:
		"""Wait for the previously submitted task to complete.

		This is used for sequential submission mode (tianyan provider).
		Does nothing if no task was previously submitted.
		"""
		import time
		if self._last_pending_task_id is not None:
			status = self._wait_task(self._last_pending_task_id)
			if status != "Finished":
				raise RuntimeError(f"previous task {self._last_pending_task_id} ended with status {status}")
			self._last_pending_task_id = None

	def _wait_task(self, task_id):
		"""Wait for a task to finish and return its final status.

		Args:
			task_id: ``ProviderTaskHandle`` returned by ``_submit_openqasm_async``.

		Returns:
			``str`` final status (``"Finished"``, ``"Failed"``, or ``"Canceled"``).

		Raises:
			RuntimeError: If active task adapter and ``ProviderTaskHandle`` are not set.
		"""
		import time
		adapter = self._active_task_adapter
		if adapter is None or not isinstance(task_id, ProviderTaskHandle):
			raise RuntimeError("active task adapter and ProviderTaskHandle are required when waiting task status")
		while True:
			status = adapter.query_status(task_id)
			if status in {"Finished", "Failed", "Canceled"}:
				return status
			time.sleep(3)

	def _get_task_result(self, task_id):
		"""Fetch normalized task result for current active adapter.

		Args:
			task_id: ``ProviderTaskHandle`` returned by ``_submit_openqasm_async``.

		Returns:
			Provider-specific task result from the active adapter.

		Raises:
			RuntimeError: If active task adapter and ``ProviderTaskHandle`` are not set.
		"""
		adapter = self._active_task_adapter
		if adapter is None or not isinstance(task_id, ProviderTaskHandle):
			raise RuntimeError("active task adapter and ProviderTaskHandle are required when fetching task result")
		return adapter.fetch_result(task_id)

	def _compact_for_sim(self, qct: QuantumCircuit, target_qubits: Optional[Sequence[int]] = None) -> Tuple[QuantumCircuit, Dict[int, int]]:
		"""Map sparse physical qubits to a dense 0..n-1 range for simulation.

		Args:
			qct (*QuantumCircuit*): Transpiled circuit with sparse qubit indices.
			target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement. Defaults to ``None``.

		Returns:
			Tuple of (compacted circuit, physical-to-dense index mapping).
		"""
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
	) -> List[int]:
		"""Recover measurement qubit order from transpiler layout mapping.

		Uses ``compiled_qc.logical_to_physical`` when available.  If the
		layout is missing (e.g. ``use_sabre_routing=False``), falls back to
		``compiled_qc.qubits`` (physical qubit list from Layout), then to
		``range(num_qubits)``.

		Args:
			compiled_qc (*QuantumCircuit*): Transpiled circuit with physical qubit layout applied.
			original_qc (*QuantumCircuit*): Original logical circuit before compilation.
			num_qubits (*int*): Number of qubits.

		Returns:
			Ordered list of physical qubit indices (never ``None``).
		"""
		layout = getattr(compiled_qc, "logical_to_physical", None)
		if isinstance(layout, dict) and layout:
			logical_qubits = original_qc.qubits
			if not logical_qubits:
				logical_qubits = list(range(num_qubits))

			ordered: List[int] = []
			for lq in logical_qubits:
				pq = layout.get(lq)
				if not isinstance(pq, int):
					break
				ordered.append(pq)
			else:
				if len(set(ordered)) == len(ordered):
					return ordered

		# Fallback: use transpiled circuit's qubit list (Layout order).
		used = list(getattr(compiled_qc, "qubits", None) or [])
		if used and len(used) >= num_qubits:
			return used[:num_qubits]
		return list(range(num_qubits))

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
		convert_single_qubit_gate_to_u: bool = True,
	) -> RunResult:
		"""Run a circuit on a specific backend with optional mitigation.

		Args:
			qc (*QuantumCircuit*): Quantum circuit.
			name (*str*): Experiment name for the submission.
			num_qubits (*int*): Number of qubits.
			backend (*Backend*): Hardware backend descriptor.
			chip_name (*str*): Name of the target chip.
			shots (*int*): Number of measurement shots. Defaults to ``1024``.
			zne (*bool*): Whether to apply zero-noise extrapolation. Defaults to ``False``.
			readout_mitigation (*bool*): Whether to apply readout error mitigation. Defaults to ``False``.
			readout_shots (*Optional[int]*): Number of shots for readout calibration. Defaults to ``None``.
			observables (*Optional[Sequence[str] | str]*): Observable operators to measure. Defaults to ``None``.
			return_probabilities (*bool*): Whether to return probability distributions. Defaults to ``False``.
			target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement. Defaults to ``None``.
			merge_groups (*bool*): Whether to batch observables by compatible measurement bases. Defaults to ``True``.
			qasm_version (*str*): OpenQASM version. Only ``'2.0'`` is supported; passing any other value raises ``ValueError``. Defaults to ``'2.0'``.
			use_dd (*bool*): Whether to insert dynamical decoupling sequences. Defaults to ``True``.
			print_true (*bool*): Whether to print progress information. Defaults to ``False``.
			transpile (*bool*): Whether to transpile the circuit for hardware. Defaults to ``True``.
			submit_options (*Optional[Dict[str, object]]*): Extra provider-specific submission options. Defaults to ``None``.
			convert_single_qubit_gate_to_u (*bool*): Whether to convert single-qubit gates to U gates. Defaults to ``True``.

		Returns:
			``RunResult`` containing counts, expectations, and metadata.

		Raises:
			ValueError: If *num_qubits* doesn't match *target_qubits* length.
			ValueError: If *qasm_version* is not ``'2.0'``.
			RuntimeError: If the submitted task ends with a non-success status.
		"""
		if qasm_version != "2.0":
			raise ValueError(f"Only OpenQASM 2.0 is supported; got qasm_version={qasm_version!r}")
		if isinstance(observables, str):
			observables = [observables]
		if observables is None:
			observables = []
		observables = list(observables)

		from .backend import is_noisy_circuit_for_backend
		noisy_circuit = is_noisy_circuit_for_backend(qc, chip_name)
		if noisy_circuit:
			transpile = False

		if print_true:
			logger.info("which hardware: %s", chip_name)

		use_simulator = str(chip_name).lower() == "simulator"

		# Auto-provision runtime when the caller skipped the high-level entry
		# points (run_auto / VQERunner / …) and jumped straight here.
		if not use_simulator and self._active_task_adapter is None:
			from .backend import infer_provider_from_chip, ResolvedBackend
			inferred = infer_provider_from_chip(chip_name)
			if inferred is None:
				raise RuntimeError(
					f"Cannot infer provider for chip '{chip_name}'. "
					"Use run_auto() or set up runtime manually."
				)
			runtime = create_provider_runtime(provider=inferred, client=self)
			# Reuse the caller-supplied Backend for transpilation.
			# Only need platform_obj from the adapter (for task submission).
			platform_obj = getattr(runtime.backend_adapter, '_platform', None)
			if platform_obj is not None and hasattr(platform_obj, 'set_machine'):
				platform_obj.set_machine(chip_name)
			resolved = ResolvedBackend(
				provider=inferred,
				hardware_name=chip_name,
				backend=backend,
				metadata={"platform_obj": platform_obj} if platform_obj is not None else {},
			)
			self.chip_name = chip_name
			self.chip_backend = backend
			self._active_task_adapter = runtime.task_adapter
			self._active_resolved_backend = resolved
			logger.info("auto-provisioned runtime for provider=%s chip=%s", inferred, chip_name)

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
			"""Translate gates to the hardware's native basis gate set.

			Args:
				qct (*QuantumCircuit*): Circuit to translate.

			Returns:
				Translated ``QuantumCircuit``.
			"""
			translator = TranslateToBasisGates(
				convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
				two_qubit_gate_basis=backend.two_qubit_gate_basis,
			)
			return translator.run(qct)

		def _prepare_circuit(
			basis_pattern: Optional[Sequence[str]],
			scale_zne: bool,
			base_qct: QuantumCircuit,
			target_qubits_in_use: Optional[Sequence[int]] = None,
		) -> QuantumCircuit:
			"""Prepare transpiled circuit with optional basis rotation and ZNE scaling.

			Args:
				basis_pattern (*Optional[Sequence[str]]*): Measurement basis rotations (e.g. ``['X', 'Y', 'Z']``), or ``None`` for computational basis.
				scale_zne (*bool*): Whether to apply ZNE CZ-tripling after transpilation.
				base_qct (*QuantumCircuit*): Transpiled circuit template to prepare variants from.
				target_qubits_in_use (*Optional[Sequence[int]]*): Physical qubits to measure, or ``None`` to measure all. Defaults to ``None``.

			Returns:
				Constructed ``QuantumCircuit``.
			"""
			qct = base_qct.deepcopy()
			if basis_pattern is not None:
				append_measurement_basis(qct, basis_pattern, target_qubits=target_qubits_in_use)
			elif not self._has_measurements(qct):
				qct.barrier()
				qct.measure(target_qubits_in_use, list(range(len(target_qubits_in_use))))
			if not noisy_circuit and (basis_pattern is not None or not self._has_measurements(qct)):
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
				convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
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
				# For tianyan provider, wait for previous task to complete before submitting next.
				if self._needs_sequential_submission():
					self._wait_for_last_task()
				task_id_1 = self._submit_circuit_async(
					name=f"{name}_g{gi}",
					circuit=qct,
					shots=shots,
					chip_name=chip_name,
					submit_options=submit_options,
				)
				if self._needs_sequential_submission():
					self._last_pending_task_id = task_id_1
				if print_true:
					logger.info("compile and run circuit: %s", f"{name}_g{gi}")
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
					if self._needs_sequential_submission():
						self._wait_for_last_task()
					task_id_3 = self._submit_circuit_async(
						name=f"{name}_g{gi}_zne3",
						circuit=qct,
						shots=shots,
						chip_name=chip_name,
						submit_options=submit_options,
					)
					if self._needs_sequential_submission():
						self._last_pending_task_id = task_id_3
					pending.append((gi, "3", task_id_3))
					task_ids.append(task_id_3)

			meta = {
				"basis": basis_pattern,
				"observables": group["observables"],
			}
			group_meta.append(meta)

		target_qubits_group = target_qubits_in_use.copy()
		
		per_qubit: Optional[Dict[int, np.ndarray]] = None
		if readout_mitigation:
			if len(target_qubits_group) != num_qubits:
				raise ValueError(
					f"num_qubits ({num_qubits}) must match len(target_qubits) ({len(target_qubits_group)}) for readout mitigation"
				)
			if self._needs_sequential_submission():
				self._wait_for_last_task()
			calibration_manager = ReadoutCalibrationManager(
				cache_dir=Path(__file__).resolve().parent / ".cache",
				submit_circuit_async=self._submit_circuit_async,
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
				print_true=print_true,
			)
			per_qubit = {k: np.asarray(v) for k, v in cal.per_qubit_confusion.items()}

		if print_true:
			logger.info("which qubits: %s", list(target_qubits_group) if target_qubits_group is not None else "auto")

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
			# Infer bit width from counts (handles partial-measurement projection).
			_sample_key = next(iter(counts_1), "")
			_num_bits = len(_sample_key) if _sample_key else num_qubits
			samples_1 = get_samples(counts_1, _num_bits)
			samples_list.append(samples_1.tolist())
			if zne:
				counts_3 = group_counts[gi]["3"]
				samples_3 = get_samples(counts_3, _num_bits)
				samples_zne_list.append(samples_3.tolist())

			if return_probabilities:
				probs_1 = get_probabilities_from_samples(samples_1, _num_bits)
				probabilities_raw_list.append(probs_1.tolist())
				probabilities_list.append(probs_1.tolist())
				if zne:
					probs_3 = get_probabilities_from_samples(samples_3, _num_bits)
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
			logger.info("returning results")

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
		clifford_fitting: bool = False,
		clifford_fitting_num_samples: int = 8,
		clifford_fitting_num_non_clifford_gates: int = 0,
		clifford_fitting_seed: Optional[int] = None,
	) -> RunResult:
		"""Automatically select hardware, run, and return results.

		Args:
			circuit (*Union[str, QuantumCircuit]*): Quantum circuit to execute.
			name (*str*): Experiment name for the submission.
			num_qubits (*int*): Number of qubits.
			provider (*str*): Platform provider name. One of ``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``, ``"origin"``, ``"fieldquantum"``, ``"simulator"`` (case-insensitive). If ``prefer_chips`` contains a known chip name, the inferred provider overrides this argument. Defaults to ``'quafu'``.
			shots (*int*): Number of measurement shots. Defaults to ``8192``.
			zne (*bool*): Whether to apply zero-noise extrapolation. Defaults to ``False``.
			readout_mitigation (*bool*): Whether to apply readout error mitigation. Defaults to ``False``.
			readout_shots (*Optional[int]*): Number of shots for readout calibration. Defaults to ``None``.
			observables (*Optional[Sequence[str] | str]*): Observable operators to measure. Defaults to ``None``.
			return_probabilities (*bool*): Whether to return probability distributions. Defaults to ``False``.
			target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement. Defaults to ``None``.
			prefer_chips (*Optional[Sequence[str] | str]*): Preferred chip names for scheduling. Defaults to ``None``.
			transpile_on_client (*bool*): Whether to transpile on the client side. Defaults to ``True``.
			max_wait_time (*int*): Maximum wait time in seconds. Defaults to ``3600``.
			sleep_time (*int*): Polling interval in seconds. Defaults to ``5``.
			print_true (*bool*): Whether to print progress information. Defaults to ``True``.
			clifford_fitting (*bool*): Whether to apply Clifford-randomized affine correction to ``observables``. No effect if *observables* is empty. Defaults to ``False``.
			clifford_fitting_num_samples (*int*): Number of Clifford-randomized calibration circuits. Defaults to ``8``.
			clifford_fitting_num_non_clifford_gates (*int*): Per-circuit count of single-qubit gates replaced with Haar-random unitaries (instead of random Cliffords). Defaults to ``0``.
			clifford_fitting_seed (*Optional[int]*): RNG seed for reproducible calibration sampling. Defaults to ``None``.

		Returns:
			``RunResult`` containing counts, expectations, and metadata.
		"""
		# Normalize input circuit; strip measurements only when observables conflict.
		qc = self._normalize_input_circuit(circuit, num_qubits, observables=observables)
		provider = resolve_provider(provider, prefer_chips)
		use_dd = provider not in {"tianyan", "guodun", "tencent", "simulator", "fieldquantum"}
		# Tencent QOS parser doesn't understand u(...) gates; keep native h/rz/x/y/z.
		convert_single_qubit_gate_to_u = provider not in {"tencent", "fieldquantum"}

		runtime = create_provider_runtime(provider=provider, client=self)

		resolved_backend = runtime.backend_adapter.resolve_backend(
			num_qubits=num_qubits,
			prefer_hardware=prefer_chips,
		)

		self.chip_name = resolved_backend.hardware_name
		self.chip_backend = resolved_backend.backend

		def _as_int(value, default):
			"""Convert *value* to ``int``, falling back to *default*.

			Args:
				value: Value to convert.
				default: Fallback used when conversion fails.

			Returns:
				``int`` converted value.
			"""
			try:
				return int(value)
			except Exception:
				return int(default)

		submit_options = {
			"max_wait_time": _as_int(max_wait_time, 3600),
			"sleep_time": _as_int(sleep_time, 5),
		}
		self._active_task_adapter = runtime.task_adapter
		self._active_resolved_backend = resolved_backend

		# When Clifford fitting is requested with observables, transpile once
		# on the client and reuse the same template for both the main run
		# and the calibration jobs (mirrors the run_vqe / run_qaoa pattern).
		observables_list: List[str] = []
		if observables is not None:
			observables_list = [observables] if isinstance(observables, str) else list(observables)
		do_clifford_fitting = bool(clifford_fitting) and bool(observables_list) and int(clifford_fitting_num_samples) > 0

		transpiled_qc: Optional[QuantumCircuit] = None
		target_qubits_in_use = target_qubits
		if do_clifford_fitting:
			from .backend import is_noisy_circuit_for_backend
			noisy_circuit = is_noisy_circuit_for_backend(qc, resolved_backend.hardware_name)
			if transpile_on_client and not noisy_circuit:
				transpiled_qc = self._transpile_with_backend(
					deepcopy(qc),
					resolved_backend.backend,
					target_qubits=target_qubits,
					use_dd=use_dd,
					convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
				)
				target_qubits_in_use = self._ordered_target_qubits_from_layout(
					compiled_qc=transpiled_qc,
					original_qc=qc,
					num_qubits=num_qubits,
				)
			else:
				transpiled_qc = deepcopy(qc)
				if target_qubits is not None:
					target_qubits_in_use = list(target_qubits)
				else:
					used = list(transpiled_qc.qubits)
					target_qubits_in_use = used if used else list(range(num_qubits))

			result = self._run_with_backend(
				transpiled_qc,
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
				target_qubits=target_qubits_in_use,
				use_dd=use_dd,
				print_true=print_true,
				transpile=False,
				submit_options=submit_options,
				convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
			)

			from ..algorithms.optimizer_utils import (
				apply_clifford_fit,
				build_clifford_fit_map,
			)
			fit_map = build_clifford_fit_map(
				self,
				name=f"{name}_cfit",
				num_qubits=num_qubits,
				backend=resolved_backend.backend,
				chip_name=resolved_backend.hardware_name,
				observables=observables_list,
				shots=shots,
				zne=zne,
				readout_mitigation=readout_mitigation,
				transpiled_template=transpiled_qc,
				num_samples=int(clifford_fitting_num_samples),
				num_non_clifford_gates=int(clifford_fitting_num_non_clifford_gates),
				seed=clifford_fitting_seed,
				target_qubits=target_qubits_in_use,
				convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
			)
			result.observable_values = apply_clifford_fit(result.observable_values, fit_map)
			return result

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
			use_dd=use_dd,
			print_true=print_true,
			transpile=bool(transpile_on_client),
			submit_options=submit_options,
			convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
		)

