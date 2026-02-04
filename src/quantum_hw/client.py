from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from quark import Task
from quark.circuit import Backend, QuantumCircuit, Transpiler

from .circuits import build_cluster, build_ghz, build_ising_time_evolution, build_qft
from .observables import (
    append_measurement_basis,
    group_observables,
    pauli_expectation,
    pauli_support,
)
from .hardware import rank_chips
from .readout import (
    apply_readout_mitigation_multi,
    expectation_from_probabilities,
    marginal_probabilities,
)
from .types import CalibrationResult, RunResult
from .utils import extract_qubits_from_openqasm2, get_probabilities, get_samples
from .zne import apply_zne_cz_tripling, zne_linear_extrapolate


class QuantumHardwareClient:
    def __init__(self, token: str):
        """Create a hardware client with an access token."""
        self.token = token
        self.chip_name = None
        self.tmgr = Task(token)
        self.chip_backend = None

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
        return Transpiler(backend).run(qc, target_qubits=list(target_qubits))

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

    def calibrate_readout(
        self,
        target_qubits: Sequence[int],
        shots: Optional[int] = None,
        *,
        chip_name: Optional[str] = None,
        backend: Optional[Backend] = None,
    ) -> CalibrationResult:
        """Calibrate readout error for selected qubits with caching."""
        target_qubits = list(target_qubits)
        if shots is None:
            shots = 1024
        if chip_name is None:
            chip_name = self.chip_name
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
            print("[readout] use cached readout calibration")
            return CalibrationResult(
                target_qubits=target_qubits,
                per_qubit_confusion=cached_confusion,
            )

        if backend is None:
            if self.chip_backend is None:
                raise RuntimeError("chip_backend is not set; use run_auto or provide backend")
            backend = self.chip_backend

        per_qubit_confusion: Dict[int, np.ndarray] = {}
        print("[readout] run readout calibration on hardware")
        pending: List[Tuple[object, int, str]] = []
        for q in missing:
            for bits, qc in self._readout_calibration_circuits():
                qct = self._transpile_with_backend(qc, backend, target_qubits=[q])
                task_id = self._submit_openqasm_async(
                    name=f"readout_cal_q{q}_{bits}",
                    qasm=qct.to_openqasm2,
                    shots=shots,
                    chip_name=chip_name,
                )
                pending.append((task_id, q, bits))

        res_map: Dict[int, Dict[str, Dict[str, int]]] = {q: {} for q in missing}
        for task_id, q, bits in pending:
            status = self._wait_task(task_id)
            if status != "Finished":
                raise RuntimeError(f"readout calibration task {task_id} ended with status {status}")
            counts = self.tmgr.result(task_id)["count"]
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

    def _readout_cache_path(self, *, chip_name: Optional[str]) -> Path:
        """Resolve the on-disk cache path for readout calibration."""
        cache_dir = Path(__file__).resolve().parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = chip_name if chip_name is not None else "unknown"
        return cache_dir / f"readout_{name}.json"

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

    def _load_readout_cache(self, target_qubits: Sequence[int], *, chip_name: Optional[str]) -> Optional[CalibrationResult]:
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

    def _readout_calibration_circuits(self):
        """Build minimal calibration circuits for a single qubit."""
        circuits = []
        for i in range(2):
            bits = format(i, "01b")
            qc = QuantumCircuit(1)
            if bits == "1":
                qc.x(0)
            else:
                qc.x(0)
                qc.x(0)
            qc.measure_all()
            circuits.append((bits, qc))
        return circuits

    def _build_confusion_matrix(self, res_list: Sequence[Dict[str, int]]) -> np.ndarray:
        """Build a 2x2 confusion matrix from two calibration results."""
        mat = np.zeros((2, 2), dtype=float)
        for i, res in enumerate(res_list):
            probs = get_probabilities(res, 1)
            mat[i, :] = probs
        return mat

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
    ) -> RunResult:
        """Run a circuit on a specific backend with optional mitigation."""
        if isinstance(observables, str):
            observables = [observables]
        if observables is None:
            observables = []
        observables = list(observables)

        print("[hardware] which hardware:", chip_name)

        supports_by_obs = {obs: pauli_support(obs, num_qubits=num_qubits) for obs in observables}

        if observables:
            groups = group_observables(observables, num_qubits=num_qubits)
        else:
            groups = [{"basis": None, "observables": []}]

        def _prepare_qasm(qc, basis_pattern: Optional[Sequence[str]], scale_zne: bool) -> str:
            """Prepare OpenQASM with optional basis rotation and ZNE scaling."""
            qc = deepcopy(qc)
            if basis_pattern is not None:
                append_measurement_basis(qc, basis_pattern)
            qct = self._transpile_with_backend(qc, backend, target_qubits=target_qubits)
            if scale_zne:
                qct = apply_zne_cz_tripling(qct)
            return qct.to_openqasm2

        pending: List[Tuple[int, str, object]] = []
        group_meta: List[Dict[str, object]] = []
        task_ids: List[object] = []

        for gi, group in enumerate(groups):
            basis_pattern = group["basis"]
            qasm_1 = _prepare_qasm(qc, basis_pattern, scale_zne=False)
            task_id_1 = self._submit_openqasm_async(
                name=f"{name}_g{gi}",
                qasm=qasm_1,
                shots=shots,
                chip_name=chip_name,
            )
            print("[run] compile and run circuit:", f"{name}_g{gi}")
            pending.append((gi, "1", task_id_1))
            task_ids.append(task_id_1)

            if zne:
                qasm_3 = _prepare_qasm(qc, basis_pattern, scale_zne=True)
                task_id_3 = self._submit_openqasm_async(
                    name=f"{name}_g{gi}_zne3",
                    qasm=qasm_3,
                    shots=shots,
                    chip_name=chip_name,
                )
                print("[run] run circuit:", f"zero-noise extrapolation")
                pending.append((gi, "3", task_id_3))
                task_ids.append(task_id_3)

            group_meta.append({"basis": basis_pattern, "observables": group["observables"], "qasm_1": qasm_1})

        if target_qubits is None:
            target_qubits_group = extract_qubits_from_openqasm2(qasm_1)
        else:
            target_qubits_group = target_qubits

        print("[run] which qubits:", list(target_qubits_group) if target_qubits_group is not None else "auto")

        group_counts: Dict[int, Dict[str, Dict[str, int]]] = {i: {} for i in range(len(group_meta))}
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
                support = supports_by_obs[obs]
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
                cal = self.calibrate_readout(
                    target_qubits_group,
                    shots=readout_shots,
                    chip_name=chip_name,
                    backend=backend,
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

        print("[finish] returning results")

        return RunResult(
            task_ids=[str(t) for t in task_ids] if task_ids else None,
            samples=samples_out,
            probabilities=probabilities_out,
            probabilities_raw=probabilities_raw_out,
            observable_values=observable_values_out,
            observable_values_raw=observable_values_raw_out,
        )

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
    ) -> RunResult:
        """Automatically select hardware, run, and return results."""
        print("[hardware] read hardware information and select")
        if circuit[:4] == "OPEN":
            qc = QuantumCircuit.from_openqasm2(openqasm2_str=circuit)
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
            # try:
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
            )
            #     break
            # except Exception as exc:
            #     last_error = exc
            #     continue

        raise RuntimeError("all candidate chips failed to transpile or run") from last_error
