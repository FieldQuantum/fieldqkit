"""Minimal cqlib adapter for submitting circuits and mapping results."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..circuit import QuantumCircuit
from ..compile import Transpiler
from ..compile.translate import TranslateToBasisGates
from ..core.observables import append_measurement_basis, group_observables, pauli_expectation
from ..core.types import RunResult
from ..core.utils import get_probabilities_from_samples, get_samples
from .provider_backend import build_cqlib_backend_bundle


def _normalize_counts(counts: Dict[str, int], num_qubits: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in counts.items():
        if not isinstance(key, str) or any(ch not in "01" for ch in key):
            continue
        if len(key) < num_qubits:
            norm_key = key.rjust(num_qubits, "0")
        elif len(key) > num_qubits:
            norm_key = key[-num_qubits:]
        else:
            norm_key = key
        out[norm_key] = out.get(norm_key, 0) + int(round(float(value)))
    return out


def _counts_from_result_status_matrix(matrix: object, num_qubits: int) -> Optional[Dict[str, int]]:
    """Parse resultStatus sample matrix to counts.

    Expected shape is (shots + 1, nqubits), where:
    - first row: measured qubit indices
    - remaining rows: shot samples with 0/1 values
    """
    if not isinstance(matrix, list) or len(matrix) < 2:
        return None
    if not all(isinstance(row, list) for row in matrix):
        return None

    header = matrix[0]
    if len(header) == 0:
        return None

    shots_rows = matrix[1:]
    counts: Dict[str, int] = {}
    for row in shots_rows:
        try:
            bits = [int(v) for v in row]
        except Exception:
            continue
        if any(v not in (0, 1) for v in bits):
            continue

        bitstring = "".join(str(b) for b in bits)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts if counts else None


def _extract_result_status_counts(item: object, num_qubits: int) -> Optional[Dict[str, int]]:
    if not isinstance(item, dict):
        return None
    return _counts_from_result_status_matrix(item.get("resultStatus"), num_qubits=num_qubits)


class CqlibAdapter:
    """Adapter that runs jobs through cqlib and returns RunResult."""

    def __init__(
        self,
        *,
        login_key: str,
        platform: str = "tianyan",
        machine_name: Optional[str] = None,
        submit_mode: str = "submit_job",
    ) -> None:
        if not login_key:
            raise ValueError("cqlib login key cannot be empty")

        from cqlib.quantum_platform import GuoDunPlatform, TianYanPlatform, QuantumLanguage
        from cqlib.utils.qasm_to_qcis.qasm_to_qcis import QasmToQcis

        self._quantum_language = QuantumLanguage
        self._converter = QasmToQcis()
        self._submit_mode = str(submit_mode).lower()
        self._platform_name = str(platform).lower()
        self._machine_name = machine_name

        platform_cls = TianYanPlatform if self._platform_name == "tianyan" else GuoDunPlatform
        self._platform = platform_cls(login_key=login_key, auto_login=True, machine_name=machine_name)
        self._backend_bundle = None

    def _submit_and_query(
        self,
        *,
        qcis: str,
        name: str,
        shots: int,
        max_wait_time: int,
        sleep_time: int,
    ) -> Tuple[List[str], List[dict]]:
        if self._submit_mode == "submit_experiment" and hasattr(self._platform, "submit_experiment"):
            query_ids = self._platform.submit_experiment(
                circuit=qcis,
                language=self._quantum_language.QCIS,
                name=name,
                num_shots=shots,
                machine_name=getattr(self._platform, "machine_name", None),
                is_verify=True,
            )
        else:
            query_ids = self._platform.submit_job(
                circuit=qcis,
                exp_name=name,
                num_shots=shots,
                language=self._quantum_language.QCIS,
                is_verify=True,
            )

        if isinstance(query_ids, str):
            qids = [query_ids]
        elif isinstance(query_ids, list):
            qids = [str(q) for q in query_ids]
        else:
            raise RuntimeError("cqlib submit returned unexpected query_ids type")
        
        query_payload = qids[0] if len(qids) == 1 else qids
        try:
            result_payload = self._platform.query_experiment(
                query_payload,
                max_wait_time=max_wait_time,
                sleep_time=sleep_time,
            )
        except Exception:
            raise RuntimeError(f"cqlib query failed for query_ids={qids}")
        if isinstance(result_payload, dict):
            items = [result_payload]
        elif isinstance(result_payload, list):
            items = [r for r in result_payload if isinstance(r, dict)]
        else:
            items = []
        return qids, items

    def _extract_counts(self, results: List[dict], *, num_qubits: int) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for item in results:
            counts = _extract_result_status_counts(item, num_qubits=num_qubits)
            if counts is None:
                continue
            counts = _normalize_counts(counts, num_qubits=num_qubits)
            for bit, cnt in counts.items():
                merged[bit] = merged.get(bit, 0) + int(cnt)
        if not merged:
            raise RuntimeError("failed to extract counts: resultStatus not found or invalid")
        return merged

    @staticmethod
    def _ordered_target_qubits_from_layout(
        *,
        compiled_qc: QuantumCircuit,
        num_qubits: int,
    ) -> Optional[List[int]]:
        layout = getattr(compiled_qc, "logical_to_physical", None)
        if not isinstance(layout, dict) or not layout:
            return None
        ordered: List[int] = []
        for lq in range(num_qubits):
            pq = layout.get(lq)
            if not isinstance(pq, int):
                return None
            ordered.append(pq)
        if len(set(ordered)) != len(ordered):
            return None
        return ordered

    def _prepare_group_circuit(
        self,
        base_qc: QuantumCircuit,
        num_qubits: int,
        basis_pattern: Optional[Sequence[str]],
    ) -> QuantumCircuit:
        qct = base_qc.deepcopy()
        target_qubits_in_use = self._ordered_target_qubits_from_layout(compiled_qc=base_qc, num_qubits=num_qubits)
        if target_qubits_in_use is None:
            target_qubits_in_use = list(range(num_qubits))

        if basis_pattern is not None:
            append_measurement_basis(qct, basis_pattern, target_qubits=target_qubits_in_use)
        else:
            qct.measure(target_qubits_in_use, list(range(len(target_qubits_in_use))))

        translator = TranslateToBasisGates(
            convert_single_qubit_gate_to_u=True,
            two_qubit_gate_basis=(
                self._backend_bundle.backend.two_qubit_gate_basis
                if self._backend_bundle is not None
                else "cz"
            ),
        )
        return translator.run(qct)

    def _transpile_base_circuit(self, qc: QuantumCircuit, *, num_qubits: int) -> QuantumCircuit:
        self._backend_bundle = build_cqlib_backend_bundle(
            platform_obj=self._platform,
            machine_name=self._machine_name,
            num_qubits=num_qubits,
        )
        target_qubits = self._backend_bundle.target_qubits
        transpiled = Transpiler(self._backend_bundle.backend).run(
            qc.deepcopy(),
            target_qubits=target_qubits,
            use_dd=False,
            use_three_qubit_decompose=True,
            use_sabre_routing=True,
            use_translate_to_basis=True,
            use_gate_compressor=True,
        )
        return transpiled

    def run(
        self,
        qc: QuantumCircuit,
        *,
        name: str,
        num_qubits: int,
        shots: int,
        observables: Optional[Sequence[str] | str] = None,
        return_probabilities: bool = False,
        merge_groups: bool = True,
        max_wait_time: int = 3600,
        sleep_time: int = 5,
        transpile_on_client: bool = True,
    ) -> RunResult:
        if isinstance(observables, str):
            observables = [observables]
        if observables is None:
            observables = []
        obs_list = list(observables)

        base_qc = qc.deepcopy()
        if any(g[0] == "measure" for g in getattr(base_qc, "gates", [])):
            base_qc.remove_gate("measure")

        if transpile_on_client:
            base_qc = self._transpile_base_circuit(base_qc, num_qubits=num_qubits)

        if obs_list:
            if merge_groups:
                groups = group_observables(obs_list, num_qubits=num_qubits)
            else:
                from ..core.observables import pauli_basis_pattern

                groups = [
                    {"basis": pauli_basis_pattern(obs, num_qubits=num_qubits), "observables": [obs]}
                    for obs in obs_list
                ]
        else:
            groups = [{"basis": None, "observables": []}]

        task_ids: List[str] = []
        samples_list: List[List[List[int]]] = []
        probabilities_list: List[List[float]] = []
        probabilities_raw_list: List[List[float]] = []
        observable_values: Dict[str, float] = {}
        observable_values_raw: Dict[str, float] = {}

        for gi, group in enumerate(groups):
            qct = self._prepare_group_circuit(
                base_qc,
                num_qubits=num_qubits,
                basis_pattern=group["basis"],
            )
            qasm2 = qct.to_openqasm2
            qcis = self._converter.convert_to_qcis(qasm2)
            qids, result_items = self._submit_and_query(
                qcis=qcis,
                name=f"{name}_g{gi}",
                shots=shots,
                max_wait_time=max_wait_time,
                sleep_time=sleep_time,
            )
            task_ids.extend(qids)
            counts = self._extract_counts(result_items, num_qubits=num_qubits)

            samples = get_samples(counts, num_qubits)
            samples_list.append(samples.tolist())

            if return_probabilities:
                probs = get_probabilities_from_samples(samples, num_qubits)
                probabilities_list.append(probs.tolist())
                probabilities_raw_list.append(probs.tolist())

            for obs in group["observables"]:
                val = float(pauli_expectation(samples, obs))
                observable_values[obs] = val
                observable_values_raw[obs] = val

        return RunResult(
            task_ids=task_ids if task_ids else None,
            samples=samples_list,
            samples_zne=None,
            probabilities=probabilities_list,
            probabilities_raw=probabilities_raw_list,
            observable_values=observable_values,
            observable_values_raw=observable_values_raw,
        )
