"""Readout calibration and mitigation utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..circuit import QuantumCircuit

from .utils import get_probabilities


def build_readout_calibration_circuits(num_qubits: int):
	"""Build calibration circuits for all computational basis states."""
	circuits = []
	for i in range(2**num_qubits):
		bits = format(i, f"0{num_qubits}b")
		qc = QuantumCircuit(num_qubits)
		for q, b in enumerate(bits[::-1]):
			if b == "1":
				qc.x(q)
		qc.measure(range(num_qubits), range(num_qubits))
		circuits.append((bits, qc))
	return circuits


def build_confusion_matrix(res_list: Sequence[Dict[str, int]], num_qubits: int) -> np.ndarray:
	"""Build a confusion matrix from calibration results."""
	dim = 2**num_qubits
	mat = np.zeros((dim, dim), dtype=float)
	for i, res in enumerate(res_list):
		probs = get_probabilities(res, num_qubits)
		mat[i, :] = probs
	return mat


def build_local_confusion_matrix(per_qubit_confusion: Dict[int, np.ndarray], target_qubits: Sequence[int]) -> np.ndarray:
	"""Tensor product local per-qubit confusion matrices."""
	if not target_qubits:
		raise ValueError("target_qubits is empty")
	mats = [per_qubit_confusion[q] for q in target_qubits]
	out = mats[0]
	for m in mats[1:]:
		out = np.kron(out, m)
	return out


def mitigate_readout(probabilities: np.ndarray, confusion_matrix: np.ndarray) -> np.ndarray:
	"""Apply readout mitigation using a pseudo-inverse."""
	if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
		raise ValueError("confusion_matrix must be square")
	pinv = np.linalg.pinv(confusion_matrix)
	mitigated = pinv @ probabilities
	mitigated = np.clip(mitigated, 0.0, 1.0)
	s = mitigated.sum()
	if s == 0:
		return mitigated
	return mitigated / s


def expectation_from_probabilities(probabilities: np.ndarray, support: Sequence[int]) -> float:
	"""Compute Z-basis expectation value from probabilities."""
	if not support:
		return 1.0
	num = len(support)
	probs = probabilities.reshape([2] * num)
	parity = np.zeros([2] * num, dtype=int)
	for i in range(num):
		shape = [1] * num
		shape[i] = 2
		parity += np.arange(2).reshape(shape)
	sign = 1.0 - 2.0 * (parity % 2)
	return float((probs * sign).sum())

def calibrate_readout(Task, Backend, Transpiler, chip_name: str, target_qubits: List[int], shots: int, qasm_version: str = "2.0") -> Dict[int, np.ndarray]:
	"""Standalone readout calibration helper (legacy interface)."""
	tmgr = Task()
	chip_backend = Backend(chip_name)

	per_qubit_confusion: Dict[int, np.ndarray] = {}
	pending: List[Tuple[object, int, str]] = []
	for q in target_qubits:
		circuits = build_readout_calibration_circuits(num_qubits=1)
		for bits, qc in circuits:
			qct = Transpiler(chip_backend).run(qc, target_qubits=[q])
			task = {
				"chip": chip_name,
				"name": f"readout_cal_q{q}_{bits}",
				"circuit": qct.to_openqasm2 if qasm_version == "2.0" else qct.to_openqasm3,
				"shots": shots,
				"compile": False,
			}
			task_id = tmgr.run(task)
			pending.append((task_id, q, bits))

	res_map: Dict[int, Dict[str, Dict[str, int]]] = {q: {} for q in target_qubits}
	for task_id, q, bits in pending:
		while True:
			status = tmgr.status(task_id)
			if status in {"Finished", "Failed", "Canceled"}:
				break
		if status != "Finished":
			raise RuntimeError(f"readout calibration task {task_id} ended with status {status}")
		res = tmgr.result(task_id)["count"]
		res_map[q][bits] = res

	for q in target_qubits:
		counts_list = [res_map[q]["0"], res_map[q]["1"]]
		per_qubit_confusion[q] = build_confusion_matrix(counts_list, num_qubits=1)

	return per_qubit_confusion
