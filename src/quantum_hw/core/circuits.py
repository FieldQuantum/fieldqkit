from __future__ import annotations

from typing import List

import numpy as np
from ..circuit import QuantumCircuit


def build_ghz(num_qubits: int, measure: bool = False) -> QuantumCircuit:
	"""Build a GHZ state circuit with optional measurements."""
	qc = QuantumCircuit(num_qubits)
	qc.h(0)
	for i in range(num_qubits - 1):
		qc.cx(i, i + 1)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def build_cluster(num_qubits: int, measure: bool = False) -> QuantumCircuit:
	"""Build a 1D cluster-like circuit with optional measurements."""
	qc = QuantumCircuit(num_qubits)
	for i in range(num_qubits):
		qc.h(i)
	for i in range(num_qubits // 2):
		qc.cz(2 * i, 2 * i + 1)
	for i in range((num_qubits - 1) // 2):
		qc.cz(2 * i + 1, 2 * i + 2)
	for i in range((num_qubits - 1) // 2):
		qc.h(2 * i + 1)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def _apply_controlled_phase(qc: QuantumCircuit, control: int, target: int, angle: float) -> None:
	"""Apply a controlled phase gate with API fallback."""
	if hasattr(qc, "cp"):
		qc.cp(angle, control, target)
		return
	if hasattr(qc, "cu1"):
		qc.cu1(angle, control, target)
		return
	if hasattr(qc, "crz"):
		qc.crz(angle, control, target)
		return
	raise AttributeError("QuantumCircuit does not support controlled phase gate")


def build_qft(num_qubits: int, measure: bool = False, with_swaps: bool = True) -> QuantumCircuit:
	"""Build a QFT circuit with optional swaps and measurements."""
	qc = QuantumCircuit(num_qubits)
	for i in range(num_qubits):
		qc.h(i)
		for j in range(i + 1, num_qubits):
			angle = np.pi / (2 ** (j - i))
			_apply_controlled_phase(qc, j, i, angle)
	if with_swaps:
		for i in range(num_qubits // 2):
			qc.swap(i, num_qubits - 1 - i)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def build_ising_time_evolution(
	num_qubits: int,
	j: float,
	h: float,
	t: float,
	steps: int = 1,
	measure: bool = False,
) -> QuantumCircuit:
	"""Build a trotterized Ising time-evolution circuit."""
	qc = QuantumCircuit(num_qubits)
	dt = t / steps
	for _ in range(steps):
		for i in range(num_qubits - 1):
			qc.cx(i, i + 1)
			qc.rz(2 * j * dt, i + 1)
			qc.cx(i, i + 1)
		for i in range(num_qubits):
			qc.rx(2 * h * dt, i)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc
