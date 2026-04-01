"""Preset circuit builders for common algorithms."""

from __future__ import annotations

from typing import List

import numpy as np
from ..circuit import QuantumCircuit


def build_ghz(num_qubits: int, measure: bool = False) -> QuantumCircuit:
	"""Build a GHZ state circuit with optional measurements.

	Args:
		num_qubits (*int*): Number of qubits.
		measure (*bool*): Whether to append measurement gates at the end. Defaults to ``False``.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	# Prepare |+> on q0, then entangle along a chain with CX.
	qc.h(0)
	for i in range(num_qubits - 1):
		qc.cx(i, i + 1)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def build_cluster(num_qubits: int, measure: bool = False) -> QuantumCircuit:
	"""Build a 1D cluster-like circuit (H on all qubits + two CZ layers) with optional measurements.

	Args:
		num_qubits (*int*): Number of qubits.
		measure (*bool*): Whether to append measurement gates at the end. Defaults to ``False``.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	for i in range(num_qubits):
		qc.h(i)
	for i in range(num_qubits // 2):
		qc.cz(2 * i, 2 * i + 1)
	for i in range((num_qubits - 1) // 2):
		qc.cz(2 * i + 1, 2 * i + 2)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def _apply_controlled_phase(qc: QuantumCircuit, control: int, target: int, angle: float) -> None:
	"""Apply a controlled phase gate with API fallback.

	Tries ``cp`` → ``cu1`` → ``crz`` in order.  Note that ``crz`` is *not*
	exactly equivalent to ``cp`` (they differ by a global phase); the
	fallback is provided only for circuits where the global phase does
	not matter.

	Args:
		qc (*QuantumCircuit*): Quantum circuit.
		control (*int*): Control qubit index.
		target (*int*): Target qubit index.
		angle (*float*): Rotation angle in radians.

	Raises:
		AttributeError: QuantumCircuit does not support controlled phase gate
	"""
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
	"""Build a QFT circuit with optional swaps and measurements.

	Args:
		num_qubits (*int*): Number of qubits.
		measure (*bool*): Whether to append measurement gates at the end. Defaults to ``False``.
		with_swaps (*bool*): Whether to include final bit-reversal swaps for canonical QFT ordering. Defaults to ``True``.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	for i in range(num_qubits):
		# Hadamard on each qubit plus controlled-phase rotations.
		qc.h(i)
		for j in range(i + 1, num_qubits):
			angle = np.pi / (2 ** (j - i))
			_apply_controlled_phase(qc, j, i, angle)
	if with_swaps:
		# Optional bit-reversal to match canonical QFT output order.
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
	"""Build a trotterized Ising time-evolution circuit.

	Args:
		num_qubits (*int*): Number of qubits.
		j (*float*): ZZ coupling strength.
		h (*float*): Transverse field strength.
		t (*float*): Total evolution time.
		steps (*int*): Number of Trotter steps. Defaults to ``1``.
		measure (*bool*): Whether to append measurement gates at the end. Defaults to ``False``.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	# First-order Trotter: split into ZZ interactions and X rotations.
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
