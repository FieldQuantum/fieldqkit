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
	"""Apply a controlled-phase CP(θ) using the basic gate decomposition.

	``CP(λ) = Rz(λ/2, ctrl) · CX · Rz(-λ/2, tgt) · CX · Rz(λ/2, tgt)``

	Args:
		qc (*QuantumCircuit*): Quantum circuit.
		control (*int*): Control qubit index.
		target (*int*): Target qubit index.
		angle (*float*): Phase angle λ in radians.
	"""
	# CP(λ) decomposed into basic gates: rz + cx + rz + cx + rz
	qc.rz(angle / 2, control)
	qc.cx(control, target)
	qc.rz(-angle / 2, target)
	qc.cx(control, target)
	qc.rz(angle / 2, target)


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

	Implements first-order Trotter for the transverse-field Ising
	Hamiltonian ``H = J * sum_i Z_i Z_{i+1} + h * sum_i X_i`` (note the
	positive-sign convention; this differs from
	:func:`quantum_hw.algorithms.vqe.build_ising_hamiltonian` by an overall
	minus, equivalent to ``j -> -j`` and ``h -> -h``).

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
	# Native rzz(theta) = exp(-i (theta/2) Z Z) and rx(theta) = exp(-i (theta/2) X).
	# Pass 2 * j * dt and 2 * h * dt to keep behaviour identical to the original
	# CX-RZ(2 j dt)-CX + RX(2 h dt) implementation.
	dt = t / steps
	for _ in range(steps):
		for i in range(num_qubits - 1):
			qc.rzz(2 * j * dt, i, i + 1)
		for i in range(num_qubits):
			qc.rx(2 * h * dt, i)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def build_heisenberg_time_evolution(
	num_qubits: int,
	t: float,
	jx: float = 1.0,
	jy: float = 1.0,
	jz: float = 1.0,
	hz: float = 0.0,
	steps: int = 1,
	measure: bool = False,
) -> QuantumCircuit:
	"""Build a trotterized Heisenberg time-evolution circuit.

	Implements first-order Trotter for the Heisenberg Hamiltonian
	``H = sum_i (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1}) + hz * sum_i Z_i``
	(matching :func:`quantum_hw.algorithms.vqe.build_heisenberg_hamiltonian`).

	Args:
		num_qubits (*int*): Number of qubits (chain length).
		t (*float*): Total evolution time.
		jx (*float*): XX coupling.
		jy (*float*): YY coupling.
		jz (*float*): ZZ coupling.
		hz (*float*): Longitudinal field.
		steps (*int*): Number of Trotter steps.
		measure (*bool*): Whether to append measurement gates at the end.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	dt = t / steps
	for _ in range(steps):
		for i in range(num_qubits - 1):
			if jx != 0:
				qc.rxx(2 * jx * dt, i, i + 1)
			if jy != 0:
				qc.ryy(2 * jy * dt, i, i + 1)
			if jz != 0:
				qc.rzz(2 * jz * dt, i, i + 1)
		if hz != 0:
			for i in range(num_qubits):
				qc.rz(2 * hz * dt, i)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def build_xxz_time_evolution(
	num_qubits: int,
	t: float,
	jxy: float = 1.0,
	jz: float = 1.0,
	hz: float = 0.0,
	steps: int = 1,
	measure: bool = False,
) -> QuantumCircuit:
	"""Build a trotterized XXZ time-evolution circuit.

	Implements first-order Trotter for the XXZ Hamiltonian
	``H = Jxy * sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + Jz * sum_i Z_i Z_{i+1} + hz * sum_i Z_i``
	(matching :func:`quantum_hw.algorithms.vqe.build_xxz_hamiltonian`). Uses
	native ``rxx`` / ``ryy`` / ``rzz`` gates.

	Args:
		num_qubits (*int*): Number of qubits (chain length).
		t (*float*): Total evolution time.
		jxy (*float*): XX = YY coupling.
		jz (*float*): ZZ coupling.
		hz (*float*): Longitudinal field.
		steps (*int*): Number of Trotter steps.
		measure (*bool*): Whether to append measurement gates at the end.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	dt = t / steps
	for _ in range(steps):
		for i in range(num_qubits - 1):
			if jxy != 0:
				qc.rxx(2 * jxy * dt, i, i + 1)
				qc.ryy(2 * jxy * dt, i, i + 1)
			if jz != 0:
				qc.rzz(2 * jz * dt, i, i + 1)
		if hz != 0:
			for i in range(num_qubits):
				qc.rz(2 * hz * dt, i)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc


def build_xy_time_evolution(
	num_qubits: int,
	t: float,
	jx: float = 1.0,
	jy: float = 1.0,
	hz: float = 0.0,
	steps: int = 1,
	measure: bool = False,
) -> QuantumCircuit:
	"""Build a trotterized XY time-evolution circuit.

	Implements first-order Trotter for the XY Hamiltonian
	``H = Jx * sum_i X_i X_{i+1} + Jy * sum_i Y_i Y_{i+1} + hz * sum_i Z_i``
	(matching :func:`quantum_hw.algorithms.vqe.build_xy_hamiltonian`). Uses
	native ``rxx`` / ``ryy`` gates.

	Args:
		num_qubits (*int*): Number of qubits (chain length).
		t (*float*): Total evolution time.
		jx (*float*): XX coupling.
		jy (*float*): YY coupling.
		hz (*float*): Longitudinal field.
		steps (*int*): Number of Trotter steps.
		measure (*bool*): Whether to append measurement gates at the end.

	Returns:
		Constructed ``QuantumCircuit``.
	"""
	qc = QuantumCircuit(num_qubits)
	dt = t / steps
	for _ in range(steps):
		for i in range(num_qubits - 1):
			if jx != 0:
				qc.rxx(2 * jx * dt, i, i + 1)
			if jy != 0:
				qc.ryy(2 * jy * dt, i, i + 1)
		if hz != 0:
			for i in range(num_qubits):
				qc.rz(2 * hz * dt, i)
	if measure:
		qc.barrier()
		qc.measure_all()
	return qc
