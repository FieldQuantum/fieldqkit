"""Tests for batch circuit execution helper."""

from unittest.mock import patch

import pytest

from quantum_hw.api.backend import Backend
from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.circuit import QuantumCircuit


@pytest.fixture(autouse=True)
def _patch_task(monkeypatch):
	class _DummyTask:
		def __init__(self, *args, **kwargs):
			pass

	monkeypatch.setattr("quantum_hw.api.client.Task", _DummyTask)



def test_run_with_backend_batch_matches_single_simulator():
	client = QuantumHardwareClient()
	backend = Backend("Simulator")
	num_qubits = 1
	shots = 32

	qc0 = QuantumCircuit(num_qubits)
	qc1 = QuantumCircuit(num_qubits)
	qc0.x(0)
	qc0.x(0)
	qc1.x(0)

	obs = ["Z0"]

	res0 = client._run_with_backend(
		qc0,
		"single0",
		num_qubits,
		backend=backend,
		chip_name="Simulator",
		shots=shots,
		observables=obs,
		return_probabilities=True,
		print_true=False,
	)
	res1 = client._run_with_backend(
		qc1,
		"single1",
		num_qubits,
		backend=backend,
		chip_name="Simulator",
		shots=shots,
		observables=obs,
		return_probabilities=True,
		print_true=False,
	)

	batch = client._run_with_backend_batch(
		[qc0, qc1],
		"batch",
		num_qubits,
		backend=backend,
		chip_name="Simulator",
		shots=shots,
		observables=[obs, obs],
		return_probabilities=True,
		print_true=False,
	)

	assert len(batch) == 2
	assert batch[0].samples == res0.samples
	assert batch[0].probabilities == res0.probabilities
	assert batch[0].observable_values == res0.observable_values
	assert batch[1].samples == res1.samples
	assert batch[1].probabilities == res1.probabilities
	assert batch[1].observable_values == res1.observable_values


def test_run_with_backend_simulator_passes_circuit_to_simulator():
	client = QuantumHardwareClient()
	backend = Backend("Simulator")
	qc = QuantumCircuit(2)
	qc.x(1)

	with patch("quantum_hw.api.client.simulate_counts", return_value={"00": 8}) as mock_simulate_counts:
		client._run_with_backend(
			qc,
			"single",
			2,
			backend=backend,
			chip_name="Simulator",
			shots=8,
			print_true=False,
		)

	first_arg = mock_simulate_counts.call_args.args[0]
	assert isinstance(first_arg, QuantumCircuit)
