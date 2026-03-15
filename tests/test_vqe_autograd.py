import numpy as np
import pytest


torch = pytest.importorskip("torch")

from quantum_hw.algorithms.vqe import build_ising_hamiltonian, run_vqe_with_backend
from quantum_hw.api.backend import Backend
from quantum_hw.circuit import QuantumCircuit
import quantum_hw.sim as sim_pkg


def test_vqe_autograd_runs_on_simulator():
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    result = run_vqe_with_backend(
        object(),
        name="test_vqe_autograd",
        num_qubits=2,
        backend=backend,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        layers=1,
        shots=512,
        max_iters=4,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=7,
        gradient_method="autograd",
    )

    assert len(result.energy_history) == 4
    assert len(result.grad_history) == 4
    assert np.isfinite(result.best_energy)
    assert result.best_energy <= min(result.energy_history)


def test_vqe_autograd_rejects_non_simulator():
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    with pytest.raises(ValueError, match="only supported on Simulator"):
        run_vqe_with_backend(
            object(),
            name="test_vqe_autograd_bad_chip",
            num_qubits=2,
            backend=backend,
            chip_name="Baihua",
            hamiltonian=hamiltonian,
            layers=1,
            shots=512,
            max_iters=1,
            learning_rate=0.2,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
            shift=np.pi / 2,
            zne=False,
            readout_mitigation=False,
            gradient_method="autograd",
        )


def test_vqe_autograd_custom_ansatz_supports_negative_expression_param():
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    qc = QuantumCircuit(2)
    qc.ry("theta", 0)
    qc.ry("-theta", 1)
    qc.cx(0, 1)

    result = run_vqe_with_backend(
        object(),
        name="test_vqe_autograd_custom_negative",
        num_qubits=2,
        backend=backend,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        layers=1,
        shots=256,
        max_iters=2,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=7,
        gradient_method="autograd",
        ansatz="custom",
        custom_ansatz_circuit=qc,
    )

    assert len(result.energy_history) == 2
    assert np.isfinite(result.best_energy)


def test_vqe_autograd_calls_sim_facade_energy(monkeypatch):
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)
    backend = Backend("Simulator")

    calls = {"count": 0}

    def _fake_energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian):
        calls["count"] += 1
        expectations = {obs: 0.0 for _, obs in hamiltonian}
        # keep graph connected to params for autograd backward
        return (params ** 2).sum(), expectations

    monkeypatch.setattr(sim_pkg, "energy_and_expectations", _fake_energy_and_expectations)

    result = run_vqe_with_backend(
        object(),
        name="test_vqe_autograd_sim_facade",
        num_qubits=2,
        backend=backend,
        chip_name="Simulator",
        hamiltonian=hamiltonian,
        layers=1,
        shots=128,
        max_iters=1,
        learning_rate=0.2,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        seed=7,
        gradient_method="autograd",
    )

    assert calls["count"] >= 1
    assert len(result.energy_history) == 1
