import numpy as np
import pytest

from quantum_hw.algorithms.vqe import build_ising_hamiltonian, run_vqe_with_backend
from quantum_hw.api.backend import Backend
from quantum_hw.circuit import QuantumCircuit


class _DummyResult:
    def __init__(self, observable_values):
        self.observable_values = observable_values


class _DummyClient:
    def __init__(self):
        self.transpile_calls = 0
        self.run_transpile_flags = []

    def _transpile_with_backend(self, qc, backend, target_qubits=None, **kwargs):
        self.transpile_calls += 1
        return qc.deepcopy()

    def _run_with_backend(self, qc, name, num_qubits, **kwargs):
        self.run_transpile_flags.append(bool(kwargs.get("transpile", True)))
        observables = kwargs.get("observables", [])
        return _DummyResult({obs: 0.0 for obs in observables})


def test_vqe_parameter_shift_hardware_transpile_once():
    client = _DummyClient()
    backend = Backend("Simulator")
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)

    run_vqe_with_backend(
        client,
        name="test_vqe_compile_once",
        num_qubits=2,
        backend=backend,
        chip_name="Baihua",
        hamiltonian=hamiltonian,
        layers=1,
        shots=256,
        max_iters=2,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        gradient_method="parameter-shift",
        seed=1,
    )

    assert client.transpile_calls == 1
    assert client.run_transpile_flags
    assert all(flag is False for flag in client.run_transpile_flags)


def test_vqe_parameter_shift_ucc_transpile_once():
    client = _DummyClient()
    backend = Backend("Simulator")
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)

    run_vqe_with_backend(
        client,
        name="test_vqe_ucc_compile_once",
        num_qubits=2,
        backend=backend,
        chip_name="Baihua",
        hamiltonian=hamiltonian,
        layers=1,
        shots=128,
        max_iters=1,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        gradient_method="parameter-shift",
        ansatz="ucc",
        seed=2,
    )

    assert client.transpile_calls == 1
    assert client.run_transpile_flags
    assert all(flag is False for flag in client.run_transpile_flags)


def _build_custom_ansatz_qc(num_qubits: int, layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.ry(f"alpha_{idx}", q)
            idx += 1
    return qc


def _build_custom_ansatz_with_negative_tied_param() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.ry("theta", 0)
    qc.ry("-theta", 1)
    qc.cx(0, 1)
    return qc


def test_vqe_custom_ansatz_requires_circuit():
    client = _DummyClient()
    backend = Backend("Simulator")
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)

    with pytest.raises(ValueError, match="custom ansatz requires custom_ansatz_circuit"):
        run_vqe_with_backend(
            client,
            name="test_vqe_custom_missing",
            num_qubits=2,
            backend=backend,
            chip_name="Baihua",
            hamiltonian=hamiltonian,
            layers=1,
            shots=128,
            max_iters=1,
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
            shift=np.pi / 2,
            zne=False,
            readout_mitigation=False,
            gradient_method="parameter-shift",
            ansatz="custom",
        )


def test_vqe_custom_ansatz_uses_circuit():
    client = _DummyClient()
    backend = Backend("Simulator")
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)

    custom_qc = _build_custom_ansatz_qc(num_qubits=2, layers=1)

    run_vqe_with_backend(
        client,
        name="test_vqe_custom_ansatz",
        num_qubits=2,
        backend=backend,
        chip_name="Baihua",
        hamiltonian=hamiltonian,
        layers=1,
        shots=128,
        max_iters=1,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        gradient_method="parameter-shift",
        ansatz="custom",
        custom_ansatz_circuit=custom_qc,
        seed=3,
    )

    assert client.transpile_calls == 1


def test_vqe_custom_ansatz_negative_tied_parameter_uses_single_variable():
    client = _DummyClient()
    backend = Backend("Simulator")
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)

    custom_qc = _build_custom_ansatz_with_negative_tied_param()

    run_vqe_with_backend(
        client,
        name="test_vqe_custom_negative_tied_param",
        num_qubits=2,
        backend=backend,
        chip_name="Baihua",
        hamiltonian=hamiltonian,
        layers=1,
        shots=128,
        max_iters=1,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        gradient_method="parameter-shift",
        ansatz="custom",
        custom_ansatz_circuit=custom_qc,
        init_params=[0.2],
        seed=3,
    )

    assert client.transpile_calls == 1


def test_vqe_parameter_shift_clifford_fitting_returns_coefficients():
    client = _DummyClient()
    backend = Backend("Simulator")
    hamiltonian = build_ising_hamiltonian(num_qubits=2, j=1.0, h=1.0)

    result = run_vqe_with_backend(
        client,
        name="test_vqe_clifford_fitting",
        num_qubits=2,
        backend=backend,
        chip_name="Baihua",
        hamiltonian=hamiltonian,
        layers=1,
        shots=64,
        max_iters=1,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        shift=np.pi / 2,
        zne=False,
        readout_mitigation=False,
        gradient_method="parameter-shift",
        seed=5,
        clifford_fitting=True,
        clifford_fitting_num_samples=2,
    )

    assert client.transpile_calls == 1
    assert result.clifford_fitting is not None
    assert set(result.clifford_fitting.keys()) == {obs for _, obs in hamiltonian}
    for coeffs in result.clifford_fitting.values():
        assert "a" in coeffs and "b" in coeffs
