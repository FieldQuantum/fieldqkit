import numpy as np

from quantum_hw.algorithms.vqe import build_ising_hamiltonian, run_vqe_with_backend
from quantum_hw.api.backend import Backend


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
