import types

from quantum_hw.algorithms.shadow import ShadowTomography
from quantum_hw.algorithms.vqe import VQERunner
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.api.backend import HardwareCalibration, HardwareProfile, HardwareTopology, ResolvedBackend


class _FakeBackendAdapter:
    def __init__(self, provider):
        self.provider = provider
        self.discover_calls = []
        self.resolve_calls = []

    def discover_hardware(self, *, num_qubits, prefer_hardware=None):
        self.discover_calls.append((num_qubits, prefer_hardware))
        return [
            HardwareProfile(
                provider=self.provider,
                hardware_name="chip_a",
                nqubits_available=max(2, num_qubits),
                two_qubit_gate_basis="cz",
                topology=HardwareTopology(qubits=list(range(max(2, num_qubits))), couplers=[]),
                calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=None),
                raw_info={"global_info": {"nqubits_available": max(2, num_qubits)}},
            )
        ]

    def resolve_backend(self, *, num_qubits, prefer_hardware=None):
        self.resolve_calls.append((num_qubits, prefer_hardware))
        chip = prefer_hardware[0] if isinstance(prefer_hardware, list) and prefer_hardware else "chip_a"
        return ResolvedBackend(
            provider=self.provider,
            hardware_name=chip,
            backend=types.SimpleNamespace(two_qubit_gate_basis="cz"),
            metadata={"platform_name": self.provider},
        )


class _FakeClient:
    def __init__(self):
        self.chip_name = None
        self.chip_backend = None

    @staticmethod
    def _default_qasm_version_for_provider(provider):
        provider_name = str(provider).lower()
        if provider_name in {"tianyan", "guodun"}:
            return "3.0"
        return "2.0"

    def _normalize_input_circuit(self, circuit, num_qubits):
        if isinstance(circuit, QuantumCircuit):
            return circuit
        qc = QuantumCircuit(num_qubits)
        return qc


def test_vqe_runner_run_model_supports_tianyan_provider(monkeypatch):
    import quantum_hw.algorithms.vqe as vqe_module

    seen = {}

    def _fake_runtime(*, provider, client):
        seen["provider"] = provider
        seen["client"] = client
        return types.SimpleNamespace(
            provider=provider,
            backend_adapter=_FakeBackendAdapter(provider=provider),
            task_adapter=types.SimpleNamespace(),
        )

    def _fake_run_vqe_with_backend(client, *, chip_name, target_qubits, **kwargs):
        seen["chip_name"] = chip_name
        seen["target_qubits"] = target_qubits
        seen["qasm_version"] = kwargs.get("qasm_version")
        seen["use_dd"] = kwargs.get("use_dd")
        return types.SimpleNamespace(best_energy=-1.0, energy_history=[-1.0], grad_history=[0.0])

    monkeypatch.setattr(vqe_module, "create_provider_runtime", _fake_runtime)
    monkeypatch.setattr(vqe_module, "run_vqe_with_backend", _fake_run_vqe_with_backend)

    runner = VQERunner(client=_FakeClient(), max_iters=1)
    result = runner.run_model(
        name="vqe",
        num_qubits=2,
        provider="tianyan",
        model="custom",
        hamiltonian=[(1.0, "Z0")],
    )

    assert result.best_energy == -1.0
    assert seen["provider"] == "tianyan"
    assert seen["chip_name"] == "chip_a"
    assert seen["target_qubits"] is None
    assert seen["qasm_version"] == "3.0"
    assert seen["use_dd"] is False


def test_shadow_tomography_run_supports_guodun_provider(monkeypatch):
    import quantum_hw.algorithms.shadow as shadow_module

    seen = {}

    def _fake_runtime(*, provider, client):
        seen["provider"] = provider
        seen["client"] = client
        return types.SimpleNamespace(
            provider=provider,
            backend_adapter=_FakeBackendAdapter(provider=provider),
            task_adapter=types.SimpleNamespace(),
        )

    def _fake_run_shadow_with_backend(client, qc, *, chip_name, target_qubits, **kwargs):
        del client, qc
        seen["chip_name"] = chip_name
        seen["target_qubits"] = target_qubits
        seen["qasm_version"] = kwargs.get("qasm_version")
        seen["use_dd"] = kwargs.get("use_dd")
        return types.SimpleNamespace(task_ids=[], observable_estimates={})

    monkeypatch.setattr(shadow_module, "create_provider_runtime", _fake_runtime)
    monkeypatch.setattr(shadow_module, "run_shadow_with_backend", _fake_run_shadow_with_backend)

    tomo = ShadowTomography(client=_FakeClient())
    qc = QuantumCircuit(2)
    result = tomo.run(
        qc,
        name="shadow",
        num_qubits=2,
        provider="guodun",
        shots=4,
        shots_per_basis=1,
        observables=["Z0"],
    )

    assert result.observable_estimates == {}
    assert seen["provider"] == "guodun"
    assert seen["chip_name"] == "chip_a"
    assert seen["target_qubits"] is None
    assert seen["qasm_version"] == "3.0"
    assert seen["use_dd"] is False
