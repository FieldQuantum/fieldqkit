from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.api.quantum_platform import ProviderRuntime
from quantum_hw.api.backend import ResolvedBackend
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.core.types import RunResult


class _FakeBackendAdapter:
    def __init__(self, *, provider="quafu"):
        self.provider = provider
        self.calls = []

    def resolve_backend(self, *, num_qubits, prefer_hardware=None):
        self.calls.append(
            {
                "num_qubits": num_qubits,
                "prefer_hardware": prefer_hardware,
            }
        )
        return ResolvedBackend(
            provider=self.provider,
            hardware_name="fake_machine",
            backend={"fake": True},
            metadata={"platform_obj": object()},
        )


class _FakeTaskAdapter:
    pass


def _install_runtime_mocks(monkeypatch, *, backend_adapter, task_adapter):
    import quantum_hw.api.client as client_module

    seen = {}

    def fake_create_provider_runtime(*, provider, client):
        seen["provider"] = provider
        seen["client"] = client
        return ProviderRuntime(provider=provider, backend_adapter=backend_adapter, task_adapter=task_adapter)

    monkeypatch.setattr(client_module, "create_provider_runtime", fake_create_provider_runtime)
    return seen


def _install_run_with_backend_mock(monkeypatch):
    seen = {}

    def fake_run_with_backend(self, qc, name, num_qubits, **kwargs):
        seen["qc"] = qc
        seen["name"] = name
        seen["num_qubits"] = num_qubits
        seen["kwargs"] = kwargs
        return RunResult(
            task_ids=["rid"],
            samples=[[[0]]],
            samples_zne=None,
            probabilities=[[1.0]],
            probabilities_raw=[[1.0]],
            observable_values={"Z0": 1.0},
            observable_values_raw={"Z0": 1.0},
        )

    monkeypatch.setattr(QuantumHardwareClient, "_run_with_backend", fake_run_with_backend)
    return seen


def test_run_auto_quafu_routes_to_quafu_adapters(monkeypatch):
    backend_adapter = _FakeBackendAdapter(provider="quafu")
    task_adapter = _FakeTaskAdapter()
    seen = _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)
    run_seen = _install_run_with_backend_mock(monkeypatch)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(1)

    res = client.run_auto(
        circuit=qc,
        name="n1",
        num_qubits=1,
        provider="quafu",
        shots=321,
        observables=["Z0"],
        return_probabilities=True,
        prefer_chips=["A", "B"],
        print_true=False,
    )

    assert res.task_ids == ["rid"]
    assert client.chip_name == "fake_machine"
    assert client.chip_backend == {"fake": True}
    assert seen["provider"] == "quafu"
    assert backend_adapter.calls[0]["prefer_hardware"] == ["A", "B"]
    assert run_seen["name"] == "n1"
    assert run_seen["kwargs"]["qasm_version"] == "2.0"
    assert run_seen["kwargs"]["use_dd"] is True


def test_run_auto_tianyan_routes_to_tianyan_runtime(monkeypatch):
    backend_adapter = _FakeBackendAdapter(provider="tianyan")
    task_adapter = _FakeTaskAdapter()
    seen = _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)
    run_seen = _install_run_with_backend_mock(monkeypatch)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(2)

    client.run_auto(circuit=qc, name="job_ty", num_qubits=2, provider="tianyan", print_true=False)

    assert seen["provider"] == "tianyan"
    assert backend_adapter.calls[0]["num_qubits"] == 2
    assert run_seen["kwargs"]["qasm_version"] == "3.0"
    assert run_seen["kwargs"]["use_dd"] is False


def test_run_auto_guodun_routes_to_guodun_runtime(monkeypatch):
    backend_adapter = _FakeBackendAdapter(provider="guodun")
    task_adapter = _FakeTaskAdapter()
    seen = _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)
    run_seen = _install_run_with_backend_mock(monkeypatch)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(2)

    client.run_auto(circuit=qc, name="job_gd", num_qubits=2, provider="guodun", print_true=False)

    assert seen["provider"] == "guodun"
    assert backend_adapter.calls[0]["num_qubits"] == 2
    assert run_seen["kwargs"]["qasm_version"] == "3.0"
    assert run_seen["kwargs"]["use_dd"] is False


def test_run_auto_invalid_provider_raises():
    from quantum_hw.api import quantum_platform as runtime_module

    client = QuantumHardwareClient()

    try:
        runtime_module.create_provider_runtime(provider="unknown", client=client)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "provider must be one of" in str(exc)


def test_run_auto_sets_client_backend_state(monkeypatch):
    backend_adapter = _FakeBackendAdapter(provider="quafu")
    task_adapter = _FakeTaskAdapter()
    _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)
    _install_run_with_backend_mock(monkeypatch)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(1)

    client.run_auto(circuit=qc, name="state", num_qubits=1, provider="quafu", print_true=False)

    assert client.chip_name == "fake_machine"
    assert client.chip_backend == {"fake": True}


def test_run_auto_passes_target_qubits_and_flags_to_run_flow(monkeypatch):
    backend_adapter = _FakeBackendAdapter(provider="quafu")
    task_adapter = _FakeTaskAdapter()
    run_seen = _install_run_with_backend_mock(monkeypatch)
    _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(3)

    client.run_auto(
        circuit=qc,
        name="flags",
        num_qubits=3,
        provider="quafu",
        zne=True,
        readout_mitigation=True,
        readout_shots=77,
        target_qubits=[0, 2, 1],
        print_true=False,
    )

    assert run_seen["kwargs"]["zne"] is True
    assert run_seen["kwargs"]["readout_mitigation"] is True
    assert run_seen["kwargs"]["readout_shots"] == 77
    assert run_seen["kwargs"]["target_qubits"] == [0, 2, 1]


def test_run_auto_request_contains_provider_options(monkeypatch):
    backend_adapter = _FakeBackendAdapter(provider="quafu")
    task_adapter = _FakeTaskAdapter()
    run_seen = _install_run_with_backend_mock(monkeypatch)
    _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(1)

    client.run_auto(
        circuit=qc,
        name="opts",
        num_qubits=1,
        provider="quafu",
        transpile_on_client=True,
        max_wait_time=12,
        sleep_time=9,
        print_true=False,
    )

    assert run_seen["kwargs"]["transpile"] is True
    assert run_seen["kwargs"]["submit_options"]["max_wait_time"] == 12
    assert run_seen["kwargs"]["submit_options"]["sleep_time"] == 9