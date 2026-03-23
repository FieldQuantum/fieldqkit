from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.api.unified_backend import ResolvedBackend
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.core.types import RunResult


class _FakeBackendAdapter:
    def __init__(self, *, tmgr=None, login_key=None, platform=None):
        self.tmgr = tmgr
        self.login_key = login_key
        self.platform = platform
        self.calls = []

    def resolve_backend(self, *, num_qubits, prefer_hardware=None, rank_weights=None):
        self.calls.append(
            {
                "num_qubits": num_qubits,
                "prefer_hardware": prefer_hardware,
                "rank_weights": rank_weights,
            }
        )
        return ResolvedBackend(
            provider="quafu" if self.login_key is None else "cqlib",
            hardware_name="fake_machine",
            backend={"fake": True},
            target_qubits=[9, 8, 7],
            metadata={"platform_name": "tianyan", "machine_name": "fake_machine"},
        )


class _FakeTaskAdapter:
    def __init__(self, *, client=None, login_key=None):
        self.client = client
        self.login_key = login_key
        self.calls = []

    def run_task(self, request, backend):
        self.calls.append({"request": request, "backend": backend})
        return RunResult(
            task_ids=["rid"],
            samples=[[[0]]],
            samples_zne=None,
            probabilities=[[1.0]],
            probabilities_raw=[[1.0]],
            observable_values={"Z0": 1.0},
            observable_values_raw={"Z0": 1.0},
        )


def test_run_auto_quafu_routes_to_quafu_adapters(monkeypatch):
    import quantum_hw.api.client as client_module

    backend_adapter = _FakeBackendAdapter(tmgr=object())
    task_adapter = _FakeTaskAdapter(client=object())

    monkeypatch.setattr(client_module, "QuafuBackendAdapter", lambda tmgr: backend_adapter)
    monkeypatch.setattr(client_module, "QuafuTaskAdapter", lambda client: task_adapter)

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
        rank_weights={"error": 1.0},
        print_true=False,
    )

    assert res.task_ids == ["rid"]
    assert client.chip_name == "fake_machine"
    assert client.chip_backend == {"fake": True}
    assert backend_adapter.calls[0]["prefer_hardware"] == ["A", "B"]
    assert backend_adapter.calls[0]["rank_weights"] == {"error": 1.0}
    req = task_adapter.calls[0]["request"]
    assert req.name == "n1"
    assert req.shots == 321
    assert req.observables == ["Z0"]


def test_run_auto_cqlib_routes_to_cqlib_adapters(monkeypatch):
    import quantum_hw.api.client as client_module

    backend_adapter = _FakeBackendAdapter(login_key="lk", platform="tianyan")
    task_adapter = _FakeTaskAdapter(login_key="lk")

    monkeypatch.setattr(client_module, "CqlibBackendAdapter", lambda login_key, platform: backend_adapter)
    monkeypatch.setattr(client_module, "CqlibTaskAdapter", lambda login_key: task_adapter)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(2)

    client.run_auto(
        circuit=qc,
        name="job_cq",
        num_qubits=2,
        provider="cqlib",
        cqlib_platform="tianyan",
        cqlib_submit_mode="submit_experiment",
        cqlib_transpile=False,
        cqlib_max_wait_time=88,
        cqlib_sleep_time=3,
        prefer_chips="tianyan176",
        print_true=False,
    )

    req = task_adapter.calls[0]["request"]
    assert req.provider_options["submit_mode"] == "submit_experiment"
    assert req.provider_options["transpile_on_client"] is False
    assert req.provider_options["max_wait_time"] == 88
    assert req.provider_options["sleep_time"] == 3
    assert backend_adapter.calls[0]["prefer_hardware"] == "tianyan176"


def test_run_auto_invalid_provider_raises():
    client = QuantumHardwareClient()
    qc = QuantumCircuit(1)

    try:
        client.run_auto(circuit=qc, name="bad", num_qubits=1, provider="unknown")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "provider must be 'quafu' or 'cqlib'" in str(exc)


def test_run_auto_sets_client_backend_state(monkeypatch):
    import quantum_hw.api.client as client_module

    backend_adapter = _FakeBackendAdapter(tmgr=object())
    task_adapter = _FakeTaskAdapter(client=object())
    monkeypatch.setattr(client_module, "QuafuBackendAdapter", lambda tmgr: backend_adapter)
    monkeypatch.setattr(client_module, "QuafuTaskAdapter", lambda client: task_adapter)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(1)

    client.run_auto(circuit=qc, name="state", num_qubits=1, provider="quafu", print_true=False)

    assert client.chip_name == "fake_machine"
    assert client.chip_backend == {"fake": True}


def test_run_auto_passes_target_qubits_and_flags_to_task_request(monkeypatch):
    import quantum_hw.api.client as client_module

    backend_adapter = _FakeBackendAdapter(tmgr=object())
    task_adapter = _FakeTaskAdapter(client=object())
    monkeypatch.setattr(client_module, "QuafuBackendAdapter", lambda tmgr: backend_adapter)
    monkeypatch.setattr(client_module, "QuafuTaskAdapter", lambda client: task_adapter)

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

    req = task_adapter.calls[0]["request"]
    assert req.zne is True
    assert req.readout_mitigation is True
    assert req.readout_shots == 77
    assert req.target_qubits == [0, 2, 1]


def test_run_auto_request_contains_cqlib_options_even_for_quafu(monkeypatch):
    import quantum_hw.api.client as client_module

    backend_adapter = _FakeBackendAdapter(tmgr=object())
    task_adapter = _FakeTaskAdapter(client=object())
    monkeypatch.setattr(client_module, "QuafuBackendAdapter", lambda tmgr: backend_adapter)
    monkeypatch.setattr(client_module, "QuafuTaskAdapter", lambda client: task_adapter)

    client = QuantumHardwareClient()
    qc = QuantumCircuit(1)

    client.run_auto(
        circuit=qc,
        name="opts",
        num_qubits=1,
        provider="quafu",
        cqlib_submit_mode="submit_job",
        cqlib_transpile=True,
        cqlib_max_wait_time=12,
        cqlib_sleep_time=9,
        print_true=False,
    )

    req = task_adapter.calls[0]["request"]
    assert req.provider_options["submit_mode"] == "submit_job"
    assert req.provider_options["transpile_on_client"] is True
    assert req.provider_options["max_wait_time"] == 12
    assert req.provider_options["sleep_time"] == 9
