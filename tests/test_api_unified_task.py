import types

import pytest

from quantum_hw.api import unified_task as ut
from quantum_hw.api.unified_backend import ResolvedBackend
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.core.types import RunResult


class _DummyClient:
    def __init__(self):
        self.called = None

    def _run_with_backend(self, *args, **kwargs):
        self.called = {"args": args, "kwargs": kwargs}
        return RunResult(
            task_ids=["t1"],
            samples=[[[0]]],
            samples_zne=None,
            probabilities=[[1.0]],
            probabilities_raw=[[1.0]],
            observable_values={"Z0": 1.0},
            observable_values_raw={"Z0": 1.0},
        )


class _FakeCqlibAdapter:
    def __init__(self, *, login_key, platform, machine_name, submit_mode):
        self.ctor = {
            "login_key": login_key,
            "platform": platform,
            "machine_name": machine_name,
            "submit_mode": submit_mode,
        }

    def run(self, qc, **kwargs):
        return RunResult(
            task_ids=["cq1"],
            samples=[[[1]]],
            samples_zne=None,
            probabilities=[[0.0, 1.0]],
            probabilities_raw=[[0.0, 1.0]],
            observable_values={"Z0": -1.0},
            observable_values_raw={"Z0": -1.0},
        )


def _build_request(**overrides):
    req = ut.TaskRequest(
        qc=QuantumCircuit(1),
        name="job",
        num_qubits=1,
        shots=1024,
        zne=False,
        readout_mitigation=False,
        readout_shots=None,
        observables=["Z0"],
        return_probabilities=True,
        target_qubits=None,
        print_true=False,
        provider_options={},
    )
    for k, v in overrides.items():
        setattr(req, k, v)
    return req


def test_quafu_task_adapter_forwards_explicit_target_qubits():
    client = _DummyClient()
    adapter = ut.QuafuTaskAdapter(client=client)
    backend = ResolvedBackend(provider="quafu", hardware_name="chip", backend="backend_obj", target_qubits=[3])
    request = _build_request(target_qubits=[7], shots=2048)

    res = adapter.run_task(request, backend)

    assert res.task_ids == ["t1"]
    assert client.called["kwargs"]["target_qubits"] == [7]
    assert client.called["kwargs"]["shots"] == 2048
    assert client.called["kwargs"]["chip_name"] == "chip"


def test_quafu_task_adapter_falls_back_to_backend_target_qubits():
    client = _DummyClient()
    adapter = ut.QuafuTaskAdapter(client=client)
    backend = ResolvedBackend(provider="quafu", hardware_name="chip", backend="backend_obj", target_qubits=[1, 2])
    request = _build_request(target_qubits=None)

    adapter.run_task(request, backend)

    assert client.called["kwargs"]["target_qubits"] == [1, 2]


def test_quafu_task_adapter_passes_flags_and_observables():
    client = _DummyClient()
    adapter = ut.QuafuTaskAdapter(client=client)
    backend = ResolvedBackend(provider="quafu", hardware_name="chip", backend="backend_obj", target_qubits=None)
    request = _build_request(zne=True, readout_mitigation=True, readout_shots=300, observables=["Z0", "Z1"])

    adapter.run_task(request, backend)

    kwargs = client.called["kwargs"]
    assert kwargs["zne"] is True
    assert kwargs["readout_mitigation"] is True
    assert kwargs["readout_shots"] == 300
    assert kwargs["observables"] == ["Z0", "Z1"]


def test_cqlib_task_adapter_rejects_zne():
    adapter = ut.CqlibTaskAdapter(login_key="k")
    backend = ResolvedBackend(provider="cqlib", hardware_name="m", backend="b")
    request = _build_request(zne=True)

    with pytest.raises(ValueError, match="does not support zne"):
        adapter.run_task(request, backend)


def test_cqlib_task_adapter_rejects_readout_mitigation():
    adapter = ut.CqlibTaskAdapter(login_key="k")
    backend = ResolvedBackend(provider="cqlib", hardware_name="m", backend="b")
    request = _build_request(readout_mitigation=True)

    with pytest.raises(ValueError, match="does not support readout mitigation"):
        adapter.run_task(request, backend)


def test_cqlib_task_adapter_rejects_explicit_target_qubits():
    adapter = ut.CqlibTaskAdapter(login_key="k")
    backend = ResolvedBackend(provider="cqlib", hardware_name="m", backend="b")
    request = _build_request(target_qubits=[0])

    with pytest.raises(ValueError, match="does not support explicit target_qubits"):
        adapter.run_task(request, backend)


def test_cqlib_task_adapter_uses_defaults(monkeypatch):
    import quantum_hw.api.cqlib_adapter as cqlib_adapter_module

    seen = {}

    class FakeAdapter(_FakeCqlibAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            seen["ctor"] = self.ctor

        def run(self, qc, **kwargs):
            seen["run_kwargs"] = kwargs
            return super().run(qc, **kwargs)

    monkeypatch.setattr(cqlib_adapter_module, "CqlibAdapter", FakeAdapter)

    adapter = ut.CqlibTaskAdapter(login_key="key123")
    backend = ResolvedBackend(provider="cqlib", hardware_name="machine_x", backend="b", metadata={})
    request = _build_request(provider_options={})

    res = adapter.run_task(request, backend)

    assert res.task_ids == ["cq1"]
    assert seen["ctor"]["login_key"] == "key123"
    assert seen["ctor"]["platform"] == "tianyan"
    assert seen["ctor"]["machine_name"] == "machine_x"
    assert seen["ctor"]["submit_mode"] == "submit_job"
    assert seen["run_kwargs"]["transpile_on_client"] is True
    assert seen["run_kwargs"]["max_wait_time"] == 3600
    assert seen["run_kwargs"]["sleep_time"] == 5


def test_cqlib_task_adapter_uses_metadata_and_provider_options(monkeypatch):
    import quantum_hw.api.cqlib_adapter as cqlib_adapter_module

    seen = {}

    class FakeAdapter(_FakeCqlibAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            seen["ctor"] = self.ctor

        def run(self, qc, **kwargs):
            seen["run_kwargs"] = kwargs
            return super().run(qc, **kwargs)

    monkeypatch.setattr(cqlib_adapter_module, "CqlibAdapter", FakeAdapter)

    adapter = ut.CqlibTaskAdapter(login_key="lk")
    backend = ResolvedBackend(
        provider="cqlib",
        hardware_name="fallback_machine",
        backend="b",
        metadata={"platform_name": "guodun", "machine_name": "m777"},
    )
    request = _build_request(
        provider_options={
            "submit_mode": "submit_experiment",
            "transpile_on_client": False,
            "max_wait_time": "12",
            "sleep_time": "1",
        }
    )

    adapter.run_task(request, backend)

    assert seen["ctor"]["platform"] == "guodun"
    assert seen["ctor"]["machine_name"] == "m777"
    assert seen["ctor"]["submit_mode"] == "submit_experiment"
    assert seen["run_kwargs"]["transpile_on_client"] is False
    assert seen["run_kwargs"]["max_wait_time"] == 12
    assert seen["run_kwargs"]["sleep_time"] == 1
