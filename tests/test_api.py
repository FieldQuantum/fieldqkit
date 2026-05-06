"""Tests for the API module: exports, provider runtime, backend adapters, task adapters, and run_auto flow."""

import json
import math
import os
import types

import pytest

import quantum_hw.api as api
import quantum_hw.api.quantum_platform as qp
from quantum_hw.api import backend as bmod
from quantum_hw.api import task as ut
from quantum_hw.api.backend import (
    Backend,
    HardwareCalibration,
    HardwareProfile,
    HardwareTopology,
    ResolvedBackend,
)
from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.api.quantum_platform import ProviderRuntime
from quantum_hw.api.quantum_platform import cqlib as cq
from quantum_hw.api.quantum_platform import guodun as gd
from quantum_hw.api.quantum_platform import origin as og
from quantum_hw.api.quantum_platform import quafu as qf
from quantum_hw.api.quantum_platform import tencent as tc
from quantum_hw.api.quantum_platform import tianyan as ty
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.qasm_to_qcis import QasmToQcis
from quantum_hw.core.types import RunResult


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


class _DummyClient:
    def __init__(self):
        self.tmgr = types.SimpleNamespace(
            run=self._run,
            status=lambda tid: "Finished",
            result=lambda tid: {"count": {"0": 10}},
            cancel=self._cancel,
        )
        self.submitted_task = None
        self.canceled = None

    def _run(self, task):
        self.submitted_task = task
        return 123

    def _cancel(self, tid):
        self.canceled = tid


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


# ═══════════════════════════════════════════════════════════
#  API exports
# ═══════════════════════════════════════════════════════════


def test_api_exports_include_unified_symbols():
    expected = {
        "Backend",
        "QuantumHardwareClient",
        "ResolvedBackend",
        "HardwareTopology",
        "HardwareCalibration",
        "HardwareProfile",
        "BackendAdapter",
        "QuafuBackendAdapter",
        "TianYanBackendAdapter",
        "GuoDunBackendAdapter",
        "OpenQasmSubmitRequest",
        "ProviderTaskHandle",
        "TaskAdapter",
        "QuafuTaskAdapter",
        "TianYanTaskAdapter",
        "GuoDunTaskAdapter",
        "ProviderRuntime",
        "create_provider_runtime",
        "QuafuPlatform",
        "TianYanPlatform",
        "GuoDunPlatform",
        "QuantumLanguage",
    }
    assert expected.issubset(set(api.__all__))


def test_api_unified_symbols_are_accessible():
    assert api.Backend is not None
    assert api.ResolvedBackend is not None
    assert api.HardwareTopology is not None
    assert api.HardwareCalibration is not None
    assert api.HardwareProfile is not None
    assert api.BackendAdapter is not None
    assert api.QuafuBackendAdapter is not None
    assert api.TianYanBackendAdapter is not None
    assert api.GuoDunBackendAdapter is not None
    assert api.OpenQasmSubmitRequest is not None
    assert api.ProviderTaskHandle is not None
    assert api.TaskAdapter is not None
    assert api.QuafuTaskAdapter is not None
    assert api.TianYanTaskAdapter is not None
    assert api.GuoDunTaskAdapter is not None
    assert api.ProviderRuntime is not None
    assert api.create_provider_runtime is not None
    assert api.QuafuPlatform is not None
    assert api.TianYanPlatform is not None
    assert api.GuoDunPlatform is not None
    assert api.QuantumLanguage is not None


def test_api_exports_include_tencent_symbols():
    assert "TencentBackendAdapter" in api.__all__
    assert "TencentTaskAdapter" in api.__all__
    assert "TencentPlatform" in api.__all__
    assert api.TencentBackendAdapter is not None
    assert api.TencentTaskAdapter is not None
    assert api.TencentPlatform is not None


def test_api_exports_include_origin_symbols():
    assert "OriginPlatform" in api.__all__
    assert "OriginBackendAdapter" in api.__all__
    assert "OriginTaskAdapter" in api.__all__
    assert api.OriginPlatform is og.OriginPlatform
    assert api.OriginBackendAdapter is og.OriginBackendAdapter
    assert api.OriginTaskAdapter is og.OriginTaskAdapter


# ═══════════════════════════════════════════════════════════
#  Provider runtime creation
# ═══════════════════════════════════════════════════════════


def test_create_provider_runtime_for_quafu(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "QuafuBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "QuafuTaskAdapter", lambda client: dummy_task)

    client = _DummyClient()
    runtime = module.create_provider_runtime(provider="quafu", client=client)

    assert runtime.provider == "quafu"
    assert runtime.backend_adapter is dummy_backend
    assert runtime.task_adapter is dummy_task


def test_create_provider_runtime_for_tianyan(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "TianYanBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "TianYanTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="tianyan", client=_DummyClient())

    assert runtime.provider == "tianyan"
    assert runtime.backend_adapter is dummy_backend
    assert runtime.task_adapter is dummy_task


def test_create_provider_runtime_for_guodun(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "GuoDunBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "GuoDunTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="guodun", client=_DummyClient())

    assert runtime.provider == "guodun"
    assert runtime.backend_adapter is dummy_backend
    assert runtime.task_adapter is dummy_task


def test_create_provider_runtime_for_tencent(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "TencentBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "TencentTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="tencent", client=_DummyClient())

    assert runtime.provider == "tencent"
    assert runtime.backend_adapter is dummy_backend
    assert runtime.task_adapter is dummy_task


def test_create_provider_runtime_invalid_provider_raises():
    client = _DummyClient()
    try:
        qp.create_provider_runtime(provider="x", client=client)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "provider must be one of" in str(exc)


def test_create_provider_runtime_provider_name_is_case_insensitive(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "QuafuBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "QuafuTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="QuAfU", client=_DummyClient())
    assert runtime.provider == "quafu"


def test_create_provider_runtime_tencent_case_insensitive(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "TencentBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "TencentTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="Tencent", client=_DummyClient())
    assert runtime.provider == "tencent"


def test_create_provider_runtime_for_origin(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()
    monkeypatch.setattr(module, "OriginBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "OriginTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="origin", client=_DummyClient())
    assert runtime.provider == "origin"
    assert runtime.backend_adapter is dummy_backend
    assert runtime.task_adapter is dummy_task


def test_create_provider_runtime_origin_case_insensitive(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()
    monkeypatch.setattr(module, "OriginBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "OriginTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="Origin", client=_DummyClient())
    assert runtime.provider == "origin"


def test_provider_runtime_dataclass_fields_accessible():
    runtime = qp.ProviderRuntime(provider="p", backend_adapter=types.SimpleNamespace(), task_adapter=types.SimpleNamespace())
    assert runtime.provider == "p"
    assert runtime.backend_adapter is not None
    assert runtime.task_adapter is not None


# ═══════════════════════════════════════════════════════════
#  Hardware discovery
# ═══════════════════════════════════════════════════════════


def test_list_available_hardware_quafu(monkeypatch):
    import quantum_hw.api.quantum_platform.quafu as _qf

    class _FakeQuafuPlatform:
        def list_available_hardware(self):
            return [
                {"provider": "quafu", "hardware_name": "chip_a", "queue_length": 1, "status": None, "is_toll": None, "raw": {"queue_length": 1}},
                {"provider": "quafu", "hardware_name": "chip_b", "queue_length": 2, "status": None, "is_toll": None, "raw": {"queue_length": 2}},
            ]

    monkeypatch.setattr(_qf, "QuafuPlatform", lambda: _FakeQuafuPlatform())

    rows = qp.list_available_hardware("quafu")

    assert [row["hardware_name"] for row in rows] == ["chip_a", "chip_b"]
    assert rows[0]["provider"] == "quafu"
    assert rows[0]["queue_length"] == 1
    assert rows[0]["status"] is None
    assert rows[0]["is_toll"] is None


def test_list_available_hardware_tianyan(monkeypatch):
    import quantum_hw.api.quantum_platform.tianyan as _ty
    import quantum_hw.api.platform_credentials as _creds

    class _FakeTianYanPlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login, machine_name

        def list_available_hardware(self):
            return [
                {"provider": "tianyan", "hardware_name": "tianyan176", "queue_length": 3, "status": "running", "is_toll": "free", "raw": {"machineName": "tianyan176"}},
                {"provider": "tianyan", "hardware_name": "tianyan24", "queue_length": None, "status": "calibration", "is_toll": "paid", "raw": {"machineName": "tianyan24"}},
            ]

    monkeypatch.setattr(_creds, "get_tianyan_api_token", lambda: "k")
    monkeypatch.setattr(_ty, "get_tianyan_api_token", lambda: "k")
    monkeypatch.setattr(_ty, "TianYanPlatform", _FakeTianYanPlatform)

    rows = qp.list_available_hardware("tianyan")

    assert len(rows) == 2
    assert rows[0]["provider"] == "tianyan"
    assert rows[0]["hardware_name"] == "tianyan176"
    assert rows[0]["queue_length"] == 3
    assert rows[0]["status"] == "running"
    assert rows[0]["is_toll"] == "free"


def test_list_available_hardware_invalid_provider_raises():
    try:
        qp.list_available_hardware("x")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "provider must be one of" in str(exc)


# ═══════════════════════════════════════════════════════════
#  Backend adapters
# ═══════════════════════════════════════════════════════════


def test_quafu_backend_adapter_resolve_backend_success(monkeypatch):
    class _FakePlatform:
        def list_available_hardware(self):
            return [
                {"provider": "quafu", "hardware_name": "chip_b", "queue_length": 1, "status": None, "is_toll": None, "raw": {"queue_length": 1}},
                {"provider": "quafu", "hardware_name": "chip_a", "queue_length": 2, "status": None, "is_toll": None, "raw": {"queue_length": 2}},
            ]

    monkeypatch.setattr(bmod, "Backend", lambda chip: {"chip": chip})
    monkeypatch.setattr(
        bmod,
        "build_hardware_profile",
        lambda **kwargs: HardwareProfile(
            provider="quafu",
            hardware_name=kwargs["hardware_name"],
            nqubits_available=5,
            two_qubit_gate_basis="cz",
            topology=HardwareTopology(qubits=[0, 1, 2, 3, 4], couplers=[]),
            calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=1),
            raw_info={},
        ),
    )

    platform_obj = _FakePlatform()
    adapter = qf.QuafuBackendAdapter(platform_obj=platform_obj)
    resolved = adapter.resolve_backend(num_qubits=5, prefer_hardware=["chip_b", "chip_a"])

    assert resolved.provider == "quafu"
    assert resolved.hardware_name == "chip_b"
    assert resolved.backend == {"chip": "chip_b"}
    assert resolved.metadata["platform_obj"] is platform_obj


def test_backend_adapter_discovery_requires_bound_platform():
    class _Adapter(bmod.BackendAdapter):
        provider = "quafu"

    with pytest.raises(RuntimeError, match="requires a bound platform"):
        _Adapter().list_available_hardware()


def test_quafu_backend_adapter_discovery_uses_bound_platform(monkeypatch):
    class _FakePlatform:
        def list_available_hardware(self):
            return [
                {"provider": "quafu", "hardware_name": "chip_bound", "queue_length": 2, "status": None, "is_toll": None, "raw": {"queue_length": 2}}
            ]

    monkeypatch.setattr(bmod, "list_available_hardware", lambda provider: (_ for _ in ()).throw(AssertionError(f"unexpected fallback for {provider}")))
    monkeypatch.setattr(bmod, "Backend", lambda chip: {"chip": chip})
    monkeypatch.setattr(
        bmod,
        "build_hardware_profile",
        lambda **kwargs: HardwareProfile(
            provider="quafu",
            hardware_name=kwargs["hardware_name"],
            nqubits_available=5,
            two_qubit_gate_basis="cz",
            topology=HardwareTopology(qubits=[0, 1, 2, 3, 4], couplers=[]),
            calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=kwargs["queue_length"]),
            raw_info={},
        ),
    )

    adapter = qf.QuafuBackendAdapter(platform_obj=_FakePlatform())
    profiles = adapter.discover_hardware(num_qubits=2)

    assert [profile.hardware_name for profile in profiles] == ["chip_bound"]
    assert profiles[0].calibration.queue_length == 2


def test_tianyan_backend_adapter_selects_platform_and_resolves(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            self.login_key = login_key
            self.machine_name = machine_name
            self.set_machine_called = None

        def set_machine(self, machine_name):
            self.set_machine_called = machine_name

        def query_quantum_computer_list(self):
            return [["machineName"], ["tianyan176"]]

        def download_config(self, machine):
            return {"machine": machine}

        def list_available_hardware(self):
            return [
                {"provider": "tianyan", "hardware_name": "tianyan176", "queue_length": None, "status": "running", "is_toll": "free", "raw": {"machineName": "tianyan176"}}
            ]

    monkeypatch.setattr(ty, "TianYanPlatform", _FakePlatform)
    monkeypatch.setattr(ty, "get_tianyan_api_token", lambda: "k")
    monkeypatch.setattr(bmod, "Backend", lambda chip: {"chip": chip})
    monkeypatch.setattr(
        bmod,
        "build_hardware_profile",
        lambda **kwargs: HardwareProfile(
            provider="tianyan",
            hardware_name=kwargs["hardware_name"],
            nqubits_available=4,
            two_qubit_gate_basis="cz",
            topology=HardwareTopology(qubits=[0, 1, 2, 3], couplers=[]),
            calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=None),
            raw_info={"machine": kwargs["hardware_name"]},
        ),
    )

    adapter = ty.TianYanBackendAdapter(machine_name="m0")
    resolved = adapter.resolve_backend(num_qubits=4, prefer_hardware="tianyan176")

    assert resolved.provider == "tianyan"
    assert resolved.hardware_name == "tianyan176"
    assert resolved.backend == {"chip": "tianyan176"}
    assert resolved.metadata["platform_obj"] is adapter._platform


def test_guodun_backend_adapter_selects_platform_and_resolves(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            self.login_key = login_key
            self.machine_name = machine_name
            self.set_machine_called = None

        def set_machine(self, machine_name):
            self.set_machine_called = machine_name

        def query_quantum_computer_list(self):
            return [["machineName"], ["gd_qc1"]]

        def download_config(self, machine):
            return {"machine": machine}

        def list_available_hardware(self):
            return [
                {"provider": "guodun", "hardware_name": "gd_qc1", "queue_length": None, "status": "running", "is_toll": "free", "raw": {"machineName": "gd_qc1"}}
            ]

    monkeypatch.setattr(gd, "GuoDunPlatform", _FakePlatform)
    monkeypatch.setattr(gd, "get_guodun_api_token", lambda: "k")
    monkeypatch.setattr(bmod, "Backend", lambda chip: {"chip": chip})
    monkeypatch.setattr(
        bmod,
        "build_hardware_profile",
        lambda **kwargs: HardwareProfile(
            provider="guodun",
            hardware_name=kwargs["hardware_name"],
            nqubits_available=2,
            two_qubit_gate_basis="cz",
            topology=HardwareTopology(qubits=[0, 1], couplers=[]),
            calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=None),
            raw_info={"machine": kwargs["hardware_name"]},
        ),
    )

    adapter = gd.GuoDunBackendAdapter(machine_name="m2")
    resolved = adapter.resolve_backend(num_qubits=1)

    assert resolved.provider == "guodun"
    assert resolved.hardware_name == "gd_qc1"
    assert resolved.backend == {"chip": "gd_qc1"}
    assert resolved.metadata["platform_obj"] is adapter._platform


def test_tianyan_backend_adapter_supports_simulator_preference(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login, machine_name

    monkeypatch.setattr(ty, "TianYanPlatform", _FakePlatform)
    monkeypatch.setattr(ty, "get_tianyan_api_token", lambda: "k")

    adapter = ty.TianYanBackendAdapter(machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=3, prefer_hardware="Simulator")
    assert resolved.hardware_name == "Simulator"


def test_guodun_backend_adapter_supports_simulator_preference(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login, machine_name

    monkeypatch.setattr(gd, "GuoDunPlatform", _FakePlatform)
    monkeypatch.setattr(gd, "get_guodun_api_token", lambda: "k")

    adapter = gd.GuoDunBackendAdapter(machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=2, prefer_hardware="Simulator")
    assert resolved.hardware_name == "Simulator"


def test_tianyan_backend_adapter_discovery_uses_bound_platform(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login, machine_name

        def list_available_hardware(self):
            return [
                {"provider": "tianyan", "hardware_name": "tianyan176", "queue_length": 3, "status": "running", "is_toll": "free", "raw": {"machineName": "tianyan176"}}
            ]

    monkeypatch.setattr(ty, "TianYanPlatform", _FakePlatform)
    monkeypatch.setattr(ty, "get_tianyan_api_token", lambda: "k")
    monkeypatch.setattr(bmod, "list_available_hardware", lambda provider: (_ for _ in ()).throw(AssertionError(f"unexpected fallback for {provider}")))
    monkeypatch.setattr(bmod, "Backend", lambda chip: {"chip": chip})
    monkeypatch.setattr(
        bmod,
        "build_hardware_profile",
        lambda **kwargs: HardwareProfile(
            provider="tianyan",
            hardware_name=kwargs["hardware_name"],
            nqubits_available=4,
            two_qubit_gate_basis="cz",
            topology=HardwareTopology(qubits=[0, 1, 2, 3], couplers=[]),
            calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=kwargs["queue_length"]),
            raw_info={},
        ),
    )

    adapter = ty.TianYanBackendAdapter(api_token="custom_key")
    profiles = adapter.discover_hardware(num_qubits=2)

    assert [profile.hardware_name for profile in profiles] == ["tianyan176"]
    assert profiles[0].calibration.queue_length == 3


def test_tianyan_backend_adapter_does_not_request_extra_overview(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login
            self.machine_name = machine_name

        def query_quantum_computer_list(self):
            return [["machineName"], ["tianyan176"]]

        def download_config(self, machine):
            return {"machine": machine, "overview": {"coupler_map": {"G0": ["Q0", "Q1"]}, "qubits": ["Q0", "Q1"]}, "disabledQubits": "", "disabledCouplers": "", "twoQubitGate": {"czGate": {}}}

        def get_machine_config(self, params):
            del params
            raise AssertionError("get_machine_config should not be called")

        def set_machine(self, machine_name):
            self.machine_name = machine_name

        def list_available_hardware(self):
            return [
                {"provider": "tianyan", "hardware_name": "tianyan176", "queue_length": None, "status": "running", "is_toll": "free", "raw": {"machineName": "tianyan176"}}
            ]

    monkeypatch.setattr(ty, "TianYanPlatform", _FakePlatform)
    monkeypatch.setattr(ty, "get_tianyan_api_token", lambda: "k")

    adapter = ty.TianYanBackendAdapter(machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=2)

    assert resolved.hardware_name == "tianyan176"


def test_guodun_backend_adapter_does_not_request_extra_overview(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login
            self.machine_name = machine_name

        def query_quantum_computer_list(self):
            return [["machineName"], ["gd_qc1"]]

        def download_config(self, machine):
            return {"machine": machine, "overview": {"coupler_map": {"G0": ["Q0", "Q1"]}, "qubits": ["Q0", "Q1"]}, "disabledQubits": "", "disabledCouplers": "", "twoQubitGate": {"czGate": {}}}

        def get_machine_config(self, params):
            del params
            raise AssertionError("get_machine_config should not be called")

        def set_machine(self, machine_name):
            self.machine_name = machine_name

        def list_available_hardware(self):
            return [
                {"provider": "guodun", "hardware_name": "gd_qc1", "queue_length": None, "status": "running", "is_toll": "free", "raw": {"machineName": "gd_qc1"}}
            ]

    monkeypatch.setattr(gd, "GuoDunPlatform", _FakePlatform)
    monkeypatch.setattr(gd, "get_guodun_api_token", lambda: "k")
    monkeypatch.setattr(bmod, "Backend", lambda chip: {"chip": chip})
    monkeypatch.setattr(
        bmod,
        "build_hardware_profile",
        lambda **kwargs: HardwareProfile(
            provider="guodun",
            hardware_name=kwargs["hardware_name"],
            nqubits_available=2,
            two_qubit_gate_basis="cz",
            topology=HardwareTopology(qubits=[0, 1], couplers=[]),
            calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=None),
            raw_info={"machine": kwargs["hardware_name"]},
        ),
    )

    adapter = gd.GuoDunBackendAdapter(machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=2)

    assert resolved.hardware_name == "gd_qc1"


def test_cqlib_parser_supports_fsim_value_and_download_config_shape():
    chip_info = cq.chip_info_from_config(
        {
            "disabledQubits": "Q3",
            "disabledCouplers": "G1",
            "twoQubitGate": {"fsim_value": {}},
            "overview": {
                "qubits": ["Q0", "Q1", "Q2", "Q3"],
                "coupler_map": {
                    "G0": ["Q0", "Q1"],
                    "G1": ["Q1", "Q2"],
                    "G2": ["Q0", "Q2"],
                },
            },
        },
        machine_name="m",
    )
    backend = Backend(chip_info)

    assert backend.two_qubit_gate_basis == "fsim"
    assert 3 not in [int(q[1:]) for q in backend.chip_info["qubits_info"].keys()]
    assert len(backend.chip_info["couplers_info"]) == 2


def test_quafu_loader_preserves_qubit_coordinate(monkeypatch):
    payload = {
        "qubits_info": {
            "Q0": {"fidelity": 0.99, "coordinate": [1, 2]},
            "Q1": {"fidelity": 0.98, "coordinate": {"x": 3, "y": 4}},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.95},
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }

    class _Resp:
        def __init__(self, data):
            self.content = json.dumps(data).encode()

    class _Session:
        def get(self, url):
            del url
            return _Resp(payload)

    monkeypatch.setattr(qf.requests, "Session", _Session)

    chip_info = qf.load_quafu_chip_info("Baihua")
    assert chip_info["qubits_info"]["Q0"]["coordinate"] == [1.0, 2.0]
    assert chip_info["qubits_info"]["Q1"]["coordinate"] == [3.0, 4.0]


def test_quafu_loader_filters_low_fidelity_coupler(monkeypatch):
    payload = {
        "qubits_info": {
            "Q0": {"fidelity": 0.99},
            "Q1": {"fidelity": 0.98},
            "Q2": {"fidelity": 0.97},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.89},
            "C1": {"qubits_index": [1, 2], "fidelity": 0.91},
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }

    class _Resp:
        def __init__(self, data):
            self.content = json.dumps(data).encode()

    class _Session:
        def get(self, url):
            del url
            return _Resp(payload)

    monkeypatch.setattr(qf.requests, "Session", _Session)

    chip_info = qf.load_quafu_chip_info("Baihua")

    assert "C0" not in chip_info["couplers_info"]
    assert chip_info["couplers_info"]["C1"]["qubits_index"] == [1, 2]


# ═══════════════════════════════════════════════════════════
#  Task adapters
# ═══════════════════════════════════════════════════════════


def test_quafu_task_adapter_submit_query_fetch_cancel_lifecycle():
    client = _DummyClient()
    adapter = qf.QuafuTaskAdapter(client=client)
    backend = ResolvedBackend(
        provider="quafu",
        hardware_name="chip",
        backend="backend_obj",
        metadata={"platform_obj": client.tmgr},
    )

    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(name="n", qasm="OPENQASM 2.0;", shots=100, chip_name="chip"),
        backend,
    )

    assert isinstance(handle, ut.ProviderTaskHandle)
    assert handle.provider == "quafu"
    assert client.submitted_task["name"] == "n"
    assert client.submitted_task["chip"] == "chip"

    assert adapter.query_status(handle) == "Finished"
    assert adapter.fetch_result(handle) == {"count": {"0": 10}}

    adapter.cancel_task(handle)
    assert client.canceled == 123


def test_vendored_adapter_submit_openqasm_submit_job_and_fetch_result(monkeypatch):
    class _FakeConverter:
        def convert_to_qcis(self, qasm):
            return f"QCIS::{qasm}"

    import quantum_hw.circuit.qasm_to_qcis as _qcis_mod
    monkeypatch.setattr(_qcis_mod, "QasmToQcis", _FakeConverter)

    class _Platform:
        def __init__(self):
            self.last_submit = None

        def submit_job(self, **kwargs):
            self.last_submit = kwargs
            return ["qid1"]

        def query_experiment(self, query_id, max_wait_time, sleep_time):
            del max_wait_time, sleep_time
            assert query_id == "qid1"
            return [{"resultStatus": [[0], [1], [0], [1]]}]

        def stop_running_experiments(self, query_id=None):
            self.stopped = query_id

    platform = _Platform()
    backend = ResolvedBackend(
        provider="tianyan",
        hardware_name="m",
        backend="b",
        metadata={"platform_obj": platform},
    )

    adapter = ty.TianYanTaskAdapter(client=_DummyClient(), api_token="k")
    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(
            name="exp",
            qasm="OPENQASM 2.0;",
            shots=20,
            chip_name="m",
            submit_options={"num_qubits": 1},
        ),
        backend,
    )

    assert handle.task_id == "qid1"
    assert platform.last_submit["exp_name"] == "exp"
    assert platform.last_submit["circuit"].startswith("QCIS::")

    status = adapter.query_status(handle)
    assert status == "Finished"

    result = adapter.fetch_result(handle)
    assert result["count"] == {"1": 2, "0": 1}

    adapter.cancel_task(handle)
    assert getattr(platform, "stopped", None) == "qid1"


def test_tianyan_adapter_provider_value():
    adapter = ty.TianYanTaskAdapter(client=_DummyClient(), api_token="k")
    assert adapter.provider == "tianyan"


def test_guodun_adapter_provider_value():
    adapter = gd.GuoDunTaskAdapter(client=_DummyClient(), api_token="k")
    assert adapter.provider == "guodun"


def test_guodun_task_adapter_submit_query_fetch_cancel_lifecycle(monkeypatch):
    class _FakeConverter:
        def convert_to_qcis(self, qasm):
            return f"QCIS::{qasm}"

    import quantum_hw.circuit.qasm_to_qcis as _qcis_mod
    monkeypatch.setattr(_qcis_mod, "QasmToQcis", _FakeConverter)

    class _Platform:
        def __init__(self):
            self.last_submit = None
            self.stopped = None

        def submit_job(self, **kwargs):
            self.last_submit = kwargs
            return ["gd_qid1"]

        def query_experiment(self, query_id, max_wait_time, sleep_time):
            return [{"resultStatus": [[0], [0], [1], [0], [1]]}]

        def stop_running_experiments(self, query_id=None):
            self.stopped = query_id

    platform = _Platform()
    backend = ResolvedBackend(
        provider="guodun",
        hardware_name="gd_qc1",
        backend="b",
        metadata={"platform_obj": platform},
    )

    adapter = gd.GuoDunTaskAdapter(client=_DummyClient(), api_token="k")
    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(
            name="gd_exp",
            qasm="OPENQASM 2.0;",
            shots=50,
            chip_name="gd_qc1",
            submit_options={"num_qubits": 1},
        ),
        backend,
    )

    assert handle.provider == "guodun"
    assert handle.task_id == "gd_qid1"
    assert platform.last_submit["exp_name"] == "gd_exp"
    assert platform.last_submit["circuit"].startswith("QCIS::")

    status = adapter.query_status(handle)
    assert status == "Finished"

    result = adapter.fetch_result(handle)
    assert isinstance(result["count"], dict)
    assert sum(result["count"].values()) > 0

    adapter.cancel_task(handle)
    assert platform.stopped == "gd_qid1"


def test_tianyan_task_adapter_submit_query_fetch_lifecycle(monkeypatch):
    class _FakeConverter:
        def convert_to_qcis(self, qasm):
            return f"QCIS::{qasm}"

    import quantum_hw.circuit.qasm_to_qcis as _qcis_mod
    monkeypatch.setattr(_qcis_mod, "QasmToQcis", _FakeConverter)

    class _Platform:
        def __init__(self):
            self.last_submit = None
            self.stopped = None

        def submit_job(self, **kwargs):
            self.last_submit = kwargs
            return ["ty_qid1", "ty_qid2"]

        def query_experiment(self, query_id, max_wait_time, sleep_time):
            return [
                {"resultStatus": [[0], [1], [0]]},
                {"resultStatus": [[0], [0], [1]]},
            ]

        def stop_running_experiments(self, query_id=None):
            self.stopped = query_id

    platform = _Platform()
    backend = ResolvedBackend(
        provider="tianyan",
        hardware_name="tianyan176",
        backend="b",
        metadata={"platform_obj": platform},
    )

    adapter = ty.TianYanTaskAdapter(client=_DummyClient(), api_token="k")
    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(
            name="ty_exp",
            qasm="OPENQASM 2.0;",
            shots=100,
            chip_name="tianyan176",
            submit_options={"num_qubits": 1},
        ),
        backend,
    )

    assert handle.provider == "tianyan"
    assert handle.task_id == "ty_qid1,ty_qid2"
    assert platform.last_submit["exp_name"] == "ty_exp"

    status = adapter.query_status(handle)
    assert status == "Finished"

    result = adapter.fetch_result(handle)
    assert isinstance(result["count"], dict)

    adapter.cancel_task(handle)
    assert platform.stopped is not None


def test_tencent_task_adapter_submit_query_fetch_lifecycle(monkeypatch):
    submitted_tasks = []

    class _FakePlatform:
        def submit_task(self, source, device_name, shots=1024):
            submitted_tasks.append({"source": source, "device": device_name, "shots": shots})
            return "task-abc-123"

        def query_task_state(self, task_id, device_name):
            return "completed"

        def fetch_task_result(self, task_id, device_name):
            return {"10": 500, "01": 30, "00": 470, "11": 24}

    backend = ResolvedBackend(
        provider="tencent",
        hardware_name="tianji_s2",
        backend="backend_obj",
        metadata={"platform_obj": _FakePlatform()},
    )

    monkeypatch.setattr(tc, "_get_tencent_token", lambda: "fake_token")
    monkeypatch.setattr(tc, "_ensure_token", lambda token=None: "fake_token")

    adapter = tc.TencentTaskAdapter(client=_DummyClient(), token="fake_token")

    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(
            name="test_ghz",
            qasm="OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n",
            shots=1024,
            chip_name="tianji_s2",
            submit_options={"num_qubits": 2},
        ),
        backend,
    )

    assert isinstance(handle, ut.ProviderTaskHandle)
    assert handle.provider == "tencent"
    assert handle.task_id == "task-abc-123"
    assert len(submitted_tasks) == 1
    assert submitted_tasks[0]["device"] == "tianji_s2"
    assert submitted_tasks[0]["shots"] == 1024

    status = adapter.query_status(handle)
    assert status == "Finished"

    result = adapter.fetch_result(handle)
    assert "count" in result
    counts = result["count"]
    assert counts["10"] == 500
    assert counts["01"] == 30
    assert counts["00"] == 470
    assert counts["11"] == 24


def test_tencent_task_adapter_pending_maps_to_running(monkeypatch):
    class _FakePlatform:
        def query_task_state(self, task_id, device_name):
            return "pending"

    monkeypatch.setattr(tc, "_get_tencent_token", lambda: "fake_token")
    monkeypatch.setattr(tc, "_ensure_token", lambda token=None: "fake_token")

    adapter = tc.TencentTaskAdapter(client=_DummyClient(), token="fake_token")
    handle = ut.ProviderTaskHandle(
        provider="tencent",
        task_id="task-xyz",
        payload={"platform_obj": _FakePlatform(), "device_name": "tianji_s2"},
    )

    status = adapter.query_status(handle)
    assert status == "Running"


def test_tencent_task_adapter_failed_maps_to_failed(monkeypatch):
    class _FakePlatform:
        def query_task_state(self, task_id, device_name):
            return "failed"

    monkeypatch.setattr(tc, "_get_tencent_token", lambda: "fake_token")
    monkeypatch.setattr(tc, "_ensure_token", lambda token=None: "fake_token")

    adapter = tc.TencentTaskAdapter(client=_DummyClient(), token="fake_token")
    handle = ut.ProviderTaskHandle(
        provider="tencent",
        task_id="task-xyz",
        payload={"platform_obj": _FakePlatform(), "device_name": "tianji_s2"},
    )

    status = adapter.query_status(handle)
    assert status == "Failed"


# ═══════════════════════════════════════════════════════════
#  QASM / QCIS conversion
# ═══════════════════════════════════════════════════════════


def test_qasm3_delay_can_convert_to_qcis_idle_instruction():
    qasm = """
OPENQASM 3.0;
include \"stdgates.inc\";
qubit[1] q;
bit[1] c;
delay[5] q[0];
c[0] = measure q[0];
"""
    qcis = QasmToQcis().convert_to_qcis(qasm)
    lines = [line.strip().upper() for line in qcis.splitlines() if line.strip()]
    assert any(line.startswith("I Q0 ") for line in lines)
    assert any(line == "M Q0" for line in lines)


def test_qasm3_generated_with_defcalgrammar_and_delay_can_convert_to_qcis():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.delay(2e-6, 0)
    qc.measure([0], [0])

    qcis = QasmToQcis().convert_to_qcis(qc.to_openqasm3)
    lines = [line.strip().upper() for line in qcis.splitlines() if line.strip()]

    assert any(line.startswith("I Q0 ") for line in lines)
    assert any(line == "M Q0" for line in lines)


def test_qasm3_duration_literal_ns_converts_to_seconds_in_qcis_delay():
    qasm = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
delay[5ns] q[0];
"""
    qcis = QasmToQcis().convert_to_qcis(qasm)
    first_line = [line.strip() for line in qcis.splitlines() if line.strip()][0]
    duration = float(first_line.split()[-1])
    assert math.isclose(duration, 5e-9, rel_tol=0.0, abs_tol=1e-15)


def test_quantumcircuit_delay_unit_argument_is_normalized_to_seconds():
    qc = QuantumCircuit(1)
    qc.delay(5, 0, unit="ns")
    gate = qc.gates[-1]
    assert gate[0] == "delay"
    assert math.isclose(gate[1], 5e-9, rel_tol=0.0, abs_tol=1e-15)


def test_qasm_to_qcis_basic_conversion():
    qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""
    qcis = QasmToQcis().convert_to_qcis(qasm)
    lines = [line.strip().upper() for line in qcis.splitlines() if line.strip()]
    assert any("Q0" in line for line in lines)
    assert any("CZ" in line for line in lines)


def test_cqlib_request_error_has_status_code():
    from quantum_hw.api.quantum_platform.cqlib import CqlibRequestError

    exc = CqlibRequestError("test error", status_code=401)
    assert exc.status_code == 401
    assert "401" in exc.message

    exc_no_code = CqlibRequestError("generic error")
    assert exc_no_code.status_code is None


def test_assign_parameters_inline():
    from quantum_hw.api.quantum_platform.cqlib import _assign_parameters

    circuits = ["X2P Q0\nRZ Q0 {THETA}"]
    result = _assign_parameters(circuits, [["theta"]], [[1.5]])
    assert "1.5" in result[0]
    assert "{" not in result[0]


# ═══════════════════════════════════════════════════════════
#  run_auto routing
# ═══════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════
#  _normalize_input_circuit measurement handling
# ═══════════════════════════════════════════════════════════


class TestNormalizeMeasurements:
    """Verify _normalize_input_circuit preserves or strips measurements
    depending on whether observables are provided."""

    def _make_client(self):
        return QuantumHardwareClient()

    # -- observables + measurements → warn and strip --

    def test_measurements_stripped_when_observables_provided(self):
        client = self._make_client()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        assert client._has_measurements(qc)

        result = client._normalize_input_circuit(qc, 3, observables=["Z0", "X1"])
        assert not client._has_measurements(result)

    def test_warning_emitted_when_observables_conflict(self, caplog):
        import logging
        client = self._make_client()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        with caplog.at_level(logging.WARNING):
            client._normalize_input_circuit(qc, 2, observables=["Z0"])
        assert any("conflict" in r.message.lower() for r in caplog.records)

    # -- no observables + measurements → preserve --

    def test_measurements_preserved_when_no_observables(self):
        client = self._make_client()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        result = client._normalize_input_circuit(qc, 3, observables=None)
        assert client._has_measurements(result)

    def test_measurements_preserved_when_observables_empty_list(self):
        client = self._make_client()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result = client._normalize_input_circuit(qc, 2, observables=[])
        assert client._has_measurements(result)

    # -- no measurements + no observables → nothing to strip --

    def test_no_measurements_and_no_observables(self):
        client = self._make_client()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = client._normalize_input_circuit(qc, 2, observables=None)
        assert not client._has_measurements(result)

    # -- partial measurements with scrambled cbit order, no observables --

    def test_partial_measure_scrambled_cbits_preserved(self):
        """Circuit measures only qubits 0 and 2 (out of 4), mapping them
        to classical bits in a non-trivial order (cbit 1 ← qubit 0,
        cbit 0 ← qubit 2).  Without observables the measure gate must
        be kept exactly as the user specified."""
        client = self._make_client()
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        # Scrambled cbit mapping: qubit 0 → cbit 1, qubit 2 → cbit 0
        qc.measure([0, 2], [1, 0])

        result = client._normalize_input_circuit(qc, 4, observables=None)
        # Measurement must survive
        assert client._has_measurements(result)
        # Find the measure gate and verify qubit/cbit mapping unchanged
        measure_gates = [g for g in result.gates if g[0] == "measure"]
        assert len(measure_gates) == 1
        gate_name, qubits, cbits = measure_gates[0]
        assert qubits == [0, 2], f"measured qubits changed: {qubits}"
        assert cbits == [1, 0], f"cbit mapping changed: {cbits}"

    def test_partial_measure_scrambled_cbits_stripped_with_observables(self):
        """Same scrambled partial measurement but WITH observables —
        the user-specified measure must be stripped."""
        client = self._make_client()
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.measure([0, 2], [1, 0])

        result = client._normalize_input_circuit(qc, 4, observables=["Z0 Z2"])
        assert not client._has_measurements(result)

    # -- run_auto integration: qc forwarded to _run_with_backend keeps measurements --

    def test_run_auto_preserves_user_measurements_no_observables(self):
        """Run a partial-measurement circuit through the real simulator path
        (chip_name='simulator', transpile=False) and verify the results.

        Circuit: H(0)→CX(0,1) produces (|000⟩+|110⟩)/√2.
        User adds measure([0, 2], [1, 0]) — partial and scrambled:
            qubit 0 → cbit 1, qubit 2 → cbit 0.
        Without observables the pipeline must keep the user measure gates,
        project simulator counts to the 2-cbit subspace, and return
        2-bit samples with the scrambled cbit ordering.

        Expected outcomes (cbit order [c0, c1]):
            |000⟩ → c0=q2=0, c1=q0=0 → [0, 0]
            |110⟩ → c0=q2=0, c1=q0=1 → [0, 1]
        """
        client = QuantumHardwareClient()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        # Partial measure with scrambled cbit order: qubit 0→cbit 1, qubit 2→cbit 0
        qc.measure([0, 2], [1, 0])

        # Normalize — measurements must survive without observables
        qc_norm = client._normalize_input_circuit(qc, 3, observables=None)
        assert client._has_measurements(qc_norm)
        measure_gates = [g for g in qc_norm.gates if g[0] == "measure"]
        assert measure_gates[0][1] == [0, 2]
        assert measure_gates[0][2] == [1, 0]

        # Run through the real _run_with_backend using the simulator
        result = client._run_with_backend(
            qc_norm,
            name="sim_partial_meas",
            num_qubits=3,
            backend=Backend("simulator"),
            chip_name="simulator",
            shots=2048,
            transpile=False,
            observables=None,
            return_probabilities=True,
            print_true=False,
        )

        # Samples should be 2-bit (only 2 cbits from the partial measure).
        assert result.samples is not None and len(result.samples) == 1
        samples = result.samples[0]  # list of [c0, c1] rows
        assert len(samples[0]) == 2, f"expected 2-bit samples, got {len(samples[0])}-bit"

        unique_rows = {tuple(row) for row in samples}
        # |000⟩ → [0, 0], |110⟩ → [0, 1]  (qubit 2 is always 0 → c0 always 0)
        assert unique_rows <= {(0, 0), (0, 1)}, f"unexpected outcomes: {unique_rows}"

        # Probabilities over 2 cbits → length 4 (2**2)
        probs = result.probabilities[0]
        assert len(probs) == 4, f"expected 4 probabilities, got {len(probs)}"
        # P("00")≈0.5, P("01")≈0.5, the rest zero
        assert probs[0] > 0.3   # P(c0c1=00) ≈ 0.5
        assert probs[1] > 0.3   # P(c0c1=01) ≈ 0.5
        assert probs[2] + probs[3] < 1e-9  # no outcomes with c0=1

    def test_run_auto_strips_measurements_with_observables(self, monkeypatch):
        backend_adapter = _FakeBackendAdapter(provider="quafu")
        task_adapter = _FakeTaskAdapter()
        _install_runtime_mocks(monkeypatch, backend_adapter=backend_adapter, task_adapter=task_adapter)
        run_seen = _install_run_with_backend_mock(monkeypatch)

        client = QuantumHardwareClient()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        client.run_auto(
            circuit=qc,
            name="strip_meas",
            num_qubits=2,
            provider="quafu",
            observables=["Z0"],
            print_true=False,
        )

        forwarded_qc = run_seen["qc"]
        assert not QuantumHardwareClient._has_measurements(forwarded_qc)

    def test_empty_circuit_with_z_observable_expectation_is_one(self):
        """Empty circuit (|000⟩ state) with ZZZ observable should yield expectation 1.0."""
        client = QuantumHardwareClient()
        qc = QuantumCircuit(3)
        # No gates — state is |000⟩

        result = client._run_with_backend(
            qc,
            name="empty_zzz",
            num_qubits=3,
            backend=Backend("simulator"),
            chip_name="simulator",
            shots=1024,
            observables=["ZZZ"],
            transpile=False,
        )

        assert result.observable_values["ZZZ"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════
#  OriginQ (本源量子) provider — fixtures + tests
# ═══════════════════════════════════════════════════════════


class _OriginFakeJobStatus:
    def __init__(self, name: str) -> None:
        self.name = name


class _OriginFakeQubitInfo:
    def __init__(self, qid: int) -> None:
        self._qid = qid

    def get_qubit_id(self) -> str:
        return str(self._qid)

    def get_frequency(self) -> float:
        return 4000.0 + self._qid

    def get_readout_fidelity(self) -> float:
        return 0.95

    def get_single_gate_fidelity(self) -> float:
        return 0.999

    def get_t1(self) -> float:
        return 30.0

    def get_t2(self) -> float:
        return 5.0


class _OriginFakeDoubleQubitsInfo:
    def __init__(self, a: int, b: int, fidelity: float = 0.97) -> None:
        self._pair = [a, b]
        self._fid = fidelity

    def get_qubits(self):
        return list(self._pair)

    def get_fidelity(self) -> float:
        return self._fid


class _OriginFakeChipInfo:
    def __init__(self, *, nq: int = 4, edges=((0, 1), (1, 2), (2, 3))):
        self._nq = nq
        self._edges = list(edges)

    def chip_id(self) -> str:
        return "FAKE_CHIP"

    def qubits_num(self) -> int:
        return self._nq

    def available_qubits(self):
        return list(range(self._nq))

    def get_basic_gates(self):
        return ["RPhi", "CZ"]

    def get_chip_topology(self, qubits=None):
        return [list(e) for e in self._edges]

    def single_qubit_info(self):
        return [_OriginFakeQubitInfo(i) for i in range(self._nq)]

    def double_qubits_info(self):
        return [_OriginFakeDoubleQubitsInfo(a, b) for a, b in self._edges]

    def get_single_gate_timing(self) -> int:
        return 30

    def get_double_gate_timing(self) -> int:
        return 40

    def high_frequency_qubits(self):
        return []


class _OriginFakeQCloudResult:
    def __init__(self, counts, status="FINISHED", err=""):
        self._counts = dict(counts)
        self._status = _OriginFakeJobStatus(status)
        self._err = err

    def get_counts(self):
        return dict(self._counts)

    def job_status(self):
        return self._status

    def error_message(self) -> str:
        return self._err


class _OriginFakeQCloudJob:
    _registry: dict = {}

    def __init__(self, job_id: str) -> None:
        self._jid = str(job_id)

    def job_id(self) -> str:
        return self._jid

    def status(self):
        prog = self._registry.get(self._jid, {}).get("statuses", ["FINISHED"])
        if len(prog) > 1:
            prog.pop(0)
        return _OriginFakeJobStatus(prog[0])

    def result(self):
        return self._registry[self._jid]["result"]


class _OriginFakeQCloudOptions:
    def set_amend(self, _):
        pass
    def set_mapping(self, _):
        pass
    def set_optimization(self, _):
        pass
    def set_is_prob_counts(self, _):
        pass


class _OriginFakeQCloudBackend:
    last_run = None

    def __init__(self, name: str) -> None:
        self._name = name
        self._chip = _OriginFakeChipInfo()

    def name(self) -> str:
        return self._name

    def chip_info(self) -> _OriginFakeChipInfo:
        return self._chip

    def run(self, prog, shots, options=None, *args, **kwargs):
        type(self).last_run = {
            "prog": prog,
            "shots": int(shots),
            "options": options,
            "args": args,
            "kwargs": kwargs,
        }
        jid = f"FAKEJOB-{int(shots)}"
        _OriginFakeQCloudJob._registry[jid] = {
            "statuses": ["WAITING", "COMPUTING", "FINISHED"],
            "result": _OriginFakeQCloudResult({"00": int(shots) // 2, "11": int(shots) // 2}),
        }
        return _OriginFakeQCloudJob(jid)


class _OriginFakeQCloudService:
    def __init__(self, api_key: str, url: str = "fake://"):
        self._key = api_key
        self._url = url

    def backends(self):
        return {
            "PQPUMESH8": True,
            "WK_C180": True,
            "HanYuan_01": False,
            "full_amplitude": True,
        }

    def backend(self, name: str) -> _OriginFakeQCloudBackend:
        return _OriginFakeQCloudBackend(name)


def _build_origin_fake_qcloud_module():
    return types.SimpleNamespace(
        QCloudService=_OriginFakeQCloudService,
        QCloudBackend=_OriginFakeQCloudBackend,
        QCloudJob=_OriginFakeQCloudJob,
        QCloudResult=_OriginFakeQCloudResult,
        QCloudOptions=_OriginFakeQCloudOptions,
        JobStatus=_OriginFakeJobStatus,
    )


@pytest.fixture
def fake_origin_sdk(monkeypatch):
    """Patch the lazy SDK loaders so origin tests are fully offline."""
    fake_mod = _build_origin_fake_qcloud_module()
    monkeypatch.setattr(og, "_import_qcloud", lambda: fake_mod)
    monkeypatch.setattr(og, "_import_qasm_to_qprog", lambda: (lambda s: s))
    monkeypatch.setattr(og, "_get_origin_token", lambda: "fake-token")
    yield fake_mod


def test_origin_chip_names_registered_in_provider_inference():
    from quantum_hw.api.backend import infer_provider_from_chip
    assert infer_provider_from_chip("PQPUMESH8") == "origin"
    assert infer_provider_from_chip("WK_C180") == "origin"
    assert infer_provider_from_chip("HanYuan_01") == "origin"


def test_origin_credential_helper_reads_yaml(monkeypatch, tmp_path):
    from quantum_hw.api import platform_credentials as pc

    cfg = tmp_path / ".quantum_hw.yaml"
    cfg.write_text(
        "credentials:\n  origin:\n    api_token: 'my-origin-token'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANTUM_HW_CONFIG", str(cfg))
    pc.reload_config()
    try:
        assert pc.get_origin_api_token() == "my-origin-token"
    finally:
        pc.reload_config()


def test_origin_platform_list_available_hardware_filters_simulators(fake_origin_sdk):
    p = og.OriginPlatform(token="fake-token")
    rows = p.list_available_hardware()
    names = {r["hardware_name"] for r in rows}
    assert "PQPUMESH8" in names
    assert "WK_C180" in names
    assert "HanYuan_01" in names
    assert "full_amplitude" not in names
    statuses = {r["hardware_name"]: r["status"] for r in rows}
    assert statuses["PQPUMESH8"] == "online"
    assert statuses["HanYuan_01"] == "offline"


def test_load_origin_chip_info_normalizes_unified_layout(fake_origin_sdk):
    info = og.load_origin_chip_info("PQPUMESH8", token="fake-token")
    assert info["chip_name"] == "PQPUMESH8"
    assert info["global_info"]["two_qubit_gate_basis"] == "cz"
    assert info["global_info"]["nqubits_available"] == 4
    assert set(info["qubits_info"].keys()) == {"Q0", "Q1", "Q2", "Q3"}
    pairs = sorted(tuple(c["qubits_index"]) for c in info["couplers_info"].values())
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    fid = next(iter(info["couplers_info"].values()))["fidelity"]
    assert 0.0 < fid <= 1.0


def test_backend_loads_origin_chip_via_sdk(fake_origin_sdk):
    from quantum_hw.api.backend import Backend
    b = Backend("PQPUMESH8")
    assert b.chip_name == "PQPUMESH8"
    assert b.two_qubit_gate_basis == "cz"
    assert len(b.qubits_with_attributes) == 4
    assert len(b.couplers_with_attributes) == 3


def test_origin_status_map():
    assert og._map_status(_OriginFakeJobStatus("FINISHED")) == "Finished"
    assert og._map_status(_OriginFakeJobStatus("FAILED")) == "Failed"
    assert og._map_status(_OriginFakeJobStatus("WAITING")) == "Running"
    assert og._map_status(_OriginFakeJobStatus("COMPUTING")) == "Running"
    assert og._map_status(_OriginFakeJobStatus("QUEUING")) == "Running"
    assert og._map_status(_OriginFakeJobStatus("BOGUS")) == "Running"


def test_origin_platform_submit_query_and_fetch(fake_origin_sdk):
    p = og.OriginPlatform(token="fake-token")
    qasm = (
        'OPENQASM 2.0;\nincludeXX "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        "h q[0];\ncx q[0],q[1];\nmeasure q[0]->c[0];\nmeasure q[1]->c[1];\n"
    ).replace("includeXX", "include")
    job_id = p.submit_task(source=qasm, device_name="PQPUMESH8", shots=64)
    assert job_id == "FAKEJOB-64"

    seen = [p.query_task_state(job_id, "PQPUMESH8") for _ in range(3)]
    assert seen[0] == "Running"
    assert seen[-1] == "Finished"
    assert "Finished" in seen

    counts = p.fetch_task_result(job_id, "PQPUMESH8")
    assert counts == {"00": 32, "11": 32}


def test_origin_platform_fetch_raises_when_failed(fake_origin_sdk):
    p = og.OriginPlatform(token="fake-token")
    _OriginFakeQCloudJob._registry["FAILJOB"] = {
        "statuses": ["FAILED"],
        "result": _OriginFakeQCloudResult({}, status="FAILED", err="reason"),
    }
    with pytest.raises(RuntimeError, match="not finished"):
        p.fetch_task_result("FAILJOB", "PQPUMESH8")


def test_origin_task_adapter_submit_and_query(fake_origin_sdk):
    from quantum_hw.api.task import OpenQasmSubmitRequest

    platform_obj = og.OriginPlatform(token="fake-token")
    resolved = ResolvedBackend(
        provider="origin",
        hardware_name="PQPUMESH8",
        backend=object(),
        metadata={"platform_obj": platform_obj},
    )
    adapter = og.OriginTaskAdapter(client=None, token="fake-token")

    req = OpenQasmSubmitRequest(
        name="t",
        qasm='OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0]->c[0];\n',
        shots=128,
        chip_name="PQPUMESH8",
        submit_options={"num_qubits": 1},
    )
    handle = adapter.submit_openqasm(req, resolved)
    assert handle.provider == "origin"
    assert handle.task_id == "FAKEJOB-128"
    assert handle.payload["device_name"] == "PQPUMESH8"
    assert handle.payload["platform_obj"] is platform_obj

    statuses = [adapter.query_status(handle) for _ in range(3)]
    assert statuses[-1] == "Finished"
    out = adapter.fetch_result(handle)
    assert out == {"count": {"00": 64, "11": 64}}


def test_origin_task_adapter_creates_platform_when_missing(fake_origin_sdk):
    from quantum_hw.api.task import OpenQasmSubmitRequest

    resolved = ResolvedBackend(
        provider="origin",
        hardware_name="PQPUMESH8",
        backend=object(),
        metadata={},
    )
    adapter = og.OriginTaskAdapter(client=None, token="fake-token")
    req = OpenQasmSubmitRequest(name="t", qasm="qasm", shots=8, chip_name="PQPUMESH8", submit_options={})
    handle = adapter.submit_openqasm(req, resolved)
    assert handle.task_id == "FAKEJOB-8"


def test_origin_cancel_task_logs_warning(fake_origin_sdk, caplog):
    handle = api.ProviderTaskHandle(provider="origin", task_id="X")
    adapter = og.OriginTaskAdapter(client=None, token="fake-token")
    with caplog.at_level("WARNING", logger=og.__name__):
        adapter.cancel_task(handle)
    assert any("does not support task cancellation" in m for m in caplog.messages)


def test_list_available_hardware_dispatches_to_origin(monkeypatch, fake_origin_sdk):
    from quantum_hw.api import backend as bmod
    rows = bmod.list_available_hardware("origin")
    names = {r["hardware_name"] for r in rows}
    assert {"PQPUMESH8", "WK_C180", "HanYuan_01"} <= names
