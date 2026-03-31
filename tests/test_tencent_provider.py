"""Tests for Tencent provider adapter (tencent.py)."""

import types

import quantum_hw.api.quantum_platform as qp
from quantum_hw.api.quantum_platform import tencent as tc
from quantum_hw.api import task as ut
from quantum_hw.api.backend import ResolvedBackend


class _DummyClient:
    def __init__(self):
        self.tmgr = object()


# ── Provider runtime integration ────────────────────────────────────────


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


def test_create_provider_runtime_tencent_case_insensitive(monkeypatch):
    import quantum_hw.api.quantum_platform as module

    dummy_backend = object()
    dummy_task = object()

    monkeypatch.setattr(module, "TencentBackendAdapter", lambda: dummy_backend)
    monkeypatch.setattr(module, "TencentTaskAdapter", lambda client: dummy_task)

    runtime = module.create_provider_runtime(provider="Tencent", client=_DummyClient())
    assert runtime.provider == "tencent"


# ── TencentPlatform ─────────────────────────────────────────────────────


def test_tencent_platform_list_available_hardware(monkeypatch):
    class _FakeDevice:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(tc.tc_apis, "list_devices", lambda provider: [_FakeDevice("tianji_s2"), _FakeDevice("tianxuan_s2")])
    monkeypatch.setattr(tc, "_ensure_token", lambda token=None: "fake_token")

    platform = tc.TencentPlatform.__new__(tc.TencentPlatform)
    platform._token = "fake_token"

    rows = platform.list_available_hardware()
    assert len(rows) == 2
    assert rows[0]["provider"] == "tencent"
    assert rows[0]["hardware_name"] == "tianji_s2"
    assert rows[1]["hardware_name"] == "tianxuan_s2"


# ── TencentTaskAdapter ──────────────────────────────────────────────────


def test_tencent_task_adapter_submit_query_fetch_lifecycle(monkeypatch):
    submitted_tasks = []
    queried_states = {}
    fetched_results = {}

    class _FakePlatform:
        def submit_task(self, source, device_name, shots=1024):
            submitted_tasks.append({"source": source, "device": device_name, "shots": shots})
            return "task-abc-123"

        def query_task_state(self, task_id, device_name):
            return "completed"

        def fetch_task_result(self, task_id, device_name):
            # Big-endian: q[0] is leftmost
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

    # Query status
    status = adapter.query_status(handle)
    assert status == "Finished"

    # Fetch result — should flip bitstrings
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


# ── API exports ──────────────────────────────────────────────────────────


def test_api_exports_include_tencent_symbols():
    import quantum_hw.api as api

    assert "TencentBackendAdapter" in api.__all__
    assert "TencentTaskAdapter" in api.__all__
    assert "TencentPlatform" in api.__all__
    assert api.TencentBackendAdapter is not None
    assert api.TencentTaskAdapter is not None
    assert api.TencentPlatform is not None
