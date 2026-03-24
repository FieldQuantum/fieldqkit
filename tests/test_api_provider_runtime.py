import types

from quantum_hw.api import quantum_platform as qp


class _DummyClient:
    def __init__(self):
        self.tmgr = object()


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


def test_provider_runtime_dataclass_fields_accessible():
    runtime = qp.ProviderRuntime(provider="p", backend_adapter=types.SimpleNamespace(), task_adapter=types.SimpleNamespace())
    assert runtime.provider == "p"
    assert runtime.backend_adapter is not None
    assert runtime.task_adapter is not None


def test_list_available_hardware_quafu(monkeypatch):
    import quantum_hw.api.quantum_platform.quafu as qf

    class _FakeQuafuPlatform:
        def list_available_hardware(self):
            return [
                {"provider": "quafu", "hardware_name": "chip_a", "queue_length": 1, "status": None, "is_toll": None, "raw": {"queue_length": 1}},
                {"provider": "quafu", "hardware_name": "chip_b", "queue_length": 2, "status": None, "is_toll": None, "raw": {"queue_length": 2}},
            ]

    monkeypatch.setattr(qf, "QuafuPlatform", lambda: _FakeQuafuPlatform())

    rows = qp.list_available_hardware("quafu")

    assert [row["hardware_name"] for row in rows] == ["chip_a", "chip_b"]
    assert rows[0]["provider"] == "quafu"
    assert rows[0]["queue_length"] == 1
    assert rows[0]["status"] is None
    assert rows[0]["is_toll"] is None


def test_list_available_hardware_tianyan(monkeypatch):
    import quantum_hw.api.quantum_platform.tianyan as ty

    class _FakeTianYanPlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login, machine_name

        def list_available_hardware(self):
            return [
                {"provider": "tianyan", "hardware_name": "tianyan176", "queue_length": 3, "status": "running", "is_toll": "free", "raw": {"machineName": "tianyan176"}},
                {"provider": "tianyan", "hardware_name": "tianyan24", "queue_length": None, "status": "calibration", "is_toll": "paid", "raw": {"machineName": "tianyan24"}},
            ]

    monkeypatch.setattr(ty, "get_tianyan_login_key", lambda: "k")
    monkeypatch.setattr(ty, "TianYanPlatform", _FakeTianYanPlatform)

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