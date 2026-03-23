import types

import pytest

from quantum_hw.api import unified_backend as ub


class _DummyPlatform:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _DummyPlatform.instances.append(self)


def _install_fake_cqlib(monkeypatch):
    qp = types.ModuleType("cqlib.quantum_platform")

    class FakeTianYanPlatform(_DummyPlatform):
        pass

    class FakeGuoDunPlatform(_DummyPlatform):
        pass

    qp.TianYanPlatform = FakeTianYanPlatform
    qp.GuoDunPlatform = FakeGuoDunPlatform

    cqlib_mod = types.ModuleType("cqlib")
    cqlib_mod.quantum_platform = qp

    monkeypatch.setitem(__import__("sys").modules, "cqlib", cqlib_mod)
    monkeypatch.setitem(__import__("sys").modules, "cqlib.quantum_platform", qp)
    return FakeTianYanPlatform, FakeGuoDunPlatform


def test_quafu_backend_adapter_resolve_backend_success(monkeypatch):
    called = {}

    def fake_rank_chips(tmgr, num_qubits, prefer_chips, weights):
        called["args"] = (tmgr, num_qubits, prefer_chips, weights)
        return ["chip_a", "chip_b"]

    monkeypatch.setattr(ub, "rank_chips", fake_rank_chips)
    monkeypatch.setattr(ub, "Backend", lambda chip: {"chip": chip})

    tmgr = object()
    adapter = ub.QuafuBackendAdapter(tmgr=tmgr)
    resolved = adapter.resolve_backend(num_qubits=5, prefer_hardware=["chip_a"], rank_weights={"queue": 1.0})

    assert resolved.provider == "quafu"
    assert resolved.hardware_name == "chip_a"
    assert resolved.backend == {"chip": "chip_a"}
    assert resolved.metadata["ranked_chips"] == ["chip_a", "chip_b"]
    assert called["args"] == (tmgr, 5, ["chip_a"], {"queue": 1.0})


def test_quafu_backend_adapter_no_candidates_raises(monkeypatch):
    monkeypatch.setattr(ub, "rank_chips", lambda *args, **kwargs: [])
    adapter = ub.QuafuBackendAdapter(tmgr=object())
    with pytest.raises(RuntimeError, match="no available chips"):
        adapter.resolve_backend(num_qubits=2)


def test_quafu_backend_adapter_passes_preferences_and_weights(monkeypatch):
    seen = {}

    def fake_rank_chips(tmgr, num_qubits, prefer_chips, weights):
        seen["prefer"] = prefer_chips
        seen["weights"] = weights
        return ["chip_only"]

    monkeypatch.setattr(ub, "rank_chips", fake_rank_chips)
    monkeypatch.setattr(ub, "Backend", lambda chip: chip)

    adapter = ub.QuafuBackendAdapter(tmgr=object())
    adapter.resolve_backend(num_qubits=3, prefer_hardware="Simulator", rank_weights={"error": 0.7})

    assert seen["prefer"] == "Simulator"
    assert seen["weights"] == {"error": 0.7}


def test_cqlib_backend_adapter_requires_login_key():
    with pytest.raises(ValueError, match="login key cannot be empty"):
        ub.CqlibBackendAdapter(login_key="")


def test_cqlib_backend_adapter_selects_tianyan_platform_class(monkeypatch):
    FakeTianYanPlatform, _ = _install_fake_cqlib(monkeypatch)
    _DummyPlatform.instances.clear()

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="tianyan", machine_name="m1")

    assert isinstance(adapter._platform_obj, FakeTianYanPlatform)
    assert adapter._platform_obj.kwargs["login_key"] == "k"
    assert adapter._platform_obj.kwargs["machine_name"] == "m1"


def test_cqlib_backend_adapter_selects_guodun_platform_class(monkeypatch):
    _, FakeGuoDunPlatform = _install_fake_cqlib(monkeypatch)
    _DummyPlatform.instances.clear()

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="guodun", machine_name="m2")

    assert isinstance(adapter._platform_obj, FakeGuoDunPlatform)
    assert adapter._platform_obj.kwargs["machine_name"] == "m2"


def test_cqlib_backend_adapter_resolve_backend_prefers_str_hardware(monkeypatch):
    _install_fake_cqlib(monkeypatch)
    called = {}

    def fake_bundle(platform_obj, machine_name, num_qubits):
        called["machine_name"] = machine_name
        called["num_qubits"] = num_qubits
        return types.SimpleNamespace(backend="b", target_qubits=[1, 2])

    monkeypatch.setattr(ub, "build_cqlib_backend_bundle", fake_bundle)

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="tianyan", machine_name="m0")
    resolved = adapter.resolve_backend(num_qubits=4, prefer_hardware="preferred_machine")

    assert called["machine_name"] == "preferred_machine"
    assert called["num_qubits"] == 4
    assert resolved.hardware_name == "preferred_machine"
    assert resolved.target_qubits == [1, 2]


def test_cqlib_backend_adapter_resolve_backend_prefers_sequence_first(monkeypatch):
    _install_fake_cqlib(monkeypatch)
    called = {}

    def fake_bundle(platform_obj, machine_name, num_qubits):
        called["machine_name"] = machine_name
        return types.SimpleNamespace(backend="b", target_qubits=None)

    monkeypatch.setattr(ub, "build_cqlib_backend_bundle", fake_bundle)

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="tianyan", machine_name=None)
    resolved = adapter.resolve_backend(num_qubits=2, prefer_hardware=["first", "second"])

    assert called["machine_name"] == "first"
    assert resolved.hardware_name == "first"


def test_cqlib_backend_adapter_resolve_backend_uses_default_machine_name(monkeypatch):
    _install_fake_cqlib(monkeypatch)
    called = {}

    def fake_bundle(platform_obj, machine_name, num_qubits):
        called["machine_name"] = machine_name
        return types.SimpleNamespace(backend="b", target_qubits=None)

    monkeypatch.setattr(ub, "build_cqlib_backend_bundle", fake_bundle)

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="tianyan", machine_name=None)
    resolved = adapter.resolve_backend(num_qubits=2)

    assert called["machine_name"] == "tianyan176"
    assert resolved.hardware_name == "tianyan176"


def test_cqlib_backend_adapter_ignores_rank_weights(monkeypatch):
    _install_fake_cqlib(monkeypatch)

    monkeypatch.setattr(
        ub,
        "build_cqlib_backend_bundle",
        lambda platform_obj, machine_name, num_qubits: types.SimpleNamespace(backend="b", target_qubits=[0]),
    )

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="tianyan", machine_name="m")
    resolved = adapter.resolve_backend(num_qubits=1, rank_weights={"queue": 9.9})

    assert resolved.metadata["platform_name"] == "tianyan"
    assert resolved.metadata["machine_name"] == "m"


def test_cqlib_backend_adapter_metadata_contains_platform_obj(monkeypatch):
    _install_fake_cqlib(monkeypatch)

    monkeypatch.setattr(
        ub,
        "build_cqlib_backend_bundle",
        lambda platform_obj, machine_name, num_qubits: types.SimpleNamespace(backend="b", target_qubits=[0, 1]),
    )

    adapter = ub.CqlibBackendAdapter(login_key="k", platform="tianyan", machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=2)

    assert resolved.metadata["platform_obj"] is adapter._platform_obj
    assert resolved.metadata["platform_name"] == "tianyan"
    assert resolved.metadata["machine_name"] == "abc"
