import pytest
import json

from quantum_hw.api.quantum_platform import cqlib as cq
from quantum_hw.api.quantum_platform import guodun as gd
from quantum_hw.api.quantum_platform import quafu as qf
from quantum_hw.api.quantum_platform import tianyan as ty
from quantum_hw.api import backend as bmod
from quantum_hw.api.backend import Backend, HardwareCalibration, HardwareProfile, HardwareTopology


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
    monkeypatch.setattr(ty, "get_tianyan_login_key", lambda: "k")
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
    monkeypatch.setattr(gd, "get_guodun_login_key", lambda: "k")
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
    monkeypatch.setattr(ty, "get_tianyan_login_key", lambda: "k")

    adapter = ty.TianYanBackendAdapter(machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=3, prefer_hardware="Simulator")
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
    monkeypatch.setattr(ty, "get_tianyan_login_key", lambda: "k")
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

    adapter = ty.TianYanBackendAdapter(login_key="custom_key")
    profiles = adapter.discover_hardware(num_qubits=2)

    assert [profile.hardware_name for profile in profiles] == ["tianyan176"]
    assert profiles[0].calibration.queue_length == 3


def test_guodun_backend_adapter_supports_simulator_preference(monkeypatch):
    class _FakePlatform:
        def __init__(self, login_key, auto_login, machine_name):
            del login_key, auto_login, machine_name

    monkeypatch.setattr(gd, "GuoDunPlatform", _FakePlatform)
    monkeypatch.setattr(gd, "get_guodun_login_key", lambda: "k")

    adapter = gd.GuoDunBackendAdapter(machine_name="abc")
    resolved = adapter.resolve_backend(num_qubits=2, prefer_hardware="Simulator")
    assert resolved.hardware_name == "Simulator"


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
    monkeypatch.setattr(ty, "get_tianyan_login_key", lambda: "k")

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
    monkeypatch.setattr(gd, "get_guodun_login_key", lambda: "k")
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


def test_cqlib_parser_preserves_qubit_coordinate_when_present():
    chip_info = cq.chip_info_from_config(
        {
            "twoQubitGate": {"czGate": {}},
            "overview": {
                "qubits": [
                    {"name": "Q0", "coordinate": [0, 1]},
                    {"name": "Q1", "x": 2, "y": 3},
                ],
                "coupler_map": {
                    "G0": ["Q0", "Q1"],
                },
            },
        },
        machine_name="m",
    )
    backend = Backend(chip_info)

    assert backend.chip_info["qubits_info"]["Q0"]["coordinate"] == [0.0, 1.0]
    assert backend.chip_info["qubits_info"]["Q1"]["coordinate"] == [2.0, 3.0]
    assert backend.graph.nodes[0]["coordinate"] == [0.0, 1.0]
    assert backend.graph.nodes[1]["coordinate"] == [2.0, 3.0]


def test_cqlib_parser_does_not_invent_qubits_or_couplers_from_missing_fields():
    chip_info = cq.chip_info_from_config({}, machine_name="m")
    backend = Backend(chip_info)

    assert backend.chip_info["qubits_info"] == {}
    assert backend.chip_info["couplers_info"] == {}


def test_cqlib_parser_does_not_invent_connectivity_without_coupler_map():
    chip_info = cq.chip_info_from_config(
        {
            "twoQubitGate": {"czGate": {}},
            "overview": {
                "qubits": ["Q0", "Q1", "Q2"],
            },
        },
        machine_name="m",
    )
    backend = Backend(chip_info)

    assert sorted(backend.graph.nodes()) == [0, 1, 2]
    assert list(backend.graph.edges()) == []
