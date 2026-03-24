import pytest

from quantum_hw.api import backend as hardware_module
from quantum_hw.api.backend import Backend
from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.circuit import QuantumCircuit


def test_backend_graph_from_dict():
    chip_info = {
        "size": (1, 2),
        "priority_qubits": [[0, 1]],
        "qubits_info": {
            "Q0": {"fidelity": 0.99},
            "Q1": {"fidelity": 0.98},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.95}
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }

    backend = Backend(chip_info)
    graph = backend.graph

    assert set(graph.nodes()) == {0, 1}
    assert set(map(tuple, graph.edges())) == {(0, 1)}
    assert backend.two_qubit_gate_basis == "cz"

    filtered = backend.edge_filtered_graph(thres=0.96)
    assert filtered.number_of_edges() == 0


def test_normalize_hardware_preferences_accepts_string_or_list():
    assert hardware_module.normalize_hardware_preferences("chip_a") == ["chip_a"]
    assert hardware_module.normalize_hardware_preferences(["chip_a", "chip_b"]) == ["chip_a", "chip_b"]


def test_backend_treats_low_fidelity_coupler_as_disconnected():
    chip_info = {
        "qubits_info": {
            "Q0": {"fidelity": 0.99},
            "Q1": {"fidelity": 0.98},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.89},
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }

    backend = Backend(chip_info)

    assert sorted(backend.graph.nodes()) == [0, 1]
    assert list(backend.graph.edges()) == []


def test_build_hardware_profile_excludes_low_fidelity_coupler():
    chip_info = {
        "chip_name": "x",
        "qubits_info": {
            "Q0": {"fidelity": 0.99},
            "Q1": {"fidelity": 0.98},
            "Q2": {"fidelity": 0.97},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.89},
            "C1": {"qubits_index": [1, 2], "fidelity": 0.95},
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }

    backend = Backend(chip_info)
    profile = hardware_module.build_hardware_profile(
        provider="quafu",
        hardware_name="x",
        backend=backend,
        queue_length=None,
        raw_info=chip_info,
    )

    assert profile.topology.couplers == [(1, 2)]
    assert "C0" not in profile.calibration.coupler_fidelity
    assert profile.calibration.coupler_fidelity["C1"] == 0.95


def test_transpiler_layout_smoke():
    from quantum_hw.compile import Transpiler

    chip_info = {
        "size": (1, 2),
        "priority_qubits": [[0, 1]],
        "qubits_info": {
            "Q0": {"fidelity": 0.99},
            "Q1": {"fidelity": 0.98},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.95}
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }

    backend = Backend(chip_info)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    transpiled = Transpiler(backend).run(qc, use_gate_compressor=False)

    assert transpiled.nqubits >= 2
    assert len(transpiled.gates) > 0


def test_run_with_backend_uses_layout_mapping_for_measurement_order():
    client = QuantumHardwareClient()
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1):
        qc.cz(i, 2)
    for i in range(num_qubits - 2, -1, -1):
        qc.cz(i, 2)
    qc.x(1)
    res = client.run_auto(qc, "test", num_qubits, observables=["Z0", "Z1", "Z2"], prefer_chips="Simulator")
    assert res.observable_values["Z0"] > 0.9
    assert res.observable_values["Z1"] < - 0.9
    assert res.observable_values["Z2"] > 0.9
