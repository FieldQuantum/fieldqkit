import pytest

from quantum_hw.api import backend as hardware_module
from quantum_hw.api.backend import Backend


def test_backend_graph_from_dict():
    chip_info = {
        "size": (1, 2),
        "priority_qubits": [[0, 1]],
        "qubits_info": {
            "Q0": {"fidelity": 0.99, "coordinate": [0, 0]},
            "Q1": {"fidelity": 0.98, "coordinate": [1, 0]},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.95, "index": 0}
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


def test_rank_chips_uses_local_backend_info(monkeypatch):
    def fake_status(_tmgr):
        return {"chip_a": 1, "chip_b": 5}

    def fake_info(name):
        if name == "chip_a":
            return {"global_info": {"nqubits_available": 6, "error_rate_2q": 0.02}}
        return {"global_info": {"nqubits_available": 8, "error_rate_2q": 0.1}}

    monkeypatch.setattr(hardware_module, "get_available_chip_status", fake_status)
    monkeypatch.setattr(hardware_module, "get_chip_info", fake_info)

    ranked = hardware_module.rank_chips(object(), num_qubits=4)
    assert ranked[0] == "chip_a"

    ranked_pref = hardware_module.rank_chips(object(), num_qubits=4, prefer_chips=["chip_b"])
    assert ranked_pref == ["chip_b"]


def test_transpiler_layout_smoke():
    from quantum_hw.compile import Transpiler
    from quantum_hw.circuit import QuantumCircuit

    chip_info = {
        "size": (1, 2),
        "priority_qubits": [[0, 1]],
        "qubits_info": {
            "Q0": {"fidelity": 0.99, "coordinate": [0, 0]},
            "Q1": {"fidelity": 0.98, "coordinate": [1, 0]},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.95, "index": 0}
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
