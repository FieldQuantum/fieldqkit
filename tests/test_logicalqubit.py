"""Offline tests for the LogicalQubit (逻辑比特) direct-REST provider.

These exercise the pure, network-free pieces: the QuantumCircuit -> IR
converter, chip_info normalization, readout-cache priming, and status
mapping. Live submission is validated separately against hardware.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fieldqkit.circuit import QuantumCircuit
from fieldqkit.api.quantum_platform import logicalqubit as lq


# --- a representative QZ02-shaped backend config (trimmed) ----------------

_QZ02_CFG = {
    "name": "QZ02",
    "type": "real_qpu",
    "qubits": 4,
    "supported_measurement_types": ["measure2"],
    "topology": {
        "coupling_map": [[0, 1], [1, 2], [2, 3]],
        "qubit_coordinates": {"0": [0, 0], "1": [1, 0], "2": [2, 0], "3": [3, 0]},
    },
    "properties": {
        "qubit_metrics": [
            {"id": 0, "xeb_fidelity": 0.999, "measure_f0": 0.976, "measure_f1": 0.968},
            {"id": 1, "xeb_fidelity": 0.998, "measure_f0": 0.961, "measure_f1": 0.978},
            {"id": 2, "xeb_fidelity": 0.997, "measure_f0": 0.95, "measure_f1": 0.96},
            {"id": 3, "xeb_fidelity": 0.996, "measure_f0": 0.94, "measure_f1": 0.95},
        ],
    },
}


class _FakePlatform:
    def get_backend_config(self, name):
        return _QZ02_CFG if name == "QZ02" else None


# --- chip_info mapping ----------------------------------------------------

def test_load_chip_info_maps_topology_and_fidelities():
    info = lq.load_logicalqubit_chip_info("QZ02", platform=_FakePlatform())
    assert info["chip_name"] == "QZ02"
    assert info["global_info"]["two_qubit_gate_basis"] == "cz"
    assert info["global_info"]["nqubits_available"] == 4
    assert info["global_info"]["supported_measurement_types"] == ["measure2"]
    # per-qubit gate fidelity from xeb_fidelity, readout f0/f1 carried through
    assert info["qubits_info"]["Q0"]["fidelity"] == pytest.approx(0.999)
    assert info["qubits_info"]["Q0"]["readout_f0"] == pytest.approx(0.976)
    assert info["qubits_info"]["Q0"]["readout_f1"] == pytest.approx(0.968)
    assert info["qubits_info"]["Q0"]["coordinate"] == [0.0, 0.0]
    # coupling_map -> couplers_info
    edges = {tuple(c["qubits_index"]) for c in info["couplers_info"].values()}
    assert edges == {(0, 1), (1, 2), (2, 3)}


def test_two_state_readout_detection():
    info = lq.load_logicalqubit_chip_info("QZ02", platform=_FakePlatform())
    assert lq._two_state_readout(info) is True
    info["global_info"]["supported_measurement_types"] = ["measure", "measure2"]
    assert lq._two_state_readout(info) is False


# --- circuit -> IR converter ---------------------------------------------

def _build_circuit():
    qc = QuantumCircuit(6)
    qc.u(0.5, 0.1, 0.2, 3)
    qc.cz(3, 4)
    qc.cz(4, 5)
    qc.barrier()
    qc.measure([3, 4, 5], [0, 1, 2])
    return qc


def test_converter_structure_and_su2():
    cmd, shots = lq.circuit_to_lqcloud_command(_build_circuit(), two_state_readout=False, shots=1000)
    c = cmd["circuit"]
    assert cmd["action"] == "run_circuit"
    assert c["n_qubits"] == 3 and c["n_clbits"] == 3
    assert c["initial_layout"] == [3, 4, 5]  # physical qubits, compacted->dense
    assert c["result_format"] == "counts" and shots == 1000
    names = [i["name"] for i in c["instructions"]]
    # 3 leading resets, one su2 for the u, two cz, a barrier, three measures
    assert names[:3] == ["reset", "reset", "reset"]
    assert names.count("su2") == 1
    assert names.count("cz") == 2
    assert names.count("measure") == 3
    assert "x21" not in names  # 1-state readout
    su2 = next(i for i in c["instructions"] if i["name"] == "su2")
    assert len(su2["params"]) == 8


def test_converter_inserts_x21_for_two_state_readout():
    cmd, _ = lq.circuit_to_lqcloud_command(_build_circuit(), two_state_readout=True, shots=1000)
    names = [i["name"] for i in cmd["circuit"]["instructions"]]
    # one x21 immediately before each measure
    assert names.count("x21") == 3
    for i, n in enumerate(names):
        if n == "measure":
            assert names[i - 1] == "x21"


def test_converter_barrier_before_measure_is_guaranteed():
    qc = QuantumCircuit(2)
    qc.u(0.3, 0.0, 0.0, 0)
    qc.measure([0, 1], [0, 1])  # no explicit barrier
    cmd, _ = lq.circuit_to_lqcloud_command(qc, two_state_readout=False, shots=100)
    names = [i["name"] for i in cmd["circuit"]["instructions"]]
    assert "barrier" in names
    assert names.index("barrier") < names.index("measure")


def test_converter_memory_mode_and_shot_clipping():
    qc = QuantumCircuit(13)
    for q in range(13):
        qc.u(0.1, 0.0, 0.0, q)
    qc.barrier()
    qc.measure(list(range(13)), list(range(13)))
    cmd, shots = lq.circuit_to_lqcloud_command(qc, two_state_readout=False, shots=100_000)
    assert cmd["circuit"]["result_format"] == "memory"
    assert shots == 50_000  # clipped to per-shot ceiling


# --- readout-cache priming ------------------------------------------------

def test_prime_readout_cache_writes_confusion_matrices(tmp_path, monkeypatch):
    from fieldqkit.api.backend import Backend

    info = lq.load_logicalqubit_chip_info("QZ02", platform=_FakePlatform())
    backend = Backend(info)

    written = {}

    def fake_save(path, *, payload_key, timestamps, payload):
        written["path"] = path
        written["payload_key"] = payload_key
        written["timestamps"] = timestamps
        written["payload"] = payload

    monkeypatch.setattr(lq, "save_timestamped_payload", fake_save, raising=False)
    # patch the imported symbol inside prime_readout_cache
    import fieldqkit.calibration._cache as cache_mod
    monkeypatch.setattr(cache_mod, "save_timestamped_payload", fake_save)

    lq.prime_readout_cache(backend, "QZ02")

    assert written["payload_key"] == "per_qubit_confusion"
    # M = [[f0, 1-f1], [1-f0, f1]]
    m0 = np.array(written["payload"]["0"])
    assert m0[0, 0] == pytest.approx(0.976)  # f0
    assert m0[1, 1] == pytest.approx(0.968)  # f1
    assert m0[1, 0] == pytest.approx(1 - 0.976)
    assert m0[0, 1] == pytest.approx(1 - 0.968)
    # every column sums to 1 (valid confusion matrix)
    assert np.allclose(m0.sum(axis=0), 1.0)


# --- status mapping -------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("queued", "Running"),
    ("running", "Running"),
    ("completed", "Finished"),
    ("failed", "Failed"),
    ("cancelled", "Canceled"),
    ("weird", "Running"),
])
def test_status_mapping(raw, expected):
    assert lq._STATUS_MAP.get(raw, "Running") == expected
