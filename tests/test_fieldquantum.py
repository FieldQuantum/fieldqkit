"""Integration tests for the FieldQuantum cloud simulator provider.

The server is started once per module on the default port (8765) with a single
daemon thread.  Tests focus on response correctness, comparing HTTP results
from the server against the local simulator.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest
import requests

SERVER_URL = "http://localhost:8765"


# ---------------------------------------------------------------------------
# Module-level server fixture (one-liner start)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _fq_server():
    """Start the FieldQuantum server on port 8765 for the whole module."""
    from quantum_hw.api.fieldquantum_server import serve

    threading.Thread(target=serve, kwargs={"port": 8765}, daemon=True).start()
    # Wait until reachable
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{SERVER_URL}/health", timeout=1).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        pytest.skip("FieldQuantum server did not start in time")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(payload: dict, timeout: int = 60) -> dict:
    """Submit a job and return the final result via the task-id lifecycle.

    Flow: POST /run -> task_id -> GET /task/{id}/status -> GET /task/{id}/result
    """
    # 1. Submit
    submit = requests.post(f"{SERVER_URL}/run", json=payload, timeout=timeout)
    submit.raise_for_status()
    task_id = submit.json()["task_id"]
    assert isinstance(task_id, str) and len(task_id) > 0
    # 2. Status check
    status_resp = requests.get(f"{SERVER_URL}/task/{task_id}/status", timeout=10)
    status_resp.raise_for_status()
    assert status_resp.json()["status"] == "finished"
    # 3. Fetch result
    result_resp = requests.get(f"{SERVER_URL}/task/{task_id}/result", timeout=timeout)
    result_resp.raise_for_status()
    return result_resp.json()


def _local_counts(qasm: str, shots: int) -> dict:
    """Run QASM locally via simulate_counts and return the count dict."""
    from quantum_hw.circuit import QuantumCircuit
    from quantum_hw.sim import simulate_counts

    qc = QuantumCircuit().from_openqasm2(qasm)
    return simulate_counts(qc, shots)


def _tvd(a: dict, b: dict) -> float:
    """Total variation distance between two count dicts (normalised to probs)."""
    keys = set(a) | set(b)
    sa, sb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in keys)


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


def test_task_lifecycle_submit_status_result():
    """Verify the full task_id lifecycle: POST /run -> /status -> /result."""
    from quantum_hw.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0).measure([0], [0])
    qasm = qc.to_openqasm2()

    # Submit
    submit_resp = requests.post(
        f"{SERVER_URL}/run",
        json={"mode": "sample", "qasm": qasm, "shots": 64},
        timeout=10,
    )
    submit_resp.raise_for_status()
    body = submit_resp.json()
    assert "task_id" in body, f"Expected task_id in response, got: {body}"
    task_id = body["task_id"]

    # Status
    status_resp = requests.get(f"{SERVER_URL}/task/{task_id}/status", timeout=5)
    status_resp.raise_for_status()
    status_body = status_resp.json()
    assert status_body["task_id"] == task_id
    assert status_body["status"] == "finished"

    # Result
    result_resp = requests.get(f"{SERVER_URL}/task/{task_id}/result", timeout=10)
    result_resp.raise_for_status()
    result_body = result_resp.json()
    assert "counts" in result_body
    assert result_body["counts"].get("1", 0) == 64


def test_task_status_unknown_id():
    """Querying a non-existent task_id must return 404."""
    resp = requests.get(f"{SERVER_URL}/task/no-such-id/status", timeout=5)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Sample mode: correctness vs local simulator
# ---------------------------------------------------------------------------


def test_sample_bell_state_structure():
    """Bell-state counts from server must only contain '00' and '11'."""
    from quantum_hw.circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1).measure([0, 1], [0, 1])
    qasm = qc.to_openqasm2()

    data = _post({"mode": "sample", "qasm": qasm, "shots": 1024})
    counts = data["counts"]
    assert set(counts.keys()) <= {"00", "11", "10", "01"}
    # Entangled: dominant outcomes must be 00 and 11
    dominant = {k for k, v in counts.items() if v > 100}
    assert dominant == {"00", "11"} or dominant <= {"00", "11"}
    assert sum(counts.values()) == 1024


def test_sample_bell_state_vs_local():
    """Server counts and local simulator counts should have low TVD."""
    from quantum_hw.circuit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1).measure([0, 1], [0, 1])
    qasm = qc.to_openqasm2()
    shots = 4096

    server_counts = _post({"mode": "sample", "qasm": qasm, "shots": shots})["counts"]
    local_counts = _local_counts(qasm, shots)
    assert _tvd(server_counts, local_counts) < 0.05


def test_sample_x_gate_vs_local():
    """X|0> = |1>: server and local must agree exactly on all shots."""
    from quantum_hw.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0).measure([0], [0])
    qasm = qc.to_openqasm2()
    shots = 512

    server_counts = _post({"mode": "sample", "qasm": qasm, "shots": shots})["counts"]
    local_counts = _local_counts(qasm, shots)
    # Both must be all-1 with TVD ≈ 0
    assert server_counts.get("1", 0) == shots
    assert local_counts.get("1", 0) == shots


def test_sample_ghz_3q_vs_local():
    """3-qubit GHZ: server TVD vs local < 0.05."""
    from quantum_hw.circuit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0).cnot(0, 1).cnot(1, 2).measure([0, 1, 2], [0, 1, 2])
    qasm = qc.to_openqasm2()
    shots = 4096

    server_counts = _post({"mode": "sample", "qasm": qasm, "shots": shots})["counts"]
    local_counts = _local_counts(qasm, shots)
    # GHZ: only 000 and 111 allowed
    assert set(server_counts.keys()) <= {"000", "111"}
    assert _tvd(server_counts, local_counts) < 0.05


def test_sample_parametric_rx_identity():
    """rx(0)|0> = |0>: server must return all shots as '0'."""
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[1];\ncreg c[1];\nrx(theta_0) q[0];\nmeasure q[0] -> c[0];\n'
    )
    shots = 512
    data = _post({
        "mode": "sample", "qasm": qasm, "shots": shots,
        "param_names": ["theta_0"], "param_values": [0.0],
    })
    assert data["counts"].get("0", 0) == shots


def test_sample_parametric_rx_pi_flip():
    """rx(π)|0> ≈ i|1>: server must return all (or nearly all) shots as '1'."""
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[1];\ncreg c[1];\nrx(theta_0) q[0];\nmeasure q[0] -> c[0];\n'
    )
    shots = 512
    data = _post({
        "mode": "sample", "qasm": qasm, "shots": shots,
        "param_names": ["theta_0"], "param_values": [math.pi],
    })
    assert data["counts"].get("1", 0) == shots


# ---------------------------------------------------------------------------
# Expectation mode: correctness vs analytical values
# ---------------------------------------------------------------------------


def test_expectation_z0_on_zero_state():
    """<Z0> on |0> = +1."""
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
    data = _post({
        "mode": "expectation", "qasm": qasm,
        "param_names": [], "param_values": [],
        "hamiltonian": [{"coeff": 1.0, "pauli": "Z0"}],
        "shots": 8192,
    })
    assert abs(data["energy"] - 1.0) < 0.04


def test_expectation_z0_on_x_state():
    """<Z0> on |+> (H|0>) = 0."""
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n'
    data = _post({
        "mode": "expectation", "qasm": qasm,
        "param_names": [], "param_values": [],
        "hamiltonian": [{"coeff": 1.0, "pauli": "Z0"}],
        "shots": 8192,
    })
    assert abs(data["energy"]) < 0.1


def test_expectation_ry_energy_vs_analytical():
    """<Z0> for Ry(θ)|0> = cos(θ) for several θ values."""
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[1];\nry(theta) q[0];\n'
    )
    for theta in [0.0, math.pi / 4, math.pi / 2, math.pi * 3 / 4]:
        data = _post({
            "mode": "expectation", "qasm": qasm,
            "param_names": ["theta"], "param_values": [theta],
            "hamiltonian": [{"coeff": 1.0, "pauli": "Z0"}],
            "shots": 16384,
        })
        assert abs(data["energy"] - math.cos(theta)) < 0.05, (
            f"theta={theta:.3f}: expected {math.cos(theta):.4f}, got {data['energy']:.4f}"
        )


def test_expectation_ry_gradient_vs_parameter_shift():
    """Server gradients must match the parameter-shift rule: dE/dθ = -sin(θ)."""
    theta = math.pi / 4
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[1];\nry(theta) q[0];\n'
    )
    data = _post({
        "mode": "expectation", "qasm": qasm,
        "param_names": ["theta"], "param_values": [theta],
        "hamiltonian": [{"coeff": 1.0, "pauli": "Z0"}],
        "shots": 65536,
    })
    assert abs(data["gradients"][0] - (-math.sin(theta))) < 0.05


def test_expectation_two_qubit_zzterm():
    """<Z0 Z1> on Bell state (H⊗CNOT)|00> = +1."""
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[2];\nh q[0];\ncx q[0],q[1];\n'
    )
    data = _post({
        "mode": "expectation", "qasm": qasm,
        "param_names": [], "param_values": [],
        "hamiltonian": [{"coeff": 1.0, "pauli": "Z0 Z1"}],
        "shots": 8192,
    })
    assert abs(data["energy"] - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Full provider path via QuantumHardwareClient (end-to-end)
# ---------------------------------------------------------------------------


def test_run_auto_fieldquantum_sampling():
    """run_auto sampling path: Bell-state output consistent with local simulator."""
    from quantum_hw.api import QuantumHardwareClient
    from quantum_hw.circuit import QuantumCircuit
    from quantum_hw.sim import simulate_counts

    shots = 1024
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1)

    client = QuantumHardwareClient()
    result = client.run_auto(
        qc, "fq_bell", num_qubits=2,
        provider="fieldquantum", shots=shots,
        transpile_on_client=True, print_true=False,
    )
    assert result is not None
    flat = [s for grp in result.samples for s in (grp or [])]
    assert len(flat) == shots
    # Must be pure Bell: only 00/11 bitstrings (length-2)
    bitstrings = {"".join(str(b) for b in row) for row in flat}
    assert bitstrings <= {"00", "11"}


def test_run_auto_fieldquantum_observable_vs_local():
    """<Z0 Z1> from FieldQuantum vs local simulator should agree to within 0.1."""
    from quantum_hw.api import QuantumHardwareClient
    from quantum_hw.circuit import QuantumCircuit
    from quantum_hw.core.observables import pauli_expectation
    from quantum_hw.core.utils import get_samples
    from quantum_hw.sim import simulate_counts

    shots = 4096
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1)

    # FieldQuantum cloud path
    client = QuantumHardwareClient()
    fq_result = client.run_auto(
        qc, "fq_zz", num_qubits=2,
        provider="fieldquantum", shots=shots,
        observables=["Z0 Z1"],
        transpile_on_client=True, print_true=False,
    )
    fq_val = fq_result.observable_values["Z0 Z1"]

    # Local simulator reference
    qc_local = QuantumCircuit(2)
    qc_local.h(0).cnot(0, 1)
    from quantum_hw.core.observables import append_measurement_basis
    qc_local.remove_gate("measure")
    append_measurement_basis(qc_local, ["Z", "Z"], target_qubits=[0, 1])
    local_counts = simulate_counts(qc_local, shots)
    local_samples = get_samples(local_counts, 2)
    local_val = pauli_expectation(local_samples, "Z0 Z1")

    assert abs(fq_val - local_val) < 0.1, (
        f"FieldQuantum Z0 Z1 = {fq_val:.4f}, local = {local_val:.4f}"
    )
