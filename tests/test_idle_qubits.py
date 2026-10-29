"""High-level execution must retain declared, idle logical qubits."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from fieldqkit.api.backend import Backend, ResolvedBackend
from fieldqkit.api.client import QuantumHardwareClient
from fieldqkit.api.quantum_platform import ProviderRuntime
from fieldqkit.api.task import ProviderTaskHandle
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.sim import simulate_counts


def _run(circuit, num_qubits, transpile, **kwargs):
    return QuantumHardwareClient().run_auto(
        circuit, name="idle_register", num_qubits=num_qubits,
        provider="simulator", shots=8, return_probabilities=True,
        transpile_on_client=transpile, print_true=False, **kwargs,
    )


def _assert_basis_result(result, bits):
    expected = [int(bit) for bit in bits]
    probabilities = np.zeros(2 ** len(bits))
    probabilities[int(bits, 2)] = 1.0
    assert result.samples == [[expected] * 8]
    assert len(result.probabilities) == 1
    np.testing.assert_allclose(result.probabilities[0], probabilities, atol=1e-12)


@pytest.mark.parametrize("has_gate", [False, True])
def test_normalization_retains_idle_metadata_without_adding_gates(has_gate):
    circuit = QuantumCircuit(3)
    if has_gate:
        circuit.x(0)
    before = deepcopy(circuit.__dict__)
    normalized = QuantumHardwareClient()._normalize_input_circuit(circuit, 3)
    assert normalized is not circuit
    assert normalized.qubits == [0, 1, 2]
    assert normalized.gates == before["gates"]
    assert normalized.nqubits == 3
    assert circuit.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
@pytest.mark.parametrize("active_qubit", [0, 1, 2])
def test_default_measurement_includes_idle_register_qubits(transpile, active_qubit):
    circuit = QuantumCircuit(3).x(active_qubit)
    before = deepcopy(circuit.__dict__)
    result = _run(circuit, 3, transpile)
    bits = ["0"] * 3
    bits[active_qubit] = "1"
    _assert_basis_result(result, "".join(bits))
    assert circuit.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
@pytest.mark.parametrize("active_qubit", [0, 1, 2])
def test_observables_keep_idle_logical_qubit_indices(transpile, active_qubit):
    circuit = QuantumCircuit(3).x(active_qubit)
    before = deepcopy(circuit.__dict__)
    result = _run(circuit, 3, transpile, observables=["Z0", "Z1", "Z2"])
    bits = ["0"] * 3
    bits[active_qubit] = "1"
    _assert_basis_result(result, "".join(bits))
    assert result.observable_values == {
        f"Z{qubit}": -1.0 if qubit == active_qubit else 1.0
        for qubit in range(3)
    }
    assert circuit.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
def test_empty_declared_register_executes_as_zero_state(transpile):
    circuit = QuantumCircuit(3)
    before = deepcopy(circuit.__dict__)
    result = _run(circuit, 3, transpile, observables=["Z0", "Z2"])
    _assert_basis_result(result, "000")
    assert result.observable_values == {"Z0": 1.0, "Z2": 1.0}
    assert circuit.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
def test_explicit_partial_measurement_and_classical_order_stay_intact(transpile):
    circuit = QuantumCircuit(4).x(0).measure([0, 2], [1, 0])
    before = deepcopy(circuit.__dict__)
    result = _run(circuit, 4, transpile)
    # Only two classical outcomes are requested: c0=q2=0, c1=q0=1.
    _assert_basis_result(result, "01")
    assert circuit.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
def test_observable_replacement_of_partial_measurement_keeps_idle_qubits(transpile):
    circuit = QuantumCircuit(4).x(0).measure([0, 2], [1, 0])
    before = deepcopy(circuit.__dict__)
    result = _run(circuit, 4, transpile, observables=["Z0", "Z3"])
    _assert_basis_result(result, "1000")
    assert result.observable_values == {"Z0": -1.0, "Z3": 1.0}
    assert circuit.__dict__ == before


@pytest.mark.parametrize("transpile", [False, True])
@pytest.mark.parametrize("observables", [None, ["Z0", "Z2"]])
def test_openqasm_declared_idle_register_is_preserved(transpile, observables):
    source = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[3];\ncreg c[3];\nx q[2];\n'
    )
    result = _run(source, 3, transpile, observables=observables)
    _assert_basis_result(result, "001")
    if observables:
        assert result.observable_values == {"Z0": 1.0, "Z2": -1.0}


def test_density_matrix_execution_preserves_idle_logical_qubits():
    circuit = QuantumCircuit(3).x(0).depolarize1(0.0, 0)
    before = deepcopy(circuit.__dict__)
    result = _run(circuit, 3, True, observables=["Z0", "Z2"])
    _assert_basis_result(result, "100")
    assert result.observable_values == {"Z0": -1.0, "Z2": 1.0}
    assert circuit.__dict__ == before


def test_compiled_gate_cancellation_retains_declared_register_width():
    circuit = QuantumCircuit(3).x(0).x(0)
    before = deepcopy(circuit.__dict__)
    _assert_basis_result(_run(circuit, 3, True), "000")
    assert circuit.__dict__ == before


@pytest.mark.parametrize("observables", [None, ["Z0", "Z1", "Z2"]])
@pytest.mark.parametrize("pattern", ["single", "connected"])
def test_idle_qubits_survive_sparse_physical_hardware_mapping(monkeypatch, observables, pattern):
    """Exercise actual layout, submission, polling, and result collection offline."""
    import fieldqkit.api.client as client_module

    physical_qubits = [2, 5, 7]
    backend = Backend({
        "chip_name": "idle_fixture",
        "priority_qubits": [physical_qubits],
        "qubits_info": {f"Q{q}": {"fidelity": 1.0} for q in physical_qubits},
        "couplers_info": {
            "C0": {"qubits_index": [2, 5], "fidelity": 1.0},
            "C1": {"qubits_index": [5, 7], "fidelity": 1.0},
        },
        "global_info": {"two_qubit_gate_basis": "cz", "nqubits_available": 3},
    })
    submitted = []

    class OfflineTasks:
        provider = "tencent"
        native_ir = True

        def submit_native_circuit(self, request, resolved_backend):
            submitted.append(request.circuit.deepcopy())
            dense, _ = QuantumHardwareClient()._compact_for_sim(
                request.circuit, physical_qubits,
            )
            counts = simulate_counts(dense, request.shots, seed=0, device="cpu")
            return ProviderTaskHandle(
                self.provider, "offline-task", payload={"counts": counts},
            )

        def query_status(self, handle):
            return "Finished"

        def fetch_result(self, handle):
            return {"count": handle.payload["counts"]}

    resolved = ResolvedBackend(
        provider="tencent", hardware_name="idle_fixture", backend=backend,
    )
    runtime = ProviderRuntime(
        provider="tencent",
        backend_adapter=SimpleNamespace(resolve_backend=lambda **kwargs: resolved),
        task_adapter=OfflineTasks(),
    )
    monkeypatch.setattr(client_module, "create_provider_runtime", lambda **kwargs: runtime)
    circuit = QuantumCircuit(3)
    if pattern == "single":
        circuit.x(1)
        expected_bits = "010"
    else:
        # Logical qubits 0 and 2 form one interaction component; idle logical
        # qubit 1 must retain its position even when layout groups components.
        circuit.x(0).cx(0, 2)
        expected_bits = "101"
    before = deepcopy(circuit.__dict__)
    result = QuantumHardwareClient().run_auto(
        circuit, name="sparse_idle_register", num_qubits=3, provider="tencent",
        target_qubits=physical_qubits, shots=8, observables=observables,
        return_probabilities=True, print_true=False,
    )
    _assert_basis_result(result, expected_bits)
    if observables:
        assert result.observable_values == {
            f"Z{qubit}": -1.0 if bit == "1" else 1.0
            for qubit, bit in enumerate(expected_bits)
        }
    assert submitted
    measurements = [gate for gate in submitted[0].gates if gate[0] == "measure"]
    measured_qubits = {qubit for gate in measurements for qubit in gate[1]}
    assert measured_qubits == set(physical_qubits)
    assert circuit.__dict__ == before
