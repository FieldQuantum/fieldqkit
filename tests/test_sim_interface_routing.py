import pytest


torch = pytest.importorskip("torch")

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim import interface
from quantum_hw.sim import mps as mps_mod
import quantum_hw.sim as sim_pkg


@pytest.fixture(autouse=True)
def _restore_sim_globals():
    old_threshold = interface.MPS_THRESHOLD_QUBITS
    old_fallback = mps_mod.ENABLE_STATEVECTOR_FALLBACK
    yield
    interface.MPS_THRESHOLD_QUBITS = old_threshold
    mps_mod.ENABLE_STATEVECTOR_FALLBACK = old_fallback


def test_sim_pkg_exports_unified_sim_apis():
    assert hasattr(sim_pkg, "simulate_counts")
    assert hasattr(sim_pkg, "expectation_pauli")
    assert hasattr(sim_pkg, "energy_and_expectations")


def test_simulate_counts_routes_by_threshold(monkeypatch):
    interface.MPS_THRESHOLD_QUBITS = 12

    def _sv(qc, shots, *, seed=None, param_values=None):
        return {"sv": int(shots), "n": int(qc.nqubits)}

    def _mps(qc, shots, *, seed=None, param_values=None):
        return {"mps": int(shots), "n": int(qc.nqubits)}

    monkeypatch.setattr(interface, "_simulate_counts_statevector", _sv)
    monkeypatch.setattr(interface, "_simulate_counts_mps", _mps)

    small_qc = QuantumCircuit(11)
    large_qc = QuantumCircuit(12)

    assert interface.simulate_counts(small_qc, 7) == {"sv": 7, "n": 11}
    assert interface.simulate_counts(large_qc, 9) == {"mps": 9, "n": 12}


def test_expectation_pauli_routes_by_threshold(monkeypatch):
    interface.MPS_THRESHOLD_QUBITS = 12

    monkeypatch.setattr(
        interface,
        "_expectation_pauli_statevector",
        lambda state, pauli, *, num_qubits: ("sv", pauli, int(num_qubits)),
    )
    monkeypatch.setattr(
        interface,
        "_expectation_pauli_mps",
        lambda state, pauli, *, num_qubits: ("mps", pauli, int(num_qubits)),
    )

    assert interface.expectation_pauli(object(), "Z0", num_qubits=11) == ("sv", "Z0", 11)
    assert interface.expectation_pauli(object(), "Z0", num_qubits=12) == ("mps", "Z0", 12)


def test_energy_and_expectations_routes_by_threshold(monkeypatch):
    interface.MPS_THRESHOLD_QUBITS = 12

    monkeypatch.setattr(
        interface,
        "_energy_and_expectations_statevector",
        lambda symbolic_qc, *, params, param_names, hamiltonian: ("sv", int(symbolic_qc.nqubits)),
    )
    monkeypatch.setattr(
        interface,
        "_energy_and_expectations_mps",
        lambda symbolic_qc, *, params, param_names, hamiltonian: ("mps", int(symbolic_qc.nqubits)),
    )

    small_qc = QuantumCircuit(11)
    large_qc = QuantumCircuit(12)

    assert interface.energy_and_expectations(
        small_qc,
        params=torch.tensor([0.0], dtype=torch.float64),
        param_names=["theta_0"],
        hamiltonian=[(1.0, "Z0")],
    ) == ("sv", 11)

    assert interface.energy_and_expectations(
        large_qc,
        params=torch.tensor([0.0], dtype=torch.float64),
        param_names=["theta_0"],
        hamiltonian=[(1.0, "Z0")],
    ) == ("mps", 12)


def test_mps_dummy_functions_use_statevector_fallback(monkeypatch):
    mps_mod.set_statevector_fallback(True)

    monkeypatch.setattr(
        mps_mod,
        "_simulate_counts_statevector",
        lambda qc, shots, *, seed=None, param_values=None: {"ok": int(shots)},
    )
    monkeypatch.setattr(
        mps_mod,
        "_expectation_pauli_statevector",
        lambda state, pauli, *, num_qubits: 0.5,
    )
    monkeypatch.setattr(
        mps_mod,
        "_energy_and_expectations_statevector",
        lambda symbolic_qc, *, params, param_names, hamiltonian: (1.23, {"Z0": 0.5}),
    )

    qc = QuantumCircuit(2)

    assert mps_mod.simulate_counts(qc, 13) == {"ok": 13}
    assert mps_mod.expectation_pauli(object(), "Z0", num_qubits=2) == 0.5
    assert mps_mod.energy_and_expectations(
        qc,
        params=torch.tensor([0.0], dtype=torch.float64),
        param_names=["theta_0"],
        hamiltonian=[(1.0, "Z0")],
    ) == (1.23, {"Z0": 0.5})


def test_mps_dummy_functions_raise_when_fallback_disabled():
    mps_mod.set_statevector_fallback(False)
    qc = QuantumCircuit(2)

    with pytest.raises(NotImplementedError, match="simulate_counts"):
        mps_mod.simulate_counts(qc, 8)

    with pytest.raises(NotImplementedError, match="expectation_pauli"):
        mps_mod.expectation_pauli(object(), "Z0", num_qubits=2)

    with pytest.raises(NotImplementedError, match="energy_and_expectations"):
        mps_mod.energy_and_expectations(
            qc,
            params=torch.tensor([0.0], dtype=torch.float64),
            param_names=["theta_0"],
            hamiltonian=[(1.0, "Z0")],
        )


def test_interface_extreme_large_threshold_routes_to_statevector(monkeypatch):
    interface.MPS_THRESHOLD_QUBITS = 10**9
    qc = QuantumCircuit(64)

    monkeypatch.setattr(
        interface,
        "_simulate_counts_statevector",
        lambda qc, shots, *, seed=None, param_values=None: {"path": "sv"},
    )
    monkeypatch.setattr(
        interface,
        "_simulate_counts_mps",
        lambda qc, shots, *, seed=None, param_values=None: {"path": "mps"},
    )

    assert interface.simulate_counts(qc, 1) == {"path": "sv"}
