import pytest


torch = pytest.importorskip("torch")

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from quantum_hw.sim.common import materialize_gate_matrix, resolve_param
from quantum_hw.sim.mpo import simulate_mpo_process


def _mpo_to_matrix(mpo):
    if not mpo:
        return torch.eye(1, dtype=torch.complex128)

    current = mpo[0]
    for next_tensor in mpo[1:]:
        current = torch.einsum("lprq,rstu->lpstqu", current, next_tensor)
        dl, p0, p1, dr, q0, q1 = current.shape
        current = current.reshape(dl, p0 * p1, dr, q0 * q1)
    return current.reshape(current.shape[1], current.shape[3])


def _left_apply_gate_to_matrix(unitary, gate, qubits, num_qubits):
    k = len(qubits)
    if k == 0:
        return unitary

    out_shape = [2] * num_qubits + [2**num_qubits]
    tensor = unitary.reshape(out_shape)
    tensor = torch.moveaxis(tensor, list(qubits), list(range(k)))
    tensor = tensor.reshape(2**k, -1)
    tensor = gate @ tensor
    tensor = tensor.reshape(out_shape)
    tensor = torch.moveaxis(tensor, list(range(k)), list(qubits))
    return tensor.reshape(2**num_qubits, 2**num_qubits)


def _reference_unitary_from_circuit(qc):
    n = int(qc.nqubits)
    dtype = torch.complex128
    device = torch.device("cpu")
    unitary = torch.eye(2**n, dtype=dtype, device=device)

    for gate_info in qc.gates:
        gate = gate_info[0]

        if gate in functional_gates_available:
            if gate == "reset":
                raise ValueError("reset is not unitary and cannot be mapped to dense unitary reference")
            continue

        if gate in one_qubit_gates_available:
            q = int(gate_info[1])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q], n)
            continue

        if gate in one_qubit_parameter_gates_available:
            q = int(gate_info[-1])
            params = [resolve_param(qc, p, None) for p in gate_info[1:-1]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q], n)
            continue

        if gate in two_qubit_gates_available:
            q0 = int(gate_info[1])
            q1 = int(gate_info[2])
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q0, q1], n)
            continue

        if gate in two_qubit_parameter_gates_available:
            q0 = int(gate_info[-2])
            q1 = int(gate_info[-1])
            params = [resolve_param(qc, p, None) for p in gate_info[1:-2]]
            mat = materialize_gate_matrix(gate, params, dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, [q0, q1], n)
            continue

        if gate in three_qubit_gates_available:
            qubits = [int(gate_info[1]), int(gate_info[2]), int(gate_info[3])]
            mat = materialize_gate_matrix(gate, [], dtype=dtype, device=device)
            unitary = _left_apply_gate_to_matrix(unitary, mat, qubits, n)
            continue

        raise ValueError(f"unsupported gate for reference builder: {gate}")

    return unitary


def test_simulate_mpo_process_matches_dense_reference_on_nonadjacent_and_three_qubit_gates():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.ry(0.37, 2)
    qc.cx(0, 3)
    qc.rzz(-0.52, 1, 3)
    qc.ccx(0, 2, 3)

    mpo = simulate_mpo_process(qc)
    actual = _mpo_to_matrix(mpo)
    expected = _reference_unitary_from_circuit(qc)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_simulate_mpo_process_respects_max_bond_dim_cap():
    qc = QuantumCircuit(6)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.cx(0, 5)
    qc.cx(1, 4)
    qc.cx(2, 3)
    qc.rxx(0.31, 0, 5)
    qc.rzz(-0.29, 1, 4)

    mpo = simulate_mpo_process(qc, max_bond_dim=4)
    max_bond = max(int(t.shape[2]) for t in mpo)

    assert max_bond <= 4
