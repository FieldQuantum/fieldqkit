from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.quantumcircuit_helpers import three_qubit_gates_available
from quantum_hw.compile.decompose import ThreeQubitGateDecompose
from quantum_hw.compile.translate import TranslateToBasisGates


def test_three_qubit_decompose_removes_three_qubit_gates():
    qc = QuantumCircuit(3, 3)
    qc.ccx(0, 1, 2)
    qc.ccz(0, 1, 2)
    qc.cswap(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    new_qc = ThreeQubitGateDecompose().run(qc)
    assert new_qc.nqubits == qc.nqubits
    assert new_qc.ncbits == qc.ncbits
    assert all(gate[0] not in three_qubit_gates_available for gate in new_qc.gates)


def test_translate_to_basis_gates_cz_and_u_only():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.ry(0.2, 1)
    qc.cx(0, 1)
    qc.swap(0, 1)
    qc.rxx(0.3, 0, 1)
    qc.ryy(-0.4, 0, 1)
    qc.rzz(0.5, 0, 1)
    qc.cp(0.6, 0, 1)
    qc.measure([0, 1], [0, 1])

    new_qc = TranslateToBasisGates(convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz").run(qc)
    allowed = {"u", "cz", "measure"}
    for gate in new_qc.gates:
        assert gate[0] in allowed
