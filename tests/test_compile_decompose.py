import numpy as np
import pytest

from fieldqkit.compile import decompose
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.circuit.matrix import gate_matrix_dict, u_mat
from fieldqkit.circuit.utils import is_equiv_unitary


ONE_QUBIT_GATES = {
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
}
ONE_QUBIT_PARAM_GATES = {"rx", "ry", "rz", "u"}
TWO_QUBIT_GATES = {"cx", "cy", "cz", "swap", "iswap", "ecr"}
TWO_QUBIT_PARAM_GATES = {"rxx", "ryy", "rzz"}


def _bits_from_index(idx: int, nqubits: int) -> list[int]:
    # Big-endian: qubit 0 is the most-significant bit (matches circuit.matrix).
    return [(idx >> (nqubits - 1 - q)) & 1 for q in range(nqubits)]


def _index_from_bits(bits: list[int]) -> int:
    n = len(bits)
    idx = 0
    for q, bit in enumerate(bits):
        idx |= (bit & 1) << (n - 1 - q)
    return idx


def _expand_gate(nqubits: int, gate_mat: np.ndarray, qubits: list[int]) -> np.ndarray:
    dim = 2**nqubits
    k = len(qubits)
    full = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        in_bits = _bits_from_index(col, nqubits)
        sub_bits = [in_bits[q] for q in qubits]
        sub_idx = _index_from_bits(sub_bits)
        for row_sub in range(2**k):
            out_bits = in_bits.copy()
            out_sub_bits = _bits_from_index(row_sub, k)
            for q, bit in zip(qubits, out_sub_bits):
                out_bits[q] = bit
            row = _index_from_bits(out_bits)
            full[row, col] += gate_mat[row_sub, sub_idx]
    return full


def _gate_qubits(gate_info: tuple) -> list[int]:
    name = gate_info[0]
    if name in ONE_QUBIT_GATES:
        return [gate_info[1]]
    if name in ONE_QUBIT_PARAM_GATES:
        return [gate_info[-1]]
    if name in TWO_QUBIT_GATES:
        return [gate_info[1], gate_info[2]]
    if name in TWO_QUBIT_PARAM_GATES:
        return [gate_info[-2], gate_info[-1]]
    raise ValueError(f"Unsupported gate {name}")


def _gate_matrix(gate_info: tuple) -> np.ndarray:
    name = gate_info[0]
    if name == "u":
        theta, phi, lamda = gate_info[1:-1]
        return u_mat(theta, phi, lamda)
    mat = gate_matrix_dict[name]
    if callable(mat):
        params = gate_info[1:-1]
        return mat(*params)
    return mat


def _circuit_unitary(nqubits: int, gates: list[tuple]) -> np.ndarray:
    unitary = np.eye(2**nqubits, dtype=complex)
    for gate_info in gates:
        gate_mat = _gate_matrix(gate_info)
        qubits = _gate_qubits(gate_info)
        full = _expand_gate(nqubits, gate_mat, qubits)
        unitary = full @ unitary
    return unitary


def test_u_dot_u_composes_in_order():
    u1 = ("u", 0.3, -0.2, 1.1, 0)
    u2 = ("u", -0.7, 0.5, -0.4, 0)
    combined = decompose.u_dot_u(u1, u2)
    combined_mat = u_mat(*combined[1:-1])
    expected = u_mat(*u2[1:-1]) @ u_mat(*u1[1:-1])
    assert is_equiv_unitary(combined_mat, expected)


@pytest.mark.parametrize(
    "builder, gate_name, theta",
    [
        (decompose.x2u, "x", None),
        (decompose.y2u, "y", None),
        (decompose.z2u, "z", None),
        (decompose.h2u, "h", None),
        (decompose.s2u, "s", None),
        (decompose.sdg2u, "sdg", None),
        (decompose.t2u, "t", None),
        (decompose.tdg2u, "tdg", None),
        (decompose.sx2u, "sx", None),
        (decompose.sxdg2u, "sxdg", None),
        (decompose.rx2u, "rx", 0.7),
        (decompose.ry2u, "ry", -1.2),
        (decompose.rz2u, "rz", 0.4),
    ],
)
def test_single_gate_to_u_matches_matrix(builder, gate_name, theta):
    gate_info = builder(0) if theta is None else builder(theta, 0)
    mat = u_mat(*gate_info[1:-1])
    base = gate_matrix_dict[gate_name]
    expected = base(theta) if callable(base) else base
    assert is_equiv_unitary(mat, expected)


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_cz_decompose_matrix(basis, convert):
    gates = decompose.cz_decompose(0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["cz"])


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_cx_decompose_matrix(basis, convert):
    gates = decompose.cx_decompose(0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["cx"])


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_cy_decompose_matrix(basis, convert):
    gates = decompose.cy_decompose(0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["cy"])


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_swap_decompose_matrix(basis, convert):
    gates = decompose.swap_decompose(0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["swap"])


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_iswap_decompose_matrix(basis, convert):
    gates = decompose.iswap_decompose(0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["iswap"])


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_ecr_decompose_matrix(basis, convert):
    gates = decompose.ecr_decompose(0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["ecr"])


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_rxx_decompose_matrix(basis, convert):
    theta = 0.37
    gates = decompose.rxx_decompose(theta, 0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["rxx"](theta))


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_ryy_decompose_matrix(basis, convert):
    theta = -0.52
    gates = decompose.ryy_decompose(theta, 0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["ryy"](theta))


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
def test_rzz_decompose_matrix(basis, convert):
    theta = 0.9
    gates = decompose.rzz_decompose(theta, 0, 1, convert, basis)
    unitary = _circuit_unitary(2, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["rzz"](theta))


def test_ccx_decompose_matrix():
    gates = decompose.ccx_decompose(0, 1, 2)
    unitary = _circuit_unitary(3, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["ccx"])


def test_ccz_decompose_matrix():
    gates = decompose.ccz_decompose(0, 1, 2)
    unitary = _circuit_unitary(3, gates)
    assert is_equiv_unitary(unitary, gate_matrix_dict["ccz"])


# ════════════════════════════════════════════════════════════════════
#  Appended: boundary cases, invariants, and large-scale decomposition
# ════════════════════════════════════════════════════════════════════


def test_u_dot_u_is_associative():
    """Composing three U gates is associative up to global phase."""
    import random

    random.seed(0)

    def rnd():
        return ("u", random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3), 0)

    for _ in range(20):
        a, b, c = rnd(), rnd(), rnd()
        left = decompose.u_dot_u(decompose.u_dot_u(a, b), c)
        right = decompose.u_dot_u(a, decompose.u_dot_u(b, c))
        assert is_equiv_unitary(u_mat(*left[1:-1]), u_mat(*right[1:-1]))


def test_u_dot_u_with_identity_is_noop():
    """Composing a U with the identity U (all-zero angles) leaves it unchanged."""
    base = ("u", 0.6, -0.3, 0.9, 0)
    identity = ("u", 0.0, 0.0, 0.0, 0)
    composed = decompose.u_dot_u(base, identity)
    assert is_equiv_unitary(u_mat(*composed[1:-1]), u_mat(*base[1:-1]))
    composed2 = decompose.u_dot_u(identity, base)
    assert is_equiv_unitary(u_mat(*composed2[1:-1]), u_mat(*base[1:-1]))


def test_u_dot_u_preserves_qubit_index():
    """The composed U gate keeps the operand qubit index."""
    u1 = ("u", 0.3, -0.2, 1.1, 7)
    u2 = ("u", -0.7, 0.5, -0.4, 7)
    combined = decompose.u_dot_u(u1, u2)
    assert combined[0] == "u"
    assert combined[-1] == 7


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
@pytest.mark.parametrize("qubits", [(2, 5), (5, 2), (3, 0)])
def test_cx_decompose_on_nonzero_qubits(basis, convert, qubits):
    """CX decomposition is correct on arbitrary (and reversed) qubit indices."""
    nq = max(qubits) + 1
    control, target = qubits
    gates = decompose.cx_decompose(control, target, convert, basis)
    unitary = _circuit_unitary(nq, gates)
    expected = _expand_gate(nq, gate_matrix_dict["cx"], [control, target])
    assert is_equiv_unitary(unitary, expected)


@pytest.mark.parametrize("basis", ["cz", "cx", "iswap", "ecr"])
@pytest.mark.parametrize("convert", [False, True])
@pytest.mark.parametrize("qubits", [(2, 5), (4, 1)])
def test_swap_decompose_on_nonzero_qubits(basis, convert, qubits):
    """SWAP decomposition is correct on arbitrary qubit indices."""
    nq = max(qubits) + 1
    q1, q2 = qubits
    gates = decompose.swap_decompose(q1, q2, convert, basis)
    unitary = _circuit_unitary(nq, gates)
    expected = _expand_gate(nq, gate_matrix_dict["swap"], [q1, q2])
    assert is_equiv_unitary(unitary, expected)


def test_ccx_decompose_on_nonzero_qubits():
    """Toffoli decomposition is correct on non-trivial qubit indices."""
    gates = decompose.ccx_decompose(1, 3, 4)
    unitary = _circuit_unitary(5, gates)
    expected = _expand_gate(5, gate_matrix_dict["ccx"], [1, 3, 4])
    assert is_equiv_unitary(unitary, expected)


def test_decompose_pass_is_identity_on_two_qubit_circuit():
    """ThreeQubitGateDecompose leaves a circuit with no 3-qubit gates unchanged."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.5, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    out = decompose.ThreeQubitGateDecompose().run(qc)
    assert out.gates == qc.gates


def test_decompose_pass_on_empty_circuit():
    """ThreeQubitGateDecompose on an empty circuit yields an empty gate list."""
    qc = QuantumCircuit(3, 3)
    out = decompose.ThreeQubitGateDecompose().run(qc)
    assert out.gates == []
    assert out.nqubits == 3
    assert out.ncbits == 3


def test_decompose_pass_removes_only_three_qubit_gates():
    """Mixed circuit: 3-qubit gates expand, others survive verbatim."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.ccx(0, 1, 2)
    qc.cx(1, 2)
    qc.ccz(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    out = decompose.ThreeQubitGateDecompose().run(qc)
    names = [g[0] for g in out.gates]
    assert "ccx" not in names
    assert "ccz" not in names
    # Surviving non-3q gates: original h, cx and measure are still present.
    assert names.count("measure") == 1
    # The decomposed gates introduce only 1q and 2q gates.
    for g in out.gates:
        if g[0] == "measure":
            continue
        assert g[0] in {"h", "cx", "t", "tdg"}


def test_decompose_pass_is_idempotent():
    """Running ThreeQubitGateDecompose twice equals running it once."""
    qc = QuantumCircuit(3, 3)
    qc.ccx(0, 1, 2)
    qc.ccz(0, 1, 2)
    once = decompose.ThreeQubitGateDecompose().run(qc)
    twice = decompose.ThreeQubitGateDecompose().run(once)
    assert once.gates == twice.gates

