import numpy as np
import pytest

from quantum_hw.compile import decompose
from quantum_hw.circuit.matrix import gate_matrix_dict, u_mat
from quantum_hw.circuit.utils import is_equiv_unitary


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

