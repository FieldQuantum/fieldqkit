r"""The matrices corresponding to single-qubit, two-qubit and three-qubit gates."""

import numpy as np

ket0 = np.array([[1.0], [0.0]], dtype=complex)
ket1 = np.array([[0.0], [1.0]], dtype=complex)


def ketn0(nqubits: int) -> np.ndarray:
    state = ket0
    for _ in range(nqubits - 1):
        state = np.kron(state, ket0)
    return state


def ketn1(nqubits: int) -> np.ndarray:
    state = ket1
    for _ in range(nqubits - 1):
        state = np.kron(state, ket1)
    return state


id_mat = np.eye(2, dtype=complex)
x_mat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
y_mat = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
z_mat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
h_mat = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
s_mat = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
sdg_mat = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=complex)
t_mat = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4)]], dtype=complex)
tdg_mat = np.array([[1.0, 0.0], [0.0, np.exp(-1.0j * np.pi / 4)]], dtype=complex)
sx_mat = np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=complex) / 2
sxdg_mat = np.array([[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]], dtype=complex) / 2

swap_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)

iswap_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0j, 0.0],
        [0.0, 1.0j, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)

ecr_mat = np.array(
    [
        [0.0, 0.0, 1.0, 1.0j],
        [0.0, 0.0, 1.0j, 1.0],
        [1.0, -1.0j, 0.0, 0.0],
        [-1.0j, 1.0, 0.0, 0.0],
    ],
    dtype=complex,
) / np.sqrt(2)

cx_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=complex,
)

xc_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=complex,
)

cy_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, 1.0j, 0.0],
    ],
    dtype=complex,
)

yc_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 1.0j, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=complex,
)

cz_mat = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ],
    dtype=complex,
)

ccz_mat = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ],
    dtype=complex,
)

ccx_mat = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=complex,
)

cxc_mat = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    dtype=complex,
)

cswap_mat = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=complex,
)

swapc_mat = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=complex,
)


def r_mat(theta, phi):
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.exp(-1j * phi) * np.sin(theta / 2)],
            [-1j * np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def rx_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(0.5 * theta), -1.0j * np.sin(0.5 * theta)],
            [-1.0j * np.sin(0.5 * theta), np.cos(0.5 * theta)],
        ],
        dtype=complex,
    )


def ry_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(0.5 * theta), -np.sin(0.5 * theta)],
            [np.sin(0.5 * theta), np.cos(0.5 * theta)],
        ],
        dtype=complex,
    )


def rz_mat(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex)


def p_mat(theta: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * theta)]], dtype=complex)


def u_mat(theta: float, phi: float, lamda: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1.0j * lamda) * np.sin(theta / 2)],
            [np.exp(1.0j * phi) * np.sin(theta / 2), np.exp(1.0j * (phi + lamda)) * np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def u1_mat(lamda: float):
    return u_mat(0.0, 0.0, lamda)


def u2_mat(phi: float, lamda: float):
    return u_mat(np.pi / 2, phi, lamda)


def rxx_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def ryy_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2), 0, 0, 1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def rzz_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-1j * theta / 2), 0, 0, 0],
            [0, np.exp(1j * theta / 2), 0, 0],
            [0, 0, np.exp(1j * theta / 2), 0],
            [0, 0, 0, np.exp(-1j * theta / 2)],
        ],
        dtype=complex,
    )


def cp_mat(theta: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * theta)],
        ],
        dtype=complex,
    )


gate_matrix_dict = {
    "id": id_mat,
    "x": x_mat,
    "y": y_mat,
    "z": z_mat,
    "h": h_mat,
    "s": s_mat,
    "sdg": sdg_mat,
    "t": t_mat,
    "tdg": tdg_mat,
    "sx": sx_mat,
    "sxdg": sxdg_mat,
    "swap": swap_mat,
    "iswap": iswap_mat,
    "ecr": ecr_mat,
    "cx": cx_mat,
    "cnot": cx_mat,
    "cy": cy_mat,
    "cz": cz_mat,
    "rx": rx_mat,
    "ry": ry_mat,
    "rz": rz_mat,
    "p": p_mat,
    "u": u_mat,
    "r": r_mat,
    "rxx": rxx_mat,
    "ryy": ryy_mat,
    "rzz": rzz_mat,
    "cp": cp_mat,
    "ccz": ccz_mat,
    "ccx": ccx_mat,
    "cswap": cswap_mat,
}
