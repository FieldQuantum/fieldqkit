# The following function is adapted from MindQuantum.
# Original function is licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original copyright notice:
# Copyright 2021 Huawei Technologies Co., Ltd

# Modified by XX Xiao on 2024.

"""A toolkit for KAK and U3 decompositions."""

import numpy as np
from scipy import linalg
from math import atan2

def generate_random_unitary_matrix(dim: int, seed: int | None = None) -> np.ndarray:
    r"""
    Generate a random complex unitary matrix of the specified dimension.

    Args:
        dim (int): The dimension of the unitary matrix.
        seed (int | None): A seed for the random number generator to ensure reproducibility. Defaults to None.

    Returns:
        np.ndarray: A dim x dim complex unitary matrix.
    """
    from scipy.stats import unitary_group
    U = unitary_group.rvs(dim, size=1, random_state=seed)
    return U

def is_equiv_unitary(mat1: np.ndarray, mat2: np.ndarray) -> bool:
    r"""
    Distinguish whether two unitary operators are equivalent, regardless of the global phase.

    Args:
        mat1 (np.ndarray): The first unitary matrix to compare.
        mat2 (np.ndarray): The second unitary matrix to compare.

    Raises:
        ValueError: If mat1 and mat2 have different dimensions.
        ValueError: If mat1 is not unitary.
        ValueError: If mat2 is not unitary.

    Returns:
        bool: True if mat1 and mat2 are equivalent; False otherwise.
    """
    if mat1.shape != mat2.shape:
        raise ValueError(f'Input matrices have different dimensions: {mat1.shape}, {mat2.shape}.')
    d = mat1.shape[0]
    if not np.allclose(mat1 @ mat1.conj().T, np.identity(d)):
        raise ValueError('mat1 is not unitary')
    if not np.allclose(mat2 @ mat2.conj().T, np.identity(d)):
        raise ValueError('mat2 is not unitary')
    mat1f = mat1.ravel()
    mat2f = mat2.ravel()
    idx_uf = np.flatnonzero(mat1f.round(4))  # cut to some precision
    idx_vf = np.flatnonzero(mat2f.round(4))
    try:
        if np.array_equal(idx_uf, idx_vf):
            coe = mat1f[idx_uf] / mat2f[idx_vf]
            return np.allclose(coe / coe[0], np.ones(len(idx_uf)), atol=1e-6)
        return False
    except ValueError:
        return False
    
def simult_svd(mat1: np.ndarray, mat2: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    r"""
    Simultaneous SVD of two matrices, based on Eckart-Young theorem.
    Given two real matrices A and B who satisfy the condition of simultaneous SVD, then $A = U D_1 V^{\dagger}, B = U D_2 V^{\dagger}$.

    Args:
        mat1: real matrix
        mat2: real matrix

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
            A tuple containing two tuples:
            The first tuple contains the orthogonal matrices U and V (both in SO(2)).
            The second tuple contains the diagonal matrices D1 and D2.
    """
    if mat1.shape != mat2.shape:
        raise ValueError(f'mat1 and mat2 have different dimensions: {mat1.shape}, {mat2.shape}.')
    d = mat1.shape[0]

    # real orthogonal matrices decomposition
    u_a, d_a, v_a_h = linalg.svd(mat1)
    u_a_h = u_a.conj().T
    v_a = v_a_h.conj().T

    if np.count_nonzero(d_a) != d:
        raise ValueError('Not implemented yet for the situation that mat1 is not full-rank')
    # g commutes with d
    g = u_a_h @ mat2 @ v_a
    # because g is hermitian, eigen-decomposition is its spectral decomposition
    _, p = linalg.eigh(g)  # p is unitary or orthogonal
    u = u_a @ p
    v = v_a @ p

    # ensure det(u_a) == det(v_a) == +1
    if linalg.det(u) < 0:
        u[:, 0] *= -1
    if linalg.det(v) < 0:
        v[:, 0] *= -1

    d1 = u.conj().T @ mat1 @ v
    d2 = u.conj().T @ mat2 @ v
    return (u, v), (d1, d2) # four real matrix

def glob_phase(mat: np.ndarray) -> float:
    r"""
    Extract the global phase $\alpha$ from a d x d matrix. $U = e^{i\alpha} S$ in which S is in SU(d).

    Args:
        mat: A d x d unitary matrix.

    Returns:
        float: Global phase rad, in range of (-pi, pi].
    """
    d = mat.shape[0]
    if d == 0:
        raise ZeroDivisionError("Dimension of mat can not be zero.")
    exp_alpha = linalg.det(mat) ** (1 / d)
    return np.angle(exp_alpha)

def remove_glob_phase(mat: np.ndarray) -> np.ndarray:
    r"""
    Remove the global phase of a 2 x 2 unitary matrix by means of ZYZ decomposition.

    That is, remove $e^{i\alpha}$ from $U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)$ and return $R_z(\phi) R_y(\theta) R_z(\lambda)$.

    Args:
        mat: A 2 x 2 unitary matrix.

    Returns:
        np.ndarray: A 2 x 2 matrix without global phase.
    """
    alpha = glob_phase(mat)
    return mat * np.exp(-1j * alpha)

def kron_factor_4x4_to_2x2s(mat: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]:
    r"""
    Decompose a 4 x 4 matrix U into the Kronecker product of two 2 x 2 unitary matrices $U = A \otimes B$ and a global scalar factor.

    This function assumes the input matrix is the Kronecker product of two 2x2 unitary matrices.
    If the matrix is not factorizable as a Kronecker product of two 2x2 unitaries, 
    or if the matrix has a zero determinant, the output may be incorrect or an error is raised.

    Args:
        mat (np.ndarray): A 4 x 4 unitary matrix to be factored.

    Returns:
        tuple[complex, np.ndarray, np.ndarray]: 
            A complex scalar g representing the global factor.
            f1 (np.ndarray): A 2 x 2 matrix, representing part of the Kronecker product.
            f2 (np.ndarray): Another 2 x 2 matrix, representing part of the Kronecker product.

    Raises:
        ValueError: If the input matrix cannot be tensor-factored into two 2 x 2 matrices.
        ZeroDivisionError: If a zero determinant causes a division by zero during factor extraction.

    """

    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(mat[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype = np.complex128)
    f2 = np.zeros((2, 2), dtype = np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = mat[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = mat[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    f1 /= np.sqrt(np.linalg.det(f1)) or 1
    f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase.
    denominator = f1[a >> 1, b >> 1] * f2[a & 1, b & 1]
    if denominator == 0:
        raise ZeroDivisionError("denominator cannot be zero.")
    g = mat[a, b] / denominator
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    return g, f1, f2

def kak_decompose(mat: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""
    Perform KAK decomposition of an arbitrary two-qubit gate.

    For more detail, please refer to `An Introduction to Cartan's KAK Decomposition for QC
    Programmers` [click here](https://arxiv.org/abs/quant-ph/0406176).

    Args:
        mat (np.ndarray): A 4 x 4 unitary matrix.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: 
            rots1: A list of four 2 x 2 matrices representing the rotation gates acting on the first qubit.
            rots2: A list of four 2 x 2 matrices representing the rotation gates acting on the second qubit.

    Raises:
        ValueError: If the input matrix is not a valid 4x4 unitary matrix.
    """
    M = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex) / np.sqrt(2)
    M_DAG = M.conj().T
    A = np.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]],dtype=complex)
    pauli_i = np.eye(2, dtype=complex)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    # construct a new matrix replacing U
    u_su4 = M_DAG @ remove_glob_phase(mat) @ M  # ensure the decomposed object is in SU(4)
    ur = np.real(u_su4)  # real part of u_su4
    ui = np.imag(u_su4)  # imagine part of u_su4

    # simultaneous SVD decomposition
    (q_left, q_right), (dr, di) = simult_svd(ur, ui)
    d = dr + 1j * di

    _, a1, a0 = kron_factor_4x4_to_2x2s(M @ q_left @ M_DAG)
    _, b1, b0 = kron_factor_4x4_to_2x2s(M @ q_right.T @ M_DAG)

    k = linalg.inv(A) @ np.angle(np.diag(d))
    h1, h2, h3 = -k[1:]

    u0 = 1j / np.sqrt(2) * (pauli_x + pauli_z) @ linalg.expm(-1j * (h1 - np.pi / 4) * pauli_x)
    v0 = -1j / np.sqrt(2) * (pauli_x + pauli_z)
    u1 = linalg.expm(-1j * h3 * pauli_z)
    v1 = linalg.expm(1j * h2 * pauli_z)
    w = (pauli_i - 1j * pauli_x) / np.sqrt(2)

    # list of operators
    rots1 = [b0, u0, v0, a0 @ w]  # rotation gate on idx1
    rots2 = [b1, u1, v1, a1 @ w.conj().T]
    return rots1, rots2

def zyz_decompose(mat: np.ndarray) -> tuple[float, float, float, float]:
    r"""
    Perform the ZYZ decomposition of a 2 x 2 unitary matrix.

    The decomposition is based on the relation:

    $$
        U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)
    $$

    Args:
        mat: A 2 x 2 unitary matrix to decompose.

    Returns:
        tuple[float, float, float, float]: A tuple of four phase angles:
            theta: The rotation angle of the R_y gate.
            phi: The first rotation angle of the R_z gate.
            lambda: The second rotation angle of the R_z gate.
            alpha: The global phase factor.
    """
    mat = mat.astype(np.complex128)
    if mat.shape != (2, 2):
        raise ValueError('Input matrix should be a 2*2 matrix')
    coe = linalg.det(mat) ** (-0.5)
    alpha = -np.angle(coe)
    v = coe * mat
    v = v.round(10)
    theta = 2 * atan2(abs(v[1, 0]), abs(v[0, 0]))
    phi_lam_sum = 2 * np.angle(v[1, 1])
    phi_lam_diff = 2 * np.angle(v[1, 0])
    phi = (phi_lam_sum + phi_lam_diff) / 2
    lam = (phi_lam_sum - phi_lam_diff) / 2
    return float(theta), float(phi), float(lam), float(alpha)

def u3_decompose(mat: np.ndarray) -> tuple[float, float, float, float]:
    r"""
    Decompose a 2 x 2 unitary matrix into U3 gate and obtain the parameters and global phase.
    
    The decomposition is based on the relation:

    $$
        U = e^{i \cdot p} U3(\theta, \phi, \lambda)
    $$

    Args:
        mat (np.ndarray): A 2 x 2 unitary matrix to decompose.

    Returns:
        tuple[float, float, float, float]: A tuple containing the three parameters $\theta$, $\phi$, $\lambda$ of a standard U3 gate and global phase $p$.
    """
    theta, phi, lam, alpha = zyz_decompose(mat)
    phase = alpha - (phi + lam) / 2
    return theta, phi, lam, phase