r"""Torch matrix utilities for simulator gates."""

from __future__ import annotations

import math

import torch


def _complex_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Map a real or complex torch dtype to its complex counterpart.

    Args:
        dtype (*torch.dtype | None*): Input dtype; ``None`` defaults to ``complex128``.

    Returns:
        Complex torch dtype (``complex64`` or ``complex128``).
    """
    if dtype in {torch.complex64, torch.complex128}:
        return dtype
    if dtype == torch.float32:
        return torch.complex64
    return torch.complex128


def _as_angle(theta, *, device: torch.device | None):
    """Convert a rotation angle to a float64 tensor on the given device.

    Args:
        theta: Rotation angle in radians (scalar or tensor).
        device (*torch.device | None*): Target device.

    Returns:
        ``torch.Tensor`` of dtype ``float64``.
    """
    if isinstance(theta, torch.Tensor):
        return theta.to(device=device)
    return torch.tensor(float(theta), device=device, dtype=torch.float64)


ket0 = torch.tensor([[1.0], [0.0]], dtype=torch.complex128)
ket1 = torch.tensor([[0.0], [1.0]], dtype=torch.complex128)


def ketn0(nqubits: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Build the all-zero computational basis state |0⋯0⟩ as a column vector.

    Args:
        nqubits (*int*): Number of qubits.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Torch tensor of shape ``(2**nqubits, 1)``.
    """
    k0 = ket0 if device is None else ket0.to(device=device)
    state = k0
    for _ in range(nqubits - 1):
        state = torch.kron(state, k0)
    return state


def ketn1(nqubits: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Build the all-one computational basis state |1⋯1⟩ as a column vector.

    Args:
        nqubits (*int*): Number of qubits.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Torch tensor of shape ``(2**nqubits, 1)``.
    """
    k1 = ket1 if device is None else ket1.to(device=device)
    state = k1
    for _ in range(nqubits - 1):
        state = torch.kron(state, k1)
    return state


id_mat = torch.eye(2, dtype=torch.complex128)
x_mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
y_mat = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128)
z_mat = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
h_mat = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2.0)
s_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex128)
sdg_mat = torch.tensor([[1.0, 0.0], [0.0, -1.0j]], dtype=torch.complex128)
t_mat = torch.tensor([[1.0, 0.0], [0.0, torch.exp(1.0j * torch.tensor(math.pi / 4.0))]], dtype=torch.complex128)
tdg_mat = torch.tensor([[1.0, 0.0], [0.0, torch.exp(-1.0j * torch.tensor(math.pi / 4.0))]], dtype=torch.complex128)
sx_mat = torch.tensor(
    [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=torch.complex128
) / 2.0
sxdg_mat = torch.tensor(
    [[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]], dtype=torch.complex128
) / 2.0

swap_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.complex128,
)

iswap_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0j, 0.0],
        [0.0, 1.0j, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.complex128,
)

ecr_mat = torch.tensor(
    [
        [0.0, 0.0, 1.0, 1.0j],
        [0.0, 0.0, 1.0j, 1.0],
        [1.0, -1.0j, 0.0, 0.0],
        [-1.0j, 1.0, 0.0, 0.0],
    ],
    dtype=torch.complex128,
) / math.sqrt(2.0)

cx_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=torch.complex128,
)



cy_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0j],
        [0.0, 0.0, 1.0j, 0.0],
    ],
    dtype=torch.complex128,
)



cz_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ],
    dtype=torch.complex128,
)

ccz_mat = torch.diag(torch.tensor([1, 1, 1, 1, 1, 1, 1, -1], dtype=torch.complex128))

ccx_mat = torch.tensor(
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
    dtype=torch.complex128,
)






def rx_mat(theta, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the single-qubit Rx(θ) rotation gate matrix.

    Args:
        theta: Rotation angle in radians.
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        2×2 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    c = torch.cos(0.5 * t)
    s = torch.sin(0.5 * t)
    out = torch.zeros((2, 2), dtype=cdtype, device=device)
    out[0, 0] = c
    out[1, 1] = c
    out[0, 1] = -1j * s
    out[1, 0] = -1j * s
    return out


def ry_mat(theta, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the single-qubit Ry(θ) rotation gate matrix.

    Args:
        theta: Rotation angle in radians.
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        2×2 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    c = torch.cos(0.5 * t)
    s = torch.sin(0.5 * t)
    out = torch.zeros((2, 2), dtype=cdtype, device=device)
    out[0, 0] = c
    out[1, 1] = c
    out[0, 1] = -s
    out[1, 0] = s
    return out


def rz_mat(theta, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the single-qubit Rz(θ) rotation gate matrix.

    Args:
        theta: Rotation angle in radians.
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        2×2 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    out = torch.zeros((2, 2), dtype=cdtype, device=device)
    out[0, 0] = torch.exp(-0.5j * t)
    out[1, 1] = torch.exp(0.5j * t)
    return out


def u_mat(theta, phi, lamda, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the single-qubit U(θ,φ,λ) general unitary gate matrix.

    Args:
        theta: Polar angle in radians.
        phi: First azimuthal angle in radians.
        lamda: Lambda angle in radians (spelled ``lamda`` to avoid Python keyword).
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        2×2 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    p = _as_angle(phi, device=device)
    l = _as_angle(lamda, device=device)
    out = torch.zeros((2, 2), dtype=cdtype, device=device)
    out[0, 0] = torch.cos(t / 2)
    out[0, 1] = -torch.exp(1.0j * l) * torch.sin(t / 2)
    out[1, 0] = torch.exp(1.0j * p) * torch.sin(t / 2)
    out[1, 1] = torch.exp(1.0j * (p + l)) * torch.cos(t / 2)
    return out


def rxx_mat(theta, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the two-qubit Rxx(θ) rotation gate matrix.

    Args:
        theta: Rotation angle in radians.
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        4×4 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    out = torch.zeros((4, 4), dtype=cdtype, device=device)
    out[0, 0] = c
    out[1, 1] = c
    out[2, 2] = c
    out[3, 3] = c
    out[0, 3] = -1j * s
    out[1, 2] = -1j * s
    out[2, 1] = -1j * s
    out[3, 0] = -1j * s
    return out


def ryy_mat(theta, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the two-qubit Ryy(θ) rotation gate matrix.

    Args:
        theta: Rotation angle in radians.
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        4×4 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    c = torch.cos(t / 2)
    s = torch.sin(t / 2)
    out = torch.zeros((4, 4), dtype=cdtype, device=device)
    out[0, 0] = c
    out[1, 1] = c
    out[2, 2] = c
    out[3, 3] = c
    out[0, 3] = 1j * s
    out[1, 2] = -1j * s
    out[2, 1] = -1j * s
    out[3, 0] = 1j * s
    return out


def rzz_mat(theta, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
    """Construct the two-qubit Rzz(θ) rotation gate matrix.

    Args:
        theta: Rotation angle in radians.
        device (*torch.device | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.
        dtype (*torch.dtype | None*): Torch data type. Defaults to ``None``.

    Returns:
        4×4 complex ``torch.Tensor``.
    """
    cdtype = _complex_dtype(dtype)
    t = _as_angle(theta, device=device)
    out = torch.zeros((4, 4), dtype=cdtype, device=device)
    out[0, 0] = torch.exp(-1j * t / 2)
    out[1, 1] = torch.exp(1j * t / 2)
    out[2, 2] = torch.exp(1j * t / 2)
    out[3, 3] = torch.exp(-1j * t / 2)
    return out


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
    "cy": cy_mat,
    "cz": cz_mat,
    "rx": rx_mat,
    "ry": ry_mat,
    "rz": rz_mat,
    "u": u_mat,
    "rxx": rxx_mat,
    "ryy": ryy_mat,
    "rzz": rzz_mat,
    "ccz": ccz_mat,
    "ccx": ccx_mat,
}
