"""Kraus operator definitions for noise channels."""

import torch


def _pauli_matrices_torch(dtype=torch.complex64, device=None):
    """Return single-qubit Pauli matrices as torch tensors."""
    I = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    return I, X, Y, Z


def depolarize1_kraus(p: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for single-qubit depolarizing channel.

    ρ' = (1-p)ρ + (p/3)(Xρ X + Yρ Y + Zρ Z)

    Args:
        p (float): Error probability (0 ≤ p ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Depolarizing probability must be in [0, 1], got {p}")

    I, X, Y, Z = _pauli_matrices_torch(dtype, device)
    K0 = (1.0 - p) ** 0.5 * I
    K1 = (p / 3.0) ** 0.5 * X
    K2 = (p / 3.0) ** 0.5 * Y
    K3 = (p / 3.0) ** 0.5 * Z

    return [K0, K1, K2, K3]


def depolarize2_kraus(p: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for two-qubit depolarizing channel.

    ρ' = (1-p)ρ + (p/15)·sum_{P ≠ I⊗I} P ρ P†

    Args:
        p (float): Error probability (0 ≤ p ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Depolarizing probability must be in [0, 1], got {p}")

    I, X, Y, Z = _pauli_matrices_torch(dtype, device)
    paulis = [I, X, Y, Z]
    names = ['I', 'X', 'Y', 'Z']

    # K0: identity (no error)
    K0 = (1.0 - p) ** 0.5 * torch.kron(I, I)
    kraus = [K0]

    # K1..K15: the 15 non-identity 2q Paulis
    for name1, P1 in zip(names, paulis):
        for name2, P2 in zip(names, paulis):
            if name1 == 'I' and name2 == 'I':
                continue
            K = (p / 15.0) ** 0.5 * torch.kron(P1, P2)
            kraus.append(K)

    return kraus


def x_error_kraus(p: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for single-qubit bit-flip (X) error.

    ρ' = (1-p)ρ + p·Xρ X

    Args:
        p (float): Error probability (0 ≤ p ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Error probability must be in [0, 1], got {p}")

    I, X, _, _ = _pauli_matrices_torch(dtype, device)
    K0 = (1.0 - p) ** 0.5 * I
    K1 = p ** 0.5 * X

    return [K0, K1]


def y_error_kraus(p: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for single-qubit Y error.

    ρ' = (1-p)ρ + p·Yρ Y

    Args:
        p (float): Error probability (0 ≤ p ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Error probability must be in [0, 1], got {p}")

    I, _, Y, _ = _pauli_matrices_torch(dtype, device)
    K0 = (1.0 - p) ** 0.5 * I
    K1 = p ** 0.5 * Y

    return [K0, K1]


def z_error_kraus(p: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for single-qubit phase-flip (Z) error.

    ρ' = (1-p)ρ + p·Zρ Z

    Args:
        p (float): Error probability (0 ≤ p ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Error probability must be in [0, 1], got {p}")

    I, _, _, Z = _pauli_matrices_torch(dtype, device)
    K0 = (1.0 - p) ** 0.5 * I
    K1 = p ** 0.5 * Z

    return [K0, K1]


def amplitude_damping_kraus(gamma: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for amplitude damping (energy dissipation).

    Models decay of excited state to ground state.

    Args:
        gamma (float): Damping parameter (0 ≤ γ ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"Damping parameter must be in [0, 1], got {gamma}")

    K0 = torch.tensor([[1.0, 0.0], [0.0, (1.0 - gamma) ** 0.5]], dtype=dtype, device=device)
    K1 = torch.tensor([[0.0, gamma ** 0.5], [0.0, 0.0]], dtype=dtype, device=device)

    return [K0, K1]


def phase_damping_kraus(gamma: float, *, dtype=torch.complex64, device=None) -> list:
    """Kraus operators for phase damping (dephasing).

    Models loss of coherence without energy dissipation.

    Args:
        gamma (float): Dephasing parameter (0 ≤ γ ≤ 1).
        dtype (torch.dtype): Torch data type.
        device: Torch device.

    Returns:
        List of Kraus operators as torch tensors.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"Dephasing parameter must be in [0, 1], got {gamma}")

    K0 = torch.tensor([[1.0, 0.0], [0.0, (1.0 - gamma) ** 0.5]], dtype=dtype, device=device)
    K1 = torch.tensor([[0.0, 0.0], [0.0, gamma ** 0.5]], dtype=dtype, device=device)

    return [K0, K1]


def get_kraus_ops(gate_name: str, param: float, *, dtype=torch.complex64, device=None) -> list:
    """Return Kraus operators for a named noise channel.

    Args:
        gate_name (str): Name of the noise channel (e.g., 'depolarize1', 'x_error').
        param (float): Channel parameter (probability or damping coefficient).
        dtype (torch.dtype): Torch data type. Defaults to ``torch.complex64``.
        device: Torch device. Defaults to ``None``.

    Returns:
        List of Kraus operator matrices as torch tensors.

    Raises:
        ValueError: If gate_name is not recognized.
    """
    if gate_name == 'depolarize1':
        return depolarize1_kraus(param, dtype=dtype, device=device)
    elif gate_name == 'depolarize2':
        return depolarize2_kraus(param, dtype=dtype, device=device)
    elif gate_name == 'x_error':
        return x_error_kraus(param, dtype=dtype, device=device)
    elif gate_name == 'y_error':
        return y_error_kraus(param, dtype=dtype, device=device)
    elif gate_name == 'z_error':
        return z_error_kraus(param, dtype=dtype, device=device)
    elif gate_name == 'amplitude_damping':
        return amplitude_damping_kraus(param, dtype=dtype, device=device)
    elif gate_name == 'phase_damping':
        return phase_damping_kraus(param, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown noise channel: {gate_name}")
