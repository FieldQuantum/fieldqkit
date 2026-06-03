"""Some common decomposition methods for two-qubit and three-qubit gates.

SPDX-License-Identifier: Apache-2.0
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

import numpy as np
from typing import Literal

from ..circuit.utils import u3_decompose
from ..circuit.matrix import u_mat
from ..circuit import QuantumCircuit
from .basepasses import TranspilerPass


def u_dot_u(u_info1: tuple, u_info2: tuple) -> tuple:
    """Compose two single-qubit U gates by multiplying their matrices and re-decomposing.

    Args:
        u_info1 (tuple): First U gate info tuple ('u', theta, phi, lambda, qubit).
        u_info2 (tuple): Second U gate info tuple ('u', theta, phi, lambda, qubit).

    Returns:
        tuple: The composed U gate info tuple.
    """
    assert u_info1[-1] == u_info2[-1]
    u_mat1 = u_mat(*u_info1[1:-1])
    u_mat2 = u_mat(*u_info2[1:-1])

    new_u = u_mat2 @ u_mat1
    theta, phi, lamda, _ = u3_decompose(new_u)
    return ("u", theta, phi, lamda, u_info1[-1])


def x2u(qubit: int) -> tuple:
    """Convert a Pauli-X gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", np.pi, np.pi / 2, -np.pi / 2, qubit)

def y2u(qubit: int) -> tuple:
    """Convert a Pauli-Y gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", np.pi, 0.0, 0, qubit)

def z2u(qubit: int) -> tuple:
    """Convert a Pauli-Z gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", 0.0, 0.0, np.pi, qubit)

def h2u(qubit: int) -> tuple:
    """Convert a Hadamard gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", np.pi / 2, 0.0, np.pi, qubit)

def s2u(qubit: int) -> tuple:
    """Convert an S gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", 0.0, np.pi / 4, np.pi / 4, qubit)

def sdg2u(qubit: int) -> tuple:
    """Convert an S-dagger gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", 0.0, -np.pi / 4, -np.pi / 4, qubit)

def t2u(qubit: int) -> tuple:
    """Convert a T gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", 0.0, np.pi / 8, np.pi / 8, qubit)

def tdg2u(qubit: int) -> tuple:
    """Convert a T-dagger gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", 0.0, -np.pi / 8, -np.pi / 8, qubit)

def sx2u(qubit: int) -> tuple:
    """Convert an SX (√X) gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", np.pi / 2, -np.pi / 2, np.pi / 2, qubit)

def sxdg2u(qubit: int) -> tuple:
    """Convert an SX-dagger (√X†) gate to its equivalent U(θ, φ, λ) representation.

    Args:
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", np.pi / 2, np.pi / 2, -np.pi / 2, qubit)

def rx2u(theta: float, qubit: int) -> tuple:
    """Convert an RX(θ) rotation gate to its equivalent U(θ, φ, λ) representation.

    Args:
        theta (float): Rotation angle in radians.
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", theta, -np.pi / 2, np.pi / 2, qubit)

def ry2u(theta: float, qubit: int) -> tuple:
    """Convert an RY(θ) rotation gate to its equivalent U(θ, φ, λ) representation.

    Args:
        theta (float): Rotation angle in radians.
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", theta, 0.0, 0.0, qubit)

def rz2u(theta: float, qubit: int) -> tuple:
    """Convert an RZ(θ) rotation gate to its equivalent U(θ, φ, λ) representation.

    Args:
        theta (float): Rotation angle in radians.
        qubit (int): Target qubit index.

    Returns:
        tuple: U gate info tuple.
    """
    return ("u", 0.0, 0.0, theta, qubit)


def cz_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose a CZ gate into the specified two-qubit basis gate set.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if two_qubit_gate_basis == "cz":
        gates = [("cz", control_qubit, target_qubit)]
    elif two_qubit_gate_basis == "cx":
        if convert_single_qubit_gate_to_u:
            gates = [
                h2u(target_qubit), 
                ("cx", control_qubit, target_qubit), 
                h2u(target_qubit)
            ]
        else:
            gates = [
                ("h", target_qubit), 
                ("cx", control_qubit, target_qubit), 
                ("h", target_qubit)
            ]
    elif two_qubit_gate_basis == "ecr":
        if convert_single_qubit_gate_to_u:
            gates = [
                h2u(target_qubit),
                sdg2u(control_qubit),
                sxdg2u(target_qubit),
                ("ecr", control_qubit, target_qubit),
                x2u(control_qubit),
                h2u(target_qubit),
            ]
        else:
            gates = [
                ("h", target_qubit),
                ("sdg", control_qubit),
                ("sxdg", target_qubit),
                ("ecr", control_qubit, target_qubit),
                ("x", control_qubit),
                ("h", target_qubit),
            ]
    elif two_qubit_gate_basis == "iswap":
        if convert_single_qubit_gate_to_u:
            gates = [
                rz2u(np.pi / 2, target_qubit),
                rx2u(np.pi / 2, target_qubit),
                ("iswap", control_qubit, target_qubit),
                rx2u(np.pi / 2, control_qubit),
                rz2u(-np.pi / 2, control_qubit),
                rz2u(np.pi / 2, target_qubit),
                ("iswap", control_qubit, target_qubit),
                rz2u(np.pi / 2, target_qubit),
                rx2u(np.pi / 2, target_qubit),
            ]
        else:
            gates = [
                ("rz", np.pi / 2, target_qubit),
                ("rx", np.pi / 2, target_qubit),
                ("iswap", control_qubit, target_qubit),
                ("rx", np.pi / 2, control_qubit),
                ("rz", -np.pi / 2, control_qubit),
                ("rz", np.pi / 2, target_qubit),
                ("iswap", control_qubit, target_qubit),
                ("rz", np.pi / 2, target_qubit),
                ("rx", np.pi / 2, target_qubit),
            ]
    return gates

def cx_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose a CX (CNOT) gate into the specified two-qubit basis gate set.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if two_qubit_gate_basis in ["cz", "iswap", "ecr"]:
        if convert_single_qubit_gate_to_u:
            gates = [
                h2u(target_qubit),
            ] + cz_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
                h2u(target_qubit),
            ]
        else:
            gates = [
                ("h", target_qubit),
            ] + cz_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
                ("h", target_qubit),
            ]
    elif two_qubit_gate_basis == "cx":
        gates = [("cx", control_qubit, target_qubit)]
    
    return gates

def cy_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose a CY gate into CX and S/S-dagger gates in the specified basis.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if convert_single_qubit_gate_to_u:
        gates = [
            sdg2u(target_qubit),
        ] + cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            s2u(target_qubit),
        ]
    else:
        gates = [
            ("sdg", target_qubit),
        ] + cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("s", target_qubit),
        ]
    return gates

def swap_decompose(
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose a SWAP gate into the specified two-qubit basis gate set.

    Args:
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if two_qubit_gate_basis in ["cz", "cx", "ecr"]:
        gates = (
            cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
            + cx_decompose(qubit2, qubit1, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
            + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
        )
    elif two_qubit_gate_basis == "iswap":
        if convert_single_qubit_gate_to_u:
            gates = [
                ("iswap", qubit1, qubit2),
                sx2u(qubit2),
                ("iswap", qubit1, qubit2),
                sx2u(qubit1),
                ("iswap", qubit1, qubit2),
                sx2u(qubit2),
            ]
        else:
            gates = [
                ("iswap", qubit1, qubit2),
                ("sx", qubit2),
                ("iswap", qubit1, qubit2),
                ("sx", qubit1),
                ("iswap", qubit1, qubit2),
                ("sx", qubit2),
            ]
    return gates


def iswap_decompose(
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose an iSWAP gate into the specified two-qubit basis gate set.

    Args:
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if two_qubit_gate_basis == "iswap":
        gates = [("iswap", qubit1, qubit2)]
    elif two_qubit_gate_basis in ["cz", "cx", "ecr"]:
        if convert_single_qubit_gate_to_u:
            gates = [
                s2u(qubit1),
                s2u(qubit2),
                h2u(qubit1),
            ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ] + cx_decompose(qubit2, qubit1, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
                h2u(qubit2),
            ]
        else:
            gates = [
                ("s", qubit1),
                ("s", qubit2),
                ("h", qubit1),
            ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ] + cx_decompose(qubit2, qubit1, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
                ("h", qubit2),
            ]

    return gates


def ecr_decompose(
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose an ECR (echoed cross-resonance) gate into the specified two-qubit basis gate set.

    Args:
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if two_qubit_gate_basis == "ecr":
        gates = [("ecr", qubit1, qubit2)]
    elif two_qubit_gate_basis in ["cz", "cx", "iswap"]:
        if convert_single_qubit_gate_to_u:
            gates = [
                s2u(qubit1),
                sx2u(qubit2),
            ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
                x2u(qubit1),
            ]
        else:
            gates = [
                ("s", qubit1),
                ("sx", qubit2),
            ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
                ("x", qubit1),
            ]
    return gates


def rxx_decompose(
    theta: float,
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose an RXX(θ) parametric gate into the specified two-qubit basis gate set.

    Args:
        theta (float): Rotation angle in radians.
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if convert_single_qubit_gate_to_u:
        gates = [
            h2u(qubit1),
            h2u(qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            rz2u(theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            h2u(qubit1),
            h2u(qubit2),
        ]
    else:
        gates = [
            ("h", qubit1),
            ("h", qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("rz", theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("h", qubit1),
            ("h", qubit2),
        ]
    return gates


def ryy_decompose(
    theta: float,
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose an RYY(θ) parametric gate into the specified two-qubit basis gate set.

    Args:
        theta (float): Rotation angle in radians.
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if convert_single_qubit_gate_to_u:
        gates = [
            rx2u(np.pi / 2, qubit1),
            rx2u(np.pi / 2, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            rz2u(theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            rx2u(-np.pi / 2, qubit1),
            rx2u(-np.pi / 2, qubit2),
        ]
    else:
        gates = [
            ("rx", np.pi / 2, qubit1),
            ("rx", np.pi / 2, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("rz", theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("rx", -np.pi / 2, qubit1),
            ("rx", -np.pi / 2, qubit2),
        ]
    return gates


def rzz_decompose(
    theta: float,
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    """Decompose an RZZ(θ) parametric gate into the specified two-qubit basis gate set.

    Args:
        theta (float): Rotation angle in radians.
        qubit1 (int): First qubit index.
        qubit2 (int): Second qubit index.
        convert_single_qubit_gate_to_u (bool): Whether to convert single-qubit gates to U gates.
        two_qubit_gate_basis (Literal['cz', 'cx', 'iswap', 'ecr']): Target two-qubit basis gate.

    Returns:
        list: Decomposed gate info tuples.
    """
    if convert_single_qubit_gate_to_u:
        gates = cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            rz2u(theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    else:
        gates = cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("rz", theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    return gates


def ccx_decompose(control_qubit1: int, control_qubit2: int, target_qubit: int):
    """Decompose a Toffoli (CCX) gate into single- and two-qubit gates.

    Args:
        control_qubit1 (int): First control qubit index.
        control_qubit2 (int): Second control qubit index.
        target_qubit (int): Target qubit index.

    Returns:
        list: Decomposed gate info tuples.
    """
    gates = [
        ("h", target_qubit),
        ("cx", control_qubit2, target_qubit),
        ("tdg", target_qubit),
        ("cx", control_qubit1, target_qubit),
        ("t", target_qubit),
        ("cx", control_qubit2, target_qubit),
        ("t", control_qubit2),
        ("tdg", target_qubit),
        ("cx", control_qubit1, target_qubit),
        ("cx", control_qubit1, control_qubit2),
        ("t", target_qubit),
        ("t", control_qubit1),
        ("tdg", control_qubit2),
        ("h", target_qubit),
        ("cx", control_qubit1, control_qubit2),
    ]
    return gates


def ccz_decompose(control_qubit1: int, control_qubit2: int, target_qubit: int):
    """Decompose a CCZ gate into single- and two-qubit gates.

    Args:
        control_qubit1 (int): First control qubit index.
        control_qubit2 (int): Second control qubit index.
        target_qubit (int): Target qubit index.

    Returns:
        list: Decomposed gate info tuples.
    """
    gates = [
        ("cx", control_qubit2, target_qubit),
        ("tdg", target_qubit),
        ("cx", control_qubit1, target_qubit),
        ("t", target_qubit),
        ("cx", control_qubit2, target_qubit),
        ("t", control_qubit2),
        ("tdg", target_qubit),
        ("cx", control_qubit1, target_qubit),
        ("cx", control_qubit1, control_qubit2),
        ("t", target_qubit),
        ("t", control_qubit1),
        ("tdg", control_qubit2),
        ("h", target_qubit),
        ("cx", control_qubit1, control_qubit2),
        ("h", target_qubit),
    ]
    return gates


class ThreeQubitGateDecompose(TranspilerPass):
    """A transpiler pass that decomposes three-qubit gates into combinations of single- and two-qubit gates."""

    def __init__(self):
        """Initialize the three-qubit gate decomposition pass."""
        super().__init__()

    def run(self, qc: QuantumCircuit):
        """Decompose three-qubit gates (ccx, ccz) into one- and two-qubit gates.

        Args:
            qc (*QuantumCircuit*): Quantum circuit.

        Returns:
            ``QuantumCircuit`` with three-qubit gates decomposed.
        """
        new = []
        for gate_info in qc.gates:
            if gate_info[0] == "ccx":
                new += ccx_decompose(*gate_info[1:])
            elif gate_info[0] == "ccz":
                new += ccz_decompose(*gate_info[1:])
            else:
                new.append(gate_info)
        new_qc = qc.deepcopy()
        new_qc.gates = new
        return new_qc
