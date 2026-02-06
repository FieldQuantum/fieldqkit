# Copyright (c) 2024 XX Xiao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

r"""Some common decomposition methods for two-qubit and three-qubit gates."""

import numpy as np
from typing import Literal

from ..circuit.utils import u3_decompose
from ..circuit.matrix import u_mat
from ..circuit import QuantumCircuit
from .basepasses import TranspilerPass


def u_dot_u(u_info1: tuple, u_info2: tuple) -> tuple:
    assert u_info1[-1] == u_info2[-1]
    u_mat1 = u_mat(*u_info1[1:-1])
    u_mat2 = u_mat(*u_info2[1:-1])

    new_u = u_mat2 @ u_mat1
    theta, phi, lamda, _ = u3_decompose(new_u)
    return ("u", theta, phi, lamda, u_info1[-1])


def x2u(qubit: int) -> tuple:
    return ("u", np.pi, np.pi / 2, -np.pi / 2, qubit)

def y2u(qubit: int) -> tuple:
    return ("u", np.pi, 0.0, 0, qubit)

def z2u(qubit: int) -> tuple:
    return ("u", 0.0, 0.0, np.pi, qubit)

def h2u(qubit: int) -> tuple:
    return ("u", np.pi / 2, 0.0, np.pi, qubit)

def s2u(qubit: int) -> tuple:
    return ("u", 0.0, np.pi / 4, np.pi / 4, qubit)

def sdg2u(qubit: int) -> tuple:
    return ("u", 0.0, -np.pi / 4, -np.pi / 4, qubit)

def t2u(qubit: int) -> tuple:
    return ("u", 0.0, np.pi / 8, np.pi / 8, qubit)

def tdg2u(qubit: int) -> tuple:
    return ("u", 0.0, -np.pi / 8, -np.pi / 8, qubit)

def sx2u(qubit: int) -> tuple:
    return ("u", np.pi / 2, -np.pi / 2, np.pi / 2, qubit)

def sxdg2u(qubit: int) -> tuple:
    return ("u", np.pi / 2, np.pi / 2, -np.pi / 2, qubit)

def rx2u(theta: float, qubit: int) -> tuple:
    return ("u", theta, -np.pi / 2, np.pi / 2, qubit)

def ry2u(theta: float, qubit: int) -> tuple:
    return ("u", theta, 0.0, 0.0, qubit)

def rz2u(theta: float, qubit: int) -> tuple:
    return ("u", 0.0, 0.0, theta, qubit)


def cz_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
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
    if convert_single_qubit_gate_to_u:
        gates = cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            rz2u(theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    else:
        gates = cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("rz", theta, qubit2),
        ] + cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    return gates


def cp_decompose(
    theta: float,
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],
) -> list:
    if convert_single_qubit_gate_to_u:
        gates = [
            rz2u(theta / 2, control_qubit),
            rz2u(theta / 2, target_qubit),
        ] + cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            rz2u(-theta / 2, target_qubit),
        ] + cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    else:
        gates = [
            ("rz", theta / 2, control_qubit),
            ("rz", theta / 2, target_qubit),
        ] + cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis) + [
            ("rz", -theta / 2, target_qubit),
        ] + cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    return gates


def ccx_decompose(control_qubit1: int, control_qubit2: int, target_qubit: int):
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


def cswap_decompose(control_qubit1: int, control_qubit2: int, target_qubit: int):
    gates = [
        ("cx", target_qubit, control_qubit2),
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
        ("cx", target_qubit, control_qubit2),
    ]
    return gates


def ccz_decompose(control_qubit1: int, control_qubit2: int, target_qubit: int):
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


def ccx_decompose_mute_phase(control_qubit1: int, control_qubit2: int, target_qubit: int):
    gates = [
        ("u", np.pi / 4, 0, 0, target_qubit),
        ("cx", control_qubit2, target_qubit),
        ("u", np.pi / 4, 0, 0, target_qubit),
        ("cx", control_qubit1, target_qubit),
        ("u", np.pi / 4, -np.pi, -np.pi, target_qubit),
        ("cx", control_qubit2, target_qubit),
        ("u", np.pi / 4, -np.pi, -np.pi, target_qubit),
    ]
    return gates[::-1]


class ThreeQubitGateDecompose(TranspilerPass):
    """A transpiler pass that decomposes three-qubit gates into combinations of single- and two-qubit gates."""

    def __init__(self):
        super().__init__()

    def run(self, qc: QuantumCircuit):
        new = []
        for gate_info in qc.gates:
            if gate_info[0] == "ccx":
                new += ccx_decompose(*gate_info[1:])
            elif gate_info[0] == "ccz":
                new += ccz_decompose(*gate_info[1:])
            elif gate_info[0] == "cswap":
                new += cswap_decompose(*gate_info[1:])
            else:
                new.append(gate_info)
        new_qc = qc.deepcopy()
        new_qc.gates = new
        return new_qc
