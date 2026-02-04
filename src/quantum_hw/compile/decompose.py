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
from quark.circuit.utils import u3_decompose
from quark.circuit.matrix import u_mat
from quark.circuit.quantumcircuit import QuantumCircuit
from .basepasses import TranspilerPass


def u_dot_u(u_info1: tuple, u_info2: tuple) -> tuple:
    assert u_info1[-1] == u_info2[-1]
    u_mat1 = u_mat(*u_info1[1:-1])
    u_mat2 = u_mat(*u_info2[1:-1])

    new_u = u_mat2 @ u_mat1
    theta, phi, lamda, _ = u3_decompose(new_u)
    return ("u", theta, phi, lamda, u_info1[-1])


def h2u(qubit: int) -> tuple:
    return ("u", np.pi / 2, 0.0, np.pi, qubit)


def x2u(qubit: int) -> tuple:
    return ("u", np.pi, np.pi / 2, -np.pi / 2, qubit)


def sdg2u(qubit: int) -> tuple:
    return ("u", 0.0, -0.7853981633974483, -0.7853981633974483, qubit)


def s2u(qubit: int) -> tuple:
    return ("u", 0.0, 0.7853981633974483, 0.7853981633974483, qubit)


def rx2u(theta: float, qubit: int) -> tuple:
    return ("u", theta, -np.pi / 2, np.pi / 2, qubit)


def ry2u(theta: float, qubit: int) -> tuple:
    return ("u", theta, 0.0, 0.0, qubit)


def rz2u(theta: float, qubit: int) -> tuple:
    return ("u", 0.0, 0.0, theta, qubit)


def convert_cx_to_iswap(control_qubit, target_qubit, convert_single_qubit_gate_to_u: bool):
    gates0 = [
        ("rz", -1.5707963267948966, control_qubit),
        ("ry", 1.5707963267948966, control_qubit),
        ("rz", 1.5707963267948966, control_qubit),
        ("ry", -1.5707963267948966, target_qubit),
        ("rz", -1.0094094858814842, target_qubit),
        ("iswap", control_qubit, target_qubit),
        ("rz", -1.5707963267948966, control_qubit),
        ("ry", 3.141592653589793, control_qubit),
        ("rz", 1.5707963267948966, target_qubit),
        ("ry", -1.5707963267948966, target_qubit),
        ("iswap", control_qubit, target_qubit),
        ("rz", -3.141592653589793, control_qubit),
        ("ry", -1.5707963267948966, control_qubit),
        ("rz", -1.0094094858814842, target_qubit),
        ("ry", -1.5707963267948966, target_qubit),
    ]

    new = []
    if convert_single_qubit_gate_to_u:
        for gate_info in gates0:
            if gate_info[0] == "rz":
                new_gate_info = rz2u(gate_info[1], gate_info[2])
                new.append(new_gate_info)
            elif gate_info[0] == "ry":
                new_gate_info = ry2u(gate_info[1], gate_info[2])
                new.append(new_gate_info)
            else:
                new.append(gate_info)
        gates0 = new
    return gates0


def convert_cz_to_iswap(control_qubit, target_qubit, convert_single_qubit_gate_to_u: bool):
    gates0 = [
        ("rz", -1.5707963267948966, control_qubit),
        ("ry", 1.5707963267948966, control_qubit),
        ("rz", 1.5707963267948966, control_qubit),
        ("rz", -1.5707963267948966, target_qubit),
        ("ry", 3.141592653589793, target_qubit),
        ("iswap", control_qubit, target_qubit),
        ("rz", -1.5707963267948966, control_qubit),
        ("ry", 3.141592653589793, control_qubit),
        ("rz", 1.5707963267948966, target_qubit),
        ("ry", -1.5707963267948966, target_qubit),
        ("iswap", control_qubit, target_qubit),
        ("ry", 1.5707963267948966, control_qubit),
        ("rz", 1.5707963267948966, target_qubit),
    ]
    new = []
    if convert_single_qubit_gate_to_u:
        for gate_info in gates0:
            if gate_info[0] == "rz":
                new_gate_info = rz2u(gate_info[1], gate_info[2])
                new.append(new_gate_info)
            elif gate_info[0] == "ry":
                new_gate_info = ry2u(gate_info[1], gate_info[2])
                new.append(new_gate_info)
            else:
                new.append(gate_info)
        gates0 = new
    return gates0


def cz_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cx", "cz", "iswap"],
) -> list:
    gates = []
    if two_qubit_gate_basis == "cz":
        gates.append(("cz", control_qubit, target_qubit))
    elif two_qubit_gate_basis == "cx":
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(target_qubit))
        else:
            gates.append(("h", target_qubit))
        gates.append(("cx", control_qubit, target_qubit))
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(target_qubit))
        else:
            gates.append(("h", target_qubit))
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cz_to_iswap(control_qubit, target_qubit, convert_single_qubit_gate_to_u)

    return gates


def cx_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cx", "cz", "iswap"],
) -> list:
    gates = []
    if two_qubit_gate_basis == "cz":
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(target_qubit))
        else:
            gates.append(("h", target_qubit))
        gates.append(("cz", control_qubit, target_qubit))
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(target_qubit))
        else:
            gates.append(("h", target_qubit))
    elif two_qubit_gate_basis == "cx":
        gates.append(("cx", control_qubit, target_qubit))
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(control_qubit, target_qubit, convert_single_qubit_gate_to_u)
    return gates


def cy_decompose(
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    gates = []
    if convert_single_qubit_gate_to_u:
        gates.append(sdg2u(target_qubit))
    else:
        gates.append(("sdg", target_qubit))

    if two_qubit_gate_basis == "cz":
        gates += cx_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "cx":
        gates.append(("cx", control_qubit, target_qubit))
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(control_qubit, target_qubit, convert_single_qubit_gate_to_u)

    if convert_single_qubit_gate_to_u:
        gates.append(s2u(target_qubit))
    else:
        gates.append(("s", target_qubit))
    return gates


def swap_decompose(
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    if two_qubit_gate_basis == "cz":
        gates = []
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(qubit2))
        else:
            gates.append(("h", qubit2))
        gates.append(("cz", qubit1, qubit2))
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(qubit2))
            gates.append(h2u(qubit1))
        else:
            gates.append(("h", qubit2))
            gates.append(("h", qubit1))
        gates.append(("cz", qubit1, qubit2))
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(qubit1))
            gates.append(h2u(qubit2))
        else:
            gates.append(("h", qubit1))
            gates.append(("h", qubit2))
        gates.append(("cz", qubit1, qubit2))
        if convert_single_qubit_gate_to_u:
            gates.append(h2u(qubit2))
        else:
            gates.append(("h", qubit2))
    elif two_qubit_gate_basis == "cx":
        gates = []
        gates.append(("cx", qubit1, qubit2))
        gates.append(("cx", qubit2, qubit1))
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "iswap":
        gates = []
        gates.append(("iswap", qubit1, qubit2))
        gates.append(("sx", qubit2))
        gates.append(("iswap", qubit1, qubit2))
        gates.append(("sx", qubit1))
        gates.append(("iswap", qubit1, qubit2))
        gates.append(("sx", qubit2))
    return gates


def iswap_decompose(
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    if two_qubit_gate_basis == "iswap":
        gates = []
        gates.append(("iswap", qubit1, qubit2))
    else:
        gates = []
        if convert_single_qubit_gate_to_u:
            gates.append(x2u(qubit1))
            gates.append(rx2u(np.pi / 2, qubit1))
            gates.append(rx2u(-np.pi / 2, qubit2))
        else:
            gates.append(("x", qubit1))
            gates.append(("rx", np.pi / 2, qubit1))
            gates.append(("rx", -np.pi / 2, qubit2))
        if two_qubit_gate_basis == "cx":
            gates.append(("cx", qubit1, qubit2))
        elif two_qubit_gate_basis == "cz":
            gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
        if convert_single_qubit_gate_to_u:
            gates.append(rx2u(-np.pi / 2, qubit1))
            gates.append(rz2u(-np.pi / 2, qubit2))
        else:
            gates.append(("rx", -np.pi / 2, qubit1))
            gates.append(("rz", -np.pi / 2, qubit2))
        if two_qubit_gate_basis == "cx":
            gates.append(("cx", qubit1, qubit2))
        elif two_qubit_gate_basis == "cz":
            gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
        if convert_single_qubit_gate_to_u:
            gates.append(rx2u(-np.pi / 2, qubit1))
            gates.append(rx2u(np.pi / 2, qubit2))
            gates.append(x2u(qubit1))
        else:
            gates.append(("rx", -np.pi / 2, qubit1))
            gates.append(("rx", np.pi / 2, qubit2))
            gates.append(("x", qubit1))
    return gates


def rxx_decompose(
    theta: float,
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    gates = []
    if convert_single_qubit_gate_to_u:
        gates.append(h2u(qubit1))
        gates.append(h2u(qubit2))
    else:
        gates.append(("h", qubit1))
        gates.append(("h", qubit2))
    if two_qubit_gate_basis == "cx":
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "cz":
        gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(qubit1, qubit2, convert_single_qubit_gate_to_u)

    if convert_single_qubit_gate_to_u:
        gates.append(rz2u(theta, qubit2))
    else:
        gates.append(("rz", theta, qubit2))
    if two_qubit_gate_basis == "cx":
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "cz":
        gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(qubit1, qubit2, convert_single_qubit_gate_to_u)

    if convert_single_qubit_gate_to_u:
        gates.append(h2u(qubit1))
        gates.append(h2u(qubit2))
    else:
        gates.append(("h", qubit1))
        gates.append(("h", qubit2))
    return gates


def ryy_decompose(
    theta: float,
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    gates = []
    if convert_single_qubit_gate_to_u:
        gates.append(rx2u(np.pi / 2, qubit1))
        gates.append(rx2u(np.pi / 2, qubit2))
    else:
        gates.append(("rx", np.pi / 2, qubit1))
        gates.append(("rx", np.pi / 2, qubit2))
    if two_qubit_gate_basis == "cx":
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "cz":
        gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(qubit1, qubit2, convert_single_qubit_gate_to_u)
    if convert_single_qubit_gate_to_u:
        gates.append(rz2u(theta, qubit2))
    else:
        gates.append(("rz", theta, qubit2))
    if two_qubit_gate_basis == "cx":
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "cz":
        gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(qubit1, qubit2, convert_single_qubit_gate_to_u)
    if convert_single_qubit_gate_to_u:
        gates.append(rx2u(-np.pi / 2, qubit1))
        gates.append(rx2u(-np.pi / 2, qubit2))
    else:
        gates.append(("rx", -np.pi / 2, qubit1))
        gates.append(("rx", -np.pi / 2, qubit2))
    return gates


def rzz_decompose(
    theta: float,
    qubit1: int,
    qubit2: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    gates = []
    if two_qubit_gate_basis == "cx":
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "cz":
        gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(qubit1, qubit2, convert_single_qubit_gate_to_u)
    if convert_single_qubit_gate_to_u:
        gates.append(rz2u(theta, qubit2))
    else:
        gates.append(("rz", theta, qubit2))
    if two_qubit_gate_basis == "cx":
        gates.append(("cx", qubit1, qubit2))
    elif two_qubit_gate_basis == "cz":
        gates += cx_decompose(qubit1, qubit2, convert_single_qubit_gate_to_u, two_qubit_gate_basis)
    elif two_qubit_gate_basis == "iswap":
        gates += convert_cx_to_iswap(qubit1, qubit2, convert_single_qubit_gate_to_u)
    return gates


def cp_decompose(
    theta: float,
    control_qubit: int,
    target_qubit: int,
    convert_single_qubit_gate_to_u: bool,
    two_qubit_gate_basis: Literal["cz", "cx", "iswap"],
) -> list:
    gates = []
    if convert_single_qubit_gate_to_u:
        gates.append(h2u(target_qubit))
    else:
        gates.append(("h", target_qubit))

    gates += cz_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis)

    if convert_single_qubit_gate_to_u:
        gates.append(rx2u(theta / 2, target_qubit))
    else:
        gates.append(("rx", theta / 2, target_qubit))

    gates += cz_decompose(control_qubit, target_qubit, convert_single_qubit_gate_to_u, two_qubit_gate_basis)

    if convert_single_qubit_gate_to_u:
        gates.append(rz2u(-1 * theta / 2, control_qubit))
        gates.append(h2u(target_qubit))
        gates.append(rz2u(-1 * theta / 2, target_qubit))
    else:
        gates.append(("rz", -1 * theta / 2, control_qubit))
        gates.append(("h", target_qubit))
        gates.append(("rz", -1 * theta / 2, target_qubit))
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
