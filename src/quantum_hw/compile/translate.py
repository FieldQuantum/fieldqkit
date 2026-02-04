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

r"""Translate single- and two-qubit gates in the quantum circuit into basis gates."""

from typing import Literal
import numpy as np
from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
    functional_gates_available,
)
from ..circuit.matrix import gate_matrix_dict
from .decompose import (
    cx_decompose,
    cz_decompose,
    cy_decompose,
    swap_decompose,
    iswap_decompose,
    rxx_decompose,
    ryy_decompose,
    rzz_decompose,
    cp_decompose,
    u3_decompose,
)
from .basepasses import TranspilerPass


class TranslateToBasisGates(TranspilerPass):
    """Transpiler pass for converting quantum gates to hardware-specific basis gates."""

    def __init__(
        self,
        convert_single_qubit_gate_to_u: bool = True,
        two_qubit_gate_basis: Literal["cz", "cx", "iswap"] = "cz",
    ):
        super().__init__()
        self.convert_single_qubit_gate_to_u = convert_single_qubit_gate_to_u
        self.two_qubit_gate_basis = two_qubit_gate_basis

    def run(self, qc: QuantumCircuit) -> QuantumCircuit:
        new_qc = qc.deepcopy()

        new = []
        for gate_info in qc.gates:
            gate = gate_info[0]
            if gate in one_qubit_gates_available.keys():
                if self.convert_single_qubit_gate_to_u:
                    gate_matrix = gate_matrix_dict[gate]
                    theta, phi, lamda, _ = u3_decompose(gate_matrix)
                    new.append(("u", theta, phi, lamda, gate_info[-1]))
                else:
                    new.append(gate_info)
            elif gate in one_qubit_parameter_gates_available.keys():
                if self.convert_single_qubit_gate_to_u:
                    if gate == "u":
                        new.append(gate_info)
                    elif gate == "r":
                        theta, phi, qubit = gate_info[1:]
                        new.append(("u", theta, phi - np.pi / 2, np.pi / 2 - phi, qubit))
                    else:
                        gate_matrix = gate_matrix_dict[gate](*gate_info[1:-1])
                        theta, phi, lamda, _ = u3_decompose(gate_matrix)
                        new.append(("u", theta, phi, lamda, gate_info[-1]))
                else:
                    new.append(gate_info)
            elif gate in two_qubit_gates_available.keys():
                if gate in ["cz"]:
                    if self.two_qubit_gate_basis in ["cx", "cz"]:
                        new += [gate_info]
                    else:
                        _cz = cz_decompose(
                            gate_info[1],
                            gate_info[2],
                            self.convert_single_qubit_gate_to_u,
                            self.two_qubit_gate_basis,
                        )
                        new += _cz
                elif gate in ["cx", "cnot"]:
                    _cx = cx_decompose(
                        gate_info[1],
                        gate_info[2],
                        self.convert_single_qubit_gate_to_u,
                        self.two_qubit_gate_basis,
                    )
                    new += _cx
                elif gate in ["swap"]:
                    _swap = swap_decompose(
                        gate_info[1],
                        gate_info[2],
                        self.convert_single_qubit_gate_to_u,
                        self.two_qubit_gate_basis,
                    )
                    new += _swap
                elif gate in ["iswap"]:
                    _iswap = iswap_decompose(
                        gate_info[1],
                        gate_info[2],
                        self.convert_single_qubit_gate_to_u,
                        self.two_qubit_gate_basis,
                    )
                    new += _iswap
                elif gate in ["cy"]:
                    _cy = cy_decompose(
                        gate_info[1],
                        gate_info[2],
                        self.convert_single_qubit_gate_to_u,
                        self.two_qubit_gate_basis,
                    )
                    new += _cy
                else:
                    raise TypeError(f"Input {gate} gate is not support now. Try kak please")
            elif gate in two_qubit_parameter_gates_available.keys():
                if gate == "rxx":
                    new += rxx_decompose(*gate_info[1:], self.convert_single_qubit_gate_to_u, self.two_qubit_gate_basis)
                elif gate == "ryy":
                    new += ryy_decompose(*gate_info[1:], self.convert_single_qubit_gate_to_u, self.two_qubit_gate_basis)
                elif gate == "rzz":
                    new += rzz_decompose(*gate_info[1:], self.convert_single_qubit_gate_to_u, self.two_qubit_gate_basis)
                elif gate == "cp":
                    new += cp_decompose(*gate_info[1:], self.convert_single_qubit_gate_to_u, self.two_qubit_gate_basis)
            elif gate in functional_gates_available.keys():
                new.append(gate_info)
            else:
                raise TypeError(f"Input {gate} gate is not support to basic gates now.")

        new_qc.gates = new
        return new_qc
