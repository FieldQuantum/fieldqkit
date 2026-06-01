r"""Translate single- and two-qubit gates in the quantum circuit into basis gates.

SPDX-License-Identifier: Apache-2.0
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

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
    ecr_decompose,
    rxx_decompose,
    ryy_decompose,
    rzz_decompose,
    u3_decompose,
)
from .basepasses import TranspilerPass


class TranslateToBasisGates(TranspilerPass):
    """Transpiler pass for converting quantum gates to hardware-specific basis gates."""

    def __init__(
        self,
        convert_single_qubit_gate_to_u: bool = True,
        two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"] = "cz",
    ):
        """Initialize the basis gate translator.

        Args:
            convert_single_qubit_gate_to_u (*bool*): Whether to convert single-qubit gates to U gates. Defaults to ``True``.
            two_qubit_gate_basis (*Literal['cz', 'cx', 'iswap', 'ecr']*): Target native two-qubit gate for decomposition. Defaults to ``'cz'``.
        """
        super().__init__()
        self.convert_single_qubit_gate_to_u = convert_single_qubit_gate_to_u
        self.two_qubit_gate_basis = two_qubit_gate_basis

    def run(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Translate all gates in the circuit to the target basis gate set.

        Args:
            qc (*QuantumCircuit*): Quantum circuit.

        Returns:
            ``QuantumCircuit`` with all gates translated to the target basis gate set.

        Raises:
            TypeError: If a gate is not supported for basis gate translation.
        """
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
                    elif isinstance(gate_info[1], str) and gate in {"rx", "ry", "rz"}:
                        # Symbolic parameter: keep structural form, express as u
                        theta = gate_info[1]
                        qubit = gate_info[-1]
                        if gate == "rx":
                            new.append(("u", theta, -np.pi / 2, np.pi / 2, qubit))
                        elif gate == "ry":
                            new.append(("u", theta, 0.0, 0.0, qubit))
                        else:  # rz
                            new.append(("u", 0.0, 0.0, theta, qubit))
                    else:
                        gate_matrix = gate_matrix_dict[gate](*gate_info[1:-1])
                        theta, phi, lamda, _ = u3_decompose(gate_matrix)
                        new.append(("u", theta, phi, lamda, gate_info[-1]))
                else:
                    new.append(gate_info)
            elif gate in two_qubit_gates_available.keys():
                if gate in ["cz"]:
                    _cz = cz_decompose(
                        gate_info[1],
                        gate_info[2],
                        self.convert_single_qubit_gate_to_u,
                        self.two_qubit_gate_basis,
                    )
                    new += _cz
                elif gate in ["cx"]:
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
                elif gate in ["ecr"]:
                    _ecr = ecr_decompose(
                        gate_info[1],
                        gate_info[2],
                        self.convert_single_qubit_gate_to_u,
                        self.two_qubit_gate_basis,
                    )
                    new += _ecr
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
            elif gate in functional_gates_available.keys():
                new.append(gate_info)
            else:
                raise TypeError(f"Input {gate} gate is not support to basic gates now.")

        new_qc.gates = new
        return new_qc
