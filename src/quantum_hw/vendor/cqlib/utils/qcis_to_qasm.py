# This code is part of cqlib.
#
# Copyright (C) 2024 China Telecom Quantum Group, QuantumCTek Co., Ltd.,
# Center for Excellence in Quantum Information and Quantum Physics.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import re
from numpy import pi
from typing import Callable

# Define half pi
HALF_PI = pi / 2


class QcisToQasm:
    """Class for converting quantum circuit instructions from qcis format to qasm format."""

    def _rxy_qcis_to_qasm(*args) -> str:
        """Converts a rxy gate instruction from qcis format to qasm format.

        Args:
            *args: Variable length argument list containing:
                q_num (int): The target qubit index.
                phase1 (float): The first phase angle.
                phase2 (float): The second phase angle.

        Returns:
            str: A string representing the converted qasm instruction for the rxy gate
        """
        # Unpack arguments to get the qubit number and phase angles
        try:
            q_num, phase1, phase2 = args
        except ValueError:
            raise QcisToQasm.InputParaNumError(
                f"For gate rxy, requires 3 input parameters but {len(args)} has given."
            )
        return f"u({phase2},{float(phase1) - HALF_PI},{HALF_PI - float(phase1)}) q[{q_num}];"

    def _xy_qcis_to_qasm(*args) -> str:
        """Converts a xy gate instruction from qcis format to qasm format.

        Args:
            *args: Variable length argument list containing:
                q_num (int): The target qubit index.
                phase (float): The phase angle.

        Returns:
            str: A string representing the converted qasm instruction for the xy gate
        """
        # Unpack arguments to get the qubit number and phase angles
        try:
            q_num, phase = args
        except ValueError:
            raise QcisToQasm.InputParaNumError(
                f"For gate xy, requires 2 input parameters but {len(args)} has given."
            )
        qasm_xy = f"rz({HALF_PI - float(phase)}) q[{q_num}];\n"
        qasm_xy += f"y q[{q_num}];\n"
        qasm_xy += f"rz({float(phase) - HALF_PI}) q[{q_num}];"
        return qasm_xy

    def _xy2p_qcis_to_qasm(*args) -> str:
        """Converts a xy2p gate instruction from qcis format to qasm format.

        Args:
            *args: Variable length argument list containing:
                q_num (int): The target qubit index.
                phase (float): The phase angle.

        Returns:
            str: A string representing the converted qasm instruction for the xy2p gate
        """
        # Unpack arguments to get the qubit number and phase angles
        try:
            q_num, phase = args
        except ValueError:
            raise QcisToQasm.InputParaNumError(
                f"For gate xy2p, requires 2 input parameters but {len(args)} has given."
            )
        qasm_xy2p = f"rz({HALF_PI - float(phase)}) q[{q_num}];\n"
        qasm_xy2p += f"ry({HALF_PI}) q[{q_num}];\n"
        qasm_xy2p += f"rz({float(phase) - HALF_PI}) q[{q_num}];"
        return qasm_xy2p

    def _xy2m_qcis_to_qasm(*args) -> str:
        """Converts a xy2m gate instruction from qcis format to qasm format.

        Args:
            *args: Variable length argument list containing:
                q_num (int): The target qubit index.
                phase (float): The phase angle.

        Returns:
            str: A string representing the converted qasm instruction for the xy2m gate
        """
        # Unpack arguments to get the qubit number and phase angles
        try:
            q_num, phase = args
        except ValueError:
            raise QcisToQasm.InputParaNumError(
                f"For gate xy2m, requires 2 input parameters but {len(args)} has given."
            )
        qasm_xy2m = f"rz({-(HALF_PI + float(phase))}) q[{q_num}];\n"
        qasm_xy2m += f"ry({HALF_PI}) q[{q_num}];\n"
        qasm_xy2m += f"rz({float(phase) + HALF_PI}) q[{q_num}];"
        return qasm_xy2m

    convert_rule = {
        "X": "x q[{0}];",
        "Y": "y q[{0}];",
        "Z": "z q[{0}];",
        "H": "h q[{0}];",
        "S": "s q[{0}];",
        "SD": "sdg q[{0}];",
        "T": "t q[{0}];",
        "TD": "tdg q[{0}];",
        "X2P": "sx q[{0}];",
        "X2M": "sxdg q[{0}];",
        "Y2P": f"ry({HALF_PI})" + " q[{0}];",
        "Y2M": f"ry({-HALF_PI})" + " q[{0}];",
        "CZ": "cz q[{0}],q[{1}];",
        "RX": "rx({1}) q[{0}];",
        "RY": "ry({1}) q[{0}];",
        "RZ": "rz({1}) q[{0}];",
        "RXY": _rxy_qcis_to_qasm,
        "I": "",
        "B": "barrier q[{0}],q[{1}];",
        "XY": _xy_qcis_to_qasm,
        "XY2P": _xy2p_qcis_to_qasm,
        "XY2M": _xy2m_qcis_to_qasm,
    }

    class GateNotImplemented(Exception):
        """Exception raised for gates that are not implemented in the conversion process."""

    class InputParaNumError(Exception):
        """Exception raised when the number of input parameters for a gate does not match the requirement."""

    @staticmethod
    def convert_qcis_to_qasm(qcis_input: str) -> str:
        """
        Converts a string of qcis formatted quantum circuit instructions to qasm format.

        Args:
            qcis_input: A string containing qcis formatted quantum circuit instructions.

        Returns:
            A string of qasm formatted quantum circuit instructions.
        """
        qcis_input = qcis_input.upper()
        qcis_instruction_list = qcis_input.split("\n")
        qcis_instruction_list = [
            inst.strip() for inst in qcis_instruction_list if qcis_input.strip()
        ]
        qubit_idx = re.findall(r"Q(\d+)", qcis_input)
        qreg_qubit = max([int(idx) for idx in qubit_idx])
        qasm_output = f"""OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{qreg_qubit + 1}];\ncreg c[{qreg_qubit + 1}];\n"""
        measure_qcis = ""
        for qcis_line_num, qcis_item in enumerate(qcis_instruction_list):
            # If current line is empty, skip
            if not qcis_item:
                continue
            # If current line starts with M, add it to measure_qcis and handle it afterward
            if qcis_item.startswith("M"):
                measure_qcis += f"{qcis_item}\n"
                continue
            # unpack qcis order as gate and parameters
            # lstrip Q to only keep quantum index
            gate, *para = [temp.lstrip("Q") for temp in qcis_item.split(" ")]
            if gate not in QcisToQasm.convert_rule:
                raise QcisToQasm.GateNotImplemented(f"{gate} is not supported")
            # if converting rule of the gate is a function, call it with parameters
            if isinstance(QcisToQasm.convert_rule[gate], Callable):
                qasm_line = QcisToQasm.convert_rule[gate](*para)
            else:
                # else if converting rule of the gate is a string, format it with parameters
                required_params = QcisToQasm.convert_rule[gate].count("{")
                if len(para) < required_params:
                    raise QcisToQasm.InputParaNumError(
                        f"Gate {gate} requires {required_params} parameters but {len(para)} were given with qcis order: {qcis_item}."
                    )
                qasm_line = QcisToQasm.convert_rule[gate].format(*para)

            # if qasm_line is not empty, concatenate it to qasm_output
            if qasm_line:
                qasm_output += qasm_line
                qasm_output += "\n"
        # add measure instruction at the end of qasm_output
        measure_qubit_list = re.findall(r"Q(\d+)", measure_qcis)
        for idx, measure_qubit_idx in enumerate(measure_qubit_list):
            qasm_output += f"measure q[{int(measure_qubit_idx)}] -> c[{idx}];\n"
        return qasm_output
