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

from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class Instruction:
    """
    Basic data class. Used for storing data from qasm ast and output data with QCIS.
    """

    name: str
    qubit_index: List[int]
    arguments: Optional[List[Union[int, float]]] = None

    def __str__(self):
        """
                output instruction string with QCIS format.

                Returns:
        `           str: QCIS instruction string.
        """
        instr_str = self.name.upper() + " "
        for i in self.qubit_index:
            instr_str += f"Q{i} "
        if self.arguments:
            for i in self.arguments:
                instr_str += f"{i} "
        return instr_str.rstrip()
