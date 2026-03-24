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

import math
import random
from .data import Instruction


class NativeQcisRules:
    """
    Decompose basic QASM gate into native gates from QCIS.
    QCIS native gates include X2P, Y2P, RZ, X2M, Y2M, CZ
    """

    pi = round(math.pi, 6)
    i_duration = 60

    @staticmethod
    def x(input_instruction: Instruction):
        return [
            Instruction("x2p", input_instruction.qubit_index),
            Instruction("x2p", input_instruction.qubit_index),
        ]

    @staticmethod
    def y(input_instruction: Instruction):
        return [
            Instruction("y2p", input_instruction.qubit_index),
            Instruction("y2p", input_instruction.qubit_index),
        ]

    @staticmethod
    def z(input_instruction: Instruction):
        return [Instruction("rz", input_instruction.qubit_index, [NativeQcisRules.pi])]

    @staticmethod
    def h(input_instruction: Instruction):
        return [
            Instruction("y2m", input_instruction.qubit_index),
            Instruction("rz", input_instruction.qubit_index, [NativeQcisRules.pi]),
        ]

    @staticmethod
    def sx(input_instruction: Instruction):
        return [Instruction("x2p", input_instruction.qubit_index)]

    @staticmethod
    def sxdg(input_instruction: Instruction):
        return [Instruction("x2m", input_instruction.qubit_index)]

    @staticmethod
    def s(input_instruction: Instruction):
        return [
            Instruction("rz", input_instruction.qubit_index, [NativeQcisRules.pi / 2])
        ]

    @staticmethod
    def sdg(input_instruction: Instruction):
        return [
            Instruction("rz", input_instruction.qubit_index, [-NativeQcisRules.pi / 2])
        ]

    @staticmethod
    def t(input_instruction: Instruction):
        return [
            Instruction("rz", input_instruction.qubit_index, [NativeQcisRules.pi / 4])
        ]

    @staticmethod
    def tdg(input_instruction: Instruction):
        return [
            Instruction("rz", input_instruction.qubit_index, [-NativeQcisRules.pi / 4])
        ]

    @staticmethod
    def rx(input_instruction: Instruction):
        return [
            Instruction("y2m", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, input_instruction.arguments
            ),
            Instruction("y2p", input_instruction.qubit_index),
        ]

    @staticmethod
    def ry(input_instruction: Instruction):
        return [
            Instruction("x2p", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, input_instruction.arguments
            ),
            Instruction("x2m", input_instruction.qubit_index),
        ]

    @staticmethod
    def rz(input_instruction: Instruction):
        return [input_instruction]

    @staticmethod
    def u(input_instruction: Instruction):
        return [
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[1]]
            ),
            Instruction("x2p", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[0]]
            ),
            Instruction("x2m", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[2]]
            ),
        ]

    @staticmethod
    def u1(input_instruction: Instruction):
        return [
            Instruction(
                "rz", input_instruction.qubit_index, input_instruction.arguments
            )
        ]

    @staticmethod
    def u2(input_instruction: Instruction):
        return [
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[0]]
            ),
            Instruction("y2p", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[1]]
            ),
        ]

    @staticmethod
    def u3(input_instruction: Instruction):
        return [
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[2]]
            ),
            Instruction("x2p", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[0]]
            ),
            Instruction("x2m", input_instruction.qubit_index),
            Instruction(
                "rz", input_instruction.qubit_index, [input_instruction.arguments[1]]
            ),
        ]

    @staticmethod
    def id(input_instruction: Instruction):
        return [Instruction("i", input_instruction.qubit_index, [NativeQcisRules.i_duration])]

    @staticmethod
    def delay(input_instruction: Instruction):
        if input_instruction.arguments:
            duration = input_instruction.arguments[0]
        else:
            duration = NativeQcisRules.i_duration
        return [Instruction("i", input_instruction.qubit_index, [duration])]

    @staticmethod
    def cx(input_instruction: Instruction):
        return [
            Instruction("y2m", [input_instruction.qubit_index[1]]),
            Instruction("cz", input_instruction.qubit_index),
            Instruction("y2p", [input_instruction.qubit_index[1]]),
        ]

    @staticmethod
    def cz(input_instruction: Instruction):
        return [Instruction("cz", input_instruction.qubit_index)]

    @staticmethod
    def cy(input_instruction: Instruction):
        return [
            Instruction("x2p", [input_instruction.qubit_index[1]]),
            Instruction("cz", input_instruction.qubit_index),
            Instruction("x2m", [input_instruction.qubit_index[1]]),
        ]

    @staticmethod
    def ch(input_instruction: Instruction):
        res = []
        res.extend(
            NativeQcisRules.s(Instruction("s", [input_instruction.qubit_index[1]]))
        )
        res.extend(
            NativeQcisRules.h(Instruction("h", [input_instruction.qubit_index[1]]))
        )
        res.extend(
            NativeQcisRules.t(Instruction("t", [input_instruction.qubit_index[1]]))
        )
        res.extend(NativeQcisRules.cx(Instruction("cx", input_instruction.qubit_index)))
        res.extend(
            NativeQcisRules.tdg(Instruction("tdg", [input_instruction.qubit_index[1]]))
        )
        res.extend(
            NativeQcisRules.h(Instruction("h", [input_instruction.qubit_index[1]]))
        )
        res.extend(
            NativeQcisRules.sdg(Instruction("sdg", [input_instruction.qubit_index[1]]))
        )
        return res

    @staticmethod
    def swap(input_instruction: Instruction):
        res = []
        rever_qubit_index = list(input_instruction.qubit_index)
        rever_qubit_index.reverse()
        res.extend(NativeQcisRules.cx(Instruction("cx", input_instruction.qubit_index)))
        res.extend(NativeQcisRules.cx(Instruction("cx", rever_qubit_index)))
        res.extend(NativeQcisRules.cx(Instruction("cx", input_instruction.qubit_index)))
        return res

    @staticmethod
    def crz(input_instruction: Instruction):
        res = [
            Instruction(
                "rz",
                [input_instruction.qubit_index[1]],
                [i / 2 for i in input_instruction.arguments],
            )
        ]
        res.extend(NativeQcisRules.cx(Instruction("cx", input_instruction.qubit_index)))
        res.append(
            Instruction(
                "rz",
                [input_instruction.qubit_index[1]],
                [-i / 2 for i in input_instruction.arguments],
            )
        )
        res.extend(NativeQcisRules.cx(Instruction("cx", input_instruction.qubit_index)))
        return res

    @staticmethod
    def cp(input_instruction: Instruction):
        res = [
            Instruction(
                "rz",
                [input_instruction.qubit_index[0]],
                [i / 2 for i in input_instruction.arguments],
            )
        ]
        res.extend(
            NativeQcisRules.crz(
                Instruction(
                    "crz", input_instruction.qubit_index, input_instruction.arguments
                )
            )
        )
        return res

    @staticmethod
    def ccx(input_instruction: Instruction):
        res = []
        res.extend(
            NativeQcisRules.h(Instruction("h", [input_instruction.qubit_index[2]]))
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[1],
                        input_instruction.qubit_index[2],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.tdg(Instruction("tdg", [input_instruction.qubit_index[2]]))
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[0],
                        input_instruction.qubit_index[2],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.t(Instruction("t", [input_instruction.qubit_index[2]]))
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[1],
                        input_instruction.qubit_index[2],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.tdg(Instruction("tdg", [input_instruction.qubit_index[2]]))
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[0],
                        input_instruction.qubit_index[2],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.t(Instruction("t", [input_instruction.qubit_index[1]]))
        )
        res.extend(
            NativeQcisRules.t(Instruction("t", [input_instruction.qubit_index[2]]))
        )
        res.extend(
            NativeQcisRules.h(Instruction("h", [input_instruction.qubit_index[2]]))
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[0],
                        input_instruction.qubit_index[1],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.t(Instruction("t", [input_instruction.qubit_index[0]]))
        )
        res.extend(
            NativeQcisRules.tdg(Instruction("tdg", [input_instruction.qubit_index[1]]))
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[0],
                        input_instruction.qubit_index[1],
                    ],
                )
            )
        )
        return res

    @staticmethod
    def cu3(input_instruction: Instruction):
        res = [
            Instruction(
                "rz",
                [input_instruction.qubit_index[1]],
                [(input_instruction.arguments[1] - input_instruction.arguments[2]) / 2],
            )
        ]
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[0],
                        input_instruction.qubit_index[1],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.u(
                Instruction(
                    "u",
                    [input_instruction.qubit_index[1]],
                    [
                        -input_instruction.arguments[0] / 2,
                        0,
                        -(
                                input_instruction.arguments[1]
                                + input_instruction.arguments[2]
                        )
                        / 2,
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.cx(
                Instruction(
                    "cx",
                    [
                        input_instruction.qubit_index[0],
                        input_instruction.qubit_index[1],
                    ],
                )
            )
        )
        res.extend(
            NativeQcisRules.u(
                Instruction(
                    "u",
                    [input_instruction.qubit_index[1]],
                    [
                        input_instruction.arguments[0] / 2,
                        input_instruction.arguments[2],
                        0,
                    ],
                )
            )
        )
        return res
