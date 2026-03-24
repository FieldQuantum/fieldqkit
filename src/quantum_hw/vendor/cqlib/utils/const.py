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

qasm2qcis = {
    "single_gate": {
        "x": ["X2P", "X2P"],
        "y": ["Y2P", "Y2P"],
        "z": "RZ [math.pi]",
        "h": {"0": ["Y2M", "RZ [math.pi]"], "1": ["RZ [math.pi]", "Y2P"]},
        "sx": "X2P",
        "sxdg": "X2M",
        "h_sx_h": "Y2P",
        "h_sxdg_h": "Y2M",
        "s": "RZ [math.pi/2]",
        "sdg": "RZ [-math.pi/2]",
        "t": "RZ [math.pi/4]",
        "tdg": "RZ [-math.pi/4]",
        "rx": ["Y2M", "RZ [0]", "Y2P"],
        "ry": ["X2P", "RZ [0]", "X2M"],
        "rz": "RZ [0]",
        "u": ["RZ [1]", "X2P", "RZ [0]", "X2M", "RZ [2]"],
        "u2": ["RZ [0]", "Y2P", "RZ [1]"],
        "u1": "RZ [0]",
        "barrier": "B",
        "id": "I",
    },
    "couper_gate": {
        "cx": ["Y2M 1", "CZ 0 1", "Y2P 1"],
        "cz": ["CZ 0 1"],
        "cy": ["X2P 1", "CZ 0 1", "X2M 1"],
        "ch": [
            {"single_gate": "s 1"},
            {"single_gate": "h 1"},
            {"single_gate": "t 1"},
            {"couper_gate": "cx 0 1"},
            {"single_gate": "tdg 1"},
            {"single_gate": "h 1"},
            {"single_gate": "sdg 1"},
        ],
        "swap": [
            {"couper_gate": "cx 0 1"},
            {"couper_gate": "cx 1 0"},
            {"couper_gate": "cx 0 1"},
        ],
        "crz": [
            "RZ 1 [0]/2",
            {"couper_gate": "cx 0 1"},
            "RZ 1 -[0]/2",
            {"couper_gate": "cx 0 1"},
        ],
        "cp": ["RZ 0 [0]/2", {"couper_gate": "crz([0]) 0 1"}],
        "ccx": [
            {"single_gate": "h 2"},
            {"couper_gate": "cx 1 2"},
            {"single_gate": "tdg 2"},
            {"couper_gate": "cx 0 2"},
            {"single_gate": "t 2"},
            {"couper_gate": "cx 1 2"},
            {"single_gate": "tdg 2"},
            {"couper_gate": "cx 0 2"},
            {"single_gate": "t 1"},
            {"single_gate": "t 2"},
            {"single_gate": "h 2"},
            {"couper_gate": "cx 0 1"},
            {"single_gate": "t 0"},
            {"single_gate": "tdg 1"},
            {"couper_gate": "cx 0 1"},
        ],
        "cu3": [
            "RZ 1 ([1]-[2])/2",
            {"couper_gate": "cx 0 1"},
            {"single_gate": "u(-[0]/2,0,-([2]+[1])/2) 1"},
            {"couper_gate": "cx 0 1"},
            {"single_gate": "u([0]/2,[2],0) 1"},
        ],
    },
    "circuit_simplify": {
        "repeat": {"RZ": ["n", "RZ"], "X": "I", "Y": "I", "Z": "I", "S": "Z", "T": "S"}
    },
}

RZ_PERIOD = 2 * math.pi
RZ_PARAMS = {"-pi/2": -math.pi / 2, "pi/2": math.pi / 2, "-pi": math.pi, "pi": math.pi}
SIMPLIFY_RULE = {
    "rz_param_as_pi": {
        r"RZ Q(\d+) -pi(\n|$)": "RZ Q[1] pi",
        r"RZ Q(\d+) pi(\n|$)": "RZ Q[1] pi/2\nRZ Q[1] pi/2",
    },
    "gate_conversion": {
        r"Y2M Q(\d+)\nY2P Q\1": "I Q[1] 0",
        r"Y2P Q(\d+)\nY2M Q\1": "I Q[1] 0",
        r"X2M Q(\d+)\nX2P Q\1": "I Q[1] 0",
        r"X2P Q(\d+)\nX2M Q\1": "I Q[1] 0",
        r"Y2M Q(\d+)\nY2M Q\1\nY2M Q\1": "Y2P Q[1]",
        r"Y2P Q(\d+)\nY2P Q\1\nY2P Q\1": "Y2M Q[1]",
        r"X2M Q(\d+)\nX2M Q\1\nX2M Q\1": "X2P Q[1]",
        r"X2P Q(\d+)\nX2P Q\1\nX2P Q\1": "X2M Q[1]",
    },
    "gate_param_conversion": {
        r"Y2P Q(\d+)\nX2M Q\1": "X2M Q[1]\nRZ Q[1] pi/2",
        r"Y2M Q(\d+)\nX2M Q\1": "X2M Q[1]\nRZ Q[1] -pi/2",
        r"Y2M Q(\d+)\nX2P Q\1": "X2P Q[1]\nRZ Q[1] pi/2",
        r"Y2P Q(\d+)\nX2P Q\1": "X2P Q[1]\nRZ Q[1] -pi/2",
        r"X2P Q(\d+)\nY2P Q\1": "Y2P Q[1]\nRZ Q[1] pi/2",
        r"X2P Q(\d+)\nY2M Q\1": "Y2M Q[1]\nRZ Q[1] -pi/2",
        r"X2M Q(\d+)\nY2M Q\1": "Y2M Q[1]\nRZ Q[1] pi/2",
        r"X2M Q(\d+)\nY2P Q\1": "Y2P Q[1]\nRZ Q[1] -pi/2",
        r"RZ Q(\d+) pi/2\nX2P Q\1": "Y2P Q[1]\nRZ Q[1] pi/2",
        r"RZ Q(\d+) pi/2\nX2M Q\1": "Y2M Q[1]\nRZ Q[1] pi/2",
        r"RZ Q(\d+) -pi/2\nX2P Q\1": "Y2M Q[1]\nRZ Q[1] -pi/2",
        r"RZ Q(\d+) -pi/2\nX2M Q\1": "Y2P Q[1]\nRZ Q[1] -pi/2",
        r"RZ Q(\d+) pi/2\nY2P Q\1": "X2M Q[1]\nRZ Q[1] pi/2",
        r"RZ Q(\d+) pi/2\nY2M Q\1": "X2P Q[1]\nRZ Q[1] pi/2",
        r"RZ Q(\d+) -pi/2\nY2P Q\1": "X2P Q[1]\nRZ Q[1] -pi/2",
        r"RZ Q(\d+) -pi/2\nY2M Q\1": "X2M Q[1]\nRZ Q[1] -pi/2",
    },
    "rz_param_conversion": {
        r"Y2P Q(\d+)\nY2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2P Q\1": "RZ Q[1] -[2]\nY2M Q[1]",
        r"Y2P Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nY2P Q\1\nY2P Q\1": "Y2M Q[1]\nRZ Q[1] -[2]",
        r"Y2P Q(\d+)\nY2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2M Q\1": "RZ Q[1] -[2]\nY2P Q[1]",
        r"Y2P Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nY2M Q\1\nY2M Q\1": "Y2M Q[1]\nRZ Q[1] -[2]",
        r"Y2M Q(\d+)\nY2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2P Q\1": "RZ Q[1] -[2]\nY2M Q[1]",
        r"Y2M Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nY2P Q\1\nY2P Q\1": "Y2P Q[1]\nRZ Q[1] -[2]",
        r"Y2M Q(\d+)\nY2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2M Q\1": "RZ Q[1] -[2]\nY2P Q[1]",
        r"Y2M Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nY2M Q\1\nY2M Q\1": "Y2P Q[1]\nRZ Q[1] -[2]",
        r"X2P Q(\d+)\nX2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2P Q\1": "RZ Q[1] -[2]\nX2M Q[1]",
        r"X2P Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nX2P Q\1\nX2P Q\1": "X2M Q[1]\nRZ Q[1] -[2]",
        r"X2P Q(\d+)\nX2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2M Q\1": "RZ Q[1] -[2]\nX2P Q[1]",
        r"X2P Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nX2M Q\1\nX2M Q\1": "X2M Q[1]\nRZ Q[1] -[2]",
        r"X2M Q(\d+)\nX2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2P Q\1": "RZ Q[1] -[2]\nX2M Q[1]",
        r"X2M Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nX2P Q\1\nX2P Q\1": "X2P Q[1]\nRZ Q[1] -[2]",
        r"X2M Q(\d+)\nX2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2M Q\1": "RZ Q[1] -[2]\nX2P Q[1]",
        r"X2M Q(\d+)\nRZ Q\1 (\d+(\.\d*)?)\nX2M Q\1\nX2M Q\1": "X2P Q[1]\nRZ [1] -[2]",
        r"Y2M Q(\d+)\nX2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2P Q\1\nX2P Q\1": "X2M Q[1]\nRZ Q[1] [2]\nX2P Q[1]",
        r"Y2P Q(\d+)\nX2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2P Q\1\nY2M Q\1": "X2M Q[1]\nRZ Q[1] [2]\nX2P Q[1]",
        r"Y2M Q(\d+)\nX2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2M Q\1\nY2P Q\1": "X2P Q[1]\nRZ Q[1] [2]\nX2M Q[1]",
        r"Y2P Q(\d+)\nX2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nX2M Q\1\nY2M Q\1": "X2P Q[1]\nRZ Q[1] [2]\nX2M Q[1]",
        r"X2M Q(\d+)\nY2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2M Q\1\nX2P Q\1": "Y2P Q[1]\nRZ Q[1] [2]\nY2M Q[1]",
        r"X2P Q(\d+)\nY2P Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2M Q\1\nX2M Q\1": "Y2P Q[1]\nRZ Q[1] [2]\nY2M Q[1]",
        r"X2M Q(\d+)\nY2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2P Q\1\nY2P Q\1": "Y2M Q[1]\nRZ Q[1] [2]\nY2P Q[1]",
        r"X2P Q(\d+)\nY2M Q\1\nRZ Q\1 (\d+(\.\d*)?)\nY2P Q\1\nX2M Q\1": "Y2M Q[1]\nRZ Q[1] [2]\nY2P Q[1]",
    },
}


GATE_PULSE_QCIS_TEMPLATE_QUBIT = "PXY {qubit} {wave_class} {length} {amplitude} {phase} {drag_alpha} {extra_param}"
GATE_PULSE_QCIS_TEMPLATE_COUPLER = """
B {qubit0} {qubit1} {coupler}
I {qubit0} {pad_time_left}
PZ {qubit0} {detune_wave_class} {detune_length} {qubit0_detune_amplitude} {detune_extra_param}
I {qubit0} {pad_time_right}
RZ {qubit0} {qubit0_dynamic_phase}
I {qubit1} {pad_time_left}
PZ {qubit1} {detune_wave_class} {detune_length} {qubit1_detune_amplitude} {detune_extra_param}
I {qubit1} {pad_time_right}
RZ {qubit1} {qubit1_dynamic_phase}
I {coupler} {pad_time_left}
PZ0 {coupler} {detune_wave_class} {detune_length} {coupler_detune_amplitude} {detune_extra_param}
I {coupler} {coupler_detune_longer_left}
PZ {coupler} {cz_wave_class} {cz_length} {cz_amplitude} {cz_extra_param}
I {coupler} {coupler_detune_longer_right}
I {coupler} {pad_time_right}
B {qubit0} {qubit1} {coupler}
"""


GATE_PHASES = {
    "X2P": 0,
    "X2M": math.pi,
    "Y2P": math.pi / 2,
    "Y2M": -math.pi / 2
}

