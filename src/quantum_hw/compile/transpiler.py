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

r"""This module contains the Transpiler class, which is designed to convert quantum circuits into formats that are more suitable for execution on hardware backends"""

from quark.circuit.quantumcircuit import QuantumCircuit
from quark.circuit.backend import Backend
from .decompose import ThreeQubitGateDecompose
from .layout import Layout
from .routing import SabreRouting
from .translate import TranslateToBasisGates
from .optimize import GateCompressor
from .schedule import DynamicalDecoupling
from .dag import split_qubits


class Transpiler:
    r"""The transpilation process involves converting the operations
    in the circuit to those supported by the device and swapping
    qubits (via swap gates) within the circuit to overcome limited
    qubit connectivity.
    """

    def __init__(self, chip_backend: Backend | None = None):
        self.chip_backend = chip_backend

    def run(self, qc: QuantumCircuit | str | list, target_qubits: list = [], optimize_level=1, niter=5, use_dd=False):
        if isinstance(qc, QuantumCircuit):
            pass
        elif isinstance(qc, str):
            qc = QuantumCircuit().from_openqasm2(qc)
        elif isinstance(qc, list):
            qc = QuantumCircuit().from_qlisp(qc)
        else:
            raise TypeError("Expected a Quark QuantumCircuit or OpenQASM 2.0 or qlisp, but got a {}.".format(type(qc)))

        if self.chip_backend is None:
            print("Warning: No chip specified, defaulting to a linearly connected layout for simulation.")
            import networkx as nx

            subgraph = nx.Graph()
            qubits = list(sorted(qc.qubits))
            subgraph.add_edges_from([(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)])
            subgraph.graph["normal_order"] = qc.qubits
            self.two_qubit_gate_basis = "cx"
            self.convert_single_qubit_gate_to_u = False
        else:
            self.two_qubit_gate_basis = self.chip_backend.two_qubit_gate_basis
            self.convert_single_qubit_gate_to_u = True
            subgraph = Layout(self.chip_backend).select_layout(
                qc,
                target_qubits=target_qubits,
                use_chip_priority=True,
                select_criteria={
                    "key": "fidelity_var",
                    "topology": "linear",
                },
            )
        if optimize_level == 0:
            passes = [
                ThreeQubitGateDecompose(),
                SabreRouting(subgraph, initial_mapping="trivial", do_random_choice=False, iterations=1),
                TranslateToBasisGates(
                    convert_single_qubit_gate_to_u=self.convert_single_qubit_gate_to_u,
                    two_qubit_gate_basis=self.two_qubit_gate_basis,
                ),
            ]
        elif optimize_level == 1:
            passes = [
                ThreeQubitGateDecompose(),
                SabreRouting(subgraph, initial_mapping="trivial", do_random_choice=False, iterations=niter),
                TranslateToBasisGates(
                    convert_single_qubit_gate_to_u=self.convert_single_qubit_gate_to_u,
                    two_qubit_gate_basis=self.two_qubit_gate_basis,
                ),
                GateCompressor(),
            ]
        elif optimize_level == 2:
            if len(split_qubits(qc)) > 1:
                raise ValueError(
                    "If quantum circuit can be divided along the qubits, the optimize_level is restricted to 0 or 1"
                )
            passes = [
                ThreeQubitGateDecompose(),
                SabreRouting(subgraph, initial_mapping="random", do_random_choice=True, iterations=niter),
                TranslateToBasisGates(
                    convert_single_qubit_gate_to_u=self.convert_single_qubit_gate_to_u,
                    two_qubit_gate_basis=self.two_qubit_gate_basis,
                ),
                GateCompressor(),
            ]
        else:
            raise ValueError("Currently, only optimize_level values of 0 or 1 or 2 are supported.")
        if use_dd:
            try:
                t1g = self.chip_backend.chip_info["global_info"]["one_qubit_gate_length"]
                t2g = self.chip_backend.chip_info["global_info"]["two_qubit_gate_length"]
                passes.append(DynamicalDecoupling(t1g, t2g))
            except Exception:
                pass
        for pass_obj in passes:
            qc = pass_obj.run(qc)
        return qc
