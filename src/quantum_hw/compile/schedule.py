"""A toolkit for applying dynamical decoupling (DD) sequences to quantum circuits.

SPDX-License-Identifier: MIT
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

import copy
import networkx as nx
from typing import Literal
from .basepasses import TranspilerPass
from .dag import qc2dag, dag2qc
from ..circuit.quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
)


class DynamicalDecoupling(TranspilerPass):
    """Transpiler pass that inserts DD sequences into idle windows of all two-qubit gate slots."""

    def __init__(self, t1g, t2g):
        """Initialize the dynamical decoupling pass with single- and two-qubit gate durations.

        Args:
            t1g: Single-qubit gate duration (seconds).
            t2g: Two-qubit gate duration (seconds).
        """
        self.t1g = t1g
        self.t2g = t2g
        self._count = 86751

    def counter(self):
        """Increment and return the internal unique-ID counter.

        Returns:
            int: Next unique ID.
        """
        self._count += 1
        return self._count

    def _get_max_idle_time(self, nodes):
        """Determine the maximum idle-time unit for a DAG generation based on the heaviest gate type present.

        Args:
            nodes: Nodes.

        Returns:
            Result.
        """
        gates = [node.split("_")[0] for node in nodes]
        one_qubit_gates = list(one_qubit_gates_available.keys()) + list(one_qubit_parameter_gates_available.keys())
        two_qubit_gates = list(two_qubit_gates_available.keys()) + list(two_qubit_parameter_gates_available.keys())
        if bool(set(two_qubit_gates) & set(gates)):
            max_idle_time = self.t2g
        elif bool(set(one_qubit_gates) & set(gates)):
            max_idle_time = self.t1g
        else:
            max_idle_time = 0
        return max_idle_time

    def _update_idle_time(self, node, max_idle_time):
        """Subtract the gate's duration from the remaining idle time for a qubit.

        Args:
            node: Node.
            max_idle_time: Max idle time.

        Returns:
            Result.
        """
        gate = node.split("_")[0]
        if gate in one_qubit_gates_available.keys() or gate in one_qubit_parameter_gates_available.keys():
            return max_idle_time - self.t1g
        if gate in two_qubit_gates_available.keys():
            return max_idle_time - self.t2g
        return 0.0

    def run(self, qc, sequence: Literal["XY4", "CPMG"] = "XY4", align_right: bool = True, insert_before_barrier: bool = False):
        """Insert dynamical decoupling sequences (XY4 or CPMG) into idle windows of the circuit.

        Args:
            qc: Quantum circuit.
            sequence (*Literal['XY4', 'CPMG']*): Sequence (``Literal['XY4', 'CPMG']``). Defaults to ``'XY4'``.
            align_right (*bool*): Align right (``bool``). Defaults to ``True``.
            insert_before_barrier (*bool*): Insert before barrier (``bool``). Defaults to ``False``.

        Returns:
            ``QuantumCircuit`` with DD sequences inserted into idle windows.

        Raises:
            ValueError: f'Sequence {sequence} is not support now!
        """
        if sequence == "XY4":
            sequence_length = 4
        elif sequence == "CPMG":
            sequence_length = 2
        else:
            raise ValueError(f"Sequence {sequence} is not support now!")

        dag = qc2dag(qc, show_qubits=False)
        qubit_idle_time = {k: {"current_node": None, "idle_time": 0} for k in qc.qubits}
        dag_copy = copy.deepcopy(dag)

        if align_right:
            topological_generations = []
            rev_dag = dag_copy.reverse()
            for nodes in nx.topological_generations(rev_dag):
                topological_generations.insert(0, nodes)
        else:
            topological_generations = nx.topological_generations(dag_copy)

        for nodes in topological_generations:
            max_idle_time = self._get_max_idle_time(nodes)
            node_qubits_dic = {node: dag_copy.nodes[node]["qubits"] for node in nodes}
            qubit_node_dic = {}
            for k, vv in node_qubits_dic.items():
                for v in vv:
                    qubit_node_dic[v] = k
            for qubit, node in qubit_node_dic.items():
                pre_node = qubit_idle_time[qubit]["current_node"]
                idle_time = qubit_idle_time[qubit]["idle_time"]
                if pre_node is None:
                    if idle_time > 0:
                        delay_nodes = [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": idle_time})]
                        delay_edges = [(delay_nodes[0][0], node, {"qubit": [qubit]})]
                        dag.add_nodes_from(delay_nodes)
                        dag.add_edges_from(delay_edges)
                    qubit_idle_time[qubit]["idle_time"] = self._update_idle_time(node, max_idle_time)
                    qubit_idle_time[qubit]["current_node"] = node
                else:
                    if idle_time >= self.t1g * sequence_length:
                        if node.split("_")[0] == "barrier" and not insert_before_barrier:
                            qubit_idle_time[qubit]["idle_time"] = self._update_idle_time(node, max_idle_time)
                            qubit_idle_time[qubit]["current_node"] = node
                        else:
                            dag.remove_edge(pre_node, node)
                            n_dd = int(idle_time // (self.t1g * sequence_length))
                            GRID_NS = 0.1
                            tgap_units = round((idle_time - n_dd * sequence_length * self.t1g) / sequence_length / n_dd / (GRID_NS * 1e-9))
                            tgap = tgap_units * GRID_NS * 1e-9
                            tgap_half = tgap / 2
                            if sequence == "XY4":
                                dd_nodes = []
                                for idx in range(n_dd):
                                    if idx == 0:
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap_half})]
                                    else:
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                    if idx % 2 == 0:
                                        dd_nodes += [(f"x_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                        dd_nodes += [(f"y_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                        dd_nodes += [(f"x_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                        dd_nodes += [(f"y_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                    else:
                                        dd_nodes += [(f"y_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                        dd_nodes += [(f"x_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                        dd_nodes += [(f"y_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                        dd_nodes += [(f"x_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                    if idx == n_dd - 1:
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap_half})]
                            else:
                                dd_nodes = []
                                for idx in range(n_dd):
                                    if idx == 0:
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap_half})]
                                    else:
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                    dd_nodes += [(f"x_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                    if tgap > 0:
                                        dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap})]
                                    dd_nodes += [(f"x_{self.counter()}_[{qubit}]", {"qubits": [qubit]})]
                                    if idx == n_dd - 1:
                                        if tgap > 0:
                                            dd_nodes += [(f"delay_{self.counter()}_[{qubit}]", {"qubits": [qubit], "duration": tgap_half})]
                            dd_edges = [(dd_nodes[i][0], dd_nodes[i + 1][0], {"qubit": [qubit]}) for i in range(len(dd_nodes) - 1)]
                            dd_edges.append((pre_node, dd_nodes[0][0], {"qubit": [qubit]}))
                            dd_edges.append((dd_nodes[-1][0], node, {"qubit": [qubit]}))
                            dag.add_nodes_from(dd_nodes)
                            dag.add_edges_from(dd_edges)
                            qubit_idle_time[qubit]["idle_time"] = self._update_idle_time(node, max_idle_time)
                            qubit_idle_time[qubit]["current_node"] = node
                    else:
                        qubit_idle_time[qubit]["idle_time"] = self._update_idle_time(node, max_idle_time)
                        qubit_idle_time[qubit]["current_node"] = node
            for q in qubit_idle_time.keys():
                if q not in qubit_node_dic.keys():
                    qubit_idle_time[q]["idle_time"] += max_idle_time
        qc_new = dag2qc(dag)
        return qc_new
