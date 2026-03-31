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

r""" 
This module contains the `GateCompressor` class, which is designed to optimize quantum circuits by 
merging or compressing adjacent gates.
"""

import numpy as np

from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    three_qubit_gates_available,
)
from .dag import qc2dag, dag2qc
from ..circuit.utils import u3_decompose
from ..circuit.matrix import u_mat, gate_matrix_dict
from .basepasses import TranspilerPass

# Gates that are diagonal in the computational basis.
_DIAGONAL_1Q_GATES = frozenset({'id', 'z', 's', 'sdg', 't', 'tdg', 'rz', 'p'})
# Functional / non-unitary instructions that must never be reordered.
_NON_REORDERABLE = frozenset({'barrier', 'measure', 'reset', 'delay'})


class GateCompressor(TranspilerPass):
    """A transpiler pass that merges or compresses adjacent gates in a quantum circuit."""

    def __init__(self):
        super().__init__()
        self.compressible_gates = [
            "id",
            "x",
            "y",
            "z",
            "h",
            "cx",
            "cnot",
            "cy",
            "cz",
            "swap",
            "rx",
            "ry",
            "rz",
            "p",
            "u",
            "rxx",
            "ryy",
            "rzz",
            "ccx",
            "ccz",
            "cswap",
        ]
        self._idx = 1000000
        self._single_qubit_gates = (
            set(one_qubit_gates_available.keys())
            | set(one_qubit_parameter_gates_available.keys())
        )

    # ---- commutation helpers ----

    @staticmethod
    def _get_gate_qubits(gate_info):
        """Return the qubit indices for a gate as a list."""
        gate = gate_info[0]
        if gate in one_qubit_gates_available or gate in one_qubit_parameter_gates_available:
            return [gate_info[-1]]
        if gate in two_qubit_gates_available or gate in two_qubit_parameter_gates_available:
            return [gate_info[-2], gate_info[-1]]
        if gate in three_qubit_gates_available:
            return [gate_info[-3], gate_info[-2], gate_info[-1]]
        return []  # unknown / functional gate

    @staticmethod
    def _get_any_gate_matrix(gate_info):
        """Return the unitary matrix for any gate, or None if symbolic."""
        gate = gate_info[0]
        if gate in one_qubit_gates_available:
            return gate_matrix_dict.get(gate)
        if gate in one_qubit_parameter_gates_available:
            params = gate_info[1:-1]
            if any(isinstance(p, str) for p in params):
                return None
            return gate_matrix_dict[gate](*params)
        if gate in two_qubit_gates_available:
            return gate_matrix_dict.get(gate)
        if gate in two_qubit_parameter_gates_available:
            params = gate_info[1:-2]
            if any(isinstance(p, str) for p in params):
                return None
            return gate_matrix_dict[gate](*params)
        if gate in three_qubit_gates_available:
            return gate_matrix_dict.get(gate)
        return None

    @staticmethod
    def _expand_matrix(gate_mat, positions, n_total):
        """Expand *gate_mat* acting on *positions* to the full *n_total*-qubit space."""
        dim = 2 ** n_total
        gate_n = len(positions)
        result = np.zeros((dim, dim), dtype=complex)
        pos_set = set(positions)
        for i in range(dim):
            for j in range(dim):
                # Non-gate bits must be equal.
                ok = True
                for pos in range(n_total):
                    if pos not in pos_set:
                        if ((i >> (n_total - 1 - pos)) & 1) != ((j >> (n_total - 1 - pos)) & 1):
                            ok = False
                            break
                if not ok:
                    continue
                ig = jg = 0
                for k, pos in enumerate(positions):
                    ig |= ((i >> (n_total - 1 - pos)) & 1) << (gate_n - 1 - k)
                    jg |= ((j >> (n_total - 1 - pos)) & 1) << (gate_n - 1 - k)
                result[i, j] = gate_mat[ig, jg]
        return result

    @classmethod
    def _check_commutation(cls, gate_info1, gate_info2):
        """Return True if *gate_info1* and *gate_info2* commute."""
        g1, g2 = gate_info1[0], gate_info2[0]
        # Functional gates never commute with anything.
        if g1 in _NON_REORDERABLE or g2 in _NON_REORDERABLE:
            return False
        q1 = cls._get_gate_qubits(gate_info1)
        q2 = cls._get_gate_qubits(gate_info2)
        s1, s2 = set(q1), set(q2)
        # Disjoint qubits -> always commute.
        if s1.isdisjoint(s2):
            return True
        # Fast path: both diagonal single-qubit on same qubit.
        if g1 in _DIAGONAL_1Q_GATES and g2 in _DIAGONAL_1Q_GATES:
            return True
        # Diagonal single-qubit + CZ (diagonal 2-qubit).
        if g1 in _DIAGONAL_1Q_GATES and g2 == 'cz':
            return True
        if g2 in _DIAGONAL_1Q_GATES and g1 == 'cz':
            return True
        # CZ-CZ (both diagonal).
        if g1 == 'cz' and g2 == 'cz':
            return True
        # Matrix fallback.
        m1 = cls._get_any_gate_matrix(gate_info1)
        m2 = cls._get_any_gate_matrix(gate_info2)
        if m1 is None or m2 is None:
            return False
        all_qubits = sorted(s1 | s2)
        p1 = [all_qubits.index(q) for q in q1]
        p2 = [all_qubits.index(q) for q in q2]
        n = len(all_qubits)
        full1 = cls._expand_matrix(m1, p1, n)
        full2 = cls._expand_matrix(m2, p2, n)
        return np.allclose(full1 @ full2, full2 @ full1)

    # ---- commutation-based gate reordering ----

    # Two-qubit gates that are self-inverse: G·G = I.
    _SELF_INVERSE_2Q = frozenset({'cx', 'cnot', 'cy', 'cz', 'swap'})

    def commutation_reorder(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Bubble single-qubit gates past commuting gates to form longer
        single-qubit runs on the same qubit, enabling more merges."""
        gates = list(qc.gates)
        for i in range(1, len(gates)):
            if gates[i][0] not in self._single_qubit_gates:
                continue
            qubit = self._gate_qubit(gates[i])
            j = i
            while j > 0:
                prev_gate = gates[j - 1][0]
                # Never cross functional instructions.
                if prev_gate in _NON_REORDERABLE:
                    break
                # Stop next to a single-qubit gate on the same qubit (merge target).
                if prev_gate in self._single_qubit_gates and self._gate_qubit(gates[j - 1]) == qubit:
                    break
                # Check commutation (handles disjoint and overlapping cases).
                if not self._check_commutation(gates[j], gates[j - 1]):
                    break
                gates[j], gates[j - 1] = gates[j - 1], gates[j]
                j -= 1
        new_qc = qc.deepcopy()
        new_qc.gates = gates
        return new_qc

    # ---- two-qubit pair cancellation ----

    def cancel_two_qubit_pairs(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Cancel pairs of identical self-inverse two-qubit gates when all
        intermediate gates commute with them.

        For example, CZ(a,b) ... CZ(a,b) can be removed if every gate
        between them acts on qubits disjoint from {a, b} (or otherwise
        commutes with CZ).
        """
        gates = list(qc.gates)
        cancelled: set[int] = set()
        changed = True
        while changed:
            changed = False
            for i in range(len(gates)):
                if i in cancelled:
                    continue
                name_i = gates[i][0]
                if name_i not in self._SELF_INVERSE_2Q:
                    continue
                qubits_i = self._get_gate_qubits(gates[i])
                if len(qubits_i) != 2:
                    continue
                # Look forward for a matching gate.
                for j in range(i + 1, len(gates)):
                    if j in cancelled:
                        continue
                    name_j = gates[j][0]
                    # Must be the same gate on the same qubits (order-insensitive for symmetric gates).
                    if name_j != name_i:
                        # Check commutation of this intermediate gate with gate i.
                        if not self._check_commutation(gates[i], gates[j]):
                            break
                        continue
                    qubits_j = self._get_gate_qubits(gates[j])
                    same_qubits = (qubits_j == qubits_i)
                    # CZ and SWAP are symmetric in qubit order.
                    if not same_qubits and name_i in ('cz', 'swap'):
                        same_qubits = (qubits_j == qubits_i[::-1])
                    if same_qubits:
                        # All intermediate gates commuted — cancel the pair.
                        cancelled.add(i)
                        cancelled.add(j)
                        changed = True
                        break
                    else:
                        # Same gate type but different qubits; check commutation.
                        if not self._check_commutation(gates[i], gates[j]):
                            break
                if changed:
                    break
        if not cancelled:
            return qc
        new_gates = [g for idx, g in enumerate(gates) if idx not in cancelled]
        new_qc = qc.deepcopy()
        new_qc.gates = new_gates
        return new_qc

    # ---- single-qubit block merge ----

    @staticmethod
    def _gate_matrix(gate_info):
        """Return the 2×2 unitary for a single-qubit gate, or None if symbolic."""
        gate = gate_info[0]
        if gate in one_qubit_gates_available:
            return gate_matrix_dict[gate]
        if gate in one_qubit_parameter_gates_available:
            params = gate_info[1:-1]
            if any(isinstance(p, str) for p in params):
                return None
            return gate_matrix_dict[gate](*params)
        return None

    @staticmethod
    def _gate_qubit(gate_info):
        """Return the qubit index for a single-qubit gate."""
        return gate_info[-1]

    def merge_single_qubit_runs(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Merge consecutive single-qubit gates on the same qubit into one u gate."""
        new_qc = qc.deepcopy()
        gates = list(qc.gates)
        result = []
        i = 0
        while i < len(gates):
            gate = gates[i][0]
            if gate not in self._single_qubit_gates:
                result.append(gates[i])
                i += 1
                continue
            # Start collecting a run of single-qubit gates on the same qubit.
            qubit = self._gate_qubit(gates[i])
            run_start = i
            mat = self._gate_matrix(gates[i])
            if mat is None:
                # Symbolic parameter — can't merge, emit as-is.
                result.append(gates[i])
                i += 1
                continue
            accumulated = np.array(mat, dtype=complex)
            i += 1
            while i < len(gates):
                g = gates[i][0]
                if g not in self._single_qubit_gates:
                    break
                if self._gate_qubit(gates[i]) != qubit:
                    break
                m = self._gate_matrix(gates[i])
                if m is None:
                    break
                accumulated = np.array(m, dtype=complex) @ accumulated
                i += 1
            run_length = i - run_start
            if run_length == 1:
                # Single gate, no merge needed.
                result.append(gates[run_start])
                continue
            # Check if the accumulated matrix is identity.
            if np.allclose(accumulated, np.eye(2)):
                continue  # Entire run cancels out.
            theta, phi, lamda, _ = u3_decompose(accumulated)
            result.append(("u", theta, phi, lamda, qubit))
        new_qc.gates = result
        return new_qc

    def remove_identity_gates(self, qc: QuantumCircuit):
        new_qc = qc.deepcopy()
        new = []
        for gate_info in qc.gates:
            gate = gate_info[0]
            if gate in one_qubit_parameter_gates_available.keys():
                params = gate_info[1:-1]
                mat = gate_matrix_dict[gate](*params)
                idm = np.eye(mat.shape[0])
                if np.allclose(mat, idm) is False:
                    new.append(gate_info)
            elif gate in two_qubit_parameter_gates_available.keys():
                params = gate_info[1:-2]
                mat = gate_matrix_dict[gate](*params)
                idm = np.eye(mat.shape[0])
                if np.allclose(mat, idm) is False:
                    new.append(gate_info)
            else:
                new.append(gate_info)
        new_qc.gates = new
        return new_qc

    def is_adjacent_gates(self, node1: str, node2: str):
        gate1 = node1.split("_")[0]
        gate2 = node2.split("_")[0]
        qubits1 = self.dag.nodes[node1]["qubits"]
        qubits2 = self.dag.nodes[node2]["qubits"]
        if (
            gate1 == gate2
            and gate1 in self.compressible_gates
            and qubits1 == qubits2
            and list(self.dag.out_edges(node1)) == list(self.dag.in_edges(node2))
        ):
            return True
        return False

    def has_adjacent_gates(self):
        for edge in self.dag.edges():
            if self.is_adjacent_gates(edge[0], edge[1]):
                return True
        return False

    def compress_adjacent_single_qubit_gates(self, node1: str, node2: str):
        nodes_remove = [node1, node2]
        nodes_added = []
        edges_added = []
        node1_predecessors = list(self.dag.predecessors(node1))
        if len(node1_predecessors) == 0:
            node1_pre = None
        elif len(node1_predecessors) == 1:
            node1_pre = node1_predecessors[0]
        node2_successors = list(self.dag.successors(node2))
        if len(node2_successors) == 0:
            node2_suc = None
        elif len(node1_predecessors) == 1:
            node2_suc = node2_successors[0]
        if node1_pre is not None and node2_suc is not None:
            if self.dag.has_edge(node1_pre, node2_suc):
                qubit = self.dag.nodes[node1_pre]["qubits"]
                edges_added.append((node1_pre, node2_suc, {"qubit": list(sorted(qubit))}))
            else:
                qubit = self.dag.get_edge_data(node1, node2)["qubit"]
                edges_added.append((node1_pre, node2_suc, {"qubit": qubit}))
        return nodes_remove, nodes_added, edges_added

    def compress_adjacent_single_parameter_qubit_gates(self, node1: str, node2: str):
        nodes_remove = [node1, node2]
        nodes_added = []
        edges_added = []
        node1_predecessors = list(self.dag.predecessors(node1))
        if len(node1_predecessors) == 0:
            node1_pre = None
        elif len(node1_predecessors) == 1:
            node1_pre = node1_predecessors[0]
        node2_successors = list(self.dag.successors(node2))
        if len(node2_successors) == 0:
            node2_suc = None
        elif len(node1_predecessors) == 1:
            node2_suc = node2_successors[0]
        gate = node1.split("_")[0]
        params1 = self.dag.nodes[node1]["params"]
        params2 = self.dag.nodes[node2]["params"]
        if gate == "u":
            u_mat1 = u_mat(*params1)
            u_mat2 = u_mat(*params2)
            new_u = u_mat2 @ u_mat1
            theta, phi, lamda, _ = u3_decompose(new_u)
            params = [theta, phi, lamda]
        else:
            params = [params1[indx] + params2[indx] for indx in range(len(params1))]
        mat = gate_matrix_dict[gate](*params)
        idm = np.eye(mat.shape[0])
        if np.allclose(mat, idm):
            if node1_pre is not None and node2_suc is not None:
                if self.dag.has_edge(node1_pre, node2_suc):
                    qubit = self.dag.nodes[node1_pre]["qubits"]
                    edges_added.append((node1_pre, node2_suc, {"qubit": list(sorted(qubit))}))
                else:
                    qubit = self.dag.get_edge_data(node1, node2)["qubit"]
                    edges_added.append((node1_pre, node2_suc, {"qubit": qubit}))
        else:
            qubits = self.dag.nodes[node1]["qubits"]
            new_node_info = (gate + "_" + str(self.idx) + "_" + str(qubits), {"qubits": qubits, "params": params})
            nodes_added.append(new_node_info)
            if node1_pre is not None:
                qubit = self.dag.get_edge_data(node1_pre, node1)["qubit"]
                edges_added.append((node1_pre, new_node_info[0], {"qubit": qubit}))
            if node2_suc is not None:
                qubit = self.dag.get_edge_data(node2, node2_suc)["qubit"]
                edges_added.append((new_node_info[0], node2_suc, {"qubit": qubit}))
        return nodes_remove, nodes_added, edges_added

    def compress_adjacent_two_qubit_gates(self, node1: str, node2: str):
        nodes_remove = [node1, node2]
        nodes_added = []
        edges_added = []
        node1_predecessors = list(self.dag.predecessors(node1))
        if len(node1_predecessors) == 0:
            node1_pre_dic = None
        else:
            node1_pre_dic = {}
            for node1_pre in node1_predecessors:
                qubit = self.dag.get_edge_data(node1_pre, node1)["qubit"]
                node1_pre_dic[node1_pre] = qubit
        node2_successors = list(self.dag.successors(node2))
        if len(node2_successors) == 0:
            node2_suc_dic = None
        else:
            node2_suc_dic = {}
            for node2_suc in node2_successors:
                qubit = self.dag.get_edge_data(node2, node2_suc)["qubit"]
                node2_suc_dic[node2_suc] = qubit
        if node1_pre_dic is not None and node2_suc_dic is not None:
            for node1_pre, qubits1 in node1_pre_dic.items():
                for node2_suc, qubits2 in node2_suc_dic.items():
                    common_qubits = [q for q in qubits1 if q in qubits2]
                    if len(common_qubits) > 0:
                        if self.dag.has_edge(node1_pre, node2_suc):
                            common_qubits += self.dag.get_edge_data(node1_pre, node2_suc)["qubit"]
                            common_qubits = list(set(common_qubits))
                        edges_added.append((node1_pre, node2_suc, {"qubit": common_qubits}))
        return nodes_remove, nodes_added, edges_added

    def compress_adjacent_two_qubit_parameter_gates(self, node1: str, node2: str):
        nodes_remove = [node1, node2]
        nodes_added = []
        edges_added = []
        node1_predecessors = list(self.dag.predecessors(node1))
        if len(node1_predecessors) == 0:
            node1_pre_dic = None
        else:
            node1_pre_dic = {}
            for node1_pre in node1_predecessors:
                qubit = self.dag.get_edge_data(node1_pre, node1)["qubit"]
                node1_pre_dic[node1_pre] = qubit
        node2_successors = list(self.dag.successors(node2))
        if len(node2_successors) == 0:
            node2_suc_dic = None
        else:
            node2_suc_dic = {}
            for node2_suc in node2_successors:
                qubit = self.dag.get_edge_data(node2, node2_suc)["qubit"]
                node2_suc_dic[node2_suc] = qubit

        gate = node1.split("_")[0]
        params1 = self.dag.nodes[node1]["params"]
        params2 = self.dag.nodes[node2]["params"]
        params = [params1[indx] + params2[indx] for indx in range(len(params1))]
        mat = gate_matrix_dict[gate](*params)
        idm = np.eye(mat.shape[0])
        if np.allclose(mat, idm):
            if node1_pre_dic is not None and node2_suc_dic is not None:
                for node1_pre, qubits1 in node1_pre_dic.items():
                    for node2_suc, qubits2 in node2_suc_dic.items():
                        common_qubits = [q for q in qubits1 if q in qubits2]
                        if len(common_qubits) > 0:
                            if self.dag.has_edge(node1_pre, node2_suc):
                                common_qubits += self.dag.get_edge_data(node1_pre, node2_suc)["qubit"]
                                common_qubits = list(set(common_qubits))
                            edges_added.append((node1_pre, node2_suc, {"qubit": common_qubits}))
        else:
            qubits = self.dag.nodes[node1]["qubits"]
            new_node_info = (gate + "_" + str(self.idx) + "_" + str(qubits), {"qubits": qubits, "params": params})
            nodes_added.append(new_node_info)
            if node1_pre_dic is not None:
                for node1_pre, qubits1 in node1_pre_dic.items():
                    edges_added.append((node1_pre, new_node_info[0], {"qubit": qubits1}))
            if node2_suc_dic is not None:
                for node2_suc, qubits2 in node2_suc_dic.items():
                    edges_added.append((new_node_info[0], node2_suc, {"qubit": qubits2}))
        return nodes_remove, nodes_added, edges_added

    def compress_adjacent_three_qubit_gates(self, node1, node2):
        nodes_remove = [node1, node2]
        nodes_added = []
        edges_added = []
        node1_predecessors = list(self.dag.predecessors(node1))
        if len(node1_predecessors) == 0:
            node1_pre_dic = None
        else:
            node1_pre_dic = {}
            for node1_pre in node1_predecessors:
                qubit = self.dag.get_edge_data(node1_pre, node1)["qubit"]
                node1_pre_dic[node1_pre] = qubit
        node2_successors = list(self.dag.successors(node2))
        if len(node2_successors) == 0:
            node2_suc_dic = None
        else:
            node2_suc_dic = {}
            for node2_suc in node2_successors:
                qubit = self.dag.get_edge_data(node2, node2_suc)["qubit"]
                node2_suc_dic[node2_suc] = qubit
        if node1_pre_dic is not None and node2_suc_dic is not None:
            for node1_pre, qubits1 in node1_pre_dic.items():
                for node2_suc, qubits2 in node2_suc_dic.items():
                    common_qubits = [q for q in qubits1 if q in qubits2]
                    if len(common_qubits) > 0:
                        if self.dag.has_edge(node1_pre, node2_suc):
                            common_qubits += self.dag.get_edge_data(node1_pre, node2_suc)["qubit"]
                            common_qubits = list(set(common_qubits))
                        edges_added.append((node1_pre, node2_suc, {"qubit": common_qubits}))
        return nodes_remove, nodes_added, edges_added

    def run_compress_once(self, node1: str, node2: str):
        gate = node1.split("_")[0]
        if gate in one_qubit_gates_available.keys():
            return self.compress_adjacent_single_qubit_gates(node1, node2)
        if gate in one_qubit_parameter_gates_available.keys():
            return self.compress_adjacent_single_parameter_qubit_gates(node1, node2)
        if gate in two_qubit_gates_available.keys():
            return self.compress_adjacent_two_qubit_gates(node1, node2)
        if gate in two_qubit_parameter_gates_available.keys():
            return self.compress_adjacent_two_qubit_parameter_gates(node1, node2)
        if gate in three_qubit_gates_available.keys():
            return self.compress_adjacent_three_qubit_gates(node1, node2)
        return None

    @property
    def idx(self):
        self._idx += 1
        return self._idx

    def run(self, qc: QuantumCircuit):
        qubits = qc.qubits
        qc1 = self.remove_identity_gates(qc)
        qc1 = self.commutation_reorder(qc1)
        qc1 = self.merge_single_qubit_runs(qc1)
        qc1 = self.cancel_two_qubit_pairs(qc1)
        self.dag = qc2dag(qc1)

        compress = self.has_adjacent_gates()
        while compress:
            for edge in self.dag.edges():
                node1, node2 = edge
                if self.is_adjacent_gates(node1, node2):
                    nodes_remove, nodes_added, edges_added = self.run_compress_once(node1, node2)
                    self.dag.remove_nodes_from(nodes_remove)
                    self.dag.add_nodes_from(nodes_added)
                    self.dag.add_edges_from(edges_added)
                    break
            compress = self.has_adjacent_gates()
        new_qc = dag2qc(self.dag, qc1.nqubits, qc1.ncbits)
        new_qc.qubits = qubits
        return new_qc
