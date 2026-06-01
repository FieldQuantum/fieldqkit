"""A toolkit for the SABRE algorithm.

SPDX-License-Identifier: Apache-2.0
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import math
import random
from typing import Literal
import networkx as nx
from networkx import floyd_warshall_numpy
from ..circuit.quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    three_qubit_gates_available,
    functional_gates_available,
)
from ..circuit import QuantumCircuit
from .dag import qc2dag, split_qubits
from .basepasses import TranspilerPass
import re


def extract_qubits(node_name: str):
    """Parse qubit indices from a DAG node name string.

    Args:
        node_name (*str*): DAG node name containing qubit indices in brackets, e.g. ``'cx_42_[0, 1]'``.

    Returns:
        List of integer qubit indices parsed from the node name.
    """
    bracket_content = re.search(r"\[([^\]]*)\]", node_name)
    if not bracket_content:
        return []
    return list(map(int, re.findall(r"\d+", bracket_content.group(1))))


def update_v2p_and_p2v_mapping(v2p: dict, swap_gate_info: tuple):
    """Return updated virtual-to-physical and physical-to-virtual qubit mappings after applying a SWAP.

    Args:
        v2p (*dict*): Current virtual-to-physical qubit mapping ``{virtual: physical}``.
        swap_gate_info (*tuple*): SWAP gate tuple ``('swap', vq1, vq2)``.

    Returns:
        ``(v2p, p2v)`` —updated mapping dictionaries.
    """
    v2p = copy.deepcopy(v2p)
    vq1, vq2 = swap_gate_info[1:]
    v2p[vq1], v2p[vq2] = v2p[vq2], v2p[vq1]
    p2v = {p: v for v, p in v2p.items()}
    return v2p, p2v


class SabreRouting(TranspilerPass):
    """SABRE-based routing pass for quantum circuit transpilation."""

    def __init__(
        self,
        subgraph: nx.Graph,
        initial_mapping: Literal["random", "trivial"] | list = "trivial",
        do_random_choice: bool = False,
        iterations: int = 5,
        heuristic: Literal["basic", "lookahead", "basic_decay", "lookahead_decay"] = "lookahead_decay",
        max_extended_set_weight: float = 0.5,
        noise_aware: bool = False,
        n_trials: int = 1,
    ):
        """Initialize the SABRE router with coupling topology, distance matrices, heuristic parameters, and mapping strategy.

        Args:
            subgraph (*nx.Graph*): Device coupling graph whose nodes are physical qubits.
            initial_mapping (*Literal['random', 'trivial'] | list*): Strategy or explicit list for the initial virtual-to-physical qubit mapping. Defaults to ``'trivial'``.
            do_random_choice (*bool*): If ``True``, break ties randomly when selecting SWAPs. Defaults to ``False``.
            iterations (*int*): Number of forward/backward SABRE iterations. Defaults to ``5``.
            heuristic (*Literal['basic', 'lookahead', 'basic_decay', 'lookahead_decay']*): SWAP scoring heuristic. Defaults to ``'lookahead_decay'``.
            max_extended_set_weight (*float*): Weight factor for the extended successor set in lookahead scoring. Defaults to ``0.5``.
            noise_aware (*bool*): Whether to use noise-aware strategies. Defaults to ``False``.
            n_trials (*int*): Number of independent routing trials (best result kept). Defaults to ``1``.
        """
        super().__init__()
        self.coupling_graph = subgraph
        if noise_aware:
            # Weight edges by -log(fidelity) so high-fidelity paths are cheaper.
            wg = subgraph.copy()
            all_perfect = True
            for u, v, data in wg.edges(data=True):
                f = data.get("fidelity", 1.0)
                f = max(f, 1e-6)  # avoid log(0)
                w = -math.log(f)
                if w > 1e-12:
                    all_perfect = False
                wg[u][v]["weight"] = w
            if all_perfect:
                # All fidelities ≈1 →noise-aware gives no useful gradient;
                # fall back to hop-count distances.
                self.distance_matrix = floyd_warshall_numpy(subgraph)
            else:
                self.distance_matrix = floyd_warshall_numpy(wg, weight="weight")
        else:
            self.distance_matrix = floyd_warshall_numpy(subgraph)
        # Hop-count matrix (always unweighted) for adjacency checks.
        self.hop_matrix = floyd_warshall_numpy(subgraph)
        self.n_trials = n_trials
        if "normal_order" not in subgraph.graph:
            subgraph.graph["normal_order"] = list(subgraph.nodes())
        self.physical_qubits = [int(n) for n in subgraph.graph["normal_order"]]
        self.physical_qubits_index = dict(zip(list(subgraph.nodes), range(len(subgraph.nodes))))
        self.initial_mapping = initial_mapping
        self.do_random_choice = do_random_choice
        self.iterations = iterations
        self.heuristic = heuristic
        self.extended_successor_set = []
        self.max_extended_set_weight = max_extended_set_weight
        self.decay_parameter = {}
        self._cache = OrderedDict()

    def _distance_matrix_element(self, pq1: int, pq2: int):
        """Return the shortest-path distance between two physical qubits from the precomputed distance matrix.

        Args:
            pq1 (*int*): First physical qubit index.
            pq2 (*int*): Second physical qubit index.

        Returns:
            ``float`` — shortest-path distance (weighted by ``-log(fidelity)`` when
            ``noise_aware=True``, unweighted hop-count otherwise).
        """
        idx1 = self.physical_qubits_index[pq1]
        idx2 = self.physical_qubits_index[pq2]
        return self.distance_matrix[idx1][idx2]

    def _dag_successors(self, node: str):
        """Return the cached successor nodes of a DAG node.

        Args:
            node (*str*): DAG node name.

        Returns:
            List of successor node names.
        """
        return self._get_nodes(node, "successors")

    def _dag_predecessors(self, node: str):
        """Return the cached predecessor nodes of a DAG node.

        Args:
            node (*str*): DAG node name.

        Returns:
            List of predecessor node names.
        """
        return self._get_nodes(node, "predecessors")

    def _get_nodes(self, node: str, query_type: Literal["successors", "predecessors"]):
        """Retrieve and cache successor or predecessor nodes for a DAG node using an LRU cache.

        Args:
            node (*str*): DAG node name.
            query_type (*Literal['successors', 'predecessors']*): Direction to query.

        Returns:
            List of neighboring node names.
        """
        key = (self.dag_id, id(node), query_type)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if query_type == "successors":
            nodes = list(self.dag.successors(node))
        else:
            nodes = list(self.dag.predecessors(node))
        self._cache[key] = nodes
        if len(self._cache) > 10000:
            self._cache.popitem(last=False)
        return nodes

    def _initialize_v2p_p2v(self, virtual_qubits: list):
        """Create initial virtual-to-physical qubit mappings from the configured strategy (trivial, random, or explicit list).

        Args:
            virtual_qubits (*list*): List of virtual qubit indices from the circuit.

        Returns:
            ``(v2p, p2v)`` —virtual-to-physical and physical-to-virtual mapping dictionaries.

        Raises:
            ValueError: If *initial_mapping* length doesn't match the physical qubits,
                or the mapping strategy string is unrecognised.
        """
        if isinstance(self.initial_mapping, list):
            if len(set(self.initial_mapping)) != len(set(self.physical_qubits)):
                raise ValueError(
                    f"The number of initial_mapping does not match the number of physical qubits.{self.initial_mapping} {self.physical_qubits}"
                )
            v2p = dict(zip(virtual_qubits, self.initial_mapping))
        elif isinstance(self.initial_mapping, str):
            if len(set(virtual_qubits)) != len(set(self.physical_qubits)):
                raise ValueError(
                    f"The number of virtual qubits does not match the number of physical qubits.{virtual_qubits} {self.physical_qubits}"
                )
            if self.initial_mapping == "trivial":
                v2p = dict(zip(virtual_qubits, self.physical_qubits))
            elif self.initial_mapping == "random":
                shuffle_physical_qubits = random.sample(self.physical_qubits, len(self.physical_qubits))
                v2p = dict(zip(virtual_qubits, shuffle_physical_qubits))
            else:
                raise ValueError(f"There is a spelling error in the input of initial_mapping {self.initial_mapping}.")
        else:
            raise ValueError("Invalid input type for initial_mapping —only str and list are supported.")
        p2v = {p: v for v, p in v2p.items()}
        return v2p, p2v

    def _mapping_node_to_gate_info(self, node: str):
        """Convert a DAG node into a gate-info tuple with physical qubit indices using the current v2p mapping.

        Args:
            node (*str*): DAG node identifier (e.g. ``"cx_0_3"``).

        Returns:
            Gate-info tuple with physical qubit indices.

        Raises:
            ValueError: If the gate has not been decomposed into supported basis gates.
        """
        gate = node.split("_")[0]
        if gate in one_qubit_gates_available.keys():
            qubit0 = extract_qubits(node)[0]
            gate_info = (gate, self.v2p[qubit0])
        elif gate in two_qubit_gates_available.keys():
            qubit1, qubit2 = extract_qubits(node)
            gate_info = (gate, self.v2p[qubit1], self.v2p[qubit2])
        elif gate in three_qubit_gates_available.keys():
            raise ValueError(
                f"Please first decompose the {gate} gate into a combination of single- and two-qubit gates."
            )
        elif gate in one_qubit_parameter_gates_available.keys():
            qubit0 = extract_qubits(node)[0]
            paramslst = self.dag.nodes[node]["params"]
            gate_info = (gate, *paramslst, self.v2p[qubit0])
        elif gate in two_qubit_parameter_gates_available.keys():
            paramslst = self.dag.nodes[node]["params"]
            qubit1, qubit2 = extract_qubits(node)
            gate_info = (gate, *paramslst, self.v2p[qubit1], self.v2p[qubit2])
        elif gate in functional_gates_available.keys():
            if gate == "measure":
                qubitlst = self.dag.nodes[node]["qubits"]
                cbitlst = self.dag.nodes[node]["cbits"]
                gate_info = (gate, [self.v2p[qubit] for qubit in qubitlst], cbitlst)
            elif gate == "barrier":
                qubitlst = extract_qubits(node)
                phy_qubitlst = [self.v2p[qubit] for qubit in qubitlst]
                gate_info = (gate, tuple(phy_qubitlst))
            elif gate == "delay":
                qubitlst = self.dag.nodes[node]["qubits"]
                phy_qubitlst = [self.v2p[qubit] for qubit in qubitlst]
                duration = self.dag.nodes[node]["duration"]
                gate_info = (gate, duration, tuple(phy_qubitlst))
            elif gate == "reset":
                qubit0 = self.dag.nodes[node]["qubits"][0]
                gate_info = (gate, self.v2p[qubit0])
        return gate_info

    def _get_extended_successor_set(self, front_layer: list):
        """Collect two-qubit-gate successors of the front layer as the SABRE lookahead set.

        Args:
            front_layer (*list*): Front layer (``list``).
        """
        if "lookahead" in self.heuristic:
            two_qubit_gates = list(two_qubit_gates_available.keys()) + list(two_qubit_parameter_gates_available.keys())
            E = set()
            for node in front_layer:
                for node_successor in self._dag_successors(node):
                    gate = node_successor.split("_")[0]
                    if gate in two_qubit_gates and len(E) <= len(self.v2p):
                        E.update([node_successor])
            self.extended_successor_set = list(E)

    def _hop_distance(self, pq1: int, pq2: int):
        """Return the unweighted hop-count distance between two physical qubits.

        Args:
            pq1 (*int*): First physical qubit index.
            pq2 (*int*): Second physical qubit index.

        Returns:
            ``float`` hop-count distance.
        """
        idx1 = self.physical_qubits_index[pq1]
        idx2 = self.physical_qubits_index[pq2]
        return self.hop_matrix[idx1][idx2]

    def _get_execute_node_list(self, front_layer: list):
        """Return the subset of front-layer gates that can execute immediately on adjacent physical qubits.

        Args:
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            List of executable DAG node names.
        """
        execute_node_list = []
        for node in front_layer:
            gate = node.split("_")[0]
            if gate not in two_qubit_gates_available.keys() and gate not in two_qubit_parameter_gates_available.keys():
                execute_node_list.append(node)
            else:
                vq1, vq2 = extract_qubits(node)
                pq1, pq2 = self.v2p[vq1], self.v2p[vq2]
                if self._hop_distance(pq1, pq2) == 1:
                    execute_node_list.append(node)
        return execute_node_list

    def _has_no_correlation_on_front_layer(self, node: str, front_layer: list):
        """Check whether a node's qubits are disjoint from all qubits currently in the front layer.

        Args:
            node (*str*): DAG node name.
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            ``bool`` —``True`` if the node's qubits are disjoint from front-layer qubits.
        """
        qubitlst = []
        for fnode in front_layer:
            qubits = extract_qubits(fnode)
            qubitlst += qubits
        qubitlst = set(qubitlst)

        node_qubits = set(extract_qubits(node))
        if qubitlst.intersection(node_qubits):
            return False
        return True

    def _get_swap_candidate_list(self, front_layer: list):
        """Generate candidate SWAP gates from physical neighbors of unresolved front-layer two-qubit gates.

        Args:
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            List of ``('swap', vq_min, vq_max)`` candidate tuples.
        """
        swap_candidate_list = []
        for hard_node in front_layer:
            vq1, vq2 = extract_qubits(hard_node)
            pq1_neighbours = self.coupling_graph.neighbors(self.v2p[vq1])
            pq2_neighbours = self.coupling_graph.neighbors(self.v2p[vq2])
            vq1_neighbours = [self.p2v[pq] for pq in pq1_neighbours]
            vq2_neighbours = [self.p2v[pq] for pq in pq2_neighbours]
            for vq in vq1_neighbours:
                poss = [vq, vq1]
                if ("swap", min(poss), max(poss)) not in swap_candidate_list:
                    swap_candidate_list.append(("swap", min(poss), max(poss)))
            for vq in vq2_neighbours:
                poss = [vq, vq2]
                if ("swap", min(poss), max(poss)) not in swap_candidate_list:
                    swap_candidate_list.append(("swap", min(poss), max(poss)))
        return swap_candidate_list

    def _reset_decay_parameter(self):
        """Reset per-physical-qubit decay weights to 1 for the decay-based heuristic.
        """
        if "decay" in self.heuristic:
            self.decay_parameter = {k: 1 for k in self.physical_qubits}

    def _update_decay_parameter(self, min_score_swap_gate_info: tuple):
        """Increment the decay weight of both physical qubits involved in the chosen SWAP.

        Args:
            min_score_swap_gate_info (*tuple*): ``(score, qubit1, qubit2)`` tuple for the best SWAP candidate.
        """
        if "decay" in self.heuristic:
            min_score_swap_qubits = list(min_score_swap_gate_info[1:])
            pq1 = self.v2p[min_score_swap_qubits[0]]
            pq2 = self.v2p[min_score_swap_qubits[1]]
            self.decay_parameter[pq1] = self.decay_parameter[pq1] + 0.01
            self.decay_parameter[pq2] = self.decay_parameter[pq2] + 0.01

    def _heuristic_score_basic(self, swap_gate_info: tuple, front_layer: list):
        """Compute the basic SABRE heuristic score: mean front-layer gate distance after a candidate SWAP.

        Args:
            swap_gate_info (*tuple*): Candidate SWAP gate tuple ``('swap', vq1, vq2)``.
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            ``float`` mean distance score.
        """
        v2p, _ = update_v2p_and_p2v_mapping(self.v2p, swap_gate_info)
        F = front_layer
        size_F = len(F)
        f_distance = 0
        for node in F:
            vq1, vq2 = extract_qubits(node)
            f_distance += self._distance_matrix_element(v2p[vq1], v2p[vq2])
        return f_distance / size_F

    def _heuristic_score_lookahead(self, swap_gate_info: tuple, front_layer: list):
        """Compute the SABRE lookahead heuristic: front-layer distance plus weighted extended-set distance.

        Args:
            swap_gate_info (*tuple*): Candidate SWAP gate tuple ``('swap', vq1, vq2)``.
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            ``float`` combined heuristic score.
        """
        v2p, _ = update_v2p_and_p2v_mapping(self.v2p, swap_gate_info)
        F = front_layer
        E = self.extended_successor_set
        size_E = len(E)
        if size_E == 0:
            size_E = 1
        size_F = len(F)
        W = min(self.max_extended_set_weight, size_E / size_F)
        f_distance = 0
        e_distance = 0
        for node in F:
            vq1, vq2 = extract_qubits(node)
            f_distance += self._distance_matrix_element(v2p[vq1], v2p[vq2])
        for node in E:
            vq1, vq2 = extract_qubits(node)
            e_distance += self._distance_matrix_element(v2p[vq1], v2p[vq2])
        f_distance = f_distance / size_F
        e_distance = e_distance / size_E
        H = f_distance + W * e_distance
        return H

    def _heuristic_score_basic_decay(self, swap_gate_info: tuple, front_layer: list):
        """Compute the SABRE basic-decay heuristic: mean front-layer distance scaled by qubit decay weights.

        Args:
            swap_gate_info (*tuple*): Candidate SWAP gate tuple ``('swap', vq1, vq2)``.
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            ``float`` decay-weighted distance score.
        """
        v2p, _ = update_v2p_and_p2v_mapping(self.v2p, swap_gate_info)
        F = front_layer
        size_F = len(F)
        max_decay = max(self.decay_parameter[v2p[swap_gate_info[1]]], self.decay_parameter[v2p[swap_gate_info[2]]])
        f_distance = 0
        for node in F:
            vq1, vq2 = extract_qubits(node)
            f_distance += self._distance_matrix_element(v2p[vq1], v2p[vq2])
        f_distance = f_distance / size_F
        H = max_decay * f_distance
        return H

    def _heuristic_score_lookahead_decay(self, swap_gate_info: tuple, front_layer: list):
        """Compute the SABRE lookahead-decay heuristic: lookahead distances scaled by qubit decay weights.

        Args:
            swap_gate_info (*tuple*): Candidate SWAP gate tuple ``('swap', vq1, vq2)``.
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            ``float`` decay-weighted lookahead score.
        """
        v2p, _ = update_v2p_and_p2v_mapping(self.v2p, swap_gate_info)
        F = front_layer
        E = self.extended_successor_set
        size_E = len(E)
        if size_E == 0:
            size_E = 1
        size_F = len(F)
        W = min(self.max_extended_set_weight, size_E / size_F)
        max_decay = max(self.decay_parameter[v2p[swap_gate_info[1]]], self.decay_parameter[v2p[swap_gate_info[2]]])
        f_distance = 0
        e_distance = 0
        for node in F:
            vq1, vq2 = extract_qubits(node)
            f_distance += self._distance_matrix_element(v2p[vq1], v2p[vq2])
        for node in E:
            vq1, vq2 = extract_qubits(node)
            e_distance += self._distance_matrix_element(v2p[vq1], v2p[vq2])
        f_distance = f_distance / size_F
        e_distance = e_distance / size_E
        H = max_decay * (f_distance + W * e_distance)
        return H

    def _heuristic_score(self, swap_gate_info: tuple, front_layer: list):
        """Dispatch to the configured SABRE heuristic scoring function and return the score.

        Args:
            swap_gate_info (*tuple*): Candidate SWAP gate tuple ``('swap', vq1, vq2)``.
            front_layer (*list*): Current front layer of the DAG.

        Returns:
            ``float`` heuristic score.
        """
        if self.heuristic == "basic":
            H = self._heuristic_score_basic(swap_gate_info, front_layer)
        elif self.heuristic == "lookahead":
            H = self._heuristic_score_lookahead(swap_gate_info, front_layer)
        elif self.heuristic == "basic_decay":
            H = self._heuristic_score_basic_decay(swap_gate_info, front_layer)
        elif self.heuristic == "lookahead_decay":
            H = self._heuristic_score_lookahead_decay(swap_gate_info, front_layer)
        return H

    def _single_sabre_routing(self):
        """Execute a single forward pass of SABRE routing.

        Returns:
            ``(new, nswap)`` — routed gate list and number of SWAP gates inserted.
            Final qubit mappings are stored in ``self.v2p`` and ``self.p2v``.
        """
        front_layer = list(nx.topological_generations(self.dag))
        if front_layer != []:
            front_layer = front_layer[0]
        self._reset_decay_parameter()
        self._get_extended_successor_set(front_layer)

        nswap = 0
        front_layer_repeat = copy.deepcopy(front_layer)
        new = []
        collect_execute = []
        while len(front_layer) != 0:
            if front_layer_repeat == front_layer:
                pass
            else:
                front_layer_repeat = copy.deepcopy(front_layer)
                self._get_extended_successor_set(front_layer)
            execute_node_list = self._get_execute_node_list(front_layer)
            if execute_node_list:
                for execute_node in execute_node_list:
                    collect_execute.append(execute_node)
                    front_layer.remove(execute_node)
                    if self.do_map_node_to_gate:
                        gate_info = self._mapping_node_to_gate_info(execute_node)
                        new.append(gate_info)
                    for successor_node in self._dag_successors(execute_node):
                        if self._has_no_correlation_on_front_layer(successor_node, front_layer):
                            predecessors = self._dag_predecessors(successor_node)
                            if all(x in (front_layer + collect_execute) for x in predecessors):
                                front_layer.append(successor_node)
                self._reset_decay_parameter()
            else:
                swap_candidate_list = self._get_swap_candidate_list(front_layer)
                swap_heuristic_score = {}
                for swap_gate_info in swap_candidate_list:
                    score = self._heuristic_score(swap_gate_info, front_layer)
                    swap_heuristic_score[swap_gate_info] = score

                min_score = min(swap_heuristic_score.values())
                best_swap = [swap for swap, score in swap_heuristic_score.items() if score == min_score]
                if len(best_swap) > 1:
                    if self.do_random_choice:
                        min_score_swap_gate_info = random.choice(best_swap)
                    else:
                        min_score_swap_gate_info = best_swap[0]
                else:
                    min_score_swap_gate_info = best_swap[0]

                if self.do_map_node_to_gate:
                    vq1 = min_score_swap_gate_info[1]
                    vq2 = min_score_swap_gate_info[2]
                    pq1 = self.v2p[vq1]
                    pq2 = self.v2p[vq2]
                    new.append(("swap", pq1, pq2))
                    nswap += 1

                self._update_decay_parameter(min_score_swap_gate_info)
                self.v2p, self.p2v = update_v2p_and_p2v_mapping(self.v2p, min_score_swap_gate_info)

        return new, nswap

    def _run_once(self, qc, virtual_qubits, dag, rev_dag):
        """Execute one complete forward/reverse SABRE routing attempt.

        Args:
            qc: Quantum circuit.
            virtual_qubits: List of virtual qubit indices.
            dag: Forward DAG representation of the circuit.
            rev_dag: Reversed DAG for backward passes.

        Returns:
            ``(new, nswap, v2p, final_p2v)`` —routed gate list, swap count, and final mappings.
        """
        self.v2p, self.p2v = self._initialize_v2p_p2v(virtual_qubits)
        init_p2v = {p: v for v, p in self.v2p.items()}

        self.do_map_node_to_gate = False
        for idx in range(self.iterations):
            if idx == self.iterations - 1:
                self.do_map_node_to_gate = True
            if idx % 2 == 0:
                self.dag_id = "forward"
                self.dag = dag
            else:
                self.dag_id = "reverse"
                self.dag = rev_dag

            new, nswap = self._single_sabre_routing()

            if self.iterations == 1:
                best_p2v = init_p2v
            else:
                if idx == self.iterations - 2:
                    best_p2v = {p: v for v, p in self.v2p.items()}

        final_p2v = {p: v for v, p in self.v2p.items()}
        return new, nswap, dict(self.v2p), final_p2v

    def run(self, qc: QuantumCircuit):
        """Run the full SABRE routing pipeline: build DAGs, execute multi-trial forward/reverse routing, and return the routed circuit.

        Args:
            qc (*QuantumCircuit*): Quantum circuit.

        Returns:
            Routed ``QuantumCircuit`` with SWAP gates inserted.
        """
        all_qubits = split_qubits(qc)
        virtual_qubits = [x for sub in all_qubits for x in sub]

        dag = qc2dag(qc, show_qubits=False)
        rev_qc = qc.deepcopy()
        rev_qc.gates.reverse()
        rev_dag = qc2dag(rev_qc, show_qubits=False)

        saved_initial_mapping = self.initial_mapping

        best_new, best_nswap, best_v2p, best_final_p2v = None, float("inf"), None, None
        for trial in range(self.n_trials):
            if trial > 0:
                # After the first trial, use random initial mapping.
                self.initial_mapping = "random"
            new, nswap, v2p, final_p2v = self._run_once(qc, virtual_qubits, dag, rev_dag)
            if nswap < best_nswap:
                best_new, best_nswap, best_v2p, best_final_p2v = new, nswap, v2p, final_p2v

        self.initial_mapping = saved_initial_mapping
        self.v2p = best_v2p

        new_qc = QuantumCircuit(max(self.physical_qubits) + 1, qc.ncbits)
        new_qc.gates = best_new
        new_qc.params_value = qc.params_value
        new_qc.qubits = self.physical_qubits
        # Expose final layout mapping for downstream measurement alignment.
        new_qc.logical_to_physical = best_v2p
        new_qc.physical_to_logical = best_final_p2v
        return new_qc
