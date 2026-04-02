"""This module contains the Layout class, which is designed to select suitable layouts
for quantum circuits on hardware backends.

SPDX-License-Identifier: MIT
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

from __future__ import annotations

import logging
import os
import copy
import networkx as nx
import numpy as np
from typing import Literal

logger = logging.getLogger(__name__)
from itertools import combinations, zip_longest, product
from multiprocessing import Pool
from functools import partial
from ..api.backend import Backend

from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from .dag import split_qubits


class Layout:
    """Responsible for selecting suitable qubit layouts from a given chip for a quantum circuit."""

    def __init__(self, chip_backend: Backend):
        """Initialize the layout selector with hardware graph, fidelity thresholds, and parallelism settings.

        Args:
            chip_backend (Backend): The backend providing hardware connectivity and fidelity data.
        """
        self.priority_qubits = chip_backend.priority_qubits
        self.graph = chip_backend.edge_filtered_graph(thres=0.6)
        self.ncore = os.cpu_count() // 2
        self.fidelity_mean_threshold = 0.9
        self.edge_fidelitys = nx.get_edge_attributes(self.graph, "fidelity")
        self.algorithm_switch_threshold = 10

    # ---- circuit interaction analysis ----

    _TWO_QUBIT_GATES = frozenset(
        list(two_qubit_gates_available.keys()) + list(two_qubit_parameter_gates_available.keys())
    )

    @classmethod
    def _extract_interaction_graph(cls, qc: QuantumCircuit):
        """Build a weighted graph of virtual-qubit interactions from the circuit.

        Each edge (i, j) has weight = number of two-qubit gates between
virtual qubits i and j.  Returns a ``networkx.Graph``.

        Args:
            qc (*QuantumCircuit*): Quantum circuit.

        Returns:
            ``networkx.Graph`` with edge weights representing gate counts.
        """
        G = nx.Graph()
        G.add_nodes_from(qc.qubits)
        for gate_info in qc.gates:
            gate = gate_info[0]
            if gate in cls._TWO_QUBIT_GATES:
                q1, q2 = gate_info[-2], gate_info[-1]
                if G.has_edge(q1, q2):
                    G[q1][q2]["weight"] += 1
                else:
                    G.add_edge(q1, q2, weight=1)
        return G

    @staticmethod
    def _estimate_routing_cost(interaction_graph, subgraph):
        """Estimate the routing cost of mapping *interaction_graph* onto *subgraph*.

        Uses a greedy heuristic: map virtual qubits with the most
interactions to the most central physical qubits, then sum
``weight * shortest_path_distance`` for every interacting pair.
Lower is better.

        Args:
            interaction_graph: Weighted graph of virtual-qubit interactions.
            subgraph: Candidate hardware subgraph for layout mapping.

        Returns:
            ``float`` estimated routing cost (lower is better).
        """
        ig = interaction_graph
        if ig.number_of_edges() == 0:
            return 0.0
        # Rank virtual qubits by total interaction weight (descending).
        v_qubits = sorted(
            ig.nodes(),
            key=lambda n: sum(d.get("weight", 1) for _, _, d in ig.edges(n, data=True)),
            reverse=True,
        )
        # Rank physical qubits by degree centrality (descending).
        p_qubits = sorted(
            subgraph.nodes(),
            key=lambda n: subgraph.degree(n),
            reverse=True,
        )
        # Greedy mapping: highest-interaction virtual → highest-degree physical.
        v_to_p = {}
        used = set()
        for vq in v_qubits:
            for pq in p_qubits:
                if pq not in used:
                    v_to_p[vq] = pq
                    used.add(pq)
                    break
        # Pre-compute shortest path lengths in the subgraph.
        sp = dict(nx.all_pairs_shortest_path_length(subgraph))
        total = 0.0
        for u, v, data in ig.edges(data=True):
            w = data.get("weight", 1)
            pu, pv = v_to_p.get(u), v_to_p.get(v)
            if pu is not None and pv is not None:
                dist = sp.get(pu, {}).get(pv, len(subgraph))
                total += w * dist
        return total

    def _get_node_neighbours(self, node: int):
        """Return the list of neighboring physical qubits for a node in the hardware graph.

        Args:
            node (*int*): Physical qubit index.

        Returns:
            List of neighboring physical qubit indices.
        """
        return list(self.graph.neighbors(node))

    def _get_node_connect_dict(self, node: int, nqubits: int):
        """Build a forward-reachable neighbor dictionary from a starting node up to *nqubits* hops deep.

        Args:
            node (*int*): Starting physical qubit index.
            nqubits (*int*): Maximum depth (number of qubits in target subgraph).

        Returns:
            ``dict`` mapping each visited node to its forward neighbors.
        """
        current_neighbours = [i for i in self._get_node_neighbours(node) if i > node]
        dd = {node: current_neighbours}
        remove = list(range(node + 1))
        for _ in range(nqubits - 2):
            current = []
            for node0 in current_neighbours:
                node0_neighbours = self._get_node_neighbours(node0)
                node0_neighbours = [i for i in node0_neighbours if i not in remove]
                current.append(node0_neighbours)
                dd[node0] = node0_neighbours
            current_neighbours = list(set(item for sublist in current for item in sublist))
        return dd

    def get_one_node_subgraph(self, node: int, nqubits: int):
        """Enumerate all connected subgraphs of size *nqubits* containing the given start node.

        Args:
            node (*int*): Starting physical qubit index.
            nqubits (*int*): Required subgraph size.

        Returns:
            List of tuples, each a sorted set of physical qubit indices forming a connected subgraph.
        """
        def post_combinations(mid, dd, cut):
            """Generate neighbor combinations up to *cut* elements for subgraph expansion.

            Args:
                mid: Current set of nodes in the subgraph.
                dd: Forward-neighbor dictionary.
                cut: Maximum number of additional nodes to add.

            Returns:
                List of candidate node lists for expansion.
            """
            rr = set([elem for node in mid if node in dd for elem in dd[node]])
            cc = []
            mm = min(cut, len(dd)) + 1
            for idx in range(1, mm):
                cc += [list(comb) for comb in combinations(rr, idx)]
            return cc

        dd = self._get_node_connect_dict(node, nqubits)
        collect = []
        init = [{"pre": [], "mid": [node], "post": post_combinations([node], dd, nqubits - 1)}]
        for _ in range(nqubits):
            update = []
            for c0 in init:
                new_pre = c0["pre"] + c0["mid"]
                new_pre.sort()
                new_pre = list(set(new_pre))
                if len(new_pre) == nqubits:
                    new_pre.sort()
                    collect.append(tuple(new_pre))
                elif len(new_pre) < nqubits:
                    if c0["post"] == []:
                        continue
                    else:
                        for mid0 in c0["post"]:
                            mid = [i for i in mid0 if i not in new_pre]
                            c1 = {
                                "pre": new_pre,
                                "mid": mid,
                                "post": post_combinations(mid, dd, nqubits - len(new_pre + mid)),
                            }
                            update.append(c1)
            init = update
        return list(set(collect))

    def collect_all_subgraph_in_parallel(self, nqubits):
        """Enumerate connected subgraphs of size *nqubits* from every node using multiprocessing.

        Args:
            nqubits (*int*): Number of qubits.

        Returns:
            List of subgraph tuples (sorted qubit index tuples).
        """
        collect_all = []
        try:
            with Pool(processes=self.ncore) as pool:
                res = pool.map(partial(self.get_one_node_subgraph, nqubits=nqubits), self.graph.nodes())
        except Exception:
            res = [self.get_one_node_subgraph(node, nqubits) for node in self.graph.nodes()]
        for collect in res:
            collect_all += collect
        return collect_all

    def get_one_subgraph_info(self, nodes: tuple | list):
        """Compute edge-fidelity statistics for a subgraph and return its info if mean fidelity exceeds the threshold.

        Args:
            nodes (*tuple | list*): Physical qubit indices forming the subgraph.

        Returns:
            ``(nodes, degree_dict, fidelity_mean, fidelity_var)`` tuple, or ``None`` if below threshold.
        """
        subgraph = self.graph.subgraph(nodes)
        subgraph_degree = dict(subgraph.degree())
        subgraph_fidelity = np.array([self.edge_fidelitys[(min(edge), max(edge))] for edge in subgraph.edges])
        fidelity_mean = np.mean(subgraph_fidelity)
        fidelity_var = np.var(subgraph_fidelity)
        if fidelity_mean >= self.fidelity_mean_threshold:
            nodes_info = (nodes, subgraph_degree, fidelity_mean, fidelity_var)
            return nodes_info
        return None

    def collect_all_subgraph_info_in_parallel(self, nqubits: int):
        """Compute fidelity statistics for all enumerated subgraphs using multiprocessing.

        Args:
            nqubits (*int*): Number of qubits.

        Returns:
            List of subgraph info tuples (or ``None`` entries for below-threshold subgraphs).
        """
        all_subgraph = self.collect_all_subgraph_in_parallel(nqubits)
        try:
            with Pool(processes=self.ncore) as pool:
                res = pool.map(partial(self.get_one_subgraph_info), all_subgraph)
        except Exception:
            res = [self.get_one_subgraph_info(sg) for sg in all_subgraph]
        return res

    def classify_all_subgraph_according_topology(self, nqubits: int):
        """Classify qualifying subgraphs into linear (chain) and nonlinear topology groups.

        Args:
            nqubits (*int*): Number of qubits.

        Returns:
            ``(linear_list, nonlinear_list)`` two lists of ``(nodes, fidelity_mean, fidelity_var)`` tuples.
        """
        linear_subgraph_list = []
        nonlinear_subgraph_list = []
        all_subgraph_info = self.collect_all_subgraph_info_in_parallel(nqubits)

        for subgraph_info in filter(lambda x: x is not None, all_subgraph_info):
            nodes, subgraph_degree, fidelity_mean, fidelity_var = subgraph_info
            nodes_info = (nodes, fidelity_mean, fidelity_var)
            if max(subgraph_degree.values()) <= 2:
                linear_subgraph_list.append(nodes_info)
            else:
                nonlinear_subgraph_list.append(nodes_info)
        return linear_subgraph_list, nonlinear_subgraph_list

    def sort_subgraph_according_mean_fidelity(self, nqubits: int, num: int = 1, printdetails: bool = True):
        """Rank subgraph layouts by mean edge fidelity (descending) and return the top *num* candidates.

        Args:
            nqubits (*int*): Number of qubits.
            num (*int*): Maximum number of candidate layouts to return. Defaults to ``1``.
            printdetails (*bool*): Whether to print detailed progress. Defaults to ``True``.

        Returns:
            ``(linear_sorted[:num], nonlinear_sorted[:num])`` — top candidates by mean fidelity.
        """
        linear_subgraph_list, nonlinear_subgraph_list = self.classify_all_subgraph_according_topology(nqubits)
        linear_subgraph_list_sort = sorted(linear_subgraph_list, key=lambda x: x[1], reverse=True)
        nonlinear_subgraph_list_sort = sorted(nonlinear_subgraph_list, key=lambda x: x[1], reverse=True)
        if printdetails:
            logger.debug("%d linear, %d nonlinear subgraphs", len(linear_subgraph_list_sort), len(nonlinear_subgraph_list_sort))
            logger.info("The average fidelity is arranged in descending order,only print the first ten.")
            length = nqubits * 5 + 22

            logger.info(
                "{:<3} | {:^{}} | {:^{}} ".format(
                    "idx",
                    "subgraph with linear topology",
                    length,
                    "subgraph with nonlinear topology",
                    length,
                )
            )
            for i, (linear, nonlinear) in enumerate(
                zip_longest(linear_subgraph_list_sort, nonlinear_subgraph_list_sort, fillvalue=" ")
            ):
                if i >= len(linear_subgraph_list_sort):
                    linear = ("(                  )", 0.0, 0.0)
                if i >= len(nonlinear_subgraph_list_sort):
                    nonlinear = ("(                  )", 0.0, 0.0)
                if i <= num:
                    logger.info(
                        "{:<3} | {:<{}} {:<10.6f} {:<10.6f} | {:<{}} {:<10.6f} {:<10.6f} ".format(
                            i,
                            str(linear[0]),
                            nqubits * 5,
                            linear[1],
                            linear[2],
                            str(nonlinear[0]),
                            nqubits * 5,
                            nonlinear[1],
                            nonlinear[2],
                        )
                    )

        return linear_subgraph_list_sort[:num], nonlinear_subgraph_list_sort[:num]

    def sort_subgraph_according_var_fidelity(self, nqubits: int, num: int = 1, printdetails: bool = True):
        """Rank subgraph layouts by fidelity variance (ascending, most uniform first) and return the top *num* candidates.

        Args:
            nqubits (*int*): Number of qubits.
            num (*int*): Maximum number of candidate layouts to return. Defaults to ``1``.
            printdetails (*bool*): Whether to print detailed progress. Defaults to ``True``.

        Returns:
            ``(linear_sorted[:num], nonlinear_sorted[:num])`` — top candidates by lowest variance.
        """
        linear_subgraph_list, nonlinear_subgraph_list = self.classify_all_subgraph_according_topology(nqubits)
        linear_subgraph_list_sort = sorted(linear_subgraph_list, key=lambda x: x[2])
        nonlinear_subgraph_list_sort = sorted(nonlinear_subgraph_list, key=lambda x: x[2])

        if printdetails:
            logger.debug("%d linear, %d nonlinear subgraphs", len(linear_subgraph_list_sort), len(nonlinear_subgraph_list_sort))
            logger.info("The fidelity variance is arranged in ascending order, only print the first ten.")
            length = nqubits * 5 + 22

            logger.info(
                "{:<3} | {:^{}} | {:^{}} ".format(
                    "idx",
                    "subgraph with linear topology",
                    length,
                    "subgraph with nonlinear topology",
                    length,
                )
            )
            for i, (linear, nonlinear) in enumerate(
                zip_longest(linear_subgraph_list_sort, nonlinear_subgraph_list_sort, fillvalue=" ")
            ):
                if i >= len(linear_subgraph_list_sort):
                    linear = ("(                  )", 0.0, 0.0)
                if i >= len(nonlinear_subgraph_list_sort):
                    nonlinear = ("(                  )", 0.0, 0.0)

                if i <= num:
                    logger.info(
                        "{:<3} | {:<{}} {:<10.6f} {:<10.6f} | {:<{}} {:<10.6f} {:<10.6f} ".format(
                            i,
                            str(linear[0]),
                            nqubits * 5,
                            linear[1],
                            linear[2],
                            str(nonlinear[0]),
                            nqubits * 5,
                            nonlinear[1],
                            nonlinear[2],
                        )
                    )

        return linear_subgraph_list_sort[:num], nonlinear_subgraph_list_sort[:num]

    def select_one_qubit_from_backend(self):
        """Select the highest-fidelity qubit from the backend.

        Returns:
            List containing the selected qubit index.
        """
        for nodes in nx.connected_components(self.graph):
            if len(nodes) > 1:
                subgraph = self.graph.subgraph(nodes)
                break
        node_fidelity_dic = nx.get_node_attributes(subgraph, "fidelity")
        sorted_dict = dict(sorted(node_fidelity_dic.items(), key=lambda item: item[1], reverse=True))
        qubit = [list(sorted_dict.keys())[0]]
        return qubit

    def select_few_qubits_from_backend(
        self,
        nqubits: int,
        key: Literal["fidelity_mean", "fidelity_var"] = "fidelity_var",
        topology: Literal["linear", "nonlinear"] = "linear",
        printdetails: bool = False,
        interaction_graph=None,
    ):
        """Select the best physical qubit layout for a small circuit by fidelity ranking and optional routing-cost re-scoring.

        Args:
            nqubits (*int*): Number of qubits.
            key (*Literal['fidelity_mean', 'fidelity_var']*): Lookup key. Defaults to ``'fidelity_var'``.
            topology (*Literal['linear', 'nonlinear']*): Subgraph topology preference. Defaults to ``'linear'``.
            printdetails (*bool*): Whether to print detailed progress. Defaults to ``False``.
            interaction_graph: Circuit interaction graph for routing-cost re-ranking. Defaults to ``None``.

        Returns:
            List of selected physical qubit indices.

        Raises:
            ValueError: If no subgraph with the required number of qubits can be found.
        """
        # When circuit-aware, collect more candidates for re-ranking.
        num = 10 if interaction_graph is not None and interaction_graph.number_of_edges() > 0 else 1
        if key == "fidelity_mean":
            linear_list, nonlinear_list = self.sort_subgraph_according_mean_fidelity(
                nqubits, num=num, printdetails=printdetails
            )
        elif key == "fidelity_var":
            linear_list, nonlinear_list = self.sort_subgraph_according_var_fidelity(
                nqubits, num=num, printdetails=printdetails
            )

        if topology == "linear":
            layouts = linear_list
        elif topology == "nonlinear":
            layouts = nonlinear_list

        if len(layouts) == 0:
            raise ValueError(
                f"There is no {nqubits} qubits that meets both key = {key} and topology = {topology}. Please change the conditions."
            )

        # Circuit-aware re-ranking: when an interaction graph is provided,
        # re-score the top-K candidates by combining fidelity with estimated
        # routing cost so that the chosen layout minimises expected SWAPs.
        if interaction_graph is not None and interaction_graph.number_of_edges() > 0 and len(layouts) > 1:
            scored = []
            for nodes, fid_mean, fid_var in layouts:
                sg = self.graph.subgraph(nodes)
                cost = self._estimate_routing_cost(interaction_graph, sg)
                # Normalise routing cost by total interaction weight so it
                # stays in a comparable range to fidelity values.
                total_weight = sum(d.get("weight", 1) for _, _, d in interaction_graph.edges(data=True))
                norm_cost = cost / max(total_weight, 1)
                # Combined score: higher fidelity is better (maximise),
                # lower routing cost is better (minimise).
                score = fid_mean - 0.05 * norm_cost
                scored.append((nodes, fid_mean, fid_var, score))
            scored.sort(key=lambda x: x[3], reverse=True)
            return list(scored[0][0])

        return list(layouts[0][0])

    def _get_largest_component(self):
        """Return the largest connected component of the hardware graph.

        Returns:
            ``networkx.Graph`` subgraph of the largest component.
        """
        components = list(nx.connected_components(self.graph))
        len_comp = [len(comp) for comp in components]
        idx = len_comp.index(max(len_comp))
        return self.graph.subgraph(components[idx])

    def select_much_qubits_from_backend(self, nqubits):
        """Select physical qubits for a large circuit via BFS expansion on the largest connected component.

        Args:
            nqubits (*int*): Number of qubits.

        Returns:
            List of selected physical qubit indices.

        Raises:
            ValueError: If the circuit requires more qubits than the largest connected subgraph.
        """
        one_subgraph = self._get_largest_component()
        if len(one_subgraph.nodes()) < nqubits:
            raise ValueError(
                f"The user circuit requires {nqubits} qubits exceeds the qubit capacity of the largest connected subgraph."
            )
        start_node = np.random.choice(list(one_subgraph.nodes))

        visited = set([start_node])
        queue = [(start_node, 0)]
        while queue and len(visited) < nqubits:
            current_node, depth = queue.pop(0)
            if depth >= nqubits - 1:
                continue
            for neighbor in one_subgraph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    if len(visited) == nqubits:
                        break
        return list(visited)

    def select_qubits_by_local_algorithm(self, nqubits, select_criteria, interaction_graph=None):
        """Select physical qubits using size-adaptive heuristics: single-qubit, small-enumeration, or BFS-based strategies.

        Args:
            nqubits (*int*): Number of qubits.
            select_criteria (*dict*): Selection criteria dict with keys ``'key'`` and ``'topology'``.
            interaction_graph (*nx.Graph | None*): Circuit interaction graph for routing-cost re-ranking. Defaults to ``None``.

        Returns:
            List of selected physical qubit indices.

        Raises:
            ValueError: Wrong qubits error!
        """
        if nqubits == 1:
            qubit = self.select_one_qubit_from_backend()
            return qubit
        if 1 < nqubits <= self.algorithm_switch_threshold:
            key_first = select_criteria["key"]
            topology_first = select_criteria["topology"]
            all_keys = ["fidelity_var", "fidelity_mean"]
            all_topologys = ["linear", "nonlinear"]
            all_keys.remove(key_first)
            all_topologys.remove(topology_first)
            sorted_keys = [key_first] + all_keys
            sorted_topologys = [topology_first] + all_topologys
            physical_qubits_layout = []
            for key, topology in product(sorted_keys, sorted_topologys):
                try:
                    physical_qubits_layout = self.select_few_qubits_from_backend(
                        nqubits, key=key, topology=topology, interaction_graph=interaction_graph,
                    )
                except Exception as e:
                    physical_qubits_layout = []
                    logger.warning("Layout selection failed: %s", e)
                if physical_qubits_layout:
                    break
                    break
            if physical_qubits_layout == []:
                raise ValueError("Unable to find a suitable layout.")
            return physical_qubits_layout
        if nqubits > self.algorithm_switch_threshold:
            physical_qubits_layout = self.select_much_qubits_from_backend(nqubits)
            return physical_qubits_layout
        raise ValueError("Wrong qubits error!")

    def select_layout(
        self,
        qc: QuantumCircuit,
        target_qubits: list = [],
        use_chip_priority: bool = True,
        select_criteria: dict = {"key": "fidelity_var", "topology": "linear"},
        skip_split_qc: bool = True,
    ):
        """Select a hardware qubit layout for the circuit, using priority lists, target qubits, or fidelity-based algorithmic search.

        Args:
            qc (*QuantumCircuit*): Quantum circuit.
            target_qubits (*list*): Physical qubit indices to constrain the layout to. Defaults to ``[]``.
            use_chip_priority (*bool*): Whether to prefer the backend's pre-ranked priority qubit list. Defaults to ``True``.
            select_criteria (*dict*): Selection strategy with keys ``'key'`` and ``'topology'``. Defaults to ``{'key': 'fidelity_var', 'topology': 'linear'}``.
            skip_split_qc (*bool*): If ``True``, treat the circuit as a single block. Defaults to ``True``.

        Returns:
            ``networkx.Graph`` subgraph representing the selected hardware layout.

        Raises:
            ValueError: If *target_qubits* length doesn't match the circuit's qubit count.
        """
        nqubits = len(qc.qubits)
        if skip_split_qc:
            all_qubits = [qc.qubits]
        else:
            all_qubits = split_qubits(qc)
        self.source_graph = copy.deepcopy(self.graph)

        # Extract interaction graph for circuit-aware layout selection.
        interaction_graph = self._extract_interaction_graph(qc)

        if target_qubits != []:
            if len(set(target_qubits)) != nqubits:
                raise ValueError(
                    f"The number of qubits {len(target_qubits)} in target_qubits does not match the number of qubits {nqubits} in the circuit."
                )
            lose_nodes = []
            for qubit in target_qubits:
                if not self.graph.has_node(qubit):
                    lose_nodes.append(qubit)
            if lose_nodes != []:
                raise ValueError(
                    f"These qubit(s) {lose_nodes} does not exist.This maybe due to an incorrected input index or low fidelity filtering."
                )
            idx = 0
            for qubits0 in all_qubits:
                part_target_qubits = target_qubits[idx : idx + len(qubits0)]
                idx += len(qubits0)
                if len(part_target_qubits) > 1:
                    subgraph0 = self.graph.subgraph(part_target_qubits).copy()
                    if not nx.is_connected(subgraph0):
                        raise ValueError(
                            f"The target physical qubits {part_target_qubits} corresponding to virtual qubits {qubits0} are not connected."
                        )
            subgraph = self.graph.subgraph(target_qubits).copy()
            subgraph.graph["normal_order"] = target_qubits
            if len(subgraph.edges()) > 0:
                subgraph_fidelity = np.array([self.edge_fidelitys[(min(edge), max(edge))] for edge in subgraph.edges])
                fidelity_mean = np.mean(subgraph_fidelity)
                fidelity_var = np.var(subgraph_fidelity)
            return subgraph

        if use_chip_priority:
            priority_qubits_list = self.priority_qubits
            new_qubits = []
            for qubits0 in all_qubits:
                is_priority_provided = False
                priority_qubits_list = [q for q in priority_qubits_list if q not in new_qubits]
                for qubits in priority_qubits_list:
                    if len(qubits0) == len(qubits):
                        subgraph0 = self.source_graph.subgraph(qubits).copy()
                        is_overlap = any(x in qubits for sub in new_qubits for x in sub)
                        if nx.is_connected(subgraph0) and not is_overlap:
                            new_qubits.append(qubits)
                            is_priority_provided = True
                        break
                    continue
                if not is_priority_provided:
                    self.graph.remove_nodes_from([x for sub in new_qubits for x in sub])
                    qubits = self.select_qubits_by_local_algorithm(len(qubits0), select_criteria, interaction_graph)
                    new_qubits.append(qubits)
            subgraph = self.source_graph.subgraph([x for sub in new_qubits for x in sub])
            subgraph.graph["normal_order"] = [x for sub in new_qubits for x in sub]
            return subgraph

        new_qubits = []
        for qubits0 in all_qubits:
            self.graph.remove_nodes_from(list(set(e for sub in new_qubits for e in sub)))
            try:
                qubits = self.select_qubits_by_local_algorithm(len(qubits0), select_criteria, interaction_graph)
                new_qubits.append(qubits)
            except Exception as e:
                raise ValueError(f"Local algorithm search layout Faild {e}")
        subgraph = self.source_graph.subgraph([x for sub in new_qubits for x in sub])
        subgraph.graph["normal_order"] = [x for sub in new_qubits for x in sub]
        return subgraph
