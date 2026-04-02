"""Tools for converting between directed acyclic graphs (DAGs) and quantum circuits.

SPDX-License-Identifier: MIT
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    three_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    functional_gates_available,
    convert_gate_info_to_dag_info,
)


def draw_dag(dag, output="dag_figure.png"):
    """Draws a directed acyclic graph (DAG) representation of a quantum circuit and saves it as an image.

    Args:
        dag (*nx.DiGraph*): Directed acyclic graph representation of the circuit.
        output (*str*): File path for the saved image. Defaults to ``'dag_figure.png'``.
    """
    import matplotlib.pyplot as plt

    A = nx.nx_agraph.to_agraph(dag)

    for node in A.nodes():
        gate = node.split("_")[0]
        if gate == "measure":
            cbit = dag.nodes[node]["cbits"][0]
            node.attr["label"] = gate + f" [c{cbit}]"
        else:
            node.attr["label"] = gate
    for u, v, data in dag.edges(data=True):
        edge = A.get_edge(u, v)
        line = ""
        for qubit in data["qubit"]:
            line += "q" + str(qubit)
        edge.attr["label"] = line

    A.graph_attr["dpi"] = "300"
    A.layout(prog="dot")

    A.draw(output)

    img = plt.imread(output)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def qc2dag(qc: QuantumCircuit, show_qubits: bool = True) -> nx.DiGraph:
    """Converts a quantum circuit into a directed acyclic graph (DAG).

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        show_qubits (*bool*): Whether to include ``start``/``end`` sentinel nodes per qubit. Defaults to ``True``.

    Returns:
        ``nx.DiGraph`` representing the circuit's gate dependency graph.
    """
    node_list, edge_list = convert_gate_info_to_dag_info(qc.nqubits, qc.qubits, qc.gates, show_qubits=show_qubits)
    dag = nx.DiGraph()
    dag.add_nodes_from(node_list)
    dag.add_edges_from(edge_list)
    dag.graph["qubits"] = qc.qubits
    return dag


def dag2qc(dag: nx.DiGraph, nqubits: int | None = None, ncbits: int | None = None) -> QuantumCircuit:
    """Converts a directed acyclic graph (DAG) back into a QuantumCircuit.

    Args:
        dag (*nx.DiGraph*): Directed acyclic graph representation of the circuit.
        nqubits (*int | None*): Number of qubits. Defaults to ``None``.
        ncbits (*int | None*): Number of classical bits. Defaults to ``None``.

    Returns:
        ``QuantumCircuit`` reconstructed from the topologically sorted DAG nodes.
    """
    current_qubits = []
    new = []
    for node in nx.topological_sort(dag):
        gate = node.split("_")[0]
        qubits = dag.nodes[node]["qubits"]
        current_qubits += qubits
        if gate in one_qubit_gates_available.keys():
            new.append((gate, qubits[0]))
        elif gate in two_qubit_gates_available.keys():
            new.append((gate, qubits[0], qubits[1]))
        elif gate in three_qubit_gates_available.keys():
            new.append((gate, qubits[0], qubits[1], qubits[2]))
        elif gate in one_qubit_parameter_gates_available.keys():
            params = dag.nodes[node]["params"]
            new.append((gate, *params, qubits[0]))
        elif gate in two_qubit_parameter_gates_available.keys():
            params = dag.nodes[node]["params"]
            new.append((gate, *params, qubits[0], qubits[1]))
        elif gate in functional_gates_available.keys():
            if gate == "measure":
                cbits = dag.nodes[node]["cbits"]
                new.append((gate, qubits, cbits))
            elif gate == "barrier":
                new.append((gate, tuple(qubits)))
            elif gate == "delay":
                duration = dag.nodes[node]["duration"]
                new.append((gate, duration, tuple(qubits)))
            elif gate == "reset":
                new.append((gate, qubits[0]))
    if nqubits is None:
        nqubits = max(current_qubits) + 1
    if ncbits is None:
        ncbits = nqubits
    qc = QuantumCircuit(nqubits, ncbits)
    qc.gates = new
    qc.qubits = dag.graph["qubits"]
    return qc


def get_qcgraph_edges(gates):
    """Extract qubit-pair connectivity edges from multi-qubit gates in a gate list.

    Args:
        gates: List of gate info tuples.

    Returns:
        list: List of (qubit1, qubit2) edge tuples.

    Raises:
        ValueError: f'wrong gate type {gate}'
    """
    edges = []
    for gate_info in gates:
        gate = gate_info[0]
        if gate in one_qubit_gates_available.keys():
            continue
        elif gate in one_qubit_parameter_gates_available.keys():
            continue
        elif gate in two_qubit_gates_available.keys():
            edges.append(gate_info[1:])
        elif gate in two_qubit_parameter_gates_available.keys():
            edges.append(gate_info[2:])
        elif gate in three_qubit_gates_available.keys():
            edges.append(gate_info[1:3])
            edges.append(gate_info[2:4])
        elif gate in functional_gates_available.keys():
            continue
        else:
            raise ValueError(f"wrong gate type {gate}")
    return edges


def qc2graph(qc):
    """Convert a quantum circuit into an undirected qubit interaction graph.

    Args:
        qc: Quantum circuit instance.

    Returns:
        nx.Graph: Undirected graph with qubits as nodes and two-qubit gates as edges.
    """
    graph = nx.Graph()
    graph.add_nodes_from(qc.qubits)
    edges = get_qcgraph_edges(qc.gates)
    graph.add_edges_from(edges)
    return graph


def split_qubits(qc):
    """Partition the circuit's qubits into independent groups based on two-qubit gate connectivity.

    Args:
        qc: Quantum circuit instance.

    Returns:
        list[list[int]]: List of connected qubit groups.
    """
    graph = qc2graph(qc)
    all_qubits = [list(q) for q in nx.connected_components(graph)]
    return all_qubits


def draw_graph(G):
    """Render and display a networkx graph using matplotlib with a shell layout.

    Args:
        G: A networkx Graph instance.
    """
    import matplotlib.pyplot as plt

    pos = nx.shell_layout(G)
    plt.figure(figsize=(7, 6))
    nx.draw_networkx(
        G,
        pos,
        node_color="skyblue",
        node_size=1500,
        node_shape="o",
        with_labels=True,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        width=2,
        alpha=0.7,
    )
    plt.axis("off")
    plt.show()
