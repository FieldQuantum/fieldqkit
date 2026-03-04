r"""
This module contains the Backend class, which processes superconducting chip information into an undirected graph representation. It also supports the creation of custom undirected graphs to serve as virtual chips.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import requests
import json
from pathlib import Path


def _build_simulator_chip_info(nqubits: int = 12) -> dict:
    qubits_info = {
        f"Q{i}": {
            "fidelity": 1.0,
            "coordinate": [float(i), 0.0],
            "T1": 0.0,
            "T2": 0.0,
            "frequency": 0.0,
        }
        for i in range(nqubits)
    }
    couplers_info = {}
    idx = 0
    for i in range(nqubits-1):
        couplers_info[f"C{idx}"] = {
            "qubits_index": [i, i+1],
            "fidelity": 1.0,
            "index": idx,
        }
        idx += 1
    global_info = {
        "two_qubit_gate_basis": "cz",
        "nqubits_available": nqubits,
        "error_rate_2q": 0.0,
        "one_qubit_gate_length": 1.0,
        "two_qubit_gate_length": 1.0,
    }
    return {
        "size": (nqubits, 1),
        "priority_qubits": [list(range(nqubits))],
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": global_info,
        "calibration_time": "simulator",
    }

def load_chip_basic_info(chip_name):
    # Pull chip metadata from the Quafu backend service.
    session = requests.Session()
    URL = 'https://quafu-sqc.baqis.ac.cn'
    info0 = session.get(f'{URL}/task/backendtest/{chip_name}1') 
    chip_info = json.loads(info0.content.decode())
    if chip_info:
        print(f'{chip_name} configuration loading done!')
        return chip_info
    else:
        print(f'{chip_name} configuration loading failed!')
        return None


class Backend:
    """A class to represent a quantum hardware backend as a nx.Graph.
    """
    def __init__(self,chip: Literal['Baihua','Dongling','Haituo','Yunmeng','Miaofeng','Yudu','Hongluo','Simulator','Custom'] | dict):
        """Initialize a Backend object.

        Args:
            chip (str): Chip name, currently 'Baihua','Dongling','Haituo','Yunmeng','Miaofeng','Yudu','Hongluo', and 'Simulator' are supported
        """
        if isinstance(chip,dict):
            # Custom chip payload (useful for tests or local topology overrides).
            self.chip_name = ' '
            self.chip_info = chip
            self.size = self.chip_info['size']
            self.priority_qubits = self.chip_info['priority_qubits']
            self.qubits_with_attributes = self._collect_qubits_with_attributes()
            self.couplers_with_attributes = self._collect_couplers_with_attributes()
            self.two_qubit_gate_basis = self.chip_info['global_info']['two_qubit_gate_basis'].lower()
        elif chip in ['Baihua','Dongling','Haituo','Yunmeng','Miaofeng','Yudu','Hongluo']:
            # Live hardware chip configuration fetched from Quafu services.
            self.chip_name = chip
            try:
                self.chip_info = load_chip_basic_info(chip)
            except:
                raise(ValueError(f'{chip} is under maintenance, configuration information is unavailable'))
            print('The last calibration time was',self.chip_info['calibration_time'])
            self.size = self.chip_info['size']
            self.priority_qubits = self.chip_info['priority_qubits']
            self.qubits_with_attributes = self._collect_qubits_with_attributes()
            self.couplers_with_attributes = self._collect_couplers_with_attributes()
            self.two_qubit_gate_basis = self.chip_info['global_info']['two_qubit_gate_basis'].lower()
            
        elif chip == 'Custom':
            self.chip_name = chip
            self.chip_info = dict()
            self.size = (0,0)
            self.qubits_with_attributes = list()
            self.couplers_with_attributes = list()
            self.priority_qubits = []
            self.two_qubit_gate_basis = 'cz'
        elif chip in ['Simulator', 'simulator']:
            # Built-in simulator chip (fully connected).
            self.chip_name = 'Simulator'
            self.chip_info = _build_simulator_chip_info()
            self.size = self.chip_info['size']
            self.priority_qubits = self.chip_info['priority_qubits']
            self.qubits_with_attributes = self._collect_qubits_with_attributes()
            self.couplers_with_attributes = self._collect_couplers_with_attributes()
            self.two_qubit_gate_basis = self.chip_info['global_info']['two_qubit_gate_basis'].lower()
        else:
            raise(ValueError(f'Wrong chip name! {chip}'))
    
    @property
    def graph(self):
        """Returns the graph representation of the object.
        
        This property method calls `self.get_graph()` to generate and return the graph with nodes and edges.

        Returns:
            networkx.Graph: The graph with nodes and weighted edges.
        """
        return self.get_graph()
    
    def edge_filtered_graph(self,thres = 0.6):
        """Create a subgraph by filtering out edges with fidelity below a specified threshold.

        Args:
            thres (float, optional): The fidelity threshold. Defaults to 0.6.

        Returns:
            networkx.Graph: A new NetworkX graph object containing only edges with fidelity greater than or equal to the threshold.
        """
        def edge_filter(u,v):
            return self.graph[u][v].get("fidelity") >= thres
        def node_filter(n):
            return self.graph.nodes[n].get("fidelity") >= thres
        subgraph_view = nx.subgraph_view(self.graph,filter_node=node_filter,filter_edge=edge_filter)
        return nx.Graph(subgraph_view)
    
    def _collect_qubits_with_attributes(self):
        """Collect qubit indices and their associated attributes from chip information.

        Returns:
            list: A list of tuples, where each tuple contains a qubit index (int) and its attributes (dict)
              extracted from self.chip_info['qubits_info']
        """
        qubits_with_attributes = []
        for key in self.chip_info['qubits_info'].keys():
            qubit = int(key.split('Q')[1])
            qubits_with_attributes.append((qubit, self.chip_info['qubits_info'][key]))
        return qubits_with_attributes
    
    def _collect_couplers_with_attributes(self):
        """Collect coupler information including qubit indices and their associated attributes from chip information.

        Returns:
            list: A list of tuples, where each tuple contains two qubit indices (int, int) and the coupler attributes (dict)
              extracted from self.chip_info['couplers_info'].
        """
        couplers_with_attributes = []
        for key in self.chip_info['couplers_info'].keys():
            qubit1, qubit2 = self.chip_info['couplers_info'][key]['qubits_index']
            couplers_with_attributes.append((qubit1,qubit2,self.chip_info['couplers_info'][key]))
        return couplers_with_attributes
    
    def get_graph(self):
        """Constructs and returns an undirected graph with nodes and weighted edges.

        Returns:
            networkx.Graph: An undirected graph with nodes and weighted edges.
        """
        G = nx.Graph()
        G.add_nodes_from(self.qubits_with_attributes)
        G.add_edges_from(self.couplers_with_attributes)
        return G
        
    def draw(self, save_svg_fname: str|None = None, edge_fidelity_thres=0.9):
        """Draw the chip layout using a fixed standard style.

        Args:
            save_svg_fname (str | None, optional):
                The filename for saving the drawing as a svg. If None, the drawing will not be saved.
                Defaults to None.
            edge_fidelity_thres (float, optional):
                Minimum 2-qubit fidelity required to keep an edge in the visualization.
                Defaults to 0.9.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        graph_show = self.edge_filtered_graph(thres=edge_fidelity_thres)
        pos = nx.get_node_attributes(graph_show, 'coordinate')
        node_colors = ['#083776' for _ in graph_show.nodes()]
        edge_colors = ['#083776' for _ in graph_show.edges()]
        node_labels = {node: node for node in graph_show.nodes()}
        edge_labels = {edge: '' for edge in graph_show.edges()}

        fig, ax = plt.subplots(figsize=(15, 13))
        nx.draw(
            graph_show,
            pos,
            ax=ax,
            with_labels=False,
            node_color=node_colors,
            node_size=800,
            edgecolors='white',
            edge_color=edge_colors,
            width=18,
        )
        nx.draw_networkx_labels(graph_show, pos, labels=node_labels, font_size=10, font_color='white')
        nx.draw_networkx_edge_labels(
            graph_show,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='white',
            bbox=dict(facecolor='none', edgecolor='none'),
            rotate=False,
        )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xpos = (xlim[0] + xlim[1]) / 2
        ypos = ylim[0] - (ylim[1] - ylim[0]) * -0.03
        ax.text(xpos, ypos, f'{self.chip_name}', va='center', ha='center', fontsize=24, fontweight='bold', color='k', family='serif')

        ax.invert_yaxis()

        if save_svg_fname:
            plt.savefig(save_svg_fname + '.svg', bbox_inches='tight')
        plt.clf()
        plt.close()
        return None
    

def _cache_chip_topology_figure(backend: Backend, chip_name: str) -> None:
    """Best-effort topology rendering for local cache."""
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cache_dir / f"{chip_name}_chip"
    backend.draw(
        save_svg_fname=str(fig_path),
        edge_fidelity_thres=0.9,
    )


def get_available_chip_status(tmgr) -> Dict[str, int]:
    """Fetch chip queue status from task manager."""
    status = tmgr.status()
    if not isinstance(status, dict):
        raise RuntimeError("tmgr.status() must return a dict of chip -> queue length")
    return {k: v for k, v in status.items() if isinstance(v, int)}


def get_chip_info(chip_name: str) -> Dict[str, Union[int, float]]:
    """Get chip metadata and optionally cache a topology drawing."""
    try:
        backend = Backend(chip_name)
        info = backend.chip_info
        _cache_chip_topology_figure(backend, chip_name)
        return info
    except Exception:
        return {}


def rank_chips(
    tmgr,
    *,
    num_qubits: int,
    prefer_chips: Optional[Sequence[str] | str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Rank chips by queue length, size, and error rate with weights."""
    if isinstance(prefer_chips, str):
        prefer_chips = [prefer_chips]
    if prefer_chips is not None:
        prefer_lower = {c.lower() for c in prefer_chips}
        if "simulator" in prefer_lower:
            return ["Simulator"] if num_qubits <= 12 else []

    status = get_available_chip_status(tmgr)
    if prefer_chips is not None:
        prefer_set = {c for c in prefer_chips}
        status = {k: v for k, v in status.items() if k in prefer_set}

    ranked: List[Tuple[str, int, int, float]] = []
    for chip_name, queue_len in status.items():
        info = get_chip_info(chip_name)
        global_info = info.get("global_info", {}) if isinstance(info, dict) else {}
        nqubits = int(global_info.get("nqubits_available", info.get("nqubits_available", 0)) or 0)
        if nqubits < num_qubits:
            continue
        error_rate_2q = float(global_info.get("error_rate_2q", info.get("error_rate_2q", float("inf"))))
        ranked.append((chip_name, queue_len, nqubits, error_rate_2q))

    if not ranked:
        return []

    if weights is None:
        weights = {"queue": 0.2, "nqubits": 0.3, "error": 0.5}
    w_queue = float(weights.get("queue", 0.2))
    w_nqubits = float(weights.get("nqubits", 0.3))
    w_error = float(weights.get("error", 0.5))

    queues = [r[1] for r in ranked]
    nqubits_list = [r[2] for r in ranked]
    errors = [r[3] for r in ranked]

    def _normalize(values: List[float]) -> List[float]:
        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    q_norm = _normalize([float(v) for v in queues])
    n_norm = _normalize([float(v) for v in nqubits_list])
    e_norm = _normalize([float(v) for v in errors])

    scored: List[Tuple[str, float]] = []
    for (chip_name, _, _, _), qn, nn, en in zip(ranked, q_norm, n_norm, e_norm):
        score = w_queue * qn + w_nqubits * (1.0 - nn) + w_error * en
        scored.append((chip_name, score))

    scored.sort(key=lambda x: x[1])
    return [name for name, _ in scored]