# Copyright (c) 2024 XX Xiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

r"""
This module contains the Backend class, which processes superconducting chip information into an undirected graph representation. It also supports the creation of custom undirected graphs to serve as virtual chips.
"""

import networkx as nx
import numpy as np
from typing import Literal
import requests
import json


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
    def __init__(self,chip: Literal['Baihua','Custom'] | dict):
        """Initialize a Backend object.

        Args:
            chip (str): Chip name, currently only 'Baihua' is avaliable.
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
        
    def draw(self,show_couplers_fidelity:bool = False, show_qubits_attributes:Literal['T1','T2','fidelity','frequency','']='', 
             show_qubits_index:bool=False,show_couplers_index:bool=False,
             highlight_nodes:list = [], save_svg_fname: str|None = None,edge_fidelity_thres=0.0):
        """Draw the chip layout.
    
        Args:
            show_couplers_fidelity (bool, optional): Whether to display the fidelity of couplers. Defaults to False.
            show_qubits_attributes (Literal['T1', 'T2', 'fidelity', 'frequancy', ''], optional): 
                Specify which qubit attribute to display. Options include:
                - 'T1': Display the T1 time of qubits (in microseconds).
                - 'T2': Display the T2 time of qubits (in microseconds).
                - 'fidelity': Display the fidelity of qubits.
                - 'frequancy': Display the frequency of qubits (in GHz).
                - '': Display no attributes.
                Defaults to ''.
            show_qubits_index (bool, optional): Whether to display index for each qubit. Defaults to False.
            show_couplers_index (bool, optional): Whether to display index for each coupler. Defaults to False.
            highlight_nodes (list, optional): A list of qubits to highlight. Defaults to an empty list [].
            save_svg_fname (str | None, optional): 
                The filename for saving the drawing as an SVG. If None, the drawing will not be saved. 
                The user is responsible for ensuring the validity of the provided file path. Defaults to None.
    
        Returns:
            None: This function does not return a value but generates and optionally saves the chip layout.
    
        Notes:
            - `highlight_nodes` specifies the qubits to be visually highlighted.
            - `show_couplers_fidelity` and `show_qubits_attributes` can be enabled simultaneously without conflicts.
            - When `save_svg_fname` is provided, the drawing will be saved as an SVG file.
        """
        import matplotlib.pyplot as plt 
        from matplotlib.colors import Normalize,LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable 

        graph_show = self.edge_filtered_graph(thres=edge_fidelity_thres)
        edge_fidelity = nx.get_edge_attributes(graph_show, 'fidelity') 
        node_attributes = nx.get_node_attributes(graph_show,show_qubits_attributes)
        if show_couplers_fidelity:
            if set(list(edge_fidelity.values())) == {0}:
                print('The two-qubit gate fidelity is N/A now.')
                is_edge_info_avaliable = False
            else:
                is_edge_info_avaliable = True
        if show_qubits_attributes:
            if set(list(node_attributes.values())) == {0}:
                print('The qubit attributes is N/A now.')
                is_node_info_avaliable = False
            else:
                is_node_info_avaliable = True            

        if show_couplers_fidelity is True and is_edge_info_avaliable is True:
            min_fidelity = sorted(list(edge_fidelity.values()))[0]
            max_fidelity = sorted(list(edge_fidelity.values()))[-1]
            edge_norm = Normalize(vmin = min_fidelity, vmax = max_fidelity)
            edge_cmap = LinearSegmentedColormap.from_list('truncated_blues', plt.get_cmap('Blues')(np.linspace(0.31, 1.0, 1000)))  #plt.get_cmap('Blues')
            edge_colors = [ScalarMappable(norm = edge_norm, cmap = edge_cmap).to_rgba(fidelity) for fidelity in edge_fidelity.values()]
        else:
            edge_colors = ['#083776' for edge in graph_show.edges()]

        edge_labels = {}
        for k,v in edge_fidelity.items():
            if v == None:
                fidelity = ''
            else:
                fidelity = np.round(v, 3)
            if show_couplers_fidelity is True and is_edge_info_avaliable is True:
                if show_couplers_index:
                    edge_labels[k] = graph_show.edges[k].get('index')
                else:
                    edge_labels[k] = fidelity
            else:
                edge_labels[k] = graph_show.edges[k].get('index')

        if show_qubits_attributes != '' and is_node_info_avaliable is True:
            min_attributes = sorted(list(node_attributes.values()))[0]
            max_attributes = sorted(list(node_attributes.values()))[-1]
            node_norm = Normalize(vmin = min_attributes, vmax = max_attributes)
            node_cmap = LinearSegmentedColormap.from_list('truncated_blues', plt.get_cmap('Blues')(np.linspace(0.31, 1.0, 1000))) #plt.get_cmap('Blues') 
            node_colors = [ScalarMappable(norm = node_norm, cmap = node_cmap).to_rgba(attribute) for attribute in node_attributes.values()]
            if show_qubits_attributes == 'fidelity':
                if show_qubits_index:
                    node_labels = {node:node for node in graph_show.nodes()} 
                else:
                    node_labels =  {node: np.round(attr, 3) for node, attr in node_attributes.items()} # 保留三位有效数字
            else:
                if show_qubits_index:
                    node_labels = {node:node for node in graph_show.nodes()} 
                else:    
                    node_labels =  {node: np.round(attr, 2) for node, attr in node_attributes.items()}  # 保留两位有效数字
            node_font_size = 8 
            if show_couplers_fidelity is True and is_edge_info_avaliable is True:
                figsize = (15, 15)
            else:
                figsize = (15, 13.5)
        else:
            node_colors = ['#083776' for node in graph_show.nodes()]
            node_labels = {node:node for node in graph_show.nodes()}   
            node_font_size = 10 
            if show_couplers_fidelity is True and is_edge_info_avaliable is True:
                figsize = (15, 13.5)
            else:
                figsize = (15, 13)

        if len(highlight_nodes) > 0:
            if isinstance(highlight_nodes, list) and any(isinstance(i, list) for i in highlight_nodes):
                highlight_nodes = list(set(e for sub in highlight_nodes for e in sub))
            node_colors_update = []
            node_labels_update = {}
            for idx, node in enumerate(graph_show.nodes()):
                if node in highlight_nodes:
                    node_colors_update.append(node_colors[idx])
                    node_labels_update[node] = node_labels[node]
                else:
                    node_colors_update.append('#f3f3f3')
                    node_labels_update[node] = ''
            node_colors = node_colors_update
            node_labels = node_labels_update

            subgraph = graph_show.subgraph(highlight_nodes)
            edge_colors_update = []
            edge_labels_update = {}
            for idx, edge in enumerate(graph_show.edges()):
                if edge in subgraph.edges():
                    edge_colors_update.append(edge_colors[idx])
                    edge_labels_update[edge] = edge_labels[edge]
                else:
                    edge_colors_update.append('#f3f3f3')
                    edge_labels_update[edge] = ''
            edge_colors = edge_colors_update
            edge_labels = edge_labels_update

        fig, ax = plt.subplots(figsize=figsize) #facecolor='none'
        pos = nx.get_node_attributes(graph_show, 'coordinate') 
        nx.draw(graph_show, pos,ax=ax, with_labels=False, node_color=node_colors, node_size=800, edgecolors='white',edge_color = edge_colors ,width = 18,) 
        nx.draw_networkx_labels(graph_show,pos,labels=node_labels,font_size=node_font_size, font_color='white')
        nx.draw_networkx_edge_labels(graph_show, pos, edge_labels=edge_labels,font_size=8, font_color='white',bbox=dict(facecolor='none', edgecolor='none'),rotate=False)

        if show_qubits_attributes != '' and is_node_info_avaliable is True:
            if show_qubits_attributes == 'T1':
                show_label = r'T1 ($\mu s$)'
            elif show_qubits_attributes == 'T2':
                show_label = r'T2 ($\mu s$)'
            elif show_qubits_attributes == 'fidelity':
                show_label = 'Single qubit gate fidelity'
            elif show_qubits_attributes == 'frequency':
                show_label = 'Frequency (GHz)'
            node_sm = ScalarMappable(cmap=node_cmap, norm=node_norm)
            node_sm.set_array(list(node_attributes.values()))  # 设置包含数据的数组
            if show_couplers_fidelity and is_edge_info_avaliable is True:
                node_cbar = fig.colorbar(node_sm, ax=ax, orientation='horizontal',pad=0.07, fraction=0.03, aspect=25)
            else:
                node_cbar = fig.colorbar(node_sm, ax=ax, orientation='horizontal',pad=0.001, fraction=0.0333, aspect=25)
            node_cbar.set_label(show_label)

        if show_couplers_fidelity is True and is_edge_info_avaliable is True:
            edge_sm = ScalarMappable(cmap=edge_cmap, norm=edge_norm)# 创建颜色条
            edge_sm.set_array(list(edge_fidelity.values()))  # 设置包含数据的数组
            edge_cbar = fig.colorbar(edge_sm, ax=ax, orientation='horizontal',  pad=0.001, fraction=0.0333, aspect=25)# 调整颜色条大小和位置，放置在底部
            edge_cbar.set_label('CZ fidelity')

        xlim = ax.get_xlim() 
        ylim = ax.get_ylim()
        xpos = (xlim[0] + xlim[1]) / 2
        ypos = ylim[0] - (ylim[1] - ylim[0]) * -0.03  # 计算标题位置
        ax.text(xpos, ypos, f'{self.chip_name}', va='center', ha='center', fontsize=24, fontweight='bold', color='k', family='serif')

        ax.invert_yaxis() # y 轴正半轴朝下

        if save_svg_fname:
            # plt.savefig(save_svg_fname + '.svg', transparent=True, dpi=300, bbox_inches='tight') 
            plt.savefig(save_svg_fname + '.pdf', bbox_inches='tight') 
        plt.clf()
        plt.close()
        return None