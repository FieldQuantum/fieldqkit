"""Unified backend abstractions and generic graph-based backend helpers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx


MIN_CONNECTED_COUPLER_FIDELITY = 0.9


def _as_float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def is_connected_coupler(coupler_info: Any, *, min_fidelity: float = MIN_CONNECTED_COUPLER_FIDELITY) -> bool:
    if not isinstance(coupler_info, dict):
        return False
    fidelity = _as_float_or_default(coupler_info.get("fidelity", 1.0), 1.0)
    return fidelity >= float(min_fidelity)


def _fallback_priority_qubits(qubits_info: Any) -> List[List[int]]:
    if not isinstance(qubits_info, dict):
        return []
    scored: List[Tuple[float, int]] = []
    for key, value in qubits_info.items():
        if not isinstance(value, dict):
            continue
        try:
            qubit = int(str(key).replace("Q", ""))
        except Exception:
            continue
        fidelity = _as_float_or_default(value.get("fidelity", 1.0), 1.0)
        scored.append((fidelity, qubit))
    if not scored:
        return []
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [[qubit for _, qubit in scored]]


def _build_simulator_chip_info(nqubits: int = 16) -> dict:
    qubits_info = {
        f"Q{i}": {"fidelity": 1.0}
        for i in range(nqubits)
    }
    couplers_info = {}
    idx = 0
    for i in range(nqubits - 1):
        couplers_info[f"C{idx}"] = {
            "qubits_index": [i, i + 1],
            "fidelity": 1.0,
        }
        idx += 1
    global_info = {
        "two_qubit_gate_basis": "cz",
        "nqubits_available": nqubits,
        "error_rate_2q": 0.0,
        "one_qubit_gate_length": 0.01,
        "two_qubit_gate_length": 0.01,
    }
    return {
        "chip_name": "Simulator",
        "size": (1, nqubits),
        "priority_qubits": [list(range(nqubits))],
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": global_info,
    }


class Backend:
    """Graph-based hardware backend abstraction."""

    def __init__(self, chip: Union[str, dict]):
        if isinstance(chip, dict):
            self.chip_name = chip.get("chip_name", " ")
            self.chip_info = chip
        elif chip in ["Baihua", "Dongling", "Haituo", "Yunmeng", "Miaofeng", "Yudu", "Hongluo"]:
            from .quantum_platform.quafu import load_quafu_chip_info
            self.chip_name = str(chip)
            self.chip_info = load_quafu_chip_info(self.chip_name)
        elif chip in ["tianyan176", "tianyan176-2", "tianyan24", "tianyan504", "tianyan287", "gd_qc1", "chmy176", "gd_sim1"]:
            from .quantum_platform.cqlib import load_cqlib_chip_info
            self.chip_name = str(chip)
            self.chip_info = load_cqlib_chip_info(self.chip_name)
        elif chip in ["Simulator", "simulator"]:
            self.chip_name = "Simulator"
            self.chip_info = _build_simulator_chip_info()
        else:
            raise ValueError(f"Wrong chip name! {chip}")

        self.priority_qubits = (
            self.chip_info.get("priority_qubits")
            if isinstance(self.chip_info, dict)
            else None
        )
        if not self.priority_qubits:
            self.priority_qubits = _fallback_priority_qubits(
                self.chip_info.get("qubits_info") if isinstance(self.chip_info, dict) else {}
            )
        self.qubits_with_attributes = self._collect_qubits_with_attributes()
        self.couplers_with_attributes = self._collect_couplers_with_attributes()
        self.two_qubit_gate_basis = self.chip_info['global_info']['two_qubit_gate_basis'].lower()

    @property
    def graph(self):
        return self.get_graph()

    def edge_filtered_graph(self, thres: float = 0.6):
        def edge_filter(u, v):
            return self.graph[u][v].get("fidelity") >= thres

        def node_filter(n):
            return self.graph.nodes[n].get("fidelity") >= thres

        subgraph_view = nx.subgraph_view(self.graph, filter_node=node_filter, filter_edge=edge_filter)
        return nx.Graph(subgraph_view)

    def _collect_qubits_with_attributes(self):
        qubits_with_attributes = []
        for key in self.chip_info['qubits_info'].keys():
            qubit = int(key.split('Q')[1])
            qubits_with_attributes.append((qubit, self.chip_info['qubits_info'][key]))
        return qubits_with_attributes

    def _collect_couplers_with_attributes(self):
        couplers_with_attributes = []
        for key in self.chip_info['couplers_info'].keys():
            coupler_info = self.chip_info['couplers_info'][key]
            if not is_connected_coupler(coupler_info):
                continue
            if not isinstance(coupler_info, dict):
                continue
            pair = coupler_info.get('qubits_index')
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            qubit1, qubit2 = pair
            couplers_with_attributes.append((qubit1,qubit2,coupler_info))
        return couplers_with_attributes

    def get_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.qubits_with_attributes)
        graph.add_edges_from(self.couplers_with_attributes)
        return graph

    def draw(self, save_svg_fname: str | None = None, edge_fidelity_thres: float = 0.9):
        import matplotlib.pyplot as plt

        graph_show = self.edge_filtered_graph(thres=edge_fidelity_thres)
        pos = nx.get_node_attributes(graph_show, "coordinate")
        if not pos:
            pos = nx.spring_layout(graph_show, seed=42)
        elif len(pos) != graph_show.number_of_nodes():
            fallback_pos = nx.spring_layout(graph_show, seed=42)
            for node in graph_show.nodes():
                if node not in pos:
                    pos[node] = fallback_pos[node]
        node_colors = ["#083776" for _ in graph_show.nodes()]
        edge_colors = ["#083776" for _ in graph_show.edges()]
        node_labels = {node: node for node in graph_show.nodes()}
        edge_labels = {edge: "" for edge in graph_show.edges()}

        fig, ax = plt.subplots(figsize=(15, 13))
        nx.draw(
            graph_show,
            pos,
            ax=ax,
            with_labels=False,
            node_color=node_colors,
            node_size=800,
            edgecolors="white",
            edge_color=edge_colors,
            width=18,
        )
        nx.draw_networkx_labels(graph_show, pos, labels=node_labels, font_size=10, font_color="white")
        nx.draw_networkx_edge_labels(
            graph_show,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="white",
            bbox=dict(facecolor="none", edgecolor="none"),
            rotate=False,
        )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xpos = (xlim[0] + xlim[1]) / 2
        ypos = ylim[0] - (ylim[1] - ylim[0]) * -0.03
        ax.text(xpos, ypos, f"{self.chip_name}", va="center", ha="center", fontsize=24, fontweight="bold", color="k", family="serif")

        ax.invert_yaxis()

        if save_svg_fname:
            plt.savefig(save_svg_fname + ".svg", bbox_inches="tight")
        plt.clf()
        plt.close()

    def cache_topology_figure(self, edge_fidelity_thres: float = 0.9) -> None:
        cache_dir = Path(__file__).resolve().parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_path = cache_dir / f"{self.chip_name}_chip"
        self.draw(save_svg_fname=str(fig_path), edge_fidelity_thres=edge_fidelity_thres)


def normalize_hardware_preferences(prefer_hardware: Optional[Sequence[str] | str]) -> List[str]:
    """Normalize preferred hardware names into a compact list."""
    if isinstance(prefer_hardware, str):
        items = [prefer_hardware]
    elif isinstance(prefer_hardware, Sequence):
        items = [str(item) for item in prefer_hardware]
    else:
        items = []
    return [item.strip() for item in items if str(item).strip()]


def is_simulator_preferred(prefer_hardware: Optional[Sequence[str] | str]) -> bool:
    """Return True when hardware preference explicitly asks for simulator."""
    return any(item.lower() == "simulator" for item in normalize_hardware_preferences(prefer_hardware))


@dataclass(frozen=True)
class HardwareTopology:
    qubits: List[int]
    couplers: List[Tuple[int, int]]


@dataclass(frozen=True)
class HardwareCalibration:
    qubit_fidelity: Dict[int, float]
    coupler_fidelity: Dict[str, float]
    queue_length: Optional[int] = None


@dataclass(frozen=True)
class HardwareProfile:
    provider: str
    hardware_name: str
    nqubits_available: int
    two_qubit_gate_basis: str
    topology: HardwareTopology
    calibration: HardwareCalibration
    raw_info: Dict[str, Any] = field(default_factory=dict)


def build_simulator_profile(*, provider: str, num_qubits: int) -> HardwareProfile:
    """Build a synthetic hardware profile for the local simulator."""
    nqubits = max(int(num_qubits), 1)
    target = list(range(nqubits))
    return HardwareProfile(
        provider=provider,
        hardware_name="Simulator",
        nqubits_available=nqubits,
        two_qubit_gate_basis="cz",
        topology=HardwareTopology(qubits=target, couplers=[]),
        calibration=HardwareCalibration(qubit_fidelity={}, coupler_fidelity={}, queue_length=0),
        raw_info=_build_simulator_chip_info(nqubits=nqubits),
    )


@dataclass
class ResolvedBackend:
    provider: str
    hardware_name: str
    backend: Backend
    profile: Optional[HardwareProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    provider: str
    default_hardware_name: Optional[str] = None

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Return normalized hardware rows from the adapter's bound platform."""
        platform_obj = getattr(self, "_platform", None)
        if platform_obj is None or not hasattr(platform_obj, "list_available_hardware"):
            raise RuntimeError(f"{self.__class__.__name__} requires a bound platform with list_available_hardware()")
        return platform_obj.list_available_hardware()

    def discover_hardware(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
    ) -> List[HardwareProfile]:
        """Discover candidate hardware profiles from unified provider listing."""
        if is_simulator_preferred(prefer_hardware):
            return [build_simulator_profile(provider=self.provider, num_qubits=num_qubits)]

        rows = self.list_available_hardware()
        queue_by_name = {
            str(row.get("hardware_name") or "").strip(): row.get("queue_length")
            for row in rows
            if str(row.get("hardware_name") or "").strip()
        }

        preferred = normalize_hardware_preferences(prefer_hardware)
        if preferred:
            candidate_names = preferred
        else:
            candidate_names = [name for name in queue_by_name.keys() if name]

        fallback_hardware = self._fallback_hardware_name()
        if not candidate_names and fallback_hardware:
            candidate_names = [fallback_hardware]

        profiles: List[HardwareProfile] = []
        for machine_name in candidate_names:
            backend_obj = Backend(machine_name)
            profile = build_hardware_profile(
                provider=self.provider,
                hardware_name=machine_name,
                backend=backend_obj,
                queue_length=queue_by_name.get(machine_name),
                raw_info=getattr(backend_obj, "chip_info", {}),
            )
            if profile.nqubits_available >= num_qubits:
                profiles.append(profile)
        return profiles

    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
    ) -> ResolvedBackend:
        """Resolve a concrete backend target for one provider."""
        candidates = self.discover_hardware(num_qubits=num_qubits, prefer_hardware=prefer_hardware)
        if not candidates:
            raise RuntimeError("no available chips satisfy num_qubits requirement")

        chosen = candidates[0]
        is_simulator = str(chosen.hardware_name).lower() == "simulator"

        platform_obj = getattr(self, "_platform", None)
        if (not is_simulator) and platform_obj is not None and hasattr(platform_obj, "set_machine"):
            platform_obj.set_machine(chosen.hardware_name)

        backend_obj = Backend(chosen.hardware_name)
        if (not is_simulator) and hasattr(backend_obj, "cache_topology_figure"):
            backend_obj.cache_topology_figure()

        profile = build_hardware_profile(
            provider=self.provider,
            hardware_name=chosen.hardware_name,
            backend=backend_obj,
            queue_length=chosen.calibration.queue_length,
            raw_info=getattr(backend_obj, "chip_info", {}),
        )

        return ResolvedBackend(
            provider=self.provider,
            hardware_name=chosen.hardware_name,
            backend=backend_obj,
            profile=profile,
            metadata={"platform_obj": platform_obj} if platform_obj is not None else {},
        )

    def _fallback_hardware_name(self) -> Optional[str]:
        machine_name = getattr(self, "_machine_name", None)
        if machine_name:
            return str(machine_name)
        if self.default_hardware_name:
            return str(self.default_hardware_name)
        return None


def as_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def build_hardware_profile(
    *,
    provider: str,
    hardware_name: str,
    backend: Backend,
    queue_length: Optional[int],
    raw_info: Dict[str, Any],
) -> HardwareProfile:
    chip_info = getattr(backend, "chip_info", {}) if isinstance(backend, Backend) else {}
    qubits_info = chip_info.get("qubits_info", {}) if isinstance(chip_info, dict) else {}
    couplers_info = chip_info.get("couplers_info", {}) if isinstance(chip_info, dict) else {}
    global_info = chip_info.get("global_info", {}) if isinstance(chip_info, dict) else {}

    qubits: List[int] = []
    qubit_fidelity: Dict[int, float] = {}
    for key, value in qubits_info.items():
        try:
            qid = int(str(key).lstrip("Q"))
        except Exception:
            continue
        qubits.append(qid)
        if isinstance(value, dict):
            try:
                qubit_fidelity[qid] = float(value.get("fidelity", 1.0))
            except Exception:
                qubit_fidelity[qid] = 1.0

    couplers: List[Tuple[int, int]] = []
    coupler_fidelity: Dict[str, float] = {}
    for key, value in couplers_info.items():
        if not isinstance(value, dict):
            continue
        fidelity = _as_float_or_default(value.get("fidelity", 1.0), 1.0)
        if fidelity < MIN_CONNECTED_COUPLER_FIDELITY:
            continue
        pair = value.get("qubits_index")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            q1 = int(pair[0])
            q2 = int(pair[1])
        except Exception:
            continue
        couplers.append((q1, q2))
        coupler_fidelity[str(key)] = fidelity

    if not qubits:
        qubits = sorted(set(list(getattr(backend, "graph", {}).nodes) if hasattr(getattr(backend, "graph", None), "nodes") else []))
    if not qubits:
        try:
            nqubits_available = int(global_info.get("nqubits_available", 0) or 0)
        except Exception:
            nqubits_available = 0
        qubits = list(range(max(nqubits_available, 1)))

    try:
        basis = str(global_info.get("two_qubit_gate_basis", backend.two_qubit_gate_basis)).lower()
    except Exception:
        basis = "cz"

    return HardwareProfile(
        provider=provider,
        hardware_name=hardware_name,
        nqubits_available=len(qubits),
        two_qubit_gate_basis=basis,
        topology=HardwareTopology(qubits=sorted(qubits), couplers=couplers),
        calibration=HardwareCalibration(
            qubit_fidelity=qubit_fidelity,
            coupler_fidelity=coupler_fidelity,
            queue_length=queue_length,
        ),
        raw_info=raw_info if isinstance(raw_info, dict) else {},
    )


def list_available_hardware(provider: str) -> List[Dict[str, Any]]:
    provider_name = str(provider).lower()

    if provider_name == "quafu":
        from .quantum_platform.quafu import QuafuPlatform

        platform_obj = QuafuPlatform()
        return platform_obj.list_available_hardware()

    if provider_name == "tianyan":
        from .platform_credentials import get_tianyan_login_key
        from .quantum_platform.tianyan import TianYanPlatform

        login_key = get_tianyan_login_key()
        platform_obj = TianYanPlatform(login_key=login_key, auto_login=True, machine_name=None)
        return platform_obj.list_available_hardware()

    if provider_name == "guodun":
        from .platform_credentials import get_guodun_login_key
        from .quantum_platform.guodun import GuoDunPlatform

        login_key = get_guodun_login_key()
        platform_obj = GuoDunPlatform(login_key=login_key, auto_login=True, machine_name=None)
        return platform_obj.list_available_hardware()

    raise ValueError("provider must be one of: 'quafu', 'tianyan', or 'guodun'")
