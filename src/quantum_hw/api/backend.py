"""Unified backend abstractions and generic graph-based backend helpers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx


MIN_CONNECTED_COUPLER_FIDELITY = 0.9

# ---------------------------------------------------------------------------
# Canonical chip-name registry (single source of truth)
# ---------------------------------------------------------------------------

QUAFU_HARDWARE_NAMES = {"Baihua", "Dongling", "Yudu", "Hongluo"}
TIANYAN_HARDWARE_NAMES = {"supremacy_sample", "tianyan-287", "tianyan176", "tianyan176-2", "tianyan24", "tianyan504", "tianyan_s", "tianyan_sa", "tianyan_sw", "tianyan_swn", "tianyan_tn"}
GUODUN_HARDWARE_NAMES = {"chmy176", "gd_qc1", "gd_sim1", "gd_test"}
CQLIB_HARDWARE_NAMES = TIANYAN_HARDWARE_NAMES | GUODUN_HARDWARE_NAMES
TENCENT_HARDWARE_NAMES = {"simulator:tc", "tianji_m2", "tianji_m2v14s2", "tianji_m2v14s4", "tianji_m2v15s3", "tianji_m2v16s1", "tianji_s2", "tianji_s2v6", "tianji_s2v7", "tianxuan_s2", "tianxuan_s2v20s1", "tianxuan_s2v20s2"}
ORIGIN_HARDWARE_NAMES = {"PQPUMESH8", "WK_C180"}
SIMULATOR_HARDWARE_NAMES = {"Simulator", "simulator"}
FIELDQUANTUM_HARDWARE_NAMES = {"fieldquantum_sim"}

# Cloud-side simulators registered under a real-hardware provider. The provider's
# config endpoint refuses to return topology for these (e.g. cqlib returns
# "Only quantum physics machines can obtain configuration parameters"), so we
# substitute a synthetic full-connectivity chip_info and still route jobs
# through the provider's task adapter.
TIANYAN_CLOUD_SIM_NAMES = {
    "supremacy_sample",
    "tianyan_s",
    "tianyan_sa",
    "tianyan_sw",
    "tianyan_swn",
    "tianyan_tn",
}
GUODUN_CLOUD_SIM_NAMES: set[str] = set()
TENCENT_CLOUD_SIM_NAMES = {"simulator:tc"}
CLOUD_SIM_HARDWARE_NAMES = (
    TIANYAN_CLOUD_SIM_NAMES | GUODUN_CLOUD_SIM_NAMES | TENCENT_CLOUD_SIM_NAMES
)


def _as_float_or_default(value: Any, default: float) -> float:
    """Convert *value* to float, returning *default* on failure.

    Args:
        value (*Any*): Value to convert.
        default (*float*): Fallback value if conversion fails.

    Returns:
        ``float`` value of *value*, or *default* on failure.
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def is_connected_coupler(coupler_info: Any, *, min_fidelity: float = MIN_CONNECTED_COUPLER_FIDELITY) -> bool:
    """Return True if *coupler_info* has fidelity >= *min_fidelity*.

    Args:
        coupler_info (*Any*): Dictionary with coupler metadata (must contain a ``"fidelity"`` key).
        min_fidelity (*float*): Minimum fidelity threshold. Defaults to ``MIN_CONNECTED_COUPLER_FIDELITY``.

    Returns:
        ``True`` if the condition is satisfied.
    """
    if not isinstance(coupler_info, dict):
        return False
    fidelity = _as_float_or_default(coupler_info.get("fidelity", 1.0), 1.0)
    return fidelity >= float(min_fidelity)


def _fallback_priority_qubits(qubits_info: Any) -> List[List[int]]:
    """Build a priority qubit list sorted by descending fidelity.

    Args:
        qubits_info (*Any*): Dictionary mapping qubit keys to metadata dicts.

    Returns:
        Nested list ``[[q0, q1, ...]]`` with qubits sorted by descending fidelity.
    """
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
    """Build a synthetic chip_info dict for the local simulator.

    Args:
        nqubits (*int*): Number of qubits. Defaults to ``16``.

    Returns:
        Synthetic chip-info dictionary with perfect fidelities.
    """
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
        "one_qubit_gate_length": 1e-8,
        "two_qubit_gate_length": 5e-8,
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
        """Initialize Backend from a chip name or chip info dict.

        Args:
            chip (*Union[str, dict]*): Chip name string or pre-built chip-info dictionary.

        Raises:
            ValueError: f'Wrong chip name! {chip}'
        """
        if isinstance(chip, dict):
            self.chip_name = chip.get("chip_name", " ")
            self.chip_info = chip
        elif chip in CLOUD_SIM_HARDWARE_NAMES:
            self.chip_name = str(chip)
            self.chip_info = _build_simulator_chip_info()
            self.chip_info["chip_name"] = self.chip_name
        elif chip in QUAFU_HARDWARE_NAMES:
            from .quantum_platform.quafu import load_quafu_chip_info
            self.chip_name = str(chip)
            self.chip_info = load_quafu_chip_info(self.chip_name)
        elif chip in CQLIB_HARDWARE_NAMES:
            from .quantum_platform.cqlib import load_cqlib_chip_info
            self.chip_name = str(chip)
            self.chip_info = load_cqlib_chip_info(self.chip_name)
        elif chip in TENCENT_HARDWARE_NAMES:
            from .quantum_platform.tencent import _load_tencent_chip_info
            self.chip_name = str(chip)
            self.chip_info = _load_tencent_chip_info(self.chip_name)
        elif chip in ORIGIN_HARDWARE_NAMES:
            from .quantum_platform.origin import load_origin_chip_info
            self.chip_name = str(chip)
            self.chip_info = load_origin_chip_info(self.chip_name)
        elif chip in SIMULATOR_HARDWARE_NAMES:
            self.chip_name = "Simulator"
            self.chip_info = _build_simulator_chip_info()
        elif chip in FIELDQUANTUM_HARDWARE_NAMES:
            self.chip_name = "fieldquantum_sim"
            self.chip_info = _build_simulator_chip_info()
            self.chip_info["chip_name"] = "fieldquantum_sim"
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
        """Return the hardware coupling graph (``networkx.Graph``).

        Returns:
            ``networkx.Graph`` instance.
        """
        return self.get_graph()

    def edge_filtered_graph(self, thres: float = 0.6):
        """Return a subgraph keeping only edges/nodes with fidelity >= *thres*.

        Args:
            thres (*float*): Fidelity threshold. Defaults to ``0.6``.

        Returns:
            ``networkx.Graph`` containing only edges and nodes above the threshold.
        """
        def edge_filter(u, v):
            """Return ``True`` if edge fidelity >= threshold.

            Args:
                u: Source node index.
                v: Target node index.

            Returns:
                ``bool`` whether the edge passes the fidelity filter.
            """
            return self.graph[u][v].get("fidelity") >= thres

        def node_filter(n):
            """Return ``True`` if node fidelity >= threshold.

            Args:
                n: Node index.

            Returns:
                ``bool`` whether the node passes the fidelity filter.
            """
            return self.graph.nodes[n].get("fidelity") >= thres

        subgraph_view = nx.subgraph_view(self.graph, filter_node=node_filter, filter_edge=edge_filter)
        return nx.Graph(subgraph_view)

    def _collect_qubits_with_attributes(self):
        """Collect qubit index / attribute pairs from chip_info.

        Returns:
            List of ``(qubit_index, attributes_dict)`` tuples.
        """
        qubits_with_attributes = []
        for key in self.chip_info['qubits_info'].keys():
            qubit = int(key.split('Q')[1])
            qubits_with_attributes.append((qubit, self.chip_info['qubits_info'][key]))
        return qubits_with_attributes

    def _collect_couplers_with_attributes(self):
        """Collect connected coupler triples from chip_info.

        Returns:
            List of ``(q1, q2, attributes_dict)`` tuples.
        """
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
            qubit1, qubit2 = int(pair[0]), int(pair[1])
            couplers_with_attributes.append((qubit1,qubit2,coupler_info))
        return couplers_with_attributes

    def get_graph(self):
        """Build and return a new ``networkx.Graph`` from qubits and couplers.

        Returns:
            ``networkx.Graph`` with qubit nodes and coupler edges.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.qubits_with_attributes)
        graph.add_edges_from(self.couplers_with_attributes)
        return graph

    def draw(self, save_svg_fname: str | None = None, edge_fidelity_thres: float = 0.9):
        """Draw the hardware topology graph and optionally save as SVG.

        Args:
            save_svg_fname (*str | None*): File path to save the topology as SVG, or ``None`` to skip. Defaults to ``None``.
            edge_fidelity_thres (*float*): Minimum fidelity threshold for rendering coupler edges. Defaults to ``0.9``.
        """
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
        """Draw and cache the topology figure as SVG in the .cache directory.

        Args:
            edge_fidelity_thres (*float*): Minimum fidelity threshold for rendering coupler edges. Defaults to ``0.9``.
        """
        cache_dir = Path(__file__).resolve().parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fig_path = cache_dir / f"{self.chip_name}_chip"
        self.draw(save_svg_fname=str(fig_path), edge_fidelity_thres=edge_fidelity_thres)


def normalize_hardware_preferences(prefer_hardware: Optional[Sequence[str] | str]) -> List[str]:
    """Normalize preferred hardware names into a compact list.

    Args:
        prefer_hardware (*Optional[Sequence[str] | str]*): Raw hardware preference (string, sequence, or ``None``).

    Returns:
        Normalised list of trimmed, non-empty hardware name strings.
    """
    if isinstance(prefer_hardware, str):
        items = [prefer_hardware]
    elif isinstance(prefer_hardware, Sequence):
        items = [str(item) for item in prefer_hardware]
    else:
        items = []
    return [item.strip() for item in items if str(item).strip()]


def is_simulator_preferred(prefer_hardware: Optional[Sequence[str] | str]) -> bool:
    """Return True when hardware preference explicitly asks for simulator.

    Args:
        prefer_hardware (*Optional[Sequence[str] | str]*): Hardware name(s) to check, or ``None``.

    Returns:
        ``True`` if the condition is satisfied.
    """
    return any(item.lower() == "simulator" for item in normalize_hardware_preferences(prefer_hardware))


@dataclass(frozen=True)
class HardwareTopology:
    """Physical qubit connectivity of a quantum chip."""

    qubits: List[int]
    couplers: List[Tuple[int, int]]


@dataclass(frozen=True)
class HardwareCalibration:
    """Calibration fidelity data and queue status for a quantum chip."""

    qubit_fidelity: Dict[int, float]
    coupler_fidelity: Dict[str, float]
    queue_length: Optional[int] = None


@dataclass(frozen=True)
class HardwareProfile:
    """Unified hardware description combining topology, calibration, and provider metadata."""

    provider: str
    hardware_name: str
    nqubits_available: int
    two_qubit_gate_basis: str
    topology: HardwareTopology
    calibration: HardwareCalibration
    raw_info: Dict[str, Any] = field(default_factory=dict)


def build_simulator_profile(*, provider: str, num_qubits: int) -> HardwareProfile:
    """Build a synthetic hardware profile for the local simulator.

    Args:
        provider (*str*): Platform provider name (``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``).
        num_qubits (*int*): Number of qubits.

    Returns:
        ``HardwareProfile`` for the simulated backend.
    """
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


def _build_backend_for_chip(chip_name: str, *, num_qubits: int) -> "Backend":
    """Build a ``Backend`` for *chip_name*, scaling cloud simulators to *num_qubits*.
    """
    if str(chip_name) in CLOUD_SIM_HARDWARE_NAMES:
        chip_info = _build_simulator_chip_info(nqubits=max(int(num_qubits), 1))
        chip_info["chip_name"] = str(chip_name)
        return Backend(chip_info)
    return Backend(chip_name)


@dataclass
class ResolvedBackend:
    provider: str
    hardware_name: str
    backend: Backend
    profile: Optional[HardwareProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    """Abstract adapter bridging provider-specific platforms to the unified backend interface."""

    provider: str
    default_hardware_name: Optional[str] = None

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Return normalized hardware rows from the adapter's bound platform.

        Returns:
            List of hardware description dictionaries.

        Raises:
            RuntimeError: If no bound platform with ``list_available_hardware()`` is available.
        """
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
        """Discover candidate hardware profiles from unified provider listing.

        Args:
            num_qubits (*int*): Number of qubits.
            prefer_hardware (*Optional[Sequence[str] | str]*): Preferred hardware name(s) to filter candidates. Defaults to ``None``.

        Returns:
            List of ``HardwareProfile`` instances matching the qubit requirement.
        """
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
            backend_obj = _build_backend_for_chip(machine_name, num_qubits=num_qubits)
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
        """Resolve a concrete backend target for one provider.

        Args:
            num_qubits (*int*): Number of qubits.
            prefer_hardware (*Optional[Sequence[str] | str]*): Preferred hardware name(s) to filter candidates. Defaults to ``None``.

        Returns:
            Resolved ``ResolvedBackend`` instance containing provider, backend, and profile.

        Raises:
            RuntimeError: no available chips satisfy num_qubits requirement
        """
        candidates = self.discover_hardware(num_qubits=num_qubits, prefer_hardware=prefer_hardware)
        if not candidates:
            raise RuntimeError("no available chips satisfy num_qubits requirement")

        chosen = candidates[0]
        is_simulator = str(chosen.hardware_name).lower() == "simulator"

        platform_obj = getattr(self, "_platform", None)
        if (not is_simulator) and platform_obj is not None and hasattr(platform_obj, "set_machine"):
            platform_obj.set_machine(chosen.hardware_name)

        backend_obj = _build_backend_for_chip(chosen.hardware_name, num_qubits=num_qubits)

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
        """Return fallback hardware name from instance or class default.

        Returns:
            ``Optional[str]``: Hardware name or ``None``.
        """
        machine_name = getattr(self, "_machine_name", None)
        if machine_name:
            return str(machine_name)
        if self.default_hardware_name:
            return str(self.default_hardware_name)
        return None


def as_int_or_none(value: Any) -> Optional[int]:
    """Convert *value* to int, returning None on failure.

    Args:
        value (*Any*): Value to set.

    Returns:
        ``int`` value, or ``None`` if conversion fails.
    """
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
    """Build a ``HardwareProfile`` from a ``Backend`` instance.

    Args:
        provider (*str*): Platform provider name (``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``).
        hardware_name (*str*): Human-readable identifier for the quantum hardware.
        backend (*Backend*): Hardware backend descriptor.
        queue_length (*Optional[int]*): Current job queue depth on the hardware.
        raw_info (*Dict[str, Any]*): Unprocessed chip metadata from the platform.

    Returns:
        Constructed ``HardwareProfile`` for the given backend.
    """
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
    """List available hardware for the given provider.

    Args:
        provider (*str*): Platform provider name (``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``).

    Returns:
        List of normalized hardware description dictionaries.

    Raises:
        ValueError: If *provider* is not one of the supported platform names.
    """
    provider_name = str(provider).lower()

    if provider_name == "quafu":
        from .quantum_platform.quafu import QuafuPlatform

        platform_obj = QuafuPlatform()
        return platform_obj.list_available_hardware()

    if provider_name == "tianyan":
        from .platform_credentials import get_tianyan_api_token
        from .quantum_platform.tianyan import TianYanPlatform

        api_token = get_tianyan_api_token()
        platform_obj = TianYanPlatform(login_key=api_token, auto_login=True, machine_name=None)
        return platform_obj.list_available_hardware()

    if provider_name == "guodun":
        from .platform_credentials import get_guodun_api_token
        from .quantum_platform.guodun import GuoDunPlatform

        api_token = get_guodun_api_token()
        platform_obj = GuoDunPlatform(login_key=api_token, auto_login=True, machine_name=None)
        return platform_obj.list_available_hardware()

    if provider_name == "tencent":
        from .quantum_platform.tencent import TencentPlatform

        platform_obj = TencentPlatform()
        return platform_obj.list_available_hardware()

    if provider_name == "origin":
        from .platform_credentials import get_origin_api_token
        from .quantum_platform.origin import OriginPlatform

        api_token = get_origin_api_token()
        platform_obj = OriginPlatform(token=api_token)
        return platform_obj.list_available_hardware()

    raise ValueError("provider must be one of: 'quafu', 'tianyan', 'guodun', 'tencent', or 'origin'")


# ---------------------------------------------------------------------------
# Chip name → provider inference
# ---------------------------------------------------------------------------

_CHIP_PROVIDER_MAP: Dict[str, str] = {}

def _register_chips(provider: str, names: Sequence[str]) -> None:
    for n in names:
        _CHIP_PROVIDER_MAP[n.lower()] = provider

_register_chips("quafu", QUAFU_HARDWARE_NAMES)
_register_chips("tianyan", TIANYAN_HARDWARE_NAMES)
_register_chips("guodun", GUODUN_HARDWARE_NAMES)
_register_chips("tencent", TENCENT_HARDWARE_NAMES)
_register_chips("origin", ORIGIN_HARDWARE_NAMES)
_register_chips("simulator", SIMULATOR_HARDWARE_NAMES)
_register_chips("fieldquantum", FIELDQUANTUM_HARDWARE_NAMES)


def infer_provider_from_chip(chip_name: str) -> Optional[str]:
    """Infer the provider name from a chip/hardware name.

    Args:
        chip_name (*str*): Name of the chip or hardware.

    Returns:
        Provider name string, or ``None`` if the chip is unknown.
    """
    return _CHIP_PROVIDER_MAP.get(str(chip_name).lower())


def resolve_provider(
    provider: str,
    prefer_chips: Optional[Sequence[str] | str] = None,
) -> str:
    """Resolve the effective provider from *prefer_chips* or fall back to *provider*.

    If *prefer_chips* contains a single known chip name whose provider can be
    inferred, that inferred provider is returned.  Otherwise *provider* is
    returned as-is.

    Args:
        provider (*str*): Caller-supplied provider name.
        prefer_chips (*Optional[Sequence[str] | str]*): Preferred chip names. Defaults to ``None``.

    Returns:
        Effective provider name string (lower-cased).
    """
    chips = normalize_hardware_preferences(prefer_chips)
    if chips:
        inferred = infer_provider_from_chip(chips[0])
        if inferred:
            return inferred
    return str(provider).lower()


# ---------------------------------------------------------------------------
# Simulator-only backend adapter (no credentials required)
# ---------------------------------------------------------------------------

class SimulatorBackendAdapter(BackendAdapter):
    """Lightweight backend adapter for the local simulator (no API token needed)."""

    provider = "simulator"
    default_hardware_name = "Simulator"

    def __init__(self) -> None:
        self._machine_name = "Simulator"

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        return []

    def discover_hardware(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
    ) -> List[HardwareProfile]:
        return [build_simulator_profile(provider=self.provider, num_qubits=num_qubits)]

    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
    ) -> ResolvedBackend:
        backend_obj = Backend(_build_simulator_chip_info(nqubits=num_qubits))
        profile = build_simulator_profile(provider=self.provider, num_qubits=num_qubits)
        return ResolvedBackend(
            provider=self.provider,
            hardware_name="Simulator",
            backend=backend_obj,
            profile=profile,
            metadata={},
        )
