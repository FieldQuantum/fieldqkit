"""Provider backend abstractions for multi-cloud integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backend import Backend


@dataclass
class ProviderBackendBundle:
    """Resolved backend plus preferred target qubits."""

    backend: Backend
    target_qubits: Optional[List[int]] = None


def _to_int_qubit(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("Q"):
            s = s[1:]
        if s.isdigit():
            return int(s)
    return None


def _infer_two_qubit_basis(config: Dict[str, Any]) -> str:
    global_info = config.get("global_info", {}) if isinstance(config, dict) else {}
    basis = global_info.get("two_qubit_gate_basis")
    if isinstance(basis, str) and basis.strip():
        b = basis.strip().lower()
        if b == "cnot":
            return "cx"
        return b

    twoq = config.get("twoQubitGate", {}) if isinstance(config, dict) else {}
    if isinstance(twoq, dict):
        key_to_basis = {
            "czGate": "cz",
            "cnotGate": "cx",
            "cxGate": "cx",
            "iswapGate": "iswap",
            "ecrGate": "ecr",
            "fsimGate": "fsim",
        }
        for key, basis_name in key_to_basis.items():
            if key in twoq:
                return basis_name

    overview = config.get("overview", {}) if isinstance(config, dict) else {}
    basis = overview.get("basisGate") if isinstance(overview, dict) else None
    if isinstance(basis, str) and basis.strip():
        b = basis.strip().lower()
        if b == "cnot":
            return "cx"
        return b
    return "cz"


def _extract_couplers(config: Dict[str, Any]) -> List[Tuple[int, int]]:
    disabled_qubits = _extract_disabled_qubits(config)
    disabled_couplers = _extract_disabled_couplers(config)
    out: List[Tuple[int, int]] = []

    couplers_info = config.get("couplers_info") if isinstance(config, dict) else None
    if isinstance(couplers_info, dict):
        for coupler_key, item in couplers_info.items():
            if str(coupler_key) in disabled_couplers:
                continue
            if not isinstance(item, dict):
                continue
            pair = item.get("qubits_index")
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                q1 = _to_int_qubit(pair[0])
                q2 = _to_int_qubit(pair[1])
                if q1 is not None and q2 is not None and q1 != q2 and q1 not in disabled_qubits and q2 not in disabled_qubits:
                    out.append((q1, q2))

    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        coupler_map = overview.get("coupler_map")
        if isinstance(coupler_map, dict):
            for coupler_key, pair in coupler_map.items():
                if str(coupler_key) in disabled_couplers:
                    continue
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    q1 = _to_int_qubit(pair[0])
                    q2 = _to_int_qubit(pair[1])
                    if q1 is not None and q2 is not None and q1 != q2 and q1 not in disabled_qubits and q2 not in disabled_qubits:
                        out.append((q1, q2))

    coupler_map = config.get("coupler_map") if isinstance(config, dict) else None
    if isinstance(coupler_map, dict):
        for coupler_key, pair in coupler_map.items():
            if str(coupler_key) in disabled_couplers:
                continue
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                q1 = _to_int_qubit(pair[0])
                q2 = _to_int_qubit(pair[1])
                if q1 is not None and q2 is not None and q1 != q2 and q1 not in disabled_qubits and q2 not in disabled_qubits:
                    out.append((q1, q2))

    # Remove duplicates while preserving order.
    dedup: List[Tuple[int, int]] = []
    seen = set()
    for q1, q2 in out:
        key = tuple(sorted((q1, q2)))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((q1, q2))
    return dedup


def _extract_qubits(config: Dict[str, Any], couplers: Sequence[Tuple[int, int]]) -> List[int]:
    qubits = set()
    disabled_qubits = _extract_disabled_qubits(config)

    qubits_info = config.get("qubits_info") if isinstance(config, dict) else None
    if isinstance(qubits_info, dict):
        for key in qubits_info.keys():
            q = _to_int_qubit(key)
            if q is not None:
                qubits.add(q)

    for q1, q2 in couplers:
        qubits.add(int(q1))
        qubits.add(int(q2))

    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        qlist = overview.get("qubits")
        if isinstance(qlist, list):
            for q in qlist:
                qi = _to_int_qubit(q)
                if qi is not None:
                    qubits.add(qi)

    nqubits = None
    global_info = config.get("global_info") if isinstance(config, dict) else None
    if isinstance(global_info, dict):
        nqubits = global_info.get("nqubits_available")
    if nqubits is None and isinstance(config, dict):
        nqubits = config.get("nqubits_available")
    try:
        nqubits_i = int(nqubits) if nqubits is not None else None
    except Exception:
        nqubits_i = None

    if nqubits_i is not None and nqubits_i > 0:
        qubits.update(range(nqubits_i))

    if not qubits:
        return [0]
    return sorted(int(q) for q in qubits if int(q) not in disabled_qubits)


def _parse_comma_ids(value: Any) -> List[str]:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _extract_disabled_qubits(config: Dict[str, Any]) -> set[int]:
    keys = ["disabledQubits", "disabled_qubits"]
    values: List[str] = []
    for k in keys:
        values.extend(_parse_comma_ids(config.get(k)))
    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        values.extend(_parse_comma_ids(overview.get("disabled_qubits")))
    out = set()
    for v in values:
        q = _to_int_qubit(v)
        if q is not None:
            out.add(q)
    return out


def _extract_disabled_couplers(config: Dict[str, Any]) -> set[str]:
    keys = ["disabledCouplers", "disabled_couplers"]
    values: List[str] = []
    for k in keys:
        values.extend(_parse_comma_ids(config.get(k)))
    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        values.extend(_parse_comma_ids(overview.get("disabled_couplers")))
    return {v.strip() for v in values if v.strip()}


def _select_connected_target_qubits(
    *,
    qubits: Sequence[int],
    couplers: Sequence[Tuple[int, int]],
    num_qubits: int,
    preferred: Optional[Sequence[int]] = None,
) -> Optional[List[int]]:
    if num_qubits <= 0:
        return []
    qset = set(int(q) for q in qubits)
    if len(qset) < num_qubits:
        return None
    if num_qubits == 1:
        if preferred:
            for q in preferred:
                if int(q) in qset:
                    return [int(q)]
        return [sorted(qset)[0]]

    adjacency: Dict[int, set[int]] = {int(q): set() for q in qset}
    for q1, q2 in couplers:
        a = int(q1)
        b = int(q2)
        if a in adjacency and b in adjacency and a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    def _bfs(start: int) -> List[int]:
        visited: List[int] = []
        seen = set([start])
        queue = [start]
        while queue and len(visited) < num_qubits:
            cur = queue.pop(0)
            visited.append(cur)
            for nb in sorted(adjacency.get(cur, [])):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
                    if len(seen) >= num_qubits and len(visited) + len(queue) >= num_qubits:
                        pass
        return visited[:num_qubits]

    starts: List[int] = []
    if preferred:
        for q in preferred:
            qi = int(q)
            if qi in qset and qi not in starts:
                starts.append(qi)
    for q in sorted(qset):
        if q not in starts:
            starts.append(q)

    for s in starts:
        if not adjacency.get(s):
            continue
        cand = _bfs(s)
        if len(cand) == num_qubits:
            return cand

    # Last fallback: if no couplers available, return first N qubits.
    if not couplers:
        return sorted(qset)[:num_qubits]
    return None


def _extract_priority_qubits(config: Dict[str, Any], qubits: Sequence[int], num_qubits: int) -> Optional[List[int]]:
    couplers = _extract_couplers(config)
    preferred_seq: Optional[List[int]] = None

    if isinstance(config, dict):
        pq = config.get("priority_qubits")
        if isinstance(pq, list) and pq:
            first = pq[0]
            if isinstance(first, (list, tuple)):
                seq = [q for q in (_to_int_qubit(x) for x in first) if q is not None]
                if len(seq) >= num_qubits:
                    preferred_seq = list(seq)

    if preferred_seq is not None:
        picked = _select_connected_target_qubits(
            qubits=qubits,
            couplers=couplers,
            num_qubits=num_qubits,
            preferred=preferred_seq,
        )
        if picked is not None:
            return picked

    return _select_connected_target_qubits(qubits=qubits, couplers=couplers, num_qubits=num_qubits, preferred=None)


def _build_backend_chip_info(config: Dict[str, Any], *, num_qubits: int) -> Tuple[dict, Optional[List[int]]]:
    couplers = _extract_couplers(config)
    qubits = _extract_qubits(config, couplers)

    # Fallback: when topology is unavailable, use a simple linear chain.
    if not couplers and len(qubits) > 1:
        couplers = [(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)]

    target_qubits = _extract_priority_qubits(config, qubits, num_qubits)
    twoq_basis = _infer_two_qubit_basis(config)

    qubits_info = {
        f"Q{q}": {
            "fidelity": 1.0,
            "coordinate": [float(i), 0.0],
            "T1": 0.0,
            "T2": 0.0,
            "frequency": 0.0,
        }
        for i, q in enumerate(qubits)
    }

    couplers_info = {}
    for idx, (q1, q2) in enumerate(couplers):
        couplers_info[f"C{idx}"] = {
            "qubits_index": [int(q1), int(q2)],
            "fidelity": 1.0,
            "index": idx,
        }

    chip_info = {
        "size": (len(qubits), 1),
        "priority_qubits": [target_qubits] if target_qubits else [list(qubits)],
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": {
            "two_qubit_gate_basis": twoq_basis,
            "nqubits_available": len(qubits),
            "error_rate_2q": 0.0,
            "one_qubit_gate_length": 1.0,
            "two_qubit_gate_length": 1.0,
        },
        "calibration_time": "cqlib",
    }
    return chip_info, target_qubits


def _has_topology_payload(config: Dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        return False
    if isinstance(config.get("couplers_info"), dict) and config.get("couplers_info"):
        return True
    if isinstance(config.get("coupler_map"), dict) and config.get("coupler_map"):
        return True
    overview = config.get("overview")
    if isinstance(overview, dict):
        cmap = overview.get("coupler_map")
        if isinstance(cmap, dict) and cmap:
            return True
    return False


def build_cqlib_backend_bundle(
    *,
    platform_obj: Any,
    machine_name: Optional[str],
    num_qubits: int,
) -> ProviderBackendBundle:
    """Build a local Backend object from cqlib machine config payloads."""
    configs: List[Dict[str, Any]] = []

    if machine_name is None:
        machine_name = getattr(platform_obj, "machine_name", None)

    primary_has_topology = False
    try:
        cfg = platform_obj.download_config(machine=machine_name)
        if isinstance(cfg, dict):
            configs.append(cfg)
            primary_has_topology = _has_topology_payload(cfg)
    except Exception:
        pass

    # Fallback: only request overview payload when primary config lacks topology.
    if not primary_has_topology:
        try:
            cfg = platform_obj.get_machine_config(
                params={
                    "type": "overview",
                    "computerCode": machine_name,
                    "label": "qpu_coordinate,coupler_map,disabled_couplers,disabled_qubits",
                }
            )
            if isinstance(cfg, dict):
                configs.append(cfg)
        except Exception:
            pass

    merged: Dict[str, Any] = {}
    for cfg in configs:
        for key, value in cfg.items():
            if key not in merged:
                merged[key] = value

    chip_info, target_qubits = _build_backend_chip_info(merged, num_qubits=num_qubits)
    backend = Backend(chip_info)
    return ProviderBackendBundle(backend=backend, target_qubits=target_qubits)
