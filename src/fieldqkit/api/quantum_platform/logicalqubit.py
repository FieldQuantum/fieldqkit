"""LogicalQubit (逻辑比特) cloud provider — direct REST integration.

This adapter talks to the LogicalQubit quantum cloud
(https://cloud.logicalqubit.com) directly over its JSON REST API and does
**not** depend on the vendor ``lqcloud`` SDK. The wire protocol was
reverse-engineered from that SDK; the shapes below mirror what the SDK
sends on the wire:

Endpoints
    GET  /api/v1/qpus                     -> [{name, topology, properties, ...}]
    POST /api/v1/tasks/async              -> {task_id, status, queue_position}
    GET  /api/v1/tasks/async/{task_id}    -> {status, result, ...}

Auth is a single ``X-API-Key`` header (``qk_<...>`` token issued by the
platform). Circuits are transmitted as a plain instruction-list IR under
``command.circuit`` — every single-qubit gate is emitted as one native
``su2`` (arbitrary 2x2 unitary, matching how the SDK lowers numeric
RX/RY/U), two-qubit gates as native ``cz``. On backends whose only
readout mode is ``measure2`` (2-state / qutrit-aware readout) an ``x21``
pulse is inserted before every ``measure``.

Counts come back big-endian (clbit 0 = leftmost character), which already
matches fieldqkit's :func:`fieldqkit.core.utils.get_samples` convention —
no bit-reversal is applied (verified against QZ02 hardware).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests

from ..backend import BackendAdapter, ResolvedBackend, as_int_or_none
from ..task import NativeCircuitSubmitRequest, ProviderTaskHandle, TaskAdapter

logger = logging.getLogger(__name__)

LOGICALQUBIT_DEFAULT_URL = "https://cloud.logicalqubit.com"

_HTTP_TIMEOUT = (10, 300)
_MAX_RETRIES = 5
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

# Per-shot (memory) policy — mirrors the vendor SDK's per_shot_config.
_PER_SHOT_QUBIT_THRESHOLD = 12
_PER_SHOT_MAX_SHOTS = 50_000


# ---------------------------------------------------------------------------
# Credentials / config
# ---------------------------------------------------------------------------

def _get_logicalqubit_token(token: Optional[str] = None) -> str:
    """Resolve the LogicalQubit API token (arg -> credential store)."""
    if token:
        return token
    from ..platform_credentials import get_logicalqubit_api_token
    return get_logicalqubit_api_token()


# ---------------------------------------------------------------------------
# HTTP platform client
# ---------------------------------------------------------------------------

class LogicalQubitPlatform:
    """Thin REST client for the LogicalQubit cloud (no vendor SDK)."""

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = _get_logicalqubit_token(token)
        if not self._token:
            raise ValueError("logicalqubit api token cannot be empty")
        self._url = LOGICALQUBIT_DEFAULT_URL
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self._token,
            "User-Agent": "fieldqkit-logicalqubit/0.1",
        })
        self._qpus_cache: Optional[List[Dict[str, Any]]] = None
        self._machine_name: Optional[str] = None

    # -- low-level request with retry ----------------------------------

    def _request(self, method: str, path: str, json_body: Any = None) -> Any:
        url = f"{self._url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            headers = {}
            if method.upper() == "POST":
                headers["Idempotency-Key"] = uuid.uuid4().hex
            try:
                resp = self._session.request(
                    method.upper(), url, json=json_body, headers=headers, timeout=_HTTP_TIMEOUT
                )
                if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                    time.sleep(min(30.0, 0.5 * (2 ** attempt)))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(min(30.0, 0.5 * (2 ** attempt)))
                    continue
        raise RuntimeError(f"LogicalQubit request {method} {path} failed after {_MAX_RETRIES} retries") from last_exc

    # -- backend discovery ---------------------------------------------

    def get_qpus(self, *, refresh: bool = False) -> List[Dict[str, Any]]:
        """Return the raw QPU config list (cached)."""
        if self._qpus_cache is not None and not refresh:
            return self._qpus_cache
        data = self._request("GET", "/api/v1/qpus")
        rows = data if isinstance(data, list) else data.get("qpus", data.get("backends", []))
        self._qpus_cache = [r for r in rows if isinstance(r, dict)]
        return self._qpus_cache

    def get_backend_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the config dict for one backend (or ``None``)."""
        for row in self.get_qpus():
            if row.get("name") == name:
                return row
        return None

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Return normalized hardware rows for the unified adapter layer."""
        rows: List[Dict[str, Any]] = []
        for row in self.get_qpus():
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            rows.append({
                "provider": "logicalqubit",
                "hardware_name": name,
                "queue_length": as_int_or_none(row.get("queue_position")),
                "status": row.get("status"),
                "is_toll": None,
                "raw": {"type": row.get("type"), "qubits": row.get("qubits")},
            })
        return rows

    def set_machine(self, name: str) -> None:
        self._machine_name = str(name)

    # -- task lifecycle -------------------------------------------------

    def submit(self, command: Dict[str, Any], chip_name: str) -> str:
        """Submit a ``run_circuit`` command; return the task id."""
        resp = self._request(
            "POST", "/api/v1/tasks/async",
            json_body={"command": command, "qpu_name": chip_name},
        )
        task_id = resp.get("task_id") or resp.get("job_id")
        if not task_id:
            raise RuntimeError(f"LogicalQubit submission returned no task id: {resp!r}")
        return str(task_id)

    def task_detail(self, task_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/tasks/async/{task_id}")

    def status(self, task_id: str) -> str:
        """Return the raw server status string (lower-case)."""
        return str(self.task_detail(task_id).get("status", "")).lower()

    def result_counts(self, task_id: str) -> Dict[str, int]:
        """Fetch a completed task's measurement counts (``{bitstring: n}``)."""
        detail = self.task_detail(task_id)
        st = str(detail.get("status", "")).lower()
        if st != "completed":
            raise RuntimeError(
                f"LogicalQubit task {task_id} not completed (status={st}): "
                f"{detail.get('error') or detail.get('error_type')}"
            )
        res = detail.get("result")
        if isinstance(res, str):
            if res.startswith("http"):
                res = self._request("GET", res) if res.startswith(self._url) else requests.get(res, timeout=_HTTP_TIMEOUT).json()
            else:
                import json as _json
                try:
                    res = _json.loads(res)
                except Exception:
                    pass
        if not isinstance(res, dict):
            raise RuntimeError(f"LogicalQubit task {task_id}: unexpected result payload {res!r}")
        if isinstance(res.get("counts"), dict):
            return {str(k): int(v) for k, v in res["counts"].items()}
        if res.get("result_format") == "memory" or "memory" in res:
            counts: Dict[str, int] = {}
            for bit in res.get("memory", []):
                counts[str(bit)] = counts.get(str(bit), 0) + 1
            return counts
        raise RuntimeError(f"LogicalQubit task {task_id}: no counts/memory in result {res!r}")


# ---------------------------------------------------------------------------
# Status normalisation
# ---------------------------------------------------------------------------

_STATUS_MAP: Dict[str, str] = {
    "queued": "Running",
    "pending": "Running",
    "running": "Running",
    "completed": "Finished",
    "failed": "Failed",
    "cancelled": "Canceled",
}


# ---------------------------------------------------------------------------
# chip_info loader
# ---------------------------------------------------------------------------

def load_logicalqubit_chip_info(
    chip_name: str,
    token: Optional[str] = None,
    *,
    platform: Optional[LogicalQubitPlatform] = None,
) -> Dict[str, Any]:
    """Fetch and normalise a LogicalQubit backend into fieldqkit ``chip_info``.

    Maps ``topology.coupling_map`` / ``qubit_coordinates`` and per-qubit
    ``properties.qubit_metrics`` (``xeb_fidelity`` -> gate fidelity,
    ``measure_f0`` / ``measure_f1`` -> readout confusion) into the dict
    consumed by :class:`fieldqkit.api.backend.Backend`.
    """
    plat = platform or LogicalQubitPlatform(token=token)
    cfg = plat.get_backend_config(chip_name)
    if not isinstance(cfg, dict):
        raise ValueError(f"LogicalQubit backend not found: {chip_name}")

    topo = cfg.get("topology", {}) if isinstance(cfg.get("topology"), dict) else {}
    coords = topo.get("qubit_coordinates", {}) if isinstance(topo.get("qubit_coordinates"), dict) else {}
    coupling_map = topo.get("coupling_map", []) if isinstance(topo.get("coupling_map"), list) else []
    props = cfg.get("properties", {}) if isinstance(cfg.get("properties"), dict) else {}
    qubit_metrics = props.get("qubit_metrics", []) if isinstance(props.get("qubit_metrics"), list) else []

    nqubits = cfg.get("qubits")
    if not isinstance(nqubits, int):
        nqubits = len(qubit_metrics) or (max((int(q) for pair in coupling_map for q in pair), default=-1) + 1)

    metric_by_id: Dict[int, Dict[str, Any]] = {}
    for m in qubit_metrics:
        if isinstance(m, dict) and m.get("id") is not None:
            try:
                metric_by_id[int(m["id"])] = m
            except (TypeError, ValueError):
                continue

    def _coord(qid: int) -> Optional[List[float]]:
        c = coords.get(str(qid), coords.get(qid))
        if isinstance(c, (list, tuple)) and len(c) == 2:
            try:
                return [float(c[0]), float(c[1])]
            except (TypeError, ValueError):
                return None
        return None

    qubits_info: Dict[str, Dict[str, Any]] = {}
    for qid in range(int(nqubits)):
        m = metric_by_id.get(qid, {})
        try:
            fidelity = float(m.get("xeb_fidelity", 1.0))
        except (TypeError, ValueError):
            fidelity = 1.0
        entry: Dict[str, Any] = {"fidelity": fidelity}
        coord = _coord(qid)
        if coord is not None:
            entry["coordinate"] = coord
        # Readout confusion inputs, used to prime the readout cache.
        if "measure_f0" in m:
            try:
                entry["readout_f0"] = float(m["measure_f0"])
            except (TypeError, ValueError):
                pass
        if "measure_f1" in m:
            try:
                entry["readout_f1"] = float(m["measure_f1"])
            except (TypeError, ValueError):
                pass
        qubits_info[f"Q{qid}"] = entry

    couplers_info: Dict[str, Dict[str, Any]] = {}
    for idx, pair in enumerate(coupling_map):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            continue
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        couplers_info[f"C{idx}"] = {"qubits_index": [a, b], "fidelity": 1.0}

    supported = cfg.get("supported_measurement_types")
    supported = list(supported) if isinstance(supported, list) else []

    return {
        "chip_name": chip_name,
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": {
            "two_qubit_gate_basis": "cz",
            "nqubits_available": int(nqubits),
            "error_rate_2q": 0.0,
            "one_qubit_gate_length": 3e-8,
            "two_qubit_gate_length": 6e-8,
            "supported_measurement_types": supported,
        },
        "priority_qubits": None,
    }


def _two_state_readout(chip_info: Dict[str, Any]) -> bool:
    """True when the backend supports ONLY ``measure2`` (2-state readout)."""
    try:
        supported = set(chip_info["global_info"].get("supported_measurement_types", []) or [])
    except Exception:
        return False
    return "measure" not in supported and "measure2" in supported


# ---------------------------------------------------------------------------
# Readout-cache priming (skip on-hardware readout calibration)
# ---------------------------------------------------------------------------

def prime_readout_cache(backend, chip_name: str) -> None:
    """Write vendor readout confusion matrices into fieldqkit's cache.

    Builds each qubit's 2x2 confusion matrix ``[[f0, 1-f1], [1-f0, f1]]``
    from ``measure_f0`` / ``measure_f1`` and stores it under
    ``api/.cache/readout_<chip>.json`` with a fresh timestamp. This makes
    :meth:`ReadoutCalibrationManager.calibrate_readout` treat every qubit
    as fresh-cached and skip submitting calibration circuits entirely.
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from ...calibration._cache import cache_file, save_timestamped_payload

    chip_info = getattr(backend, "chip_info", None)
    if not isinstance(chip_info, dict):
        return
    qubits_info = chip_info.get("qubits_info", {})
    if not isinstance(qubits_info, dict):
        return

    now = datetime.now(timezone.utc).isoformat()
    timestamps: Dict[str, str] = {}
    per_qubit: Dict[str, List[List[float]]] = {}
    for key, entry in qubits_info.items():
        if not isinstance(entry, dict) or "readout_f0" not in entry or "readout_f1" not in entry:
            continue
        try:
            qid = int(str(key).lstrip("Q"))
        except (TypeError, ValueError):
            continue
        f0 = float(entry["readout_f0"])
        f1 = float(entry["readout_f1"])
        # M[measure, prepare]: column = prepared state, row = measured outcome.
        mat = [[f0, 1.0 - f1], [1.0 - f0, f1]]
        per_qubit[str(qid)] = mat
        timestamps[str(qid)] = now

    if not per_qubit:
        return

    cache_dir = Path(__file__).resolve().parents[1] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file(cache_dir, stem="readout", chip_name=chip_name)
    save_timestamped_payload(path, payload_key="per_qubit_confusion", timestamps=timestamps, payload=per_qubit)
    logger.info("primed readout cache for %s (%d qubits) at %s", chip_name, len(per_qubit), path)


# ---------------------------------------------------------------------------
# fieldqkit QuantumCircuit -> LogicalQubit instruction-list IR
# ---------------------------------------------------------------------------

def _su2_params(mat: np.ndarray) -> List[float]:
    """Flatten a 2x2 unitary into the native ``su2`` param order."""
    m = np.asarray(mat, dtype=complex)
    return [
        float(m[0, 0].real), float(m[0, 0].imag),
        float(m[0, 1].real), float(m[0, 1].imag),
        float(m[1, 0].real), float(m[1, 0].imag),
        float(m[1, 1].real), float(m[1, 1].imag),
    ]


def circuit_to_lqcloud_command(qc, *, two_state_readout: bool = False, shots: int = 1024):
    """Convert a transpiled fieldqkit circuit to a ``run_circuit`` command.

    The circuit must already be transpiled to the ``cz`` basis on physical
    (adjacent) qubits — the platform does not route. Physical qubits are
    compacted to a dense ``0..k-1`` logical range and pinned back via
    ``initial_layout``; every single-qubit gate becomes one ``su2``.

    Returns ``(command_dict, effective_shots)``.
    """
    from ...circuit.quantumcircuit_helpers import (
        one_qubit_gates_available,
        one_qubit_parameter_gates_available,
        two_qubit_gates_available,
        three_qubit_gates_available,
        two_qubit_parameter_gates_available,
    )
    from ...circuit.matrix import gate_matrix_dict

    # 1) collect the physical qubits actually used, then compact them.
    used: set = set()
    for g in qc.gates:
        name = g[0]
        if name in one_qubit_gates_available:
            used.add(g[1])
        elif name in one_qubit_parameter_gates_available:
            used.add(g[-1])
        elif name in two_qubit_gates_available:
            used.update((g[1], g[2]))
        elif name in two_qubit_parameter_gates_available:
            used.update((g[2], g[3]))
        elif name in three_qubit_gates_available:
            used.update((g[1], g[2], g[3]))
        elif name == "measure":
            used.update(g[1])
        elif name == "reset":
            used.add(g[1])
        elif name == "barrier":
            used.update(g[1])
        elif name == "delay":
            used.update(g[2])
    used_sorted: List[int] = sorted(int(q) for q in used)
    dense = {q: i for i, q in enumerate(used_sorted)}
    k = len(used_sorted)
    if k == 0:
        raise ValueError("cannot submit an empty circuit to LogicalQubit")

    instrs: List[Dict[str, Any]] = []
    # 2) SDK-parity auto-reset: one reset per (dense) qubit up front.
    for i in range(k):
        instrs.append({"name": "reset", "qubits": [i], "clbits": [], "params": []})

    last_was_barrier = False
    n_clbits = 0
    measured: set = set()

    for g in qc.gates:
        name = g[0]
        if name == "id":
            continue
        if name in one_qubit_gates_available:
            if name not in gate_matrix_dict:
                raise NotImplementedError(f"no matrix for single-qubit gate '{name}'")
            instrs.append({"name": "su2", "qubits": [dense[g[1]]], "clbits": [],
                           "params": _su2_params(gate_matrix_dict[name])})
            last_was_barrier = False
        elif name in one_qubit_parameter_gates_available:
            factory = gate_matrix_dict.get(name)
            if factory is None:
                raise NotImplementedError(f"no matrix factory for gate '{name}'")
            params = list(g[1:-1])
            instrs.append({"name": "su2", "qubits": [dense[g[-1]]], "clbits": [],
                           "params": _su2_params(factory(*params))})
            last_was_barrier = False
        elif name in two_qubit_gates_available:
            if name != "cz":
                raise NotImplementedError(
                    f"two-qubit gate '{name}' must be translated to 'cz' before submission"
                )
            instrs.append({"name": "cz", "qubits": [dense[g[1]], dense[g[2]]], "clbits": [], "params": []})
            last_was_barrier = False
        elif name == "barrier":
            instrs.append({"name": "barrier", "qubits": [dense[q] for q in g[1]], "clbits": [], "params": []})
            last_was_barrier = True
        elif name == "reset":
            instrs.append({"name": "reset", "qubits": [dense[g[1]]], "clbits": [], "params": []})
            last_was_barrier = False
        elif name == "delay":
            duration_ns = round(float(g[1]) * 1e9)
            for q in g[2]:
                instrs.append({"name": "delay", "qubits": [dense[q]], "clbits": [], "params": [duration_ns]})
            last_was_barrier = False
        elif name == "measure":
            qubits = list(g[1])
            cbits = list(g[2])
            # Barrier before the readout window (hardware scheduling boundary).
            if not last_was_barrier:
                instrs.append({"name": "barrier", "qubits": [dense[q] for q in qubits], "clbits": [], "params": []})
            for q, c in zip(qubits, cbits):
                dq = dense[q]
                if two_state_readout:
                    instrs.append({"name": "x21", "qubits": [dq], "clbits": [], "params": []})
                instrs.append({"name": "measure", "qubits": [dq], "clbits": [int(c)], "params": []})
                measured.add(dq)
                n_clbits = max(n_clbits, int(c) + 1)
            last_was_barrier = False
        elif name in three_qubit_gates_available or name in two_qubit_parameter_gates_available:
            raise NotImplementedError(f"gate '{name}' must be decomposed before submission")
        else:
            raise NotImplementedError(f"unsupported gate '{name}' for LogicalQubit")

    n_measured = len(measured)
    result_format = "memory" if n_measured >= _PER_SHOT_QUBIT_THRESHOLD else "counts"
    eff_shots = int(shots)
    if result_format == "memory" and eff_shots > _PER_SHOT_MAX_SHOTS:
        logger.warning(
            "LogicalQubit: %d measured qubits -> per-shot memory mode; shots clipped %d -> %d",
            n_measured, eff_shots, _PER_SHOT_MAX_SHOTS,
        )
        eff_shots = _PER_SHOT_MAX_SHOTS

    command = {
        "action": "run_circuit",
        "circuit": {
            "n_qubits": k,
            "n_clbits": n_clbits,
            "instructions": instrs,
            "shots": eff_shots,
            "result_format": result_format,
            "initial_layout": used_sorted,
        },
    }
    return command, eff_shots


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class LogicalQubitBackendAdapter(BackendAdapter):
    provider = "logicalqubit"
    default_hardware_name = "QZ02"

    def __init__(self, *, machine_name: Optional[str] = None, token: Optional[str] = None) -> None:
        self._token = token or _get_logicalqubit_token()
        self._machine_name = machine_name
        self._platform = LogicalQubitPlatform(token=self._token)

    def resolve_backend(self, *, num_qubits: int, prefer_hardware=None) -> ResolvedBackend:
        resolved = super().resolve_backend(num_qubits=num_qubits, prefer_hardware=prefer_hardware)
        # Prime the readout cache from vendor f0/f1 so client-side readout
        # mitigation reuses it instead of running on-hardware calibration.
        try:
            prime_readout_cache(resolved.backend, resolved.hardware_name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to prime LogicalQubit readout cache: %s", exc)
        return resolved


class LogicalQubitTaskAdapter(TaskAdapter):
    provider = "logicalqubit"
    native_ir = True

    def __init__(self, *, client: Any, token: Optional[str] = None) -> None:
        self._client = client
        self._token = token or _get_logicalqubit_token()

    def submit_native_circuit(self, submit_request: NativeCircuitSubmitRequest,
                              backend: ResolvedBackend) -> ProviderTaskHandle:
        platform: LogicalQubitPlatform = backend.metadata.get("platform_obj")
        if platform is None:
            platform = LogicalQubitPlatform(token=self._token)
        two_state = _two_state_readout(getattr(backend.backend, "chip_info", {}) or {})
        command, _ = circuit_to_lqcloud_command(
            submit_request.circuit, two_state_readout=two_state, shots=submit_request.shots,
        )
        task_id = platform.submit(command, submit_request.chip_name)
        return ProviderTaskHandle(provider=self.provider, task_id=task_id,
                                  payload={"platform_obj": platform})

    def query_status(self, handle: ProviderTaskHandle) -> str:
        platform: LogicalQubitPlatform = handle.payload.get("platform_obj")
        if platform is None:
            platform = LogicalQubitPlatform(token=self._token)
        return _STATUS_MAP.get(platform.status(handle.task_id), "Running")

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        platform: LogicalQubitPlatform = handle.payload.get("platform_obj")
        if platform is None:
            platform = LogicalQubitPlatform(token=self._token)
        return {"count": platform.result_counts(handle.task_id)}

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        platform: LogicalQubitPlatform = handle.payload.get("platform_obj")
        if platform is None:
            return
        try:
            platform._request("POST", f"/api/v1/jobs/{handle.task_id}/cancel")
        except Exception as exc:  # pragma: no cover
            logger.warning("LogicalQubit cancel failed for %s: %s", handle.task_id, exc)
