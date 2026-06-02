"""Origin Quantum (本源量子) cloud provider integration.

This adapter targets the OriginQ public cloud (https://qcloud.originqc.com.cn/)
and uses the official ``pyqpanda3.qcloud`` SDK as the transport layer.

Why we depend on pyqpanda3:
    The OriginQ cloud uses a proprietary HTTP/binary protocol whose details
    are encapsulated entirely inside the compiled C++ extension
    ``pyqpanda3.qcloud.qcloud.cpXXX-...pyd``.  In particular, neither the
    REST endpoint contract nor the QProg→QASM serializer are exposed in
    pure-Python source.  Re-implementing the protocol from scratch would
    require reverse-engineering and would be extremely fragile across
    pyqpanda3 releases.  Therefore this module **calls** pyqpanda3 as our
    cloud SDK, while the rest of the integration (chip-info normalization,
    task submission flow, status mapping, result shaping) is implemented in
    this repo to match the unified ``BackendAdapter``/``TaskAdapter``
    contract used by the other providers (Quafu, TianYan, GuoDun, Tencent).

OpenQASM 2.0 → QProg conversion is provided by
``pyqpanda3.intermediate_compiler.convert_qasm_string_to_qprog`` (also a
compiled C++ implementation).  Source:
``D:\\Programs\\anaconda\\envs\\quantum\\Lib\\site-packages\\pyqpanda3\\intermediate_compiler\\intermediate_compiler.pyi``
(``pyqpanda3 >= 0.3``).

Endpoints exercised (all via pyqpanda3 SDK):
    QCloudService(api_key, url).backends()      -> dict[str, bool]
    QCloudService.backend(name).chip_info()     -> ChipInfo
    QCloudBackend.run(prog, shots, options)     -> QCloudJob
    QCloudJob.status() / job_id() / result()    -> JobStatus / str / QCloudResult
    QCloudResult.get_counts() / job_status() / error_message()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..backend import BackendAdapter, ResolvedBackend
from ..task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORIGIN_DEFAULT_URL = "http://pyqanda-admin.qpanda.cn"
"""Default OriginQ cloud admin endpoint used by pyqpanda3.qcloud."""

ORIGIN_HARDWARE_NAMES = {
    "PQPUMESH8",
    "WK_C180",
    "HanYuan_01",
}
"""Canonical OriginQ chip names observed from the live ``backends()`` listing."""

ORIGIN_SIMULATOR_NAMES = {"full_amplitude", "partial_amplitude", "single_amplitude"}
"""Server-side simulator backend names exposed by the OriginQ cloud."""

# ---------------------------------------------------------------------------
# Lazy SDK loader
# ---------------------------------------------------------------------------

def _import_qcloud():
    """Import :mod:`pyqpanda3.qcloud`.

    Returns:
        Module object.

    Raises:
        ImportError: if pyqpanda3 is not installed.
    """
    try:
        from pyqpanda3 import qcloud  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "pyqpanda3 is required for the OriginQ provider; "
            "install via `pip install pyqpanda3`"
        ) from exc
    return qcloud


def _import_qasm_to_qprog():
    """Return ``pyqpanda3.intermediate_compiler.convert_qasm_string_to_qprog``.

    The OriginQ cloud SDK consumes ``QProg`` objects, not OpenQASM text.  This
    helper bridges our unified OpenQASM-centric submit interface to pyqpanda3.

    Raises:
        ImportError: if pyqpanda3 is not installed.
    """
    try:
        from pyqpanda3.intermediate_compiler import convert_qasm_string_to_qprog  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "pyqpanda3.intermediate_compiler is required for QASM->QProg "
            "conversion on the OriginQ provider"
        ) from exc
    return convert_qasm_string_to_qprog


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _get_origin_token() -> str:
    """Load Origin api token from credentials store.

    Returns:
        API token string (non-empty).

    Raises:
        ValueError: if no credential is configured.
    """
    from ..platform_credentials import get_origin_api_token  # noqa: PLC0415
    return get_origin_api_token()


def _ensure_token(token: Optional[str] = None) -> str:
    """Return an api_token, falling back to credentials store.

    Args:
        token: explicit token (overrides yaml/env); ``None`` to look up.

    Returns:
        Non-empty token string.

    Raises:
        ValueError: if neither argument nor stored credential is available.
    """
    tok = token or _get_origin_token()
    if not tok:
        raise ValueError("origin API token cannot be empty")
    return tok


# ---------------------------------------------------------------------------
# Chip-info normalization
# ---------------------------------------------------------------------------

def _normalize_chip_info(chip, chip_name: str) -> Dict[str, Any]:
    """Convert a pyqpanda3 ``ChipInfo`` into the unified ``Backend.chip_info`` dict.

    The unified dict layout matches the format consumed by
    :class:`fieldqkit.api.backend.Backend` (see other providers like
    ``tencent._load_tencent_chip_info``).

    Args:
        chip: pyqpanda3 ``ChipInfo`` instance.
        chip_name: hardware identifier.

    Returns:
        Normalized ``chip_info`` dictionary.
    """
    qubits_info: Dict[str, Dict[str, Any]] = {}
    for sq in chip.single_qubit_info():
        try:
            qid = int(sq.get_qubit_id())
        except Exception:
            continue
        qubits_info[f"Q{qid}"] = {
            "fidelity": float(sq.get_single_gate_fidelity()),
            "readout_fidelity": float(sq.get_readout_fidelity()),
            "frequency": float(sq.get_frequency()),
            "T1": float(sq.get_t1()),
            "T2": float(sq.get_t2()),
        }

    couplers_info: Dict[str, Dict[str, Any]] = {}
    seen_pairs: set = set()
    for idx, dq in enumerate(chip.double_qubits_info()):
        pair = dq.get_qubits()
        if not pair or len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        # de-duplicate undirected pairs
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        couplers_info[f"C{idx}"] = {
            "qubits_index": [a, b],
            "fidelity": float(dq.get_fidelity()),
        }

    # Some chips do not surface coupler entries via double_qubits_info() but
    # only via the topology graph; back-fill so that the routing layer has
    # something to work with.
    if not couplers_info:
        try:
            edges = chip.get_chip_topology()
        except Exception:
            edges = []
        for idx, edge in enumerate(edges or []):
            if not edge or len(edge) != 2:
                continue
            a, b = int(edge[0]), int(edge[1])
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            couplers_info[f"C{idx}"] = {
                "qubits_index": [a, b],
                "fidelity": 1.0,
            }

    try:
        nqubits = int(chip.qubits_num())
    except Exception:
        nqubits = len(qubits_info)

    basic_gates = []
    try:
        basic_gates = list(chip.get_basic_gates())
    except Exception:
        pass
    two_q_basis = "cz"
    for g in basic_gates:
        if str(g).lower() == "cz":
            two_q_basis = "cz"
            break

    global_info = {
        "two_qubit_gate_basis": two_q_basis,
        "nqubits_available": nqubits,
        "basic_gates": basic_gates,
        "single_gate_timing_ns": _safe_int(getattr(chip, "get_single_gate_timing", lambda: None)()),
        "double_gate_timing_ns": _safe_int(getattr(chip, "get_double_gate_timing", lambda: None)()),
    }

    return {
        "chip_name": chip_name,
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": global_info,
    }


def _safe_int(value: Any) -> Optional[int]:
    """Best-effort int conversion; ``None`` on failure."""
    try:
        return int(value)
    except Exception:
        return None


def load_origin_chip_info(chip_name: str, token: Optional[str] = None, *, url: str = ORIGIN_DEFAULT_URL) -> Dict[str, Any]:
    """Fetch and normalize chip info for an OriginQ device.

    Args:
        chip_name: target chip identifier (e.g. ``"PQPUMESH8"``).
        token: optional explicit api_token.
        url: cloud admin URL.

    Returns:
        Normalized ``chip_info`` dict consumed by ``Backend``.
    """
    qcloud = _import_qcloud()
    tok = _ensure_token(token)
    service = qcloud.QCloudService(api_key=tok, url=url)
    backend = service.backend(chip_name)
    return _normalize_chip_info(backend.chip_info(), chip_name)


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

_STATUS_MAP: Dict[str, str] = {
    "FINISHED": "Finished",
    "FAILED": "Failed",
    "WAITING": "Running",
    "QUEUING": "Running",
    "COMPUTING": "Running",
}


def _map_status(job_status_obj: Any) -> str:
    """Map a pyqpanda3 ``JobStatus`` enum value to the unified status string."""
    name = getattr(job_status_obj, "name", str(job_status_obj))
    return _STATUS_MAP.get(str(name).upper(), "Running")


# ---------------------------------------------------------------------------
# Platform / Adapter classes
# ---------------------------------------------------------------------------

class OriginPlatform:
    """Thin wrapper around ``pyqpanda3.qcloud.QCloudService`` for our adapters."""

    def __init__(self, token: Optional[str] = None, url: str = ORIGIN_DEFAULT_URL) -> None:
        """Create a connected cloud service handle.

        Args:
            token: optional explicit api_token; falls back to credentials store.
            url: cloud admin URL.
        """
        self._token = _ensure_token(token)
        self._url = url
        qcloud = _import_qcloud()
        self._qcloud = qcloud
        self._service = qcloud.QCloudService(api_key=self._token, url=url)
        self._backend_cache: Dict[str, Any] = {}
        self._machine_name: Optional[str] = None

    # --- introspection ---------------------------------------------------

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Return online hardware rows (excluding server-side simulators).

        Returns:
            List of normalized hardware description dicts.
        """
        rows: List[Dict[str, Any]] = []
        try:
            backends = self._service.backends() or {}
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning("origin backends() failed: %s", exc)
            return rows
        for name, online in backends.items():
            if name in ORIGIN_SIMULATOR_NAMES:
                continue
            rows.append({
                "provider": "origin",
                "hardware_name": str(name),
                "queue_length": None,
                "status": "online" if online else "offline",
                "is_toll": None,
                "raw": {"backend_name": str(name), "online": bool(online)},
            })
        return rows

    def set_machine(self, name: str) -> None:
        """Cache the active machine name for downstream task submissions."""
        self._machine_name = str(name)

    def get_backend(self, chip_name: str):
        """Return (and cache) a pyqpanda3 ``QCloudBackend`` for *chip_name*.

        Args:
            chip_name: hardware identifier.

        Returns:
            pyqpanda3 ``QCloudBackend`` instance.
        """
        cached = self._backend_cache.get(chip_name)
        if cached is not None:
            return cached
        be = self._service.backend(chip_name)
        self._backend_cache[chip_name] = be
        return be

    # --- task submission -------------------------------------------------

    def _build_options(self):
        """Build a default ``QCloudOptions`` for hardware submissions.

        We disable server-side mapping/optimization so that the result
        bitstrings line up with the qubits chosen by the local transpiler.
        Server-side amend (readout post-processing) is also disabled to
        return raw counts.
        """
        opts = self._qcloud.QCloudOptions()
        try:
            opts.set_amend(False)
            opts.set_mapping(False)
            opts.set_optimization(False)
            opts.set_is_prob_counts(False)
        except Exception:
            pass
        return opts

    def submit_task(
        self,
        source: str,
        device_name: str,
        shots: int = 1024,
    ) -> str:
        """Submit an OpenQASM 2.0 program; return the job id.

        Args:
            source: OpenQASM 2.0 program string.
            device_name: target chip name.
            shots: number of measurement shots.

        Returns:
            Job ID string.
        """
        convert = _import_qasm_to_qprog()
        prog = convert(source)
        backend = self.get_backend(device_name)
        opts = self._build_options()
        # Overload #4: run(prog, shots, options, ...)
        try:
            job = backend.run(prog, int(shots), opts)
        except RuntimeError as exc:
            # Fall back to "qubits" overload for amplitude-only simulators
            # which reject the options-based overload.
            if "Full_AMPLITUDE" in str(exc) or "AMPLITUDE" in str(exc):
                job = backend.run(prog, int(shots))
            else:
                raise
        return str(job.job_id())

    def query_task_state(self, task_id: str, device_name: str) -> str:
        """Query unified status for a previously submitted job."""
        return self._fetch_status(task_id)

    def fetch_task_result(self, task_id: str, device_name: str) -> Dict[str, int]:
        """Return measurement counts for a finished job.

        Args:
            task_id: job id from :meth:`submit_task`.
            device_name: target chip name (kept for interface symmetry).

        Returns:
            ``dict`` mapping bitstring → count.

        Raises:
            RuntimeError: if the cloud reports a job-level failure.
        """
        job = self._qcloud.QCloudJob(str(task_id))
        result = job.result()
        status = result.job_status()
        if getattr(status, "name", str(status)).upper() != "FINISHED":
            err = result.error_message()
            raise RuntimeError(f"OriginQ job {task_id} not finished: status={status} err={err}")
        counts = result.get_counts()
        # OriginQ cloud returns bitstrings in little-endian order
        # (rightmost char = q[0]), but this package's convention is
        # big-endian (leftmost char = q[0], cf. core.utils.get_samples).
        # Reverse each bitstring so downstream code interprets qubits
        # consistently.  Empirically verified with an asymmetric probe
        # circuit (X q[0] only on PQPUMESH8 → dominant '001' became
        # '100' after reversal).
        return {str(k)[::-1]: int(v) for k, v in dict(counts).items()}

    # --- helpers ---------------------------------------------------------

    def _fetch_status(self, task_id: str) -> str:
        """Resolve a job id into a unified status string."""
        job = self._qcloud.QCloudJob(str(task_id))
        try:
            status = job.status()
        except Exception as exc:  # pragma: no cover
            logger.warning("origin status query failed: %s", exc)
            return "Running"
        return _map_status(status)


class OriginBackendAdapter(BackendAdapter):
    """Backend adapter for the OriginQ cloud."""

    provider = "origin"
    default_hardware_name = "WK_C180"

    def __init__(
        self,
        *,
        machine_name: Optional[str] = None,
        token: Optional[str] = None,
        url: str = ORIGIN_DEFAULT_URL,
    ) -> None:
        """Initialize the adapter.

        Args:
            machine_name: optional default machine name.
            token: optional api_token (falls back to credential store).
            url: cloud admin URL.
        """
        self._token = token or _get_origin_token()
        self._machine_name = machine_name
        self._platform = OriginPlatform(token=self._token, url=url)
        if machine_name:
            self._platform.set_machine(machine_name)


class OriginTaskAdapter(TaskAdapter):
    """Task adapter that routes OpenQASM submissions to OriginQ via pyqpanda3."""

    provider = "origin"

    def __init__(
        self,
        *,
        client: Any,
        token: Optional[str] = None,
        url: str = ORIGIN_DEFAULT_URL,
    ) -> None:
        """Initialize the adapter.

        Args:
            client: ``QuantumHardwareClient`` reference (kept for symmetry).
            token: optional api_token.
            url: cloud admin URL.
        """
        self._client = client
        self._token = token or _get_origin_token()
        self._url = url
        _ensure_token(self._token)

    def _platform_for(self, backend: ResolvedBackend) -> OriginPlatform:
        """Return the platform handle bound to *backend* (constructing on demand)."""
        platform_obj = backend.metadata.get("platform_obj") if backend is not None else None
        if isinstance(platform_obj, OriginPlatform):
            return platform_obj
        return OriginPlatform(token=self._token, url=self._url)

    def submit_openqasm(self, submit_request: OpenQasmSubmitRequest, backend: ResolvedBackend) -> ProviderTaskHandle:
        """Submit a single OpenQASM program.

        Args:
            submit_request: unified submit descriptor.
            backend: resolved backend for this submission.

        Returns:
            ``ProviderTaskHandle`` carrying the job id and platform refs.
        """
        platform_obj = self._platform_for(backend)
        task_id = platform_obj.submit_task(
            source=submit_request.qasm,
            device_name=submit_request.chip_name,
            shots=submit_request.shots,
        )
        payload = {
            "platform_obj": platform_obj,
            "device_name": submit_request.chip_name,
        }
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=str(task_id),
            payload=payload,
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        """Return the unified status (``Finished``/``Failed``/``Running``)."""
        platform_obj: Optional[OriginPlatform] = handle.payload.get("platform_obj")
        device_name = handle.payload.get("device_name", "")
        if platform_obj is None:
            platform_obj = OriginPlatform(token=self._token, url=self._url)
        return platform_obj.query_task_state(handle.task_id, device_name)

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        """Block on a finished task and return a ``{"count": ...}`` dict."""
        platform_obj: Optional[OriginPlatform] = handle.payload.get("platform_obj")
        device_name = handle.payload.get("device_name", "")
        if platform_obj is None:
            platform_obj = OriginPlatform(token=self._token, url=self._url)
        counts = platform_obj.fetch_task_result(handle.task_id, device_name)
        return {"count": counts}

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        """OriginQ SDK does not expose a cancel endpoint; emit a warning."""
        logger.warning("OriginQ provider does not support task cancellation (job_id=%s)", handle.task_id)
