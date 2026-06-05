"""Tencent quantum cloud provider direct REST integration.

SPDX-License-Identifier: Apache-2.0
Modified from TensorCircuit, Copyright (c) The TensorCircuit Authors.
This file has been altered from the original. See THIRD_PARTY_NOTICES.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from ..backend import BackendAdapter, ResolvedBackend, as_int_or_none
from ..task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level REST helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://quantum.tencent.com/cloud/quk/"
_MAX_RETRIES = 5
_RETRY_TIMEOUT = 12  # seconds per request


def _auth_headers(token: str) -> Dict[str, str]:
    """Build HTTP headers with Bearer token for Tencent API requests.

    Args:
        token (*str*): API authentication token.

    Returns:
        Headers ``dict`` with ``Authorization`` and ``user-agent`` keys.
    """
    return {
        "Authorization": f"Bearer {token}",
        "user-agent": "Mozilla/5.0",
    }


def _post_json(endpoint: str, *, token: str, json: Any = None) -> Dict[str, Any]:
    """POST to Tencent cloud API with retry, return parsed JSON.

    Args:
        endpoint (*str*): API endpoint path (appended to ``_BASE_URL``).
        token (*str*): API authentication token.
        json (*Any*): JSON-serialisable request body. Defaults to ``None``.

    Returns:
        Parsed JSON response ``dict``.

    Raises:
        RuntimeError: If the request fails after all retries.
        ValueError: If the server returns an unexpected non-dict response.
    """
    url = _BASE_URL + endpoint
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                json=json or {},
                headers=_auth_headers(token),
                timeout=_RETRY_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("unexpected non-dict response from server")
            if "err" in data:
                raise ValueError(data["err"])
            return data
        except (
            requests.exceptions.RequestException,
            ValueError,
            ConnectionResetError,
        ) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(0.5 * attempt)
                continue
    raise RuntimeError(f"Tencent API request to {endpoint} failed after {_MAX_RETRIES} retries") from last_exc


# ---------------------------------------------------------------------------
# Token helper
# ---------------------------------------------------------------------------

def _get_tencent_token() -> str:
    """Get Tencent API token from credentials.

    Returns:
        API token string.
    """
    from ..platform_credentials import get_tencent_api_token
    return get_tencent_api_token()


def _ensure_token(token: Optional[str] = None) -> str:
    """Return a valid API token, falling back to stored credentials.

    Args:
        token (*Optional[str]*): API authentication token. Defaults to ``None``.

    Returns:
        Non-empty API token string.

    Raises:
        ValueError: tencent API token cannot be empty
    """
    tok = token or _get_tencent_token()
    if not tok:
        raise ValueError("tencent API token cannot be empty")
    return tok


# ---------------------------------------------------------------------------
# Device / chip info queries
# ---------------------------------------------------------------------------

def _list_devices(token: str) -> List[str]:
    """Return device id strings from Tencent cloud.

    Args:
        token (*str*): API authentication token.

    Returns:
        List of device ID strings.
    """
    r = _post_json("device/find", token=token)
    return [d["id"] for d in r.get("devices", [])]


def _get_device_properties(device_name: str, token: str) -> Dict[str, Any]:
    """Fetch raw device properties (bits, links, etc.).

    Args:
        device_name (*str*): Name of the target Tencent device.
        token (*str*): API authentication token.

    Returns:
        Device properties ``dict`` with normalised ``bits`` and ``links``.

    Raises:
        ValueError: No device with the name: {device_name}
    """
    r = _post_json("device/detail", token=token, json={"id": device_name})
    if "device" not in r:
        raise ValueError(f"No device with the name: {device_name}")
    props = r["device"]
    # Normalize links/bits into dicts keyed by tuple/qubit-id
    if "links" in props:
        props["links"] = {
            (link["A"], link["B"]): link for link in props["links"]
        }
    if "bits" in props:
        props["bits"] = {bit["Qubit"]: bit for bit in props["bits"]}
    return props


def _load_tencent_chip_info(chip_name: str, token: Optional[str] = None) -> Optional[dict]:
    """Load chip info from Tencent Cloud and normalize to unified format.

    Args:
        chip_name (*str*): Name of the target chip.
        token (*Optional[str]*): API authentication token. Defaults to ``None``.

    Returns:
        Unified chip-info ``dict`` with ``qubits_info``, ``couplers_info``, and ``global_info`` keys.
    """
    tok = _ensure_token(token)
    props = _get_device_properties(chip_name, tok)

    qubits_info: Dict[str, Dict[str, Any]] = {}
    if "bits" in props:
        for key, bit in props["bits"].items():
            qid = int(key) if isinstance(key, (int, str)) else key
            sq_err = float(bit.get("SingleQubitErrRate", 0.001))
            fidelity = 1.0 - sq_err
            qubits_info[f"Q{qid}"] = {
                "fidelity": fidelity,
                "T1": bit.get("T1"),
                "T2": bit.get("T2"),
                "frequency": bit.get("Freqency"),
                "readout_f0_err": bit.get("ReadoutF0Err"),
                "readout_f1_err": bit.get("ReadoutF1Err"),
            }

    couplers_info: Dict[str, Dict[str, Any]] = {}
    if "links" in props:
        idx = 0
        for link_key, link in props["links"].items():
            a = int(link["A"])
            b = int(link["B"])
            cz_err = float(link.get("CZErrRate", 0.01))
            fidelity = 1.0 - cz_err
            couplers_info[f"C{idx}"] = {
                "qubits_index": [a, b],
                "fidelity": fidelity,
            }
            idx += 1

    nqubits = int(props.get("qubits", len(qubits_info)))
    global_info = {
        "two_qubit_gate_basis": "cz",
        "nqubits_available": nqubits,
        "state": props.get("state", "unknown"),
    }

    return {
        "chip_name": chip_name,
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": global_info,
    }


# ---------------------------------------------------------------------------
# Task submission / query helpers
# ---------------------------------------------------------------------------

def _strip_barrier(qasm: str) -> str:
    """Remove barrier instructions that tencent QOS cannot parse.

    Args:
        qasm (*str*): OpenQASM source string.

    Returns:
        OpenQASM string with barrier lines removed.
    """
    return "\n".join(
        line for line in qasm.split("\n")
        if not line.strip().startswith("barrier")
    )


def _submit_task(
    *,
    token: str,
    device_name: str,
    source: str,
    shots: int = 1024,
    qos_option: int = 2,
) -> str:
    """Submit an OpenQASM task, return the task id.

    Args:
        token (*str*): API authentication token.
        device_name (*str*): Target Tencent device identifier.
        source (*str*): OpenQASM source string.
        shots (*int*): Number of measurement shots. Defaults to ``1024``.
        qos_option (*int*): QOS compilation level. Defaults to ``2``.

    Returns:
        Task ID string.

    Raises:
        RuntimeError: task submission failed — no task id returned
    """
    device_str = f"{device_name}?o={qos_option}"
    payload = {
        "device": device_str,
        "shots": shots,
        "source": _strip_barrier(source),
        "version": "1",
        "lang": "OPENQASM",
        "prior": 1,
    }
    r = _post_json("task/submit", token=token, json=payload)
    tasks = r.get("tasks", [])
    for t in tasks:
        if "err" in t or "id" not in t:
            msg = t.get("err", "unknown submission error")
            logger.warning("task submission warning: %s", msg)
        else:
            return t["id"]
    raise RuntimeError("task submission failed — no task id returned")


def _get_task_detail(task_id: str, token: str) -> Dict[str, Any]:
    """Fetch raw task detail dict (state, result, etc.).

    Args:
        task_id (*str*): Task identifier.
        token (*str*): API authentication token.

    Returns:
        Task detail ``dict`` including ``state`` and ``results``.
    """
    r = _post_json("task/detail", token=token, json={"id": task_id})
    task_data = r.get("task", {})
    # Normalize results
    if "result" in task_data:
        result = task_data["result"]
        task_data["results"] = result.get("counts", result) if isinstance(result, dict) else result
    return task_data


def _query_task_state(task_id: str, token: str) -> str:
    """Return the current state string for a submitted task.

    Args:
        task_id (*str*): Task identifier.
        token (*str*): API authentication token.

    Returns:
        State string (e.g. ``"completed"``, ``"failed"``, ``"pending"``).
    """
    return _get_task_detail(task_id, token).get("state", "pending")


def _fetch_task_results_blocking(
    task_id: str,
    token: str,
    *,
    poll_interval: float = 0.5,
    timeout: float = 600.0,
) -> Dict[str, int]:
    """Poll until task completes, then return counts dict.

    Args:
        task_id (*str*): Task identifier.
        token (*str*): API authentication token.
        poll_interval (*float*): Base seconds between status polls. Defaults to ``0.5``.
        timeout (*float*): Hard timeout in seconds. Defaults to ``600.0``.

    Returns:
        Measurement counts ``dict`` mapping bitstrings to counts.

    Raises:
        TimeoutError: Task did not reach a terminal state within ``timeout`` seconds.
        RuntimeError: Task reported a failed state.
    """
    deadline = time.monotonic() + timeout
    tries = 0
    while True:
        detail = _get_task_detail(task_id, token)
        state = detail.get("state", "pending")
        if state == "completed":
            return detail.get("results", {})
        if state == "failed":
            raise RuntimeError(f"Tencent task {task_id} failed: {detail.get('err', '')}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Tencent task {task_id} did not complete within {timeout}s (last state: {state})"
            )
        time.sleep(poll_interval + tries / 10)
        tries += 1


# ---------------------------------------------------------------------------
# Platform / Adapter classes
# ---------------------------------------------------------------------------

class TencentPlatform:
    """Direct REST wrapper for the unified adapter pattern."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize Tencent platform with API token for circuit submission.

        Args:
            token (*Optional[str]*): API authentication token. Defaults to ``None``.
        """
        self._token = _ensure_token(token)

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Poll Tencent Cloud for available quantum device IDs.

        Returns:
            List of hardware description dictionaries.
        """
        device_ids = _list_devices(self._token)
        return [
            {
                "provider": "tencent",
                "hardware_name": name,
                "queue_length": None,
                "status": None,
                "is_toll": None,
                "raw": {"device_id": name},
            }
            for name in device_ids
        ]

    def submit_task(
        self,
        source: str,
        device_name: str,
        shots: int = 1024,
    ) -> str:
        """Submit an OpenQASM circuit to Tencent cloud.

        Args:
            source (*str*): OpenQASM source string.
            device_name (*str*): Target Tencent device identifier.
            shots (*int*): Number of measurement shots. Defaults to ``1024``.

        Returns:
            Task ID string.
        """
        # QOS o=2: gate decomposition only (no routing/mapping).
        # Client-side transpilation already handles routing.
        return _submit_task(
            token=self._token,
            device_name=device_name,
            source=source,
            shots=shots,
            qos_option=2,
        )

    def query_task_state(self, task_id: str, device_name: str) -> str:
        """Return the current task state string.

        Args:
            task_id (*str*): Task identifier.
            device_name (*str*): Device identifier (unused, kept for interface consistency).

        Returns:
            State string (e.g. ``"completed"``, ``"pending"``).
        """
        return _query_task_state(task_id, self._token)

    def fetch_task_result(self, task_id: str, device_name: str, *, timeout: float = 600.0) -> Dict[str, int]:
        """Block until the task completes and return measurement counts.

        Args:
            task_id (*str*): Task identifier.
            device_name (*str*): Device identifier (unused, kept for interface consistency).
            timeout (*float*): Hard timeout in seconds. Defaults to ``600.0``.

        Returns:
            Measurement counts ``dict`` mapping bitstrings to counts.
        """
        return _fetch_task_results_blocking(task_id, self._token, timeout=timeout)


class TencentBackendAdapter(BackendAdapter):
    provider = "tencent"
    default_hardware_name = "tianji_s2"

    def __init__(self, *, machine_name: Optional[str] = None, token: Optional[str] = None) -> None:
        """Initialize Tencent backend adapter with optional machine and authentication token.

        Args:
            machine_name (*Optional[str]*): Identifier of the target quantum machine. Defaults to ``None``.
            token (*Optional[str]*): API authentication token. Defaults to ``None``.
        """
        self._token = token or _get_tencent_token()
        self._machine_name = machine_name
        self._platform = TencentPlatform(token=self._token)


class TencentTaskAdapter(TaskAdapter):
    provider = "tencent"

    def __init__(self, *, client: Any, token: Optional[str] = None) -> None:
        """Initialize Tencent task adapter with quantum hardware client.

        Args:
            client (*Any*): ``QuantumHardwareClient`` instance.
            token (*Optional[str]*): API authentication token. Defaults to ``None``.
        """
        self._client = client
        self._token = token or _get_tencent_token()
        _ensure_token(self._token)

    def submit_openqasm(self, submit_request: OpenQasmSubmitRequest, backend: ResolvedBackend) -> ProviderTaskHandle:
        """Submit an OpenQASM circuit to the Tencent backend and return a task handle.

        Args:
            submit_request (*OpenQasmSubmitRequest*): Submission request descriptor.
            backend (*ResolvedBackend*): Hardware backend descriptor.

        Returns:
            ``ProviderTaskHandle``: Handle for tracking the submitted task.
        """
        platform_obj: TencentPlatform = backend.metadata.get("platform_obj")
        if platform_obj is None:
            platform_obj = TencentPlatform(token=self._token)

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
        """Map Tencent task state to a unified status string.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Returns:
            Unified status string (``"Finished"``, ``"Failed"``, or ``"Running"``).
        """
        platform_obj: TencentPlatform = handle.payload.get("platform_obj")
        device_name = handle.payload.get("device_name", "tianji_s2")
        if platform_obj is None:
            platform_obj = TencentPlatform(token=self._token)

        state = platform_obj.query_task_state(handle.task_id, device_name)
        # Map tencent states → unified states
        state_map = {
            "completed": "Finished",
            "failed": "Failed",
            "pending": "Running",
            "scheduled": "Running",
        }
        return state_map.get(state, "Running")

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        """Fetch measurement counts for a completed Tencent task.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Returns:
            ``dict`` with ``"count"`` key mapping to bitstring counts.
        """
        platform_obj: TencentPlatform = handle.payload.get("platform_obj")
        device_name = handle.payload.get("device_name", "tianji_s2")
        if platform_obj is None:
            platform_obj = TencentPlatform(token=self._token)

        counts = platform_obj.fetch_task_result(handle.task_id, device_name)
        return {"count": counts}

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        """Log a cancellation warning (Tencent API does not support task cancellation).

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.
        """
        # Tencent cloud REST API does not expose a cancel endpoint
        logger.warning("Tencent provider does not support task cancellation")
