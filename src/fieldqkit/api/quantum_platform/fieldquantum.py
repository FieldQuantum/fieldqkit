"""FieldQuantum cloud simulator provider.

Client-side adapters for the FieldQuantum HTTP-based cloud simulator backend
(https://fieldquantum.tech). The service exposes a REST API rooted at
``https://api.fieldquantum.tech/api/v1/fieldquantum`` and requires a bearer
token of the form ``fq_<32hex>`` issued at
``https://fieldquantum.tech/account/api-token/``.

Endpoints used:

    POST /task/run                -> {"task_id": int, "status": "submitted"}
    GET  /task/status/{task_id}   -> {"task_id", "status", "fqd_task_id"}
    GET  /task/result/{task_id}   -> {"task_id", "status", "result"|"error"}

Server-side status values are normalised to the unified client states
``"Running"`` / ``"Finished"`` / ``"Failed"``.

Typical usage::

    client = QuantumHardwareClient()
    result = client.run_auto(
        circuit, "demo", num_qubits=4,
        provider="fieldquantum",
        shots=1024,
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from ..backend import (
    BackendAdapter,
    Backend,
    HardwareProfile,
    ResolvedBackend,
    _build_simulator_chip_info,
    build_simulator_profile,
)
from ..platform_credentials import get_fieldquantum_api_token
from ..task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable server URL (override via env-var for staging / private deploys)
# ---------------------------------------------------------------------------

FIELDQUANTUM_DEFAULT_URL: str = os.environ.get(
    "FIELDQUANTUM_SERVER_URL",
    "https://api.fieldquantum.tech/api/v1/fieldquantum",
)


# ---------------------------------------------------------------------------
# HTTP platform client
# ---------------------------------------------------------------------------

class FieldQuantumPlatform:
    """Thin HTTP client for the FieldQuantum cloud simulator REST API.

    The API root is taken from the module-level :data:`FIELDQUANTUM_DEFAULT_URL`
    (override via the ``FIELDQUANTUM_SERVER_URL`` env var). The bearer token
    (``fq_<32hex>``) is resolved at construction time via
    :func:`get_fieldquantum_api_token` (config file → ``FIELDQUANTUM_API_TOKEN``
    env var).
    """

    def __init__(self) -> None:
        self.base_url = FIELDQUANTUM_DEFAULT_URL.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {get_fieldquantum_api_token()}",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_error(data: Any, fallback: str) -> str:
        """Pull a human-readable error string out of a server JSON body."""
        if isinstance(data, dict):
            for key in ("message", "error"):
                value = data.get(key)
                if value:
                    return str(value)
        return fallback

    def _raise_for_response(self, resp: requests.Response) -> Dict[str, Any]:
        """Parse a JSON response and raise ``RuntimeError`` on server errors.

        The FieldQuantum service signals failure via both HTTP status code
        and a JSON body containing ``"error"`` and (optionally) ``"message"``.
        """
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            msg = self._extract_error(data, fallback=resp.text or resp.reason)
            raise RuntimeError(f"FieldQuantum HTTP {resp.status_code}: {msg}")
        if isinstance(data, dict) and "error" in data and "result" not in data:
            raise RuntimeError(
                f"FieldQuantum server error: {self._extract_error(data, 'unknown')}"
            )
        return data

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Return the (single) virtual chip exposed by the FieldQuantum service.
        """
        return [{
            "provider": "fieldquantum",
            "hardware_name": "fieldquantum_sim",
            "queue_length": 0,
            "status": None,
            "is_toll": None,
            "raw": {},
        }]

    def submit_job(self, payload: Dict[str, Any]) -> str:
        """POST *payload* to ``/task/run`` and return the server task_id.

        Args:
            payload: Request body dict (must include ``"mode"`` key).

        Returns:
            Task ID as a string. (The server issues integer IDs; we keep
            the wider string type for parity with other providers and the
            ``ProviderTaskHandle`` contract.)

        Raises:
            RuntimeError: If the server returns an error body or non-2xx.
        """
        resp = self._session.post(
            f"{self.base_url}/task/run",
            json=payload,
            timeout=300,
        )
        data = self._raise_for_response(resp)
        return str(data["task_id"])

    def query_task_status(self, task_id: str) -> str:
        """Query the raw status of a submitted task.

        Returns:
            Raw server status, one of ``submitted``, ``queued``, ``running``,
            ``finished``, ``failed``, ``error``.
        """
        resp = self._session.get(
            f"{self.base_url}/task/status/{task_id}",
            timeout=10,
        )
        data = self._raise_for_response(resp)
        return data.get("status", "unknown")

    def fetch_task_result(self, task_id: str) -> Dict[str, Any]:
        """Retrieve the result of a finished task.

        Returns:
            The flat execution payload, unwrapped from the server's
            ``{"ok", "resource", "result"}`` envelope. For ``sample`` mode
            this contains ``"counts"``; for ``expectation`` mode it contains
            ``"energy"`` / ``"expectations"`` / ``"gradients"``.

        Raises:
            RuntimeError: If the task is still pending (HTTP 425), failed,
                or the request itself errored out.
        """
        resp = self._session.get(
            f"{self.base_url}/task/result/{task_id}",
            timeout=300,
        )
        # HTTP 425 means "not ready" — surface as a distinct error so callers
        # know to keep polling instead of treating it as a hard failure.
        if resp.status_code == 425:
            raise RuntimeError(
                f"FieldQuantum task {task_id} not ready yet "
                "(poll /task/status until status=finished)"
            )
        data = self._raise_for_response(resp)
        status = data.get("status")
        if status in ("failed", "error"):
            raise RuntimeError(
                f"FieldQuantum task {task_id} failed: "
                f"{self._extract_error(data, 'unknown')}"
            )
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"FieldQuantum task {task_id}: unexpected result payload {data!r}"
            )
        if "ok" in result and isinstance(result.get("result"), dict):
            if not result.get("ok", True):
                raise RuntimeError(
                    f"FieldQuantum task {task_id} failed: "
                    f"{self._extract_error(result, 'unknown')}"
                )
            result = result["result"]
        return result

    def run_expectation(
        self,
        qasm: str,
        param_names: List[str],
        param_values: List[float],
        hamiltonian: List[Dict[str, Any]],
        *,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Submit an ``expectation`` job and block until results are ready.

        Args:
            qasm: OpenQASM 2.0 circuit template using symbolic parameter
                placeholders (e.g. ``rx(theta_0) q[0];``).
            param_names: Ordered list of symbolic parameter names.
            param_values: Numeric values matching *param_names*.
            hamiltonian: List of ``{"coeff": float, "pauli": str}`` dicts.
            poll_interval: Seconds between status polls (>=3s recommended
                by the service; internal reconciliation is 10s).
            timeout: Hard timeout in seconds.

        Returns:
            ``{"energy", "expectations", "gradients"}``.
        """
        task_id = self.submit_job({
            "mode": "expectation",
            "qasm": qasm,
            "param_names": param_names,
            "param_values": list(param_values),
            "hamiltonian": hamiltonian,
        })
        deadline = time.monotonic() + timeout
        while True:
            status = self.query_task_status(task_id)
            if status == "finished":
                return self.fetch_task_result(task_id)
            if status in ("failed", "error"):
                # Let fetch_task_result surface the server's error message.
                return self.fetch_task_result(task_id)
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"FieldQuantum task {task_id} timed out in status={status!r}"
                )
            time.sleep(max(poll_interval, 3.0))


# ---------------------------------------------------------------------------
# Status normalisation
# ---------------------------------------------------------------------------

_STATUS_MAP: Dict[str, str] = {
    "submitted": "Running",
    "pending":   "Running",
    "queued":    "Running",
    "running":   "Running",
    "finished":  "Finished",
    "failed":    "Failed",
    "error":     "Failed",
}


# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------

class FieldQuantumBackendAdapter(BackendAdapter):
    """Backend adapter for the FieldQuantum cloud simulator.

    Args:
        num_qubits: Default qubit count for the synthetic chip. Defaults to 16.
    """

    provider = "fieldquantum"
    default_hardware_name = "fieldquantum_sim"

    def __init__(self, *, num_qubits: int = 16) -> None:
        self._num_qubits = num_qubits
        self._machine_name = "fieldquantum_sim"
        self._platform = FieldQuantumPlatform()

    # ------------------------------------------------------------------
    # BackendAdapter interface
    # ------------------------------------------------------------------

    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Any] = None,
    ) -> ResolvedBackend:
        """Build a synthetic chip backend for the given qubit count."""
        nq = max(int(num_qubits), 1)
        chip_info = _build_simulator_chip_info(nqubits=nq)
        chip_info["chip_name"] = "fieldquantum_sim"
        backend_obj = Backend(chip_info)
        profile = build_simulator_profile(provider=self.provider, num_qubits=nq)
        return ResolvedBackend(
            provider=self.provider,
            hardware_name="fieldquantum_sim",
            backend=backend_obj,
            profile=profile,
            metadata={"platform_obj": self._platform},
        )


# ---------------------------------------------------------------------------
# Task adapter
# ---------------------------------------------------------------------------

class FieldQuantumTaskAdapter(TaskAdapter):
    """Submit OpenQASM circuits to the FieldQuantum cloud simulator.

    Args:
        client: ``QuantumHardwareClient`` instance (kept for interface parity).
    """

    provider = "fieldquantum"

    def __init__(self, *, client: Any) -> None:
        self._client = client
        self._platform = FieldQuantumPlatform()

    # ------------------------------------------------------------------
    # TaskAdapter interface
    # ------------------------------------------------------------------

    def submit_openqasm(
        self,
        submit_request: OpenQasmSubmitRequest,
        backend: ResolvedBackend,
    ) -> ProviderTaskHandle:
        """POST QASM (sample mode) and return a handle with the server task_id."""
        platform: FieldQuantumPlatform = (
            backend.metadata.get("platform_obj") or self._platform
        )
        logger.info(
            "FieldQuantum: submitting %d-shot job to %s",
            submit_request.shots,
            platform.base_url,
        )
        task_id = platform.submit_job({
            "mode": "sample",
            "qasm": submit_request.qasm,
            "shots": submit_request.shots,
        })
        logger.info("FieldQuantum: task submitted, task_id=%s", task_id)
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=task_id,
            payload={},
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        """Normalise the server status to ``Running`` / ``Finished`` / ``Failed``."""
        raw = self._platform.query_task_status(handle.task_id)
        return _STATUS_MAP.get(raw, "Running")

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        """Fetch counts for a finished sample-mode task."""
        result = self._platform.fetch_task_result(handle.task_id)
        return {"count": result.get("counts", {})}
