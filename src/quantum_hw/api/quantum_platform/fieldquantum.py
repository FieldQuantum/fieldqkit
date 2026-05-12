"""FieldQuantum cloud simulator provider.

Client-side adapters for the FieldQuantum HTTP-based cloud simulator backend.
The companion server (``fieldquantum_server.py``) must be running before
jobs can be submitted through this provider.

Typical usage::

    # start server once:
    #   python -m quantum_hw.api.fieldquantum_server

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
from ..task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable server URL (override via env-var for non-localhost setups)
# ---------------------------------------------------------------------------

FIELDQUANTUM_DEFAULT_URL: str = os.environ.get(
    "FIELDQUANTUM_SERVER_URL", "http://localhost:8765"
)


# ---------------------------------------------------------------------------
# HTTP platform client
# ---------------------------------------------------------------------------

class FieldQuantumPlatform:
    """Thin HTTP client that wraps the FieldQuantum simulator REST API.

    Args:
        base_url: Root URL of the server, e.g. ``"http://localhost:8765"``.
    """

    def __init__(self, base_url: str = FIELDQUANTUM_DEFAULT_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return ``True`` if the server is reachable and healthy."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def run_expectation(
        self,
        qasm: str,
        param_names: List[str],
        param_values: List[float],
        hamiltonian: List[Dict[str, Any]],
        shots: int = 8192,
    ) -> Dict[str, Any]:
        """Submit an *expectation* request and return energy + gradients.

        The server computes Pauli expectation values via sampling and returns
        parameter-shift gradients.

        Args:
            qasm: OpenQASM 2.0 circuit template with symbolic parameter names
                  (e.g. ``rx(theta_0) q[0];``).
            param_names: Ordered list of symbolic parameter names.
            param_values: Numeric values corresponding to *param_names*.
            hamiltonian: List of ``{"coeff": float, "pauli": str}`` dicts.
            shots: Shots per expectation evaluation.

        Returns:
            Dict with keys ``"energy"``, ``"expectations"``, and ``"gradients"``.
        """
        task_id = self.submit_job({
            "mode": "expectation",
            "qasm": qasm,
            "param_names": param_names,
            "param_values": list(param_values),
            "hamiltonian": hamiltonian,
            "shots": shots,
        })
        return self.fetch_task_result(task_id)

    def submit_job(self, payload: Dict[str, Any]) -> str:
        """POST *payload* to ``/run`` and return the server-issued *task_id*.

        Args:
            payload: Request body dict (must include ``"mode"`` key).

        Returns:
            Task ID string issued by the server.

        Raises:
            requests.HTTPError: On non-2xx HTTP status.
            RuntimeError: If the server returns an error body.
        """
        resp = self._session.post(
            f"{self.base_url}/run",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Server error: {data['error']}")
        return data["task_id"]

    def query_task_status(self, task_id: str) -> str:
        """Query the execution status of a submitted task.

        Args:
            task_id: Task ID returned by :meth:`submit_job`.

        Returns:
            Status string: ``"finished"``, ``"running"``, ``"pending"``, or ``"error"``.
        """
        resp = self._session.get(
            f"{self.base_url}/task/{task_id}/status",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("status", "unknown")

    def fetch_task_result(self, task_id: str) -> Dict[str, Any]:
        """Retrieve the result of a finished task.

        Args:
            task_id: Task ID returned by :meth:`submit_job`.

        Returns:
            Result dict (structure depends on mode: ``"counts"`` for sample,
            ``"energy"`` / ``"gradients"`` for expectation).

        Raises:
            requests.HTTPError: On non-2xx HTTP status.
            RuntimeError: If the server stored an error for this task.
        """
        resp = self._session.get(
            f"{self.base_url}/task/{task_id}/result",
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Server error: {data['error']}")
        return data


# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------

class FieldQuantumBackendAdapter(BackendAdapter):
    """Backend adapter for the FieldQuantum cloud simulator.

    No credentials are required; the server must be reachable at *base_url*.

    Args:
        base_url: Server URL. Defaults to ``FIELDQUANTUM_DEFAULT_URL``.
        num_qubits: Default qubit count for the synthetic chip. Defaults to 16.
    """

    provider = "fieldquantum"
    default_hardware_name = "fieldquantum_sim"

    def __init__(
        self,
        *,
        base_url: str = FIELDQUANTUM_DEFAULT_URL,
        num_qubits: int = 16,
    ) -> None:
        self._base_url = base_url
        self._num_qubits = num_qubits
        self._machine_name = "fieldquantum_sim"
        self._platform = FieldQuantumPlatform(base_url=base_url)

    # ------------------------------------------------------------------
    # BackendAdapter interface
    # ------------------------------------------------------------------

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        return [{"hardware_name": "fieldquantum_sim", "queue_length": 0}]

    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Any] = None,
    ) -> ResolvedBackend:
        """Build a synthetic chip backend for the given qubit count.

        Args:
            num_qubits: Number of qubits to allocate on the synthetic chip.
            prefer_hardware: Ignored for the cloud simulator.

        Returns:
            ``ResolvedBackend`` backed by a synthetic chip matching *num_qubits*.
        """
        nq = max(int(num_qubits), 1)
        chip_info = _build_simulator_chip_info(nqubits=nq)
        # Tag the chip_info so Backend.__init__ can identify it from a dict.
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
    """Task adapter that submits OpenQASM circuits to the FieldQuantum server.

    Implements the full task lifecycle:
    1. :meth:`submit_openqasm` — POST ``/run``, receive *task_id*.
    2. :meth:`query_status` — GET ``/task/{task_id}/status``.
    3. :meth:`fetch_result` — GET ``/task/{task_id}/result``.

    Args:
        client: ``QuantumHardwareClient`` instance (not used directly, kept
                for interface parity with other adapters).
        base_url: Server URL. Defaults to ``FIELDQUANTUM_DEFAULT_URL``.
    """

    provider = "fieldquantum"

    def __init__(
        self,
        *,
        client: Any,
        base_url: str = FIELDQUANTUM_DEFAULT_URL,
    ) -> None:
        self._client = client
        self._platform = FieldQuantumPlatform(base_url=base_url)

    # ------------------------------------------------------------------
    # TaskAdapter interface
    # ------------------------------------------------------------------

    def submit_openqasm(
        self,
        submit_request: OpenQasmSubmitRequest,
        backend: ResolvedBackend,
    ) -> ProviderTaskHandle:
        """POST QASM to the server (sample mode) and return a handle with the server task_id.

        Args:
            submit_request: Submission descriptor with QASM, shots, etc.
            backend: Resolved backend (provides ``platform_obj`` in metadata).

        Returns:
            ``ProviderTaskHandle`` with the server-issued *task_id*.
        """
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
        """Poll ``/task/{task_id}/status`` and return a normalised status string.

        Returns:
            ``"Finished"``, ``"Running"``, or ``"Failed"``.
        """
        raw = self._platform.query_task_status(handle.task_id)
        return {
            "finished": "Finished",
            "error": "Failed",
            "pending": "Running",
            "running": "Running",
        }.get(raw, "Running")

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        """Fetch the result from ``/task/{task_id}/result``.

        Args:
            handle: Task handle from :meth:`submit_openqasm`.

        Returns:
            ``{"count": {bitstring: int, ...}}`` dict consumed by the client.
        """
        result = self._platform.fetch_task_result(handle.task_id)
        return {"count": result.get("counts", {})}
