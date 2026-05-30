"""Quafu provider integration."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Literal, Optional

import requests

from ..platform_credentials import get_quafu_api_token
from ..backend import BackendAdapter, ResolvedBackend, as_int_or_none, MIN_CONNECTED_COUPLER_FIDELITY
from ..task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter


def _normalize_coordinate(value: Any) -> Optional[List[float]]:
    """Parse a coordinate value (list, tuple, or dict) into a two-element ``[x, y]`` float list.

    Args:
        value (*Any*): Raw coordinate value (list, tuple, or dict with ``x``/``y`` keys).

    Returns:
        Two-element ``[x, y]`` list, or ``None`` if parsing fails.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except Exception:
            return None
    if isinstance(value, dict):
        x = value.get("x", value.get("X"))
        y = value.get("y", value.get("Y"))
        if x is not None and y is not None:
            try:
                return [float(x), float(y)]
            except Exception:
                return None
    return None


def _flip_bitstring(bs: str) -> str:
    """Reverse a bitstring to convert big-endian to little-endian.

    Args:
        bs (*str*): Bitstring to reverse.

    Returns:
        Reversed bitstring.
    """
    return bs[::-1]


def _flip_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """Flip all bitstrings in a count dict from little-endian to big-endian.

    Args:
        counts (*Dict[str, int]*): Measurement count dictionary.

    Returns:
        New dictionary with all bitstrings reversed.
    """
    return {_flip_bitstring(k): v for k, v in counts.items()}


def load_quafu_chip_info(chip_name: str):
    """Fetch chip specifications via HTTP and return normalized qubit/coupler topology.

    Args:
        chip_name (*str*): Name of the target chip.

    Returns:
        Dictionary containing chip topology, qubit fidelities and coupler information.
    """
    session = requests.Session()
    url = "https://quafu-sqc.baqis.ac.cn"
    info = session.get(f"{url}/task/backendtest/{chip_name}1")
    raw = json.loads(info.content.decode())
    if not isinstance(raw, dict) or not raw:
        return None

    qubits_raw = raw.get("qubits_info") if isinstance(raw.get("qubits_info"), dict) else {}
    couplers_raw = raw.get("couplers_info") if isinstance(raw.get("couplers_info"), dict) else {}

    qubits_info: Dict[str, Dict[str, float]] = {}
    for key, value in qubits_raw.items():
        if not isinstance(value, dict):
            continue
        try:
            fidelity = float(value.get("fidelity", 1.0))
        except Exception:
            fidelity = 1.0
        normalized_qubit = {"fidelity": fidelity}
        coordinate = _normalize_coordinate(value.get("coordinate"))
        if coordinate is not None:
            normalized_qubit["coordinate"] = coordinate
        qubits_info[str(key)] = normalized_qubit

    couplers_info: Dict[str, Dict[str, Any]] = {}
    for key, value in couplers_raw.items():
        if not isinstance(value, dict):
            continue
        pair = value.get("qubits_index")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            q1 = int(pair[0])
            q2 = int(pair[1])
        except Exception:
            continue
        try:
            fidelity = float(value.get("fidelity", 1.0))
        except Exception:
            fidelity = 1.0
        if fidelity < MIN_CONNECTED_COUPLER_FIDELITY:
            continue
        couplers_info[str(key)] = {
            "qubits_index": [q1, q2],
            "fidelity": fidelity,
        }

    global_info = raw.get("global_info") if isinstance(raw.get("global_info"), dict) else {}
    priority_qubits = raw.get("priority_qubits") if isinstance(raw.get("priority_qubits"), list) else None

    return {
        "chip_name": chip_name,
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": dict(global_info),
        "priority_qubits": priority_qubits,
    }


def get_available_chip_status(platform_obj) -> Dict[str, int]:
    """Return a dict mapping active chip names to their current queue lengths.

    Args:
        platform_obj: Authenticated Quafu platform instance.

    Returns:
        Dictionary mapping chip names to queue lengths.

    Raises:
        RuntimeError: If ``platform_obj.status()`` does not return a dict.
    """
    status = platform_obj.status()
    if not isinstance(status, dict):
        raise RuntimeError("platform_obj.status() must return a dict of chip -> queue length")
    return {k: v for k, v in status.items() if isinstance(v, int)}


class QuafuPlatform:
    URL = "https://quafu-sqc.baqis.ac.cn"
    session = requests.Session()

    def __new__(cls, *args, **kwargs):
        """Return singleton instance of the Quafu platform.

        Args:
            *args: Positional arguments (unused, forwarded to ``super().__new__``).
            **kwargs: Keyword arguments (unused, forwarded to ``super().__new__``).

        Returns:
            ``QuafuPlatform`` singleton instance.
        """
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        """Initialize Quafu platform with API token and session.

        Raises:
            ValueError: quafu token cannot be empty
        """
        self.token = get_quafu_api_token()
        if not self.token:
            raise ValueError("quafu token cannot be empty")
        self.tasks = {}

    def request(self, url: str, data: dict = {}, method: str = "get"):
        """Send an HTTP request to the Quafu API.

        Args:
            url (*str*): Full API endpoint URL.
            data (*dict*): Request payload dictionary. Defaults to ``{}``.
            method (*str*): HTTP method (``'get'`` or ``'post'``). Defaults to ``'get'``.

        Returns:
            ``dict`` parsed JSON response.

        Raises:
            ValueError: f'unsupported method: {method}'
        """
        if method == "get":
            res = self.session.get(url, headers={"token": self.token})
        elif method == "post":
            res = self.session.post(url, data=json.dumps(data), headers={"token": self.token})
        else:
            raise ValueError(f"unsupported method: {method}")
        return json.loads(res.content.decode())

    def verify(self):
        """Verify the current API session.

        Returns:
            Verification response from the server.
        """
        return self.request(f"{self.URL}/task/verify")

    def query(
        self,
        tid: int = 2,
        chips: str = "Baihua",
        status: str = "Finished,Failed",
        start: str = "2024-04-01",
        end: str = time.strftime("%Y-%m-%d"),
        offset: int = 0,
        limit: int = 10,
        sort: Literal["taskId", "taskName", "chipName", "status", "submitTime"] = "submitTime",
        order: Literal["asc", "desc"] = "desc",
    ):
        """Query submitted tasks with filtering, pagination, and sorting options.

        Args:
            tid (*int*): Task type ID. Defaults to ``2``.
            chips (*str*): Comma-separated chip names to filter. Defaults to ``'Baihua'``.
            status (*str*): Comma-separated status filters. Defaults to ``'Finished,Failed'``.
            start (*str*): Start date (``YYYY-MM-DD``). Defaults to ``'2024-04-01'``.
            end (*str*): End date (``YYYY-MM-DD``). Defaults to ``time.strftime('%Y-%m-%d')``.
            offset (*int*): Pagination offset. Defaults to ``0``.
            limit (*int*): Maximum records to return. Defaults to ``10``.
            sort (*Literal['taskId', 'taskName', 'chipName', 'status', 'submitTime']*): Sort field. Defaults to ``'submitTime'``.
            order (*Literal['asc', 'desc']*): Sort order. Defaults to ``'desc'``.

        Returns:
            ``dict`` parsed JSON query response.
        """
        return self.request(f"{self.URL}/task/query/?tid={tid}&chips={chips}&status={status}&start={start}&end={end}&offset={offset}&limit={limit}&sort={sort}&order={order}")

    def delete(self, tid: int):
        """Delete a submitted task by its ID.

        Args:
            tid (*int*): Task ID.

        Returns:
            ``dict`` parsed JSON response.
        """
        return self.request(f"{self.URL}/task/delete/{tid}")

    def result(self, tid: int, timeout: float = 0.0):
        """Retrieve task result with optional timeout and automatic polling.

        Args:
            tid (*int*): Task ID.
            timeout (*float*): Maximum wait time in seconds; ``0`` for single request. Defaults to ``0.0``.

        Returns:
            ``dict`` parsed JSON task result.

        Raises:
            TimeoutError: f'Task {tid} result timeout after {timeout} seconds
        """
        if timeout:
            st = time.time()
            while True:
                res = self.request(f"{self.URL}/task/result/{tid}")
                if isinstance(res, dict) and res:
                    return res
                if time.time() - st > timeout:
                    raise TimeoutError(f"Task {tid} result timeout after {timeout} seconds")
                time.sleep(0.2)
        else:
            time.sleep(0.2)
        return self.request(f"{self.URL}/task/result/{tid}")

    def status(self, tid: int = 0):
        """Query the status of a submitted task.

        Args:
            tid (*int*): Task ID. Defaults to ``0``.

        Returns:
            ``dict`` parsed JSON status response.
        """
        time.sleep(0.2)
        return self.request(f"{self.URL}/task/status/{tid}")

    def cancel(self, tid: int):
        """Cancel a running task by its ID.

        Args:
            tid (*int*): Task ID.

        Returns:
            ``dict`` parsed JSON response.
        """
        time.sleep(0.2)
        return self.request(f"{self.URL}/task/cancel/{tid}")

    def run(self, task: dict, repeat: int = 1):
        """Submit a quantum circuit task to the Quafu backend and return its task ID.

        Args:
            task (*dict*): Task configuration with keys ``'chip'``, ``'circuit'``, and optionally ``'name'``, ``'shots'``.
            repeat (*int*): Number of measurement repetitions. Defaults to ``1``.

        Returns:
            ``dict`` parsed JSON response containing the task ID.
        """
        time.sleep(0.2)
        name = task.get("name", "MyQuantumJob")
        chip = task["chip"]
        shots = task.get("shots", repeat * 1024)
        circuit = str(task["circuit"])
        tid = self.request(
            f"{self.URL}/task/run/?name={name}&chip={chip}&shots={shots}",
            data={
                "circuit": circuit,
                "compile": task.get("compile", True),
                "options": task.get("options", {"clientip": os.getenv("CLIENT_REAL_IP", "")}),
            },
            method="post",
        )
        if isinstance(tid, int):
            self.tasks[tid] = task
        return tid

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """List available hardware.

        Returns:
            List of hardware description dictionaries.
        """
        queue_map = get_available_chip_status(self)
        rows: List[Dict[str, Any]] = []
        for hardware_name in sorted(queue_map.keys()):
            rows.append(
                {
                    "provider": "quafu",
                    "hardware_name": hardware_name,
                    "queue_length": as_int_or_none(queue_map.get(hardware_name)),
                    "status": None,
                    "is_toll": None,
                    "raw": {"queue_length": queue_map.get(hardware_name)},
                }
            )
        return rows


class QuafuBackendAdapter(BackendAdapter):
    provider = "quafu"
    default_hardware_name = "Baihua"

    def __init__(self, *, machine_name: Optional[str] = None, platform_obj: Optional[QuafuPlatform] = None) -> None:
        """Initialize Quafu backend adapter with optional machine and platform instance.

        Args:
            machine_name (*Optional[str]*): Identifier of the target quantum machine. Defaults to ``None``.
            platform_obj (*Optional[QuafuPlatform]*): Existing platform instance to reuse. Defaults to ``None``.
        """
        self._machine_name = machine_name
        self._platform = platform_obj or QuafuPlatform()


class QuafuTaskAdapter(TaskAdapter):
    provider = "quafu"

    def __init__(self, *, client: Any) -> None:
        """Initialize Quafu task adapter with quantum hardware client.

        Args:
            client (*Any*): ``QuantumHardwareClient`` instance.
        """
        self._client = client

    def submit_openqasm(self, submit_request: OpenQasmSubmitRequest, backend: ResolvedBackend) -> ProviderTaskHandle:
        """Submit an OpenQASM circuit to the Quafu backend and return a task handle.

        Args:
            submit_request (*OpenQasmSubmitRequest*): Submission request descriptor.
            backend (*ResolvedBackend*): Hardware backend descriptor.

        Returns:
            ``ProviderTaskHandle`` for tracking the submitted task.

        Raises:
            RuntimeError: platform_obj is missing in backend metadata
        """
        platform_obj = backend.metadata.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in backend metadata")
        task = {
            "chip": submit_request.chip_name,
            "name": submit_request.name,
            "circuit": submit_request.qasm,
            "shots": submit_request.shots,
            "compile": False,
        }
        task_id = platform_obj.run(task)
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=task_id,
            payload={"platform_obj": platform_obj},
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        """Return the current task status from the Quafu platform.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Returns:
            Status string or parsed JSON response from the Quafu API.

        Raises:
            RuntimeError: platform_obj is missing in task handle payload
        """
        task_id = handle.task_id
        platform_obj = handle.payload.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in task handle payload")
        return platform_obj.status(task_id)

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        """Fetch measurement counts for a completed Quafu task.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Returns:
            ``dict`` with ``"count"`` key mapping to big-endian bitstring counts.

        Raises:
            RuntimeError: platform_obj is missing in task handle payload
        """
        task_id = handle.task_id
        platform_obj = handle.payload.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in task handle payload")
        res = platform_obj.result(task_id)
        if not isinstance(res, dict) or "count" not in res:
            raise RuntimeError(f"quafu task result missing count for task_id={handle.task_id}")
        res["count"] = _flip_counts(res["count"])
        return res

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        """Abort a submitted Quafu task via its platform handle.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Raises:
            RuntimeError: platform_obj is missing in task handle payload
        """
        task_id = handle.task_id
        platform_obj = handle.payload.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in task handle payload")
        platform_obj.cancel(task_id)
