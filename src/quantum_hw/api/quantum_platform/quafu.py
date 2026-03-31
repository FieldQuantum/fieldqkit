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
    """Reverse a bitstring to convert big-endian ↔ little-endian."""
    return bs[::-1]


def _flip_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """Flip all bitstrings in a count dict from little-endian to big-endian."""
    return {_flip_bitstring(k): v for k, v in counts.items()}


def load_quafu_chip_info(chip_name: str):
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
    status = platform_obj.status()
    if not isinstance(status, dict):
        raise RuntimeError("platform_obj.status() must return a dict of chip -> queue length")
    return {k: v for k, v in status.items() if isinstance(v, int)}


class QuafuPlatform:
    URL = "https://quafu-sqc.baqis.ac.cn"
    session = requests.Session()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.token = get_quafu_api_token()
        if not self.token:
            raise ValueError("quafu token cannot be empty")
        self.tasks = {}

    def request(self, url: str, data: dict = {}, method: str = "get"):
        if method == "get":
            res = self.session.get(url, headers={"token": self.token})
        elif method == "post":
            res = self.session.post(url, data=json.dumps(data), headers={"token": self.token})
        else:
            raise ValueError(f"unsupported method: {method}")
        return json.loads(res.content.decode())

    def verify(self):
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
        return self.request(f"{self.URL}/task/query/?tid={tid}&chips={chips}&status={status}&start={start}&end={end}&offset={offset}&limit={limit}&sort={sort}&order={order}")

    def delete(self, tid: int):
        return self.request(f"{self.URL}/task/delete/{tid}")

    def result(self, tid: int, timeout: float = 0.0):
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
        time.sleep(0.2)
        return self.request(f"{self.URL}/task/status/{tid}")

    def cancel(self, tid: int):
        time.sleep(0.2)
        return self.request(f"{self.URL}/task/cancel/{tid}")

    def run(self, task: dict, repeat: int = 1):
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
        self._machine_name = machine_name
        self._platform = platform_obj or QuafuPlatform()


class QuafuTaskAdapter(TaskAdapter):
    provider = "quafu"

    def __init__(self, *, client: Any) -> None:
        self._client = client

    def submit_openqasm(self, submit_request: OpenQasmSubmitRequest, backend: ResolvedBackend) -> ProviderTaskHandle:
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
        task_id = handle.task_id
        platform_obj = handle.payload.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in task handle payload")
        return platform_obj.status(task_id)

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
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
        task_id = handle.task_id
        platform_obj = handle.payload.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in task handle payload")
        platform_obj.cancel(task_id)
