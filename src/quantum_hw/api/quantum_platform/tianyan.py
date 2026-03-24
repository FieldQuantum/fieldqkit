"""TianYan provider integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .cqlib import QuantumLanguage, RemotePlatformClient, extract_counts_from_result_items, normalize_hardware_rows, records_from_platform_list_query
from ..platform_credentials import get_tianyan_login_key
from ..backend import BackendAdapter, ResolvedBackend
from ..task import OpenQasmSubmitRequest, ProviderTaskHandle, TaskAdapter


logger = logging.getLogger("cqlib")


class TianYanPlatform(RemotePlatformClient):
    SCHEME = "https"
    DOMAIN = "qc.zdxlz.com"
    LOGIN_PATH = "/qccp-auth/oauth2/sdk/opnId"
    CREATE_LAB_PATH = "/qccp-quantum/sdk/experiment/save"
    SAVE_EXP_PATH = "/qccp-quantum/sdk/experiment/detail/save"
    RUN_EXP_PATH = "/qccp-quantum/sdk/experiment/detail/run"
    SUBMIT_EXP_PATH = "/qccp-quantum/sdk/experiment/submit"
    CREATE_EXP_AND_RUN_PATH = "/qccp-quantum/sdk/experiment/temporary/save"
    QUERY_EXP_PATH = "/qccp-quantum/sdk/experiment/result/find"
    DOWNLOAD_CONFIG_PATH = "/qccp-quantum/sdk/experiment/download/config"
    QCIS_CHECK_REGULAR_PATH = "/qccp-quantum/sdk/experiment/qcis/rule/verify"
    GET_EXP_CIRCUIT_PATH = "/qccp-quantum/sdk/experiment/getQcis/by/taskIds"
    MACHINE_LIST_PATH = "/qccp-quantum/sdk/quantumComputer/list"
    RE_EXECUTE_TASK_PATH = "/qccp-quantum/sdk/experiment/resubmit"
    STOP_RUNNING_EXP_PATH = ""

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        records = records_from_platform_list_query(self)
        return normalize_hardware_rows(provider="tianyan", records=records)

    def query_experiment(
        self,
        query_id: str | list[str],
        max_wait_time: int = 3600,
        sleep_time: int = 5,
        readout_calibration: bool = False,
        machine_config: dict = None,
    ):
        data = super().query_experiment(query_id, max_wait_time, sleep_time)
        if readout_calibration:
            from ...vendor.cqlib.exceptions import CqlibRequestError
            from ...vendor.cqlib.utils.laboratory_utils import LaboratoryUtils

            lu = LaboratoryUtils()
            if machine_config is None:
                try:
                    machine_config = self.download_config()
                except CqlibRequestError:
                    logger.warning("Only quantum physics machines can obtain configuration, please check the machine type.")
            if machine_config is not None:
                for item in data:
                    calibration_result = lu.probability_calibration(result=item, laboratory=self, config_json=machine_config)
                    item["probability"] = lu.probability_correction(calibration_result)
        return data


class TianYanBackendAdapter(BackendAdapter):
    provider = "tianyan"
    default_hardware_name = "tianyan176"

    def __init__(self, *, machine_name: Optional[str] = None, login_key: Optional[str] = None) -> None:
        self._login_key = login_key or get_tianyan_login_key()
        if not self._login_key:
            raise ValueError("tianyan login key cannot be empty")
        self._machine_name = machine_name
        self._platform = TianYanPlatform(login_key=self._login_key, auto_login=True, machine_name=machine_name)


class TianYanTaskAdapter(TaskAdapter):
    provider = "tianyan"

    def __init__(self, *, client: Any, login_key: Optional[str] = None) -> None:
        self._client = client
        self._login_key = login_key or get_tianyan_login_key()
        self._handle_cache: Dict[str, Dict[str, Any]] = {}

    def submit_openqasm(self, submit_request: OpenQasmSubmitRequest, backend: ResolvedBackend) -> ProviderTaskHandle:
        from ...vendor.cqlib.utils.qasm_to_qcis.qasm_to_qcis import QasmToQcis

        platform_obj = backend.metadata.get("platform_obj")
        if platform_obj is None:
            raise RuntimeError("platform_obj is missing in backend metadata")
        options = dict(submit_request.submit_options or {})

        def _as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        max_wait_time = _as_int(options.get("max_wait_time", 3600), 3600)
        sleep_time = _as_int(options.get("sleep_time", 5), 5)
        qcis = QasmToQcis().convert_to_qcis(submit_request.qasm)
        submitted_task_ids = platform_obj.submit_job(circuit=qcis, exp_name=submit_request.name, num_shots=submit_request.shots, language=QuantumLanguage.QCIS, is_verify=True)
        task_ids = [submitted_task_ids] if isinstance(submitted_task_ids, str) else [str(q) for q in submitted_task_ids]
        task_id = task_ids[0] if len(task_ids) == 1 else ",".join(task_ids)
        payload = {"task_ids": task_ids, "platform_obj": platform_obj, "max_wait_time": max_wait_time, "sleep_time": sleep_time, "num_qubits": submit_request.submit_options.get("num_qubits", 0)}
        handle = ProviderTaskHandle(provider=self.provider, task_id=task_id, payload=payload)
        self._handle_cache[task_id] = payload
        return handle

    def query_status(self, handle: ProviderTaskHandle) -> str:
        payload = dict(self._handle_cache.get(handle.task_id, {}))
        payload.update(handle.payload)
        if payload.get("result_items") is not None:
            return "Finished"
        result = payload["platform_obj"].query_experiment(payload["task_ids"][0] if len(payload["task_ids"]) == 1 else payload["task_ids"], max_wait_time=payload["max_wait_time"], sleep_time=payload["sleep_time"])
        result_items = [result] if isinstance(result, dict) else [item for item in result if isinstance(item, dict)] if isinstance(result, list) else []
        payload["result_items"] = result_items
        self._handle_cache[handle.task_id] = payload
        return "Finished" if result_items else "Failed"

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        payload = dict(self._handle_cache.get(handle.task_id, {}))
        payload.update(handle.payload)
        if payload.get("result_items") is None and self.query_status(handle) != "Finished":
            raise RuntimeError(f"task {handle.task_id} ended with status Failed")
        return {"count": extract_counts_from_result_items(payload.get("result_items", []), num_qubits=int(payload.get("num_qubits", 0) or 0))}

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        payload = dict(self._handle_cache.get(handle.task_id, {}))
        payload.update(handle.payload)
        platform_obj = payload["platform_obj"]
        if hasattr(platform_obj, "stop_running_experiments"):
            for qid in payload["task_ids"]:
                try:
                    platform_obj.stop_running_experiments(query_id=qid)
                except Exception:
                    continue
