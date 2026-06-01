"""GuoDun provider integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cqlib import CqlibTaskAdapter, RemotePlatformClient, normalize_hardware_rows, records_from_platform_list_query
from ..platform_credentials import get_guodun_api_token
from ..backend import BackendAdapter


class GuoDunPlatform(RemotePlatformClient):
    SCHEME = "https"
    DOMAIN = "quantumctek-cloud.com"
    LOGIN_PATH = "/api-uaa/oauth/token"
    MACHINE_LIST_PATH = "/experiment/sdk/quantumComputer/list"
    CREATE_LAB_PATH = "/experiment/sdk/experiment/save"
    SAVE_EXP_PATH = "/experiment/sdk/experiment/detail/save"
    QUERY_EXP_PATH = "/experiment/sdk/experiment/result/find"
    DOWNLOAD_CONFIG_PATH = "/experiment/sdk/experiment/download/config"
    QCIS_CHECK_REGULAR_PATH = "/experiment/sdk/experiment/qcis/rule/verify"
    GET_EXP_CIRCUIT_PATH = "/experiment/sdk/experiment/getQcis/by/taskIds"
    RE_EXECUTE_TASK_PATH = "/experiment/sdk/experiment/resubmit"
    STOP_RUNNING_EXP_PATH = "/experiment/sdk/experiment/discontinue"
    CREATE_EXP_AND_RUN_PATH = "/experiment/sdk/experiment/temporary/save"
    CREATE_WAVEFORM_DIAGRAM = "/experiment/sdk/generateWaveformDiagram"
    DOWN_WAVEFORM_DIAGRAM = "/experiment/sdk/getWaveformDiagram"

    def list_available_hardware(self) -> List[Dict[str, Any]]:
        """Query the GuoDun platform and return a normalized hardware catalog.

        Returns:
            List of hardware description dictionaries.
        """
        records = records_from_platform_list_query(self)
        return normalize_hardware_rows(provider="guodun", records=records)

    def re_execute_task(self, query_id: Optional[str] = None, lab_id: Optional[str] = None):
        """Re-execute a previously submitted task.

        Args:
            query_id (*Optional[str]*): Experiment query identifier. Defaults to ``None``.
            lab_id (*Optional[str]*): Laboratory identifier. Defaults to ``None``.

        Returns:
            Task data ``dict`` from the response.

        Raises:
            ValueError: Please provide lab_id or query_id.
        """
        if not lab_id and not query_id:
            raise ValueError("Please provide lab_id or query_id.")
        try:
            lab_id = int(lab_id)
        except TypeError:
            pass
        try:
            query_id = int(query_id)
        except TypeError:
            pass
        data = {"lab_id": lab_id, "query_id": query_id}
        result = self._send_request(self.RE_EXECUTE_TASK_PATH, method="POST", data=data)
        return result.get("data")

    def stop_running_experiments(self, lab_id: Optional[str] = None, query_id: Optional[str] = None):
        """Stop currently running experiments.

        Args:
            lab_id (*Optional[str]*): Laboratory identifier. Defaults to ``None``.
            query_id (*Optional[str]*): Experiment query identifier. Defaults to ``None``.

        Returns:
            Task data ``dict`` from the response.

        Raises:
            ValueError: Please provide lab_id or query_id.
        """
        if not lab_id and not query_id:
            raise ValueError("Please provide lab_id or query_id.")
        try:
            lab_id = int(lab_id)
        except TypeError:
            pass
        try:
            query_id = int(query_id)
        except TypeError:
            pass
        data = {"lab_id": lab_id, "query_id": query_id}
        result = self._send_request(self.STOP_RUNNING_EXP_PATH, method="POST", data=data)
        return result.get("data")

    def create_waveform_data(self, circuit, circuit_name: Optional[str] = None) -> int:
        """Submit circuit waveform data to the platform.

        Args:
            circuit: Quantum circuit to execute.
            circuit_name (*Optional[str]*): Optional name for the waveform entry. Defaults to ``None``.

        Returns:
            Waveform diagram ID (``int``).

        Raises:
            ValueError: Please provide circuit.
        """
        if not circuit:
            raise ValueError("Please provide circuit.")
        if not self.machine_name:
            raise ValueError("The platform is missing machine_name parameter.")
        data = {"circuit": circuit, "qcCode": self.machine_name, "circuit_name": circuit_name}
        res = self._send_request(self.CREATE_WAVEFORM_DIAGRAM, method="POST", data=data)
        return res.get("data").get("id")

    def query_waveform_data(self, query_id: int) -> str:
        """Retrieve the waveform visualisation URL for a diagram.

        Args:
            query_id (*int*): Waveform diagram ID.

        Returns:
            Waveform visible URL string.
        """
        params = {"id": query_id}
        res = self._send_request(self.DOWN_WAVEFORM_DIAGRAM, method="GET", params=params)
        return res.get("data").get("visibleUrl")


class GuoDunBackendAdapter(BackendAdapter):
    provider = "guodun"
    default_hardware_name = "gd_qc1"

    def __init__(self, *, machine_name: Optional[str] = None, api_token: Optional[str] = None) -> None:
        """Initialize GuoDun backend adapter with optional machine and login credentials.

        Args:
            machine_name (*Optional[str]*): Identifier of the target quantum machine. Defaults to ``None``.
            api_token (*Optional[str]*): API token for authentication. Defaults to ``None``.

        Raises:
            ValueError: guodun api token cannot be empty
        """
        self._api_token = api_token or get_guodun_api_token()
        if not self._api_token:
            raise ValueError("guodun api token cannot be empty")
        self._machine_name = machine_name
        self._platform = GuoDunPlatform(login_key=self._api_token, auto_login=True, machine_name=machine_name)


class GuoDunTaskAdapter(CqlibTaskAdapter):
    """TaskAdapter for the GuoDun provider (shared cqlib QCIS protocol)."""

    provider = "guodun"

    def _default_api_token(self) -> Optional[str]:
        return get_guodun_api_token()
