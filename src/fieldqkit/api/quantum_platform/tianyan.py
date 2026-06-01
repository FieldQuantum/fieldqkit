"""TianYan provider integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .cqlib import CqlibTaskAdapter, RemotePlatformClient, normalize_hardware_rows, records_from_platform_list_query
from ..platform_credentials import get_tianyan_api_token
from ..backend import BackendAdapter


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
        """Query the TianYan platform and return a normalized hardware catalog.

        Returns:
            List of hardware description dictionaries.
        """
        records = records_from_platform_list_query(self)
        return normalize_hardware_rows(provider="tianyan", records=records)


class TianYanBackendAdapter(BackendAdapter):
    provider = "tianyan"
    default_hardware_name = "tianyan176"

    def __init__(self, *, machine_name: Optional[str] = None, api_token: Optional[str] = None) -> None:
        """Initialize TianYan backend adapter with optional machine and login credentials.

        Args:
            machine_name (*Optional[str]*): Identifier of the target quantum machine. Defaults to ``None``.
            api_token (*Optional[str]*): API token for authentication. Defaults to ``None``.

        Raises:
            ValueError: tianyan api token cannot be empty
        """
        self._api_token = api_token or get_tianyan_api_token()
        if not self._api_token:
            raise ValueError("tianyan api token cannot be empty")
        self._machine_name = machine_name
        self._platform = TianYanPlatform(login_key=self._api_token, auto_login=True, machine_name=machine_name)


class TianYanTaskAdapter(CqlibTaskAdapter):
    """TaskAdapter for the TianYan provider (shared cqlib QCIS protocol)."""

    provider = "tianyan"

    def _default_api_token(self) -> Optional[str]:
        return get_tianyan_api_token()
