"""Shared cqlib-style platform helpers for TianYan and GuoDun providers."""

from __future__ import annotations

from enum import Enum
import functools
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import requests

from ..backend import as_int_or_none

logger = logging.getLogger("cqlib")


class CqlibRequestError(Exception):
    """Request error with optional HTTP status code."""

    def __init__(self, message, status_code=None):
        """Initialize request error exception with optional HTTP status code.

        Args:
            message: Error message string.
            status_code: HTTP status code, if applicable. Defaults to ``None``.
        """
        super().__init__(message)
        self.status_code = status_code
        if status_code is not None:
            self.message = f"Request failed with status code {status_code}: {message}"
        else:
            self.message = message


def _assign_parameters(
    circuits: List[str], parameters: List[List], values: List[List]
) -> List[str]:
    """Assign parameter values to QCIS circuit template strings.

    Args:
        circuits (*List[str]*): QCIS circuit strings with ``{PARAM}`` placeholders.
        parameters (*List[List]*): Per-circuit lists of parameter names.
        values (*List[List]*): Per-circuit lists of numeric values to substitute.

    Returns:
        Result list.

    Raises:
        ValueError: f'Circuit has parameters {circuit_parameters}, but no val...
    """
    new_circuit: List[str] = []
    for circuit, parameter, value in zip(circuits, parameters, values):
        circuit = circuit.upper()
        p = re.compile(r"\{(\w+)\}")
        circuit_parameters = p.findall(circuit)
        if circuit_parameters:
            after_parameter = [p.upper() for p in parameter]
            if not value:
                raise ValueError(
                    f"Circuit has parameters {circuit_parameters}, but no values provided."
                )
            if len(circuit_parameters) != len(value):
                raise ValueError(
                    f"Circuit has {len(circuit_parameters)} parameters, but {len(value)} values provided."
                )
            if after_parameter and len(circuit_parameters) != len(after_parameter):
                raise ValueError(
                    f"Circuit has {len(circuit_parameters)} parameters, but {len(after_parameter)} parameter names provided."
                )
            if set(after_parameter) != set(circuit_parameters):
                raise ValueError(
                    "Parameter names in circuit do not match the provided parameter names."
                )
            param_dic = dict(zip(after_parameter, value))
            new_circuit.append(circuit.format(**param_dic))
        elif parameter or value:
            raise ValueError(
                "Circuit has no parameters, but parameter names or values were provided."
            )
        else:
            new_circuit.append(circuit)
    return new_circuit

TIANYAN_HARDWARE_NAMES = {"tianyan176", "tianyan176-2", "tianyan24", "tianyan504", "tianyan287"}
GUODUN_HARDWARE_NAMES = {"gd_qc1", "chmy176", "gd_sim1"}


class QuantumLanguage(Enum):
    QCIS = "qcis"
    ISQ = "isq"
    QUINGO = "quingo"


def records_from_platform_list_query(platform_obj: Any) -> List[Dict[str, Any]]:
    """Query the platform object for available quantum computers and return raw records.

    Args:
        platform_obj (*Any*): Platform client instance with a ``query_quantum_computer_records`` method.

    Returns:
        List of hardware record dictionaries.
    """
    try:
        records = platform_obj.query_quantum_computer_records()
        if isinstance(records, list):
            out = [row for row in records if isinstance(row, dict)]
            if out:
                return out
    except Exception:
        pass

    try:
        rows = platform_obj.query_quantum_computer_list()
    except Exception:
        return []

    if not isinstance(rows, list) or not rows:
        return []

    records: List[Dict[str, Any]] = []
    first = rows[0]
    if isinstance(first, (list, tuple)) and "machineName" in first:
        headers = [str(v) for v in first]
        name_idx = headers.index("machineName")
        status_idx = headers.index("status") if "status" in headers else None
        toll_idx = headers.index("isToll") if "isToll" in headers else None
        for row in rows[1:]:
            if not isinstance(row, (list, tuple)):
                continue
            if name_idx >= len(row):
                continue
            record: Dict[str, Any] = {"machineName": row[name_idx]}
            if status_idx is not None and status_idx < len(row):
                record["status"] = row[status_idx]
            if toll_idx is not None and toll_idx < len(row):
                record["isToll"] = row[toll_idx]
            records.append(record)
        return records

    for row in rows:
        if isinstance(row, (list, tuple)) and row:
            records.append({"machineName": row[-1]})
    return records


def normalize_hardware_rows(*, provider: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize heterogeneous hardware records into a uniform schema.

    Args:
        provider (*str*): Platform provider name (``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``).
        records (*List[Dict[str, Any]]*): Raw hardware records from the platform.

    Returns:
        List of normalized hardware dictionaries with keys
        ``provider``, ``hardware_name``, ``queue_length``, ``status``, ``is_toll``, ``raw``.
    """
    rows = []
    for record in records:
        hardware_name = str(record.get("machineName") or "").strip()
        if not hardware_name:
            continue
        queue_length = as_int_or_none(
            record.get("queueLength")
            if "queueLength" in record
            else record.get("queueNum")
            if "queueNum" in record
            else record.get("waitingNum")
        )
        rows.append(
            {
                "provider": provider,
                "hardware_name": hardware_name,
                "queue_length": queue_length,
                "status": record.get("status"),
                "is_toll": record.get("isToll"),
                "raw": record,
            }
        )
    return rows


def extract_counts_from_result_items(result_items: Sequence[Dict[str, Any]], *, num_qubits: int) -> Dict[str, int]:
    """Extract counts from result items.

    Args:
        result_items (*Sequence[Dict[str, Any]]*): Result items (``Sequence[Dict[str, Any]]``).
        num_qubits (*int*): Number of qubits.

    Returns:
        Result dictionary.

    Raises:
        RuntimeError: failed to extract counts from platform result payload
    """
    merged: Dict[str, int] = {}
    width = max(int(num_qubits), 1)
    for item in result_items:
        matrix = item.get("resultStatus")
        if not isinstance(matrix, list) or len(matrix) < 2:
            continue
        for row in matrix[1:]:
            if not isinstance(row, list):
                continue
            try:
                bits = [int(v) for v in row]
            except Exception:
                continue
            if any(v not in (0, 1) for v in bits):
                continue
            bitstring = "".join(str(v) for v in bits)
            if len(bitstring) < width:
                bitstring = bitstring.rjust(width, "0")
            elif len(bitstring) > width:
                bitstring = bitstring[-width:]
            merged[bitstring] = merged.get(bitstring, 0) + 1

    if merged:
        return merged

    for item in result_items:
        maybe_count = item.get("count")
        if isinstance(maybe_count, dict):
            for bit, cnt in maybe_count.items():
                normalized = str(bit)
                merged[normalized] = merged.get(normalized, 0) + int(round(float(cnt)))
    if not merged:
        raise RuntimeError("failed to extract counts from platform result payload")
    return merged


def format_circuit(circuit: str):
    """Format circuit.

    Args:
        circuit (*str*): Quantum circuit to execute.

    Returns:
        Result.
    """
    content = []
    for line in circuit.split("\n"):
        line = line.strip()
        if line:
            content.append(line)
    return "\n".join(content)


def _reconnect_on_failure(func, max_retries=2, retry_delay=1):
    """Retry decorator for RemotePlatformClient methods.

    Args:
        func: Func.
        max_retries: Max retries. Defaults to ``2``.
        retry_delay: Retry delay. Defaults to ``1``.

    Returns:
        Result.

    Raises:
        CqlibRequestError: f'function:[{func.__name__}] Max retries exceeded. Attemp...
        last_error: If an error condition is met.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper.

        Args:
            *args: *args.
            **kwargs: **kwargs.

        Returns:
            Result.

        Raises:
            CqlibRequestError: f'function:[{func.__name__}] Max retries exceeded. Attemp...
            last_error: If an error condition is met.
        """
        retries = 0
        last_error = None
        while retries < max_retries:
            retries += 1
            try:
                return func(self, *args, **kwargs)
            except CqlibRequestError as exc:
                last_error = exc
                logger.warning("%s execution failed\ntry count:%s \nerror info: \n%s", func.__name__, retries, exc)
                if exc.status_code == 401:
                    logger.warning("user's token has expired, try to log in again.")
                    self.login()
                    time.sleep(retry_delay)
            except Exception as exc:
                last_error = exc

        if last_error:
            raise last_error

        raise CqlibRequestError(
            f"function:[{func.__name__}] Max retries exceeded. Attempt {max_retries} times failed. "
        )

    return wrapper


class RemotePlatformClient:
    """Shared HTTP client for TianYan and GuoDun style providers."""

    SCHEME = "https"
    DOMAIN = ""
    LOGIN_PATH = ""
    CREATE_LAB_PATH = ""
    SAVE_EXP_PATH = ""
    CREATE_EXP_AND_RUN_PATH = ""
    QUERY_EXP_PATH = ""
    DOWNLOAD_CONFIG_PATH = ""
    QCIS_CHECK_REGULAR_PATH = ""
    GET_EXP_CIRCUIT_PATH = ""
    MACHINE_LIST_PATH = ""
    RE_EXECUTE_TASK_PATH = ""
    STOP_RUNNING_EXP_PATH = ""

    def __init__(self, login_key: str, auto_login: bool = True, machine_name: str = None):
        """Initialize CQLIB platform with login credentials and optional machine selection.

        Args:
            login_key (*str*): Login key for authentication.
            auto_login (*bool*): Whether to log in immediately on initialization. Defaults to ``True``.
            machine_name (*str*): Identifier of the target quantum machine. Defaults to ``None``.
        """
        self.login_key = login_key
        self.auto_login = auto_login
        self.machine_name = machine_name
        self.access_token = ""
        if self.auto_login:
            self.login()

    def login(self, timeout=60) -> int:
        """Authenticate with the platform and store the access token.

        Args:
            timeout: HTTP request timeout in seconds. Defaults to ``60``.

        Returns:
            The access token string on success.

        Raises:
            CqlibRequestError: Login failed: request interface failed
        """
        data = {
            "grant_type": "openId",
            "openId": self.login_key,
            "account_type": "member",
        }
        res = requests.post(
            url=f"{self.SCHEME}://{self.DOMAIN}{self.LOGIN_PATH}",
            data=data,
            timeout=timeout,
        )
        if res.status_code != 200:
            raise CqlibRequestError("Login failed: request interface failed", res.status_code)

        data = res.json()
        if data.get("code", -1) != 0:
            raise CqlibRequestError("Login failed")
        self.access_token = data.get("data").get("access_token")
        return self.access_token

    def set_machine(self, machine_name: str):
        """Set the target quantum machine by name for subsequent operations.

        Args:
            machine_name (*str*): Identifier of the target quantum machine.
        """
        self.machine_name = machine_name

    def create_lab(self, name: str, remark: str = "") -> str:
        """Create a new laboratory workspace and return its unique identifier.

        Args:
            name (*str*): Descriptive name / identifier.
            remark (*str*): Remark (``str``). Defaults to ``''``.

        Returns:
            Formatted string.
        """
        data = {"name": name, "remark": remark}
        result = self._send_request(path=self.CREATE_LAB_PATH, data=data, method="POST")
        return result.get("data").get("lab_id")

    def save_experiment(
        self,
        lab_id: str,
        circuit: str,
        name: Optional[str] = "",
        language: QuantumLanguage = QuantumLanguage.QCIS,
        **kwargs,
    ):
        """Save a quantum circuit as an experiment to a specific laboratory.

        Args:
            lab_id (*str*): Lab id (``str``).
            circuit (*str*): Quantum circuit to execute.
            name (*Optional[str]*): Descriptive name / identifier. Defaults to ``''``.
            language (*QuantumLanguage*): Language (``QuantumLanguage``). Defaults to ``QuantumLanguage.QCIS``.
            **kwargs: **kwargs.

        Returns:
            Result.
        """
        if language.value == "qcis":
            circuit = circuit.upper()
        exp_data = format_circuit(circuit)
        data = {
            "inputCode": exp_data,
            "lab_id": lab_id,
            "languageCode": language.value,
            "name": name,
            "source": "SDK",
            "computerCode": self.machine_name,
        }
        if "noise" in kwargs:
            data["noise"] = kwargs["noise"]
        elif "quantum_state" in kwargs:
            data["quantumState"] = kwargs["quantum_state"]
        result = self._send_request(path=self.SAVE_EXP_PATH, data=data, method="post")
        return result.get("data").get("exp_id")

    def submit_job(
        self,
        circuit: Optional[Union[List, str]] = None,
        exp_name: Optional[str] = "",
        parameters: Optional[List[List]] = None,
        values: Optional[List[List]] = None,
        num_shots: Optional[int] = 12000,
        lab_id: Optional[str] = None,
        exp_id: Optional[str] = None,
        language: QuantumLanguage = QuantumLanguage.QCIS,
        version: Optional[str] = "1",
        is_verify: Optional[bool] = True,
        **kwargs,
    ):
        """Submit job.

        Args:
            circuit (*Optional[Union[List, str]]*): Quantum circuit to execute. Defaults to ``None``.
            exp_name (*Optional[str]*): Exp name (``Optional[str]``). Defaults to ``''``.
            parameters (*Optional[List[List]]*): Parameter values. Defaults to ``None``.
            values (*Optional[List[List]]*): Values (``Optional[List[List]]``). Defaults to ``None``.
            num_shots (*Optional[int]*): Num shots (``Optional[int]``). Defaults to ``12000``.
            lab_id (*Optional[str]*): Lab id (``Optional[str]``). Defaults to ``None``.
            exp_id (*Optional[str]*): Exp id (``Optional[str]``). Defaults to ``None``.
            language (*QuantumLanguage*): Language (``QuantumLanguage``). Defaults to ``QuantumLanguage.QCIS``.
            version (*Optional[str]*): Version (``Optional[str]``). Defaults to ``'1'``.
            is_verify (*Optional[bool]*): Is verify (``Optional[bool]``). Defaults to ``True``.
            **kwargs: **kwargs.

        Returns:
            Result.

        Raises:
            ValueError: When circuit is not defined, experiment id should be defi...
        """
        if isinstance(circuit, str):
            circuit = [circuit]
        if circuit is not None:
            if len(circuit) > 1:
                version = None
        else:
            if exp_id is None:
                raise ValueError(
                    "When circuit is not defined, experiment id should be defined but None has been given."
                )
            data = {"exp_id": exp_id, "shots": num_shots, "is_verify": is_verify, "source": "SDK"}
            return self.handler_run_experiment_result(data)

        if circuit and parameters and values and len(parameters) == len(circuit) == len(values):
            new_circuit = _assign_parameters(circuit, parameters, values)
            if not new_circuit:
                logger.error("Unable to assign a value to the circuits")
                return 0
        else:
            new_circuit = circuit

        data = {
            "exp_id": exp_id,
            "lab_id": lab_id,
            "inputCode": new_circuit,
            "languageCode": language.value,
            "name": exp_name,
            "shots": num_shots,
            "source": "SDK",
            "computerCode": self.machine_name,
            "experimentDetailName": version,
            "is_verify": is_verify,
        }
        if "noise" in kwargs:
            data["noise"] = kwargs["noise"]
        elif "quantum_state" in kwargs:
            data["quantumState"] = kwargs["quantum_state"]
        return self.handler_run_experiment_result(data)

    def handler_run_experiment_result(self, data):
        """Handler run experiment result.

        Args:
            data: Input data array.

        Returns:
            Result.
        """
        result = self._send_request(path=self.CREATE_EXP_AND_RUN_PATH, data=data, method="POST")
        if result == 0:
            return 0
        return result.get("data").get("query_ids")

    def query_experiment(self, query_id: Union[str, List[str]], max_wait_time: int = 120, sleep_time: int = 5):
        """Query experiment.

        Args:
            query_id (*Union[str, List[str]]*): Query id (``Union[str, List[str]]``).
            max_wait_time (*int*): Maximum wait time in seconds. Defaults to ``120``.
            sleep_time (*int*): Polling interval in seconds. Defaults to ``5``.

        Returns:
            Result.

        Raises:
            CqlibRequestError: Failed to query the experimental result.
        """
        if isinstance(query_id, str):
            query_id = [query_id]
        last_time = time.time() + max_wait_time
        while time.time() < last_time:
            try:
                data = {"query_ids": query_id}
                result = self._send_request(path=self.QUERY_EXP_PATH, data=data, method="POST")
                query_exp = result.get("data").get("experimentResultModelList")
                if query_exp and len(query_exp) == len(query_id):
                    return query_exp
            except Exception as exc:
                logger.error(exc)
            logger.info("waiting for %d seconds", sleep_time)
            time.sleep(sleep_time)
        raise CqlibRequestError("Failed to query the experimental result.")

    def download_config(self, read_time: str = None, machine: str = None):
        """Download config.

        Args:
            read_time (*str*): Read time (``str``). Defaults to ``None``.
            machine (*str*): Machine (``str``). Defaults to ``None``.

        Returns:
            Result.

        Raises:
            ValueError: f"The machine '{machine}' is not supported.
        """
        if not machine:
            machine = self.machine_name
        if not re.fullmatch(r"^[a-zA-Z0-9_-]+$", machine):
            raise ValueError(f"The machine '{machine}' is not supported.")
        result = self._send_request(
            path=f"{self.DOWNLOAD_CONFIG_PATH}/{machine}",
            method="GET",
            params={"readTime": read_time},
        )
        cfg = result.get("data")
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        return cfg

    def qcis_check_regular(self, qcis_raw: str):
        """Qcis check regular.

        Args:
            qcis_raw (*str*): Qcis raw (``str``).

        Returns:
            Result.
        """
        data = {"computerCode": self.machine_name, "qcis": qcis_raw}
        resp = self._send_request(path=self.QCIS_CHECK_REGULAR_PATH, method="POST", data=data, raise_for_code=False)
        return resp["code"] == 0

    def get_experiment_circuit(self, query_id: Union[str, List[str]]):
        """Retrieve the quantum circuit(s) associated with experiment query IDs.

        Args:
            query_id (*Union[str, List[str]]*): Query id (``Union[str, List[str]]``).

        Returns:
            Retrieved data.
        """
        if isinstance(query_id, str):
            query_id = [query_id]
        data = {"query_ids": query_id}
        result = self._send_request(path=self.GET_EXP_CIRCUIT_PATH, method="POST", data=data)
        return result.get("data")

    def query_quantum_computer_list(self):
        """Query quantum computer list.

        Returns:
            List of lists with computer information rows.
        """
        computer_list_data = self.query_quantum_computer_records()
        if not computer_list_data:
            return []
        headers = list(computer_list_data[0].keys())
        table_data = []
        for row in computer_list_data:
            row_values = [row.get(key, None) for key in headers]
            table_data.append(row_values)
        return table_data

    def query_quantum_computer_records(self) -> List[Dict[str, Any]]:
        """Query quantum computer records.

        Returns:
            List of dictionaries with computer records.
        """
        result = self._send_request(self.MACHINE_LIST_PATH)
        computer_list_data = result.get("data")
        if not isinstance(computer_list_data, list):
            return []
        status_mapping = {0: "running", 1: "calibration", 2: "under maintenance", 3: "off-line"}
        toll_mapping = {1: "free", 2: "paid"}

        normalized_rows: List[Dict[str, Any]] = []
        for item in computer_list_data:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            if "code" in item:
                normalized["machineName"] = normalized.pop("code")
            normalized["status"] = status_mapping.get(normalized.get("status"), "unknown")
            normalized["isToll"] = toll_mapping.get(normalized.get("isToll"), "unknown")
            for key in list(normalized.keys()):
                if normalized[key] is None:
                    del normalized[key]
            normalized_rows.append(normalized)
        return normalized_rows

    def re_execute_task(self, query_id: Optional[str] = None, lab_id: Optional[str] = None):
        """Re execute task.

        Args:
            query_id (*Optional[str]*): Query id (``Optional[str]``). Defaults to ``None``.
            lab_id (*Optional[str]*): Lab id (``Optional[str]``). Defaults to ``None``.

        Returns:
            Result.

        Raises:
            ValueError: Please provide lab_id or query_id.
        """
        if not lab_id and not query_id:
            raise ValueError("Please provide lab_id or query_id.")
        data = {"lab_id": lab_id, "query_id": query_id}
        result = self._send_request(self.RE_EXECUTE_TASK_PATH, method="POST", data=data)
        return result.get("data")

    def stop_running_experiments(self, lab_id: Optional[str] = None, query_id: Optional[str] = None):
        """Stop running experiments.

        Args:
            lab_id (*Optional[str]*): Lab id (``Optional[str]``). Defaults to ``None``.
            query_id (*Optional[str]*): Query id (``Optional[str]``). Defaults to ``None``.

        Returns:
            Result.

        Raises:
            ValueError: Please provide lab_id or query_id.
        """
        if not lab_id and not query_id:
            raise ValueError("Please provide lab_id or query_id.")
        data = {"lab_id": lab_id, "query_id": query_id}
        result = self._send_request(self.STOP_RUNNING_EXP_PATH, method="POST", data=data)
        return result.get("data")

    @_reconnect_on_failure
    def _send_request(self, path: str, method: str = "GET", data=None, params=None, raise_for_code=True):
        """Send an authenticated HTTP request to the platform API.

        Args:
            path (*str*): API endpoint path (appended to the base URL).
            method (*str*): HTTP method (``'GET'``, ``'POST'``, etc.). Defaults to ``'GET'``.
            data: JSON-serialisable request body. Defaults to ``None``.
            params: URL query parameters. Defaults to ``None``.
            raise_for_code: Whether to raise on non-zero response codes. Defaults to ``True``.

        Returns:
            Result.

        Raises:
            CqlibRequestError: f'Request API failed: {res.text}
        """
        url = f"{self.SCHEME}://{self.DOMAIN}{path}"
        headers = {"basicToken": self.access_token, "Authorization": f"Bearer {self.access_token}"}
        res = requests.request(method.upper(), url, json=data, headers=headers, params=params, timeout=60)
        if res.status_code != 200:
            raise CqlibRequestError(f"Request API failed: {res.text}", res.status_code)
        result = res.json()
        if raise_for_code and result.get("code", -1) != 0:
            raise CqlibRequestError(f"Request error: {res.text}")
        return result


def _to_int_qubit(value: Any) -> Optional[int]:
    """To int qubit.

    Args:
        value (*Any*): Value to set.

    Returns:
        ``Optional[int]`` result.
    """
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("Q"):
            s = s[1:]
        if s.isdigit():
            return int(s)
    return None


def _extract_qubit_id(value: Any) -> Optional[int]:
    """Extract qubit id.

    Args:
        value (*Any*): Value to set.

    Returns:
        ``Optional[int]`` result.
    """
    qid = _to_int_qubit(value)
    if qid is not None:
        return qid
    if isinstance(value, dict):
        for key in ("qubit", "qubit_id", "id", "name", "label", "index"):
            qid = _to_int_qubit(value.get(key))
            if qid is not None:
                return qid
            if key == "index":
                try:
                    return int(value.get(key))
                except Exception:
                    continue
    return None


def _normalize_coordinate(value: Any) -> Optional[List[float]]:
    """Normalize coordinate.

    Args:
        value (*Any*): Value to set.

    Returns:
        Result list.
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


def _extract_qubit_coordinates(config: Dict[str, Any]) -> Dict[int, List[float]]:
    """Extract qubit coordinates.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        Result dictionary.
    """
    out: Dict[int, List[float]] = {}

    qubits_info = config.get("qubits_info") if isinstance(config, dict) else None
    if isinstance(qubits_info, dict):
        for qkey, qvalue in qubits_info.items():
            qid = _extract_qubit_id(qkey)
            if qid is None or not isinstance(qvalue, dict):
                continue
            coordinate = _normalize_coordinate(
                qvalue.get("coordinate", qvalue.get("position", qvalue.get("location")))
            )
            if coordinate is not None:
                out[qid] = coordinate

    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        qlist = overview.get("qubits")
        if isinstance(qlist, list):
            for qitem in qlist:
                qid = _extract_qubit_id(qitem)
                if qid is None:
                    continue
                if isinstance(qitem, dict):
                    coordinate = _normalize_coordinate(
                        qitem.get("coordinate", qitem.get("position", qitem.get("location", qitem)))
                    )
                    if coordinate is not None:
                        out[qid] = coordinate

    return out


def _infer_two_qubit_basis(config: Dict[str, Any]) -> str:
    """Infer two qubit basis.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        Formatted string.
    """
    twoq = config.get("twoQubitGate", {}) if isinstance(config, dict) else {}
    if isinstance(twoq, dict):
        key_to_basis = {
            "czGate": "cz",
            "cnotGate": "cx",
            "cxGate": "cx",
            "iswapGate": "iswap",
            "ecrGate": "ecr",
            "fsimGate": "fsim",
            "fsim_value": "fsim",
        }
        for key, basis_name in key_to_basis.items():
            if key in twoq:
                return basis_name
    return "cz"


def _parse_comma_ids(value: Any) -> List[str]:
    """Parse comma ids.

    Args:
        value (*Any*): Value to set.

    Returns:
        Result list.
    """
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _extract_disabled_qubits(config: Dict[str, Any]) -> set[int]:
    """Extract disabled qubits.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        ``set[int]`` result.
    """
    keys = ["disabledQubits"]
    values: List[str] = []
    for key in keys:
        values.extend(_parse_comma_ids(config.get(key)))
    out = set()
    for value in values:
        qid = _to_int_qubit(value)
        if qid is not None:
            out.add(qid)
    return out


def _extract_disabled_couplers(config: Dict[str, Any]) -> set[str]:
    """Extract disabled couplers.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        ``set[str]`` result.
    """
    keys = ["disabledCouplers"]
    values: List[str] = []
    for key in keys:
        values.extend(_parse_comma_ids(config.get(key)))
    return {v.strip() for v in values if v.strip()}


def _extract_couplers(config: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """Extract couplers.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        Result list.
    """
    disabled_qubits = _extract_disabled_qubits(config)
    disabled_couplers = _extract_disabled_couplers(config)
    out: List[Tuple[str, int, int]] = []

    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        coupler_map = overview.get("coupler_map")
        if isinstance(coupler_map, dict):
            for coupler_key, pair in coupler_map.items():
                if str(coupler_key) in disabled_couplers:
                    continue
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    q1 = _to_int_qubit(pair[0])
                    q2 = _to_int_qubit(pair[1])
                    if q1 is not None and q2 is not None and q1 != q2 and q1 not in disabled_qubits and q2 not in disabled_qubits:
                        out.append((str(coupler_key), q1, q2))

    dedup: List[Tuple[str, int, int]] = []
    seen = set()
    for coupler_name, q1, q2 in out:
        key = tuple(sorted((q1, q2)))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((coupler_name, q1, q2))
    return dedup


def _extract_qubits(config: Dict[str, Any], couplers: Sequence[Tuple[str, int, int]]) -> List[int]:
    """Extract qubits.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.
        couplers (*Sequence[Tuple[str, int, int]]*): List of qubit coupler pairs.

    Returns:
        Result list.
    """
    qubits = set()
    disabled_qubits = _extract_disabled_qubits(config)

    for _, q1, q2 in couplers:
        qubits.add(int(q1))
        qubits.add(int(q2))

    overview = config.get("overview") if isinstance(config, dict) else None
    if isinstance(overview, dict):
        qlist = overview.get("qubits")
        if isinstance(qlist, list):
            for qubit in qlist:
                qid = _extract_qubit_id(qubit)
                if qid is not None:
                    qubits.add(qid)

    return sorted(int(q) for q in qubits if int(q) not in disabled_qubits)


def _normalize_error_to_fidelity(value: Any) -> float:
    """Normalize error to fidelity.

    Args:
        value (*Any*): Value to set.

    Returns:
        Computed float result.
    """
    try:
        error_value = float(value)
    except Exception:
        return 1.0

    error_rate = error_value / 100.0

    fidelity = 1.0 - error_rate
    if fidelity < 0.0:
        return 0.0
    if fidelity > 1.0:
        return 1.0
    return fidelity


def _extract_qubit_fidelity_map(config: Dict[str, Any]) -> Dict[int, float]:
    """Extract qubit fidelity map.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        Result dictionary.
    """
    gate_error = (
        config.get("qubit", {})
        .get("singleQubit", {})
        .get("gate error", {})
        if isinstance(config, dict)
        else {}
    )
    if not isinstance(gate_error, dict):
        return {}

    qubit_used = gate_error.get("qubit_used")
    param_list = gate_error.get("param_list")
    if not isinstance(qubit_used, list) or not isinstance(param_list, list):
        return {}

    out: Dict[int, float] = {}
    for qubit_name, error_value in zip(qubit_used, param_list):
        qid = _to_int_qubit(qubit_name)
        if qid is None:
            continue
        out[qid] = _normalize_error_to_fidelity(error_value)
    return out


def _extract_coupler_fidelity_map(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract coupler fidelity map.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.

    Returns:
        Result dictionary.
    """
    if not isinstance(config, dict):
        return {}

    twoq = config.get("twoQubitGate")
    if not isinstance(twoq, dict):
        return {}

    out: Dict[str, float] = {}
    for gate_payload in twoq.values():
        if not isinstance(gate_payload, dict):
            continue
        gate_error = gate_payload.get("gate error")
        if not isinstance(gate_error, dict):
            continue
        qubit_used = gate_error.get("qubit_used")
        param_list = gate_error.get("param_list")
        if not isinstance(qubit_used, list) or not isinstance(param_list, list):
            continue
        for coupler_name, error_value in zip(qubit_used, param_list):
            name = str(coupler_name).strip()
            if not name:
                continue
            out[name] = _normalize_error_to_fidelity(error_value)
        if out:
            return out
    return out


def chip_info_from_config(config: Dict[str, Any], *, machine_name: Optional[str] = None) -> Dict[str, Any]:
    """Chip info from config.

    Args:
        config (*Dict[str, Any]*): Configuration dictionary.
        machine_name (*Optional[str]*): Identifier of the target quantum machine. Defaults to ``None``.

    Returns:
        Result dictionary.
    """
    couplers = _extract_couplers(config)
    qubits = _extract_qubits(config, couplers)
    qubit_coordinates = _extract_qubit_coordinates(config)
    qubit_fidelity_map = _extract_qubit_fidelity_map(config)
    coupler_fidelity_map = _extract_coupler_fidelity_map(config)
    twoq_basis = _infer_two_qubit_basis(config)
    qubits_info = {}
    for q in qubits:
        qentry: Dict[str, Any] = {"fidelity": qubit_fidelity_map.get(int(q), 1.0)}
        if int(q) in qubit_coordinates:
            qentry["coordinate"] = qubit_coordinates[int(q)]
        qubits_info[f"Q{q}"] = qentry

    couplers_info = {}
    for idx, (coupler_name, q1, q2) in enumerate(couplers):
        couplers_info[f"C{idx}"] = {
            "qubits_index": [int(q1), int(q2)],
            "fidelity": coupler_fidelity_map.get(coupler_name, 1.0),
        }

    chip_info = {
        "chip_name": machine_name or " ",
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": {
            "two_qubit_gate_basis": twoq_basis,
            "nqubits_available": len(qubits),
            "error_rate_2q": 0.0,
            "one_qubit_gate_length": 1e-8,
            "two_qubit_gate_length": 1e-6,
        },
        "priority_qubits": None
    }
    return chip_info


def _infer_provider_from_chip_name(chip_name: str) -> str:
    """Infer provider from chip name.

    Args:
        chip_name (*str*): Name of the target chip.

    Returns:
        Formatted string.

    Raises:
        ValueError: f'Wrong chip name! {chip_name}
    """
    normalized = str(chip_name).strip()
    if normalized in TIANYAN_HARDWARE_NAMES:
        return "tianyan"
    if normalized in GUODUN_HARDWARE_NAMES:
        return "guodun"
    raise ValueError(f"Wrong chip name! {chip_name}")


def load_cqlib_chip_info(
    chip_name: str,
    *,
    provider: Optional[str] = None,
    platform: Optional[RemotePlatformClient] = None,
) -> Dict[str, Any]:
    """Load cqlib chip info.

    Args:
        chip_name (*str*): Name of the target chip.
        provider (*Optional[str]*): Platform provider name (``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``). Defaults to ``None``.
        platform (*Optional[RemotePlatformClient]*): Platform (``Optional[RemotePlatformClient]``). Defaults to ``None``.

    Returns:
        Result dictionary.

    Raises:
        ValueError: chip_name cannot be empty
    """
    machine_name = str(chip_name).strip()
    if not machine_name:
        raise ValueError("chip_name cannot be empty")

    if platform is None:
        resolved_provider = provider or _infer_provider_from_chip_name(machine_name)
        if resolved_provider == "tianyan":
            from .tianyan import TianYanPlatform, get_tianyan_login_key

            login_key = get_tianyan_login_key()
            if not login_key:
                raise ValueError("tianyan login key cannot be empty")
            platform = TianYanPlatform(login_key=login_key, auto_login=True, machine_name=machine_name)
        elif resolved_provider == "guodun":
            from .guodun import GuoDunPlatform, get_guodun_login_key

            login_key = get_guodun_login_key()
            if not login_key:
                raise ValueError("guodun login key cannot be empty")
            platform = GuoDunPlatform(login_key=login_key, auto_login=True, machine_name=machine_name)
        else:
            raise ValueError(f"Unsupported provider: {resolved_provider}")

    config = load_backend_config(platform, machine_name=machine_name)
    return chip_info_from_config(config, machine_name=machine_name)


def load_backend_config(platform: RemotePlatformClient, *, machine_name: str) -> Dict[str, Any]:
    """Load backend config.

    Args:
        platform (*RemotePlatformClient*): Platform (``RemotePlatformClient``).
        machine_name (*str*): Identifier of the target quantum machine.

    Returns:
        Result dictionary.
    """
    try:
        config = platform.download_config(machine=machine_name)
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


__all__ = [
    "TIANYAN_HARDWARE_NAMES",
    "GUODUN_HARDWARE_NAMES",
    "QuantumLanguage",
    "format_circuit",
    "extract_counts_from_result_items",
    "RemotePlatformClient",
    "chip_info_from_config",
    "load_cqlib_chip_info",
    "load_backend_config",
]