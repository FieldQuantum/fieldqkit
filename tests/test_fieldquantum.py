"""Tests for the FieldQuantum cloud-simulator provider adapter.

All HTTP traffic is mocked: ``get_fieldquantum_api_token`` is patched so the
platform constructs without credentials, and ``requests``-style responses /
sessions are replaced with in-memory fakes. No network calls are made.
"""

import types

import pytest

import fieldqkit.api.quantum_platform.fieldquantum as fq
from fieldqkit.api.quantum_platform.fieldquantum import (
    FieldQuantumPlatform,
    FieldQuantumBackendAdapter,
    FieldQuantumTaskAdapter,
    _STATUS_MAP,
)
from fieldqkit.api.task import ProviderTaskHandle


@pytest.fixture(autouse=True)
def _patch_token(monkeypatch):
    """Avoid needing real credentials when constructing the platform."""
    monkeypatch.setattr(fq, "get_fieldquantum_api_token", lambda: "fq_" + "0" * 32)


# ─────────────────────────────────────────────────────────────
#  HTTP fakes
# ─────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", reason="OK", raise_json=False):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.reason = reason
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("no json body")
        return self._json


class FakeSession:
    """Routes POST/GET by URL suffix to queued FakeResponses and records calls."""

    def __init__(self):
        self.headers = {}
        self.post_response = None
        self.status_response = None
        self.result_response = None
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self.post_response

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        if "/task/status/" in url:
            return self.status_response
        if "/task/result/" in url:
            return self.result_response
        raise AssertionError(f"unexpected GET {url}")


def _platform_with_session():
    platform = FieldQuantumPlatform()
    session = FakeSession()
    platform._session = session
    return platform, session


# ─────────────────────────────────────────────────────────────
#  _extract_error / _raise_for_response
# ─────────────────────────────────────────────────────────────

class TestResponseHandling:
    def test_extract_error_precedence(self):
        assert FieldQuantumPlatform._extract_error({"message": "m", "error": "e"}, "fb") == "m"
        assert FieldQuantumPlatform._extract_error({"error": "e"}, "fb") == "e"
        assert FieldQuantumPlatform._extract_error({}, "fb") == "fb"
        assert FieldQuantumPlatform._extract_error("not a dict", "fb") == "fb"

    def test_raise_for_response_ok(self):
        platform, _ = _platform_with_session()
        data = platform._raise_for_response(FakeResponse(200, {"task_id": 7}))
        assert data == {"task_id": 7}

    def test_raise_for_response_http_error(self):
        platform, _ = _platform_with_session()
        resp = FakeResponse(500, {"message": "boom"}, text="boom")
        with pytest.raises(RuntimeError, match="HTTP 500: boom"):
            platform._raise_for_response(resp)

    def test_raise_for_response_error_body(self):
        platform, _ = _platform_with_session()
        resp = FakeResponse(200, {"error": "bad request"})
        with pytest.raises(RuntimeError, match="server error: bad request"):
            platform._raise_for_response(resp)

    def test_raise_for_response_error_with_result_is_ok(self):
        # "error" alongside "result" must not raise.
        platform, _ = _platform_with_session()
        data = platform._raise_for_response(FakeResponse(200, {"error": "x", "result": {"counts": {}}}))
        assert data["result"] == {"counts": {}}

    def test_raise_for_response_non_json_ok(self):
        platform, _ = _platform_with_session()
        data = platform._raise_for_response(FakeResponse(200, raise_json=True))
        assert data == {}


# ─────────────────────────────────────────────────────────────
#  Platform API methods
# ─────────────────────────────────────────────────────────────

class TestPlatformApi:
    def test_list_available_hardware(self):
        platform, _ = _platform_with_session()
        hw = platform.list_available_hardware()
        assert len(hw) == 1
        assert hw[0]["provider"] == "fieldquantum"
        assert hw[0]["hardware_name"] == "fieldquantum_sim"

    def test_submit_job_returns_str_task_id(self):
        platform, session = _platform_with_session()
        session.post_response = FakeResponse(200, {"task_id": 123, "status": "submitted"})
        task_id = platform.submit_job({"mode": "sample", "qasm": "...", "shots": 100})
        assert task_id == "123"  # integer id coerced to str
        assert session.calls[0][0] == "POST"
        assert session.calls[0][1].endswith("/task/run")
        assert session.calls[0][2]["mode"] == "sample"

    def test_query_task_status(self):
        platform, session = _platform_with_session()
        session.status_response = FakeResponse(200, {"task_id": 1, "status": "running"})
        assert platform.query_task_status("1") == "running"

    def test_query_task_status_missing_defaults_unknown(self):
        platform, session = _platform_with_session()
        session.status_response = FakeResponse(200, {"task_id": 1})
        assert platform.query_task_status("1") == "unknown"

    def test_fetch_task_result_success(self):
        platform, session = _platform_with_session()
        session.result_response = FakeResponse(
            200, {"task_id": 1, "status": "finished", "result": {"counts": {"00": 50, "11": 50}}}
        )
        result = platform.fetch_task_result("1")
        assert result == {"counts": {"00": 50, "11": 50}}

    def test_fetch_task_result_not_ready_425(self):
        platform, session = _platform_with_session()
        session.result_response = FakeResponse(425, {})
        with pytest.raises(RuntimeError, match="not ready yet"):
            platform.fetch_task_result("1")

    def test_fetch_task_result_failed_status(self):
        platform, session = _platform_with_session()
        # Uses "message" (not "error") so _raise_for_response passes and the
        # status=="failed" branch of fetch_task_result is what raises.
        session.result_response = FakeResponse(
            200, {"task_id": 1, "status": "failed", "message": "circuit too large"}
        )
        with pytest.raises(RuntimeError, match="failed: circuit too large"):
            platform.fetch_task_result("1")

    def test_fetch_task_result_bad_payload(self):
        platform, session = _platform_with_session()
        session.result_response = FakeResponse(200, {"task_id": 1, "status": "finished", "result": "oops"})
        with pytest.raises(RuntimeError, match="unexpected result payload"):
            platform.fetch_task_result("1")


# ─────────────────────────────────────────────────────────────
#  run_expectation polling loop
# ─────────────────────────────────────────────────────────────

class TestRunExpectation:
    def test_polls_until_finished(self, monkeypatch):
        platform, _ = _platform_with_session()
        monkeypatch.setattr(fq.time, "sleep", lambda *_: None)  # no real waiting

        statuses = iter(["submitted", "running", "finished"])
        monkeypatch.setattr(platform, "submit_job", lambda payload: "55")
        monkeypatch.setattr(platform, "query_task_status", lambda tid: next(statuses))
        monkeypatch.setattr(
            platform, "fetch_task_result",
            lambda tid: {"energy": -1.0, "expectations": {"ZZ": -1.0}, "gradients": [0.0]},
        )
        out = platform.run_expectation("qasm", ["t0"], [0.1], [{"coeff": 1.0, "pauli": "ZZ"}])
        assert out["energy"] == -1.0

    def test_times_out(self, monkeypatch):
        platform, _ = _platform_with_session()
        monkeypatch.setattr(fq.time, "sleep", lambda *_: None)
        monkeypatch.setattr(platform, "submit_job", lambda payload: "55")
        monkeypatch.setattr(platform, "query_task_status", lambda tid: "running")
        # monotonic jumps past the deadline on the second read.
        ticks = iter([0.0, 1.0, 1000.0, 1000.0])
        monkeypatch.setattr(fq.time, "monotonic", lambda: next(ticks))
        with pytest.raises(TimeoutError, match="timed out"):
            platform.run_expectation("qasm", [], [], [], timeout=10.0)


# ─────────────────────────────────────────────────────────────
#  Status normalisation
# ─────────────────────────────────────────────────────────────

class TestStatusMap:
    @pytest.mark.parametrize("raw,expected", [
        ("submitted", "Running"),
        ("queued", "Running"),
        ("running", "Running"),
        ("finished", "Finished"),
        ("failed", "Failed"),
        ("error", "Failed"),
    ])
    def test_status_map(self, raw, expected):
        assert _STATUS_MAP[raw] == expected


# ─────────────────────────────────────────────────────────────
#  Backend adapter
# ─────────────────────────────────────────────────────────────

class TestBackendAdapter:
    def test_resolve_backend(self):
        adapter = FieldQuantumBackendAdapter(num_qubits=8)
        resolved = adapter.resolve_backend(num_qubits=4)
        assert resolved.provider == "fieldquantum"
        assert resolved.hardware_name == "fieldquantum_sim"
        assert isinstance(resolved.metadata.get("platform_obj"), FieldQuantumPlatform)
        assert resolved.backend.chip_info["chip_name"] == "fieldquantum_sim"

    def test_resolve_backend_min_one_qubit(self):
        adapter = FieldQuantumBackendAdapter()
        resolved = adapter.resolve_backend(num_qubits=0)
        # nq is clamped to >= 1.
        assert resolved.backend is not None


# ─────────────────────────────────────────────────────────────
#  Task adapter
# ─────────────────────────────────────────────────────────────

class _StubPlatform:
    def __init__(self):
        self.base_url = "http://stub"
        self.submitted = None
        self.raw_status = "running"
        self.result = {"counts": {"00": 10, "11": 6}}

    def submit_job(self, payload):
        self.submitted = payload
        return "777"

    def query_task_status(self, task_id):
        return self.raw_status

    def fetch_task_result(self, task_id):
        return self.result


class TestTaskAdapter:
    def test_submit_openqasm_uses_backend_platform(self):
        adapter = FieldQuantumTaskAdapter(client=None)
        stub = _StubPlatform()
        backend = types.SimpleNamespace(metadata={"platform_obj": stub})
        req = types.SimpleNamespace(qasm="OPENQASM 2.0;", shots=2048)
        handle = adapter.submit_openqasm(req, backend)
        assert isinstance(handle, ProviderTaskHandle)
        assert handle.provider == "fieldquantum"
        assert handle.task_id == "777"
        assert stub.submitted == {"mode": "sample", "qasm": "OPENQASM 2.0;", "shots": 2048}

    def test_query_status_normalises(self):
        adapter = FieldQuantumTaskAdapter(client=None)
        stub = _StubPlatform()
        adapter._platform = stub
        handle = ProviderTaskHandle(provider="fieldquantum", task_id="1", payload={})

        stub.raw_status = "finished"
        assert adapter.query_status(handle) == "Finished"
        stub.raw_status = "queued"
        assert adapter.query_status(handle) == "Running"
        stub.raw_status = "error"
        assert adapter.query_status(handle) == "Failed"
        # Unknown status falls back to "Running" (keep polling).
        stub.raw_status = "something_new"
        assert adapter.query_status(handle) == "Running"

    def test_fetch_result_wraps_counts(self):
        adapter = FieldQuantumTaskAdapter(client=None)
        stub = _StubPlatform()
        adapter._platform = stub
        handle = ProviderTaskHandle(provider="fieldquantum", task_id="1", payload={})
        assert adapter.fetch_result(handle) == {"count": {"00": 10, "11": 6}}

    def test_fetch_result_missing_counts(self):
        adapter = FieldQuantumTaskAdapter(client=None)
        stub = _StubPlatform()
        stub.result = {}  # no "counts" key
        adapter._platform = stub
        handle = ProviderTaskHandle(provider="fieldquantum", task_id="1", payload={})
        assert adapter.fetch_result(handle) == {"count": {}}
