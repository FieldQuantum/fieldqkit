import types
import math

from quantum_hw.api.quantum_platform import guodun as gd
from quantum_hw.api.quantum_platform import quafu as qf
from quantum_hw.api.quantum_platform import tianyan as ty
from quantum_hw.api import task as ut
from quantum_hw.api.backend import ResolvedBackend
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.vendor.cqlib.utils.qasm_to_qcis.qasm_to_qcis import QasmToQcis
class _DummyClient:
    def __init__(self):
        self.tmgr = types.SimpleNamespace(
            run=self._run,
            status=lambda tid: "Finished",
            result=lambda tid: {"count": {"0": 10}},
            cancel=self._cancel,
        )
        self.submitted_task = None
        self.canceled = None

    def _run(self, task):
        self.submitted_task = task
        return 123

    def _cancel(self, tid):
        self.canceled = tid

def test_quafu_task_adapter_submit_query_fetch_cancel_lifecycle():
    client = _DummyClient()
    adapter = qf.QuafuTaskAdapter(client=client)
    backend = ResolvedBackend(
        provider="quafu",
        hardware_name="chip",
        backend="backend_obj",
        metadata={"platform_obj": client.tmgr},
    )

    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(name="n", qasm="OPENQASM 2.0;", shots=100, chip_name="chip"),
        backend,
    )

    assert isinstance(handle, ut.ProviderTaskHandle)
    assert handle.provider == "quafu"
    assert client.submitted_task["name"] == "n"
    assert client.submitted_task["chip"] == "chip"

    assert adapter.query_status(handle) == "Finished"
    assert adapter.fetch_result(handle) == {"count": {"0": 10}}

    adapter.cancel_task(handle)
    assert client.canceled == 123

def test_vendored_adapter_submit_openqasm_submit_job_and_fetch_result(monkeypatch):
    qasm_module = types.ModuleType("quantum_hw.vendor.cqlib.utils.qasm_to_qcis.qasm_to_qcis")

    class _FakeConverter:
        def convert_to_qcis(self, qasm):
            return f"QCIS::{qasm}"

    qasm_module.QasmToQcis = _FakeConverter

    sysmod = __import__("sys").modules
    monkeypatch.setitem(sysmod, "quantum_hw.vendor.cqlib.utils.qasm_to_qcis.qasm_to_qcis", qasm_module)

    class _Platform:
        def __init__(self):
            self.last_submit = None

        def submit_job(self, **kwargs):
            self.last_submit = kwargs
            return ["qid1"]

        def query_experiment(self, query_id, max_wait_time, sleep_time):
            del max_wait_time, sleep_time
            assert query_id == "qid1"
            return [{"resultStatus": [[0], [1], [0], [1]]}]

        def stop_running_experiments(self, query_id=None):
            self.stopped = query_id

    platform = _Platform()
    backend = ResolvedBackend(
        provider="tianyan",
        hardware_name="m",
        backend="b",
        metadata={"platform_obj": platform},
    )

    adapter = ty.TianYanTaskAdapter(client=_DummyClient(), login_key="k")
    handle = adapter.submit_openqasm(
        ut.OpenQasmSubmitRequest(
            name="exp",
            qasm="OPENQASM 2.0;",
            shots=20,
            chip_name="m",
            submit_options={"num_qubits": 1},
        ),
        backend,
    )

    assert handle.task_id == "qid1"
    assert platform.last_submit["exp_name"] == "exp"
    assert platform.last_submit["circuit"].startswith("QCIS::")

    status = adapter.query_status(handle)
    assert status == "Finished"

    result = adapter.fetch_result(handle)
    assert result["count"] == {"1": 2, "0": 1}

    adapter.cancel_task(handle)
    assert getattr(platform, "stopped", None) == "qid1"


def test_tianyan_adapter_provider_value():
    adapter = ty.TianYanTaskAdapter(client=_DummyClient(), login_key="k")
    assert adapter.provider == "tianyan"


def test_guodun_adapter_provider_value():
    adapter = gd.GuoDunTaskAdapter(client=_DummyClient(), login_key="k")
    assert adapter.provider == "guodun"


def test_qasm3_delay_can_convert_to_qcis_idle_instruction():
    qasm = """
OPENQASM 3.0;
include \"stdgates.inc\";
qubit[1] q;
bit[1] c;
delay[5] q[0];
c[0] = measure q[0];
"""
    qcis = QasmToQcis().convert_to_qcis(qasm)
    lines = [line.strip().upper() for line in qcis.splitlines() if line.strip()]
    assert any(line.startswith("I Q0 ") for line in lines)
    assert any(line == "M Q0" for line in lines)


def test_qasm3_generated_with_defcalgrammar_and_delay_can_convert_to_qcis():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.delay(2e-6, 0)
    qc.measure([0], [0])

    qcis = QasmToQcis().convert_to_qcis(qc.to_openqasm3)
    lines = [line.strip().upper() for line in qcis.splitlines() if line.strip()]

    assert any(line.startswith("I Q0 ") for line in lines)
    assert any(line == "M Q0" for line in lines)


def test_qasm3_duration_literal_ns_converts_to_seconds_in_qcis_delay():
    qasm = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
delay[5ns] q[0];
"""
    qcis = QasmToQcis().convert_to_qcis(qasm)
    first_line = [line.strip() for line in qcis.splitlines() if line.strip()][0]
    duration = float(first_line.split()[-1])
    assert math.isclose(duration, 5e-9, rel_tol=0.0, abs_tol=1e-15)


def test_quantumcircuit_delay_unit_argument_is_normalized_to_seconds():
    qc = QuantumCircuit(1)
    qc.delay(5, 0, unit="ns")
    gate = qc.gates[-1]
    assert gate[0] == "delay"
    assert math.isclose(gate[1], 5e-9, rel_tol=0.0, abs_tol=1e-15)
