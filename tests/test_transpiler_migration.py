import pytest

try:
    from quark.circuit import QuantumCircuit
    from quark.circuit import Transpiler as QuarkTranspiler
    from quark.circuit import Backend
except Exception:
    QuantumCircuit = None
    QuarkTranspiler = None
    Backend = None

from quantum_hw.compile import Transpiler as LocalTranspiler


def _build_reference_circuit() -> "QuantumCircuit":
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(1.234, 1)
    qc.rz(-0.25, 2)
    qc.swap(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _build_complex_circuit() -> "QuantumCircuit":
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.ccx(0, 1, 2)
    qc.ccz(0, 1, 2)
    qc.cswap(0, 1, 2)
    qc.cy(1, 2)
    qc.rxx(0.7, 0, 1)
    qc.ryy(-0.4, 1, 2)
    qc.rzz(0.9, 0, 2)
    qc.cp(0.25, 0, 1)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _build_param_circuit() -> "QuantumCircuit":
    qc = QuantumCircuit(1, 1)
    qc.rx(0.125, 0)
    qc.measure([0], [0])
    return qc


def _build_custom_backend() -> "Backend":
    chip_info = {
        "size": (1, 3),
        "priority_qubits": [[0, 1, 2]],
        "global_info": {"two_qubit_gate_basis": "cz"},
        "qubits_info": {
            "Q0": {"fidelity": 0.999, "coordinate": (0, 0)},
            "Q1": {"fidelity": 0.999, "coordinate": (0, 1)},
            "Q2": {"fidelity": 0.999, "coordinate": (0, 2)},
        },
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.999, "index": 0},
            "C1": {"qubits_index": [1, 2], "fidelity": 0.999, "index": 1},
        },
    }
    return Backend(chip_info)


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
@pytest.mark.parametrize("optimize_level", [0, 1])
def test_transpiler_matches_quark_on_quantumcircuit(optimize_level):
    qc = _build_reference_circuit()

    qct_ref = QuarkTranspiler(None).run(qc, optimize_level=optimize_level, niter=2, use_dd=False)
    qct_new = LocalTranspiler(None).run(qc, optimize_level=optimize_level, niter=2, use_dd=False)

    assert qct_new.gates == qct_ref.gates
    assert qct_new.qubits == qct_ref.qubits
    assert qct_new.nqubits == qct_ref.nqubits
    assert qct_new.ncbits == qct_ref.ncbits


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
@pytest.mark.parametrize("optimize_level", [0, 1])
def test_transpiler_matches_quark_on_openqasm(optimize_level):
    qc = _build_reference_circuit()
    qasm = qc.to_openqasm2

    qct_ref = QuarkTranspiler(None).run(qasm, optimize_level=optimize_level, niter=2, use_dd=False)
    qct_new = LocalTranspiler(None).run(qasm, optimize_level=optimize_level, niter=2, use_dd=False)

    assert qct_new.gates == qct_ref.gates
    assert qct_new.qubits == qct_ref.qubits
    assert qct_new.nqubits == qct_ref.nqubits
    assert qct_new.ncbits == qct_ref.ncbits


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
@pytest.mark.parametrize("optimize_level", [0, 1])
def test_transpiler_matches_quark_on_qlisp(optimize_level):
    qc = _build_reference_circuit()
    qlisp = qc.to_qlisp

    qct_ref = QuarkTranspiler(None).run(qlisp, optimize_level=optimize_level, niter=2, use_dd=False)
    qct_new = LocalTranspiler(None).run(qlisp, optimize_level=optimize_level, niter=2, use_dd=False)

    assert qct_new.gates == qct_ref.gates
    assert qct_new.qubits == qct_ref.qubits
    assert qct_new.nqubits == qct_ref.nqubits
    assert qct_new.ncbits == qct_ref.ncbits


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_optimize_level_2_rejects_split_qubits():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.z(1)
    with pytest.raises(ValueError):
        QuarkTranspiler(None).run(qc, optimize_level=2)
    with pytest.raises(ValueError):
        LocalTranspiler(None).run(qc, optimize_level=2)


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_invalid_input_type():
    with pytest.raises(TypeError):
        QuarkTranspiler(None).run(123)
    with pytest.raises(TypeError):
        LocalTranspiler(None).run(123)


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_use_dd_flag_consistency():
    qc = _build_reference_circuit()
    qct_ref = QuarkTranspiler(None).run(qc, optimize_level=1, niter=2, use_dd=True)
    qct_new = LocalTranspiler(None).run(qc, optimize_level=1, niter=2, use_dd=True)

    assert qct_new.gates == qct_ref.gates
    assert qct_new.qubits == qct_ref.qubits
    assert qct_new.nqubits == qct_ref.nqubits
    assert qct_new.ncbits == qct_ref.ncbits


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
@pytest.mark.parametrize("optimize_level", [0, 1])
def test_transpiler_matches_quark_on_complex_circuit(optimize_level):
    qc = _build_complex_circuit()

    qct_ref = QuarkTranspiler(None).run(qc, optimize_level=optimize_level, niter=2, use_dd=False)
    qct_new = LocalTranspiler(None).run(qc, optimize_level=optimize_level, niter=2, use_dd=False)

    assert qct_new.gates == qct_ref.gates
    assert qct_new.qubits == qct_ref.qubits
    assert qct_new.nqubits == qct_ref.nqubits
    assert qct_new.ncbits == qct_ref.ncbits


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_preserves_params_value():
    qc = _build_param_circuit()

    qct_ref = QuarkTranspiler(None).run(qc, optimize_level=1, niter=2, use_dd=False)
    qct_new = LocalTranspiler(None).run(qc, optimize_level=1, niter=2, use_dd=False)

    assert qct_new.params_value == qct_ref.params_value


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_matches_quark_with_custom_backend():
    backend = _build_custom_backend()
    qc = _build_reference_circuit()
    target_qubits = [0, 1, 2]

    qct_ref = QuarkTranspiler(backend).run(qc, target_qubits=target_qubits, optimize_level=1, niter=2, use_dd=False)
    qct_new = LocalTranspiler(backend).run(qc, target_qubits=target_qubits, optimize_level=1, niter=2, use_dd=False)

    assert qct_new.gates == qct_ref.gates
    assert qct_new.qubits == qct_ref.qubits
    assert qct_new.nqubits == qct_ref.nqubits
    assert qct_new.ncbits == qct_ref.ncbits


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_target_qubits_length_mismatch():
    backend = _build_custom_backend()
    qc = _build_reference_circuit()

    with pytest.raises(ValueError):
        QuarkTranspiler(backend).run(qc, target_qubits=[0, 1], optimize_level=1, niter=2, use_dd=False)
    with pytest.raises(ValueError):
        LocalTranspiler(backend).run(qc, target_qubits=[0, 1], optimize_level=1, niter=2, use_dd=False)


@pytest.mark.skipif(QuarkTranspiler is None, reason="quark is not installed")
def test_transpiler_target_qubits_missing_in_backend():
    backend = _build_custom_backend()
    qc = _build_reference_circuit()

    with pytest.raises(ValueError):
        QuarkTranspiler(backend).run(qc, target_qubits=[0, 1, 9], optimize_level=1, niter=2, use_dd=False)
    with pytest.raises(ValueError):
        LocalTranspiler(backend).run(qc, target_qubits=[0, 1, 9], optimize_level=1, niter=2, use_dd=False)
