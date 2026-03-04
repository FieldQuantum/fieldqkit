import pytest

try:
    from quark.circuit import QuantumCircuit as QuarkCircuit
    from quark.circuit.quantumcircuit_helpers import (
        add_gates_to_lines as quark_add_gates_to_lines,
        convert_gate_info_to_dag_info as quark_convert_gate_info_to_dag_info,
        parse_openqasm2_to_gates as quark_parse_openqasm2_to_gates,
    )
except Exception:
    QuarkCircuit = None
    quark_add_gates_to_lines = None
    quark_convert_gate_info_to_dag_info = None
    quark_parse_openqasm2_to_gates = None

from quantum_hw.circuit import QuantumCircuit as LocalCircuit
from quantum_hw.circuit.qasm2 import parse_openqasm2_to_gates as local_parse_openqasm2_to_gates
from quantum_hw.circuit.quantumcircuit_helpers import (
    add_gates_to_lines as local_add_gates_to_lines,
    convert_gate_info_to_dag_info as local_convert_gate_info_to_dag_info,
)


@pytest.mark.skipif(QuarkCircuit is None, reason="quark is not installed")
def test_circuit_matches_quark_on_gate_sequence():
    qc_ref = _build_full_circuit(QuarkCircuit)
    qc_local = _build_full_circuit(LocalCircuit)

    assert qc_local.gates == qc_ref.gates
    assert qc_local.qubits == qc_ref.qubits
    assert qc_local.nqubits == qc_ref.nqubits
    assert qc_local.ncbits == qc_ref.ncbits
    assert qc_local.depth == qc_ref.depth
    assert qc_local.ncz == qc_ref.ncz


@pytest.mark.skipif(QuarkCircuit is None, reason="quark is not installed")
def test_circuit_roundtrip_openqasm2():
    qc_ref = _build_io_circuit(QuarkCircuit)
    qasm = qc_ref.to_openqasm2

    qc_ref_parsed = QuarkCircuit().from_openqasm2(qasm)
    qc_local_parsed = LocalCircuit().from_openqasm2(qasm)

    assert qc_local_parsed.gates == qc_ref_parsed.gates
    assert qc_local_parsed.qubits == qc_ref_parsed.qubits
    assert qc_local_parsed.nqubits == qc_ref_parsed.nqubits
    assert qc_local_parsed.ncbits == qc_ref_parsed.ncbits


@pytest.mark.skipif(QuarkCircuit is None, reason="quark is not installed")
def test_circuit_roundtrip_openqasm2_with_pi_and_delay():
    qasm = """
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
rx(pi/2) q[0];
rz(-pi/3) q[1];
cp(pi/4) q[0],q[1];
delay(1e-6) q[0];
barrier q[0],q[1];
reset q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
""".strip()

    qc_ref_parsed = QuarkCircuit().from_openqasm2(qasm)
    qc_local_parsed = LocalCircuit().from_openqasm2(qasm)

    assert qc_local_parsed.gates == qc_ref_parsed.gates
    assert qc_local_parsed.qubits == qc_ref_parsed.qubits
    assert qc_local_parsed.nqubits == qc_ref_parsed.nqubits
    assert qc_local_parsed.ncbits == qc_ref_parsed.ncbits


@pytest.mark.skipif(QuarkCircuit is None, reason="quark is not installed")
def test_circuit_params_value_and_apply():
    qc_ref = _build_param_circuit(QuarkCircuit)
    qc_local = _build_param_circuit(LocalCircuit)

    assert qc_local.params_value == qc_ref.params_value

    params = {"theta": 0.123, "phi": -0.456, "gamma": 0.789}
    qc_ref.shallow_apply_value(params)
    qc_local.shallow_apply_value(params)

    assert qc_local.gates == qc_ref.gates

    qc_ref.deep_apply_value(params)
    qc_local.deep_apply_value(params)

    assert qc_local.gates == qc_ref.gates
    assert qc_local.params_value["theta"] == params["theta"]
    assert qc_local.params_value["phi"] == params["phi"]
    assert qc_local.params_value["gamma"] == params["gamma"]


@pytest.mark.skipif(QuarkCircuit is None, reason="quark is not installed")
def test_circuit_remove_and_count_gates():
    qc_ref = _build_barrier_circuit(QuarkCircuit)
    qc_local = _build_barrier_circuit(LocalCircuit)

    qc_ref.remove_barrier()
    qc_local.remove_barrier()
    assert qc_local.gates == qc_ref.gates

    qc_ref.remove_gate("rx")
    qc_local.remove_gate("rx")
    assert qc_local.gates == qc_ref.gates


@pytest.mark.skipif(QuarkCircuit is None, reason="quark is not installed")
def test_helpers_match_quark_outputs():
    qc_ref = _build_io_circuit(QuarkCircuit)
    qasm = qc_ref.to_openqasm2

    local_gates, local_qubits, local_cbits = local_parse_openqasm2_to_gates(qasm)
    ref_gates, ref_qubits, ref_cbits = quark_parse_openqasm2_to_gates(qasm)
    assert local_gates == ref_gates
    assert local_qubits == ref_qubits
    assert local_cbits == ref_cbits

    local_nodes, local_edges = local_convert_gate_info_to_dag_info(
        qc_ref.nqubits,
        qc_ref.qubits,
        qc_ref.gates,
        show_qubits=True,
    )
    ref_nodes, ref_edges = quark_convert_gate_info_to_dag_info(
        qc_ref.nqubits,
        qc_ref.qubits,
        qc_ref.gates,
        show_qubits=True,
    )
    
    assert local_nodes.tolist() == ref_nodes.tolist()
    assert local_edges.tolist() == ref_edges.tolist()

    local_lines, local_lines_use = local_add_gates_to_lines(
        qc_ref.nqubits,
        qc_ref.ncbits,
        qc_ref.gates,
        qc_ref.params_value,
        width=4,
    )
    ref_lines, ref_lines_use = quark_add_gates_to_lines(
        qc_ref.nqubits,
        qc_ref.ncbits,
        qc_ref.gates,
        qc_ref.params_value,
        width=4,
    )
    assert local_lines == ref_lines
    assert local_lines_use == ref_lines_use


def _build_full_circuit(circuit_cls):
    qc = circuit_cls(3, 3)
    qc.id(0)
    qc.x(1)
    qc.y(2)
    qc.z(0)
    qc.h(1)
    qc.s(0)
    qc.sdg(1)
    qc.t(2)
    qc.tdg(0)
    qc.sx(1)
    qc.sxdg(2)

    qc.rx(0.1, 0)
    qc.ry(-0.2, 1)
    qc.rz(0.3, 2)
    qc.p(0.4, 0)
    qc.u(0.5, 0.6, 0.7, 1)
    qc.r(0.8, 0.9, 2)

    qc.cx(0, 1)
    qc.cy(1, 2)
    qc.cz(0, 2)
    qc.swap(0, 2)
    qc.iswap(1, 2)

    qc.rxx(0.11, 0, 1)
    qc.ryy(-0.12, 1, 2)
    qc.rzz(0.13, 0, 2)
    qc.cp(0.14, 0, 1)

    qc.ccx(0, 1, 2)
    qc.ccz(0, 1, 2)
    qc.cswap(0, 1, 2)

    qc.reset(2)
    qc.delay(1e-6, 0, 1)
    qc.barrier(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _build_io_circuit(circuit_cls):
    qc = circuit_cls(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(1.234, 1)
    qc.rz(-0.25, 2)
    qc.swap(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _build_param_circuit(circuit_cls):
    qc = circuit_cls(2, 2)
    qc.rx("theta", 0)
    qc.rz("phi", 1)
    qc.cp("gamma", 0, 1)
    return qc


def _build_mapping_circuit(circuit_cls):
    qc = circuit_cls(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _build_barrier_circuit(circuit_cls):
    qc = circuit_cls(2, 2)
    qc.rx(0.3, 0)
    qc.barrier(0, 1)
    qc.rx(-0.6, 1)
    qc.barrier(0, 1)
    qc.measure_all()
    return qc
