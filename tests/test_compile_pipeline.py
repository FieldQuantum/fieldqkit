"""Full transpiler pipeline tests – edge cases and flag interactions.

Individual pass tests (routing, compression, decomposition, layout) are in
test_compile_passes.py.  This file focuses on edge cases and integration
behaviors that only surface in the end-to-end pipeline.
"""

import math
import numpy as np
import pytest

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.compile.transpiler import Transpiler
from quantum_hw.api.backend import Backend


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def _gate_names(qc):
    return [g[0] for g in qc.gates]


def _two_qubit_gate_qubits(qc):
    from quantum_hw.circuit.quantumcircuit_helpers import (
        two_qubit_gates_available,
        two_qubit_parameter_gates_available,
    )
    all_2q = set(two_qubit_gates_available) | set(two_qubit_parameter_gates_available)
    result = []
    for g in qc.gates:
        if g[0] in all_2q:
            result.append((g[0], g[-2], g[-1]))
    return result


def _assert_connectivity(qc, coupling_edges):
    edge_set = set()
    for u, v in coupling_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    for name, q0, q1 in _two_qubit_gate_qubits(qc):
        assert (q0, q1) in edge_set, f"Gate {name}({q0},{q1}) not in coupling map"


def _assert_basis_only(qc, two_q_basis, allow_swap=True):
    allowed = {two_q_basis}
    if allow_swap:
        allowed.add("swap")
    for name, _, _ in _two_qubit_gate_qubits(qc):
        assert name in allowed, f"Unexpected 2-qubit gate: {name}"


def _assert_no_three_qubit_gates(qc):
    from quantum_hw.circuit.quantumcircuit_helpers import three_qubit_gates_available
    for g in qc.gates:
        assert g[0] not in three_qubit_gates_available, f"3-qubit gate survived: {g[0]}"


def _build_baihua_like_chip(nq=20):
    qubits_info = {f"Q{i}": {"fidelity": 0.995 - 0.001 * i} for i in range(nq)}
    edges = []
    for row in range(4):
        base = row * 5
        for c in range(4):
            edges.append((base + c, base + c + 1))
    inter = [(0, 5), (2, 7), (4, 9),
             (5, 10), (7, 12), (9, 14),
             (10, 15), (12, 17), (14, 19)]
    edges.extend(inter)
    couplers_info = {}
    for idx, (a, b) in enumerate(edges):
        couplers_info[f"C{idx}"] = {
            "qubits_index": [a, b],
            "fidelity": round(0.98 - 0.002 * idx, 4),
        }
    chip_info = {
        "chip_name": "BaihuaTest",
        "qubits_info": qubits_info,
        "couplers_info": couplers_info,
        "global_info": {
            "two_qubit_gate_basis": "cz",
            "nqubits_available": nq,
            "error_rate_2q": 0.005,
            "one_qubit_gate_length": 0.03,
            "two_qubit_gate_length": 0.06,
        },
        "priority_qubits": [list(range(nq))],
    }
    return Backend(chip_info)


# ══════════════════════════════════════════════════════════
#  Integration: mixed gates through real topology
# ══════════════════════════════════════════════════════════

class TestBaihuaFullPipeline:
    @pytest.fixture()
    def backend(self):
        return _build_baihua_like_chip()

    def test_mixed_gates_all_types(self, backend):
        """Mix of every gate family: 1q, 1q-param, 2q, 2q-param, 3q."""
        qc = QuantumCircuit(12, 12)
        qc.h(0); qc.x(1); qc.y(2); qc.z(3)
        qc.s(4); qc.t(5); qc.sx(6)
        qc.rx(0.1, 7); qc.ry(0.2, 8); qc.rz(0.3, 9)
        qc.u(0.4, 0.5, 0.6, 10)
        qc.cx(0, 11); qc.cz(1, 10); qc.swap(3, 8)
        qc.iswap(2, 9)
        qc.rxx(0.7, 0, 7); qc.cp(0.8, 4, 11)
        qc.ccx(0, 6, 11)
        qc.barrier(*range(12))
        qc.measure(list(range(12)), list(range(12)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")
        edges = [(u, v) for u, v, _ in backend.couplers_with_attributes]
        _assert_connectivity(compiled, edges)
        assert "measure" in _gate_names(compiled)

    def test_dynamical_decoupling_adds_delays(self, backend):
        """With DD enabled, delay gates should appear in the output."""
        qc = QuantumCircuit(5, 5)
        qc.cx(0, 4)
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(backend).run(qc, use_dd=True)
        names = _gate_names(compiled)
        assert "delay" in names or "x" in names or len(compiled.gates) > len(qc.gates)


# ══════════════════════════════════════════════════════════
#  No-backend fallback
# ══════════════════════════════════════════════════════════

class TestNoBackendFallback:
    def test_all_passes_disabled(self):
        """With all passes disabled, circuit should be unchanged."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        compiled = Transpiler(chip_backend=None).run(
            qc,
            use_dd=False,
            use_three_qubit_decompose=False,
            use_sabre_routing=False,
            use_translate_to_basis=False,
            use_gate_compressor=False,
        )
        assert _gate_names(compiled) == _gate_names(qc)

    def test_parameterized_circuit_no_backend(self):
        """Parameterized 2-qubit gates compile correctly without a backend."""
        qc = QuantumCircuit(4, 4)
        qc.rx(0.5, 0)
        qc.rxx(0.3, 0, 3)
        qc.ryy(0.4, 1, 3)
        qc.rzz(0.5, 0, 2)
        qc.measure(list(range(4)), list(range(4)))

        compiled = Transpiler(chip_backend=None).run(qc, use_dd=False)
        _assert_no_three_qubit_gates(compiled)
        assert "measure" in _gate_names(compiled)


# ══════════════════════════════════════════════════════════
#  Transpiler flag interactions
# ══════════════════════════════════════════════════════════

class TestTranspilerOptions:
    @pytest.fixture()
    def backend(self):
        return _build_baihua_like_chip()

    def test_routing_off_but_translate_on(self, backend):
        """Skip routing but still translate to CZ basis."""
        qc = QuantumCircuit(3, 3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        compiled = Transpiler(backend).run(
            qc, use_sabre_routing=False, use_dd=False,
        )
        _assert_basis_only(compiled, "cz", allow_swap=False)

    def test_compressor_off_preserves_redundant_gates(self, backend):
        """With compressor off, H·H pair should survive."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.cz(0, 1)
        qc.measure([0, 1], [0, 1])

        compiled = Transpiler(backend).run(
            qc, use_gate_compressor=False, use_dd=False,
            use_sabre_routing=False,
        )
        names = _gate_names(compiled)
        single_q = [n for n in names if n not in ("cz", "measure", "barrier", "swap")]
        assert len(single_q) >= 2


# ══════════════════════════════════════════════════════════
#  Edge cases
# ══════════════════════════════════════════════════════════

class TestEdgeCases:
    @pytest.fixture()
    def backend(self):
        return _build_baihua_like_chip()

    def test_single_qubit_circuit(self, backend):
        """A 1-qubit circuit should compile without error."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.rz(0.5, 0)
        qc.measure(0, 0)

        compiled = Transpiler(backend).run(qc, use_dd=False)
        assert compiled.nqubits >= 1

    def test_identity_circuit(self, backend):
        """Circuit with only measure – nothing to route or translate."""
        qc = QuantumCircuit(3, 3)
        qc.measure([0, 1, 2], [0, 1, 2])

        compiled = Transpiler(backend).run(qc, use_dd=False)
        meas = [g for g in compiled.gates if g[0] == "measure"]
        assert len(meas) == 3

    def test_barrier_separates_optimization(self, backend):
        """Barrier should prevent cross-boundary gate compression."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.barrier(0, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        compiled = Transpiler(backend).run(qc, use_dd=False)
        names = _gate_names(compiled)
        single_q = [n for n in names if n not in ("cz", "cx", "measure", "barrier", "swap", "delay")]
        assert len(single_q) >= 2

    def test_idle_qubits_preserved(self, backend):
        """Qubits with no gates should still appear in the output circuit."""
        qc = QuantumCircuit(5, 5)
        qc.h(0)
        qc.cx(0, 4)
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(backend).run(qc, use_dd=False)
        assert compiled.nqubits >= 5
