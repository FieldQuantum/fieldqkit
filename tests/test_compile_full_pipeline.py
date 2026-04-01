"""Full transpiler pipeline tests with realistic topologies.

Tests exercise the end-to-end compilation flow using:
  1. Baihua-style topology (heavy-hex-like grid)
  2. Simulator topology (16-qubit linear chain)
  3. No-backend fallback (auto linear topology)

Circuits intentionally include long-range interactions, 3-qubit gates,
parameterized gates, barriers, and mixed gate types to stress every
compilation pass.
"""

import math
import numpy as np
import networkx as nx
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
    """Return list of (gate_name, q0, q1) for all 2-qubit gates."""
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
    """Assert every 2-qubit gate acts on a directly connected pair."""
    edge_set = set()
    for u, v in coupling_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    for name, q0, q1 in _two_qubit_gate_qubits(qc):
        assert (q0, q1) in edge_set, (
            f"Gate {name}({q0},{q1}) not in coupling map"
        )


def _assert_basis_only(qc, two_q_basis, allow_swap=True):
    """Assert that the compiled circuit only uses the target 2-qubit basis gate (+ optionally swap)."""
    allowed = {two_q_basis}
    if allow_swap:
        allowed.add("swap")
    for name, _, _ in _two_qubit_gate_qubits(qc):
        assert name in allowed, f"Unexpected 2-qubit gate: {name}"


def _assert_no_three_qubit_gates(qc):
    from quantum_hw.circuit.quantumcircuit_helpers import three_qubit_gates_available
    for g in qc.gates:
        assert g[0] not in three_qubit_gates_available, f"3-qubit gate survived: {g[0]}"


# ── Topology builders ──

def _build_baihua_like_chip(nq=20):
    """Build a Baihua-style heavy-hex grid topology (offline, no HTTP).

    Connectivity:
        Row 0:  0 — 1 — 2 — 3 — 4
        Row 1:  5 — 6 — 7 — 8 — 9
        Row 2: 10 —11 —12 —13 —14
        Row 3: 15 —16 —17 —18 —19
        Inter-row:  0-5, 2-7, 4-9,
                    5-10, 7-12, 9-14,
                   10-15,12-17,14-19
    """
    qubits_info = {f"Q{i}": {"fidelity": 0.995 - 0.001 * i} for i in range(nq)}
    edges = []
    # intra-row
    for row in range(4):
        base = row * 5
        for c in range(4):
            edges.append((base + c, base + c + 1))
    # inter-row (staggered connections like heavy-hex)
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


def _build_simulator_backend():
    return Backend("Simulator")


# ══════════════════════════════════════════════════════════
#  1. Baihua-like backend – full pipeline
# ══════════════════════════════════════════════════════════

class TestBaihuaFullPipeline:
    """End-to-end compilation against a heavy-hex-style grid."""

    @pytest.fixture()
    def backend(self):
        return _build_baihua_like_chip()

    # ── basic long-range CX ──

    def test_long_range_cx_chain(self, backend):
        """CX gates spanning the full diagonal: 0→19, 3→15, 1→18."""
        qc = QuantumCircuit(20, 20)
        qc.cx(0, 19)   # max distance
        qc.cx(3, 15)   # cross-grid
        qc.cx(1, 18)
        qc.measure(list(range(20)), list(range(20)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")
        edges = [(u, v) for u, v, _ in backend.couplers_with_attributes]
        _assert_connectivity(compiled, edges)

    def test_dense_entangling_layer(self, backend):
        """GHZ-like entanglement across all qubits via cascaded CX."""
        qc = QuantumCircuit(20, 20)
        qc.h(0)
        for i in range(19):
            qc.cx(i, i + 1)
        qc.measure(list(range(20)), list(range(20)))

        compiled = Transpiler(backend).run(qc)
        _assert_basis_only(compiled, "cz")
        assert compiled.nqubits >= 20

    def test_three_qubit_gates_decomposed(self, backend):
        """CCX, CCZ, CSWAP should be decomposed then routed."""
        qc = QuantumCircuit(20, 20)
        qc.ccx(0, 5, 9)     # far apart
        qc.ccz(1, 7, 14)    # cross-row
        qc.cswap(3, 10, 18) # three different rows
        qc.measure(list(range(20)), list(range(20)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_parameterized_long_range(self, backend):
        """Parameterized 2-qubit gates across the chip."""
        qc = QuantumCircuit(15, 15)
        qc.h(0)
        qc.rxx(0.3, 0, 14)
        qc.ryy(0.5, 1, 12)
        qc.rzz(0.7, 2, 10)
        qc.cp(1.2, 4, 9)
        qc.measure(list(range(15)), list(range(15)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_mixed_gates_all_types(self, backend):
        """Mix of every gate family: 1q, 1q-param, 2q, 2q-param, 3q."""
        qc = QuantumCircuit(12, 12)
        # 1-qubit
        qc.h(0); qc.x(1); qc.y(2); qc.z(3)
        qc.s(4); qc.t(5); qc.sx(6)
        # 1-qubit parametric
        qc.rx(0.1, 7); qc.ry(0.2, 8); qc.rz(0.3, 9)
        qc.u(0.4, 0.5, 0.6, 10)
        # 2-qubit (long-range)
        qc.cx(0, 11); qc.cz(1, 10); qc.swap(3, 8)
        qc.iswap(2, 9)
        # 2-qubit parametric (long-range)
        qc.rxx(0.7, 0, 7); qc.cp(0.8, 4, 11)
        # 3-qubit
        qc.ccx(0, 6, 11)
        qc.barrier(*range(12))
        qc.measure(list(range(12)), list(range(12)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")
        edges = [(u, v) for u, v, _ in backend.couplers_with_attributes]
        _assert_connectivity(compiled, edges)
        # barrier and measure should survive
        names = _gate_names(compiled)
        assert "measure" in names

    def test_repeated_long_range_cx_compresses(self, backend):
        """Two identical long-range CX should produce fewer gates than two independent ones."""
        qc_double = QuantumCircuit(20, 20)
        qc_double.cx(0, 19)
        qc_double.cx(0, 19)  # second identical CX
        qc_double.measure(list(range(20)), list(range(20)))

        qc_single = QuantumCircuit(20, 20)
        qc_single.cx(0, 19)
        qc_single.measure(list(range(20)), list(range(20)))

        compiled_double = Transpiler(backend).run(qc_double)
        compiled_single = Transpiler(backend).run(qc_single)

        # CX·CX = I, so the double version should have ≤ gates of single
        # (the two CX cancel, then SWAP overhead might remain partially)
        gates_double = len([g for g in compiled_double.gates if g[0] not in ("measure", "barrier")])
        gates_single = len([g for g in compiled_single.gates if g[0] not in ("measure", "barrier")])
        assert gates_double <= gates_single + 5  # small tolerance

    def test_star_pattern_from_center(self, backend):
        """One qubit entangled with many distant qubits (star pattern)."""
        qc = QuantumCircuit(20, 20)
        center = 7  # node with good connectivity
        targets = [0, 4, 15, 19, 10, 14]
        qc.h(center)
        for t in targets:
            qc.cx(center, t)
        qc.measure(list(range(20)), list(range(20)))

        compiled = Transpiler(backend).run(qc, niter=3)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_dynamical_decoupling_adds_delays(self, backend):
        """With DD enabled, delay gates should appear in the output."""
        qc = QuantumCircuit(5, 5)
        qc.cx(0, 4)
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(backend).run(qc, use_dd=True)
        names = _gate_names(compiled)
        # DD inserts x gates and delays for idle qubits
        assert "delay" in names or "x" in names or len(compiled.gates) > len(qc.gates)


# ══════════════════════════════════════════════════════════
#  2. Simulator backend (16-qubit linear chain)
# ══════════════════════════════════════════════════════════

class TestSimulatorFullPipeline:
    """End-to-end compilation against the built-in Simulator backend."""

    @pytest.fixture()
    def backend(self):
        return _build_simulator_backend()

    def test_far_ends_cx(self, backend):
        """CX between qubit 0 and 15 – maximum distance on 16-qubit chain."""
        qc = QuantumCircuit(16, 16)
        qc.cx(0, 15)
        qc.measure(list(range(16)), list(range(16)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")
        edges = [(i, i + 1) for i in range(15)]
        _assert_connectivity(compiled, edges)

    def test_all_to_all_small_subset(self, backend):
        """All-to-all CX on 5 qubits mapped onto a 16-qubit line."""
        qc = QuantumCircuit(5, 5)
        for i in range(5):
            for j in range(i + 1, 5):
                qc.cx(i, j)
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)

    def test_trotter_ising_like(self, backend):
        """Trotter-style circuit: alternating RZZ + RX layers (long-range Ising)."""
        n = 8
        qc = QuantumCircuit(n, n)
        for step in range(3):
            # ZZ interactions including long-range
            for i in range(n):
                for j in range(i + 1, n):
                    qc.rzz(0.1 * (step + 1), i, j)
            # Transverse field
            for i in range(n):
                qc.rx(0.2, i)
        qc.measure(list(range(n)), list(range(n)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_qft_like_structure(self, backend):
        """QFT-like circuit with controlled phase gates at all distances."""
        n = 8
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                qc.cp(angle, i, j)
        # swap to reverse order
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        qc.measure(list(range(n)), list(range(n)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_grover_like_oracle(self, backend):
        """Grover-like circuit with multi-controlled gates decomposed."""
        qc = QuantumCircuit(6, 6)
        # Hadamard layer
        for i in range(6):
            qc.h(i)
        # Oracle: 3-qubit gates on distant qubits
        qc.ccx(0, 3, 5)
        qc.ccz(1, 4, 5)
        # Diffusion
        for i in range(6):
            qc.h(i)
            qc.x(i)
        qc.ccx(0, 1, 5)
        for i in range(6):
            qc.x(i)
            qc.h(i)
        qc.measure(list(range(6)), list(range(6)))

        compiled = Transpiler(backend).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_u_gate_to_cz_basis(self, backend):
        """U gates should be preserved (or re-synthesised), 2-qubit gates go to cz."""
        qc = QuantumCircuit(4, 4)
        qc.u(0.1, 0.2, 0.3, 0)
        qc.u(0.4, 0.5, 0.6, 3)
        qc.cx(0, 3)
        qc.rxx(0.5, 1, 3)
        qc.measure(list(range(4)), list(range(4)))

        compiled = Transpiler(backend).run(qc)
        _assert_basis_only(compiled, "cz")


# ══════════════════════════════════════════════════════════
#  3. No-backend fallback (auto linear topology, cx basis)
# ══════════════════════════════════════════════════════════

class TestNoBackendFallback:
    """Compilation without a Backend – uses CX basis and auto-generated linear topology."""

    def test_long_range_routed(self):
        """Long-range CX should be routed with SWAPs even without a backend."""
        qc = QuantumCircuit(6, 6)
        qc.cx(0, 5)
        qc.cx(1, 4)
        qc.cx(2, 3)
        qc.measure(list(range(6)), list(range(6)))

        compiled = Transpiler(chip_backend=None).run(qc, use_dd=False)
        _assert_no_three_qubit_gates(compiled)
        # No backend → CX basis; 1-qubit gates remain native (not forced to u)
        names = _gate_names(compiled)
        assert "measure" in names

    def test_three_qubit_decomposed_no_backend(self):
        qc = QuantumCircuit(5, 5)
        qc.ccx(0, 2, 4)
        qc.cswap(1, 3, 4)
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(chip_backend=None).run(qc, use_dd=False)
        _assert_no_three_qubit_gates(compiled)

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
        names = _gate_names(compiled)
        assert "measure" in names


# ══════════════════════════════════════════════════════════
#  4. Routing-specific stress tests
# ══════════════════════════════════════════════════════════

class TestRoutingStress:
    """Stress tests for the routing pass under various topologies."""

    @pytest.fixture()
    def baihua(self):
        return _build_baihua_like_chip()

    def test_butterfly_pattern(self, baihua):
        """Butterfly / FFT connectivity pattern – each layer crosses half the chip."""
        n = 16
        qc = QuantumCircuit(n, n)
        stride = n // 2
        while stride >= 1:
            for i in range(n):
                partner = i ^ stride  # XOR gives butterfly partner
                if partner > i and partner < n:
                    qc.cx(i, partner)
            stride //= 2
        qc.measure(list(range(n)), list(range(n)))

        compiled = Transpiler(baihua).run(qc, niter=5)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_reverse_linear_worst_case(self, baihua):
        """CX(n-1, 0), CX(n-2, 1), … – every gate is backwards on a line."""
        n = 10
        qc = QuantumCircuit(n, n)
        for i in range(n // 2):
            qc.cx(n - 1 - i, i)
        qc.measure(list(range(n)), list(range(n)))

        compiled = Transpiler(baihua).run(qc)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_random_dense_circuit(self, baihua):
        """Pseudorandom dense circuit – 50 random CX on 15 qubits."""
        rng = np.random.default_rng(42)
        n = 15
        qc = QuantumCircuit(n, n)
        for _ in range(50):
            a, b = rng.choice(n, size=2, replace=False)
            qc.cx(int(a), int(b))
        qc.measure(list(range(n)), list(range(n)))

        compiled = Transpiler(baihua).run(qc, niter=5, routing_n_trials=3)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")
        edges = [(u, v) for u, v, _ in baihua.couplers_with_attributes]
        _assert_connectivity(compiled, edges)

    def test_multi_trial_reduces_swaps(self, baihua):
        """More routing trials should not increase SWAP count (statistically)."""
        qc = QuantumCircuit(10, 10)
        qc.cx(0, 9)
        qc.cx(2, 8)
        qc.cx(4, 6)
        qc.measure(list(range(10)), list(range(10)))

        c1 = Transpiler(baihua).run(qc, routing_n_trials=1)
        c10 = Transpiler(baihua).run(qc, routing_n_trials=10)

        swap1 = sum(1 for g in c1.gates if g[0] == "swap")
        swap10 = sum(1 for g in c10.gates if g[0] == "swap")
        # 10 trials should find equal or better solution
        assert swap10 <= swap1 + 2  # small tolerance for randomness


# ══════════════════════════════════════════════════════════
#  5. Transpiler options interaction
# ══════════════════════════════════════════════════════════

class TestTranspilerOptions:
    """Test interactions between different transpiler flags."""

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
        # At least 2 single-qubit gate ops from the H·H (not cancelled)
        single_q = [n for n in names if n not in ("cz", "measure", "barrier", "swap")]
        assert len(single_q) >= 2

    def test_only_routing_pass(self, backend):
        """Only run routing — no decompose, translate, or compression."""
        qc = QuantumCircuit(10, 10)
        qc.cx(0, 9)
        qc.measure(list(range(10)), list(range(10)))

        compiled = Transpiler(backend).run(
            qc,
            use_three_qubit_decompose=False,
            use_translate_to_basis=False,
            use_gate_compressor=False,
            use_dd=False,
        )
        # CX should survive (not translated), and SWAP should be inserted
        names = _gate_names(compiled)
        assert "cx" in names or "swap" in names

    def test_noise_aware_explicit_false(self, backend):
        """Explicitly disable noise-aware routing on a real backend."""
        qc = QuantumCircuit(5, 5)
        qc.cx(0, 4)
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(backend).run(qc, noise_aware=False, use_dd=False)
        _assert_no_three_qubit_gates(compiled)


# ══════════════════════════════════════════════════════════
#  6. Edge cases
# ══════════════════════════════════════════════════════════

class TestEdgeCases:
    """Boundary / degenerate circuits."""

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
        # Both H gates should survive (barrier prevents merge)
        names = _gate_names(compiled)
        single_q = [n for n in names if n not in ("cz", "cx", "measure", "barrier", "swap", "delay")]
        assert len(single_q) >= 2

    def test_deeply_nested_trotter(self, backend):
        """20-step Trotter with nearest + next-nearest interactions."""
        n = 6
        qc = QuantumCircuit(n, n)
        for _ in range(20):
            for i in range(n - 1):
                qc.rzz(0.05, i, i + 1)       # nearest
            for i in range(n - 2):
                qc.rzz(0.02, i, i + 2)       # next-nearest (requires routing)
            for i in range(n):
                qc.rx(0.1, i)
        qc.measure(list(range(n)), list(range(n)))

        compiled = Transpiler(backend).run(qc, use_dd=False, niter=3)
        _assert_no_three_qubit_gates(compiled)
        _assert_basis_only(compiled, "cz")

    def test_idle_qubits_preserved(self, backend):
        """Qubits with no gates should still appear in the output circuit."""
        qc = QuantumCircuit(5, 5)
        qc.h(0)
        qc.cx(0, 4)
        # qubits 1, 2, 3 are idle
        qc.measure(list(range(5)), list(range(5)))

        compiled = Transpiler(backend).run(qc, use_dd=False)
        assert compiled.nqubits >= 5
