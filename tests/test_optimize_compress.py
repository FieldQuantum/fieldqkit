"""Tests for GateCompressor – focusing on the compress_adjacent_* functions.

The two functions had a copy-paste bug where ``len(node2_successors)``
was incorrectly written as ``len(node1_predecessors)``.  These tests
verify the fix by exercising all predecessor / successor combinations:

  (a) both present   – edge should be reconnected
  (b) no predecessor – gate at circuit start
  (c) no successor   – gate at circuit end
  (d) neither        – two-gate circuit that collapses entirely
"""

import math
import numpy as np
import pytest

from quantum_hw.circuit import QuantumCircuit
from quantum_hw.compile.optimize import GateCompressor


# ────────────────────────────────────────────────────────
#  helpers
# ────────────────────────────────────────────────────────

def _gate_names(qc):
    return [g[0] for g in qc.gates]


def _unitary_for_circuit(qc):
    """Brute-force unitary from gate matrices (for small circuits)."""
    from quantum_hw.circuit.matrix import gate_matrix_dict
    from quantum_hw.circuit.quantumcircuit_helpers import (
        one_qubit_gates_available,
        one_qubit_parameter_gates_available,
        two_qubit_gates_available,
        two_qubit_parameter_gates_available,
    )

    n = qc.nqubits
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)

    for gate_info in qc.gates:
        name = gate_info[0]
        if name in ("measure", "barrier", "reset", "delay"):
            continue

        if name in one_qubit_gates_available:
            mat = gate_matrix_dict[name]
            qubits = [gate_info[-1]]
        elif name in one_qubit_parameter_gates_available:
            params = gate_info[1:-1]
            mat = gate_matrix_dict[name](*params)
            qubits = [gate_info[-1]]
        elif name in two_qubit_gates_available:
            mat = gate_matrix_dict[name]
            qubits = [gate_info[-2], gate_info[-1]]
        elif name in two_qubit_parameter_gates_available:
            params = gate_info[1:-2]
            mat = gate_matrix_dict[name](*params)
            qubits = [gate_info[-2], gate_info[-1]]
        else:
            raise ValueError(f"Unknown gate: {name}")

        if callable(mat) and not isinstance(mat, np.ndarray):
            mat = mat()
        mat = np.array(mat, dtype=complex)
        full = GateCompressor._expand_matrix(mat, qubits, n)
        U = full @ U

    return U


# ────────────────────────────────────────────────────────
#  1. Single-qubit self-inverse cancellation (H·H = I, X·X = I …)
# ────────────────────────────────────────────────────────

class TestCompressAdjacentSingleQubitGates:
    """Tests for compress_adjacent_single_qubit_gates (self-inverse pairs)."""

    def test_hh_cancel_middle(self):
        """H-H in the middle of a circuit should cancel, preserving surrounding gates."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.h(0)
        qc.h(0)
        qc.z(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert "h" not in names
        # x then z should remain (or a merged u gate)
        assert len(compressed.gates) <= 2  # merged into u or kept as x, z

    def test_hh_cancel_at_start(self):
        """H-H at the very start – no predecessor."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.h(0)
        qc.x(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert names.count("h") == 0
        # x should survive
        assert any(n in ("x", "u") for n in names)

    def test_hh_cancel_at_end(self):
        """H-H at the very end – no successor."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.h(0)
        qc.h(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert names.count("h") == 0

    def test_xx_only_circuit(self):
        """X-X is the entire circuit – collapses to empty (no predecessors or successors)."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.x(0)
        compressed = GateCompressor().run(qc)
        # Should have no gates (or only measure-like)
        names = _gate_names(compressed)
        non_functional = [n for n in names if n not in ("measure", "barrier")]
        assert len(non_functional) == 0

    def test_multiple_pairs(self):
        """Multiple self-inverse pairs all cancel."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.h(0)
        qc.x(0)
        qc.x(0)
        qc.y(0)
        qc.y(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        non_functional = [n for n in names if n not in ("measure", "barrier")]
        assert len(non_functional) == 0


# ────────────────────────────────────────────────────────
#  2. Parametric single-qubit merge (rz(a)·rz(b) = rz(a+b))
# ────────────────────────────────────────────────────────

class TestCompressAdjacentParametricGates:
    """Tests for compress_adjacent_single_parameter_qubit_gates."""

    def test_rz_merge_middle(self):
        """Two rz in the middle merge into one."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.rz(0.3, 0)
        qc.rz(0.5, 0)
        qc.h(0)
        compressed = GateCompressor().run(qc)
        # Should have at most 3 gates: h, rz/u, h  (or merged further)
        assert len(compressed.gates) <= 3

    def test_rz_merge_cancels_to_identity(self):
        """rz(π) · rz(-π) = rz(0) ≈ I, should be eliminated."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.rz(math.pi, 0)
        qc.rz(-math.pi, 0)
        qc.x(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        non_functional = [n for n in names if n not in ("measure", "barrier")]
        # x·x also cancels → empty
        assert len(non_functional) == 0

    def test_rz_merge_at_start(self):
        """Parametric merge at circuit start – no predecessor."""
        qc = QuantumCircuit(1, 1)
        qc.rz(0.1, 0)
        qc.rz(0.2, 0)
        qc.h(0)
        compressed = GateCompressor().run(qc)
        # The two rz should merge; the circuit should be ≤ 2 gates
        assert len(compressed.gates) <= 2

    def test_rz_merge_at_end(self):
        """Parametric merge at circuit end – no successor."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.rz(0.1, 0)
        qc.rz(0.2, 0)
        compressed = GateCompressor().run(qc)
        assert len(compressed.gates) <= 2

    def test_rz_merge_full_circuit(self):
        """Two rz gates as the entire circuit – neither predecessor nor successor."""
        qc = QuantumCircuit(1, 1)
        qc.rz(0.3, 0)
        qc.rz(0.7, 0)
        compressed = GateCompressor().run(qc)
        assert len(compressed.gates) <= 1

    def test_ry_merge(self):
        """ry gates should also merge correctly."""
        qc = QuantumCircuit(1, 1)
        qc.ry(0.4, 0)
        qc.ry(0.6, 0)
        compressed = GateCompressor().run(qc)
        assert len(compressed.gates) <= 1

    def test_rx_cancellation_2pi(self):
        """rx(π) · rx(π) = rx(2π) ≈ -I ≈ global phase → identity up to global phase."""
        qc = QuantumCircuit(1, 1)
        qc.rx(math.pi, 0)
        qc.rx(math.pi, 0)
        compressed = GateCompressor().run(qc)
        # rx(2π) = -I which is identity up to global phase; may or may not be removed
        # but at minimum should have ≤ 1 gate
        assert len(compressed.gates) <= 1


# ────────────────────────────────────────────────────────
#  3. Two-qubit self-inverse cancellation (CX·CX = I, CZ·CZ = I)
# ────────────────────────────────────────────────────────

class TestCompressTwoQubitGates:
    """Tests for compress_adjacent_two_qubit_gates."""

    def test_cx_pair_cancels(self):
        """CX·CX on the same qubits should cancel."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(1)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert "cx" not in names and "cnot" not in names

    def test_cz_pair_cancels(self):
        """CZ·CZ on the same qubits should cancel."""
        qc = QuantumCircuit(2, 2)
        qc.cz(0, 1)
        qc.cz(0, 1)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert "cz" not in names

    def test_swap_pair_cancels(self):
        """SWAP·SWAP on the same qubits should cancel."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.swap(0, 1)
        qc.swap(0, 1)
        qc.h(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert "swap" not in names


# ────────────────────────────────────────────────────────
#  4. Unitary equivalence: compression must not change the circuit semantics
# ────────────────────────────────────────────────────────

class TestCompressionPreservesUnitary:
    """Verify that the compressed circuit has the same unitary as the original."""

    @pytest.mark.parametrize("gates_fn", [
        # Case 1: H-H cancel
        lambda qc: (qc.h(0), qc.h(0), qc.x(0)),
        # Case 2: rz merge
        lambda qc: (qc.rz(0.3, 0), qc.rz(0.7, 0)),
        # Case 3: mixed single-qubit
        lambda qc: (qc.h(0), qc.rz(0.5, 0), qc.rz(0.3, 0), qc.h(0)),
        # Case 4: ry-ry merge
        lambda qc: (qc.ry(1.2, 0), qc.ry(0.4, 0)),
    ], ids=["hh_cancel", "rz_merge", "mixed_1q", "ry_merge"])
    def test_single_qubit_unitary_preserved(self, gates_fn):
        qc = QuantumCircuit(1, 1)
        gates_fn(qc)
        compressed = GateCompressor().run(qc)
        u_orig = _unitary_for_circuit(qc)
        u_comp = _unitary_for_circuit(compressed)
        # Equal up to global phase
        if not np.allclose(u_orig, u_comp):
            # Find global phase from largest-magnitude element
            idx = np.argmax(np.abs(u_orig))
            if np.abs(u_orig.flat[idx]) < 1e-10:
                # Both should be near-zero matrices
                assert np.allclose(u_orig, u_comp, atol=1e-8)
            else:
                phase = u_comp.flat[idx] / u_orig.flat[idx]
                assert np.allclose(u_orig * phase, u_comp, atol=1e-8)

    def test_two_qubit_cx_cancel_unitary(self):
        """CX·CX cancellation preserves the 2-qubit unitary."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        compressed = GateCompressor().run(qc)
        u_orig = _unitary_for_circuit(qc)
        u_comp = _unitary_for_circuit(compressed)
        if not np.allclose(u_orig, u_comp):
            idx = np.argmax(np.abs(u_orig))
            phase = u_comp.flat[idx] / u_orig.flat[idx]
            assert np.allclose(u_orig * phase, u_comp, atol=1e-8)


# ────────────────────────────────────────────────────────
#  5. Multi-qubit circuit: ensure compression on one qubit
#     doesn't corrupt another qubit's gates
# ────────────────────────────────────────────────────────

class TestMultiQubitCompression:
    """Compression on disjoint qubits should be independent."""

    def test_independent_qubit_compression(self):
        """Gates on different qubits are compressed independently."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)       # should cancel
        qc.x(1)
        qc.x(1)       # should cancel
        qc.rz(0.3, 0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert "h" not in names
        assert "x" not in names

    def test_interleaved_gates_no_false_cancel(self):
        """H on qubit 0 and H on qubit 1 must NOT cancel each other."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        compressed = GateCompressor().run(qc)
        # Both should survive (on different qubits)
        assert len(compressed.gates) >= 2

    def test_three_qubit_circuit_compression(self):
        """Compression in a 3-qubit circuit with mixed operations."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)   # cancel with above
        qc.rz(0.5, 2)
        qc.rz(-0.5, 2)  # cancel (sum=0 → identity)
        qc.h(0)        # combined with first h → may merge
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        assert "cx" not in names
        # rz(0) should be eliminated
        # h·h should cancel or merge
        assert len(compressed.gates) <= 2


# ────────────────────────────────────────────────────────
#  6. Edge case: measure gates block compression
# ────────────────────────────────────────────────────────

class TestMeasureBarrier:
    """Measure and barrier should prevent gate cancellation across them."""

    def test_measure_blocks_cancellation(self):
        """H·measure·H must NOT cancel the two H gates."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.h(0)
        compressed = GateCompressor().run(qc)
        names = _gate_names(compressed)
        h_count = sum(1 for n in names if n in ("h", "u"))
        assert h_count >= 2  # both H (or u equivalents) must survive
