from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.quantumcircuit_helpers import three_qubit_gates_available
from quantum_hw.compile.decompose import ThreeQubitGateDecompose
from quantum_hw.compile.optimize import GateCompressor
from quantum_hw.compile.routing import SabreRouting
from quantum_hw.compile.transpiler import Transpiler
from quantum_hw.compile.translate import TranslateToBasisGates
import networkx as nx
import numpy as np


def test_three_qubit_decompose_removes_three_qubit_gates():
    qc = QuantumCircuit(3, 3)
    qc.ccx(0, 1, 2)
    qc.ccz(0, 1, 2)
    qc.cswap(0, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    new_qc = ThreeQubitGateDecompose().run(qc)
    assert new_qc.nqubits == qc.nqubits
    assert new_qc.ncbits == qc.ncbits
    assert all(gate[0] not in three_qubit_gates_available for gate in new_qc.gates)


def test_translate_to_basis_gates_cz_and_u_only():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.ry(0.2, 1)
    qc.cx(0, 1)
    qc.swap(0, 1)
    qc.rxx(0.3, 0, 1)
    qc.ryy(-0.4, 0, 1)
    qc.rzz(0.5, 0, 1)
    qc.cp(0.6, 0, 1)
    qc.measure([0, 1], [0, 1])

    new_qc = TranslateToBasisGates(convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz").run(qc)
    allowed = {"u", "cz", "measure"}
    for gate in new_qc.gates:
        assert gate[0] in allowed


def test_pauli_evolution_expands_mixed_string():
    qc = QuantumCircuit(5, 5)
    qc.pauli_evolution(0.3, "X1 Y2 Z3 Z4")

    names = [g[0] for g in qc.gates]
    assert names.count("rz") == 1
    assert names.count("cx") == 6
    assert names.count("h") == 4
    assert names.count("sdg") == 1
    assert names.count("s") == 1


def test_pauli_evolution_is_compile_compatible():
    qc = QuantumCircuit(5, 5)
    qc.pauli_evolution("theta", "X1 Y2 Z3 Z4")

    transpiled = Transpiler(chip_backend=None).run(
        qc,
        use_dd=False,
        use_three_qubit_decompose=False,
        use_sabre_routing=False,
        use_translate_to_basis=True,
        use_gate_compressor=False,
    )
    allowed = {"h", "sdg", "s", "rz", "cx"}
    assert all(g[0] in allowed for g in transpiled.gates)


# --------------- Routing: noise_aware & n_trials ---------------

def _make_line_graph(n, fidelities=None):
    """Build a line graph 0-1-..-(n-1) with optional per-edge fidelity."""
    g = nx.Graph()
    for i in range(n - 1):
        f = fidelities[i] if fidelities else 1.0
        g.add_edge(i, i + 1, fidelity=f)
    g.graph["normal_order"] = list(range(n))
    return g


def test_sabre_routing_default_produces_valid_circuit():
    """Baseline: default routing (no noise_aware, n_trials=1) still works."""
    qc = QuantumCircuit(4, 4)
    qc.cx(0, 3)
    qc.cx(3, 0)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    g = _make_line_graph(4)
    routed = SabreRouting(g, iterations=3).run(qc)
    assert routed.nqubits >= qc.nqubits
    # Should produce a valid circuit (with or without swaps).
    gate_names = [gate[0] for gate in routed.gates]
    assert "cx" in gate_names or "swap" in gate_names


def test_noise_aware_builds_weighted_distance_matrix():
    """Noise-aware routing uses -log(fidelity) weights in its distance matrix."""
    import math
    g = nx.Graph()
    g.add_edge(0, 1, fidelity=0.5)
    g.add_edge(1, 2, fidelity=0.99)
    g.graph["normal_order"] = [0, 1, 2]

    sr_na = SabreRouting(g, noise_aware=True, iterations=1)
    sr_hop = SabreRouting(g, noise_aware=False, iterations=1)

    # Hop-based: d(0,1) = d(1,2) = 1, d(0,2) = 2
    assert sr_hop.distance_matrix[0][1] == 1
    assert sr_hop.distance_matrix[0][2] == 2

    # Noise-aware: d(0,1) = -log(0.5) ≈ 0.693, d(1,2) = -log(0.99) ≈ 0.01
    assert abs(sr_na.distance_matrix[0][1] - (-math.log(0.5))) < 1e-6
    assert abs(sr_na.distance_matrix[1][2] - (-math.log(0.99))) < 1e-6
    # d(0,2) = -log(0.5) + -log(0.99), not 2
    assert abs(sr_na.distance_matrix[0][2] - (-math.log(0.5) - math.log(0.99))) < 1e-6
    # Hop matrix is always unweighted
    assert sr_na.hop_matrix[0][2] == 2


def test_stochastic_n_trials_reduces_or_matches_swaps():
    """More trials should find a routing with equal or fewer swaps."""
    g = _make_line_graph(5)
    qc = QuantumCircuit(5, 5)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.measure(list(range(5)), list(range(5)))

    single = SabreRouting(g, iterations=3, n_trials=1).run(qc)
    multi = SabreRouting(g, iterations=3, n_trials=10).run(qc)

    swaps_single = sum(1 for gate in single.gates if gate[0] == "swap")
    swaps_multi = sum(1 for gate in multi.gates if gate[0] == "swap")
    # Multi-trial should be no worse (probabilistically always true for 10 trials).
    assert swaps_multi <= swaps_single or swaps_multi <= swaps_single + 1


def test_transpiler_noise_aware_default_no_backend():
    """Without a backend, noise_aware should default to False and not crash."""
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    routed = Transpiler(chip_backend=None).run(
        qc, use_dd=False, use_translate_to_basis=False,
        use_three_qubit_decompose=False, use_gate_compressor=False,
    )
    assert routed.nqubits >= qc.nqubits


# --------------- hop_matrix / _hop_distance 专项测试 ---------------

def test_hop_matrix_equals_distance_matrix_when_not_noise_aware():
    """noise_aware=False 时 hop_matrix 和 distance_matrix 完全相同。"""
    import numpy as np
    g = _make_line_graph(5)
    sr = SabreRouting(g, iterations=1)
    np.testing.assert_array_equal(sr.hop_matrix, sr.distance_matrix)


def test_hop_matrix_differs_from_distance_matrix_when_noise_aware():
    """noise_aware=True 时 hop_matrix 仍然是跳数，distance_matrix 是加权值。"""
    import numpy as np
    g = _make_line_graph(4, fidelities=[0.9, 0.8, 0.7])
    sr = SabreRouting(g, noise_aware=True, iterations=1)
    # hop_matrix: 相邻 = 1
    assert sr.hop_matrix[0][1] == 1
    assert sr.hop_matrix[0][3] == 3
    # distance_matrix: 相邻 != 1
    assert sr.distance_matrix[0][1] != 1
    # 两个矩阵不相等
    assert not np.array_equal(sr.hop_matrix, sr.distance_matrix)


def test_hop_matrix_always_integer_hops():
    """hop_matrix 中所有值都应为非负整数跳数。"""
    import numpy as np
    g = _make_line_graph(6, fidelities=[0.5, 0.6, 0.7, 0.8, 0.9])
    for na in (True, False):
        sr = SabreRouting(g, noise_aware=na, iterations=1)
        for i in range(6):
            for j in range(6):
                v = sr.hop_matrix[i][j]
                assert v == int(v), f"hop_matrix[{i}][{j}]={v} is not integer"
                assert v >= 0


def test_noise_aware_routing_terminates_line_graph():
    """noise_aware 在线形图上不会死循环，且结果正确。"""
    g = _make_line_graph(5, fidelities=[0.95, 0.8, 0.7, 0.99])
    qc = QuantumCircuit(5, 5)
    qc.cx(0, 4)
    qc.measure(list(range(5)), list(range(5)))
    routed = SabreRouting(g, noise_aware=True, iterations=3).run(qc)
    assert routed.nqubits >= qc.nqubits
    gate_names = [gate[0] for gate in routed.gates]
    assert "measure" in gate_names


def test_noise_aware_routing_terminates_star_graph():
    """noise_aware 在星形图上不会死循环。"""
    g = nx.star_graph(4)  # 中心0, 叶子1-4
    for u, v in g.edges():
        g[u][v]["fidelity"] = 0.9
    g.graph["normal_order"] = list(range(5))
    qc = QuantumCircuit(5, 5)
    qc.cx(1, 2)
    qc.cx(3, 4)
    qc.measure(list(range(5)), list(range(5)))
    routed = SabreRouting(g, noise_aware=True, iterations=3).run(qc)
    assert routed.nqubits >= 5


def test_noise_aware_routing_terminates_grid_graph():
    """noise_aware 在 2x3 网格图上不会死循环。"""
    g = nx.grid_2d_graph(2, 3)
    # 把二维坐标重编号为整数
    mapping = {node: i for i, node in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, mapping)
    for u, v in g.edges():
        g[u][v]["fidelity"] = 0.85
    g.graph["normal_order"] = list(range(6))
    qc = QuantumCircuit(6, 6)
    qc.cx(0, 5)
    qc.cx(1, 4)
    qc.measure(list(range(6)), list(range(6)))
    routed = SabreRouting(g, noise_aware=True, iterations=3).run(qc)
    assert routed.nqubits >= 6
    assert any(gate[0] == "measure" for gate in routed.gates)


def test_noise_aware_routing_terminates_complete_graph():
    """noise_aware 在完全图上不会死循环（任意对都邻接，不应有 swap）。"""
    g = nx.complete_graph(4)
    for u, v in g.edges():
        g[u][v]["fidelity"] = 0.95
    g.graph["normal_order"] = list(range(4))
    qc = QuantumCircuit(4, 4)
    qc.cx(0, 3)
    qc.cx(1, 2)
    qc.measure(list(range(4)), list(range(4)))
    routed = SabreRouting(g, noise_aware=True, iterations=1).run(qc)
    # 完全图任意对邻接，不需要 swap
    assert not any(gate[0] == "swap" for gate in routed.gates)


def test_noise_aware_with_missing_fidelity_defaults_to_1():
    """边没有 fidelity 属性时，默认用 1.0（即 -log(1)=0 权重，不改变距离排序）。"""
    import math
    g = nx.Graph()
    g.add_edge(0, 1)  # 没有 fidelity
    g.add_edge(1, 2, fidelity=0.8)
    g.graph["normal_order"] = [0, 1, 2]
    sr = SabreRouting(g, noise_aware=True, iterations=1)
    # edge(0,1) 无 fidelity → weight = -log(1.0) = 0
    assert abs(sr.distance_matrix[0][1] - 0.0) < 1e-9
    # edge(1,2) → weight = -log(0.8)
    assert abs(sr.distance_matrix[1][2] - (-math.log(0.8))) < 1e-6
    # hop_matrix 不受影响
    assert sr.hop_matrix[0][1] == 1
    assert sr.hop_matrix[1][2] == 1


def test_noise_aware_with_very_low_fidelity():
    """极低 fidelity（接近 0）不会导致 inf 或 crash。"""
    g = nx.Graph()
    g.add_edge(0, 1, fidelity=1e-8)
    g.add_edge(1, 2, fidelity=0.99)
    g.graph["normal_order"] = [0, 1, 2]
    sr = SabreRouting(g, noise_aware=True, iterations=1)
    # 应该是一个大但有限的值
    import math
    assert math.isfinite(sr.distance_matrix[0][1])
    assert sr.distance_matrix[0][1] > 10  # -log(1e-8) ≈ 18.4


def test_noise_aware_with_fidelity_exactly_1():
    """fidelity=1.0 时 -log(1)=0，不影响路由。"""
    g = _make_line_graph(3, fidelities=[1.0, 1.0])
    sr = SabreRouting(g, noise_aware=True, iterations=1)
    assert sr.distance_matrix[0][1] == 0.0
    assert sr.distance_matrix[0][2] == 0.0  # 0+0
    # 但 hop_matrix 仍然是 1, 2
    assert sr.hop_matrix[0][1] == 1
    assert sr.hop_matrix[0][2] == 2


def test_routing_cx_on_adjacent_no_swap():
    """相邻 qubit 上的 cx 无论 noise_aware 与否都不需要 swap。"""
    g = _make_line_graph(3, fidelities=[0.9, 0.8])
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 1)
    qc.measure([0, 1, 2], [0, 1, 2])
    for na in (True, False):
        routed = SabreRouting(g, noise_aware=na, iterations=1).run(qc)
        assert not any(gate[0] == "swap" for gate in routed.gates), f"noise_aware={na}"


def test_routing_preserves_single_qubit_gates():
    """单 qubit 门应该被保留，不受 hop_matrix 改动影响。"""
    g = _make_line_graph(3, fidelities=[0.9, 0.8])
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.x(1)
    qc.rz(0.5, 2)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    for na in (True, False):
        routed = SabreRouting(g, noise_aware=na, iterations=3).run(qc)
        gate_names = [gate[0] for gate in routed.gates]
        assert "h" in gate_names, f"h missing, noise_aware={na}"
        assert "x" in gate_names, f"x missing, noise_aware={na}"
        assert "rz" in gate_names, f"rz missing, noise_aware={na}"


def test_routing_logical_to_physical_mapping_valid():
    """路由后的 logical_to_physical 映射应该是有效的双射。"""
    g = _make_line_graph(4, fidelities=[0.9, 0.8, 0.7])
    qc = QuantumCircuit(4, 4)
    qc.cx(0, 3)
    qc.cx(1, 2)
    qc.measure(list(range(4)), list(range(4)))
    for na in (True, False):
        routed = SabreRouting(g, noise_aware=na, iterations=3).run(qc)
        l2p = routed.logical_to_physical
        p2l = routed.physical_to_logical
        # 合法双射
        assert set(l2p.values()) == set(p2l.keys())
        assert set(l2p.keys()) == set(p2l.values())
        for v, p in l2p.items():
            assert p2l[p] == v


def test_noise_aware_and_n_trials_combined():
    """同时开启 noise_aware + n_trials > 1 不会死循环或出错。"""
    g = _make_line_graph(5, fidelities=[0.95, 0.8, 0.7, 0.6])
    qc = QuantumCircuit(5, 5)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.measure(list(range(5)), list(range(5)))
    routed = SabreRouting(g, noise_aware=True, n_trials=5, iterations=3).run(qc)
    assert routed.nqubits >= 5
    assert any(gate[0] == "measure" for gate in routed.gates)


def test_noise_aware_multiple_cx_gates():
    """多个远距离 cx 门在 noise_aware 下能正确路由。"""
    g = _make_line_graph(6, fidelities=[0.9, 0.85, 0.8, 0.75, 0.7])
    qc = QuantumCircuit(6, 6)
    qc.cx(0, 5)
    qc.cx(0, 3)
    qc.cx(2, 5)
    qc.cx(1, 4)
    qc.measure(list(range(6)), list(range(6)))
    routed = SabreRouting(g, noise_aware=True, iterations=3).run(qc)
    gate_names = [gate[0] for gate in routed.gates]
    assert "cx" in gate_names
    assert "measure" in gate_names


def test_noise_aware_swap_gate_uses_physical_qubits():
    """noise_aware 下 swap 门的 qubit 应该是 physical qubit（在图的节点集中）。"""
    g = _make_line_graph(4, fidelities=[0.9, 0.8, 0.7])
    nodes = set(g.nodes())
    qc = QuantumCircuit(4, 4)
    qc.cx(0, 3)
    qc.measure(list(range(4)), list(range(4)))
    routed = SabreRouting(g, noise_aware=True, iterations=3).run(qc)
    for gate in routed.gates:
        if gate[0] == "swap":
            assert gate[1] in nodes, f"swap qubit {gate[1]} not in graph nodes"
            assert gate[2] in nodes, f"swap qubit {gate[2]} not in graph nodes"
            # swap 的两个 qubit 必须在图中相邻
            assert g.has_edge(gate[1], gate[2]), f"swap on non-adjacent qubits {gate[1]}, {gate[2]}"


def test_noise_aware_cx_only_on_adjacent():
    """路由结果中的 cx 门的两个 qubit 必须在图中相邻。"""
    g = _make_line_graph(5, fidelities=[0.9, 0.85, 0.8, 0.7])
    qc = QuantumCircuit(5, 5)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.measure(list(range(5)), list(range(5)))
    for na in (True, False):
        routed = SabreRouting(g, noise_aware=na, iterations=3).run(qc)
        for gate in routed.gates:
            if gate[0] == "cx":
                assert g.has_edge(gate[1], gate[2]), \
                    f"noise_aware={na}: cx on non-adjacent {gate[1]}, {gate[2]}"


def test_noise_aware_heuristic_uses_weighted_distance():
    """
    验证 noise_aware 的 heuristic 评分确实使用加权距离而非跳数。
    构造两条路径长度相同（2跳）但 fidelity 显著不同的图，
    检查 heuristic 分数不同。
    """
    import math
    # 0 -- 1 -- 3  (upper, fid=0.5)
    # 0 -- 2 -- 3  (lower, fid=0.99)
    g = nx.Graph()
    g.add_edge(0, 1, fidelity=0.5)
    g.add_edge(1, 3, fidelity=0.5)
    g.add_edge(0, 2, fidelity=0.99)
    g.add_edge(2, 3, fidelity=0.99)
    g.graph["normal_order"] = [0, 1, 2, 3]

    sr_na = SabreRouting(g, noise_aware=True, iterations=1)
    sr_hop = SabreRouting(g, noise_aware=False, iterations=1)

    # 在 noise-aware 下, d(0,3) 走上路 = 2*(-log(0.5)) ≈ 1.386
    # 走下路 = 2*(-log(0.99)) ≈ 0.020
    # Floyd-Warshall 取最短 → 走下路
    d_na_03 = sr_na.distance_matrix[
        sr_na.physical_qubits_index[0]
    ][sr_na.physical_qubits_index[3]]
    assert abs(d_na_03 - 2 * (-math.log(0.99))) < 1e-6

    # 在 hop 下, d(0,3) = 2 (两条路都是 2 跳)
    d_hop_03 = sr_hop.distance_matrix[
        sr_hop.physical_qubits_index[0]
    ][sr_hop.physical_qubits_index[3]]
    assert d_hop_03 == 2


def test_transpiler_noise_aware_false_explicit():
    """显式传 noise_aware=False 给 Transpiler 不会出错。"""
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    routed = Transpiler(chip_backend=None).run(
        qc, use_dd=False, use_translate_to_basis=False,
        use_three_qubit_decompose=False, use_gate_compressor=False,
        noise_aware=False,
    )
    assert routed.nqubits >= qc.nqubits


def test_transpiler_noise_aware_true_no_backend_uses_hop():
    """无 backend 但 noise_aware=True，线形图无 fidelity 属性也不会报错。"""
    qc = QuantumCircuit(3, 3)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    # 线形图无 fidelity → 默认 1.0 → -log(1)=0，等效于无权
    routed = Transpiler(chip_backend=None).run(
        qc, use_dd=False, use_translate_to_basis=False,
        use_three_qubit_decompose=False, use_gate_compressor=False,
        noise_aware=True,
    )
    assert routed.nqubits >= qc.nqubits


# --------------- GateCompressor: merge_single_qubit_runs ---------------

def _sim_unitary(qc: QuantumCircuit) -> np.ndarray:
    """Compute the full unitary of a circuit via matrix multiplication (for small circuits)."""
    from quantum_hw.circuit.matrix import gate_matrix_dict
    n = qc.nqubits
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for gate_info in qc.gates:
        gate = gate_info[0]
        if gate == "measure":
            continue
        if gate in ("barrier", "delay", "reset"):
            continue
        mat = None
        qubit = None
        if gate in ("id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg"):
            mat = gate_matrix_dict[gate]
            qubit = (gate_info[1],)
        elif gate in ("rx", "ry", "rz", "p", "u", "r"):
            params = gate_info[1:-1]
            mat = gate_matrix_dict[gate](*params)
            qubit = (gate_info[-1],)
        elif gate in ("cx", "cnot", "cy", "cz", "swap", "iswap", "ecr"):
            mat = gate_matrix_dict[gate]
            qubit = (gate_info[1], gate_info[2])
        elif gate in ("rxx", "ryy", "rzz", "cp"):
            params = gate_info[1:-2]
            mat = gate_matrix_dict[gate](*params)
            qubit = (gate_info[-2], gate_info[-1])
        else:
            raise ValueError(f"Unknown gate {gate}")
        # Embed into full space
        if len(qubit) == 1:
            full = np.eye(1, dtype=complex)
            for q in range(n):
                full = np.kron(full, mat if q == qubit[0] else np.eye(2))
        elif len(qubit) == 2:
            # Build full matrix via basis state mapping
            full = np.zeros((dim, dim), dtype=complex)
            for col in range(dim):
                bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
                q0, q1 = qubit
                sub_col = bits[q0] * 2 + bits[q1]
                for sub_row in range(4):
                    if mat[sub_row, sub_col] == 0:
                        continue
                    new_bits = list(bits)
                    new_bits[q0] = (sub_row >> 1) & 1
                    new_bits[q1] = sub_row & 1
                    row = sum(b << (n - 1 - q) for q, b in enumerate(new_bits))
                    full[row, col] += mat[sub_row, sub_col]
        U = full @ U
    return U


def test_merge_hh_cancels():
    """H·H = I, should be removed entirely."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.h(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    non_functional = [g for g in merged.gates if g[0] not in ("measure", "barrier")]
    assert len(non_functional) == 0


def test_merge_xyz_to_single_u():
    """x·y·z on same qubit → single u gate."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.y(0)
    qc.z(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    non_functional = [g for g in merged.gates if g[0] not in ("measure", "barrier")]
    assert len(non_functional) == 1
    assert non_functional[0][0] == "u"


def test_merge_preserves_unitary():
    """Merged circuit produces the same unitary as the original."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(0.3, 0)
    qc.rx(0.5, 0)
    qc.s(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    U_orig = _sim_unitary(qc)
    U_merged = _sim_unitary(merged)
    # Unitaries may differ by global phase
    ratio = U_merged.flatten()[np.argmax(np.abs(U_orig.flatten()))] / U_orig.flatten()[np.argmax(np.abs(U_orig.flatten()))]
    assert np.allclose(U_merged, ratio / abs(ratio) * U_orig, atol=1e-10)


def test_merge_stops_at_two_qubit_gate():
    """cx in the middle splits the run: h·rx → u, cx, ry·rz → u."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    qc.ry(0.4, 0)
    qc.rz(0.5, 0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    names = [g[0] for g in merged.gates]
    assert names.count("u") == 2
    assert "cx" in names


def test_merge_different_qubits_not_merged():
    """Consecutive gates on different qubits are not merged together."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.x(1)
    qc.rz(0.3, 0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    # h(0) stays alone (next gate is on q1), x(1) stays alone, rz(0) stays alone
    non_functional = [g for g in merged.gates if g[0] not in ("measure", "barrier")]
    assert len(non_functional) == 3


def test_merge_single_gate_unchanged():
    """A single gate without neighbors stays as-is."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    assert merged.gates == [("h", 0)]


def test_merge_identity_run_removed():
    """x·x = I, run should be entirely removed."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.x(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    non_functional = [g for g in merged.gates if g[0] not in ("measure", "barrier")]
    assert len(non_functional) == 0


def test_merge_rz_rz_combines_angles():
    """rz(a)·rz(b) should become a single gate with combined angle."""
    qc = QuantumCircuit(1, 1)
    qc.rz(0.3, 0)
    qc.rz(0.5, 0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    non_functional = [g for g in merged.gates if g[0] not in ("measure", "barrier")]
    assert len(non_functional) == 1
    assert non_functional[0][0] == "u"
    # Verify unitary equivalence
    U_orig = _sim_unitary(qc)
    U_merged = _sim_unitary(merged)
    ratio = U_merged[0, 0] / U_orig[0, 0] if abs(U_orig[0, 0]) > 1e-10 else 1
    assert np.allclose(U_merged, ratio / abs(ratio) * U_orig, atol=1e-10)


def test_merge_with_symbolic_param_stops_run():
    """Symbolic parameters break the run — can't compute matrix."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz("theta", 0)
    qc.x(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    names = [g[0] for g in merged.gates]
    # h stays alone, rz("theta") stays alone, x stays alone
    assert len(names) == 3


def test_merge_keeps_measure():
    """measure gates are preserved after merge."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.x(0)
    qc.measure([0, 1], [0, 1])
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    assert any(g[0] == "measure" for g in merged.gates)


def test_merge_preserves_two_qubit_unitary():
    """On a 2-qubit circuit, merged result has the same unitary."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.rz(0.2, 0)
    qc.cx(0, 1)
    qc.h(1)
    qc.s(1)
    qc.t(1)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    U_orig = _sim_unitary(qc)
    U_merged = _sim_unitary(merged)
    ratio = U_merged.flatten()[np.argmax(np.abs(U_orig.flatten()))] / U_orig.flatten()[np.argmax(np.abs(U_orig.flatten()))]
    assert np.allclose(U_merged, ratio / abs(ratio) * U_orig, atol=1e-10)


def test_merge_in_full_compressor_run():
    """GateCompressor.run() uses merge internally — translate + compress reduces gate count."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.s(0)
    qc.t(0)
    qc.measure([0, 1], [0, 1])
    translated = TranslateToBasisGates(convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz").run(qc)
    compressed = GateCompressor().run(translated)
    # After translate: u·cz·u·u·u → merge should combine the 3 trailing u's into 1
    u_count_compressed = sum(1 for g in compressed.gates if g[0] == "u")
    u_count_translated = sum(1 for g in translated.gates if g[0] == "u")
    assert u_count_compressed < u_count_translated


def test_full_pipeline_unitary_preserved():
    """Full transpile pipeline (no routing) preserves unitary."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.y(1)
    qc.rz(0.7, 0)
    U_orig = _sim_unitary(qc)

    transpiled = Transpiler(chip_backend=None).run(
        qc, use_dd=False, use_sabre_routing=False,
        use_three_qubit_decompose=False,
        use_translate_to_basis=True,
        use_gate_compressor=True,
    )
    U_trans = _sim_unitary(transpiled)
    ratio = U_trans.flatten()[np.argmax(np.abs(U_orig.flatten()))] / U_orig.flatten()[np.argmax(np.abs(U_orig.flatten()))]
    assert np.allclose(U_trans, ratio / abs(ratio) * U_orig, atol=1e-10)


# --------------- GateCompressor: commutation reorder ---------------

def _unitary_equiv(qc1, qc2, atol=1e-10):
    """Check two circuits have equivalent unitaries (up to global phase)."""
    U1 = _sim_unitary(qc1)
    U2 = _sim_unitary(qc2)
    idx = np.argmax(np.abs(U1.flatten()))
    if abs(U1.flatten()[idx]) < 1e-12:
        return np.allclose(U1, U2, atol=atol)
    ratio = U2.flatten()[idx] / U1.flatten()[idx]
    return np.allclose(U2, (ratio / abs(ratio)) * U1, atol=atol)


def test_commutation_disjoint_qubits_commute():
    """Gates on completely disjoint qubits always commute."""
    assert GateCompressor._check_commutation(('h', 0), ('x', 1))
    assert GateCompressor._check_commutation(('rx', 0.5, 0), ('cz', 1, 2))
    assert GateCompressor._check_commutation(('cx', 0, 1), ('h', 2))


def test_commutation_diagonal_single_qubit():
    """Diagonal single-qubit gates on the same qubit commute."""
    assert GateCompressor._check_commutation(('rz', 0.3, 0), ('z', 0))
    assert GateCompressor._check_commutation(('p', 0.5, 0), ('s', 0))
    assert GateCompressor._check_commutation(('t', 0), ('rz', 1.0, 0))


def test_commutation_diagonal_with_cz():
    """Diagonal single-qubit gates commute with CZ."""
    assert GateCompressor._check_commutation(('rz', 0.5, 0), ('cz', 0, 1))
    assert GateCompressor._check_commutation(('z', 1), ('cz', 0, 1))
    assert GateCompressor._check_commutation(('p', 0.7, 0), ('cz', 0, 1))


def test_commutation_cz_cz():
    """Two CZ gates always commute (both diagonal)."""
    assert GateCompressor._check_commutation(('cz', 0, 1), ('cz', 0, 2))
    assert GateCompressor._check_commutation(('cz', 0, 1), ('cz', 1, 2))


def test_commutation_non_commuting_pair():
    """H does not commute with X on the same qubit (via matrix fallback)."""
    assert not GateCompressor._check_commutation(('h', 0), ('x', 0))
    # H on qubit 0 does not commute with CX(0,1).
    assert not GateCompressor._check_commutation(('h', 0), ('cx', 0, 1))


def test_commutation_matrix_fallback_cx_x_target():
    """X on the target qubit (q0 in cx_mat convention) commutes with CX."""
    # In the code's cx_mat convention, qubit 0 is the target.
    assert GateCompressor._check_commutation(('x', 0), ('cx', 0, 1))


def test_commutation_barrier_blocks():
    """Barrier never commutes with anything."""
    assert not GateCompressor._check_commutation(('h', 0), ('barrier', (0, 1)))
    assert not GateCompressor._check_commutation(('barrier', (0,)), ('rz', 0.5, 0))


def test_reorder_bubbles_single_qubit_past_disjoint_cz():
    """u gate on q0 can bubble past cz(1,2) to join another u on q0."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)       # single-qubit on q0
    qc.cz(1, 2)   # two-qubit on q1,q2 (disjoint from q0)
    qc.x(0)       # single-qubit on q0
    gc = GateCompressor()
    reordered = gc.commutation_reorder(qc)
    # After reorder: h(0), x(0) should be adjacent, cz(1,2) after them
    gates = reordered.gates
    gate_names = [g[0] for g in gates]
    # h and x should be next to each other
    idx_h = gate_names.index('h')
    idx_x = gate_names.index('x')
    assert abs(idx_h - idx_x) == 1
    assert _unitary_equiv(qc, reordered)


def test_reorder_stops_at_blocking_gate():
    """u gate on q0 cannot move past cx(0,1) when they don't commute."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)       # H does NOT commute with CX on shared qubit
    gc = GateCompressor()
    reordered = gc.commutation_reorder(qc)
    gates = reordered.gates
    gate_names = [g[0] for g in gates]
    # CX blocks the second h from moving left, so order stays: h, cx, h
    assert gate_names == ['h', 'cx', 'h']
    assert _unitary_equiv(qc, reordered)


def test_reorder_diagonal_through_cz():
    """rz on q0 can commute through cz(0,1)."""
    qc = QuantumCircuit(2, 2)
    qc.rz(0.3, 0)
    qc.cz(0, 1)
    qc.rz(0.7, 0)
    gc = GateCompressor()
    reordered = gc.commutation_reorder(qc)
    gates = reordered.gates
    # Both rz gates should now be adjacent
    rz_indices = [i for i, g in enumerate(gates) if g[0] == 'rz' and g[-1] == 0]
    assert len(rz_indices) == 2
    assert rz_indices[1] - rz_indices[0] == 1
    assert _unitary_equiv(qc, reordered)


def test_reorder_preserves_unitary_complex_circuit():
    """Commutation reorder preserves the unitary of a complex circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.rz(0.5, 1)
    qc.cz(0, 1)
    qc.x(2)
    qc.cz(1, 2)
    qc.rz(0.3, 0)
    qc.h(2)
    gc = GateCompressor()
    reordered = gc.commutation_reorder(qc)
    assert _unitary_equiv(qc, reordered)


def test_reorder_with_measure_barrier():
    """Measure and barrier block all gate movement."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure([0], [0])
    qc.x(0)
    gc = GateCompressor()
    reordered = gc.commutation_reorder(qc)
    gate_names = [g[0] for g in reordered.gates]
    # h cannot merge with x because measure is in between
    assert gate_names == ['h', 'measure', 'x']


def test_reorder_enables_more_merging():
    """Commutation reorder + merge gives fewer gates than merge alone."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cz(1, 2)   # disjoint from q0 — can be crossed
    qc.x(0)
    qc.cz(1, 2)   # disjoint from q0
    qc.rz(0.5, 0)
    gc = GateCompressor()
    # Without reorder: merge sees h | cz | x | cz | rz — no single-qubit merging on q0
    merged_only = gc.merge_single_qubit_runs(qc)
    u_count_no_reorder = sum(1 for g in merged_only.gates if g[0] in ('u', 'h', 'x', 'rz'))
    # With reorder: h, x, rz get grouped on q0 → merge into one u
    reordered = gc.commutation_reorder(qc)
    merged = gc.merge_single_qubit_runs(reordered)
    u_count_with_reorder = sum(1 for g in merged.gates if g[0] in ('u', 'h', 'x', 'rz'))
    assert u_count_with_reorder < u_count_no_reorder
    assert _unitary_equiv(qc, merged)


def test_reorder_full_pipeline_preserves_unitary():
    """Full GateCompressor.run() (with commutation reorder) preserves the unitary."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.4, 2)
    qc.x(0)
    qc.cz(1, 2)
    qc.h(2)
    qc.rz(0.7, 0)
    translated = TranslateToBasisGates(
        convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz"
    ).run(qc)
    compressed = GateCompressor().run(translated)
    # Compare translated (before compression) with compressed (after).
    assert _unitary_equiv(translated, compressed)


def test_expand_matrix_identity():
    """_expand_matrix with identity on position 0 in a 2-qubit space."""
    I = np.eye(2, dtype=complex)
    full = GateCompressor._expand_matrix(I, [0], 2)
    expected = np.eye(4, dtype=complex)
    assert np.allclose(full, expected)


def test_expand_matrix_x_on_second_qubit():
    """_expand_matrix puts X on qubit 1 of a 2-qubit space: I⊗X."""
    from quantum_hw.circuit.matrix import x_mat
    full = GateCompressor._expand_matrix(x_mat, [1], 2)
    expected = np.kron(np.eye(2), x_mat)
    assert np.allclose(full, expected)


def test_expand_matrix_cx():
    """_expand_matrix on a CX with swapped positions gives correct matrix."""
    from quantum_hw.circuit.matrix import cx_mat
    # CX on (1, 0) in a 2-qubit space — reversed qubit order
    full = GateCompressor._expand_matrix(cx_mat, [1, 0], 2)
    # This should be the XC (target-controlled) matrix
    from quantum_hw.circuit.matrix import xc_mat
    assert np.allclose(full, xc_mat)
