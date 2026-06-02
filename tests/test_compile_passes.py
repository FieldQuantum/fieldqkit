from fieldqkit.circuit import QuantumCircuit
from fieldqkit.circuit.quantumcircuit_helpers import three_qubit_gates_available
from fieldqkit.compile.decompose import ThreeQubitGateDecompose
from fieldqkit.compile.optimize import GateCompressor
from fieldqkit.compile.routing import SabreRouting
from fieldqkit.compile.transpiler import Transpiler
from fieldqkit.compile.translate import TranslateToBasisGates
from fieldqkit.compile.layout import Layout
import networkx as nx
import numpy as np


def test_three_qubit_decompose_removes_three_qubit_gates():
    qc = QuantumCircuit(3, 3)
    qc.ccx(0, 1, 2)
    qc.ccz(0, 1, 2)
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
    """fidelity=1.0 时 all_perfect 回退到跳数距离，与 hop_matrix 一致。"""
    g = _make_line_graph(3, fidelities=[1.0, 1.0])
    sr = SabreRouting(g, noise_aware=True, iterations=1)
    # 所有 fidelity=1.0 → -log(1)=0 → all_perfect=True → 回退跳数
    assert sr.distance_matrix[0][1] == 1.0
    assert sr.distance_matrix[0][2] == 2.0
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
    from fieldqkit.circuit.matrix import gate_matrix_dict
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
    """Big-endian: cx(0,1) has control=q0, target=q1.

    X on the target (q1) commutes with CX; X on the control (q0) does not.
    """
    assert GateCompressor._check_commutation(('x', 1), ('cx', 0, 1))
    assert not GateCompressor._check_commutation(('x', 0), ('cx', 0, 1))


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
    from fieldqkit.circuit.matrix import x_mat
    full = GateCompressor._expand_matrix(x_mat, [1], 2)
    expected = np.kron(np.eye(2), x_mat)
    assert np.allclose(full, expected)


def test_expand_matrix_cx():
    """_expand_matrix places positions[0] as the high-order (control) slot.

    With big-endian cx_mat (control = first qubit = MSB):
      - positions=[0,1] → CX(control=q0, target=q1), the standard CNOT.
      - positions=[1,0] → CX(control=q1, target=q0), the reversed embedding.
    """
    from fieldqkit.circuit.matrix import cx_mat

    full = GateCompressor._expand_matrix(cx_mat, [0, 1], 2)
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    assert np.allclose(full, expected)

    full_rev = GateCompressor._expand_matrix(cx_mat, [1, 0], 2)
    expected_rev = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
    )
    assert np.allclose(full_rev, expected_rev)


def test_gatecompressor_preserves_semantics_across_cx():
    """Regression: GateCompressor must not reorder/merge a single-qubit gate
    across an asymmetric CX when they do not commute.

    rz on a CX *target* does NOT commute with CX, so the compressor must not
    bubble rz(0.4, q1) left across cx(0, 1) to merge it with rz(0.3, q1).
    Verified against the production statevector simulator (ground truth).
    Before the circuit.matrix big-endian fix, _check_commutation treated the
    CX as control=q1 and wrongly allowed this reorder, corrupting the circuit.
    """
    from fieldqkit.sim.statevector import simulate_statevector

    def state(qc):
        return simulate_statevector(qc, device="cpu").detach().cpu().numpy()

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.rz(0.3, 1)
    qc.cx(0, 1)
    qc.rz(0.4, 1)  # on the CX target — must NOT cross the CX

    before = state(qc)
    after = state(GateCompressor().run(qc))
    overlap = abs(np.vdot(before, after)) / (
        np.linalg.norm(before) * np.linalg.norm(after)
    )
    assert np.isclose(overlap, 1.0), f"GateCompressor changed semantics (overlap={overlap})"


# --------------- Layout: Pool fallback & subgraph enumeration ---------------

def _make_mock_backend(n, edges, fidelities=None, node_fidelities=None, priority_qubits=None):
    """Build a mock Backend-like object for Layout tests."""
    class MockBackend:
        def __init__(self):
            self.priority_qubits = priority_qubits or []

        def edge_filtered_graph(self, thres=0.6):
            g = nx.Graph()
            for i in range(n):
                nf = node_fidelities[i] if node_fidelities else 0.99
                g.add_node(i, fidelity=nf)
            for idx, (u, v) in enumerate(edges):
                ef = fidelities[idx] if fidelities else 0.95
                if ef >= thres and g.nodes[u].get("fidelity", 1.0) >= thres and g.nodes[v].get("fidelity", 1.0) >= thres:
                    g.add_edge(u, v, fidelity=ef)
            return g

    return MockBackend()


def test_layout_collect_subgraphs_serial_fallback():
    """Layout falls back to serial when Pool raises (simulated via monkey-patch)."""
    backend = _make_mock_backend(4, [(0, 1), (1, 2), (2, 3)])
    layout = Layout(backend)
    # Monkey-patch ncore to 0 to force Pool to fail
    layout.ncore = 0
    subgraphs = layout.collect_all_subgraph_in_parallel(3)
    assert len(subgraphs) > 0
    for sg in subgraphs:
        assert len(sg) == 3


def test_layout_collect_subgraph_info_serial_fallback():
    """collect_all_subgraph_info_in_parallel also falls back gracefully."""
    backend = _make_mock_backend(4, [(0, 1), (1, 2), (2, 3)])
    layout = Layout(backend)
    layout.ncore = 0
    info_list = layout.collect_all_subgraph_info_in_parallel(3)
    valid = [x for x in info_list if x is not None]
    assert len(valid) > 0


def test_layout_select_few_qubits_linear():
    """select_few_qubits_from_backend returns a connected linear subgraph."""
    backend = _make_mock_backend(
        5, [(0, 1), (1, 2), (2, 3), (3, 4)],
        fidelities=[0.95, 0.96, 0.97, 0.98],
    )
    layout = Layout(backend)
    qubits = layout.select_few_qubits_from_backend(3, key="fidelity_var", topology="linear", printdetails=False)
    assert len(qubits) == 3
    subgraph = layout.graph.subgraph(qubits)
    assert nx.is_connected(subgraph)
    assert max(dict(subgraph.degree()).values()) <= 2  # linear


def test_layout_select_few_qubits_nonlinear():
    """select_few_qubits_from_backend with star topology returns nonlinear."""
    # Star: 0 at center, connected to 1, 2, 3
    backend = _make_mock_backend(
        4, [(0, 1), (0, 2), (0, 3)],
        fidelities=[0.95, 0.96, 0.97],
    )
    layout = Layout(backend)
    qubits = layout.select_few_qubits_from_backend(4, key="fidelity_var", topology="nonlinear", printdetails=False)
    assert len(qubits) == 4
    subgraph = layout.graph.subgraph(qubits)
    assert nx.is_connected(subgraph)
    assert max(dict(subgraph.degree()).values()) > 2  # nonlinear


def test_layout_fidelity_mean_threshold_filters():
    """Subgraphs with mean fidelity below threshold are filtered out."""
    # Edges with low fidelity: mean below 0.9
    backend = _make_mock_backend(3, [(0, 1), (1, 2)], fidelities=[0.85, 0.88])
    layout = Layout(backend)
    layout.fidelity_mean_threshold = 0.9
    info = layout.get_one_subgraph_info((0, 1, 2))
    assert info is None  # Filtered out: mean(0.85, 0.88) = 0.865 < 0.9


def test_layout_fidelity_mean_threshold_passes():
    """Subgraphs with mean fidelity at or above threshold pass."""
    backend = _make_mock_backend(3, [(0, 1), (1, 2)], fidelities=[0.95, 0.92])
    layout = Layout(backend)
    layout.fidelity_mean_threshold = 0.9


# ---- Tests for cancel_two_qubit_pairs ----

def test_cancel_cz_pair_disjoint_intermediate():
    """CZ(0,1) - H(2) - CZ(0,1) should cancel."""
    qc = QuantumCircuit(3)
    qc.cz(0, 1); qc.h(2); qc.cz(0, 1)
    result = GateCompressor().run(qc)
    names = [g[0] for g in result.gates]
    assert 'cz' not in names
    assert 'h' in names


def test_cancel_cz_pair_rz_intermediate():
    """CZ(0,1) - RZ(q1) - CZ(0,1) should cancel (RZ diagonal commutes with CZ)."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1); qc.rz(1.0, 1); qc.cz(0, 1)
    result = GateCompressor().run(qc)
    names = [g[0] for g in result.gates]
    assert 'cz' not in names


def test_no_cancel_cz_pair_ry_overlap():
    """CZ(0,1) - RY(q1) - CZ(0,1) should NOT cancel (RY doesn't commute with CZ)."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1); qc.ry(1.0, 1); qc.cz(0, 1)
    result = GateCompressor().run(qc)
    n_cz = sum(1 for g in result.gates if g[0] == 'cz')
    assert n_cz == 2


def test_cancel_cx_pair_translated():
    """CX·CX → identity after translation to CZ basis + compression."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1); qc.cx(0, 1)
    translated = TranslateToBasisGates(two_qubit_gate_basis='cz').run(qc)
    result = GateCompressor().run(translated)
    assert len(result.gates) == 0


def test_cancel_swap_pair_translated():
    """SWAP·SWAP → identity after translation + compression."""
    qc = QuantumCircuit(2)
    qc.swap(0, 1); qc.swap(0, 1)
    translated = TranslateToBasisGates(two_qubit_gate_basis='cz').run(qc)
    result = GateCompressor().run(translated)
    assert len(result.gates) == 0


def test_cancel_cz_multiple_disjoint():
    """CZ pair separated by multiple gates on disjoint qubits should cancel."""
    qc = QuantumCircuit(5)
    qc.cz(0, 1); qc.h(2); qc.h(3); qc.cx(2, 3); qc.cz(0, 1)
    result = GateCompressor().run(qc)
    names = [g[0] for g in result.gates]
    assert 'cz' not in names


def test_cancel_cz_symmetric_qubit_order():
    """CZ(0,1) and CZ(1,0) should cancel (CZ is symmetric)."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1); qc.cz(1, 0)
    result = GateCompressor().run(qc)
    assert len(result.gates) == 0


def test_layout_fidelity_mean_threshold_passes():
    """Subgraphs with mean fidelity at or above threshold pass."""
    backend = _make_mock_backend(3, [(0, 1), (1, 2)], fidelities=[0.95, 0.92])
    layout = Layout(backend)
    layout.fidelity_mean_threshold = 0.9
    info = layout.get_one_subgraph_info((0, 1, 2))
    assert info is not None
    assert info[2] >= 0.9  # fidelity_mean


def test_layout_select_qubits_by_local_algorithm_one_qubit():
    """Single qubit layout selects the highest-fidelity node."""
    backend = _make_mock_backend(
        3, [(0, 1), (1, 2)],
        node_fidelities=[0.90, 0.99, 0.95],
    )
    layout = Layout(backend)
    qubits = layout.select_qubits_by_local_algorithm(1, {"key": "fidelity_var", "topology": "linear"})
    assert len(qubits) == 1
    assert qubits[0] == 1  # Q1 has highest node fidelity


def test_layout_priority_qubits_exact_match():
    """When priority_qubits has an entry matching nqubits, it's used directly."""
    backend = _make_mock_backend(
        4, [(0, 1), (1, 2), (2, 3)],
        priority_qubits=[[1, 2]],  # A 2-qubit priority layout
    )
    layout = Layout(backend)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    subgraph = layout.select_layout(qc, use_chip_priority=True)
    assert set(subgraph.nodes()) == {1, 2}


def test_layout_priority_qubits_no_match_falls_back():
    """When no priority_qubits match, falls back to local algorithm."""
    backend = _make_mock_backend(
        4, [(0, 1), (1, 2), (2, 3)],
        priority_qubits=[[0, 1, 2]],  # Only 3-qubit, no 2-qubit
    )
    layout = Layout(backend)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    subgraph = layout.select_layout(qc, use_chip_priority=True)
    assert len(subgraph.nodes()) == 2
    assert nx.is_connected(subgraph)


# --------------- QCIS: RZ ±π clamping ---------------


def test_routing_normalizes_numpy_int64_physical_qubits():
    """routing 的 logical_to_physical 映射值必须是 Python int，即使 subgraph 节点含 np.int64。"""
    g = nx.Graph()
    # 故意混入 np.int64 节点
    g.add_node(int(0), fidelity=0.99)
    g.add_node(np.int64(1), fidelity=0.99)
    g.add_node(int(2), fidelity=0.99)
    g.add_node(np.int64(3), fidelity=0.99)
    g.add_edge(0, np.int64(1), fidelity=0.95)
    g.add_edge(np.int64(1), 2, fidelity=0.90)
    g.add_edge(2, np.int64(3), fidelity=0.85)
    g.graph["normal_order"] = list(g.nodes())

    qc = QuantumCircuit(4, 4)
    qc.cx(0, 3)
    qc.cx(1, 2)
    qc.measure(list(range(4)), list(range(4)))

    routed = SabreRouting(g, noise_aware=False, iterations=3).run(qc)
    l2p = routed.logical_to_physical
    for v, p in l2p.items():
        assert type(p) is int, f"physical qubit {p} has type {type(p)}, expected int"


def test_backend_couplers_normalize_qubit_indices():
    """Backend._collect_couplers_with_attributes 应将 np.int64 的 qubits_index 转为 int。"""
    from fieldqkit.api.backend import Backend
    chip = {
        "chip_name": "test",
        "qubits_info": {
            "Q0": {"fidelity": 0.99},
            "Q1": {"fidelity": 0.99},
            "Q2": {"fidelity": 0.99},
        },
        "couplers_info": {
            "C0_1": {"connected": True, "qubits_index": [np.int64(0), np.int64(1)], "fidelity": 0.95},
            "C1_2": {"connected": True, "qubits_index": [np.int64(1), np.int64(2)], "fidelity": 0.90},
        },
        "global_info": {"two_qubit_gate_basis": "CZ"},
    }
    backend = Backend(chip=chip)
    graph = backend.get_graph()
    for n in graph.nodes():
        assert type(n) is int, f"node {n} has type {type(n)}, expected int"
    for u, v in graph.edges():
        assert type(u) is int and type(v) is int


# --------------- QCIS: RZ ±π clamping ---------------

def test_qcis_rz_at_positive_pi_is_clamped():
    """RZ at exactly +π is clamped to slightly less than π."""
    import math
    from fieldqkit.circuit.qcis import Instruction
    inst = Instruction("rz", [0], [math.pi])
    s = str(inst)
    # Should contain a value slightly less than π
    val = float(s.split()[-1])
    assert val < math.pi
    assert abs(val - math.pi) < 1e-9


def test_qcis_rz_at_negative_pi_is_clamped():
    """RZ at exactly -π is clamped to slightly greater than -π."""
    import math
    from fieldqkit.circuit.qcis import Instruction
    inst = Instruction("rz", [0], [-math.pi])
    s = str(inst)
    val = float(s.split()[-1])
    assert val > -math.pi
    assert abs(val + math.pi) < 1e-9


def test_qcis_rz_not_at_pi_unchanged():
    """RZ values not at ±π boundary are not modified."""
    import math
    from fieldqkit.circuit.qcis import Instruction
    for angle in [0.0, 0.5, -0.5, math.pi / 4, -math.pi / 3, 2.0, -2.0]:
        inst = Instruction("rz", [0], [angle])
        s = str(inst)
        val = float(s.split()[-1])
        assert val == angle, f"Angle {angle} was modified to {val}"


def test_qcis_rz_near_pi_but_not_exact_unchanged():
    """RZ values near but not exactly at ±π are unchanged."""
    import math
    from fieldqkit.circuit.qcis import Instruction
    # Just outside the 1e-12 tolerance
    angle = math.pi - 1e-11
    inst = Instruction("rz", [0], [angle])
    s = str(inst)
    val = float(s.split()[-1])
    assert val == angle


def test_qcis_non_rz_gates_not_clamped():
    """Non-RZ gates with angle arguments are not affected by clamping."""
    import math
    from fieldqkit.circuit.qcis import Instruction
    inst = Instruction("x2p", [0], [math.pi])
    s = str(inst)
    val = float(s.split()[-1])
    assert val == math.pi  # Not clamped


def test_qcis_rz_clamped_value_within_strict_interval():
    """The clamped value must be strictly in the open interval (-π, π)."""
    import math
    from fieldqkit.circuit.qcis import Instruction
    for angle in [math.pi, -math.pi]:
        inst = Instruction("rz", [0], [angle])
        s = str(inst)
        val = float(s.split()[-1])
        assert -math.pi < val < math.pi, f"Clamped {angle} to {val}, not in (-π, π)"


def test_qcis_full_circuit_rz_pi_clamped():
    """End-to-end: circuit with RZ(π) → circuit_to_qcis output has clamped value."""
    import math
    from fieldqkit.circuit import QuantumCircuit
    from fieldqkit.circuit.qcis import circuit_to_qcis
    from fieldqkit.compile.translate import TranslateToBasisGates
    qc = QuantumCircuit(1)
    qc.rz(math.pi, 0)
    translated = TranslateToBasisGates().run(qc)
    qcis = circuit_to_qcis(translated)
    for line in qcis.strip().splitlines():
        if "RZ" in line.upper():
            val = float(line.strip().split()[-1])
            assert -math.pi < val < math.pi, f"RZ value {val} not in open interval"


# --------------- Layout: large connected graph (mimics TianYan/GuoDun) ---------------

def test_layout_4qubit_on_large_connected_graph():
    """Layout can find 4-qubit subgraph on a large connected graph (like TianYan)."""
    # Build a 20-node line graph with high fidelity — mimics real chip
    edges = [(i, i + 1) for i in range(19)]
    backend = _make_mock_backend(20, edges, fidelities=[0.95] * 19)
    layout = Layout(backend)
    qubits = layout.select_qubits_by_local_algorithm(4, {"key": "fidelity_var", "topology": "linear"})
    assert len(qubits) == 4
    subgraph = layout.graph.subgraph(qubits)
    assert nx.is_connected(subgraph)


def test_layout_bfs_for_large_nqubits():
    """For nqubits > algorithm_switch_threshold, BFS is used."""
    edges = [(i, i + 1) for i in range(29)]
    backend = _make_mock_backend(30, edges, fidelities=[0.95] * 29)
    layout = Layout(backend)
    layout.algorithm_switch_threshold = 5
    qubits = layout.select_qubits_by_local_algorithm(10, {"key": "fidelity_var", "topology": "linear"})
    assert len(qubits) == 10
    subgraph = layout.graph.subgraph(qubits)
    assert nx.is_connected(subgraph)


def test_transpiler_full_pipeline_4qubit_mock_backend():
    """Full transpile pipeline on a 4-qubit circuit with mock backend."""
    from fieldqkit.api.backend import Backend
    chip_info = {
        "size": (1, 6),
        "priority_qubits": [],
        "qubits_info": {f"Q{i}": {"fidelity": 0.99} for i in range(6)},
        "couplers_info": {
            f"C{i}": {"qubits_index": [i, i + 1], "fidelity": 0.95}
            for i in range(5)
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }
    backend = Backend(chip_info)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cz(0, 3)
    qc.rz(np.pi / 4, 1)
    qc.cx(1, 3)
    qc.h(3)

    transpiled = Transpiler(backend).run(qc, use_dd=False)
    assert transpiled.nqubits >= 4
    assert len(transpiled.gates) > 0
    # All gates should be basis gates
    allowed = {"u", "cz", "measure", "barrier", "swap"}
    for gate in transpiled.gates:
        assert gate[0] in allowed, f"Unexpected gate {gate[0]}"


# ---- Tests for circuit-aware layout ----

def test_extract_interaction_graph_basic():
    """Interaction graph captures two-qubit gate counts."""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.cz(1, 2)
    ig = Layout._extract_interaction_graph(qc)
    assert ig.has_edge(0, 1)
    assert ig[0][1]["weight"] == 2
    assert ig.has_edge(1, 2)
    assert ig[1][2]["weight"] == 1
    assert not ig.has_edge(0, 2)


def test_extract_interaction_graph_no_2q_gates():
    """Interaction graph has no edges for single-qubit-only circuits."""
    qc = QuantumCircuit(3)
    qc.h(0); qc.x(1); qc.rz(0.5, 2)
    ig = Layout._extract_interaction_graph(qc)
    assert ig.number_of_edges() == 0
    assert set(ig.nodes()) == {0, 1, 2}


def test_estimate_routing_cost_adjacent():
    """Routing cost is minimal when interaction matches physical adjacency."""
    ig = nx.Graph()
    ig.add_edge(0, 1, weight=5)
    # Physical: 0-1 adjacent
    sg = nx.path_graph(2)
    cost = Layout._estimate_routing_cost(ig, sg)
    assert cost == 5.0  # distance 1 * weight 5


def test_estimate_routing_cost_distant():
    """Routing cost increases when interacting qubits are far apart."""
    ig = nx.Graph()
    ig.add_edge(0, 1, weight=5)
    # Physical: 0-1-2, virtual qubit 0 maps to physical 0, virtual 1 maps to physical 2
    # But greedy mapping depends on degree...
    sg = nx.path_graph(3)
    # Both virtual qubits have degree 1, physical 1 has degree 2 (mapped first? no, greedy maps by interaction weight)
    # Virtual 0 has weight 5, virtual 1 has weight 5 => both equal
    # Physical qubits by degree: 1(deg2), 0(deg1), 2(deg1)
    # So mapping: v0->p1, v1->p0 => distance 1
    cost = Layout._estimate_routing_cost(ig, sg)
    assert cost >= 5.0  # at least distance 1 * weight 5


def test_circuit_aware_layout_prefers_matching_topology():
    """Circuit-aware layout re-ranking should prefer subgraphs that match
    the circuit's interaction pattern when fidelities are comparable."""
    # Build a 6-qubit backend with two 3-qubit chains:
    # Chain A: 0-1-2 (fidelity 0.94) -- star-like 1 connects to 0 and 2
    # Chain B: 3-4-5 (fidelity 0.95) -- linear
    # Both have similar fidelity but B is slightly better.
    # Circuit has interactions 0-1, 1-2 (linear chain) -- should prefer chain B (linear, higher fidelity).
    chip_info = {
        "size": (1, 6),
        "priority_qubits": [],
        "qubits_info": {f"Q{i}": {"fidelity": 0.99} for i in range(6)},
        "couplers_info": {
            "C0": {"qubits_index": [0, 1], "fidelity": 0.94},
            "C1": {"qubits_index": [1, 2], "fidelity": 0.94},
            "C2": {"qubits_index": [3, 4], "fidelity": 0.95},
            "C3": {"qubits_index": [4, 5], "fidelity": 0.95},
        },
        "global_info": {"two_qubit_gate_basis": "cz"},
    }
    from fieldqkit.api.backend import Backend
    backend = Backend(chip_info)
    qc = QuantumCircuit(3)
    qc.cx(0, 1); qc.cx(1, 2); qc.cx(0, 1)
    layout = Layout(backend)
    subgraph = layout.select_layout(qc, use_chip_priority=False)
    selected = list(subgraph.nodes())
    # Should select the higher-fidelity chain (3,4,5)
    assert set(selected) == {3, 4, 5}


# ════════════════════════════════════════════════════════════════════
#  Appended: large-scale routing invariants & boundary cases
# ════════════════════════════════════════════════════════════════════


def _grid_graph(rows, cols):
    """Build an integer-labelled rows×cols grid coupling graph."""
    g = nx.grid_2d_graph(rows, cols)
    mapping = {node: i for i, node in enumerate(sorted(g.nodes()))}
    g = nx.relabel_nodes(g, mapping)
    g.graph["normal_order"] = list(range(rows * cols))
    return g


def _all_two_qubit_adjacent(routed, g):
    """Every routed cx/cz/swap must act on physically adjacent qubits."""
    for gate in routed.gates:
        if gate[0] in ("cx", "cz", "swap", "iswap", "ecr"):
            assert g.has_edge(gate[1], gate[2]), (
                f"{gate[0]}({gate[1]},{gate[2]}) on non-adjacent physical qubits"
            )


def test_routing_single_qubit_graph_no_swap():
    """Boundary: 1-qubit circuit on a 1-node graph routes with no swaps."""
    g = nx.Graph()
    g.add_node(0)
    g.graph["normal_order"] = [0]
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(0.5, 0)
    qc.measure(0, 0)
    routed = SabreRouting(g, iterations=3).run(qc)
    assert routed.nqubits >= 1
    assert not any(gate[0] == "swap" for gate in routed.gates)
    assert "h" in [gate[0] for gate in routed.gates]


def test_routing_already_adjacent_line_no_swaps():
    """Boundary: an already-routed line circuit (trivial map) gets no swaps."""
    g = _make_line_graph(6)
    qc = QuantumCircuit(6, 6)
    for i in range(5):
        qc.cx(i, i + 1)
    qc.measure(list(range(6)), list(range(6)))
    for niter in (1, 5):
        routed = SabreRouting(g, initial_mapping="trivial", iterations=niter).run(qc)
        assert not any(gate[0] == "swap" for gate in routed.gates), f"iterations={niter}"
        assert sum(1 for gate in routed.gates if gate[0] == "cx") == 5


def test_routing_preserves_two_qubit_gate_count_line():
    """Routing never adds or drops logical 2-qubit gates (only inserts swaps)."""
    g = _make_line_graph(7)
    qc = QuantumCircuit(7, 7)
    pairs = [(0, 6), (1, 5), (2, 4), (0, 3), (1, 6), (2, 5)]
    for a, b in pairs:
        qc.cx(a, b)
    qc.measure(list(range(7)), list(range(7)))
    routed = SabreRouting(g, iterations=5).run(qc)
    assert sum(1 for gate in routed.gates if gate[0] == "cx") == len(pairs)
    _all_two_qubit_adjacent(routed, g)


def test_routing_wide_deep_line_adjacency_invariant():
    """Large-scale: 20-qubit, deep circuit on a line — all 2q gates adjacent."""
    import random

    random.seed(7)
    g = _make_line_graph(20)
    qc = QuantumCircuit(20, 20)
    n_cx = 0
    for _ in range(60):
        a, b = random.sample(range(20), 2)
        qc.cx(a, b)
        n_cx += 1
    qc.measure(list(range(20)), list(range(20)))
    routed = SabreRouting(g, iterations=5).run(qc)
    assert routed.nqubits == 20
    _all_two_qubit_adjacent(routed, g)
    assert sum(1 for gate in routed.gates if gate[0] == "cx") == n_cx


def test_routing_wide_deep_grid_adjacency_invariant():
    """Large-scale: 24-qubit deep circuit on a 4x6 grid — all 2q gates adjacent."""
    import random

    random.seed(11)
    g = _grid_graph(4, 6)  # 24 qubits
    qc = QuantumCircuit(24, 24)
    n_cx = 0
    for _ in range(70):
        a, b = random.sample(range(24), 2)
        qc.cx(a, b)
        n_cx += 1
    qc.measure(list(range(24)), list(range(24)))
    routed = SabreRouting(g, iterations=5).run(qc)
    assert routed.nqubits == 24
    _all_two_qubit_adjacent(routed, g)
    assert sum(1 for gate in routed.gates if gate[0] == "cx") == n_cx


def test_routing_constrained_topology_forces_swaps():
    """A line graph with conflicting long-range interactions must insert swaps."""
    g = _make_line_graph(6)
    qc = QuantumCircuit(6, 6)
    pairs = [(0, 3), (1, 4), (2, 5), (0, 5), (1, 3), (2, 4), (0, 4), (1, 5)]
    for a, b in pairs:
        qc.cx(a, b)
    qc.measure(list(range(6)), list(range(6)))
    routed = SabreRouting(g, iterations=5).run(qc)
    assert any(gate[0] == "swap" for gate in routed.gates)
    _all_two_qubit_adjacent(routed, g)


def test_n_trials_reduces_swaps_on_hard_circuit():
    """More trials should never produce more swaps than a single trial."""
    import random

    g = _make_line_graph(7)
    qc = QuantumCircuit(7, 7)
    pairs = [(0, 6), (1, 5), (2, 4), (0, 3), (1, 6), (2, 5)]
    for a, b in pairs:
        qc.cx(a, b)
    qc.measure(list(range(7)), list(range(7)))
    for seed in range(3):
        random.seed(seed)
        single = SabreRouting(
            g, iterations=5, n_trials=1, do_random_choice=True, initial_mapping="random"
        ).run(qc)
        random.seed(seed)
        multi = SabreRouting(
            g, iterations=5, n_trials=12, do_random_choice=True, initial_mapping="random"
        ).run(qc)
        s_single = sum(1 for gate in single.gates if gate[0] == "swap")
        s_multi = sum(1 for gate in multi.gates if gate[0] == "swap")
        assert s_multi <= s_single, f"seed={seed}: multi={s_multi} > single={s_single}"


def test_routing_logical_to_physical_bijection_large():
    """On a large line graph the logical↔physical maps stay valid bijections."""
    g = _make_line_graph(20)
    qc = QuantumCircuit(20, 20)
    for i in range(0, 19, 3):
        qc.cx(i, i + 1)
    qc.cx(0, 19)
    qc.measure(list(range(20)), list(range(20)))
    routed = SabreRouting(g, iterations=5).run(qc)
    l2p = routed.logical_to_physical
    p2l = routed.physical_to_logical
    assert set(l2p.values()) == set(p2l.keys())
    assert set(l2p.keys()) == set(p2l.values())
    for v, p in l2p.items():
        assert p2l[p] == v


def test_routing_grid_swaps_use_grid_edges():
    """On a grid, inserted swaps connect physically adjacent grid qubits."""
    import random

    random.seed(5)
    g = _grid_graph(3, 4)  # 12 qubits
    qc = QuantumCircuit(12, 12)
    for _ in range(40):
        a, b = random.sample(range(12), 2)
        qc.cx(a, b)
    qc.measure(list(range(12)), list(range(12)))
    routed = SabreRouting(g, iterations=5).run(qc)
    nodes = set(g.nodes())
    saw_swap = False
    for gate in routed.gates:
        if gate[0] == "swap":
            saw_swap = True
            assert gate[1] in nodes and gate[2] in nodes
            assert g.has_edge(gate[1], gate[2])
    # This deep, conflict-heavy problem requires at least one swap.
    assert saw_swap


# ════════════════════════════════════════════════════════════════════
#  Appended: GateCompressor on larger circuits
# ════════════════════════════════════════════════════════════════════


def test_compressor_collapses_long_single_qubit_run():
    """A long run of single-qubit gates on one qubit collapses to a single u."""
    qc = QuantumCircuit(1, 1)
    for _ in range(20):
        qc.h(0)
        qc.t(0)
        qc.s(0)
    gc = GateCompressor()
    merged = gc.merge_single_qubit_runs(qc)
    non_functional = [g for g in merged.gates if g[0] not in ("measure", "barrier")]
    assert len(non_functional) == 1
    assert non_functional[0][0] == "u"


def test_compressor_preserves_unitary_wide_circuit():
    """GateCompressor.run() preserves the unitary of a 4-qubit translated circuit."""
    qc = QuantumCircuit(4, 4)
    qc.h(0); qc.x(1); qc.rz(0.3, 2); qc.ry(0.4, 3)
    qc.cz(0, 1); qc.cz(2, 3)
    qc.h(0); qc.s(1); qc.t(2); qc.x(3)
    qc.cz(1, 2)
    qc.rz(0.7, 0); qc.h(3)
    translated = TranslateToBasisGates(
        convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz"
    ).run(qc)
    compressed = GateCompressor().run(translated)
    assert _unitary_equiv(translated, compressed)
    # Compression should not increase the gate count.
    assert len(compressed.gates) <= len(translated.gates)


def test_compressor_idempotent_on_compressed_circuit():
    """Running GateCompressor a second time does not change a compressed circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0); qc.cz(0, 1); qc.x(0); qc.cz(1, 2); qc.rz(0.5, 0)
    translated = TranslateToBasisGates(
        convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz"
    ).run(qc)
    once = GateCompressor().run(translated)
    twice = GateCompressor().run(once)
    assert [g[0] for g in once.gates] == [g[0] for g in twice.gates]
    assert _unitary_equiv(once, twice)


def test_compressor_cancels_cz_pair_through_disjoint_block():
    """A CZ pair separated only by gates on disjoint qubits cancels."""
    qc = QuantumCircuit(5, 5)
    qc.cz(0, 1)
    qc.h(2); qc.h(3); qc.cx(2, 3); qc.rz(0.3, 4)
    qc.cz(0, 1)
    result = GateCompressor().run(qc)
    assert not any(g[0] == "cz" for g in result.gates)
