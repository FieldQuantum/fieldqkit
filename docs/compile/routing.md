# SabreRouting — SABRE 路由

## 概览

- **模块**：`fieldqkit.compile.routing`（约650 行）
- **作用**：基于 SABRE 算法插入 SWAP 门，使所有两比特门满足芯片拓扑连通性约束。
- **继承**：`TranspilerPass`（实现 `run()` 方法）
- **依赖**：`networkx`（Floyd-Warshall 距离矩阵）、`qc2dag` / `split_qubits`（DAG 构建与比特分割）

---

## 辅助函数

### `extract_qubits(node_name: str) -> list`

从 DAG 节点名中提取比特索引。例如 `"cx_3_[0, 1]"` → `[0, 1]`。使用正则表达式解析方括号内的数字。

---

### `update_v2p_and_p2v_mapping(v2p: dict, swap_gate_info: tuple) -> (dict, dict)`

**用途：** 对映射字典应用一次 SWAP 操作。

**参数：**
- `v2p`：当前虚拟→物理映射
- `swap_gate_info`：`("swap", vq1, vq2)` 三元组

**返回值：** 更新后的 `(v2p, p2v)` 映射字典对（`deepcopy`，不修改原始）。

---

## 类签名

```python
class SabreRouting(TranspilerPass):
    def __init__(
        self,
        subgraph: nx.Graph,
        initial_mapping: Literal["random", "trivial"] | list = "trivial",
        do_random_choice: bool = False,
        iterations: int = 5,
        heuristic: Literal["basic", "lookahead", "basic_decay", "lookahead_decay"] = "lookahead_decay",
        max_extended_set_weight: float = 0.5,
        noise_aware: bool = False,
        n_trials: int = 1,
    )
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `subgraph` | `nx.Graph` | — | 物理比特耦合子图（由 `Layout.select_layout` 返回）。需包含 `graph["normal_order"]` 属性。 |
| `initial_mapping` | `str \| list` | `"trivial"` | 初始映射策略。`"trivial"` = 按物理比特顺序一一对应；`"random"` = 随机打乱物理比特后映射；`list` = 显式映射列表（长度必须等于物理比特数）。 |
| `do_random_choice` | `bool` | `False` | SWAP 候选中有多个最优时是否随机选择。`False` 时取第一个（确定性）。 |
| `iterations` | `int` | `5` | SABRE 前后向迭代次数。必须为**正奇数**，否则抛 `ValueError`。越多结果越稳定。 |
| `heuristic` | `str` | `"lookahead_decay"` | 启发式函数名称（详见下方）。 |
| `max_extended_set_weight` | `float` | `0.5` | lookahead 启发式中，前瞻集权重系数 $W$ 的上界。实际 $W = \min(0.5,\, |E|/|F|)$。 |
| `noise_aware` | `bool` | `False` | 是否使用保真度加权距离矩阵。 |
| `n_trials` | `int` | `1` | 多随机初始映射试验次数。 |

**异常：**
- `ValueError`：`initial_mapping` 为列表时，长度与物理比特数不匹配。
- `ValueError`：`initial_mapping` 为字符串时，虚拟比特数与物理比特数不匹配。

---

### 初始化属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `coupling_graph` | `nx.Graph` | 保存的物理拓扑图引用 |
| `distance_matrix` | `np.ndarray` | Floyd-Warshall 距离矩阵（噪声感知或均匀） |
| `hop_matrix` | `np.ndarray` | 跳数矩阵（始终无权），用于邻接判断 |
| `physical_qubits` | `List[int]` | 物理比特列表（按 `normal_order`） |
| `physical_qubits_index` | `Dict[int, int]` | 物理比特→矩阵索引映射 |
| `n_trials` | `int` | 试验次数 |
| `initial_mapping` | `str \| list` | 保存的初始映射策略 |
| `do_random_choice` | `bool` | SWAP 随机选择标志 |
| `iterations` | `int` | 前后向迭代次数 |
| `heuristic` | `str` | 启发式函数名 |
| `extended_successor_set` | `list` | lookahead 前瞻集（运行时动态更新） |
| `max_extended_set_weight` | `float` | 前瞻权重上界 |
| `decay_parameter` | `dict` | decay 启发式的比特衰减参数（运行时更新） |
| `_cache` | `OrderedDict` | DAG 前驱/后继查询缓存（LRU，上限 10000） |

---

## 启发式函数详解

SABRE 在每一步评估所有 SWAP 候选的启发式分数，选择分数最低的 SWAP 执行。

### `basic` — 基础距离

$$H = \frac{1}{|F|} \sum_{(v_1, v_2) \in F} d(p_{v_1}', p_{v_2}')$$

仅考虑前沿层门在 SWAP 后的平均距离。

---

### `lookahead` — 基础 + 前瞻

$$H = \frac{1}{|F|} \sum_F d(p_{v_1}', p_{v_2}') + W \cdot \frac{1}{|E|} \sum_E d(p_{v_1}', p_{v_2}')$$

其中 $E$ 是前沿层后继中的两比特门集合（前瞻集），$W = \min(\text{max\_extended\_set\_weight},\, |E|/|F|)$。

---

### `basic_decay` — 基础 + 衰减

$$H = \max(\delta_{p_1'}, \delta_{p_2'}) \cdot \frac{1}{|F|} \sum_F d(p_{v_1}', p_{v_2}')$$

其中 $\delta_q$ 是物理比特 $q$ 的衰减因子，初始为 1，每次 SWAP 涉及的两比特 $+0.01$。惩罚频繁使用的比特。

---

### `lookahead_decay` — 前瞻 + 衰减（默认，推荐）

$$H = \max(\delta_{p_1'}, \delta_{p_2'}) \cdot \left(\frac{1}{|F|}\sum_F d + W \cdot \frac{1}{|E|}\sum_E d\right)$$

结合前瞻和衰减，效果最好。

---

## 噪声感知路由

当 `noise_aware=True` 时：

```python
# 对耦合图每条边计算 -log(保真度) 权重
for u, v, data in wg.edges(data=True):
    f = max(data.get("fidelity", 1.0), 1e-6)
    wg[u][v]["weight"] = -math.log(f)
distance_matrix = floyd_warshall_numpy(wg, weight="weight")
```

- 高保真度路径权重低（$-\log(0.99) \approx 0.01$），优先选择
- 低保真度路径权重高（$-\log(0.5) \approx 0.69$），被惩罚
- SWAP 倾向经过高质量耦合器
- 始终维持一个无权跳数矩阵 `hop_matrix` 用于邻接判断（`_hop_distance` 方法）

---

## 多试验模式 (n_trials)

当 `n_trials > 1` 时：

1. 第一次试验使用 `initial_mapping` 指定的策略
2. 后续试验强制使用 `"random"` 初始映射
3. 每次执行完整的前后向 SABRE 迭代
4. 取插入 SWAP 数最少的结果
5. 试验完成后恢复 `initial_mapping` 为原始值

**适用场景：** 大线路或复杂拓扑时，不同初始映射可显著影响 SWAP 数。推荐 `n_trials=8~16`。

---

## `run(...)` 方法

**签名：**

```python
def run(self, qc: QuantumCircuit) -> QuantumCircuit
```

**执行流程：**

1. 提取所有比特分组（`split_qubits(qc)`），展平为 `virtual_qubits`
2. 构建正向 DAG（`qc2dag`）和反向 DAG（反转 `qc.gates` 后 `qc2dag`）
3. 循环 `n_trials` 次，每次调用 `_run_once`
4. 选取 SWAP 数最少的结果
5. 构建新的 `QuantumCircuit`，物理比特为 `self.physical_qubits`

**返回值：** 路由后的 `QuantumCircuit`，附带：
- `logical_to_physical: Dict[int, int]` — 最终虚拟→物理映射
- `physical_to_logical: Dict[int, int]` — 最终物理→虚拟映射
- `params_value` — 继承自输入 `qc`
- `qubits` — 设为 `self.physical_qubits`

---

### `_run_once(qc, virtual_qubits, dag, rev_dag)`

**用途：** 执行一次完整的前后向 SABRE 路由。

**流程：**
1. 初始化虚拟→物理映射（`_initialize_v2p_p2v`）
2. 交替执行 `iterations` 次正向/反向路由（`_single_sabre_routing`）
3. 最后一次迭代设 `do_map_node_to_gate=True`，输出物理门序列
4. 返回 `(new_gates, nswap, v2p, final_p2v)`

---

### `_single_sabre_routing()`

**核心 SABRE 循环：**

```
while front_layer 非空:
    execute_list = 前沿层中可直接执行的门（单比特 或 两比特且比特相邻）
    if execute_list 非空:
        执行这些门，推进 DAG 后继到前沿层
        重置 decay 参数
    else:
        枚举所有 SWAP 候选
        用启发式函数评分
        选择最优 SWAP（分数最低）
        应用 SWAP 到 v2p/p2v 映射
        更新 decay 参数
```

---

## 内部辅助方法

| 方法 | 说明 |
|---|---|
| `_distance_matrix_element(pq1, pq2)` | 查距离矩阵元素（考虑噪声加权） |
| `_hop_distance(pq1, pq2)` | 查跳数矩阵元素（始终无权） |
| `_dag_successors(node)` / `_dag_predecessors(node)` | 带缓存的 DAG 前驱/后继查询 |
| `_get_nodes(node, query_type)` | 底层 DAG 查询，带 LRU 缓存（上限 10000） |
| `_initialize_v2p_p2v(virtual_qubits)` | 根据 `initial_mapping` 策略初始化映射 |
| `_mapping_node_to_gate_info(node)` | 将 DAG 节点转换为物理门信息元组 |
| `_get_execute_node_list(front_layer)` | 返回前沿层中可直接执行的门列表 |
| `_has_no_correlation_on_front_layer(node, front_layer)` | 检查 `node` 与当前前沿层无比特冲突 |
| `_get_swap_candidate_list(front_layer)` | 枚举前沿层所有非直接执行门的 SWAP 候选 |
| `_get_extended_successor_set(front_layer)` | 构建前瞻集 $E$（前沿层后继中的两比特门） |
| `_reset_decay_parameter()` | 重置所有比特的衰减因子为 1 |
| `_update_decay_parameter(swap_gate_info)` | 将 SWAP 涉及的两比特衰减因子各 $+0.01$ |
| `_heuristic_score(swap_gate_info, front_layer)` | 根据 `self.heuristic` 分派到对应启发式函数 |

---

## SWAP 分解

路由产生的 `("swap", pq1, pq2)` 门在后续 `TranslateToBasisGates` Pass 中被分解为 3 个 CX（或等效的本征两比特门），具体见 [decompose.md](./decompose.md)。

---

## 示例

```python
import networkx as nx
from fieldqkit.compile.routing import SabreRouting
from fieldqkit.circuit import QuantumCircuit

# 构造简单拓扑
subgraph = nx.Graph()
subgraph.add_edges_from([(0, 1), (1, 2), (2, 3)])
subgraph.graph["normal_order"] = [0, 1, 2, 3]

# 构造线路（包含非相邻两比特门）
qc = QuantumCircuit(4)
qc.cx(0, 3)  # 比特 0 和 3 不相邻
qc.cx(1, 2)

# 路由
router = SabreRouting(subgraph, heuristic="lookahead_decay", iterations=5)
routed = router.run(qc)
print(f"路由后门数: {len(routed.gates)}")
print(f"映射: {routed.logical_to_physical}")

# 噪声感知 + 多试验
for u, v in subgraph.edges():
    subgraph[u][v]["fidelity"] = 0.95
router = SabreRouting(subgraph, noise_aware=True, n_trials=8)
routed = router.run(qc)
```

---

## 相关页面

- [编译模块总览](./README.md)
- [Layout — 比特布局选择](./layout.md)
- [Transpiler — 编译流水线](./transpiler.md)
- [Decompose — 门分解](./decompose.md)（SWAP 分解）
- [DAG — 有向无环图工具](./dag.md)
