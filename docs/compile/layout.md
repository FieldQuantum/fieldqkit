# Layout — 比特布局选择

## 概览

- **模块**：`quantum_hw.compile.layout`（约670 行）
- **作用**：根据芯片拓扑和耦合保真度，为量子线路选择最优的物理比特子图。
- **依赖**：`Backend`、`QuantumCircuit`、`split_qubits`、`networkx`、`numpy`、`multiprocessing`
- **继承**：无（独立类）

---

## 类签名

```python
class Layout:
    def __init__(self, chip_backend: Backend)
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `chip_backend` | `Backend` | 硬件后端对象，提供芯片拓扑图和校准信息。 |

---

### 初始化属性

初始化时自动设置以下属性：

| 属性 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `priority_qubits` | `List[List[int]]` | `chip_backend.priority_qubits` | 芯片推荐比特优先级列表 |
| `graph` | `nx.Graph` | `chip_backend.edge_filtered_graph(thres=0.6)` | 保真度过滤后的耦合图（边保真度 ≥ 0.6 的子图，同时过滤节点） |
| `ncore` | `int` | `os.cpu_count() // 2` | 并行枚举子图时使用的进程数 |
| `fidelity_mean_threshold` | `float` | 硬编码 `0.9` | 候选子图的平均保真度筛选阈值 |
| `edge_fidelitys` | `Dict[Tuple, float]` | `nx.get_edge_attributes(graph, "fidelity")` | 边保真度字典 |
| `algorithm_switch_threshold` | `int` | 硬编码 `10` | 小规模枚举 vs 大规模 BFS 的分界比特数 |

---

### 类变量

| 变量 | 类型 | 说明 |
|---|---|---|
| `_TWO_QUBIT_GATES` | `frozenset` | 所有两比特门名称集合（含参数化），用于交互图提取 |

---

## 核心方法

### `select_layout(...)`

**签名：**

```python
def select_layout(
    self,
    qc: QuantumCircuit,
    target_qubits: list = [],
    use_chip_priority: bool = True,
    select_criteria: dict = {"key": "fidelity_var", "topology": "linear"},
    skip_split_qc: bool = True,
) -> nx.Graph
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `qc` | `QuantumCircuit` | — | 输入量子线路 |
| `target_qubits` | `list` | `[]` | 手动指定物理比特列表；非空时跳过自动选择，直接校验连通性和比特数 |
| `use_chip_priority` | `bool` | `True` | 优先使用芯片推荐比特（`Backend.priority_qubits`） |
| `select_criteria` | `dict` | `{"key": "fidelity_var", "topology": "linear"}` | 自动选择参数。`key` ∈ {`"fidelity_var"`, `"fidelity_mean"`}；`topology` ∈ {`"linear"`, `"nonlinear"`} |
| `skip_split_qc` | `bool` | `True` | 是否跳过线路分割（默认将所有比特视为一组） |

**返回值：** `nx.Graph` 子图，节点为选中的物理比特。

**返回子图属性：**
- `graph["normal_order"]`：`List[int]` — 物理比特顺序列表，定义虚拟→物理映射的对应关系

**异常：**
- `ValueError`：`target_qubits` 比特数与线路不匹配。
- `ValueError`：`target_qubits` 中有比特不在过滤后的图中（可能因保真度过低被移除）。
- `ValueError`：目标物理比特不连通。

---

#### `select_layout` 执行流程

```
select_layout(qc, target_qubits, ...)
  │
  ├─ [A] target_qubits 非空？
  │     ├─ 校验比特数匹配
  │     ├─ 校验比特存在于 graph
  │     ├─ 校验各分区连通性
  │     └─ 返回 graph.subgraph(target_qubits)
  │
  ├─ [B] use_chip_priority = True？
  │     ├─ 遍历 priority_qubits_list 寻找匹配长度的优先组
  │     ├─ 优先组可用 → 使用之
  │     └─ 无匹配 → fallback 到 select_qubits_by_local_algorithm
  │
  └─ [C] 自动搜索
        └─ select_qubits_by_local_algorithm(nqubits, select_criteria, interaction_graph)
```

---

### `_extract_interaction_graph(qc)` (classmethod)

**签名：**

```python
@classmethod
def _extract_interaction_graph(cls, qc: QuantumCircuit) -> nx.Graph
```

**用途：** 从线路中提取虚拟比特交互加权图。

**返回值：** `nx.Graph`，其中：
- 节点 = 虚拟比特索引
- 边 $(i, j)$ 的属性 `weight` = 比特 $i$ 和 $j$ 之间的两比特门数量

**用途：** 提供给 `_estimate_routing_cost` 做线路感知布局评分。

---

### `_estimate_routing_cost(interaction_graph, subgraph)` (staticmethod)

**签名：**

```python
@staticmethod
def _estimate_routing_cost(interaction_graph: nx.Graph, subgraph: nx.Graph) -> float
```

**用途：** 贪心估算将 `interaction_graph` 映射到 `subgraph` 的路由代价。

**算法：**
1. 按交互权重降序排列虚拟比特 $v$
2. 按度中心性降序排列物理比特 $p$
3. 贪心映射：最高交互权的虚拟比特 → 最高度的物理比特
4. 代价 = $\sum_{(u,v) \in E} w_{uv} \cdot d(p_u, p_v)$，其中 $d$ 是物理图上的最短路径长度

**返回值：** 估算代价（`float`），越小越好。当 `interaction_graph` 无边时返回 `0.0`。

---

## 线路感知布局选择（Circuit-Aware Layout）

当比特数 $\leq$ `algorithm_switch_threshold`（默认 10）且线路有两比特门交互时，布局选择流程为：

1. **提取交互图**：`_extract_interaction_graph(qc)` 构建虚拟比特加权图
2. **候选枚举**：枚举所有满足 `fidelity_mean_threshold` 的连通子图（`collect_all_subgraph_in_parallel`，利用 `multiprocessing.Pool` 并行化）
3. **排序选 Top-K**（K=10）：按 `fidelity_var` 或 `fidelity_mean` 排序
4. **路由代价重排序**：对 Top-K 候选调用 `_estimate_routing_cost`
5. **综合评分**：

$$\text{score} = \bar{f} - 0.05 \times \frac{\text{routing\_cost}}{\sum w}$$

选最高分候选。

---

## 自动比特选择策略

### `select_qubits_by_local_algorithm(nqubits, select_criteria, interaction_graph=None)`

根据 `nqubits` 自动选择算法：

| 条件 | 方法 | 说明 |
|---|---|---|
| `nqubits == 1` | `select_one_qubit_from_backend()` | 选单比特门保真度最高的比特 |
| `1 < nqubits ≤ threshold` | `select_few_qubits_from_backend(...)` | 子图枚举 + 保真度/路由代价排序 |
| `nqubits > threshold` | `select_much_qubits_from_backend(n)` | BFS 随机起点扩展 |

**容错机制：** 小规模模式下，若首选 (key, topology) 组合找不到候选，自动 fallback 尝试所有 (key, topology) 排列（共 4 种组合）。

---

### `select_one_qubit_from_backend()`

**用途：** 在最大连通分量中选择单比特门保真度最高的物理比特。

**返回值：** `[int]` — 单元素列表。

---

### `select_few_qubits_from_backend(...)`

**签名：**

```python
def select_few_qubits_from_backend(
    self,
    nqubits: int,
    key: Literal["fidelity_mean", "fidelity_var"] = "fidelity_var",
    topology: Literal["linear", "nonlinear"] = "linear",
    printdetails: bool = False,
    interaction_graph=None,
) -> list
```

| 参数 | 说明 |
|---|---|
| `nqubits` | 所需比特数 |
| `key` | 排序键：`"fidelity_var"`（方差最小优先）或 `"fidelity_mean"`（均值最大优先） |
| `topology` | `"linear"`（最大度 ≤ 2）或 `"nonlinear"`（有分支） |
| `printdetails` | 是否打印排序结果 |
| `interaction_graph` | 非 None 时启用线路感知重排序（Top-10 候选） |

**异常：** `ValueError` — 无满足条件的候选子图。

---

### `select_much_qubits_from_backend(nqubits)`

**用途：** 大规模比特选择（>10 比特），使用 BFS 从最大连通分量中随机起点扩展。

**异常：** `ValueError` — 所需比特数超过最大连通分量容量。

---

## 子图枚举辅助方法

| 方法 | 签名 | 说明 |
|---|---|---|
| `get_one_node_subgraph(node, nqubits)` | → `List[tuple]` | 以 `node` 为起点枚举所有大小为 `nqubits` 的连通子图 |
| `collect_all_subgraph_in_parallel(nqubits)` | → `List[tuple]` | 并行枚举所有起点的子图，合并去重 |
| `get_one_subgraph_info(nodes)` | → `Tuple \| None` | 计算子图度分布、平均保真度、保真度方差；低于阈值返回 `None` |
| `collect_all_subgraph_info_in_parallel(nqubits)` | → `List` | 并行计算所有的子图信息 |
| `classify_all_subgraph_according_topology(nqubits)` | → `(linear_list, nonlinear_list)` | 按最大度分类为线性（≤2）/非线性 |
| `sort_subgraph_according_mean_fidelity(nqubits, num, printdetails)` | → `(linear_top, nonlinear_top)` | 按平均保真度降序排列 |
| `sort_subgraph_according_var_fidelity(nqubits, num, printdetails)` | → `(linear_top, nonlinear_top)` | 按保真度方差升序排列 |

---

## 内部辅助方法

| 方法 | 说明 |
|---|---|
| `_get_node_neighbours(node)` | 返回 `node` 在 `self.graph` 中的邻居列表 |
| `_get_node_connect_dict(node, nqubits)` | 返回从 `node` 开始的 BFS 邻居字典 |
| `_get_largest_component()` | 返回 `self.graph` 最大连通分量的子图 |

---

## 示例

```python
from quantum_hw.api.backend import Backend
from quantum_hw.compile.layout import Layout
from quantum_hw.circuit import QuantumCircuit

backend = Backend("Baihua")
layout = Layout(backend)

qc = QuantumCircuit(6)
qc.h(0)
for i in range(5):
    qc.cx(i, i + 1)

# 自动选择（线路感知）
subgraph = layout.select_layout(qc)
print(f"选中比特: {list(subgraph.nodes())}")
print(f"比特顺序: {subgraph.graph['normal_order']}")

# 手动指定比特
subgraph = layout.select_layout(qc, target_qubits=[0, 1, 2, 3, 4, 5])

# 查看交互图
ig = Layout._extract_interaction_graph(qc)
print(f"交互边: {list(ig.edges(data=True))}")

# 查看排序结果
linear_top, nonlinear_top = layout.sort_subgraph_according_mean_fidelity(6, num=5, printdetails=True)
```

---

## 相关页面

- [编译模块总览](./README.md)
- [SabreRouting — SABRE 路由](./routing.md)
- [Transpiler — 编译流水线](./transpiler.md)
- [DAG — 有向无环图工具](./dag.md)（提供 `split_qubits`）
