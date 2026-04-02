# DAG — 有向无环图工具

## 概览

- **模块**：`quantum_hw.compile.dag`（约220 行）
- **作用**：提供量子线路与有向无环图（DAG）、无向交互图之间的互转工具，供编译 Pass 内部使用。
- **依赖**：`networkx`、`convert_gate_info_to_dag_info`（来自 `quantumcircuit_helpers`）

---

## 核心函数

### `qc2dag(qc, show_qubits=True) -> nx.DiGraph`

**签名：**

```python
def qc2dag(qc: QuantumCircuit, show_qubits: bool = True) -> nx.DiGraph
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `qc` | `QuantumCircuit` | — | 输入量子线路 |
| `show_qubits` | `bool` | `True` | 是否在节点名中包含比特信息 |

**返回值：** `nx.DiGraph`

**DAG 结构：**
- **节点名格式**：`"{gate}_{idx}_{[qubits]}"` （如 `"cx_3_[0, 1]"`）
- **节点属性**：
  - `qubits: list[int]` — 门作用的比特列表
  - `params: list` — 参数化门的参数列表（仅参数化门有此属性）
  - `cbits: list[int]` — 经典比特（仅 `measure` 门有此属性）
  - `duration: float` — 持续时间（仅 `delay` 门有此属性）
- **边属性**：
  - `qubit: list[int]` — 该边对应的物理比特
- **图属性**：
  - `graph["qubits"]` — 保存 `qc.qubits` 比特列表

**实现：** 调用 `convert_gate_info_to_dag_info` 生成节点列表和边列表，构建 `nx.DiGraph`。

---

### `dag2qc(dag, nqubits=None, ncbits=None) -> QuantumCircuit`

**签名：**

```python
def dag2qc(dag: nx.DiGraph, nqubits: int | None = None, ncbits: int | None = None) -> QuantumCircuit
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `dag` | `nx.DiGraph` | — | 输入 DAG |
| `nqubits` | `int \| None` | `None` | 比特数。`None` 时自动取 `max(qubit) + 1` |
| `ncbits` | `int \| None` | `None` | 经典比特数。`None` 时等于 `nqubits` |

**返回值：** `QuantumCircuit`（`qubits` 属性从 `dag.graph["qubits"]` 继承）。

**算法：** 使用 `nx.topological_sort(dag)` 遍历节点，每个节点根据门名从节点属性还原为门信息元组。

**支持的门类型还原：**

| 门类型 | 还原方式 |
|---|---|
| 固定单比特门 | `(gate, qubit)` |
| 参数化单比特门 | `(gate, *params, qubit)` |
| 固定两比特门 | `(gate, qubit1, qubit2)` |
| 参数化两比特门 | `(gate, *params, qubit1, qubit2)` |
| 三比特门 | `(gate, qubit1, qubit2, qubit3)` |
| `measure` | `(gate, qubits, cbits)` |
| `barrier` | `(gate, tuple(qubits))` |
| `delay` | `(gate, duration, tuple(qubits))` |
| `reset` | `(gate, qubit)` |

---

### `qc2graph(qc) -> nx.Graph`

**签名：**

```python
def qc2graph(qc: QuantumCircuit) -> nx.Graph
```

将线路转为无向交互图：
- **节点** = `qc.qubits` 中的每个比特
- **边** = 存在两比特（或三比特）门交互的比特对
- 通过 `get_qcgraph_edges` 提取边列表

**用途：** 用于 `split_qubits()` 查找连通分量，以及 Layout 的交互图分析。

---

### `split_qubits(qc) -> list[list[int]]`

**签名：**

```python
def split_qubits(qc: QuantumCircuit) -> list[list[int]]
```

基于 `qc2graph` 找连通分量，返回比特分组列表。每个分组是一个比特列表。

**用途：** 被 `SabreRouting.run()` 调用，将不互相交互的比特组独立处理。

---

## 辅助函数

### `get_qcgraph_edges(gates) -> list[tuple]`

**签名：**

```python
def get_qcgraph_edges(gates: list) -> list[tuple]
```

从门信息列表中提取所有两比特门的比特对。

**门类型处理：**

| 门类型 | 行为 |
|---|---|
| 固定/参数化单比特门 | 跳过 |
| 固定两比特门 | 取 `gate_info[1:]` 作为边 |
| 参数化两比特门 | 取 `gate_info[2:]`（跳过参数）作为边 |
| 三比特门 | 拆分为两条边：`gate_info[1:3]` 和 `gate_info[2:4]` |
| 功能性指令 | 跳过 |
| 未知门 | `ValueError` |

---

## 可视化函数

### `draw_dag(dag, output="dag_figure.png")`

**依赖：** `pygraphviz`（`nx.nx_agraph`）、`matplotlib`。

**功能：**
- 使用 Graphviz `dot` 布局引擎渲染 DAG
- 节点标签显示门名（`measure` 门额外显示经典比特 `c{bit}`）
- 边标签显示比特名（`q0`, `q1` 等）
- 输出 300 DPI 图片并通过 `plt.show()` 显示
- 图片同时保存到 `output` 指定的文件

---

### `draw_graph(G)`

**依赖：** `matplotlib`。

**功能：**
- 使用 `nx.shell_layout` 布局
- 天蓝色圆形节点，灰色边，粗体标签
- 图片大小 7×6 英寸

---

## 示例

```python
from quantum_hw.compile.dag import qc2dag, dag2qc, qc2graph, split_qubits
from quantum_hw.circuit import QuantumCircuit

# 线路 → DAG → 线路
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

dag = qc2dag(qc)
print(f"DAG 节点数: {dag.number_of_nodes()}")
print(f"DAG 边数: {dag.number_of_edges()}")

# DAG 还原
qc_restored = dag2qc(dag, nqubits=3)
assert len(qc_restored.gates) == len(qc.gates)

# 交互图与连通分量
qc2 = QuantumCircuit(4)
qc2.cx(0, 1)
qc2.cx(2, 3)  # 两组独立的比特
graph = qc2graph(qc2)
groups = split_qubits(qc2)
print(f"连通分量: {groups}")  # [[0, 1], [2, 3]]
```

---

## 相关页面

- [编译模块总览](./README.md)
- [GateCompressor — 门压缩](./optimize.md)（DAG 压缩核心使用 `qc2dag` / `dag2qc`）
- [DynamicalDecoupling — 动力学去耦](./schedule.md)（在 DAG 上插入 DD 序列）
- [SabreRouting — 路由](./routing.md)（使用 `qc2dag` 和 `split_qubits`）
