# DynamicalDecoupling — 动力学去耦

## 概览

- **模块**：`fieldqkit.compile.schedule`（约210 行）
- **作用**：在量子线路的空闲时段插入动力学去耦（DD）序列，抑制退相干噪声。
- **继承**：`TranspilerPass`（实现 `run()` 方法）
- **依赖**：`qc2dag` / `dag2qc`（DAG 构建）、`networkx.topological_generations`

---

## 类签名

```python
class DynamicalDecoupling(TranspilerPass):
    def __init__(self, t1g: float, t2g: float)
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `t1g` | `float` | 单比特门耗时（秒），也用作 DD 序列中每个脉冲的最小间隔 |
| `t2g` | `float` | 两比特门耗时（秒），用于确定包含两比特门的拓扑层的最大空闲时间 |

---

### 初始化属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `t1g` | `float` | 单比特门耗时 |
| `t2g` | `float` | 两比特门耗时 |
| `_count` | `int` | 节点命名计数器，起始值 `86751`（避免与现有 DAG 节点冲突） |

---

## `run(...)` 方法

**签名：**

```python
def run(
    self,
    qc: QuantumCircuit,
    sequence: Literal["XY4", "CPMG"] = "XY4",
    align_right: bool = True,
    insert_before_barrier: bool = False,
) -> QuantumCircuit
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `qc` | `QuantumCircuit` | — | 输入量子线路 |
| `sequence` | `str` | `"XY4"` | DD 序列类型，支持 `"XY4"` 或 `"CPMG"` |
| `align_right` | `bool` | `True` | 右对齐模式：使用反向拓扑排序，使 DD 序列靠近线路末尾 |
| `insert_before_barrier` | `bool` | `False` | 为 `True` 时在 `barrier` 前也尝试插入 DD 序列 |

**异常：**
- `ValueError`：`sequence` 不是 `"XY4"` 或 `"CPMG"` 时抛出。

**返回值：** 插入 DD 序列后的新 `QuantumCircuit`。

---

## DD 序列详解

### XY4（默认，推荐）

- **序列长度**：4 个脉冲，最小耗时 = $4 \times t_{1g}$
- **偶数周期**（`idx % 2 == 0`）：`X → delay → Y → delay → X → delay → Y`
- **奇数周期**（`idx % 2 == 1`）：`Y → delay → X → delay → Y → delay → X`
- 交替 XY/YX 排列提供对 X 和 Y 方向噪声的对称抑制

### CPMG

- **序列长度**：2 个脉冲，最小耗时 = $2 \times t_{1g}$
- **所有周期**：`X → delay → X`
- 适用于 T₂ 退相干主导的场景

---

## 插入算法

```
输入 qc
  │
  ▼
① 构建 DAG（qc2dag, show_qubits=False）
  │
  ▼
② 计算拓扑代排序
   ├─ align_right=True  → 反向 DAG 的拓扑代，再反转列表
   └─ align_right=False → 正向 DAG 的拓扑代
  │
  ▼
③ 逐层扫描：
   for each 拓扑层的 nodes:
     max_idle_time = _get_max_idle_time(nodes)
     for each qubit 涉及的 node:
       idle_time = 上一个门以来的累积空闲时间
       if idle_time ≥ t1g × sequence_length:
         计算 n_dd 和 tgap
         插入 DD 节点链到 DAG
       else:
         更新 idle 时间，不插入
     不在当前层出现的 qubit：idle_time += max_idle_time
  │
  ▼
④ dag2qc(dag) → 输出 qc_new
```

---

## DD 插入参数计算

给定空闲时间 `idle_time`：

$$n_{\text{dd}} = \left\lfloor \frac{\text{idle\_time}}{L_{\text{seq}} \times t_{1g}} \right\rfloor$$

其中 $L_{\text{seq}}$ 为序列长度（XY4 = 4，CPMG = 2）。

**间隔计算：**

```python
GRID_NS = 0.1  # 0.1 ns 时间网格
tgap_units = round((idle_time - n_dd * L_seq * t1g) / L_seq / n_dd / (GRID_NS * 1e-9))
tgap = tgap_units * GRID_NS * 1e-9
tgap_half = tgap / 2  # 首尾半间隔
```

- DD 序列在首尾使用 `tgap_half` 的 delay，中间使用 `tgap` 的 delay
- 时间量化到 0.1 ns 网格，匹配硬件脉冲精度

---

## 内部辅助方法

### `counter() -> int`

自增计数器，每次调用 `_count += 1` 并返回当前值。用于生成唯一的 DAG 节点名。

---

### `_get_max_idle_time(nodes: list) -> float`

根据拓扑层中的门类型确定最大空闲时间：

| 层内包含门类型 | 返回值 |
|---|---|
| 两比特门 | `self.t2g` |
| 仅单比特门 | `self.t1g` |
| 均无 | `0` |

两比特门优先（检查顺序：两比特门 → 单比特门）。

---

### `_update_idle_time(node, max_idle_time) -> float`

从 `max_idle_time` 中减去当前门的执行时间：

| 门类型 | 减去的时间 |
|---|---|
| 单比特门/参数单比特门 | `self.t1g` |
| 两比特门 | `self.t2g` |
| 其他 | 返回 `0` |

---

## 示例

```python
from fieldqkit.compile.schedule import DynamicalDecoupling
from fieldqkit.circuit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)  # 此时 qubit 2 空闲
qc.cx(1, 2)  # 此时 qubit 0 空闲
qc.h(2)

# 门耗时：单比特 25ns，两比特 50ns
dd = DynamicalDecoupling(t1g=25e-9, t2g=50e-9)

# XY4 序列，右对齐
qc_dd = dd.run(qc, sequence="XY4", align_right=True)
print(f"DD 前门数: {len(qc.gates)}")
print(f"DD 后门数: {len(qc_dd.gates)}")

# CPMG 序列，允许在 barrier 前插入
qc_dd2 = dd.run(qc, sequence="CPMG", insert_before_barrier=True)
```

---

## 相关页面

- [编译模块总览](./README.md)
- [Transpiler — 编译流水线](./transpiler.md)（DD 为最后一步 Pass）
- [DAG — 有向无环图工具](./dag.md)（DD 在 DAG 上操作）
