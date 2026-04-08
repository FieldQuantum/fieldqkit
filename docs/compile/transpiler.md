# Transpiler

## 概览

- **模块**：`quantum_hw.compile.transpiler`（约170 行）
- **作用**：编译流水线管理器，按顺序执行各编译 Pass，将逻辑线路转换为物理芯片可执行的门序列。
- **依赖**：`Backend`、`Layout`、`SabreRouting`、`TranslateToBasisGates`、`GateCompressor`、`DynamicalDecoupling`、`ThreeQubitGateDecompose`、`split_qubits`
- **继承**：无（独立类，非 `TranspilerPass` 子类）

---

## 类签名

```python
class Transpiler:
    def __init__(self, chip_backend: Backend | None = None, *, convert_single_qubit_gate_to_u: bool | None = None)
```

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `chip_backend` | `Backend \| None` | 硬件后端对象；提供芯片拓扑和校准信息。`None` 时退化为简单线性拓扑（仅供模拟器使用）。 |
| `convert_single_qubit_gate_to_u` | `bool \| None` | 是否将单比特门转换为 U 门；`None` 时自动推断（真机 `True`，无 backend `False`）。显式传 `False` 可保留原始门名（如 Tencent 平台需要）。 |

**属性（初始化后可访问）：**

| 属性 | 类型 | 说明 |
|---|---|---|
| `chip_backend` | `Backend \| None` | 保存的后端引用 |
| `two_qubit_gate_basis` | `str` | 在 `run()` 中设置；取自 `chip_backend.two_qubit_gate_basis`（真机）或 `"cx"`（无 backend） |
| `convert_single_qubit_gate_to_u` | `bool` | 在 `run()` 中设置；真机为 `True`，无 backend 为 `False` |

---

## `run(...)` 方法

**签名：**

```python
def run(
    self,
    qc: QuantumCircuit,
    target_qubits: list | None = None,
    niter: int = 5,
    use_dd: bool = True,
    use_three_qubit_decompose: bool = True,
    use_sabre_routing: bool = True,
    use_translate_to_basis: bool = True,
    use_gate_compressor: bool = True,
    routing_initial_mapping: str = "random",
    routing_random_choice: bool = True,
    noise_aware: bool | None = None,
    routing_n_trials: int = 1,
) -> QuantumCircuit
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `qc` | `QuantumCircuit` | — | 输入线路。必须为 `QuantumCircuit` 类型，否则抛出 `TypeError`。 |
| `target_qubits` | `list \| None` | `None` | 手动指定物理比特列表；`None` 时由 Layout 自动选择。非空时会校验数量和连通性。 |
| `niter` | `int` | `5` | SABRE 路由前后向迭代次数。越多结果越稳定，但耗时更长。 |
| `use_dd` | `bool` | `True` | 是否启用动力学去耦。需要 `chip_backend.chip_info` 中有 `one_qubit_gate_length` 和 `two_qubit_gate_length`，否则静默跳过。 |
| `use_three_qubit_decompose` | `bool` | `True` | 是否将 CCX/CCZ/CSWAP 分解为单/两比特门组合。 |
| `use_sabre_routing` | `bool` | `True` | 是否启用 SABRE 路由。关闭时不插入 SWAP。 |
| `use_translate_to_basis` | `bool` | `True` | 是否将所有门翻译到芯片本征门集。 |
| `use_gate_compressor` | `bool` | `True` | 是否启用门压缩（对易重排 + 单比特合并 + 两比特对消 + DAG 压缩）。 |
| `routing_initial_mapping` | `str` | `"random"` | 初始映射策略：`"trivial"`（按顺序映射）或 `"random"`（随机映射）。注意：当线路可沿比特分割（`split_qubits` 返回多组）时，`"random"` 会自动降级为 `"trivial"` 并发出 `logger.warning`。 |
| `routing_random_choice` | `bool` | `True` | SABRE 选择 SWAP 时是否随机（而非确定性贪心选最优）。 |
| `noise_aware` | `bool \| None` | `None` | 路由时使用 $-\log(f)$ 保真度加权距离；`None` 时自动推断——有真机 backend 时启用，否则关闭。 |
| `routing_n_trials` | `int` | `1` | SABRE 多随机初始映射试验次数。>1 时，第一次用指定策略，后续用 `"random"`，取 SWAP 数最少的结果。 |

**返回值：** 编译后的 `QuantumCircuit`。

**异常：**
- `TypeError`：`qc` 类型不是 `QuantumCircuit`。

---

### 返回值附带属性

编译后的 `QuantumCircuit` 会继承路由产生的映射信息：

| 属性 | 类型 | 说明 |
|---|---|---|
| `logical_to_physical` | `Dict[int, int]` | 逻辑比特 → 物理比特映射（由 `SabreRouting` 写入） |
| `physical_to_logical` | `Dict[int, int]` | 物理比特 → 逻辑比特映射 |

这些属性在各 Pass 之间通过 `deepcopy` 传递，确保 API 层可用于测量比特对齐。

---

## Pass 执行顺序

```
输入 QuantumCircuit
  │
  ▼
[1] ThreeQubitGateDecompose        CCX/CCZ/CSWAP → 单/两比特门组合
  │
  ▼
[2] Layout.select_layout()         选择物理比特子图（线路感知 + 保真度优先）
  │
  ▼
[3] SabreRouting                   SWAP 插入（噪声感知 + 多试验模式）
  │                                → 写入 logical_to_physical / physical_to_logical
  ▼
[4] TranslateToBasisGates          所有门 → {U, CZ} 本征门集
  │
  ▼
[5] GateCompressor                 对易重排 → 单比特合并 → 两比特对消 → DAG 压缩
  │
  ▼
[6] DynamicalDecoupling            在空闲时隙插入 DD 序列（XY4/CPMG）
  │
  ▼
输出 QuantumCircuit（附带映射属性）
```

**注意：** Layout 选择集成在 `run()` 方法内部（在 `SabreRouting` 之前调用），不是独立的 Pass 对象。

---

## 内部 Layout 选择逻辑

在 `run()` 方法中，Layout 选择发生在 Pass 链构建之前：

```python
subgraph = Layout(self.chip_backend).select_layout(
    qc,
    target_qubits=target_qubits,
    use_chip_priority=True,
    select_criteria={"key": "fidelity_var", "topology": "linear"},
)
```

- 固定使用 `fidelity_var`（保真度方差最小）+ `linear`（线性拓扑优先）策略
- 返回的 `subgraph` 传入 `SabreRouting` 作为物理拓扑约束
- 当 `target_qubits` 非空时，由 `Layout.select_layout` 校验连通性

---

## 无 Backend 退化行为

当 `chip_backend=None` 时进入模拟器模式：

| 行为 | 说明 |
|---|---|
| 拓扑 | 按逻辑比特顺序构建线性链：`0-1-2-...-N` |
| `two_qubit_gate_basis` | `"cx"` |
| `convert_single_qubit_gate_to_u` | `False`（保留原始门名） |
| `noise_aware` | 默认 `False` |
| DD | 静默跳过（无 `chip_info` 中的门长信息） |
| `graph["normal_order"]` | 设为 `qc.qubits` |

---

## DD Pass 的容错处理

DD 的初始化需要从 `chip_backend.chip_info["global_info"]` 中读取：
- `one_qubit_gate_length`：单比特门耗时
- `two_qubit_gate_length`：两比特门耗时

若读取失败（`chip_info` 不含这些字段，或 `chip_backend=None`），DD Pass 被 **静默跳过**（`try/except` 捕获）。

---

## 示例

### 基础用法

```python
from quantum_hw.api.backend import Backend
from quantum_hw.compile.transpiler import Transpiler
from quantum_hw.circuit import QuantumCircuit

backend = Backend("Baihua")
transpiler = Transpiler(backend)

qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

compiled = transpiler.run(qc)
print(f"编译后门数: {len(compiled.gates)}")
print(f"逻辑→物理映射: {compiled.logical_to_physical}")
```

### 噪声感知 + 多试验

```python
compiled = transpiler.run(
    qc,
    noise_aware=True,
    routing_n_trials=8,
    niter=7,
)
print(f"SWAP 路径偏好高保真耦合器")
```

### 指定物理比特

```python
compiled = transpiler.run(qc, target_qubits=[10, 11, 12, 13])
```

### 模拟器模式

```python
transpiler = Transpiler(chip_backend=None)
compiled = transpiler.run(qc, use_dd=False)
```

### 选择性禁用 Pass

```python
compiled = transpiler.run(
    qc,
    use_three_qubit_decompose=False,   # 已无三比特门
    use_dd=False,                       # 不需要 DD
    use_gate_compressor=False,          # 跳过压缩
)
```

---

## 相关页面

- [编译模块总览](./README.md)
- [Layout — 比特布局选择](./layout.md)
- [SabreRouting — SABRE 路由](./routing.md)
- [TranslateToBasisGates — 本征门翻译](./translate.md)
- [GateCompressor — 门压缩优化](./optimize.md)
- [DynamicalDecoupling — 动力学去耦](./schedule.md)
- [ThreeQubitGateDecompose — 门分解](./decompose.md)
- [DAG — 有向无环图工具](./dag.md)
