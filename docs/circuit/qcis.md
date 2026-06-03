# QCIS 原生指令

## 概览

- **模块**：`fieldqkit.circuit.qcis`
- **源文件**：`qcis.py`
- **作用**：提供 QCIS 原生指令原语，以及将 `QuantumCircuit` 直接转换为 QCIS 指令字符串的能力
- **QCIS 原生门集**：`X2P`、`X2M`、`Y2P`、`Y2M`、`RZ`、`CZ`（以及 `I` 用于 delay）

> 注意：本模块直接读取 `QuantumCircuit.gates`（tuple IR），不经过 QASM 文本中转。

---

## 核心类

### `Instruction`（dataclass）

表示一条 QCIS 指令。

```python
@dataclass
class Instruction:
    name: str                                          # 指令名（小写，如 'x2p', 'rz', 'cz'）
    qubit_index: List[int]                             # 作用 qubit 索引列表
    arguments: Optional[List[Union[int, float]]] = None  # 参数列表（如旋转角），默认 None
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 指令名称 |
| `qubit_index` | `List[int]` | 作用的 qubit 索引列表 |
| `arguments` | `Optional[List[Union[int, float]]]` | 指令参数（角度、delay 时长等），默认 `None` |

`__str__` 方法生成标准 QCIS 格式字符串，例如：

- `Instruction("x2p", [0])` → `"X2P Q0"`
- `Instruction("rz", [1], [1.5708])` → `"RZ Q1 1.5708"`
- `Instruction("cz", [0, 1])` → `"CZ Q0 Q1"`

---

### `NativeQcisRules`

将标准量子门分解为 QCIS 原生门序列的规则集。每个方法为 `@staticmethod`，输入一条 `Instruction`，返回 `list[Instruction]`。

**类常量**：

| 常量 | 值 | 说明 |
|------|-----|------|
| `pi` | `round(math.pi, 6)` | 使用 6 位精度的 π |
| `i_duration` | `60` | 默认 identity delay 时长 |

**分解规则一览**：

#### 单比特离散门

| 方法 | 门 | QCIS 分解 |
|------|---------|-----------|
| `x(inp)` | X | X2P · X2P |
| `y(inp)` | Y | Y2P · Y2P |
| `z(inp)` | Z | RZ(π) |
| `h(inp)` | H | Y2M · RZ(π) |
| `sx(inp)` | √X | X2P |
| `sxdg(inp)` | √X† | X2M |
| `s(inp)` | S | RZ(π/2) |
| `sdg(inp)` | S† | RZ(−π/2) |
| `t(inp)` | T | RZ(π/4) |
| `tdg(inp)` | T† | RZ(−π/4) |
| `id(inp)` | I | I(60) |

#### 单比特参数门

| 方法 | 门 | QCIS 分解 |
|------|---------|-----------|
| `rx(inp)` | Rx(θ) | Y2M · RZ(θ) · Y2P |
| `ry(inp)` | Ry(θ) | X2P · RZ(θ) · X2M |
| `rz(inp)` | Rz(θ) | RZ(θ)（直通） |
| `u(inp)` | U(θ,φ,λ) | RZ(λ) · X2P · RZ(θ) · X2M · RZ(φ) |

#### 延时

`delay` 门由 `circuit_to_qcis` 直接处理，不经过 `NativeQcisRules`。`QuantumCircuit.delay()` 以秒存储时长；`circuit_to_qcis` 自动转换为纳秒后写入 QCIS `I` 指令。

#### 双比特门

| 方法 | 门 | QCIS 分解 |
|------|---------|-----------|
| `cx(inp)` | CX | Y2M(target) · CZ · Y2P(target) |
| `cz(inp)` | CZ | CZ（直通） |
| `cy(inp)` | CY | X2P(target) · CZ · X2M(target) |
| `swap(inp)` | SWAP | CX · CX(反向) · CX |

#### 三比特门

| 方法 | 门 | QCIS 分解 |
|------|---------|-----------|
| `ccx(inp)` | CCX (Toffoli) | 标准 H-CX-T-T† 分解（15 条原生指令） |

---

## 主转换函数

### `circuit_to_qcis(qc: QuantumCircuit) -> str`

将 `QuantumCircuit` 直接转换为 QCIS 指令字符串，无需经过 QASM 文本中转。

| 参数 | 类型 | 说明 |
|------|------|------|
| `qc` | `QuantumCircuit` | 待转换线路。应已完成 transpile（仅含 basis 门）。 |

**返回**：QCIS 指令字符串（每条指令一行）。

**异常**：
- `NotImplementedError`：线路含 `NativeQcisRules` 未覆盖的门。

**功能门处理**：

| gate tuple | QCIS 输出 |
|-----------|-----------|
| `('measure', [q...], [c...])` | 每个 qubit 生成 `M Q<i>` |
| `('reset', q)` | `RST Q<q>` |
| `('barrier', qubits)` | `B Q<q0> Q<q1> ...` |
| `('delay', t, qubits)` | 每个 qubit 生成 `I Q<i> <t>` |

---

## 示例

### 基本用法

```python
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.circuit.qcis import circuit_to_qcis

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print(circuit_to_qcis(qc))
# Y2M Q0
# RZ Q0 3.141593
# Y2M Q1
# CZ Q0 Q1
# Y2P Q1
# M Q0
# M Q1
```

### 使用自定义分解规则

```python
from fieldqkit.circuit.qcis import NativeQcisRules, Instruction, circuit_to_qcis

class MyRules(NativeQcisRules):
    @staticmethod
    def h(inp):
        # 自定义 H 门分解
        return [Instruction("x2p", inp.qubit_index), Instruction("rz", inp.qubit_index, [NativeQcisRules.pi / 2])]
```

---

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [OpenQASM 解析](./qasm.md)
