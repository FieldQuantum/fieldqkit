# QASM-to-QCIS 转换器

## 概览

- **模块**：`quantum_hw.circuit.qasm_to_qcis`
- **源文件**：`qasm_to_qcis.py`（约930 行）
- **作用**：将 OpenQASM 2.0/3.0 程序转换为 QCIS（Quantum Circuit Instruction Set）原生指令字符串
- **QCIS 原生门集**：`X2P`、`X2M`、`Y2P`、`Y2M`、`RZ`、`CZ`（以及 `I` 用于 delay）

---

## 核心类

### `Instruction`（dataclass）

表示一条 QCIS 指令。

```python
@dataclass
class Instruction:
    name: str                              # 指令名（小写，如 'x2p', 'rz', 'cz'）
    qubit_index: list[int] | int           # 作用 qubit 索引
    arguments: list[float] = field(...)    # 参数列表（如旋转角）
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 指令名称 |
| `qubit_index` | `list[int]` | 作用的 qubit 索引列表 |
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

**分解规则一览（29 个静态方法）**：

#### 单比特离散门

| 方法 | QASM 门 | QCIS 分解 |
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

| 方法 | QASM 门 | QCIS 分解 |
|------|---------|-----------|
| `rx(inp)` | Rx(θ) | Y2M · RZ(θ) · Y2P |
| `ry(inp)` | Ry(θ) | X2P · RZ(θ) · X2M |
| `rz(inp)` | Rz(θ) | RZ(θ)（直通） |
| `u(inp)` | U(θ,φ,λ) | RZ(φ) · X2P · RZ(θ) · X2M · RZ(λ) |
| `u1(inp)` | U1(λ) | RZ(λ) |
| `u2(inp)` | U2(φ,λ) | RZ(φ) · Y2P · RZ(λ) |
| `u3(inp)` | U3(θ,φ,λ) | RZ(λ) · X2P · RZ(θ) · X2M · RZ(φ) |

#### 延时

| 方法 | QASM 门 | QCIS 分解 |
|------|---------|-----------|
| `delay(inp)` | delay(t) | I(t) |

#### 双比特门

| 方法 | QASM 门 | QCIS 分解 |
|------|---------|-----------|
| `cx(inp)` | CX | Y2M(target) · CZ · Y2P(target) |
| `cz(inp)` | CZ | CZ（直通） |
| `cy(inp)` | CY | X2P(target) · CZ · X2M(target) |
| `ch(inp)` | CH | S · H · T · CX · T† · H · S†（均作用于 target） |
| `swap(inp)` | SWAP | CX · CX(反向) · CX |
| `crz(inp)` | CRz(θ) | RZ(θ/2, target) · CX · RZ(−θ/2, target) · CX |
| `cp(inp)` | CP(θ) | RZ(θ/2, control) · CRz(θ) |

#### 三比特门

| 方法 | QASM 门 | QCIS 分解 |
|------|---------|-----------|
| `ccx(inp)` | CCX (Toffoli) | 标准 H-CX-T-T† 分解（15 条原生指令） |
| `cu3(inp)` | CU3(θ,φ,λ) | RZ · CX · U · CX · U 序列 |

---

### `QasmToQcis`

OpenQASM → QCIS 的主转换器类。

#### `__init__(self, rule=None)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rule` | `class` | `NativeQcisRules` | 分解规则类，可替换为自定义规则 |

初始化内部状态：
- `instruct_convert_rule_dict`：从规则类中提取所有静态方法作为分解函数字典
- `qcis_str`：累积输出字符串
- `qubit_map`：qubit 寄存器到全局索引映射
- `var_map`：变量映射（预置 `pi`、`π`）

#### `convert_to_qcis(qasm: str) -> str`

将 OpenQASM 程序转换为 QCIS 指令字符串。

| 参数 | 类型 | 说明 |
|------|------|------|
| `qasm` | `str` | OpenQASM 2.0 或 3.0 程序字符串 |

**返回**：QCIS 指令字符串（每条指令一行）。

**内部流程**：

```
qasm → openqasm3.parse() → AST 遍历
  → 寄存器声明 → 更新 qubit_map
  → 门操作 → 查找分解规则 → 生成 Instruction 列表
  → 累积到 qcis_str
```

#### 内部方法

| 方法 | 说明 |
|------|------|
| `_parse_argument(argument, var_map)` | 解析 AST 参数节点为数值（支持字面量、标识符、表达式） |
| `_parse_qubit(qubit, qubit_map)` | 解析 AST qubit 引用为全局索引 |
| `_parse_ast_statement(statement, var_map, qubit_map)` | 分发处理各 AST 语句类型 |

支持的 AST 语句类型：
- `Include`：处理 `include "qelib1.inc"` / `stdgates.inc`
- `QubitDeclaration` / `QuantumDeclaration`：注册 qubit 映射
- `ClassicalDeclaration` / `QuantumPhase`：静默忽略
- `QuantumGate` / `QuantumGateDefinition`：门操作分解
- `QuantumMeasurementStatement` / `QuantumMeasurement`：测量指令（生成 `M` 指令）
- `QuantumBarrier`：生成 `B` 指令
- `DelayInstruction` / `QuantumDelay`：延时指令

---

## 辅助函数

### `_traversal_binary_tree(node, var_map)`

递归遍历 OpenQASM 3 AST 表达式树，求值为 Python 数值。支持：
- 字面量（`IntegerLiteral`、`FloatLiteral`、`ImaginaryLiteral`、`BooleanLiteral`、`DurationLiteral`）
- 标识符（通过 `var_map` 查找）
- 二元表达式（`+`/`-`/`*`/`/`/`%`/`**`/`&`/`|`/`^`/`<<`/`>>`）
- 一元表达式（`-`/`~`/`!`）

### `_duration_literal_to_seconds(duration_literal) -> float`

将 OpenQASM 3 的 `DurationLiteral` 节点转换为秒。支持单位：`s`、`ms`、`us`、`ns`、`ps`。

### `_meth_dispatch(func)`

方法级 single-dispatch 装饰器，按第二参数（`args[1]`）的类型分发。用于实现 `_parse_argument`、`_parse_qubit`、`_parse_ast_statement` 的多态处理。

---

## 常量

### `_unary_operator`

一元运算符字典：`{"-": operator.neg, "~": operator.invert, "!": operator.not_}`

### `_binary_operator_map`

二元运算符字典，映射 13 种 OpenQASM 3 运算符名到 Python `operator` 函数。

### `_include_file_path_map`

include 文件路径映射：`{"qelib1.inc": Path(...) / "include/qelib1.inc"}`。

---

## 示例

```python
from quantum_hw.circuit.qasm_to_qcis import QasmToQcis

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
"""

converter = QasmToQcis()
qcis_str = converter.convert_to_qcis(qasm)
print(qcis_str)
# Y2M Q0
# RZ Q0 3.141593
# Y2M Q1
# CZ Q0 Q1
# Y2P Q1
# M Q0
```

### 使用自定义规则

```python
from quantum_hw.circuit.qasm_to_qcis import QasmToQcis, NativeQcisRules, Instruction

class MyRules(NativeQcisRules):
    @staticmethod
    def h(inp):
        # 自定义 H 门分解
        return [Instruction("x2p", inp.qubit_index), Instruction("rz", inp.qubit_index, [NativeQcisRules.pi / 2])]

converter = QasmToQcis(rule=MyRules)
```

## 相关页面

- [OpenQASM 解析](./qasm.md)
- [QuantumCircuit](./quantumcircuit.md)
