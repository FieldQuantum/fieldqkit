# OpenQASM 解析

## 概览

- **模块**：`fieldqkit.circuit.qasm2`
- **源文件**：`qasm2.py`（约410 行）
- **作用**：将 OpenQASM 2.0 程序解析为统一的 gate tuple IR
- **调用入口**：`QuantumCircuit.from_openqasm2`

> `QuantumCircuit` → QCIS 的转换见 [QCIS 原生指令](./qcis.md)。

---

## qasm2 模块

### 导出接口

```python
__all__ = [
    "parse_expression",
    "parse_openqasm2_regs",
    "parse_openqasm2_custom_gates",
    "parse_openqasm2_to_gates",
]
```

### `parse_openqasm2_to_gates(openqasm2_str: str)`

主入口。将完整 OpenQASM 2.0 字符串解析为 gate tuple IR。

| 参数 | 类型 | 说明 |
|------|------|------|
| `openqasm2_str` | `str` | 完整 OpenQASM 2.0 程序 |

**返回**：`(new, qubit_used, cbit_used)`

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `new` | `list` | 解析后的语句行列表（gate tuple IR） |
| `qubit_used` | `set[int]` | 使用的 qubit 索引集合 |
| `cbit_used` | `set[int]` | 使用的 cbit 索引集合 |

**解析能力**：

- `qreg`/`creg` 声明解析与全局索引映射
- 所有内置门（离散门、参数门、三比特门）
- 噪声信道门（`depolarize1`/`depolarize2`/`x_error`/`y_error`/`z_error`/`amplitude_damping`/`phase_damping`），以 `opaque` 声明承载
- `measure` / `barrier` / `reset`
- 自定义 `gate` / `opaque` 定义的内联展开

**不支持**：

- 经典条件门 `if (c == val) ...`：解析器遇到此类语句会抛出 `ValueError`。

### `parse_openqasm2_regs(openqasm2_str: str)`

解析 `qreg` 和 `creg` 声明，将多寄存器定义提取并从源码中移除。

**返回**：`(qregs, cregs, new_qasm)`

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `qregs` | `list[tuple[str, int]]` | 量子寄存器名与大小 |
| `cregs` | `list[tuple[str, int]]` | 经典寄存器名与大小 |
| `new_qasm` | `str` | 移除寄存器声明后的 QASM |

### `parse_openqasm2_custom_gates(openqasm2_str: str)`

解析并缓存自定义门定义（`gate ... { ... }`），在主解析中内联展开。

### 辅助函数

#### `_record_qubits(qubit_used: list, *qubits: int) -> None`

将 qubit 索引追加到已使用列表。

#### `generate_reg_map(regs)`

将多个寄存器按顺序平铺为全局索引映射字典。例如 `[('q', 3), ('r', 2)]` → `{('q', 0): 0, ('q', 1): 1, ('q', 2): 2, ('r', 0): 3, ('r', 1): 4}`。

#### `sparse_gate_params_qregs(line: str)`

将一行门指令拆解为 `(gate_name, params_str, qregs_str)` 三元组。

#### `get_positions_list(gate, qregs_str, qreg_map, creg_map)`

根据门名和寄存器引用字符串，查找全局 qubit/cbit 索引列表。

---

## 与 QuantumCircuit 的关系

| QuantumCircuit 方法 | 调用链 |
|---------------------|--------|
| `from_openqasm2(qasm)` | 校验 `OPENQASM 2.0` 头 → `parse_openqasm2_to_gates` → 回填属性 |
| `to_openqasm2(symbolic=False)` | 将 gate tuple IR 导出为 QASM 2.0 字符串；`symbolic=True` 保留字符串参数 |

## 行为说明

- 解析过程默认静默，不输出调试信息
- 输出门序列采用统一 tuple IR，便于导出、绘图和编译复用
- 参数表达式求值使用 `parse_expression`（来自 `quantumcircuit_helpers`，安全 AST 求值）
- 噪声信道在 `to_openqasm2` 中先以 `opaque` 声明，再按门发射；单比特形如 `opaque depolarize1(p) q;` 与 `depolarize1(0.1) q[0];`，双比特形如 `opaque depolarize2(p) q0,q1;` 与 `depolarize2(0.05) q[0],q[1];`。阻尼信道（`amplitude_damping`/`phase_damping`）的形参名用 `gamma`，其余用 `p`。这些门同样可经 `from_openqasm2` 往返还原。

## 示例

```python
from fieldqkit.circuit.qasm2 import parse_openqasm2_to_gates

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rx(pi/4) q[0];
cx q[0],q[1];
measure q[0] -> c[0];
"""

new, qubit_used, cbit_used = parse_openqasm2_to_gates(qasm)
print(new)
```

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [Helpers 与渲染](./helpers_render.md)
- [QCIS 原生指令](./qcis.md)
