# OpenQASM 解析

## 概览

- 模块：`quantum_hw.circuit.qasm2`、`quantum_hw.circuit.qasm3`
- 作用：将 OpenQASM 2/3 程序解析为统一的 gate tuple IR
- 调用入口：`QuantumCircuit.from_openqasm2`、`QuantumCircuit.from_openqasm3`

## qasm2 主要接口

### `parse_openqasm2_to_gates(openqasm2_str)`

- 输入：完整 OpenQASM2 字符串
- 输出：`(new, qubits, cbits, gates, params_value)`
- 能力：
  - `qreg/creg` 声明解析
  - 内置门与参数门解析
  - `measure` / `barrier` / `reset`
  - `if (...)` 条件前缀附着
  - 自定义 `gate` / `opaque` 内联展开

### `parse_openqasm2_regs(regs)`

- 将多寄存器定义平铺为全局索引映射

### `parse_openqasm2_custom_gates(gates)`

- 解析并缓存自定义门定义

### `parse_expression(expr)`

- 解析参数表达式，支持 `pi` 与常见数学函数

## qasm3 主要接口

### `parse_openqasm3_to_gates(openqasm3_str)`

- 将 OpenQASM3 程序解析为与 qasm2 一致的 IR 结构

### `convert_qasm_pi_to_decimal(qasm)`

- 统一 `pi` 表达式形式，便于后续求值

## 与 QuantumCircuit 的关系

- `from_openqasm2`：校验头部后调用 `parse_openqasm2_to_gates`
- `from_openqasm3`：校验头部后调用 `parse_openqasm3_to_gates`
- 两者都会回填 `nqubits/ncbits/qubits/gates/params_value`

## 行为说明

- 解析过程默认静默，不输出调试信息
- 输出门序列采用统一 tuple IR，便于导出、绘图和编译复用

## 示例

```python
from quantum_hw.circuit.qasm2 import parse_openqasm2_to_gates

qasm = """
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
rx(pi/4) q[0];
cx q[0],q[1];
measure q[0] -> c[0];
"""

new, qubits, cbits, gates, params_value = parse_openqasm2_to_gates(qasm)
print(gates)
```

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [Helpers 与渲染](./helpers_render.md)
