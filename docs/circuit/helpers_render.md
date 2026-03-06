# Helpers 与渲染

## 概览

- 模块：`quantum_hw.circuit.quantumcircuit_helpers`、`quantum_hw.circuit.render`
- 作用：
  - 管理门元信息与参数解析
  - 提供 gate tuple 到文本图的转换
  - 提供完整/简洁两种线路渲染能力

## quantumcircuit_helpers 主要接口

### 门集合常量

- `one_qubit_gates_available`
- `two_qubit_gates_available`
- `three_qubit_gates_available`
- `one_qubit_parameter_gates_available`
- `two_qubit_parameter_gates_available`

这些集合用于门名校验、门类型识别和渲染布局。

### 参数处理

### `handle_expression(exp, para_dict)`

- 将字符串表达式按参数字典求值

### `parse_gate_params(gate, para_dict)`

- 解析单个 gate tuple 的参数并返回标准化结果

### `judge_phase_num(params)`

- 判断参数数量并辅助构造 gate tuple

### 绘图相关

### `gate_map(gates, one_qubit_gate_map, two_qubit_gate_map)`

- 按 qubit 构造门层依赖映射

### `format_gates_layerd(*layers)`

- 将层化结构整理为统一渲染输入

### `convert_gate_info_to_drawing_format(gate, qbit_used)`

- 将 gate tuple 转换为绘图 token

### `draw_circuit(...)`

- 输出完整 ASCII 线路图

## render 主要接口

### `draw_circuit_simply(gates, qubits, width=4)`

- 只渲染活跃 qubit，输出更紧凑

### `print_simple_quantumcircuit(circuit, width=4)`

- 面向 `QuantumCircuit` 的简洁绘图包装函数

## 参数显示约定

- 整数按整数文本显示
- 接近整数的浮点会归一为整数展示
- 其余浮点保持标准字符串表示
- 未绑定参数保持占位符名称（如 `theta`）

## 示例

```python
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0).cx(0, 1).rzz("phi", 1, 2)
qc.apply_value({"phi": 0.5}, deep=True)

print(qc.draw(width=5))
print(qc.draw_simply(width=5))
```

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [OpenQASM 解析](./qasm.md)
