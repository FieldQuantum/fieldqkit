# simulator common helpers

## 模块

- quantum_hw.sim.common

## 概览

该模块提供各仿真后端共享的基础工具：

- 参数解析与符号表达式求值
- 门矩阵物化
- Pauli 基元矩阵
- 参数张量到名字映射

主要被 statevector/mps/mpo 复用。

## 关键函数

### resolve_param(qc, param, param_values=None)

- 支持：
  - 数值标量（float/int）
  - 参数名字符串
  - 参数表达式字符串（通过 qc._eval_param_expression）
- 内置符号：pi
- 解析优先级（字符串参数）：
  - 显式 param_values
  - qc.params_value
  - 表达式解析（symbol_resolver）

### materialize_gate_matrix(gate, params, *, dtype, device)

- 从 matrix.gate_matrix_dict 取门定义。
- 固定门：直接 to(dtype/device)
- 参数门：调用函数生成矩阵

### single_pauli(op, *, dtype, device)

- 返回 X/Y/Z 的 2x2 复矩阵。
- 不支持 I（通常由调用侧跳过）。

### build_param_values_from_tensor(*, params, param_names)

- 把参数张量展平后映射为 name -> value。
- 长度不匹配时抛 ValueError。

## 常见报错

- missing parameter value for ...
- unsupported parameter type
- unsupported Pauli
- params length must be ...

## 相关页面

- matrix utilities（matrix.md）
- statevector simulator（statevector.md）
- mps simulator（mps.md）
- mpo process simulator（mpo.md）
