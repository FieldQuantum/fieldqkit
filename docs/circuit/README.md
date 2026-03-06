# Circuit

## 概览

- 模块路径：`quantum_hw.circuit`
- 模块定位：量子线路表示与转换能力
- 主要能力：
  - `QuantumCircuit` 线路构建与编辑
  - OpenQASM 2/3 导入解析与导出
  - 文本线路渲染（完整/简洁）
  - 门矩阵与分解工具（u3/zyz/kak）

## 页面导航

- [QuantumCircuit](./quantumcircuit.md)
- [OpenQASM 解析](./qasm.md)
- [Helpers 与渲染](./helpers_render.md)
- [matrix 与 utils](./matrix_utils.md)

## 数据模型约定

`QuantumCircuit.gates` 使用 tuple IR：

- 单比特离散门：`(gate, q)`
- 双比特离散门：`(gate, q0, q1)`
- 三比特门：`(gate, q0, q1, q2)`
- 单比特参数门：`(gate, theta, q)` / `("r", theta, phi, q)` / `("u", theta, phi, lam, q)`
- 双比特参数门：`(gate, theta, q0, q1)`
- 功能门：`delay` / `barrier` / `measure` / `reset`

参数支持占位符字符串，后续通过 `apply_value(...)` 绑定。

## 相关页面

- [circuits builders](../core/circuits.md)
- [statevector simulator](../sim/statevector.md)
- [matrix utilities](../sim/matrix.md)
