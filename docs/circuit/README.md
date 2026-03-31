# Circuit

## 概览

- **模块路径**：`quantum_hw.circuit`
- **模块定位**：量子线路表示与转换能力
- **源文件**：`quantumcircuit.py`（1364 行）、`quantumcircuit_helpers.py`（580 行）、`matrix.py`（370 行）、`utils.py`（245 行）、`qasm2.py`（307 行）、`qasm3.py`（353 行）、`qasm_to_qcis.py`（426 行）、`render.py`（25 行）
- **主要能力**：
  - `QuantumCircuit` 线路构建与编辑
  - OpenQASM 2/3 导入解析与导出
  - OpenQASM → QCIS 原生指令转换
  - 文本线路渲染（完整/简洁）
  - 门矩阵与分解工具（u3/zyz/kak）

## 页面导航

- [QuantumCircuit](./quantumcircuit.md) — 核心线路类、门操作 API、参数绑定、分析与可视化
- [OpenQASM 解析](./qasm.md) — OpenQASM 2/3 解析器
- [QASM-to-QCIS 转换器](./qasm_to_qcis.md) — QASM 到 QCIS 原生指令的转换
- [Helpers 与渲染](./helpers_render.md) — 门集合常量、参数格式化、DAG 转换、ASCII 渲染管线
- [matrix 与 utils](./matrix_utils.md) — 门矩阵表示、分解算法、幺正等价判定

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
