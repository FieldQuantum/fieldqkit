# simulator interface

## 模块

- quantum_hw.sim.interface

## 概览

该模块是仿真后端分发层，负责在 statevector 与 MPS 间路由：

- simulate_counts(...)
- expectation_pauli(...)
- energy_and_expectations(...)

MPO 过程模拟不在该分发层中自动路由，需要显式调用 quantum_hw.sim.mpo.simulate_mpo_process。

## 分发规则

- 常量：MPS_THRESHOLD_QUBITS = 12
- 规则：
  - num_qubits > 12: 使用 MPS 后端
  - 否则: 使用 statevector 后端

## 公开函数

### simulate_counts(qc, shots, *, seed=None, param_values=None)

- 返回 Dict[str, int]。
- bitstring 序与后端保持一致（当前均为小端序）。

### expectation_pauli(state, pauli, *, num_qubits)

- 按 num_qubits 选择后端对应实现。

### energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian)

- 返回 (energy, expectations)。
- VQE 训练路径通常通过此函数统一进入仿真后端。

## 包级导出

quantum_hw.sim.__init__ 当前导出：

- simulate_counts
- expectation_pauli
- energy_and_expectations
- simulate_statevector
- simulate_mps
- simulate_mpo_process

## 相关页面

- statevector simulator（statevector.md）
- mps simulator（mps.md）
- mpo process simulator（mpo.md）
