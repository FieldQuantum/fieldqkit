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

- 常量：MPS_THRESHOLD_QUBITS = 16
- 规则：
  - num_qubits > 16: 使用 MPS 后端
  - 否则: 使用 statevector 后端

## 公开函数

### simulate_counts(qc, shots, *, seed=None, param_values=None, device=None)

- 返回 Dict[str, int]。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- bitstring 采用大端序（qubit 0 对应字符串最左位）。

### expectation_pauli(state, pauli, *, num_qubits)

- 按 num_qubits 选择后端对应实现。

### sample_probabilities(state, samples, *, num_qubits)

- 返回给定样本向量的概率 $P(b_i|\psi)$。
- 输入 `samples` 为 `(N, n_qubits)` 整数张量/数组（元素 0/1，big-endian）。
- 返回 1-D 张量，长度 N，支持自动微分。
- 按 num_qubits 阈值分派到 statevector 或 MPS 后端。
- 用于无监督 QNN 的 NLL 损失计算。

### energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian, device=None)

- 返回 (energy, expectations)。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- VQE 训练路径通常通过此函数统一进入仿真后端。

## 包级导出

quantum_hw.sim.__init__ 当前导出：

- simulate_counts
- expectation_pauli
- sample_probabilities
- energy_and_expectations
- simulate_statevector
- simulate_mps
- simulate_mpo_process
- auto_sim_device

## 相关页面

- statevector simulator（statevector.md）
- mps simulator（mps.md）
- mpo process simulator（mpo.md）
