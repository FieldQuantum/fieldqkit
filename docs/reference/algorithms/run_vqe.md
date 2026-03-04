# run_vqe

## 概览

- **模块**：`quantum_hw.algorithms.vqe`
- **作用**：基于参数移位梯度 + Adam 的变分优化流程。

## 支持模型

- `ising`
- `heisenberg`
- `xy`
- `xxz`
- `custom`（显式传入哈密顿量）

## 返回

- `VQEResult`
  - `best_energy`
  - `best_params`
  - `energy_history`

## 待补

- 梯度评估次数与 `layers / num_qubits` 的关系
- `target_qubits` 与并行梯度打包策略说明
