# statevector simulator

## 模块

- `fieldqkit.sim.statevector`

## 概览

该模块是 Torch 实现的状态向量模拟器，并同时提供 VQE 自动微分所需的能量计算辅助函数。

在高层调用中，通常通过 `fieldqkit.sim.interface` 进行后端选择；当 qubit 数不超过阈值时才会路由到本模块。

核心能力：

- 线路前向演化：`simulate_statevector(...)`
- 采样计数：`simulate_counts(...)`
- 可微分能量评估：`expectation_pauli(...)`、`energy_and_expectations(...)`
- 样本概率计算：`sample_probabilities(...)`——用于无监督 QNN 的 NLL 损失

## 关键函数

### `simulate_statevector(qc, *, param_values=None, device=None) -> torch.Tensor`

- 从 `|0...0>` 初态按门序演化。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- 支持离散门与参数门。
- **不支持 `reset`**：模拟器遇到 `reset` 会抛出 `NotImplementedError`（reset 是非酉信道，纯态后端无法正确表示纠缠比特的 reset）。含 `reset` 的电路仍可构造并提交到真机。
- 返回一维态向量，形状 `(2**n,)`。

### `simulate_counts(qc, shots, *, seed=None, param_values=None, device=None) -> Dict[str, int]`

- 基于 `simulate_statevector` 的概率分布进行多项式采样。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- bitstring 输出采用大端序（qubit 0 对应字符串最左位）。

### `expectation_pauli(state, pauli, *, num_qubits) -> torch.Tensor`

- 计算 `⟨psi|P|psi⟩`，其中 `P` 是 Pauli string。
- 内部先将 Pauli 字符串展开到逐比特模式，再逐位施加局部 `X/Y/Z` 算符。
- 返回复标量（VQE 能量中通常取 `.real`）。

### `sample_probabilities(state, samples) -> torch.Tensor`

- 输入态向量 `state`（长度 $2^n$）和样本数组 `samples`（`(N, n_qubits)` 整数，元素 0/1，big-endian）。
- 返回 1-D 张量，长度 N，$P(b_i) = |\langle b_i|\psi\rangle|^2$。
- 完全可微分，支持 autograd 回传。
- 主要用途：无监督 QNN 的负对数似然（NLL）损失计算。

### `energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian, device=None)`

- 先调用 `build_param_values_from_tensor(...)` 得到参数绑定，再调用 `simulate_statevector(...)` 得到当前态。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- 遍历哈密顿量项，调用 `expectation_pauli(...)` 计算每个可观测量期望。
- 返回：
  - `energy`：可微分的 Torch 标量。
  - `expectations`：`Dict[str, float]`，用于日志与结果记录。

## 常见报错

- `ValueError("missing parameter value for ...")`
- `TypeError("unsupported parameter type: ...")`
- `ValueError("unsupported gate for simulator: ...")`
- `ValueError("params length must be ...")`

## 示例

```python
import torch
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.sim.statevector import (
    energy_and_expectations,
    simulate_counts,
    simulate_statevector,
)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

state = simulate_statevector(qc)
counts = simulate_counts(qc, shots=1024, seed=42)

symbolic = QuantumCircuit(1)
symbolic.rx("theta_0", 0)
params = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
energy, exps = energy_and_expectations(
    symbolic,
    params=params,
    param_names=["theta_0"],
    hamiltonian=[(1.0, "Z0")],
)
energy.backward()

print(state.shape)
print(counts)
print(float(energy.detach().cpu().item()), exps)
```

## 相关页面

- [matrix utilities](./matrix.md)
- [simulator common helpers](./common.md)
- [simulator interface](./interface.md)
- [mps simulator](./mps.md)
- [VQERunner.run_model](../algorithms/vqe_runner.md)
