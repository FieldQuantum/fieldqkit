# statevector simulator

## 模块

- `quantum_hw.sim.statevector`

## 概览

该模块是 Torch 实现的状态向量模拟器，并同时提供 VQE 自动微分所需的能量计算辅助函数。

在高层调用中，通常通过 `quantum_hw.sim.interface` 进行后端选择；当 qubit 数不超过阈值时才会路由到本模块。

核心能力：

- 线路前向演化：`simulate_statevector(...)`
- 采样计数：`simulate_counts(...)`
- 可微分能量评估：`build_state_from_symbolic(...)`、`expectation_pauli(...)`、`energy_and_expectations(...)`

## 关键函数

### `simulate_statevector(qc, *, param_values=None) -> torch.Tensor`

- 从 `|0...0>` 初态按门序演化。
- 支持离散门、参数门和 `reset`。
- 返回一维态向量，形状 `(2**n,)`。

### `simulate_counts(qc, shots, *, seed=None, param_values=None) -> Dict[str, int]`

- 基于 `simulate_statevector` 的概率分布进行多项式采样。
- bitstring 输出采用小端序。

### `build_state_from_symbolic(symbolic_qc, *, params, param_names) -> torch.Tensor`

- 输入符号参数线路和可微 `params` 张量。
- 将 `param_names[i] -> params[i]` 映射为 `param_values`，再复用 `simulate_statevector(...)`。
- 这是 autograd 路径构建态向量的统一入口。

### `expectation_pauli(state, pauli, *, num_qubits) -> torch.Tensor`

- 计算 `⟨psi|P|psi⟩`，其中 `P` 是 Pauli string。
- 内部先将 Pauli 字符串展开到逐比特模式，再逐位施加局部 `X/Y/Z` 算符。
- 返回复标量（VQE 能量中通常取 `.real`）。

### `energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian)`

- 先调用 `build_state_from_symbolic(...)` 得到当前态。
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
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim.statevector import (
    build_state_from_symbolic,
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
