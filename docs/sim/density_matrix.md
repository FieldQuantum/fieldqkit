# density matrix simulator

## 模块

- `quantum_hw.sim.density_matrix`

## 概览

该模块是 Torch 实现的密度矩阵模拟器，用于含噪线路（噪声信道）的前向演化与可观测量计算。它与 statevector / MPS 后端保持同构接口，并被 `quantum_hw.sim.interface` 自动选用：当线路包含噪声信道（`has_noise_channels(qc)` 为真）时，分发层会路由到本模块，而非按 qubit 数阈值选择 statevector / MPS。

核心能力：

- 含噪线路前向演化：`simulate_density_matrix(...)`，返回密度矩阵 $\rho$
- 采样计数：`simulate_noisy_counts(...)`，从 $\rho$ 对角线采样
- Pauli 期望：`expectation_pauli_dm(...)`，计算 $\mathrm{tr}(P\rho)$
- 样本概率：`sample_probabilities_dm(...)`
- 可微分能量评估：`energy_and_expectations(...)`

噪声信道的 Kraus 算符由 [noise_kraus](./noise_kraus.md) 模块提供。

## 表示约定

- 密度矩阵以 `(2,)*2n` 张量内部存储：前 n 个轴为「行」指标，后 n 个轴为「列」指标；矩阵形态为 `reshape(2**n, 2**n)`。
- 初态为 $|0\dots0\rangle\langle0\dots0|$。
- 默认 dtype：`torch.complex64`。
- 酉门施加为 $\rho' = U\rho U^\dagger$；噪声信道施加为 $\rho' = \sum_k K_k\rho K_k^\dagger$；`reset` 将目标比特的 $|1\rangle\langle1|$ 块并入 $|0\rangle\langle0|$ 并丢弃相关相干项。

## 关键函数

### `simulate_density_matrix(qc, *, param_values=None, device=None) -> torch.Tensor`

- 从 $|0\rangle\langle0|$ 按门序演化，处理酉门、参数门、三比特门、噪声信道与 `reset`。
- `barrier` / `delay` / `measure` 在该层不改变 $\rho$。
- `param_values`：参数名到值的映射；保留 autograd，对可微变分参数可回传梯度。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- 返回形状 `(2**n, 2**n)` 的密度矩阵张量。
- 噪声率必须是 [0,1] 的具体数值（不可微，不接受符号参数）。

### `simulate_noisy_counts(qc, shots, *, seed=None, param_values=None, device=None) -> Dict[str, int]`

- 先调用 `simulate_density_matrix`，再对 $\rho$ 对角线（截断至非负并归一化）做多项式采样。
- bitstring 采用大端序（qubit 0 对应字符串最左位，与其他后端一致）。
- `seed=None` 时采样非确定性（每次调用结果可不同）；传入整数种子可复现。

### `expectation_pauli_dm(state, pauli, *, num_qubits) -> torch.Tensor`

- 计算 $\mathrm{tr}(P\rho)$，其中 `P` 为 Pauli string（支持 `'XZI'` 紧凑式与 `'Z0Z1'` 索引式）。
- `state` 为密度矩阵张量（`(2**n, 2**n)` 或 `(2,)*2n`）。
- 返回实标量（Hermitian Pauli 的期望为实）。

### `sample_probabilities_dm(state, samples) -> torch.Tensor`

- 对每个计算基态 $|i\rangle$ 计算 $P(i) = \langle i|\rho|i\rangle$（取 $\rho$ 对角元的实部）。
- `samples`：`(N, n_qubits)` 整数张量/数组（元素 0/1，big-endian）。
- 返回 1-D 张量，长度 N。

### `energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian, device=None)`

- 将 `params` + `param_names` 映射为 `param_values`，调用 `simulate_density_matrix(...)` 得到 $\rho$。
- 遍历哈密顿量项，调用 `expectation_pauli_dm(...)` 累加能量。
- 返回：
  - `energy`：可微分 Torch 标量。
  - `expectations`：`Dict[str, float]`，用于日志与结果记录。
- 含噪 VQE 训练路径通过 `interface.energy_and_expectations` 自动进入此函数（导出名为 `energy_and_expectations_dm`）。

## 常见报错

- `ValueError("Unsupported gate for DM simulator: ...")`
- `ValueError("Unknown noise channel: ...")`（来自 `noise_kraus.get_kraus_ops`）

## 示例

```python
import torch
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim import (
    simulate_density_matrix,
    simulate_noisy_counts,
    expectation_pauli_dm,
)

qc = QuantumCircuit(2)
qc.h(0).cx(0, 1).depolarize1(0.1, 0)

rho = simulate_density_matrix(qc)          # (4, 4) 密度矩阵
print(torch.trace(rho).real)               # ≈ 1.0
print(expectation_pauli_dm(rho, "ZZ", num_qubits=2))
print(simulate_noisy_counts(qc, shots=1000, seed=42))
```

## 相关页面

- [noise kraus operators](./noise_kraus.md)
- [simulator interface](./interface.md)
- [simulator common helpers](./common.md)
- [statevector simulator](./statevector.md)
