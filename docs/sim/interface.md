# simulator interface

## 模块

- quantum_hw.sim.interface

## 概览

该模块是仿真后端分发层，负责在 statevector、MPS 与密度矩阵（DM）后端间路由：

- simulate_counts(...)
- expectation_pauli(...)
- sample_probabilities(...)
- energy_and_expectations(...)
- build_state_from_symbolic(...)

MPO 过程模拟不在该分发层中自动路由，需要显式调用 quantum_hw.sim.mpo.simulate_mpo_process。

## 分发规则

- 常量：MPS_THRESHOLD_QUBITS = 16
- 规则（按优先级）：
  - 线路含噪声信道（`has_noise_channels(qc)` 为真）: 使用密度矩阵后端（[density_matrix.md](./density_matrix.md)），优先于 qubit 数阈值。
  - 否则 num_qubits > 16: 使用 MPS 后端。
  - 否则: 使用 statevector 后端。
- 对 `expectation_pauli` / `sample_probabilities`，状态对象的类型决定后端：1D 张量为 statevector，list 为 MPS，2D 张量（`state.dim()==2`）为密度矩阵。

## 公开函数

### get_sim_config() -> dict

返回当前模拟器配置，包含 `'mps_threshold_qubits'` 和 `'max_bond_dim'`。

### set_sim_config(*, mps_threshold_qubits=None, max_bond_dim=...)

运行时修改模拟器超参。

- `mps_threshold_qubits`：超过此值时使用 MPS，`None` 不变。
- `max_bond_dim`：MPS 最大键维，`None` 不截断，`...`（默认）不变。

```python
from quantum_hw.sim import set_sim_config
set_sim_config(mps_threshold_qubits=20, max_bond_dim=512)
```

### build_state_from_symbolic(symbolic_qc, *, params, param_names, max_bond_dim=MAX_BOND_DIM, device=None)

- 从符号线路和可微 `params` 张量构建模拟器状态：含噪线路返回密度矩阵（2D tensor），否则按 qubit 数阈值分派到 statevector（flat tensor）或 MPS（site tensor list）。
- 返回值可直接传入同层的 `expectation_pauli` 和 `sample_probabilities`（它们按状态类型做相同分派）。
- `max_bond_dim`：MPS 后端的最大键维，默认 `MAX_BOND_DIM`（256）。statevector 后端忽略此参数。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。

### simulate_counts(qc, shots, *, seed=None, param_values=None, max_bond_dim=MAX_BOND_DIM, device=None)

- 返回 Dict[str, int]。
- 含噪线路自动路由到密度矩阵后端（从 $\rho$ 对角线采样）；否则按 qubit 数阈值选 statevector / MPS。
- `max_bond_dim`：MPS 后端的最大键维，默认 `MAX_BOND_DIM`（256）。传 `None` 表示不截断。statevector / DM 后端忽略此参数。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- bitstring 采用大端序（qubit 0 对应字符串最左位）。
- **测量投影**：当线路包含显式 `measure` 门且指定了 qubit→cbit 映射时，返回的 bitstring 会被投影到经典比特子空间（宽度 = `max(cbit) + 1`）。未测量的 qubit 被 marginalize 掉。若无显式测量门，行为不变（全 qubit 空间）。

### expectation_pauli(state, pauli, *, num_qubits)

- 按状态类型选择后端实现：2D 张量（`state.dim()==2`）走密度矩阵实现 $\mathrm{tr}(P\rho)$，否则按 num_qubits 阈值选 statevector / MPS。

### sample_probabilities(state, samples, *, num_qubits)

- 返回给定样本向量的概率 $P(b_i|\psi)$。
- 输入 `samples` 为 `(N, n_qubits)` 整数张量/数组（元素 0/1，big-endian）。
- 返回 1-D 张量，长度 N，支持自动微分。
- 按状态类型分派：2D 张量走密度矩阵实现，否则按 num_qubits 阈值选 statevector / MPS。
- 用于无监督 QNN 的 NLL 损失计算。

### energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian, max_bond_dim=MAX_BOND_DIM, device=None)

- 返回 (energy, expectations)。
- 含噪线路自动路由到密度矩阵后端（导出名 `energy_and_expectations_dm`）；否则按 qubit 数阈值选 statevector / MPS。
- `max_bond_dim`：MPS 后端的最大键维，默认 `MAX_BOND_DIM`（256）。传 `None` 表示不截断。statevector / DM 后端忽略此参数。
- `device`：torch 设备（`'cpu'` / `'cuda'`），默认 `None`（自动选择）。
- VQE 训练路径通常通过此函数统一进入仿真后端。

## 包级导出

`quantum_hw.sim.__init__` 当前导出：

- 分发层：`get_sim_config`、`set_sim_config`、`build_state_from_symbolic`、`simulate_counts`、`expectation_pauli`、`sample_probabilities`、`energy_and_expectations`
- 后端入口：`simulate_statevector`、`simulate_mps`、`simulate_mpo_process`
- 密度矩阵（含噪）：`simulate_density_matrix`、`simulate_noisy_counts`、`expectation_pauli_dm`、`sample_probabilities_dm`、`energy_and_expectations_dm`
- Clifford / Clifford+T：`CliffordError`、`is_clifford_circuit`、`simulate_clifford_expectation`、`simulate_clifford_expectations`、`count_non_clifford_gates`、`count_t_gates`、`simulate_clifford_t_expectation`、`simulate_clifford_t_expectations`
- 设备：`auto_sim_device`

## 相关页面

- statevector simulator（statevector.md）
- mps simulator（mps.md）
- mpo process simulator（mpo.md）
- density matrix simulator（density_matrix.md）
- noise kraus operators（noise_kraus.md）
