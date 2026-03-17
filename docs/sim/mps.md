# mps simulator

## 模块

- quantum_hw.sim.mps

## 概览

该模块提供 Torch 实现的 MPS 后端，并与 statevector 后端保持同构接口：

- 线路前向演化：simulate_mps(...)
- 采样计数：simulate_counts(...)
- Pauli 期望：expectation_pauli(...)
- VQE 能量计算：energy_and_expectations(...)

同时该模块也包含一组内部张量网络工具函数，用于：

- 通过门的 MPO 分解施加多比特门
- 维护/移动 canonical center
- 在指定脏区间执行按需压缩

## 张量形状约定

- 单站点 MPS 张量：A[l, p, r]
- 维度含义：
  - l: 左虚拟键
  - p: 物理维（2）
  - r: 右虚拟键

## 关键函数

### simulate_mps(qc, *, param_values=None, max_bond_dim=None, device=None) -> List[torch.Tensor]

- 从 |0...0> 初态构造 MPS 并按门序演化。
- 支持：
  - 固定门：one_qubit/two_qubit/three_qubit
  - 参数门：one_qubit_param/two_qubit_param
  - 功能门：reset（barrier/measure/delay 在该层不改变态）
- max_bond_dim:
  - None: 不做显式截断
  - int: 在脏区间触发 sweep 压缩时截断键维

### simulate_counts(qc, shots, *, seed=None, param_values=None, device=None) -> Dict[str, int]

- 基于 MPS 逐位条件采样。
- 输出 bitstring 采用小端序（与 statevector 后端一致）。

### expectation_pauli(state, pauli, *, num_qubits)

- 输入 state 必须是非空 MPS 张量列表。
- 计算 <psi|P|psi>，其中 P 为 Pauli string。

### energy_and_expectations(symbolic_qc, *, params, param_names, hamiltonian, device=None)

- 将 params + param_names 映射到 param_values。
- 调用 simulate_mps(...) 得到态后，逐项收缩哈密顿量。
- 返回：
  - energy: 可微分 Torch 标量
  - expectations: Dict[str, float]

## 设备与 dtype

- 默认 dtype: torch.complex128（态张量）
- 自动设备选择：
  - 如果显式传入 device，优先使用
  - 否则优先 torch 默认设备/可用 CUDA

## 与接口层关系

- 当 qubit 数超过 interface.MPS_THRESHOLD_QUBITS（当前为 12）时，
  interface 层会把 counts/expectation/energy 路由到本模块。

## 常见限制与报错

- unsupported gate for simulator
- gate qubits must be distinct
- params must be a torch.Tensor（energy_and_expectations）

## 相关页面

- statevector simulator（statevector.md）
- simulator interface（interface.md）
- mpo process simulator（mpo.md）
