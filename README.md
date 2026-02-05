# Quantum Hardware Interface

该项目提供面向用户的量子硬件控制接口，涵盖：

- 预置线路：GHZ / Cluster / QFT / Ising 时间演化
- 自定义线路：OpenQASM2 / OpenQASM3 字符串
- ZNE：将编译后所有 CZ 门三倍插入并做线性外推
- Readout 误差缓解：按物理比特做校准并缓存
- 结果处理：采样、Pauli observables、概率分布 $p$
- Shadow tomography：随机测量基的可观测量估计
- VQE：基于量子测量的变分优化框架（Adam 优化，默认 Ising 模型）
- QAOA：经典组合优化问题到量子电路的接口（MaxCut）

## 安装

```bash
pip install -e .
```

> 依赖：Python >= 3.9，`numpy>=1.24`。
> OpenQASM3 解析依赖 `openqasm3`。
> 需要访问硬件时请安装 `quark` 及其环境依赖。
> 作图示例需要 `matplotlib`。
> 可选依赖：`pip install -e .[quark,viz]`。

### Quafu 安装与 Token

使用 Quafu 前请先安装依赖：

```bash
pip install quarkstudio
pip install quarkcircuit
```

Token 建议通过环境变量或配置文件注入，避免硬编码：

- 运行时显式传入 `QuantumHardwareClient(token=...)`
- 或按 Quafu 官方文档配置 Token

## 快速开始

完整示例见 [examples/demo.py](examples/demo.py)。

```python
from quantum_hw import QuantumHardwareClient

client = QuantumHardwareClient(token="...")
result = client.run_auto(
    circuit="ghz",
    name="demo",
    num_qubits=6,
    shots=8192,
    observables=["IIZZII", "ZZIIII"],
    readout_mitigation=True,
    return_probabilities=True,
)

print(result.observable_values)
print(result.probabilities)
```

## 模块结构

- `quantum_hw.api`：面向用户的 API 层（`QuantumHardwareClient`）。
- `quantum_hw.core`：通用工具与数据结构（circuits / observables / readout / zne / plotting / types）。
- `quantum_hw.compile`：编译与转译入口（`Transpiler`）。
- `quantum_hw.circuit.qasm2` / `quantum_hw.circuit.qasm3`：OpenQASM2/3 解析实现。

> 旧的平铺模块已移除，请改用 `quantum_hw.core.*` 与 `quantum_hw.api`。

## 关键设计与行为

- 推荐入口：`run_auto()` 自动选择硬件并执行。
- `observables` 支持单个字符串或列表；内部会自动合并测量基并批量提交。
- `samples`、`probabilities`、`observable_values` 会随输入形态自动折叠（单个 vs 多个）。
- Readout 缓存按芯片单文件保存，按比特更新时间戳，默认有效期 1 小时。
- `run_auto()` 支持 `prefer_chips`（字符串或列表）与 `rank_weights`（加权排序）。
- 当 `readout_mitigation=True` 且未提供 `target_qubits` 时，会使用转译后 QASM 中的物理比特集合；
    为避免逻辑-物理映射不一致，建议显式传入 `target_qubits`。

## Pauli string 格式

支持两种格式：

- 固定长度字符串：`"ZZIX"`
- 显式索引：`"Z0 X2 Y3 I4"`

## API 入口

### `QuantumHardwareClient.run_auto(...)`

常用参数：

- `circuit: str`：线路名称（如 `"ghz"`）或 OpenQASM2 / OpenQASM3 字符串
- `name: str`：任务名称
- `num_qubits: int`：逻辑比特数
- `shots: int = 8192`
- `zne: bool = False`
- `readout_mitigation: bool = False`
- `readout_shots: Optional[int] = None`
- `observables: Optional[Sequence[str] | str] = None`
- `return_probabilities: bool = False`
- `target_qubits: Optional[Sequence[int]] = None`
- `prefer_chips: Optional[Sequence[str] | str] = None`
- `rank_weights: Optional[Dict[str, float]] = None`

返回 `RunResult`，包含：

- `task_ids`：任务 ID 列表
- `samples`：采样结果
- `probabilities` / `probabilities_raw`
- `observable_values` / `observable_values_raw`

### `QuantumHardwareClient.run_shadow(...)`

用于 classical shadow tomography。

常用参数：

- `circuit: str`：线路名称或 OpenQASM2 / OpenQASM3
- `name: str`
- `num_qubits: int`
- `shots: int = 8192`
- `observables: Optional[Sequence[str] | str] = None`
- `batch_size: int = 1`（每个随机基的 shots，经典 shadow 常用 1）
- `seed: Optional[int] = None`
- `zne: bool = False`
- `estimator: str = "mean"`（可选：`"mom"`）
- `mom_groups: Optional[int] = None`

返回 `ShadowResult`，包含：

- `task_ids`
- `samples`、`basis_patterns`
- `observables`
- `observable_estimates`、`observable_stderr`
- `observable_estimates_raw`、`observable_stderr_raw`（仅 ZNE 时）
- `num_samples`

### `QuantumHardwareClient.run_vqe(...)`

VQE 变分优化，当前默认模型为 Ising（横场 Ising），优化器为 Adam。

常用参数：

- `name: str`
- `num_qubits: int`
- `model: str = "ising"`
- `j: float = 1.0`, `h: float = 1.0`（Ising）
- `jx/jy/jz/hz`（Heisenberg/XY）
- `jxy/jz/hz`（XXZ）
- `layers: int = 1`
- `shots: int = 1024`
- `max_iters: int = 20`
- `learning_rate: float = 0.1`
- `beta1: float = 0.9`, `beta2: float = 0.999`, `eps: float = 1e-8`
- `shift: float = π/2`
- `target_qubits: Optional[Sequence[int]] = None`
- `prefer_chips: Optional[Sequence[str] | str] = None`
- `rank_weights: Optional[Dict[str, float]] = None`

常用哈密顿量构建：

- `build_ising_hamiltonian`
- `build_heisenberg_hamiltonian`
- `build_xxz_hamiltonian`
- `build_xy_hamiltonian`
- `build_custom_hamiltonian`

返回 `VQEResult`，包含：

- `best_energy`、`best_params`
- `energy_history`
- `last_expectations`（最后一次评估的可观测量期望值）

也可使用 `VQERunner`：

```python
from quantum_hw import QuantumHardwareClient, VQERunner

client = QuantumHardwareClient(token="...")
runner = VQERunner(client, layers=2, shots=1024)
result = runner.run_ising(name="vqe", num_qubits=4)
```

### `QuantumHardwareClient.run_qaoa(...)`

QAOA 组合优化（当前支持 MaxCut）。

常用参数：

- `name: str`
- `num_qubits: int`
- `problem: str = "maxcut"`
- `edges: List[Tuple[int,int]]`
- `weights: Optional[List[float]] = None`
- `p: int = 1`
- `learning_rate: float = 0.1`
- `target_qubits: Optional[Sequence[int]] = None`
- `prefer_chips: Optional[Sequence[str] | str] = None`
- `rank_weights: Optional[Dict[str, float]] = None`

返回 `QAOAResult`，包含：

- `best_cost`、`best_params`
- `cost_history`
- `last_expectations`（最后一次评估的可观测量期望值）

也可使用 `QAOARunner`：

```python
from quantum_hw import QuantumHardwareClient, QAOARunner

client = QuantumHardwareClient(token="...")
runner = QAOARunner(client, p=2, shots=1024)
result = runner.run_maxcut(name="qaoa", num_qubits=4, edges=[(0, 1), (1, 2)])
```

更详细的函数说明请见 [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)。