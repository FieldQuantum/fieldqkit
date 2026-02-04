# 文档

本文档给出更详细的使用说明、API 参数、返回结构、以及实现细节。示例代码见 [examples/demo.py](../examples/demo.py)。

## 快速流程

1. 配置线路与实验参数（`num_qubits`、`shots`、`zne`、`readout_mitigation`、`observables`）。
2. 创建 `QuantumHardwareClient`（传入 Token）。
3. 调用 `run_auto()` 自动选择硬件并运行。
4. 读取 `RunResult` 中的概率分布或可观测量，并根据需要绘图。

当 `observables` 为列表时，会自动合并可共用测量基的线路并批量提交任务。

## 核心类与结果结构

### `QuantumHardwareClient`

**用途**：封装硬件任务提交、自动选芯片、readout 校准缓存、ZNE 等逻辑。

**构造函数**

- `QuantumHardwareClient(token: str)`
  - `token`：硬件访问凭据。

**主要方法**

#### `run_auto(...) -> RunResult`

自动选择硬件并执行任务，是推荐入口。

参数：

- `circuit: str`：线路名称（如 `"ghz"`）或 OpenQASM2 字符串（以 `OPENQASM` 开头）。
- `name: str`：任务名称。
- `num_qubits: int`：线路逻辑比特数。
- `shots: int = 8192`：采样次数。
- `zne: bool = False`：是否启用 ZNE（CZ 三倍插入 + 线性外推）。
- `readout_mitigation: bool = False`：是否启用 readout 误差缓解。
- `readout_shots: Optional[int] = None`：readout 校准的 shots 数。
- `observables: Optional[Sequence[str] | str] = None`：Pauli string 或列表。
- `return_probabilities: bool = False`：是否返回概率分布。
- `target_qubits: Optional[Sequence[int]] = None`：指定物理比特映射。
- `prefer_chips: Optional[Sequence[str] | str] = None`：偏好芯片。
- `rank_weights: Optional[Dict[str, float]] = None`：芯片排序权重（`queue`/`nqubits`/`error`）。

返回：`RunResult`。

**返回结构 `RunResult`**

- `task_ids: Optional[List[str]]`：提交任务 ID 列表。
- `samples: Optional[List[List[int]] | List[List[List[int]]]]`
  - 单个测量基时为 `List[List[int]]`；多个测量基时为 `List[List[List[int]]]`。
- `probabilities: Optional[List[float] | List[List[float]]]`
  - 单个测量基时为一维概率分布；多个测量基时为二维列表。
- `probabilities_raw: Optional[List[float] | List[List[float]]]`
  - 仅在开启 readout mitigation 时返回，表示缓解前的概率分布。
- `observable_values: Optional[float | Dict[str, float]]`
  - 单个 observable 时为标量；多 observable 时为 `{observable: value}`。
- `observable_values_raw: Optional[float | Dict[str, float]]`
  - 仅在开启 readout mitigation 时返回，表示缓解前的可观测量。

**常见注意点**

- 当 `return_probabilities=False` 且 `observables` 为空时，`probabilities` 可能为 `None`。
- Readout 缓存默认有效期 1 小时，按芯片存单文件，按比特更新时间戳。

## 线路构建函数（`quantum_hw.circuits`）

- `build_ghz(num_qubits: int, measure: bool = False)`
  - 构建 GHZ 线路。
- `build_cluster(num_qubits: int, measure: bool = False)`
  - 构建 1D cluster 线路。
- `build_qft(num_qubits: int, measure: bool = False, with_swaps: bool = True)`
  - 构建 QFT 线路。
- `build_ising_time_evolution(num_qubits: int, j: float, h: float, t: float, steps: int = 1, measure: bool = False)`
  - 构建 Ising 模型时间演化线路。

`measure=True` 时自动追加测量。

## Observables（`quantum_hw.observables`）

主要函数：

- `pauli_support(pauli: str, num_qubits: int | None = None) -> List[int]`
- `group_observables(observables: Sequence[str], num_qubits: int) -> List[Dict[str, object]]`
- `append_measurement_basis(qc, basis_pattern: Sequence[str]) -> None`
- `pauli_expectation(samples: np.ndarray, pauli: str) -> float`

Pauli string 支持两种格式：

- 显式索引：`"Z0 X2 Y3 I4"`
- 固定长度字符串：`"ZZIX"`

## Readout（`quantum_hw.readout`）

主要函数：

- `build_readout_calibration_circuits(num_qubits: int)`
- `build_local_confusion_matrix(per_qubit_confusion: Dict[int, np.ndarray], target_qubits: Sequence[int])`
- `mitigate_readout(probabilities: np.ndarray, confusion_matrix: np.ndarray)`
- `apply_readout_mitigation_multi(...)`

**缓存与校准策略**

- 按芯片单文件缓存（`readout_<chip>.json`）。
- 按比特独立时间戳，缺失/过期只重跑该比特。

## ZNE（`quantum_hw.zne`）

- `apply_zne_cz_tripling(qct)`：对编译后电路 CZ 门做三倍插入。
- `zne_linear_extrapolate(probs_1, probs_3)`：线性外推去噪。

## Hardware 选择（`quantum_hw.hardware`）

- `rank_chips(tmgr, num_qubits, prefer_chips=None, weights=None) -> List[str]`
  - 排序逻辑：`queue` 越低越好、`nqubits` 越多越好、`error` 越低越好。
  - `weights` 为归一化后的线性权重。

## Plotting（`quantum_hw.plotting`）

- `plot_probabilities_compare(raw, mitigated, num_qubits, max_labels=16)`
  - 以柱状图对比缓解前后的概率。
- `plot_observables_compare(raw, mitigated)`
  - 对比可观测量期望值。
