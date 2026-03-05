# result types

## 模块

- `quantum_hw.core.types`

## 数据结构

- `RunResult`
- `CalibrationResult`
- `ShadowResult`
- `VQEResult`
- `QAOAResult`

## `RunResult`

用于 `QuantumHardwareClient.run_auto(...)` 及其内部执行链路。

字段：

- `task_ids: Optional[List[str]]`
	- 硬件任务 ID 列表；模拟器路径通常为 `None`。
- `samples: List[List[List[int]]]`
	- 每个测量组一份二维样本；即使只有单组也会以“按组分块”三维结构返回。
- `samples_zne: Optional[List[List[List[int]]]]`
	- 仅 ZNE 开启时返回，对应噪声缩放线路样本。
- `probabilities: List[List[float]]`
	- 按测量组返回概率向量列表；未请求概率输出时为 `[]`。
	- 若启用 readout/ZNE，此字段为处理后的结果。
- `probabilities_raw: List[List[float]]`
	- 缓解/外推前的概率，主要用于对比。
- `observable_values: Dict[str, float]`
	- 当前实现下，单个或多个 observable 都统一返回字典形状。
- `observable_values_raw: Dict[str, float]`
	- 对应 `observable_values` 的 raw 版本（缓解/外推前）。

何时为 `None`（当前实现）：

- `samples_zne` 在未开启 ZNE 时返回 `None`。
- `task_ids` 在模拟器路径通常为 `None`。

## `CalibrationResult`

用于 readout 校准结果。

- `target_qubits: List[int]`：校准覆盖的物理比特。
- `per_qubit_confusion: Dict[int, List[List[float]]]`：每个比特对应的 $2\times2$ confusion matrix。

## `ShadowResult`

用于 classical shadow 结果。

- `task_ids: Optional[List[str]]`
- `samples: Optional[List[List[int]]]`：展平后的样本列表。
- `basis_patterns: Optional[List[List[str]]]`：与样本等长的随机测量基（`X/Y/Z`）。
- `observables: Optional[List[str]]`
- `observable_estimates: Optional[Dict[str, float]]`
- `observable_estimates_raw: Optional[Dict[str, float]]`
	- 仅 ZNE 开启时有意义。
- `observable_stderr: Optional[Dict[str, float]]`
- `observable_stderr_raw: Optional[Dict[str, float]]`
	- 仅 ZNE 开启时有意义。
- `num_samples: Optional[int]`

## `VQEResult`

- `best_energy: float`：历史最优能量。
- `best_params: List[float]`：最优参数。
- `energy_history: List[float]`：每轮能量。
- `params_history: Optional[List[List[float]]]`：每轮更新后的参数。
- `grad_history: Optional[List[List[float]]]`：每轮梯度。
- `last_expectations: Optional[Dict[str, float]]`：最后一轮的各 observable 估计。

## `QAOAResult`

- `best_cost: float`：历史最优 cost（当前实现默认最大化）。
- `best_params: List[float]`：最优参数。
- `cost_history: List[float]`：每轮 cost。
- `params_history: Optional[List[List[float]]]`：每轮更新后的参数。
- `grad_history: Optional[List[List[float]]]`：每轮梯度。
- `last_expectations: Optional[Dict[str, float]]`：最后一轮的各 observable 估计。

## 单/多 observable 形状说明

- `RunResult.observable_values` 与 `RunResult.observable_values_raw` 统一为 `Dict[str, float]`。
- 即使只有一个 observable，也返回 `{observable: value}` 结构，避免调用端做分支判断。
- 算法结果（`ShadowResult/VQEResult/QAOAResult`）同样采用映射或列表，接口形状稳定。

## raw 字段与 zne/readout 关系

- `raw` 字段表示处理前估计：通常是 readout 缓解前或 ZNE 外推前。
- 未启用对应处理链路时，`raw` 字段可能为空，或与最终值等价但不单独返回。

## 相关页面

- [QuantumHardwareClient](../api/QuantumHardwareClient.md)
- [ShadowTomography.run](../algorithms/shadow_tomography.md)
- [VQERunner.run_model](../algorithms/vqe_runner.md)
- [QAOARunner.run_model](../algorithms/qaoa_runner.md)
