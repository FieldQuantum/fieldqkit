# ReadoutCalibrationManager

## 概览

- **模块**：`quantum_hw.calibration.readout`
- **作用**：执行单比特 readout 校准（`|0⟩/|1⟩`），生成每比特 confusion matrix，并按芯片做增量缓存。

## 构造函数

```python
ReadoutCalibrationManager(
		*,
		cache_dir: Path,
		submit_openqasm_async: Callable[[str, str, int, Optional[str]], object],
		wait_task: Callable[[object], str],
		get_task_result: Callable[[object], Dict[str, object]],
		compact_for_sim: Callable[[QuantumCircuit], object],
		simulate_counts: Callable[[QuantumCircuit, int], Dict[str, int]],
)
```

- 该类是“执行器 + 缓存器”，构造时注入任务提交/等待/取结果能力。
- `cache_dir` 下缓存文件名形如：`readout_<chip_name>.json`。

## 关键方法

### `calibrate_readout(...) -> CalibrationResult`

- **签名**

```python
calibrate_readout(
		target_qubits: Optional[Sequence[int]],
		shots: Optional[int] = None,
		*,
		chip_name: Optional[str] = None,
		backend: Optional[Backend] = None,
		qasm_version: str = "2.0",
		print_true: bool = False,
) -> CalibrationResult
```

- **参数要点**
	- `target_qubits=None`：会自动从 `backend.qubits_with_attributes` 推导，并过滤 fidelity 为 0 的比特。
	- `shots=None`：默认使用 `1024`。
	- `qasm_version`：硬件提交时 `"2.0"` 走 `to_openqasm2`，否则走 `to_openqasm3`。
	- `chip_name="Simulator"`：走本地模拟分支，不发硬件任务。

- **返回值 `CalibrationResult`**
	- `target_qubits: List[int]`
	- `per_qubit_confusion: Dict[int, List[List[float]]]`
		- 每个比特对应 $2\times2$ confusion matrix。

- **缓存策略**
	- 缓存文件：`readout_<chip>.json`
	- 结构：
		- `timestamps: {"<qubit>": ISO8601时间}`
		- `per_qubit_confusion: {"<qubit>": [[...],[...]]}`
	- TTL：12 小时（`cache_is_fresh(..., ttl_hours=12)`）。
	- 增量刷新：仅缺失/过期的比特会重新校准。

- **与 `run_auto(readout_mitigation=True)` 的联动**
	- `QuantumHardwareClient._run_with_backend` 会实例化该管理器并复用缓存，随后将每比特 confusion matrix 用于概率/observable 缓解。

## 示例

```python
from pathlib import Path
from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.calibration.readout import ReadoutCalibrationManager
from quantum_hw.sim.statevector import simulate_counts

client = QuantumHardwareClient()
chip_name = "Simulator"
client.chip_name = chip_name
client.chip_backend = Backend(chip_name)

manager = ReadoutCalibrationManager(
		cache_dir=Path("src/quantum_hw/api/.cache"),
		submit_openqasm_async=client._submit_openqasm_async,
		wait_task=client._wait_task,
		get_task_result=client._get_task_result,
		compact_for_sim=client._compact_for_sim,
		simulate_counts=simulate_counts,
)

res = manager.calibrate_readout(
		target_qubits=None,
		shots=1024,
		chip_name=chip_name,
		backend=client.chip_backend,
)

print(res.target_qubits)
print(res.per_qubit_confusion[res.target_qubits[0]])
```

## 状态与副作用

- 会读写 `cache_dir` 下 readout 缓存文件。
- 硬件路径下会提交并等待多个任务（每个待校准比特两条线路）。
- 依赖 `chip_name` 与 `backend`，两者缺失会抛错。

## 相关辅助函数

- `build_confusion_matrix(res_list, num_qubits)`
	- 将计数字典转换为 confusion matrix；是本类产出矩阵的底层构造函数。
- 缓存辅助（`quantum_hw.calibration._cache`）
	- `cache_file`：缓存文件命名。
	- `load_timestamped_payload` / `save_timestamped_payload`：时间戳缓存读写。
	- `cache_is_fresh`：TTL 判定（12 小时）。

## 常见问题

- Q: 为什么只重跑了部分比特？
- A: 该模块按比特时间戳增量刷新，命中且未过期的比特直接复用缓存。

- Q: `target_qubits=None` 为什么报错？
- A: 需要 `backend.qubits_with_attributes` 可用；否则无法自动推导候选比特。

- Q: Simulator 模式也会生成缓存吗？
- A: 会，流程一致，只是任务执行从硬件提交切换为本地模拟。

## 相关页面

- [build_confusion_matrix](./build_confusion_matrix.md)
- [NativeTwoQubitRBManager](./NativeTwoQubitRBManager.md)
- [NativeTwoQubitTomographyManager](./NativeTwoQubitTomographyManager.md)
