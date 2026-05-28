# NativeTwoQubitRBManager

## 概览

- **模块**：`quantum_hw.calibration.rb`
- **作用**：对后端 native 双比特基门做 randomized benchmarking（RB），输出每条 coupler 的衰减拟合与 fidelity。

## 构造函数

```python
NativeTwoQubitRBManager(
		*,
		cache_dir: Path,
		submit_circuit_async: Callable[[str, QuantumCircuit, int, Optional[str], Optional[Dict]], object],
		wait_task: Callable[[object], str],
		get_task_result: Callable[[object], Dict[str, object]],
		compact_for_sim: Callable[[QuantumCircuit], object],
		simulate_counts: Callable[[QuantumCircuit, int], Dict[str, int]],
)
```

- 缓存文件名：`rb_two_qubit_<chip_name>.json`。

## 关键方法

### `calibrate_native_two_qubit_rb(...) -> Dict[str, Dict[str, object]]`

- **签名**

```python
calibrate_native_two_qubit_rb(
		couplers: Optional[Sequence[Tuple[int, int]]] = None,
		*,
		lengths: Optional[Sequence[int]] = None,
		num_sequences: int = 20,
		shots: int = 1024,
		chip_name: Optional[str] = None,
		backend: Optional[Backend] = None,
		readout_mitigation: bool = True,
		readout_shots: Optional[int] = None,
		seed: Optional[int] = None,
		print_true: bool = False,
) -> Dict[str, Dict[str, object]]
```

- **参数要点**
	- `couplers=None`：自动选择 `backend.couplers_with_attributes` 中 `fidelity>0` 的连线。
	- 若自动筛选后没有可用连线，会抛出 `RuntimeError("no available couplers with fidelity > 0")`。
	- `lengths=None`：默认 `[1, 2, 4, 8, 16, 32]`。
	- `num_sequences`：每个长度采样多少条随机序列。
	- `readout_mitigation=True`：会先调用 `ReadoutCalibrationManager`，构建 coupler 2 比特 confusion matrix。
	- `seed`：随机序列生成器种子。

- **返回值结构**
	- 顶层按 coupler key（`"q1-q2"`，按升序归一化）索引。
	- 常规运行返回字段：
		- `lengths: List[int]`
		- `total_lengths: List[int]`
		- `num_sequences: int`
		- `shots: int`
		- `survival_samples: Dict[str, List[float]]`
		- `survival_avg: Dict[int, float]`
		- `fit: {"p", "epc", "fidelity", "A", "B"}`
	- 缓存命中时为轻量返回：`{"fit": {"fidelity": ...}}`。

- **统计与拟合说明**
	- 每个长度先构造前向随机序列，再显式追加逆序列。
	- `total_lengths` 是用于拟合横轴的“等效总门数”（由基门类型映射得到 scale）。
	- 生存概率取 `|00⟩` 概率（可先做 readout 缓解再统计）。
	- 拟合模型本质为指数衰减线性化，输出平均门保真度：
		- 维度 $d=4$，$f_{avg}=\frac{(d-1)p+1}{d}$，`epc=1-f_avg`。

- **缓存策略**
	- 文件：`rb_two_qubit_<chip>.json`
	- 仅缓存每条 coupler 的 `fidelity`（减小体积）。
	- TTL 12 小时（`cache_is_fresh` 默认值）；过期后会重跑该 coupler。

## 示例

```python
from pathlib import Path
from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.calibration import NativeTwoQubitRBManager
from quantum_hw.sim.statevector import simulate_counts

client = QuantumHardwareClient()
chip_name = "Simulator"
client.chip_name = chip_name
client.chip_backend = Backend(chip_name)

rb = NativeTwoQubitRBManager(
		cache_dir=Path("src/quantum_hw/api/.cache"),
		submit_circuit_async=client._submit_circuit_async,
		wait_task=client._wait_task,
		get_task_result=client._get_task_result,
		compact_for_sim=client._compact_for_sim,
		simulate_counts=simulate_counts,
)

results = rb.calibrate_native_two_qubit_rb(
		couplers=None,
		lengths=[1, 2, 4, 8],
		num_sequences=30,
		shots=1024,
		chip_name=chip_name,
		backend=client.chip_backend,
		readout_mitigation=True,
		seed=42,
)

for key, payload in results.items():
		print(key, payload.get("fit", {}).get("fidelity"))
```

## 状态与副作用

- 会读写 RB 缓存文件；并在 readout 缓解开启时联动 readout 缓存。
- 硬件路径下会提交大量任务（约为 `len(couplers) * len(lengths) * num_sequences`）。
- 依赖 `chip_name`、`backend`、`backend.two_qubit_gate_basis`。
- 当前实现会在每条 coupler 完成后打印一行保真度日志：`Coupler <q1-q2>: fidelity=<value>`（与 `print_true` 无关）。

## 相关辅助函数

- coupler 辅助（`quantum_hw.calibration._coupler_utils`）
	- `resolve_positive_fidelity_couplers`：`couplers=None` 时自动筛选 `fidelity>0` 连线。
	- `coupler_key`：规范化 coupler 标识为 `min-max`（如 `3-7`）。
- 缓存辅助（`quantum_hw.calibration._cache`）
	- `cache_file` / `load_timestamped_payload` / `save_timestamped_payload` / `cache_is_fresh`。
	- 说明：当前 RB 缓存仅持久化 `fit.fidelity`。

## 常见问题

- Q: `lengths` 和 `total_lengths` 有什么区别？
- A: `lengths` 是随机序列长度；`total_lengths` 是考虑“前向+逆序列”和基门分解尺度后的拟合横轴。

- Q: 为什么缓存命中结果字段变少？
- A: 当前实现缓存只保存 `fidelity`，命中后返回最小结构；若需完整轨迹需强制重跑。

- Q: 哪些双比特门支持 RB？
- A: 当前支持 `cz/cx(cnot)/iswap/ecr`（`cnot` 会归一到 `cx`）。

## 相关页面

- [ReadoutCalibrationManager](./ReadoutCalibrationManager.md)
- [NativeTwoQubitTomographyManager](./NativeTwoQubitTomographyManager.md)
