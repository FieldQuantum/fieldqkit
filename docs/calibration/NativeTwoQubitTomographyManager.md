# NativeTwoQubitTomographyManager

## 概览

- **模块**：`quantum_hw.calibration.tomography`
- **作用**：对 native 双比特门执行 process tomography，重建实际 PTM 并输出 error channel 的 Choi 矩阵。

## 构造函数

```python
NativeTwoQubitTomographyManager(
		*,
		cache_dir: Path,
		submit_circuit_async: Callable[[str, QuantumCircuit, int, Optional[str], Optional[Dict]], object],
		wait_task: Callable[[object], str],
		get_task_result: Callable[[object], Dict[str, object]],
		compact_for_sim: Callable[[QuantumCircuit], object],
		simulate_counts: Callable[[QuantumCircuit, int], Dict[str, int]],
)
```

- 缓存文件名：`tomo_two_qubit_<chip_name>.json`。

## 关键方法

### `calibrate_native_two_qubit_tomography(...) -> Dict[str, Dict[str, object]]`

- **签名**

```python
calibrate_native_two_qubit_tomography(
		couplers: Optional[Sequence[Tuple[int, int]]] = None,
		*,
		shots: int = 1024,
		chip_name: Optional[str] = None,
		backend: Optional[Backend] = None,
		readout_mitigation: bool = True,
		readout_shots: Optional[int] = None,
		print_true: bool = False,
) -> Dict[str, Dict[str, object]]
```

- **参数要点**
	- `couplers=None`：自动选择 `fidelity>0` 的 coupler。
	- `shots`：每个“输入态 × 测量基”实验配置的采样数。
	- `readout_mitigation=True`：会先校准并构造 2 比特局部 confusion matrix。

- **实验规模**
	- 输入态：每比特 6 个（`0/1/+/−/+i/−i`），共 $6\times6=36$ 组。
	- 测量基：每比特 `X/Y/Z`，共 $3\times3=9$ 组。
	- 单个 coupler 总实验配置：$36\times9=324$。

- **重建流程（实现级）**
	- 从测量概率提取 Pauli 期望值。
	- 将输入/输出密度矩阵映射到 Pauli 向量，线性拟合得到 `ptm_actual`。
	- 用理想基门 `ptm_ideal` 右逆组合得到误差通道：
		- `ptm_error = ptm_actual @ pinv(ptm_ideal)`。
	- 再将 `ptm_error` 转为 Choi 矩阵 `choi_error`。

- **返回值结构**
	- 顶层按 coupler key（`"q1-q2"`）索引。
	- 每条 coupler 结果：`{"choi_error": np.ndarray(shape=(16,16), dtype=complex)}`。
	- 缓存落盘时保存为：`{"real": [...], "imag": [...]}`，加载时恢复为复数矩阵。

- **缓存策略**
	- 文件：`tomo_two_qubit_<chip>.json`
	- 按 coupler 粒度缓存。
	- TTL 1 小时，过期后重跑。

## 示例

```python
from pathlib import Path
from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.calibration import NativeTwoQubitTomographyManager
from quantum_hw.sim.statevector import simulate_counts

client = QuantumHardwareClient()
chip_name = "Simulator"
client.chip_name = chip_name
client.chip_backend = Backend(chip_name)

tomo = NativeTwoQubitTomographyManager(
		cache_dir=Path("src/quantum_hw/api/.cache"),
		submit_circuit_async=client._submit_circuit_async,
		wait_task=client._wait_task,
		get_task_result=client.tmgr.result,
		compact_for_sim=client._compact_for_sim,
		simulate_counts=simulate_counts,
)

res = tomo.calibrate_native_two_qubit_tomography(
		couplers=None,
		shots=256,
		chip_name=chip_name,
		backend=client.chip_backend,
		readout_mitigation=True,
)

for key, payload in res.items():
		print(key, payload["choi_error"].shape)
```

## 状态与副作用

- 会读写 tomography 缓存；若启用缓解还会读写 readout 缓存。
- 硬件路径任务量较大（按 coupler 乘以 324 个实验配置）。
- 依赖后端基门信息 `backend.two_qubit_gate_basis`，并仅支持 `cz/cx(cnot)/iswap/ecr`。

## 相关辅助函数

- coupler 辅助（`quantum_hw.calibration._coupler_utils`）
	- `resolve_positive_fidelity_couplers`：自动筛选有效 coupler。
	- `coupler_key`：统一 coupler key。
- 缓存辅助（`quantum_hw.calibration._cache`）
	- 统一管理 `tomo_two_qubit_<chip>.json` 的读写与 TTL。
- readout 联动
	- 通过 `ReadoutCalibrationManager` 获取每比特 confusion matrix，再构造局部 2 比特缓解矩阵。

## 常见问题

- Q: 返回的是 PTM 还是 Choi？
- A: 对外返回的是误差通道 `choi_error`；PTM 在内部中间步骤使用。

- Q: 缓存文件里为什么是 `real/imag` 两个数组？
- A: JSON 不支持复数，落盘时拆分实部和虚部，读取时再合并。

- Q: `readout_mitigation=True` 会发生什么？
- A: 先做单比特 readout 校准，再对每次 tomography 测量概率做局部 2 比特缓解。

## 相关页面

- [ReadoutCalibrationManager](./ReadoutCalibrationManager.md)
- [NativeTwoQubitRBManager](./NativeTwoQubitRBManager.md)
