# readout

## 模块

- `fieldqkit.core.readout`

## 关键函数

- `build_local_confusion_matrix`
- `mitigate_readout`
- `expectation_from_samples_unbiased`
- `mitigate_observable_from_samples`

## 函数说明

### `build_local_confusion_matrix(per_qubit_confusion, target_qubits) -> np.ndarray`

- 作用：对目标比特的单比特 confusion matrix 做 Kronecker 张量积，构造多比特局部 confusion matrix。
- 要求：`target_qubits` 不能为空。

### `mitigate_readout(probabilities, confusion_matrix) -> np.ndarray`

- 作用：通过 confusion matrix 伪逆做 readout 概率缓解。
- 约定：`confusion_matrix[i, j] = P(测得 i | 制备 j)`（即 `[measure, prepare]`）
- 数值处理：
	- 先 `pinv(confusion_matrix) @ p`。
	- 再裁剪到 `[0,1]`。
	- 最后归一化（若总和非零）。

### `expectation_from_samples_unbiased(local_samples, local_confusion_matrices) -> float`

- 作用：基于样本直接构造无偏估计，避免显式构造 `2^k` 概率向量。
- 适用：support 很大时降低内存压力，但方差通常更大。

### `mitigate_observable_from_samples(samples, support, per_qubit, target_qubits_group, marginal_max_support=10) -> float`

- 作用：对 observable 做读出缓解的“自适应入口”。
- 策略：
	- 当 `|support| <= marginal_max_support`：走边缘概率路径（先构造局部概率，再缓解，再算期望）。
	- 当 `|support| > marginal_max_support`：走无偏样本估计路径。

## 自适应路径说明

- 小 support：显式边缘概率法更稳定直观。
- 大 support：无偏样本法避免 `2^k` 维向量构造，时间/内存更可控。
- `run_auto` 中 observable 的缓解即使用该自适应策略。

## 常见报错

- `ValueError("target_qubits is empty")`
- `ValueError("confusion_matrix must be square")`
- `ValueError("local_samples must be a 2D array...")`
- `ValueError("local_confusion_matrices length must equal local_samples.shape[1]")`
- `ValueError("each local confusion matrix must have shape (2, 2)")`
- `ValueError("local_samples must contain only 0/1 outcomes")`

## 示例

```python
import numpy as np
from fieldqkit.core.readout import (
		build_local_confusion_matrix,
		mitigate_readout,
		mitigate_observable_from_samples,
)

per_qubit = {
		3: np.array([[0.97, 0.03], [0.04, 0.96]]),
		5: np.array([[0.98, 0.02], [0.05, 0.95]]),
}
cm = build_local_confusion_matrix(per_qubit, [3, 5])

probs = np.array([0.80, 0.10, 0.07, 0.03])
probs_m = mitigate_readout(probs, cm)

samples = np.array([[0, 0], [0, 1], [1, 0], [0, 0], [1, 1]], dtype=int)
val = mitigate_observable_from_samples(
		samples=samples,
		support=[0, 1],
		per_qubit=per_qubit,
		target_qubits_group=[3, 5],
)
print(probs_m, val)
```

## 与校准模块联动

- `ReadoutCalibrationManager` 负责生成 `per_qubit_confusion`。
- 本模块负责将该校准矩阵应用到概率或 observable 估计。

## 相关页面

- [ReadoutCalibrationManager](../calibration/ReadoutCalibrationManager.md)
- [observables](./observables.md)
- [utils](./utils.md)
