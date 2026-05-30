# utils

## 模块

- `fieldqkit.core.utils`

## 概览

该模块提供“计数 ↔ 样本 ↔ 概率”转换与局部边缘处理工具，是 `core.readout`、`calibration` 和算法模块的数据基础层。

## 关键函数

- `get_probabilities`
- `get_samples`
- `get_probabilities_from_samples`
- `marginal_samples`
- `get_local_probabilities_from_samples`
- `expectation_from_probabilities`

## 函数说明

### `get_samples(result: Dict[str, int], num_qubits: int) -> np.ndarray`

- 作用：将计数字典展开成逐 shot 的二维样本数组。
- 输出形状：`(nshots, num_qubits)`。
- 位序：直接按 bitstring 原序映射到样本列，和本项目概率计算位序保持一致。

### `get_probabilities_from_samples(samples: np.ndarray, num_qubits: int) -> np.ndarray`

- 作用：从样本数组计算全局基态概率向量。
- 输出长度：`2**num_qubits`。
- 行为：
  - 空样本返回全零向量。
  - 输入维度不匹配时抛 `ValueError`。

### `get_probabilities(result: Dict[str, int], num_qubits: int) -> np.ndarray`

- 作用：`counts -> samples -> probabilities` 的便捷封装。
- 常用于从硬件/模拟器计数直接得到概率分布。

### `marginal_samples(samples: np.ndarray, support: Sequence[int]) -> np.ndarray`

- 作用：从全量样本抽取 support 子集列。
- 特殊情况：`support` 为空时返回形状 `(nshots, 0)` 的空列数组。

### `get_local_probabilities_from_samples(samples: np.ndarray, support: Sequence[int]) -> np.ndarray`

- 作用：计算 support 上的边缘概率向量。
- 特殊情况：`support` 为空时返回 `np.array([1.0])`。

### `expectation_from_probabilities(probabilities: np.ndarray, support: Sequence[int]) -> float`

- 作用：从 Z 基概率分布计算 support 上 parity 期望值。
- 常用于 readout 缓解后，从边缘概率直接得到 observable 期望。

## 常见报错

- `ValueError("samples must be a 2D array with shape (nshots, num_qubits)")`
  - 触发于 `get_probabilities_from_samples` 输入形状不匹配。

## 示例

```python
import numpy as np
from fieldqkit.core.utils import (
    get_samples,
    get_probabilities,
    get_local_probabilities_from_samples,
    expectation_from_probabilities,
)

counts = {"00": 80, "01": 10, "10": 7, "11": 3}
samples = get_samples(counts, num_qubits=2)
probs = get_probabilities(counts, num_qubits=2)

local_probs = get_local_probabilities_from_samples(samples, support=[0, 1])
exp_zz = expectation_from_probabilities(local_probs, support=[0, 1])

print(samples.shape, probs, exp_zz)
```

## 相关页面

- [readout](./readout.md)
- [observables](./observables.md)
- [build_confusion_matrix](../calibration/build_confusion_matrix.md)
