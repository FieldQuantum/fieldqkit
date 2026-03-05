# zne

## 模块

- `quantum_hw.core.zne`

## 关键函数

- `apply_zne_cz_tripling`
- `zne_linear_extrapolate`

## 函数说明

### `apply_zne_cz_tripling(qct)`

- 作用：对电路中的每个 `cz` 门做三倍插入（原门后追加两个同门）。
- 当前缩放方案：`1x -> 3x`（仅针对 `cz`）。
- 返回：浅拷贝后的新电路对象（`gates` 列表已替换）。

### `zne_linear_extrapolate(probs_1, probs_3)`

- 作用：利用 scale=1 与 scale=3 的结果做线性零噪声外推。
- 公式：

$$
f(0) \approx \frac{3f(1)-f(3)}{2}
$$

- 可用于概率向量、标量期望值等可线性处理量。

## 边界与数值行为

- `zne_linear_extrapolate` 本身不做裁剪/归一化。
- 在 `run_auto` 路径中：
	- 概率外推后会做 `clip([0,1])` + 归一化。
	- observable 外推会按业务路径做必要裁剪（部分分支会限制在 `[-1, 1]`）。

## 与运行流程联动

- `QuantumHardwareClient._run_with_backend` 在 `zne=True` 时：
	- 额外构造一条 CZ-tripled 线路（scale=3）。
	- 收集 `samples_zne`。
	- 对概率/observable 调用 `zne_linear_extrapolate` 形成最终输出。
- `shadow` 算法同样沿用 `1x/3x + 线性外推` 机制。

## 示例

```python
import numpy as np
from quantum_hw.core.zne import zne_linear_extrapolate

probs_1 = np.array([0.52, 0.48])
probs_3 = np.array([0.56, 0.44])
probs_0 = zne_linear_extrapolate(probs_1, probs_3)
print(probs_0)

e1, e3 = 0.31, 0.22
e0 = zne_linear_extrapolate(e1, e3)
print(e0)
```

## 相关页面

- [QuantumHardwareClient](../api/QuantumHardwareClient.md)
- [readout](./readout.md)
- [ShadowTomography.run](../algorithms/shadow_tomography.md)
