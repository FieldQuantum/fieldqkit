# noise kraus operators

## 模块

- `fieldqkit.sim.noise_kraus`

## 概览

该模块提供噪声信道的 Kraus 算符构造函数，供 [density matrix simulator](./density_matrix.md) 在演化含噪线路时调用。每个信道都是保迹的（CPTP），即 $\sum_k K_k^\dagger K_k = I$。算符以 `torch.complex64` 张量返回，可指定 `dtype` / `device`。

## 信道与 Kraus 算符

| 函数 | 信道 | 作用 |
|---|---|---|
| `depolarize1_kraus(p)` | 单比特去极化 | $\rho' = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$ |
| `depolarize2_kraus(p)` | 双比特去极化 | $\rho' = (1-p)\rho + \frac{p}{15}\sum_{P\neq II} P\rho P^\dagger$ |
| `x_error_kraus(p)` | 比特翻转 (X) | $\rho' = (1-p)\rho + p\,X\rho X$ |
| `y_error_kraus(p)` | Y 错误 | $\rho' = (1-p)\rho + p\,Y\rho Y$ |
| `z_error_kraus(p)` | 相位翻转 (Z) | $\rho' = (1-p)\rho + p\,Z\rho Z$ |
| `amplitude_damping_kraus(gamma)` | 振幅阻尼（能量耗散） | 激发态以 $\gamma$ 衰减到基态 |
| `phase_damping_kraus(gamma)` | 相位阻尼（退相干） | 损失相干性而不耗散能量 |

所有概率/阻尼参数须在 $[0, 1]$ 内，否则抛 `ValueError`。

## 分发入口

### `get_kraus_ops(gate_name, param, *, dtype=torch.complex64, device=None) -> list`

- 按 `gate_name` 返回对应信道的 Kraus 算符列表。
- 支持：`'depolarize1'`、`'depolarize2'`、`'x_error'`、`'y_error'`、`'z_error'`、`'amplitude_damping'`、`'phase_damping'`。
- `param`：信道参数（概率或阻尼系数）。
- 未知 `gate_name` 抛 `ValueError("Unknown noise channel: ...")`。

## 示例

```python
import torch
from fieldqkit.sim.noise_kraus import get_kraus_ops

kraus = get_kraus_ops("amplitude_damping", 0.3)
total = sum(K.conj().T @ K for K in kraus)
print(torch.allclose(total, torch.eye(2, dtype=torch.complex64)))  # True（保迹）
```

## 相关页面

- [density matrix simulator](./density_matrix.md)
- [QuantumCircuit 噪声信道方法](../circuit/quantumcircuit.md)
