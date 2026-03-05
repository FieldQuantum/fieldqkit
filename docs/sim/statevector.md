# statevector simulator

## 模块

- `quantum_hw.sim.statevector`

## 概览

该模块提供基于状态向量的本地模拟能力，核心入口为：

- `simulate_statevector(qc)`：返回末态向量
- `simulate_counts(qc, shots, seed=None)`：按末态概率采样得到计数

## 关键函数

### `simulate_statevector(qc: QuantumCircuit) -> np.ndarray`

- 作用：从 `|0...0⟩` 初态出发，按线路门序依次演化得到最终状态向量。
- 支持门类：
  - 单比特、双比特、三比特离散门
  - 含参数门（参数可来自数值或 `qc.params_value`）
  - `reset`（功能门）
- 返回：形状 `(2**n,)` 的复数向量。

### `simulate_counts(qc: QuantumCircuit, shots: int, seed: int | None = None) -> Dict[str, int]`

- 作用：先调用 `simulate_statevector` 得到概率，再按 `shots` 采样生成计数字典。
- 输出 bitstring：小端序（实现中对 `format(idx, ...)[::-1]` 做反转）。
- `seed`：使用 `np.random.default_rng(seed)` 控制可复现采样。

## 常见报错

- `ValueError("missing parameter value for ...")`
- `ValueError("invalid parameter value for ...")`
- `TypeError("unsupported parameter type: ...")`
- `ValueError("unsupported gate for simulator: ...")`

## 示例

```python
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim.statevector import simulate_statevector, simulate_counts

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

state = simulate_statevector(qc)
counts = simulate_counts(qc, shots=1024, seed=42)

print(state.shape)
print(counts)
```

## 相关页面

- [matrix utilities](./matrix.md)
- [utils](../core/utils.md)
