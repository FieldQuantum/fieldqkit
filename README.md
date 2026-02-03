# Quantum Hardware Interface

这个包用于给用户提供量子硬件控制接口，支持：

- 预置线路：GHZ、Cluster、QFT、Ising 模型时间演化
- 其它线路：用户提供 OpenQASM2 字符串
- ZNE（将编译后所有 CZ 门替换为 3 个 CZ 门）
- Readout 误差缓解（根据编译后的真实物理比特进行 readout benchmark，按比特缓存）
- 结果处理：样本（比特顺序反转）、Pauli string observables、完整概率分布 $p$

## 安装

```bash
pip install -e .
```

> 需要安装 `quark` 以及能访问硬件的依赖（根据你的环境）。
> 作图示例需要 `matplotlib`。
> 可选依赖：`pip install -e .[quark,viz]`。

## 快速使用

见 [examples/demo.py](examples/demo.py)。

示例（自动选芯片、批量可合并测量）：

```python
from quantum_hw import QuantumHardwareClient

client = QuantumHardwareClient(token="...")
result = client.run_auto(
  circuit="ghz",
  name="demo",
  num_qubits=6,
  shots=8192,
  observables=["IIZZII", "ZZIIII"],
  readout_mitigation=True,
  rank_weights={"queue": 0.2, "nqubits": 0.3, "error": 0.5},
)

print(result.observable_values)
print(result.probabilities)
```

## 设计要点

- 推荐使用 `run_auto()` 自动选择硬件并执行。
- `observables` 支持单个字符串或列表；可自动合并测量基并批量提交。
- 输出字段统一为复数：`task_ids`、`samples`、`probabilities`、`observable_values`，单个 observable 时返回标量，多 observable 时返回列表/字典。
- Readout 缓存按芯片存单文件、按比特更新时间戳，自动复用。
- `run_auto()` 支持 `prefer_chips`（字符串或列表）与 `rank_weights`（加权排序）。

## 说明

- Pauli string 暂定格式：
  - 带显式索引：`"Z0 X2 Y3 I4"`
  - 或固定长度字符串：`"ZZIX"`
- 对于 X/Y 测量，请确保样本来自对应测量基（例如已做基变换）。
