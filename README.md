# Quantum Hardware Interface

该项目提供面向用户的量子硬件控制接口，涵盖：

- 预置线路：GHZ / Cluster / QFT / Ising 时间演化
- 自定义线路：OpenQASM2 字符串
- ZNE：将编译后所有 CZ 门三倍插入并做线性外推
- Readout 误差缓解：按物理比特做校准并缓存
- 结果处理：采样、Pauli observables、概率分布 $p$

## 安装

```bash
pip install -e .
```

> 依赖：Python >= 3.9，`numpy>=1.24`。
> 需要访问硬件时请安装 `quark` 及其环境依赖。
> 作图示例需要 `matplotlib`。
> 可选依赖：`pip install -e .[quark,viz]`。

### Quafu 安装与 Token

使用 Quafu 前请先安装依赖：

```bash
pip install quarkstudio
pip install quarkcircuit
```

Token 建议通过环境变量或配置文件注入，避免硬编码：

- 运行时显式传入 `QuantumHardwareClient(token=...)`
- 或按 Quafu 官方文档配置 Token

## 快速开始

完整示例见 [examples/demo.py](examples/demo.py)。

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
    return_probabilities=True,
)

print(result.observable_values)
print(result.probabilities)
```

## 关键设计与行为

- 推荐入口：`run_auto()` 自动选择硬件并执行。
- `observables` 支持单个字符串或列表；内部会自动合并测量基并批量提交。
- `samples`、`probabilities`、`observable_values` 会随输入形态自动折叠（单个 vs 多个）。
- Readout 缓存按芯片单文件保存，按比特更新时间戳，默认有效期 1 小时。
- `run_auto()` 支持 `prefer_chips`（字符串或列表）与 `rank_weights`（加权排序）。

## Pauli string 格式

支持两种格式：

- 固定长度字符串：`"ZZIX"`
- 显式索引：`"Z0 X2 Y3 I4"`

## API 入口

### `QuantumHardwareClient.run_auto(...)`

常用参数：

- `circuit: str`：线路名称（如 `"ghz"`）或 OpenQASM2 字符串
- `name: str`：任务名称
- `num_qubits: int`：逻辑比特数
- `shots: int = 8192`
- `zne: bool = False`
- `readout_mitigation: bool = False`
- `readout_shots: Optional[int] = None`
- `observables: Optional[Sequence[str] | str] = None`
- `return_probabilities: bool = False`
- `target_qubits: Optional[Sequence[int]] = None`
- `prefer_chips: Optional[Sequence[str] | str] = None`
- `rank_weights: Optional[Dict[str, float]] = None`

返回 `RunResult`，包含：

- `task_ids`：任务 ID 列表
- `samples`：采样结果
- `probabilities` / `probabilities_raw`
- `observable_values` / `observable_values_raw`

更详细的函数说明请见 [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)。