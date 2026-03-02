# Quantum Hardware Interface

该项目提供面向用户的量子硬件控制接口，涵盖：

- 预置线路：GHZ / Cluster / QFT / Ising 时间演化
- 自定义线路：OpenQASM2 / OpenQASM3 字符串或 `QuantumCircuit`
- ZNE：将编译后所有 CZ 门三倍插入并做线性外推
- Readout 误差缓解：按物理比特做校准并缓存
- Calibration：readout 校准、native two-qubit RB、two-qubit process tomography
- 结果处理：采样、Pauli observables、概率分布 $p$
- Shadow tomography：随机测量基的可观测量估计
- VQE：基于量子测量的变分优化框架（Adam 优化）
- QAOA：经典组合优化问题到量子电路的接口（MaxCut + 自定义 Z/ZZ 代价项）

## 安装

```bash
pip install -e .
```

> 依赖：Python >= 3.9，`numpy>=1.24`，`networkx>=3.0`，`requests>=2.31`。
> OpenQASM3 解析依赖 `openqasm3`。
> 作图示例需要 `matplotlib`。
> 一键安装依赖：`pip install -e .[viz]`。

### Quafu 安装与 Token

Token 建议通过环境变量或配置文件注入，避免硬编码：

- 按 Quafu 官方文档配置 Token（例如环境变量或配置文件）

## 快速开始

完整示例见 [examples/demo.py](examples/demo.py)。

```python
from quantum_hw import QuantumHardwareClient

client = QuantumHardwareClient()
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

也可以直接传入 `QuantumCircuit`；若线路里已有测量，会被自动移除并在执行时按测量基重新追加。

校准示例：

- Readout calibration: [examples/demo_calibrate.py](examples/demo_calibrate.py)
- Native two-qubit RB: [examples/demo_rb.py](examples/demo_rb.py)
- Native two-qubit process tomography: see `NativeTwoQubitTomographyManager`

RB 缓存文件只保存每个 coupler 的 `fidelity`，避免大体积缓存。

## 模块结构

- `quantum_hw.api`：面向用户的 API 层（`QuantumHardwareClient`）。
- `quantum_hw.core`：通用工具与数据结构（circuits / observables / readout / zne / plotting / types）。
- `quantum_hw.calibration`：校准模块（readout / native two-qubit RB / two-qubit tomography）。
- `quantum_hw.compile`：编译与转译入口（`Transpiler`）。
- `quantum_hw.circuit.qasm2` / `quantum_hw.circuit.qasm3`：OpenQASM2/3 解析实现。

更多接口参数、执行流程与实现细节请见 [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)。