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

> 运行依赖：Python >= 3.9，`numpy>=1.24`，`torch>=2.1`，`scipy>=1.10`，`ipython>=8.0`，`networkx>=3.0`，`requests>=2.31`。
> OpenQASM3 解析依赖 `openqasm3[parser]`。
> 作图示例依赖 `matplotlib`，当前已包含在默认安装依赖中。
> 测试依赖组 `[test]`：`pytest>=7.4`。
> `quarkstudio`，`quarkcircuit` 因依赖声明问题（错误依赖 `clirk>=1.2.0`）拆分到 `[test-quark]`。

如果你需要本地运行测试，建议使用：

```bash
pip install -e .[test]
```

如果你需要 quark 相关测试，建议在安装基础测试依赖后单独安装：

```bash
pip install -e .[test]
pip install -e .[test-quark] --no-deps
```

## 快速开始

完整示例见 [examples/demo_full.ipynb](examples/demo_full.ipynb)。

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

## 模块结构

- `quantum_hw.api`：面向用户的 API 层（`QuantumHardwareClient`）。
- `quantum_hw.core`：通用工具与数据结构（circuits / observables / readout / zne / plotting / types）。
- `quantum_hw.calibration`：校准模块（readout / native two-qubit RB / two-qubit tomography）。
- `quantum_hw.compile`：编译与转译入口（`Transpiler`）。
- `quantum_hw.circuit`：线路表示与转换模块（`QuantumCircuit`、OpenQASM2/3、渲染、矩阵与分解工具）。

## 教程导航（Notebook）

- [全览入门：run_auto + mitigation + 可视化](examples/demo_full.ipynb)
- [QuantumCircuit 与 core 函数拆解](examples/demo_circuit_core.ipynb)
- [Shadow tomography 分层教程](examples/demo_shadow.ipynb)
- [Readout calibration + ZNE 专项](examples/demo_readout_zne.ipynb)
- [VQE：顶层接口 + parameter-shift 手动梯度下降](examples/demo_vqe.ipynb)
- [Backend 拓扑与芯片排序](examples/demo_backend.ipynb)
- [cqlib Provider 最小接入示例](examples/demo_cqlib_provider.ipynb)

## 学习路径（入门 → 进阶 → 硬件 → 优化）

1. 入门：先看 [全览入门：run_auto + mitigation + 可视化](examples/demo_full.ipynb)
2. 进阶：继续 [QuantumCircuit 与 core 函数拆解](examples/demo_circuit_core.ipynb)
3. 硬件：再看 [Readout calibration + ZNE 专项](examples/demo_readout_zne.ipynb)
4. 优化：按顺序学习
    - [Shadow tomography 分层教程](examples/demo_shadow.ipynb)
    - [VQE：顶层接口 + parameter-shift 手动梯度下降](examples/demo_vqe.ipynb)
5. 硬件拓扑补充：参考 [Backend 拓扑与芯片排序](examples/demo_backend.ipynb)

## 文档 (Docs)

Docs 总览见 [docs/README.md](docs/README.md)。

## Chemistry 应用

化学相关的数据、脚本、示例和说明文档已独立收纳在 [chemistry](chemistry)。
