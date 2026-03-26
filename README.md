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

## 模块全景

```
quantum_hw/                          入口 __init__.py（导出顶层 API）
├── api/         (~2,200 行)         硬件 API 层
│   ├── client.py                    QuantumHardwareClient — 唯一用户入口
│   ├── backend.py                   Backend / HardwareProfile / BackendAdapter (ABC)
│   ├── task.py                      OpenQasmSubmitRequest / TaskAdapter (ABC) / ProviderTaskHandle
│   ├── platform_credentials.py      凭证管理（Quafu / TianYan / GuoDun）
│   └── quantum_platform/            三平台具体适配
│       ├── quafu.py                 Quafu REST API（BAQIS）
│       ├── tianyan.py               天衍（cqlib 协议）
│       ├── guodun.py                国盾（cqlib 协议）
│       └── cqlib.py                 cqlib 公共 HTTP 客户端 + QASM↔QCIS 转换
│
├── circuit/     (~3,200 行)         线路表示
│   ├── quantumcircuit.py            QuantumCircuit 类（门操作、参数化、deepcopy）
│   ├── qasm2.py / qasm3.py          OpenQASM 2/3 解析器
│   ├── matrix.py                    门矩阵定义（numpy）
│   ├── render.py                    线路文本可视化
│   └── utils.py                     辅助工具
│
├── compile/     (~2,100 行)         编译转译
│   ├── transpiler.py                Transpiler — pass 管理器
│   ├── basepasses.py                TranspilerPass (ABC)
│   ├── decompose.py                 门分解（CX/SWAP/iSWAP/ECR/CCX… → U+CZ）
│   ├── layout.py                    Layout（逻辑↔物理比特映射）
│   ├── routing.py                   SabreRouting（SWAP 插入）
│   ├── translate.py                 TranslateToBasisGates（翻译到 {U, CZ} 本征门集）
│   ├── optimize.py                  GateCompressor（单比特门合并）
│   ├── schedule.py                  DynamicalDecoupling（CZ 间隙填充 DD 序列）
│   └── dag.py                       DAG 转换与可视化
│
├── algorithms/  (~1,900 行)         量子算法
│   ├── vqe.py                       VQERunner — Ising/Heisenberg/XXZ/XY/自定义 Hamiltonian
│   │                                parameter-shift / autograd 梯度, Adam 优化, Clifford fitting
│   ├── shadow.py                    ShadowTomography — classical shadow 协议
│   ├── ansatz_templates.py          Hardware-efficient / UCC ansatz 构建
│   └── circuit_compression.py       MPS/MPO 混合后缀压缩（降低线路深度）
│
├── core/        (~500 行)           通用工具
│   ├── circuits.py                  预置线路（GHZ / Cluster / QFT / Ising 演化）
│   ├── observables.py               Pauli 字符串解析、期望值计算、测量基转换
│   ├── readout.py                   Readout 误差缓解（逆混淆矩阵）
│   ├── zne.py                       ZNE（CZ 三倍插入 + 线性外推）
│   ├── types.py                     RunResult / VQEResult / ShadowResult / QAOAResult
│   └── plotting.py                  概率分布和可观测量对比图
│
├── calibration/ (~900 行)           校准
│   ├── readout.py                   ReadoutCalibrationManager（带缓存的 readout 校准）
│   ├── rb.py                        NativeTwoQubitRBManager（原生两比特 RB）
│   ├── tomography.py                NativeTwoQubitTomographyManager（过程层析）
│   └── _cache.py / _coupler_utils   缓存 TTL / coupler 过滤
│
├── sim/         (~1,400 行)         模拟器
│   ├── statevector.py               全态矢量模拟（torch，支持 autograd）
│   ├── mps.py                       MPS 张量网络模拟器（可微）
│   ├── mpo.py                       MPO 量子过程模拟器
│   ├── matrix.py                    torch 门矩阵（支持梯度）
│   ├── interface.py                 统一模拟入口 simulate_counts / expectation_pauli
│   └── common.py                    参数解析工具
│
└── vendor/      (~1,700 行)         内置第三方代码
    └── cqlib/                       QASM↔QCIS 转换器（天衍/国盾平台指令集）
```

## 核心调用链路

```
用户代码
  │
  ▼
QuantumHardwareClient.run_auto(provider="quafu", circuit=..., observables=...)
  │
  ├─ create_provider_runtime(provider)
  │    → ProviderRuntime(backend_adapter, task_adapter)
  │
  ├─ backend_adapter.discover_hardware()
  │    → [HardwareProfile, ...]
  │
  ├─ backend_adapter.resolve_backend()
  │    → ResolvedBackend(backend, profile)
  │
  └─ _run_with_backend()
       ├─ Transpiler pipeline
       │    decompose → layout → route → translate → DD
       ├─ QuantumCircuit.to_openqasm2/3
       ├─ TaskAdapter.submit_openqasm()   → 提交任务
       ├─ TaskAdapter.query_status()      → 轮询状态
       ├─ TaskAdapter.fetch_result()      → 获取 counts
       ├─ ReadoutCalibrationManager       → readout 缓解
       ├─ ZNE                             → CZ 三倍插入 + 外推
       └─ pauli_expectation()             → observable values → RunResult
```

## 教程导航（Notebook）

- [全览入门：run_auto + mitigation + 可视化](examples/demo_full.ipynb)
- [QuantumCircuit 与 core 函数拆解](examples/demo_circuit_core.ipynb)
- [Shadow tomography 分层教程](examples/demo_shadow.ipynb)
- [Readout calibration + ZNE 专项](examples/demo_readout_zne.ipynb)
- [VQE：顶层接口 + parameter-shift 手动梯度下降](examples/demo_vqe.ipynb)
- [Backend 拓扑与芯片排序](examples/demo_backend.ipynb)

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

完整 API 文档见 [docs/api/](docs/api/) —— 包含 QuantumHardwareClient、硬件发现、后端操作、任务管理、Provider 实现等详细说明。

## Chemistry 应用

化学相关的数据、脚本、示例和说明文档已独立收纳。

## 下一步发展方向

### 算法扩充

- **QAOA 实现**：实现 `QAOARunner`，复用 `_run_with_backend` 链路，支持 MaxCut 及自定义 Z/ZZ 代价项（类型 `QAOAResult` 已定义）。
- **QML 支持**：增加参数化线路分类器（PQC classifier），复用 `sim` 的 autograd 做本地训练。
- **动态线路**：在 `QuantumCircuit` 中支持 mid-circuit measurement + classical feedforward。

### 噪声建模与仿真

- **噪声模拟器**：在 `sim/` 增加退极化 / 振幅阻尼 / 读出翻转噪声通道（Kraus 算子或 MPO 密度矩阵），在本地评估缓解策略效果。
- **芯片噪声导入**：从 `HardwareProfile.calibration` 自动构建噪声模型，实现"数字孪生"式仿真。

### 编译器增强

- **多策略 routing**：引入 stochastic routing / 噪声感知 routing（优先使用高保真 coupler）。
- **本征门集扩展**：支持 iSWAP / √iSWAP 本征门（部分硬件原生支持），减少不必要的门分解。
- **电路等价性验证**：提供编译前后的酉矩阵比较工具（已有 `sim/mpo.py` 基础）。

### 工程质量

- **凭证管理**：引入 `~/.quantum_hw/config.toml` 或 keyring 方案，移除硬编码 debug token。
- **异步任务**：`asyncio` 化的 submit / poll，支持 batch 并行提交。
- **CLI 入口**：`python -m quantum_hw run --circuit ghz --qubits 6 --provider quafu`。
- **CI / 自动测试**：GitHub Actions 配合 mock provider，每次提交自动跑 pytest。
- **日志体系**：统一 `logging` 替代散布的 `print()`。
