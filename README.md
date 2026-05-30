# Quantum Hardware Interface

> 版本：0.1.0 · 许可：MIT · Python ≥ 3.9

---

## 项目定位

`fieldqkit`（包名 `fieldqkit`）是一个面向用户的**量子硬件控制接口**，提供从量子线路构建、编译转译、提交执行、误差缓解到变分算法的完整工作流。项目以统一 API 屏蔽多量子云平台（夸父 / 天衍 / 国盾 / 腾讯 / 本源）的差异，并内置基于 PyTorch 的本地模拟器，支持自动微分和大规模张量网络仿真。

核心目标：

| 目标 | 说明 |
|---|---|
| **统一硬件访问** | 单一 `QuantumHardwareClient` 对接多平台（夸父/天衍/国盾/腾讯/本源/FieldQuantum） |
| **自动编译** | 逻辑电路 → 物理芯片的完整转译流程 |
| **误差缓解** | Readout 校准 + 零噪声外推（ZNE） |
| **变分算法** | VQE、QAOA、Shadow Tomography、QML |
| **量子机器学习** | PQC 监督分类 + 无监督 QNN 分布学习 |
| **硬件校准** | Readout、原生两比特 RB、过程层析 |
| **高效仿真** | 全态矢量 + MPS + MPO，支持梯度计算 |

## 安装

```bash
pip install -e .
```

> 核心依赖：Python >= 3.9，`numpy>=1.24`，`scipy>=1.10`，`networkx>=3.0`，`requests>=2.31`，`matplotlib>=3.7`，`openqasm3[parser]>=0.5`。

如果需要使用**本地模拟器**（`fieldqkit.sim`），需要额外安装 PyTorch：

```bash
pip install -e .[sim]       # 核心 + 模拟器（torch>=2.1）
```

如果需要接入**本源量子云**（`fieldqkit` Origin provider），需要额外安装 pyqpanda3：

```bash
pip install -e .[origin]    # 核心 + pyqpanda3（本源量子云 SDK）
```

> **量坤云端模拟器**（`fieldquantum` provider）无需额外依赖，仅需在配置文件或环境变量中填入 `fq_<32hex>` 形式的 API token。详见下文 [量坤云端模拟器](#量坤云端模拟器fieldquantum-provider) 小节。

其他可选依赖组：

```bash
pip install -e .[test]      # 核心 + pytest
```

## 快速开始

完整示例见 [examples/demo_full.ipynb](examples/demo_full.ipynb)。

```python
from fieldqkit import QuantumHardwareClient

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

## 真机使用

使用真机前需要配置对应平台的 API 凭证。推荐通过**配置文件**管理 token：

### 1. 创建配置文件

将项目根目录的模板文件复制为配置文件：

```bash
cp .quantum_hw.example.yaml .quantum_hw.yaml
```

### 2. 填入你的 token

编辑配置文件，将购买或申请到的 token 填入对应字段：

```yaml
credentials:
  quafu:
    api_token: "your-quafu-token-here"
```

### 各平台链接

- 夸父量子云：https://quafu-sqc.baqis.ac.cn/
- 天衍量子云： https://qc.zdxlz.com/
- 国盾量子云： https://quantumctek-cloud.com/
- 腾讯量子云： https://quantum.tencent.com/cloud/
- 本源量子云： https://qcloud.originqc.com.cn/
- 量坤云端模拟器： https://fieldquantum.tech/

各平台政策不同，优先推荐使用夸父量子云的免费资源（不限时）进行体验和学习。

## 模块全景

```
fieldqkit/                          入口 __init__.py（导出顶层 API）
├── api/                             硬件 API 层
│   ├── client.py                    QuantumHardwareClient — 唯一用户入口
│   ├── backend.py                   Backend / HardwareProfile / BackendAdapter (ABC)
│   ├── task.py                      OpenQasmSubmitRequest / TaskAdapter (ABC) / ProviderTaskHandle
│   ├── platform_credentials.py      凭证管理（夸父 / 天衍 / 国盾 / 腾讯 / 本源）
│   └── quantum_platform/            平台具体适配
│       ├── quafu.py                 夸父
│       ├── tianyan.py               天衍
│       ├── guodun.py                国盾
│       ├── tencent.py               腾讯
│       ├── origin.py                本源
│       ├── fieldquantum.py          量坤云端模拟器
│       └── cqlib.py                 cqlib 公共 HTTP 客户端（天衍 / 国盾共用）
│
├── circuit/                         线路表示
│   ├── quantumcircuit.py            QuantumCircuit 类（门操作、参数化、deepcopy）
│   ├── quantumcircuit_helpers.py    门名称字典、DAG 信息转换、门→线路渲染辅助
│   ├── qasm2.py                     OpenQASM 2 解析器
│   ├── qcis.py                      QASM ↔ QCIS 原生指令转换
│   ├── matrix.py                    门矩阵定义（numpy）
│   ├── render.py                    线路文本可视化
│   └── utils.py                     辅助工具
│
├── compile/                         编译转译
│   ├── transpiler.py                Transpiler — pass 管理器
│   ├── basepasses.py                TranspilerPass (ABC)
│   ├── decompose.py                 门分解（CX/SWAP/iSWAP/ECR/CCX… → U+CZ）
│   ├── layout.py                    Layout（线路感知布局选择，保真度+路由代价联合排序）
│   ├── routing.py                   SabreRouting（SWAP 插入，噪声感知 + 多试验模式）
│   ├── translate.py                 TranslateToBasisGates（翻译到 {U, CZ} 本征门集）
│   ├── optimize.py                  GateCompressor（对易重排 + 单比特合并 + 两比特对消）
│   ├── schedule.py                  DynamicalDecoupling（XY4 / CPMG DD 序列）
│   └── dag.py                       DAG 转换与可视化
│
├── algorithms/                      量子算法
│   ├── vqe.py                       VQERunner — Ising/Heisenberg/XXZ/XY/自定义 Hamiltonian
│   │                                parameter-shift / autograd 梯度, Adam 优化, Clifford fitting
│   ├── qaoa.py                      QAOARunner — MaxCut / 自定义 Z/ZZ 代价项
│   │                                parameter-shift / autograd 梯度, Adam 优化, Clifford fitting
│   ├── qml.py                       QML — PQC 监督分类 + 无监督 QNN + 条件 QNN
│   │                                autograd / parameter-shift, Adam 优化
│   ├── qml_runner.py                QMLRunner — 高层 QML 入口（自动 provider/芯片解析）
│   ├── qml_encoding.py              编码线路模板：Angle / IQP（含符号参数版本）
│   ├── optimizer_utils.py            共享优化工具（能量计算、参数移位梯度、Adam、
│   │                                Clifford fitting、run_variational_loop 通用优化循环）
│   ├── shadow.py                    ShadowTomography — classical shadow 协议
│   ├── ansatz_templates.py          Hardware-efficient ansatz 构建
│   └── circuit_compression.py       MPS/MPO 混合后缀压缩（降低线路深度）
│                                    + build_compression_transform（可复用压缩变换工厂）
│
├── core/                            通用工具
│   ├── circuits.py                  预置线路（GHZ / Cluster / QFT / Ising 演化）
│   ├── observables.py               Pauli 字符串解析、期望值计算、测量基转换
│   ├── readout.py                   Readout 误差缓解（逆混淆矩阵）
│   ├── zne.py                       ZNE（CZ 三倍插入 + 线性外推）
│   ├── types.py                     RunResult / CalibrationResult / VQEResult / ShadowResult /
│   │                                QAOAResult / QMLResult / QBMResult
│   ├── utils.py                     概率/采样辅助函数
│   └── plotting.py                  概率分布和可观测量对比图
│
├── calibration/                     校准
│   ├── readout.py                   ReadoutCalibrationManager（带缓存的 readout 校准）
│   ├── rb.py                        NativeTwoQubitRBManager（原生两比特 RB）
│   ├── tomography.py                NativeTwoQubitTomographyManager（过程层析）
│   └── _cache.py / _coupler_utils   缓存 TTL / coupler 过滤
│
├── sim/                             模拟器（需安装 [sim] 依赖组）
│   ├── statevector.py               全态矢量模拟（torch，支持 autograd）
│   ├── mps.py                       MPS 张量网络模拟器（可微）
│   ├── mpo.py                       MPO 量子过程模拟器
│   ├── clifford.py                  Clifford stabilizer 模拟器
│   ├── clifford_t.py                Clifford+T branching 模拟器
│   ├── matrix.py                    torch 门矩阵（支持梯度）
│   ├── interface.py                 统一模拟入口 simulate_counts / expectation_pauli /
│   │                                sample_probabilities / energy_and_expectations
│   │                                + set_sim_config / get_sim_config（运行时调参）
│   └── common.py                    参数解析工具
│
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
- [Clifford fitting 闭环（tianyan176）](examples/demo_clifford_fitting.ipynb)
- [VQE：顶层接口 + parameter-shift 手动梯度下降](examples/demo_vqe.ipynb)
- [QAOA：MaxCut + 自定义哈密顿量 + VQE 对比](examples/demo_qaoa.ipynb)
- [QML Iris：Iris 数据集多分类](examples/demo_qml_iris.ipynb)
- [QNN BAS：Born Machine 分布学习](examples/demo_qnn_bas.ipynb)
- [QNN 无监督：量子分布学习](examples/demo_qnn_unsupervised.ipynb)
- [VQE H₂ 4-qubit：氢分子势能面扫描](examples/demo_vqe_h2_4q.ipynb)
- [Backend 拓扑与芯片排序](examples/demo_backend.ipynb)

## 学习路径（入门 → 进阶 → 硬件 → 优化）

1. 入门：先看 [全览入门：run_auto + mitigation + 可视化](examples/demo_full.ipynb)
2. 进阶：继续 [QuantumCircuit 与 core 函数拆解](examples/demo_circuit_core.ipynb)
3. 硬件：再看 [Readout calibration + ZNE 专项](examples/demo_readout_zne.ipynb)
4. 优化：按顺序学习
    - [Shadow tomography 分层教程](examples/demo_shadow.ipynb)
    - [VQE：顶层接口 + parameter-shift 手动梯度下降](examples/demo_vqe.ipynb)
    - [QAOA：MaxCut + 自定义哈密顿量 + VQE 对比](examples/demo_qaoa.ipynb)
5. 量子机器学习：按顺序学习
    - [QML Iris：Iris 数据集多分类](examples/demo_qml_iris.ipynb)
    - [QNN BAS：Born Machine 分布学习](examples/demo_qnn_bas.ipynb)
    - [QNN 无监督：量子分布学习](examples/demo_qnn_unsupervised.ipynb)
6. VQE 进阶：[VQE H₂ 4-qubit：氢分子势能面扫描](examples/demo_vqe_h2_4q.ipynb)
7. 硬件拓扑补充：参考 [Backend 拓扑与芯片排序](examples/demo_backend.ipynb)

## 文档 (Docs)

Docs 总览见 [docs/README.md](docs/README.md)。
