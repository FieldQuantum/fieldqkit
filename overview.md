# Quantum Hardware Interface — 项目全览

> 版本：0.1.0 · 许可：MIT · Python ≥ 3.9

---

## 一、项目定位

`quantum-hw`（包名 `quantum_hw`）是一个面向用户的**量子硬件控制接口**，提供从量子线路构建、编译转译、提交执行、误差缓解到变分算法的完整工作流。项目以统一 API 屏蔽多量子云平台（Quafu / 天衍 / 国盾 / 腾讯量子云）的差异，并内置基于 PyTorch 的本地模拟器，支持自动微分和大规模张量网络仿真。

核心目标：

| 目标 | 说明 |
|---|---|
| **统一硬件访问** | 单一 `QuantumHardwareClient` 对接多平台（Quafu/天衍/国盾/腾讯） |
| **自动编译** | 逻辑电路 → 物理芯片的完整转译流水线 |
| **误差缓解** | Readout 校准 + 零噪声外推（ZNE） |
| **变分算法** | VQE、QAOA、Shadow Tomography、QML |
| **量子机器学习** | PQC 监督分类 + 无监督 QNN 分布学习 |
| **硬件校准** | Readout、原生两比特 RB、过程层析 |
| **高效仿真** | 全态矢量 + MPS + MPO，支持梯度计算 |

代码规模约 **19,292 行**，分布于 61 个 Python 文件。

---

## 二、顶层目录结构

```
Quantum_control/
├── README.md                  项目说明（中文）
├── pyproject.toml             包管理配置
├── overview.md                本文件
├── docs/                      API 参考文档
│   ├── README.md              文档导航 & 学习路径
│   ├── api/                   API 层各模块说明
│   ├── algorithms/            算法模块说明
│   ├── calibration/           校准模块说明
│   ├── circuit/               线路层说明
│   ├── compile/               编译模块说明
│   ├── core/                  核心工具说明
│   └── sim/                   模拟器说明
├── examples/                  Jupyter Notebook 教程
│   ├── demo_full.ipynb        端到端完整示例
│   ├── demo_circuit_core.ipynb 线路 & 核心工具
│   ├── demo_shadow.ipynb      Shadow Tomography
│   ├── demo_vqe.ipynb         VQE 变分优化
│   ├── demo_qaoa.ipynb        QAOA 组合优化
│   ├── demo_qml.ipynb         PQC 监督分类器
│   ├── demo_qml_iris.ipynb    Iris 多分类
│   ├── demo_qnn_bas.ipynb     BAS 数据集 QNN 分布学习
│   ├── demo_qnn_unsupervised.ipynb 无监督 QNN 分布学习
│   ├── demo_readout_zne.ipynb Readout 缓解 + ZNE
│   └── demo_backend.ipynb     硬件拓扑与后端
├── scripts/                   辅助脚本
└── src/quantum_hw/            主源码（详见下节）
tests/                         测试套件（20 个 Python 测试文件）
```

---

## 三、源码模块详解

### 3.1 整体布局


```
quantum_hw/                          入口 __init__.py（导出顶层 API）
├── api/         (~4,399 行)         硬件 API 层
│   ├── client.py                    QuantumHardwareClient — 唯一用户入口
│   ├── backend.py                   Backend / HardwareProfile / BackendAdapter (ABC)
│   ├── task.py                      OpenQasmSubmitRequest / TaskAdapter (ABC) / ProviderTaskHandle
│   ├── platform_credentials.py      凭证管理（Quafu / TianYan / GuoDun / Tencent）
│   └── quantum_platform/            四平台具体适配
│       ├── quafu.py                 Quafu REST API（BAQIS）
│       ├── tianyan.py               天衍（cqlib 协议）
│       ├── guodun.py                国盾（cqlib 协议）
│       ├── tencent.py               腾讯量子云（tensorcircuit cloud API）
│       └── cqlib.py                 cqlib 公共 HTTP 客户端 + QASM↔QCIS 转换
│
├── circuit/     (~4,917 行)         线路表示
│   ├── quantumcircuit.py            QuantumCircuit 类（门操作、参数化、deepcopy）
│   ├── quantumcircuit_helpers.py    门名称字典、DAG 信息转换、门→线路渲染辅助
│   ├── qasm2.py / qasm3.py          OpenQASM 2/3 解析器
│   ├── matrix.py                    门矩阵定义（numpy）
│   ├── render.py                    线路文本可视化
│   └── utils.py                     U3/ZYZ/KAK 分解、酉矩阵等价性检验
│
├── compile/     (~3,580 行)         编译转译
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
├── algorithms/  (~4,795 行)         量子算法
│   ├── vqe.py                       VQERunner — Ising/Heisenberg/XXZ/XY/自定义 Hamiltonian
│   │                                parameter-shift / autograd 梯度, Adam 优化, Clifford fitting
│   ├── qaoa.py                      QAOARunner — MaxCut / 自定义 Z/ZZ 代价项
│   │                                parameter-shift / autograd 梯度, Adam 优化, Clifford fitting
│   ├── qml.py                       QML — PQC 监督分类 + 无监督 QNN 分布学习
│   │                                autograd / parameter-shift, Adam 优化
│   ├── qml_encoding.py              编码线路模板：Angle / IQP（含符号参数版本）
│   ├── optimizer_utils.py            共享优化工具（能量计算、参数移位梯度、Adam、
│   │                                Clifford fitting、run_variational_loop 通用优化循环）
│   ├── shadow.py                    ShadowTomography — classical shadow 协议
│   ├── ansatz_templates.py          Hardware-efficient / UCC ansatz 构建
│   └── circuit_compression.py       MPS/MPO 混合后缀压缩（降低线路深度）
│                                    + build_compression_transform（可复用压缩变换工厂）
│
├── core/        (~954 行)           通用工具
│   ├── circuits.py                  预置线路（GHZ / Cluster / QFT / Ising 演化）
│   ├── observables.py               Pauli 字符串解析、期望值计算、测量基转换
│   ├── readout.py                   Readout 误差缓解（逆混淆矩阵）
│   ├── zne.py                       ZNE（CZ 三倍插入 + 线性外推）
│   ├── types.py                     RunResult / CalibrationResult / VQEResult / ShadowResult /
│   │                                QAOAResult / QMLResult / QBMResult
│   ├── utils.py                     概率/采样辅助函数
│   └── plotting.py                  概率分布、可观测量对比、能量收敛曲线（matplotlib）
│
├── calibration/ (~1,446 行)          校准
│   ├── readout.py                   ReadoutCalibrationManager（带缓存的 readout 校准）
│   ├── rb.py                        NativeTwoQubitRBManager（原生两比特 RB）
│   ├── tomography.py                NativeTwoQubitTomographyManager（过程层析）
│   └── _cache.py / _coupler_utils   缓存 TTL / coupler 过滤
│
├── sim/         (~2,491 行)         模拟器
│   ├── statevector.py               全态矢量模拟（torch，支持 autograd）
│   ├── mps.py                       MPS 张量网络模拟器（可微，ComplexSVD 自定义 autograd）
│   ├── mpo.py                       MPO 量子过程模拟器
│   ├── matrix.py                    torch 门矩阵（支持梯度）
│   ├── interface.py                 统一模拟入口 simulate_counts / expectation_pauli /
│   │                                sample_probabilities / energy_and_expectations
│   └── common.py                    参数解析、设备选择工具
```

---

### 3.2 API 层（`api/`）

**核心类：`QuantumHardwareClient`**（`api/client.py`，527 行）

用户的唯一入口。负责线路归一化、Provider 运行时创建、后端解析、调用转译流水线、任务提交/轮询、结果后处理。

主要方法：

| 方法 | 功能 |
|---|---|
| `run_auto(circuit, num_qubits, shots, observables, ...)` | 一键执行完整工作流 |
| `_run_with_backend(qc, backend, runtime, ...)` | 底层执行：转译→提交→轮询→后处理 |
| `build_circuit(kind, **kwargs)` | 构建预置线路（ghz/cluster/qft/ising） |
| `_transpile_with_backend(qc, backend, ...)` | 调用编译流水线 |
| `_normalize_input_circuit(circuit, num_qubits)` | 统一化输入（字符串/对象） |

**硬件抽象**（`api/backend.py`，407 行）：

- `Backend`：基于图的硬件拓扑，提供比特距离、连通性、保真度查询
- `HardwareTopology` / `HardwareCalibration` / `HardwareProfile`：完整硬件描述（拓扑 + 校准元数据）
- `BackendAdapter`（ABC）：各 Provider 后端适配器的抽象基类
- `ResolvedBackend`：任务绑定的后端实例
- `list_available_hardware` / `build_hardware_profile`：硬件发现与 profile 构建

**任务管理**（`api/task.py`）：

- `TaskAdapter`（ABC）：提交/查询/获取/取消任务的标准接口
- `ProviderTaskHandle`：跨 Provider 的任务句柄（含 provider 名、task_id、payload）
- `OpenQasmSubmitRequest`：提交请求数据类

**Platform Provider 实现**（`api/quantum_platform/`）：

| 文件 | Provider | 协议 |
|---|---|---|
| `quafu.py`（220 行） | Quafu（北京量子信息科学研究院） | Quafu REST API |
| `tianyan.py`（113 行） | 天衍平台 | cqlib（QASM↔QCIS） |
| `guodun.py`（131 行） | 国盾平台 | cqlib（QASM↔QCIS），含 waveform 功能 |
| `cqlib.py`（659 行） | 公共 HTTP 客户端 | cqlib 共享实现（`RemotePlatformClient`） |

`ProviderRuntime`（`quantum_platform/__init__.py`）：dataclass 封装 backend adapter + task adapter，由 `create_provider_runtime` 工厂按 provider 名创建。

---

### 3.3 量子线路层（`circuit/`）

**核心类：`QuantumCircuit`**（`quantumcircuit.py`，1,364 行，68 个方法）

| 门类型 | 包含 |
|---|---|
| 单比特门 | Id、H、X、Y、Z、S、Sdg、T、Tdg、SX、SXdg、Rx、Ry、Rz、P、U、R、Reset |
| 两比特门 | CX/CNOT、CY、CZ、SWAP、iSWAP、ECR、RXX、RYY、RZZ、CP、CRZ |
| 三比特门 | CCX（Toffoli）、CCZ、CSWAP |
| 功能门 | measure / measure_all、barrier、delay、pauli_evolution |

重要功能：
- `from_openqasm2()` / `from_openqasm3()` 解析 OpenQASM 字符串
- `to_openqasm2()` / `to_openqasm3()` 导出字符串
- `draw()` 文本可视化
- `deepcopy()` 深拷贝
- 参数化门支持：通过 `params_value` 字典绑定符号参数
- `logical_to_physical` 映射（由转译器写入）
- `depth()` / `ncz()` 线路指标
- `apply_value()` 参数绑定
- `kak_for_unitary()` / `u3_for_unitary()` / `zyz_for_unitary()` 矩阵→门分解

**其他子模块：**

| 文件 | 功能 |
|---|---|
| `quantumcircuit_helpers.py`（534 行） | 门名称字典、`convert_gate_info_to_dag_info`、`add_gates_to_lines` 渲染辅助、`parse_expression` 表达式解析 |
| `qasm2.py`（307 行）/ `qasm3.py`（353 行） | OpenQASM 2.0 / 3.0 解析器 |
| `matrix.py`（318 行） | 门矩阵定义（NumPy），`gate_matrix_dict` 统一索引 |
| `utils.py`（245 行） | `u3_decompose`、`zyz_decompose`、`kak_decompose`、`simult_svd`、`is_equiv_unitary`、`generate_random_unitary_matrix` |
| `render.py`（25 行） | `draw_circuit` / `draw_circuit_simply` ASCII 线路可视化 |

---

### 3.4 编译 / 转译层（`compile/`）

**核心类：`Transpiler`**（`transpiler.py`，141 行）

编译流水线按以下顺序执行：

```
1. 三比特门分解        CCX/CCZ → U+CZ 原生门
2. 布局（Layout）      线路感知布局选择（保真度 + 路由代价联合排序）
3. SABRE 路由          噪声感知 SWAP 插入（多试验模式 + 保真度加权距离）
4. 基础门翻译          所有门 → {U, CZ} 本征门集
5. 门压缩              对易重排 + 单比特合并 + 两比特对消 + DAG 压缩
6. 动力学去耦（DD）    在双比特门空闲时雙插入 DD 序列（XY4 / CPMG）
```

各 Pass 均实现 `TranspilerPass` 抽象基类，可独立运行或组合。

| 文件 | 功能 |
|---|---|
| `decompose.py`（482 行） | 三比特/两比特门分解（含 Toffoli、各种二比特→CZ 分解） |
| `layout.py`（516 行） | 线路感知的物理比特分配（子图枚举 + 保真度路由代价联合排序） |
| `routing.py`（456 行） | SABRE 启发式 SWAP 插入（噪声感知 + 多试验模式） |
| `translate.py`（162 行） | 翻译到 {U, CZ} 本征门集 |
| `optimize.py`（598 行） | 对易重排、单比特合并、两比特对消、DAG 压缩（GateCompressor） |
| `schedule.py`（179 行） | 动力学去耦序列（DynamicalDecoupling，XY4/CPMG） |
| `dag.py`（176 行） | DAG（有向无环图）转换与可视化 |
| `basepasses.py`（42 行） | `TranspilerPass` ABC |

---

### 3.5 量子算法层（`algorithms/`）

#### 共享优化引擎（`optimizer_utils.py`，731 行）

`run_variational_loop()` 是 VQE 和 QAOA 共用的通用变分优化循环，支持：
- parameter-shift 梯度（硬件）/ autograd（模拟器）两条路径
- Adam 优化器（`adam_update`）
- 可插拔 `circuit_transform` 回调（供压缩等后处理使用）
- Clifford fitting 仿射噪声校正（`build_clifford_fit_map` / `apply_clifford_fit`）
- `evaluate_energy_with_backend` / `instantiate_transpiled_template` 硬件路径辅助

类型定义：
- `Hamiltonian = List[Tuple[float, str]]` — 通用哈密顿量表示

#### VQE（`vqe.py`，662 行）—— 变分量子本征求解器

`VQERunner` 薄封装 `run_vqe_with_backend`，后者负责：
- Hamiltonian：Ising / Heisenberg / XXZ / XY / 自定义 Pauli 字符串
- Ansatz 构建（Hardware-efficient / UCC / 自定义）
- 调用 `run_variational_loop` 执行优化循环
- 可选路径：Clifford fitting、混合后缀规划压缩（`build_compression_transform`）

#### QAOA（`qaoa.py`，374 行）—— 经典组合优化

`QAOARunner` 薄封装 `run_qaoa_with_backend`，后者负责：
- MaxCut（`build_maxcut_hamiltonian`）和自定义 Z/ZZ 代价项
- QAOA 参数化 ansatz 构建（RZZ + RX 交替层，`build_qaoa_ansatz_symbolic`）
- 调用 `run_variational_loop` 执行优化循环
- 可选 Clifford fitting 噪声校正

#### QML — 量子机器学习（`qml.py`，691 行）

`run_pqc_classifier`：参数化量子线路（PQC）监督分类器
- 支持 angle / IQP / 自定义编码，autograd 或 parameter-shift 梯度
- 支持 train/test 分离评估，返回 `QMLResult`（含 test accuracy）

`run_qnn_unsupervised`：无监督 QNN 分布学习
- autograd 路径：NLL 损失（通过 `sample_probabilities` 计算 $P(b|\theta)$）
- parameter-shift 路径：MMD² 损失（RBF 核）
- 返回 `QBMResult`（含生成样本）

#### QML 编码模板（`qml_encoding.py`，131 行）

- `angle_encoding_circuit` / `angle_encoding_circuit_symbolic`：角度编码
- `iqp_encoding_circuit` / `iqp_encoding_circuit_symbolic`：IQP 编码
- 每个编码提供数值版（直接编码特征向量）和符号版（返回参数化线路）

#### Shadow Tomography（`shadow.py`，276 行）

`ShadowTomography` 实现 **Classical Shadow 协议**：
- 随机单比特测量基采样
- 从有限测量快照重建可观测量期望值
- 返回 `ShadowResult`（含估计值与标准误）

#### 线路压缩（`circuit_compression.py`，623 行）

`MPS/MPO 混合后缀压缩` + 可复用压缩变换工厂：
- `plan_hybrid_suffix_blocks`：基于 MPS 键维分析，将线路后缀规划为多段压缩块
- `compress_circuit_with_hybrid_objective`：单段 MPS/MPO 目标优化压缩
- `build_compression_transform`：工厂函数，封装压缩状态（warm-start、block plan 缓存）并返回可直接传入 `run_variational_loop` 的 `circuit_transform` 回调，以及预转译的压缩模板
- `HybridCompressionPlan` / `SuffixCompressionBlock`：压缩方案数据类

#### Ansatz 模板（`ansatz_templates.py`，106 行）

- Hardware-efficient ansatz：分层 Ry 旋转 + CZ 纠缠（含符号参数版本）
- UCC（Unitary Coupled Cluster）ansatz（含符号参数版本）

---

### 3.6 核心工具层（`core/`）

| 文件 | 功能 |
|---|---|
| `circuits.py`（81 行） | 预置线路构建：`build_ghz`、`build_cluster`、`build_qft`、`build_ising_time_evolution` |
| `observables.py`（126 行） | Pauli 字符串解析（"ZZIX" 紧凑格式 / "Z0 X2" 索引格式）、测量基转换（`append_measurement_basis` / `apply_measurement_basis_rotations`）、期望值计算、可观测量分组（`group_observables`）、`shift_pauli_string` |
| `readout.py`（76 行） | Readout 误差缓解：`build_local_confusion_matrix`、`mitigate_readout`（伪逆校正）、`mitigate_observable_from_samples`、`expectation_from_samples_unbiased`（无偏奇偶估计器） |
| `zne.py`（20 行） | ZNE：`apply_zne_cz_tripling` — CZ 三倍化 + 线性外推至零噪声 |
| `types.py`（65 行） | 结果数据类：`RunResult`、`CalibrationResult`、`VQEResult`、`ShadowResult`、`QAOAResult`、`QMLResult`、`QBMResult` |
| `utils.py`（55 行） | 概率/采样辅助函数：`get_probabilities` / `get_samples` / `get_probabilities_from_samples` / `marginal_samples` / `get_local_probabilities_from_samples` |
| `plotting.py`（113 行） | `plot_probabilities_compare`、`plot_observables_compare`、`plot_energy_history`（matplotlib） |

---

### 3.7 硬件校准层（`calibration/`）

#### Readout 校准（`readout.py`，200 行）

`ReadoutCalibrationManager`：
- 为目标物理比特制备 |0⟩ 和 |1⟩ 态并测量，获得每比特 2×2 混淆矩阵
- 结果带 TTL 缓存（默认按芯片+比特索引存储），避免频繁重新校准
- 支持模拟器模式（返回理想对角矩阵）

#### 原生两比特 RB（`rb.py`，302 行）

`NativeTwoQubitRBManager`：
- 在指定 coupler 上执行随机 Clifford 序列，拟合存活概率衰减曲线 p(L) ≈ A·λ^L
- 输出每对比特的平均保真度和拟合参数

#### 过程层析（`tomography.py`，307 行）

`NativeTwoQubitTomographyManager`：
- 制备 16 种输入态，测量 16 种输出基，通过最小二乘重建 4×4 Choi 矩阵
- 计算过程保真度

#### 辅助模块：

| 文件 | 功能 |
|---|---|
| `_cache.py`（50 行） | TTL 文件缓存（`cache_file` / `load_timestamped_payload` / `save_timestamped_payload`，带 ISO8601 时间戳） |
| `_coupler_utils.py`（22 行） | `coupler_key` 规范化命名（如 "q0-q1"）、`resolve_positive_fidelity_couplers` 按保真度过滤 |

---

### 3.8 量子模拟器层（`sim/`）

统一入口 `sim/interface.py`（92 行）根据比特数自动分派：
- **≤ 16 比特** → 全态矢量模拟器（`statevector.py`）
- **> 16 比特** → MPS 张量网络模拟器（`mps.py`）

| 模块 | 描述 |
|---|---|
| `statevector.py`（198 行） | 基于 PyTorch 的全态矢量模拟，支持 autograd，顺序应用门矩阵 |
| `mps.py`（537 行） | MPS 张量网络模拟器，支持键维截断（默认 256）和梯度传播，`ComplexSVD` 自定义 autograd 反向，适合大量子比特 |
| `mpo.py`（242 行） | MPO 量子过程模拟器，用于构建幺正矩阵（线路等价性验证） |
| `matrix.py`（316 行） | PyTorch 格式门矩阵（与 `circuit/matrix.py` 对应），`gate_matrix_dict` 支持 `requires_grad=True` |
| `common.py`（66 行） | `auto_sim_device` 设备选择、`resolve_param` / `materialize_gate_matrix` 参数解析、`build_param_values_from_tensor` tensor↔dict 转换、`single_pauli` Pauli 算符构造 |

主要 API：
- `simulate_counts(qc, shots, seed, param_values)` → 测量计数字典
- `expectation_pauli(state, pauli, num_qubits)` → Pauli 期望值
- `sample_probabilities(state, samples, num_qubits)` → 样本概率（可微，用于 QNN NLL 损失）
- `energy_and_expectations(symbolic_qc, params, param_names, hamiltonian)` → VQE 能量评估

---

### 3.9 模块代码量分布

| 子包 | 行数 | 占比 |
|---|---|---|
| `algorithms/` | 4,150 | 21.5% |
| `circuit/` | 4,261 | 22.1% |
| `compile/` | 3,127 | 16.2% |
| `api/`（含 `quantum_platform/`） | 3,644 | 18.9% |
| `sim/` | 2,046 | 10.6% |
| `calibration/` | 1,233 | 6.4% |
| `core/` | 767 | 4.0% |
| 根 `__init__.py` | 64 | 0.3% |
| **合计** | **19,292** | **100%** |

---

## 四、核心调用链路

以最常用的 `run_auto()` 为例：

```
用户代码
  │
  ▼
QuantumHardwareClient.run_auto(
    circuit="ghz",       ← 预置名称 / QASM 字符串 / QuantumCircuit
    num_qubits=6,
    shots=8192,
    observables=["IIZZII", "ZZIIII"],
    readout_mitigation=True,
    provider="quafu"
)
  │
  ├─ [1] 归一化输入线路（字符串 → QuantumCircuit）
  │
  ├─ [2] 创建 ProviderRuntime（Quafu/TianYan/GuoDun）
  │
  ├─ [3] 发现硬件 & 解析后端
  │       backend_adapter.discover_hardware()
  │       backend_adapter.resolve_backend(num_qubits)
  │
  ├─ [4] 编译转译流水线
  │       三比特门分解 → 布局 → SABRE 路由 → 基础门翻译
  │       → 门压缩 → 动力学去耦
  │
  ├─ [5] 按测量基分组可观测量
  │
  ├─ [6] 对每组测量基：
  │       - 追加测量旋转门
  │       - 导出 OpenQASM
  │       - submit_openqasm() → 提交任务
  │       - query_status()   → 轮询状态
  │       - fetch_result()   → 获取计数
  │
  └─ [7] 后处理
          - Readout 误差缓解（混淆矩阵伪逆）
          - ZNE（CZ 三倍 + 线性外推，可选）
          - pauli_expectation() → 期望值
          ↓
      RunResult(task_ids, samples, probabilities, observable_values)
```

---

## 五、数据类型（结果结构）

| 类型 | 字段 |
|---|---|
| `RunResult` | `task_ids`、`samples`、`samples_zne`、`probabilities`、`probabilities_raw`、`observable_values`、`observable_values_raw` |
| `CalibrationResult` | `target_qubits`、`per_qubit_confusion` |
| `VQEResult` | `best_energy`、`best_params`、`energy_history`、`params_history`、`grad_history`、`last_expectations`、`clifford_fitting` |
| `ShadowResult` | `task_ids`、`samples`、`basis_patterns`、`observables`、`observable_estimates`、`observable_stderr` |
| `QAOAResult` | `best_cost`、`best_params`、`cost_history`、`params_history`、`grad_history`、`last_expectations`、`clifford_fitting` |
| `QMLResult` | `task`、`best_loss`、`best_params`、`loss_history`、`params_history`、`accuracy`、`test_loss_history`、`test_accuracy` |
| `QBMResult` | `best_loss`、`best_params`、`loss_history`、`test_loss_history`、`params_history`、`generated_samples` |

核心中间类型：

| 类型 | 位置 | 用途 |
|---|---|---|
| `Hamiltonian = List[Tuple[float, str]]` | `optimizer_utils` | 通用哈密顿量表示 |
| `CliffordFitMap = Dict[str, Tuple[float, float]]` | `optimizer_utils` | 噪声线性拟合系数 |
| `ProviderRuntime` | `quantum_platform/__init__` | Provider 运行时（backend adapter + task adapter） |
| `ResolvedBackend` | `api/backend` | 解析后硬件后端 |
| `HardwareProfile` | `api/backend` | 硬件参数 profile（拓扑 + 校准） |
| `HybridCompressionPlan` / `SuffixCompressionBlock` | `circuit_compression` | 压缩方案 |
| `OpenQasmSubmitRequest` / `ProviderTaskHandle` | `api/task` | 提交请求 / 任务句柄 |

---

## 六、依赖项

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| `numpy` | ≥1.24 | 数值计算、矩阵运算 |
| `scipy` | ≥1.10 | 线性代数、优化 |
| `networkx` | ≥3.0 | 硬件拓扑图 |
| `openqasm3[parser]` | ≥0.5 | OpenQASM 3.0 解析 |
| `requests` | ≥2.31 | 量子云平台 HTTP 通信 |
| `matplotlib` | ≥3.7 | 结果可视化 |
| `ipython` | ≥8.0 | 交互环境 |

可选依赖：
- `[test]`：`pytest≥7.4`（单元测试）
- `[sim]`：`torch≥2.1`（模拟器张量运算）
- `[full]`：`torch≥2.1`、`ipython≥8.0`（全功能安装）

---

## 七、快速开始

```bash
# 安装
pip install -e .

# 运行测试
pip install -e .[test]
pytest
```

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

print(result.observable_values)   # {'IIZZII': ..., 'ZZIIII': ...}
print(result.probabilities)       # numpy array of length 2^6
```

---

## 八、教程导航

| Notebook | 内容 |
|---|---|
| `demo_full.ipynb` | 端到端完整流程（推荐入门） |
| `demo_circuit_core.ipynb` | QuantumCircuit API + 核心工具详解 |
| `demo_shadow.ipynb` | Classical Shadow Tomography |
| `demo_readout_zne.ipynb` | Readout 校准 + ZNE 误差缓解 |
| `demo_vqe.ipynb` | VQE 变分优化（parameter-shift 梯度） |
| `demo_qaoa.ipynb` | QAOA MaxCut + 自定义代价项 |
| `demo_qml.ipynb` | PQC 监督分类器 |
| `demo_qml_iris.ipynb` | Iris 数据集多分类 |
| `demo_qnn_bas.ipynb` | BAS 数据集 QNN 分布学习 |
| `demo_qnn_unsupervised.ipynb` | 无监督 QNN 分布学习 |
| `demo_backend.ipynb` | 硬件拓扑与后端排序 |

**推荐学习路径：**

```
入门 → demo_full
进阶 → demo_circuit_core
硬件 → demo_readout_zne
优化 → demo_shadow → demo_vqe → demo_qaoa
QML  → demo_qml → demo_qml_iris → demo_qnn_bas → demo_qnn_unsupervised
拓扑 → demo_backend
```

---

## 九、测试覆盖

测试文件位于 `tests/`（20 个文件），主要覆盖：

| 测试组 | 文件 |
|---|---|
| API 层 | `test_api_exports_unified`、`test_api_provider_runtime`、`test_api_run_auto_unified`、`test_api_unified_backend`、`test_api_unified_task` |
| 算法 | `test_algorithms_provider_symmetry`、`test_vqe_autograd`、`test_vqe_hybrid_suffix_planner`、`test_qaoa`、`test_qml` |
| 线路 | `test_circuit_openqasm_advanced`、`test_circuit_refactor`、`test_circuit_safety` |
| 编译器 | `test_compile_passes`、`test_decompose_matrices` |
| 模拟器 | `test_sim_mps`、`test_sim_mpo` |
| 其他 | `test_qasm_parsing_modules`、`test_tencent_provider` |

---

## 十、未来发展方向

### 算法扩充

- **QAOA 实现**：已实现 `QAOARunner`，通过 `run_variational_loop` 与 VQE 共享优化核心，支持 MaxCut 及自定义 Z/ZZ 代价项。✅
- **QML 支持**：增加参数化线路分类器（PQC classifier）和无监督 QNN 分布学习，复用 `sim` 的 autograd 做本地训练。✅
- **动态线路**：在 `QuantumCircuit` 中支持 mid-circuit measurement + classical feedforward。

### 噪声建模与仿真

- **噪声模拟器**：在 `sim/` 增加退极化 / 振幅阻尼 / 读出翻转噪声通道（Kraus 算子或 MPO 密度矩阵），在本地评估缓解策略效果。
- **芯片噪声导入**：从 `HardwareProfile.calibration` 自动构建噪声模型，实现"数字孪生"式仿真。

### 编译器增强

- **噪声感知路由**：已实现 noise-aware routing（`-log(f)` 保真度加权距离）和多试验模式（`n_trials`）。✅
- **线路感知布局**：已实现 circuit-aware layout 选择（交互图 + 路由代价联合排序）。✅
- **两比特对消除**：已实现 `cancel_two_qubit_pairs`（自逆门对 + 对易判断消除）。✅
- **对易重排序**：已实现 `commutation_reorder`（利用对易关系聚集同比特单比特门以增强合并）。✅
- **本征门集扩展**：支持 iSWAP / √iSWAP 本征门（部分硬件原生支持），减少不必要的门分解。
- **电路等价性验证**：提供编译前后的酉矩阵比较工具（已有 `sim/mpo.py` 基础）。

### 工程质量

- **凭证管理**：引入 `~/.quantum_hw/config.toml` 或 keyring 方案，移除硬编码 debug token。
- **异步任务**：`asyncio` 化的 submit / poll，支持 batch 并行提交。
- **CLI 入口**：`python -m quantum_hw run --circuit ghz --qubits 6 --provider quafu`。
- **CI / 自动测试**：GitHub Actions 配合 mock provider，每次提交自动跑 pytest。
- **日志体系**：统一 `logging` 替代散布的 `print()`。

---

## 十一、设计模式总结

| 模式 | 应用位置 |
|---|---|
| **适配器模式** | `BackendAdapter`/`TaskAdapter` 屏蔽多平台差异 |
| **策略模式** | `TranspilerPass` 编译 Pass 可插拔组合 |
| **模板方法** | `QuantumHardwareClient` 固化工作流骨架，各步可覆盖 |
| **工厂模式** | `create_provider_runtime()` 按 provider 名创建运行时 |
| **装饰器/缓存** | TTL 缓存装饰校准结果，避免重复执行 |
| **数据类** | 大量使用 Python `@dataclass` 描述请求/响应/配置结构 |
