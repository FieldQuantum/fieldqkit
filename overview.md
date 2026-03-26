# Quantum Hardware Interface — 项目全览

> 版本：0.1.0 · 许可：MIT · Python ≥ 3.9

---

## 一、项目定位

`quantum-hw-interface`（包名 `quantum_hw`）是一个面向用户的**量子硬件控制接口**，提供从量子线路构建、编译转译、提交执行、误差缓解到变分算法的完整工作流。项目以统一 API 屏蔽多量子云平台（Quafu / 天衍 / 国盾）的差异，并内置基于 PyTorch 的本地模拟器，支持自动微分和大规模张量网络仿真。

核心目标：

| 目标 | 说明 |
|---|---|
| **统一硬件访问** | 单一 `QuantumHardwareClient` 对接多平台 |
| **自动编译** | 逻辑电路 → 物理芯片的完整转译流水线 |
| **误差缓解** | Readout 校准 + 零噪声外推（ZNE） |
| **变分算法** | VQE、QAOA、Shadow Tomography |
| **硬件校准** | Readout、原生两比特 RB、过程层析 |
| **高效仿真** | 全态矢量 + MPS + MPO，支持梯度计算 |

代码规模约 **12,521 行**，分布于 68 个 Python 文件。

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
│   ├── core/                  核心工具说明
│   └── sim/                   模拟器说明
├── examples/                  Jupyter Notebook 教程
│   ├── demo_full.ipynb        端到端完整示例
│   ├── demo_circuit_core.ipynb 线路 & 核心工具
│   ├── demo_shadow.ipynb      Shadow Tomography
│   ├── demo_vqe.ipynb         VQE 变分优化
│   ├── demo_readout_zne.ipynb Readout 缓解 + ZNE
│   └── demo_backend.ipynb     硬件拓扑与后端
├── scripts/                   辅助脚本
└── src/quantum_hw/            主源码（详见下节）
    └── tests/                 测试套件（19 个测试文件）
```

---

## 三、源码模块详解

### 3.1 整体布局

```
src/quantum_hw/
├── __init__.py                顶层公开 API 导出
├── api/        (~2,200 行)   硬件 API 层
├── circuit/    (~3,200 行)   量子线路表示
├── compile/    (~2,100 行)   编译 / 转译
├── algorithms/ (~1,900 行)   量子算法
├── core/       (~500 行)     通用工具
├── calibration/(~900 行)    硬件校准
├── sim/        (~1,400 行)   量子模拟器
└── vendor/     (~1,700 行)   内置第三方代码（QASM↔QCIS）
```

---

### 3.2 API 层（`api/`）

**核心类：`QuantumHardwareClient`**（`api/client.py`）

用户的唯一入口。负责线路归一化、Provider 运行时创建、后端解析、调用转译流水线、任务提交/轮询、结果后处理。

主要方法：

| 方法 | 功能 |
|---|---|
| `run_auto(circuit, num_qubits, shots, observables, ...)` | 一键执行完整工作流 |
| `build_circuit(kind, **kwargs)` | 构建预置线路（ghz/cluster/qft/ising） |
| `_transpile_with_backend(qc, backend, ...)` | 调用编译流水线 |
| `_normalize_input_circuit(circuit, num_qubits)` | 统一化输入（字符串/对象） |

**硬件抽象**（`api/backend.py`）：

- `Backend`：基于图的硬件拓扑，提供比特距离、连通性、保真度查询
- `HardwareProfile`：完整硬件描述（拓扑 + 校准元数据）
- `BackendAdapter`（ABC）：各 Provider 后端适配器的抽象基类
- `ResolvedBackend`：任务绑定的后端实例

**任务管理**（`api/task.py`）：

- `TaskAdapter`（ABC）：提交/查询/获取/取消任务的标准接口
- `ProviderTaskHandle`：跨 Provider 的任务句柄（含 provider 名、task_id、payload）
- `OpenQasmSubmitRequest`：提交请求数据类

**Platform Provider 实现**（`api/quantum_platform/`）：

| 文件 | Provider | 协议 |
|---|---|---|
| `quafu.py` | Quafu（北京量子信息科学研究院） | Quafu REST API |
| `tianyan.py` | 天衍平台 | cqlib（QASM↔QCIS） |
| `guodun.py` | 国盾平台 | cqlib（QASM↔QCIS） |
| `cqlib.py` | 公共 HTTP 客户端 | cqlib 共享实现 |

---

### 3.3 量子线路层（`circuit/`）

**核心类：`QuantumCircuit`**（`quantumcircuit.py`）

| 门类型 | 包含 |
|---|---|
| 单比特门 | H、X、Y、Z、S、Sdg、T、Tdg、SX、SXdg、Rx、Ry、Rz、P、U3、Reset |
| 两比特门 | CX/CNOT、CY、CZ、SWAP、iSWAP、ECR、RXX、RYY、RZZ、CP、CRZ |
| 三比特门 | CCX（Toffoli）、CCZ、CSWAP |
| 功能门 | measure_all、barrier、delay |

重要功能：
- `from_openqasm2()` / `from_openqasm3()` 解析 OpenQASM 字符串
- `to_openqasm2()` / `to_openqasm3()` 导出字符串
- `draw()` 文本可视化
- `deepcopy()` 深拷贝
- 参数化门支持：通过 `params_value` 字典绑定符号参数
- `logical_to_physical` 映射（由转译器写入）

**其他子模块：**

| 文件 | 功能 |
|---|---|
| `qasm2.py` / `qasm3.py` | OpenQASM 2.0 / 3.0 解析器 |
| `matrix.py` | 门矩阵定义（NumPy） |
| `utils.py` | U3 分解、ZYZ Euler 分解、KAK 分解 |
| `render.py` | ASCII 线路可视化 |

---

### 3.4 编译 / 转译层（`compile/`）

**核心类：`Transpiler`**（`transpiler.py`）

编译流水线按以下顺序执行：

```
1. 三比特门分解        CCX/CCZ → U+CZ 原生门
2. 布局（Layout）      逻辑比特 → 物理比特映射
3. SABRE 路由          插入 SWAP 门以满足连通性约束
4. 基础门翻译          所有门 → {U, CZ} 本征门集
5. 门压缩              合并相邻单比特门（减少深度）
6. 动力学去耦（DD）    在 CZ 空闲时隙插入 DD 序列
```

各 Pass 均实现 `TranspilerPass` 抽象基类，可独立运行或组合。

| 文件 | 功能 |
|---|---|
| `decompose.py` | 门分解（含 Toffoli） |
| `layout.py` | 保真度优先的物理比特分配 |
| `routing.py` | SABRE 启发式 SWAP 插入 |
| `translate.py` | 翻译到 {U, CZ} 本征门集 |
| `optimize.py` | 单比特门合并（GateCompressor） |
| `schedule.py` | 动力学去耦序列（DynamicalDecoupling） |
| `dag.py` | DAG（有向无环图）转换与可视化 |

---

### 3.5 量子算法层（`algorithms/`）

#### VQE（`vqe.py`）—— 变分量子本征求解器

`VQERunner` 支持：
- Hamiltonian：Ising / Heisenberg / XXZ / XY / 自定义 Pauli 字符串
- 梯度策略：parameter-shift 规则 或 PyTorch autograd
- 优化器：Adam
- Ansatz：Hardware-efficient / UCC（来自 `ansatz_templates.py`）
- Clifford fitting 加速

#### Shadow Tomography（`shadow.py`）

`ShadowTomography` 实现 **Classical Shadow 协议**：
- 随机单比特测量基采样
- 从有限测量快照重建可观测量期望值
- 返回 `ShadowResult`（含估计值与标准误）

#### 线路压缩（`circuit_compression.py`）

`MPS/MPO 混合后缀压缩`：
- 将深层参数化线路的后缀压缩为等效 MPS 形式
- 降低 VQE 等算法的线路深度，节省量子资源

#### Ansatz 模板（`ansatz_templates.py`）

- Hardware-efficient ansatz：分层 Ry 旋转 + CZ 纠缠
- UCC（Unitary Coupled Cluster）ansatz

---

### 3.6 核心工具层（`core/`）

| 文件 | 功能 |
|---|---|
| `circuits.py` | 预置线路构建：`build_ghz`、`build_cluster`、`build_qft`、`build_ising_time_evolution` |
| `observables.py` | Pauli 字符串解析（"ZZIX" 紧凑格式 / "Z0 X2" 索引格式）、测量基转换、期望值计算、可观测量分组 |
| `readout.py` | Readout 误差缓解：构建混淆矩阵、伪逆校正、无偏奇偶估计器 |
| `zne.py` | ZNE：CZ 三倍插入（1× → 3× 噪声）+ 线性外推至零噪声 |
| `types.py` | 结果数据类：`RunResult`、`VQEResult`、`ShadowResult`、`QAOAResult` |
| `utils.py` | 概率/采样辅助函数：counts 转概率/样本、边缘分布、Z 期望值 |
| `plotting.py` | 概率分布与可观测量对比可视化（matplotlib） |

---

### 3.7 硬件校准层（`calibration/`）

#### Readout 校准（`readout.py`）

`ReadoutCalibrationManager`：
- 为目标物理比特制备 |0⟩ 和 |1⟩ 态并测量，获得每比特 2×2 混淆矩阵
- 结果带 TTL 缓存（默认按芯片+比特索引存储），避免频繁重新校准
- 支持模拟器模式（返回理想对角矩阵）

#### 原生两比特 RB（`rb.py`）

`NativeTwoQubitRBManager`：
- 在指定 coupler 上执行随机 Clifford 序列，拟合存活概率衰减曲线 p(L) ≈ A·λ^L
- 输出每对比特的平均保真度和拟合参数

#### 过程层析（`tomography.py`）

`NativeTwoQubitTomographyManager`：
- 制备 16 种输入态，测量 16 种输出基，通过最小二乘重建 4×4 Choi 矩阵
- 计算过程保真度

#### 辅助模块：

| 文件 | 功能 |
|---|---|
| `_cache.py` | TTL 文件缓存（保存/加载带 ISO8601 时间戳的数据） |
| `_coupler_utils.py` | Coupler 规范化命名（如 "q0-q1"）、按保真度过滤 |

---

### 3.8 量子模拟器层（`sim/`）

统一入口 `sim/interface.py` 根据比特数自动分派：
- **≤ 12 比特** → 全态矢量模拟器（`statevector.py`）
- **> 12 比特** → MPS 张量网络模拟器（`mps.py`）

| 模块 | 描述 |
|---|---|
| `statevector.py` | 基于 PyTorch 的全态矢量模拟，支持 autograd，顺序应用门矩阵 |
| `mps.py` | MPS 张量网络模拟器，支持键维截断（默认 256）和梯度传播，适合大量子比特 |
| `mpo.py` | MPO 量子过程模拟器，用于构建幺正矩阵（线路等价性验证） |
| `matrix.py` | PyTorch 格式门矩阵，支持 `requires_grad=True` |
| `common.py` | 参数绑定、门矩阵实例化、Pauli 算符构造等共享工具 |

主要 API：
- `simulate_counts(qc, shots, seed, param_values)` → 测量计数字典
- `expectation_pauli(state, pauli, num_qubits)` → Pauli 期望值
- `energy_and_expectations(symbolic_qc, params, param_names, hamiltonian)` → VQE 能量评估

---

### 3.9 内置第三方代码（`vendor/`）

`vendor/cqlib/` 实现 **OpenQASM ↔ QCIS** 双向转换，服务于天衍和国盾平台的专有指令集：

| 子模块 | 功能 |
|---|---|
| `qasm_to_qcis/` | OpenQASM → QCIS 转换（规则映射 + 数据表） |
| `qcis_to_qasm.py` | QCIS → OpenQASM 逆向转换 |
| `laboratory_utils.py` | 平台特定工具函数 |
| `const.py` | 指令名称常量 |

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
| `RunResult` | `task_ids`、`samples`、`probabilities`、`observable_values` |
| `VQEResult` | `energy`、`params`、`energy_history`、`expectation_history` |
| `ShadowResult` | `task_ids`、`samples`、`basis_patterns`、`observables`、`observable_estimates`、`observable_stderr` |
| `QAOAResult` | `best_cost`、`best_params`、`cost_history` |

---

## 六、依赖项

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| `numpy` | ≥1.24 | 数值计算、矩阵运算 |
| `torch` | ≥2.1 | 可微模拟、VQE 自动微分 |
| `scipy` | ≥1.10 | 线性代数、优化 |
| `networkx` | ≥3.0 | 硬件拓扑图 |
| `openqasm3[parser]` | ≥0.5 | OpenQASM 3.0 解析 |
| `requests` | ≥2.31 | 量子云平台 HTTP 通信 |
| `matplotlib` | ≥3.7 | 结果可视化 |
| `ipython` | ≥8.0 | 交互环境 |

可选依赖：
- `[test]`：`pytest≥7.4`（单元测试）
- `[test-quark]`：`quarkcircuit`、`quarkstudio`（Quark 平台集成）

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
| `demo_backend.ipynb` | 硬件拓扑与后端排序 |

**推荐学习路径：**

```
入门 → demo_full
进阶 → demo_circuit_core
硬件 → demo_readout_zne
优化 → demo_shadow → demo_vqe
拓扑 → demo_backend
```

---

## 九、测试覆盖

测试文件位于 `tests/`（19 个文件），主要覆盖：

| 测试组 | 文件 |
|---|---|
| API 层 | `test_api_exports_unified`、`test_api_provider_runtime`、`test_api_run_auto_unified`、`test_api_unified_backend`、`test_api_unified_task` |
| 算法 | `test_algorithms_provider_symmetry`、`test_vqe_autograd`、`test_vqe_hybrid_suffix_planner` |
| 线路 | `test_circuit_migration`、`test_circuit_openqasm_advanced`、`test_circuit_refactor`、`test_circuit_safety` |
| 编译器 | `test_compile_passes`、`test_decompose_matrices` |
| 模拟器 | `test_sim_mps`、`test_sim_mpo` |
| 其他 | `test_qasm_parsing_modules`、`test_backend_migration` |

---

## 十、未来发展方向

### 算法扩充
- **QAOA Runner**：复用 `_run_with_backend` 链路，支持 MaxCut 及自定义 Z/ZZ 代价项
- **QML**：参数化线路分类器（PQC），复用 `sim` autograd 本地训练
- **动态线路**：支持 mid-circuit measurement + classical feedforward

### 噪声建模与仿真
- **噪声模拟器**：退极化 / 振幅阻尼 / 读出翻转噪声通道（Kraus 算子或 MPO 密度矩阵）
- **芯片噪声导入**：从 `HardwareProfile.calibration` 自动构建噪声模型（"数字孪生"仿真）

### 编译器增强
- **多策略路由**：随机路由 / 噪声感知路由（优先高保真 coupler）
- **本征门集扩展**：支持 iSWAP / √iSWAP 原生门
- **等价性验证**：编译前后幺正矩阵比较工具（已有 `sim/mpo.py` 基础）

### 工程质量
- **凭证管理**：`~/.quantum_hw/config.toml` 或 keyring 方案
- **异步任务**：`asyncio` 化的 submit/poll，支持批量并行提交
- **CLI 入口**：`python -m quantum_hw run --circuit ghz --qubits 6 --provider quafu`
- **CI/CD**：GitHub Actions + mock provider 自动回归测试
- **日志体系**：统一 `logging` 模块替代散布的 `print()`

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
