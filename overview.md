# quantum-hw: 统一量子硬件控制与算法框架

> **版本**: 0.1.0 · **许可证**: MIT · **Python**: ≥ 3.9  
> **作者**: FieldQuantum · **仓库**: https://github.com/FieldQuantum/Quantum_control

---

## 一、项目定位

**quantum-hw** 是一个面向真实量子硬件的统一编程框架，核心目标是：

- **一套 API，多平台硬件**：通过统一的 `QuantumHardwareClient` 接口，屏蔽四大国产量子云平台的差异
- **从线路设计到结果分析的完整工作流**：涵盖线路构建 → 编译优化 → 硬件执行 → 误差缓解 → 算法求解全链路
- **本地仿真与硬件执行双路径**：所有变分算法同时支持 PyTorch 自动微分（仿真）和参数平移规则（硬件）

---

## 二、整体架构

```
quantum_hw/
├── api/                  硬件抽象层
│   ├── client.py         → QuantumHardwareClient（统一入口）
│   ├── backend.py        → 后端拓扑 / 硬件画像
│   ├── task.py           → 任务提交与结果查询
│   └── quantum_platform/ → 四大平台适配器
│       ├── quafu.py          夸父
│       ├── tianyan.py        天衍
│       ├── guodun.py         国盾
│       └── tencent.py        腾讯
│
├── circuit/              量子线路表示
│   ├── quantumcircuit.py → QuantumCircuit 类（门操作 API）
│   ├── qasm2.py / qasm3.py → OpenQASM 2.0/3.0 解析与导出
│   ├── qasm_to_qcis.py  → QASM → QCIS 硬件指令翻译
│   ├── matrix.py         → 门矩阵定义（NumPy）
│   └── render.py         → ASCII 线路可视化
│
├── compile/              编译与转译流水线
│   ├── transpiler.py     → Transpiler 编排器 + PassManager
│   ├── decompose.py      → 三量子比特门分解
│   ├── layout.py         → 保真度感知的量子比特布局
│   ├── routing.py        → SABRE 路由（噪声感知 SWAP 插入）
│   ├── translate.py      → 翻译至原生基门集 {U, CZ}
│   ├── optimize.py       → 门压缩（对易重排 + Clifford 合并）
│   └── schedule.py       → 动态解耦（XY4 / CPMG）
│
├── core/                 核心工具
│   ├── circuits.py       → 典型线路构造器（GHZ / Cluster / QFT / Ising）
│   ├── observables.py    → Pauli 字符串解析与测量基映射
│   ├── readout.py        → 读出误差缓解（无偏/有偏）
│   ├── zne.py            → 零噪声外推（ZNE）
│   └── types.py          → 类型安全的结果数据类
│
├── calibration/          硬件标定
│   ├── readout.py        → 读出混淆矩阵标定（含缓存）
│   ├── rb.py             → 二量子比特随机基准测试（RB）
│   └── tomography.py     → 二量子比特过程层析
│
├── algorithms/           量子算法与机器学习
│   ├── vqe.py            → VQE（变分量子本征求解器）
│   ├── qaoa.py           → QAOA（量子近似优化）
│   ├── shadow.py         → 经典影子层析
│   ├── qml.py            → 参数化量子线路分类器 / 无监督 QNN
│   ├── ansatz_templates.py → 硬件高效拟设 + UCC 拟设
│   └── circuit_compression.py → MPS/MPO 混合线路压缩
│
└── sim/                  PyTorch 仿真器
    ├── statevector.py    → 全状态向量仿真
    ├── mps.py            → 矩阵积态（MPS）近似仿真
    ├── mpo.py            → 矩阵积算符（MPO）演化
    └── interface.py      → 阈值自动分派（≤16 qubit→精确，>16→MPS）
```

---

## 三、核心能力

### 3.1 多平台硬件统一接入

| 平台 | 适配器 | 说明 |
|------|--------|------|
| **夸父（Quafu）** | `quafu.py` | 北京量子院 |
| **天衍（TianYan）** | `tianyan.py` | 中电信量子 |
| **国盾（GuoDun）** | `guodun.py` | 国盾量子 |
| **腾讯（Tencent）** | `tencent.py` | 腾讯 |

- 自动发现可用硬件、量子比特数、拓扑连通性
- YAML 凭证管理 + 环境变量回退
- 异步任务提交与轮询

### 3.2 自动编译流水线

Transpiler 按序执行以下 Pass：

1. **三量子比特门分解** — CCX / CSWAP 拆解为 1 & 2 量子比特门
2. **布局选择** — 保真度感知的最优子图搜索
3. **SABRE 路由** — 噪声感知 SWAP 插入，多轮试探取最优
4. **基门翻译** — 映射至 {U, CZ} 原生门集
5. **门压缩** — 对易重排 + 相邻 Clifford 门合并
6. **动态解耦** — 空闲窗口插入 XY4/CPMG 脉冲序列

### 3.3 误差缓解

| 技术 | 模块 | 要点 |
|------|------|------|
| 读出标定 | `calibration/readout.py` | 逐量子比特混淆矩阵，带时间戳缓存 |
| 零噪声外推（ZNE） | `core/zne.py` | CZ 三倍插入 + Richardson 线性外推 |
| 读出纠正 | `core/readout.py` | 无偏/有偏两种矫正模式 |
| 动态解耦（DD） | `compile/schedule.py` | 编译期自动注入 |

### 3.4 变分量法算法

| 算法 | 关键特性 |
|------|----------|
| **VQE** | Ising / Heisenberg / XXZ / XY / 自定义哈密顿量；硬件高效拟设 + UCC 拟设；Adam 优化 |
| **QAOA** | MaxCut / 通用 Ising 问题；p 层拟设，自动构造混合算符 |
| **经典影子层析** | 高效期望值估计；中位数均值估计器（抗重尾噪声）；ZNE 集成 |
| **量子机器学习** | 有监督 PQC 分类器（angle / IQP 编码）；无监督量子 Boltzmann 机 |

> 所有变分算法均支持**双梯度路径**：仿真器侧 autograd / 硬件侧 parameter-shift rule，代码一致。

### 3.5 PyTorch 仿真引擎

- **状态向量** (≤16 qubit)：精确模拟，支持 GPU 加速
- **MPS** (>16 qubit)：张量网络近似，可配置截断维度
- **MPO**：算符演化，用于过程层析
- 全流程支持 `torch.autograd` 自动微分

---

## 四、特色亮点

1. **国产量子云原生适配**  
   面向国内四大主流量子云平台深度定制，提供统一的凭证管理、硬件发现与任务调度。

2. **仿真-硬件一体化设计**  
   同一份算法代码可无缝切换仿真/真机执行，仿真使用 PyTorch autograd 求梯度，硬件使用 parameter-shift rule，无需修改算法逻辑。

3. **噪声感知的全链路编译**  
   SABRE 路由主动规避低保真度 coupler；编译期自动注入动态解耦；支持符号化编译（拟设模板编译一次，参数多次实例化）。

4. **集成式误差缓解**  
   读出标定、ZNE、动态解耦三重缓解手段内置于执行流程，用户通过参数开关即可启用。

5. **张量网络仿真核心**  
   基于 PyTorch 的 MPS/MPO 仿真器，兼顾大量子比特系统的可扩展性与自动微分需求；线路压缩算法可用于 ansatz 简化。

6. **完整标定工具链**  
   内置二量子比特 RB、过程层析、读出标定管理器，带缓存机制避免重复执行。

7. **类型安全的结果系统**  
   所有输出均为结构化数据类（`RunResult`, `VQEResult`, `QAOAResult`, `ShadowResult`, `QMLResult`），便于下游分析与序列化。

---

## 五、依赖关系

| 分类 | 依赖 | 用途 |
|------|------|------|
| 核心 | `numpy≥1.24`, `scipy≥1.10` | 数值计算 |
| 图论 | `networkx≥3.0` | 硬件拓扑建模与路由 |
| 线路 | `openqasm3[parser]≥0.5` | OpenQASM 解析 |
| 通信 | `requests≥2.31`, `pyyaml≥6.0` | 云平台 API 与配置 |
| 可视化 | `matplotlib≥3.7` | 绘图 |
| 仿真（可选） | `torch≥2.1` | 状态向量 / MPS 仿真 + autograd |

---

## 六、示例覆盖

| 示例 | 内容 |
|------|------|
| `demo_full.ipynb` | 端到端工作流：连接硬件 → 线路执行 → 误差缓解 |
| `demo_circuit_core.ipynb` | 线路构建、门操作、状态向量仿真 |
| `demo_vqe.ipynb` | VQE 求解哈密顿量基态能量 |
| `demo_qaoa.ipynb` | QAOA 求解 MaxCut 问题 |
| `demo_shadow.ipynb` | 经典影子层析协议 |
| `demo_readout_zne.ipynb` | 读出标定与 ZNE 工作流 |
| `demo_qml.ipynb` / `demo_qml_iris.ipynb` | 有监督量子机器学习 |
| `demo_qnn_unsupervised.ipynb` | 无监督量子神经网络 |
| `demo_backend.ipynb` | 硬件发现与后端枚举 |

---

## 七、快速上手

```python
from quantum_hw import QuantumHardwareClient, build_ghz

# 1. 初始化客户端
client = QuantumHardwareClient()

# 2. 构建线路
circuit = build_ghz(num_qubits=6)

# 3. 在真机上运行（自动编译 + 误差缓解）
result = client.run_auto(
    circuit=circuit,
    num_qubits=6,
    shots=8192,
    observables=["IIZZII", "ZZIIII"],
    readout_mitigation=True,
)

# 4. 查看结果
print(result.observable_estimates)
```

```bash
# 安装
pip install quantum-hw          # 核心功能
pip install quantum-hw[sim]     # 含 PyTorch 仿真
pip install quantum-hw[full]    # 全部可选依赖
```
