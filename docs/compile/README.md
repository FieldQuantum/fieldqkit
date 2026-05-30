# 编译 / 转译模块

## 概览

- **模块路径**：`fieldqkit.compile`
- **作用**：将用户构建的逻辑量子线路编译为可在物理芯片上执行的本征门序列。
- **对外导出**：`Transpiler`（编译流水线管理器）
- **内部 Pass**：`ThreeQubitGateDecompose` → `Layout` + `SabreRouting` → `TranslateToBasisGates` → `GateCompressor` → `DynamicalDecoupling`
- **基类**：`TranspilerPass`（`basepasses.py`，约30 行），定义 `run(qc) -> QuantumCircuit` 抽象方法

---

## 编译流水线

```
输入 QuantumCircuit（任意门集）
  │
  ├─ [1] ThreeQubitGateDecompose    CCX/CCZ → 1Q+2Q 门分解
  │                                  CCX: 6 CX, CCZ: 6 CX
  │
  ├─ [2] Layout                     逻辑比特 → 物理比特映射
  │       │                          保真度优先选择 + 电路感知评分
  │       │                          circuit_aware: score = f̄ − 0.05 × normalized_cost
  │       │
  │       └── SabreRouting          SWAP 插入（SABRE 启发式）
  │                                  4 种启发式函数（推荐 lookahead_decay）
  │                                  支持噪声感知（-log(f) Floyd-Warshall）
  │                                  支持多随机初始映射试验 (n_trials)
  │
  ├─ [3] TranslateToBasisGates      所有门 → {U, 本征2Q门} 门集
  │                                  支持 CZ/CX/iSWAP/ECR 四种基
  │                                  单比特门统一为 u(θ,φ,λ) 或保留原生
  │                                  支持字符串符号参数（VQE/QML）
  │
  ├─ [4] GateCompressor             门优化五步流水线：
  │                                  ① remove_identity_gates
  │                                  ② commutation_reorder（冒泡重排）
  │                                  ③ merge_single_qubit_runs（矩阵累乘→U）
  │                                  ④ cancel_two_qubit_pairs（自逆门对消）
  │                                  ⑤ DAG 压缩循环（同类相邻门合并）
  │
  └─ [5] DynamicalDecoupling        空闲时隙填充 DD 序列
  │                                  XY4: XYXY/YXYX 交替 4 脉冲
  │                                  CPMG: XX 2 脉冲
  │                                  支持右对齐 + barrier 前插入控制
  │
  ▼
输出 QuantumCircuit（本征门集）
  附带：logical_to_physical / physical_to_logical 映射
```

---

## 各 Pass 关键特性

| Pass | 核心类/函数 | 输入 | 输出 | 关键参数 |
|---|---|---|---|---|
| 分解 | `ThreeQubitGateDecompose` | 含 3Q 门的 QC | 仅 1Q+2Q 门的 QC | 无 |
| 布局 | `Layout` | 芯片拓扑 + 校准数据 | 物理比特子图 | `priority_qubits`, `circuit_aware` |
| 路由 | `SabreRouting` | 逻辑 QC + 子图 | 物理 QC（含 SWAP） | `heuristic`, `noise_aware`, `n_trials` |
| 翻译 | `TranslateToBasisGates` | 任意门集 QC | 本征门集 QC | `two_qubit_gate_basis`, `convert_single_qubit_gate_to_u` |
| 优化 | `GateCompressor` | QC | 优化后 QC | 无（自动执行全部优化） |
| 去耦 | `DynamicalDecoupling` | QC + 门耗时 | 含 DD 的 QC | `sequence`, `align_right` |

---

## 无后端退化行为

当 `Transpiler` 未绑定 `Backend` 时，跳过 Layout 和 DD，使用线性拓扑与 CX 基：

| 步骤 | 有后端 | 无后端 |
|---|---|---|
| ThreeQubitGateDecompose | ✅ | ✅ |
| Layout | ✅ | ❌ 跳过（使用线性拓扑） |
| SabreRouting | ✅ | ✅（线性拓扑） |
| TranslateToBasisGates | ✅ | ✅（CX 基、保留原生单比特门） |
| GateCompressor | ✅ | ✅ |
| DynamicalDecoupling | ✅ | ❌ 跳过 |

---

## 页面导航

- [Transpiler](./transpiler.md) — 编译流水线管理器（Pass 编排与执行）
- [Layout](./layout.md) — 物理比特布局选择（保真度优先 + 电路感知评分）
- [SabreRouting](./routing.md) — SABRE 启发式路由（SWAP 插入 + 噪声感知）
- [TranslateToBasisGates](./translate.md) — 门集翻译（4 种两比特门基 + 符号参数）
- [GateCompressor](./optimize.md) — 门优化与压缩（5 步流水线 + DAG 压缩）
- [DynamicalDecoupling](./schedule.md) — 动力学去耦（XY4 / CPMG）
- [ThreeQubitGateDecompose + 分解函数](./decompose.md) — 三比特门分解 + 10 种两比特门分解 + 单比特→U 转换
- [DAG 工具](./dag.md) — 有向无环图转换、交互图、连通分量分割、可视化

---

## 源文件统计

| 文件 | 行数 | 核心类/函数 |
|---|---|---|
| `transpiler.py` | 182 | `Transpiler` |
| `layout.py` | 674 | `Layout` |
| `routing.py` | 655 | `SabreRouting` |
| `optimize.py` | 819 | `GateCompressor` |
| `decompose.py` | 712 | `ThreeQubitGateDecompose` + 10 个两比特门分解 + 13 个 1Q→U 转换 |
| `schedule.py` | 211 | `DynamicalDecoupling` |
| `translate.py` | 160 | `TranslateToBasisGates` |
| `dag.py` | 223 | `qc2dag` / `dag2qc` / `qc2graph` / `split_qubits` |
| `basepasses.py` | 29 | `TranspilerPass`（ABC） |
| **合计** | **3,665** | |

---

## 相关页面

- [docs 索引](../README.md)
- [QuantumHardwareClient._transpile_with_backend](../api/QuantumHardwareClient.md) — API 层如何调用编译流水线
