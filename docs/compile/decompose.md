# Decompose — 门分解

## 概览

- **模块**：`fieldqkit.compile.decompose`（约750 行）
- **作用**：将高阶门（三比特门、非本征两比特门）分解为单比特 + 本征两比特门的组合。
- **设计**：三比特门分解由 `ThreeQubitGateDecompose` TranspilerPass 完成；两比特门分解由独立函数完成，被 `TranslateToBasisGates` 调用。
- **依赖**：`u3_decompose`（U 分解）、`u_mat`（U 矩阵构造）

---

## 辅助函数

### `u_dot_u(u_info1, u_info2) -> tuple`

将两个 U 门信息元组相乘，返回等效的单个 U 门：

```python
u_dot_u(("u", θ1, φ1, λ1, qubit), ("u", θ2, φ2, λ2, qubit))
# → ("u", θ_new, φ_new, λ_new, qubit)
```

**约束：** 两个 U 门必须作用于同一比特（`u_info1[-1] == u_info2[-1]`，否则 `assert` 失败）。

**计算：** $U_2 \cdot U_1 \to u3\_decompose(U_2 \cdot U_1)$。

---

## 单比特门 → U 转换函数

将标准门转换为 `("u", θ, φ, λ, qubit)` 元组。所有函数签名为 `f(qubit) -> tuple`（参数化门额外接受角度）。

### 固定门

| 函数 | 门 | $\theta$ | $\phi$ | $\lambda$ |
|---|---|---:|---:|---:|
| `x2u` | X | $\pi$ | $\pi/2$ | $-\pi/2$ |
| `y2u` | Y | $\pi$ | $0$ | $0$ |
| `z2u` | Z | $0$ | $0$ | $\pi$ |
| `h2u` | H | $\pi/2$ | $0$ | $\pi$ |
| `s2u` | S | $0$ | $\pi/4$ | $\pi/4$ |
| `sdg2u` | S† | $0$ | $-\pi/4$ | $-\pi/4$ |
| `t2u` | T | $0$ | $\pi/8$ | $\pi/8$ |
| `tdg2u` | T† | $0$ | $-\pi/8$ | $-\pi/8$ |
| `sx2u` | √X | $\pi/2$ | $-\pi/2$ | $\pi/2$ |
| `sxdg2u` | √X† | $\pi/2$ | $\pi/2$ | $-\pi/2$ |

### 参数化门

| 函数 | 签名 | $\theta$ | $\phi$ | $\lambda$ |
|---|---|---:|---:|---:|
| `rx2u` | `(theta, qubit)` | $\theta$ | $-\pi/2$ | $\pi/2$ |
| `ry2u` | `(theta, qubit)` | $\theta$ | $0$ | $0$ |
| `rz2u` | `(theta, qubit)` | $0$ | $0$ | $\theta$ |

---

## 两比特门分解函数

所有函数共享统一签名：

```python
def xxx_decompose(
    qubit1: int,                    # 控制/第一比特
    qubit2: int,                    # 目标/第二比特
    convert_single_qubit_gate_to_u: bool,   # True → 单比特门转为 U
    two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"],  # 目标本征门
) -> list[tuple]
```

`convert_single_qubit_gate_to_u` 控制分解结果中的单比特门是保留原生形式（`"h"`, `"s"` 等）还是转为 `"u"` 门。

---

### 固定两比特门分解

| 函数 | 分解电路（以 CX 为例） | 说明 |
|---|---|---|
| `cx_decompose` | CX 原生 / H·CZ·H | CX 为 `"cx"` 基时直接返回；否则通过 CZ 分解 |
| `cz_decompose` | CZ 原生 / H·CX·H / 含 ECR/iSWAP 分解 | 核心分解函数，其他函数依赖它 |
| `cy_decompose` | S†·CX·S | 通过 CX 分解 |
| `swap_decompose` | CX·CX·CX 或 iSWAP·SX·iSWAP·SX·iSWAP·SX | 3 个 CX（或 3 个 iSWAP） |
| `iswap_decompose` | iSWAP 原生 / S·S·H·CX·CX·H | iSWAP 为基时直接返回 |
| `ecr_decompose` | ECR 原生 / S·SX·CX·X | ECR 为基时直接返回 |

**分解门数（两比特门计数，以不同 `two_qubit_gate_basis` 为目标）：**

| 源门 | → CX | → CZ | → iSWAP | → ECR |
|---|---:|---:|---:|---:|
| CX | 1 | 1 | 2 | 1 |
| CZ | 1 | 1 | 2 | 1 |
| CY | 1 | 1 | 2 | 1 |
| SWAP | 3 | 3 | 3 | 3 |
| iSWAP | 2 | 2 | 1 | 2 |
| ECR | 1 | 1 | 2 | 1 |

---

### 参数化两比特门分解

| 函数 | 签名多出的参数 | 分解电路（CX 基） |
|---|---|---|
| `rxx_decompose` | `theta: float` | H·H → CX → Rz(θ) → CX → H·H |
| `ryy_decompose` | `theta: float` | Rx(π/2)·Rx(π/2) → CX → Rz(θ) → CX → Rx(-π/2)·Rx(-π/2) |
| `rzz_decompose` | `theta: float` | CX → Rz(θ) → CX |
| `cp_decompose` | `theta: float` | Rz(θ/2)·Rz(θ/2) → CX → Rz(-θ/2) → CX |

所有参数化分解均使用 2 个两比特门（CX 或等效本征门）。

---

## 三比特门分解

### `ThreeQubitGateDecompose` 类

```python
class ThreeQubitGateDecompose(TranspilerPass):
    def __init__(self)
    def run(self, qc: QuantumCircuit) -> QuantumCircuit
```

**`run()` 方法：** 遍历 `qc.gates`，将三比特门替换为分解后的门序列，其他门保留。返回 `deepcopy` 的新线路。

**分解分派：**

| 输入门 | 调用函数 | 产生的 CX 数 |
|---|---|---:|
| `ccx` (Toffoli) | `ccx_decompose(c1, c2, t)` | 6 |
| `ccz` | `ccz_decompose(c1, c2, t)` | 6 |

---

### 三比特门分解函数

#### `ccx_decompose(control_qubit1, control_qubit2, target_qubit) -> list`

标准 Toffoli 分解：15 个门（6 CX + 7 T/T† + 2 H）。

#### `ccz_decompose(control_qubit1, control_qubit2, target_qubit) -> list`

CCZ 分解：15 个门（6 CX + 7 T/T† + 2 H），结构类似 CCX 但目标比特的 H 位置不同。

#### `ccx_decompose_mute_phase(control_qubit1, control_qubit2, target_qubit) -> list`

**相位近似版 Toffoli**：仅 7 个门（3 CX + 4 U），通过牺牲全局相位精度换取更少的两比特门。注意返回的门序列是**反转**的（`gates[::-1]`）。

---

## 示例

```python
from fieldqkit.compile.decompose import (
    cx_decompose, cz_decompose, swap_decompose, ccx_decompose,
    ThreeQubitGateDecompose, x2u, h2u, u_dot_u
)
from fieldqkit.circuit import QuantumCircuit

# 两比特门分解：CX → CZ 基
gates = cx_decompose(0, 1, convert_single_qubit_gate_to_u=True, two_qubit_gate_basis="cz")
print(f"CX→CZ 分解产生 {len(gates)} 个门: {gates}")

# SWAP → iSWAP 基
gates = swap_decompose(0, 1, convert_single_qubit_gate_to_u=False, two_qubit_gate_basis="iswap")
print(f"SWAP→iSWAP 分解产生 {len(gates)} 个门")

# 三比特门分解
qc = QuantumCircuit(3)
qc.ccx(0, 1, 2)
decomposer = ThreeQubitGateDecompose()
decomposed = decomposer.run(qc)
print(f"CCX 分解后门数: {len(decomposed.gates)}")

# 单比特 U 转换与合并
u1 = x2u(0)   # X 门的 U 表示
u2 = h2u(0)   # H 门的 U 表示
merged = u_dot_u(u1, u2)  # H·X 合并为单个 U
```

---

## 相关页面

- [编译模块总览](./README.md)
- [TranslateToBasisGates — 基门翻译](./translate.md)（调用本模块的两比特分解函数）
- [GateCompressor — 门压缩](./optimize.md)（分解后的门可进一步优化）
- [Transpiler — 编译流水线](./transpiler.md)
