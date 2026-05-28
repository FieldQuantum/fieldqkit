# TranslateToBasisGates — 本征门翻译

## 概览

- **模块**：`quantum_hw.compile.translate`（约170 行）
- **作用**：将线路中所有门翻译为目标芯片的本征门集（单比特统一为 U 门，两比特统一为指定本征门）。
- **继承**：`TranspilerPass`（实现 `run()` 方法）
- **依赖**：`decompose` 模块（所有两比特门分解函数）、`u3_decompose`（矩阵→U 分解）

---

## 类签名

```python
class TranslateToBasisGates(TranspilerPass):
    def __init__(
        self,
        convert_single_qubit_gate_to_u: bool = True,
        two_qubit_gate_basis: Literal["cz", "cx", "iswap", "ecr"] = "cz",
    )
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `convert_single_qubit_gate_to_u` | `bool` | `True` | 是否将所有单比特门统一为 `("u", θ, φ, λ, qubit)` 形式 |
| `two_qubit_gate_basis` | `str` | `"cz"` | 目标两比特本征门基 |

---

### 初始化属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `convert_single_qubit_gate_to_u` | `bool` | 保存的单比特门转换标志 |
| `two_qubit_gate_basis` | `str` | 保存的两比特门基名称 |

---

## 支持的两比特门基

| 基 | 说明 | 典型硬件 |
|---|---|---|
| `"cz"` | 受控 Z 门（默认） | Quafu、TianYan、GuoDun 等超导芯片 |
| `"cx"` | 受控 NOT 门 | IBM Quantum |
| `"iswap"` | iSWAP 门 | Google Sycamore |
| `"ecr"` | Echoed Cross-Resonance 门 | IBM Eagle |

---

## `run(...)` 方法

**签名：**

```python
def run(self, qc: QuantumCircuit) -> QuantumCircuit
```

**返回值：** 翻译后的 `QuantumCircuit`（`deepcopy`）。

**异常：**
- `TypeError`：遇到不支持的两比特门（如未列出的门名）时抛出，提示 `"Try kak please"`
- `TypeError`：遇到完全不认识的门名时抛出

---

### 翻译规则（详细分派表）

#### 固定单比特门

| 条件 | 行为 |
|---|---|
| `convert_single_qubit_gate_to_u=True` | 查 `gate_matrix_dict` 获取矩阵 → `u3_decompose` → `("u", θ, φ, λ, qubit)` |
| `convert_single_qubit_gate_to_u=False` | 原样保留（如 `("h", 0)` → `("h", 0)`） |

支持的门：`id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`

---

#### 参数化单比特门

当 `convert_single_qubit_gate_to_u=True` 时：

| 门 | 处理方式 |
|---|---|
| `u` | 保留原样 |
| `rx(θ)` 且 θ 为 `str` | → `("u", θ, -π/2, π/2, qubit)` |
| `ry(θ)` 且 θ 为 `str` | → `("u", θ, 0, 0, qubit)` |
| `rz(θ)` 且 θ 为 `str` | → `("u", 0, 0, θ, qubit)` |
| 其他参数门（数值参数） | 矩阵计算 → `u3_decompose` |

当 `convert_single_qubit_gate_to_u=False` 时：原样保留。

> **符号参数处理**：当参数为字符串（符号参数，如 VQE/QML 中的可训练参数）时，使用解析规则直接映射而非矩阵计算。

---

#### 固定两比特门

| 输入门 | 调用的分解函数 |
|---|---|
| `cz` | `cz_decompose(q1, q2, ...)` |
| `cx` | `cx_decompose(q1, q2, ...)` |
| `swap` | `swap_decompose(q1, q2, ...)` |
| `iswap` | `iswap_decompose(q1, q2, ...)` |
| `ecr` | `ecr_decompose(q1, q2, ...)` |
| `cy` | `cy_decompose(q1, q2, ...)` |
| 其他 | `TypeError` |

---

#### 参数化两比特门

| 输入门 | 调用的分解函数 |
|---|---|
| `rxx` | `rxx_decompose(θ, q1, q2, ...)` |
| `ryy` | `ryy_decompose(θ, q1, q2, ...)` |
| `rzz` | `rzz_decompose(θ, q1, q2, ...)` |

---

#### 功能性指令

`barrier`, `measure`, `reset`, `delay` → 原样保留，不做翻译。

---

## 示例

```python
from quantum_hw.compile.translate import TranslateToBasisGates
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.swap(1, 2)
qc.rz(0.5, 0)

# 默认翻译（CZ 基 + 单比特统一为 U）
translator = TranslateToBasisGates()
translated = translator.run(qc)
# h → u, cx → H·CZ·H, swap → 3×CZ, rz → u

# IBM 风格翻译（CX 基，保留单比特原生门）
translator2 = TranslateToBasisGates(
    convert_single_qubit_gate_to_u=False,
    two_qubit_gate_basis="cx"
)
translated2 = translator2.run(qc)
# h → h, cx → cx, swap → 3×CX, rz → rz

# 符号参数支持
qc2 = QuantumCircuit(2)
qc2.rx("theta_0", 0)  # 字符串参数
translator.run(qc2)    # → ("u", "theta_0", -π/2, π/2, 0)
```

---

## 相关页面

- [编译模块总览](./README.md)
- [Decompose — 门分解](./decompose.md)（两比特门分解实现）
- [GateCompressor — 门压缩](./optimize.md)（翻译后的进一步优化）
- [Transpiler — 编译流水线](./transpiler.md)
