# GateCompressor — 门压缩优化

## 概览

- **模块**：`quantum_hw.compile.optimize`（约820 行）
- **作用**：通过合并、对消和重排门来减少线路中的门数量，降低噪声影响。
- **继承**：`TranspilerPass`（实现 `run()` 方法）
- **依赖**：`qc2dag` / `dag2qc`（DAG 构建）、`u3_decompose`（合并后分解）、`gate_matrix_dict`（矩阵查表）

---

## 模块级常量

| 常量 | 类型 | 值 | 说明 |
|---|---|---|---|
| `_DIAGONAL_1Q_GATES` | `frozenset` | `{id, z, s, sdg, t, tdg, rz}` | 计算基对角单比特门（互相对易） |
| `_NON_REORDERABLE` | `frozenset` | `{barrier, measure, reset, delay}` | 功能性指令，不可重排 |

---

## 类签名

```python
class GateCompressor(TranspilerPass):
    def __init__(self, convert_single_qubit_gate_to_u: bool = True)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `convert_single_qubit_gate_to_u` | `bool` | `True` | 合并后的单比特门是否输出为 `u` 门；为 `False` 时分解为 `rz`/`ry` 序列 |

| 属性 | 类型 | 说明 |
|---|---|---|
| `compressible_gates` | `list[str]` | 14 个可压缩门名列表（`x`, `y`, `z`, `h`, `cx`, `cy`, `cz`, `swap`, `ecr`, `ccx`, `ccz`, `rxx`, `ryy`, `rzz`）。单比特参数门（`rx`/`ry`/`rz`/`u`）和非自逆 1Q 门（`s`/`sdg`/`t`/`tdg`/`sx`/`sxdg`）刻意排除，由 `merge_single_qubit_runs` 的矩阵累乘处理 |
| `_idx` | `int` | DAG 压缩时新节点的自增 ID，起始值 `1000000` |
| `_single_qubit_gates` | `set[str]` | 所有单比特门名的合集（固定 + 参数化） |
| `dag` | `nx.DiGraph` | 运行时 DAG 实例（在 `run()` 中赋值） |

**类变量：**

```python
_SELF_INVERSE_2Q = frozenset({'cx', 'cy', 'cz', 'swap', 'ecr'})
```

自逆两比特门集合：$G \cdot G = I$。

---

## `run(...)` 方法

**签名：**

```python
def run(self, qc: QuantumCircuit) -> QuantumCircuit
```

执行完整优化流水线，按以下顺序：

```
输入 qc
  │
  ▼
① remove_identity_gates ─── 移除 I 门
  │
  ▼
② commutation_reorder ───── 冒泡重排
  │
  ▼
③ merge_single_qubit_runs ─ 合并单比特连续门
  │
  ▼
④ cancel_two_qubit_pairs ── 自逆 2Q 门对消
  │
  ▼
⑤ DAG 压缩循环 ──────────── 相邻同类门合并
  │
  ▼
输出 new_qc
```

**返回值：** 优化后的 `QuantumCircuit`（`deepcopy`，`qubits` 属性继承自输入）。

---

## 优化 Pass 详解

### `remove_identity_gates(qc) -> QuantumCircuit`

逐门检查所有参数化门（单比特和两比特）的矩阵是否为单位矩阵 $I$。若 `np.allclose(mat, I)` 为 `True`，则丢弃该门。非参数化门和功能性指令保留不变。

---

### `commutation_reorder(qc) -> QuantumCircuit`

**目标：** 将单比特门向前（左）冒泡移动，使同比特单比特门聚集，为后续 `merge_single_qubit_runs` 创造更多合并机会。

**冒泡规则：**
- 从左到右扫描每个单比特门
- 向前移动，跳过与其对易的门
- 遇到以下情况时停止：
  1. 同比特的另一个单比特门（合并目标，停在其后）
  2. 不对易的门
  3. 功能性指令（`barrier` / `measure` / `reset` / `delay`）

**实现：** 冒泡排序风格的原地交换。

---

### `merge_single_qubit_runs(qc) -> QuantumCircuit`

将连续作用于同一比特的单比特门序列累乘为一个 $2\times 2$ 矩阵：

$$U_{\text{merged}} = U_n \cdot U_{n-1} \cdots U_2 \cdot U_1$$

然后通过 `u3_decompose` 转为单个 `("u", θ, φ, λ, qubit)` 门。

**特殊情况：**
- 若 $U_{\text{merged}} \approx I$（`np.allclose`），整段门消除
- 若某个门含符号参数（`str` 类型），该门单独输出，不参与合并
- 单门不触发合并

---

### `cancel_two_qubit_pairs(qc) -> QuantumCircuit`

对消自逆两比特门对。

**算法：**

```
repeat:
  对每个自逆 2Q 门 G[i]：
    向后搜索匹配的 G[j]（同门名 + 同比特）
    检查中间所有门 G[k] 是否与 G[i] 对易：
      若不对易 → 停止搜索
      若对易 → 继续
    找到匹配：G[i] 和 G[j] 同时移除，restart
  无更多匹配 → 结束
```

**比特对称性：** `cz` 和 `swap` 门的比特顺序不敏感（即 `CZ(a,b) == CZ(b,a)`）。

---

### DAG 压缩循环

将线路转换为 DAG，反复寻找并合并相邻同类门，直到无更多可合并的门对。

**相邻判定 `is_adjacent_gates(node1, node2)`：**
- 同门名（`node1.split("_")[0] == node2.split("_")[0]`）
- 门名在 `compressible_gates` 列表中
- 同比特（`qubits1 == qubits2`）
- DAG 直接边连接（`out_edges(node1) == in_edges(node2)`）

**合并分派 `run_compress_once(node1, node2)`：**

| 门类型 | 合并方法 | 行为 |
|---|---|---|
| 固定单比特门 | `compress_adjacent_single_qubit_gates` | 两门对消（自逆），删除两节点 |
| 参数单比特门 | `compress_adjacent_single_parameter_qubit_gates` | 参数相加（或 `u` 门矩阵乘法 + `u3_decompose`）；若结果为 $I$ 则消除 |
| 固定两比特门 | `compress_adjacent_two_qubit_gates` | 自逆对消，删除两节点 |
| 参数两比特门 | `compress_adjacent_two_qubit_parameter_gates` | 参数相加；若结果为 $I$ 则消除 |
| 三比特门 | `compress_adjacent_three_qubit_gates` | 自逆对消（如 CCX·CCX = I） |

**DAG 更新操作：** 删除两个旧节点 → 添加新节点（如有） → 重新连接前驱/后继边。

---

## 对易判断

### `_check_commutation(gate_info1, gate_info2) -> bool`

类方法（`@classmethod`）。优先级递减的快速路径：

1. **功能性指令** → `False`（永不对易）
2. **不相交比特** → `True`
3. **两个对角单比特门**（均在 `_DIAGONAL_1Q_GATES` 中）→ `True`
4. **对角单比特 + CZ** → `True`（CZ 也对角）
5. **CZ + CZ** → `True`
6. **矩阵回退**：获取两门的矩阵，扩展到联合比特空间，检查 $[A, B] = AB - BA = 0$
   - 若任一门矩阵不可获取（符号参数）→ `False`（保守假设）

---

## 静态辅助方法

| 方法 | 签名 | 说明 |
|---|---|---|
| `_get_gate_qubits` | `(gate_info) -> list[int]` | 返回门信息元组中的比特索引列表（1Q/2Q/3Q 分别取最后 1/2/3 个元素） |
| `_get_any_gate_matrix` | `(gate_info) -> np.ndarray \| None` | 获取任意类型门的矩阵。含符号参数时返回 `None` |
| `_expand_matrix` | `(gate_mat, positions, n_total) -> np.ndarray` | 将小矩阵扩展到 $2^n \times 2^n$ 联合比特空间 |
| `_gate_matrix` | `(gate_info) -> np.ndarray \| None` | 获取单比特门矩阵（仅 1Q 门） |
| `_gate_qubit` | `(gate_info) -> int` | 返回单比特门的比特索引（`gate_info[-1]`） |

---

## `idx` 属性

```python
@property
def idx(self) -> int
```

DAG 压缩时为新节点生成唯一 ID。每次访问自增 1，起始值 `1000000`，避免与原始节点 ID 冲突。

---

## 示例

```python
from quantum_hw.compile.optimize import GateCompressor
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.h(0)        # H·H = I → 被消除
qc.cx(0, 1)
qc.cx(0, 1)    # CX·CX = I → 被对消
qc.rz(0.1, 2)
qc.rz(0.2, 2)  # 连续 Rz → 合并为 Rz(0.3)

compressor = GateCompressor()
optimized = compressor.run(qc)
print(f"优化前门数: {len(qc.gates)}")
print(f"优化后门数: {len(optimized.gates)}")

# 单独使用某个优化步骤
qc2 = compressor.commutation_reorder(qc)
qc3 = compressor.merge_single_qubit_runs(qc2)
```

---

## 相关页面

- [编译模块总览](./README.md)
- [Transpiler — 编译流水线](./transpiler.md)
- [TranslateToBasisGates — 基门翻译](./translate.md)
- [DAG — 有向无环图工具](./dag.md)（DAG 压缩依赖）
- [Decompose — 门分解](./decompose.md)
