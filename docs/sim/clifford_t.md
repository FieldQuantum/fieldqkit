# Clifford+T branching simulator

## 模块

- `fieldqkit.sim.clifford_t`

## 概览

该模块在 [Clifford 稳定子模拟器](clifford.md) 的基础上扩展支持任意非 Clifford 门——`t`/`tdg`、任意角度的 `rx`/`ry`/`rz`/`u`/`u3`、`rxx`/`ryy`/`rzz`、`cp` 等——采用 **海森堡 Pauli 分支演化**：

将可观测量保持为 Pauli 串的加权和

$$
O = \sum_i c_i\, P_i
$$

并逐门反向共轭。对于 Clifford 门，每一项仍是单个 Pauli（走稳定子快通道，O(n) 一次）；对于非 Clifford 门，单个 Pauli 会分裂为若干 Pauli 的线性组合（例如 `T^\dagger X T = (X - Y)/\sqrt{2}`），算法对所有分支保持精确的复系数与去重。

设非 Clifford 门数量为 `k`，最坏情况下分支数为 `O(4^k)`；当 `k` 较小或 Pauli 串相互合并时实际开销远低于上界。复杂度上界 `O(4^k · g · n)`，与状态向量法 `O(g · 2^n)` 互补：

| 场景 | 推荐 |
|---|---|
| 纯 Clifford | [`simulate_clifford_expectation(s)`](clifford.md) |
| 少量非 Clifford（`k ≲ 15`），比特数大（`n` 大） | 本模块 |
| 比特数小（`n ≲ 18`），非 Clifford 门多 | `simulate_statevector` + `expectation_pauli` |

## 支持的门

- 所有 Clifford 门（直接继承 [`clifford.md`](clifford.md) 列表）；
- 单比特任意旋转：`rx`, `ry`, `rz`, `u`/`u3`, `t`, `tdg`；
- 两比特旋转：`rxx`, `ryy`, `rzz`, `cp`（分解为 `rzz`+`rz` 后处理）。

## 关键函数

### `simulate_clifford_t_expectation(qc, pauli, *, num_qubits=None, max_terms=None) -> float`

- 反向逐门将 Pauli `pauli` 共轭到电路起点；非 Clifford 门触发分支。
- 终态读取 `⟨0|·|0⟩`：累加所有「只含 `I/Z`」的分支项系数实部。
- `max_terms`：若给出则在分支总数超过该阈值时抛出 `RuntimeError("...max_terms...")`，用于守护内存与时间开销。
- `num_qubits`：稀疏物理布局场景下显式声明 Pauli 串长度。

### `simulate_clifford_t_expectations(qc, paulis, *, num_qubits=None, max_terms=None) -> Dict[str, float]`

- 批量版本。

### `count_t_gates(qc) -> int`

- 统计 `t`/`tdg` 门数量。

### `count_non_clifford_gates(qc) -> int`

- 统计所有非 Clifford 门数量（含任意旋转），可用于估算分支上界。

## 算法说明

- 状态用 `Dict[Tuple[str, ...], complex]` 维护：键是 Pauli 串（长度 `n` 的 `I/X/Y/Z` 元组），值是复系数。
- 单比特旋转分支公式（围绕 `Z` 轴的例子；`X`/`Y` 同理）：

  $$
  R_z^\dagger(\theta) X R_z(\theta) = \cos\theta\, X - \sin\theta\, Y
  $$

  $$
  R_z^\dagger(\theta) Y R_z(\theta) = \cos\theta\, Y + \sin\theta\, X
  $$

- `T` / `T†` 分支：

  $$
  T^\dagger X T = \tfrac{1}{\sqrt 2}(X - Y),\quad T^\dagger Y T = \tfrac{1}{\sqrt 2}(X + Y),\quad T^\dagger Z T = Z
  $$

  （`Tdg` 翻转 `Y` 上的符号。）

- `U(θ, φ, λ)` 直接构造对应 SU(2) 矩阵，按矩阵元素分解为 4 个 Pauli 的线性组合。
- 每步处理后对同 Pauli 串系数做去重合并，避免分支爆炸。
- 每应用一个 Clifford 门时优先走稳定子快通道（`conjugate_clifford_gate`），仅当抛出 `CliffordError` 才走分支路径。

## 常见报错

- `RuntimeError("...max_terms exceeded...")`：分支总数超过 `max_terms`，应增大上限或改用 `simulate_statevector`。
- `ValueError`：Pauli 串长度与 `num_qubits` 不一致；不支持的门。

## 关联用法

- 在 `fieldqkit.algorithms.optimizer_utils._ideal_expectations_clifford_aware` 中作为 Clifford fitting 校准的可扩展理想期望路径：先尝试 `sim.clifford` 的稳定子快通道，遇到非 Clifford 门时回退到本模块的分支扩展，仍不可行时再退回 statevector。
- 闭环示例见 [`examples/demo_clifford_fitting.ipynb`](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_clifford_fitting.ipynb)（8 比特 `Baihua` 真机线路 + `run_auto` 自动校准）。

## 相关

- [`clifford.md`](clifford.md)：纯 Clifford 快通道；
- [`statevector.md`](statevector.md)：稠密状态向量模拟；
- [`interface.md`](interface.md)：自动后端选择。
