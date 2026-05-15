# Clifford stabilizer simulator

## 模块

- `quantum_hw.sim.clifford`

## 概览

该模块提供基于 **海森堡演化（Heisenberg picture）的 Pauli 共轭** 的纯 Clifford 模拟器。

对于只包含 Clifford 门的线路，任意 Pauli 算符 `P` 在演化后仍是单个 Pauli（最多差一个 ±1 符号）：

$$
U^\dagger P\, U = (-1)^s\, P'
$$

由此可以在 `O(g \cdot n)` 时间内（其中 `g` 是门数，`n` 是比特数）逐门反向共轭可观测量，再读出零态期望——完全避免显式构造 `2^n` 维态向量。

适用场景：

- Clifford 拟合校准（`build_clifford_fit_map`）的理想期望计算；
- 在大比特数（10+）下对纯 Clifford 子线路进行可观测量验证；
- 作为 Clifford+T 分支模拟器的快速通道。

## 支持的门

| 类别 | 门 |
|---|---|
| 单比特 Clifford | `h`, `s`, `sdg`, `x`, `y`, `z`, `sx`, `sxdg`, `id` |
| 两比特 Clifford | `cx`/`cnot`, `cz`, `swap` |
| 参数化（仅当角度为 `π/2` 的整数倍时） | `rx`, `ry`, `rz`, `u`/`u3` |

任何非 Clifford 门（如 `t`, `tdg`，或角度非 `π/2` 倍数的旋转）会触发 `CliffordError`；此时应改用 `simulate_clifford_t_expectation(s)`。

## 关键函数

### `simulate_clifford_expectation(qc, pauli, *, num_qubits=None) -> float`

- 反向逐门将 Pauli `pauli` 共轭到电路起点，再读取 `⟨0|P|0⟩`（仅 `I/Z` 字符的 Pauli 给出 `±1`，否则为 0）。
- `num_qubits`：若给出则覆盖 `qc.nqubits`，用于稀疏物理比特布局后再编译到致密索引时显式声明可观测量长度。
- 非 Clifford 门抛出 `CliffordError`。

### `simulate_clifford_expectations(qc, paulis, *, num_qubits=None) -> Dict[str, float]`

- 批量版本，对同一线路下的一组 Pauli 串返回 `{pauli: expectation}`。

### `is_clifford_circuit(qc) -> bool`

- 返回线路是否完全由可识别的 Clifford 操作构成。可在调用 `simulate_clifford_expectation(s)` 之前用作守护判断。

### `conjugate_clifford_gate(p, gate_info) -> int`

- 给定一个 Clifford 门 `gate_info` 与当前 Pauli 模式 `p`（长度为 `n` 的 `I/X/Y/Z` 列表），就地将 `p` 替换为共轭后的 Pauli 模式，并返回整体符号（`+1` 或 `-1`）。非 Clifford 门抛出 `CliffordError`。

## 算法说明

- 单比特表通过 `_SINGLE_QUBIT_TABLES` 静态查表；
- 两比特门通过 `_TWO_QUBIT_TABLES` 静态查表（输入是 `(P_ctrl, P_tgt)`，输出是 `(P_ctrl', P_tgt', sign)`）；
- 旋转门 `rx/ry/rz` 在角度 `θ = k·π/2` 下退化为 `id/S/S†/Z` 类等价 Clifford，由 `_angle_mod_4_halfpi` 离散化后查表；
- `u/u3` 使用相应矩阵识别为 `H`/`Z`/单比特 Clifford 之一，否则抛错。

## 常见报错

- `CliffordError("non-Clifford rotation angle ...")`
- `CliffordError("gate '<name>' is not Clifford")`
- `ValueError("pauli string length does not match num_qubits")`

## 相关

- 见 [`clifford_t.md`](clifford_t.md) 处理含非 Clifford 门的线路；
- 见 [`statevector.md`](statevector.md) 处理小比特数的稠密模拟。
