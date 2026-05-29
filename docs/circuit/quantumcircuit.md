# QuantumCircuit

## 概览

- **模块**：`quantum_hw.circuit.quantumcircuit`
- **作用**：提供统一的量子线路表示、门操作 API、QASM 导入导出、参数绑定、线路分析与文本可视化能力。
- **核心对象**：`QuantumCircuit`

`QuantumCircuit` 是项目中的主线路对象。上层算法（如 VQE、QAOA）、编译流程、模拟器和执行 API 都以该类为中心完成数据交换。

## 构造与基础属性

### `QuantumCircuit(*args)`

支持三种形式：

- `QuantumCircuit()`
- `QuantumCircuit(nqubits)`
- `QuantumCircuit(nqubits, ncbits)`

关键属性：

- `nqubits`：逻辑量子位数
- `ncbits`：经典位数
- `qubits`：线路中实际使用的 qubit（升序）
- `gates`：tuple IR 门序列
- `params_value`：参数名到数值的映射

### 内部数据约定（核心）

`gates` 使用 tuple IR 存储，常见形态如下：

- 单比特离散门：`(gate, q)`
- 双比特离散门：`(gate, q0, q1)`
- 三比特门：`(gate, q0, q1, q2)`
- 单比特参数门：`(gate, theta, q)` / `("r", theta, phi, q)` / `("u", theta, phi, lam, q)`
- 双比特参数门：`(gate, theta, q0, q1)`
- 功能门：`("delay", duration, (q...))` / `("barrier", (q...))` / `("measure", [q...], [c...])` / `("reset", q)`
- 噪声信道门：`(gate, p, q)`（单比特，如 `("depolarize1", 0.1, 0)`）/ `("depolarize2", p, q0, q1)`（双比特）

### `deepcopy() -> QuantumCircuit`

- 深拷贝线路（含 `qubits/gates/params_value`）。

### `adjust_index(num: int, *, cbit_offset: Optional[int] = None) -> QuantumCircuit`

- 作用：整体平移 qubit/cbit 索引。
- 常用于线路拼接和子线路打包。

### `cbits`（property）

- 返回测量实际使用的经典位索引（去重排序）。

## 推荐使用流程

1. 初始化线路并添加门。
2. 若使用参数占位符，先写入符号参数门（如 `rx("theta", 0)`）。
3. 使用 `apply_value(...)` 绑定参数。
4. 根据需求导出 QASM、绘图或提交执行。

说明：所有门方法均返回 `self`，支持链式调用，例如 `qc.h(0).cx(0, 1).measure_all()`。

## QASM 导入导出

### `from_openqasm2(openqasm2_str: str) -> QuantumCircuit`

- 校验 `OPENQASM 2.0` 头。
- 调用 `parse_openqasm2_to_gates` 解析。

### `to_openqasm2(symbolic=False)`

- 导出 OpenQASM 2.0 程序字符串。
- `symbolic=True` 时保留字符串参数原样输出（用于生成服务端参数模板）；默认行为先解析到数值再输出。

## 门操作接口

门操作按功能分为离散门、参数门和功能门。调用成功后会更新 `gates` 与 `qubits`。

### 单比特离散门

- `id` / `x` / `y` / `z` / `h`
- `s` / `sdg` / `t` / `tdg`
- `sx` / `sxdg`

### 双比特离散门

- `cx` / `cnot` / `cy` / `cz`
- `swap` / `iswap` / `ecr`

### 三比特门

- `ccx` / `ccz`

### 参数门

- 单比特：`u`、`rx`、`ry`、`rz`
- 双比特：`rxx`、`ryy`、`rzz`

### Pauli 演化门

- `pauli_evolution(theta, pauli)`：追加
    $$\exp\left(-i\,\frac{\theta}{2}P\right)$$
    其中 `P` 为 Pauli 串。
- 支持两种 Pauli 字符串格式：
    - 紧凑格式：`"XIZY"`、`"ZZII"`（长度需与 `nqubits` 一致）
    - 索引格式：`"X1 Y2 Z3 Z4"`
- `theta` 可为数值或字符串参数；字符串会登记到 `params_value`。
- 当 `P=I`（全单位项）时仅对应全局相位，线路中不会追加实际门。

参数可为数值或字符串占位符。占位符会自动注册到 `params_value`，供后续绑定。

### 噪声信道门

用于构造含噪线路，由密度矩阵后端（[density matrix simulator](../sim/density_matrix.md)）模拟。这些门**仅能在模拟器上运行**（`simulator` / `fieldquantum_sim`），提交真机会被拒绝；含噪线路也会跳过转译。

- 单比特：`depolarize1(p, qubit)`、`x_error(p, qubit)`、`y_error(p, qubit)`、`z_error(p, qubit)`、`amplitude_damping(gamma, qubit)`、`phase_damping(gamma, qubit)`
- 双比特：`depolarize2(p, qubit0, qubit1)`

约束：

- 噪声率（`p` / `gamma`）必须是 $[0, 1]$ 内的**具体数值**；传入符号/字符串参数会抛 `ValueError`（噪声率不可微，不参与变分优化）。
- `depolarize2` 要求两个 qubit 不同。

## 参数绑定与映射

### `apply_value(params_dic: dict, *, deep: bool = False) -> QuantumCircuit`

- `deep=False`：只更新参数表。
- `deep=True`：将 gate tuple 中占位参数物化为数值。

典型用法：

- 扫描多个参数组合时：先构造一次符号线路，再重复 `apply_value(..., deep=True)`。
- 仅维护参数仓库时：使用 `deep=False`。

### `mapping_to_others(mapping: dict) -> QuantumCircuit`

- 按映射字典重写线路 qubit 索引。

## 矩阵分解相关接口

### `u3_for_unitary(unitary: np.ndarray, qubit: int)`

- 输入：`2x2` 幺正矩阵。
- 行为：调用 `u3_decompose` 后插入 `u(...)`。

### `zyz_for_unitary(unitary: np.ndarray, qubit: int) -> QuantumCircuit`

- 输入：`2x2` 幺正矩阵。
- 行为：转为 `rz-ry-rz` 序列。

### `kak_for_unitary(unitary: np.ndarray, qubit1: int, qubit2: int) -> QuantumCircuit`

- 输入：`4x4` 幺正矩阵。
- 行为：KAK 分解并生成等价双比特线路。

## 功能门与编辑接口

- `reset(qubit)`
- `delay(duration, *qubits, unit='s')`
- `barrier(*qubits)`
- `measure(qubitlst, cbitlst)`
- `measure_all()`
- `remove_barrier()`
- `remove_gate(gate_name)`
- `count_gate(gate_name) -> int`
- `remove_noise_channels() -> QuantumCircuit`

补充说明：

- `delay` 支持 `unit='ns'/'us'/'ms'/'s'`（内部统一换算为秒）。
- `measure_all()` 仅对当前 `qubits` 中已使用量子位追加测量。
- `remove_barrier()` 不影响量子语义，可用于导出前清理线路。
- `remove_noise_channels()` 返回去除全部噪声信道门的**副本**（原线路不变），用于获取理想（无噪）参考线路，例如 Clifford 数据回归（CDR）中的理想分支。

## 分析与可视化

- `depth`：线路深度（DAG 分层，忽略 barrier）
- `ncz`：双比特门计数
- `qubits_in_use`：实际使用 qubit 列表
- `draw(width=4)`：完整文本图
- `draw_simply(width=4)`：仅活跃 qubit 文本图

## 常见异常

- 非法索引：`ValueError`
- 参数未绑定或类型错误：`ValueError` / `TypeError`
- 非法 unitary 形状：`ValueError`
- `to_latex` 当前未实现：`NotImplementedError`

## 示例 1：基础建线路与导出 QASM

```python
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.rx("theta", 0)
qc.apply_value({"theta": 0.25}, deep=True)
qc.measure([0, 1], [0, 1])

print(qc.to_openqasm2())
```

## 示例 2：参数化模板复用

```python
import numpy as np
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(2)
qc.ry("t0", 0)
qc.rz("t1", 1)
qc.cz(0, 1)

for step in range(3):
    values = {"t0": float(step) * 0.1, "t1": np.pi / 4 + step * 0.05}
    qci = qc.deepcopy().apply_value(values, deep=True)
    print("step", step, "depth", qci.depth)
```

## 示例 3：OpenQASM 导入后继续编辑

```python
from quantum_hw.circuit import QuantumCircuit

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""

qc = QuantumCircuit().from_openqasm2(qasm)
qc.barrier(0, 1)
qc.measure([0, 1], [0, 1])
print(qc.to_openqasm2())
```

## 示例 4：子线路拼接前做索引平移

```python
from quantum_hw.circuit import QuantumCircuit

left = QuantumCircuit(2, 2)
left.h(0)
left.cx(0, 1)

right = QuantumCircuit(2, 2)
right.x(0)
right.measure([0, 1], [0, 1])
right.adjust_index(2, cbit_offset=2)

merged = QuantumCircuit(4, 4)
merged.gates = left.gates + right.gates
merged.qubits = sorted(set(left.qubits + right.qubits))

print(merged.qubits)
```

## 示例 5：线路统计与可视化

```python
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cz(0, 1)
qc.rzz(0.4, 1, 2)
qc.measure([0, 1, 2], [0, 1, 2])
print("depth:", qc.depth)
print("ncz:", qc.ncz)
print("qubits_in_use:", qc.qubits_in_use)

qc.draw(width=3)
qc.draw_simply(width=3)
```

## 示例 6：Pauli 演化

```python
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(5)
qc.pauli_evolution(0.3, "X1 Y2 Z3 Z4")

# 也支持字符串参数
qc.pauli_evolution("theta", "X1 Y2 Z3 Z4")
```

## 示例 7：从幺正矩阵生成线路片段

```python
import numpy as np
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(2)

u1 = np.array([[1, 0], [0, 1j]], dtype=complex)
qc.u3_for_unitary(u1, 0)

u2 = np.eye(4, dtype=complex)
qc.kak_for_unitary(u2, 0, 1)

print("gates:", len(qc.gates))
```

## 相关页面

- [OpenQASM 解析](./qasm.md)
- [QCIS 原生指令](./qcis.md)
- [helpers 与渲染](./helpers_render.md)
- [matrix 与 utils](./matrix_utils.md)
- [density matrix simulator](../sim/density_matrix.md)
