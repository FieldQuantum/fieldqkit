# circuits builders

## 模块

- `fieldqkit.core.circuits`

## 包含函数

- `build_ghz`
- `build_cluster`
- `build_qft`
- `build_ising_time_evolution`
- `build_heisenberg_time_evolution`
- `build_xxz_time_evolution`
- `build_xy_time_evolution`

## 函数说明

### `build_ghz(num_qubits: int, measure: bool = False) -> QuantumCircuit`

- 作用：构造链式纠缠 GHZ 线路（`H(0)` + `CX` 链）。
- 参数：
	- `num_qubits`：量子比特数。
	- `measure`：为 `True` 时在末尾追加 `barrier + measure_all()`。
- 返回：`QuantumCircuit`。

### `build_cluster(num_qubits: int, measure: bool = False) -> QuantumCircuit`

- 作用：构造 1D cluster-like 线路（全 `H` 后按偶奇邻边施加 `CZ`）。
- 参数：同上。
- 返回：`QuantumCircuit`。

### `build_qft(num_qubits: int, measure: bool = False, with_swaps: bool = True) -> QuantumCircuit`

- 作用：构造 QFT 线路（Hadamard + controlled-phase）。
- 参数：
	- `with_swaps`：是否追加末尾比特反转 `SWAP`（canonical QFT 输出顺序）。
	- 其余同上。
- 返回：`QuantumCircuit`。

### `build_ising_time_evolution(num_qubits: int, j: float, h: float, t: float, steps: int = 1, measure: bool = False) -> QuantumCircuit`

- 作用：构造一阶 Trotter 的 transverse-field Ising 时间演化线路。
- 哈密顿量：$H = J \sum_i Z_i Z_{i+1} + h \sum_i X_i$（正号约定，与 `build_ising_hamiltonian` 差一个整体负号）。
- 关键结构：每步执行 `RZZ(2 j dt)` 双比特块与 `RX(2 h dt)` 单比特块（直接使用底层原生 `rzz` 门）。
- 参数：
	- `j`：ZZ 耦合强度。
	- `h`：横场强度。
	- `t`：总演化时间。
	- `steps`：Trotter 步数，步长为 $dt=t/steps$。
	- `measure`：同上。
- 返回：`QuantumCircuit`。

### `build_heisenberg_time_evolution(num_qubits, t, jx=1.0, jy=1.0, jz=1.0, hz=0.0, steps=1, measure=False) -> QuantumCircuit`

- 作用：构造一阶 Trotter 的 Heisenberg 时间演化线路。
- 哈密顿量：$H = \sum_i (J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}) + h_z \sum_i Z_i$（与 `build_heisenberg_hamiltonian` 一致）。
- 关键结构：每步顺序施加原生 `RXX(2 J_x dt)` / `RYY(2 J_y dt)` / `RZZ(2 J_z dt)` 双比特块以及 `RZ(2 h_z dt)` 纵场。
- 系数为零的项会被跳过，避免引入冗余门。
- 返回：`QuantumCircuit`。

### `build_xxz_time_evolution(num_qubits, t, jxy=1.0, jz=1.0, hz=0.0, steps=1, measure=False) -> QuantumCircuit`

- 作用：构造一阶 Trotter 的 XXZ 时间演化线路。
- 哈密顿量：$H = J_{xy} \sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + J_z \sum_i Z_i Z_{i+1} + h_z \sum_i Z_i$（与 `build_xxz_hamiltonian` 一致）。
- 关键结构：原生 `RXX` / `RYY` / `RZZ` 三种双比特门 + 可选 `RZ` 纵场。
- 返回：`QuantumCircuit`。

### `build_xy_time_evolution(num_qubits, t, jx=1.0, jy=1.0, hz=0.0, steps=1, measure=False) -> QuantumCircuit`

- 作用：构造一阶 Trotter 的 XY 时间演化线路。
- 哈密顿量：$H = J_x \sum_i X_i X_{i+1} + J_y \sum_i Y_i Y_{i+1} + h_z \sum_i Z_i$（与 `build_xy_hamiltonian` 一致）。
- 关键结构：原生 `RXX` / `RYY` 双比特门 + 可选 `RZ` 纵场。
- 返回：`QuantumCircuit`。

> 角度约定：原生 `r_PP(\theta) = \exp(-i (\theta/2) P\otimes P)`，因此每个项的耦合系数 $J$ 与时间步长 $dt$ 通过 `r_PP(2 J dt)` 传入。

## `measure=True` 行为

- 四个构造函数都会在尾部统一追加：
	- `qc.barrier()`
	- `qc.measure_all()`
- 若 `measure=False`，返回纯量子演化线路，便于后续编译、基变换或观测量分组测量。

## 常见报错

- 各函数一般不会主动做参数合法性检查；非法 `num_qubits/steps` 会在后续门操作或数值阶段暴露错误（`build_qft` 的受控相位通过 `rz`/`cx` 基本门分解实现，不依赖 `cp/cu1/crz` 方法）。

## 示例

```python
from fieldqkit.core.circuits import (
		build_ghz,
		build_cluster,
		build_qft,
		build_ising_time_evolution,
		build_heisenberg_time_evolution,
		build_xxz_time_evolution,
		build_xy_time_evolution,
)

qc1 = build_ghz(6, measure=False)
qc2 = build_cluster(6, measure=True)
qc3 = build_qft(5, with_swaps=True)
qc4 = build_ising_time_evolution(4, j=1.0, h=0.7, t=0.8, steps=2)
qc5 = build_heisenberg_time_evolution(4, t=0.8, jx=1.0, jy=1.0, jz=0.5, hz=0.1, steps=4)
qc6 = build_xxz_time_evolution(4, t=0.8, jxy=1.0, jz=0.5, steps=4)
qc7 = build_xy_time_evolution(4, t=0.8, jx=1.0, jy=1.0, steps=4)
```

## 相关页面

- [observables](./observables.md)
- [readout](./readout.md)
- [zne](./zne.md)
