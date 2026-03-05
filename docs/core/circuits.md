# circuits builders

## 模块

- `quantum_hw.core.circuits`

## 包含函数

- `build_ghz`
- `build_cluster`
- `build_qft`
- `build_ising_time_evolution`

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

- 作用：构造一阶 Trotter 的 Ising 时间演化线路。
- 关键结构：每步执行 ZZ 交互块（`CX-RZ-CX`）与 X 旋转块（`RX`）。
- 参数：
	- `j`：ZZ 耦合强度。
	- `h`：横场强度。
	- `t`：总演化时间。
	- `steps`：Trotter 步数，步长为 $dt=t/steps$。
	- `measure`：同上。
- 返回：`QuantumCircuit`。

## `measure=True` 行为

- 四个构造函数都会在尾部统一追加：
	- `qc.barrier()`
	- `qc.measure_all()`
- 若 `measure=False`，返回纯量子演化线路，便于后续编译、基变换或观测量分组测量。

## 常见报错

- `build_qft` 在底层电路对象不支持受控相位门（`cp/cu1/crz`）时会抛 `AttributeError`。
- 其余函数一般不会主动做参数合法性检查；非法 `num_qubits/steps` 会在后续门操作或数值阶段暴露错误。

## 示例

```python
from quantum_hw.core.circuits import (
		build_ghz,
		build_cluster,
		build_qft,
		build_ising_time_evolution,
)

qc1 = build_ghz(6, measure=False)
qc2 = build_cluster(6, measure=True)
qc3 = build_qft(5, with_swaps=True)
qc4 = build_ising_time_evolution(4, j=1.0, h=0.7, t=0.8, steps=2)
```

## 相关页面

- [observables](./observables.md)
- [readout](./readout.md)
- [zne](./zne.md)
