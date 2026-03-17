# ansatz_templates

## 概览

- 模块：`quantum_hw.algorithms.ansatz_templates`
- 定位：VQE 与压缩流程共用的 ansatz 构造器。
- 提供能力：
  - hardware-efficient ansatz（数值/符号）
  - 轻量 UCC-inspired ansatz（数值/符号）
  - UCC 参数计数器

## 函数列表

### `build_hardware_efficient_ansatz`

```python
build_hardware_efficient_ansatz(
    num_qubits: int,
    params: Sequence[float],
    *,
    layers: int = 1,
) -> QuantumCircuit
```

- 参数长度约束：
  $$\text{len(params)} = 2 \times \text{num\_qubits} \times (\text{layers}+1)$$
- 结构：每层 `RX` + `RY` + 邻接 `CZ`，末尾再加一轮 `RX` + `RY`。

### `build_hardware_efficient_ansatz_symbolic`

```python
build_hardware_efficient_ansatz_symbolic(
    num_qubits: int,
    param_names: Sequence[str],
    *,
    layers: int = 1,
) -> QuantumCircuit
```

- 与上面同构，但参数为字符串占位符。
- 用于：
  - VQE 预编译模板
  - 压缩优化时的可微符号线路

### `build_ucc_num_params`

```python
build_ucc_num_params(num_qubits: int, layers: int) -> int
```

- 返回：
  $$\text{layers} \times (\text{num\_qubits} + \max(\text{num\_qubits}-1, 0))$$
- 输入约束：`num_qubits > 0` 且 `layers > 0`。

### `build_ucc_ansatz`

```python
build_ucc_ansatz(
    num_qubits: int,
    params: Sequence[float],
    *,
    layers: int = 1,
) -> QuantumCircuit
```

- 参数长度必须等于 `build_ucc_num_params(num_qubits, layers)`。
- 每层结构：
  - 每个量子比特一层 `RY`
  - 每对邻接比特执行 `CX - RY(target) - CX`

### `build_ucc_ansatz_symbolic`

```python
build_ucc_ansatz_symbolic(
    num_qubits: int,
    param_names: Sequence[str],
    *,
    layers: int = 1,
) -> QuantumCircuit
```

- 与 `build_ucc_ansatz` 同拓扑，但参数为符号字符串。

## 示例

### hardware-efficient（数值）

```python
from quantum_hw.algorithms.ansatz_templates import build_hardware_efficient_ansatz

n = 4
layers = 2
param_count = 2 * n * (layers + 1)
params = [0.01] * param_count

qc = build_hardware_efficient_ansatz(n, params, layers=layers)
```

### UCC（符号）

```python
from quantum_hw.algorithms.ansatz_templates import (
    build_ucc_num_params,
    build_ucc_ansatz_symbolic,
)

n = 6
layers = 2
m = build_ucc_num_params(n, layers)
names = [f"theta_{i}" for i in range(m)]

qc = build_ucc_ansatz_symbolic(n, names, layers=layers)
```

## 典型调用方

- VQE 主流程：`quantum_hw.algorithms.vqe`
- 压缩流程：`quantum_hw.algorithms.circuit_compression`

## 注意事项

- 这些函数仅负责 ansatz 结构构建，不执行 transpile、测量或后端提交。
- 参数长度校验是强约束，不匹配会抛 `ValueError`。

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [circuit compression](./circuit_compression.md)
