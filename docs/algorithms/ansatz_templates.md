# ansatz_templates

## 概览

- 模块：`fieldqkit.algorithms.ansatz_templates`
- 定位：VQE 与压缩流程共用的 ansatz 构造器。
- 提供能力：
  - hardware-efficient ansatz（数值/符号）

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

## 示例

### hardware-efficient（数值）

```python
from fieldqkit.algorithms.ansatz_templates import build_hardware_efficient_ansatz

n = 4
layers = 2
param_count = 2 * n * (layers + 1)
params = [0.01] * param_count

qc = build_hardware_efficient_ansatz(n, params, layers=layers)
```

## 典型调用方

- VQE 主流程：`fieldqkit.algorithms.vqe`
- 压缩流程：`fieldqkit.algorithms.circuit_compression`

## 注意事项

- 这些函数仅负责 ansatz 结构构建，不执行 transpile、测量或后端提交。
- 参数长度校验是强约束，不匹配会抛 `ValueError`。

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [circuit compression](./circuit_compression.md)
