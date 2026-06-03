# observables

## 模块

- `fieldqkit.core.observables`

## 关键函数

- `pauli_support`
- `shift_pauli_string`
- `pauli_basis_pattern`
- `apply_measurement_basis_rotations`
- `append_measurement_basis`
- `group_observables`
- `pauli_expectation`

## Pauli 字符串格式

模块支持两种输入：

- 紧凑格式：`"ZZIX"`（长度固定，位置即比特索引）
- 显式索引格式：`"Z0 X2 Y3 I4"`（顺序可变，`I` 可省）

内部解析会统一到 `(idx, op)` 形式，并校验索引范围。

## 函数说明

### `pauli_support(pauli: str, num_qubits: int | None = None) -> List[int]`

- 作用：返回非单位算符作用的比特索引（升序）。
- 示例：
	- `pauli_support("ZZIX") -> [0, 1, 2]`
	- `pauli_support("X3 Z1") -> [1, 3]`

### `shift_pauli_string(pauli: str, offset: int) -> str`

- 作用：将 Pauli 字符串中的所有比特索引平移 `offset`。
- 返回索引形式的字符串，例如 `shift_pauli_string("X0 Z1", 3) -> "X3 Z4"`。

### `pauli_basis_pattern(pauli: str, num_qubits: int) -> List[str]`

- 作用：生成长度为 `num_qubits` 的测量基模式（元素为 `I/X/Y/Z`）。
- 典型用于可观测量分组与测量基附加。

### `append_measurement_basis(qc, basis_pattern: Sequence[str], target_qubits: Sequence[int] = None) -> None`

- 作用：按给定基模式先做基变换，再追加测量。
- 行为：
	- `target_qubits=None` 时，优先使用 `qc.qubits`（保持线路内记录的 qubit 顺序）；若对象无 `qubits` 属性，再退化为 `range(len(basis_pattern))`。
	- `target_qubits` 与 `basis_pattern` 长度必须一致，否则抛 `ValueError`。
	- 测量映射到紧凑经典位序：`qc.measure(target_qubits, range(len(target_qubits)))`。

### `group_observables(observables: Sequence[str], num_qubits: int) -> List[Dict[str, object]]`

- 作用：将可共测 observables 分组，减少任务提交数量。
- 策略：贪心合并到“第一个兼容组”。
- 兼容定义：在每个比特上，两者不能出现冲突的非 `I` 轴（例如 `X` 与 `Z` 冲突）。
- 返回结构：
	- `[{"basis": [...], "observables": [...]}, ...]`

### `pauli_expectation(samples: np.ndarray, pauli: str) -> float`

- 作用：从已在对应基下测量的样本估计 Pauli 期望值。
- 要求：`samples` 必须是二维数组 `(nshots, nqubits)`。
- 规则：测量位 `0 -> +1`，`1 -> -1`，按 support 上本征值乘积取均值。

## `target_qubits` 影响说明

- `apply_measurement_basis_rotations` / `append_measurement_basis` 支持 `target_qubits`。
- 当执行在物理映射后线路上时，可用 `target_qubits` 指定“模式元素与物理比特”的对应关系。
- 若不传，默认优先使用 `qc.qubits` 的顺序（常用于保留 transpiler/layout 映射语义）。

## 常见报错

- `ValueError("pauli string is empty")`
- `ValueError("pauli length mismatch with num_qubits")`
- `ValueError("pauli index out of range")`
- `ValueError("unsupported basis op: ...")`
- `ValueError("target_qubits length (...) does not match basis_pattern length (...)")`
- `ValueError("samples must be 2D")`

## 示例

```python
import numpy as np
from fieldqkit.core.observables import (
		pauli_support,
		group_observables,
		pauli_expectation,
)

print(pauli_support("Z0 X2 Y3"))

groups = group_observables(["Z0 Z1", "Z0", "X2", "X2 Z3"], num_qubits=4)
print(groups)

samples = np.array([[0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
print(pauli_expectation(samples, "Z0 Z1"))
```

## 相关页面

- [circuits builders](./circuits.md)
- [readout](./readout.md)
