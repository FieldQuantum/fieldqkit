# matrix utilities

## 模块

- `fieldqkit.sim.matrix`

## 概览

该模块维护模拟器使用的基础态与门矩阵定义：

- 初态列向量：`ket0`、`ket1`
- 多比特基态构造：`ketn0(nqubits)`、`ketn1(nqubits)`
- 固定门矩阵常量：如 `x_mat`、`h_mat`、`cx_mat`、`cz_mat`、`ccx_mat`
- 参数门矩阵函数：如 `rx_mat`、`ry_mat`、`rz_mat`、`u_mat`、`rxx_mat`
- 统一索引：`gate_matrix_dict`

## 关键函数

### `ketn0(nqubits: int, *, device=None) -> torch.Tensor`

- 作用：构造 `|0...0⟩` 列向量。
- 维度：`(2**nqubits, 1)`。

### `ketn1(nqubits: int, *, device=None) -> torch.Tensor`

- 作用：构造 `|1...1⟩` 列向量。
- 维度：`(2**nqubits, 1)`。

### 参数门矩阵函数

- 单比特：`rx_mat`、`ry_mat`、`rz_mat`、`u_mat`
- 双比特：`rxx_mat`、`ryy_mat`、`rzz_mat`

所有函数返回 `torch.Tensor(dtype=torch.complex*)`，维度与门作用比特数一致。

## `gate_matrix_dict`

- 字典键是门名字符串，值为：
  - 固定门：对应矩阵常量
  - 参数门：对应矩阵构造函数
- 完整键集合（共 26 个）：
  - 单比特：`id/x/y/z/h/s/sdg/t/tdg/sx/sxdg`
  - 双比特：`swap/iswap/ecr/cx/cy/cz`
  - 参数门：`rx/ry/rz/u/rxx/ryy/rzz`
  - 三比特：`ccz/ccx`

## 注意事项

- `statevector` 模拟器依赖本模块矩阵定义；变更 `gate_matrix_dict` 会直接影响模拟结果。
- 字典中没有 `cnot` 键；`QuantumCircuit` 的 `cnot()` 在建线路时即以 `cx` 记录，因此模拟器只会查到 `cx`。

## 示例

```python
import torch
from fieldqkit.sim.matrix import ketn0, gate_matrix_dict

psi0 = ketn0(3)
rx_fn = gate_matrix_dict["rx"]
rx = rx_fn(torch.pi / 3)
cz = gate_matrix_dict["cz"]

print(psi0.shape)
print(rx.shape, cz.shape)
```

## 相关页面

- [statevector simulator](./statevector.md)
- [mps simulator](./mps.md)
- [mpo process simulator](./mpo.md)
- [simulator common helpers](./common.md)
