# matrix 与 utils

## 概览

- 模块：`quantum_hw.circuit.matrix`、`quantum_hw.circuit.utils`
- 作用：
  - 提供标准门矩阵与线路整体矩阵计算
  - 提供参数表达式处理与单/双比特分解算法

## matrix 主要接口

### 基础门矩阵

- 单比特：`id_mat`、`x_mat`、`y_mat`、`z_mat`、`h_mat`、`s_mat`、`sdg_mat`、`t_mat`、`tdg_mat`、`sx_mat`
- 双比特：`swap_mat`、`iswap_mat`、`cx_mat`、`cz_mat`

### 参数门矩阵函数

- `rx_mat(theta)`、`ry_mat(theta)`、`rz_mat(theta)`
- `p_mat(theta)`、`u_mat(theta, phi, lamb)`
- `rxx_mat(theta)`、`ryy_mat(theta)`、`rzz_mat(theta)`

### `circuit_to_unitary(nqubits, gates)`

- 将 gate tuple 序列转换为整条线路的幺正矩阵

### `is_unitary(matrix, tol=1e-9)`

- 判断输入矩阵是否幺正

### `remove_glob_phase(matrix)`

- 去除全局相位，便于等价比较

### `is_approx(a, b, tol=1e-9)`

- 判断两个矩阵是否近似相等

## utils 主要接口

### 参数与角度

- `limit_angle(angle)`：角度归一化
- `parse_expression(expr)`：表达式求值
- `handle_expression(exp, para_dict)`：带参数求值

### 单比特分解

- `zyz_decompose(unitary)`
- `u3_decompose(unitary)`

### 双比特分解

- `kak_decompose(unitary)`
- `simult_svd(...)`
- `glob_phase(...)`

## 与 QuantumCircuit 的关系

- `u3_for_unitary` 调用 `u3_decompose`
- `zyz_for_unitary` 调用 `zyz_decompose`
- `kak_for_unitary` 调用 `kak_decompose`
- 可用 `circuit_to_unitary` 对线路语义做矩阵级验证

## 示例

```python
import numpy as np
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.circuit.matrix import circuit_to_unitary, is_unitary

qc = QuantumCircuit(2)
qc.h(0).cx(0, 1)
U = circuit_to_unitary(qc.nqubits, qc.gates)

print(U.shape)
print(is_unitary(U))
assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]))
```

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [OpenQASM 解析](./qasm.md)
