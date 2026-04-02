# matrix 与 utils

## 概览

- **模块**：`quantum_hw.circuit.matrix`、`quantum_hw.circuit.utils`
- **源文件**：`matrix.py`（约400 行）、`utils.py`（约290 行）
- **作用**：
  - 提供所有标准量子门的矩阵表示（常量 + 参数函数）
  - 提供门名到矩阵的统一查找字典 `gate_matrix_dict`
  - 提供单/双比特幺正矩阵分解算法（ZYZ、U3、KAK）
  - 提供随机幺正矩阵生成与等价判定工具

---

## matrix 模块

### 态矢量常量

| 名称 | 形状 | 说明 |
|------|------|------|
| `ket0` | `(2,1)` | $\|0\rangle = \begin{pmatrix}1\\0\end{pmatrix}$ |
| `ket1` | `(2,1)` | $\|1\rangle = \begin{pmatrix}0\\1\end{pmatrix}$ |

### 态矢量函数

#### `ketn0(nqubits: int) -> np.ndarray`

生成 $n$ 个 $\|0\rangle$ 的张量积：$\|0\rangle^{\otimes n}$。

| 参数 | 类型 | 说明 |
|------|------|------|
| `nqubits` | `int` | 量子比特数 |

**返回**：`(2^n, 1)` 形状的列向量。

#### `ketn1(nqubits: int) -> np.ndarray`

生成 $n$ 个 $\|1\rangle$ 的张量积：$\|1\rangle^{\otimes n}$。

### 单比特门矩阵常量

| 名称 | 门 | 矩阵 |
|------|-----|------|
| `id_mat` | $I$ | $\begin{pmatrix}1&0\\0&1\end{pmatrix}$ |
| `x_mat` | $X$ | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ |
| `y_mat` | $Y$ | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ |
| `z_mat` | $Z$ | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ |
| `h_mat` | $H$ | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ |
| `s_mat` | $S$ | $\begin{pmatrix}1&0\\0&i\end{pmatrix}$ |
| `sdg_mat` | $S^\dagger$ | $\begin{pmatrix}1&0\\0&-i\end{pmatrix}$ |
| `t_mat` | $T$ | $\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}$ |
| `tdg_mat` | $T^\dagger$ | $\begin{pmatrix}1&0\\0&e^{-i\pi/4}\end{pmatrix}$ |
| `sx_mat` | $\sqrt{X}$ | $\frac{1}{2}\begin{pmatrix}1+i&1-i\\1-i&1+i\end{pmatrix}$ |
| `sxdg_mat` | $\sqrt{X}^\dagger$ | $\frac{1}{2}\begin{pmatrix}1-i&1+i\\1+i&1-i\end{pmatrix}$ |

### 双比特门矩阵常量

| 名称 | 门 | 维度 | 说明 |
|------|-----|------|------|
| `cx_mat` | CX (CNOT) | $4\times4$ | 控制-X，控制位在前 |
| `xc_mat` | XC | $4\times4$ | 控制-X，控制位在后 |
| `cy_mat` | CY | $4\times4$ | 控制-Y，控制位在前 |
| `yc_mat` | YC | $4\times4$ | 控制-Y，控制位在后 |
| `cz_mat` | CZ | $4\times4$ | 控制-Z |
| `swap_mat` | SWAP | $4\times4$ | 交换门 |
| `iswap_mat` | iSWAP | $4\times4$ | $\|01\rangle\leftrightarrow i\|10\rangle$ |
| `ecr_mat` | ECR | $4\times4$ | Echoed Cross-Resonance，含 $1/\sqrt{2}$ 归一化 |

### 三比特门矩阵常量

| 名称 | 门 | 维度 | 说明 |
|------|-----|------|------|
| `ccz_mat` | CCZ | $8\times8$ | 双控-Z |
| `ccx_mat` | CCX (Toffoli) | $8\times8$ | 双控-X |
| `cxc_mat` | CXC | $8\times8$ | 控制位在两端的 CX |
| `cswap_mat` | Fredkin (CSWAP) | $8\times8$ | 控制-SWAP |
| `swapc_mat` | SWAPC | $8\times8$ | SWAP-控制 |

### 单比特参数门函数

#### `r_mat(theta, phi) -> np.ndarray`

通用旋转门：

$$R(\theta,\phi) = \begin{pmatrix}\cos\frac{\theta}{2} & -ie^{-i\phi}\sin\frac{\theta}{2} \\ -ie^{i\phi}\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}$$

#### `rx_mat(theta: float) -> np.ndarray`

$$R_x(\theta) = \begin{pmatrix}\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}$$

#### `ry_mat(theta: float) -> np.ndarray`

$$R_y(\theta) = \begin{pmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}$$

#### `rz_mat(theta: float) -> np.ndarray`

$$R_z(\theta) = \begin{pmatrix}e^{-i\theta/2} & 0 \\\ 0 & e^{i\theta/2}\end{pmatrix}$$

#### `p_mat(theta: float) -> np.ndarray`

相位门：

$$P(\theta) = \begin{pmatrix}1 & 0 \\\ 0 & e^{i\theta}\end{pmatrix}$$

#### `u_mat(theta: float, phi: float, lamda: float) -> np.ndarray`

通用 U3 门：

$$U(\theta,\phi,\lambda) = \begin{pmatrix}\cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}\end{pmatrix}$$

#### `u1_mat(lamda: float) -> np.ndarray`

等价于 `u_mat(0, 0, lamda)`，即 $U_1(\lambda) = U(0,0,\lambda)$。

#### `u2_mat(phi: float, lamda: float) -> np.ndarray`

等价于 `u_mat(π/2, phi, lamda)`，即 $U_2(\phi,\lambda) = U(\pi/2,\phi,\lambda)$。

### 双比特参数门函数

#### `rxx_mat(theta: float) -> np.ndarray`

$$R_{xx}(\theta) = \exp\bigl(-i\frac{\theta}{2} X\otimes X\bigr)$$

#### `ryy_mat(theta: float) -> np.ndarray`

$$R_{yy}(\theta) = \exp\bigl(-i\frac{\theta}{2} Y\otimes Y\bigr)$$

#### `rzz_mat(theta: float) -> np.ndarray`

$$R_{zz}(\theta) = \exp\bigl(-i\frac{\theta}{2} Z\otimes Z\bigr)$$

#### `cp_mat(theta: float) -> np.ndarray`

控制相位门：$CP(\theta) = \text{diag}(1, 1, 1, e^{i\theta})$。

### `gate_matrix_dict`

统一查找字典，将门名映射到矩阵常量或参数门函数，共 27 项（含 `cnot` → `cx_mat` 别名）：

```python
gate_matrix_dict = {
    'id': id_mat, 'x': x_mat, 'y': y_mat, 'z': z_mat, 'h': h_mat,
    's': s_mat, 'sdg': sdg_mat, 't': t_mat, 'tdg': tdg_mat,
    'sx': sx_mat, 'sxdg': sxdg_mat,
    'swap': swap_mat, 'iswap': iswap_mat, 'ecr': ecr_mat,
    'cx': cx_mat, 'cnot': cx_mat, 'cy': cy_mat, 'cz': cz_mat,
    'rx': rx_mat, 'ry': ry_mat, 'rz': rz_mat,
    'p': p_mat, 'u': u_mat, 'r': r_mat,
    'rxx': rxx_mat, 'ryy': ryy_mat, 'rzz': rzz_mat, 'cp': cp_mat,
    'ccz': ccz_mat, 'ccx': ccx_mat, 'cswap': cswap_mat,
}
```

> 离散门对应 `np.ndarray` 常量，参数门对应可调用函数。

---

## utils 模块

### `generate_random_unitary_matrix(dim: int, seed: int | None = None) -> np.ndarray`

利用 `scipy.stats.unitary_group.rvs` 生成 `dim × dim` 随机幺正矩阵。

| 参数 | 类型 | 说明 |
|------|------|------|
| `dim` | `int` | 矩阵维度 |
| `seed` | `int \| None` | 随机种子，默认 `None` |

### `is_equiv_unitary(mat1: np.ndarray, mat2: np.ndarray) -> bool`

判断两个幺正矩阵是否全局相位等价。

| 参数 | 类型 | 说明 |
|------|------|------|
| `mat1` | `np.ndarray` | 第一个幺正矩阵 |
| `mat2` | `np.ndarray` | 第二个幺正矩阵 |

**异常**：输入维度不匹配或非幺正时抛出 `ValueError`。

### `simult_svd(mat1, mat2) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]`

对两个实矩阵做联合 SVD 分解（Eckart-Young 定理）：

$$A = U D_1 V^\dagger,\quad B = U D_2 V^\dagger$$

| 参数 | 类型 | 说明 |
|------|------|------|
| `mat1` | `np.ndarray` | 实矩阵 A |
| `mat2` | `np.ndarray` | 实矩阵 B |

**返回**：`((U, V), (D1, D2))`，其中 $U, V \in SO(2)$，$D_1, D_2$ 为对角矩阵。

### `glob_phase(mat: np.ndarray) -> float`

提取 $d\times d$ 矩阵的全局相位 $\alpha$，满足 $U = e^{i\alpha} S$、$S \in SU(d)$。

$$\alpha = \arg\bigl(\det(U)^{1/d}\bigr)$$

**返回**：弧度值，范围 $(-\pi, \pi]$。

### `remove_glob_phase(mat: np.ndarray) -> np.ndarray`

去除 $2\times2$ 幺正矩阵的全局相位。利用 ZYZ 分解：

$$U = e^{i\alpha}\, R_z(\phi)\, R_y(\theta)\, R_z(\lambda) \;\Rightarrow\; \text{返回}\; R_z(\phi)\, R_y(\theta)\, R_z(\lambda)$$

### `kron_factor_4x4_to_2x2s(mat: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray]`

将 $4\times4$ 矩阵分解为两个 $2\times2$ 矩阵的 Kronecker 积：$U = g\,(A \otimes B)$。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `g` | `complex` | 全局标量因子 |
| `f1` | `np.ndarray` | $2\times2$ 因子矩阵 A |
| `f2` | `np.ndarray` | $2\times2$ 因子矩阵 B |

**异常**：不可 tensor 分解时抛出 `ValueError`；零行列式时抛出 `ZeroDivisionError`。

### `kak_decompose(mat: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]`

对任意双比特门做 KAK 分解。

参考论文：[An Introduction to Cartan's KAK Decomposition (arXiv:quant-ph/0406176)](https://arxiv.org/abs/quant-ph/0406176)

| 参数 | 类型 | 说明 |
|------|------|------|
| `mat` | `np.ndarray` | $4\times4$ 幺正矩阵 |

**返回**：`(rots1, rots2)`，各含 4 个 $2\times2$ 旋转矩阵，分别作用于 qubit 0 和 qubit 1。

内部调用链：`remove_glob_phase` → `simult_svd` → `kron_factor_4x4_to_2x2s`。

### `zyz_decompose(mat: np.ndarray) -> tuple[float, float, float, float]`

ZYZ 分解：$U = e^{i\alpha}\, R_z(\phi)\, R_y(\theta)\, R_z(\lambda)$。

**返回**：`(theta, phi, lamda, alpha)`。

### `u3_decompose(mat: np.ndarray) -> tuple[float, float, float]`

U3 分解：$U = e^{i\alpha}\, U_3(\theta, \phi, \lambda)$。

**返回**：`(theta, phi, lamda)`（忽略全局相位）。

---

## 与 QuantumCircuit 的关系

| QuantumCircuit 方法 | utils 调用 |
|---------------------|-----------|
| `u3_for_unitary(unitary, qubit)` | `u3_decompose` |
| `zyz_for_unitary(unitary, qubit)` | `zyz_decompose` |
| `kak_for_unitary(unitary, q1, q2)` | `kak_decompose` |

---

## 示例

### 使用门矩阵

```python
import numpy as np
from quantum_hw.circuit.matrix import h_mat, cx_mat, gate_matrix_dict

# 直接使用常量
print(h_mat.shape)          # (2, 2)
print(cx_mat.shape)         # (4, 4)

# 通过字典查找参数门
rz_fn = gate_matrix_dict['rz']
print(rz_fn(np.pi / 4))    # 2x2 rz(π/4) 矩阵
```

### 幺正矩阵等价判定

```python
import numpy as np
from quantum_hw.circuit.utils import generate_random_unitary_matrix, is_equiv_unitary

U = generate_random_unitary_matrix(4, seed=42)
# 添加全局相位后仍然等价
U_phased = np.exp(1j * 0.3) * U
print(is_equiv_unitary(U, U_phased))  # True
```

### KAK 分解

```python
import numpy as np
from quantum_hw.circuit.matrix import cx_mat
from quantum_hw.circuit.utils import kak_decompose

rots1, rots2 = kak_decompose(cx_mat)
print(len(rots1), len(rots2))  # 4, 4
```

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [OpenQASM 解析](./qasm.md)
- [Helpers 与渲染](./helpers_render.md)
