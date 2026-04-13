# QML — 量子机器学习

## 概览

- **模块**：`quantum_hw.algorithms.qml`
- **作用**：提供参数化量子线路（PQC）的机器学习训练框架，支持 **监督分类** 和 **无监督分布学习** 两种任务。
- **梯度方式**：`autograd`（模拟器，torch 自动微分）或 `parameter-shift`（硬件兼容）。
- **优化器**：Adam。
- **编码策略**：`angle` / `iqp` / 自定义 callable。

---

## 1. 监督分类 — `run_pqc_classifier`

### 功能

训练一个参数化量子线路分类器（PQC classifier），通过测量指定比特上的 $\langle Z \rangle$ 期望值做类别预测。

- **二分类**：单比特 $\langle Z \rangle > 0 \Rightarrow$ 类别 0，否则类别 1。
- **多分类**：$\text{argmax}_k \langle Z_k \rangle$ 作为预测类别。
- **损失函数**：multi-class cross-entropy（softmax on $\langle Z \rangle$ logits）。

### 签名

```python
run_pqc_classifier(
    num_qubits: int,
    train_data: Sequence[Tuple[Sequence[float], int]],
    *,
    test_data: Optional[Sequence[Tuple[Sequence[float], int]]] = None,
    encoding: Union[str, Callable] = "angle",
    encoding_kwargs: Optional[dict] = None,
    num_classes: int = 2,
    measurement_qubits: Optional[Sequence[int]] = None,
    layers: int = 2,
    max_iters: int = 100,
    learning_rate: float = 0.01,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    gradient_method: str = "autograd",
    # --- parameter-shift / hardware params ---
    client=None,
    backend=None,
    chip_name: str = "",
    shots: int = 4096,
    shift: float = np.pi / 2,
    zne: bool = False,
    readout_mitigation: bool = False,
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
) -> QMLResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `num_qubits` | `int` | - | 量子比特数。 |
| `train_data` | `Sequence[Tuple[Sequence[float], int]]` | - | 训练数据 `(features, label)` 对。 |
| `test_data` | `Optional[...]` | `None` | 可选验证数据；提供时以 test loss 选择最优模型，并报告 test accuracy。 |
| `encoding` | `str \| Callable` | `"angle"` | 编码方式：`"angle"` / `"iqp"` 或自定义 callable `(n_qubits, n_features) -> (QuantumCircuit, param_names)`。 |
| `encoding_kwargs` | `Optional[dict]` | `None` | 传给编码函数的额外参数。 |
| `num_classes` | `int` | `2` | 类别数。 |
| `measurement_qubits` | `Optional[Sequence[int]]` | `None` | 测量的量子比特索引；二分类默认 `[0]`，多分类默认 `range(min(num_classes, num_qubits))`。 |
| `layers` | `int` | `2` | Ansatz 层数。 |
| `max_iters` | `int` | `100` | 最大迭代轮数。 |
| `learning_rate` | `float` | `0.01` | Adam 学习率。 |
| `seed` | `Optional[int]` | `None` | 随机种子。 |
| `callback` | `Optional[Callable]` | `None` | 每轮回调 `(iter, loss)`。 |
| `gradient_method` | `str` | `"autograd"` | `"autograd"` 或 `"parameter-shift"`。 |
| `client` | - | `None` | parameter-shift 路径需要；`QuantumHardwareClient` 实例。 |
| `backend` | - | `None` | parameter-shift 路径需要；硬件后端。 |
| `chip_name` | `str` | `""` | 芯片名（parameter-shift 路径）。 |
| `shots` | `int` | `4096` | 每次评估 shots。 |
| `shift` | `float` | `π/2` | 参数移位角。 |
| `zne` | `bool` | `False` | 是否启用 ZNE。 |
| `readout_mitigation` | `bool` | `False` | 是否启用 readout 缓解。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 物理比特映射。 |
| `qasm_version` | `str` | `"2.0"` | OpenQASM 版本。 |
| `convert_single_qubit_gate_to_u` | `bool` | `True` | 转译时是否将单比特门转为 U 门。 |

### 返回值

`QMLResult`（定义于 `quantum_hw.core.types`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task` | `str` | 固定为 `"supervised"`。 |
| `best_loss` | `float` | 历史最优 loss（有 test_data 时为 test loss）。 |
| `best_params` | `List[float]` | 最优参数。 |
| `loss_history` | `List[float]` | 每轮训练 loss。 |
| `params_history` | `Optional[List[List[float]]]` | 每轮参数。 |
| `accuracy` | `Optional[float]` | 训练集 accuracy。 |
| `test_loss_history` | `Optional[List[float]]` | 每轮验证 loss（仅有 test_data 时）。 |
| `test_accuracy` | `Optional[float]` | 验证集 accuracy（仅有 test_data 时）。 |

---

## 2. 无监督分布学习 — `run_qnn_unsupervised`

### 功能

训练一个无编码的参数化量子线路（QNN / QBM），使其输出分布逼近给定的训练样本分布。

- **autograd 路径（模拟器）**：负对数似然（NLL）损失，直接通过 `sample_probabilities` 计算 $P(b|\theta)$。
- **parameter-shift 路径（硬件）**：最大均值差异（MMD²）损失，使用 RBF 核比较生成分布与训练分布。

### 签名

```python
run_qnn_unsupervised(
    num_qubits: int,
    train_samples: np.ndarray,
    *,
    test_samples: Optional[np.ndarray] = None,
    ansatz: str = "hardware_efficient",
    layers: int = 2,
    max_iters: int = 100,
    learning_rate: float = 0.01,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    gradient_method: str = "autograd",
    # --- parameter-shift / hardware params ---
    client=None,
    backend=None,
    chip_name: str = "",
    shots: int = 4096,
    shift: float = np.pi / 2,
    zne: bool = False,
    readout_mitigation: bool = False,
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
    # --- MMD params ---
    mmd_sigma: float = 1.0,
    # --- generation ---
    gen_shots: int = 1024,
) -> QBMResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `num_qubits` | `int` | - | 量子比特数。 |
| `train_samples` | `np.ndarray` | - | `(N, num_qubits)` 整数数组，元素 0/1，big-endian。 |
| `test_samples` | `Optional[np.ndarray]` | `None` | 可选验证样本数组，用于监控 test loss。 |
| `ansatz` | `str` | `"hardware_efficient"` | Ansatz 类型（当前仅支持 `"hardware_efficient"`）。 |
| `layers` | `int` | `2` | Ansatz 层数。 |
| `max_iters` | `int` | `100` | 最大迭代轮数。 |
| `learning_rate` | `float` | `0.01` | Adam 学习率。 |
| `seed` | `Optional[int]` | `None` | 随机种子。 |
| `callback` | `Optional[Callable]` | `None` | 每轮回调 `(iter, loss)`。 |
| `gradient_method` | `str` | `"autograd"` | `"autograd"`（NLL 损失）或 `"parameter-shift"`（MMD 损失）。 |
| `client` | - | `None` | parameter-shift 路径需要；`QuantumHardwareClient` 实例。 |
| `backend` | - | `None` | parameter-shift 路径需要；硬件后端。 |
| `chip_name` | `str` | `""` | 芯片名（parameter-shift 路径）。 |
| `shots` | `int` | `4096` | 每次评估 shots。 |
| `shift` | `float` | `π/2` | 参数移位角。 |
| `zne` | `bool` | `False` | 是否启用 ZNE。 |
| `readout_mitigation` | `bool` | `False` | 是否启用 readout 缓解。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 物理比特映射。 |
| `qasm_version` | `str` | `"2.0"` | OpenQASM 版本。 |
| `convert_single_qubit_gate_to_u` | `bool` | `True` | 转译时是否将单比特门转为 U 门。 |
| `mmd_sigma` | `float` | `1.0` | RBF 核带宽（仅 parameter-shift 路径）。 |
| `gen_shots` | `int` | `1024` | 训练完成后生成样本数。 |

### 返回值

`QBMResult`（定义于 `quantum_hw.core.types`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `best_loss` | `float` | 历史最优 loss（有 test_samples 时为 test loss）。 |
| `best_params` | `List[float]` | 最优参数。 |
| `loss_history` | `List[float]` | 每轮训练 loss。 |
| `test_loss_history` | `Optional[List[float]]` | 每轮验证 loss（仅有 test_samples 时）。 |
| `params_history` | `Optional[List[List[float]]]` | 每轮参数。 |
| `generated_samples` | `Optional[List[List[int]]]` | 训练完成后生成的样本（big-endian）。 |

---

## 内部辅助函数

| 函数 | 作用 |
|---|---|
| `_batch_loss_and_grads(z_values_list, labels, num_classes)` | 批量计算分类 loss 和 $\partial L/\partial \langle Z \rangle$。 |
| `_predictions_from_z(z_values_list, num_classes)` | 从 $\langle Z \rangle$ 值转换为类别预测。 |
| `_get_z_autograd(...)` | autograd 路径获取所有样本的 $\langle Z \rangle$ 值。 |
| `_get_z_backend(...)` | parameter-shift 路径获取所有样本的 $\langle Z \rangle$ 值。 |
| `_mmd_rbf(samples_p, samples_q, sigma)` | 计算 MMD²（RBF 核）。 |
| `_deduplicate_samples(samples)` | 样本去重并返回权重。 |
| `_simulate_samples(qc, shots, param_values, seed)` | 模拟器采样并返回 bitstring 数组。 |

---

## 示例

### 监督分类（Iris 数据集）

```python
from quantum_hw.algorithms import run_pqc_classifier

train = [(features_i, label_i) for features_i, label_i in zip(X_train, y_train)]
test  = [(features_i, label_i) for features_i, label_i in zip(X_test, y_test)]

result = run_pqc_classifier(
    num_qubits=4,
    train_data=train,
    test_data=test,
    encoding="angle",
    num_classes=3,
    layers=2,
    max_iters=80,
    learning_rate=0.02,
    gradient_method="autograd",
    seed=42,
)

print(f"Train accuracy: {result.accuracy:.4f}")
print(f"Test accuracy:  {result.test_accuracy:.4f}")
```

### 无监督分布学习

```python
import numpy as np
from quantum_hw.algorithms import run_qnn_unsupervised

# 目标分布：|00⟩ 和 |11⟩ 各 50%
samples = np.array([[0,0]]*50 + [[1,1]]*50)

result = run_qnn_unsupervised(
    num_qubits=2,
    train_samples=samples,
    layers=3,
    max_iters=200,
    learning_rate=0.02,
    gradient_method="autograd",
    seed=42,
)

print(f"Best loss: {result.best_loss:.6f}")
print(f"Generated {len(result.generated_samples)} samples")
```

## 相关页面

- [result types — QMLResult / QBMResult](../core/result_types.md)
- [ansatz templates](./ansatz_templates.md)
- [simulator interface — sample_probabilities](../sim/interface.md)
- [statevector simulator](../sim/statevector.md)
- [mps simulator](../sim/mps.md)
- [QML 分类教程](../../examples/demo_qml_iris.ipynb)
- [QNN BAS 教程](../../examples/demo_qnn_bas.ipynb)
- [QNN 无监督教程](../../examples/demo_qnn_unsupervised.ipynb)

---

## 3. 条件 QNN — `run_qnn_conditional`

### 功能

训练一个参数化量子线路学习条件分布 $P(y|x)$，其中 $x$ 为输入 bit-string，$y$ 为输出 bit-string。

支持 **autograd**（NLL 损失）和 **parameter-shift**（MMD 损失）两种梯度路径，与 `run_qnn_unsupervised` 框架一致。

---

## 4. 高层入口 — `QMLRunner`

模块：`quantum_hw.algorithms.qml_runner`

`QMLRunner` 是 QML 任务的高层封装，类似 `VQERunner` / `QAOARunner` 的设计：自动解析 provider → backend → chip，然后委托底层 `run_pqc_classifier` / `run_qnn_unsupervised` / `run_qnn_conditional`。

```python
from quantum_hw.algorithms.qml_runner import QMLRunner

runner = QMLRunner(client=client, layers=2, max_iters=100)
result = runner.run_classifier(train_data=train, test_data=test, num_qubits=4)
```
