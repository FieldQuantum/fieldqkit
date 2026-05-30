# qml_encoding — 数据编码线路

## 概览

- 模块：`fieldqkit.algorithms.qml_encoding`
- 作用：把经典特征向量映射为量子态的编码线路，供 QML 训练使用。
- 每种策略有两个变体：
  - **数值版**（如 `angle_encoding_circuit`）：门携带具体数值。
  - **符号版**（如 `angle_encoding_circuit_symbolic`）：门携带字符串参数名（`x_0`、`x_1`…），从而线路只需 transpile 一次即可在不同特征向量间复用（parameter-shift 路径关键优化）。

四个函数都是包级导出（`from fieldqkit.algorithms import angle_encoding_circuit, ...`）。

## Angle 编码

### `angle_encoding_circuit`

```python
angle_encoding_circuit(
    features: Sequence[float],
    num_qubits: int,
    *,
    gate: str = "ry",
) -> QuantumCircuit
```

- 对每个比特 *i* 施加 `gate(feature_i)`。
- `len(features) < num_qubits` 时多余比特保持 |0⟩。
- `gate` ∈ `{"rx", "ry", "rz"}`。

### `angle_encoding_circuit_symbolic`

```python
angle_encoding_circuit_symbolic(
    num_qubits: int,
    num_features: int,
    *,
    gate: str = "ry",
    prefix: str = "x",
) -> Tuple[QuantumCircuit, List[str]]
```

- 返回 `(circuit, encoding_param_names)`，参数名为 `prefix_0, prefix_1, …`（共 `min(num_features, num_qubits)` 个）。

## IQP 编码

### `iqp_encoding_circuit`

```python
iqp_encoding_circuit(
    features: Sequence[float],
    num_qubits: int,
    *,
    reps: int = 1,
) -> QuantumCircuit
```

- 每个 repetition 结构：所有比特 `H` → 各比特 `RZ(x_i)` → 相邻对 `RZZ(x_i · x_{i+1})`（用 `CX-RZ-CX` 实现）。
- 末尾再加一层 `H`，将相位差转为幅度差。

### `iqp_encoding_circuit_symbolic`

```python
iqp_encoding_circuit_symbolic(
    num_qubits: int,
    num_features: int,
    *,
    reps: int = 1,
    prefix: str = "x",
) -> Tuple[QuantumCircuit, List[str]]
```

- 单比特 RZ 用符号参数 `x_i`；ZZ 相互作用门用符号乘积表达式 `x_i*x_{i+1}`。
- 返回 `(circuit, encoding_param_names)`，其中 `encoding_param_names` 只列基础参数名（乘积表达式由其派生）。

## 与 QML 的关系

- `run_pqc_classifier(encoding="angle" / "iqp")` 内部调用对应的**符号版**构建模板。
- 也可传入自定义 callable `(num_qubits, num_features) -> (QuantumCircuit, param_names)` 作为 `encoding`。

## 相关页面

- [QML](./qml.md)
- [QMLRunner](./qml_runner.md)
- [ansatz templates](./ansatz_templates.md)
