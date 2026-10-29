# optimizer_utils — 变分算法共享工具

## 概览

- 模块：`fieldqkit.algorithms.optimizer_utils`
- 作用：VQE / QAOA（以及任何走 `run_variational_loop` 的算法）共享的底座，包含可观测量与能量评估、parameter-shift 梯度、Adam、Clifford fitting、通用优化循环。
- 这些函数主要是内部实现；本页面用于算法拆解与二次开发参考。

## 类型别名

```python
Hamiltonian   = List[Tuple[float, str]]          # [(coeff, pauli_string), ...]
CliffordFitMap = Dict[str, Tuple[float, float]]  # {observable: (a, b)}
```

## 可观测量与能量

### `normalize_observable_values(values)`
把 list-of-dict / 单元素 list 规整为 `Dict[str, float]`（或在单标量情形原样返回）。

### `ensure_observable_map(observables, values) -> Dict[str, float]`
从各种后端返回形状中稳妥得到 `{observable: value}`。形状无法对齐时抛 `RuntimeError("observable_values shape mismatch")`。

### `energy_from_expectations(hamiltonian, expectations) -> float`
计算 ⟨H⟩ = Σ coeff_i · ⟨O_i⟩。

## 模板实例化

### `instantiate_transpiled_template(transpiled_template, param_names, params) -> QuantumCircuit`
深拷贝预编译模板并把符号参数绑定为数值（`qc.apply_value(values, deep=True)`），避免每次迭代重新 transpile。

## 硬件能量评估

### `evaluate_energy_with_backend(...) -> Tuple[float, Dict[str, float]]`

```python
evaluate_energy_with_backend(
    client, qc, *,
    name, num_qubits, backend, chip_name, shots,
    hamiltonian, zne, readout_mitigation,
    clifford_fit_map=None, target_qubits=None,
    qasm_version="2.0", convert_single_qubit_gate_to_u=True,
    submit_options=None,
) -> Tuple[float, Dict[str, float]]
```

- 对**单条已绑定参数的线路**做一次能量前向：从 `hamiltonian` 提取 observables → `client._run_with_backend(..., transpile=False)` → `apply_clifford_fit` → `energy_from_expectations`。
- 固定 `transpile=False`，依赖外层预编译模板。
- `submit_options` 透传到 task adapter（`max_wait_time` / `sleep_time` 等）。

## Clifford fitting（仿射噪声校正）

围绕 `ideal ≈ a · noisy + b` 的逐 observable 仿射拟合。

| 函数 | 作用 |
|---|---|
| `build_single_qubit_rotation_gate_list(template)` | 收集模板里所有 `u/rx/ry/rz` 的 `(gate_index, gate_name)`，作为随机化位点。 |
| `randomize_single_qubit_gates_to_clifford(template, rng, gates, num_non_clifford_gates=0)` | 把参数化单比特门替换为随机 Clifford（或指定数量的 Haar 随机 U3），返回 `(circuit, signature)`。 |
| `sample_unique_randomized_clifford_circuits(template, *, rng, num_samples, gates, num_non_clifford_gates=0)` | 按签名去重，采样近似唯一的校准线路。 |
| `fit_linear_clifford_map(noisy, ideal) -> (a, b)` | 最小二乘拟合；方差过小时回退 `(1.0, mean_shift)`。 |
| `apply_clifford_fit(expectations, fit_map) -> Dict[str, float]` | 应用校正并裁剪到 `[-1, 1]`；`fit_map` 为空时原样返回。 |
| `build_clifford_fit_map(...) -> CliffordFitMap` | 端到端：生成 N 条 Clifford 随机线路，分别在噪声后端与理想模拟器上求期望，逐 observable 拟合 `(a, b)`。 |

`build_clifford_fit_map` 的理想分支优先走可扩展的 Heisenberg-picture 模拟器（`sim.clifford`，纯 Clifford 时 O(g·n)；含 k 个非 Clifford 旋转时 O(4^k·g·n)），不支持的门回退到 statevector。其签名同样接受 `submit_options`（仅用于噪声分支）。

## 梯度

### `parameter_shift_gradient(...) -> np.ndarray`

```python
parameter_shift_gradient(
    client, params, *,
    name, num_qubits, backend, chip_name, shots,
    hamiltonian, shift, zne, readout_mitigation,
    param_template=None, param_names=None,
    clifford_fit_map=None, target_qubits=None,
    circuit_transform=None, qasm_version="2.0",
    convert_single_qubit_gate_to_u=True, submit_options=None,
) -> np.ndarray
```

- 参数移位针对**独立门角**，不是将同名参数在整条线路中的所有出现位置一起移位。设第 `k` 个门角为 `a_k(θ)`，则按链式法则计算：

  `grad[i] = Σ_k (∂a_k/∂θ_i) · (E(a_k + shift) − E(a_k − shift)) / (2 sin(shift))`。

  每次仅改变该门的该角度，其他门角保持当前绑定值。默认 `shift=π/2` 时分母为 `2`。这种逐门角求导支持 QAOA 中共享的 `gamma` / `beta`，以及 `theta/2`、`-theta`、`theta+phi` 等已有参数表达式。
- 支持 `RX/RY/RZ/RXX/RYY/RZZ` 和 `U` 的三个角度槽；表达式 Jacobian 复用线路已有的安全表达式解析器计算，只依赖 NumPy，硬件路径无需安装 `[sim]`。
- 每轮先绑定当前参数；移位后的线路仍依次经过可选 `circuit_transform`、`GateCompressor` 和 `evaluate_energy_with_backend`。模板本身不会被修改。
- **后端求值次数**：没有 `circuit_transform` 时，每个对训练参数有非零导数的门角需要两次能量求值。一个门角同时依赖多个参数时，复用这对结果；无关参数和局部导数为零的角度不提交任务。有 `circuit_transform` 时，为保留其参数索引约定，按相关参数分别求值。因此共享参数线路的任务数通常高于原来的“每个参数两次”。
- `shift` 必须有限且 `abs(sin(shift)) >= 1e-12`；零、`π`、NaN 或无穷等无效值会在后端求值前抛出 `ValueError`。缺少模板/参数名、绑定不完整或表达式导数无定义时也会报错。

门角参数移位规则参见 [Schuld et al., Evaluating analytic gradients on quantum hardware](https://arxiv.org/abs/1811.11184)。共享或复合参数的梯度还需要乘上角度表达式的 Jacobian，并累加所有门角的贡献。

## Adam

### `adam_update(params, grads, m, v, t, *, lr, beta1, beta2, eps) -> (params, m, v)`
标准 Adam 单步（含偏差修正），`t` 为 1-based 迭代计数。

## 通用优化循环

### `run_variational_loop(...) -> dict`

```python
run_variational_loop(
    client, *,
    tag, name, num_qubits, param_names, symbolic_qc, hamiltonian, params,
    backend, chip_name, shots, max_iters,
    learning_rate, beta1, beta2, eps, shift,
    zne, readout_mitigation, gradient_method,
    seed=None, callback=None,
    transpiled_template=None, gradient_param_template=None,
    target_qubits=None, clifford_fit_map=None, circuit_transform=None,
    qasm_version="2.0", extra_info="",
    convert_single_qubit_gate_to_u=True, submit_options=None, device=None,
) -> dict
```

VQE / QAOA 共用的 Adam 优化主循环。`gradient_method` 须由调用方预先校验，按 `chip_name` 分三条路径：

| 路径 | 条件 | 行为 |
|---|---|---|
| 本地 autograd | `autograd` + `chip_name=="simulator"` | torch `energy_and_expectations` 自动微分回传梯度。 |
| 云端 autograd | `autograd` + `chip_name=="fieldquantum_sim"` | 单次 HTTP 调用 `FieldQuantumPlatform.run_expectation`，服务端完成采样 + parameter-shift，返回 `energy / expectations / gradients`。需 `metadata["platform_obj"]`（缺失抛 `RuntimeError`）。 |
| parameter-shift | 其它 | 每轮一次 `evaluate_energy_with_backend` + 一次 `parameter_shift_gradient`（含可选压缩 `circuit_transform`）。 |

> 该分发是 VQE/QAOA 支持 `fieldquantum_sim` autograd 的实现处；QML 不走本循环，因此 QML 的 autograd 仅限本地 Simulator。

**返回 dict 字段**：`best_cost`、`best_params`、`cost_history`、`params_history`、`grad_history`、`last_expectations`。

## 调用链

```
VQERunner/QAOARunner.run_model
  → run_vqe/qaoa_with_backend
      → build_clifford_fit_map (可选)
      → run_variational_loop
          → evaluate_energy_with_backend / parameter_shift_gradient
              → client._run_with_backend
```

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [QAOARunner.run_model](./qaoa_runner.md)
- [circuit compression](./circuit_compression.md)
- [api: run_with_backend](../api/run_with_backend.md)
