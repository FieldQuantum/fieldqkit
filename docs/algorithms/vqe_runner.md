# VQERunner.run_model

## 概览

- **模块**：`quantum_hw.algorithms.vqe`
- **作用**：支持参数移位或 `torch autograd` 梯度，并用 Adam 做能量最小化。
- **当前推荐入口**：`VQERunner.run_model(...)`

## 推荐签名（`VQERunner.run_model`）

```python
VQERunner(
  client,
  layers=1,
  shots=1024,
  max_iters=20,
  learning_rate=0.1,
  beta1=0.9,
  beta2=0.98,
  eps=1e-8,
  shift=np.pi / 2.0,
  zne=False,
  readout_mitigation=False,
  seed=None,
  gradient_method="parameter-shift",
)

run_model(
    name,
    num_qubits,
  *,
    model="ising",
    model_params=None,
    hamiltonian=None,
    target_qubits=None,
    init_params=None,
    callback=None,
    prefer_chips=None,
    rank_weights=None,
) -> VQEResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `client` | `QuantumHardwareClient` | - | 是 | `VQERunner` 初始化参数。 |
| `name` | `str` | - | 是 | 任务名前缀。 |
| `num_qubits` | `int` | - | 是 | 逻辑比特数。 |
| `model` | `str` | `"ising"` | 否 | 支持：`ising/heisenberg/xy/xxz/custom`。 |
| `model_params` | `Optional[Dict[str, float]]` | `None` | 否 | 内置模型参数字典。 |
| `hamiltonian` | `Optional[Sequence[Tuple[float, str]]]` | `None` | 否 | 自定义哈密顿量（仅 `model="custom"` 时使用）。 |
| `layers` | `int` | `1` | 否 | `VQERunner` 初始化参数：纠缠层数。 |
| `shots` | `int` | `1024` | 否 | `VQERunner` 初始化参数：每次评估 shots。 |
| `max_iters` | `int` | `20` | 否 | `VQERunner` 初始化参数：迭代轮数。 |
| `learning_rate` | `float` | `0.1` | 否 | `VQERunner` 初始化参数：Adam 学习率。 |
| `beta1` / `beta2` / `eps` | `float` | `0.9/0.98/1e-8` | 否 | `VQERunner` 初始化参数：Adam 超参数。 |
| `shift` | `float` | `π/2` | 否 | `VQERunner` 初始化参数：参数移位角。 |
| `zne` | `bool` | `False` | 否 | `VQERunner` 初始化参数：是否启用 ZNE。 |
| `readout_mitigation` | `bool` | `False` | 否 | `VQERunner` 初始化参数：是否启用 readout 缓解。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定物理比特映射。 |
| `seed` | `Optional[int]` | `None` | 否 | `VQERunner` 初始化参数：参数初始化随机种子。 |
| `gradient_method` | `Literal["parameter-shift", "autograd"]` | `"parameter-shift"` | 否 | 梯度计算方式。`autograd` 仅支持 `Simulator`。 |
| `init_params` | `Optional[Sequence[float]]` | `None` | 否 | 显式初始参数；长度必须等于 `2 * num_qubits * (layers + 1)`。 |
| `callback` | `Optional[Callable[[int, float, np.ndarray], None]]` | `None` | 否 | 每轮回调，参数为 `(iter_idx, energy, params)`。 |
| `prefer_chips` | `Optional[Sequence[str] \| str]` | `None` | 否 | 候选芯片限制（可传 `"Simulator"`）。 |
| `rank_weights` | `Optional[Dict[str, float]]` | `None` | 否 | 芯片排序权重（`queue/nqubits/error`）。 |

## 低层接口（手动指定后端）

```python
run_vqe_with_backend(
  client,
  *,
  name,
  num_qubits,
  backend,
  chip_name,
  hamiltonian,
  layers,
  shots,
  max_iters,
  learning_rate,
  beta1,
  beta2,
  eps,
  shift,
  zne,
  readout_mitigation,
  target_qubits=None,
  seed=None,
  init_params=None,
  callback=None,
  gradient_method="parameter-shift",
) -> VQEResult
```

## 私有方法（进阶拆解）

下面两个方法是 `run_vqe_with_backend(...)` 的核心内部步骤，适合做算法拆解和二次开发时参考。

### `_evaluate_energy_with_backend`

```python
_evaluate_energy_with_backend(
  client,
  qc,
  *,
  name,
  num_qubits,
  backend,
  chip_name,
  shots,
  hamiltonian,
  zne,
  readout_mitigation,
) -> Tuple[float, Dict[str, float]]
```

- 作用：对**单条 ansatz 线路**做一次能量前向评估。
- 做法：
  - 从 `hamiltonian` 提取待测 `observables`。
  - 调用 `client._run_with_backend(...)` 得到每个 observable 的期望值（此处固定 `transpile=False`，依赖外层预编译模板）。
  - 用 `sum(coeff * <obs>)` 合成总能量。
- 返回：
  - `energy: float`：当前参数对应能量。
  - `expectations: Dict[str, float]`：observable 到期望值映射。

### `_parameter_shift_gradient`

```python
_parameter_shift_gradient(
  client,
  params,
  *,
  name,
  num_qubits,
  backend,
  chip_name,
  shots,
  hamiltonian,
  shift,
  zne,
  readout_mitigation,
  transpiled_template,
  param_names,
) -> np.ndarray
```

- 作用：基于 parameter-shift 规则估计梯度向量。
- 公式：
  - 第 `i` 个分量使用 `0.5 * (E(theta_i + shift) - E(theta_i - shift))`。
- 要求：在当前流程中需要传入 `transpiled_template` 与 `param_names`（用于仅替换参数值、避免重复编译）。
- 返回：`np.ndarray`，长度与 `params` 相同。

### 自动微分路径（`gradient_method="autograd"`）

- 触发条件：`run_vqe_with_backend(..., gradient_method="autograd")` 或 `VQERunner(gradient_method="autograd")`。
- 限制：
  - 仅支持 `chip_name="Simulator"`
- 实现位置：`src/quantum_hw/sim/statevector.py`（在同一模拟器模块中提供可微分能量评估）。
- 依赖：需要安装 `torch`。

## 返回值

返回 `VQEResult`（定义于 `quantum_hw.core.types`）：

- `best_energy: float`
- `best_params: List[float]`
- `energy_history: List[float]`
- `params_history: Optional[List[List[float]]]`
- `grad_history: Optional[List[List[float]]]`
- `last_expectations: Optional[Dict[str, float]]`

## 支持模型与参数

- `model="ising"`：调用 `build_ising_hamiltonian(num_qubits, j=..., h=...)`
- `model="heisenberg"`：`jx/jy/jz/hz`
- `model="xy"`：`jx/jy/hz`
- `model="xxz"`：`jxy/jz/hz`
- `model="custom"`：必须提供 `hamiltonian=[(coeff, pauli), ...]`

## 异常与报错

- `ValueError`
  - `model` 不支持。
  - `model="custom"` 但未提供 `hamiltonian`。
  - `init_params` 长度不匹配。
  - 自定义 Pauli 项为空字符串或索引越界。
- `RuntimeError`
  - 无可用芯片。
  - 所有候选芯片执行失败（`run_model` 会捕获候选执行异常并继续尝试下一块芯片，最终统一抛错）。
  - 期望值结构与 observable 列表不匹配。

## 示例

```python
from quantum_hw import QuantumHardwareClient
from quantum_hw.algorithms import VQERunner

client = QuantumHardwareClient()
runner = VQERunner(
  client=client,
  layers=2,
  shots=2048,
  max_iters=15,
  learning_rate=0.15,
  readout_mitigation=True,
  seed=42,
)

result = runner.run_model(
    name="vqe_ising_4q",
    num_qubits=4,
    model="ising",
    model_params={"j": 1.0, "h": 0.8},
    prefer_chips="Simulator",
)

print(result.best_energy)
print(result.best_params)
```

## 行为细节 / 注意事项

- Ansatz 参数维度为 $2 \times \text{num\_qubits} \times (\text{layers}+1)$，文中所有初始化与回调参数都遵循该维度。
- 每个参数梯度需要两次移位评估（`+shift/-shift`）；单轮理论评估次数约为 `1 + 2 * num_params` 次。
- 当 `gradient_method="autograd"` 且使用 `Simulator` 时，梯度由 `energy_t.backward()` 回传，不再执行 parameter-shift 线路采样。
- 当 `gradient_method="parameter-shift"` 时，VQE 会先在内部对参数化 ansatz 做一次预编译，然后每次迭代/移位只替换参数值并提交，避免重复 transpile。

## 相关页面

- [QAOARunner.run_model](./qaoa_runner.md)
- [ShadowTomography.run](./shadow_tomography.md)
- [run_with_backend](../api/run_with_backend.md)
- [result types](../core/result_types.md)
