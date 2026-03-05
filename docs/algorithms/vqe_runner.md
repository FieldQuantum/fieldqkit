# VQERunner.run_model

## 概览

- **模块**：`quantum_hw.algorithms.vqe`
- **作用**：使用参数移位法估计梯度，并用 Adam 做能量最小化。
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
  beta2=0.999,
  eps=1e-8,
  shift=np.pi / 2.0,
  zne=False,
  readout_mitigation=False,
  seed=None,
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
| `beta1` / `beta2` / `eps` | `float` | `0.9/0.999/1e-8` | 否 | `VQERunner` 初始化参数：Adam 超参数。 |
| `shift` | `float` | `π/2` | 否 | `VQERunner` 初始化参数：参数移位角。 |
| `zne` | `bool` | `False` | 否 | `VQERunner` 初始化参数：是否启用 ZNE。 |
| `readout_mitigation` | `bool` | `False` | 否 | `VQERunner` 初始化参数：是否启用 readout 缓解。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定物理比特映射。 |
| `seed` | `Optional[int]` | `None` | 否 | `VQERunner` 初始化参数：参数初始化随机种子。 |
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
) -> VQEResult
```

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
  - 所有候选芯片执行失败。
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
- 当 `target_qubits is None` 且后端可用比特充足时，内部会尝试把多个梯度评估线路打包并行提交；若批次失败会自动降级缩小 batch。
- 当显式设置 `target_qubits` 时，会关闭打包路径，逐参数串行评估，保证映射固定。

## 相关页面

- [QAOARunner.run_model](./qaoa_runner.md)
- [ShadowTomography.run](./shadow_tomography.md)
- [result types](../core/result_types.md)
