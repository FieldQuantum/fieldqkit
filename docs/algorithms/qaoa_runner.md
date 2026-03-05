# QAOARunner.run_model

## 概览

- **模块**：`quantum_hw.algorithms.qaoa`
- **作用**：QAOA 组合优化入口，支持 `maxcut` 与 `custom` 两类问题。
- **优化方向**：默认做 cost 最大化（Adam ascent）。
- **当前推荐入口**：`QAOARunner.run_model(...)`

## 推荐签名（`QAOARunner.run_model`）

```python
QAOARunner(
  client,
  p=1,
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
    problem="maxcut",
    edges=None,
    weights=None,
    terms=None,
    constant=0.0,
    target_qubits=None,
    init_params=None,
    callback=None,
    prefer_chips=None,
    rank_weights=None,
) -> QAOAResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `client` | `QuantumHardwareClient` | - | 是 | `QAOARunner` 初始化参数。 |
| `name` | `str` | - | 是 | 任务名前缀。 |
| `num_qubits` | `int` | - | 是 | 逻辑比特数。 |
| `problem` | `str` | `"maxcut"` | 否 | 问题模式：`"maxcut"` 或 `"custom"`。 |
| `edges` | `Optional[Sequence[Tuple[int, int]]]` | `None` | 条件必填 | `problem="maxcut"` 时必填。 |
| `weights` | `Optional[Sequence[float]]` | `None` | 否 | MaxCut 边权；为空时按 1.0 处理。 |
| `terms` | `Optional[Sequence[Tuple[float, str]]]` | `None` | 条件必填 | `problem="custom"` 时必填。 |
| `constant` | `float` | `0.0` | 否 | custom 模式代价常数项。 |
| `p` | `int` | `1` | 否 | `QAOARunner` 初始化参数：深度，参数维度为 `2 * p`。 |
| `shots` | `int` | `1024` | 否 | `QAOARunner` 初始化参数：每次 cost 评估 shots。 |
| `max_iters` | `int` | `20` | 否 | `QAOARunner` 初始化参数：迭代轮数。 |
| `learning_rate` | `float` | `0.1` | 否 | `QAOARunner` 初始化参数：Adam 学习率。 |
| `beta1` / `beta2` / `eps` | `float` | `0.9/0.999/1e-8` | 否 | `QAOARunner` 初始化参数：Adam 超参数。 |
| `shift` | `float` | `π/2` | 否 | `QAOARunner` 初始化参数：参数移位角。 |
| `zne` | `bool` | `False` | 否 | `QAOARunner` 初始化参数：是否启用 ZNE。 |
| `readout_mitigation` | `bool` | `False` | 否 | `QAOARunner` 初始化参数：是否启用 readout 缓解。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定物理比特映射。 |
| `seed` | `Optional[int]` | `None` | 否 | `QAOARunner` 初始化参数：参数初始化随机种子。 |
| `init_params` | `Optional[Sequence[float]]` | `None` | 否 | 显式初始参数；长度必须为 `2 * p`。 |
| `callback` | `Optional[Callable[[int, float, np.ndarray], None]]` | `None` | 否 | 每轮回调 `(iter_idx, cost, params)`。 |
| `prefer_chips` | `Optional[Sequence[str] \| str]` | `None` | 否 | 候选芯片限制（可传 `"Simulator"`）。 |
| `rank_weights` | `Optional[Dict[str, float]]` | `None` | 否 | 芯片排序权重（`queue/nqubits/error`）。 |

## 低层接口（手动指定后端）

```python
run_qaoa_with_backend(
  client,
  *,
  name,
  num_qubits,
  backend,
  chip_name,
  edges,
  weights,
  terms,
  constant,
  p,
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
) -> QAOAResult
```

## 问题模式

- `problem="maxcut"`
  - 代价哈密顿量自动由 `edges/weights` 构建：
  - $$C = \sum_{(i,j)} w_{ij} \frac{1 - Z_i Z_j}{2}$$
- `problem="custom"`
  - 传入 `terms=[(coeff, pauli), ...]` 与可选 `constant`。
  - 当前仅支持 `Z` 或 `ZZ` 项，且最多二体项。

## 返回值

返回 `QAOAResult`（定义于 `quantum_hw.core.types`）：

- `best_cost: float`
- `best_params: List[float]`
- `cost_history: List[float]`
- `params_history: Optional[List[List[float]]]`
- `grad_history: Optional[List[List[float]]]`
- `last_expectations: Optional[Dict[str, float]]`

## 异常与报错

- `ValueError`
  - `problem` 非 `maxcut/custom`。
  - `maxcut` 缺少 `edges`，或边索引非法、权重长度不匹配。
  - `custom` 缺少 `terms`，或出现不支持的 Pauli 项。
  - `init_params` 长度不等于 `2*p`。
- `RuntimeError`
  - 无可用芯片。
  - 所有候选芯片执行失败。
  - 期望值结构与 observable 列表不匹配。

## 示例

```python
from quantum_hw import QuantumHardwareClient, QAOARunner

client = QuantumHardwareClient()
runner = QAOARunner(
  client=client,
  p=2,
  shots=2048,
  max_iters=25,
  learning_rate=0.12,
  seed=42,
)

result = runner.run_model(
    name="qaoa_maxcut_6q",
    num_qubits=6,
    problem="maxcut",
    edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    prefer_chips=["Simulator"],
)

print(result.best_cost)
print(result.best_params)
```

## 行为细节 / 注意事项

- 参数移位梯度每个参数需要两次评估；单轮理论评估次数约为 `1 + 4p`。
- 当 `target_qubits is None` 且后端容量足够时，梯度评估会走打包并行路径；失败时自动回退更小 batch。
- 当指定 `target_qubits` 时，会关闭打包并行，改为逐参数评估，保证固定映射。
- `QAOARunner.run_model` 已同时支持 `maxcut` 与 `custom`，无需再区分两个 runner 方法。

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [ShadowTomography.run](./shadow_tomography.md)
- [result types](../core/result_types.md)
