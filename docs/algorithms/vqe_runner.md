# VQERunner.run_model

## 概览

- **模块**：`fieldqkit.algorithms.vqe`
- **作用**：支持参数移位或 `torch autograd` 梯度，并用 Adam 做能量最小化。
- **ansatz 支持**：`hardwareefficient` / `custom`
- **当前推荐入口**：`VQERunner.run_model(...)`
- **Simulator 自动微分入口**：`fieldqkit.sim.energy_and_expectations`（由 sim 接口层按 qubit 数在 statevector/MPS 间分发）。
- **压缩能力（parameter-shift 路径）**：支持后缀分块规划 + stage 级压缩（prefix 用 `mps` 目标，suffix block 用 `mpo` 目标）。
- **硬件压缩执行路径**：压缩开启时使用“双模板”模式：
  - 梯度模板：原始 symbolic ansatz（每次参数化后再压缩）。
  - 执行模板：预编译 hardware-efficient 压缩模板（仅注入压缩参数后执行）。

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
  clifford_fitting=False,
  clifford_fitting_num_samples=8,
  clifford_fitting_num_non_clifford_gates=3,
  seed=None,
  gradient_method="parameter-shift",
  enable_block_planner=False,
  planner_bond_cap=128,
  planner_trunc_tol=1e-8,
  planner_max_layers_per_block=6,
  enable_circuit_compression=False,
  compression_block_layers=None,
  compression_optimizer_steps=20,
  compression_optimizer_lr=0.05,
  compression_verbose=False,
  compression_plot_loss=False,
)

run_model(
    name,
    num_qubits,
  *,
    provider="quafu",
    model="ising",
    model_params=None,
    hamiltonian=None,
    target_qubits=None,
    init_params=None,
    callback=None,
    prefer_chips=None,
    ansatz="hardwareefficient",
    custom_ansatz_circuit=None,
) -> VQEResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `client` | `QuantumHardwareClient` | - | 是 | `VQERunner` 初始化参数。 |
| `name` | `str` | - | 是 | 任务名前缀。 |
| `num_qubits` | `int` | - | 是 | 逻辑比特数。 |
| `provider` | `str` | `"quafu"` | 否 | 运行时 provider 名称。 |
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
| `clifford_fitting` | `bool` | `False` | 否 | `VQERunner` 初始化参数：是否启用基于 Clifford 随机线路的仿射校正。 |
| `clifford_fitting_num_samples` | `int` | `8` | 否 | `VQERunner` 初始化参数：Clifford 拟合采样条数。 |
| `clifford_fitting_num_non_clifford_gates` | `int` | `3` | 否 | `VQERunner` 初始化参数：每条拟合随机线路中保留为随机单比特 unitary（非 Clifford）的参数化门数量，其余参数化门仍替换为随机 Clifford。 |
| `enable_block_planner` | `bool` | `False` | 否 | 是否启用后缀分块规划。 |
| `planner_bond_cap` | `int` | `128` | 否 | 分块规划和压缩共用的 bond 上限。 |
| `planner_trunc_tol` | `float` | `1e-8` | 否 | 分块规划和压缩共用的截断误差阈值。 |
| `planner_max_layers_per_block` | `int` | `6` | 否 | 规划时每个后缀块最多层数。 |
| `enable_circuit_compression` | `bool` | `False` | 否 | 是否启用每次能量/梯度评估前的线路压缩。含噪线路不支持，置 `True` 会抛 `ValueError`。 |
| `compression_block_layers` | `Optional[int]` | `None` | 条件必填 | 启用压缩时必填，必须是单个正整数 `k`（压缩 ansatz 层数）。 |
| `compression_optimizer_steps` | `int` | `20` | 否 | 每次压缩优化步数。 |
| `compression_optimizer_lr` | `float` | `0.05` | 否 | 压缩优化学习率。 |
| `compression_verbose` | `bool` | `False` | 否 | 是否打印压缩统计。 |
| `compression_plot_loss` | `bool` | `False` | 否 | 是否绘制压缩 loss 曲线。 |
| `max_wait_time` | `int` | `3600` | 否 | `VQERunner` 初始化参数：任务查询最大等待时间（秒）。 |
| `sleep_time` | `int` | `5` | 否 | `VQERunner` 初始化参数：查询轮询间隔（秒）。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定物理比特映射。 |
| `seed` | `Optional[int]` | `None` | 否 | `VQERunner` 初始化参数：参数初始化随机种子。 |
| `gradient_method` | `Literal["parameter-shift", "autograd"]` | `"parameter-shift"` | 否 | 梯度计算方式。`autograd` 支持两种后端：`Simulator`（本地 torch 自动微分）和 `fieldquantum_sim`（单次 HTTP 调用服务端完成参数移位 + 梯度计算，需配置 `FIELDQUANTUM_API_TOKEN`（可选 `FIELDQUANTUM_SERVER_URL` 覆盖默认地址））。 |
| `init_params` | `Optional[Sequence[float]]` | `None` | 否 | 显式初始参数；长度必须等于 `2 * num_qubits * (layers + 1)`。 |
| `callback` | `Optional[Callable[[int, float, np.ndarray], None]]` | `None` | 否 | 每轮回调，参数为 `(iter_idx, energy, params)`。 |
| `prefer_chips` | `Optional[Sequence[str] \| str]` | `None` | 否 | 候选芯片限制（可传 `"Simulator"`）。 |
| `ansatz` | `Literal["hardwareefficient", "custom"]` | `"hardwareefficient"` | 否 | 变分线路类型。 |
| `custom_ansatz_circuit` | `Optional[QuantumCircuit]` | `None` | 否 | 当 `ansatz="custom"` 时必填；线路中的未解析字符串参数会被自动识别并优化。 |

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
  ansatz="hardwareefficient",
  custom_ansatz_circuit=None,
  clifford_fitting=False,
  clifford_fitting_num_samples=8,
  clifford_fitting_num_non_clifford_gates=3,
  enable_block_planner=False,
  planner_bond_cap=128,
  planner_trunc_tol=1e-8,
  planner_max_layers_per_block=6,
  enable_circuit_compression=False,
  compression_block_layers=None,
  compression_optimizer_steps=20,
  compression_optimizer_lr=0.05,
  compression_verbose=False,
  compression_plot_loss=False,
  qasm_version="2.0",
  use_dd=True,
  convert_single_qubit_gate_to_u=True,
  transpile=True,
  submit_options=None,
) -> VQEResult
```

> `VQERunner` 的 `max_wait_time` / `sleep_time` 会被打包进 `submit_options`，沿 `run_vqe_with_backend → run_variational_loop → evaluate_energy_with_backend → client._run_with_backend` 透传到 provider task adapter（与 `ShadowTomography.run` 一致）。

## Ansatz 行为说明

- `hardwareefficient`：默认 ansatz，参数个数为
  $$2 \times \text{num\_qubits} \times (\text{layers}+1)$$
- `custom`：由用户提供 `QuantumCircuit`，框架从 `params_value` 中自动提取仍未解析的字符串参数名作为优化变量。

## 私有方法（进阶拆解）

下面两个方法位于 `optimizer_utils.py`，是 `run_vqe_with_backend(...)` 的核心内部步骤，适合做算法拆解和二次开发时参考。

### `evaluate_energy_with_backend`

```python
evaluate_energy_with_backend(
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
  clifford_fit_map=None,
  target_qubits=None,
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

### `parameter_shift_gradient`

```python
parameter_shift_gradient(
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
  param_template,
  param_names,
) -> np.ndarray
```

- 作用：基于 parameter-shift 规则估计梯度向量。
- 公式：
  - 第 `i` 个分量使用 `0.5 * (E(theta_i + shift) - E(theta_i - shift))`。
- 要求：在当前流程中需要传入 `param_template` 与 `param_names`（用于仅替换参数值、避免重复编译）。
- 返回：`np.ndarray`，长度与 `params` 相同。

### 自动微分路径（`gradient_method="autograd"`）

- 触发条件：`run_vqe_with_backend(..., gradient_method="autograd")` 或 `VQERunner(gradient_method="autograd")`。
- 限制：
  - 仅支持 `chip_name="Simulator"`
- 实现入口：`fieldqkit.sim.energy_and_expectations`（包级导出）。
- 后端分发：由 sim 接口层按 qubit 数自动路由（`statevector` 或 `mps`）。
- 依赖：需要安装 `torch`。

### `custom` ansatz 注意事项

- 必须传入 `custom_ansatz_circuit`。
- `custom_ansatz_circuit.nqubits` 必须与 `num_qubits` 一致。
- 线路里需要存在未解析的字符串参数；若没有可优化参数会报错。
- `autograd` 模式下仍要求 `chip_name="Simulator"`。

## 返回值

返回 `VQEResult`（定义于 `fieldqkit.core.types`）：

- `best_energy: float`
- `best_params: List[float]`
- `energy_history: List[float]`
- `params_history: Optional[List[List[float]]]`
- `grad_history: Optional[List[List[float]]]`
- `last_expectations: Optional[Dict[str, float]]`
- `clifford_fitting: Optional[Dict[str, Dict[str, float]]]`
  - 形状：`{observable: {"a": float, "b": float}}`
  - 语义：每个 Hamiltonian 观测量各自拟合仿射校正关系
    $$\langle O \rangle_{ideal} \approx a \cdot \langle O \rangle_{noisy} + b$$

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
- 含噪 ansatz（含 `depolarize` / `amplitude_damping` 等噪声信道）
  - 仅可运行于 `simulator` / `fieldquantum_sim`，否则在 `is_noisy_circuit_for_backend` 抛 `ValueError`。
  - 强制 `transpile=False`。
  - 与 `enable_circuit_compression=True` 互斥：抛 `ValueError("circuit compression is not supported for noisy circuits")`。

## 示例

```python
from fieldqkit import QuantumHardwareClient
from fieldqkit.algorithms import VQERunner

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

### `custom` ansatz 示例

```python
from fieldqkit import QuantumHardwareClient
from fieldqkit.algorithms import VQERunner
from fieldqkit.circuit import QuantumCircuit

qc = QuantumCircuit(4)
qc.ry("alpha_0", 0)
qc.ry("alpha_1", 1)
qc.cz(0, 1)
qc.rx("alpha_2", 1)

client = QuantumHardwareClient()
runner = VQERunner(client=client, gradient_method="autograd", seed=7)

result = runner.run_model(
  name="h2_custom_ansatz",
  num_qubits=4,
  model="custom",
  hamiltonian=[(0.1, "Z0"), (-0.2, "Z1"), (0.05, "X0 X1")],
  ansatz="custom",
  custom_ansatz_circuit=qc,
  prefer_chips="Simulator",
)
```

## 行为细节 / 注意事项

- 参数维度随 ansatz 不同而变化：`hardwareefficient` 与 `custom` 不同。
- 每个参数梯度需要两次移位评估（`+shift/-shift`）；单轮理论评估次数约为 `1 + 2 * num_params` 次。
- 当 `gradient_method="autograd"` 且使用 `Simulator` 时，梯度由 `energy_t.backward()` 回传，不再执行 parameter-shift 线路采样。
- 当 `gradient_method="parameter-shift"` 时，VQE 会先在内部对参数化 ansatz 做一次预编译，然后每次迭代/移位只替换参数值并提交，避免重复 transpile。
- 当 `enable_circuit_compression=True` 时：
  - 仅在 `parameter-shift` 路径生效（`autograd` 路径不走压缩变换）。
  - `compression_block_layers` 必须是单个正整数。
  - 若 `enable_block_planner=True`，会先调用 `plan_hybrid_suffix_blocks(...)` 得到 `prefix + suffix blocks`，再按 stage 分别构造 target 子线路做拟合。
  - stage 目标模式固定为：prefix 使用 `objective_mode="mps"`，suffix block 使用 `objective_mode="mpo"`。
- 当 `clifford_fitting=True` 时：
  - 当前仅支持 `gradient_method="parameter-shift"`。
  - 会对编译后模板中**所有**单比特旋转门（`p`/`r`/`u`/`u3`/`rx`/`ry`/`rz`，无论参数是否仍为符号）进行 Clifford 随机化采样；与 `run_auto` 路径使用同一 `optimizer_utils.build_clifford_fit_map` 函数。
  - 拟合粒度为“每个 observable 一组 `(a,b)`”，不再是单一哈密顿量级别系数。
  - 校准电路的理想期望优先走 `sim.clifford`（Heisenberg picture，$O(g\cdot n)$），含 Haar 随机 U3 时回退到 `sim.clifford_t` 分支扩展，仍不可行时退回 statevector。

## 相关页面

- [run_with_backend](../api/run_with_backend.md)
- [result types](../core/result_types.md)
- [circuit compression](./circuit_compression.md)
- [ansatz templates](./ansatz_templates.md)
- [simulator interface](../sim/interface.md)
- [statevector simulator](../sim/statevector.md)
- [mps simulator](../sim/mps.md)
- [simulator common helpers](../sim/common.md)
