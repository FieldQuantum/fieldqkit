# QAOARunner.run_model

## 概览

- **模块**：`quantum_hw.algorithms.qaoa`
- **作用**：将 QAOA 组合优化问题（MaxCut / 自定义代价项）映射到量子线路，用 Adam 优化器最小化代价函数。
- **ansatz 结构**：初态 $|+\rangle^{\otimes n}$，每层包含 $\text{RZZ}(\gamma)$ 代价层 + $\text{RX}(\beta)$ 混合层。
- **当前推荐入口**：`QAOARunner.run_model(...)`
- **梯度方式**：`parameter-shift`（硬件兼容）或 `autograd`（Simulator only，基于 torch）。
- **噪声缓解**：支持 Clifford fitting（仅 parameter-shift 路径）。

## 推荐签名（`QAOARunner.run_model`）

```python
QAOARunner(
  client,
  p=1,
  shots=1024,
  max_iters=30,
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
)

run_model(
    name,
    num_qubits,
    edges,
  *,
    provider="quafu",
    target_qubits=None,
    init_params=None,
    callback=None,
    prefer_chips=None,
) -> QAOAResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `client` | `QuantumHardwareClient` | - | 是 | `QAOARunner` 初始化参数。 |
| `name` | `str` | - | 是 | 任务名前缀。 |
| `num_qubits` | `int` | - | 是 | 逻辑比特数（图节点数）。 |
| `edges` | `Sequence[Tuple[int, int]]` | - | 是 | 图的边列表，用于构建 ansatz 的 RZZ 层。 |
| `p` | `int` | `1` | 否 | `QAOARunner` 初始化参数：QAOA 层数。 |
| `shots` | `int` | `1024` | 否 | `QAOARunner` 初始化参数：每次评估 shots。 |
| `max_iters` | `int` | `30` | 否 | `QAOARunner` 初始化参数：迭代轮数。 |
| `learning_rate` | `float` | `0.1` | 否 | `QAOARunner` 初始化参数：Adam 学习率。 |
| `beta1` / `beta2` / `eps` | `float` | `0.9/0.98/1e-8` | 否 | `QAOARunner` 初始化参数：Adam 超参数。 |
| `shift` | `float` | `π/2` | 否 | `QAOARunner` 初始化参数：参数移位角。 |
| `zne` | `bool` | `False` | 否 | `QAOARunner` 初始化参数：是否启用 ZNE。 |
| `readout_mitigation` | `bool` | `False` | 否 | `QAOARunner` 初始化参数：是否启用 readout 缓解。 |
| `clifford_fitting` | `bool` | `False` | 否 | `QAOARunner` 初始化参数：是否启用 Clifford fitting。 |
| `seed` | `Optional[int]` | `None` | 否 | `QAOARunner` 初始化参数：RNG 种子。 |
| `gradient_method` | `Literal["parameter-shift", "autograd"]` | `"parameter-shift"` | 否 | 梯度计算方式。`autograd` 仅限 Simulator。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 物理比特映射。 |
| `init_params` | `Optional[Sequence[float]]` | `None` | 否 | 自定义初始参数（长度 = 2p）。 |
| `callback` | `Optional[Callable]` | `None` | 否 | 每轮回调 `(iter, cost, params)`。 |
| `prefer_chips` | `Optional[Sequence[str] \| str]` | `None` | 否 | 候选芯片过滤。 |

## 返回值

`QAOAResult`，包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `best_cost` | `float` | 历史最优 cost。 |
| `best_params` | `List[float]` | 最优参数。 |
| `cost_history` | `List[float]` | 每轮 cost。 |
| `params_history` | `Optional[List[List[float]]]` | 每轮更新后的参数。 |
| `grad_history` | `Optional[List[List[float]]]` | 每轮梯度。 |
| `last_expectations` | `Optional[Dict[str, float]]` | 最后一轮的各 observable 估计。 |
| `clifford_fitting` | `Optional[Dict[str, Dict[str, float]]]` | Clifford 拟合系数。 |

## 快速示例

```python
from quantum_hw import QuantumHardwareClient
from quantum_hw.algorithms.qaoa import QAOARunner

client = QuantumHardwareClient()
runner = QAOARunner(client, p=2, gradient_method="autograd")

result = runner.run_model(
    name="maxcut_demo",
    num_qubits=4,
    edges=[(0,1), (1,2), (2,3), (0,3)],
    provider="simulator",
    model="maxcut",
    max_iters=30,
)

print(f"Best cost: {result.best_cost:.4f}")
```

## 底层函数

- `build_maxcut_hamiltonian(edges, num_qubits)` — 构建 MaxCut ZZ 代价项。
- `build_qaoa_ansatz_symbolic(num_qubits, edges, p)` — 构建符号化 QAOA ansatz 线路。
- `run_qaoa_with_backend(...)` — 底层优化循环（支持 parameter-shift / autograd）。

## 共享优化工具

QAOA 与 VQE 共用 `optimizer_utils.py` 中的基础设施：

- 能量估算 (`evaluate_energy_with_backend`)
- 参数移位梯度 (`parameter_shift_gradient`)
- Adam 更新 (`adam_update`)
- Clifford fitting (`build_clifford_fit_map`)

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [result types — QAOAResult](../core/result_types.md)
- [QAOA 教程 notebook](../../examples/demo_qaoa.ipynb)
