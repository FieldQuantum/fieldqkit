# ShadowTomography.run

## 概览

- **模块**：`quantum_hw.algorithms.shadow`
- **当前推荐入口**：`ShadowTomography.run(...)`

## 推荐签名（`ShadowTomography.run`）

```python
ShadowTomography(client, seed=None)

run(
  circuit,
  name,
  num_qubits,
  *,
  provider: str = "quafu",
  shots=8192,
  shots_per_basis=1,
  observables=None,
  zne=False,
  estimator="mean",
  mom_groups=None,
  target_qubits=None,
  prefer_chips=None,
  max_wait_time: int = 3600,
  sleep_time: int = 5,
) -> ShadowResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `client` | `QuantumHardwareClient` | - | 是 | `ShadowTomography` 初始化参数。 |
| `circuit` | `str \| QuantumCircuit` | - | 是 | 线路输入（内置名称、OpenQASM2/3 字符串或 `QuantumCircuit`）。 |
| `name` | `str` | - | 是 | 任务名前缀，内部会追加 `_shadow`。 |
| `num_qubits` | `int` | - | 是 | 逻辑比特数。 |
| `provider` | `str` | `"quafu"` | 否 | 平台名，支持 `quafu/tianyan/guodun/tencent`，或指定 `"Simulator"`。 |
| `shots` | `int` | `8192` | 否 | 总采样预算。实际会被按 batch 分配。 |
| `shots_per_basis` | `int` | `1` | 否 | 每个随机测量基的 shots。batch 数量为 `ceil(shots / shots_per_basis)`。 |
| `observables` | `Optional[Sequence[str] \| str]` | `None` | 否 | 待估计的 Pauli 字符串；可传单个字符串。 |
| `zne` | `bool` | `False` | 否 | 是否启用 ZNE（会同时运行基线与 3x 噪声缩放数据）。 |
| `estimator` | `str` | `"mean"` | 否 | 估计器：`"mean"` 或 `"mom"`（median-of-means）。 |
| `mom_groups` | `Optional[int]` | `None` | 否 | `estimator="mom"` 时分组数；为空则使用 `max(1, int(sqrt(nshots)))`。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定物理比特映射。 |
| `prefer_chips` | `Optional[Sequence[str] \| str]` | `None` | 否 | 限制候选芯片（可传 `"Simulator"`）。 |
| `max_wait_time` | `int` | `3600` | 否 | 任务查询最大等待时间（秒）。 |
| `sleep_time` | `int` | `5` | 否 | 查询轮询间隔（秒）。 |
| `seed` | `Optional[int]` | `None` | 否 | `ShadowTomography` 初始化时设置的随机种子。 |

## 低层接口（手动指定后端）

```python
run_shadow_with_backend(
  client,
  qc,
  *,
  name,
  num_qubits,
  backend,
  chip_name,
  shots,
  shots_per_basis=1,
  observables=None,
  estimator="mean",
  mom_groups=None,
  target_qubits=None,
  zne=False,
  seed=None,
  qasm_version="2.0",
  use_dd=True,
  submit_options=None,
  convert_single_qubit_gate_to_u=True,
) -> ShadowResult
```

- `run_shadow_with_backend` 不做芯片筛选，适合已有 `Backend` 的高级流程。

## 返回值

返回 `ShadowResult`（定义于 `quantum_hw.core.types`）：

- `task_ids: Optional[List[str]]`
- `samples: Optional[List[List[int]]]`
  - 扁平样本列表，每条样本对应一次随机测量基下的单次测量结果。
- `basis_patterns: Optional[List[List[str]]]`
  - 与 `samples` 等长；每条样本对应一个测量基模式（元素为 `"X"/"Y"/"Z"`）。
- `observables: Optional[List[str]]`
- `observable_estimates: Optional[Dict[str, float]]`
  - 最终估计值；当 `zne=True` 时是外推后的结果。
- `observable_estimates_raw: Optional[Dict[str, float]]`
  - 仅 `zne=True` 时有值，表示未外推的基线估计。
- `observable_stderr: Optional[Dict[str, float]]`
  - 与最终估计对应的标准误。
- `observable_stderr_raw: Optional[Dict[str, float]]`
  - 仅 `zne=True` 时有值，对应基线估计的标准误。
- `num_samples: Optional[int]`

## 异常与报错

- `ValueError`
  - `estimator` 不是 `"mean"` 或 `"mom"`。
  - `estimate_observables(...)` 输入维度不合法（例如样本不是二维数组）。
- `RuntimeError`
  - 无可用芯片（`ShadowTomography.run` 路径）。
  - 返回样本块与生成的 `basis_patterns` 数量不一致。
  - 所有候选芯片都执行失败（`ShadowTomography.run` 路径）。

## 示例

```python
from quantum_hw import QuantumHardwareClient, ShadowTomography

client = QuantumHardwareClient()
shadow = ShadowTomography(client=client, seed=42)

result = shadow.run(
    circuit="ghz",
    name="shadow_ghz_6q",
    num_qubits=6,
    shots=4096,
    shots_per_basis=8,
    observables=["ZZIIII", "ZZZZII", "ZZZZZZ"],
    estimator="mom",
    mom_groups=16,
    zne=True,
    prefer_chips="Simulator",
)

print(result.observable_estimates)
print(result.observable_estimates_raw)
print(result.num_samples)
```

## 行为细节 / 注意事项

- `shots_per_basis` 越大，随机基数量越少；统计方差与测量基覆盖会相互权衡。
- `estimator="mom"` 对重尾噪声更稳健，但在样本量较小时方差可能变大。
- 当 `observables=None` 时，依旧会执行 shadow 采样，但估计结果字典为空。
- ZNE 路径下，文档中“raw”字段均对应 1x 噪声数据，最终字段为线性外推结果。
- 当 `prefer_chips="Simulator"` 时，底层测量仍通过统一仿真接口执行（详见 sim interface 文档）。

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [result types](../core/result_types.md)
- [simulator interface](../sim/interface.md)
