# run_with_backend

## 概览

- 模块：`quantum_hw.api.client`
- 实际函数名：`QuantumHardwareClient._run_with_backend(...)`
- 作用：在给定 backend/chip 条件下，统一执行编译、提交、采样、ZNE、readout 缓解并返回 `RunResult`。

> 说明：当前源码公开的高层入口是 `run_auto(...)`。本页面记录的是供算法层复用的低层执行接口。

## 签名

```python
_run_with_backend(
    qc: QuantumCircuit,
    name: str,
    num_qubits: int,
    *,
    backend: Backend,
    chip_name: str,
    shots: int = 1024,
    zne: bool = False,
    readout_mitigation: bool = False,
    readout_shots: int | None = None,
    observables: Sequence[str] | str | None = None,
    return_probabilities: bool = False,
    target_qubits: Sequence[int] | None = None,
    merge_groups: bool = True,
    qasm_version: str = "2.0",
    use_dd: bool = True,
    print_true: bool = False,
    transpile: bool = True,
    submit_options: Dict[str, object] | None = None,
    convert_single_qubit_gate_to_u: bool = True,
) -> RunResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `qc` | `QuantumCircuit` | - | 是 | 待执行线路。 |
| `name` | `str` | - | 是 | 任务前缀。 |
| `num_qubits` | `int` | - | 是 | 逻辑比特数。 |
| `backend` | `Backend` | - | 是 | 已解析的后端拓扑对象。 |
| `chip_name` | `str` | - | 是 | 芯片名；`"Simulator"` 走本地分支。 |
| `shots` | `int` | `1024` | 否 | 每个测量任务的 shots。 |
| `zne` | `bool` | `False` | 否 | 是否启用 ZNE（scale=1 与 scale=3 线性外推）。 |
| `readout_mitigation` | `bool` | `False` | 否 | 是否做读出误差缓解。 |
| `readout_shots` | `Optional[int]` | `None` | 否 | readout 校准 shots。 |
| `observables` | `Optional[Sequence[str] \| str]` | `None` | 否 | 可观测量列表；为空时仅返回采样/概率。 |
| `return_probabilities` | `bool` | `False` | 否 | 是否返回概率向量。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定目标物理比特。 |
| `merge_groups` | `bool` | `True` | 否 | 是否按可共测规则合并 observable 组。 |
| `qasm_version` | `str` | `"2.0"` | 否 | 硬件提交时导出 OpenQASM 版本。仅支持 `"2.0"`；传入其他值会抛出 `ValueError`。 |
| `use_dd` | `bool` | `True` | 否 | transpile 时是否启用 DD。 |
| `print_true` | `bool` | `False` | 否 | 是否打印日志。 |
| `transpile` | `bool` | `True` | 否 | 是否先在客户端编译。**含噪线路会被强制置为 `False`**（覆盖用户传入）。 |
| `submit_options` | `Optional[Dict[str, object]]` | `None` | 否 | 任务提交附加选项，透传到 task adapter。 |
| `convert_single_qubit_gate_to_u` | `bool` | `True` | 否 | 是否将单比特门转换为 U 门；Tencent 平台需设为 `False`。 |

## 返回值

返回 `RunResult`，字段定义见 `quantum_hw.core.types`：

- `task_ids`
- `samples`
- `samples_zne`
- `probabilities`
- `probabilities_raw`
- `observable_values`
- `observable_values_raw`

## 执行流程

1. 校验 `qasm_version == "2.0"`；标准化 `observables`，并预计算每个 observable 的 support。
2. 调用 `is_noisy_circuit_for_backend(qc, chip_name)`：含噪线路强制 `transpile=False`（且后续跳过基变换 `_translate_to_basis`）；若目标非模拟器后端则在此抛 `ValueError`。
3. 若 `_active_task_adapter` 为空且 `chip_name != "Simulator"`，通过 `infer_provider_from_chip(chip_name)` 自动建立 runtime（**会触发该 provider 的 token 解析**）。
4. 按是否可共测分组（`merge_groups=True` 时调用 `group_observables`）。
5. 预编译一次基线路（`transpile=True`，含噪线路除外），每组仅追加基变换和测量。
6. `chip_name="Simulator"` 时直接本地 `simulate_counts`（含噪线路会自动路由到密度矩阵后端；若线路含显式 `measure` 门，simulator 会自动投影到 cbit 子空间）。
7. 硬件模式下逐组异步提交任务，随后统一轮询与取结果。
   - 结果解析时从 counts key 推断 bit 宽度（支持部分测量投影场景）。
   - TianYan provider 不支持批量提交，本函数会自动切换到顺序提交模式（提交一条 → 等完成 → 再提交下一条）。
8. 若启用 ZNE，额外执行 scale=3 线路并线性外推。
9. 若启用 readout mitigation，调用 `ReadoutCalibrationManager` 获取 confusion matrix 并做概率/observable 缓解。
10. 汇总并返回 `RunResult`。

## 异常与约束

- `ValueError`
  - `qasm_version` 不是 `"2.0"`。
  - `target_qubits` 与 `num_qubits` 不匹配。
  - `target_qubits` 未覆盖线路实际使用比特。
  - `num_qubits` 与 `target_qubits` 长度不一致时启用 readout mitigation。
- `RuntimeError`
  - 硬件任务状态不是 `Finished`。
  - 获取任务结果时缺少激活 task adapter。
  - 自动 provision 时 `infer_provider_from_chip` 返回 `None`。
- 含噪线路约束
  - 含噪线路（`depolarize` / `amplitude_damping` 等）仅可在 `simulator` / `fieldquantum_sim` 上运行；目标为真机时 `is_noisy_circuit_for_backend` 抛 `ValueError`。
  - 含噪线路强制跳过转译与基变换。

## 示例

```python
from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.circuit import QuantumCircuit

client = QuantumHardwareClient()
backend = Backend("Simulator")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

res = client._run_with_backend(
    qc=qc,
    name="manual_backend",
    num_qubits=2,
    backend=backend,
    chip_name="Simulator",
    shots=2048,
    observables=["Z0 Z1"],
    return_probabilities=True,
    transpile=True,
)

print(res.observable_values)
```

## 相关页面

- [QuantumHardwareClient](./QuantumHardwareClient.md)
- [Backend](./Backend.md)
- [Task](./Task.md)
- [readout](../core/readout.md)
- [zne](../core/zne.md)
