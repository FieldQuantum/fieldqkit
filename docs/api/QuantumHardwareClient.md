# QuantumHardwareClient

## 概览

- **模块**：`quantum_hw.api.client`
- **定位**：统一封装“线路输入规范化 → 芯片选择 → 编译执行 → 结果后处理”。
- **公开入口**：推荐优先使用 `run_auto(...)`。

## 构造

```python
from quantum_hw.api.client import QuantumHardwareClient

client = QuantumHardwareClient()
```

构造后会初始化：

- `client.tmgr`：`Task` 实例（用于提交/查询任务）。
- `client.chip_name`：最近一次运行使用的芯片名。
- `client.chip_backend`：最近一次运行绑定的 `Backend`。

## 主要方法

### `build_circuit(kind: str, **kwargs) -> QuantumCircuit`

快速构建内置常见线路。

```python
build_circuit(kind: str, **kwargs) -> QuantumCircuit
```

支持的 `kind`（大小写不敏感）：

- `"ghz"` → `build_ghz(**kwargs)`
- `"cluster"` → `build_cluster(**kwargs)`
- `"qft"` → `build_qft(**kwargs)`
- `"ising" / "ising_time_evolution" / "ising_time"` → `build_ising_time_evolution(**kwargs)`

若 `kind` 不受支持会抛出 `ValueError`。

示例：

```python
qc = client.build_circuit("ghz", num_qubits=6, measure=False)
res = client.run_auto(circuit=qc, name="ghz_from_builder", num_qubits=6)
```

### `run_auto(...) -> RunResult`

```python
run_auto(
        circuit,
        name,
        num_qubits,
        *,
        shots=8192,
        zne=False,
        readout_mitigation=False,
        readout_shots=None,
        observables=None,
        return_probabilities=False,
        target_qubits=None,
        prefer_chips=None,
        rank_weights=None,
        print_true=True,
) -> RunResult
```

#### 参数说明

- `circuit: str | QuantumCircuit`
    - 支持 4 类输入：
        1. 预置线路名（`"ghz" / "cluster" / "qft" / "ising"` 等）
        2. OpenQASM2 字符串（以 `OPENQASM 2.0` 开头）
        3. OpenQASM3 字符串（以 `OPENQASM 3.0` 开头）
        4. `QuantumCircuit` 对象
    - 若传入 `QuantumCircuit` 且已包含 `measure`，会先移除，再按测量基重新追加。
- `name: str`：任务名前缀。
- `num_qubits: int`：逻辑比特数。
- `shots: int`：每次运行采样次数。
- `zne: bool`：是否启用 ZNE（通过 CZ tripling + 线性外推）。
- `readout_mitigation: bool`：是否启用读出误差缓解。
- `readout_shots: Optional[int]`：读出校准 shots，不给时使用校准模块默认值。
- `observables: Optional[Sequence[str] | str]`：可观测量（Pauli string 或其列表）。
- `return_probabilities: bool`：是否返回概率分布（含 raw 与处理后结果）。
- `target_qubits: Optional[Sequence[int]]`：指定物理比特映射。
- `prefer_chips: Optional[Sequence[str] | str]`：限制候选芯片集合。
- `rank_weights: Optional[Dict[str, float]]`：芯片排序权重，键为 `queue/nqubits/error`。
- `print_true: bool`：打印调试信息。

#### 返回值 `RunResult`

`RunResult` 定义于 `quantum_hw.core.types`，字段如下：

- `task_ids: Optional[List[str]]`
    - 硬件模式下返回任务 ID 列表；模拟器模式通常为 `None`。
- `samples: List[List[List[int]]]`
    - 每个测量组一份样本；外层列表长度等于测量组数量。
- `samples_zne: Optional[List[List[List[int]]]]`
    - 仅当 `zne=True` 时返回，对应噪声缩放 `scale=3` 的样本。
- `probabilities: List[List[float]]`
    - 若 `return_probabilities=False`，返回空列表 `[]`。
    - 若启用 readout/ZNE，此字段是处理后的结果。
- `probabilities_raw: List[List[float]]`
    - 若 `return_probabilities=False`，返回空列表 `[]`。
    - 表示未缓解的原始概率。
- `observable_values: Dict[str, float]`
    - 可观测量最终估计值（可被 readout/ZNE 更新）。
- `observable_values_raw: Dict[str, float]`
    - 可观测量原始估计值（不含缓解）。

## 执行行为说明

`run_auto` 内部流程：

1. 规范化线路输入（内置名称 / OpenQASM2/3 / `QuantumCircuit`）。
2. 调用 `rank_chips(...)` 选择候选芯片。
3. 使用 `Transpiler` 编译到目标后端门集。
4. 对 `observables` 进行分组并追加测量基。
5. 执行硬件任务或本地模拟。
6. 可选执行 readout 缓解与 ZNE 外推，最终组装 `RunResult`。

## 异常与约束

- `num_qubits` 与输入 `QuantumCircuit` 不一致时抛 `ValueError`。
- `readout_mitigation=True` 且 `len(target_qubits) != num_qubits` 时抛 `ValueError`。
- 无可用芯片时抛 `RuntimeError`。
- 若硬件任务最终状态不是 `Finished`，抛 `RuntimeError`。

## 示例

```python
from quantum_hw import QuantumHardwareClient

client = QuantumHardwareClient()

result = client.run_auto(
        circuit="ghz",
        name="ghz_demo",
        num_qubits=6,
        shots=4096,
        observables=["ZZIIII", "IIZZII"],
        return_probabilities=True,
        zne=False,
        readout_mitigation=False,
)

print(result.observable_values)
print(result.probabilities[0][:8])
```

## 相关页面

- [`rank_chips`](./rank_chips.md)
- [`Backend`](./Backend.md)
- [`Task`](./Task.md)
