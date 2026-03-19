# run_with_backend

## 概览

- **模块**：`quantum_hw.api.client`
- **源码函数**：`QuantumHardwareClient._run_with_backend(...)`
- **作用**：在已确定 `Backend` 与 `chip_name` 的前提下，完成线路执行、可观测量估计、可选 ZNE 与 readout 缓解，并返回 `RunResult`。

> 这是执行链路的核心函数。`run_auto(...)` 在选片后会调用它。

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
        readout_shots: Optional[int] = None,
        observables: Optional[Sequence[str] | str] = None,
        return_probabilities: bool = False,
        target_qubits: Optional[Sequence[int]] = None,
        merge_groups: bool = True,
        qasm_version: str = "2.0",
        print_true: bool = False,
        transpile: bool = True,
) -> RunResult
```

## 参数说明

- `qc`：待执行的 `QuantumCircuit`。
- `name`：任务名前缀。
- `num_qubits`：逻辑比特数，用于解析样本与 observable 支持。
- `backend`：目标后端对象，决定拓扑和双比特门基。
- `chip_name`：目标芯片名。`"simulator"`（大小写不敏感）会走模拟路径。
- `shots`：每组采样次数。
- `zne`：是否执行 ZNE（CZ tripling + 线性外推）。
- `readout_mitigation`：是否执行 readout 缓解。
- `readout_shots`：readout 校准 shots。
- `observables`：可观测量（单个 Pauli string 或列表）。
- `return_probabilities`：是否返回概率分布结果。
- `target_qubits`：物理比特映射。
- `merge_groups`：是否将兼容测量基的 observables 合并成一个测量组。
- `qasm_version`：硬件提交序列化版本。`"2.0"` 走 `to_openqasm2`，否则走 `to_openqasm3`。
- `print_true`：是否打印执行过程日志。
- `transpile`：是否先执行 `Transpiler`。默认 `True`；传 `False` 时直接复用输入线路（常用于上层算法已完成预编译、仅替换参数再执行的场景）。

## 行为要点

1. 处理 `observables` 输入并构造测量组。
2. 根据 `transpile` 与 `chip_name` 决定是否编译。
        - `transpile=True`：执行 `_transpile_with_backend(...)`。
        - `transpile=False`：直接使用输入线路副本，避免重复编译开销。
        - `transpile=False` 且 `target_qubits=None`：默认使用 `base_qct.qubits` 顺序作为执行/测量物理比特顺序；若为空再退化到 `list(range(num_qubits))`。
3. 对每个测量组构造线路并执行：
   - 硬件：提交异步任务并等待完成。
   - 模拟器：直接调用 `simulate_counts(...)`。
4. 若启用 `zne`，每组增加 scale=3 线路并做线性外推。
5. 若启用 `readout_mitigation`，调用 `ReadoutCalibrationManager` 标定并做缓解。
6. 汇总 `samples / probabilities / observable_values` 并返回 `RunResult`。

## 异常与约束

- `readout_mitigation=True` 时，要求有效目标物理比特数等于 `num_qubits`，否则抛 `ValueError`。
- 硬件任务若最终状态不是 `Finished`，抛 `RuntimeError`。

## 示例

```python
from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend

client = QuantumHardwareClient()
qc = client._normalize_input_circuit("ghz", num_qubits=6)

backend = Backend("Simulator")

result = client._run_with_backend(
        qc,
        name="ghz_fixed_backend",
        num_qubits=6,
        backend=backend,
        chip_name="Simulator",
        shots=4096,
        observables=["ZZIIII", "IIZZII"],
        return_probabilities=True,
        zne=False,
        readout_mitigation=False,
        target_qubits=list(range(6)),
        merge_groups=True,
        qasm_version="2.0",
        transpile=True,
)

print(result.observable_values)
print(result.probabilities[0][:8])
```

#### 适用场景

- 你已经明确要跑某个芯片，不希望自动排序切换。
- 需要精细控制执行细节（如 `merge_groups`、`qasm_version`、`transpile`）。
- 在上层算法或实验框架中复用统一执行内核。

## 相关页面

- [`QuantumHardwareClient`](./QuantumHardwareClient.md)
- [`rank_chips`](./rank_chips.md)
- [`Backend`](./Backend.md)
- [`Task`](./Task.md)
