# QuantumHardwareClient

## 概览

- 模块：`quantum_hw.api.client`
- 作用：统一封装线路输入标准化、硬件自动选择、任务提交、结果聚合，以及 ZNE/readout 缓解流程。
- 推荐入口：`run_auto(...)`
- 低层执行入口：`_run_with_backend(...)`（供算法层与高级用户复用）

## 推荐签名

```python
QuantumHardwareClient()

run_auto(
    circuit: str | QuantumCircuit,
    name: str,
    num_qubits: int,
    *,
    provider: str = "quafu",
    shots: int = 8192,
    zne: bool = False,
    readout_mitigation: bool = False,
    readout_shots: int | None = None,
    observables: Sequence[str] | str | None = None,
    return_probabilities: bool = False,
    target_qubits: Sequence[int] | None = None,
    prefer_chips: Sequence[str] | str | None = None,
    transpile_on_client: bool = True,
    clifford_fitting: bool = False,
    clifford_fitting_num_samples: int = 8,
    clifford_fitting_num_non_clifford_gates: int = 0,
    clifford_fitting_seed: int | None = None,
    max_wait_time: int = 3600,
    sleep_time: int = 5,
    print_true: bool = True,
) -> RunResult
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `circuit` | `str \| QuantumCircuit` | - | 是 | 支持三类输入：预置线路名（如 `"ghz"`）、OpenQASM2/3 字符串、`QuantumCircuit` 对象。 |
| `name` | `str` | - | 是 | 任务名前缀。 |
| `num_qubits` | `int` | - | 是 | 本次任务逻辑比特数。 |
| `provider` | `str` | `"quafu"` | 否 | 平台名，支持 `quafu/tianyan/guodun/tencent`（大小写不敏感）。 |
| `shots` | `int` | `8192` | 否 | 每个测量任务采样次数。 |
| `zne` | `bool` | `False` | 否 | 是否启用零噪声外推（当前通过 CZ tripling + 线性外推实现）。 |
| `readout_mitigation` | `bool` | `False` | 否 | 是否启用读出误差缓解。 |
| `readout_shots` | `Optional[int]` | `None` | 否 | 读出校准 shots；`None` 时使用校准模块默认值。 |
| `observables` | `Optional[Sequence[str] \| str]` | `None` | 否 | 待测 Pauli 可观测量列表或单项字符串。 |
| `return_probabilities` | `bool` | `False` | 否 | 是否返回概率向量。 |
| `target_qubits` | `Optional[Sequence[int]]` | `None` | 否 | 指定物理比特映射。 |
| `prefer_chips` | `Optional[Sequence[str] \| str]` | `None` | 否 | 候选芯片白名单，可显式传 `"Simulator"`。 |
| `transpile_on_client` | `bool` | `True` | 否 | `True` 时客户端先编译再提交。开启 `clifford_fitting` 时该编译被框架层复用为校准线路的模板。 |
| `clifford_fitting` | `bool` | `False` | 否 | 是否在框架层对 `observables` 启用 Clifford-随机化仿射校正（仅在 `observables` 非空时生效）。流程与 `run_vqe` / `run_qaoa` 对齐：先在客户端一次性编译模板，然后用该模板在硬件上提交主任务及 `clifford_fitting_num_samples` 条校准线路；理想期望由 `sim.clifford`（Heisenberg picture，$O(g\cdot n)$）计算，非 Clifford 门部分回退到 `sim.clifford_t` 的分支展开，最终落到 statevector。 |
| `clifford_fitting_num_samples` | `int` | `8` | 否 | 校准线路条数。 |
| `clifford_fitting_num_non_clifford_gates` | `int` | `0` | 否 | 每条校准线路中替换为 Haar 随机 U3 的单比特门个数（其余替换为 24 个 Clifford U3 之一）。 |
| `clifford_fitting_seed` | `Optional[int]` | `None` | 否 | 校准采样的 RNG 种子。 |
| `max_wait_time` | `int` | `3600` | 否 | 任务查询最大等待时间（秒），透传到 provider task adapter。 |
| `sleep_time` | `int` | `5` | 否 | 查询轮询间隔（秒），透传到 provider task adapter。 |
| `print_true` | `bool` | `True` | 否 | 是否打印运行日志。 |

## 返回值

返回 `RunResult`（`quantum_hw.core.types.RunResult`）：

- `task_ids: Optional[List[str]]`：硬件任务 ID 列表（Simulator 为 `None`）。
- `samples: List[List[List[int]]]`：每个观测分组对应的样本矩阵。
- `samples_zne: Optional[List[List[List[int]]]]`：启用 ZNE 时的噪声放大样本。
- `probabilities: List[List[float]]`：处理后的概率向量（含可选 ZNE/REM）。
- `probabilities_raw: List[List[float]]`：未缓解原始概率。
- `observable_values: Dict[str, float]`：处理后的可观测量期望值。
- `observable_values_raw: Dict[str, float]`：原始期望值。

## 关键方法

### `build_circuit(kind, **kwargs) -> QuantumCircuit`

**签名：**
```python
def build_circuit(self, kind: str, **kwargs) -> QuantumCircuit
```

**用途：** 构建预定义的量子线路。

**支持的线路类型及参数：**

#### `kind="ghz"` - GHZ 纠缠态
```python
client.build_circuit("ghz", num_qubits=4)
```
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---:|---|
| `num_qubits` | `int` | 是 | - | GHZ 态比特数（最少 2）。 |
| `measure` | `bool` | 否 | `False` | 是否添加末尾测量。 |

**示例：** 4 比特 GHZ 态 = H(q0) + CX(0,1) + CX(1,2) + CX(2,3)

---

#### `kind="cluster"` - 1D 簇态
```python
client.build_circuit("cluster", num_qubits=6)
```
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---:|---|
| `num_qubits` | `int` | 是 | - | 簇态比特数。 |
| `measure` | `bool` | 否 | `False` | 是否添加末尾测量。 |

**示例：** 6 比特簇态 = H(all) + CZ(0,1) + CZ(1,2) + CZ(2,3) + CZ(3,4) + CZ(4,5)

---

#### `kind="qft"` - 量子傅里叶变换
```python
client.build_circuit("qft", num_qubits=8, with_swaps=True)
```
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---:|---|
| `num_qubits` | `int` | 是 | - | QFT 比特数。 |
| `measure` | `bool` | 否 | `False` | 是否添加末尾测量。 |
| `with_swaps` | `bool` | 否 | `True` | 是否包含 bit-reversal swap。 |

**说明：** 包含受控相位转旋，可选 bit-reversal 交换来调整输出顺序。

---

#### `kind="ising"` / `"ising_time_evolution"` / `"ising_time"` - Ising 时间演化
```python
client.build_circuit("ising", num_qubits=6, j=0.5, h=1.0, t=1.0, steps=5)
```
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|:---:|---:|---|
| `num_qubits` | `int` | 是 | - | Ising 模型比特数。 |
| `j` | `float` | 是 | - | ZZ 耦合强度系数；越大两比特相互作用越强。 |
| `h` | `float` | 是 | - | X 磁场强度系数；越大量子磁场效应越强。 |
| `t` | `float` | 是 | - | 拓扑演化时间；越大演化时间越长。 |
| `steps` | `int` | 否 | `1` | Trotter 分解步数；步数越多精度越高。 |
| `measure` | `bool` | 否 | `False` | 是否添加末尾测量。 |

**说明：** 一阶 Trotter 分解：每步内先 ZZ 相互作用（CX-RZ-CX），后 X 旋转。dt = t/steps。

---

**返回值：** `QuantumCircuit` 对象。

**异常：**
- `ValueError`：`kind` 不在支持列表中；或缺少必填参数。
- `KeyError`：缺少必填的 `**kwargs` 字段。

**示例：**
```python
client = QuantumHardwareClient()

# GHZ 态
qc_ghz = client.build_circuit("ghz", num_qubits=4)

# Ising 时间演化（4 个比特，ZZ 耦合 0.5，X 磁场 1.0，演化 0.5 秒，5 步）
qc_ising = client.build_circuit("ising", num_qubits=4, j=0.5, h=1.0, t=0.5, steps=5)

# QFT 带 swap
qc_qft = client.build_circuit("qft", num_qubits=8, with_swaps=True)
```

### `_transpile_with_backend(...) -> QuantumCircuit`

- 作用：调用编译流水线对线路进行转译。
- 签名：

```python
def _transpile_with_backend(
    self, qc, backend, target_qubits=None, use_dd=True,
    use_three_qubit_decompose=True, use_sabre_routing=True,
    use_translate_to_basis=True, use_gate_compressor=True,
    noise_aware=None, routing_n_trials=1,
    convert_single_qubit_gate_to_u=None,
) -> QuantumCircuit
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `noise_aware` | `bool \| None` | `None` | 路由时使用保真度加权距离矩阵。 |
| `routing_n_trials` | `int` | `1` | SABRE 多随机初始映射试验数。 |
| `use_gate_compressor` | `bool` | `True` | 是否启用门压缩（含两比特门对消除）。 |
| `use_dd` | `bool` | `True` | 是否启用动力学去耦。 |
| `convert_single_qubit_gate_to_u` | `bool \| None` | `None` | 是否将单比特门转换为 U 门；`None` 时由 Transpiler 自动推断。 |

### `_normalize_input_circuit(circuit, num_qubits, *, observables=None) -> QuantumCircuit`

- 作用：将输入标准化为 `QuantumCircuit`，并根据 `observables` 决定是否保留用户测量门。
- 当 `observables` 不为空且线路已含 `measure` 门时：发出 warning 并移除已有测量（后续由 observable 基变换重新添加）。
- 当 `observables` 为空或 `None` 时：保留用户指定的测量门不做修改。

### `_run_with_backend(...) -> RunResult`

- 作用：在已解析 backend 条件下执行统一流程。
- 主要步骤：
  - 可观测量分组（减少任务数量）
  - 基变换与测量附加
  - 可选本地编译
  - 硬件异步提交或本地模拟
  - 可选 ZNE 与 readout mitigation
  - 统一汇总 `RunResult`
- **部分测量支持**：当用户线路包含显式 `measure` 门（含 qubit→cbit 映射）且不提供 observables 时，返回的 `samples` 和 `probabilities` 基于经典比特子空间（宽度 = `max(cbit) + 1`），而非全 qubit 空间。

### `_submit_openqasm_async(...) -> ProviderTaskHandle`

- 作用：通过当前激活 `TaskAdapter` 提交 OpenQASM 异步任务。

## 异常与约束

- `ValueError`
  - `provider` 非法（由 provider runtime 工厂抛出）。
  - `num_qubits` 与输入线路不一致。
  - `target_qubits` 覆盖不完整或长度不匹配。
- `RuntimeError`
  - 未设置激活 task adapter/后端却尝试提交任务。
  - 任务状态非 `Finished`。

## 示例

```python
from quantum_hw.api.client import QuantumHardwareClient

client = QuantumHardwareClient()
res = client.run_auto(
    circuit="ghz",
    name="ghz_demo",
    num_qubits=4,
    provider="quafu",
    prefer_chips="Simulator",
    observables=["Z0 Z1", "X0 X1 X2 X3"],
    shots=4096,
    zne=False,
    readout_mitigation=False,
    print_true=False,
)

print(res.observable_values)
```

## 相关页面

- [run_with_backend](./run_with_backend.md)
- [Backend](./Backend.md)
- [Task](./Task.md)
- [provider_runtime](./provider_runtime.md)
