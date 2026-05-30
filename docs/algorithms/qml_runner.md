# QMLRunner — QML 高层入口

## 概览

- 模块：`fieldqkit.algorithms.qml_runner`
- 作用：与 `VQERunner` / `QAOARunner` 平行的高层封装，自动解析 provider → 候选芯片 → 后端，再委托底层 `run_pqc_classifier` / `run_qnn_unsupervised` / `run_qnn_conditional`。
- 候选芯片逐块尝试；任一块失败则记录并尝试下一块，全部失败时抛 `RuntimeError("all candidate chips failed")`。

> **autograd 限制**：QML 的 `autograd` **仅支持本地 `Simulator`**。若解析到的芯片是云端模拟器（`fieldquantum_sim`）或真机，底层函数会抛 `ValueError`（被 runner 捕获后视为该芯片失败）。需要在云端/真机训练时请用 `gradient_method="parameter-shift"`。

## 构造

```python
@dataclass
class QMLRunner:
    client: object
    layers: int = 2
    shots: int = 4096
    max_iters: int = 100
    learning_rate: float = 0.01
    seed: Optional[int] = None
    gradient_method: Literal["parameter-shift", "autograd"] = "parameter-shift"
    shift: float = np.pi / 2.0
    zne: bool = False
    readout_mitigation: bool = False
    convert_single_qubit_gate_to_u: bool = True
    gen_shots: int = 1024
    mmd_sigma: float = 1.0
```

| 字段 | 默认 | 说明 |
|---|---:|---|
| `client` | - | `QuantumHardwareClient` 实例。 |
| `layers` | `2` | ansatz 层数。 |
| `shots` | `4096` | 每次评估 shots。 |
| `max_iters` | `100` | 训练迭代轮数。 |
| `learning_rate` | `0.01` | Adam 学习率。 |
| `seed` | `None` | 随机种子。 |
| `gradient_method` | `"parameter-shift"` | `"parameter-shift"` 或 `"autograd"`（autograd 仅本地 Simulator）。 |
| `shift` | `π/2` | 参数移位角。 |
| `zne` | `False` | 是否启用 ZNE。 |
| `readout_mitigation` | `False` | 是否启用 readout 缓解。 |
| `convert_single_qubit_gate_to_u` | `True` | 转译时是否将单比特门转为 U 门（仅 parameter-shift 路径生效；tencent/fieldquantum 自动关闭）。 |
| `gen_shots` | `1024` | 无监督/条件任务训练结束后生成样本的 shots。 |
| `mmd_sigma` | `1.0` | MMD RBF 核带宽（parameter-shift 路径）。 |

## 方法

### `run_classifier(...) -> QMLResult`

```python
run_classifier(
    name: str,
    num_qubits: int,
    train_data: Sequence[Tuple[Sequence[float], int]],
    *,
    test_data: Optional[Sequence[Tuple[Sequence[float], int]]] = None,
    encoding: Union[str, Callable] = "angle",
    encoding_kwargs: Optional[dict] = None,
    num_classes: int = 2,
    measurement_qubits: Optional[Sequence[int]] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    provider: str = "quafu",
    prefer_chips: Optional[Sequence[str] | str] = None,
    target_qubits: Optional[Sequence[int]] = None,
) -> QMLResult
```

委托 [`run_pqc_classifier`](./qml.md#1-监督分类--run_pqc_classifier)；参数语义见 qml.md。

### `run_unsupervised(...) -> QBMResult`

```python
run_unsupervised(
    name: str,
    num_qubits: int,
    train_samples: np.ndarray,
    *,
    test_samples: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    provider: str = "quafu",
    prefer_chips: Optional[Sequence[str] | str] = None,
    target_qubits: Optional[Sequence[int]] = None,
) -> QBMResult
```

委托 [`run_qnn_unsupervised`](./qml.md#2-无监督分布学习--run_qnn_unsupervised)。`mmd_sigma` / `gen_shots` 取自 runner 字段。

### `run_conditional(...) -> QBMResult`

```python
run_conditional(
    name: str,
    num_qubits: int,
    train_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    *,
    test_pairs: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    provider: str = "quafu",
    prefer_chips: Optional[Sequence[str] | str] = None,
    target_qubits: Optional[Sequence[int]] = None,
) -> QBMResult
```

委托 [`run_qnn_conditional`](./qml.md#3-条件-qnn--run_qnn_conditional)：学习条件分布 P(y|x)，输入 bit-string *x* 以计算基态 `|x⟩` 制备。

## provider 支持

`quafu / tianyan / guodun / tencent / origin / fieldquantum / simulator`（大小写不敏感）。若 `prefer_chips` 含已知芯片名，会由 `resolve_provider` 反查覆盖 `provider`。

## 示例

```python
from fieldqkit import QuantumHardwareClient
from fieldqkit.algorithms.qml_runner import QMLRunner

client = QuantumHardwareClient()
runner = QMLRunner(client=client, layers=2, max_iters=100, gradient_method="autograd")

result = runner.run_classifier(
    name="iris_clf",
    num_qubits=4,
    train_data=train,
    test_data=test,
    num_classes=3,
    provider="simulator",          # autograd → 必须本地 Simulator
    prefer_chips="Simulator",
)
print(result.accuracy, result.test_accuracy)
```

## 相关页面

- [QML 底层函数](./qml.md)
- [VQERunner.run_model](./vqe_runner.md)
- [QAOARunner.run_model](./qaoa_runner.md)
- [result types — QMLResult / QBMResult](../core/result_types.md)
