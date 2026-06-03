# build_confusion_matrix

## 概览

- **所属模块**：`fieldqkit.calibration.readout`
- **用途**：将校准任务得到的计数字典列表转换为 confusion matrix（每**列**对应一个输入基态的输出概率分布）。
- **约定**：`mat[i, j] = P(测得 i | 制备 j)`，即第 `j` 列是“制备输入态 `j` 后测得各输出态的概率分布”，每列求和为 1。

## 签名

```python
build_confusion_matrix(
    res_list: Sequence[Dict[str, int]],
    num_qubits: int,
) -> np.ndarray
```

## 参数

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `res_list` | `Sequence[Dict[str, int]]` | - | 是 | 每个输入态对应一次测量计数结果。 |
| `num_qubits` | `int` | - | 是 | 参与构造 confusion matrix 的比特数。 |

## 返回值

- 返回类型：`np.ndarray`
- 形状：`(2**num_qubits, 2**num_qubits)`
- 语义：`mat[i, j] = P(测得 i | 制备 j)`，第 `j` **列**是“准备输入态 `j` 后，测得各输出态的概率分布”，每列求和为 1。

## 异常与报错

- 函数本身不主动抛业务异常。
- 若 `res_list` 与 `num_qubits` 不匹配（例如计数键位宽不一致），会在内部概率换算阶段出现错误或得到不符合预期的矩阵。

## 示例

```python
from fieldqkit.calibration.readout import build_confusion_matrix

res_list = [
    {"0": 980, "1": 44},   # prepare |0>
    {"0": 37,  "1": 987},  # prepare |1>
]

cm = build_confusion_matrix(res_list, num_qubits=1)
print(cm.shape)  # (2, 2)
print(cm)
```

## 行为细节 / 注意事项

- 内部按 `get_probabilities(...)` 的位序约定填充矩阵列（`mat[:, j] = probs_j`）。
- 该函数是 `ReadoutCalibrationManager` 构建单比特 confusion matrix 的基础组件。
- 对多比特输入同样适用，前提是 `res_list` 的顺序与你的输入态编码一致。

## 相关页面

- [ReadoutCalibrationManager](./ReadoutCalibrationManager.md)
- [readout](../core/readout.md)
