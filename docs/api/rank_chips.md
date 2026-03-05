# rank_chips

## 概览

- **模块**：`quantum_hw.api.backend`
- **作用**：在满足比特数约束的芯片中，按队列长度、芯片规模与双比特误差进行加权排序。

## 签名

```python
rank_chips(
		tmgr,
		*,
		num_qubits: int,
		prefer_chips: Optional[Sequence[str] | str] = None,
		weights: Optional[Dict[str, float]] = None,
) -> List[str]
```

## 参数说明

- `tmgr`：任务管理器对象，要求提供 `status()` 方法并返回 `dict[chip_name, queue_len]`。
- `num_qubits: int`：所需最小可用比特数。
- `prefer_chips: Optional[Sequence[str] | str]`
	- 若给定，仅在指定集合内排序。
	- 特殊规则：若包含 `"simulator"`（大小写不敏感），当 `num_qubits <= 12` 直接返回 `['Simulator']`。
- `weights: Optional[Dict[str, float]]`
	- 可配置键：`queue`、`nqubits`、`error`。
	- 默认：`{"queue": 0.2, "nqubits": 0.3, "error": 0.5}`。

## 排序逻辑

候选芯片会先过滤：

1. 来自 `tmgr.status()` 的在线芯片。
2. `nqubits_available >= num_qubits`。
3. 若设置了 `prefer_chips`，进一步按集合过滤。

对每个候选芯片计算：

- `queue_len`：排队长度（越小越好）
- `nqubits`：可用比特数（越大越好）
- `error_rate_2q`：双比特误差率（越小越好）

每项做 min-max 归一化后，最终分数为：

$$
	\text{score} = w_q \cdot q_{norm} + w_n \cdot (1 - n_{norm}) + w_e \cdot e_{norm}
$$

按分数升序返回芯片名列表。

## 相关辅助函数

- `get_available_chip_status(tmgr) -> Dict[str, int]`
	- 拉取并校验队列状态字典。
- `get_chip_info(chip_name) -> Dict[str, Union[int, float]]`
	- 拉取芯片信息并尝试缓存拓扑图（失败时返回空字典）。

## 示例

```python
from quantum_hw.api.backend import rank_chips
from quantum_hw.api.task import Task

tmgr = Task()

chips = rank_chips(
		tmgr,
		num_qubits=10,
		prefer_chips=["Baihua", "Dongling"],
		weights={"queue": 0.3, "nqubits": 0.2, "error": 0.5},
)

print(chips)
```

## 注意事项

- 若没有候选芯片满足约束，会返回空列表 `[]`。
