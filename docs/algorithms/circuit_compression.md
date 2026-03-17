# circuit_compression

## 概览

- 模块：`quantum_hw.algorithms.circuit_compression`
- 目标：为 VQE parameter-shift 路径提供可复用的线路压缩工具。
- 核心思想：
	- 先把目标线路按层切成 prefix/suffix blocks（可选）。
	- 再用浅层硬件高效 ansatz 拟合每个 stage。
	- 拟合目标显式二选一：`mps`（状态保真）或 `mpo`（过程保真）。

## 数据结构

### `SuffixCompressionBlock`

```python
@dataclass
class SuffixCompressionBlock:
		start_layer: int
		end_layer: int
		max_bond: int
		relative_trunc_error: float
		num_gates: int
```

- 含义：描述一个后缀压缩块在层空间上的范围和复杂度估计。

### `HybridCompressionPlan`

```python
@dataclass
class HybridCompressionPlan:
		split_layer: int
		total_layers: int
		prefix_max_bond: int
		prefix_relative_trunc_error: float
		blocks: List[SuffixCompressionBlock]
```

- 含义：
	- `split_layer` 左侧是 prefix。
	- `split_layer` 到末层被连续划分为 `blocks`。

## 关键函数

### `build_layer_span_circuit`

```python
build_layer_span_circuit(
		qc_bound: QuantumCircuit,
		*,
		start_layer: int,
		end_layer: int,
) -> QuantumCircuit
```

- 作用：提取闭区间 `[start_layer, end_layer]` 对应的子线路。
- 行为：
	- 自动按门冲突构建 moments/layers。
	- 入参越界会被裁剪到合法范围。
	- 空区间返回空线路。
- 典型用法：VQE 在 stage 压缩前构造每个 stage 的 target 子线路。

### `plan_hybrid_suffix_blocks`

```python
plan_hybrid_suffix_blocks(
		qc_bound: QuantumCircuit,
		*,
		bond_cap: int = 128,
		trunc_tol: float = 1e-8,
		max_layers_per_block: int = 6,
		device: torch.device | str | None = None,
) -> HybridCompressionPlan
```

- 作用：依据 bond 与截断误差阈值，把线路切成 `prefix + suffix blocks`。
- 规划策略：
	- prefix：逐层扩展，使用 MPS 截断误差判断是否继续吸收。
	- suffix：在每个起点上尝试扩展到 `max_layers_per_block`，用 MPO 截断误差决定 block 终点。
- 参数约束：
	- `bond_cap > 0`
	- `trunc_tol >= 0`
	- `max_layers_per_block > 0`

### `compress_circuit_with_hybrid_objective`

```python
compress_circuit_with_hybrid_objective(
		qc_bound: QuantumCircuit,
		*,
		num_qubits: int,
		approx_layers: int,
		optimizer_steps: int,
		optimizer_lr: float,
		objective_mode: Literal["mps", "mpo"] = "mps",
		bond_cap: int,
		warm_start_params: Optional[np.ndarray],
		device: torch.device | str | None = None,
) -> Tuple[QuantumCircuit, np.ndarray, Dict[str, object]]
```

- 作用：把 `qc_bound` 拟合为浅层硬件高效线路。
- 目标模式：
	- `objective_mode="mps"`：仅计算状态 infidelity。
	- `objective_mode="mpo"`：仅计算过程 infidelity。
- 优化器：Adam，两阶段（主优化 + 保守 refine）。
- 返回：
	- `compressed_qc`：压缩后线路。
	- `best_params`：可用于下一次 warm start 的参数。
	- `summary`：包含 `objective_mode/objective_infidelity/init_loss/best_loss/loss_delta/loss_history`。

## 使用示例

### 1) 先规划后分段压缩

```python
from quantum_hw.algorithms.circuit_compression import (
		plan_hybrid_suffix_blocks,
		build_layer_span_circuit,
		compress_circuit_with_hybrid_objective,
)

plan = plan_hybrid_suffix_blocks(
		qc,
		bond_cap=128,
		trunc_tol=1e-8,
		max_layers_per_block=6,
)

prefix_qc = build_layer_span_circuit(
		qc,
		start_layer=0,
		end_layer=max(plan.split_layer - 1, -1),
)

cmp_prefix, warm, info_prefix = compress_circuit_with_hybrid_objective(
		prefix_qc,
		num_qubits=qc.nqubits,
		approx_layers=2,
		optimizer_steps=20,
		optimizer_lr=0.05,
		objective_mode="mps",
		bond_cap=128,
		warm_start_params=None,
)

for block in plan.blocks:
		block_qc = build_layer_span_circuit(
				qc,
				start_layer=block.start_layer,
				end_layer=block.end_layer,
		)
		cmp_block, warm, info_block = compress_circuit_with_hybrid_objective(
				block_qc,
				num_qubits=qc.nqubits,
				approx_layers=2,
				optimizer_steps=20,
				optimizer_lr=0.05,
				objective_mode="mpo",
				bond_cap=128,
				warm_start_params=warm,
		)
```

### 2) 直接压缩整条线路

```python
compressed_qc, warm_start, summary = compress_circuit_with_hybrid_objective(
		qc,
		num_qubits=qc.nqubits,
		approx_layers=2,
		optimizer_steps=30,
		optimizer_lr=0.03,
		objective_mode="mps",
		bond_cap=64,
		warm_start_params=None,
)
```

## 注意事项

- 该模块假设输入线路可由 `simulate_mps` / `simulate_mpo_process` 支持。
- `objective_mode` 为显式单目标，不再支持 mps/mpo 权重混合。
- `warm_start_params` 长度不匹配时会回退到随机初始化。

## 相关页面

- [VQERunner.run_model](./vqe_runner.md)
- [ansatz templates](./ansatz_templates.md)
- [mps simulator](../sim/mps.md)
- [mpo process simulator](../sim/mpo.md)
