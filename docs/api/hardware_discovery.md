# 硬件发现与选择

## 概览

- 模块：`quantum_hw.api.backend`
- 作用：在指定数量的可用比特数下，发现候选硬件、选择最优后端、获取拓扑与校准信息。
- 核心接口：`BackendAdapter.discover_hardware(...)` 与 `BackendAdapter.resolve_backend(...)`

## 核心接口

```python
BackendAdapter.discover_hardware(
    *,
    num_qubits: int,
    prefer_hardware: Sequence[str] | str | None = None,
) -> List[HardwareProfile]

BackendAdapter.resolve_backend(
    *,
    num_qubits: int,
    prefer_hardware: Sequence[str] | str | None = None,
) -> ResolvedBackend
```

## 参数详解

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|---|---|---:|:---:|---|
| `num_qubits` | `int` | - | 是 | 本次任务需要的最小可用物理比特数。 |
| `prefer_hardware` | `Optional[Sequence[str] \| str]` | `None` | 否 | 候选芯片偏好：可为单个名字、列表；传 `"Simulator"` 强制使用本地模拟器。 |

## 返回值详解

### `discover_hardware(...) -> List[HardwareProfile]`

返回符合条件的候选 `HardwareProfile` 列表（已按 `prefer_hardware` 排序），各字段含义：

```python
@dataclass
class HardwareProfile:
    provider: str                              # 平台名：quafu/tianyan/guodun/tencent/origin/fieldquantum/simulator
    hardware_name: str                         # 芯片名如 "Baihua"、"Simulator"、"fieldquantum_sim"
    nqubits_available: int                     # 该芯片的可用物理比特总数
    two_qubit_gate_basis: str                  # 两比特门基：通常为 "cz"
    topology: HardwareTopology                 # 量子芯片拓扑结构
    calibration: HardwareCalibration           # 校准数据（保真度、队列长度等）
    raw_info: Dict[str, Any]                   # 原始芯片配置（高级用途）
```

**HardwareTopology 内部结构：**
```python
@dataclass
class HardwareTopology:
    qubits: List[int]                          # 可用物理比特编号列表
    couplers: List[Tuple[int, int]]            # 两比特耦合器列表（物理比特对）
```

**HardwareCalibration 内部结构：**
```python
@dataclass
class HardwareCalibration:
    qubit_fidelity: Dict[int, float]           # 各物理比特的单比特门保真度
    coupler_fidelity: Dict[str, float]           # 各两比特门对的保真度（键为 couplers_info 原始键）
    queue_length: Optional[int]                # 该芯片的任务队列长度（可选）
```

### `resolve_backend(...) -> ResolvedBackend`

返回最终选定的后端对象：

```python
@dataclass
class ResolvedBackend:
    provider: str                              # 平台名
    hardware_name: str                         # 选中的芯片名
    backend: Backend                           # 完整后端对象（含拓扑图、编译规则等）
    profile: Optional[HardwareProfile]         # 硬件配置文件（可能为 None）
    metadata: Dict[str, Any]                   # 额外元数据（如平台对象引用）
```

## 选择与过滤逻辑

1. **Simulator 快速路径**：若 `prefer_hardware` 包含 `"Simulator"`（大小写不敏感），直接返回本地模拟器 profile。

2. **候选生成**：
   - 若用户指定 `prefer_hardware`，按用户顺序作为候选列表。
   - 否则使用平台的 `list_available_hardware()` 返回的芯片列表。
   - 若候选名单为空，尝试 fallback 默认芯片（如 Quafu 的 "Baihua"）。

3. **过滤条件**：只保留 `nqubits_available >= num_qubits` 的候选。

4. **最终选择**：`resolve_backend()` 默认返回过滤后的首个候选。

## Backend 对象与拓扑图操作

一旦获得 `ResolvedBackend`，可通过其 `backend` 字段访问完整的硬件信息与拓扑图操作：

```python
resolved = adapter.resolve_backend(num_qubits=8, prefer_hardware="Baihua")
backend_obj = resolved.backend

# 获取拓扑图（NetworkX 图对象）
graph = backend_obj.get_graph()

# 按保真度阈值过滤边（例如只保留保真度 >= 0.6 的耦合器）
filtered_graph = backend_obj.edge_filtered_graph(thres=0.6)

# 绘制并保存拓扑图
backend_obj.draw(save_svg_fname="chip_topology", edge_fidelity_thres=0.9)

# 缓存拓扑图到本地
backend_obj.cache_topology_figure(edge_fidelity_thres=0.9)
```

## 使用示例

### 示例1：发现所有可用芯片

```python
from quantum_hw.api.quantum_platform import QuafuBackendAdapter

adapter = QuafuBackendAdapter()
profiles = adapter.discover_hardware(num_qubits=8)

for p in profiles:
    print(f"{p.hardware_name}: {p.nqubits_available} qubits")
```

### 示例2：指定偏好芯片

```python
# 按优先级尝试 Baihua -> Dongling -> Simulator
profiles = adapter.discover_hardware(
    num_qubits=8,
    prefer_hardware=["Baihua", "Dongling", "Simulator"]
)
```

### 示例3：解析并获取拓扑

```python
resolved = adapter.resolve_backend(num_qubits=8, prefer_hardware="Simulator")
print(f"Selected chip: {resolved.hardware_name}")
print(f"Provider: {resolved.provider}")

# 查看拓扑信息
topo = resolved.profile.topology
print(f"Qubits: {topo.qubits}")
print(f"Couplers: {topo.couplers}")
```

### 示例4：强制选择模拟器

```python
# 传入字符串 "Simulator"，适合测试与调试
resolved = adapter.resolve_backend(num_qubits=4, prefer_hardware="Simulator")
assert resolved.hardware_name == "Simulator"
```

## 相关页面

- [Backend](./Backend.md)
- [QuantumHardwareClient](./QuantumHardwareClient.md)
- [providers](./providers.md)
