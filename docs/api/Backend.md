# Backend

## 概览

- 模块：`quantum_hw.api.backend`
- 作用：提供统一硬件拓扑抽象、硬件 profile 标准化、provider 侧后端发现/解析。
- 主要对象：`Backend`、`BackendAdapter`、`HardwareProfile`、`ResolvedBackend`。

## 核心类详解

### `Backend` 类

**签名：**
```python
def __init__(self, chip: str | dict)
```

**用途：** 将芯片配置映射为图结构，支持拓扑图构建、过滤与可视化。

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `chip` | `str \| dict` | **字符串**：支持以下芯片名——Quafu 侧：`Baihua`、`Dongling`、`Haituo`、`Yunmeng`、`Miaofeng`、`Yudu`、`Hongluo`；cqlib 侧：`tianyan176`、`tianyan176-2`、`tianyan24`、`tianyan504`、`tianyan287`、`gd_qc1`、`chmy176`、`gd_sim1`；模拟器：`Simulator`。**字典**：直接传入标准化 `chip_info` 配置（需含 `qubits_info`、`couplers_info`、`global_info` 等字段）。 |

**返回值：** `Backend` 对象，包含拓扑图和校准信息。

**异常：**
- `ValueError`：芯片名不在支持列表中。

**示例：**
```python
from quantum_hw.api.backend import Backend

# 按名字创建后端
backend_str = Backend("Baihua")

# 按配置字典创建
backend_dict = Backend({
    "hardware_name": "Baihua",
    "nqubits": 12,
    "two_qubit_gate_basis": "cz",
    ...
})
```

---

#### 关键属性

- `chip_name: str` —— 芯片名
- `chip_info: dict` —— 原始芯片配置
- `priority_qubits: List[List[int]]` —— 推荐比特优先级
- `qubits_with_attributes: List[Tuple[int, dict]]` —— 各比特及其属性（保真度等）
- `couplers_with_attributes: List[Tuple[int, int, dict]]` —— 各耦合器及其属性
- `two_qubit_gate_basis: str` —— 两比特门基（通常为 `"cz"`）
- `graph`（property）：`networkx.Graph` —— 完整拓扑图

---

#### 关键方法详解

### `Backend.get_graph()` 

**签名：**
```python
def get_graph(self) -> networkx.Graph
```

**用途：** 获取芯片拓扑的完整 NetworkX 图对象。

**返回值：** `networkx.Graph` 对象，其中：
- 节点：物理比特编号（`int`）
- 边：耦合器对 `(q_i, q_j)`，边属性包含保真度等

**示例：**
```python
backend = Backend("Baihua")
G = backend.get_graph()
print(f"比特数: {G.number_of_nodes()}")
print(f"耦合器数: {G.number_of_edges()}")
print(f"边数据: {list(G.edges(data=True))[:3]}")
```

---

### `Backend.edge_filtered_graph(thres: float = 0.6)`

**签名：**
```python
def edge_filtered_graph(self, thres: float = 0.6) -> networkx.Graph
```

**用途：** 返回只包含保真度 ≥ 阈值的边**和节点**的子图。节点和边均按 `fidelity` 属性过滤。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `thres` | `float` | `0.6` | 保真度阈值，范围 [0, 1]；`0.6` = 60% 保真度。 |

**返回值：** 过滤后的 `networkx.Graph`（可能不连通）。

**使用场景：** 编译优化、拓扑感知路由时筛选高保真度的耦合器。

**示例：**
```python
backend = Backend("Baihua")

# 仅保留保真度 >= 0.9 的耦合器
G_high_fidelity = backend.edge_filtered_graph(thres=0.9)
print(f"高保真耦合器数: {G_high_fidelity.number_of_edges()}")

# 获取低保真耦合器集合
G_all = backend.get_graph()
low_fidelity_edges = set(G_all.edges()) - set(G_high_fidelity.edges())
```

---

### `Backend.draw(save_svg_fname: str | None = None, edge_fidelity_thres: float = 0.9)`

**签名：**
```python
def draw(
    self,
    save_svg_fname: Optional[str] = None,
    edge_fidelity_thres: float = 0.9
) -> None
```

**用途：** 绘制芯片拓扑图，可选保存为 SVG 文件。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `save_svg_fname` | `Optional[str]` | `None` | SVG 文件保存路径（**不含** `.svg` 扩展名）；`None` 则仅在 Jupyter 中显示。 |
| `edge_fidelity_thres` | `float` | `0.9` | 拓扑绘制中保真度阈值；保真度低于此阈值的耦合器不绘制。 |

**返回值：** `None`

**异常：** 
- `ValueError`：`save_svg_fname` 路径创建失败。

**示例：**
```python
backend = Backend("Baihua")

# 在 Jupyter 中显示拓扑
backend.draw()

# 保存为 SVG 文件
backend.draw(save_svg_fname="baihua_topo", edge_fidelity_thres=0.85)
# 生成: baihua_topo.svg

# 仅绘制高保真度耦合器
backend.draw(save_svg_fname="baihua_hifi", edge_fidelity_thres=0.95)
```

---

### `Backend.cache_topology_figure(edge_fidelity_thres: float = 0.9)`

**签名：**
```python
def cache_topology_figure(self, edge_fidelity_thres: float = 0.9) -> None
```

**用途：** 将拓扑图缓存到本地磁盘（`.cache/` 目录），用于加速后续加载。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `edge_fidelity_thres` | `float` | `0.9` | 缓存图的保真度阈值。 |

**返回值：** `None`

**缓存位置：** `src/quantum_hw/api/.cache/{chip_name}_chip.svg`

**使用场景：** 在硬件解析 (`resolve_backend`) 时自动调用，加快后续相同芯片的加载。

**示例：**
```python
backend = Backend("Baihua")
backend.cache_topology_figure()
# 生成: src/quantum_hw/api/.cache/Baihua_chip.svg
```

---

### `HardwareTopology`

```python
@dataclass(frozen=True)
class HardwareTopology:
    qubits: List[int]
    couplers: List[Tuple[int, int]]
```

### `HardwareCalibration`

```python
@dataclass(frozen=True)
class HardwareCalibration:
    qubit_fidelity: Dict[int, float]
    coupler_fidelity: Dict[str, float]
    queue_length: Optional[int] = None
```

### `HardwareProfile`

```python
@dataclass(frozen=True)
class HardwareProfile:
    provider: str
    hardware_name: str
    nqubits_available: int
    two_qubit_gate_basis: str
    topology: HardwareTopology
    calibration: HardwareCalibration
    raw_info: Dict[str, Any] = field(default_factory=dict)
```

### `ResolvedBackend`

```python
@dataclass
class ResolvedBackend:
    provider: str
    hardware_name: str
    backend: Backend
    profile: Optional[HardwareProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `BackendAdapter`

```python
class BackendAdapter(ABC):
    provider: str
    default_hardware_name: Optional[str] = None
```

#### 关键方法

| 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `list_available_hardware` | `list_available_hardware()` | `List[Dict[str, Any]]` | 从绑定平台获取统一硬件列表。 |
| `discover_hardware` | `discover_hardware(*, num_qubits, prefer_hardware=None)` | `List[HardwareProfile]` | 候选发现与过滤。 |
| `resolve_backend` | `resolve_backend(*, num_qubits, prefer_hardware=None)` | `ResolvedBackend` | 选择最终后端。 |

## 关键函数

### `normalize_hardware_preferences(prefer_hardware) -> List[str]`

- 作用：将字符串/列表输入归一化为非空硬件名列表。

### `is_simulator_preferred(prefer_hardware) -> bool`

- 作用：判断偏好中是否显式要求 `Simulator`。

### `build_simulator_profile(provider, num_qubits) -> HardwareProfile`

- 作用：构造本地模拟器 profile。

### `build_hardware_profile(*, provider, hardware_name, backend, queue_length, raw_info) -> HardwareProfile`

- 作用：从 `Backend.chip_info` 生成统一 profile 数据结构。
- 注意：所有参数均为 keyword-only。
- 内部使用 `MIN_CONNECTED_COUPLER_FIDELITY = 0.9` 常量过滤低保真耦合器。

### `list_available_hardware(provider) -> List[Dict[str, Any]]`（模块级函数）

- 作用：按 provider 创建平台对象并返回统一硬件列表。
- 支持：`quafu/tianyan/guodun`。
- 注意：这是 `quantum_hw.api.backend` 模块级函数，与 `BackendAdapter.list_available_hardware()` 实例方法不同。

## 常见报错

- `ValueError("Wrong chip name! ...")`
- `ValueError("provider must be one of: 'quafu', 'tianyan', or 'guodun'")`
- `RuntimeError("no available chips satisfy num_qubits requirement")`

## 示例

```python
from quantum_hw.api.backend import Backend, list_available_hardware

rows = list_available_hardware("quafu")
print(rows[:2])

sim = Backend("Simulator")
g = sim.get_graph()
print(g.number_of_nodes(), g.number_of_edges())
```

## 相关页面

- [hardware_discovery](./hardware_discovery.md)
- [Task](./Task.md)
- [provider_runtime](./provider_runtime.md)
