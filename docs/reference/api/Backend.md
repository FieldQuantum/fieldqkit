# Backend

## 概览

- **模块**：`quantum_hw.api.backend`
- **作用**：将芯片信息组织为 `networkx.Graph`，并提供拓扑筛选、芯片信息查询与排序工具。
- **典型用途**：
  - 查看芯片拓扑
  - 给编译器提供后端连通图/门基信息
  - 调用 `rank_chips(...)` 做候选芯片排序
  - 在本地构造自定义后端用于测试

## 构造

```python
from quantum_hw.api.backend import Backend

backend = Backend("Baihua")
# 或 Backend("Simulator")
# 或 Backend(custom_chip_dict)
```

### 支持输入

- 真实芯片名：`Baihua / Dongling / Haituo / Yunmeng / Miaofeng / Yudu / Hongluo`
- `"Simulator"` / `"simulator"`：内置 12 比特模拟后端
- `"Custom"`：空壳后端（手动补信息）
- `dict`：自定义芯片结构（测试常用）

## 关键属性

- `chip_name: str`
- `chip_info: dict`
- `size: tuple`
- `priority_qubits: list`
- `qubits_with_attributes: list[tuple]`
- `couplers_with_attributes: list[tuple]`
- `two_qubit_gate_basis: str`
- `graph: networkx.Graph`（属性，等价于 `get_graph()`）

## 主要方法

### `get_graph() -> nx.Graph`

根据 `qubits_with_attributes` 和 `couplers_with_attributes` 构建无向图。

### `edge_filtered_graph(thres=0.6) -> nx.Graph`

按保真度阈值过滤子图：

- 仅保留 `edge.fidelity >= thres`
- 同时仅保留 `node.fidelity >= thres`

### `draw(...) -> None`

```python
draw(
    save_svg_fname=None,
    edge_fidelity_thres=0.9,
)
```

参数说明：

- `save_svg_fname`: 保存为 `*.svg`（文件名后缀由实现追加）
- `edge_fidelity_thres`: 绘图前的边保真度过滤阈值（默认 `0.9`）

该接口当前是固定风格绘图，不再暴露节点/边标签样式开关。

## 辅助函数

- `_build_simulator_chip_info(nqubits=12) -> dict`：构造模拟器后端信息。
- `load_chip_basic_info(chip_name) -> dict | None`：从远端服务拉取芯片配置。
- `get_available_chip_status(tmgr) -> Dict[str, int]`：读取任务队列状态。
- `get_chip_info(chip_name) -> Dict[str, Union[int, float]]`：获取芯片信息并尝试缓存拓扑图。
- `rank_chips(tmgr, *, num_qubits, prefer_chips=None, weights=None) -> List[str]`：芯片排序入口。

## 示例

```python
from quantum_hw.api.backend import Backend

bk = Backend("Simulator")
print(bk.two_qubit_gate_basis)  # cz
print(bk.graph.number_of_nodes())

bk.draw(
    save_svg_fname="./sim_topology",
    edge_fidelity_thres=0.9,
)
```
