# Helpers 与渲染

## 概览

- **模块**：`quantum_hw.circuit.quantumcircuit_helpers`、`quantum_hw.circuit.render`
- **源文件**：`quantumcircuit_helpers.py`（约620 行）、`render.py`（约50 行）
- **作用**：
  - 管理门元信息（门集合常量、显示符号映射）
  - 参数格式化与表达式安全求值
  - gate tuple → DAG 节点/边转换
  - gate tuple → ASCII 文本线路图的完整渲染管线

---

## quantumcircuit_helpers 模块

### 门集合常量

下列字典用于门名校验、类型识别和渲染符号映射。键为门名（str），值为 ASCII 绘图符号。

| 常量 | 说明 | 示例键值 |
|------|------|----------|
| `one_qubit_gates_available` | 单比特离散门 | `'x': 'X'`, `'h': 'H'`, `'sx': '√X'` |
| `two_qubit_gates_available` | 双比特离散门 | `'cx': '●X'`, `'swap': 'XX'`, `'ecr': '╬╬'` |
| `three_qubit_gates_available` | 三比特门 | `'ccz': '●●●'`, `'ccx': '●●X'` |
| `one_qubit_parameter_gates_available` | 单比特参数门 | `'rx': 'Rx'`, `'ry': 'Ry'`, `'rz': 'Rz'`, `'u': 'U'` |
| `two_qubit_parameter_gates_available` | 双比特参数门 | `'rxx': 'Rxx'`, `'ryy': 'Ryy'`, `'rzz': 'Rzz'` |
| `functional_gates_available` | 功能门 | `'barrier': '░'`, `'measure': 'M'`, `'reset': '\|0>'`, `'delay': 'Delay'` |

### DAG 转换

#### `convert_gate_info_to_dag_info(nqubits: int, qubits: list, gates: list, show_qubits: bool = True) -> tuple[list, list]`

将 gate tuple 列表转换为有向无环图（DAG）的节点与边列表，供编译模块的 DAG 分层使用。

| 参数 | 类型 | 说明 |
|------|------|------|
| `nqubits` | `int` | 量子比特总数 |
| `qubits` | `list` | 活跃 qubit 索引 |
| `gates` | `list` | gate tuple 列表 |
| `show_qubits` | `bool` | 是否为每个 qubit 生成初始节点，默认 `True` |

**返回**：`(node_list, edge_list)`，均为 Python `list`。

- 节点格式：`(gate_idx_qubits, {'qubits': [...], 'params': [...], ...})`
- 边格式：`(src_node, dst_node, {'qubit': [...]})`
- `measure` 指令会被拆解为逐 qubit 的单独节点
- 支持所有门类别（一/二/三比特门、参数门、功能门）

### 参数格式化

#### `is_multiple_of_pi(n, tolerance: float = 1e-9) -> str`

判断数值是否近似为 $\pi$ 的倍数，返回人类可读字符串。

| 输入 | 输出示例 |
|------|----------|
| `3.14159…` | `"1.0π"` |
| `1.5707…` | `"0.5π"` |
| `0.0` | `"0.0"` |
| `0.123` | `"0.123"` |

#### `_format_param_token(token, params_value: dict) -> str`

将参数值标准化为绘图标签。

- 数值型（`float`/`int`/`np.floating`/`np.integer`）→ 调用 `is_multiple_of_pi`
- 字符串占位符 → 先在 `params_value` 中查找绑定值，已绑定则转数值显示，未绑定保持原名

### 表达式求值

#### `_safe_eval_expression(expr: str) -> float`

基于 `ast` 模块的安全表达式求值器。仅允许：

- 数值常量（`int`/`float`）
- 名称 `pi`
- 一元运算（`+`/`-`）
- 二元运算（`+`/`-`/`*`/`/`/`**`）

非法表达式抛出 `ValueError`。

#### `parse_expression(expr: str) -> float`

解析参数表达式字符串（如 `"pi/4"`、`"2*pi"`、`"np.pi/3"`）为浮点数。

处理流程：
1. 去空白、`π` → `pi`、`np.pi` → `pi`
2. 尝试正则匹配 `[coef]pi[/denom]` 模式直接计算
3. 否则回退到 `_safe_eval_expression`

### 渲染管线

渲染管线将 gate tuple 列表转化为 ASCII 文本线路图，流程为：

```
gates → initialize_lines → generate_gates_layerd → format_gates_layerd → add_gates_to_lines → lines
```

#### `initialize_lines(nqubits: int, ncbits: int, gates: list) -> tuple[list, list]`

初始化空白线路画布。

| 参数 | 类型 | 说明 |
|------|------|------|
| `nqubits` | `int` | 量子比特数 |
| `ncbits` | `int` | 经典比特数 |
| `gates` | `list` | 门列表（仅用于确定层数上限） |

**返回**：`(gates_element, gates_layerd)`
- `gates_element`：单层占位符模板（`─`、` `、`═`）
- `gates_layerd`：包含初始标签层的二维列表

行布局：`q[0]` 占第 0 行,间隔行占第 1 行,`q[1]` 占第 2 行 …，经典线路占第 `2n` 行。

#### `generate_gates_layerd(nqubits: int, ncbits: int, gates: list, params_value: dict) -> tuple[list, list]`

将门按冲突检测分配到各绘图层（宽松布局）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `nqubits` | `int` | 量子比特数 |
| `ncbits` | `int` | 经典比特数 |
| `gates` | `list` | gate tuple 列表 |
| `params_value` | `dict` | 参数名到值映射 |

**返回**：`(gates_layerd, lines_use)`
- `gates_layerd`：各层的符号列表
- `lines_use`：实际使用的行号列表

内部逻辑：
- 对每个门，从最后一层向前扫描，找到该门占用 qubit 范围全空的层放置
- 多比特门之间的行用 `│` 连线
- `measure` 门会从量子线延伸到经典线位置

#### `format_gates_layerd(nqubits: int, ncbits: int, gates: list, params_value: dict) -> tuple[list, list]`

统一每层的字符串宽度。将短符号用 `─`（量子线）/ ` `（间隔行）/ `═`（经典线）填充至层内最大宽度。

**返回**：`(gates_layerd_format, lines_use)`

#### `add_gates_to_lines(nqubits: int, ncbits: int, gates: list, params_value: dict, width: int = 4) -> tuple[list, list]`

将格式化后的层拼接为最终的行字符串。

| 参数 | 类型 | 说明 |
|------|------|------|
| `width` | `int` | 门之间的间距字符数，默认 `4` |

**返回**：`(lines, lines_use)`
- `lines`：完整 ASCII 线路图的行列表
- `lines_use`：有门的行号列表

---

## render 模块

### `_render_lines(lines: list[str])`

底层渲染函数。若在 IPython/Jupyter 环境中使用 `<pre>` HTML 输出；否则 `print` 到终端。

### `draw_circuit(lines: list[str])`

输出完整线路图（所有 qubit 行）。直接调用 `_render_lines(lines)`。

### `draw_circuit_simply(lines: list[str], lines_use: list[int], nqubits: int)`

仅输出活跃 qubit 对应的行，提供更紧凑的视图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `lines` | `list[str]` | 完整线路图行列表 |
| `lines_use` | `list[int]` | 使用中的行号 |
| `nqubits` | `int` | 总量子比特数 |

---

## 参数显示约定

- 接近 $k\pi$ 的浮点显示为 `kπ`（如 `0.5π`）
- 接近 0 的值显示为 `0.0`
- 其余浮点保留小数点后三位
- 未绑定参数保持占位符名称（如 `theta`）

## 示例

```python
from quantum_hw.circuit import QuantumCircuit

qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.rzz("phi", 1, 2)
qc.apply_value({"phi": 0.5}, deep=True)

qc.draw(width=5)
qc.draw_simply(width=5)
```

## 相关页面

- [QuantumCircuit](./quantumcircuit.md)
- [OpenQASM 解析](./qasm.md)
- [matrix 与 utils](./matrix_utils.md)
