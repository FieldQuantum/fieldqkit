# Task

## 概览

- 模块：`quantum_hw.api.task`
- 作用：定义 provider 无关的任务请求/句柄/适配器协议，屏蔽各平台任务接口差异。
- 核心对象：`OpenQasmSubmitRequest`、`ProviderTaskHandle`、`TaskAdapter`。

## 数据结构

### `OpenQasmSubmitRequest`

```python
@dataclass
class OpenQasmSubmitRequest:
    name: str
    qasm: str
    shots: int
    chip_name: str
    submit_options: Dict[str, Any] = field(default_factory=dict)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 任务名。 |
| `qasm` | `str` | OpenQASM 源码（2.0 或 3.0）。 |
| `shots` | `int` | 采样次数。 |
| `chip_name` | `str` | 目标芯片名。 |
| `submit_options` | `Dict[str, Any]` | provider 扩展参数（如轮询超时、sleep 间隔等）。 |

### `ProviderTaskHandle`

```python
@dataclass
class ProviderTaskHandle:
    provider: str
    task_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `provider` | `str` | provider 名称。 |
| `task_id` | `str` | 任务标识。 |
| `payload` | `Dict[str, Any]` | provider 侧上下文（平台对象、批量子任务 ID 等）。 |

## 适配器协议详解

### `TaskAdapter.submit_openqasm(...)`

**签名：**
```python
def submit_openqasm(
    self,
    submit_request: OpenQasmSubmitRequest,
    backend: ResolvedBackend
) -> ProviderTaskHandle
```

**用途：** 提交 OpenQASM 任务到目标平台。

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `submit_request` | `OpenQasmSubmitRequest` | 任务请求对象（包含线路、shots、芯片名等）。 |
| `backend` | `ResolvedBackend` | 已解析的后端对象（用于获取平台特定配置）。 |

**返回值：** `ProviderTaskHandle` —— 包含任务标识和平台特定的上下文信息。

**异常：**
- `RuntimeError`：任务提交失败（网络、认证、芯片离线等）。
- `ValueError`：`submit_request` 中的芯片名或参数非法。

---

### `TaskAdapter.query_status(handle)`

**签名：**
```python
def query_status(self, handle: ProviderTaskHandle) -> str
```

**用途：** 查询任务的实时状态。

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `handle` | `ProviderTaskHandle` | 由 `submit_openqasm` 返回的任务句柄。 |

**返回值：** 状态字符串，常见值：
- `"Queued"`：排队中
- `"Running"`：执行中
- `"Finished"`：完成
- `"Failed"`：失败
- `"Canceled"`：已取消
- `"Unknown"`：未知（不推荐）

**异常：**
- `RuntimeError`：状态查询失败或任务已清理。

---

### `TaskAdapter.fetch_result(handle)`

**签名：**
```python
def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]
```

**用途：** 获取任务的标准化结果。

**返回值：** 结果字典，至少包含以下字段：

```python
{
    "count": Dict[str, int],      # 测量计数 {"bit_string": count, ...}
    ...                             # 平台特定字段（如原始测量数据）
}
```

其中 `count` 的 bit_string 格式为不带空格的二进制字符串（例如 `"0011"`）。

**异常：**
- `RuntimeError`：任务未完成或结果已过期（通常 24-48 小时）。
- `ValueError`：结果数据格式非标准。

---

### `TaskAdapter.cancel_task(handle)`

**签名：**
```python
def cancel_task(self, handle: ProviderTaskHandle) -> None
```

**用途：** 取消未完成的任务。

> **注意：** Quafu 的 `cancel_task` 实现为 `raise NotImplementedError`，暂不可用；天衍/国盾通过 `stop_running_experiments()` 支持取消；腾讯仅输出警告日志，不实际取消。

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `handle` | `ProviderTaskHandle` | 由 `submit_openqasm` 返回的任务句柄。 |

**返回值：** `None`

---

## 实现方

- Quafu：`quantum_hw.api.quantum_platform.quafu.QuafuTaskAdapter`
- TianYan：`quantum_hw.api.quantum_platform.tianyan.TianYanTaskAdapter`
- GuoDun：`quantum_hw.api.quantum_platform.guodun.GuoDunTaskAdapter`
- Tencent：`quantum_hw.api.quantum_platform.tencent.TencentTaskAdapter`

## 与 QuantumHardwareClient 的配合

- `QuantumHardwareClient._submit_openqasm_async(...)` 负责创建 `OpenQasmSubmitRequest` 并调用 `TaskAdapter.submit_openqasm(...)`。
- `_wait_task(...)` 轮询 `query_status(...)`。
- `_get_task_result(...)` 通过 `fetch_result(...)` 取结果。

## 示例

```python
from quantum_hw.api.task import OpenQasmSubmitRequest
from quantum_hw.api.quantum_platform.quafu import QuafuTaskAdapter, QuafuBackendAdapter

# 使用 BackendAdapter 解析后端
backend_adapter = QuafuBackendAdapter()
resolved = backend_adapter.resolve_backend(num_qubits=4, prefer_hardware="Simulator")

# 使用 TaskAdapter 提交任务
task_adapter = QuafuTaskAdapter()

qasm = """
OPENQASM 2.0;
qreg q[4];
creg c[4];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
measure q -> c;
"""

req = OpenQasmSubmitRequest(
    name="ghz_4",
    qasm=qasm,
    shots=1024,
    chip_name="Simulator",
    submit_options={}
)

handle = task_adapter.submit_openqasm(req, resolved)
print(f"Task ID: {handle.task_id}, Provider: {handle.provider}")

# 查询状态
status = task_adapter.query_status(handle)
print(f"Status: {status}")

# 获取结果
result = task_adapter.fetch_result(handle)
print(f"Counts: {result['count']}")
```

## 相关页面

- [QuantumHardwareClient](./QuantumHardwareClient.md)
- [run_with_backend](./run_with_backend.md)
- [provider_runtime](./provider_runtime.md)
