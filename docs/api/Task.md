# Task

## 概览

- 模块：`fieldqkit.api.task`
- 作用：定义 provider 无关的任务请求/句柄/适配器协议，屏蔽各平台任务接口差异。
- 核心对象：`OpenQasmSubmitRequest`、`QcisSubmitRequest`、`ProviderTaskHandle`、`TaskAdapter`。

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
| `qasm` | `str` | OpenQASM **2.0** 源码。 |
| `shots` | `int` | 采样次数。 |
| `chip_name` | `str` | 目标芯片名。 |
| `submit_options` | `Dict[str, Any]` | provider 扩展参数。详见下方"`submit_options` 约定键"。 |

### `QcisSubmitRequest`

```python
@dataclass
class QcisSubmitRequest:
    name: str
    qcis: str
    shots: int
    chip_name: str
    submit_options: Dict[str, Any] = field(default_factory=dict)
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 任务名。 |
| `qcis` | `str` | QCIS 原生指令字符串（多行，每行一条指令）。 |
| `shots` | `int` | 采样次数。 |
| `chip_name` | `str` | 目标芯片名。 |
| `submit_options` | `Dict[str, Any]` | provider 扩展参数（同 `OpenQasmSubmitRequest`）。 |

QCIS 文本由 `fieldqkit.circuit.qcis.circuit_to_qcis(QuantumCircuit)` 在客户端生成。

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
| `task_id` | `str` | 任务标识。**约定为字符串**——provider 返回整数 ID 时需在 adapter 内 `str(...)` 强转，保证跨 provider 类型一致。 |
| `payload` | `Dict[str, Any]` | provider 侧上下文（平台对象、批量子任务 ID、轮询缓存等）。 |

### `submit_options` 约定键

`submit_options` 是一个开放字典，但有几个键被 `QuantumHardwareClient` 与各 adapter 隐式约定：

| 键 | 类型 | 说明 |
|---|---|---|
| `max_wait_time` | `int` | 任务轮询最大等待时间（秒），透传到 `query_experiment(...)` 之类的接口。默认 3600。 |
| `sleep_time` | `int` | 轮询间隔（秒），默认 5。 |

新的 provider 实现可以读取这些键，但不要求必须实现。

## 适配器协议详解

### `TaskAdapter` 类属性

```python
class TaskAdapter(ABC):
    provider: str
    qcis_native: bool = False    # True 时客户端调用 submit_qcis 而非 submit_openqasm
```

- `qcis_native = True`：用于 TianYan / GuoDun。`QuantumHardwareClient._submit_circuit_async` 看到此开关后会先用 `circuit_to_qcis(...)` 把 `QuantumCircuit` 转 QCIS，再调 `submit_qcis(...)`。
- `qcis_native = False`（默认）：用于 Quafu / Tencent / Origin / FieldQuantum，走 `submit_openqasm(...)`。

### `TaskAdapter.submit_openqasm(...)`

**签名：**
```python
def submit_openqasm(
    self,
    submit_request: OpenQasmSubmitRequest,
    backend: ResolvedBackend,
) -> ProviderTaskHandle
```

**用途：** 提交 OpenQASM 2.0 任务到目标平台。

**参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `submit_request` | `OpenQasmSubmitRequest` | 任务请求对象。 |
| `backend` | `ResolvedBackend` | 已解析的后端对象（用于获取 `metadata.platform_obj` 等）。 |

**返回值：** `ProviderTaskHandle`。

**异常：**
- `RuntimeError`：任务提交失败（网络、认证、芯片离线等）。
- `ValueError`：`submit_request` 中的芯片名或参数非法。
- `NotImplementedError`：基类默认实现；当 provider 走 QCIS 通道时不必实现。

---

### `TaskAdapter.submit_qcis(...)`

**签名：**
```python
def submit_qcis(
    self,
    submit_request: QcisSubmitRequest,
    backend: ResolvedBackend,
) -> ProviderTaskHandle
```

**用途：** 提交 QCIS 任务到目标平台（TianYan / GuoDun 实现）。

**参数：** 同 `submit_openqasm`，仅请求体类型为 `QcisSubmitRequest`。

**异常：** `NotImplementedError` 默认；OpenQASM 通道 provider 不必实现。

---

### `TaskAdapter.query_status(handle)`

**签名：**
```python
def query_status(self, handle: ProviderTaskHandle) -> str
```

**用途：** 查询任务实时状态。

**返回值：** 状态字符串，标准取值：

- `"Queued"`：排队中
- `"Running"`：执行中
- `"Finished"`：完成
- `"Failed"`：失败
- `"Canceled"`：已取消

各 provider 在 adapter 内把云端原始状态（如 `pending / scheduled / completed / FINISHED`）映射到上述统一集合。

**异常：** `RuntimeError`（状态查询失败、任务过期）。

---

### `TaskAdapter.fetch_result(handle)`

**签名：**
```python
def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]
```

**用途：** 获取任务的标准化结果。

**返回值：** 结果字典，至少包含：

```python
{
    "count": Dict[str, int],   # 测量计数 {"bit_string": count, ...}
    ...                          # 可附带平台特定字段
}
```

`count` 中 bitstring 为不带空格的二进制字符串（如 `"0011"`），**全 provider 统一为 little-endian**（q[0] 在最右/最低位）；大端的 provider 在 adapter 内自动翻转。

**异常：** `RuntimeError`（任务未完成或结果已过期，通常 24-48 小时）。

---

### `TaskAdapter.cancel_task(handle)`

**签名：**
```python
def cancel_task(self, handle: ProviderTaskHandle) -> None
```

**用途：** 取消未完成的任务。

> **支持情况：**
> - Quafu：调用 `platform_obj.cancel(tid)`。
> - TianYan / GuoDun：通过 `stop_running_experiments(query_id=...)`。
> - Tencent：仅输出 warning，云侧未提供取消接口。
> - Origin：pyqpanda3 SDK 暂未暴露取消接口，仅 warning。
> - FieldQuantum：服务端暂未暴露取消接口，仅 warning。

---

## 实现方

- Quafu：`fieldqkit.api.quantum_platform.quafu.QuafuTaskAdapter`（OpenQASM）
- TianYan：`fieldqkit.api.quantum_platform.tianyan.TianYanTaskAdapter`（QCIS）
- GuoDun：`fieldqkit.api.quantum_platform.guodun.GuoDunTaskAdapter`（QCIS）
- Tencent：`fieldqkit.api.quantum_platform.tencent.TencentTaskAdapter`（OpenQASM）
- Origin：`fieldqkit.api.quantum_platform.origin.OriginTaskAdapter`（OpenQASM）
- FieldQuantum：`fieldqkit.api.quantum_platform.fieldquantum.FieldQuantumTaskAdapter`（OpenQASM，sample 模式）

## 与 QuantumHardwareClient 的配合

- `QuantumHardwareClient._submit_circuit_async(...)` 检查 `adapter.qcis_native`：
  - `True` → 转 QCIS 后调用 `submit_qcis(...)`
  - `False` → 转 OpenQASM 2.0 后调用 `submit_openqasm(...)`
- `_wait_task(...)` 轮询 `query_status(...)`，直到返回 `Finished / Failed / Canceled`。
- `_get_task_result(...)` 通过 `fetch_result(...)` 取结果。

## 示例

```python
from fieldqkit.api.task import OpenQasmSubmitRequest
from fieldqkit.api.quantum_platform.quafu import QuafuTaskAdapter, QuafuBackendAdapter

# 使用 BackendAdapter 解析后端
backend_adapter = QuafuBackendAdapter()
resolved = backend_adapter.resolve_backend(num_qubits=4, prefer_hardware="Baihua")

# 使用 TaskAdapter 提交任务
task_adapter = QuafuTaskAdapter(client=None)   # 普通调用可传 None

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
    chip_name="Baihua",
    submit_options={"max_wait_time": 3600, "sleep_time": 5},
)

handle = task_adapter.submit_openqasm(req, resolved)
print(f"Task ID: {handle.task_id}, Provider: {handle.provider}")

status = task_adapter.query_status(handle)
print(f"Status: {status}")

result = task_adapter.fetch_result(handle)
print(f"Counts: {result['count']}")
```

## 相关页面

- [QuantumHardwareClient](./QuantumHardwareClient.md)
- [run_with_backend](./run_with_backend.md)
- [provider_runtime](./provider_runtime.md)
- [providers](./providers.md)
