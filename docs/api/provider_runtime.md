# provider_runtime

## 概览

- 模块：`quantum_hw.api.quantum_platform`
- 作用：按 provider 名称创建统一运行时对象，绑定 `BackendAdapter` 与 `TaskAdapter`。
- 入口函数：`create_provider_runtime(...)`

## 核心数据结构

### `ProviderRuntime`

```python
@dataclass
class ProviderRuntime:
    provider: str
    backend_adapter: Any
    task_adapter: Any
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `provider` | `str` | 标准化 provider 名（小写）。 |
| `backend_adapter` | `Any` | 对应 provider 的后端发现/解析适配器。 |
| `task_adapter` | `Any` | 对应 provider 的任务提交与结果适配器。 |

## 工厂函数

### `create_provider_runtime(*, provider: str, client: Any) -> ProviderRuntime`

| 参数 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `provider` | `str` | 是 | 支持 `quafu/tianyan/guodun/tencent`。 |
| `client` | `Any` | 是 | 当前 `QuantumHardwareClient` 实例（供 task adapter 绑定上下文）。 |

### 返回值

按 provider 返回：

- `quafu`：`QuafuBackendAdapter` + `QuafuTaskAdapter`
- `tianyan`：`TianYanBackendAdapter` + `TianYanTaskAdapter`
- `guodun`：`GuoDunBackendAdapter` + `GuoDunTaskAdapter`
- `tencent`：`TencentBackendAdapter` + `TencentTaskAdapter`

### 异常

- `ValueError("provider must be one of: 'quafu', 'tianyan', 'guodun', or 'tencent'")`

## 示例

```python
from quantum_hw.api.quantum_platform import create_provider_runtime
from quantum_hw.api.client import QuantumHardwareClient

client = QuantumHardwareClient()
runtime = create_provider_runtime(provider="quafu", client=client)

print(runtime.provider)
print(type(runtime.backend_adapter).__name__)
print(type(runtime.task_adapter).__name__)
```

## 相关页面

- [QuantumHardwareClient](./QuantumHardwareClient.md)
- [Backend](./Backend.md)
- [Task](./Task.md)
- [providers](./providers.md)
