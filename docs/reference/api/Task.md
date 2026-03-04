# Task

## 概览

- **模块**：`quantum_hw.api.task`
- **作用**：Quafu REST API 的轻量客户端，负责任务提交、查询、取消、结果拉取。
- **模式**：单例（`Task()` 重复创建返回同一实例）。

## 构造与认证

```python
from quantum_hw.api.task import Task

tmgr = Task()
```

认证逻辑：

- 优先读取环境变量 `QPU_API_TOKEN`
- 若未设置，使用模块内默认 token 字符串
- 初始化时调用 `verify()` 校验 token

> 建议在生产环境显式设置环境变量，避免依赖默认 token。

## 主要方法

### `verify()`

校验当前 token 是否可用。

### `status(tid: int = 0)`

- `tid=0` 时返回整体状态（常用于芯片队列）
- `tid>0` 时查询单任务状态

### `run(task: dict, repeat: int = 1)`

提交任务，返回任务 ID（通常为 `int`）。

`task` 常用字段：

- `chip`（必填）
- `circuit`（必填，OpenQASM 字符串）
- `name`（可选）
- `shots`（可选）
- `compile`（可选，默认 `True`）
- `options`（可选）

### `result(tid: int, timeout: float = 0.0)`

- `timeout=0`：单次拉取
- `timeout>0`：轮询直到拿到非空结果或超时（抛 `TimeoutError`）

### `query(...)`

按任务 ID、芯片、状态、时间窗口等条件分页查询历史任务。

### `cancel(tid: int)` / `delete(tid: int)`

取消任务或删除任务记录。

## 请求层

底层统一走：

```python
request(url: str, data: dict = {}, method: str = "get")
```

- GET：`session.get`
- POST：`session.post(data=json.dumps(data))`
- Header 自动注入 `token`

## 示例

```python
from quantum_hw.api.task import Task

tmgr = Task()
print(tmgr.verify())

task_id = tmgr.run(
    {
        "chip": "Simulator",
        "name": "demo",
        "circuit": "OPENQASM 2.0; qreg q[1]; creg c[1]; measure q[0] -> c[0];",
        "shots": 1024,
        "compile": False,
    }
)

print("task_id:", task_id)
print("status:", tmgr.status(task_id))
print("result:", tmgr.result(task_id, timeout=30.0))
```

## 注意事项

- `Task` 是单例，多个 `Task()` 共享 `tasks/cache/session` 状态。
- 网络异常、认证失败、接口返回异常都可能导致抛错或返回错误字典。
- 方法内部包含短暂 `sleep`，用于降低轮询和提交频率。
