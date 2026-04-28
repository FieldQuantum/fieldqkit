# C 部分：API 层架构清理建议

> 仅作为后续清理工作的参考清单， 建议这些项作为独立 PR 推进。

适用偏好：
- 架构优先干净统一，不接受为兼容/可测性保留生产钩子字段。
- 尽量把 provider 差异收敛到抽象层，减少转发和重复实现。
- 无运行时收益的能力声明层（capabilities 类/方法）倾向直接删除。

---

## C1. `TaskAdapter` ABC 实质上是接口声明，但混入 `NotImplementedError` 默认实现

**位置**：[src/quantum_hw/api/task.py](src/quantum_hw/api/task.py)

**现状**
- `TaskAdapter(ABC)` 中的 `submit_openqasm` / `query_status` / `fetch_result` / `cancel_task` 全部以 `raise NotImplementedError(...)` 作为"默认实现"。
- 没有标注 `@abstractmethod`，所以子类可以"忘记"覆盖而仍然实例化成功，运行时才报错。
- `SimulatorBackendAdapter` 在 [quantum_platform/__init__.py#L47](src/quantum_hw/api/quantum_platform/__init__.py#L47) 把 `task_adapter=None`，调用方需要在所有使用点判空（违反"接口必须实现"的合约）。
- 等价于"运行时无收益的能力声明层"。

**建议（任选其一）**
1. **彻底删除 ABC**：把 `TaskAdapter` 改成 `Protocol`，依靠类型注解约束；或者直接在 `client.py` 中以 `Optional[TaskAdapter]` + `if task_adapter is None: …` 单点处理 simulator 短路。
2. **真正的 ABC**：给 4 个方法加 `@abstractmethod`，并提供显式的 `NullTaskAdapter`（simulator 用），让所有 provider 严格实现；同时把 `task_adapter` 字段类型由 `Any` 改成 `TaskAdapter`，消除 `None` 分支。

**影响范围**
- `src/quantum_hw/api/task.py`
- `src/quantum_hw/api/quantum_platform/__init__.py`（`ProviderRuntime.task_adapter` 类型与 simulator 分支）
- `src/quantum_hw/api/client.py` 中所有 `if self._active_task_adapter is not None:` 判空点
- 各 provider 的 `*TaskAdapter` 实现保持不变（已经全部覆盖了这 4 个方法）

---

## C2. `RemotePlatformClient` 公开导出但属于内部基类

**位置**：[src/quantum_hw/api/quantum_platform/__init__.py](src/quantum_hw/api/quantum_platform/__init__.py)（第 86 行附近的 `__all__`）+ [src/quantum_hw/api/__init__.py](src/quantum_hw/api/__init__.py)

**现状**
- `RemotePlatformClient` 是 cqlib 的内部抽象基类，被同时导出到 `quantum_hw.api.quantum_platform` 与 `quantum_hw.api`。
- docs 中虽然出现了 [docs/api/cqlib.md](docs/api/cqlib.md)，但用户层没有"用 cqlib 客户端发任务"这种正常使用路径，全部走 `Backend` + `run_with_backend`。

**建议**
- 重命名为 `_RemotePlatformClient` 或从 `__all__` 中移除，仅保留 `quantum_platform.cqlib` 内部 import。
- 同步删除 [docs/api/cqlib.md](docs/api/cqlib.md) 或并入 `providers.md` 的内部注解段。

---

## C3. provider Adapter 全部以 top-level 公开导出

**位置**：[src/quantum_hw/api/__init__.py](src/quantum_hw/api/__init__.py#L4-L29)

**现状**
- `QuafuBackendAdapter`、`TianYanBackendAdapter`、`GuoDunBackendAdapter`、`TencentBackendAdapter`、`FieldQuantumBackendAdapter` 以及对应的 `*TaskAdapter` / `*Platform` 类全部从 `quantum_hw.api` 顶层导出。
- 用户面唯一稳定入口是 `create_provider_runtime(provider=..., client=...)`，adapter 不是用户应直接构造的对象。
- 这些 adapter 都内置 provider-specific 状态（platform_obj、token、URL），公开它们等于把 provider 差异泄到 API 表面。

**建议**
- 顶层 `quantum_hw.api.__init__.py` 只导出：`QuantumHardwareClient`, `Backend`, `HardwareProfile`, `HardwareTopology`, `HardwareCalibration`, `ResolvedBackend`, `BackendAdapter`(可保留也可去掉), `OpenQasmSubmitRequest`, `ProviderTaskHandle`, `TaskAdapter`, `ProviderRuntime`, `create_provider_runtime`, `list_available_hardware`。
- 各 provider 的 `*BackendAdapter / *TaskAdapter / *Platform` 仅在 `quantum_hw.api.quantum_platform.<provider>` 子模块可见，不进入 `quantum_platform.__init__` 的 `__all__`。
- 保留 docs/api/providers.md，但说明 adapter 是"通过 `create_provider_runtime` 间接访问"的内部对象。

**影响**
- 上层 `examples/` 与测试需要核对是否有 `from quantum_hw.api import QuafuBackendAdapter` 类似直接 import。当前抽样未发现此类用法，影响面应当较小。

---

## C4. `BackendAdapter` ABC 同时含 abstract 与 concrete 方法

**位置**：[src/quantum_hw/api/backend.py](src/quantum_hw/api/backend.py#L425-L460)

**现状**
- 类继承 `ABC`，但成员方法 `list_available_hardware` / `discover_hardware` / `resolve_backend` / `_fallback_hardware_name` 都给了具体实现，且没有任何 `@abstractmethod`。
- 等价于"普通基类伪装成 ABC"。子类重写哪些方法、必须提供 `_platform` 属性，全靠隐式约定。

**建议**
1. 把 `BackendAdapter` 改成普通基类，去掉 `ABC` 继承；显式注释/类型提示 `_platform` 字段，作为子类 contract。
2. 或者把 `_platform` 抽象出 `@property @abstractmethod`，强制子类实现。
3. 顺带把 `_fallback_hardware_name` 单下划线方法移成 `_FallbackPolicy` 这种小工具，避免基类承担太多职责。

---

## C5. `create_provider_runtime` 中 fieldquantum 分支重复 import

**位置**：[src/quantum_hw/api/quantum_platform/__init__.py](src/quantum_hw/api/quantum_platform/__init__.py#L70-L82)

**现状**
- 文件顶部已经从 `.fieldquantum` 导入了 `FieldQuantumPlatform / FieldQuantumBackendAdapter / FieldQuantumTaskAdapter / FIELDQUANTUM_DEFAULT_URL`。
- `fieldquantum` 分支函数体内又再次 `from .fieldquantum import (FieldQuantumBackendAdapter, FieldQuantumTaskAdapter)` 并自己手动取 `os.environ.get("FIELDQUANTUM_SERVER_URL", "http://localhost:8765")`。
- 顶层已导入的 `FIELDQUANTUM_DEFAULT_URL` 没有用上，base_url 字面量在两处分别硬编码。

**建议**
- 删除函数内重复 import；统一用 `os.environ.get("FIELDQUANTUM_SERVER_URL", FIELDQUANTUM_DEFAULT_URL)`。
- 让所有 provider 分支结构对齐（quafu / tianyan / guodun / tencent 都是 3 行；fieldquantum 应同样精简）。

---

## C6. provider 之间结果/profile schema 不一致

**位置**：各 provider 的 `fetch_result` 返回结构、`list_available_hardware` rows。

**现状（来自审计）**
- `Quafu`: `{"count": {bitstring: int}}`
- `TianYan / GuoDun`: `{"count": extract_counts_from_result_items(...)}`
- `Tencent`: 多种格式经 `_get_task_detail` 处理
- `FieldQuantum`: `{"count": result.get("counts", {})}`

四个 provider 都用 `count` 键，但底层 dict 形态、元数据字段并不严格统一。`BackendAdapter.discover_hardware` 假定 rows 一定有 `hardware_name` / `queue_length`，对部分 provider 不成立。

**建议**
- 在 `task.py` 中引入 `ProviderTaskResult` dataclass，强制 provider 适配后再回到 `client.py`（当前 `client._collect_counts_from_result` 在做转换，但分布零散）。
- 在 `backend.py` 的 `HardwareProfile` 构造前，加一个 provider 内部归一化函数 `_normalize_hardware_row(row) -> dict`，让后续 `discover_hardware` 只面对统一 schema。
- 这一项是"收敛 provider 差异到抽象层"偏好的核心实现点。

---

## C7. 错误信息与日志中的敏感字段

**位置**：
- [src/quantum_hw/api/platform_credentials.py](src/quantum_hw/api/platform_credentials.py#L155-L164)：错误信息把 env var 名、yaml 路径与 token 字段名一并写出。
- [src/quantum_hw/api/fieldquantum_server.py](src/quantum_hw/api/fieldquantum_server.py#L369-L371)：将完整 traceback 写入日志，且把 `str(exc)` 直接通过 HTTP 500 响应返回客户端。

**建议**
- credentials 报错改为引用 docs 链接，不在异常字符串里枚举 env var 名/字段名。
- HTTP server 返回给客户端的 `{"error": str(exc)}` 改为通用 message，详细堆栈只写本地日志。

---

## C8. `cqlib` 401 重新登录缺失次数限制

**位置**：[src/quantum_hw/api/quantum_platform/cqlib.py](src/quantum_hw/api/quantum_platform/cqlib.py#L299-L307)

**现状**
- 401 时直接 `self.login()` 并继续重试；如果 token/账户也失效，会陷入"过期 → 登录失败 → 仍用旧 token → 再 401"的循环。

**建议**
- `login()` 应返回成功/失败标志，外层在 N 次连续 401 后直接 raise；同时为 `_reconnect_on_failure` 引入显式的 `max_relogin_attempts`。

---

## 推荐的实施顺序

1. **C5**（一处局部清理，零风险）→ 先做。
2. **C2** 与 **C3**（公开面收敛，主要是 `__all__` 与 docs 调整）。
3. **C1** + **C4**（ABC 语义统一），与 `client.py` 中 `task_adapter is None` 短路逻辑一并清理。
4. **C6**（provider 结果/硬件 schema 归一化），最大改动量；建议单独 PR + 增加 provider 间一致性测试。
5. **C7** + **C8**（安全/可靠性补强），可与 C6 合并 PR。

---

## 不建议改动的项

下面这些在审计中被子代理标记，但经源码复核后**确认是误报或当前足够安全**，不需要在 C 类清理中处理：

- `compile/optimize.py::cancel_two_qubit_pairs` 的 `qubits_j == qubits_i[::-1]`：list 反向比较对 `cz/swap` 这类对称门是正确的。
- `Backend.__init__` 中的 lazy import：分支结构清晰，缺省即 `ImportError`，无需 try/except 包装。
- `client._run_with_backend` 自动 provisioning 的 "infer→raise→set_machine" 顺序：先 raise 再 set_machine，无状态泄漏风险。
- 所有 docs 示例：抽样验证均能与现实现对得上，没有 broken example。
