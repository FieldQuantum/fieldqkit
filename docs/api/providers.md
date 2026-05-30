# providers

## 概览

- 模块：
  - `fieldqkit.api.quantum_platform.quafu`
  - `fieldqkit.api.quantum_platform.tianyan`
  - `fieldqkit.api.quantum_platform.guodun`
  - `fieldqkit.api.quantum_platform.tencent`
  - `fieldqkit.api.quantum_platform.origin`
  - `fieldqkit.api.quantum_platform.fieldquantum`
- 作用：分别实现六家 provider 的硬件列表查询、任务提交、状态查询和结果归一化。
- 通用约定：每家 provider 提供三件套 —— `XxxPlatform`（直接 HTTP / SDK 客户端）、`XxxBackendAdapter`（实现 `BackendAdapter` 的 `discover_hardware / resolve_backend`）、`XxxTaskAdapter`（实现 `TaskAdapter` 的 `submit_* / query_status / fetch_result / cancel_task`）。

> **提交语言：** Quafu / Tencent / Origin / FieldQuantum 走 `submit_openqasm`（OpenQASM 2.0）；TianYan / GuoDun 在 `TaskAdapter` 上把 `qcis_native = True`，由 `QuantumHardwareClient` 改走 `submit_qcis`（客户端把 `QuantumCircuit` 直接转 QCIS 后提交，绕过 OpenQASM 中间表示）。

## Quafu 平台

### `QuafuPlatform` 类

**作用：** Quafu 平台客户端，管理硬件列表、HTTP 请求和任务查询。

**初始化：** 单例模式，`QuafuPlatform()` 在构造时通过 [`get_quafu_api_token()`](./platform_credentials.md) 拿到 token（配置文件 → `QUAFU_API_TOKEN` 环境变量）。

**关键方法：**

| 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `verify` | `verify()` | `dict` | 验证当前 API 会话。 |
| `list_available_hardware` | `list_available_hardware()` | `List[Dict[str, Any]]` | 返回统一硬件行：`hardware_name`、`queue_length`、`provider` 等。 |
| `request` | `request(url, data={}, method="get")` | `dict` | 底层 HTTP 请求包装。 |
| `query` | `query(tid=2, chips="Baihua", status="Finished,Failed", start=..., end=..., offset=0, limit=10, sort="submitTime", order="desc")` | `dict` | 列出已提交任务。 |
| `result` | `result(tid: int, timeout: float = 0.0)` | `dict` | 取任务结果，可选阻塞轮询。 |
| `status` | `status(tid: int = 0)` | `dict` | 查询任务状态。 |
| `delete` | `delete(tid: int)` | `dict` | 删除任务记录。 |
| `cancel` | `cancel(tid: int)` | `dict` | 取消运行中的任务。 |
| `run` | `run(task: dict, repeat: int = 1)` | `int` | 提交线路任务并返回任务 ID。 |

**`result` 参数详解：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `tid` | `int` | - | Quafu 平台任务 ID。 |
| `timeout` | `float` | `0.0` | 最大等待秒数；`0` 时只发一次请求不轮询。 |

---

### `QuafuBackendAdapter` 类

**签名：**
```python
class QuafuBackendAdapter(BackendAdapter):
    def __init__(self, *, machine_name: Optional[str] = None, platform_obj: Optional[QuafuPlatform] = None)
```

**属性：**
- `provider = "quafu"`
- `default_hardware_name = "Baihua"`

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `machine_name` | `Optional[str]` | `None` | 指定默认芯片（不提供则由 `resolve_backend` 自动挑选）。 |
| `platform_obj` | `Optional[QuafuPlatform]` | `None` | 复用已有的平台单例；`None` 时新建。 |

---

### `QuafuTaskAdapter` 类

**作用：** Quafu 侧任务适配器（OpenQASM 提交，`qcis_native = False`）。

**关键方法：**

| 方法 | 说明 |
|---|---|
| `submit_openqasm(submit_request, backend)` | 通过 `platform_obj.run(...)` 提交任务，返回 `ProviderTaskHandle`。 |
| `query_status(handle)` | 返回标准状态字符串（`Running / Finished / Failed / Canceled`）。 |
| `fetch_result(handle)` | 返回含 `"count"` 字段的标准化字典。 |
| `cancel_task(handle)` | 调用 `platform_obj.cancel(tid)`。 |

---

## TianYan 平台

### `TianYanPlatform` 类

**继承：** `RemotePlatformClient`（见 [cqlib](./cqlib.md)）

**作用：** 天衍远程平台 HTTP 客户端。

**初始化：** `TianYanPlatform(login_key, auto_login=True, machine_name=None)`，`login_key` 一般通过 `get_tianyan_api_token()` 解析（配置文件 → `TIANYAN_API_TOKEN`）。

**关键方法（在 `RemotePlatformClient` 基础上）：**

| 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `list_available_hardware` | `list_available_hardware()` | `List[Dict]` | 通过 `query_quantum_computer_records()` 拉取并 `normalize_hardware_rows("tianyan", ...)`。 |
| `query_experiment` | `query_experiment(query_id, max_wait_time=120, sleep_time=5)` | `list` | 轮询查询结果（继承自 `RemotePlatformClient`）。 |
| `submit_job` | `submit_job(circuit, num_shots=12000, language=QuantumLanguage.QCIS, ...)` | `query_ids` | 提交 QCIS 线路。 |
| `download_config` | `download_config(read_time=None, machine=None)` | `dict` | 下载硬件标定快照。 |

---

### `TianYanBackendAdapter` 类

**签名：**
```python
class TianYanBackendAdapter(BackendAdapter):
    def __init__(self, *, machine_name: Optional[str] = None, api_token: Optional[str] = None)
```

**属性：**
- `provider = "tianyan"`
- `default_hardware_name = "tianyan176"`

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `machine_name` | `Optional[str]` | `None` | 指定芯片。 |
| `api_token` | `Optional[str]` | `None` | TianYan 登录密钥；`None` 时由 `get_tianyan_api_token()` 解析。 |

---

### `TianYanTaskAdapter` 类

**作用：** TianYan 侧任务适配器（QCIS 提交，`qcis_native = True`）。

**关键行为：**
- `submit_qcis(submit_request, backend)`：调用 `platform.submit_job(circuit=qcis_text, language=QuantumLanguage.QCIS, num_shots=...)` 拿到 `query_id` 并封装成 `ProviderTaskHandle`。
- `query_status(handle)`：调用 `query_experiment(...)` 并缓存最近一次结果以避免重复轮询。
- `fetch_result(handle)`：用 `extract_counts_from_result_items(...)` 归一化 counts。
- `cancel_task(handle)`：通过 `stop_running_experiments(query_id=...)` 终止任务。

---

## GuoDun 平台

### `GuoDunPlatform` 类

**继承：** `RemotePlatformClient`

**作用：** 国盾远程平台 HTTP 客户端。

**关键方法（GuoDun 特有）：**

| 方法 | 签名 | 说明 |
|---|---|---|
| `list_available_hardware` | `list_available_hardware()` | 返回国盾硬件列表（统一行结构）。 |
| `re_execute_task` | `re_execute_task(query_id=None, lab_id=None)` | 重新执行已提交任务，返回服务端响应 `data`。 |
| `stop_running_experiments` | `stop_running_experiments(lab_id=None, query_id=None)` | 停止运行中的实验。 |
| `create_waveform_data` | `create_waveform_data(circuit, circuit_name=None) -> int` | 为线路生成波形数据，返回波形 ID。 |
| `query_waveform_data` | `query_waveform_data(query_id: int) -> str` | 查询波形数据状态。 |

`re_execute_task` / `stop_running_experiments` 须至少提供 `lab_id` 或 `query_id`，否则抛 `ValueError("Please provide lab_id or query_id.")`。

---

### `GuoDunBackendAdapter` 类

**签名：**
```python
class GuoDunBackendAdapter(BackendAdapter):
    def __init__(self, *, machine_name: Optional[str] = None, api_token: Optional[str] = None)
```

**属性：**
- `provider = "guodun"`
- `default_hardware_name = "gd_qc1"`

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `machine_name` | `Optional[str]` | `None` | 指定芯片。 |
| `api_token` | `Optional[str]` | `None` | 国盾登录密钥；`None` 时由 `get_guodun_api_token()` 解析。 |

---

### `GuoDunTaskAdapter` 类

**作用：** GuoDun 侧任务适配器（QCIS 提交，`qcis_native = True`）。

**关键行为：**
- `submit_qcis(submit_request, backend)`：调用 `platform.submit_job(circuit=qcis_text, language=QuantumLanguage.QCIS, ...)` 拿到 `query_id`。
- `query_status(handle)`：轮询 `query_experiment(...)`。
- `fetch_result(handle)`：用 `extract_counts_from_result_items(...)` 归一化 counts。
- `cancel_task(handle)`：调用 `stop_running_experiments(query_id=...)`。

---

## Tencent 平台

### `TencentPlatform` 类

**作用：** 腾讯量子云客户端，基于 `tensorcircuit.cloud`。

**初始化：** 需提供 API token（配置文件 `credentials.tencent.api_token` → `TENCENT_API_TOKEN` 环境变量）。

**关键方法：**

| 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `list_available_hardware` | `list_available_hardware()` | `List[Dict]` | 返回统一硬件行。 |
| `submit_task` | `submit_task(source, device_name, shots=1024)` | `str` | 提交 OpenQASM 2.0 任务，返回任务 ID。 |
| `query_task_state` | `query_task_state(task_id, device_name)` | `str` | 查询任务状态。 |
| `fetch_task_result` | `fetch_task_result(task_id, device_name)` | `Dict[str, int]` | 获取测量计数。 |

**`submit_task` 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `source` | `str` | - | OpenQASM 2.0 线路字符串。 |
| `device_name` | `str` | - | 目标设备名称（如 `tianji_s2`）。 |
| `shots` | `int` | `1024` | 测量次数。 |

**注意事项：**
- QASM 中不可包含 `u(...)` 门和 `barrier` 指令（腾讯 QOS 解析器不支持）。`QuantumHardwareClient.run_auto` 会在 provider=tencent 时自动将 `convert_single_qubit_gate_to_u` 设为 `False`；适配器还会剥离 barrier。
- 返回 bitstring 为 big-endian（q[0] 在最左/最高位），适配器自动翻转为本包约定的 little-endian。

---

### `TencentBackendAdapter` 类

**签名：**
```python
class TencentBackendAdapter(BackendAdapter):
    def __init__(self, *, machine_name: Optional[str] = None, token: Optional[str] = None)
```

**属性：**
- `provider = "tencent"`
- `default_hardware_name = "tianji_s2"`

**支持芯片：** `simulator:tc`、`tianji_m2/_m2v14s2/_m2v14s4/_m2v15s3/_m2v16s1`、`tianji_s2/_s2v6/_s2v7`、`tianxuan_s2/_s2v20s1/_s2v20s2`（以平台实际在线为准）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `machine_name` | `Optional[str]` | `None` | 指定芯片。 |
| `token` | `Optional[str]` | `None` | API token；`None` 时从凭证管理器读取。 |

---

### `TencentTaskAdapter` 类

**作用：** 腾讯量子云任务适配器（OpenQASM 提交）。

**关键行为：**
- `submit_openqasm`：通过 tensorcircuit cloud API 提交 OpenQASM 2.0 线路。
- `query_status`：轮询任务状态（`completed/failed/pending/scheduled` → `Finished/Failed/Running`）。
- `fetch_result`：获取 counts 并自动 big-endian → little-endian 翻转。
- `cancel_task`：腾讯云不支持任务取消，仅记录 warning。

---

## Origin 平台（本源量子）

> **前置依赖：** 使用 Origin provider 前需安装 `pyqpanda3`：`pip install pyqpanda3` 或 `pip install -e .[origin]`。

### `OriginPlatform` 类

**作用：** 本源量子云客户端，封装 `pyqpanda3.qcloud.QCloudService`，对接 OriginQ 公有云（https://qcloud.originqc.com.cn/）。

**初始化：**

```python
OriginPlatform(token: Optional[str] = None, url: str = ORIGIN_DEFAULT_URL)
```

**关键方法：**

| 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `list_available_hardware` | `list_available_hardware()` | `List[Dict]` | 返回在线硬件列表（不含服务端模拟器）。 |
| `submit_task` | `submit_task(source, device_name, shots=1024)` | `str` | 提交 OpenQASM 2.0 任务，返回 job id。 |
| `query_task_state` | `query_task_state(task_id, device_name)` | `str` | 查询任务统一状态字符串。 |
| `fetch_task_result` | `fetch_task_result(task_id, device_name)` | `Dict[str, int]` | 获取测量计数。 |

**注意事项：**
- OpenQASM 2.0 由 `pyqpanda3.intermediate_compiler.convert_qasm_string_to_qprog` 转为 QProg 再提交。
- 默认关闭服务端的 mapping / optimization / amend，确保比特对应关系与本地转译一致。
- 返回 bitstring 已从 little-endian（OriginQ 默认）翻转为本包约定的 big-endian。

---

### `OriginBackendAdapter` 类

**签名：**
```python
class OriginBackendAdapter(BackendAdapter):
    def __init__(self, *, machine_name: Optional[str] = None, token: Optional[str] = None, url: str = ORIGIN_DEFAULT_URL)
```

**属性：**
- `provider = "origin"`
- `default_hardware_name = "WK_C180"`

**支持芯片：** `PQPUMESH8`、`WK_C180`（以平台实际在线为准）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `machine_name` | `Optional[str]` | `None` | 指定芯片。 |
| `token` | `Optional[str]` | `None` | API token；`None` 时由 `get_origin_api_token()` 解析。 |
| `url` | `str` | `ORIGIN_DEFAULT_URL` | 云服务端 URL。 |

---

### `OriginTaskAdapter` 类

**作用：** 本源量子云任务适配器（OpenQASM 提交）。

**关键行为：**
- `submit_openqasm`：将 OpenQASM 2.0 转换为 QProg 后通过 pyqpanda3 提交。
- `query_status`：轮询任务状态（`FINISHED/FAILED/WAITING/QUEUING/COMPUTING` → `Finished/Failed/Running`）。
- `fetch_result`：获取 counts 并自动 little-endian → big-endian 翻转。
- `cancel_task`：OriginQ SDK 不支持任务取消，仅记录 warning。

---

## FieldQuantum 平台（量坤云端模拟器）

> **简介：** 量坤云端模拟器（FieldQuantum Cloud Simulator）是云端托管的无噪声模拟服务，特别适合在没有真机配额或需要快速验证变分算法时使用。**梯度计算下放到服务端**（`expectation` 模式自动做 parameter-shift），可显著降低本地资源占用。
>
> - 服务端口：`https://api.fieldquantum.tech/api/v1/fieldquantum`（可被环境变量 `FIELDQUANTUM_SERVER_URL` 覆盖）
> - 认证方式：HTTP Bearer，token 形如 `fq_<32hex>`，从 [https://fieldquantum.tech/account/api-token/](https://fieldquantum.tech/account/api-token/) 申请
> - 默认硬件名：`fieldquantum_sim`（合成全连接拓扑，按 `num_qubits` 动态生成）

### `FieldQuantumPlatform` 类

**作用：** FieldQuantum 云端模拟器的 HTTP 客户端，封装任务提交、状态查询、结果拉取，以及 `expectation` 模式的端到端阻塞调用。

**初始化：**

```python
FieldQuantumPlatform()
```

- 服务端 URL 取自模块级 `FIELDQUANTUM_DEFAULT_URL`（环境变量 `FIELDQUANTUM_SERVER_URL` 覆盖）。
- Bearer token 在构造时通过 [`get_fieldquantum_api_token()`](./platform_credentials.md) 解析（配置文件 → `FIELDQUANTUM_API_TOKEN` 环境变量）。

**REST 端点（仅供参考）：**

| 端点 | 方法 | 说明 |
|---|---|---|
| `/task/run` | POST | 提交任务，返回 `{"task_id", "status": "submitted"}`。 |
| `/task/status/{task_id}` | GET | 查询任务原始状态。 |
| `/task/result/{task_id}` | GET | 获取已完成任务结果；HTTP 425 表示尚未就绪。 |

**关键方法：**

| 方法 | 签名 | 说明 |
|---|---|---|
| `submit_job` | `submit_job(payload: dict) -> str` | POST `/task/run`，返回服务端 `task_id`（字符串）。`payload` 必含 `"mode"`。 |
| `query_task_status` | `query_task_status(task_id: str) -> str` | GET `/task/status/{id}`，返回原始状态字符串（`submitted/queued/running/finished/failed/error`）。 |
| `fetch_task_result` | `fetch_task_result(task_id: str) -> dict` | GET `/task/result/{id}`，返回 `result` 字段内容；HTTP 425 → `RuntimeError("not ready")`。 |
| `run_expectation` | `run_expectation(qasm, param_names, param_values, hamiltonian, *, poll_interval=3.0, timeout=600.0) -> dict` | 提交 `expectation` 任务并阻塞轮询，返回 `{"energy", "expectations", "gradients"}`。服务端完成参数移位。 |

**两种 mode 的返回字段：**

| `mode` | 服务端返回 `result` 字段 | 用途 |
|---|---|---|
| `sample` | `{"counts": {...}}` | `run_auto` / Shadow 等需要测量计数的场景。 |
| `expectation` | `{"energy": float, "expectations": [...], "gradients": [...]}` | VQE / QAOA / QML 的服务端梯度评估。 |

**`run_expectation` 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `qasm` | `str` | - | 含符号参数的 OpenQASM 2.0 模板（如 `rx(theta_0) q[0];`）。 |
| `param_names` | `List[str]` | - | 参数名列表，与 `param_values` 一一对应。 |
| `param_values` | `List[float]` | - | 当前参数数值。 |
| `hamiltonian` | `List[Dict]` | - | 哈密顿量项，格式 `[{"coeff": float, "pauli": str}, ...]`。 |
| `poll_interval` | `float` | `3.0` | 状态轮询间隔（服务端建议 ≥ 3s）。 |
| `timeout` | `float` | `600.0` | 阻塞硬超时（秒），超过抛 `TimeoutError`。 |

**状态归一化：** 适配器内 `_STATUS_MAP` 将服务端状态映射为本项目统一状态：

| 服务端状态 | 统一状态 |
|---|---|
| `submitted` / `pending` / `queued` / `running` | `Running` |
| `finished` | `Finished` |
| `failed` / `error` | `Failed` |

---

### `FieldQuantumBackendAdapter` 类

**签名：**
```python
class FieldQuantumBackendAdapter(BackendAdapter):
    def __init__(self, *, num_qubits: int = 16)
```

**属性：**
- `provider = "fieldquantum"`
- `default_hardware_name = "fieldquantum_sim"`

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `num_qubits` | `int` | `16` | 合成芯片默认比特数；`resolve_backend` 实际使用调用时传入的 `num_qubits`。 |

**说明：** 构造时即建立 `FieldQuantumPlatform` 实例（要求 token 已配置）。`resolve_backend` 根据请求的 `num_qubits` 动态构建合成 chip_info（全连接、保真度 1.0）。

---

### `FieldQuantumTaskAdapter` 类

**作用：** FieldQuantum 侧任务适配器（`sample` 模式）。

**关键行为：**
- `submit_openqasm`：以 `{"mode": "sample", "qasm": ..., "shots": ...}` POST 到 `/task/run`，返回 `ProviderTaskHandle(provider="fieldquantum", task_id=...)`。
- `query_status`：调用 `query_task_status` 并经 `_STATUS_MAP` 归一化为 `Running / Finished / Failed`。
- `fetch_result`：返回 `{"count": result["counts"]}`，与其它 provider 保持字段一致。
- 该适配器不实现 `cancel_task`（服务端暂未暴露取消接口）。

> 变分算法在 `provider="fieldquantum"` 下会优先走 `FieldQuantumPlatform.run_expectation` 拿到服务端梯度，因此调用栈一般不经过 `FieldQuantumTaskAdapter`。

---

## 平台选择与创建

通过 `create_provider_runtime(provider, client)` 工厂函数（见 [provider_runtime](./provider_runtime.md)），根据 provider 名称自动创建适配器对（支持 `quafu / tianyan / guodun / tencent / origin / fieldquantum / simulator`）：

```python
from fieldqkit.api.quantum_platform import create_provider_runtime
from fieldqkit.api.client import QuantumHardwareClient

client = QuantumHardwareClient()
runtime = create_provider_runtime(provider="quafu", client=client)
# 返回 ProviderRuntime(
#     backend_adapter=QuafuBackendAdapter(...),
#     task_adapter=QuafuTaskAdapter(...),
# )
```

## 相关页面

- [Task](./Task.md)
- [cqlib](./cqlib.md)
- [platform_credentials](./platform_credentials.md)
- [provider_runtime](./provider_runtime.md)
