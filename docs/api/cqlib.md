# cqlib

## 概览

- 模块：`quantum_hw.api.quantum_platform.cqlib`
- 作用：提供 TianYan/GuoDun 共用的远端平台 HTTP 客户端、硬件列表归一化、结果计数提取、后端配置转 `chip_info`。

## 关键类型

### `QuantumLanguage`

```python
class QuantumLanguage(Enum):
    QCIS = "qcis"
    ISQ = "isq"
    QUINGO = "quingo"
```

## 关键函数

### `records_from_platform_list_query(platform_obj) -> List[Dict[str, Any]]`

- 作用：兼容不同平台列表格式，归一化为记录列表。

### `normalize_hardware_rows(provider, records) -> List[Dict[str, Any]]`

- 作用：将平台记录转为统一硬件行结构。

### `extract_counts_from_result_items(result_items, *, num_qubits) -> Dict[str, int]`

- 作用：从平台实验结果中提取并合并 bitstring 计数。
- 行为：优先读取 `resultStatus` 矩阵；失败时回退读取 `count` 字段。

### `chip_info_from_config(config, *, machine_name=None) -> Dict[str, Any]`

- 作用：将平台配置 JSON 转为 `Backend` 可消费的标准 `chip_info`。

### `load_cqlib_chip_info(chip_name, *, provider=None, platform=None) -> Dict[str, Any]`

- 作用：按芯片名/平台下载配置并生成 `chip_info`。

### `load_backend_config(platform, *, machine_name) -> Dict[str, Any]`

- 作用：下载 backend 配置，失败时返回空字典。

## `RemotePlatformClient`

```python
RemotePlatformClient(login_key: str, auto_login: bool = True, machine_name: str = None)
```

### 常用方法

| 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `login` | `login(timeout=60)` | `str` | 登录并获取 access token。 |
| `set_machine` | `set_machine(machine_name)` | `None` | 设置目标芯片。 |
| `submit_job` | `submit_job(..., language=QuantumLanguage.QCIS, ...)` | `query_ids` | 提交实验任务。 |
| `query_experiment` | `query_experiment(query_id, max_wait_time=120, sleep_time=5)` | `list/dict` | 轮询查询结果。 |
| `download_config` | `download_config(read_time=None, machine=None)` | `dict` | 下载设备配置。 |
| `query_quantum_computer_records` | `query_quantum_computer_records()` | `List[Dict[str, Any]]` | 查询硬件记录。 |
| `_send_request` | `_send_request(path, method="GET", data=None, params=None, raise_for_code=True)` | `dict` | 统一 HTTP 请求入口（带重连装饰器）。 |

## 输入输出约定

### 统一硬件行结构

```python
{
    "provider": str,
    "hardware_name": str,
    "queue_length": int | None,
    "status": Any,
    "is_toll": Any,
    "raw": Dict[str, Any],
}
```

### `chip_info` 结构

```python
{
    "chip_name": str,
    "qubits_info": Dict[str, Dict[str, Any]],
    "couplers_info": Dict[str, Dict[str, Any]],
    "global_info": Dict[str, Any],
    "priority_qubits": List[List[int]] | None,
}
```

## 常见异常

- `CqlibRequestError`
  - 登录失败、HTTP 状态码异常、平台返回 code 非 0。
- `CqlibInputParaError`
  - 输入参数缺失（如未传 `lab_id/query_id`）。
- `ValueError`
  - machine 名称非法、provider 无法推断等。

## 相关页面

- [providers](./providers.md)
- [Backend](./Backend.md)
- [Task](./Task.md)
