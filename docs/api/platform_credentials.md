# platform_credentials

## 概览

- 模块：`fieldqkit.api.platform_credentials`
- 作用：集中管理六家量子云平台（夸父 / 天衍 / 国盾 / 腾讯 / 本源 / 量坤）的 API 凭证，支持配置文件和环境变量两种方式。

## 凭证查找优先级

凭证按以下顺序查找，使用第一个找到的值（与 `_iter_config_candidates()` 实现一致）：

1. **配置文件**，按以下位置顺序搜索（找到第一个存在的文件即停止）：
  1. `$QUANTUM_HW_CONFIG` 指定的文件（显式覆盖，最高优先级）
  2. 当前工作目录及其各级父目录下的 `.quantum_hw.yaml`
  3. 用户主目录：`~/.quantum_hw.yaml`，然后 `~/.config/fieldqkit/credentials.yaml`
  4. 包安装目录（`fieldqkit`）及其父目录下的 `.quantum_hw.yaml`（源码 / editable 安装）
2. **环境变量**（见下表）
3. 以上均未找到 → 抛出 `ValueError`

> 即：显式路径 > 项目内文件 > 用户主目录文件 > 包安装目录文件 > 环境变量。配置文件优先于环境变量。
> 详见 [配置凭证](../configuration.md)。

## 配置文件格式

```yaml
credentials:
  quafu:
    api_token: "your-token"
  tianyan:
    api_token: "your-token"
  guodun:
    api_token: "your-token"
  tencent:
    api_token: "your-token"
  origin:
    api_token: "your-token"
  fieldquantum:
    api_token: "your-token"
```

项目根目录提供了模板文件 `.quantum_hw.example.yaml`，复制并填入 token 即可：

```bash
# 复制模板并填入 token
cp .quantum_hw.example.yaml .quantum_hw.yaml
```

## 函数

### `get_quafu_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 配置路径 | `credentials.quafu.api_token` |
| 环境变量 | `QUAFU_API_TOKEN` |
| 返回值 | Quafu API token 字符串 |

### `get_tianyan_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 配置路径 | `credentials.tianyan.api_token` |
| 环境变量 | `TIANYAN_API_TOKEN` |
| 返回值 | TianYan API token 字符串 |

### `get_guodun_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 配置路径 | `credentials.guodun.api_token` |
| 环境变量 | `GUODUN_API_TOKEN` |
| 返回值 | GuoDun API token 字符串 |

### `get_tencent_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 配置路径 | `credentials.tencent.api_token` |
| 环境变量 | `TENCENT_API_TOKEN` |
| 返回值 | Tencent API token 字符串 |

### `get_origin_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 配置路径 | `credentials.origin.api_token` |
| 环境变量 | `ORIGIN_API_TOKEN` |
| 返回值 | Origin API token 字符串 |

### `get_fieldquantum_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 配置路径 | `credentials.fieldquantum.api_token` |
| 环境变量 | `FIELDQUANTUM_API_TOKEN` |
| 返回值 | FieldQuantum API token 字符串 |
| 申请地址 | [https://fieldquantum.tech/account/api-token/](https://fieldquantum.tech/account/api-token/) |

### `reload_config() -> None`

强制重新加载配置文件（编辑配置后调用）。

## 示例

### 方式一：配置文件（推荐）

```bash
cp .quantum_hw.example.yaml .quantum_hw.yaml
# 编辑 .quantum_hw.yaml，填入 token
```

```python
from fieldqkit.api.platform_credentials import get_quafu_api_token

token = get_quafu_api_token()  # 自动从配置文件读取
```

### 方式二：环境变量

```python
import os
os.environ["QUAFU_API_TOKEN"] = "<token>"

from fieldqkit.api.platform_credentials import get_quafu_api_token
token = get_quafu_api_token()
```

### 运行时重新加载

```python
from fieldqkit.api.platform_credentials import reload_config

# 修改了 .quantum_hw.yaml 之后
reload_config()
token = get_quafu_api_token()  # 读取更新后的值
```

## 相关页面

- [provider_runtime](./provider_runtime.md)
- [providers](./providers.md)
