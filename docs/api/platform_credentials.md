# platform_credentials

## 概览

- 模块：`quantum_hw.api.platform_credentials`
- 作用：集中管理四家量子云平台的 API 凭证，支持配置文件和环境变量两种方式。

## 凭证查找优先级

凭证按以下顺序查找，使用第一个找到的值：

1. **配置文件** `.quantum_hw.yaml`，按以下位置顺序搜索：
  - 当前工作目录及其父目录
  - 包安装目录（`quantum_hw`）及其父目录
  - 可选显式路径：环境变量 `QUANTUM_HW_CONFIG`
2. **环境变量**（见下表）
3. 以上均未找到 → 抛出 `ValueError`

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

### `reload_config() -> None`

强制重新加载配置文件（编辑配置后调用）。

## 示例

### 方式一：配置文件（推荐）

```bash
cp .quantum_hw.example.yaml .quantum_hw.yaml
# 编辑 .quantum_hw.yaml，填入 token
```

```python
from quantum_hw.api.platform_credentials import get_quafu_api_token

token = get_quafu_api_token()  # 自动从配置文件读取
```

### 方式二：环境变量

```python
import os
os.environ["QUAFU_API_TOKEN"] = "<token>"

from quantum_hw.api.platform_credentials import get_quafu_api_token
token = get_quafu_api_token()
```

### 运行时重新加载

```python
from quantum_hw.api.platform_credentials import reload_config

# 修改了 .quantum_hw.yaml 之后
reload_config()
token = get_quafu_api_token()  # 读取更新后的值
```

## 相关页面

- [provider_runtime](./provider_runtime.md)
- [providers](./providers.md)
