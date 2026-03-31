# platform_credentials

## 概览

- 模块：`quantum_hw.api.platform_credentials`
- 作用：集中读取三家平台凭据（优先环境变量）。

## 函数

### `get_quafu_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 环境变量 | `QUAFU_API_TOKEN` |
| 返回值 | Quafu API token 字符串 |

### `get_tianyan_login_key() -> str`

| 项目 | 说明 |
|---|---|
| 环境变量 | `TIANYAN_LOGIN_KEY` |
| 返回值 | TianYan 登录 key 字符串 |

### `get_guodun_login_key() -> str`

| 项目 | 说明 |
|---|---|
| 环境变量 | `GUODUN_LOGIN_KEY` |
| 返回值 | GuoDun 登录 key 字符串 |

### `get_tencent_api_token() -> str`

| 项目 | 说明 |
|---|---|
| 环境变量 | `TENCENT_API_TOKEN` |
| 返回值 | Tencent 量子云 API token 字符串 |

## 行为说明

- 当前实现优先读取环境变量。
- 当环境变量未设置时，会回退到模块内调试常量。

## 推荐实践

- 生产环境应始终通过环境变量注入密钥。
- 建议在 CI/部署环境中显式检查三项环境变量是否已设置。

## 示例

```python
import os
from quantum_hw.api.platform_credentials import (
    get_quafu_api_token,
    get_tianyan_login_key,
    get_guodun_login_key,
)

os.environ["QUAFU_API_TOKEN"] = "<token>"
os.environ["TIANYAN_LOGIN_KEY"] = "<key>"
os.environ["GUODUN_LOGIN_KEY"] = "<key>"

print(bool(get_quafu_api_token()))
print(bool(get_tianyan_login_key()))
print(bool(get_guodun_login_key()))
```

## 相关页面

- [provider_runtime](./provider_runtime.md)
- [providers](./providers.md)
