# 配置凭证 (Configuration)

`fieldqkit` 的**本地模拟器无需任何配置**即可使用：

```python
from fieldqkit import QuantumHardwareClient

client = QuantumHardwareClient()
result = client.run_auto(
    circuit="ghz", name="demo", num_qubits=4,
    provider="simulator",      # 纯本地模拟，无需 token
    shots=4096,
    observables=["ZZII", "IIZZ"],
)
print(result.observable_values)
```

只有当你要**提交到真实量子云平台**（夸父 / 天衍 / 国盾 / 腾讯 / 本源 / 量坤）时，才需要配置该平台的 API token。下面三种方式任选其一。

---

## 方式一：环境变量（pip 用户最简单）

无需任何文件，直接设置对应平台的环境变量即可：

| 平台 | 环境变量 | 申请地址 |
|---|---|---|
| 夸父 Quafu | `QUAFU_API_TOKEN` | https://quafu-sqc.baqis.ac.cn/ |
| 天衍 TianYan | `TIANYAN_API_TOKEN` | https://qc.zdxlz.com/ |
| 国盾 GuoDun | `GUODUN_API_TOKEN` | https://quantumctek-cloud.com/ |
| 腾讯 Tencent | `TENCENT_API_TOKEN` | https://quantum.tencent.com/cloud/ |
| 本源 Origin | `ORIGIN_API_TOKEN` | https://qcloud.originqc.com.cn/ |
| 量坤 FieldQuantum | `FIELDQUANTUM_API_TOKEN` | https://fieldquantum.tech/ |

```bash
# Linux / macOS
export QUAFU_API_TOKEN="your-quafu-token"

# Windows PowerShell
$env:QUAFU_API_TOKEN = "your-quafu-token"
```

> 推荐先用**夸父量子云**：免费、不限时，适合入门体验。

## 方式二：配置文件（一键生成）

如果你更希望把 token 持久化到文件，运行内置命令生成模板（默认写到 `~/.quantum_hw.yaml`）：

```bash
fieldqkit-config-init                       # 写到 ~/.quantum_hw.yaml
fieldqkit-config-init --path ./my.yaml      # 或指定路径
fieldqkit-config-init --force               # 覆盖已存在的文件
```

也可以在 Python / Notebook 中调用：

```python
import fieldqkit
fieldqkit.init_config()                     # 返回写入的路径
```

然后编辑生成的文件，填入 token：

```yaml
credentials:
  quafu:
    api_token: "your-quafu-token"
  tianyan:
    api_token: ""
  # ... 其余平台留空即可
```

## 方式三：项目内配置文件（开发者）

从源码仓库开发时，可复制根目录的模板到项目根目录：

```bash
cp .quantum_hw.example.yaml .quantum_hw.yaml
```

`.quantum_hw.yaml` 已在 `.gitignore` 中排除，不会被提交。

---

## 查找优先级

凭证按以下顺序解析：

1. `$QUANTUM_HW_CONFIG` 指定的配置文件（显式覆盖）
2. **当前工作目录**及其各级父目录下的 `.quantum_hw.yaml`
3. **用户主目录**：`~/.quantum_hw.yaml`，然后 `~/.config/fieldqkit/credentials.yaml`
4. 包安装目录及其父目录下的 `.quantum_hw.yaml`（源码 / editable 安装）
5. 上述环境变量

> 即：项目内文件 > 用户主目录文件 > 环境变量。配置文件优先于环境变量。

编辑配置文件后，如果同一进程已经读取过缓存，可调用 `fieldqkit.api.platform_credentials.reload_config()` 强制重载。

## 安全提示

- **切勿把含真实 token 的文件提交到 Git 或公开分享。** `.quantum_hw.yaml` 默认已被忽略。
- token 泄露后请尽快到对应平台**吊销并重新生成**。
- CI / 服务器环境优先使用环境变量（或密钥管理服务），避免明文落盘。
