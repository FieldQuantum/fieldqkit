# fieldqkit

**A user-facing Python interface for controlling quantum hardware.** One unified API across multiple quantum-cloud platforms (Quafu / TianYan / GuoDun / Tencent / Origin / FieldQuantum), automatic transpilation, error mitigation (readout + ZNE), variational algorithms (VQE / QAOA / Shadow tomography / QML), and a built-in PyTorch simulator with autodiff. Full documentation: <https://fieldquantum.github.io/fieldqkit/>.

> 面向用户的**量子硬件控制接口**：统一多平台访问 · 自动编译 · 误差缓解 · 变分算法 · 内置 PyTorch 模拟器（支持自动微分）。

---

## 项目定位

`fieldqkit` 是一个面向用户的**量子硬件控制接口**，提供从量子线路构建、编译转译、提交执行、误差缓解到变分算法的完整工作流。项目以统一 API 屏蔽多量子云平台（夸父 / 天衍 / 国盾 / 腾讯 / 本源）的差异，并内置基于 PyTorch 的本地模拟器，支持自动微分和大规模张量网络仿真。

核心目标：

| 目标 | 说明 |
|---|---|
| **统一硬件访问** | 单一 `QuantumHardwareClient` 对接多平台（夸父/天衍/国盾/腾讯/本源/FieldQuantum） |
| **自动编译** | 逻辑电路 → 物理芯片的完整转译流程 |
| **误差缓解** | Readout 校准 + 零噪声外推（ZNE） |
| **变分算法** | VQE、QAOA、Shadow Tomography、QML |
| **量子机器学习** | PQC 监督分类 + 无监督 QNN 分布学习 |
| **硬件校准** | Readout、原生两比特 RB、过程层析 |
| **高效仿真** | 全态矢量 + MPS + MPO，支持梯度计算 |

## 安装

```bash
pip install fieldqkit
```

> 核心依赖：Python >= 3.9，`numpy>=1.24`，`scipy>=1.10`，`networkx>=3.0`，`requests>=2.31`，`matplotlib>=3.7`，`pyyaml>=6.0`。

按需安装可选依赖组：

```bash
pip install "fieldqkit[sim]"       # 本地模拟器（torch>=2.1，运行 fieldqkit.sim 必需）
pip install "fieldqkit[origin]"    # 接入本源量子云（pyqpanda3）
pip install "fieldqkit[test]"      # 运行测试（pytest）
```

> **量坤云端模拟器**（`fieldquantum` provider）无需额外依赖，仅需配置 `fq_<32hex>` 形式的 API token。
>
> 从源码开发：`git clone` 仓库后执行 `pip install -e ".[sim,test]"`。

## 快速开始（本地模拟器，无需 token）

最快的上手方式是用内置模拟器，**无需任何配置**（需安装 `[sim]` 依赖组）：

```python
from fieldqkit import QuantumHardwareClient

client = QuantumHardwareClient()
result = client.run_auto(
    circuit="ghz",
    name="demo",
    num_qubits=4,
    provider="simulator",      # 纯本地模拟，无需 token
    shots=8192,
    observables=["ZZII", "IIZZ"],
    return_probabilities=True,
)

print(result.observable_values)   # {'ZZII': 1.0, 'IIZZ': 1.0}
print(result.probabilities)
```

把 `provider` 换成 `"quafu"` / `"tianyan"` / `"guodun"` / `"tencent"` / `"origin"` / `"fieldquantum"`
即可提交到对应的量子云平台（需先配置 token，见下文 [真机使用](#真机使用)）。完整示例见
[examples/demo_full.ipynb](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_full.ipynb)。

## 真机使用

使用真机前需要配置对应平台的 API 凭证。任选一种方式（完整说明见 [docs/configuration.md](https://fieldquantum.github.io/fieldqkit/configuration/)）：

**方式一 · 环境变量（pip 用户最简单）**

```bash
export QUAFU_API_TOKEN="your-quafu-token"      # Linux/macOS
# Windows PowerShell: $env:QUAFU_API_TOKEN = "your-quafu-token"
```

各平台环境变量：`QUAFU_API_TOKEN` / `TIANYAN_API_TOKEN` / `GUODUN_API_TOKEN` / `TENCENT_API_TOKEN` / `ORIGIN_API_TOKEN` / `FIELDQUANTUM_API_TOKEN`。

**方式二 · 一键生成配置文件**

```bash
fieldqkit-config-init          # 在 ~/.quantum_hw.yaml 写入模板，编辑后填入 token
```

也可在 Python 中调用 `fieldqkit.init_config()`。然后编辑生成的文件：

```yaml
credentials:
  quafu:
    api_token: "your-quafu-token-here"
```

**方式三 · 从源码开发**：复制仓库根目录模板 `cp .quantum_hw.example.yaml .quantum_hw.yaml`。

> 查找优先级：`$QUANTUM_HW_CONFIG` → 当前目录 `.quantum_hw.yaml` → `~/.quantum_hw.yaml` → 环境变量。`.quantum_hw.yaml` 已在 `.gitignore` 中排除，请勿提交真实 token。

### 各平台链接

- 夸父量子云：https://quafu-sqc.baqis.ac.cn/
- 天衍量子云： https://qc.zdxlz.com/
- 国盾量子云： https://quantumctek-cloud.com/
- 腾讯量子云： https://quantum.tencent.com/cloud/
- 本源量子云： https://qcloud.originqc.com.cn/
- 量坤云端模拟器： https://fieldquantum.tech/

各平台政策不同，优先推荐使用夸父量子云的免费资源（不限时）进行体验和学习。

## 教程导航（Notebook）

> 每个 notebook 顶部带 **Open in Colab** 徽章，可一键在 Colab 打开运行（首个单元格会 `pip install fieldqkit`）。

- [全览入门：run_auto + mitigation + 可视化](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_full.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_full.ipynb)
- [QuantumCircuit 与 core 函数拆解](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_circuit_core.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_circuit_core.ipynb)
- [Shadow tomography 分层教程](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_shadow.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_shadow.ipynb)
- [Readout calibration + ZNE 专项](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_readout_zne.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_readout_zne.ipynb)
- [Clifford fitting 闭环（Baihua）](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_clifford_fitting.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_clifford_fitting.ipynb)
- [VQE：顶层接口 + parameter-shift 手动梯度下降](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_vqe.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_vqe.ipynb)
- [QAOA：MaxCut + 自定义哈密顿量 + VQE 对比](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qaoa.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_qaoa.ipynb)
- [QML Iris：Iris 数据集多分类](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qml_iris.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_qml_iris.ipynb)
- [QNN BAS：Born Machine 分布学习](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qnn_bas.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_qnn_bas.ipynb)
- [QNN 无监督：量子分布学习](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qnn_unsupervised.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_qnn_unsupervised.ipynb)
- [VQE H₂ 4-qubit：氢分子势能面扫描](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_vqe_h2_4q.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_vqe_h2_4q.ipynb)
- [Backend 拓扑与芯片排序](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_backend.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FieldQuantum/fieldqkit/blob/main/examples/demo_backend.ipynb)

## 学习路径（入门 → 进阶 → 硬件 → 优化）

1. 入门：先看 [全览入门：run_auto + mitigation + 可视化](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_full.ipynb)
2. 进阶：继续 [QuantumCircuit 与 core 函数拆解](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_circuit_core.ipynb)
3. 硬件：再看 [Readout calibration + ZNE 专项](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_readout_zne.ipynb)
4. 优化：按顺序学习
    - [Shadow tomography 分层教程](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_shadow.ipynb)
    - [VQE：顶层接口 + parameter-shift 手动梯度下降](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_vqe.ipynb)
    - [QAOA：MaxCut + 自定义哈密顿量 + VQE 对比](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qaoa.ipynb)
5. 量子机器学习：按顺序学习
    - [QML Iris：Iris 数据集多分类](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qml_iris.ipynb)
    - [QNN BAS：Born Machine 分布学习](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qnn_bas.ipynb)
    - [QNN 无监督：量子分布学习](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_qnn_unsupervised.ipynb)
6. VQE 进阶：[VQE H₂ 4-qubit：氢分子势能面扫描](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_vqe_h2_4q.ipynb)
7. 硬件拓扑补充：参考 [Backend 拓扑与芯片排序](https://github.com/FieldQuantum/fieldqkit/blob/main/examples/demo_backend.ipynb)

## 文档 (Docs)

- **用户指南**
  - [配置凭证 (Configuration)](https://fieldquantum.github.io/fieldqkit/configuration/) — 环境变量 / 一键生成配置 / 查找优先级
  - 教程 Notebook：见上文 [教程导航](#教程导航notebook) 与 [学习路径](#学习路径入门--进阶--硬件--优化)
- **开发者参考**
  - API 与模块参考总览见 [docs/README.md](https://fieldquantum.github.io/fieldqkit/)

## 参与贡献 (Contributing)

欢迎通过 [GitHub Issues](https://github.com/FieldQuantum/fieldqkit/issues) 反馈问题与需求，也欢迎提交 Pull Request。本地开发：

```bash
git clone https://github.com/FieldQuantum/fieldqkit.git
cd fieldqkit
pip install -e ".[sim,test]"
pytest
```

## 引用 (Citation)

如果 `fieldqkit` 对你的研究有帮助，请引用本项目（元数据见 [CITATION.cff](https://github.com/FieldQuantum/fieldqkit/blob/main/CITATION.cff)）。

## 许可证 (License)

本项目以 [Apache License 2.0](https://github.com/FieldQuantum/fieldqkit/blob/main/LICENSE) 开源。

项目中部分文件改编自第三方开源项目（quarkstudio / cqlib / TensorCircuit），
相关版权与许可声明见 [THIRD_PARTY_NOTICES](https://github.com/FieldQuantum/fieldqkit/blob/main/THIRD_PARTY_NOTICES) 与 [NOTICE](https://github.com/FieldQuantum/fieldqkit/blob/main/NOTICE)。
