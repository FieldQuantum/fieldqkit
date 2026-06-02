# 量坤科技开源 fieldqkit —— 一行 API，打通量子算法到国产真机的"最后一公里"

> 用 Qiskit 研究量子信息，却接入不了 IBM 真机？
> 用 TensorCircuit 做张量网络模拟，腾讯真机却不免费开放？
> 用 quarkstudio / cqlib 接入量子院、天衍平台，却没有误差缓解？
> 想设计一套量子算法去真机上验证，结果发现编译、提交、读取、纠偏全得自己手写？
> 模拟器上跑通的线路，一搬到真机就面目全非？

如果以上有任何一条戳中了你，那么 **fieldqkit** 也许正是你在找的那块拼图。

近日，量坤科技正式开源 **fieldqkit** —— 一个面向用户的**量子硬件控制接口**。它用一套统一 API 屏蔽了国内主流量子云平台的差异，把"量子线路构建 → 自动编译 → 真机提交 → 误差缓解 → 变分算法"的完整工作流封装在一起，并内置了一套基于 PyTorch、支持自动微分的本地模拟器。无论你是想在免费真机上做第一次量子实验的学生，还是需要把算法快速迁移到多平台验证的研究者，都能用极低的成本上手。

```bash
pip install fieldqkit
```

---

## 一、为什么需要 fieldqkit？

过去几年，量子计算正从"少数物理比特的原理验证"走向"硬件、软件、编译、算法、应用协同发展"的系统工程阶段。国内的夸父、天衍、国盾、本源等量子云平台相继开放，超导真机的可用性越来越好。

但对真正想做实验的人来说，痛点也很现实：

- **平台割裂**——每个云平台一套 SDK、一套指令集、一套提交接口，换一个后端就要重写一遍代码；
- **国际工具水土不服**——Qiskit、Cirq 生态成熟，却接不进国内真机；TensorCircuit 模拟强大，但真机资源并不免费开放；
- **工程链路缺失**——从逻辑线路到物理芯片的编译（布局、路由、门分解、基础门翻译）、读取误差校准、零噪声外推（ZNE）这些"脏活累活"，往往要研究者自己从头搭；
- **模拟与真机割裂**——在模拟器上验证好的线路，提交到真机后因为拓扑约束、噪声、原生门集不同而失效。

fieldqkit 的设计目标，就是把这些环节用一套统一接口连起来：**让你把精力放在算法本身，而不是平台适配和工程胶水上。**

---

## 二、fieldqkit 能做什么？

### 1. 统一硬件访问：一个 client 通吃多平台

核心入口只有一个 `QuantumHardwareClient`。同一段代码，只要切换 `provider` 参数，就能在不同后端之间无缝迁移——从本地模拟器，到夸父 / 天衍 / 国盾 / 腾讯 / 本源 / 量坤云端模拟器。

```python
from fieldqkit import QuantumHardwareClient

client = QuantumHardwareClient()
result = client.run_auto(
    circuit="ghz",
    num_qubits=4,
    provider="simulator",        # 纯本地模拟，无需任何 token
    shots=8192,
    observables=["ZZII", "IIZZ"],
    return_probabilities=True,
)
print(result.observable_values)  # {'ZZII': 1.0, 'IIZZ': 1.0}
```

把 `provider` 换成 `"quafu"`、`"tianyan"`、`"origin"` 等，配置好对应平台 token，即可一键提交到真机。**模拟器上跑通的代码，无需改写就能上真机。**

### 2. 自动编译：逻辑线路 → 物理芯片

内置完整转译流水线（Transpiler）：比特布局（Layout）、SABRE 路由（Routing）、门分解（Decompose）、基础门翻译（Translate）、门压缩优化（GateCompressor）、动态解耦（Dynamical Decoupling）。你写的是逻辑线路，落到芯片上的是符合拓扑与原生门集约束的物理线路。

### 3. 误差缓解：让真机结果更可信

- **读取误差校准**（Readout Calibration）：构建混淆矩阵并对测量结果做反演校正；
- **零噪声外推**（ZNE，基于 CZ 三倍化）：在不增加比特数的前提下提升期望值精度。

### 4. 变分算法与量子机器学习：开箱即用

- **VQE**：含氢分子（H₂）势能面扫描完整示例，支持 parameter-shift 手动梯度下降；
- **QAOA**：MaxCut、自定义哈密顿量，并与 VQE 对比；
- **Shadow Tomography**：经典阴影高效估计大量可观测量；
- **QML / QNN**：PQC 监督分类（Iris 数据集）、Born Machine 分布学习、无监督量子分布学习。

### 5. 硬件校准：面向真机用户的进阶能力

读取校准、原生两比特随机基准测试（RB）、过程层析（Process Tomography）——这些通常要自己搭的标定流程，fieldqkit 已经封装成 manager。

### 6. 高效仿真：一套内置的 PyTorch 模拟器

不止状态矢量。fieldqkit 内置了**全谱模拟器**：State Vector、MPS（矩阵乘积态）、MPO（过程模拟）、密度矩阵（含噪）、Kraus 噪声信道、Clifford 稳定子、Clifford+T 分支模拟。基于 PyTorch，**原生支持自动微分和梯度计算**，可直接用于变分算法和量子机器学习的训练。

---

## 三、与国际主流软件包对标

fieldqkit 并不是要替代谁，而是补齐"国产真机 + 完整工程链路 + 误差缓解"这条被国际工具忽略的路径。

| 能力 | 国际/已有工具 | fieldqkit |
|---|---|---|
| 统一多平台真机访问 | Qiskit（IBM 为主）、Cirq（Google）—— 难接国内真机 | 夸父/天衍/国盾/腾讯/本源/量坤，一套 API |
| 张量网络模拟 | TensorCircuit（真机不免费开放） | 内置 MPS/MPO + 免费真机入口（夸父不限时） |
| 误差缓解 | Mitiq（独立库，需自行集成） | Readout + ZNE，与提交链路原生打通 |
| 自动微分模拟器 | PennyLane / TensorCircuit | 内置 PyTorch 模拟器，原生 autodiff |
| 变分算法 / QML | Qiskit、PennyLane | VQE / QAOA / Shadow / QML 开箱即用 |
| 国产平台编译 | quarkstudio / cqlib（无误差缓解） | 完整转译流水线 + 误差缓解 + 算法层 |

一句话：**Qiskit 的算法生态 + Mitiq 的误差缓解 + PennyLane 的可微分 + 国产真机的可达性，在 fieldqkit 里被整合进了同一套工作流。**

---

## 四、五分钟上手

fieldqkit 最大的特点是**零门槛体验**：本地模拟器无需任何 token，`pip install "fieldqkit[sim]"` 后即可运行上文示例。

想上真机也很简单——优先推荐**夸父量子云的免费资源（不限时）**用于体验和学习，配置好 token 即可一键提交。

仓库提供了 **12+ 个 Jupyter 教程 Notebook**，每个顶部都带 **Open in Colab** 徽章，点开即跑（首个单元格自动 `pip install`），覆盖从入门到 VQE/QAOA/Shadow/QML 的完整学习路径。

---

## 五、获取方式

- **GitHub 仓库**：https://github.com/FieldQuantum/fieldqkit
- **PyPI 安装**：`pip install fieldqkit` （https://pypi.org/project/fieldqkit/ ）
- **在线文档**：https://fieldquantum.github.io/fieldqkit/

fieldqkit 以 **Apache License 2.0** 开源，欢迎通过 GitHub Issues 反馈需求、提交 Pull Request。如果它对你的研究有帮助，也欢迎引用本项目。

> 让设计量子算法、跑真机实验这件事，回归它本该有的样子——**简单、统一、可复现。**
