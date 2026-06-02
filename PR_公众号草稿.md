# 量坤科技开源 fieldqkit：一行 API，打通量子算法到国产真机的“最后一公里”

> 用 Qiskit 研究量子信息，却接入不了 IBM 真机？
> 用 TensorCircuit 做张量网络模拟，腾讯真机却不免费开放？
> 用 quarkstudio / cqlib 接入量子院、天衍平台，却没有误差缓解？
> 想设计一套量子算法去真机上验证，结果发现编译、提交、读取、误差缓解全得自己手写？
> 模拟器上跑通的线路，一搬到真机就面目全非？

如果你在工作中遇到过以上任何一点，那么 fieldqkit 也许正是你在找的那块拼图。

近日，量坤科技正式开源了 fieldqkit，一个面向用户的**量子硬件控制接口**。它用一套统一 API 屏蔽国内主流量子云平台的差异，把"量子算法 → 线路构建 → 自动编译 → 真机提交 → 误差缓解"的完整工作流封装在一起，并内置了基于 PyTorch、支持自动微分的本地模拟器。无论你是想在免费真机上做第一次量子实验的学生，还是需要把算法快速迁移到多平台验证的研究者，都能用很低的成本快速上手。

```bash
pip install fieldqkit
```

---

## 一、为什么需要 fieldqkit？

过去几年，量子计算逐渐走向硬件、软件、算法、应用协同发展的系统工程阶段。国内的国盾、本源、夸父、天衍等量子云平台相继开放，超导真机的可用性越来越好。

但对真正想做量子计算实验的人来说，仍然有不少现实困难：

- **平台割裂**：每个云平台一套 SDK、一套指令集、一套提交接口，换一个后端就要重写一遍代码；
- **国际工具水土不服**：Qiskit、Cirq 生态成熟，却接不进国内真机；
- **工程链路缺失**：从逻辑线路到物理芯片的编译（布局、路由、门分解、基础门翻译），还有误差缓解，这些“脏活累活”往往要研究者自己从头搭；
- **模拟与真机割裂**：模拟器上验证好的线路，提交到真机后常因拓扑约束、噪声、原生门集不同而失效。

fieldqkit 想做的，就是把这些环节用一套统一接口封装起来，让你把精力花在算法本身，而把平台适配交给框架。

---

## 二、fieldqkit 能做什么？

### 1. 统一硬件访问：一个 client 适配多平台

核心入口只有一个，同一段代码，切换 `provider` 参数就能在不同后端之间迁移，从本地模拟器，到夸父、天衍、国盾、腾讯、本源、量坤云端模拟器。

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

把 `provider` 换成 `"quafu"`、`"tianyan"`、`"origin"` 等，配置好对应平台 token，即可一键提交到真机。模拟器上跑通的代码，无需改写就能上真机。

### 2. 自动编译：逻辑线路到物理芯片

内置完整编译流水线：比特布局、路由、门分解、门压缩优化，输出符合拓扑与原生门集约束的物理线路。

### 3. 误差缓解：让真机结果更可信

- **读取误差校准**（Readout error mitigation）：构建混淆矩阵，对测量结果做反演校正；
- **零噪声外推**（Zero-noise extrapolation）：放大线路噪声强度，再外推回零噪声极限；
- **Clifford 拟合**（Clifford fitting）：用同结构的 Clifford 线路做拟合，推断噪声对结果的影响；
- **动态解耦**（Dynamical decoupling）：插入翻转脉冲，抵消比特空转期间累积的退相干。

### 4. 量子算法：开箱即用

- **VQE**：求解量子多体模型与量子化学问题；
- **QAOA**：变分线路求解组合优化问题；
- **Shadow Tomography**：经典阴影方法同时估计大量可观测量；
- **QML / QNN**：监督分类、非监督采样。

### 5. 硬件校准：面向真机用户的进阶能力

读取误差校准、两比特门随机基准测试、过程层析，这些通常要自己搭的标定流程，fieldqkit 都封装成了 manager。

### 6. 高效仿真：一套内置的 PyTorch 模拟器

fieldqkit 内置了**全套模拟器**：态矢量、密度矩阵、Clifford 线路、张量网络（MPS）。底层基于 PyTorch，原生支持自动微分和梯度计算，可直接用于变分算法和量子机器学习的训练。

---

## 三、与国际主流软件包对标

| 能力 | 国际/已有工具 | fieldqkit |
|---|---|---|
| 统一多平台真机访问 | Qiskit（IBM 为主）、Cirq（Google），难接国内真机 | 夸父/天衍/国盾/腾讯/本源/量坤，一套 API |
| 张量网络模拟 | TensorCircuit（真机不免费开放） | 内置张量网络引擎，配合免费真机入口（夸父不限时） |
| 误差缓解 | Mitiq（独立库，需自行集成） | 与提交链路原生打通 |
| 自动微分模拟器 | PennyLane / TensorCircuit | 内置 PyTorch 模拟器，原生支持自动微分 |
| 变分算法 / QML | Qiskit、PennyLane | VQE / QAOA / Shadow / QML 开箱即用 |
| 国产平台编译 | quarkstudio / cqlib（仅有真机调度） | 完整编译流水线，外加误差缓解与算法层 |

Qiskit 的算法生态、Mitiq 的误差缓解、PennyLane 的可微分、国产真机的可达性，在 fieldqkit 里被整合进了同一套工作流。

---

## 四、五分钟上手

fieldqkit 上手门槛很低。本地模拟器无需任何 token，`pip install "fieldqkit[sim]"` 之后就能运行上文的示例。

想上真机也不复杂，配置好 token 即可一键提交。

仓库提供了 **12 个 Jupyter 教程 Notebook**，既能本地运行，也支持在 Colab 中点开即跑，覆盖从入门到 VQE、QAOA、Shadow、QML 的完整学习路径。

---

## 五、获取方式

- **GitHub 仓库**：https://github.com/FieldQuantum/fieldqkit
- **PyPI 安装**：`pip install fieldqkit` （https://pypi.org/project/fieldqkit/ ）
- **在线文档**：https://fieldquantum.github.io/fieldqkit/

fieldqkit 以 Apache License 2.0 开源，欢迎通过 GitHub Issues 反馈需求、提交 Pull Request。如果它对你的研究有帮助，也欢迎引用本项目。

> 我们希望，设计量子算法和跑真机实验能变得更简单、更统一、更可复现。
