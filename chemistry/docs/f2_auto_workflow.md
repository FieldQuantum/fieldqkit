# F2 自动化计算流程说明

本文说明当前仓库中 F2 分子自动化计算流程的组织方式。目标是把构成一条清晰的三步链路：

1. 从分子结构出发，构造电子哈密顿量，并自动选择满足精度要求的 active space。
2. 在给定 active space 上生成一组经过筛选和排序的 UCC 激发项。
3. 读取前两步的结果，截取前 K 个激发项构造变分线路，随后运行 VQE。

这样做的直接好处是：化学计算、ansatz 生成和 VQE 求解被明确分层，每一层都有独立产物，可以单独检查，也可以串起来批量运行。

## 问题背景

对于 F2 这样的分子，如果直接在全空间上做精确对角化，代价会随着轨道数快速上升。更常见的做法是先在分子轨道基下写出电子哈密顿量，再冻结明显不参与相关效应的轨道，只把最重要的一部分轨道保留在 active space 中。

在二次量子化表述下，电子哈密顿量可以写成

$$
H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s.
$$

其中 $a_p^\dagger$ 和 $a_p$ 分别是费米子的产生与湮灭算符，$h_{pq}$ 和 $h_{pqrs}$ 来自单电子积分和双电子积分。经过 Jordan-Wigner、Bravyi-Kitaev 或 SCBK 等映射后，这个费米子哈密顿量会被转换成量子比特哈密顿量：

$$
H_q = c_0 I + \sum_i c_i P_i,
$$

其中 $P_i$ 是 Pauli 串，$c_i$ 是实系数。VQE 实际优化的就是这个量子比特哈密顿量的期望值。

## 第一步：自动导出 F2 哈密顿量

第一步使用 [export_f2_terms_wsl_auto.py](../scripts/export_f2_terms_wsl_auto.py) 完成。这个脚本一般在带有 PySCF 和 OpenFermion 的环境中运行，当前建议放在 WSL 中执行。

### 这一步在做什么

脚本从给定的核间距 $R$、基组、分子电荷和多重度出发，先做 RHF 和 FCI 计算，然后利用 MP2 近似得到各个分子轨道在 canonical MO 基下的占据数。这里占据数不是装饰信息，而是 active space 自动选择的依据。

脚本里使用的一个简单指标是

$$
s_i = \min(n_i, 2 - n_i),
$$

其中 $n_i$ 是第 $i$ 个空间轨道的近似占据数。这个量的含义很直接：

1. 如果一个轨道几乎满占据，即 $n_i \approx 2$，那么它更像“冻结占据轨道”。
2. 如果一个轨道几乎空轨道，即 $n_i \approx 0$，那么它通常可以先排除。
3. 如果一个轨道偏离 0 或 2 较多，往往说明它参与了相关效应，更值得放进 active space。

随后脚本会枚举一组候选 active space，并逐个计算其基态能量 $E_{\text{active}}$，再与全体系 FCI 能量 $E_{\text{FCI}}$ 比较。筛选条件是

$$
|E_{\text{active}} - E_{\text{FCI}}| \le \varepsilon,
$$

其中 $\varepsilon$ 是设定的 chemical accuracy 阈值。满足阈值的候选里，脚本优先选择量子比特数最少的那个，这就是 auto_minq 模式的含义。

### 这一步的输出

输出文件是一个 JSON，例如 [f2_R2.6_angstrom_sto-3g_auto.json](../data/f2_R2.6_angstrom_sto-3g_auto.json)。里面至少包含以下几类信息：

1. 常数项和 Pauli 项，也就是 VQE 真正要测量的量子比特哈密顿量。
2. FCI 参考能量，用来衡量后续 VQE 的误差。
3. 自动选出的 occupied_indices 和 active_indices。
4. 候选搜索记录 candidate_search_results，用来回看为什么最后选中了当前 active space。
5. 轨道对称性、轨道能量和近似占据数等辅助信息。

可以把这一步理解为：输入分子参数，输出“已经裁剪好的量子问题”。

## 第二步：生成排序后的 UCC 激发项

第二步使用 [f2_ucc_generator.py](../scripts/f2_ucc_generator.py)。它读取第一步生成的 JSON，不再重新做化学积分，而是在已经确定的 active space 上构造 ansatz 的候选项。

### 这一步在做什么

UCC 的基本思想是从 Hartree-Fock 参考态 $|\Phi_0\rangle$ 出发，施加一个反厄米激发生成元：

$$
|\Psi(\theta)\rangle = e^{T(\theta) - T^\dagger(\theta)} |\Phi_0\rangle.
$$

如果只保留单激发和双激发，那么

$$
T = \sum_{ia} \theta_i^a a_a^\dagger a_i + \sum_{ijab} \theta_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i.
$$

这里 $i,j$ 表示占据轨道，$a,b$ 表示虚轨道。理论上可选的激发很多，但并不是每一项都值得留下。当前脚本做了三层筛选：

1. 保持电子数守恒。
2. 保持自旋量子数不变，只考虑自旋守恒激发。
3. 利用空间对称性筛掉不满足 irrep 选择规则的激发项。

在筛完之后，脚本还会对每个激发项给一个重要性分数：只在 $|\Phi_0\rangle$ 和 $T_i|\Phi_0\rangle$ 张成的二维子空间内比较能量下降，记作

$$
\Delta E_i = E_0 - E_i.
$$

这里 $E_0$ 是参考态能量，$E_i$ 是对应二维子空间中的最低本征值。这个量越大，说明该激发对降低能量越重要，于是排序越靠前。

### 为什么要做排序

这一层排序的意义非常实际。VQE 并不总是需要“把所有允许的激发都放进去”。对硬件代价、优化稳定性和参数数量都比较敏感的场景，更有价值的做法往往是：

1. 先保留最重要的少量激发项。
2. 看能量误差是否已经足够小。
3. 如果不够，再逐步增加激发项数量。

因此第二步的输出不是最终电路，而是一份已经排好序的候选模板。

### 这一步的输出

更合适的输出文件是 [ucc_f2.json](../data/ucc_f2.json)。这个文件本质上是一份排好序的 ansatz 模板，包含两部分核心信息：

1. Hartree-Fock 初态占据模式，也就是哪些量子比特要先打到 $|1\rangle$。
2. 一组按重要性排序的 pauli_evolution 项。

之所以更推荐 JSON，而不是直接输出 Python，有两个原因：

1. JSON 是纯数据格式，语义更清楚，第二步输出的是“模板数据”，不是“待执行代码”。
2. 第三步读取时不需要再解析 Python 源码，后续如果做批处理、可视化、统计分析也更方便。

## 第三步：按 Top-K 构造 ansatz 并运行 VQE

第三步使用 [run_f2_auto_vqe.py](../scripts/run_f2_auto_vqe.py)。这一层不再依赖 notebook，而是直接读取前两步的产物，拼出 custom ansatz，然后运行 VQE。

### 这一步在做什么

脚本会先从第一步 JSON 中读取量子比特哈密顿量

$$
H_q = c_0 I + \sum_i c_i P_i,
$$

再从第二步生成的 [ucc_f2.json](../data/ucc_f2.json) 中读出已经排序好的 Pauli 演化项，截取前 $K$ 项构造变分线路：

$$
U_K(\theta) = \prod_{m=1}^{K} e^{-i \theta_m P_m}.
$$

随后用这个线路去最小化目标函数

$$
E(\theta) = \langle \Phi_0 | U_K^\dagger(\theta) H_q U_K(\theta) | \Phi_0 \rangle.
$$

当优化结束后，再把量子部分的最优能量加上常数项，得到总能量，并与 FCI 参考值比较。

### Top-K 参数的含义

Top-K 不是一个附加功能，而是这套流程里很重要的实验参数。它可以回答一个很具体的问题：

“为了达到某个精度，究竟需要多少个激发项？”

在数值实验里，这通常意味着以下几种比较：

1. 比较不同 $K$ 下的误差 $|E_{\text{VQE}} - E_{\text{FCI}}|$。
2. 比较不同 $K$ 下的参数数量和优化步数。
3. 观察在给定硬件噪声水平下，线路变深是否真的带来收益。

这也是第三步从 notebook 改写成脚本之后最有价值的地方，因为参数扫描和批处理会容易很多。

## 一键串联三步的总控脚本

如果不想分别调用三条命令，现在可以直接使用 [run_f2_auto_pipeline.py](../scripts/run_f2_auto_pipeline.py)。这个脚本会按顺序执行：

1. 调用第一步导出脚本，生成 Hamiltonian JSON。
2. 调用第二步生成脚本，生成排序后的 UCC 模板。
3. 调用第三步 VQE 脚本，按给定 Top-K 跑出最终结果。

需要说明的是，这个总控脚本本身不重新实现化学计算和 VQE，而是负责把已有三个步骤串起来。因此它适合两种场景：

1. 在一个已经装好全部依赖的环境里一口气跑完。
2. 作为批处理入口，统一管理文件命名和参数传递。

## 典型命令

### 分步执行

第一步：

```bash
python chemistry/scripts/export_f2_terms_wsl_auto.py \
  --R 2.6 \
  --unit angstrom \
  --reduction auto_minq \
  --encoding jw \
  --output chemistry/data/f2_R2.6_angstrom_sto-3g_auto.json
```

第二步：

```bash
python chemistry/scripts/f2_ucc_generator.py \
  --payload chemistry/data/f2_R2.6_angstrom_sto-3g_auto.json \
  --output chemistry/data/ucc_f2.json \
  --include-singles \
  --include-doubles
```

第三步：

```bash
python chemistry/scripts/run_f2_auto_vqe.py \
  --ham-json chemistry/data/f2_R2.6_angstrom_sto-3g_auto.json \
  --ucc-file chemistry/data/ucc_f2.json \
  --ucc-topk 5 \
  --importance-cutoff 1e-4 \
  --prefer-chips Simulator \
  --save-result chemistry/data/f2_R2.6_topk5_vqe_result.json
```

### 一键执行三步

```bash
python chemistry/scripts/run_f2_auto_pipeline.py \
  --R 2.6 \
  --unit angstrom \
  --encoding jw \
  --ucc-topk 5 \
  --importance-cutoff 1e-4 \
  --include-doubles \
  --prefer-chips Simulator
```
