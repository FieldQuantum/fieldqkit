# 更新日志（Changelog）

本文件记录 fieldqkit 各版本的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.2] - 2026-08-05

本次发布让编译流程默认可复现。此前同一条线路重复编译可能得到差异极大的结果——实测一条
零 SWAP 需求的 10 比特级联 GHZ，两比特门数在 9 到 40 之间波动，中位数 30。

### 修复

- **编译不再污染全局随机数状态**：`SabreRouting` 与 `Layout` 改用实例私有的
  `random.Random` / `numpy.random.Generator`，不再调用模块级 `random.sample`、
  `random.choice`、`np.random.choice`。此前编译会消耗全局 RNG 熵，导致用户自己
  `random.seed(...)` 之后的随机序列被编译过程静默扰乱（消耗量取决于线路结构和
  插入的 SWAP 次数），使得"设了种子的实验"实际不可复现。

### 变更

- **`Transpiler.run()` 的路由默认值改为确定性**：
  `routing_initial_mapping` 由 `"random"` 改为 `"trivial"`，
  `routing_random_choice` 由 `True` 改为 `False`。
  这两个默认值此前与 `SabreRouting.__init__` 自身的默认值（`"trivial"` / `False`）相互矛盾。
  实测在结构化线路上新默认值同时更优也更稳（10 比特星型 GHZ：27–44 → 稳定 16；
  硬件高效 ansatz：27–30 → 稳定 27）；仅在逻辑交互图接近全连接的稠密线路上
  可能多 2%–9% 的两比特门，此类线路可按需开启 `routing_n_trials` 找回。
  `"random"` 策略仍然保留可用。

### 新增

- `Transpiler.run()`、`SabreRouting`、`Layout` 新增 `seed` 参数。默认配置本身已确定，
  该参数用于在显式开启 `routing_n_trials > 1`、`routing_initial_mapping="random"`
  或 `routing_random_choice=True` 时恢复可复现性。
- `QuantumHardwareClient._transpile_with_backend()` 新增 `routing_initial_mapping`、
  `routing_random_choice`、`niter`、`seed` 四个参数，以及 `**transpiler_kwargs` 兜底透传。
  此前走高层 API 的调用方无法关闭路由随机性。
- `run_vqe_with_backend`、`run_qaoa_with_backend`、`run_pqc_classifier`、
  `run_qnn_unsupervised`、`run_qnn_conditional`、`build_compression_transform`
  新增 `transpile_options: dict` 参数，内容原样转发给 `_transpile_with_backend`。

### 升级提示

若需保持 0.1.1 的旧编译行为，显式传入原默认值即可：

```python
transpiler.run(qc, routing_initial_mapping="random", routing_random_choice=True)
```

做 ZNE 等误差缓解实验时，建议**编译一次**后再折叠（见 `apply_zne_cz_tripling`），
不要对每个噪声缩放因子分别编译——否则各缩放因子会落在不同的物理比特布局上。

## [0.1.1] - 2026-06-05

### 修订

- 修正第三方代码的署名与许可声明：在各改编文件中内嵌上游完整许可文本（quarkcircuit / quarkstudio 原为 MIT，SPDX 标注 `Apache-2.0 AND MIT`；cqlib / TensorCircuit 为 Apache-2.0），并更正 `THIRD_PARTY_NOTICES`、`NOTICE` 中的作者、年份与项目归属。

## [0.1.0] - 2026-06-04

首次公开发布。

### 新增

- **统一硬件接口**：`QuantumHardwareClient` 一套 API 接入多家量子云平台（夸父 / 天衍 / 国盾 / 腾讯 / 本源 / 量坤）及本地模拟器，支持自动选择后端。
- **电路构建**：`QuantumCircuit`，支持 OpenQASM 2.0 与 QCIS 的导入 / 导出。
- **编译与转译**：自动转译流水线，含布局（layout）、SABRE 路由、门分解、优化、调度等 pass。
- **误差缓解**：读出误差缓解（readout mitigation）、零噪声外推（ZNE）与Clifford 拟合。
- **变分算法**：VQE、QAOA、Shadow Tomography、量子机器学习（PQC 监督分类与无监督 QNN 分布学习）。
- **硬件校准**：读出校准、原生两比特随机基准测试（RB）、过程层析。
- **高效仿真**：基于 PyTorch 的全态矢量、密度矩阵、MPS、MPO 模拟器，支持自动微分（autodiff）梯度计算。
- **命令行工具**：`fieldqkit-config-init` 用于生成凭证配置模板。
- 中文文档站点与多份示例 Notebook（VQE / QAOA / QML / Shadow / 噪声仿真 / 读出与 ZNE 等）。
