# 更新日志（Changelog）

本文件记录 fieldqkit 各版本的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
