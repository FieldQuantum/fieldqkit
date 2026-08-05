# 更新日志（Changelog）

本文件记录 fieldqkit 各版本的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.2] - 2026-08-05

本次发布新增第 7 家云平台接入（逻辑比特），让编译流程默认可复现，并提升了模拟器在
大比特可微分与长线路场景下的稳定性。

### 新增

- **逻辑比特（LogicalQubit）云平台接入**：作为第 7 家供应商接入
  <https://cloud.logicalqubit.com>，直连其 JSON REST API（不依赖厂商 `lqcloud` SDK）。
  线路→IR 转换将单比特门归一为本征 `su2`、两比特门为 `cz`；指令中的比特下标使用**物理**
  索引（服务端按物理索引解读并据此校验耦合器），`initial_layout` 列出实际用到的物理比特；
  `measure2` 类后端在测量前插入 `x21`。`chip_info` 透传真实的耦合器 `cz_fidelity` 与比特
  `t1` / `t2`，使编译器的保真度感知布局在该平台生效；并从厂商 `measure_f0` / `measure_f1`
  预热读出缓存，跳过实机读出校准。提交侧带幂等键、大请求体 gzip 压缩与
  `ChunkedEncodingError` 重试。新增凭证项 `LOGICALQUBIT_API_TOKEN`。
- **原生线路提交通道**：`TaskAdapter.native_ir` + `NativeCircuitSubmitRequest` +
  `client._submit_native_async`，使 `QuantumCircuit` 对象可不经 OpenQASM / QCIS
  字符串序列化直接抵达适配器。
- **大比特可微分模拟的梯度检查点**：态矢量与密度矩阵模拟器在比特数达到阈值
  （`grad_checkpoint_threshold_qubits`，默认 20）且存在需要梯度的参数时，对门循环启用
  `torch.utils.checkpoint`，反向传播中间量由 O(n_gates) 降至 O(√n_gates)。
  采样 / 推理路径不受影响。阈值可通过 `set_sim_config()` 调整。
- 态矢量模拟器新增 `apply_pauli_string()`。
- `Transpiler.run()`、`SabreRouting`、`Layout` 新增 `seed` 参数。默认配置本身已确定，
  该参数用于在显式开启 `routing_n_trials > 1`、`routing_initial_mapping="random"`
  或 `routing_random_choice=True` 时恢复可复现性。
- `QuantumHardwareClient._transpile_with_backend()` 新增 `routing_initial_mapping`、
  `routing_random_choice`、`niter`、`seed` 四个参数，以及 `**transpiler_kwargs` 兜底透传。
  此前走高层 API 的调用方无法关闭路由随机性。
- `run_vqe_with_backend`、`run_qaoa_with_backend`、`run_pqc_classifier`、
  `run_qnn_unsupervised`、`run_qnn_conditional`、`build_compression_transform`
  新增 `transpile_options: dict` 参数，内容原样转发给 `_transpile_with_backend`。

### 修复

- **MPS 模拟器在长线路 / 强截断下崩溃**：
  - `ComplexSVD.forward` 在默认 LAPACK `gesdd` 驱动硬失败（`torch._C._LinAlgError`，
    常见于 CPU 上被截断的病态 / 重奇异值 MPS 键）时，回退到
    `scipy.linalg.svd(lapack_driver="gesvd")`。
  - `simulate_mps` 每 1000 个门对正则中心张量重新归一化。截断 SVD 不保范数，超长线路上
    振幅会漂移到非有限值并使下一次 SVD 崩溃。开销 O(χ²)，不影响采样概率。
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
- CUDA 空闲显存查询改用 `pynvml`（原 `torch.cuda.mem_get_info`），与设备利用率查询保持一致；
  `pynvml` 不可用时自动降级。
- 移除 VQE 的 `test_empty_hamiltonian_autograd_raises` 边界测试（空哈密顿量在 autograd
  路径下的报错行为不作为契约保证）。

### 修订

- 继续完善第三方代码署名：在 `api/backend.py`、`circuit/qasm2.py`、`circuit/render.py`
  中内嵌上游 quarkcircuit 完整 MIT 许可文本并标注 `SPDX-License-Identifier:
  Apache-2.0 AND MIT`，同步更新 `THIRD_PARTY_NOTICES`。
- `CITATION.cff` 与 `pyproject.toml` 的作者署名统一为 Yuchen Guo。

### 升级提示

若需保持 0.1.1 的旧编译行为，显式传入原默认值即可：

```python
transpiler.run(qc, routing_initial_mapping="random", routing_random_choice=True)
```

做 ZNE 等误差缓解实验时，建议**编译一次**后再折叠（见 `apply_zne_cz_tripling`），
不要对每个噪声缩放因子分别编译——否则各缩放因子会落在不同的物理比特布局上。

大比特（≥20）可微分模拟现在默认启用梯度检查点，以额外的前向重算换取反向传播显存下降。
若显存不是瓶颈、更在意速度，可关闭：

```python
from fieldqkit.sim import set_sim_config
set_sim_config(grad_checkpoint_threshold_qubits=None)
```

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
