# Algorithms 模块参考

- 模块路径：`fieldqkit.algorithms`
- 模块定位：在统一硬件 API 之上提供变分算法（VQE / QAOA）、经典阴影层析（Shadow）、量子机器学习（QML），以及它们共享的优化工具与线路压缩能力。

## 两层设计

所有变分类算法都遵循「高层 Runner + 低层 `*_with_backend`」两层结构：

- **高层 Runner**（`VQERunner` / `QAOARunner` / `QMLRunner` / `ShadowTomography`）：自动解析 provider → 候选芯片 → 后端，逐块尝试并在全部失败时统一抛错。
- **低层 `run_*_with_backend`**：在已给定 `Backend` / `chip_name` 的前提下执行单次优化/采样，供算法层与高级用户复用。
- **共享底座** `optimizer_utils`：能量评估、parameter-shift 梯度、Adam、Clifford fitting、通用优化循环 `run_variational_loop`。

## 页面导航

- [VQERunner.run_model](./vqe_runner.md) —— VQE：Ising/Heisenberg/XXZ/XY/自定义哈密顿量
- [QAOARunner.run_model](./qaoa_runner.md) —— QAOA：MaxCut + 自定义代价项
- [ShadowTomography.run](./shadow_tomography.md) —— classical shadow 协议
- [QML](./qml.md) —— PQC 监督分类 + 无监督 / 条件 QNN（底层函数）
- [QMLRunner](./qml_runner.md) —— QML 的高层入口（自动硬件选择）
- [optimizer_utils](./optimizer_utils.md) —— 变分算法共享工具
- [ansatz templates](./ansatz_templates.md) —— hardware-efficient ansatz 构造
- [qml_encoding](./qml_encoding.md) —— Angle / IQP 数据编码线路
- [circuit compression](./circuit_compression.md) —— MPS/MPO 混合后缀压缩

## 梯度方式速查

| 算法 | `autograd` 允许的后端 | `parameter-shift` |
|---|---|---|
| VQE / QAOA | 本地 `Simulator` **或** 云端 `fieldquantum_sim`（服务端梯度） | 所有 provider |
| QML | **仅**本地 `Simulator` | 所有 provider |

> QML 的 `autograd` 在云端模拟器 / 真机上会抛 `ValueError`，请改用 `parameter-shift`。

## 相关页面

- [api: run_with_backend](../api/run_with_backend.md)
- [core: result types](../core/result_types.md)
- [sim: simulator interface](../sim/interface.md)
