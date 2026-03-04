# DOCUMENTATION.md 兼容性检查（2026-03-04）

## 结论

总体**兼容性良好**：`DOCUMENTATION.md` 描述的核心模块、入口类与主要函数在当前代码中均可定位，主流程与返回结构基本一致。

## 已核对通过（抽样）

- `QuantumHardwareClient` 与 `run_auto` 存在。
- `RunResult / ShadowResult / VQEResult / QAOAResult` 数据结构存在。
- `run_shadow / run_vqe / run_qaoa` 入口存在。
- `ReadoutCalibrationManager / NativeTwoQubitRBManager / NativeTwoQubitTomographyManager` 存在。
- `core.circuits / core.observables / core.readout / core.zne` 关键函数存在。

## 发现的差异（建议后续修订）

1. `run_shadow` 类型注解当前是 `observables: Optional[Sequence[str]]`，文档写为 `Optional[Sequence[str] | str]`。
   - 代码中仍兼容字符串输入（内部做了 `str -> [str]` 归一化），所以功能无冲突，但类型签名展示可更新。

2. `run_vqe` 与 `run_qaoa` 的文档参数列表略少于代码实际参数。
   - 代码额外支持：`zne`、`readout_mitigation`、`seed`、`callback`（以及部分默认细节）。
   - 建议在分页面 API 文档里补齐。

3. 文档是单文件结构，检索粒度较粗。
   - 不影响运行兼容性，但不利于函数级维护。

## 建议

- 保留 `DOCUMENTATION.md` 作为总览。
- 逐步迁移细节到 `docs/reference/*` 页面。
- 每次改动接口后优先更新对应函数页，再回写总览页摘要。
