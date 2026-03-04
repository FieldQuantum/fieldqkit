# run_shadow

## 概览

- **模块**：`quantum_hw.algorithms.shadow`
- **作用**：执行 classical shadow tomography，并估计 observables。

## 关键参数（待补全）

- `shots`
- `shots_per_basis`
- `observables`
- `zne`
- `estimator` / `mom_groups`
- `target_qubits` / `prefer_chips` / `rank_weights`

## 返回

- `ShadowResult`
  - `observable_estimates`
  - `observable_stderr`
  - `basis_patterns`

## 待补

- 估计器选择建议（`mean` vs `mom`）
- `zne=True` 时 raw 与 extrapolated 字段解释
