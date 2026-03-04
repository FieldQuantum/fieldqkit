# NativeTwoQubitRBManager

## 概览

- **模块**：`quantum_hw.calibration.rb`
- **作用**：native 双比特门 RB，输出 coupler 粒度 fidelity 估计。

## 关键方法

### `calibrate_native_two_qubit_rb(...) -> Dict[str, Dict[str, object]]`

- `couplers=None` 时自动筛选正 fidelity 连线。
- 支持 readout mitigation。
- 支持按 coupler 缓存结果。

## 待补

- `lengths / num_sequences` 的统计意义
- `fit.fidelity` 的解释
- 缓存命中与过期规则
