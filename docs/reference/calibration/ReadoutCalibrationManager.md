# ReadoutCalibrationManager

## 概览

- **模块**：`quantum_hw.calibration.readout`
- **作用**：执行读出误差校准并按芯片缓存。

## 关键方法

### `calibrate_readout(...) -> CalibrationResult`

- 输入目标比特（可空），自动补全可用比特。
- 支持硬件路径与 Simulator 路径。
- 按比特时间戳做增量刷新。

## 待补

- 缓存文件结构说明
- `qasm_version` 行为
- 与 `run_auto(readout_mitigation=True)` 的联动关系
