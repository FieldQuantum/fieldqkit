# readout

## 模块

- `quantum_hw.core.readout`
- `quantum_hw.core.utils`（包含 `expectation_from_probabilities`）

## 关键函数

- `build_local_confusion_matrix`
- `mitigate_readout`
- `expectation_from_samples_unbiased`
- `expectation_from_probabilities`（定义在 `core.utils`）

## 待补

- 数值稳定性与非负/归一化处理
- 大 support 下无偏估计路径说明
- 与 `ReadoutCalibrationManager` 的配套示例
