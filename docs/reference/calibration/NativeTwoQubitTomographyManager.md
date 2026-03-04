# NativeTwoQubitTomographyManager

## 概览

- **模块**：`quantum_hw.calibration.tomography`
- **作用**：native 双比特门 process tomography，输出误差通道（Choi）信息。

## 关键方法

### `calibrate_native_two_qubit_tomography(...) -> Dict[str, Dict[str, object]]`

- 支持 coupler 自动选择。
- 支持 readout mitigation。
- 按 coupler 缓存 error channel。

## 待补

- 输入态与测量基组合规模
- PTM / Choi 的转换细节
- 返回字段（实部/虚部）约定
