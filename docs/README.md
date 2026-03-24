# API 参考总览

## 教程入口（Notebook）

- [全览入门：demo_full.ipynb](../examples/demo_full.ipynb)
	- 对应： [QuantumHardwareClient](./api/QuantumHardwareClient.md)、[readout](./core/readout.md)、[zne](./core/zne.md)
- [线路与 core：demo_circuit_core.ipynb](../examples/demo_circuit_core.ipynb)
	- 对应： [circuits builders](./core/circuits.md)、[observables](./core/observables.md)、[statevector simulator](./sim/statevector.md)
- [Shadow：demo_shadow.ipynb](../examples/demo_shadow.ipynb)
	- 对应： [ShadowTomography.run](./algorithms/shadow_tomography.md)、[observables](./core/observables.md)
- [Readout + ZNE：demo_readout_zne.ipynb](../examples/demo_readout_zne.ipynb)
	- 对应： [ReadoutCalibrationManager](./calibration/ReadoutCalibrationManager.md)、[readout](./core/readout.md)、[zne](./core/zne.md)
- [VQE：demo_vqe.ipynb](../examples/demo_vqe.ipynb)
	- 对应： [VQERunner.run_model](./algorithms/vqe_runner.md)、[observables](./core/observables.md)
- [Backend：demo_backend.ipynb](../examples/demo_backend.ipynb)
	- 对应： [hardware_discovery](./api/hardware_discovery.md)、[Backend](./api/Backend.md)

## 学习路径（入门 → 进阶 → 硬件 → 优化）

1. 入门： [全览入门：demo_full.ipynb](../examples/demo_full.ipynb)
2. 进阶： [线路与 core：demo_circuit_core.ipynb](../examples/demo_circuit_core.ipynb)
3. 硬件： [Readout + ZNE：demo_readout_zne.ipynb](../examples/demo_readout_zne.ipynb)
4. 优化：
	- [Shadow：demo_shadow.ipynb](../examples/demo_shadow.ipynb)
	- [VQE：demo_vqe.ipynb](../examples/demo_vqe.ipynb)
5. 拓扑补充： [Backend：demo_backend.ipynb](../examples/demo_backend.ipynb)

主 README 导航见 [../README.md](../README.md)。

## API

- [api module reference](./api/README.md)
- [QuantumHardwareClient](./api/QuantumHardwareClient.md)
- [run_with_backend](./api/run_with_backend.md)
- [hardware_discovery](./api/hardware_discovery.md)
- [Backend](./api/Backend.md)
- [Task](./api/Task.md)
- [provider runtime](./api/provider_runtime.md)
- [provider adapters](./api/providers.md)
- [shared cqlib layer](./api/cqlib.md)
- [platform credentials](./api/platform_credentials.md)

> 建议阅读顺序：`QuantumHardwareClient` → `run_with_backend` → `hardware_discovery` → `Backend` → `Task` → `provider_runtime` → `providers` → `cqlib` → `platform_credentials`。

## Algorithms

- [ShadowTomography.run](./algorithms/shadow_tomography.md)
- [VQERunner.run_model](./algorithms/vqe_runner.md)
- [circuit compression](./algorithms/circuit_compression.md)
- [ansatz templates](./algorithms/ansatz_templates.md)

## Calibration

- [ReadoutCalibrationManager](./calibration/ReadoutCalibrationManager.md)
- [build_confusion_matrix](./calibration/build_confusion_matrix.md)
- [NativeTwoQubitRBManager](./calibration/NativeTwoQubitRBManager.md)
- [NativeTwoQubitTomographyManager](./calibration/NativeTwoQubitTomographyManager.md)

## Core

- [circuits builders](./core/circuits.md)
- [observables](./core/observables.md)
- [utils](./core/utils.md)
- [readout](./core/readout.md)
- [zne](./core/zne.md)
- [result types](./core/result_types.md)

建议阅读顺序：`circuits` → `observables` → `utils` → `readout` → `zne` → `result_types`。

## Circuit

- [circuit module reference](./circuit/README.md)
- [QuantumCircuit](./circuit/quantumcircuit.md)
- [OpenQASM 解析](./circuit/qasm.md)
- [helpers 与渲染](./circuit/helpers_render.md)
- [matrix 与 utils](./circuit/matrix_utils.md)

## Sim

- [statevector simulator](./sim/statevector.md)
- [matrix utilities](./sim/matrix.md)
- [mps simulator](./sim/mps.md)
- [mpo process simulator](./sim/mpo.md)
- [simulator interface](./sim/interface.md)
- [simulator common helpers](./sim/common.md)
