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
- [VQE(H2)：demo_vqe_h2.ipynb](../examples/demo_vqe_h2.ipynb)
	- 对应： [VQERunner.run_model](./algorithms/vqe_runner.md)、[WSL Chemistry Workflow](./wsl_chemistry_workflow.md)
- [VQE(H2, 2Q taper)：demo_vqe_h2_2q.ipynb](../examples/demo_vqe_h2_2q.ipynb)
	- 对应： [VQERunner.run_model](./algorithms/vqe_runner.md)、[WSL Chemistry Workflow](./wsl_chemistry_workflow.md)
- [VQE(LiH, 6Q active-space)：demo_vqe_lih_6q.ipynb](../examples/demo_vqe_lih_6q.ipynb)
	- 对应： [VQERunner.run_model](./algorithms/vqe_runner.md)、[WSL Chemistry Workflow](./wsl_chemistry_workflow.md)
- [VQE(F2, 12Q active-space)：demo_vqe_f2_12q.ipynb](../examples/demo_vqe_f2_12q.ipynb)
	- 对应： [VQERunner.run_model](./algorithms/vqe_runner.md)、[WSL Chemistry Workflow](./wsl_chemistry_workflow.md)
- [Backend：demo_backend.ipynb](../examples/demo_backend.ipynb)
	- 对应： [Backend](./api/Backend.md)、[rank_chips](./api/rank_chips.md)

## 学习路径（入门 → 进阶 → 硬件 → 优化）

1. 入门： [全览入门：demo_full.ipynb](../examples/demo_full.ipynb)
2. 进阶： [线路与 core：demo_circuit_core.ipynb](../examples/demo_circuit_core.ipynb)
3. 硬件： [Readout + ZNE：demo_readout_zne.ipynb](../examples/demo_readout_zne.ipynb)
4. 优化：
	- [Shadow：demo_shadow.ipynb](../examples/demo_shadow.ipynb)
	- [VQE：demo_vqe.ipynb](../examples/demo_vqe.ipynb)
	- [VQE(H2)：demo_vqe_h2.ipynb](../examples/demo_vqe_h2.ipynb)
	- [VQE(H2, 2Q taper)：demo_vqe_h2_2q.ipynb](../examples/demo_vqe_h2_2q.ipynb)
	- [VQE(LiH, 6Q active-space)：demo_vqe_lih_6q.ipynb](../examples/demo_vqe_lih_6q.ipynb)
	- [VQE(F2, 12Q active-space)：demo_vqe_f2_12q.ipynb](../examples/demo_vqe_f2_12q.ipynb)
5. 拓扑补充： [Backend：demo_backend.ipynb](../examples/demo_backend.ipynb)

主 README 导航见 [../README.md](../README.md)。

## API

- [QuantumHardwareClient](./api/QuantumHardwareClient.md)
- [run_with_backend](./api/run_with_backend.md)
- [rank_chips](./api/rank_chips.md)
- [Backend](./api/Backend.md)
- [Task](./api/Task.md)

> 建议阅读顺序：`QuantumHardwareClient` → `rank_chips` → `Backend` → `Task`。

## Algorithms

- [ShadowTomography.run](./algorithms/shadow_tomography.md)
- [VQERunner.run_model](./algorithms/vqe_runner.md)
- [QAOARunner.run_model](./algorithms/qaoa_runner.md)

## Workflow

- [WSL Chemistry Workflow](./wsl_chemistry_workflow.md)

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
