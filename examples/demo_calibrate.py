import os

from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.calibration.readout import ReadoutCalibrationManager
from quantum_hw.sim.statevector import simulate_counts
from pathlib import Path


if __name__ == "__main__":
    chip_name = 'Baihua'
    
    token = "5gjq36bZsMvqFoSNomvnfPy4y[iDJWe[tBx9fIndISQ/:m{O5FEPyRkM4B{N{RkNyd{OypkJxiY[jxjJ4RkPyBkPxJEJ4FUMyBUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"
    client = QuantumHardwareClient(token=token)

    client.chip_name = chip_name
    client.chip_backend = Backend(chip_name)
    cache_dir=Path(__file__).resolve().parent.parent / "src/quantum_hw/api/.cache"

    calibration_manager = ReadoutCalibrationManager(
        cache_dir=cache_dir,
        transpile_with_backend=client._transpile_with_backend,
        submit_openqasm_async=client._submit_openqasm_async,
        wait_task=client._wait_task,
        get_task_result=client.tmgr.result,
        compact_for_sim=client._compact_for_sim,
        simulate_counts=simulate_counts,
    )

    result = calibration_manager.calibrate_readout(
        target_qubits=None,
        shots=1024,
        chip_name=chip_name,
        backend=client.chip_backend,
        qasm_version="2.0",
        print_true=True,
    )

    print("Calibrated qubits:", result.target_qubits)
    for q in result.target_qubits[:5]:
        print(f"Q{q} confusion matrix:", result.per_qubit_confusion[q])
