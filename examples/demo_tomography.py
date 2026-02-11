import os
from pathlib import Path

from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.calibration import NativeTwoQubitTomographyManager
from quantum_hw.sim.statevector import simulate_counts


if __name__ == "__main__":
    chip_name = "Simulator"

    client = QuantumHardwareClient()
    client.chip_name = chip_name
    client.chip_backend = Backend(chip_name)

    cache_dir = Path(__file__).resolve().parent.parent / "src/quantum_hw/api/.cache"

    tomo_manager = NativeTwoQubitTomographyManager(
        cache_dir=cache_dir,
        submit_openqasm_async=client._submit_openqasm_async,
        wait_task=client._wait_task,
        get_task_result=client.tmgr.result,
        compact_for_sim=client._compact_for_sim,
        simulate_counts=simulate_counts,
    )

    results = tomo_manager.calibrate_native_two_qubit_tomography(
        couplers=None,
        shots=256,
        chip_name=chip_name,
        backend=client.chip_backend,
        qasm_version="2.0",
        readout_mitigation=True,
        print_true=True,
    )

    for key, payload in results.items():
        choi = payload.get("choi_error")
        if choi is None:
            print(f"Coupler {key}: missing choi_error")
            continue
        print(f"Coupler {key}: choi_error shape={choi.shape}")
