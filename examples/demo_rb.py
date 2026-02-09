import os
from pathlib import Path
import numpy as np
from quantum_hw import QuantumHardwareClient
from quantum_hw.api.backend import Backend
from quantum_hw.calibration import NativeTwoQubitRBManager
from quantum_hw.sim.statevector import simulate_counts


if __name__ == "__main__":
    chip_name = 'Baihua'
    
    token = "5gjq36bZsMvqFoSNomvnfPy4y[iDJWe[tBx9fIndISQ/:m{O5FEPyRkM4B{N{RkNyd{OypkJxiY[jxjJ4RkPyBkPxJEJ4FUMyBUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"
    client = QuantumHardwareClient(token=token)

    client = QuantumHardwareClient(token=token)
    client.chip_name = chip_name
    client.chip_backend = Backend(chip_name)

    cache_dir = Path(__file__).resolve().parent.parent / "src/quantum_hw/api/.cache"

    rb_manager = NativeTwoQubitRBManager(
        cache_dir=cache_dir,
        transpile_with_backend=client._transpile_with_backend,
        submit_openqasm_async=client._submit_openqasm_async,
        wait_task=client._wait_task,
        get_task_result=client.tmgr.result,
        compact_for_sim=client._compact_for_sim,
        simulate_counts=simulate_counts,
    )

    results = rb_manager.calibrate_native_two_qubit_rb(
        couplers=[[39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45]],
        lengths=[1, 2, 3, 4, 5],
        num_sequences=100,
        shots=1024,
        chip_name=chip_name,
        backend=client.chip_backend,
        qasm_version="2.0",
        print_true=True,
    )

    for key, payload in results.items():
        fit = payload.get("fit", {})
        print(f"Coupler {key}: p={fit.get('p')}, fidelity={fit.get('fidelity')}, epc={fit.get('epc')}")
