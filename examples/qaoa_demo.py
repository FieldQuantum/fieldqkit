import datetime
import os

from quantum_hw import QuantumHardwareClient, QAOARunner


if __name__ == "__main__":
    num_qubits = 2
    date = datetime.date.today()
    name = f"QAOA_MaxCut_{num_qubits}_{date}"

    edges = [(0, 1)]
    weights = [1.0]

    client = QuantumHardwareClient()
    qaoa = QAOARunner(
        client=client,
        p=1,
        shots=1024,
        max_iters=10,
        learning_rate=0.1,
        seed=42,
    )

    result = qaoa.run_maxcut(
        name=name,
        num_qubits=num_qubits,
        edges=edges,
        weights=weights,
        prefer_chips=["Simulator"],
    )

    print("Best cost:", result.best_cost)
    print("Best params:", result.best_params)
