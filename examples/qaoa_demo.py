import datetime
import os

from quantum_hw import QuantumHardwareClient, QAOARunner


if __name__ == "__main__":
    num_qubits = 4
    date = datetime.date.today()
    name = f"QAOA_MaxCut_{num_qubits}_{date}"

    token = os.getenv("QUARK_TOKEN")
    if not token:
        raise RuntimeError("Missing QUARK_TOKEN in environment")

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    weights = [1.0, 1.0, 1.0, 1.0]

    client = QuantumHardwareClient(token=token)
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
        prefer_chips=["Yudu"],
    )

    print("Best cost:", result.best_cost)
    print("Best params:", result.best_params)
