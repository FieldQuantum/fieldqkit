import datetime
import os

from quantum_hw import QuantumHardwareClient, VQERunner


if __name__ == "__main__":
    num_qubits = 4
    date = datetime.date.today()
    name = f"VQE_Ising_{num_qubits}_{date}"

    token = os.getenv("QUARK_TOKEN")
    if not token:
        raise RuntimeError("Missing QUARK_TOKEN in environment")

    client = QuantumHardwareClient(token=token)
    vqe = VQERunner(
        client=client,
        layers=2,
        shots=1024,
        max_iters=10,
        learning_rate=0.1,
        zne=False,
        readout_mitigation=False,
        seed=42,
    )

    result = vqe.run_ising(
        name=name,
        num_qubits=num_qubits,
        j=1.0,
        h=1.0,
        prefer_chips=["Yudu"],
    )

    print("Best energy:", result.best_energy)
    print("Best params:", result.best_params)
