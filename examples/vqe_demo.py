import datetime
import os

from quantum_hw import QuantumHardwareClient, VQERunner


if __name__ == "__main__":
    num_qubits = 2
    date = datetime.date.today()
    model = 'ising'
    name = f"VQE_{model}_{num_qubits}_{date}"


    client = QuantumHardwareClient()
    vqe = VQERunner(
        client=client,
        layers=1,
        shots=4096,
        max_iters=20,
        learning_rate=0.1,
        zne=True,
        readout_mitigation=True,
        seed=42,
        shift=0.3,
    )

    print("[demo] start VQE:", name)

    result = vqe.run_model(
        name=name,
        num_qubits=num_qubits,
        model=model,
        model_params={"j": 1.0, "h": 1.0},
        prefer_chips="Simulator",
    )

    print("Best energy:", result.best_energy)
    print("Best params:", result.best_params)
