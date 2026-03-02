import datetime
import os

from quantum_hw import QuantumHardwareClient, ShadowTomography


if __name__ == "__main__":
    num_qubits = 7
    circuit = "ghz"
    date = datetime.date.today()
    name = f"Shadow_{circuit}_{num_qubits}_{date}"

    client = QuantumHardwareClient()
    shadow = ShadowTomography(client=client, seed=42)

    result = shadow.run(
        circuit=circuit,
        name=name,
        num_qubits=num_qubits,
        shots=4096,
        shots_per_basis=16,
        observables=["ZIIIIII", "ZZIIIII", "ZZZIIII", "ZZZZIII", "ZZZZZII", "ZZZZZZI", "ZZZZZZZ"],
        prefer_chips='Baihua', # only Baihua support small batch shots currently
        zne=True,
        estimator="mom",
        mom_groups=20,
    )

    print("Shadow estimates:", result.observable_estimates)
    print("Shadow estimates (raw):", result.observable_estimates_raw)
