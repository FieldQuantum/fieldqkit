import datetime
import os

from quantum_hw import QuantumHardwareClient, ShadowTomography


if __name__ == "__main__":
    num_qubits = 2
    circuit = "ghz"
    date = datetime.date.today()
    name = f"Shadow_{circuit}_{num_qubits}_{date}"

    token = "5gjq36bZsMvqFoSNomvnfPy4y[iDJWe[tBx9fIndISQ/:m{O5FEPyRkM4B{N{RkNyd{OypkJxiY[jxjJ4RkPyBkPxJEJ4FUMyBUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"

    client = QuantumHardwareClient(token=token)
    shadow = ShadowTomography(client=client, seed=42, batch_size=16)

    result = shadow.run(
        circuit=circuit,
        name=name,
        num_qubits=num_qubits,
        shots=4096,
        observables=["ZZ", "XX", "ZI", "IZ", "YY"],
        prefer_chips=["Baihua"], # only Baihua support small batch shots currently
        zne=True,
        estimator="mom",
        mom_groups=20,
    )

    print("Shadow estimates:", result.observable_estimates)
