import datetime
import matplotlib.pyplot as plt

from quantum_hw import QuantumHardwareClient


def _plot_probabilities(probabilities, num_qubits: int) -> None:
    if probabilities is None:
        return
    probs = probabilities
    if isinstance(probabilities, list) and probabilities and isinstance(probabilities[0], list):
        probs = probabilities[0]
    x = list(range(len(probs)))
    plt.figure(figsize=(10, 4))
    plt.bar(x, probs, color="#4C78A8")
    if len(probs) <= 32:
        labels = [format(i, f"0{num_qubits}b") for i in x]
        plt.xticks(x, labels, rotation=90)
    plt.xlabel("Basis state")
    plt.ylabel("Probability")
    plt.title("Measurement probabilities")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_qubits = 6
    circuit = 'ghz'
    date = datetime.date.today()
    name = f'Demo_{circuit}_{num_qubits}_{date}'
    zne = True
    shots = 50000
    readout_mitigation = True
    observables = ['IIZZII']
    return_probabilities = True

    token = "5gjq36bZsMvqFoSNomvnfPy4y[iDJWe[tBx9fIndISQ/:m{O5FEPyRkM4B{N{RkNyd{OypkJxiY[jxjJ4RkPyBkPxJEJ4FUMyBUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"
    client = QuantumHardwareClient(token=token)
    
    results = client.run_auto(
        circuit,
        name,
        num_qubits,
        shots=shots,
        zne=zne,
        readout_mitigation=readout_mitigation,
        observables=observables,
        return_probabilities=return_probabilities
    )
    print("Expectation Value:", results.observable_values)
    print("Probabilities:", results.probabilities)
    _plot_probabilities(results.probabilities, num_qubits)
