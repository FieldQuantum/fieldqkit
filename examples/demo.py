import datetime

from quantum_hw import QuantumHardwareClient
from quantum_hw.plotting import plot_observables_compare, plot_probabilities_compare

if __name__ == "__main__":
    num_qubits = 6
    circuit = 'ghz' # 'cluster', 'QFT', 'Ising evolution'
    date = datetime.date.today()
    name = f'Demo_{circuit}_{num_qubits}_{date}'
    zne = True
    shots = 50000
    readout_mitigation = True
    observables = ['ZIIIIZ']
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
        return_probabilities=return_probabilities,
        prefer_chips='Yudu'
    )
    print("Expectation Value (Raw):", results.observable_values_raw)
    print("Expectation Value (Mitigated):", results.observable_values)
    plot_probabilities_compare(results.probabilities_raw, results.probabilities, num_qubits)
    plot_observables_compare(results.observable_values_raw, results.observable_values)
