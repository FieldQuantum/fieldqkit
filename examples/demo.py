import datetime

from quantum_hw import QuantumHardwareClient, QuantumCircuit
from quantum_hw.core.plotting import plot_observables_compare, plot_probabilities_compare

if __name__ == "__main__":
    num_qubits = 7
    circuit_name = "ghz"
    date = datetime.date.today()
    name = f'Demo_{circuit_name}_{num_qubits}_{date}'
    zne = True
    shots = 50000
    readout_mitigation = True
    observables = ['ZZZZZZZ', 'ZIIIIIZ', 'XXXXXXX']
    return_probabilities = True

    client = QuantumHardwareClient()

    # User can also use a custom circuit or one of the built-in templates.    
    circuit = "ghz"  # "ghz", "cluster", "QFT", "Ising evolution"
    # circuit = QuantumCircuit(num_qubits)
    # circuit.h(0)
    # for i in range(num_qubits - 1):
    #     circuit.cx(i, i + 1)

    results = client.run_auto(
        circuit,
        name,
        num_qubits,
        shots=shots,
        zne=zne,
        readout_mitigation=readout_mitigation,
        observables=observables,
        return_probabilities=return_probabilities,
        prefer_chips=['Baihua']
    )
    print("Expectation Value (Raw):", results.observable_values_raw)
    print("Expectation Value (Mitigated):", results.observable_values)
    plot_probabilities_compare(results.probabilities_raw, results.probabilities, num_qubits)
    plot_observables_compare(results.observable_values_raw, results.observable_values, observables)
