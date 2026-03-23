def evaluate_vqe_result(res, circuit):
    """
    Compute reward for VQE ansatz.
    """

    # Energy
    energy = res.best_energy

    # Gate counting
    cx_count = 0
    param_gate_count = len(circuit.params_value.items())

    for gate in circuit.gates:
        name = gate[0]
        if name == "cz":
            cx_count += 1

    # Reward design
    energy_weight = 1.0
    cx_penalty = 0.01
    param_penalty = 0.01

    loss = energy_weight * energy + cx_penalty * cx_count + param_penalty * param_gate_count

    return loss
