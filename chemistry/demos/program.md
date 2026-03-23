# VQE Ansatz Search Program (F2 molecule)

You are designing a parameterized quantum circuit (ansatz) for VQE.

Your ONLY goal is to modify the function:
```
build_f2_symbolic_ansatz()
```
to minimize the final energy.

---

## 🧠 Problem Setting

- System: F2 molecule
- Number of qubits: 12
- freeze 1a1,2a1,3a1,4a1; active 1e1,2e1,3e1,4e1,5a1,6a1; 10 active electrons
- Hamiltonian: already fixed (handled internally)
- Hatree-Fork state reads as |111111111100>
- You DO NOT need to touch Hamiltonian or VQE runner

---

## ⚙️ Available Quantum Gates

You can use ONLY the following gates:

### Single-qubit parameterized gates:
- ```qc.ry(theta, qubit)```
- ```qc.rz(theta, qubit)```

### Single-qubit fixed gates:
- ```qc.h(qubit)```
- ```qc.s(qubit)```
- ```qc.sdg(qubit)```
- ```qc.x(qubit)```

### Two-qubit gates:
- ```qc.cx(control, target)```

---

## 🔢 Parameters

- Parameters must be strings like:
  - "theta_0", "theta_1", "theta_2", ...
- DO NOT reuse parameter names unless you intentionally want shared parameters

---

## 🏗️ Circuit Construction Rules

You MUST:

1. Keep function signature unchanged:

```def build_f2_symbolic_ansatz() -> QuantumCircuit:```

2. Use exactly 12 qubits:

```qc = QuantumCircuit(12)```

3. Return the circuit:

```return qc```

---

## 🚫 Forbidden Actions

- DO NOT modify any code outside this function
- DO NOT change runner logic
- DO NOT import new libraries
- DO NOT change qubit number

---

## ▶️ How Your Circuit Is Evaluated

Your circuit will be executed as:
```
custom_ansatz_qc = build_f2_symbolic_ansatz()
runner = VQERunner(
    client=QuantumHardwareClient(),
    shots=cfg['shots'],
    max_iters=cfg['max_iters'],
    learning_rate=cfg['learning_rate'],
    gradient_method=cfg['gradient_method'],
    seed=cfg['seed'],
    shift=cfg['shift'],
)
kwargs = {
    'name': 'f2_12q_custom',
    'num_qubits': nqubits,
    'model': 'custom',
    'hamiltonian': f2_12q_terms,
    'ansatz': 'custom',
    'custom_ansatz_circuit': custom_ansatz_qc,
    'prefer_chips': cfg['prefer_chips'],
    'init_params': [0.01] * len(symbolic_params),
}
res = runner.run_model(**kwargs)
```

You will receive:

- ```res.best_energy```
- ```res.energy_history```
- ```res.grad_history```

---

## 🏆 Optimization Objective (Reward)

Your goal is to:

1. Minimize energy (MOST IMPORTANT)
2. Use fewer CX gates
3. Use fewer parameterized gates (ry, rz)

---

## 🧩 Design Strategies (IMPORTANT)

You are encouraged to explore:

1. Hardware-efficient ansatz
2. Structured entanglement
3. Symmetry-inspired
4. Depth control

---

## 🧪 Example (Baseline)

```
def build_f2_symbolic_ansatz() -> QuantumCircuit:
    qc = QuantumCircuit(12)

    qc.ry("theta_0", 8)
    qc.cx(8, 9)
    qc.cx(9, 10)
    qc.cx(10, 11)

    for q in range(10):
        qc.x(q)

    return qc
```
---

## 🔁 Iteration Strategy

- Reuse good structures
- Add small modifications
- Avoid overly deep circuits

---

## 🎯 Output Requirement

You MUST output ONLY valid Python code:
- Only the function
