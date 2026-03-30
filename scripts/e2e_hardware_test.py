"""
End-to-end hardware test: complex circuit → simulator + 3 real platforms.

Constructs a 4-qubit circuit with long-range interactions (q0↔q3, q1↔q3),
submits to Quafu/Baihua, TianYan/tianyan176, GuoDun/gd_qc1, and compares
with local statevector simulation.
"""

import sys, time, traceback
sys.path.insert(0, "src")

import numpy as np
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim import simulate_counts
from quantum_hw.api.client import QuantumHardwareClient

# ──────────────────── 1. Build a complex circuit ────────────────────
# 4-qubit circuit with long-range interactions
# Includes: H, CX (nearest & long-range), RZ, CZ, non-trivial entanglement
NUM_QUBITS = 4
SHOTS = 4096
OBSERVABLES = ["ZZII", "IIZZ", "ZZZZ", "ZIIZ"]

qc = QuantumCircuit(NUM_QUBITS)

# Layer 1: superposition
qc.h(0)
qc.h(2)

# Layer 2: nearest-neighbor entanglement
qc.cx(0, 1)
qc.cx(2, 3)

# Layer 3: long-range interaction (q0 ↔ q3)
qc.cz(0, 3)

# Layer 4: parametric rotations
qc.rz(np.pi / 4, 1)
qc.rz(np.pi / 3, 2)

# Layer 5: another long-range interaction (q1 ↔ q3)
qc.cx(1, 3)

# Layer 6: final rotations
qc.h(0)
qc.h(3)

print("=" * 60)
print("Circuit gates:")
for g in qc.gates:
    print("  ", g)
print(f"Total gates: {len(qc.gates)}")
print("=" * 60)

# ──────────────────── 2. Simulator reference ────────────────────
print("\n[Simulator] Running local simulation...")
sim_counts = simulate_counts(qc, SHOTS, seed=42)
sim_total = sum(sim_counts.values())
print(f"Simulator total counts: {sim_total}")

# Compute observable expectations from simulator
def compute_expectation(counts, observable, num_qubits):
    """Compute <O> from measurement counts for a Pauli-Z string."""
    total = sum(counts.values())
    expectation = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.zfill(num_qubits)
        sign = 1
        for i, op in enumerate(observable):
            if op == 'Z':
                if bits[i] == '1':
                    sign *= -1
        expectation += sign * count
    return expectation / total

sim_expectations = {}
for obs in OBSERVABLES:
    sim_expectations[obs] = compute_expectation(sim_counts, obs, NUM_QUBITS)

print(f"\nSimulator expectations:")
for obs, val in sim_expectations.items():
    print(f"  <{obs}> = {val:.4f}")

# Show top probabilities
sorted_counts = sorted(sim_counts.items(), key=lambda x: -x[1])
print(f"\nSimulator top states:")
for bs, cnt in sorted_counts[:8]:
    print(f"  |{bs}> : {cnt}/{sim_total} = {cnt/sim_total:.4f}")

# ──────────────────── 3. Real hardware submissions ────────────────────
PLATFORMS = [
    {"provider": "quafu",   "chips": ["Baihua"],      "label": "Quafu/Baihua"},
    {"provider": "tianyan", "chips": ["tianyan176"],    "label": "TianYan/tianyan176"},
    {"provider": "guodun",  "chips": ["gd_qc1"],       "label": "GuoDun/gd_qc1"},
]

results = {}
client = QuantumHardwareClient()

for platform in PLATFORMS:
    label = platform["label"]
    print(f"\n{'=' * 60}")
    print(f"[{label}] Submitting...")
    try:
        t0 = time.time()
        result = client.run_auto(
            circuit=qc,
            name=f"e2e_test_{platform['provider']}",
            num_qubits=NUM_QUBITS,
            shots=SHOTS,
            observables=OBSERVABLES,
            return_probabilities=True,
            provider=platform["provider"],
            prefer_chips=platform["chips"],
            print_true=False,
        )
        elapsed = time.time() - t0
        results[label] = result
        print(f"[{label}] Done in {elapsed:.1f}s")
        print(f"  task_ids: {result.task_ids}")
        print(f"  Observable values:")
        for obs in OBSERVABLES:
            val = result.observable_values.get(obs, float('nan'))
            sim_val = sim_expectations[obs]
            diff = abs(val - sim_val)
            print(f"    <{obs}> = {val:.4f}  (sim: {sim_val:.4f}, diff: {diff:.4f})")
    except Exception as e:
        print(f"[{label}] FAILED: {e}")
        traceback.print_exc()

# ──────────────────── 4. Summary comparison ────────────────────
print("\n" + "=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)
header = f"{'Observable':>12}  {'Simulator':>10}"
for p in PLATFORMS:
    header += f"  {p['label']:>20}"
print(header)
print("-" * len(header))

for obs in OBSERVABLES:
    row = f"{obs:>12}  {sim_expectations[obs]:>10.4f}"
    for p in PLATFORMS:
        label = p["label"]
        if label in results:
            val = results[label].observable_values.get(obs, float('nan'))
            row += f"  {val:>20.4f}"
        else:
            row += f"  {'FAILED':>20}"
    print(row)

print("\n" + "=" * 60)
print("Test complete.")
