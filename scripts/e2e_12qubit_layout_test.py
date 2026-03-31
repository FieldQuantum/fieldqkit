"""
E2E 12-qubit stabilizer circuit validation.

Constructs a 12-qubit stabilizer (Clifford-only) circuit with long-range
interactions via a GHZ spanning tree + extra CZ gates + X flips on odd
qubits.  All chosen observables have exact expectations of +1 or -1,
making hardware fidelity comparison straightforward.
"""

import sys, time, traceback
sys.path.insert(0, "src")

import numpy as np
from quantum_hw.circuit import QuantumCircuit
from quantum_hw.sim import simulate_counts
from quantum_hw.api.client import QuantumHardwareClient


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


def build_12qubit_stabilizer_circuit():
    """Build a 12-qubit stabilizer circuit with known ±1 observable expectations.

    Construction:
      1) H(0) to create superposition on qubit 0.
      2) CX spanning tree with long-range edges to create 12-qubit GHZ state:
            0→11, 0→6, 11→10, 6→7, 7→8, 8→9, 0→1, 1→2, 2→3, 3→4, 4→5
         GHZ = (|000000000000⟩ + |111111111111⟩) / √2
      3) Long-range CZ gates for extra routing challenge:
            CZ(1,10), CZ(2,9), CZ(3,8), CZ(5,11)
         (Even number of CZs → phase unchanged; ZZ expectations preserved.)
      4) X on all odd qubits (1,3,5,7,9,11):
         → (|010101010101⟩ + |101010101010⟩) / √2
         ⟨Z_i Z_j⟩ = +1 if i,j same parity, −1 if different parity.
         ⟨Z_0 Z_1 … Z_11⟩ = (−1)^6 = +1

    2-qubit gates: 11 CX + 4 CZ = 15 total (6 long-range).
    """
    NUM_QUBITS = 12
    qc = QuantumCircuit(NUM_QUBITS)

    # ---------- GHZ via spanning tree (long-range branches) ----------
    qc.h(0)
    # Branch A: 0 → 11 → 10   (long-range start)
    qc.cx(0, 11)
    qc.cx(11, 10)
    # Branch B: 0 → 6 → 7 → 8 → 9   (long-range start)
    qc.cx(0, 6)
    qc.cx(6, 7)
    qc.cx(7, 8)
    qc.cx(8, 9)
    # Branch C: 0 → 1 → 2 → 3 → 4 → 5
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)

    # ---------- Extra long-range CZ (routing challenge) ----------
    qc.cz(1, 10)
    qc.cz(2, 9)
    qc.cz(3, 8)
    qc.cz(5, 11)

    # ---------- X-flip odd qubits → ±1 ZZ pattern ----------
    for i in range(1, NUM_QUBITS, 2):
        qc.x(i)

    return qc


# Observable string → exact expected value
OBSERVABLES_EXPECTED = {
    "Z" * 12:                              +1,   # full parity
    "ZZ" + "I" * 10:                       -1,   # Z0 Z1  (different parity)
    "Z" + "I" * 10 + "Z":                  -1,   # Z0 Z11 (different parity)
    "Z" + "I" * 5 + "Z" + "I" * 5:        +1,   # Z0 Z6  (same parity)
    "I" * 5 + "ZZ" + "I" * 5:             -1,   # Z5 Z6  (different parity)
    "I" * 2 + "Z" + "I" * 7 + "Z" + "I":  +1,   # Z2 Z10 (same parity)
}


def main():
    NUM_QUBITS = 12
    SHOTS = 4096
    OBSERVABLES = list(OBSERVABLES_EXPECTED.keys())

    qc = build_12qubit_stabilizer_circuit()

    n_2q = sum(1 for g in qc.gates if g[0] in ('cx', 'cz'))
    print("=" * 70)
    print("12-Qubit Stabilizer Circuit — Hardware Validation")
    print("=" * 70)
    print(f"Total gates     : {len(qc.gates)}")
    print(f"2-qubit gates   : {n_2q}  (11 CX + 4 CZ, 6 long-range)")
    print(f"Observables     : {len(OBSERVABLES)}")
    print("=" * 70)

    # ---------- Simulator reference ----------
    print("\n[Simulator] Running local simulation...")
    sim_counts = simulate_counts(qc, SHOTS, seed=42)
    sim_expectations = {}
    for obs in OBSERVABLES:
        sim_expectations[obs] = compute_expectation(sim_counts, obs, NUM_QUBITS)

    print(f"{'Observable':>16}  {'Expected':>8}  {'Sim':>8}  {'|Err|':>6}")
    for obs in OBSERVABLES:
        exp = OBSERVABLES_EXPECTED[obs]
        sim = sim_expectations[obs]
        short = obs if len(obs) <= 12 else obs[:10] + ".."
        print(f"  {short:>14}  {exp:>+8d}  {sim:>8.4f}  {abs(sim - exp):>6.4f}")

    # ---------- Real hardware ----------
    PLATFORMS = [
        {"provider": "quafu",   "chips": ["Baihua"],      "label": "Quafu/Baihua"},
        {"provider": "tianyan", "chips": ["tianyan176"],    "label": "TianYan/tianyan176"},
        {"provider": "guodun",  "chips": ["gd_qc1"],       "label": "GuoDun/gd_qc1"},
    ]

    results = {}
    client = QuantumHardwareClient()

    for platform in PLATFORMS:
        label = platform["label"]
        print(f"\n{'=' * 70}")
        print(f"[{label}] Submitting 12-qubit stabilizer circuit...")
        try:
            t0 = time.time()
            result = client.run_auto(
                circuit=qc,
                name=f"e2e_stab12q_{platform['provider']}_{int(time.time())}",
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
            print(f"  {'Observable':>16}  {'Expected':>8}  {'Hardware':>8}  {'|Err|':>6}")
            for obs in OBSERVABLES:
                exp = OBSERVABLES_EXPECTED[obs]
                val = result.observable_values.get(obs, float('nan'))
                short = obs if len(obs) <= 12 else obs[:10] + ".."
                print(f"    {short:>14}  {exp:>+8d}  {val:>8.4f}  {abs(val - exp):>6.4f}")
        except Exception as e:
            print(f"[{label}] FAILED: {e}")
            traceback.print_exc()

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY — deviation from exact stabilizer expectations")
    print("=" * 70)
    header = f"{'Observable':>14}  {'Exact':>6}"
    for p in PLATFORMS:
        header += f"  {p['label']:>22}"
    print(header)
    print("-" * len(header))

    for obs in OBSERVABLES:
        exp = OBSERVABLES_EXPECTED[obs]
        short = obs if len(obs) <= 12 else obs[:10] + ".."
        row = f"{short:>14}  {exp:>+6d}"
        for p in PLATFORMS:
            label = p["label"]
            if label in results:
                val = results[label].observable_values.get(obs, float('nan'))
                row += f"  {val:>22.4f}"
            else:
                row += f"  {'FAILED':>22}"
        print(row)

    n_success = len(results)
    n_total = len(PLATFORMS)
    print(f"\n{n_success}/{n_total} platforms succeeded.")
    if n_success < n_total:
        print("WARNING: Not all platforms succeeded!")
    print("\n" + "=" * 70)
    print("12-qubit stabilizer test complete.")


if __name__ == "__main__":
    main()
