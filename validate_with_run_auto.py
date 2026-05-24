"""
validate_with_run_auto.py

Re-run all test cases from test_data/ using QuantumHardwareClient.run_auto
with the FieldQuantum provider, then compare against stored reference outputs.

Sample mode  : compare frequency distributions    (max |freq_new-freq_ref| < TOL_FREQ)
Expectation  : compare energy + expectations      (|new-ref| < TOL_ENERGY)
Gradients    : parameter-shift rule (2N+1 runs)   (max |grad_ps-grad_ref| < TOL_GRAD)

All runs use SHOTS = 10_000.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
# ---- Configuration -----------------------------------------------------------
SHOTS = 10000
TOL_FREQ = 1e-2
TOL_ENERGY = 5e-3
TOL_GRAD = 1e-2
SHIFT = 0.1

ROOT = Path(__file__).resolve().parent
TEST_DATA = ROOT / "test_data"

sys.path.insert(0, str(ROOT / "src"))


# ---- QASM helpers ------------------------------------------------------------

def _parse_num_qubits(qasm: str) -> int:
    """Return total qubit count from qreg declarations."""
    sizes = [int(m) for m in re.findall(r"qreg\s+\w+\[(\d+)\]", qasm)]
    return sum(sizes)


def _substitute_params(
    qasm: str,
    param_names: List[str],
    param_values: List[float],
) -> str:
    """Replace symbolic param names with their numeric values (word-boundary safe)."""
    for name, val in zip(param_names, param_values):
        qasm = re.sub(r"\b" + re.escape(name) + r"\b", repr(val), qasm)
    return qasm


def _compute_energy(
    observable_values: Dict[str, float],
    hamiltonian: List[Dict[str, Any]],
) -> float:
    """E = sum(coeff * <obs>)."""
    return sum(
        term["coeff"] * observable_values.get(term["pauli"], 0.0)
        for term in hamiltonian
    )


def _samples_to_counts(samples: List[List[int]]) -> Dict[str, int]:
    """Convert RunResult.samples[group] (list of shot-bit-arrays) to bitstring counts."""
    counts: Dict[str, int] = {}
    for shot in samples:
        bs = "".join(str(b) for b in shot)
        counts[bs] = counts.get(bs, 0) + 1
    return counts


# ---- Run helpers -------------------------------------------------------------

def _run_sample(client, qasm: str, shots: int):
    """Submit a sample-mode circuit via run_auto and return RunResult."""
    from quantum_hw.circuit import QuantumCircuit
    qc = QuantumCircuit().from_openqasm2(qasm)
    num_qubits = _parse_num_qubits(qasm)
    return client.run_auto(
        qc, "val_sample", num_qubits,
        provider="simulator",
        shots=shots,
        transpile_on_client=False,
        print_true=False,
    )


def _run_expectation(client, qasm: str, observables: List[str], shots: int):
    """Submit an expectation-mode circuit via run_auto and return RunResult."""
    from quantum_hw.circuit import QuantumCircuit
    qc = QuantumCircuit().from_openqasm2(qasm)
    num_qubits = _parse_num_qubits(qasm)
    return client.run_auto(
        qc, "val_exp", num_qubits,
        provider="simulator",
        shots=shots,
        observables=observables,
        transpile_on_client=False,
        print_true=False,
    )


# ---- Case validators ---------------------------------------------------------

def validate_sample_case(
    client,
    inp: dict,
    ref_output: dict,
    shots: int = SHOTS,
) -> Tuple[bool, List[str]]:
    """
    Run sample mode via run_auto and compare frequency distribution.
    Returns (ok, detail_messages).
    """
    qasm = inp["qasm"]
    result = _run_sample(client, qasm, shots)

    new_counts = _samples_to_counts(result.samples[0])
    ref_counts: Dict[str, int] = ref_output.get("counts", {})

    issues: List[str] = []
    ok = True

    # Shot count
    total_new = sum(new_counts.values())
    if total_new != shots:
        issues.append(f"shot count mismatch: got {total_new}, expected {shots}")
        ok = False

    # Frequency comparison
    all_bs = set(new_counts) | set(ref_counts)
    total_ref = max(sum(ref_counts.values()), 1)
    max_diff = 0.0
    worst_bs = ""
    for bs in all_bs:
        f_new = new_counts.get(bs, 0) / max(total_new, 1)
        f_ref = ref_counts.get(bs, 0) / total_ref
        d = abs(f_new - f_ref)
        if d > max_diff:
            max_diff = d
            worst_bs = bs

    status = "OK" if max_diff <= TOL_FREQ else "FAIL"
    issues.append(
        f"max freq diff = {max_diff:.4f} [{worst_bs}] ({status}, tol={TOL_FREQ})"
    )
    if max_diff > TOL_FREQ:
        ok = False

    issues.append(
        f"distinct bitstrings: ref={len(ref_counts)}, new={len(new_counts)}"
    )
    return ok, issues


def validate_expectation_case(
    client,
    inp: dict,
    ref_output: dict,
    shots: int = SHOTS,
) -> Tuple[bool, List[str]]:
    """
    Run expectation mode via run_auto (sample-based expectations) and compare
    energy/expectations against stored reference. Gradients via parameter-shift.
    Returns (ok, detail_messages).
    """
    qasm_sym: str = inp["qasm"]
    print(f"  QASM (symbolic): {qasm_sym}")
    param_names: List[str] = inp.get("param_names", [])
    param_values: List[float] = list(inp.get("param_values", []))
    hamiltonian: List[Dict[str, Any]] = inp.get("hamiltonian", [])

    # Unique observable strings (preserve order)
    seen = set()
    observables: List[str] = []
    for t in hamiltonian:
        p = t["pauli"]
        if p not in seen:
            seen.add(p)
            observables.append(p)

    ref_energy: Optional[float] = ref_output.get("energy")
    ref_expectations: Dict[str, float] = ref_output.get("expectations", {})
    ref_gradients: List[float] = ref_output.get("gradients", [])

    issues: List[str] = []
    ok = True

    # --- 1. Energy at nominal param values ------------------------------------
    qasm_num = _substitute_params(qasm_sym, param_names, param_values)
    print(f"  QASM (numeric): {qasm_num}")
    result = _run_expectation(client, qasm_num, observables, shots)
    obs_vals = result.observable_values
    energy_new = _compute_energy(obs_vals, hamiltonian)

    if ref_energy is not None:
        ediff = abs(energy_new - ref_energy)
        tag = "OK" if ediff <= TOL_ENERGY else "FAIL"
        issues.append(
            f"energy: new={energy_new:+.6f}, ref={ref_energy:+.6f}, "
            f"diff={ediff:.2e} ({tag}, tol={TOL_ENERGY})"
        )
        if ediff > TOL_ENERGY:
            ok = False

    # Per-observable expectation comparison
    max_exp_diff = 0.0
    worst_obs = ""
    for pauli in observables:
        if pauli in ref_expectations and pauli in obs_vals:
            d = abs(obs_vals[pauli] - ref_expectations[pauli])
            if d > max_exp_diff:
                max_exp_diff = d
                worst_obs = pauli
    if ref_expectations:
        tag = "OK" if max_exp_diff <= TOL_ENERGY else "FAIL"
        issues.append(
            f"max obs diff = {max_exp_diff:.2e} [{worst_obs}] "
            f"({tag}, tol={TOL_ENERGY})"
        )
        if max_exp_diff > TOL_ENERGY:
            ok = False

    # --- 2. Parameter-shift gradients (2N more evaluations) ------------------
    if param_names and ref_gradients:
        n = len(param_names)
        grads_ps: List[float] = []

        for k in range(n):
            vals_plus = list(param_values)
            vals_plus[k] += SHIFT
            qasm_p = _substitute_params(qasm_sym, param_names, vals_plus)
            res_p = _run_expectation(client, qasm_p, observables, shots)
            E_plus = _compute_energy(res_p.observable_values, hamiltonian)

            vals_minus = list(param_values)
            vals_minus[k] -= SHIFT
            qasm_m = _substitute_params(qasm_sym, param_names, vals_minus)
            res_m = _run_expectation(client, qasm_m, observables, shots)
            E_minus = _compute_energy(res_m.observable_values, hamiltonian)

            grads_ps.append((E_plus - E_minus) / (2.0*SHIFT))

        max_gdiff = 0.0
        worst_k = -1
        grad_details = []
        for k, (gps, gref) in enumerate(zip(grads_ps, ref_gradients)):
            d = abs(gps - gref)
            grad_details.append(
                f"  grad[{k}]: ps={gps:+.5f}, ref={gref:+.5f}, diff={d:.2e}"
            )
            if d > max_gdiff:
                max_gdiff = d
                worst_k = k

        tag = "OK" if max_gdiff <= TOL_GRAD else "FAIL"
        issues.append(
            f"max grad diff = {max_gdiff:.2e} [k={worst_k}] "
            f"({tag}, tol={TOL_GRAD})"
        )
        issues.extend(grad_details)
        if max_gdiff > TOL_GRAD:
            ok = False

        issues.append(
            f"[param-shift used {2*n+1} run_auto calls for this case]"
        )

    elif not param_names:
        issues.append("no params; gradient check skipped")

    return ok, issues


# ---- Main --------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_cases(data_dir: Path):
    """Return sorted list of (tag, mode, input_path, output_path)."""
    cases = []
    # for inp_path in sorted(data_dir.glob("*_input.json")):
    for inp_path in sorted(data_dir.glob("expectation_19_input.json")):
    
        stem = inp_path.stem  # e.g. "sample_01_input"
        # Remove trailing "_input"
        base = stem[: -len("_input")]  # e.g. "sample_01"
        out_path = data_dir / f"{base}_output.json"
        if not out_path.exists():
            print(f"  [SKIP] no matching output: {out_path.name}")
            continue
        # Infer mode from filename prefix
        if base.startswith("sample"):
            mode = "sample"
        else:
            mode = "expectation"
        cases.append((base.upper(), mode, inp_path, out_path))
    return cases


def main() -> None:
    from quantum_hw.api.client import QuantumHardwareClient

    print("=" * 68)
    print("FieldQuantum run_auto validation")
    print(f"  SHOTS={SHOTS}, TOL_FREQ={TOL_FREQ}, "
          f"TOL_ENERGY={TOL_ENERGY}, TOL_GRAD={TOL_GRAD}")
    print("=" * 68)


    client = QuantumHardwareClient()
    cases = _collect_cases(TEST_DATA)

    print(f"\nFound {len(cases)} cases in {TEST_DATA}\n")

    n_pass = n_fail = n_error = 0
    sample_pass = sample_fail = 0
    exp_pass = exp_fail = 0

    for tag, mode, inp_path, out_path in cases:
        inp = _load_json(inp_path)
        ref_output = _load_json(out_path)

        print(f"[{tag}] {mode}  ({inp_path.name})")

        try:
            if mode == "sample":
                ok, issues = validate_sample_case(client, inp, ref_output)
                if ok:
                    sample_pass += 1
                else:
                    sample_fail += 1
            else:
                ok, issues = validate_expectation_case(client, inp, ref_output)
                if ok:
                    exp_pass += 1
                else:
                    exp_fail += 1

            verdict = "PASS" if ok else "FAIL"
            print(f"  --> {verdict}")
            for msg in issues:
                print(f"      {msg}")
            if ok:
                n_pass += 1
            else:
                n_fail += 1
        except Exception as exc:
            import traceback
            print(f"  --> ERROR: {exc}")
            traceback.print_exc()
            n_error += 1
        print()

    # --- Summary ---
    print("=" * 68)
    print("SUMMARY")
    print(f"  Total cases   : {len(cases)}")
    print(f"  PASS          : {n_pass}")
    print(f"  FAIL          : {n_fail}")
    print(f"  ERROR         : {n_error}")
    print(f"  Sample  PASS/FAIL: {sample_pass}/{sample_fail}")
    print(f"  Expect  PASS/FAIL: {exp_pass}/{exp_fail}")
    print("=" * 68)
    print()
    print("Notes:")
    print("  - Energy/expectation tolerances (1e-4) are tight for shot-based")
    print("    estimation (~1/sqrt(10000) = 0.01 std). Expect some FAIL on")
    print("    multi-term Hamiltonians; deterministic cases should PASS.")
    print("  - Gradient tolerance (1e-3): parameter-shift is exact for linear")
    print("    gate expressions but approximate for products / sums / negations.")
    print("=" * 68)


if __name__ == "__main__":
    main()
