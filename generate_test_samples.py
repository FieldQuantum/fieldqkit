"""
Generate test samples for the FieldQuantum cloud simulator, including complex
symbolic parameter expressions (negation, product, sum, scalar multiple).

Each expectation case carries analytically derived expected values (or None for
sanity-check-only cases), and a validation section checks results against them.
"""

from __future__ import annotations

import json
import math
import os
import sys

PI = math.pi
TOL = 1e-5  # tolerance for exact analytical comparisons

# --- Sample mode cases -------------------------------------------------------
# Each entry may include "expected_*" fields for validation.

sample_cases = [
    # S1: |0> state — always measures 0
    {
        "desc": "S1 | |0> trivial -> always 0",
        "input": {
            "mode": "sample",
            "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nmeasure q -> c;',
            "shots": 256,
        },
        "expected_only_bitstrings": ["0"],
    },
    # S2: X|0> = |1> — always measures 1
    {
        "desc": "S2 | X|0> = |1> -> always 1",
        "input": {
            "mode": "sample",
            "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nx q[0];\nmeasure q -> c;',
            "shots": 256,
        },
        "expected_only_bitstrings": ["1"],
    },
    # S3: Bell state |Phi+> — only 00 or 11
    {
        "desc": "S3 | Bell |Phi+> -> only 00 or 11",
        "input": {
            "mode": "sample",
            "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;',
            "shots": 1024,
        },
        "expected_only_bitstrings": ["00", "11"],
    },
    # S4: GHZ 5-qubit — only 00000 or 11111
    {
        "desc": "S4 | GHZ 5q -> only 00000 or 11111",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[5];\n'
                "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncx q[3],q[4];\n"
                "measure q -> c;"
            ),
            "shots": 2048,
        },
        "expected_only_bitstrings": ["00000", "11111"],
    },
    # S5: H gate on all 3 qubits — uniform distribution, ~1/8 each, all 8 bitstrings
    {
        "desc": "S5 | H^3|0> -> uniform over 3-bit strings",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
                "h q[0];\nh q[1];\nh q[2];\nmeasure q -> c;"
            ),
            "shots": 4096,
        },
        "expected_num_bitstrings": 8,  # all 8 should appear with many shots
    },
    # S6: rx(pi) ≈ X gate on 2 qubits — always 11
    {
        "desc": "S6 | rx(pi) both qubits -> always 11",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
                "rx(3.14159265) q[0];\nrx(3.14159265) q[1];\nmeasure q -> c;"
            ),
            "shots": 512,
        },
        "expected_only_bitstrings": ["11"],
    },
    # S7: 4-qubit random rotations + entanglement, sanity check
    {
        "desc": "S7 | 4q rotations + entanglement, sanity check",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\n'
                "rx(0.5) q[0];\nry(1.2) q[1];\nrz(0.8) q[2];\nh q[3];\n"
                "cz q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\nmeasure q -> c;"
            ),
            "shots": 2048,
        },
        "expected_num_bitstrings": None,  # sanity only
    },
    # S8: |10> state — always 10
    {
        "desc": "S8 | X q[0], identity q[1] -> always 10",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
                "x q[0];\nmeasure q -> c;"
            ),
            "shots": 256,
        },
        "expected_only_bitstrings": ["10"],
    },
    # S9: S*H circuit on 2 qubits (|+i> x |0>), sanity check
    {
        "desc": "S9 | s; h; cx on 2q, sanity check",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
                "s q[0];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;"
            ),
            "shots": 1024,
        },
        "expected_num_bitstrings": None,
    },
    # S10: ry(pi/2)|0> repeated — exactly balanced 0/1 (with large shots)
    {
        "desc": "S10 | ry(pi/2) -> balanced 0/1",
        "input": {
            "mode": "sample",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
                "ry(1.5707963) q[0];\nmeasure q -> c;"
            ),
            "shots": 4096,
        },
        "expected_num_bitstrings": 2,
    },
]

# --- Expectation mode cases ---------------------------------------------------
# Analytical derivations are in the comments. expected_energy/gradients=None means
# only sanity checks (finiteness, bounds) are applied.

expectation_cases = [
    # -- Baseline single-qubit ------------------------------------------------
    # E1: rx(theta_0)|0>, H=-Z, theta=pi/2
    #   <Z> = cos(theta)  ->  E = -cos(pi/2) = 0,  dE/dtheta = sin(pi/2) = 1
    {
        "desc": "E1 | rx(theta), H=-Z, theta=pi/2 -> E=0, grad=1",
        "input": {
            "mode": "expectation",
            "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(theta_0) q[0];',
            "param_names": ["theta_0"],
            "param_values": [PI / 2],
            "hamiltonian": [{"coeff": -1.0, "pauli": "Z0"}],
        },
        "expected_energy": 0.0,
        "expected_gradients": [1.0],
    },
    # -- Negation: -theta_1 ----------------------------------------------------
    # E2: ry(-theta_1)|0>, H=-Z, theta1=1.0
    #   angle=-theta1, <Z>=cos(-theta1)=cos(theta1)  (cos is even)
    #   E = -cos(theta1),  dE/dtheta1 = sin(theta1)
    {
        "desc": "E2 | ry(-theta_1), H=-Z, theta1=1.0 -> E=-cos(1), grad=sin(1)",
        "input": {
            "mode": "expectation",
            "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nry(-theta_1) q[0];',
            "param_names": ["theta_1"],
            "param_values": [1.0],
            "hamiltonian": [{"coeff": -1.0, "pauli": "Z0"}],
        },
        "expected_energy": -math.cos(1.0),
        "expected_gradients": [math.sin(1.0)],
    },
    # -- Product: theta_1 * theta_2 --------------------------------------------
    # E3: H q[0]; rz(theta_1*theta_2) q[0], H=-X, theta1=1.0, theta2=0.5
    #   |+> = H|0>;  rz(phi)|+> rotates in XY plane;  <X> = cos(phi),  phi=theta1theta2
    #   E = -cos(theta1theta2)
    #   dE/dtheta1 = sin(theta1theta2)*theta2,  dE/dtheta2 = sin(theta1theta2)*theta1
    {
        "desc": "E3 | h;rz(theta_1*theta_2), H=-X, theta1=1,theta2=0.5 -> E=-cos(0.5)",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
                "h q[0];\nrz(theta_1*theta_2) q[0];"
            ),
            "param_names": ["theta_1", "theta_2"],
            "param_values": [1.0, 0.5],
            "hamiltonian": [{"coeff": -1.0, "pauli": "X0"}],
        },
        "expected_energy": -math.cos(0.5),
        "expected_gradients": [math.sin(0.5) * 0.5, math.sin(0.5) * 1.0],
    },
    # -- Sum: theta_0 + theta_1 ------------------------------------------------
    # E4: rx(theta_0+theta_1)|0>, H=-Z, theta0=0.5, theta1=0.3
    #   <Z> = cos(theta0+theta1)  ->  E = -cos(0.8)
    #   dE/dtheta0 = dE/dtheta1 = sin(0.8)  (equal because symmetric)
    {
        "desc": "E4 | rx(theta_0+theta_1), H=-Z, theta0=0.5,theta1=0.3 -> equal grads",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
                "rx(theta_0+theta_1) q[0];"
            ),
            "param_names": ["theta_0", "theta_1"],
            "param_values": [0.5, 0.3],
            "hamiltonian": [{"coeff": -1.0, "pauli": "Z0"}],
        },
        "expected_energy": -math.cos(0.8),
        "expected_gradients": [math.sin(0.8), math.sin(0.8)],
    },
    # -- Scalar multiple: 2*theta_0 --------------------------------------------
    # E5: rx(2*theta_0)|0>, H=-Z, theta0=pi/4
    #   angle=pi/2  ->  E=-cos(pi/2)=0,  dE/dtheta0=2*sin(pi/2)=2  (chain rule)
    {
        "desc": "E5 | rx(2*theta_0), H=-Z, theta0=pi/4 -> E=0, grad=2",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
                "rx(2*theta_0) q[0];"
            ),
            "param_names": ["theta_0"],
            "param_values": [PI / 4],
            "hamiltonian": [{"coeff": -1.0, "pauli": "Z0"}],
        },
        "expected_energy": 0.0,
        "expected_gradients": [2.0],
    },
    # -- Entangled + negation ---------------------------------------------------
    # E6: ry(theta_0) q[0]; CX; ry(-theta_1) q[1], H=-Z0Z1
    #   After the circuit: <Z0Z1> = cos(theta1) (independent of theta0; derivation in comments)
    #   Derivation:
    #     |psi> = cos(theta0/2)*[cos(theta1/2)|00> - sin(theta1/2)|01>]
    #          + sin(theta0/2)*[sin(theta1/2)|10> + cos(theta1/2)|11>]
    #     <Z0Z1> = [cos^2(theta0/2)+sin^2(theta0/2)]*[cos^2(theta1/2)-sin^2(theta1/2)] = cos(theta1)
    #   E=-cos(theta1),  dE/dtheta0=0,  dE/dtheta1=sin(theta1)
    {
        "desc": "E6 | ry(theta0); CX; ry(-theta1), H=-Z0Z1, theta0=0.8,theta1=0.6 -> grad[0]=0",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                "ry(theta_0) q[0];\ncx q[0],q[1];\nry(-theta_1) q[1];"
            ),
            "param_names": ["theta_0", "theta_1"],
            "param_values": [0.8, 0.6],
            "hamiltonian": [{"coeff": -1.0, "pauli": "Z0 Z1"}],
        },
        "expected_energy": -math.cos(0.6),
        "expected_gradients": [0.0, math.sin(0.6)],
    },
    # -- Difference: theta_0 - theta_1 -----------------------------------------
    # E7: rx(theta_0+theta_1) q[0]; rx(theta_0-theta_1) q[1], H=-Z0-Z1
    #   E = -cos(theta0+theta1) - cos(theta0-theta1)
    #   At theta0=theta1=0.5:
    #     E = -cos(1.0) - cos(0) = -cos(1) - 1
    #     dE/dtheta0 = sin(1.0) + sin(0) = sin(1)
    #     dE/dtheta1 = sin(1.0) - sin(0) = sin(1)   <- also equal at this point
    {
        "desc": "E7 | rx(theta0+theta1) q[0]; rx(theta0-theta1) q[1], H=-Z0-Z1, theta0=theta1=0.5",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                "rx(theta_0+theta_1) q[0];\nrx(theta_0-theta_1) q[1];"
            ),
            "param_names": ["theta_0", "theta_1"],
            "param_values": [0.5, 0.5],
            "hamiltonian": [
                {"coeff": -1.0, "pauli": "Z0"},
                {"coeff": -1.0, "pauli": "Z1"},
            ],
        },
        "expected_energy": -math.cos(1.0) - math.cos(0.0),
        "expected_gradients": [
            math.sin(1.0) + math.sin(0.0),
            math.sin(1.0) - math.sin(0.0),
        ],
    },
    # -- 3-qubit product state — fully factorisable ----------------------------
    # E8: 3 independent ry(theta_i), H=-Z0-Z1-Z2
    #   E = -cos(theta0)-cos(theta1)-cos(theta2),  grad_i = sin(thetai)
    {
        "desc": "E8 | 3xry(thetai), H=-Z0-Z1-Z2, theta=[0.3,0.6,0.9]",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
                "ry(theta_0) q[0];\nry(theta_1) q[1];\nry(theta_2) q[2];"
            ),
            "param_names": ["theta_0", "theta_1", "theta_2"],
            "param_values": [0.3, 0.6, 0.9],
            "hamiltonian": [
                {"coeff": -1.0, "pauli": "Z0"},
                {"coeff": -1.0, "pauli": "Z1"},
                {"coeff": -1.0, "pauli": "Z2"},
            ],
        },
        "expected_energy": -math.cos(0.3) - math.cos(0.6) - math.cos(0.9),
        "expected_gradients": [math.sin(0.3), math.sin(0.6), math.sin(0.9)],
    },
    # -- Mixed products on 2 qubits --------------------------------------------
    # E9: rz(theta_0*theta_1) q[0]; ry(theta_1*theta_2) q[1], H=-Z0-Z1
    #   q[0]: rz(phi)|0> = e^{-iphi/2}|0> -> <Z0>=1  (rz on |0> is a global phase)
    #   q[1]: ry(theta1theta2)|0> -> <Z1>=cos(theta1theta2)
    #   E = -1 - cos(theta1theta2)
    #   dE/dtheta0 = 0   dE/dtheta1 = sin(theta1theta2)*theta2   dE/dtheta2 = sin(theta1theta2)*theta1
    {
        "desc": "E9 | rz(theta0theta1) q[0]; ry(theta1theta2) q[1], H=-Z0-Z1, theta=[pi,1,0.5]",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                "rz(theta_0*theta_1) q[0];\nry(theta_1*theta_2) q[1];"
            ),
            "param_names": ["theta_0", "theta_1", "theta_2"],
            "param_values": [PI, 1.0, 0.5],
            "hamiltonian": [
                {"coeff": -1.0, "pauli": "Z0"},
                {"coeff": -1.0, "pauli": "Z1"},
            ],
        },
        "expected_energy": -1.0 - math.cos(1.0 * 0.5),
        "expected_gradients": [0.0, math.sin(0.5) * 0.5, math.sin(0.5) * 1.0],
    },
    # -- Complex 4-qubit VQE with all expression types -------------------------
    # E10: 4-qubit Ising-like ansatz mixing ry, rz, CX gates with
    #   rz(theta_0*theta_1), ry(-theta_2), rx(theta_2+theta_3), ry(theta_3)
    #   H = -Z0Z1 - Z1Z2 - Z2Z3   (1D transverse Ising, ferromagnetic)
    #   No closed-form -> sanity checks: |E| <= 3, all gradients finite
    {
        "desc": "E10 | 4q Ising VQE with mixed expressions, sanity check",
        "input": {
            "mode": "expectation",
            "qasm": (
                'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\n'
                "ry(theta_0) q[0];\nry(theta_1) q[1];\n"
                "cx q[0],q[1];\n"
                "rz(theta_0*theta_1) q[0];\nry(-theta_2) q[1];\n"
                "cx q[1],q[2];\n"
                "rx(theta_2+theta_3) q[2];\n"
                "cx q[2],q[3];\n"
                "ry(theta_3) q[3];"
            ),
            "param_names": ["theta_0", "theta_1", "theta_2", "theta_3"],
            "param_values": [0.5, 0.8, 1.2, 0.3],
            "hamiltonian": [
                {"coeff": -1.0, "pauli": "Z0 Z1"},
                {"coeff": -1.0, "pauli": "Z1 Z2"},
                {"coeff": -1.0, "pauli": "Z2 Z3"},
            ],
        },
        "expected_energy": None,
        "expected_gradients": None,
    },
]


# --- Validation helpers ------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:+.6f}"


def validate_sample(case: dict, result: dict) -> tuple[bool, list[str]]:
    ok = True
    issues: list[str] = []
    counts = result.get("counts", {})

    # Total shots must match
    shots = case["input"]["shots"]
    total = sum(counts.values())
    if total != shots:
        issues.append(f"  shot count mismatch: got {total}, expected {shots}")
        ok = False

    # Bitstring length must be consistent with creg
    import re
    creg_sizes = [int(m) for m in re.findall(r"creg\s+\w+\[(\d+)\]", case["input"]["qasm"])]
    expected_width = sum(creg_sizes)
    for bs in counts:
        if len(bs) != expected_width:
            issues.append(f"  bitstring '{bs}' has wrong length (expected {expected_width})")
            ok = False
            break

    # Only allowed bitstrings
    allowed = case.get("expected_only_bitstrings")
    if allowed is not None:
        bad = [bs for bs in counts if bs not in allowed]
        if bad:
            issues.append(f"  unexpected bitstrings: {bad}")
            ok = False

    # Expected number of distinct bitstrings
    expected_n = case.get("expected_num_bitstrings")
    if expected_n is not None and len(counts) != expected_n:
        issues.append(f"  distinct bitstrings: got {len(counts)}, expected {expected_n}")
        ok = False

    return ok, issues


def validate_expectation(case: dict, result: dict) -> tuple[bool, list[str]]:
    ok = True
    issues: list[str] = []

    energy = result.get("energy")
    grads = result.get("gradients", [])
    n_params = len(case["input"].get("param_names", []))

    # Energy must be finite
    if energy is None or not math.isfinite(energy):
        issues.append(f"  energy not finite: {energy}")
        ok = False

    # Gradient count must match param count
    if len(grads) != n_params:
        issues.append(f"  gradient count: got {len(grads)}, expected {n_params}")
        ok = False

    # All gradients must be finite
    for i, g in enumerate(grads):
        if not math.isfinite(g):
            issues.append(f"  grad[{i}] not finite: {g}")
            ok = False

    exp_e = case.get("expected_energy")
    exp_g = case.get("expected_gradients")

    # Exact energy check
    if exp_e is not None and energy is not None and math.isfinite(energy):
        diff = abs(energy - exp_e)
        if diff > TOL:
            issues.append(
                f"  energy: got {_fmt(energy)}, expected {_fmt(exp_e)}, diff={diff:.2e}"
            )
            ok = False

    # Exact gradient checks
    if exp_g is not None and len(grads) == len(exp_g):
        for i, (ag, eg) in enumerate(zip(grads, exp_g)):
            if not math.isfinite(ag):
                continue
            diff = abs(ag - eg)
            if diff > TOL:
                issues.append(
                    f"  grad[{i}]: got {_fmt(ag)}, expected {_fmt(eg)}, diff={diff:.2e}"
                )
                ok = False

    # Sanity bound: |energy| <= sum(|coeff|)
    h = case["input"].get("hamiltonian", [])
    max_e = sum(abs(t["coeff"]) for t in h) if h else float("inf")
    if energy is not None and math.isfinite(energy) and abs(energy) > max_e + TOL:
        issues.append(f"  |energy|={abs(energy):.4f} exceeds Hamiltonian bound {max_e:.4f}")
        ok = False

    return ok, issues


# --- Main --------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from quantum_hw.api.fieldquantum_server import _handle_sample, _handle_expectation

    output_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(output_dir, exist_ok=True)

    sample_outputs: list[dict] = []
    expectation_outputs: list[dict] = []

    # -- Sample cases ----------------------------------------------------------
    print("=" * 65)
    print("SAMPLE MODE")
    print("=" * 65)

    n_pass_s = n_fail_s = 0
    for i, case in enumerate(sample_cases):
        tag = f"[S{i+1:02d}]"
        print(f"\n{tag} {case['desc']}")
        try:
            result = _handle_sample(case["input"])
            counts = result.get("counts", {})
            total = sum(counts.values())
            print(f"       shots={total}, distinct={len(counts)}")
            top = sorted(counts.items(), key=lambda x: -x[1])[:4]
            for bs, cnt in top:
                print(f"       {bs}: {cnt} ({100*cnt/max(total,1):.1f}%)")
            ok, issues = validate_sample(case, result)
        except Exception as exc:
            result = {"error": str(exc)}
            ok, issues = False, [f"  EXCEPTION: {exc}"]

        sample_outputs.append({"case": case["desc"], "input": case["input"], "output": result})

        if ok:
            print(f"       PASS")
            n_pass_s += 1
        else:
            for msg in issues:
                print(f"       FAIL: {msg.strip()}")
            n_fail_s += 1

    # -- Expectation cases -----------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPECTATION MODE  (autograd, exact statevector)")
    print("=" * 65)

    n_pass_e = n_fail_e = 0
    for i, case in enumerate(expectation_cases):
        tag = f"[E{i+1:02d}]"
        print(f"\n{tag} {case['desc']}")
        try:
            result = _handle_expectation(case["input"])
            energy = result.get("energy")
            grads = result.get("gradients", [])
            exp_e = case.get("expected_energy")
            exp_g = case.get("expected_gradients")

            print(f"       energy  : {_fmt(energy) if energy is not None else 'N/A'}", end="")
            if exp_e is not None:
                print(f"  (expected {_fmt(exp_e)})", end="")
            print()

            for j, g in enumerate(grads):
                eg_str = f"  (expected {_fmt(exp_g[j])})" if (exp_g and j < len(exp_g)) else ""
                print(f"       grad[{j}] : {_fmt(g)}{eg_str}")

            ok, issues = validate_expectation(case, result)
        except Exception as exc:
            import traceback as tb
            result = {"error": str(exc)}
            ok, issues = False, [f"  EXCEPTION: {exc}"]
            tb.print_exc()

        expectation_outputs.append({"case": case["desc"], "input": case["input"], "output": result})

        if ok:
            print(f"       PASS")
            n_pass_e += 1
        else:
            for msg in issues:
                print(f"       FAIL: {msg.strip()}")
            n_fail_e += 1

    # -- Save JSON -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SAVING JSON")
    print("=" * 65)

    def _save(path: str, obj: object) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"  saved: {os.path.basename(path)}")

    for i, item in enumerate(sample_outputs):
        _save(os.path.join(output_dir, f"sample_{i+1:02d}_input.json"), item["input"])
        _save(os.path.join(output_dir, f"sample_{i+1:02d}_output.json"), item["output"])
    for i, item in enumerate(expectation_outputs):
        _save(os.path.join(output_dir, f"expectation_{i+1:02d}_input.json"), item["input"])
        _save(os.path.join(output_dir, f"expectation_{i+1:02d}_output.json"), item["output"])

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"SUMMARY")
    print(f"  Sample     : {n_pass_s}/{n_pass_s+n_fail_s} passed")
    print(f"  Expectation: {n_pass_e}/{n_pass_e+n_fail_e} passed")
    total_pass = n_pass_s + n_pass_e
    total = n_pass_s + n_fail_s + n_pass_e + n_fail_e
    print(f"  Total      : {total_pass}/{total} passed")
    print("=" * 65)


if __name__ == "__main__":
    main()

