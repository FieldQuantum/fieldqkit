#!/usr/bin/env python3
"""Run the three-step F2 automation pipeline end-to-end.

This orchestrator sequentially runs:
1) export_f2_terms_wsl_auto.py
2) f2_ucc_generator.py
3) run_f2_auto_vqe.py

It relies on the existing step scripts rather than reimplementing their logic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], label: str) -> None:
    print(f"=== {label} ===")
    print("command:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def default_ham_output(args: argparse.Namespace) -> Path:
    return Path(
        f"chemistry/data/f2_R{args.R:.1f}_{args.unit}_{args.basis}_auto.json"
    )


def default_result_output(args: argparse.Namespace) -> Path:
    return Path(
        f"chemistry/data/f2_R{args.R:.1f}_{args.unit}_{args.basis}_topk{args.ucc_topk}_vqe_result.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the F2 export -> UCC generation -> VQE pipeline")

    parser.add_argument("--R", type=float, default=2.6, help="F-F bond distance value")
    parser.add_argument("--unit", type=str, default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)

    parser.add_argument("--encoding", type=str, default="jw", choices=["jw", "bk", "scbk"])
    parser.add_argument("--reduction", type=str, default="auto_minq", choices=["none", "paper12", "auto_minq"])
    parser.add_argument("--chemical-accuracy", type=float, default=1e-3)
    parser.add_argument("--min-active-orbs", type=int, default=2)
    parser.add_argument("--max-active-orbs", type=int, default=None)
    parser.add_argument("--score-cut", type=float, default=2e-3)
    parser.add_argument("--frozen-occ-cut", type=float, default=1.99)
    parser.add_argument("--frozen-virt-cut", type=float, default=0.01)
    parser.add_argument("--respect-symmetry-groups", action="store_true")
    parser.add_argument("--group-energy-tol", type=float, default=1e-8)

    parser.add_argument("--ham-output", type=str, default="", help="Optional path for step-1 JSON output")
    parser.add_argument("--ucc-output", type=str, default="chemistry/data/ucc_f2.json", help="Path for step-2 generated UCC file")
    parser.add_argument("--result-output", type=str, default="", help="Optional path for step-3 result JSON")

    parser.add_argument("--include-singles", action="store_true", help="Include single excitations in step 2")
    parser.add_argument("--include-doubles", action="store_true", help="Include double excitations in step 2")

    parser.add_argument("--ucc-topk", type=int, default=5)
    parser.add_argument("--importance-cutoff", type=float, default=1e-4)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--gradient-method",
        type=str,
        default="autograd",
        choices=["autograd", "parameter-shift"],
    )
    parser.add_argument("--shift", type=float, default=1.5707963267948966)
    parser.add_argument("--prefer-chips", type=str, default="Simulator")
    parser.add_argument("--name", type=str, default="f2_auto_pipeline")
    parser.add_argument("--init-value", type=float, default=0.01)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.include_singles and not args.include_doubles:
        args.include_doubles = True

    ham_output = Path(args.ham_output) if args.ham_output else default_ham_output(args)
    ucc_output = Path(args.ucc_output)
    result_output = Path(args.result_output) if args.result_output else default_result_output(args)

    python_exe = sys.executable

    step1 = [
        python_exe,
        "chemistry/scripts/export_f2_terms_wsl_auto.py",
        "--R",
        str(args.R),
        "--unit",
        args.unit,
        "--basis",
        args.basis,
        "--multiplicity",
        str(args.multiplicity),
        "--charge",
        str(args.charge),
        "--encoding",
        args.encoding,
        "--reduction",
        args.reduction,
        "--chemical-accuracy",
        str(args.chemical_accuracy),
        "--min-active-orbs",
        str(args.min_active_orbs),
        "--score-cut",
        str(args.score_cut),
        "--frozen-occ-cut",
        str(args.frozen_occ_cut),
        "--frozen-virt-cut",
        str(args.frozen_virt_cut),
        "--group-energy-tol",
        str(args.group_energy_tol),
        "--output",
        str(ham_output),
    ]
    if args.max_active_orbs is not None:
        step1.extend(["--max-active-orbs", str(args.max_active_orbs)])
    if args.respect_symmetry_groups:
        step1.append("--respect-symmetry-groups")

    step2 = [
        python_exe,
        "chemistry/scripts/f2_ucc_generator.py",
        "--payload",
        str(ham_output),
        "--output",
        str(ucc_output),
    ]
    if args.include_singles:
        step2.append("--include-singles")
    if args.include_doubles:
        step2.append("--include-doubles")

    step3 = [
        python_exe,
        "chemistry/scripts/run_f2_auto_vqe.py",
        "--ham-json",
        str(ham_output),
        "--ucc-file",
        str(ucc_output),
        "--ucc-topk",
        str(args.ucc_topk),
        "--importance-cutoff",
        str(args.importance_cutoff),
        "--shots",
        str(args.shots),
        "--max-iters",
        str(args.max_iters),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--gradient-method",
        args.gradient_method,
        "--shift",
        str(args.shift),
        "--prefer-chips",
        args.prefer_chips,
        "--name",
        args.name,
        "--init-value",
        str(args.init_value),
        "--save-result",
        str(result_output),
    ]

    run_step(step1, "Step 1: Export Hamiltonian")
    run_step(step2, "Step 2: Generate ranked UCC")
    run_step(step3, "Step 3: Run VQE")

    print("=== Pipeline finished ===")
    print(f"ham_output: {ham_output.resolve()}")
    print(f"ucc_output: {ucc_output.resolve()}")
    print(f"result_output: {result_output.resolve()}")


if __name__ == "__main__":
    main()