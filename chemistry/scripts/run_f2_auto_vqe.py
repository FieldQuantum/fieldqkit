#!/usr/bin/env python3
"""Run F2 custom-ansatz VQE from exported JSON + generated UCC template.

This script is designed as a notebook-free step-3 pipeline:
1) read Hamiltonian payload from step-1 exporter output,
2) read ranked UCC Pauli terms from step-2 generated template file,
3) pick Top-K excitation terms and run VQE.

It avoids PySCF/OpenFermion dependencies at runtime by consuming artifacts
already generated in earlier steps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from quantum_hw import QuantumHardwareClient
from quantum_hw.algorithms import VQERunner
from quantum_hw.circuit import QuantumCircuit


@dataclass
class RankedPauliTerm:
    pauli: str
    importance: float | None


@dataclass
class UCCTemplate:
    nqubits: int
    hf_occupied: list[int]
    pauli_terms: list[RankedPauliTerm]


def load_hamiltonian_payload(path: Path) -> tuple[float, list[tuple[float, str]], int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    constant = float(data["constant"])
    terms = [(float(c), str(obs)) for c, obs in data["terms"]]
    nqubits = int(data["nqubits"])
    fci_energy = float(data["fci_energy"])
    return constant, terms, nqubits, fci_energy


def load_ucc_template_from_json(path: Path) -> UCCTemplate:
    data = json.loads(path.read_text(encoding="utf-8"))
    nqubits = int(data["nqubits"])
    hf_occupied = [int(q) for q in data["hf_occupied"]]
    raw_terms = data.get("terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError(f"No ranked UCC terms found in {path}")

    pauli_terms: list[RankedPauliTerm] = []
    for entry in raw_terms:
        pauli_string = entry.get("pauli_string")
        if not isinstance(pauli_string, str) or not pauli_string:
            raise ValueError(f"Invalid pauli_string entry in {path}")
        raw_importance = entry.get("importance")
        importance = None if raw_importance is None else float(raw_importance)
        pauli_terms.append(RankedPauliTerm(pauli=pauli_string, importance=importance))

    return UCCTemplate(nqubits=nqubits, hf_occupied=hf_occupied, pauli_terms=pauli_terms)


def load_ucc_template(path: Path) -> UCCTemplate:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported UCC template format for {path}; expected .json")
    return load_ucc_template_from_json(path)


def build_topk_ucc_circuit(
    template: UCCTemplate,
    topk: int,
    importance_cutoff: float,
) -> tuple[QuantumCircuit, list[str], list[RankedPauliTerm]]:
    if topk <= 0:
        raise ValueError("--ucc-topk must be >= 1")
    if importance_cutoff < 0.0:
        raise ValueError("--importance-cutoff must be >= 0")

    if importance_cutoff > 0.0:
        missing_importance = [term.pauli for term in template.pauli_terms if term.importance is None]
        if missing_importance:
            raise ValueError(
                "The UCC template does not provide parseable importance annotations, "
                "so --importance-cutoff cannot be applied."
            )

    filtered_terms = [
        term
        for term in template.pauli_terms
        if term.importance is None or term.importance >= importance_cutoff
    ]
    if not filtered_terms:
        raise ValueError(
            "No UCC excitation terms remain after applying --importance-cutoff="
            f"{importance_cutoff:.6g}"
        )

    k = min(topk, len(filtered_terms))
    selected_terms = filtered_terms[:k]

    qc = QuantumCircuit(template.nqubits)
    for q in template.hf_occupied:
        qc.x(q)

    for i, term in enumerate(selected_terms):
        qc.pauli_evolution(f"theta_{i}", term.pauli)

    symbolic_params = sorted(
        [name for name, value in qc.params_value.items() if isinstance(name, str) and isinstance(value, str)]
    )
    return qc, symbolic_params, selected_terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F2 auto VQE with configurable Top-K UCC excitations")

    parser.add_argument(
        "--ham-json",
        type=str,
        default="chemistry/data/f2_R2.6_angstrom_sto-3g_auto.json",
        help="Step-1 payload JSON path",
    )
    parser.add_argument(
        "--ucc-file",
        type=str,
        default="chemistry/data/ucc_f2.json",
        help="Step-2 generated UCC template JSON path",
    )
    parser.add_argument(
        "--ucc-topk",
        type=int,
        default=5,
        help="Use only the first K ranked UCC excitation terms",
    )
    parser.add_argument(
        "--importance-cutoff",
        type=float,
        default=1e-4,
        help="Discard ranked UCC terms whose importance is below this threshold before applying Top-K",
    )

    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gradient-method", type=str, default="autograd", choices=["autograd", "parameter-shift"])
    parser.add_argument("--shift", type=float, default=1.5707963267948966)
    parser.add_argument("--prefer-chips", type=str, default="Simulator")
    parser.add_argument("--name", type=str, default="f2_auto_custom_topk")
    parser.add_argument("--init-value", type=float, default=0.01)

    parser.add_argument(
        "--save-result",
        type=str,
        default="",
        help="Optional output JSON path for final VQE summary",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ham_json = Path(args.ham_json)
    ucc_file = Path(args.ucc_file)

    constant, hamiltonian_terms, nqubits_ham, fci_energy = load_hamiltonian_payload(ham_json)
    template = load_ucc_template(ucc_file)

    if template.nqubits != nqubits_ham:
        raise ValueError(
            "Qubit count mismatch between Hamiltonian and UCC template: "
            f"ham={nqubits_ham}, ucc={template.nqubits}"
        )

    custom_ansatz_qc, symbolic_params, selected_terms = build_topk_ucc_circuit(
        template,
        topk=args.ucc_topk,
        importance_cutoff=args.importance_cutoff,
    )

    runner = VQERunner(
        client=QuantumHardwareClient(),
        shots=args.shots,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        gradient_method=args.gradient_method,
        seed=args.seed,
        shift=args.shift,
    )

    run_kwargs = {
        "name": args.name,
        "num_qubits": nqubits_ham,
        "model": "custom",
        "hamiltonian": hamiltonian_terms,
        "ansatz": "custom",
        "custom_ansatz_circuit": custom_ansatz_qc,
        "prefer_chips": args.prefer_chips,
        "init_params": [args.init_value] * len(symbolic_params),
    }

    result = runner.run_model(**run_kwargs)
    e_total = constant + result.best_energy
    abs_error_fci = abs(e_total - fci_energy)

    print("=== F2 Auto VQE (Top-K UCC) ===")
    print(f"ham_json: {ham_json.resolve()}")
    print(f"ucc_file: {ucc_file.resolve()}")
    print(f"nqubits: {nqubits_ham}")
    print(f"importance_cutoff: {args.importance_cutoff:.6g}")
    print(f"terms_after_importance_filter: {len([t for t in template.pauli_terms if t.importance is None or t.importance >= args.importance_cutoff])}")
    print(f"ucc_topk_used: {len(selected_terms)}")
    print(f"total_ranked_terms_available: {len(template.pauli_terms)}")
    print(f"parameter_count: {len(symbolic_params)}")
    print(f"estimated_total_energy: {e_total:.10f}")
    print(f"fci_reference_energy:   {fci_energy:.10f}")
    print(f"absolute_error_vs_fci:  {abs_error_fci:.10e}")

    if args.save_result:
        out = {
            "ham_json": str(ham_json),
            "ucc_file": str(ucc_file),
            "nqubits": nqubits_ham,
            "importance_cutoff": float(args.importance_cutoff),
            "terms_after_importance_filter": int(
                len([t for t in template.pauli_terms if t.importance is None or t.importance >= args.importance_cutoff])
            ),
            "ucc_topk_used": int(len(selected_terms)),
            "total_ranked_terms_available": int(len(template.pauli_terms)),
            "parameter_count": int(len(symbolic_params)),
            "selected_pauli_terms": [term.pauli for term in selected_terms],
            "selected_term_importances": [
                None if term.importance is None else float(term.importance) for term in selected_terms
            ],
            "estimated_total_energy": float(e_total),
            "fci_reference_energy": float(fci_energy),
            "absolute_error_vs_fci": float(abs_error_fci),
            "best_energy_no_constant": float(result.best_energy),
            "energy_history_no_constant": [float(x) for x in result.energy_history],
            "energy_history_total": [float(x + constant) for x in result.energy_history],
        }
        save_path = Path(args.save_result)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved_result: {save_path.resolve()}")


if __name__ == "__main__":
    main()
