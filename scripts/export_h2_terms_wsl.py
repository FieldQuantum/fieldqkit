"""Export H2 qubit Hamiltonian data with OpenFermion+PySCF (intended for WSL/Linux).

Usage examples:
  python scripts/export_h2_terms_wsl.py --R 2.6 --unit angstrom
  python scripts/export_h2_terms_wsl.py --R 2.6 --unit bohr --output examples/data/chemistry/h2_R2.6_bohr_sto-3g.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf

BOHR_TO_ANGSTROM = 0.529177210903


def build_h2_payload(
    R_value: float,
    unit: str,
    basis: str,
    multiplicity: int,
    charge: int,
) -> dict:
    if unit.lower() == "bohr":
        r_ang = float(R_value) * BOHR_TO_ANGSTROM
    elif unit.lower() == "angstrom":
        r_ang = float(R_value)
    else:
        raise ValueError("unit must be 'angstrom' or 'bohr'")

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r_ang))]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    qham = jordan_wigner(molecule.get_molecular_hamiltonian())

    constant = 0.0
    terms: list[list[float | str]] = []
    max_q = -1

    for pauli_term, coeff in qham.terms.items():
        cre = float(coeff.real)
        cim = float(coeff.imag)
        if abs(cim) > 1e-10 or abs(cre) < 1e-12:
            continue

        if len(pauli_term) == 0:
            constant += cre
            continue

        obs = " ".join(f"{p}{i}" for i, p in pauli_term)
        terms.append([cre, obs])
        max_q = max(max_q, max(i for i, _ in pauli_term))

    terms.sort(key=lambda x: (len(str(x[1]).split()), str(x[1])))

    return {
        "schema": "quantum_control.hamiltonian.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "molecule": "H2",
        "R_input": float(R_value),
        "unit_input": unit,
        "bond_length_angstrom": r_ang,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
        "mapping": "jordan_wigner",
        "nqubits": max_q + 1,
        "constant": constant,
        "terms": terms,
        "fci_energy": float(molecule.fci_energy),
        "term_count": len(terms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export H2 qubit Hamiltonian JSON")
    parser.add_argument("--R", type=float, default=2.6, help="Bond distance value")
    parser.add_argument("--unit", type=str, default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="examples/data/chemistry/h2_R2.6_sto-3g.json",
        help="Output JSON path (relative to project root or absolute)",
    )
    args = parser.parse_args()

    payload = build_h2_payload(
        R_value=args.R,
        unit=args.unit,
        basis=args.basis,
        multiplicity=args.multiplicity,
        charge=args.charge,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved: {out_path}")
    print(f"nqubits: {payload['nqubits']}, terms: {payload['term_count']}")
    print(f"fci_energy: {payload['fci_energy']:.10f}")


if __name__ == "__main__":
    main()
