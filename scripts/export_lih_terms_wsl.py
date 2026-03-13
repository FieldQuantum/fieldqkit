"""Export LiH qubit Hamiltonian data with OpenFermion+PySCF (intended for WSL/Linux).

Usage examples:
    python scripts/export_lih_terms_wsl.py --R 1.6 --unit angstrom
  python scripts/export_lih_terms_wsl.py --R 3.0 --unit bohr --encoding jw --output examples/data/chemistry/lih_R3.0_bohr_sto-3g.json
    python scripts/export_lih_terms_wsl.py --reduction paper4 --encoding jw --output examples/data/chemistry/lih_R1.6_angstrom_sto-3g_4q.json
    python scripts/export_lih_terms_wsl.py --reduction none --encoding jw --output examples/data/chemistry/lih_full_sto-3g.json

Notes:
    - Default reduction is the 6-qubit active-space setting:
            occupied [0], active [1,2,3]
    - Optional paper4 reduction keeps the previous 4-qubit mapping:
            occupied [0,1], active [2,3]
  - You can override orbital selection with --occupied-indices/--active-indices.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf
from pyscf import gto, scf, symm, ao2mo
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh

BOHR_TO_ANGSTROM = 0.529177210903


def _parse_index_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return []
    values = [s.strip() for s in text.split(",") if s.strip()]
    return [int(v) for v in values]


def _qubit_operator_to_payload(
    qham,
    *,
    r_input: float,
    unit_input: str,
    bond_length_angstrom: float,
    basis: str,
    multiplicity: int,
    charge: int,
    fci_energy: float,
    reduction: str,
    occupied_indices: list[int] | None,
    active_indices: list[int] | None,
) -> dict:
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
        "molecule": "LiH",
        "R_input": float(r_input),
        "unit_input": unit_input,
        "bond_length_angstrom": bond_length_angstrom,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
        "mapping": "jordan_wigner",
        "reduction": reduction,
        "occupied_indices": occupied_indices,
        "active_indices": active_indices,
        "nqubits": max_q + 1,
        "constant": constant,
        "terms": terms,
        "fci_energy": float(fci_energy),
        "term_count": len(terms),
    }


def build_lih_payload(
    R_value: float,
    unit: str,
    basis: str,
    multiplicity: int,
    charge: int,
    reduction: str,
    occupied_indices_override: list[int] | None,
    active_indices_override: list[int] | None,
) -> dict:
    if unit.lower() == "bohr":
        r_ang = float(R_value) * BOHR_TO_ANGSTROM
    elif unit.lower() == "angstrom":
        r_ang = float(R_value)
    else:
        raise ValueError("unit must be 'angstrom' or 'bohr'")

    # Put Li-H bond on z-axis, consistent with the H2 script style.
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r_ang))]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    if reduction == "none":
        occupied_indices = occupied_indices_override
        active_indices = active_indices_override
    elif reduction == "paper":
        # 6-qubit preset: 3 spatial orbitals in active space.
        # Common LiH setup in STO-3G is freeze core orbital and keep 3 active orbitals.
        occupied_indices = [0]
        active_indices = [1, 2, 5]
        if occupied_indices_override is not None:
            occupied_indices = occupied_indices_override
        if active_indices_override is not None:
            active_indices = active_indices_override
    else:
        raise ValueError("reduction must be 'none' or 'paper'")

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    qham = jordan_wigner(molecular_hamiltonian)
    sparse_ham = get_sparse_operator(qham)
    eigvals, _ = eigsh(sparse_ham, k=6, which='SA')
    print(f"Eigenvalues: {eigvals}")

    return _qubit_operator_to_payload(
        qham,
        r_input=R_value,
        unit_input=unit,
        bond_length_angstrom=r_ang,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        fci_energy=float(molecule.fci_energy),
        reduction=reduction,
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LiH Jordan-Wigner Hamiltonian JSON")
    parser.add_argument("--R", type=float, default=1.6, help="Li-H bond distance value")
    parser.add_argument("--unit", type=str, default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--encoding",
        type=str,
        default="jw",
        choices=["jw"],
        help="Only jw is currently implemented for LiH export",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="paper",
        choices=["none", "paper"],
        help="paper: occupied [0], active [1,2,3] (~6 qubits)",
    )
    parser.add_argument(
        "--occupied-indices",
        type=str,
        default=None,
        help="Optional comma-separated occupied (frozen) spatial orbital indices, e.g. 0,1",
    )
    parser.add_argument(
        "--active-indices",
        type=str,
        default=None,
        help="Optional comma-separated active spatial orbital indices, e.g. 2,3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/data/chemistry/lih_R1.6_angstrom_sto-3g_6q.json",
        help="Output JSON path (relative to project root or absolute)",
    )
    args = parser.parse_args()

    occupied_indices = _parse_index_list(args.occupied_indices)
    active_indices = _parse_index_list(args.active_indices)

    payload = build_lih_payload(
        R_value=args.R,
        unit=args.unit,
        basis=args.basis,
        multiplicity=args.multiplicity,
        charge=args.charge,
        reduction=args.reduction,
        occupied_indices_override=occupied_indices,
        active_indices_override=active_indices,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved: {out_path}")
    print(f"nqubits: {payload['nqubits']}, terms: {payload['term_count']}")
    print(f"fci_energy: {payload['fci_energy']:.10f}")
    print(f"reduction: {payload['reduction']}")
    print(f"occupied_indices: {payload['occupied_indices']}")
    print(f"active_indices: {payload['active_indices']}")


if __name__ == "__main__":
    main()
