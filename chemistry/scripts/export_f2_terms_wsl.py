"""Export F2 qubit Hamiltonian data with OpenFermion+PySCF (intended for WSL/Linux).

Usage examples:
    python chemistry/scripts/export_f2_terms_wsl.py --R 1.4 --unit angstrom
    python chemistry/scripts/export_f2_terms_wsl.py --R 2.6 --unit bohr --output chemistry/data/f2_R2.6_bohr_sto-3g_12q.json
    python chemistry/scripts/export_f2_terms_wsl.py --reduction paper12 --encoding jw --output chemistry/data/f2_R1.4_angstrom_sto-3g_12q.json
    python chemistry/scripts/export_f2_terms_wsl.py --reduction none --encoding jw --output chemistry/data/f2_full_sto-3g.json

Notes:
    - Default reduction is the 12-qubit active-space setting requested in the paper-style spec:
      freeze irreps: 1a1, 2a1, 3a1, 4a1
      active irreps: 1e1, 2e1, 3e1, 4e1, 5a1, 6a1
      implemented as occupied_indices=[0,1,2,3], active_indices=[4,5,6,7,8,9] in STO-3G.
    - You can override orbital selection with --occupied-indices/--active-indices.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf
from openfermion.linalg import get_sparse_operator
from pyscf import gto, scf, symm
from scipy.sparse.linalg import eigsh

BOHR_TO_ANGSTROM = 0.529177210903


def _print_orbital_symmetry_report(
    *,
    r_ang: float,
    basis: str,
    multiplicity: int,
    charge: int,
) -> None:
    spin = multiplicity - 1
    mol = gto.M(
        atom=[("F", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, r_ang))],
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=True,
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    if not mf.converged:
        print("warning: PySCF RHF did not fully converge when printing orbital symmetry report")

    orb_syms = symm.label_orb_symm(
        mol,
        mol.irrep_name,
        mol.symm_orb,
        mf.mo_coeff,
        check=False,
    )

    print("=== PySCF Orbital Symmetry Report (canonical MOs) ===")
    print(f"point_group: {mol.groupname}")
    print("idx  occ    mo_energy(Ha)    irrep")
    for i, (occ, energy, irrep) in enumerate(zip(mf.mo_occ, mf.mo_energy, orb_syms)):
        print(f"{i:>3d}  {occ:>3.1f}    {energy:>+12.8f}    {irrep}")


def _diagonalize_and_validate(
    *,
    qham,
    fci_energy: float,
    reduction: str,
    nqubits: int,
) -> tuple[list[float], float]:
    sparse_ham = get_sparse_operator(qham)

    # Keep the eigensolver practical for large encodings.
    if nqubits > 16:
        print(
            "warning: skip exact JW diagonalization because Hilbert space is too large "
            f"(nqubits={nqubits}, dim={sparse_ham.shape[0]})."
        )
        return [], float("nan")

    k = min(6, max(1, sparse_ham.shape[0] - 2))
    eigvals, _ = eigsh(sparse_ham, k=k, which="SA")
    eigvals = sorted(float(v) for v in eigvals)

    e0 = eigvals[0]
    delta_vs_fci = abs(e0 - float(fci_energy))

    print("=== JW Hamiltonian Diagonalization Check ===")
    print(f"lowest_{k}_eigenvalues: {eigvals}")
    print(f"active_space_ground_energy: {e0:.10f}")
    print(f"full_system_fci_energy:     {float(fci_energy):.10f}")
    print(f"|active_space - fci|:       {delta_vs_fci:.10f}")

    # A light-weight numerical sanity check for the active-space choice.
    if reduction == "paper12":
        if delta_vs_fci < 5e-2:
            print("validation: PASS (active-space ground energy is close to full FCI)")
        else:
            print(
                "validation: WARN (active-space error is noticeable; consider tuning orbital selection)"
            )

    return eigvals, delta_vs_fci


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
        "molecule": "F2",
        "R_input": float(r_input),
        "unit_input": unit_input,
        "bond_length_angstrom": bond_length_angstrom,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
        "mapping": "jordan_wigner",
        "reduction": reduction,
        "irrep_freeze": ["1a1", "2a1", "3a1", "4a1"],
        "irrep_active": ["1e1", "2e1", "3e1", "4e1", "5a1", "6a1"],
        "occupied_indices": occupied_indices,
        "active_indices": active_indices,
        "nqubits": max_q + 1,
        "constant": constant,
        "terms": terms,
        "fci_energy": float(fci_energy),
        "term_count": len(terms),
    }


def build_f2_payload(
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

    geometry = [("F", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, r_ang))]

    _print_orbital_symmetry_report(
        r_ang=r_ang,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
    )

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    if reduction == "none":
        occupied_indices = occupied_indices_override
        active_indices = active_indices_override
    elif reduction == "paper12":
        # Paper-style 12-qubit active space for F2 in STO-3G.
        occupied_indices = [0, 1, 2, 3]
        active_indices = [4, 5, 6, 7, 8, 9]
        if occupied_indices_override is not None:
            occupied_indices = occupied_indices_override
        if active_indices_override is not None:
            active_indices = active_indices_override
    else:
        raise ValueError("reduction must be 'none' or 'paper12'")

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    qham = jordan_wigner(molecular_hamiltonian)

    eigvals, delta_vs_fci = _diagonalize_and_validate(
        qham=qham,
        fci_energy=float(molecule.fci_energy),
        reduction=reduction,
        nqubits=2 * len(active_indices) if active_indices is not None else 0,
    )

    payload = _qubit_operator_to_payload(
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

    # Guardrail for the requested setup.
    if reduction == "paper12" and payload["nqubits"] != 12:
        print(
            "warning: paper12 preset is expected to give 12 qubits, "
            f"but got {payload['nqubits']} qubits. "
            "Consider overriding indices with --occupied-indices/--active-indices."
        )

    if eigvals:
        payload["jw_lowest_eigenvalues"] = eigvals
    if delta_vs_fci == delta_vs_fci:
        payload["active_space_error_vs_fci"] = float(delta_vs_fci)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export F2 Jordan-Wigner Hamiltonian JSON")
    parser.add_argument("--R", type=float, default=1.4, help="F-F bond distance value")
    parser.add_argument("--unit", type=str, default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--encoding",
        type=str,
        default="jw",
        choices=["jw"],
        help="Only jw is currently implemented for F2 export",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="paper12",
        choices=["none", "paper12"],
        help="paper12: freeze [0,1,2,3], active [4,5,6,7,8,9] (~12 qubits)",
    )
    parser.add_argument(
        "--occupied-indices",
        type=str,
        default=None,
        help="Optional comma-separated occupied (frozen) spatial orbital indices, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--active-indices",
        type=str,
        default=None,
        help="Optional comma-separated active spatial orbital indices, e.g. 4,5,6,7,8,9",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chemistry/data/f2_R1.4_angstrom_sto-3g_12q.json",
        help="Output JSON path (relative to project root or absolute)",
    )
    args = parser.parse_args()

    occupied_indices = _parse_index_list(args.occupied_indices)
    active_indices = _parse_index_list(args.active_indices)

    payload = build_f2_payload(
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
