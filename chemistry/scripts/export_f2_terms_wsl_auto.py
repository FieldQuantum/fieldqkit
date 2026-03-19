#!/usr/bin/env python3
"""Export F2 qubit Hamiltonian data with OpenFermion+PySCF.

Features added:
  - Automatic active-space selection with minimal qubit count under a target
    chemical accuracy vs full-system FCI.
  - Optional symmetry-aware grouping so degenerate / same-symmetry orbitals are
    added together during the active-space search.
  - Supports JW / BK / SCBK encodings. SCBK can remove two qubits by exploiting
    electron-number and spin symmetries.

Usage examples:
    python export_f2_terms_wsl_auto.py --R 1.4 --unit angstrom --reduction auto_minq --encoding scbk
    python export_f2_terms_wsl_auto.py --R 2.6 --unit bohr --reduction auto_minq --encoding scbk
    python export_f2_terms_wsl_auto.py --reduction paper12 --encoding jw
    python export_f2_terms_wsl_auto.py --reduction none --encoding jw --output f2_full.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from openfermion import (
    MolecularData,
    bravyi_kitaev,
    jordan_wigner,
)
from openfermion.linalg import (
    get_number_preserving_sparse_operator,
    get_sparse_operator,
)
from openfermion.transforms import get_fermion_operator, symmetry_conserving_bravyi_kitaev
from openfermionpyscf import run_pyscf
from pyscf import gto, mp, scf, symm
from scipy.sparse.linalg import eigsh

BOHR_TO_ANGSTROM = 0.529177210903


@dataclass
class OrbitalInfo:
    index: int
    occ: float
    energy: float
    irrep: str
    score: float


def infer_nqubits(qham) -> int:
    max_q = -1
    for pauli_term in qham.terms:
        if pauli_term:
            max_q = max(max_q, max(i for i, _ in pauli_term))
    return max_q + 1 if max_q >= 0 else 0


def build_pyscf_rhf(
    *,
    r_ang: float,
    basis: str,
    multiplicity: int,
    charge: int,
) -> tuple[gto.Mole, scf.hf.RHF]:
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
    return mol, mf


def compute_canonical_mp2_occupations(mf: scf.hf.RHF) -> np.ndarray:
    """Return approximate correlated occupations in the canonical MO basis.

    We intentionally use the diagonal of the MP2 1-RDM in the canonical MO basis
    rather than natural-orbital indices, because OpenFermion active_indices refer
    to the original MO ordering.
    """
    pt = mp.MP2(mf)
    pt.kernel()
    dm1 = pt.make_rdm1()

    dm1 = np.asarray(dm1)
    if dm1.ndim != 2 or dm1.shape[0] != dm1.shape[1]:
        raise RuntimeError(f"Unexpected MP2 RDM1 shape: {dm1.shape}")

    occ = np.real(np.diag(dm1))
    return occ


def label_orbital_info(
    mol: gto.Mole,
    mf: scf.hf.RHF,
    occs: np.ndarray,
) -> list[OrbitalInfo]:
    orb_syms = symm.label_orb_symm(
        mol,
        mol.irrep_name,
        mol.symm_orb,
        mf.mo_coeff,
        check=False,
    )
    infos: list[OrbitalInfo] = []
    for i, (occ, energy, irrep) in enumerate(zip(occs, mf.mo_energy, orb_syms)):
        score = float(min(float(occ), 2.0 - float(occ)))
        infos.append(
            OrbitalInfo(
                index=i,
                occ=float(occ),
                energy=float(energy),
                irrep=str(irrep),
                score=score,
            )
        )
    return infos


def print_orbital_report(
    mol: gto.Mole,
    mf: scf.hf.RHF,
    orbital_infos: list[OrbitalInfo],
) -> None:
    print("=== PySCF Orbital Report (canonical MOs + MP2 occupations) ===")
    print(f"point_group: {mol.groupname}")
    print("idx  rhf_occ   mp2_occ    mo_energy(Ha)    irrep    score=min(n,2-n)")
    for info, rhf_occ in zip(orbital_infos, mf.mo_occ):
        print(
            f"{info.index:>3d}  {float(rhf_occ):>6.2f}   {info.occ:>8.5f}   "
            f"{info.energy:>+12.8f}    {info.irrep:<6s}   {info.score:>8.5f}"
        )


def build_symmetry_groups(
    orbital_infos: list[OrbitalInfo],
    energy_tol: float = 1e-8,
) -> list[list[int]]:
    """Group orbitals by (irrep, near-degenerate energy).

    This is a conservative heuristic so the active-space search does not split
    obviously degenerate same-symmetry orbitals.
    """
    sorted_infos = sorted(orbital_infos, key=lambda x: (x.irrep, round(x.energy, 12), x.index))
    groups: list[list[int]] = []

    current: list[OrbitalInfo] = []
    for info in sorted_infos:
        if not current:
            current = [info]
            continue
        same_irrep = info.irrep == current[-1].irrep
        close_energy = abs(info.energy - current[-1].energy) <= energy_tol
        if same_irrep and close_energy:
            current.append(info)
        else:
            groups.append([x.index for x in current])
            current = [info]
    if current:
        groups.append([x.index for x in current])

    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: min(g))
    return groups


def generate_candidate_spaces(
    orbital_infos: list[OrbitalInfo],
    *,
    min_active_orbs: int,
    max_active_orbs: int | None,
    score_cut: float,
    frozen_occ_cut: float,
    frozen_virt_cut: float,
    respect_symmetry_groups: bool,
    symmetry_groups: list[list[int]] | None,
) -> list[tuple[list[int], list[int]]]:
    """Generate candidate (occupied_indices, active_indices)."""
    norb = len(orbital_infos)
    if max_active_orbs is None:
        max_active_orbs = norb
    max_active_orbs = min(max_active_orbs, norb)

    mandatory_active = [
        x.index
        for x in orbital_infos
        if x.score >= score_cut
    ]

    preferred_frozen_occ = [
        x.index
        for x in orbital_infos
        if x.occ >= frozen_occ_cut and x.index not in mandatory_active
    ]
    ignored_frozen_virt = {
        x.index
        for x in orbital_infos
        if x.occ <= frozen_virt_cut and x.index not in mandatory_active
    }

    if respect_symmetry_groups and symmetry_groups:
        group_by_member: dict[int, list[int]] = {}
        for group in symmetry_groups:
            for idx in group:
                group_by_member[idx] = group

        seed_groups: list[list[int]] = []
        seen = set()
        for idx in mandatory_active:
            g = tuple(group_by_member.get(idx, [idx]))
            if g not in seen:
                seed_groups.append(list(g))
                seen.add(g)

        candidate_groups: list[list[int]] = []
        for info in sorted(orbital_infos, key=lambda x: (-x.score, abs(x.energy), x.index)):
            if info.index in ignored_frozen_virt:
                continue
            g = tuple(group_by_member.get(info.index, [info.index]))
            if g not in seen:
                candidate_groups.append(list(g))
                seen.add(g)

        spaces: list[tuple[list[int], list[int]]] = []
        active_set = set()
        for g in seed_groups:
            active_set.update(g)

        if len(active_set) >= min_active_orbs:
            frozen = sorted(i for i in preferred_frozen_occ if i not in active_set)
            spaces.append((frozen, sorted(active_set)))

        for g in candidate_groups:
            active_set.update(g)
            if len(active_set) > max_active_orbs:
                break
            if len(active_set) >= min_active_orbs:
                frozen = sorted(i for i in preferred_frozen_occ if i not in active_set)
                spaces.append((frozen, sorted(active_set)))

    else:
        ranking = [
            x.index
            for x in sorted(orbital_infos, key=lambda x: (-x.score, abs(x.energy), x.index))
            if x.index not in ignored_frozen_virt
        ]
        active_seed = sorted(set(mandatory_active))
        spaces = []
        active_set = set(active_seed)

        if len(active_set) >= min_active_orbs:
            frozen = sorted(i for i in preferred_frozen_occ if i not in active_set)
            spaces.append((frozen, sorted(active_set)))

        for idx in ranking:
            if idx in active_set:
                continue
            active_set.add(idx)
            if len(active_set) > max_active_orbs:
                break
            if len(active_set) >= min_active_orbs:
                frozen = sorted(i for i in preferred_frozen_occ if i not in active_set)
                spaces.append((frozen, sorted(active_set)))

    # Deduplicate and enforce valid electron count.
    deduped: list[tuple[list[int], list[int]]] = []
    seen_pairs = set()
    for frozen, active in spaces:
        key = (tuple(frozen), tuple(active))
        if not active:
            continue
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append((frozen, active))
    return deduped


def exact_active_space_energy(
    molecule: MolecularData,
    *,
    occupied_indices: list[int] | None,
    active_indices: list[int] | None,
) -> float:
    if active_indices is None:
        raise ValueError("active_indices must not be None for exact_active_space_energy")

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    fermion_ham = get_fermion_operator(molecular_hamiltonian)

    n_active_orbs = len(active_indices)
    n_spin_orbs = 2 * n_active_orbs
    n_frozen_occ = len(occupied_indices or [])
    n_active_electrons = molecule.n_electrons - 2 * n_frozen_occ

    if n_active_electrons < 0 or n_active_electrons > n_spin_orbs:
        raise RuntimeError(
            f"Invalid active-electron count: {n_active_electrons} "
            f"(n_spin_orbs={n_spin_orbs}, frozen_occ={n_frozen_occ})"
        )

    sparse_op = get_number_preserving_sparse_operator(
        fermion_ham,
        num_qubits=n_spin_orbs,
        num_electrons=n_active_electrons,
        spin_preserving=False,
    )

    dim = sparse_op.shape[0]
    if dim == 0:
        raise RuntimeError("Empty number-preserving subspace")
    if dim <= 8:
        evals = np.linalg.eigvalsh(sparse_op.toarray())
        return float(np.min(evals))

    k = min(4, dim - 1)
    evals, _ = eigsh(sparse_op, k=k, which="SA")
    return float(np.min(evals))


def build_qubit_hamiltonian(
    molecule: MolecularData,
    *,
    occupied_indices: list[int] | None,
    active_indices: list[int] | None,
    encoding: str,
):
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    fermion_ham = get_fermion_operator(molecular_hamiltonian)

    if active_indices is None:
        # full-space path
        if encoding == "jw":
            qham = jordan_wigner(molecular_hamiltonian)
        elif encoding == "bk":
            qham = bravyi_kitaev(fermion_ham)
        elif encoding == "scbk":
            raise ValueError("encoding=scbk requires an explicit active space")
        else:
            raise ValueError("encoding must be one of: jw, bk, scbk")
        return qham

    n_active_orbs = len(active_indices)
    n_frozen_occ = len(occupied_indices or [])
    n_active_electrons = molecule.n_electrons - 2 * n_frozen_occ

    if encoding == "jw":
        qham = jordan_wigner(molecular_hamiltonian)
    elif encoding == "bk":
        qham = bravyi_kitaev(fermion_ham)
    elif encoding == "scbk":
        qham = symmetry_conserving_bravyi_kitaev(
            fermion_ham,
            active_orbitals=2 * n_active_orbs,
            active_fermions=n_active_electrons,
        )
    else:
        raise ValueError("encoding must be one of: jw, bk, scbk")

    return qham, fermion_ham


def diagonalize_qubit_hamiltonian_if_small(
    qham,
    *,
    max_qubits: int = 16,
) -> tuple[list[float], float]:
    nqubits = infer_nqubits(qham)
    sparse_ham = get_sparse_operator(qham)

    if nqubits > max_qubits:
        print(
            f"warning: skip qubit-Hamiltonian exact diagonalization because "
            f"nqubits={nqubits} > {max_qubits}."
        )
        return [], float("nan")

    dim = sparse_ham.shape[0]
    if dim <= 8:
        evals = np.linalg.eigvalsh(sparse_ham.toarray())
        eigvals = sorted(float(v) for v in evals)
        return eigvals[: min(6, len(eigvals))], eigvals[0]

    k = min(6, max(1, dim - 2))
    eigvals, _ = eigsh(sparse_ham, k=k, which="SA")
    eigvals = sorted(float(v) for v in eigvals)
    return eigvals, eigvals[0]


def qubit_operator_to_payload(
    qham,
    fermion_ham,
    *,
    r_input: float,
    unit_input: str,
    bond_length_angstrom: float,
    basis: str,
    multiplicity: int,
    charge: int,
    fci_energy: float,
    reduction: str,
    encoding: str,
    occupied_indices: list[int] | None,
    active_indices: list[int] | None,
    active_space_energy: float | None = None,
    active_space_error_vs_fci: float | None = None,
    extra: dict | None = None,
) -> dict:
    constant = 0.0
    terms: list[list[float | str]] = []

    for pauli_term, coeff in qham.terms.items():
        cre = float(np.real(coeff))
        cim = float(np.imag(coeff))
        if abs(cim) > 1e-10 or abs(cre) < 1e-12:
            continue

        if len(pauli_term) == 0:
            constant += cre
            continue

        obs = " ".join(f"{p}{i}" for i, p in pauli_term)
        terms.append([cre, obs])

    terms.sort(key=lambda x: (len(str(x[1]).split()), str(x[1])))

    payload = {
        "schema": "quantum_control.hamiltonian.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "molecule": "F2",
        "R_input": float(r_input),
        "unit_input": unit_input,
        "bond_length_angstrom": bond_length_angstrom,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
        "mapping": encoding,
        "reduction": reduction,
        "occupied_indices": occupied_indices,
        "active_indices": active_indices,
        "nqubits": infer_nqubits(qham),
        "constant": constant,
        "terms": terms,
        "fermion_terms": serialize_fermion_operator(fermion_ham),
        "fci_energy": float(fci_energy),
        "term_count": len(terms),
    }
    if active_space_energy is not None:
        payload["active_space_energy"] = float(active_space_energy)
    if active_space_error_vs_fci is not None:
        payload["active_space_error_vs_fci"] = float(active_space_error_vs_fci)
    if extra:
        payload.update(extra)
    return payload


def auto_select_min_qubit_space(
    *,
    molecule: MolecularData,
    mol: gto.Mole,
    mf: scf.hf.RHF,
    encoding: str,
    chemical_accuracy: float,
    min_active_orbs: int,
    max_active_orbs: int | None,
    score_cut: float,
    frozen_occ_cut: float,
    frozen_virt_cut: float,
    respect_symmetry_groups: bool,
    group_energy_tol: float,
) -> tuple[dict, list[dict], list[OrbitalInfo], list[list[int]]]:
    occs = compute_canonical_mp2_occupations(mf)
    orbital_infos = label_orbital_info(mol, mf, occs)
    print_orbital_report(mol, mf, orbital_infos)

    symmetry_groups = build_symmetry_groups(orbital_infos, energy_tol=group_energy_tol)
    if respect_symmetry_groups:
        print("=== Symmetry / degeneracy groups used in auto search ===")
        for g in symmetry_groups:
            labels = [(i, orbital_infos[i].irrep, orbital_infos[i].energy) for i in g]
            print(labels)

    candidates = generate_candidate_spaces(
        orbital_infos,
        min_active_orbs=min_active_orbs,
        max_active_orbs=max_active_orbs,
        score_cut=score_cut,
        frozen_occ_cut=frozen_occ_cut,
        frozen_virt_cut=frozen_virt_cut,
        respect_symmetry_groups=respect_symmetry_groups,
        symmetry_groups=symmetry_groups,
    )

    print("=== Auto active-space search ===")
    print(f"candidate_count: {len(candidates)}")
    ref_fci = float(molecule.fci_energy)
    results: list[dict] = []

    for occupied_indices, active_indices in candidates:
        try:
            e_active = exact_active_space_energy(
                molecule,
                occupied_indices=occupied_indices,
                active_indices=active_indices,
            )
            qham, _ = build_qubit_hamiltonian(
                molecule,
                occupied_indices=occupied_indices,
                active_indices=active_indices,
                encoding=encoding,
            )
            nqubits = infer_nqubits(qham)
            term_count = len(qham.terms)
            err = abs(e_active - ref_fci)
            accepted = err <= chemical_accuracy

            rec = {
                "occupied_indices": occupied_indices,
                "active_indices": active_indices,
                "n_active_orbs": len(active_indices),
                "n_active_electrons": int(molecule.n_electrons - 2 * len(occupied_indices)),
                "nqubits": nqubits,
                "term_count": term_count,
                "active_space_energy": float(e_active),
                "error_vs_fci": float(err),
                "accepted": bool(accepted),
            }
            results.append(rec)

            print(
                f"candidate active={active_indices}, frozen={occupied_indices}, "
                f"nq={nqubits}, terms={term_count}, err={err:.6e}, accepted={accepted}"
            )
        except Exception as exc:
            print(
                f"warning: skip candidate active={active_indices}, frozen={occupied_indices} "
                f"because evaluation failed: {exc}"
            )

    feasible = [r for r in results if r["accepted"]]
    if not feasible:
        raise RuntimeError(
            f"No candidate met chemical accuracy <= {chemical_accuracy:.3e} Ha. "
            "Try increasing --max-active-orbs or relaxing thresholds."
        )

    best = min(
        feasible,
        key=lambda r: (r["nqubits"], r["error_vs_fci"], r["term_count"], r["n_active_orbs"]),
    )

    print("=== Best auto selection ===")
    print(json.dumps(best, indent=2))

    return best, results, orbital_infos, symmetry_groups


def parse_index_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return []
    values = [s.strip() for s in text.split(",") if s.strip()]
    return [int(v) for v in values]


def serialize_orbital_infos(orbital_infos: Iterable[OrbitalInfo]) -> list[dict]:
    return [
        {
            "index": x.index,
            "mp2_occ": float(x.occ),
            "mo_energy": float(x.energy),
            "irrep": x.irrep,
            "score": float(x.score),
        }
        for x in orbital_infos
    ]


def serialize_fermion_operator(op):
    terms = []
    for term, coeff in op.terms.items():
        # term: ((p,1),(q,0),...)
        # 转成 string: "p^ q ..."
        term_str = " ".join(
            f"{i}^" if a == 1 else f"{i}"
            for i, a in term
        )
        terms.append((term_str, float(coeff)))
    return terms


def canonicalize_irrep_name(name: str) -> str:
    return str(name).strip()


def irrep_name_to_id_map(point_group: str, irrep_names: list[str]) -> dict[str, int]:
    """
    Build a deterministic integer id map for irreps in the given point group.

    For Abelian groups used in standard molecular symmetry for RHF (e.g. D2h, C2v, C2h, Ci, Cs, C2),
    each irrep is 1D and the direct product group is Abelian. We only need a stable id assignment
    plus a multiplication table. The actual multiplication table is defined separately below.
    """
    names = [canonicalize_irrep_name(x) for x in irrep_names]
    unique = sorted(set(names))
    return {name: idx for idx, name in enumerate(unique)}


def build_f2_payload(
    *,
    R_value: float,
    unit: str,
    basis: str,
    multiplicity: int,
    charge: int,
    reduction: str,
    encoding: str,
    chemical_accuracy: float,
    occupied_indices_override: list[int] | None,
    active_indices_override: list[int] | None,
    min_active_orbs: int,
    max_active_orbs: int | None,
    score_cut: float,
    frozen_occ_cut: float,
    frozen_virt_cut: float,
    respect_symmetry_groups: bool,
    group_energy_tol: float,
) -> dict:
    if unit.lower() == "bohr":
        r_ang = float(R_value) * BOHR_TO_ANGSTROM
    elif unit.lower() == "angstrom":
        r_ang = float(R_value)
    else:
        raise ValueError("unit must be 'angstrom' or 'bohr'")

    geometry = [("F", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, r_ang))]

    mol, mf = build_pyscf_rhf(
        r_ang=r_ang,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
    )
    if not mf.converged:
        print("warning: PySCF RHF did not fully converge")

    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    orbital_infos: list[OrbitalInfo] = []
    symmetry_groups: list[list[int]] = []
    candidate_results: list[dict] = []
    active_space_energy: float | None = None
    active_space_error_vs_fci: float | None = None

    if reduction == "none":
        occupied_indices = occupied_indices_override
        active_indices = active_indices_override

    elif reduction == "paper12":
        occupied_indices = [0, 1, 2, 3]
        active_indices = [4, 5, 6, 7, 8, 9]
        if occupied_indices_override is not None:
            occupied_indices = occupied_indices_override
        if active_indices_override is not None:
            active_indices = active_indices_override

    elif reduction == "auto_minq":
        best, candidate_results, orbital_infos, symmetry_groups = auto_select_min_qubit_space(
            molecule=molecule,
            mol=mol,
            mf=mf,
            encoding=encoding,
            chemical_accuracy=chemical_accuracy,
            min_active_orbs=min_active_orbs,
            max_active_orbs=max_active_orbs,
            score_cut=score_cut,
            frozen_occ_cut=frozen_occ_cut,
            frozen_virt_cut=frozen_virt_cut,
            respect_symmetry_groups=respect_symmetry_groups,
            group_energy_tol=group_energy_tol,
        )
        occupied_indices = list(best["occupied_indices"])
        active_indices = list(best["active_indices"])
        active_space_energy = float(best["active_space_energy"])
        active_space_error_vs_fci = float(best["error_vs_fci"])
    else:
        raise ValueError("reduction must be 'none', 'paper12', or 'auto_minq'")

    # Build symmetry metadata for payload.
    point_group = str(mol.groupname)

    if not orbital_infos:
        occs = compute_canonical_mp2_occupations(mf)
        orbital_infos = label_orbital_info(mol, mf, occs)

    full_orbital_irreps = [canonicalize_irrep_name(x.irrep) for x in orbital_infos]
    irrep_id_map = irrep_name_to_id_map(point_group, full_orbital_irreps)
    full_orbital_irrep_ids = [irrep_id_map[x] for x in full_orbital_irreps]

    active_space_orbital_irreps = None
    active_space_orbital_irrep_ids = None
    if active_indices is not None:
        active_space_orbital_irreps = [full_orbital_irreps[i] for i in active_indices]
        active_space_orbital_irrep_ids = [full_orbital_irrep_ids[i] for i in active_indices]

    qham, fermion_ham = build_qubit_hamiltonian(
        molecule,
        occupied_indices=occupied_indices,
        active_indices=active_indices,
        encoding=encoding,
    )

    if active_indices is not None and active_space_energy is None:
        active_space_energy = exact_active_space_energy(
            molecule,
            occupied_indices=occupied_indices,
            active_indices=active_indices,
        )
        active_space_error_vs_fci = abs(active_space_energy - float(molecule.fci_energy))

    eigvals, e0 = diagonalize_qubit_hamiltonian_if_small(qham)
    if eigvals:
        print("=== Qubit Hamiltonian Diagonalization Check ===")
        print(f"lowest_eigenvalues: {eigvals}")
        if active_space_energy is not None:
            print(f"qubit_ground_energy:        {e0:.10f}")
            print(f"active_space_ground_energy: {active_space_energy:.10f}")
            print(f"|qubit - active|:           {abs(e0 - active_space_energy):.10e}")

    # === build orbital_irreps aligned with active space ===
    orbital_irreps = None
    if active_indices is not None and orbital_infos:
        # map full index -> irrep
        irrep_map = {info.index: info.irrep for info in orbital_infos}

        orbital_irreps = [irrep_map[i] for i in active_indices]
    
    extra = {
        "chemical_accuracy_target": chemical_accuracy,
        "candidate_search_results": candidate_results,
        "orbital_infos": serialize_orbital_infos(orbital_infos),
        "symmetry_groups": symmetry_groups,
        "respect_symmetry_groups": respect_symmetry_groups,
        "selection_thresholds": {
            "min_active_orbs": min_active_orbs,
            "max_active_orbs": max_active_orbs,
            "score_cut": score_cut,
            "frozen_occ_cut": frozen_occ_cut,
            "frozen_virt_cut": frozen_virt_cut,
            "group_energy_tol": group_energy_tol,
        },
        "point_group": point_group,
        "orbital_irreps_all": full_orbital_irreps,
        "orbital_irrep_ids_all": full_orbital_irrep_ids,
        "irrep_id_map": irrep_id_map,
    }

    if active_space_orbital_irreps is not None:
        extra["orbital_irreps"] = active_space_orbital_irreps
    if active_space_orbital_irrep_ids is not None:
        extra["orbital_irrep_ids"] = active_space_orbital_irrep_ids
    if eigvals:
        extra["qubit_lowest_eigenvalues"] = eigvals

    payload = qubit_operator_to_payload(
        qham,
        fermion_ham,
        r_input=R_value,
        unit_input=unit,
        bond_length_angstrom=r_ang,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        fci_energy=float(molecule.fci_energy),
        reduction=reduction,
        encoding=encoding,
        occupied_indices=occupied_indices,
        active_indices=active_indices,
        active_space_energy=active_space_energy,
        active_space_error_vs_fci=active_space_error_vs_fci,
        extra=extra,
    )

    if reduction == "paper12" and payload["nqubits"] not in (10, 12):
        print(
            "warning: paper12 is often expected to be around 12 qubits (or 10 with SCBK), "
            f"but got {payload['nqubits']} qubits."
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export F2 qubit Hamiltonian JSON with optional automatic active-space selection"
    )
    parser.add_argument("--R", type=float, default=1.4, help="F-F bond distance value")
    parser.add_argument("--unit", type=str, default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)

    parser.add_argument(
        "--encoding",
        type=str,
        default="scbk",
        choices=["jw", "bk", "scbk"],
        help="Qubit encoding: jw, bk, or scbk",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="auto_minq",
        choices=["none", "paper12", "auto_minq"],
        help=(
            "none: no active-space reduction; "
            "paper12: manual preset; "
            "auto_minq: search smallest-qubit active space satisfying chemical accuracy"
        ),
    )

    parser.add_argument(
        "--occupied-indices",
        type=str,
        default=None,
        help="Optional comma-separated frozen occupied spatial orbital indices",
    )
    parser.add_argument(
        "--active-indices",
        type=str,
        default=None,
        help="Optional comma-separated active spatial orbital indices",
    )

    parser.add_argument(
        "--chemical-accuracy",
        type=float,
        default=5e-4,
        help="Target |E_active - E_FCI| in Hartree",
    )
    parser.add_argument(
        "--min-active-orbs",
        type=int,
        default=2,
        help="Minimum number of active spatial orbitals for auto search",
    )
    parser.add_argument(
        "--max-active-orbs",
        type=int,
        default=None,
        help="Maximum number of active spatial orbitals for auto search",
    )
    parser.add_argument(
        "--score-cut",
        type=float,
        default=2e-3,
        help="Orbitals with score=min(n,2-n) above this are mandatory-active in auto search",
    )
    parser.add_argument(
        "--frozen-occ-cut",
        type=float,
        default=1.99,
        help="Orbitals with MP2 occupation >= this are preferred frozen occupied",
    )
    parser.add_argument(
        "--frozen-virt-cut",
        type=float,
        default=0.01,
        help="Orbitals with MP2 occupation <= this are ignored as frozen virtual",
    )
    parser.add_argument(
        "--respect-symmetry-groups",
        action="store_true",
        help="Add same-irrep / near-degenerate orbitals as groups during auto search",
    )
    parser.add_argument(
        "--group-energy-tol",
        type=float,
        default=1e-8,
        help="Energy tolerance (Ha) used to define near-degenerate symmetry groups",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chemistry/data/f2_auto.json",
        help="Output JSON path (relative to project root or absolute)",
    )

    args = parser.parse_args()

    occupied_indices = parse_index_list(args.occupied_indices)
    active_indices = parse_index_list(args.active_indices)

    payload = build_f2_payload(
        R_value=args.R,
        unit=args.unit,
        basis=args.basis,
        multiplicity=args.multiplicity,
        charge=args.charge,
        reduction=args.reduction,
        encoding=args.encoding,
        chemical_accuracy=args.chemical_accuracy,
        occupied_indices_override=occupied_indices,
        active_indices_override=active_indices,
        min_active_orbs=args.min_active_orbs,
        max_active_orbs=args.max_active_orbs,
        score_cut=args.score_cut,
        frozen_occ_cut=args.frozen_occ_cut,
        frozen_virt_cut=args.frozen_virt_cut,
        respect_symmetry_groups=args.respect_symmetry_groups,
        group_energy_tol=args.group_energy_tol,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved: {out_path}")
    print(f"mapping: {payload['mapping']}")
    print(f"reduction: {payload['reduction']}")
    print(f"nqubits: {payload['nqubits']}, terms: {payload['term_count']}")
    print(f"fci_energy: {payload['fci_energy']:.10f}")
    if "active_space_energy" in payload:
        print(f"active_space_energy: {payload['active_space_energy']:.10f}")
        print(f"active_space_error_vs_fci: {payload['active_space_error_vs_fci']:.10e}")
    print(f"occupied_indices: {payload['occupied_indices']}")
    print(f"active_indices: {payload['active_indices']}")


if __name__ == "__main__":
    main()