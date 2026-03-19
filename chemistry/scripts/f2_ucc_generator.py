from __future__ import annotations

"""
Generate a symmetry-filtered, importance-ranked UCC-style circuit description
from an OpenFermion/PySCF Hamiltonian payload JSON.

What this script does
---------------------
1. Reads the JSON payload produced by your F2 exporter.
2. Infers the Hartree-Fock reference bitstring in interleaved spin-orbital order:
      0a, 0b, 1a, 1b, 2a, 2b, ...
3. Builds spin-conserving single and double fermionic excitations.
4. Filters excitations using spatial symmetry with a true irrep multiplication table.
5. Reconstructs the active-space JW qubit Hamiltonian from the payload.
6. For each excitation, computes a dominant-operator importance score
      ΔE_i = E0 - E_i
   using a 2x2 subspace spanned by |HF> and T_i|HF>.
7. Sorts excitations by importance from large to small.
8. Maps each anti-Hermitian excitation generator tau = T - T^dagger with Jordan-Wigner.
9. Keeps only the largest-magnitude Pauli term per excitation.
10. Emits a ranked UCC template in JSON format.

Important notes
---------------
- This script assumes JW ordering on the ACTIVE SPACE qubits is interleaved:
      spatial-orbital p -> qubits (2p = alpha, 2p+1 = beta)
- For SCBK / tapered encodings, the simple HF bitstring no longer stays as a plain
  product state on the reduced register. This script therefore generates the ansatz
  on the unreduced JW active-space register.
- The generated ansatz is a first-order Trotterized product of Pauli evolutions.
- Payload must contain:
      active_indices
      occupied_indices
      orbital_irreps   # aligned with active_indices ordering
      point_group
      constant
      terms

Example
-------
python chemistry/scripts/f2_ucc_generator.py \
    --payload chemistry/data/f2_R2.6_angstrom_sto-3g_auto.json \
    --output chemistry/data/ucc_f2.json \
    --include-doubles
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from openfermion.linalg import get_sparse_operator, get_number_preserving_sparse_operator

# =============================================================================
# Symmetry utilities
# =============================================================================

TOTALLY_SYMMETRIC_IRREP: dict[str, str] = {
    "D2H": "Ag",
    "C2V": "A1",
    "C2H": "Ag",
    "CI": "Ag",
    "CS": "A'",
    "C2": "A",
}

IRREP_PRODUCT_TABLES: dict[str, dict[tuple[str, str], str]] = {
    "D2H": {
        ("Ag", "Ag"): "Ag",
        ("Ag", "B1g"): "B1g",
        ("Ag", "B2g"): "B2g",
        ("Ag", "B3g"): "B3g",
        ("Ag", "Au"): "Au",
        ("Ag", "B1u"): "B1u",
        ("Ag", "B2u"): "B2u",
        ("Ag", "B3u"): "B3u",

        ("B1g", "B1g"): "Ag",
        ("B1g", "B2g"): "B3g",
        ("B1g", "B3g"): "B2g",
        ("B1g", "Au"): "B1u",
        ("B1g", "B1u"): "Au",
        ("B1g", "B2u"): "B3u",
        ("B1g", "B3u"): "B2u",

        ("B2g", "B2g"): "Ag",
        ("B2g", "B3g"): "B1g",
        ("B2g", "Au"): "B2u",
        ("B2g", "B1u"): "B3u",
        ("B2g", "B2u"): "Au",
        ("B2g", "B3u"): "B1u",

        ("B3g", "B3g"): "Ag",
        ("B3g", "Au"): "B3u",
        ("B3g", "B1u"): "B2u",
        ("B3g", "B2u"): "B1u",
        ("B3g", "B3u"): "Au",

        ("Au", "Au"): "Ag",
        ("Au", "B1u"): "B1g",
        ("Au", "B2u"): "B2g",
        ("Au", "B3u"): "B3g",

        ("B1u", "B1u"): "Ag",
        ("B1u", "B2u"): "B3g",
        ("B1u", "B3u"): "B2g",

        ("B2u", "B2u"): "Ag",
        ("B2u", "B3u"): "B1g",

        ("B3u", "B3u"): "Ag",
    },
    "C2V": {
        ("A1", "A1"): "A1",
        ("A1", "A2"): "A2",
        ("A1", "B1"): "B1",
        ("A1", "B2"): "B2",

        ("A2", "A2"): "A1",
        ("A2", "B1"): "B2",
        ("A2", "B2"): "B1",

        ("B1", "B1"): "A1",
        ("B1", "B2"): "A2",

        ("B2", "B2"): "A1",
    },
    "C2H": {
        ("Ag", "Ag"): "Ag",
        ("Ag", "Bg"): "Bg",
        ("Ag", "Au"): "Au",
        ("Ag", "Bu"): "Bu",

        ("Bg", "Bg"): "Ag",
        ("Bg", "Au"): "Bu",
        ("Bg", "Bu"): "Au",

        ("Au", "Au"): "Ag",
        ("Au", "Bu"): "Bg",

        ("Bu", "Bu"): "Ag",
    },
    "CI": {
        ("Ag", "Ag"): "Ag",
        ("Ag", "Au"): "Au",
        ("Au", "Au"): "Ag",
    },
    "CS": {
        ("A'", "A'"): "A'",
        ("A'", "A''"): "A''",
        ("A''", "A''"): "A'",
    },
    "C2": {
        ("A", "A"): "A",
        ("A", "B"): "B",
        ("B", "B"): "A",
    },
}


def reduce_point_group(point_group: str) -> str:
    pg = point_group.strip().upper()
    if pg in ("DOOH", "D∞H"):
        return "D2H"
    if pg in ("COOV", "C∞V"):
        return "C2V"
    return pg


def reduce_irrep(irrep: str, point_group: str) -> str:
    pg = point_group.strip().upper()
    ir = irrep.strip()

    if pg == "DOOH":
        if ir in ("A1g", "Sigma_g+", "SIGMA_G+"):
            return "Ag"
        if ir in ("A2g", "Sigma_g-", "SIGMA_G-"):
            return "B1g"
        if ir in ("A1u", "Sigma_u+", "SIGMA_U+"):
            return "B1u"
        if ir in ("A2u", "Sigma_u-", "SIGMA_U-"):
            return "Au"
        if ir in ("E1g", "PI_G"):
            return "B2g"
        if ir in ("E1u", "PI_U"):
            return "B2u"
        return "Ag"

    if pg == "COOV":
        if ir in ("A1", "Sigma+", "SIGMA+"):
            return "A1"
        if ir in ("A2", "Sigma-", "SIGMA-"):
            return "A2"
        if ir in ("E1", "PI"):
            return "B1"
        return "A1"

    return ir


def canonicalize_point_group(point_group: str) -> str:
    return reduce_point_group(point_group)


def canonicalize_irrep_name(irrep: str, point_group: str) -> str:
    return reduce_irrep(irrep, point_group)


def build_full_product_table(point_group: str) -> dict[tuple[str, str], str]:
    pg = canonicalize_point_group(point_group)
    if pg not in IRREP_PRODUCT_TABLES:
        raise ValueError(
            f"Unsupported point group '{point_group}'. "
            f"Supported groups: {sorted(IRREP_PRODUCT_TABLES.keys())}"
        )

    base = IRREP_PRODUCT_TABLES[pg]
    full: dict[tuple[str, str], str] = {}
    for (a, b), c in base.items():
        a = canonicalize_irrep_name(a, point_group)
        b = canonicalize_irrep_name(b, point_group)
        c = canonicalize_irrep_name(c, point_group)
        full[(a, b)] = c
        full[(b, a)] = c
    return full


def irrep_product(ir1: str, ir2: str, point_group: str) -> str:
    pg = canonicalize_point_group(point_group)
    ir1 = canonicalize_irrep_name(ir1, point_group)
    ir2 = canonicalize_irrep_name(ir2, point_group)

    table = build_full_product_table(pg)
    key = (ir1, ir2)
    if key not in table:
        raise ValueError(f"Missing irrep product for {key} in point group {pg}")
    return table[key]


def multiply_irreps(irreps: list[str], point_group: str) -> str:
    if not irreps:
        raise ValueError("multiply_irreps requires at least one irrep.")
    result = canonicalize_irrep_name(irreps[0], point_group)
    for ir in irreps[1:]:
        result = irrep_product(result, canonicalize_irrep_name(ir, point_group), point_group)
    return result


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class Excitation:
    occupied: tuple[int, ...]
    virtual: tuple[int, ...]
    kind: str  # 'single' or 'double'


@dataclass(frozen=True)
class PauliEvolutionTerm:
    excitation: Excitation
    parameter: str
    coefficient: float
    pauli_string: str
    importance: float


# =============================================================================
# Payload / active-space metadata
# =============================================================================

def load_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_active_space_metadata(
    payload: dict,
) -> tuple[int, int, list[int], list[int], list[str], str]:
    active_indices = payload.get("active_indices")
    occupied_indices = payload.get("occupied_indices") or []
    orbital_irreps = payload.get("orbital_irreps")
    point_group = payload.get("point_group")

    if not active_indices:
        raise ValueError("Payload must contain active_indices for ansatz generation.")
    if orbital_irreps is None:
        raise ValueError("Payload must contain orbital_irreps for symmetry filtering.")
    if point_group is None:
        raise ValueError("Payload must contain point_group for symmetry filtering.")
    if len(orbital_irreps) != len(active_indices):
        raise ValueError(
            f"Length mismatch: len(orbital_irreps)={len(orbital_irreps)} "
            f"!= len(active_indices)={len(active_indices)}"
        )

    n_active_spatial = len(active_indices)
    n_active_spin_orbitals = 2 * n_active_spatial

    charge = int(payload.get("charge", 0))
    total_electrons = 18 - charge  # F2-specific, consistent with your current script
    n_active_electrons = total_electrons - 2 * len(occupied_indices)

    if n_active_electrons < 0 or n_active_electrons > n_active_spin_orbitals:
        raise ValueError(
            f"Invalid active electron count: {n_active_electrons} for "
            f"{n_active_spin_orbitals} spin orbitals"
        )

    return (
        n_active_spatial,
        n_active_electrons,
        occupied_indices,
        active_indices,
        [canonicalize_irrep_name(x, point_group) for x in orbital_irreps],
        point_group,
    )


def hf_occupied_spin_orbitals(n_active_electrons: int) -> list[int]:
    return list(range(n_active_electrons))


def spin_label(spin_orbital: int) -> str:
    return "alpha" if spin_orbital % 2 == 0 else "beta"


# =============================================================================
# Symmetry-aware excitation screening
# =============================================================================

def spatial_irrep(spin_orbital: int, orbital_irreps: list[str], point_group: str) -> str:
    raw = orbital_irreps[spin_orbital // 2]
    return canonicalize_irrep_name(raw, point_group)


def symmetry_allowed_single(
    i: int,
    a: int,
    orbital_irreps: list[str],
    point_group: str,
) -> bool:
    ir_i = spatial_irrep(i, orbital_irreps, point_group)
    ir_a = spatial_irrep(a, orbital_irreps, point_group)
    totally_sym = TOTALLY_SYMMETRIC_IRREP[canonicalize_point_group(point_group)]
    return irrep_product(ir_i, ir_a, point_group) == totally_sym


def symmetry_allowed_double(
    i: int,
    j: int,
    a: int,
    b: int,
    orbital_irreps: list[str],
    point_group: str,
) -> bool:
    left = multiply_irreps(
        [
            spatial_irrep(i, orbital_irreps, point_group),
            spatial_irrep(j, orbital_irreps, point_group),
        ],
        point_group,
    )
    right = multiply_irreps(
        [
            spatial_irrep(a, orbital_irreps, point_group),
            spatial_irrep(b, orbital_irreps, point_group),
        ],
        point_group,
    )
    return left == right


def generate_spin_conserving_singles(
    n_active_spin_orbitals: int,
    n_active_electrons: int,
    orbital_irreps: list[str],
    point_group: str,
) -> list[Excitation]:
    occ = list(range(n_active_electrons))
    virt = list(range(n_active_electrons, n_active_spin_orbitals))
    singles: list[Excitation] = []

    for i in occ:
        for a in virt:
            if (i % 2) != (a % 2):
                continue
            if not symmetry_allowed_single(i, a, orbital_irreps, point_group):
                continue
            singles.append(Excitation((i,), (a,), "single"))
    return singles


def generate_spin_conserving_doubles(
    n_active_spin_orbitals: int,
    n_active_electrons: int,
    orbital_irreps: list[str],
    point_group: str,
) -> list[Excitation]:
    occ = list(range(n_active_electrons))
    virt = list(range(n_active_electrons, n_active_spin_orbitals))
    doubles: list[Excitation] = []

    for i_idx in range(len(occ)):
        for j_idx in range(i_idx + 1, len(occ)):
            i, j = occ[i_idx], occ[j_idx]
            for a_idx in range(len(virt)):
                for b_idx in range(a_idx + 1, len(virt)):
                    a, b = virt[a_idx], virt[b_idx]

                    if sorted([i % 2, j % 2]) != sorted([a % 2, b % 2]):
                        continue

                    if not symmetry_allowed_double(i, j, a, b, orbital_irreps, point_group):
                        continue

                    doubles.append(Excitation((i, j), (a, b), "double"))
    return doubles


# =============================================================================
# Dominant-excitation ranking
# =============================================================================

def apply_excitation_det(occ_set, ex: Excitation):
    occ = set(occ_set)

    # annihilate
    for i in ex.occupied:
        if i not in occ:
            return None
        occ.remove(i)

    # create
    for a in ex.virtual:
        if a in occ:
            return None
        occ.add(a)

    return tuple(sorted(occ))


def fermion_matrix_element(op_terms, bra_occ, ket_occ):
    val = 0.0

    for term_str, coeff in op_terms:
        ops = term_str.split()

        occ = list(ket_occ)
        sign = 1
        valid = True

        # apply from right to left
        for op in reversed(ops):
            if op.endswith("^"):
                i = int(op[:-1])
                if i in occ:
                    valid = False
                    break
                # fermionic sign
                sign *= (-1) ** sum(o < i for o in occ)
                occ.append(i)
                occ.sort()
            else:
                i = int(op)
                if i not in occ:
                    valid = False
                    break
                sign *= (-1) ** sum(o < i for o in occ)
                occ.remove(i)

        if valid and tuple(sorted(occ)) == tuple(sorted(bra_occ)):
            val += sign * coeff

    return val


def compute_excitation_deltaE_det(
    ex: Excitation,
    fermion_terms,
    hf_occ,
):
    hf_occ = tuple(sorted(hf_occ))

    ex_occ = apply_excitation_det(hf_occ, ex)
    if ex_occ is None:
        return 0.0

    # matrix elements
    E0 = fermion_matrix_element(fermion_terms, hf_occ, hf_occ)
    H11 = fermion_matrix_element(fermion_terms, ex_occ, ex_occ)
    V = fermion_matrix_element(fermion_terms, hf_occ, ex_occ)

    mat = np.array([[E0, V], [V, H11]])
    eigvals = np.linalg.eigvalsh(mat)

    Emin = np.min(eigvals)
    deltaE = E0 - Emin

    return float(deltaE)


# =============================================================================
# Excitation -> JW Pauli evolution
# =============================================================================

def excitation_to_antihermitian_fermion_operator(ex: Excitation) -> FermionOperator:
    if ex.kind == "single":
        i = ex.occupied[0]
        a = ex.virtual[0]
        t = FermionOperator(f"{a}^ {i}")
        td = FermionOperator(f"{i}^ {a}")
        return t - td

    i, j = ex.occupied
    a, b = ex.virtual
    t = FermionOperator(f"{a}^ {b}^ {j} {i}")
    td = FermionOperator(f"{i}^ {j}^ {b} {a}")
    return t - td


def pauli_string_from_term(term: tuple[tuple[int, str], ...]) -> str:
    if not term:
        return "I"
    return " ".join(f"{pauli}{idx}" for idx, pauli in term)


def jw_pauli_terms_for_excitation(ex: Excitation) -> list[tuple[float, str]]:
    qop = jordan_wigner(excitation_to_antihermitian_fermion_operator(ex))
    terms: list[tuple[float, str]] = []

    for term, coeff in qop.terms.items():
        if abs(coeff.imag) < 1e-12:
            continue
        c = float(coeff.imag)
        pstr = pauli_string_from_term(term)
        if pstr == "I":
            continue
        terms.append((c, pstr))

    if not terms:
        return []

    terms.sort(key=lambda x: (-abs(x[0]), len(x[1].split()), x[1]))
    return terms[-1:]


def build_ranked_excitations(
    n_active_spin_orbitals: int,
    n_active_electrons: int,
    include_singles: bool,
    include_doubles: bool,
    orbital_irreps: list[str],
    point_group: str,
    fermion_terms,
) -> list[tuple[float, Excitation]]:
    excitations: list[Excitation] = []

    if include_singles:
        excitations.extend(
            generate_spin_conserving_singles(
                n_active_spin_orbitals,
                n_active_electrons,
                orbital_irreps,
                point_group,
            )
        )

    if include_doubles:
        excitations.extend(
            generate_spin_conserving_doubles(
                n_active_spin_orbitals,
                n_active_electrons,
                orbital_irreps,
                point_group,
            )
        )

    hf_occ = hf_occupied_spin_orbitals(n_active_electrons)

    scored: list[tuple[float, Excitation]] = []
    for ex in excitations:
        delta_e = compute_excitation_deltaE_det(ex, fermion_terms, hf_occ)
        scored.append((delta_e, ex))

    scored.sort(
        key=lambda x: (
            -x[0],
            0 if x[1].kind == "double" else 1,
            x[1].occupied,
            x[1].virtual,
        )
    )
    return scored


def build_pauli_evolution_terms(
    ranked_excitations: list[tuple[float, Excitation]],
) -> list[PauliEvolutionTerm]:
    evo_terms: list[PauliEvolutionTerm] = []
    theta_counter = 0

    for importance, ex in ranked_excitations:
        pauli_terms = jw_pauli_terms_for_excitation(ex)
        for coeff, pauli_string in pauli_terms:
            evo_terms.append(
                PauliEvolutionTerm(
                    excitation=ex,
                    parameter=f"theta_{theta_counter}",
                    coefficient=coeff,
                    pauli_string=pauli_string,
                    importance=importance,
                )
            )
            theta_counter += 1

    return evo_terms


def render_json(
    nqubits: int,
    n_active_electrons: int,
    hf_occ: list[int],
    evo_terms: list[PauliEvolutionTerm],
) -> str:
    payload = {
        "schema": "ranked_ucc_v1",
        "nqubits": int(nqubits),
        "n_active_electrons": int(n_active_electrons),
        "hf_occupied": [int(q) for q in hf_occ],
        "terms": [
            {
                "parameter": term.parameter,
                "coefficient": float(term.coefficient),
                "pauli_string": term.pauli_string,
                "importance": float(term.importance),
                "excitation": {
                    "kind": term.excitation.kind,
                    "occupied": [int(x) for x in term.excitation.occupied],
                    "virtual": [int(x) for x in term.excitation.virtual],
                },
            }
            for term in evo_terms
        ],
    }
    return json.dumps(payload, indent=2)


def render_summary(
    nqubits: int,
    n_active_electrons: int,
    hf_occ: list[int],
    ranked_excitations: list[tuple[float, Excitation]],
    evo_terms: list[PauliEvolutionTerm],
    orbital_irreps: list[str],
    point_group: str,
) -> str:
    lines: list[str] = []
    lines.append("=== UCC generator summary ===")
    lines.append(f"point_group(original): {point_group}")
    lines.append(f"point_group(reduced): {canonicalize_point_group(point_group)}")
    lines.append(f"active-space orbital irreps(reduced): {orbital_irreps}")
    lines.append(f"nqubits (JW active-space register): {nqubits}")
    lines.append(f"n_active_electrons: {n_active_electrons}")
    lines.append(f"HF occupied spin orbitals: {hf_occ}")
    lines.append(f"number_of_ranked_excitations: {len(ranked_excitations)}")
    lines.append(f"number_of_pauli_evolution_terms: {len(evo_terms)}")

    preview = evo_terms[:10]
    if preview:
        lines.append("preview (sorted by importance descending):")
        for t in preview:
            lines.append(
                f"  {t.parameter}: ΔE={t.importance:.12e}, coeff={t.coefficient:+.6f}, "
                f"pauli='{t.pauli_string}', excitation={t.excitation.kind} "
                f"occ={t.excitation.occupied} virt={t.excitation.virtual}"
            )
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a symmetry-filtered, importance-ranked UCC-style circuit from active-space payload JSON"
    )
    parser.add_argument("--payload", type=str, required=True, help="Input Hamiltonian payload JSON")
    parser.add_argument("--output", type=str, default="chemistry/data/ucc_f2.json", help="Output ranked UCC JSON file")
    parser.add_argument("--include-singles", action="store_true", help="Include spin-conserving single excitations")
    parser.add_argument("--include-doubles", action="store_true", help="Include spin-conserving double excitations")
    args = parser.parse_args()

    if not args.include_singles and not args.include_doubles:
        parser.error("At least one of --include-singles or --include-doubles must be specified.")

    payload = load_payload(args.payload)
    (
        n_active_spatial,
        n_active_electrons,
        occupied_indices,
        active_indices,
        orbital_irreps,
        point_group,
    ) = get_active_space_metadata(payload)

    nqubits = 2 * n_active_spatial
    hf_occ = hf_occupied_spin_orbitals(n_active_electrons)
    
    ranked_excitations = build_ranked_excitations(
        n_active_spin_orbitals=nqubits,
        n_active_electrons=n_active_electrons,
        include_singles=args.include_singles,
        include_doubles=args.include_doubles,
        orbital_irreps=orbital_irreps,
        point_group=point_group,
        fermion_terms=payload["fermion_terms"],
    )

    evo_terms = build_pauli_evolution_terms(ranked_excitations)

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".json":
        raise ValueError("The UCC generator now writes JSON only; please use a .json output path.")
    rendered = render_json(nqubits, n_active_electrons, hf_occ, evo_terms)
    output_path.write_text(rendered, encoding="utf-8")

    print(
        render_summary(
            nqubits=nqubits,
            n_active_electrons=n_active_electrons,
            hf_occ=hf_occ,
            ranked_excitations=ranked_excitations,
            evo_terms=evo_terms,
            orbital_irreps=orbital_irreps,
            point_group=point_group,
        )
    )
    print(f"saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()