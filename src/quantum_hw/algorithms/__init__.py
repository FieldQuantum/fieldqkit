"""Algorithm entry points for VQE, QAOA, and shadow tomography."""

from .shadow import ShadowTomography, estimate_observables, run_shadow, run_shadow_with_backend
from .vqe import (
    VQERunner,
    build_custom_hamiltonian,
    build_heisenberg_hamiltonian,
    build_ising_hamiltonian,
    build_xy_hamiltonian,
    build_xxz_hamiltonian,
    run_vqe,
    run_vqe_with_backend,
)
from .qaoa import (
    QAOARunner,
    build_custom_cost_hamiltonian,
    build_maxcut_hamiltonian,
    build_qaoa_circuit,
    build_qaoa_circuit_from_terms,
    run_qaoa,
    run_qaoa_with_backend,
)

__all__ = [
    "ShadowTomography",
    "estimate_observables",
    "run_shadow",
    "run_shadow_with_backend",
    "VQERunner",
    "build_custom_hamiltonian",
    "build_heisenberg_hamiltonian",
    "build_ising_hamiltonian",
    "build_xy_hamiltonian",
    "build_xxz_hamiltonian",
    "run_vqe",
    "run_vqe_with_backend",
    "QAOARunner",
    "build_custom_cost_hamiltonian",
    "build_maxcut_hamiltonian",
    "build_qaoa_circuit",
    "build_qaoa_circuit_from_terms",
    "run_qaoa",
    "run_qaoa_with_backend",
]
