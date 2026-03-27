"""Algorithm entry points for VQE, QAOA, and shadow tomography."""

from .circuit_compression import (
    HybridCompressionPlan,
    SuffixCompressionBlock,
    compress_circuit_with_hybrid_objective,
    plan_hybrid_suffix_blocks,
)
from .qml import (
    run_pqc_classifier,
    run_qnn_unsupervised,
)
from .qml_encoding import (
    angle_encoding_circuit,
    angle_encoding_circuit_symbolic,
    iqp_encoding_circuit,
    iqp_encoding_circuit_symbolic,
)
from .qaoa import (
    QAOARunner,
    build_maxcut_hamiltonian,
    run_qaoa_with_backend,
)
from .shadow import ShadowTomography, estimate_observables, run_shadow_with_backend
from .vqe import (
    VQERunner,
    build_custom_hamiltonian,
    build_heisenberg_hamiltonian,
    build_ucc_ansatz,
    build_ising_hamiltonian,
    build_xy_hamiltonian,
    build_xxz_hamiltonian,
    run_vqe_with_backend,
)

__all__ = [
    "ShadowTomography",
    "estimate_observables",
    "run_shadow_with_backend",
    "VQERunner",
    "build_custom_hamiltonian",
    "build_heisenberg_hamiltonian",
    "build_ucc_ansatz",
    "build_ising_hamiltonian",
    "build_xy_hamiltonian",
    "build_xxz_hamiltonian",
    "run_vqe_with_backend",
    "QAOARunner",
    "build_maxcut_hamiltonian",
    "run_qaoa_with_backend",
    "HybridCompressionPlan",
    "SuffixCompressionBlock",
    "plan_hybrid_suffix_blocks",
    "compress_circuit_with_hybrid_objective",
    "run_pqc_classifier",
    "run_qnn_unsupervised",
    "angle_encoding_circuit",
    "angle_encoding_circuit_symbolic",
    "iqp_encoding_circuit",
    "iqp_encoding_circuit_symbolic",
]
