"""Algorithm entry points for VQE, QAOA, and shadow tomography."""

from __future__ import annotations

# circuit_compression requires torch + sim; lazy-load to keep the base
# package importable without torch.
_COMPRESSION_NAMES = {
    "HybridCompressionPlan",
    "SuffixCompressionBlock",
    "compile_tn_1d",
    "compress_circuit_with_hybrid_objective",
    "plan_hybrid_suffix_blocks",
}


def __getattr__(name: str):
    """Lazily import circuit compression symbols on first access.

    Args:
        name (*str*): Attribute name to look up.

    Returns:
        The requested attribute from ``circuit_compression``.

    Raises:
        AttributeError: module {__name__!r} has no attribute {name!r}
    """
    if name in _COMPRESSION_NAMES:
        from . import circuit_compression as _cc
        return getattr(_cc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .qml import (
    run_pqc_classifier,
    run_qnn_conditional,
    run_qnn_unsupervised,
)
from .qml_runner import QMLRunner
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
    "run_qnn_conditional",
    "run_qnn_unsupervised",
    "QMLRunner",
    "angle_encoding_circuit",
    "angle_encoding_circuit_symbolic",
    "iqp_encoding_circuit",
    "iqp_encoding_circuit_symbolic",
]
