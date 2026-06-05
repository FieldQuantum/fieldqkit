"""Public package exports for the quantum hardware interface."""

__version__ = "0.1.1"

from .api import QuantumHardwareClient
from .api.platform_credentials import init_config
from .circuit import QuantumCircuit
from .core.circuits import (
    build_ghz,
    build_cluster,
    build_qft,
    build_ising_time_evolution,
    build_heisenberg_time_evolution,
    build_xxz_time_evolution,
    build_xy_time_evolution,
)
from .core.observables import pauli_expectation
from .core.readout import mitigate_readout
from .calibration import ReadoutCalibrationManager, NativeTwoQubitRBManager, NativeTwoQubitTomographyManager
from .core.types import QAOAResult, QMLResult, QBMResult, ShadowResult, VQEResult, RunResult, CalibrationResult
from .core.zne import apply_zne_cz_tripling, zne_linear_extrapolate
from .algorithms.shadow import ShadowTomography, estimate_observables, run_shadow_with_backend
from .algorithms.vqe import (
    VQERunner,
    build_custom_hamiltonian,
    build_heisenberg_hamiltonian,
    build_ising_hamiltonian,
    build_xy_hamiltonian,
    build_xxz_hamiltonian,
    run_vqe_with_backend,
)
from .algorithms.qaoa import (
    QAOARunner,
    build_maxcut_hamiltonian,
    run_qaoa_with_backend,
)
from .algorithms.qml import (
    run_pqc_classifier,
    run_qnn_unsupervised,
)
from .algorithms.qml_encoding import (
    angle_encoding_circuit,
    angle_encoding_circuit_symbolic,
    iqp_encoding_circuit,
    iqp_encoding_circuit_symbolic,
)

__all__ = [
    "QuantumHardwareClient",
    "init_config",
    "QuantumCircuit",
    "build_ghz",
    "build_cluster",
    "build_qft",
    "build_ising_time_evolution",
    "build_heisenberg_time_evolution",
    "build_xxz_time_evolution",
    "build_xy_time_evolution",
    "pauli_expectation",
    "mitigate_readout",
    "ReadoutCalibrationManager",
    "NativeTwoQubitRBManager",
    "NativeTwoQubitTomographyManager",
    "apply_zne_cz_tripling",
    "zne_linear_extrapolate",
    "ShadowTomography",
    "ShadowResult",
    "estimate_observables",
    "run_shadow_with_backend",
    "VQERunner",
    "VQEResult",
    "run_vqe_with_backend",
    "QAOAResult",
    "QAOARunner",
    "run_qaoa_with_backend",
    "build_custom_hamiltonian",
    "build_heisenberg_hamiltonian",
    "build_ising_hamiltonian",
    "build_xy_hamiltonian",
    "build_xxz_hamiltonian",
    "build_maxcut_hamiltonian",
    "QMLResult",
    "QBMResult",
    "RunResult",
    "CalibrationResult",
    "run_pqc_classifier",
    "run_qnn_unsupervised",
    "angle_encoding_circuit",
    "angle_encoding_circuit_symbolic",
    "iqp_encoding_circuit",
    "iqp_encoding_circuit_symbolic",
]
