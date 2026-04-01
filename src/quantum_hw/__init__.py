"""Public package exports for the quantum hardware interface."""

__version__ = "0.1.0"

from .api import QuantumHardwareClient
from .circuit import QuantumCircuit
from .core.circuits import build_ghz, build_cluster, build_qft, build_ising_time_evolution
from .core.observables import pauli_expectation
from .core.readout import mitigate_readout
from .calibration import ReadoutCalibrationManager, NativeTwoQubitRBManager, NativeTwoQubitTomographyManager
from .core.types import QAOAResult, QMLResult, QBMResult, ShadowResult, VQEResult, RunResult, CalibrationResult
from .core.zne import apply_zne_cz_tripling
from .algorithms.shadow import ShadowTomography
from .algorithms.vqe import (
    VQERunner,
    build_custom_hamiltonian,
    build_heisenberg_hamiltonian,
    build_ising_hamiltonian,
    build_xy_hamiltonian,
    build_xxz_hamiltonian,
)
from .algorithms.qaoa import (
    QAOARunner,
    build_maxcut_hamiltonian,
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
    "QuantumCircuit",
    "build_ghz",
    "build_cluster",
    "build_qft",
    "build_ising_time_evolution",
    "pauli_expectation",
    "mitigate_readout",
    "ReadoutCalibrationManager",
    "NativeTwoQubitRBManager",
    "NativeTwoQubitTomographyManager",
    "apply_zne_cz_tripling",
    "ShadowTomography",
    "ShadowResult",
    "VQERunner",
    "VQEResult",
    "QAOAResult",
    "QAOARunner",
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
