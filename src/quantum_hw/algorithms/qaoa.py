"""QAOA Hamiltonian builders and optimization routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ..api.backend import Backend
from ..api.quantum_platform import create_provider_runtime
from ..circuit import QuantumCircuit
from ..core.types import QAOAResult
from .optimizer_utils import (
    Hamiltonian,
    CliffordFitMap,
    build_clifford_fit_map as _build_clifford_fit_map,
    run_variational_loop as _run_variational_loop,
)


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------


def build_maxcut_hamiltonian(
    edges: Sequence[Tuple[int, int]],
    num_qubits: int,
) -> Hamiltonian:
    """Build a MaxCut cost Hamiltonian from a graph edge list.

    The cost function is  ``C = Σ_{(i,j)} 0.5 (I - Z_i Z_j)``.
    The constant offset is dropped so that minimising ⟨H⟩ maximises the cut:
    ``H = Σ_{(i,j)} +0.5 Z_i Z_j``.

    Args:
        edges: Sequence of ``(i, j)`` undirected edges.
        num_qubits: Total number of qubits (graph vertices).

    Returns:
        Hamiltonian term list.

    Raises:
        ValueError: If an edge references out-of-range qubits, contains a
            self-loop, or *num_qubits* is non-positive.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    hamiltonian: Hamiltonian = []
    for i, j in edges:
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"edge ({i}, {j}) out of range for {num_qubits} qubits")
        if i == j:
            raise ValueError(f"self-loop edge ({i}, {j}) not allowed")
        pauli = ["I"] * num_qubits
        pauli[i] = "Z"
        pauli[j] = "Z"
        hamiltonian.append((0.5, "".join(pauli)))
    return hamiltonian


# ---------------------------------------------------------------------------
# QAOA ansatz
# ---------------------------------------------------------------------------


def _qaoa_num_params(p: int) -> int:
    """Number of variational parameters for *p* QAOA layers: 2p (gamma + beta)."""
    return 2 * p


def build_qaoa_ansatz_symbolic(
    num_qubits: int,
    edges: Sequence[Tuple[int, int]],
    p: int,
) -> Tuple[List[str], QuantumCircuit]:
    """Build a symbolic QAOA ansatz circuit.

    Structure per layer *l*:
      - Cost unitary:  ``RZZ(gamma_l, i, j)`` for every edge ``(i, j)``
      - Mixer unitary: ``RX(beta_l, q)`` for every qubit ``q``

    The initial state is ``|+>^n`` prepared with Hadamard gates.

    Returns ``(param_names, circuit)`` where *param_names* lists
    ``["gamma_0", "beta_0", "gamma_1", "beta_1", ...]``.
    """
    if p <= 0:
        raise ValueError("QAOA depth p must be positive")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    param_names: List[str] = []
    for l in range(p):
        param_names.append(f"gamma_{l}")
        param_names.append(f"beta_{l}")

    qc = QuantumCircuit(num_qubits)

    # Initial |+> state
    for q in range(num_qubits):
        qc.h(q)

    for l in range(p):
        gamma = f"gamma_{l}"
        beta = f"beta_{l}"
        # Cost layer: RZZ(gamma) on each edge
        for i, j in edges:
            qc.rzz(gamma, i, j)
        # Mixer layer: RX(beta) on each qubit
        for q in range(num_qubits):
            qc.rx(beta, q)

    return param_names, qc


# ---------------------------------------------------------------------------
# Core QAOA optimization
# ---------------------------------------------------------------------------


def run_qaoa_with_backend(
    client,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    hamiltonian: Hamiltonian,
    edges: Sequence[Tuple[int, int]],
    p: int = 1,
    shots: int = 1024,
    max_iters: int = 30,
    learning_rate: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.98,
    eps: float = 1e-8,
    shift: float = np.pi / 2.0,
    zne: bool = False,
    readout_mitigation: bool = False,
    target_qubits: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    init_params: Optional[Sequence[float]] = None,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    gradient_method: Literal["parameter-shift", "autograd"] = "parameter-shift",
    clifford_fitting: bool = False,
    clifford_fitting_num_samples: int = 8,
    clifford_fitting_num_non_clifford_gates: int = 3,
    qasm_version: str = "2.0",
    use_dd: bool = True,
    convert_single_qubit_gate_to_u: bool = True,
) -> QAOAResult:
    """Run QAOA optimization on a specific backend.

    Supports ``"parameter-shift"`` (hardware) and ``"autograd"`` (Simulator)
    gradient methods.  Optionally enables Clifford fitting for noise
    mitigation on real hardware.

    Args:
        client: ``QuantumHardwareClient`` instance.
        name: Task-name prefix.
        num_qubits: Number of logical qubits.
        backend: Target ``Backend``.
        chip_name: Chip identifier.
        hamiltonian: Cost Hamiltonian terms.
        edges: Graph edges for QAOA ansatz construction.
        p: Number of QAOA layers (depth).
        shots: Measurement shots per evaluation.
        max_iters: Maximum optimisation iterations.
        learning_rate: Adam learning rate.
        beta1 / beta2 / eps: Adam hyper-parameters.
        shift: Parameter-shift magnitude.
        zne: Enable zero-noise extrapolation.
        readout_mitigation: Enable readout error mitigation.
        target_qubits: Physical qubit mapping.
        seed: RNG seed.
        init_params: Explicit initial parameters (length ``2 * p``).
        callback: Per-iteration callback ``(iter, cost, params)``.
        gradient_method: ``"parameter-shift"`` or ``"autograd"``.
        clifford_fitting: Enable Clifford-based noise correction.
        clifford_fitting_num_samples: Calibration circuit count.
        clifford_fitting_num_non_clifford_gates: Haar-random gates in
            calibration circuits.
        qasm_version: OpenQASM serialisation version.
        use_dd: Enable dynamical decoupling.

    Returns:
        ``QAOAResult`` with best cost, parameters, and optimisation history.
    """
    method = str(gradient_method).lower()
    if method not in {"parameter-shift", "autograd"}:
        raise ValueError("gradient_method must be 'parameter-shift' or 'autograd'")

    param_names, symbolic_qc = build_qaoa_ansatz_symbolic(num_qubits, edges, p)
    num_params = len(param_names)

    if init_params is None:
        rng = np.random.default_rng(seed)
        params = rng.uniform(0.0, 2.0 * np.pi, size=num_params)
    else:
        params = np.asarray(init_params, dtype=float)
        if params.size != num_params:
            raise ValueError(f"init_params length must be {num_params}")

    target_qubits_in_use = target_qubits
    transpiled_template: Optional[QuantumCircuit] = None

    if method == "autograd":
        if str(chip_name).lower() != "simulator":
            raise ValueError("autograd mode is only supported on Simulator backend")
    else:
        transpiled_template = client._transpile_with_backend(
            symbolic_qc,
            backend,
            target_qubits=target_qubits,
            use_dd=use_dd,
            use_gate_compressor=False,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        target_qubits_in_use = client._ordered_target_qubits_from_layout(
            compiled_qc=transpiled_template,
            original_qc=symbolic_qc,
            num_qubits=num_qubits,
        )

    # Clifford fitting
    clifford_fit_map: Optional[CliffordFitMap] = None
    clifford_fitting_summary: Optional[Dict[str, Dict[str, float]]] = None
    if clifford_fitting:
        if method != "parameter-shift":
            raise ValueError("clifford_fitting requires gradient_method='parameter-shift'")
        if transpiled_template is None:
            raise RuntimeError("transpiled_template is required for clifford_fitting")
        clifford_fit_map = _build_clifford_fit_map(
            client,
            name=name,
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            hamiltonian=hamiltonian,
            shots=shots,
            zne=zne,
            readout_mitigation=readout_mitigation,
            transpiled_template=transpiled_template.deepcopy(),
            num_samples=int(clifford_fitting_num_samples),
            num_non_clifford_gates=int(clifford_fitting_num_non_clifford_gates),
            seed=None if seed is None else int(seed) + 7919,
            target_qubits=target_qubits_in_use,
            qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        clifford_fitting_summary = {
            obs: {"a": float(c[0]), "b": float(c[1])}
            for obs, c in clifford_fit_map.items()
        }
        print(
            "[qaoa] clifford fitting prepared:",
            f"terms={len(clifford_fit_map)}",
            f"samples={int(clifford_fitting_num_samples)}",
        )

    loop_result = _run_variational_loop(
        client,
        tag="qaoa",
        name=name,
        num_qubits=num_qubits,
        param_names=param_names,
        symbolic_qc=symbolic_qc,
        hamiltonian=hamiltonian,
        params=params,
        backend=backend,
        chip_name=chip_name,
        shots=shots,
        max_iters=max_iters,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        shift=shift,
        zne=zne,
        readout_mitigation=readout_mitigation,
        gradient_method=method,
        seed=seed,
        callback=callback,
        transpiled_template=transpiled_template,
        target_qubits=target_qubits_in_use,
        clifford_fit_map=clifford_fit_map,
        qasm_version=qasm_version,
        extra_info=f"p={p}",
        convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
    )

    return QAOAResult(
        best_cost=loop_result["best_cost"],
        best_params=loop_result["best_params"],
        cost_history=loop_result["cost_history"],
        params_history=loop_result["params_history"],
        grad_history=loop_result["grad_history"],
        last_expectations=loop_result["last_expectations"],
        clifford_fitting=clifford_fitting_summary,
    )


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


@dataclass
class QAOARunner:
    """High-level QAOA runner with automatic hardware selection."""

    client: object
    p: int = 1
    shots: int = 1024
    max_iters: int = 30
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    shift: float = np.pi / 2.0
    zne: bool = False
    readout_mitigation: bool = False
    clifford_fitting: bool = False
    clifford_fitting_num_samples: int = 8
    clifford_fitting_num_non_clifford_gates: int = 3
    seed: Optional[int] = None
    gradient_method: Literal["parameter-shift", "autograd"] = "parameter-shift"
    max_wait_time: int = 3600
    sleep_time: int = 5

    def run_model(
        self,
        name: str,
        num_qubits: int,
        edges: Sequence[Tuple[int, int]],
        *,
        provider: str = "quafu",
        target_qubits: Optional[Sequence[int]] = None,
        init_params: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
    ) -> QAOAResult:
        """Select hardware and run QAOA optimization.

        Args:
            name: Task name prefix.
            num_qubits: Number of logical qubits.
            edges: Graph edge list for the QAOA ansatz (RZZ cost layer).
            provider: Hardware provider (``"quafu"`` / ``"tianyan"`` / ``"guodun"``).
            target_qubits: Optional physical qubit mapping.
            init_params: Explicit initial parameters (length must be ``2 * p``).
            callback: Per-iteration callback ``(iter, cost, params)``.
            prefer_chips: Candidate chip filter (e.g. ``"Simulator"``).
        """
        print(
            "[qaoa] prepare run:",
            f"name={name}",
            f"num_qubits={num_qubits}",
            f"provider={provider}",
            f"p={self.p}",
            f"shots={self.shots}",
            f"max_iters={self.max_iters}",
        )
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits)

        provider_name = str(provider).lower()
        qasm_version = self.client._default_qasm_version_for_provider(provider_name)
        use_dd = provider_name not in {"tianyan", "guodun", "tencent"}
        convert_single_qubit_gate_to_u = provider_name not in {"tencent"}
        runtime = create_provider_runtime(provider=provider_name, client=self.client)
        profiles = runtime.backend_adapter.discover_hardware(
            num_qubits=num_qubits,
            prefer_hardware=prefer_chips,
        )
        print("[qaoa] candidate chips:", [p.hardware_name for p in profiles])
        if not profiles:
            raise RuntimeError(
                f"no available {provider_name} hardware for num_qubits={num_qubits}"
            )

        last_error: Optional[Exception] = None
        for profile in profiles:
            resolved = runtime.backend_adapter.resolve_backend(
                num_qubits=num_qubits,
                prefer_hardware=[profile.hardware_name],
            )
            self.client.chip_name = resolved.hardware_name
            self.client.chip_backend = resolved.backend

            self.client._active_task_adapter = runtime.task_adapter
            self.client._active_resolved_backend = resolved
            self.client._active_num_qubits = num_qubits
            try:
                print("[qaoa] running on chip:", resolved.hardware_name)
                return run_qaoa_with_backend(
                    self.client,
                    name=name,
                    num_qubits=num_qubits,
                    backend=resolved.backend,
                    chip_name=resolved.hardware_name,
                    hamiltonian=hamiltonian,
                    edges=edges,
                    p=self.p,
                    shots=self.shots,
                    max_iters=self.max_iters,
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                    shift=self.shift,
                    zne=self.zne,
                    readout_mitigation=self.readout_mitigation,
                    target_qubits=target_qubits,
                    seed=self.seed,
                    init_params=init_params,
                    callback=callback,
                    gradient_method=self.gradient_method,
                    clifford_fitting=self.clifford_fitting,
                    clifford_fitting_num_samples=self.clifford_fitting_num_samples,
                    clifford_fitting_num_non_clifford_gates=self.clifford_fitting_num_non_clifford_gates,
                    qasm_version=qasm_version,
                    use_dd=use_dd,
                    convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("all candidate chips failed to run QAOA") from last_error
