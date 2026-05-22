"""VQE Hamiltonian builders and optimization routines."""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from ..api.backend import Backend, resolve_provider
from ..api.quantum_platform import create_provider_runtime

from ..circuit import QuantumCircuit

from ..core.observables import pauli_support
from ..core.types import VQEResult
from .ansatz_templates import build_hardware_efficient_ansatz_symbolic
from .optimizer_utils import (
    Hamiltonian,
    CliffordFitMap,
    build_clifford_fit_map as _build_clifford_fit_map,
    run_variational_loop as _run_variational_loop,
)

AnsatzKind = Literal["hardwareefficient", "custom"]


def build_ising_hamiltonian(num_qubits: int, j: float = 1.0, h: float = 1.0) -> Hamiltonian:
    """Build transverse-field Ising Hamiltonian.

    H = -J * sum_i(Z_i Z_{i+1}) - h * sum_i(X_i)

    Args:
        num_qubits: Number of qubits (chain length).
        j: ZZ coupling strength.
        h: Transverse-field strength.

    Returns:
        Hamiltonian term list.
    """
    terms: Hamiltonian = []
    for i in range(num_qubits - 1):
        terms.append((-float(j), f"Z{i} Z{i + 1}"))
    for i in range(num_qubits):
        terms.append((-float(h), f"X{i}"))
    return terms


def build_heisenberg_hamiltonian(
    num_qubits: int,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    hz: float = 0.0,
) -> Hamiltonian:
    """Build Heisenberg Hamiltonian.

    H = sum_i(J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}) + h_z * sum_i(Z_i)

    Args:
        num_qubits: Chain length.
        jx: XX coupling.
        jy: YY coupling.
        jz: ZZ coupling.
        hz: Longitudinal field.

    Returns:
        Hamiltonian term list.
    """
    terms: Hamiltonian = []
    for i in range(num_qubits - 1):
        if jx != 0:
            terms.append((float(jx), f"X{i} X{i + 1}"))
        if jy != 0:
            terms.append((float(jy), f"Y{i} Y{i + 1}"))
        if jz != 0:
            terms.append((float(jz), f"Z{i} Z{i + 1}"))
    if hz != 0:
        for i in range(num_qubits):
            terms.append((float(hz), f"Z{i}"))
    return terms


def build_xxz_hamiltonian(
    num_qubits: int,
    jxy: float = 1.0,
    jz: float = 1.0,
    hz: float = 0.0,
) -> Hamiltonian:
    """Build XXZ Hamiltonian.

    H = J_xy * (XX + YY) + J_z * ZZ + h_z * Z

    Args:
        num_qubits: Chain length.
        jxy: XX / YY coupling.
        jz: ZZ coupling.
        hz: Longitudinal field.

    Returns:
        Hamiltonian term list.
    """
    terms: Hamiltonian = []
    for i in range(num_qubits - 1):
        if jxy != 0:
            terms.append((float(jxy), f"X{i} X{i + 1}"))
            terms.append((float(jxy), f"Y{i} Y{i + 1}"))
        if jz != 0:
            terms.append((float(jz), f"Z{i} Z{i + 1}"))
    if hz != 0:
        for i in range(num_qubits):
            terms.append((float(hz), f"Z{i}"))
    return terms


def build_xy_hamiltonian(
    num_qubits: int,
    jx: float = 1.0,
    jy: float = 1.0,
    hz: float = 0.0,
) -> Hamiltonian:
    """Build XY Hamiltonian.

    H = J_x * XX + J_y * YY + h_z * Z

    Args:
        num_qubits: Chain length.
        jx: XX coupling.
        jy: YY coupling.
        hz: Longitudinal field.

    Returns:
        Hamiltonian term list.
    """
    terms: Hamiltonian = []
    for i in range(num_qubits - 1):
        if jx != 0:
            terms.append((float(jx), f"X{i} X{i + 1}"))
        if jy != 0:
            terms.append((float(jy), f"Y{i} Y{i + 1}"))
    if hz != 0:
        for i in range(num_qubits):
            terms.append((float(hz), f"Z{i}"))
    return terms


def build_custom_hamiltonian(terms: Sequence[Tuple[float, str]], num_qubits: int) -> Hamiltonian:
    """Validate and return a user-supplied custom Hamiltonian.

    Args:
        terms: Sequence of ``(coefficient, pauli_string)`` pairs.
        num_qubits: Number of qubits (used for Pauli-string validation).

    Returns:
        Validated Hamiltonian term list.

    Raises:
        ValueError: If any Pauli string is empty or invalid.
    """
    out: Hamiltonian = []
    for coeff, pauli in terms:
        if not isinstance(pauli, str) or not pauli.strip():
            raise ValueError("pauli term must be a non-empty string")
        _ = pauli_support(pauli, num_qubits=num_qubits)
        out.append((float(coeff), pauli))
    return out


def _extract_names_from_expr(expr: str) -> List[str]:
    """Extract symbolic variable names from a parameter expression.

    Parses *expr* as a Python expression and returns all ``ast.Name``
    nodes except the constant ``pi``.

    Args:
        expr: String expression such as ``"theta_0"`` or ``"2*pi + alpha"``.

    Returns:
        List of symbol names found in the expression.
    """
    expr = str(expr).strip().replace('π', 'pi').replace('np.pi', 'pi')
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        # Backward compatibility: if not a valid expression, treat it as a raw symbol.
        return [expr] if expr else []

    out: List[str] = []

    def _walk(node):
        """Recursively collect symbolic parameter names from an AST node.

        Args:
            node: AST node to inspect (``Expression``, ``Name``, etc.).

        Raises:
            ValueError: f'unsupported symbolic parameter expression: {expr}'
        """
        if isinstance(node, ast.Expression):
            _walk(node.body)
            return
        if isinstance(node, ast.Name):
            if node.id != "pi":
                out.append(node.id)
            return
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            _walk(node.operand)
            return
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
        ):
            _walk(node.left)
            _walk(node.right)
            return
        raise ValueError(f"unsupported symbolic parameter expression: {expr}")

    _walk(tree)
    return out


def _extract_symbolic_params_from_circuit(qc: QuantumCircuit) -> List[str]:
    """Extract unresolved symbolic parameter names from a circuit template.

    Iterates over ``qc.params_value`` and collects symbol names that
    have not yet been bound to numeric values.

    Args:
        qc: A ``QuantumCircuit`` with symbolic parameter entries.

    Returns:
        Ordered, deduplicated list of symbolic parameter names.
    """
    names: List[str] = []
    seen = set()

    for key, value in qc.params_value.items():
        if isinstance(key, str) and isinstance(value, str):
            for symbol in _extract_names_from_expr(key):
                if symbol not in seen:
                    names.append(symbol)
                    seen.add(symbol)
    return names



def _resolve_ansatz_layout(
    *,
    ansatz: AnsatzKind,
    num_qubits: int,
    layers: int,
    custom_ansatz_circuit: Optional[QuantumCircuit] = None,
) -> Tuple[List[str], QuantumCircuit]:
    """Build an ansatz circuit from a kind name.

    Supported kinds:

    - ``"hardwareefficient"``: RX/RY + CZ entangling layers.
    - ``"custom"``: User-supplied ``QuantumCircuit`` with symbolic params.

    Args:
        ansatz: Ansatz type identifier.
        num_qubits: Number of qubits.
        layers: Circuit depth (ignored for ``"custom"``).
        custom_ansatz_circuit: Required when *ansatz* is ``"custom"``.

    Returns:
        ``(param_names, symbolic_circuit)``.

    Raises:
        ValueError: For unknown ansatz kinds or invalid custom circuits.
    """
    ansatz_name = str(ansatz).lower()
    if ansatz_name == "hardwareefficient":
        num_params = 2 * num_qubits * (layers + 1)
        param_names = [f"theta_{i}" for i in range(num_params)]
        symbolic_qc = build_hardware_efficient_ansatz_symbolic(
            num_qubits,
            param_names,
            layers=layers,
        )
        return param_names, symbolic_qc
    if ansatz_name == "custom":
        if custom_ansatz_circuit is None:
            raise ValueError("custom ansatz requires custom_ansatz_circuit")
        if not isinstance(custom_ansatz_circuit, QuantumCircuit):
            raise ValueError("custom_ansatz_circuit must be a QuantumCircuit instance")
        if int(custom_ansatz_circuit.nqubits) != int(num_qubits):
            raise ValueError("custom_ansatz_circuit.nqubits must equal num_qubits")
        param_names = _extract_symbolic_params_from_circuit(custom_ansatz_circuit)
        if not param_names:
            raise ValueError("custom ansatz circuit has no unresolved symbolic parameters")
        return param_names, custom_ansatz_circuit.deepcopy()
    raise ValueError("ansatz must be 'hardwareefficient' or 'custom'")



def run_vqe_with_backend(
    client,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    hamiltonian: Hamiltonian,
    layers: int,
    shots: int,
    max_iters: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    target_qubits: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    init_params: Optional[Sequence[float]] = None,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    gradient_method: Literal["parameter-shift", "autograd"] = "parameter-shift",
    ansatz: AnsatzKind = "hardwareefficient",
    custom_ansatz_circuit: Optional[QuantumCircuit] = None,
    clifford_fitting: bool = False,
    clifford_fitting_num_samples: int = 8,
    clifford_fitting_num_non_clifford_gates: int = 3,
    enable_block_planner: bool = False,
    planner_bond_cap: int = 128,
    planner_trunc_tol: float = 1e-8,
    planner_max_layers_per_block: int = 6,
    enable_circuit_compression: bool = False,
    compression_block_layers: Optional[int] = None,
    compression_optimizer_steps: int = 20,
    compression_optimizer_lr: float = 0.05,
    compression_verbose: bool = False,
    compression_plot_loss: bool = False,
    qasm_version: str = "2.0",
    use_dd: bool = True,
    convert_single_qubit_gate_to_u: bool = True,
    transpile: bool = True,
) -> VQEResult:
    """Run VQE optimization on a specific backend.

    Supports two gradient methods:

    - ``"parameter-shift"``: hardware-compatible finite-difference via
      the parameter-shift rule.
    - ``"autograd"``: PyTorch automatic differentiation (Simulator only).

    Optional features include Clifford fitting for noise mitigation,
    hybrid block-planner suffix compression, and full circuit compression.

    Args:
        client: ``QuantumHardwareClient`` instance.
        name: Task-name prefix for submitted jobs.
        num_qubits: Number of logical qubits.
        backend: Target ``Backend`` (Simulator or hardware).
        chip_name: Chip identifier string.
        hamiltonian: Cost Hamiltonian terms ``(coeff, pauli_str)``.
        layers: Ansatz depth.
        shots: Measurement shots per evaluation.
        max_iters: Maximum optimisation iterations.
        learning_rate: Adam learning rate.
        beta1 / beta2 / eps: Adam hyper-parameters.
        shift: Parameter-shift magnitude (default ``π/2``).
        zne: Enable zero-noise extrapolation.
        readout_mitigation: Enable readout error mitigation.
        target_qubits: Physical qubit mapping.
        seed: RNG seed.
        init_params: Explicit initial parameter values.
        callback: Per-iteration callback ``(iter, energy, params)``.
        gradient_method: ``"parameter-shift"`` or ``"autograd"``.
        ansatz: Ansatz kind (``"hardwareefficient"`` / ``"custom"``).
        custom_ansatz_circuit: Required when *ansatz* is ``"custom"``.
        clifford_fitting: Enable Clifford-based noise mitigation.
        clifford_fitting_num_samples: Calibration circuit count.
        clifford_fitting_num_non_clifford_gates: Haar-random gates per
            calibration circuit.
        enable_block_planner: Enable suffix block planner for compression.
        planner_bond_cap: Bond-dimension cap used by planner/compression.
        planner_trunc_tol: Truncation tolerance used by planner/compression.
        planner_max_layers_per_block: Planner max layers per suffix block.
        enable_circuit_compression: Enable per-iteration circuit compression.
            When combined with ``enable_block_planner``, the prefix uses
            ``objective_mode='mps'`` and each suffix block uses ``'mpo'``.
            Requires ``compression_block_layers`` to be set.
        compression_block_layers: Compression block depth (positive integer).
            **Required** when ``enable_circuit_compression=True``.
        compression_optimizer_steps: Optimization steps per compression run.
        compression_optimizer_lr: Learning rate for compression optimizer.
        compression_verbose: Print compression diagnostics.
        compression_plot_loss: Plot compression optimization loss.
        qasm_version: OpenQASM serialisation version.
        use_dd: Enable dynamical decoupling.
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates
            to ``U`` during transpilation.
        transpile: Whether to transpile the circuit for hardware on the client
            side.  When ``False`` the symbolic template is used as-is and no
            layout mapping is performed.  Defaults to ``True``.

    Returns:
        ``VQEResult`` with best energy, parameters, and full history.

    Raises:
        ValueError: If *gradient_method* is invalid, or numeric hyperparameters
            (*planner_bond_cap*, *planner_trunc_tol*, *planner_max_layers_per_block*,
            *compression_optimizer_steps*, *compression_optimizer_lr*,
            *clifford_fitting_num_non_clifford_gates*) are out of range,
            or *compression_block_layers* is missing/invalid when compression is enabled.
    """
    method = str(gradient_method).lower()
    if method not in {"parameter-shift", "autograd"}:
        raise ValueError("gradient_method must be 'parameter-shift' or 'autograd'")
    if planner_bond_cap <= 0:
        raise ValueError("planner_bond_cap must be positive")
    if planner_trunc_tol < 0.0:
        raise ValueError("planner_trunc_tol must be non-negative")
    if planner_max_layers_per_block <= 0:
        raise ValueError("planner_max_layers_per_block must be positive")
    if compression_optimizer_steps <= 0:
        raise ValueError("compression_optimizer_steps must be positive")
    if compression_optimizer_lr <= 0.0:
        raise ValueError("compression_optimizer_lr must be positive")
    if clifford_fitting_num_non_clifford_gates < 0:
        raise ValueError("clifford_fitting_num_non_clifford_gates must be non-negative")
    block_depth_k: Optional[int] = None
    if compression_block_layers is not None:
        if isinstance(compression_block_layers, (list, tuple, np.ndarray)) or isinstance(compression_block_layers, bool):
            raise ValueError("compression_block_layers must be a single positive integer k")
        block_depth_k = int(compression_block_layers)
        if block_depth_k <= 0:
            raise ValueError("compression_block_layers must be positive")
    if enable_circuit_compression and block_depth_k is None:
        raise ValueError("compression_block_layers must be provided when compression is enabled")
    unified_bond_cap = int(planner_bond_cap)
    unified_trunc_tol = float(planner_trunc_tol)

    param_names, symbolic_qc = _resolve_ansatz_layout(
        ansatz=ansatz,
        num_qubits=num_qubits,
        layers=layers,
        custom_ansatz_circuit=custom_ansatz_circuit,
    )
    num_params = len(param_names)
    if init_params is None:
        rng = np.random.default_rng(seed)
        init_values = rng.uniform(0.0, 2.0 * np.pi, size=num_params)
    else:
        init_values = np.asarray(init_params, dtype=float)
        if init_values.size != num_params:
            raise ValueError(f"init_params length must be {num_params}")

    params = init_values.copy()
    transpiled_template: Optional[QuantumCircuit] = None
    gradient_param_template: Optional[QuantumCircuit] = None
    compressed_transpiled_template: Optional[QuantumCircuit] = None
    target_qubits_in_use: Optional[Sequence[int]] = target_qubits
    circuit_transform_fn = None
    if method == "autograd":
        if (str(chip_name).lower() not in ["simulator", "fieldquantum_sim"]):
            raise ValueError("autograd mode is only supported on Simulator backend")
    else:
        if not enable_circuit_compression:
            if transpile:
                transpiled_template = client._transpile_with_backend(
                    symbolic_qc,
                    backend,
                    target_qubits=target_qubits,
                    use_dd=use_dd,
                    use_gate_compressor=False,
                    convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
                )
                gradient_param_template = transpiled_template
                target_qubits_in_use = client._ordered_target_qubits_from_layout(
                    compiled_qc=transpiled_template,
                    original_qc=symbolic_qc,
                    num_qubits=num_qubits,
                )
            else:
                transpiled_template = symbolic_qc
                gradient_param_template = symbolic_qc
                target_qubits_in_use = list(target_qubits) if target_qubits is not None else list(range(num_qubits))
        else:
            gradient_param_template = symbolic_qc
            from .circuit_compression import build_compression_transform as _build_compression_transform
            comp_ctx = _build_compression_transform(
                client,
                num_qubits=num_qubits,
                layers=layers,
                backend=backend,
                target_qubits=target_qubits,
                use_dd=use_dd,
                enable_block_planner=enable_block_planner,
                planner_bond_cap=planner_bond_cap,
                planner_trunc_tol=planner_trunc_tol,
                planner_max_layers_per_block=planner_max_layers_per_block,
                compression_block_layers=compression_block_layers,
                compression_optimizer_steps=compression_optimizer_steps,
                compression_optimizer_lr=compression_optimizer_lr,
                compression_verbose=compression_verbose,
                compression_plot_loss=compression_plot_loss,
                tag="vqe",
                convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
                transpile=transpile,
            )
            circuit_transform_fn = comp_ctx["transform"]
            compressed_transpiled_template = comp_ctx["compressed_transpiled_template"]
            target_qubits_in_use = comp_ctx["target_qubits_in_use"]

    clifford_fit_map: Optional[CliffordFitMap] = None
    clifford_fitting_summary: Optional[Dict[str, Dict[str, float]]] = None
    if clifford_fitting:
        if method != "parameter-shift":
            raise ValueError("clifford_fitting currently requires gradient_method='parameter-shift'")
        if enable_circuit_compression:
            if compressed_transpiled_template is None:
                raise RuntimeError("compressed_transpiled_template is required when compression is enabled")
            clifford_transpiled_template = compressed_transpiled_template.deepcopy()
        else:
            if transpiled_template is None:
                raise RuntimeError("transpiled_template is required when compression is disabled")
            clifford_transpiled_template = transpiled_template.deepcopy()
        clifford_fit_map = _build_clifford_fit_map(
            client,
            name=name,
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            observables=list(dict.fromkeys(obs for _, obs in hamiltonian)),
            shots=shots,
            zne=zne,
            readout_mitigation=readout_mitigation,
            transpiled_template=clifford_transpiled_template,
            num_samples=int(clifford_fitting_num_samples),
            num_non_clifford_gates=int(clifford_fitting_num_non_clifford_gates),
            seed=None if seed is None else int(seed) + 7919,
            target_qubits=target_qubits_in_use,
            qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        clifford_fitting_summary = {
            obs: {"a": float(coeffs[0]), "b": float(coeffs[1])}
            for obs, coeffs in clifford_fit_map.items()
        }
        logger.info(
            "clifford fitting prepared: terms=%d samples=%d non_clifford_gates=%d",
            len(clifford_fit_map),
            int(clifford_fitting_num_samples),
            int(clifford_fitting_num_non_clifford_gates),
        )

    loop_result = _run_variational_loop(
        client,
        tag="vqe",
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
        gradient_param_template=gradient_param_template,
        target_qubits=target_qubits_in_use,
        clifford_fit_map=clifford_fit_map,
        circuit_transform=circuit_transform_fn,
        qasm_version=qasm_version,
        extra_info=f"layers={layers} ansatz={str(ansatz).lower()}",
        convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
    )

    return VQEResult(
        best_energy=loop_result["best_cost"],
        best_params=loop_result["best_params"],
        energy_history=loop_result["cost_history"],
        params_history=loop_result["params_history"],
        grad_history=loop_result["grad_history"],
        last_expectations=loop_result["last_expectations"],
        clifford_fitting=clifford_fitting_summary,
    )


@dataclass
class VQERunner:
    """High-level VQE runner."""

    client: object
    layers: int = 1
    shots: int = 1024
    max_iters: int = 20
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
    enable_block_planner: bool = False
    planner_bond_cap: int = 128
    planner_trunc_tol: float = 1e-8
    planner_max_layers_per_block: int = 6
    enable_circuit_compression: bool = False
    compression_block_layers: Optional[int] = None
    compression_optimizer_steps: int = 20
    compression_optimizer_lr: float = 0.05
    compression_verbose: bool = False
    compression_plot_loss: bool = False
    max_wait_time: int = 3600
    sleep_time: int = 5
    transpile_on_client: bool = True

    def run_model(
        self,
        name: str,
        num_qubits: int,
        *,
        provider: str = "quafu",
        model: str = "ising",
        model_params: Optional[Dict[str, float]] = None,
        hamiltonian: Optional[Sequence[Tuple[float, str]]] = None,
        target_qubits: Optional[Sequence[int]] = None,
        init_params: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        ansatz: AnsatzKind = "hardwareefficient",
        custom_ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> VQEResult:
        """Select hardware and run VQE optimization.

        Dispatches to a built-in model Hamiltonian (``"ising"``,
        ``"heisenberg"``, ``"xxz"``, ``"xy"``) or accepts a custom
        term list when ``model="custom"``.

        Args:
            name: Task name prefix.
            num_qubits: Number of logical qubits.
            provider: Hardware provider name.
            model: Hamiltonian model.
            model_params: Extra kwargs forwarded to the model builder.
            hamiltonian: Required when ``model="custom"``.
            target_qubits: Physical qubit mapping.
            init_params: Explicit initial parameters.
            callback: Per-iteration callback.
            prefer_chips: Candidate chip filter.
            ansatz: Ansatz kind.
            custom_ansatz_circuit: Required when ``ansatz="custom"``.

        Returns:
            ``VQEResult`` with optimisation history.
        """
        logger.info(
            "prepare run: name=%s num_qubits=%d provider=%s model=%s layers=%d shots=%d max_iters=%d",
            name, num_qubits, provider, model, self.layers, self.shots, self.max_iters,
        )
        model = model.lower()
        params = model_params or {}

        def _filter(fn):
            """Return only the kwargs accepted by *fn*."""
            sig = inspect.signature(fn)
            return {k: v for k, v in params.items() if k in sig.parameters}

        if model == "ising":
            hamiltonian = build_ising_hamiltonian(num_qubits, **_filter(build_ising_hamiltonian))
        elif model == "heisenberg":
            hamiltonian = build_heisenberg_hamiltonian(num_qubits, **_filter(build_heisenberg_hamiltonian))
        elif model == "xxz":
            hamiltonian = build_xxz_hamiltonian(num_qubits, **_filter(build_xxz_hamiltonian))
        elif model == "xy":
            hamiltonian = build_xy_hamiltonian(num_qubits, **_filter(build_xy_hamiltonian))
        elif model == "custom":
            if hamiltonian is None:
                raise ValueError("custom model requires hamiltonian")
            hamiltonian = build_custom_hamiltonian(hamiltonian, num_qubits)
        else:
            raise ValueError(f"unsupported model: {model}")

        provider_name = resolve_provider(provider, prefer_chips)
        qasm_version = self.client._default_qasm_version_for_provider(provider_name)
        use_dd = provider_name not in {"tianyan", "guodun", "tencent", "simulator", "fieldquantum"}
        convert_single_qubit_gate_to_u = provider_name not in {"tencent", "fieldquantum"}
        runtime = create_provider_runtime(provider=provider_name, client=self.client)
        profiles = runtime.backend_adapter.discover_hardware(
            num_qubits=num_qubits,
            prefer_hardware=prefer_chips,
        )
        logger.info("candidate chips: %s", [p.hardware_name for p in profiles])
        if not profiles:
            raise RuntimeError(f"no available {provider_name} hardware for num_qubits={num_qubits}")

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
                logger.info("running on chip: %s", resolved.hardware_name)
                return run_vqe_with_backend(
                    self.client,
                    name=name,
                    num_qubits=num_qubits,
                    backend=resolved.backend,
                    chip_name=resolved.hardware_name,
                    hamiltonian=hamiltonian,
                    layers=self.layers,
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
                    ansatz=ansatz,
                    custom_ansatz_circuit=custom_ansatz_circuit,
                    clifford_fitting=self.clifford_fitting,
                    clifford_fitting_num_samples=self.clifford_fitting_num_samples,
                    clifford_fitting_num_non_clifford_gates=self.clifford_fitting_num_non_clifford_gates,
                    enable_block_planner=self.enable_block_planner,
                    planner_bond_cap=self.planner_bond_cap,
                    planner_trunc_tol=self.planner_trunc_tol,
                    planner_max_layers_per_block=self.planner_max_layers_per_block,
                    enable_circuit_compression=self.enable_circuit_compression,
                    compression_block_layers=self.compression_block_layers,
                    compression_optimizer_steps=self.compression_optimizer_steps,
                    compression_optimizer_lr=self.compression_optimizer_lr,
                    compression_verbose=self.compression_verbose,
                    compression_plot_loss=self.compression_plot_loss,
                    qasm_version=qasm_version,
                    use_dd=use_dd,
                    convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
                    transpile=bool(self.transpile_on_client),
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("all candidate chips failed to run VQE") from last_error


