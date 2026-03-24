"""VQE Hamiltonian builders and optimization routines."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import torch
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from ..api.backend import Backend
from ..api.quantum_platform import create_provider_runtime

from ..circuit import QuantumCircuit

from ..core.observables import pauli_support
from ..core.types import VQEResult
from .ansatz_templates import build_hardware_efficient_ansatz_symbolic
from .ansatz_templates import build_ucc_ansatz_symbolic, build_ucc_ansatz
from .ansatz_templates import build_ucc_num_params
from .circuit_compression import HybridCompressionPlan
from .circuit_compression import build_layer_span_circuit
from .circuit_compression import compress_circuit_with_hybrid_objective
from .circuit_compression import plan_hybrid_suffix_blocks

Hamiltonian = List[Tuple[float, str]]
AnsatzKind = Literal["hardwareefficient", "ucc", "custom"]
CliffordFitMap = Dict[str, Tuple[float, float]]


def build_ising_hamiltonian(num_qubits: int, j: float = 1.0, h: float = 1.0) -> Hamiltonian:
    """Build transverse-field Ising Hamiltonian: -J sum ZiZi - h sum Xi."""
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
    """Build Heisenberg Hamiltonian: sum (Jx XX + Jy YY + Jz ZZ) + hz Z."""
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
    """Build XXZ Hamiltonian: Jxy(XX+YY) + Jz ZZ + hz Z."""
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
    """Build XY Hamiltonian: Jx XX + Jy YY + hz Z."""
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
    """Validate and return a custom Hamiltonian list."""
    out: Hamiltonian = []
    for coeff, pauli in terms:
        if not isinstance(pauli, str) or not pauli.strip():
            raise ValueError("pauli term must be a non-empty string")
        _ = pauli_support(pauli, num_qubits=num_qubits)
        out.append((float(coeff), pauli))
    return out


def _extract_names_from_expr(expr: str) -> List[str]:
    """Extract symbol names from a parameter expression, excluding pi."""
    expr = str(expr).strip().replace('π', 'pi').replace('np.pi', 'pi')
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        # Backward compatibility: if not a valid expression, treat it as a raw symbol.
        return [expr] if expr else []

    out: List[str] = []

    def _walk(node):
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
    """Extract unresolved symbolic parameter names from a parameterized circuit template."""
    names: List[str] = []
    seen = set()

    for key, value in qc.params_value.items():
        if isinstance(key, str) and isinstance(value, str):
            for symbol in _extract_names_from_expr(key):
                if symbol not in seen:
                    names.append(symbol)
                    seen.add(symbol)
    return names


def _build_parameterized_single_qubit_gate_list(transpiled_template: QuantumCircuit) -> List[Tuple[int, str]]:
    """Collect (gate_index, gate_name) for symbol-bearing parameterized 1q gates.

    This intentionally excludes constant-parameter decompositions (e.g. H -> u(const,const,const)).
    """
    one_qubit_param_gates = {"p", "r", "u", "u3", "rx", "ry", "rz"}
    out: List[Tuple[int, str]] = []
    for gate_index, gate_info in enumerate(transpiled_template.gates):
        gate = str(gate_info[0]).lower()
        if gate not in one_qubit_param_gates:
            continue

        gate_params = gate_info[1:-1]
        has_symbol = False
        for param in gate_params:
            if not isinstance(param, str):
                continue
            if len(_extract_names_from_expr(param)) > 0:
                has_symbol = True
                break

        if has_symbol:
            out.append((gate_index, gate))
    return out


def _resolve_ansatz_layout(
    *,
    ansatz: AnsatzKind,
    num_qubits: int,
    layers: int,
    custom_ansatz_circuit: Optional[QuantumCircuit] = None,
) -> Tuple[List[str], QuantumCircuit]:
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
    if ansatz_name == "ucc":
        num_params = build_ucc_num_params(num_qubits, layers)
        param_names = [f"theta_{i}" for i in range(num_params)]
        symbolic_qc = build_ucc_ansatz_symbolic(
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
    raise ValueError("ansatz must be 'hardwareefficient', 'ucc', or 'custom'")


def _instantiate_transpiled_template(
    transpiled_template: QuantumCircuit,
    param_names: Sequence[str],
    params: np.ndarray,
) -> QuantumCircuit:
    """Clone transpiled template and materialize numeric parameter values in-place."""
    qc = transpiled_template.deepcopy()
    values = {name: float(params[i]) for i, name in enumerate(param_names)}
    qc.apply_value(values, deep=True)
    return qc


def _normalize_observable_values(values):
    if isinstance(values, list) and values:
        if isinstance(values[0], dict):
            merged: Dict[str, float] = {}
            for item in values:
                merged.update(item)
            return merged
        if len(values) == 1:
            return values[0]
    return values


def _ensure_observable_map(observables: Sequence[str], values) -> Dict[str, float]:
    if not observables:
        return {}
    values = _normalize_observable_values(values)
    if isinstance(values, dict):
        return {k: float(v) for k, v in values.items()}
    if len(observables) == 1:
        return {observables[0]: float(values)}
    raise RuntimeError("observable_values shape mismatch")


def _energy_from_expectations(hamiltonian: Hamiltonian, expectations: Dict[str, float]) -> float:
    return float(sum(coeff * expectations.get(obs, 0.0) for coeff, obs in hamiltonian))


# 24 single-qubit Clifford elements represented as U3(theta, phi, lambda)
# following Qiskit's u3 convention.
_CLIFFORD_U3_PARAMS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (np.pi, 0.0, np.pi),
    (np.pi, 0.0, 0.0),
    (np.pi, np.pi / 2.0, np.pi / 2.0),
    (np.pi, np.pi / 2.0, -np.pi / 2.0),
    (np.pi, -np.pi / 2.0, np.pi / 2.0),
    (np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0),
    (np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0),
    (np.pi / 2.0, np.pi / 2.0, np.pi / 2.0),
    (np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0),
    (np.pi / 2.0, 0.0, 0.0),
    (np.pi / 2.0, np.pi, 0.0),
    (np.pi / 2.0, 0.0, np.pi),
    (np.pi / 2.0, np.pi, np.pi),
    (-np.pi / 2.0, 0.0, 0.0),
    (-np.pi / 2.0, np.pi, 0.0),
    (-np.pi / 2.0, 0.0, np.pi),
    (-np.pi / 2.0, np.pi, np.pi),
    (0.0, np.pi / 2.0, np.pi / 2.0),
    (0.0, -np.pi / 2.0, -np.pi / 2.0),
    (0.0, np.pi / 2.0, -np.pi / 2.0),
    (0.0, -np.pi / 2.0, np.pi / 2.0),
    (0.0, np.pi, 0.0),
    (0.0, 0.0, np.pi),
)


def _randomize_single_qubit_gates_to_clifford(
    transpiled_template: QuantumCircuit,
    rng: np.random.Generator,
    single_qubit_gates: Sequence[Tuple[int, str]],
    num_non_clifford_gates: int = 0,
) -> Tuple[QuantumCircuit, Tuple[tuple, ...]]:
    """Randomize parameterized 1q gates with a mixed Clifford/non-Clifford policy.

    A configurable subset of gates is replaced by random single-qubit unitaries
    (sampled from a Haar-inspired U3 parameterization), while the remaining gates
    are replaced by random single-qubit Clifford instances.
    """
    qc = transpiled_template.deepcopy()
    signature: List[tuple] = []
    if not single_qubit_gates:
        raise ValueError("clifford fitting requires at least one parameterized single-qubit gate")

    n_total = len(single_qubit_gates)
    n_non_clifford = int(max(0, min(int(num_non_clifford_gates), n_total)))
    non_clifford_indices = set()
    if n_non_clifford > 0:
        chosen = rng.choice(np.arange(n_total), size=n_non_clifford, replace=False)
        non_clifford_indices = {int(i) for i in np.asarray(chosen).tolist()}

    new_gates = list(qc.gates)
    for local_idx, (gate_index, gate) in enumerate(single_qubit_gates):
        gate_info = new_gates[gate_index]
        qubit = int(gate_info[-1])

        if local_idx in non_clifford_indices:
            # Haar-inspired sampling for U3 parameterization.
            u = float(rng.uniform(0.0, 1.0))
            v = float(rng.uniform(0.0, 1.0))
            w = float(rng.uniform(0.0, 1.0))
            theta = float(2.0 * np.arcsin(np.sqrt(u)))
            phi = float(2.0 * np.pi * v)
            lam = float(2.0 * np.pi * w)
            kind = "u_haar"
        else:
            theta, phi, lam = _CLIFFORD_U3_PARAMS[int(rng.integers(0, len(_CLIFFORD_U3_PARAMS)))]
            theta = float(theta)
            phi = float(phi)
            lam = float(lam)
            kind = "u_clifford"

        new_gates[gate_index] = ("u", theta, phi, lam, qubit)
        signature.append((gate_index, gate, kind, theta, phi, lam, qubit))

    qc.gates = new_gates
    return qc, tuple(signature)


def _sample_unique_randomized_clifford_circuits(
    transpiled_template: QuantumCircuit,
    *,
    rng: np.random.Generator,
    num_samples: int,
    single_qubit_gates: Sequence[Tuple[int, str]],
    num_non_clifford_gates: int = 0,
) -> List[QuantumCircuit]:
    """Sample approximately unique mixed-randomized calibration circuits."""
    target = max(int(num_samples), 0)
    if target == 0:
        return []

    sampled: List[QuantumCircuit] = []
    seen = set()
    attempts = 0
    max_attempts = max(target * 20, 100)
    while len(sampled) < target and attempts < max_attempts:
        attempts += 1
        qc, signature = _randomize_single_qubit_gates_to_clifford(
            transpiled_template,
            rng,
            single_qubit_gates,
            num_non_clifford_gates=int(num_non_clifford_gates),
        )
        if signature in seen:
            continue
        seen.add(signature)
        sampled.append(qc)

    return sampled


def _fit_linear_clifford_map(noisy_values: Sequence[float], ideal_values: Sequence[float]) -> Tuple[float, float]:
    """Fit affine map ideal ~= a * noisy + b from calibration samples."""
    x = np.asarray(noisy_values, dtype=float)
    y = np.asarray(ideal_values, dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 1.0, 0.0
    if float(np.std(x)) <= 0.05 or float(np.std(y)) <= 0.05:
        a, b = 1.0, float(np.mean(y) - np.mean(x))
    else:
        try:
            a, b = np.polyfit(x, y, 1)
            if not (np.isfinite(a) and np.isfinite(b)):
                a, b = 1.0, float(np.mean(y) - np.mean(x))
        except (np.linalg.LinAlgError, ValueError):
            a, b = 1.0, float(np.mean(y) - np.mean(x))

    return float(a), float(b)


def _apply_clifford_fit(expectations: Dict[str, float], fit_map: Optional[CliffordFitMap]) -> Dict[str, float]:
    if not fit_map:
        return expectations
    corrected = dict(expectations)
    for obs, (a, b) in fit_map.items():
        if obs in corrected:
            corrected[obs] = float(np.clip(a * corrected[obs] + b, -1.0, 1.0))
    return corrected


def _build_clifford_fit_map(
    client,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    hamiltonian: Hamiltonian,
    shots: int,
    zne: bool,
    readout_mitigation: bool,
    transpiled_template: QuantumCircuit,
    num_samples: int,
    num_non_clifford_gates: int,
    seed: Optional[int],
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
) -> CliffordFitMap:
    """Pre-fit per-observable affine correction using Clifford calibration circuits."""
    if num_samples <= 0:
        return {}
    observables = list(dict.fromkeys(obs for _, obs in hamiltonian))
    fit_inputs_noisy: Dict[str, List[float]] = {obs: [] for obs in observables}
    fit_inputs_ideal: Dict[str, List[float]] = {obs: [] for obs in observables}

    rng = np.random.default_rng(seed)
    sim_backend = Backend("Simulator")
    single_qubit_gates = _build_parameterized_single_qubit_gate_list(transpiled_template)
    if not single_qubit_gates:
        raise ValueError("clifford fitting requires at least one parameterized single-qubit gate")

    sampled_clifford_circuits = _sample_unique_randomized_clifford_circuits(
        transpiled_template,
        rng=rng,
        num_samples=num_samples,
        single_qubit_gates=single_qubit_gates,
        num_non_clifford_gates=int(num_non_clifford_gates),
    )
    for si, clifford_qc in enumerate(sampled_clifford_circuits):
        noisy_qc = clifford_qc.deepcopy()
        ideal_qc = clifford_qc.deepcopy()

        _, noisy_expectations = _evaluate_energy_with_backend(
            client,
            noisy_qc,
            name=f"{name}_clifford_noisy_{si}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
        )

        _, ideal_expectations = _evaluate_energy_with_backend(
            client,
            ideal_qc,
            name=f"{name}_clifford_ideal_{si}",
            num_qubits=num_qubits,
            backend=sim_backend,
            chip_name="Simulator",
            shots=shots*10,
            hamiltonian=hamiltonian,
            zne=False,
            readout_mitigation=False,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
        )

        for obs in observables:
            fit_inputs_noisy[obs].append(float(noisy_expectations.get(obs, 0.0)))
            fit_inputs_ideal[obs].append(float(ideal_expectations.get(obs, 0.0)))

    fit_map: CliffordFitMap = {}
    for obs in observables:
        fit_map[obs] = _fit_linear_clifford_map(fit_inputs_noisy[obs], fit_inputs_ideal[obs])
    return fit_map


def _evaluate_energy_with_backend(
    client,
    qc: QuantumCircuit,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    shots: int,
    hamiltonian: Hamiltonian,
    zne: bool,
    readout_mitigation: bool,
    clifford_fit_map: Optional[CliffordFitMap] = None,
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
) -> Tuple[float, Dict[str, float]]:

    observables = [term[1] for term in hamiltonian]

    result = client._run_with_backend(
        qc,
        name,
        num_qubits,
        backend=backend,
        chip_name=chip_name,
        shots=shots,
        zne=zne,
        readout_mitigation=readout_mitigation,
        observables=observables,
        return_probabilities=False,
        print_true=False,
        transpile=False,
        target_qubits=target_qubits,
        qasm_version=qasm_version,
    )
    expectations_raw = _ensure_observable_map(observables, result.observable_values)
    expectations = _apply_clifford_fit(expectations_raw, clifford_fit_map)
    energy = _energy_from_expectations(hamiltonian, expectations)
    return energy, expectations


def _parameter_shift_gradient(
    client,
    params: np.ndarray,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    shots: int,
    hamiltonian: Hamiltonian,
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    param_template: Optional[QuantumCircuit] = None,
    param_names: Optional[Sequence[str]] = None,
    clifford_fit_map: Optional[CliffordFitMap] = None,
    target_qubits: Optional[Sequence[int]] = None,
    circuit_transform: Optional[Callable[[QuantumCircuit, Optional[int]], QuantumCircuit]] = None,
    qasm_version: str = "2.0",
) -> np.ndarray:
    """Compute gradients via parameter-shift rule."""
    if param_template is None or param_names is None:
        raise ValueError(
            "_parameter_shift_gradient requires param_template and param_names in current VQE flow"
        )

    grads = np.zeros_like(params, dtype=float)
    for i in range(params.size):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        qc_plus = _instantiate_transpiled_template(param_template, param_names, params_plus)
        qc_minus = _instantiate_transpiled_template(param_template, param_names, params_minus)
        if circuit_transform is not None:
            qc_plus = circuit_transform(qc_plus, i)
            qc_minus = circuit_transform(qc_minus, i)
        e_plus, _ = _evaluate_energy_with_backend(
            client,
            qc_plus,
            name=f"{name}_grad_p{i}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            clifford_fit_map=clifford_fit_map,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
        )
        e_minus, _ = _evaluate_energy_with_backend(
            client,
            qc_minus,
            name=f"{name}_grad_m{i}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            clifford_fit_map=clifford_fit_map,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
        )
        grads[i] = 0.5 * (e_plus - e_minus)
    return grads


def _adam_update(
    params: np.ndarray,
    grads: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * (grads ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v


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
) -> VQEResult:
    """Run VQE optimization on a specific backend."""
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
    compressed_param_names: Optional[List[str]] = None
    compressed_layers: Optional[int] = None
    if method == "autograd":
        if str(chip_name).lower() != "simulator":
            raise ValueError("autograd mode is only supported on Simulator backend")
        from ..sim import energy_and_expectations as _energy_and_expectations
        if seed is not None:
            torch.manual_seed(int(seed))
    else:
        if not enable_circuit_compression:
            # No compression: keep the existing fast path by transpiling symbolic template once.
            transpiled_template = client._transpile_with_backend(
                symbolic_qc,
                backend,
                target_qubits=target_qubits,
                use_dd=use_dd,
                use_gate_compressor=False,
            )
            gradient_param_template = transpiled_template
            target_qubits_in_use = client._ordered_target_qubits_from_layout(
                compiled_qc=transpiled_template,
                original_qc=symbolic_qc,
                num_qubits=num_qubits,
            )
        else:
            # Compression enabled:
            # 1) do NOT transpile original symbolic ansatz; instantiate it per-iteration/per-shift then compress.
            # 2) transpile one hardware-efficient compressed template, and only inject compressed params for execution.
            gradient_param_template = symbolic_qc
            compressed_layers = int(block_depth_k) if block_depth_k is not None else max(1, int(np.ceil(float(layers) * 0.5)))
            compressed_param_count = 2 * int(num_qubits) * (int(compressed_layers) + 1)
            compressed_param_names = [f"cmp_phi_{i}" for i in range(compressed_param_count)]
            compressed_symbolic_qc = build_hardware_efficient_ansatz_symbolic(
                num_qubits,
                compressed_param_names,
                layers=int(compressed_layers),
            )
            compressed_transpiled_template = client._transpile_with_backend(
                compressed_symbolic_qc,
                backend,
                target_qubits=target_qubits,
                use_dd=use_dd,
                use_gate_compressor=False,
            )
            target_qubits_in_use = client._ordered_target_qubits_from_layout(
                compiled_qc=compressed_transpiled_template,
                original_qc=compressed_symbolic_qc,
                num_qubits=num_qubits,
            )

    compression_warm_start: Optional[np.ndarray] = None
    compression_base_params: Optional[np.ndarray] = None
    compression_last_plan: Optional[HybridCompressionPlan] = None

    def _planner_stage_ids(plan: Optional[HybridCompressionPlan]) -> List[int]:
        if plan is None or plan.total_layers <= 0 or not plan.blocks:
            return [-1]
        return [-1] + [int(i) for i in range(len(plan.blocks))]

    def _stage_target_circuit(qc_bound: QuantumCircuit, plan: Optional[HybridCompressionPlan], stage_id: int) -> QuantumCircuit:
        if plan is None or plan.total_layers <= 0 or not plan.blocks:
            return qc_bound.deepcopy()

        if int(stage_id) == -1:
            if int(plan.split_layer) <= 0:
                return QuantumCircuit(int(num_qubits))
            return build_layer_span_circuit(
                qc_bound,
                start_layer=0,
                end_layer=int(plan.split_layer) - 1,
            )

        bid = int(stage_id)
        if bid < 0 or bid >= len(plan.blocks):
            return qc_bound.deepcopy()
        block = plan.blocks[bid]
        return build_layer_span_circuit(
            qc_bound,
            start_layer=int(block.start_layer),
            end_layer=int(block.end_layer),
        )

    def _compose_stage_circuits(stage_circuits: Sequence[QuantumCircuit]) -> QuantumCircuit:
        if not stage_circuits:
            return QuantumCircuit(int(num_qubits))

        out_nqubits = int(stage_circuits[0].nqubits)
        out_ncbits = int(stage_circuits[0].ncbits)
        out = QuantumCircuit(out_nqubits, out_ncbits)
        out.qubits = stage_circuits[0].qubits.copy()
        merged: List[tuple] = []
        for stage_qc in stage_circuits:
            if int(stage_qc.nqubits) != out_nqubits:
                raise ValueError("all stage circuits must have the same nqubits")
            merged.extend(list(stage_qc.gates))
        out.gates = merged
        return out

    def _maybe_compress_qc(qc_bound: QuantumCircuit, changed_param_index: Optional[int] = None) -> QuantumCircuit:
        nonlocal compression_warm_start
        nonlocal compression_base_params
        nonlocal compression_last_plan
        if method != "parameter-shift" or not enable_circuit_compression:
            return qc_bound

        plan: Optional[HybridCompressionPlan] = None
        if enable_block_planner:
            if changed_param_index is None or compression_last_plan is None:
                plan = plan_hybrid_suffix_blocks(
                    qc_bound,
                    bond_cap=int(unified_bond_cap),
                    trunc_tol=float(unified_trunc_tol),
                    max_layers_per_block=int(planner_max_layers_per_block),
                )
                compression_last_plan = plan
            else:
                plan = compression_last_plan

        stage_ids = _planner_stage_ids(plan)
        local_steps = int(compression_optimizer_steps)
        if changed_param_index is not None:
            local_steps = max(2, int(np.ceil(float(compression_optimizer_steps) * 0.7)))

        approx_depth = int(block_depth_k) if block_depth_k is not None else max(1, int(np.ceil(float(layers) * 0.5)))
        local_warm_start = compression_base_params if changed_param_index is not None else compression_warm_start
        compressed_stage_exec_circuits: List[QuantumCircuit] = []
        for stage_id in stage_ids:
            target_stage_qc = _stage_target_circuit(qc_bound, plan, int(stage_id))
            compressed_stage_qc, next_warm_start, summary = compress_circuit_with_hybrid_objective(
                target_stage_qc,
                num_qubits=num_qubits,
                approx_layers=approx_depth,
                optimizer_steps=int(local_steps),
                optimizer_lr=float(compression_optimizer_lr),
                objective_mode="mpo" if bool(plan is not None and int(stage_id) >= 0) else "mps",
                bond_cap=int(unified_bond_cap),
                warm_start_params=local_warm_start,
            )
            local_warm_start = next_warm_start

            if compressed_transpiled_template is not None and compressed_param_names is not None:
                stage_exec_qc = _instantiate_transpiled_template(
                    compressed_transpiled_template,
                    compressed_param_names,
                    next_warm_start,
                )
                compressed_stage_exec_circuits.append(stage_exec_qc)
            else:
                compressed_stage_exec_circuits.append(compressed_stage_qc)
            if bool(compression_verbose):
                call_tag = "base" if changed_param_index is None else f"shift(param={int(changed_param_index)})"
                print(
                    "[vqe] circuit compressed:",
                    f"call={call_tag}",
                    f"stage={int(stage_id)}",
                    f"layers={approx_depth}",
                    f"mode={summary['objective_mode']}",
                    f"init={summary['init_loss']:.3e}",
                    f"loss={summary['best_loss']:.3e}",
                    f"delta={summary['loss_delta']:.3e}",
                    f"inf={summary['objective_infidelity']:.3e}",
                )

            do_plot = bool(compression_plot_loss)
            if do_plot:
                try:
                    import matplotlib.pyplot as _plt

                    ys = summary.get("loss_history", [])
                    if isinstance(ys, list) and len(ys) > 0:
                        _plt.figure(figsize=(5.0, 3.2))
                        _plt.plot(range(len(ys)), ys, marker="o", ms=2)
                        _plt.xlabel("Compression Step")
                        _plt.ylabel("Loss")
                        _plt.title(f"Compression Loss (stage={int(stage_id)})")
                        _plt.grid(alpha=0.3)
                        _plt.tight_layout()
                        _plt.show()
                except Exception as _plot_exc:
                    if bool(compression_verbose):
                        print("[vqe] compression loss plotting skipped:", _plot_exc)

        compression_warm_start = local_warm_start
        if changed_param_index is None and compression_warm_start is not None:
            compression_base_params = compression_warm_start.copy()

        if plan is not None and bool(compression_verbose):
            print(
                "[vqe] hybrid block plan:",
                f"split={plan.split_layer}/{plan.total_layers}",
                f"seed_bond={plan.prefix_max_bond}",
                f"seed_err={plan.prefix_relative_trunc_error:.3e}",
                f"blocks={len(plan.blocks)}",
            )
        return _compose_stage_circuits(compressed_stage_exec_circuits)

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
            hamiltonian=hamiltonian,
            shots=shots,
            zne=zne,
            readout_mitigation=readout_mitigation,
            transpiled_template=clifford_transpiled_template,
            num_samples=int(clifford_fitting_num_samples),
            num_non_clifford_gates=int(clifford_fitting_num_non_clifford_gates),
            seed=None if seed is None else int(seed) + 7919,
            target_qubits=target_qubits_in_use,
            qasm_version=qasm_version,
        )
        clifford_fitting_summary = {
            obs: {"a": float(coeffs[0]), "b": float(coeffs[1])}
            for obs, coeffs in clifford_fit_map.items()
        }
        print(
            "[vqe] clifford fitting prepared:",
            f"terms={len(clifford_fit_map)}",
            f"samples={int(clifford_fitting_num_samples)}",
            f"non_clifford_gates={int(clifford_fitting_num_non_clifford_gates)}",
        )

    energy_history: List[float] = []
    params_history: List[List[float]] = []
    grad_history: List[List[float]] = []
    best_energy = float("inf")
    best_params = params.copy()
    last_expectations: Dict[str, float] = {}
    m = np.zeros_like(params, dtype=float)
    v = np.zeros_like(params, dtype=float)

    print(
        "[vqe] start optimization:",
        f"iters={max_iters}",
        f"layers={layers}",
        f"params={num_params}",
        f"ansatz={str(ansatz).lower()}",
        f"shots={shots}",
        f"shift={shift}",
        f"gradient={method}",
    )

    for it in range(max_iters):
        print(f"[vqe] iter {it} start")
        if method == "autograd":
            params_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
            energy_t, expectations = _energy_and_expectations(
                symbolic_qc,
                params=params_t,
                param_names=param_names,
                hamiltonian=hamiltonian,
            )
            energy_t.backward()
            energy = float(energy_t.detach().cpu().item())
            grads = params_t.grad.detach().cpu().numpy().astype(float, copy=True)
        else:
            if gradient_param_template is None:
                raise RuntimeError("gradient_param_template is not prepared for parameter-shift flow")
            qc = _instantiate_transpiled_template(gradient_param_template, param_names, params)
            qc = _maybe_compress_qc(qc, None)
            energy, expectations = _evaluate_energy_with_backend(
                client,
                qc,
                name=f"{name}_iter{it}",
                num_qubits=num_qubits,
                backend=backend,
                chip_name=chip_name,
                shots=shots,
                hamiltonian=hamiltonian,
                zne=zne,
                readout_mitigation=readout_mitigation,
                clifford_fit_map=clifford_fit_map,
                target_qubits=target_qubits_in_use,
                qasm_version=qasm_version,
            )
            grads = _parameter_shift_gradient(
                client,
                params,
                name=f"{name}_iter{it}",
                num_qubits=num_qubits,
                backend=backend,
                chip_name=chip_name,
                shots=shots,
                hamiltonian=hamiltonian,
                shift=shift,
                zne=zne,
                readout_mitigation=readout_mitigation,
                param_template=gradient_param_template,
                param_names=param_names,
                clifford_fit_map=clifford_fit_map,
                target_qubits=target_qubits_in_use,
                circuit_transform=_maybe_compress_qc if enable_circuit_compression else None,
                qasm_version=qasm_version,
            )

        grad_norm = float(np.linalg.norm(grads))
        print(f"[vqe] iter {it} energy={energy:.6f} grad_norm={grad_norm:.6f}")

        params, m, v = _adam_update(
            params,
            grads,
            m,
            v,
            it + 1,
            lr=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        energy_history.append(float(energy))
        params_history.append(params.tolist())
        grad_history.append(grads.tolist())
        last_expectations = expectations
        if energy < best_energy:
            best_energy = float(energy)
            best_params = params.copy()
            print(f"[vqe] iter {it} new best energy={best_energy:.6f}")

        if callback is not None:
            callback(it, float(energy), params)

    return VQEResult(
        best_energy=best_energy,
        best_params=best_params.tolist(),
        energy_history=energy_history,
        params_history=params_history,
        grad_history=grad_history,
        last_expectations=last_expectations,
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
    compression_optimizer_steps: int = 50
    compression_optimizer_lr: float = 0.05
    compression_verbose: bool = False
    compression_plot_loss: bool = False
    max_wait_time: int = 3600
    sleep_time: int = 5

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
        """Select hardware and run VQE optimization."""
        print(
            "[vqe] prepare run:",
            f"name={name}",
            f"num_qubits={num_qubits}",
            f"provider={provider}",
            f"model={model}",
            f"layers={self.layers}",
            f"shots={self.shots}",
            f"max_iters={self.max_iters}",
        )
        model = model.lower()
        params = model_params or {}
        if model == "ising":
            hamiltonian = build_ising_hamiltonian(num_qubits, **params)
        elif model == "heisenberg":
            hamiltonian = build_heisenberg_hamiltonian(num_qubits, **params)
        elif model == "xxz":
            hamiltonian = build_xxz_hamiltonian(num_qubits, **params)
        elif model == "xy":
            hamiltonian = build_xy_hamiltonian(num_qubits, **params)
        elif model == "custom":
            if hamiltonian is None:
                raise ValueError("custom model requires hamiltonian")
            hamiltonian = build_custom_hamiltonian(hamiltonian, num_qubits)
        else:
            raise ValueError(f"unsupported model: {model}")

        provider_name = str(provider).lower()
        qasm_version = self.client._default_qasm_version_for_provider(provider_name)
        use_dd = provider_name not in {"tianyan", "guodun"}
        runtime = create_provider_runtime(provider=provider_name, client=self.client)
        profiles = runtime.backend_adapter.discover_hardware(
            num_qubits=num_qubits,
            prefer_hardware=prefer_chips,
        )
        print("[vqe] candidate chips:", [p.hardware_name for p in profiles])
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
                print("[vqe] running on chip:", resolved.hardware_name)
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
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("all candidate chips failed to run VQE") from last_error


