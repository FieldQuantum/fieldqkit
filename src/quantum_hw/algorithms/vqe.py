"""VQE Hamiltonian builders and optimization routines."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import torch
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from ..api.backend import Backend
from ..api.backend import rank_chips

from ..circuit import QuantumCircuit

from ..core.observables import pauli_support
from ..core.types import VQEResult

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


def build_hardware_efficient_ansatz(
    num_qubits: int,
    params: Sequence[float],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    """Hardware-efficient ansatz with RX/RY rotations and linear CZ entanglers."""
    expected = 2 * num_qubits * (layers + 1)
    if len(params) != expected:
        raise ValueError(f"params length must be {expected} (2 * num_qubits * (layers + 1))")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        # Single-qubit rotations.
        for q in range(num_qubits):
            qc.rx(float(params[idx]), q)
            idx += 1
        for q in range(num_qubits):
            qc.ry(float(params[idx]), q)
            idx += 1
        # Linear entangling layer.
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    # Single-qubit rotations.
    for q in range(num_qubits):
        qc.rx(float(params[idx]), q)
        idx += 1
    for q in range(num_qubits):
        qc.ry(float(params[idx]), q)
        idx += 1
    return qc


def _ucc_num_params(num_qubits: int, layers: int) -> int:
    """Return parameter count for a lightweight UCC-inspired ansatz."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if layers <= 0:
        raise ValueError("layers must be positive")
    # Per layer: singles on each qubit + nearest-neighbor pair excitations.
    return layers * (num_qubits + max(num_qubits - 1, 0))


def build_ucc_ansatz(
    num_qubits: int,
    params: Sequence[float],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    """Build a lightweight UCC-inspired ansatz with singles and pair excitations."""
    expected = _ucc_num_params(num_qubits, layers)
    if len(params) != expected:
        raise ValueError(f"params length must be {expected} for ucc ansatz")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        # Singles excitation proxy.
        for q in range(num_qubits):
            qc.ry(float(params[idx]), q)
            idx += 1
        # Pair excitation proxy on linear neighbors.
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            qc.ry(float(params[idx]), q + 1)
            qc.cx(q, q + 1)
            idx += 1
    return qc


def _build_hardware_efficient_ansatz_symbolic(
    num_qubits: int,
    param_names: Sequence[str],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    """Build the same ansatz as build_hardware_efficient_ansatz but with symbolic parameters."""
    expected = 2 * num_qubits * (layers + 1)
    if len(param_names) != expected:
        raise ValueError(f"param_names length must be {expected} (2 * num_qubits * (layers + 1))")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.rx(param_names[idx], q)
            idx += 1
        for q in range(num_qubits):
            qc.ry(param_names[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    for q in range(num_qubits):
        qc.rx(param_names[idx], q)
        idx += 1
    for q in range(num_qubits):
        qc.ry(param_names[idx], q)
        idx += 1
    return qc


def _build_ucc_ansatz_symbolic(
    num_qubits: int,
    param_names: Sequence[str],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    """Build UCC-inspired ansatz using symbolic parameter names."""
    expected = _ucc_num_params(num_qubits, layers)
    if len(param_names) != expected:
        raise ValueError(f"param_names length must be {expected} for ucc ansatz")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.ry(param_names[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            qc.ry(param_names[idx], q + 1)
            qc.cx(q, q + 1)
            idx += 1
    return qc


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


def _build_trainable_parametric_1q_gate_list(
    transpiled_template: QuantumCircuit,
    trainable_symbols: Sequence[str],
) -> List[Tuple[int, str]]:
    """Collect (gate_index, gate_name) for trainable 1q parameterized gates."""
    trainable_set = set(trainable_symbols)
    out: List[Tuple[int, str]] = []
    for gate_index, gate_info in enumerate(transpiled_template.gates):
        gate = gate_info[0]
        if gate not in {"rx", "ry", "rz", "p", "u", "u3", "r"}:
            continue
        depends_on_trainable = False
        for param in gate_info[1:-1]:
            if not isinstance(param, str):
                continue
            symbols = _extract_names_from_expr(param)
            if any(symbol in trainable_set for symbol in symbols):
                depends_on_trainable = True
                break
        if depends_on_trainable:
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
        symbolic_qc = _build_hardware_efficient_ansatz_symbolic(
            num_qubits,
            param_names,
            layers=layers,
        )
        return param_names, symbolic_qc
    if ansatz_name == "ucc":
        num_params = _ucc_num_params(num_qubits, layers)
        param_names = [f"theta_{i}" for i in range(num_params)]
        symbolic_qc = _build_ucc_ansatz_symbolic(
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


_CLIFFORD_ANGLES = np.array([0.0, np.pi / 2.0, np.pi, 1.5 * np.pi], dtype=float)

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


def _randomize_parametric_1q_gates_to_clifford(
    transpiled_template: QuantumCircuit,
    rng: np.random.Generator,
    trainable_parametric_1q_gates: Sequence[Tuple[int, str]],
) -> Tuple[QuantumCircuit, Tuple[tuple, ...]]:
    """Replace each 1q parametric gate with a random 1q Clifford gate instance."""
    qc = transpiled_template.deepcopy()
    signature: List[tuple] = []
    if not trainable_parametric_1q_gates:
        raise ValueError("clifford fitting requires at least one single-qubit parameterized gate")

    new_gates = list(qc.gates)
    for gate_index, gate in trainable_parametric_1q_gates:
        gate_info = new_gates[gate_index]
        qubit = int(gate_info[-1])

        if gate in {"rx", "ry", "rz", "p"}:
            theta = float(_CLIFFORD_ANGLES[int(rng.integers(0, len(_CLIFFORD_ANGLES)))])
            new_gates[gate_index] = (gate, theta, qubit)
            signature.append((gate_index, gate, theta, qubit))
            continue

        if gate in {"u", "u3"}:
            theta, phi, lam = _CLIFFORD_U3_PARAMS[int(rng.integers(0, len(_CLIFFORD_U3_PARAMS)))]
            theta = float(theta)
            phi = float(phi)
            lam = float(lam)
            new_gates[gate_index] = (gate, theta, phi, lam, qubit)
            signature.append((gate_index, gate, theta, phi, lam, qubit))
            continue

        if gate == "r":
            theta = float(_CLIFFORD_ANGLES[int(rng.integers(0, len(_CLIFFORD_ANGLES)))])
            phi = float(_CLIFFORD_ANGLES[int(rng.integers(0, len(_CLIFFORD_ANGLES)))])
            new_gates[gate_index] = (gate, theta, phi, qubit)
            signature.append((gate_index, gate, theta, phi, qubit))
            continue

        raise ValueError(f"unsupported trainable gate in clifford fitting: {gate}")

    qc.gates = new_gates
    return qc, tuple(signature)


def _sample_unique_randomized_clifford_circuits(
    transpiled_template: QuantumCircuit,
    *,
    rng: np.random.Generator,
    num_samples: int,
    trainable_parametric_1q_gates: Sequence[Tuple[int, str]],
) -> List[QuantumCircuit]:
    """Sample approximately unique Clifford-randomized circuits."""
    target = max(int(num_samples), 0)
    if target == 0:
        return []

    sampled: List[QuantumCircuit] = []
    seen = set()
    attempts = 0
    max_attempts = max(target * 20, 100)
    while len(sampled) < target and attempts < max_attempts:
        attempts += 1
        qc, signature = _randomize_parametric_1q_gates_to_clifford(
            transpiled_template,
            rng,
            trainable_parametric_1q_gates,
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
    if float(np.std(x)) <= 1e-2 or float(np.std(y)) <= 1e-2:
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
    param_names: Sequence[str],
    num_samples: int,
    seed: Optional[int],
    target_qubits: Optional[Sequence[int]] = None,
) -> CliffordFitMap:
    """Pre-fit per-observable affine correction using Clifford calibration circuits."""
    if num_samples <= 0:
        return {}
    fit_inputs_noisy: Dict[str, List[float]] = {obs: [] for _, obs in hamiltonian}
    fit_inputs_ideal: Dict[str, List[float]] = {obs: [] for _, obs in hamiltonian}

    rng = np.random.default_rng(seed)
    sim_backend = Backend("Simulator")
    trainable_parametric_1q_gates = _build_trainable_parametric_1q_gate_list(
        transpiled_template,
        param_names,
    )
    if not trainable_parametric_1q_gates:
        raise ValueError("clifford fitting requires at least one trainable single-qubit parameterized gate")

    sampled_clifford_circuits = _sample_unique_randomized_clifford_circuits(
        transpiled_template,
        rng=rng,
        num_samples=num_samples,
        trainable_parametric_1q_gates=trainable_parametric_1q_gates,
    )

    for si, clifford_qc in enumerate(sampled_clifford_circuits):

        _, noisy_expectations = _evaluate_energy_with_backend(
            client,
            clifford_qc.deepcopy(),
            name=f"{name}_clifford_noisy_{si}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
        )

        _, ideal_expectations = _evaluate_energy_with_backend(
            client,
            clifford_qc.deepcopy(),
            name=f"{name}_clifford_ideal_{si}",
            num_qubits=num_qubits,
            backend=sim_backend,
            chip_name="Simulator",
            shots=shots*10,
            hamiltonian=hamiltonian,
            zne=False,
            readout_mitigation=False,
            target_qubits=target_qubits,
        )

        for _, obs in hamiltonian:
            fit_inputs_noisy[obs].append(float(noisy_expectations.get(obs, 0.0)))
            fit_inputs_ideal[obs].append(float(ideal_expectations.get(obs, 0.0)))

    fit_map: CliffordFitMap = {}
    for _, obs in hamiltonian:
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
    transpiled_template: Optional[QuantumCircuit] = None,
    param_names: Optional[Sequence[str]] = None,
    clifford_fit_map: Optional[CliffordFitMap] = None,
    target_qubits: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute gradients via parameter-shift rule."""
    if transpiled_template is None or param_names is None:
        raise ValueError(
            "_parameter_shift_gradient requires transpiled_template and param_names in current VQE flow"
        )

    grads = np.zeros_like(params, dtype=float)
    for i in range(params.size):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        qc_plus = _instantiate_transpiled_template(transpiled_template, param_names, params_plus)
        qc_minus = _instantiate_transpiled_template(transpiled_template, param_names, params_minus)

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
) -> VQEResult:
    """Run VQE optimization on a specific backend."""
    method = str(gradient_method).lower()
    if method not in {"parameter-shift", "autograd"}:
        raise ValueError("gradient_method must be 'parameter-shift' or 'autograd'")

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
    if method == "autograd":
        if str(chip_name).lower() != "simulator":
            raise ValueError("autograd mode is only supported on Simulator backend")
        from ..sim.statevector import energy_and_expectations as _energy_and_expectations
        if seed is not None:
            torch.manual_seed(int(seed))
    else:
        # For parameter-shift mode, transpile a symbolic ansatz once and then only update values.
        transpiled_template = client._transpile_with_backend(
            symbolic_qc,
            backend,
            target_qubits=target_qubits,
            use_gate_compressor=False,
        )
        target_qubits_in_use = client._ordered_target_qubits_from_layout(
            compiled_qc=transpiled_template,
            original_qc=symbolic_qc,
            num_qubits=num_qubits,
        )

    clifford_fit_map: Optional[CliffordFitMap] = None
    clifford_fitting_summary: Optional[Dict[str, Dict[str, float]]] = None
    if clifford_fitting:
        if method != "parameter-shift":
            raise ValueError("clifford_fitting currently requires gradient_method='parameter-shift'")
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
            transpiled_template=transpiled_template,
            param_names=param_names,
            num_samples=int(clifford_fitting_num_samples),
            seed=None if seed is None else int(seed) + 7919,
            target_qubits=target_qubits_in_use
        )
        clifford_fitting_summary = {
            obs: {"a": float(coeffs[0]), "b": float(coeffs[1])}
            for obs, coeffs in clifford_fit_map.items()
        }
        print(
            "[vqe] clifford fitting prepared:",
            f"terms={len(clifford_fit_map)}",
            f"samples={int(clifford_fitting_num_samples)}",
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
            qc = _instantiate_transpiled_template(transpiled_template, param_names, params)
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
                transpiled_template=transpiled_template,
                param_names=param_names,
                clifford_fit_map=clifford_fit_map,
                target_qubits=target_qubits_in_use,

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
    seed: Optional[int] = None
    gradient_method: Literal["parameter-shift", "autograd"] = "parameter-shift"

    def run_model(
        self,
        name: str,
        num_qubits: int,
        *,
        model: str = "ising",
        model_params: Optional[Dict[str, float]] = None,
        hamiltonian: Optional[Sequence[Tuple[float, str]]] = None,
        target_qubits: Optional[Sequence[int]] = None,
        init_params: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
        ansatz: AnsatzKind = "hardwareefficient",
        custom_ansatz_circuit: Optional[QuantumCircuit] = None,
    ) -> VQEResult:
        """Select hardware and run VQE optimization."""
        print(
            "[vqe] prepare run:",
            f"name={name}",
            f"num_qubits={num_qubits}",
            f"model={model}",
            f"layers={self.layers}",
            f"shots={self.shots}",
            f"max_iters={self.max_iters}",
        )
        run_ansatz = str(ansatz).lower()
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

        ranked_chips = rank_chips(
            self.client.tmgr,
            num_qubits=num_qubits,
            prefer_chips=prefer_chips,
            weights=rank_weights,
        )
        print("[vqe] candidate chips:", ranked_chips)
        if not ranked_chips:
            raise RuntimeError("no available chips satisfy num_qubits requirement")

        last_error: Optional[Exception] = None
        for chip_name in ranked_chips:
            backend = Backend(chip_name)
            self.client.chip_name = chip_name
            self.client.chip_backend = backend
            try:
                print("[vqe] running on chip:", chip_name)
                return run_vqe_with_backend(
                    self.client,
                    name=name,
                    num_qubits=num_qubits,
                    backend=backend,
                    chip_name=chip_name,
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
                    ansatz=run_ansatz,
                    custom_ansatz_circuit=custom_ansatz_circuit,
                    clifford_fitting=self.clifford_fitting,
                    clifford_fitting_num_samples=self.clifford_fitting_num_samples,
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("all candidate chips failed to run VQE") from last_error


