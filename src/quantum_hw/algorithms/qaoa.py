"""QAOA construction and optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ..api.backend import Backend

from ..circuit import QuantumCircuit

from ..core.observables import pauli_support
from ..core.types import QAOAResult


Edge = Tuple[int, int]
Hamiltonian = List[Tuple[float, str]]


def _parse_pauli_string(pauli: str, num_qubits: int | None = None) -> List[Tuple[int, str]]:
    pauli = pauli.strip()
    if not pauli:
        raise ValueError("pauli string is empty")

    tokens = pauli.split()
    if len(tokens) == 1 and tokens[0].isalpha():
        if num_qubits is None:
            num_qubits = len(tokens[0])
        if len(tokens[0]) != num_qubits:
            raise ValueError("pauli length mismatch with num_qubits")
        return [(i, p.upper()) for i, p in enumerate(tokens[0]) if p.upper() != "I"]

    parsed = []
    for tok in tokens:
        op = tok[0].upper()
        idx = int(tok[1:])
        if op not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"unsupported Pauli: {op}")
        if op != "I":
            parsed.append((idx, op))
    if num_qubits is not None:
        for idx, _ in parsed:
            if idx < 0 or idx >= num_qubits:
                raise ValueError("pauli index out of range")
    return parsed


def _validate_edges(num_qubits: int, edges: Sequence[Edge], weights: Optional[Sequence[float]] = None):
    if not edges:
        raise ValueError("edges must be non-empty")
    for i, j in edges:
        if i == j:
            raise ValueError("edge endpoints must be different")
        if i < 0 or j < 0 or i >= num_qubits or j >= num_qubits:
            raise ValueError("edge index out of range")
    if weights is not None and len(weights) != len(edges):
        raise ValueError("weights length must match edges length")


def build_maxcut_hamiltonian(
    num_qubits: int,
    edges: Sequence[Edge],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[Hamiltonian, float]:
    """Build MaxCut cost Hamiltonian: sum w (1 - ZiZj) / 2."""
    _validate_edges(num_qubits, edges, weights)
    if weights is None:
        weights = [1.0 for _ in edges]
    terms: Hamiltonian = []
    constant = 0.0
    for (i, j), w in zip(edges, weights):
        constant += 0.5 * float(w)
        terms.append((-0.5 * float(w), f"Z{i} Z{j}"))
    return terms, constant


def build_custom_cost_hamiltonian(
    terms: Sequence[Tuple[float, str]],
    num_qubits: int,
    constant: float = 0.0,
) -> Tuple[Hamiltonian, float]:
    out: Hamiltonian = []
    for coeff, pauli in terms:
        if not isinstance(pauli, str) or not pauli.strip():
            raise ValueError("pauli term must be a non-empty string")
        _ = pauli_support(pauli, num_qubits=num_qubits)
        out.append((float(coeff), pauli))
    return out, float(constant)


def build_qaoa_circuit(
    num_qubits: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    edges: Sequence[Edge],
    weights: Optional[Sequence[float]] = None,
) -> QuantumCircuit:
    """Build a standard QAOA circuit for MaxCut."""
    _validate_edges(num_qubits, edges, weights)
    if weights is None:
        weights = [1.0 for _ in edges]
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")

    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)

    for gamma, beta in zip(gammas, betas):
        # Cost unitary for MaxCut.
        for (i, j), w in zip(edges, weights):
            qc.cx(i, j)
            qc.rz(2.0 * float(gamma) * float(w), j)
            qc.cx(i, j)
        # Mixer unitary.
        for q in range(num_qubits):
            qc.rx(2.0 * float(beta), q)
    return qc


def _apply_cost_unitary_terms(qc: QuantumCircuit, gamma: float, terms: Hamiltonian, num_qubits: int) -> None:
    for coeff, pauli in terms:
        parsed = _parse_pauli_string(pauli, num_qubits=num_qubits)
        if not parsed:
            continue
        if any(op != "Z" for _, op in parsed):
            raise ValueError("custom QAOA only supports Z/ZZ terms")
        if len(parsed) == 1:
            q = parsed[0][0]
            qc.rz(2.0 * float(gamma) * float(coeff), q)
        elif len(parsed) == 2:
            i, _ = parsed[0]
            j, _ = parsed[1]
            qc.cx(i, j)
            qc.rz(2.0 * float(gamma) * float(coeff), j)
            qc.cx(i, j)
        else:
            raise ValueError("custom QAOA supports at most 2-body Z terms")


def build_qaoa_circuit_from_terms(
    num_qubits: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    terms: Hamiltonian,
) -> QuantumCircuit:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)
    for gamma, beta in zip(gammas, betas):
        _apply_cost_unitary_terms(qc, gamma, terms, num_qubits)
        for q in range(num_qubits):
            qc.rx(2.0 * float(beta), q)
    return qc


def _ensure_observable_map(observables: Sequence[str], values) -> Dict[str, float]:
    if not observables:
        return {}
    if isinstance(values, dict):
        return {k: float(v) for k, v in values.items()}
    if len(observables) == 1:
        return {observables[0]: float(values)}
    raise RuntimeError("observable_values shape mismatch")


def _evaluate_maxcut_cost_with_backend(
    client,
    qc: QuantumCircuit,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    shots: int,
    terms: Hamiltonian,
    constant: float,
    zne: bool,
    readout_mitigation: bool,
    target_qubits: Optional[Sequence[int]],
) -> Tuple[float, Dict[str, float]]:
    observables = [term[1] for term in terms]
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
        target_qubits=target_qubits,
    )
    expectations = _ensure_observable_map(observables, result.observable_values)
    cost = constant + sum(coeff * expectations.get(obs, 0.0) for coeff, obs in terms)
    return float(cost), expectations


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
    ascent: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * (grads ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    step = lr * m_hat / (np.sqrt(v_hat) + eps)
    if ascent:
        params = params + step
    else:
        params = params - step
    return params, m, v


def _parameter_shift_gradient_qaoa(
    client,
    params: np.ndarray,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    shots: int,
    edges: Sequence[Edge],
    weights: Optional[Sequence[float]],
    terms: Hamiltonian,
    constant: float,
    use_terms: bool,
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    target_qubits: Optional[Sequence[int]],
) -> np.ndarray:
    """Parameter-shift gradient for QAOA parameters (gammas, betas)."""
    p = params.size // 2
    grads = np.zeros_like(params, dtype=float)
    for i in range(params.size):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift
        gammas_plus = params_plus[:p]
        betas_plus = params_plus[p:]
        gammas_minus = params_minus[:p]
        betas_minus = params_minus[p:]

        if use_terms:
            qc_plus = build_qaoa_circuit_from_terms(num_qubits, gammas_plus, betas_plus, terms)
            qc_minus = build_qaoa_circuit_from_terms(num_qubits, gammas_minus, betas_minus, terms)
        else:
            qc_plus = build_qaoa_circuit(num_qubits, gammas_plus, betas_plus, edges, weights=weights)
            qc_minus = build_qaoa_circuit(num_qubits, gammas_minus, betas_minus, edges, weights=weights)

        c_plus, _ = _evaluate_maxcut_cost_with_backend(
            client,
            qc_plus,
            name=f"{name}_grad_p{i}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            terms=terms,
            constant=constant,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
        )
        c_minus, _ = _evaluate_maxcut_cost_with_backend(
            client,
            qc_minus,
            name=f"{name}_grad_m{i}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            terms=terms,
            constant=constant,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
        )
        grads[i] = 0.5 * (c_plus - c_minus)
    return grads


def run_qaoa_with_backend(
    client,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    edges: Sequence[Edge],
    weights: Optional[Sequence[float]],
    terms: Optional[Hamiltonian],
    constant: float,
    p: int,
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
) -> QAOAResult:
    rng = np.random.default_rng(seed)
    num_params = 2 * p
    if init_params is None:
        params = rng.uniform(0.0, 2.0 * np.pi, size=num_params)
    else:
        params = np.asarray(init_params, dtype=float)
        if params.size != num_params:
            raise ValueError(f"init_params length must be {num_params}")

    use_terms = terms is not None
    if terms is None:
        terms, constant = build_maxcut_hamiltonian(num_qubits, edges, weights=weights)
    cost_history: List[float] = []
    params_history: List[List[float]] = []
    grad_history: List[List[float]] = []
    best_cost = float("-inf")
    best_params = params.copy()
    last_expectations: Dict[str, float] = {}
    m = np.zeros_like(params, dtype=float)
    v = np.zeros_like(params, dtype=float)

    for it in range(max_iters):
        gammas = params[:p]
        betas = params[p:]
        # Build the cost + mixer circuit at current parameters.
        if use_terms:
            qc = build_qaoa_circuit_from_terms(num_qubits, gammas, betas, terms)
        else:
            qc = build_qaoa_circuit(num_qubits, gammas, betas, edges, weights=weights)
        cost, expectations = _evaluate_maxcut_cost_with_backend(
            client,
            qc,
            name=f"{name}_iter{it}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            terms=terms,
            constant=constant,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
        )
        # Gradient via parameter-shift evaluations.
        grads = _parameter_shift_gradient_qaoa(
            client,
            params,
            name=f"{name}_iter{it}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            edges=edges,
            weights=weights,
            terms=terms,
            constant=constant,
            use_terms=use_terms,
            shift=shift,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
        )

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
            ascent=True,
        )

        cost_history.append(float(cost))
        params_history.append(params.tolist())
        grad_history.append(grads.tolist())
        last_expectations = expectations
        if cost > best_cost:
            best_cost = float(cost)
            best_params = params.copy()

        if callback is not None:
            callback(it, float(cost), params)

    return QAOAResult(
        best_cost=best_cost,
        best_params=best_params.tolist(),
        cost_history=cost_history,
        params_history=params_history,
        grad_history=grad_history,
        last_expectations=last_expectations,
    )


@dataclass
class QAOARunner:
    """High-level QAOA runner (MaxCut)."""

    client: object
    p: int = 1
    shots: int = 1024
    max_iters: int = 20
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    shift: float = np.pi / 2.0
    zne: bool = False
    readout_mitigation: bool = False
    seed: Optional[int] = None

    def run_maxcut(
        self,
        name: str,
        num_qubits: int,
        *,
        edges: Sequence[Edge],
        weights: Optional[Sequence[float]] = None,
        target_qubits: Optional[Sequence[int]] = None,
        init_params: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> QAOAResult:
        return self.client.run_qaoa(
            name=name,
            num_qubits=num_qubits,
            problem="maxcut",
            edges=edges,
            weights=weights,
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
            init_params=init_params,
            callback=callback,
            seed=self.seed,
            prefer_chips=prefer_chips,
            rank_weights=rank_weights,
        )
