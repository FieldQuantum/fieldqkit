"""VQE Hamiltonian builders and optimization routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ..api.backend import Backend

from ..circuit import QuantumCircuit

from ..core.observables import pauli_support
from ..core.types import VQEResult


Hamiltonian = List[Tuple[float, str]]


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
    expected = 2 * num_qubits * layers
    if len(params) != expected:
        raise ValueError(f"params length must be {expected} (2 * num_qubits * layers)")

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
    return qc


def _ensure_observable_map(observables: Sequence[str], values) -> Dict[str, float]:
    if not observables:
        return {}
    if isinstance(values, dict):
        return {k: float(v) for k, v in values.items()}
    if len(observables) == 1:
        return {observables[0]: float(values)}
    raise RuntimeError("observable_values shape mismatch")


def _energy_from_expectations(hamiltonian: Hamiltonian, expectations: Dict[str, float]) -> float:
    return float(sum(coeff * expectations.get(obs, 0.0) for coeff, obs in hamiltonian))


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
    target_qubits: Optional[Sequence[int]],
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
        target_qubits=target_qubits,
    )
    expectations = _ensure_observable_map(observables, result.observable_values)
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
    layers: int,
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    target_qubits: Optional[Sequence[int]],
) -> np.ndarray:
    """Compute gradients via parameter-shift rule."""
    grads = np.zeros_like(params, dtype=float)
    for i in range(params.size):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        qc_plus = build_hardware_efficient_ansatz(num_qubits, params_plus, layers=layers)
        qc_minus = build_hardware_efficient_ansatz(num_qubits, params_minus, layers=layers)

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
) -> VQEResult:
    """Run VQE optimization on a specific backend using parameter-shift gradients."""
    rng = np.random.default_rng(seed)
    num_params = 2 * num_qubits * layers

    if init_params is None:
        params = rng.uniform(0.0, 2.0 * np.pi, size=num_params)
    else:
        params = np.asarray(init_params, dtype=float)
        if params.size != num_params:
            raise ValueError(f"init_params length must be {num_params}")

    energy_history: List[float] = []
    params_history: List[List[float]] = []
    grad_history: List[List[float]] = []
    best_energy = float("inf")
    best_params = params.copy()
    last_expectations: Dict[str, float] = {}
    m = np.zeros_like(params, dtype=float)
    v = np.zeros_like(params, dtype=float)

    for it in range(max_iters):
        # Forward energy evaluation.
        qc = build_hardware_efficient_ansatz(num_qubits, params, layers=layers)
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
            target_qubits=target_qubits,
        )
        # Gradient via two shifted evaluations per parameter.
        grads = _parameter_shift_gradient(
            client,
            params,
            name=f"{name}_iter{it}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            layers=layers,
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
        )

        energy_history.append(float(energy))
        params_history.append(params.tolist())
        grad_history.append(grads.tolist())
        last_expectations = expectations
        if energy < best_energy:
            best_energy = float(energy)
            best_params = params.copy()

        if callback is not None:
            callback(it, float(energy), params)

    return VQEResult(
        best_energy=best_energy,
        best_params=best_params.tolist(),
        energy_history=energy_history,
        params_history=params_history,
        grad_history=grad_history,
        last_expectations=last_expectations,
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
    beta2: float = 0.999
    eps: float = 1e-8
    shift: float = np.pi / 2.0
    zne: bool = False
    readout_mitigation: bool = False
    seed: Optional[int] = None

    def run_ising(
        self,
        name: str,
        num_qubits: int,
        *,
        j: float = 1.0,
        h: float = 1.0,
        target_qubits: Optional[Sequence[int]] = None,
        init_params: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> VQEResult:
        return self.client.run_vqe(
            name=name,
            num_qubits=num_qubits,
            model="ising",
            j=j,
            h=h,
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
            init_params=init_params,
            callback=callback,
            seed=self.seed,
            prefer_chips=prefer_chips,
            rank_weights=rank_weights,
        )

    def run_custom(
        self,
        name: str,
        num_qubits: int,
        *,
        hamiltonian: Sequence[Tuple[float, str]],
        target_qubits: Optional[Sequence[int]] = None,
        init_params: Optional[Sequence[float]] = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> VQEResult:
        return self.client.run_vqe(
            name=name,
            num_qubits=num_qubits,
            model="custom",
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
            init_params=init_params,
            callback=callback,
            seed=self.seed,
            prefer_chips=prefer_chips,
            rank_weights=rank_weights,
        )
