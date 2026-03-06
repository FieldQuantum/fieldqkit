"""VQE Hamiltonian builders and optimization routines."""

from __future__ import annotations

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
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    transpiled_template: Optional[QuantumCircuit] = None,
    param_names: Optional[Sequence[str]] = None,
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
) -> VQEResult:
    """Run VQE optimization on a specific backend."""
    method = str(gradient_method).lower()
    if method not in {"parameter-shift", "autograd"}:
        raise ValueError("gradient_method must be 'parameter-shift' or 'autograd'")

    num_params = 2 * num_qubits * (layers + 1)
    if init_params is None:
        rng = np.random.default_rng(seed)
        init_values = rng.uniform(0.0, 2.0 * np.pi, size=num_params)
    else:
        init_values = np.asarray(init_params, dtype=float)
        if init_values.size != num_params:
            raise ValueError(f"init_params length must be {num_params}")

    params = init_values.copy()
    param_names = [f"theta_{i}" for i in range(num_params)]
    symbolic_qc = _build_hardware_efficient_ansatz_symbolic(
        num_qubits,
        param_names,
        layers=layers,
    )
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
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("all candidate chips failed to run VQE") from last_error


