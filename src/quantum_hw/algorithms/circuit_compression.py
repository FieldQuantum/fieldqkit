"""Reusable circuit compression utilities based on hybrid MPS/MPO objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from .ansatz_templates import build_hardware_efficient_ansatz as _build_hardware_efficient_ansatz
from .ansatz_templates import build_hardware_efficient_ansatz_symbolic as _build_hardware_efficient_ansatz_symbolic
from ..circuit import QuantumCircuit
from ..circuit.quantumcircuit_helpers import (
    functional_gates_available,
    one_qubit_gates_available,
    one_qubit_parameter_gates_available,
    three_qubit_gates_available,
    two_qubit_gates_available,
    two_qubit_parameter_gates_available,
)
from ..sim.mpo import simulate_mpo_process
from ..sim.mps import simulate_mps


@dataclass
class SuffixCompressionBlock:
    start_layer: int
    end_layer: int
    max_bond: int
    relative_trunc_error: float
    num_gates: int


@dataclass
class HybridCompressionPlan:
    split_layer: int
    total_layers: int
    prefix_max_bond: int
    prefix_relative_trunc_error: float
    blocks: List[SuffixCompressionBlock]


def _gate_qubits(gate_info) -> Tuple[int, ...]:
    gate = gate_info[0]

    if gate in one_qubit_gates_available:
        return (int(gate_info[1]),)
    if gate in one_qubit_parameter_gates_available:
        return (int(gate_info[-1]),)
    if gate in two_qubit_gates_available:
        return (int(gate_info[1]), int(gate_info[2]))
    if gate in two_qubit_parameter_gates_available:
        return (int(gate_info[-2]), int(gate_info[-1]))
    if gate in three_qubit_gates_available:
        return (int(gate_info[1]), int(gate_info[2]), int(gate_info[3]))

    if gate in functional_gates_available:
        if gate in {"reset", "measure"} and len(gate_info) > 1:
            return (int(gate_info[1]),)
        if gate == "barrier" and len(gate_info) > 1:
            qs = gate_info[1]
            if isinstance(qs, (list, tuple)):
                return tuple(int(q) for q in qs)
    return ()


def _circuit_to_moments(qc: QuantumCircuit) -> List[List[tuple]]:
    moments: List[List[tuple]] = []
    used_qubits: List[set[int]] = []
    for gate_info in qc.gates:
        qs = set(_gate_qubits(gate_info))

        if not qs:
            moments.append([gate_info])
            used_qubits.append(set())
            continue

        placed = False
        for i, used in enumerate(used_qubits):
            if used.isdisjoint(qs):
                moments[i].append(gate_info)
                used.update(qs)
                placed = True
                break
        if not placed:
            moments.append([gate_info])
            used_qubits.append(set(qs))
    return moments


def _build_circuit_from_moments(
    *,
    num_qubits: int,
    moments: Sequence[Sequence[tuple]],
    start: int,
    end: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    gates: List[tuple] = []
    for i in range(int(start), int(end) + 1):
        gates.extend(list(moments[i]))
    qc.gates = gates
    return qc


def build_layer_span_circuit(
    qc_bound: QuantumCircuit,
    *,
    start_layer: int,
    end_layer: int,
) -> QuantumCircuit:
    """Extract a contiguous layer span (inclusive) as a standalone circuit."""
    num_qubits = int(qc_bound.nqubits)
    moments = _circuit_to_moments(qc_bound)
    total_layers = len(moments)
    if total_layers == 0:
        return QuantumCircuit(num_qubits)

    s = int(start_layer)
    e = int(end_layer)
    if e < s:
        return QuantumCircuit(num_qubits)

    s = max(0, min(total_layers - 1, s))
    e = max(0, min(total_layers - 1, e))
    if e < s:
        return QuantumCircuit(num_qubits)

    return _build_circuit_from_moments(
        num_qubits=num_qubits,
        moments=moments,
        start=s,
        end=e,
    )


def _mps_max_bond(mps: Sequence[torch.Tensor]) -> int:
    if not mps:
        return 1
    return max(int(t.shape[2]) for t in mps)


def _mps_inner(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(a) != len(b):
        raise ValueError("MPS lengths must match")
    if not a:
        return torch.ones((), dtype=torch.complex128)
    env = torch.ones((1, 1), dtype=a[0].dtype, device=a[0].device)
    for ta, tb in zip(a, b):
        env = torch.einsum("ab,api,bpj->ij", env, torch.conj(ta), tb)
    return env.squeeze()


def _mps_relative_trunc_error(reference: Sequence[torch.Tensor], approx: Sequence[torch.Tensor]) -> float:
    num = torch.abs(_mps_inner(reference, approx)) ** 2
    den = torch.real(_mps_inner(reference, reference) * _mps_inner(approx, approx))
    den_val = float(den.detach().cpu().item())
    if den_val <= 0.0:
        return 0.0
    fidelity = float((num / den).real.detach().cpu().item())
    fidelity = max(0.0, min(1.0, fidelity))
    return float(1.0 - fidelity)


def _mps_infidelity_tensor(reference: Sequence[torch.Tensor], approx: Sequence[torch.Tensor]) -> torch.Tensor:
    num = torch.abs(_mps_inner(reference, approx)) ** 2
    den = torch.real(_mps_inner(reference, reference) * _mps_inner(approx, approx))
    den = torch.clamp(den, min=1e-15)
    fidelity = torch.real(num / den)
    fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
    return 1.0 - fidelity


def _mpo_inner(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(a) != len(b):
        raise ValueError("MPO lengths must match")
    if not a:
        return torch.ones((), dtype=torch.complex128)
    env = torch.ones((1, 1), dtype=a[0].dtype, device=a[0].device)
    for ta, tb in zip(a, b):
        env = torch.einsum("ab,apiq,bpjq->ij", env, torch.conj(ta), tb)
    return env.squeeze()


def _mpo_relative_trunc_error(reference: Sequence[torch.Tensor], approx: Sequence[torch.Tensor]) -> float:
    num = torch.abs(_mpo_inner(reference, approx)) ** 2
    den = torch.real(_mpo_inner(reference, reference) * _mpo_inner(approx, approx))
    den_val = float(den.detach().cpu().item())
    if den_val <= 0.0:
        return 0.0
    fidelity = float((num / den).real.detach().cpu().item())
    fidelity = max(0.0, min(1.0, fidelity))
    return float(1.0 - fidelity)


def _mpo_infidelity_tensor(reference: Sequence[torch.Tensor], approx: Sequence[torch.Tensor]) -> torch.Tensor:
    num = torch.abs(_mpo_inner(reference, approx)) ** 2
    den = torch.real(_mpo_inner(reference, reference) * _mpo_inner(approx, approx))
    den = torch.clamp(den, min=1e-15)
    fidelity = torch.real(num / den)
    fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
    return 1.0 - fidelity


def plan_hybrid_suffix_blocks(
    qc_bound: QuantumCircuit,
    *,
    bond_cap: int = 128,
    trunc_tol: float = 1e-8,
    max_layers_per_block: int = 6,
    device: torch.device | str | None = None,
) -> HybridCompressionPlan:
    """Build a coarse-grained suffix plan from bond and truncation-error thresholds."""
    if bond_cap <= 0:
        raise ValueError("bond_cap must be positive")
    if trunc_tol < 0.0:
        raise ValueError("trunc_tol must be non-negative")
    if max_layers_per_block <= 0:
        raise ValueError("max_layers_per_block must be positive")

    num_qubits = int(qc_bound.nqubits)
    moments = _circuit_to_moments(qc_bound)
    total_layers = len(moments)
    if total_layers == 0:
        return HybridCompressionPlan(
            split_layer=0,
            total_layers=0,
            prefix_max_bond=1,
            prefix_relative_trunc_error=0.0,
            blocks=[],
        )

    split_layer = 0
    prefix_max_bond = 1
    prefix_rel_err = 0.0
    for end in range(0, total_layers):
        qc_prefix = _build_circuit_from_moments(
            num_qubits=num_qubits,
            moments=moments,
            start=0,
            end=end,
        )
        mps_full = simulate_mps(qc_prefix, max_bond_dim=2*int(bond_cap), device=device)
        mps_cap = simulate_mps(qc_prefix, max_bond_dim=int(bond_cap), device=device)
        trial_max_bond = _mps_max_bond(mps_cap)
        trial_err = _mps_relative_trunc_error(mps_full, mps_cap)

        if trial_err <= float(trunc_tol):
            split_layer = end + 1
            prefix_max_bond = trial_max_bond
            prefix_rel_err = trial_err
            continue
        break

    blocks: List[SuffixCompressionBlock] = []
    i = split_layer
    while i < total_layers:
        best_end = i
        best_bond = 1
        best_err = 0.0

        limit = min(total_layers - 1, i + int(max_layers_per_block) - 1)
        for end in range(i, limit + 1):
            qc_block = _build_circuit_from_moments(
                num_qubits=num_qubits,
                moments=moments,
                start=i,
                end=end,
            )
            mpo_full = simulate_mpo_process(qc_block, max_bond_dim=2*int(bond_cap), device=device)
            mpo_cap = simulate_mpo_process(qc_block, max_bond_dim=int(bond_cap), device=device)
            trial_bond = max(int(t.shape[2]) for t in mpo_cap) if mpo_cap else 1
            trial_err = _mpo_relative_trunc_error(mpo_full, mpo_cap)

            if trial_err <= float(trunc_tol):
                best_end = end
                best_bond = trial_bond
                best_err = trial_err
                continue

            if end == i:
                best_end = i
                best_bond = trial_bond
                best_err = trial_err
            break

        num_gates = sum(len(moments[layer]) for layer in range(i, best_end + 1))
        blocks.append(
            SuffixCompressionBlock(
                start_layer=i,
                end_layer=best_end,
                max_bond=int(best_bond),
                relative_trunc_error=float(best_err),
                num_gates=int(num_gates),
            )
        )
        i = best_end + 1

    return HybridCompressionPlan(
        split_layer=int(split_layer),
        total_layers=int(total_layers),
        prefix_max_bond=int(prefix_max_bond),
        prefix_relative_trunc_error=float(prefix_rel_err),
        blocks=blocks,
    )


def compress_circuit_with_hybrid_objective(
    qc_bound: QuantumCircuit,
    *,
    num_qubits: int,
    approx_layers: int,
    optimizer_steps: int,
    optimizer_lr: float,
    objective_mode: Literal["mps", "mpo"] = "mps",
    bond_cap: int,
    warm_start_params: Optional[np.ndarray],
    device: torch.device | str | None = None,
) -> Tuple[QuantumCircuit, np.ndarray, Dict[str, object]]:
    """Fit a shallow hardware-efficient circuit to a bound circuit using MPS/MPO objectives."""
    if approx_layers <= 0:
        raise ValueError("approx_layers must be positive")
    if optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be positive")
    if optimizer_lr <= 0.0:
        raise ValueError("optimizer_lr must be positive")
    mode = str(objective_mode).lower()
    if mode not in {"mps", "mpo"}:
        raise ValueError("objective_mode must be 'mps' or 'mpo'")

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    dtype = torch.float64

    param_count = 2 * int(num_qubits) * (int(approx_layers) + 1)
    he_param_names = [f"phi_{i}" for i in range(param_count)]
    he_symbolic_qc = _build_hardware_efficient_ansatz_symbolic(
        int(num_qubits),
        he_param_names,
        layers=int(approx_layers),
    )

    if warm_start_params is not None and np.asarray(warm_start_params).size == param_count:
        init = np.asarray(warm_start_params, dtype=float).copy()
    else:
        # Use a small random init and pick the best seed by initial MPS infidelity.
        rng = np.random.default_rng(7)
        init = rng.normal(loc=0.0, scale=0.5, size=(param_count,)).astype(float)

    active_idx = np.arange(param_count, dtype=int)

    target_mps: Optional[List[torch.Tensor]] = None
    target_mpo: Optional[List[torch.Tensor]] = None
    if mode == "mps":
        target_mps = simulate_mps(
            qc_bound,
            max_bond_dim=int(bond_cap),
            device=device_obj,
        )
    else:
        target_mpo = simulate_mpo_process(
            qc_bound,
            max_bond_dim=int(bond_cap),
            device=device_obj,
        )

    def _objective_from_full_params(full_params_t: torch.Tensor) -> torch.Tensor:
        param_values = {name: full_params_t[i] for i, name in enumerate(he_param_names)}
        if mode == "mps":
            assert target_mps is not None
            approx_mps = simulate_mps(
                he_symbolic_qc,
                param_values=param_values,
                max_bond_dim=int(bond_cap),
                device=device_obj,
            )
            return _mps_infidelity_tensor(target_mps, approx_mps)
        else:
            assert target_mpo is not None
            approx_mpo = simulate_mpo_process(
                he_symbolic_qc,
                param_values=param_values,
                max_bond_dim=int(bond_cap),
                device=device_obj,
            )
            return _mpo_infidelity_tensor(target_mpo, approx_mpo)

    if warm_start_params is None:
        candidates = [init]
        rng = np.random.default_rng(17)
        for _ in range(2):
            candidates.append(rng.normal(loc=0.0, scale=0.12, size=(param_count,)).astype(float))

        best_seed = candidates[0]
        best_seed_loss = float("inf")
        with torch.no_grad():
            for cand in candidates:
                cand_t = torch.tensor(cand, dtype=dtype, device=device_obj)
                seed_loss = float(_objective_from_full_params(cand_t).detach().cpu().item())
                if seed_loss < best_seed_loss:
                    best_seed_loss = seed_loss
                    best_seed = cand
        init = best_seed

    init_t = torch.tensor(init, dtype=dtype, device=device_obj)
    active_t = torch.tensor(init[active_idx], dtype=dtype, device=device_obj, requires_grad=True)
    optimizer = torch.optim.Adam([active_t], lr=float(optimizer_lr))

    best_loss = float("inf")
    best_params = init.copy()
    loss_history: List[float] = []

    init_full_params = init_t.clone()
    init_loss = float(_objective_from_full_params(init_full_params).detach().cpu().item())
    loss_history.append(float(init_loss))

    for _ in range(int(optimizer_steps)):
        optimizer.zero_grad()
        full_params_t = init_t.clone()
        full_params_t[active_idx] = active_t
        loss = _objective_from_full_params(full_params_t)

        loss.backward()
        optimizer.step()

        cur = float(loss.detach().cpu().item())
        loss_history.append(float(cur))
        if cur < best_loss:
            best_loss = cur
            best_params = init.copy()
            best_params[active_idx] = active_t.detach().cpu().numpy().astype(float, copy=True)

    # If Adam gives little improvement, run a conservative refinement pass.
    if best_loss > init_loss * 0.995:
        refine_t = torch.tensor(best_params[active_idx], dtype=dtype, device=device_obj, requires_grad=True)
        refine_opt = torch.optim.Adam([refine_t], lr=float(optimizer_lr) * 0.2)
        refine_steps = max(4, int(np.ceil(float(optimizer_steps) * 0.5)))
        for _ in range(refine_steps):
            refine_opt.zero_grad()
            full_params_t = init_t.clone()
            full_params_t[active_idx] = refine_t
            refine_loss = _objective_from_full_params(full_params_t)
            refine_loss.backward()
            refine_opt.step()

            cur = float(refine_loss.detach().cpu().item())
            loss_history.append(float(cur))
            if cur < best_loss:
                best_loss = cur
                best_params = init.copy()
                best_params[active_idx] = refine_t.detach().cpu().numpy().astype(float, copy=True)

    compressed_qc = _build_hardware_efficient_ansatz(
        int(num_qubits),
        best_params.tolist(),
        layers=int(approx_layers),
    )

    with torch.no_grad():
        best_t = torch.tensor(best_params, dtype=dtype, device=device_obj)
        best_map = {name: best_t[i] for i, name in enumerate(he_param_names)}
        if mode == "mps":
            assert target_mps is not None
            approx_mps_best = simulate_mps(
                he_symbolic_qc,
                param_values=best_map,
                max_bond_dim=int(bond_cap),
                device=device_obj,
            )
            objective_inf = float(_mps_infidelity_tensor(target_mps, approx_mps_best).detach().cpu().item())
        else:
            assert target_mpo is not None
            approx_mpo_best = simulate_mpo_process(
                he_symbolic_qc,
                param_values=best_map,
                max_bond_dim=int(bond_cap),
                device=device_obj,
            )
            objective_inf = float(_mpo_infidelity_tensor(target_mpo, approx_mpo_best).detach().cpu().item())

    summary = {
        "objective_mode": mode,
        "objective_infidelity": float(objective_inf),
        "init_loss": float(init_loss),
        "best_loss": float(best_loss),
        "loss_delta": float(init_loss - best_loss),
        "loss_history": loss_history,
    }
    return compressed_qc, best_params, summary
