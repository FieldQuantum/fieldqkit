"""Reusable circuit compression utilities based on hybrid MPS/MPO objectives."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


@dataclass
class SuffixCompressionBlock:
    """A contiguous block of circuit layers that can be compressed via MPS truncation."""

    start_layer: int
    end_layer: int
    max_bond: int
    relative_trunc_error: float
    num_gates: int


@dataclass
class HybridCompressionPlan:
    """Plan splitting a circuit into an exact prefix and MPS-compressed suffix blocks."""

    split_layer: int
    total_layers: int
    prefix_max_bond: int
    prefix_relative_trunc_error: float
    blocks: List[SuffixCompressionBlock]


def _gate_qubits(gate_info) -> Tuple[int, ...]:
    """Return the qubits involved in a gate.

    Args:
        gate_info: Tuple describing a single gate, e.g. ``('rx', theta, qubit)``.

    Returns:
        Tuple of qubit indices touched by the gate.
    """
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
    """Convert a quantum circuit to a list of moments (first-fit greedy).

    Args:
        qc (*QuantumCircuit*): Quantum circuit.

    Returns:
        List of moments, each a list of gate-info tuples.
    """
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
    """Reassemble a sub-range of circuit moments into a new circuit.

    Args:
        num_qubits (*int*): Number of qubits.
        moments (*Sequence[Sequence[tuple]]*): Full list of circuit moments.
        start (*int*): First moment index (inclusive).
        end (*int*): Last moment index (inclusive).

    Returns:
        Constructed ``QuantumCircuit``.
    """
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
    """Extract a contiguous layer span (inclusive) as a standalone circuit.

    Args:
        qc_bound (*QuantumCircuit*): Source circuit with concrete parameter values.
        start_layer (*int*): First layer index (inclusive).
        end_layer (*int*): Last layer index (inclusive).

    Returns:
        Constructed ``QuantumCircuit``.
    """
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
    """Return the maximum bond dimension in an MPS.

    Args:
        mps (*Sequence[torch.Tensor]*): List of MPS site tensors.

    Returns:
        Maximum bond dimension across all MPS bonds.
    """
    if not mps:
        return 1
    return max(int(t.shape[2]) for t in mps)


def _mps_inner(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the inner product of two MPS.

    Args:
        a (*Sequence[torch.Tensor]*): First MPS (bra).
        b (*Sequence[torch.Tensor]*): Second MPS (ket).

    Returns:
        Scalar ``torch.Tensor`` with the inner product ``⟨a|b⟩``.

    Raises:
        ValueError: MPS lengths must match
    """
    if len(a) != len(b):
        raise ValueError("MPS lengths must match")
    if not a:
        return torch.ones((), dtype=torch.complex128)
    env = torch.ones((1, 1), dtype=a[0].dtype, device=a[0].device)
    for ta, tb in zip(a, b):
        env = torch.einsum("ab,api,bpj->ij", env, torch.conj(ta), tb)
    return env.squeeze()


def _relative_trunc_error(
    inner_fn,
    reference: Sequence[torch.Tensor],
    approx: Sequence[torch.Tensor],
) -> float:
    """Return the relative truncation error between two tensor networks.

    Works for both MPS (using ``_mps_inner``) and MPO (using ``_mpo_inner``).

    Args:
        inner_fn: Inner product function (``_mps_inner`` or ``_mpo_inner``).
        reference (*Sequence[torch.Tensor]*): Reference (exact) tensor network.
        approx (*Sequence[torch.Tensor]*): Approximate (truncated) tensor network.

    Returns:
        Relative truncation error ``1 − |⟨ref|approx⟩|²/(⟨ref|ref⟩⟨approx|approx⟩)``.
    """
    num = torch.abs(inner_fn(reference, approx)) ** 2
    den = torch.real(inner_fn(reference, reference) * inner_fn(approx, approx))
    den_val = float(den.detach().cpu().item())
    if den_val <= 0.0:
        return 0.0
    fidelity = float((num / den).real.detach().cpu().item())
    fidelity = max(0.0, min(1.0, fidelity))
    return float(1.0 - fidelity)


def _mpo_inner(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the inner product of two MPO.

    Args:
        a (*Sequence[torch.Tensor]*): First MPO (bra).
        b (*Sequence[torch.Tensor]*): Second MPO (ket).

    Returns:
        Scalar ``torch.Tensor`` with the Hilbert–Schmidt inner product ``Tr(A†B)``.

    Raises:
        ValueError: MPO lengths must match
    """
    if len(a) != len(b):
        raise ValueError("MPO lengths must match")
    if not a:
        return torch.ones((), dtype=torch.complex128)
    env = torch.ones((1, 1), dtype=a[0].dtype, device=a[0].device)
    for ta, tb in zip(a, b):
        env = torch.einsum("ab,apiq,bpjq->ij", env, torch.conj(ta), tb)
    return env.squeeze()


def plan_hybrid_suffix_blocks(
    qc_bound: QuantumCircuit,
    *,
    bond_cap: int = 128,
    trunc_tol: float = 1e-8,
    max_layers_per_block: int = 6,
    device: torch.device | str | None = None,
) -> HybridCompressionPlan:
    """Build a coarse-grained suffix plan from bond and truncation-error thresholds.

    Args:
        qc_bound (*QuantumCircuit*): Bound circuit to analyse for compression.
        bond_cap (*int*): Maximum allowed bond dimension per block. Defaults to ``128``.
        trunc_tol (*float*): Truncation error tolerance for prefix/block splitting. Defaults to ``1e-08``.
        max_layers_per_block (*int*): Maximum number of circuit layers per suffix block. Defaults to ``6``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``HybridCompressionPlan`` describing the prefix split point and suffix blocks.

    Raises:
        ValueError: bond_cap must be positive
        ValueError: trunc_tol must be non-negative
        ValueError: max_layers_per_block must be positive
    """
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
        trial_err = _relative_trunc_error(_mps_inner, mps_full, mps_cap)

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
            trial_err = _relative_trunc_error(_mpo_inner, mpo_full, mpo_cap)

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


def compile_tn_1d(
    target_tn: List[torch.Tensor],
    *,
    num_qubits: int,
    approx_layers: int,
    optimizer_steps: int,
    optimizer_lr: float,
    objective_mode: Literal["mps", "mpo"] = "mps",
    bond_cap: Optional[int] = None,
    warm_start_params: Optional[np.ndarray] = None,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> Tuple[QuantumCircuit, np.ndarray, Dict[str, object]]:
    """Compile a 1-D tensor network (MPS or MPO) into a hardware-efficient ansatz circuit.

    For ``objective_mode='mps'``, *target_tn* is a list of MPS site tensors
    of shape ``(bond_l, 2, bond_r)``.
    For ``objective_mode='mpo'``, *target_tn* is a list of MPO site tensors
    of shape ``(bond_l, 2, bond_r, 2)``.

    Optimization is two-stage:
      1. Main: Adam with full *optimizer_lr* for *optimizer_steps* iterations.
      2. Refine (only if ``best_loss > init_loss * 0.995``): lr × 0.2 for
         ``max(4, ceil(optimizer_steps * 0.5))`` additional steps.

    Args:
        target_tn: Target TN as a list of site tensors.
        num_qubits: Number of qubits.
        approx_layers: Number of HEA layers.
        optimizer_steps: Number of Adam steps.
        optimizer_lr: Learning rate.
        objective_mode: ``'mps'`` for state infidelity, ``'mpo'`` for process
            infidelity. Defaults to ``'mps'``.
        bond_cap: Bond dimension cap for the ansatz simulation.
            ``None`` keeps all bond dimensions (no truncation). Defaults to ``None``.
        warm_start_params: Optional initial parameter array.
        device: Torch device. Defaults to ``None`` (CPU).
        verbose: Emit per-step optimization progress via ``logging`` (INFO
            level on this module's logger). Defaults to ``False`` (silent).

    Returns:
        Tuple of ``(compiled_circuit, optimized_params, summary_dict)`` where
        *summary_dict* contains keys: ``'objective_mode'`` (*str*),
        ``'objective_infidelity'`` (*float*), ``'init_loss'`` (*float*),
        ``'best_loss'`` (*float*), ``'loss_delta'`` (*float*),
        ``'loss_history'`` (*List[float]*).

    Raises:
        ValueError: target_tn must be a non-empty list of site tensors
        ValueError: approx_layers must be positive
        ValueError: optimizer_steps must be positive
        ValueError: optimizer_lr must be positive
        ValueError: objective_mode must be 'mps' or 'mpo'
    """
    if not target_tn:
        raise ValueError("target_tn must be a non-empty list of site tensors")
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

    target = [t.to(device=device_obj) for t in target_tn]

    if mode == "mps":
        _inner_fn = _mps_inner
        _simulate_ansatz = lambda pv: simulate_mps(
            he_symbolic_qc, param_values=pv,
            max_bond_dim=bond_cap, device=device_obj,
        )
    else:
        _inner_fn = _mpo_inner
        _simulate_ansatz = lambda pv: simulate_mpo_process(
            he_symbolic_qc, param_values=pv,
            max_bond_dim=bond_cap, device=device_obj,
        )

    param_count = 2 * int(num_qubits) * (int(approx_layers) + 1)
    he_param_names = [f"phi_{i}" for i in range(param_count)]
    he_symbolic_qc = _build_hardware_efficient_ansatz_symbolic(
        int(num_qubits), he_param_names, layers=int(approx_layers),
    )

    # --- seed selection ---
    if warm_start_params is not None and np.asarray(warm_start_params).size == param_count:
        init = np.asarray(warm_start_params, dtype=float).copy()
    else:
        rng = np.random.default_rng(7)
        init = rng.uniform(-np.pi, np.pi, size=(param_count,)).astype(float)

    # --- optimization ---
    init_t = torch.tensor(init, dtype=dtype, device=device_obj)
    active_t = torch.tensor(init, dtype=dtype, device=device_obj, requires_grad=True)
    optimizer = torch.optim.Adam([active_t], lr=float(optimizer_lr))

    # Precompute target norm (constant across all optimization steps)
    with torch.no_grad():
        _target_norm_sq = torch.real(_inner_fn(target, target))

    def _objective(params_t: torch.Tensor) -> torch.Tensor:
        pv = {name: params_t[i] for i, name in enumerate(he_param_names)}
        approx = _simulate_ansatz(pv)
        num = torch.abs(_inner_fn(target, approx)) ** 2
        den = torch.clamp(_target_norm_sq * torch.real(_inner_fn(approx, approx)), min=1e-300)
        fid = torch.clamp(num / den, min=1e-300, max=1.0)
        return -torch.log(fid)

    init_loss = float(_objective(init_t).detach().cpu().item())
    best_loss = init_loss
    best_params = init.copy()
    loss_history: List[float] = [init_loss]

    _total_main = int(optimizer_steps)
    for _step in range(1, _total_main + 1):
        optimizer.zero_grad()
        loss = _objective(active_t)
        loss.backward()
        optimizer.step()

        cur = float(loss.detach().cpu().item())
        loss_history.append(cur)
        if cur < best_loss:
            best_loss = cur
            best_params = active_t.detach().cpu().numpy().astype(float, copy=True)
        if verbose and (_step % 20 == 0 or _step == _total_main):
            cur_fid = float(np.exp(-cur))
            best_fid = float(np.exp(-best_loss))
            logger.info("    compile_tn_1d main %d/%d  fid=%.8f  best_fid=%.8f  -logF=%.4f", _step, _total_main, cur_fid, best_fid, cur)

    # Refinement if little improvement.
    if best_loss > init_loss * 0.995:
        refine_t = torch.tensor(best_params, dtype=dtype, device=device_obj, requires_grad=True)
        refine_opt = torch.optim.Adam([refine_t], lr=float(optimizer_lr) * 0.2)
        refine_steps = max(4, int(np.ceil(float(optimizer_steps) * 0.5)))
        if verbose:
            logger.info("    compile_tn_1d entering refinement (%d steps, lr=%.4f)", refine_steps, float(optimizer_lr) * 0.2)
        for _rstep in range(1, refine_steps + 1):
            refine_opt.zero_grad()
            loss = _objective(refine_t)
            loss.backward()
            refine_opt.step()

            cur = float(loss.detach().cpu().item())
            loss_history.append(cur)
            if cur < best_loss:
                best_loss = cur
                best_params = refine_t.detach().cpu().numpy().astype(float, copy=True)
            if verbose and (_rstep % 100 == 0 or _rstep == refine_steps):
                cur_fid = float(np.exp(-cur))
                best_fid = float(np.exp(-best_loss))
                logger.info("    compile_tn_1d refine %d/%d  fid=%.8f  best_fid=%.8f  -logF=%.4f", _rstep, refine_steps, cur_fid, best_fid, cur)

    # --- build final circuit ---
    compiled_qc = _build_hardware_efficient_ansatz(
        int(num_qubits), best_params.tolist(), layers=int(approx_layers),
    )

    best_fid_final = float(np.exp(-best_loss))
    objective_inf = 1.0 - best_fid_final
    if verbose:
        logger.info("    compile_tn_1d done: infidelity=%.6e, fidelity=%.8f", objective_inf, best_fid_final)

    summary = {
        "objective_mode": mode,
        "objective_infidelity": objective_inf,
        "init_loss": init_loss,
        "best_loss": best_loss,
        "loss_delta": init_loss - best_loss,
        "loss_history": loss_history,
    }
    return compiled_qc, best_params, summary


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
    verbose: bool = False,
) -> Tuple[QuantumCircuit, np.ndarray, Dict[str, object]]:
    """Fit a shallow hardware-efficient circuit to a bound circuit using MPS/MPO objectives.

    Simulates the target circuit as an MPS or MPO, then delegates to
    :func:`compile_tn_1d` for the actual optimization.

    Args:
        qc_bound (*QuantumCircuit*): Target bound circuit to approximate.
        num_qubits (*int*): Number of qubits.
        approx_layers (*int*): Number of layers in the approximating HEA circuit.
        optimizer_steps (*int*): Number of Adam optimization steps (main stage).
        optimizer_lr (*float*): Learning rate for the Adam optimizer.
        objective_mode (*Literal['mps', 'mpo']*): ``'mps'`` for state infidelity, ``'mpo'`` for process infidelity. Defaults to ``'mps'``.
        bond_cap (*int*): Maximum bond dimension for MPS/MPO truncation.
        warm_start_params (*Optional[np.ndarray]*): Optional initial parameters; used as the seed when provided and length-matching, otherwise a fixed-seed random init is used.
        device (*torch.device | str | None*): Torch device. Defaults to ``None`` (CPU).
        verbose (*bool*): Forward per-step optimization progress logging to
            :func:`compile_tn_1d`. Defaults to ``False`` (silent).

    Returns:
        Tuple of ``(QuantumCircuit, np.ndarray, dict)`` — the compressed circuit,
        optimized parameters, and a metadata dictionary.

    Raises:
        ValueError: objective_mode must be 'mps' or 'mpo'
    """
    mode = str(objective_mode).lower()
    if mode not in {"mps", "mpo"}:
        raise ValueError("objective_mode must be 'mps' or 'mpo'")

    device_obj = torch.device(device) if device is not None else torch.device("cpu")

    if mode == "mps":
        target_tn = simulate_mps(qc_bound, max_bond_dim=int(bond_cap), device=device_obj)
    else:
        target_tn = simulate_mpo_process(qc_bound, max_bond_dim=int(bond_cap), device=device_obj)

    return compile_tn_1d(
        target_tn,
        num_qubits=num_qubits,
        approx_layers=approx_layers,
        optimizer_steps=optimizer_steps,
        optimizer_lr=optimizer_lr,
        objective_mode=mode,
        bond_cap=bond_cap,
        warm_start_params=warm_start_params,
        device=device,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# High-level compression transform builder
# ---------------------------------------------------------------------------

def _planner_stage_ids(plan: Optional[HybridCompressionPlan]) -> List[int]:
    """Return stage IDs for hybrid block planner (prefix = -1, then block indices).

    Args:
        plan (*Optional[HybridCompressionPlan]*): Compression plan produced by ``plan_hybrid_suffix_blocks``.

    Returns:
        List of integer stage IDs (``-1`` for prefix, then ``0, 1, …`` for suffix blocks).
    """
    if plan is None or plan.total_layers <= 0 or not plan.blocks:
        return [-1]
    return [-1] + [int(i) for i in range(len(plan.blocks))]


def _stage_target_circuit(
    qc_bound: QuantumCircuit,
    plan: Optional[HybridCompressionPlan],
    stage_id: int,
    num_qubits: int,
) -> QuantumCircuit:
    """Extract the sub-circuit for a given planner stage.

    Args:
        qc_bound (*QuantumCircuit*): Full bound circuit to extract from.
        plan (*Optional[HybridCompressionPlan]*): Compression plan (or ``None`` to return the full circuit).
        stage_id (*int*): Stage index (``-1`` for prefix, ``≥0`` for suffix block).
        num_qubits (*int*): Number of qubits.

    Returns:
        Constructed ``QuantumCircuit``.
    """
    if plan is None or plan.total_layers <= 0 or not plan.blocks:
        return qc_bound.deepcopy()

    if int(stage_id) == -1:
        if int(plan.split_layer) <= 0:
            return QuantumCircuit(int(num_qubits))
        return build_layer_span_circuit(
            qc_bound, start_layer=0, end_layer=int(plan.split_layer) - 1,
        )

    bid = int(stage_id)
    if bid < 0 or bid >= len(plan.blocks):
        return qc_bound.deepcopy()
    block = plan.blocks[bid]
    return build_layer_span_circuit(
        qc_bound, start_layer=int(block.start_layer), end_layer=int(block.end_layer),
    )


def _compose_stage_circuits(
    stage_circuits: Sequence[QuantumCircuit],
    num_qubits: int,
) -> QuantumCircuit:
    """Compose multiple stage sub-circuits into one circuit.

    All stages must share the same ``nqubits`` *and* the same ``qubits`` list,
    so the resulting circuit preserves the transpiler/layout ordering carried
    by each stage.

    Args:
        stage_circuits (*Sequence[QuantumCircuit]*): Ordered sub-circuits to concatenate.
        num_qubits (*int*): Number of qubits.

    Returns:
        Constructed ``QuantumCircuit`` whose ``qubits`` list is taken from the
        first stage and gates are concatenated in stage order.

    Raises:
        ValueError: If stages disagree on ``nqubits`` or on the ``qubits`` layout.
    """
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
        if list(stage_qc.qubits) != list(out.qubits):
            raise ValueError(
                "all stage circuits must share the same qubits layout to preserve "
                "transpiler/physical-qubit mapping"
            )
        merged.extend(list(stage_qc.gates))
    out.gates = merged
    return out


def build_compression_transform(
    client,
    *,
    num_qubits: int,
    layers: int,
    backend,
    target_qubits: Optional[Sequence[int]] = None,
    use_dd: bool = True,
    enable_block_planner: bool = False,
    planner_bond_cap: int = 128,
    planner_trunc_tol: float = 1e-8,
    planner_max_layers_per_block: int = 6,
    compression_block_layers: Optional[int] = None,
    compression_optimizer_steps: int = 20,
    compression_optimizer_lr: float = 0.05,
    compression_verbose: bool = False,
    compression_plot_loss: bool = False,
    tag: str = "compress",
    convert_single_qubit_gate_to_u: bool = True,
    transpile: bool = True,
) -> dict:
    """Build a circuit compression transform callable and its associated templates.

    Creates a stateful compression callback compatible with the
    ``circuit_transform`` parameter of
    :func:`~fieldqkit.algorithms.optimizer_utils.run_variational_loop`.
    Also prepares a compressed hardware-efficient template that is
    transpiled once and reused across iterations.

    Args:
        client: ``QuantumHardwareClient`` instance.
        num_qubits: Number of logical qubits.
        layers: Original ansatz depth (used for default compressed depth).
        backend: Target ``Backend``.
        target_qubits: Physical qubit mapping.
        use_dd: Enable dynamical decoupling during transpilation.
        enable_block_planner: Enable hybrid block planner.
        planner_bond_cap: MPS bond dimension cap.
        planner_trunc_tol: MPS truncation tolerance.
        planner_max_layers_per_block: Max layers per block.
        compression_block_layers: Compressed circuit depth ``k``
            (required; validated by caller).
        compression_optimizer_steps: Optimiser steps per compression.
        compression_optimizer_lr: Optimiser learning rate.
        compression_verbose: Print compression diagnostics.
        compression_plot_loss: Plot compression loss curves.
        tag: Log prefix for verbose output.
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates to U during transpilation.
        transpile: Whether to transpile the compressed template on the client
            side.  When ``False`` the template is used as-is and no layout
            mapping is performed.  Defaults to ``True``.

    Returns:
        Dict with keys:

        - ``transform``: callable ``(qc, param_index) -> qc``.
          **Note:** this callable is stateful — it maintains mutable closure
          variables (warm-start params, base params, last block plan) across
          successive calls to enable warm-starting between iterations.
        - ``compressed_transpiled_template``: transpiled compressed template
        - ``target_qubits_in_use``: resolved physical qubit mapping
    """
    from .optimizer_utils import instantiate_transpiled_template

    block_depth_k = int(compression_block_layers) if compression_block_layers is not None else None
    compressed_layers = block_depth_k if block_depth_k is not None else max(1, int(np.ceil(float(layers) * 0.5)))
    compressed_param_count = 2 * int(num_qubits) * (int(compressed_layers) + 1)
    compressed_param_names = [f"cmp_phi_{i}" for i in range(compressed_param_count)]
    compressed_symbolic_qc = _build_hardware_efficient_ansatz_symbolic(
        num_qubits, compressed_param_names, layers=int(compressed_layers),
    )
    if transpile:
        compressed_transpiled_template = client._transpile_with_backend(
            compressed_symbolic_qc, backend,
            target_qubits=target_qubits, use_dd=use_dd, use_gate_compressor=False,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        target_qubits_in_use = client._ordered_target_qubits_from_layout(
            compiled_qc=compressed_transpiled_template,
            original_qc=compressed_symbolic_qc,
            num_qubits=num_qubits,
        )
    else:
        compressed_transpiled_template = compressed_symbolic_qc
        target_qubits_in_use = list(target_qubits) if target_qubits is not None else list(range(num_qubits))

    unified_bond_cap = int(planner_bond_cap)
    unified_trunc_tol = float(planner_trunc_tol)
    approx_depth = compressed_layers

    # Mutable compression state (captured by closure)
    warm_start: List[Optional[np.ndarray]] = [None]
    base_params: List[Optional[np.ndarray]] = [None]
    last_plan: List = [None]

    def _transform(qc_bound: QuantumCircuit, changed_param_index: Optional[int] = None) -> QuantumCircuit:
        """Compression transform function to be called at each iteration.

        Args:
            qc_bound (*QuantumCircuit*): Bound circuit with concrete parameter values.
            changed_param_index (*Optional[int]*): Index of the single parameter that changed since the last call, enabling incremental re-planning. ``None`` forces a full recomputation. Defaults to ``None``.

        Returns:
            Constructed ``QuantumCircuit``.
        """
        plan = None
        if enable_block_planner:
            if changed_param_index is None or last_plan[0] is None:
                plan = plan_hybrid_suffix_blocks(
                    qc_bound,
                    bond_cap=unified_bond_cap,
                    trunc_tol=unified_trunc_tol,
                    max_layers_per_block=int(planner_max_layers_per_block),
                )
                last_plan[0] = plan
            else:
                plan = last_plan[0]

        stage_ids = _planner_stage_ids(plan)
        local_steps = int(compression_optimizer_steps)
        if changed_param_index is not None:
            local_steps = max(2, int(np.ceil(float(compression_optimizer_steps) * 0.7)))

        local_warm_start = base_params[0] if changed_param_index is not None else warm_start[0]
        compressed_stage_exec_circuits: List[QuantumCircuit] = []
        for stage_id in stage_ids:
            target_stage_qc = _stage_target_circuit(qc_bound, plan, int(stage_id), num_qubits)
            compressed_stage_qc, next_warm_start, summary = compress_circuit_with_hybrid_objective(
                target_stage_qc,
                num_qubits=num_qubits,
                approx_layers=approx_depth,
                optimizer_steps=int(local_steps),
                optimizer_lr=float(compression_optimizer_lr),
                objective_mode="mpo" if (plan is not None and int(stage_id) >= 0) else "mps",
                bond_cap=unified_bond_cap,
                warm_start_params=local_warm_start,
                verbose=compression_verbose,
            )
            local_warm_start = next_warm_start

            stage_exec_qc = instantiate_transpiled_template(
                compressed_transpiled_template, compressed_param_names, next_warm_start,
            )
            compressed_stage_exec_circuits.append(stage_exec_qc)

            if compression_verbose:
                call_tag = "base" if changed_param_index is None else f"shift(param={int(changed_param_index)})"
                logger.info(
                    "[%s] circuit compressed: call=%s stage=%d layers=%d mode=%s "
                    "init=%.3e loss=%.3e delta=%.3e inf=%.3e",
                    tag, call_tag, int(stage_id), approx_depth,
                    summary['objective_mode'], summary['init_loss'],
                    summary['best_loss'], summary['loss_delta'],
                    summary['objective_infidelity'],
                )

            if compression_plot_loss:
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
                    if compression_verbose:
                        logger.info("[%s] compression loss plotting skipped: %s", tag, _plot_exc)

        warm_start[0] = local_warm_start
        if changed_param_index is None and warm_start[0] is not None:
            base_params[0] = warm_start[0].copy()

        if plan is not None and compression_verbose:
            logger.info(
                "[%s] hybrid block plan: split=%d/%d seed_bond=%s seed_err=%.3e blocks=%d",
                tag, plan.split_layer, plan.total_layers,
                plan.prefix_max_bond, plan.prefix_relative_trunc_error,
                len(plan.blocks),
            )
        return _compose_stage_circuits(compressed_stage_exec_circuits, num_qubits)

    return {
        "transform": _transform,
        "compressed_transpiled_template": compressed_transpiled_template,
        "target_qubits_in_use": target_qubits_in_use,
    }
