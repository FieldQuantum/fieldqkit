"""Classical shadow tomography utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..api.backend import Backend
from ..api.hardware import rank_chips
from ..circuit import QuantumCircuit
from ..core.observables import pauli_basis_pattern
from ..core.types import ShadowResult
from ..core.zne import zne_linear_extrapolate


_BASIS_CHOICES = ("X", "Y", "Z")


def _basis_to_code(basis: Sequence[str]) -> np.ndarray:
    mapping = {"X": 0, "Y": 1, "Z": 2}
    return np.array([mapping[b] for b in basis], dtype=int)


def _observable_to_codes(observable: str, num_qubits: int) -> np.ndarray:
    mapping = {"X": 0, "Y": 1, "Z": 2, "I": -1}
    pattern = pauli_basis_pattern(observable, num_qubits=num_qubits)
    return np.array([mapping[p] for p in pattern], dtype=int)


def _median_of_means(values: np.ndarray, groups: int) -> Tuple[float, float]:
    """Median-of-means estimator for heavy-tailed noise robustness."""
    if groups <= 1:
        mean = float(values.mean())
        stderr = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        return mean, stderr

    values = np.random.permutation(values)
    groups = min(groups, values.size)
    splits = np.array_split(values, groups)
    means = np.array([s.mean() if s.size > 0 else 0.0 for s in splits], dtype=float)
    median = float(np.median(means))
    if means.size > 1:
        stderr = float(means.std(ddof=1) / np.sqrt(means.size))
    else:
        stderr = 0.0
    return median, stderr


def estimate_observables(
    samples: np.ndarray,
    basis_patterns: Sequence[Sequence[str]],
    observables: Sequence[str],
    *,
    num_qubits: int,
    estimator: str = "mean",
    mom_groups: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Estimate Pauli observables from classical shadow data."""
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if len(basis_patterns) != samples.shape[0]:
        raise ValueError("basis_patterns length must match number of samples")

    nshots = samples.shape[0]
    if nshots == 0:
        return {}, {}

    basis_codes = np.vstack([_basis_to_code(p) for p in basis_patterns])
    estimates: Dict[str, float] = {}
    stderrs: Dict[str, float] = {}

    for obs in observables:
        op_codes = _observable_to_codes(obs, num_qubits=num_qubits)
        mask = op_codes != -1
        if not np.any(mask):
            sample_values = np.ones(nshots, dtype=float)
        else:
            # Only samples measured in the matching basis contribute.
            basis_match = np.all(basis_codes[:, mask] == op_codes[mask], axis=1)
            eigen = 1.0 - 2.0 * samples[:, mask]
            values = np.prod(3.0 * eigen, axis=1)
            sample_values = np.where(basis_match, values, 0.0)

        if estimator == "mom":
            groups = mom_groups if mom_groups is not None else max(1, int(np.sqrt(nshots)))
            mean, stderr = _median_of_means(sample_values, groups)
        else:
            mean = float(sample_values.mean())
            if nshots > 1:
                stderr = float(sample_values.std(ddof=1) / np.sqrt(nshots))
            else:
                stderr = 0.0

        estimates[obs] = mean
        stderrs[obs] = stderr

    return estimates, stderrs


def run_shadow_with_backend(
    client,
    qc,
    *,
    name: str,
    num_qubits: int,
    backend,
    chip_name: str,
    shots: int,
    shots_per_basis: int = 1,
    observables: Optional[Sequence[str]] = None,
    estimator: str = "mean",
    mom_groups: Optional[int] = None,
    target_qubits: Optional[Sequence[int]] = None,
    zne: bool = False,
    seed: Optional[int] = None,
) -> ShadowResult:
    """Run classical shadow tomography on a specific backend."""
    if observables is None:
        observables = []
    observables = list(observables)
    if estimator not in {"mean", "mom"}:
        raise ValueError("estimator must be 'mean' or 'mom'")

    rng = np.random.default_rng(seed)

    basis_patterns: List[List[str]] = []
    basis_number = int(np.ceil(shots / float(shots_per_basis)))
    for _ in range(basis_number):
        # Randomly draw a measurement basis per batch.
        basis_pattern = rng.choice(_BASIS_CHOICES, size=num_qubits).tolist()
        basis_patterns.append(basis_pattern)
    batch_name = f"{name}_shadow"

    def _basis_pattern_to_pauli(pattern: Sequence[str]) -> str:
        return " ".join(f"{op}{i}" for i, op in enumerate(pattern))

    all_samples_1: List[List[int]] = []
    all_basis_1: List[List[str]] = []
    all_samples_3: List[List[int]] = []
    all_basis_3: List[List[str]] = []
    task_ids: List[str] = []

    basis_observables = [_basis_pattern_to_pauli(p) for p in basis_patterns]
    res = client._run_with_backend(
        qc,
        batch_name,
        num_qubits,
        backend=backend,
        chip_name=chip_name,
        shots=shots_per_basis,
        observables=basis_observables,
        return_probabilities=False,
        target_qubits=target_qubits,
        merge_groups=False,
        zne=zne,
        print_true=False,
    )
    if res.task_ids:
        task_ids.extend([str(t) for t in res.task_ids])

    samples_blocks = res.samples or []
    samples_zne_blocks = res.samples_zne or []
    if len(samples_blocks) != len(basis_patterns):
        raise RuntimeError("shadow samples length mismatch with basis_patterns")
    if zne and len(samples_zne_blocks) != len(basis_patterns):
        raise RuntimeError("shadow ZNE samples length mismatch with basis_patterns")

    for basis_pattern, block in zip(basis_patterns, samples_blocks):
        if block:
            all_samples_1.extend(block)
            all_basis_1.extend([basis_pattern] * len(block))
    if zne:
        for basis_pattern, block in zip(basis_patterns, samples_zne_blocks):
            if block:
                all_samples_3.extend(block)
                all_basis_3.extend([basis_pattern] * len(block))

    samples_arr_1 = np.asarray(all_samples_1, dtype=int)
    estimates_1, stderrs_1 = estimate_observables(
        samples_arr_1,
        all_basis_1,
        observables,
        num_qubits=num_qubits,
        estimator=estimator,
        mom_groups=mom_groups,
    )

    estimates = estimates_1
    stderrs = stderrs_1
    estimates_raw = None

    if zne:
        samples_arr_3 = np.asarray(all_samples_3, dtype=int)
        estimates_3, _ = estimate_observables(
            samples_arr_3,
            all_basis_3,
            observables,
            num_qubits=num_qubits,
            estimator=estimator,
            mom_groups=mom_groups,
        )
        estimates = {
            obs: float(zne_linear_extrapolate(estimates_1[obs], estimates_3[obs]))
            for obs in observables
        }
        estimates_raw = estimates_1

    return ShadowResult(
        task_ids=task_ids,
        samples=all_samples_1,
        basis_patterns=all_basis_1,
        observables=observables,
        observable_estimates=estimates,
        observable_estimates_raw=estimates_raw,
        observable_stderr=stderrs,
        observable_stderr_raw=stderrs_1 if zne else None,
        num_samples=len(all_samples_1),
    )


def run_shadow(
    client,
    circuit: str,
    name: str,
    num_qubits: int,
    *,
    shots: int = 8192,
    shots_per_basis: int = 1,
    observables: Optional[Sequence[str]] = None,
    zne: bool = False,
    estimator: str = "mean",
    mom_groups: Optional[int] = None,
    target_qubits: Optional[Sequence[int]] = None,
    prefer_chips: Optional[Sequence[str] | str] = None,
    rank_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> ShadowResult:
    """Select hardware and run classical shadow tomography."""
    print("[shadow] read hardware information and select")
    if client._is_openqasm2(circuit):
        qc = QuantumCircuit().from_openqasm2(openqasm2_str=circuit)
    elif client._is_openqasm3(circuit):
        qc = QuantumCircuit().from_openqasm3(openqasm3_str=circuit)
    else:
        qc = client.build_circuit(circuit, num_qubits=num_qubits)

    ranked_chips = rank_chips(
        client.tmgr,
        num_qubits=num_qubits,
        prefer_chips=prefer_chips,
        weights=rank_weights,
    )
    if not ranked_chips:
        raise RuntimeError("no available chips satisfy num_qubits requirement")

    if isinstance(observables, str):
        observables = [observables]

    last_error: Optional[Exception] = None
    for chip_name in ranked_chips:
        backend = Backend(chip_name)
        client.chip_name = chip_name
        client.chip_backend = backend
        try:
            return run_shadow_with_backend(
                client,
                qc,
                name=name,
                num_qubits=num_qubits,
                backend=backend,
                chip_name=chip_name,
                shots=shots,
                shots_per_basis=shots_per_basis,
                observables=observables,
                target_qubits=target_qubits,
                zne=zne,
                estimator=estimator,
                mom_groups=mom_groups,
                seed=seed,
            )
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError("all candidate chips failed to run shadow tomography") from last_error


@dataclass
class ShadowTomography:
    """High-level helper for classical shadow tomography."""

    client: object
    seed: Optional[int] = None

    def run(
        self,
        circuit: str,
        name: str,
        num_qubits: int,
        *,
        shots: int = 8192,
        shots_per_basis: int = 1,
        observables: Optional[Sequence[str]] = None,
        zne: bool = False,
        estimator: str = "mean",
        mom_groups: Optional[int] = None,
        target_qubits: Optional[Sequence[int]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> ShadowResult:
        return run_shadow(
            self.client,
            circuit,
            name,
            num_qubits,
            shots=shots,
            shots_per_basis=shots_per_basis,
            observables=observables,
            zne=zne,
            estimator=estimator,
            mom_groups=mom_groups,
            target_qubits=target_qubits,
            prefer_chips=prefer_chips,
            rank_weights=rank_weights,
            seed=self.seed,
        )
