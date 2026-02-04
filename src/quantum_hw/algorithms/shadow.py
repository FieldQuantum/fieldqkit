from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..observables import append_measurement_basis, pauli_basis_pattern
from ..types import ShadowResult
from ..utils import get_samples
from ..zne import apply_zne_cz_tripling, zne_linear_extrapolate


_BASIS_CHOICES = ("X", "Y", "Z")


def _basis_to_code(basis: Sequence[str]) -> np.ndarray:
    mapping = {"X": 0, "Y": 1, "Z": 2}
    return np.array([mapping[b] for b in basis], dtype=int)


def _observable_to_codes(observable: str, num_qubits: int) -> np.ndarray:
    mapping = {"X": 0, "Y": 1, "Z": 2, "I": -1}
    pattern = pauli_basis_pattern(observable, num_qubits=num_qubits)
    return np.array([mapping[p] for p in pattern], dtype=int)


def _median_of_means(values: np.ndarray, groups: int) -> Tuple[float, float]:
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
    batch_size: int,
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
    if shots <= 0:
        raise ValueError("shots must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if estimator not in {"mean", "mom"}:
        raise ValueError("estimator must be 'mean' or 'mom'")

    rng = np.random.default_rng(seed)

    pending: List[Tuple[List[str], str, object]] = []
    task_ids: List[str] = []
    remaining = shots
    batch_index = 0

    while remaining > 0:
        shots_batch = min(batch_size, remaining)
        basis_pattern = rng.choice(_BASIS_CHOICES, size=num_qubits).tolist()
        qc_batch = deepcopy(qc)
        append_measurement_basis(qc_batch, basis_pattern)
        qct = client._transpile_with_backend(qc_batch, backend, target_qubits=target_qubits)
        qasm_1 = qct.to_openqasm2
        task_id_1 = client._submit_openqasm_async(
            name=f"{name}_shadow{batch_index}",
            qasm=qasm_1,
            shots=shots_batch,
            chip_name=chip_name,
        )
        pending.append((basis_pattern, "1", task_id_1))
        task_ids.append(str(task_id_1))

        if zne:
            qct_3 = apply_zne_cz_tripling(qct)
            qasm_3 = qct_3.to_openqasm2
            task_id_3 = client._submit_openqasm_async(
                name=f"{name}_shadow{batch_index}_zne3",
                qasm=qasm_3,
                shots=shots_batch,
                chip_name=chip_name,
            )
            pending.append((basis_pattern, "3", task_id_3))
            task_ids.append(str(task_id_3))
        remaining -= shots_batch
        batch_index += 1

    all_samples_1: List[List[int]] = []
    all_basis_1: List[List[str]] = []
    all_samples_3: List[List[int]] = []
    all_basis_3: List[List[str]] = []

    for basis_pattern, scale, task_id in pending:
        status = client._wait_task(task_id)
        if status != "Finished":
            raise RuntimeError(f"shadow task {task_id} ended with status {status}")
        counts = client.tmgr.result(task_id)["count"]
        samples = get_samples(counts, num_qubits)
        if samples.size == 0:
            continue
        sample_list = samples.tolist()
        if scale == "1":
            all_samples_1.extend(sample_list)
            all_basis_1.extend([basis_pattern] * len(sample_list))
        else:
            all_samples_3.extend(sample_list)
            all_basis_3.extend([basis_pattern] * len(sample_list))

    samples_arr_1 = np.asarray(all_samples_1, dtype=int)
    estimates_1, stderrs_1 = estimate_observables(
        samples_arr_1,
        all_basis_1,
        observables,
        num_qubits=num_qubits,
        estimator=estimator,
        mom_groups=mom_groups,
    )
    print("Shadow estimates (1x):", estimates_1)

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
        print("Shadow estimates (3x):", estimates_3)
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


@dataclass
class ShadowTomography:
    """High-level helper for classical shadow tomography."""

    client: object
    seed: Optional[int] = None
    batch_size: int = 1

    def run(
        self,
        circuit: str,
        name: str,
        num_qubits: int,
        *,
        shots: int = 8192,
        observables: Optional[Sequence[str]] = None,
        zne: bool = False,
        estimator: str = "mean",
        mom_groups: Optional[int] = None,
        target_qubits: Optional[Sequence[int]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> ShadowResult:
        return self.client.run_shadow(
            circuit,
            name,
            num_qubits,
            shots=shots,
            observables=observables,
            zne=zne,
            estimator=estimator,
            mom_groups=mom_groups,
            target_qubits=target_qubits,
            prefer_chips=prefer_chips,
            rank_weights=rank_weights,
            seed=self.seed,
            batch_size=self.batch_size,
        )
