"""Classical shadow tomography utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


import numpy as np

from ..api.backend import resolve_provider
from ..api.quantum_platform import create_provider_runtime
from ..circuit import QuantumCircuit
from ..core.observables import pauli_basis_pattern
from ..core.types import ShadowResult
from ..core.zne import zne_linear_extrapolate

logger = logging.getLogger(__name__)


_BASIS_CHOICES = ("X", "Y", "Z")


def _basis_to_code(basis: Sequence[str]) -> np.ndarray:
    """Convert a sequence of Pauli basis labels to integer codes.

    Args:
        basis (*Sequence[str]*): Sequence of basis labels (``'X'``, ``'Y'``, or ``'Z'``).

    Returns:
        Integer array where X→0, Y→1, Z→2.
    """
    mapping = {"X": 0, "Y": 1, "Z": 2}
    return np.array([mapping[b] for b in basis], dtype=int)


def _observable_to_codes(observable: str, num_qubits: int) -> np.ndarray:
    """Convert a Pauli observable string to per-qubit integer codes.

    Args:
        observable (*str*): Pauli observable string (e.g. ``'XZI'``).
        num_qubits (*int*): Number of qubits.

    Returns:
        Integer array where X→0, Y→1, Z→2, I→-1.
    """
    mapping = {"X": 0, "Y": 1, "Z": 2, "I": -1}
    pattern = pauli_basis_pattern(observable, num_qubits=num_qubits)
    return np.array([mapping[p] for p in pattern], dtype=int)


def _median_of_means(
    values: np.ndarray, groups: int, rng: Optional[np.random.Generator] = None
) -> Tuple[float, float]:
    """Median-of-means estimator for heavy-tailed noise robustness.

    Args:
        values (*np.ndarray*): 1-D array of per-shot estimator values.
        groups (*int*): Number of groups for the median-of-means split.
        rng (*Optional[np.random.Generator]*): Generator used to shuffle the
            values before splitting. If ``None``, a fresh default generator is
            used (non-reproducible). Defaults to ``None``.

    Returns:
        Tuple of ``(median, stderr)`` —the median-of-means estimate
        and its standard error.
    """
    if groups <= 1:
        mean = float(values.mean())
        stderr = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        return mean, stderr

    if rng is None:
        rng = np.random.default_rng()
    values = rng.permutation(values)
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
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Estimate Pauli observables from classical shadow data.

    Args:
        samples (*np.ndarray*): Measurement samples.
        basis_patterns (*Sequence[Sequence[str]]*): Basis patterns (``Sequence[Sequence[str]]``).
        observables (*Sequence[str]*): Observable operators to measure.
        num_qubits (*int*): Number of qubits.
        estimator (*str*): Estimation method: ``'mean'`` for sample mean, ``'mom'`` for median-of-means. Defaults to ``'mean'``.
        mom_groups (*Optional[int]*): Number of groups for median-of-means. If ``None``, uses ``max(1, int(sqrt(nshots)))``. Defaults to ``None``.
        rng (*Optional[np.random.Generator]*): Generator used by the ``'mom'`` estimator to shuffle samples. Pass a seeded generator for reproducibility. Defaults to ``None``.

    Returns:
        Tuple of ``(estimates, stderrs)`` where each is a
        ``Dict[str, float]`` mapping observable strings to their
        estimated expectation values and standard errors respectively.

    Raises:
        ValueError: samples must be a 2D array
    """
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
            mean, stderr = _median_of_means(sample_values, groups, rng=rng)
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
    qc: QuantumCircuit,
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
    qasm_version: str = "2.0",
    use_dd: bool = True,
    submit_options: Optional[Dict] = None,
    convert_single_qubit_gate_to_u: bool = True,
    transpile: bool = True,
) -> ShadowResult:
    """Run classical shadow tomography on a specific backend.

    Args:
        client: ``QuantumHardwareClient`` instance.
        qc (*QuantumCircuit*): Quantum circuit.
        name (*str*): Experiment name for the submission.
        num_qubits (*int*): Number of qubits.
        backend: Hardware backend descriptor.
        chip_name (*str*): Name of the target chip.
        shots (*int*): Number of measurement shots.
        shots_per_basis (*int*): Number of shots per measurement basis. Defaults to ``1``.
        observables (*Optional[Sequence[str]]*): Observable operators to measure. Defaults to ``None``.
        estimator (*str*): Estimation method: ``'mean'`` for sample mean, ``'mom'`` for median-of-means. Defaults to ``'mean'``.
        mom_groups (*Optional[int]*): Number of groups for median-of-means. If ``None``, uses ``max(1, int(sqrt(nshots)))``. Defaults to ``None``.
        target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement. Defaults to ``None``.
        zne (*bool*): Whether to apply zero-noise extrapolation. Defaults to ``False``.
        seed (*Optional[int]*): Random seed for reproducibility. Defaults to ``None``.
        qasm_version (*str*): OpenQASM version. Only ``'2.0'`` is supported. Defaults to ``'2.0'``.
        use_dd (*bool*): Whether to apply dynamical decoupling. Defaults to ``True``.
        submit_options (*Optional[Dict]*): Extra provider-specific submission options. Defaults to ``None``.
        convert_single_qubit_gate_to_u (*bool*): Whether to convert single-qubit gates to U gates. Defaults to ``True``.
        transpile (*bool*): Whether to transpile the circuit for hardware. Defaults to ``True``.

    Returns:
        ``ShadowResult`` containing shadow samples, bases, and observable estimates.

    Raises:
        ValueError: estimator must be 'mean' or 'mom'
        RuntimeError: shadow samples length mismatch with basis_patterns
    """
    if observables is None:
        observables = []
    observables = list(observables)
    if estimator not in {"mean", "mom"}:
        raise ValueError("estimator must be 'mean' or 'mom'")

    rng = np.random.default_rng(seed)

    # Draw random measurement bases for each batch.
    basis_patterns: List[List[str]] = []
    basis_number = int(np.ceil(shots / float(shots_per_basis)))
    for _ in range(basis_number):
        # Randomly draw a measurement basis per batch.
        basis_pattern = rng.choice(_BASIS_CHOICES, size=num_qubits).tolist()
        basis_patterns.append(basis_pattern)
    batch_name = f"{name}_shadow"

    def _basis_pattern_to_pauli(pattern: Sequence[str]) -> str:
        """Convert a per-qubit basis pattern to an indexed Pauli string.

        Args:
            pattern (*Sequence[str]*): Per-qubit basis choices (``'X'``/``'Y'``/``'Z'``).

        Returns:
            Indexed Pauli string (e.g. ``"X0 Y1 Z2"``).
        """
        return " ".join(f"{op}{i}" for i, op in enumerate(pattern))

    all_samples_1: List[List[int]] = []
    all_basis_1: List[List[str]] = []
    all_samples_3: List[List[int]] = []
    all_basis_3: List[List[str]] = []
    task_ids: List[str] = []

    # Treat each basis pattern as a Pauli observable to drive basis rotations.
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
        qasm_version=qasm_version,
        use_dd=use_dd,
        submit_options=submit_options,
        convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        transpile=transpile,
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
        rng=rng,
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
            rng=rng,
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


@dataclass
class ShadowTomography:
    """High-level helper for classical shadow tomography."""

    client: object
    seed: Optional[int] = None

    def run(
        self,
        circuit: Union[str, QuantumCircuit],
        name: str,
        num_qubits: int,
        *,
        provider: str = "quafu",
        shots: int = 8192,
        shots_per_basis: int = 1,
        observables: Optional[Sequence[str]] = None,
        zne: bool = False,
        estimator: str = "mean",
        mom_groups: Optional[int] = None,
        target_qubits: Optional[Sequence[int]] = None,
        prefer_chips: Optional[Sequence[str] | str] = None,
        transpile_on_client: bool = True,
        max_wait_time: int = 3600,
        sleep_time: int = 5,
    ) -> ShadowResult:
        """Select hardware and run classical shadow tomography.

        Args:
            circuit (*Union[str, QuantumCircuit]*): Quantum circuit to execute.
            name (*str*): Experiment name for the submission.
            num_qubits (*int*): Number of qubits.
            provider (*str*): Platform provider name. One of ``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``, ``"origin"``, ``"fieldquantum"``, ``"simulator"``. Defaults to ``'quafu'``.
            shots (*int*): Number of measurement shots. Defaults to ``8192``.
            shots_per_basis (*int*): Number of shots per measurement basis. Defaults to ``1``.
            observables (*Optional[Sequence[str]]*): Observable operators to measure. Defaults to ``None``.
            zne (*bool*): Whether to apply zero-noise extrapolation. Defaults to ``False``.
            estimator (*str*): Estimation method: ``'mean'`` for sample mean, ``'mom'`` for median-of-means. Defaults to ``'mean'``.
            mom_groups (*Optional[int]*): Number of groups for median-of-means. If ``None``, uses ``max(1, int(sqrt(nshots)))``. Defaults to ``None``.
            target_qubits (*Optional[Sequence[int]]*): Qubit indices for partial measurement. Defaults to ``None``.
            prefer_chips (*Optional[Sequence[str] | str]*): Preferred chip names for scheduling. Defaults to ``None``.
            transpile_on_client (*bool*): Whether to transpile on the client side. Defaults to ``True``.
            max_wait_time (*int*): Maximum wait time in seconds. Defaults to ``3600``.
            sleep_time (*int*): Polling interval in seconds. Defaults to ``5``.

        Returns:
            ``ShadowResult`` containing shadow samples, bases, and observable estimates.

        Raises:
            ValueError: If *observables* is empty.
            RuntimeError: all candidate chips failed to run shadow tomography
        """
        provider_name = resolve_provider(provider, prefer_chips)
        use_dd = provider_name not in {"tianyan", "guodun", "tencent", "simulator", "fieldquantum"}
        convert_single_qubit_gate_to_u = provider_name not in {"tencent", "fieldquantum"}
        logger.info("read hardware information and select provider=%s", provider_name)
        if not observables:
            raise ValueError("shadow tomography requires at least one observable")
        # Shadow always uses observables; pass them to strip any user measurements.
        qc = self.client._normalize_input_circuit(circuit, num_qubits, observables=observables)

        runtime = create_provider_runtime(provider=provider_name, client=self.client)
        profiles = runtime.backend_adapter.discover_hardware(
            num_qubits=num_qubits,
            prefer_hardware=prefer_chips,
        )
        if not profiles:
            raise RuntimeError(f"no available {provider_name} hardware for num_qubits={num_qubits}")

        if isinstance(observables, str):
            observables = [observables]

        last_error: Optional[Exception] = None
        for profile in profiles:
            resolved = runtime.backend_adapter.resolve_backend(
                num_qubits=num_qubits,
                prefer_hardware=[profile.hardware_name],
            )
            self.client.chip_name = resolved.hardware_name
            self.client.chip_backend = resolved.backend

            def _as_int(value, default):
                """Convert *value* to ``int``, falling back to *default* on failure.

                Args:
                    value: Value to convert.
                    default: Fallback value.

                Returns:
                    ``int`` converted value.
                """
                try:
                    return int(value)
                except Exception:
                    return int(default)

            submit_options = {
                "max_wait_time": _as_int(max_wait_time, 3600),
                "sleep_time": _as_int(sleep_time, 5),
            }
            self.client._active_task_adapter = runtime.task_adapter
            self.client._active_resolved_backend = resolved

            try:
                return run_shadow_with_backend(
                    self.client,
                    qc,
                    name=name,
                    num_qubits=num_qubits,
                    backend=resolved.backend,
                    chip_name=resolved.hardware_name,
                    shots=shots,
                    shots_per_basis=shots_per_basis,
                    observables=observables,
                    target_qubits=target_qubits,
                    zne=zne,
                    estimator=estimator,
                    mom_groups=mom_groups,
                    seed=self.seed,
                    use_dd=use_dd,
                    submit_options=submit_options,
                    convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
                    transpile=bool(transpile_on_client),
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("all candidate chips failed to run shadow tomography") from last_error