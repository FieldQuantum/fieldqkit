"""High-level QML runner with automatic hardware selection.

Mirrors the ``VQERunner`` / ``QAOARunner`` two-tier design: the runner
resolves provider → backend → chip, then delegates to the low-level
``run_pqc_classifier``, ``run_qnn_unsupervised``, or ``run_qnn_conditional``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ..api.backend import resolve_provider
from ..api.quantum_platform import create_provider_runtime
from ..core.types import QBMResult, QMLResult
from .qml import (
    run_pqc_classifier,
    run_qnn_conditional,
    run_qnn_unsupervised,
)

logger = logging.getLogger(__name__)


@dataclass
class QMLRunner:
    """High-level QML runner with automatic hardware selection.

    Wraps ``run_pqc_classifier``, ``run_qnn_unsupervised``, and
    ``run_qnn_conditional`` with automatic provider resolution and
    chip fallback, following the same pattern as ``VQERunner`` and
    ``QAOARunner``.
    """

    client: object
    layers: int = 2
    shots: int = 4096
    max_iters: int = 100
    learning_rate: float = 0.01
    seed: Optional[int] = None
    gradient_method: Literal["parameter-shift", "autograd"] = "parameter-shift"
    shift: float = np.pi / 2.0
    zne: bool = False
    readout_mitigation: bool = False
    convert_single_qubit_gate_to_u: bool = True
    gen_shots: int = 1024
    mmd_sigma: float = 1.0

    # ------------------------------------------------------------------ #
    #  Internal: resolve provider and iterate over candidate chips
    # ------------------------------------------------------------------ #

    def _resolve_and_run(
        self,
        func: Callable,
        name: str,
        num_qubits: int,
        *,
        provider: str,
        prefer_chips: Optional[Sequence[str] | str],
        target_qubits: Optional[Sequence[int]],
        extra_kwargs: dict,
    ):
        """Resolve provider → backend, try each chip, call *func*.

        Args:
            func: Low-level training function to call.
            name: Task name prefix.
            num_qubits: Number of logical qubits.
            provider: Hardware provider name.
            prefer_chips: Candidate chip filter.
            target_qubits: Physical qubit mapping.
            extra_kwargs: Algorithm-specific keyword arguments.

        Returns:
            Result from *func*.

        Raises:
            RuntimeError: If all candidate chips fail.
        """
        provider_name = resolve_provider(provider, prefer_chips)
        qasm_version = self.client._default_qasm_version_for_provider(provider_name)
        convert_u = provider_name not in {"tencent"} and self.convert_single_qubit_gate_to_u
        runtime = create_provider_runtime(provider=provider_name, client=self.client)
        profiles = runtime.backend_adapter.discover_hardware(
            num_qubits=num_qubits,
            prefer_hardware=prefer_chips,
        )
        logger.info("candidate chips: %s", [p.hardware_name for p in profiles])
        if not profiles:
            raise RuntimeError(
                f"no available {provider_name} hardware for num_qubits={num_qubits}"
            )

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
                logger.info("running on chip: %s", resolved.hardware_name)
                return func(
                    num_qubits,
                    client=self.client,
                    backend=resolved.backend,
                    chip_name=resolved.hardware_name,
                    shots=self.shots,
                    layers=self.layers,
                    max_iters=self.max_iters,
                    learning_rate=self.learning_rate,
                    seed=self.seed,
                    gradient_method=self.gradient_method,
                    shift=self.shift,
                    zne=self.zne,
                    readout_mitigation=self.readout_mitigation,
                    target_qubits=target_qubits,
                    qasm_version=qasm_version,
                    convert_single_qubit_gate_to_u=convert_u,
                    **extra_kwargs,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("chip %s failed: %s", resolved.hardware_name, exc)
                continue

        raise RuntimeError("all candidate chips failed") from last_error

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run_classifier(
        self,
        name: str,
        num_qubits: int,
        train_data: Sequence[Tuple[Sequence[float], int]],
        *,
        test_data: Optional[Sequence[Tuple[Sequence[float], int]]] = None,
        encoding: Union[str, Callable] = "angle",
        encoding_kwargs: Optional[dict] = None,
        num_classes: int = 2,
        measurement_qubits: Optional[Sequence[int]] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        provider: str = "quafu",
        prefer_chips: Optional[Sequence[str] | str] = None,
        target_qubits: Optional[Sequence[int]] = None,
    ) -> QMLResult:
        """Train a parameterized quantum classifier.

        Args:
            name: Task name prefix.
            num_qubits: Number of qubits.
            train_data: List of ``(features, label)`` pairs.
            test_data: Optional validation data.
            encoding: Encoding strategy.
            encoding_kwargs: Extra encoding kwargs.
            num_classes: Number of classes.
            measurement_qubits: Qubits to measure for readout.
            callback: Per-iteration callback.
            provider: Hardware provider.
            prefer_chips: Candidate chip filter.
            target_qubits: Physical qubit mapping.

        Returns:
            ``QMLResult`` with loss history and accuracy.
        """
        extra = dict(
            train_data=train_data,
            test_data=test_data,
            encoding=encoding,
            encoding_kwargs=encoding_kwargs,
            num_classes=num_classes,
            measurement_qubits=measurement_qubits,
            callback=callback,
        )
        return self._resolve_and_run(
            run_pqc_classifier, name, num_qubits,
            provider=provider, prefer_chips=prefer_chips,
            target_qubits=target_qubits, extra_kwargs=extra,
        )

    def run_unsupervised(
        self,
        name: str,
        num_qubits: int,
        train_samples: np.ndarray,
        *,
        test_samples: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        provider: str = "quafu",
        prefer_chips: Optional[Sequence[str] | str] = None,
        target_qubits: Optional[Sequence[int]] = None,
    ) -> QBMResult:
        """Train a QNN to reproduce an unlabelled distribution.

        Args:
            name: Task name prefix.
            num_qubits: Number of qubits.
            train_samples: ``(N, num_qubits)`` binary array.
            test_samples: Optional validation samples.
            callback: Per-iteration callback.
            provider: Hardware provider.
            prefer_chips: Candidate chip filter.
            target_qubits: Physical qubit mapping.

        Returns:
            ``QBMResult`` with loss history and generated samples.
        """
        extra = dict(
            train_samples=train_samples,
            test_samples=test_samples,
            callback=callback,
            mmd_sigma=self.mmd_sigma,
            gen_shots=self.gen_shots,
        )
        return self._resolve_and_run(
            run_qnn_unsupervised, name, num_qubits,
            provider=provider, prefer_chips=prefer_chips,
            target_qubits=target_qubits, extra_kwargs=extra,
        )

    def run_conditional(
        self,
        name: str,
        num_qubits: int,
        train_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
        *,
        test_pairs: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        provider: str = "quafu",
        prefer_chips: Optional[Sequence[str] | str] = None,
        target_qubits: Optional[Sequence[int]] = None,
    ) -> QBMResult:
        """Train a QNN to learn conditional distribution P(y|x).

        The input bit-string *x* is prepared directly as a computational basis
        state ``|x⟩`` (X gates on qubits where ``xᵢ = 1``), then the
        parameterized ansatz is applied.

        Args:
            name: Task name prefix.
            num_qubits: Number of qubits.
            train_pairs: List of ``(input_bits, output_bits)`` pairs.
            test_pairs: Optional validation pairs.
            callback: Per-iteration callback.
            provider: Hardware provider.
            prefer_chips: Candidate chip filter.
            target_qubits: Physical qubit mapping.

        Returns:
            ``QBMResult`` with loss history and generated samples.
        """
        extra = dict(
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            callback=callback,
            mmd_sigma=self.mmd_sigma,
            gen_shots=self.gen_shots,
        )
        return self._resolve_and_run(
            run_qnn_conditional, name, num_qubits,
            provider=provider, prefer_chips=prefer_chips,
            target_qubits=target_qubits, extra_kwargs=extra,
        )
