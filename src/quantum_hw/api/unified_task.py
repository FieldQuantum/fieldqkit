"""Unified task abstraction and provider-specific task adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from ..circuit import QuantumCircuit
from ..core.types import RunResult
from .unified_backend import ResolvedBackend


@dataclass
class TaskRequest:
    """Normalized task payload for all providers."""

    qc: QuantumCircuit
    name: str
    num_qubits: int
    shots: int
    zne: bool = False
    readout_mitigation: bool = False
    readout_shots: Optional[int] = None
    observables: Optional[Sequence[str] | str] = None
    return_probabilities: bool = False
    target_qubits: Optional[Sequence[int]] = None
    print_true: bool = True
    provider_options: Dict[str, Any] = field(default_factory=dict)


class TaskAdapter(ABC):
    """Common interface for executing tasks across providers."""

    provider: str

    @abstractmethod
    def run_task(self, request: TaskRequest, backend: ResolvedBackend) -> RunResult:
        """Execute a task request on the resolved backend."""


class QuafuTaskAdapter(TaskAdapter):
    provider = "quafu"

    def __init__(self, *, client: Any) -> None:
        self._client = client

    def run_task(self, request: TaskRequest, backend: ResolvedBackend) -> RunResult:
        target_qubits = request.target_qubits
        if target_qubits is None:
            target_qubits = backend.target_qubits

        return self._client._run_with_backend(
            request.qc,
            request.name,
            request.num_qubits,
            backend=backend.backend,
            chip_name=backend.hardware_name,
            shots=request.shots,
            zne=request.zne,
            readout_mitigation=request.readout_mitigation,
            readout_shots=request.readout_shots,
            observables=request.observables,
            return_probabilities=request.return_probabilities,
            target_qubits=target_qubits,
            print_true=request.print_true,
        )


class CqlibTaskAdapter(TaskAdapter):
    provider = "cqlib"

    def __init__(self, *, login_key: str) -> None:
        self._login_key = login_key

    def run_task(self, request: TaskRequest, backend: ResolvedBackend) -> RunResult:
        if request.zne:
            raise ValueError("cqlib provider does not support zne in minimal adapter")
        if request.readout_mitigation:
            raise ValueError("cqlib provider does not support readout mitigation in minimal adapter")
        if request.target_qubits is not None:
            raise ValueError("cqlib provider does not support explicit target_qubits in minimal adapter")

        from .cqlib_adapter import CqlibAdapter

        platform_name = str(backend.metadata.get("platform_name", "tianyan"))
        machine_name = str(backend.metadata.get("machine_name", backend.hardware_name))
        submit_mode = str(request.provider_options.get("submit_mode", "submit_job"))
        transpile_on_client = bool(request.provider_options.get("transpile_on_client", True))
        max_wait_time = int(request.provider_options.get("max_wait_time", 3600))
        sleep_time = int(request.provider_options.get("sleep_time", 5))

        adapter = CqlibAdapter(
            login_key=self._login_key,
            platform=platform_name,
            machine_name=machine_name,
            submit_mode=submit_mode,
        )
        return adapter.run(
            request.qc,
            name=request.name,
            num_qubits=request.num_qubits,
            shots=request.shots,
            observables=request.observables,
            return_probabilities=request.return_probabilities,
            transpile_on_client=transpile_on_client,
            max_wait_time=max_wait_time,
            sleep_time=sleep_time,
        )
