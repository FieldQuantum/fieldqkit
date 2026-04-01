"""Unified task abstractions and provider-neutral request helpers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict

from .backend import ResolvedBackend


@dataclass
class OpenQasmSubmitRequest:
    name: str
    qasm: str
    shots: int
    chip_name: str
    submit_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderTaskHandle:
    provider: str
    task_id: str
    payload: Dict[str, Any] = field(default_factory=dict)


class TaskAdapter(ABC):
    provider: str

    def submit_openqasm(self, submit_request: OpenQasmSubmitRequest, backend: ResolvedBackend) -> ProviderTaskHandle:
        """Submit an OpenQASM program and return a task handle.

        Args:
            submit_request (*OpenQasmSubmitRequest*): Submission request descriptor.
            backend (*ResolvedBackend*): Hardware backend descriptor.

        Raises:
            NotImplementedError: f'{self.provider} submit_openqasm is not implemented
        """
        raise NotImplementedError(f"{self.provider} submit_openqasm is not implemented")

    def query_status(self, handle: ProviderTaskHandle) -> str:
        """Query the execution status of a submitted task.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Raises:
            NotImplementedError: f'{self.provider} query_status is not implemented
        """
        raise NotImplementedError(f"{self.provider} query_status is not implemented")

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        """Fetch the result of a completed task.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Raises:
            NotImplementedError: f'{self.provider} fetch_result is not implemented
        """
        raise NotImplementedError(f"{self.provider} fetch_result is not implemented")

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        """Cancel a running task.

        Args:
            handle (*ProviderTaskHandle*): Task handle from a prior submission.

        Raises:
            NotImplementedError: f'{self.provider} cancel_task is not implemented
        """
        raise NotImplementedError(f"{self.provider} cancel_task is not implemented")