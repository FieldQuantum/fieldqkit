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
        raise NotImplementedError(f"{self.provider} submit_openqasm is not implemented")

    def query_status(self, handle: ProviderTaskHandle) -> str:
        raise NotImplementedError(f"{self.provider} query_status is not implemented")

    def fetch_result(self, handle: ProviderTaskHandle) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.provider} fetch_result is not implemented")

    def cancel_task(self, handle: ProviderTaskHandle) -> None:
        raise NotImplementedError(f"{self.provider} cancel_task is not implemented")