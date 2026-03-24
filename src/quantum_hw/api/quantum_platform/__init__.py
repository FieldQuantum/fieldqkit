"""Provider-specific hardware integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cqlib import QuantumLanguage, RemotePlatformClient
from ..backend import list_available_hardware
from .quafu import QuafuBackendAdapter, QuafuPlatform, QuafuTaskAdapter
from .tianyan import TianYanBackendAdapter, TianYanPlatform, TianYanTaskAdapter
from .guodun import GuoDunBackendAdapter, GuoDunPlatform, GuoDunTaskAdapter


@dataclass
class ProviderRuntime:
    provider: str
    backend_adapter: Any
    task_adapter: Any


def create_provider_runtime(*, provider: str, client: Any) -> ProviderRuntime:
    provider_name = str(provider).lower()
    if provider_name == "quafu":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=QuafuBackendAdapter(),
            task_adapter=QuafuTaskAdapter(client=client),
        )
    if provider_name == "tianyan":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=TianYanBackendAdapter(),
            task_adapter=TianYanTaskAdapter(client=client),
        )
    if provider_name == "guodun":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=GuoDunBackendAdapter(),
            task_adapter=GuoDunTaskAdapter(client=client),
        )
    raise ValueError("provider must be one of: 'quafu', 'tianyan', or 'guodun'")


__all__ = [
    "QuantumLanguage",
    "RemotePlatformClient",
    "ProviderRuntime",
    "create_provider_runtime",
    "list_available_hardware",
    "QuafuPlatform",
    "QuafuBackendAdapter",
    "QuafuTaskAdapter",
    "TianYanPlatform",
    "TianYanBackendAdapter",
    "TianYanTaskAdapter",
    "GuoDunPlatform",
    "GuoDunBackendAdapter",
    "GuoDunTaskAdapter",
]
