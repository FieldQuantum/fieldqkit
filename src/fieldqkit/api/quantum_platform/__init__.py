"""Provider-specific hardware integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cqlib import QuantumLanguage, RemotePlatformClient
from ..backend import list_available_hardware, SimulatorBackendAdapter
from .quafu import QuafuBackendAdapter, QuafuPlatform, QuafuTaskAdapter
from .tianyan import TianYanBackendAdapter, TianYanPlatform, TianYanTaskAdapter
from .guodun import GuoDunBackendAdapter, GuoDunPlatform, GuoDunTaskAdapter
from .tencent import TencentBackendAdapter, TencentPlatform, TencentTaskAdapter
from .fieldquantum import (
    FieldQuantumPlatform,
    FieldQuantumBackendAdapter,
    FieldQuantumTaskAdapter,
    FIELDQUANTUM_DEFAULT_URL,
)
from .origin import (
    OriginPlatform,
    OriginBackendAdapter,
    OriginTaskAdapter,
    ORIGIN_DEFAULT_URL,
)


@dataclass
class ProviderRuntime:
    provider: str
    backend_adapter: Any
    task_adapter: Any


def create_provider_runtime(*, provider: str, client: Any) -> ProviderRuntime:
    """Create a ``ProviderRuntime`` for the given provider name.

    Args:
        provider (*str*): Platform provider name. One of ``"quafu"``, ``"tianyan"``, ``"guodun"``, ``"tencent"``, ``"origin"``, ``"fieldquantum"``, ``"simulator"`` (case-insensitive).
        client (*Any*): ``QuantumHardwareClient`` instance.

    Returns:
        ``ProviderRuntime`` bound to the requested provider's backend and task adapters.

    Raises:
        ValueError: If *provider* is not one of the supported platform names.
    """
    provider_name = str(provider).lower()
    if provider_name == "simulator":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=SimulatorBackendAdapter(),
            task_adapter=None,
        )
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
    if provider_name == "tencent":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=TencentBackendAdapter(),
            task_adapter=TencentTaskAdapter(client=client),
        )
    if provider_name == "fieldquantum":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=FieldQuantumBackendAdapter(),
            task_adapter=FieldQuantumTaskAdapter(client=client),
        )
    if provider_name == "origin":
        return ProviderRuntime(
            provider=provider_name,
            backend_adapter=OriginBackendAdapter(),
            task_adapter=OriginTaskAdapter(client=client),
        )
    raise ValueError("provider must be one of: 'quafu', 'tianyan', 'guodun', 'tencent', 'simulator', 'fieldquantum', or 'origin'")


__all__ = [
    "QuantumLanguage",
    "RemotePlatformClient",
    "ProviderRuntime",
    "create_provider_runtime",
    "list_available_hardware",
    "SimulatorBackendAdapter",
    "QuafuPlatform",
    "QuafuBackendAdapter",
    "QuafuTaskAdapter",
    "TianYanPlatform",
    "TianYanBackendAdapter",
    "TianYanTaskAdapter",
    "GuoDunPlatform",
    "GuoDunBackendAdapter",
    "GuoDunTaskAdapter",
    "TencentPlatform",
    "TencentBackendAdapter",
    "TencentTaskAdapter",
    "FieldQuantumPlatform",
    "FieldQuantumBackendAdapter",
    "FieldQuantumTaskAdapter",
    "FIELDQUANTUM_DEFAULT_URL",
    "OriginPlatform",
    "OriginBackendAdapter",
    "OriginTaskAdapter",
    "ORIGIN_DEFAULT_URL",
]
