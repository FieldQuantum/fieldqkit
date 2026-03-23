"""Unified backend abstraction and provider-specific backend adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .backend import Backend, rank_chips
from .provider_backend import build_cqlib_backend_bundle


@dataclass
class ResolvedBackend:
    """Unified backend descriptor for all providers."""

    provider: str
    hardware_name: str
    backend: Backend
    target_qubits: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    """Common interface for resolving provider backends."""

    provider: str

    @abstractmethod
    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> ResolvedBackend:
        """Resolve a concrete backend target for one provider."""


class QuafuBackendAdapter(BackendAdapter):
    provider = "quafu"

    def __init__(self, *, tmgr: Any) -> None:
        self._tmgr = tmgr

    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> ResolvedBackend:
        ranked_chips = rank_chips(
            self._tmgr,
            num_qubits=num_qubits,
            prefer_chips=prefer_hardware,
            weights=rank_weights,
        )
        if not ranked_chips:
            raise RuntimeError("no available chips satisfy num_qubits requirement")

        chip_name = ranked_chips[0]
        backend = Backend(chip_name)
        return ResolvedBackend(
            provider=self.provider,
            hardware_name=chip_name,
            backend=backend,
            target_qubits=None,
            metadata={"ranked_chips": ranked_chips},
        )


class CqlibBackendAdapter(BackendAdapter):
    provider = "cqlib"

    def __init__(
        self,
        *,
        login_key: str,
        platform: str = "tianyan",
        machine_name: Optional[str] = None,
    ) -> None:
        if not login_key:
            raise ValueError("cqlib login key cannot be empty")

        from cqlib.quantum_platform import GuoDunPlatform, TianYanPlatform

        platform_name = str(platform).lower()
        platform_cls = TianYanPlatform if platform_name == "tianyan" else GuoDunPlatform
        self._platform_name = platform_name
        self._machine_name = machine_name
        self._platform_obj = platform_cls(
            login_key=login_key,
            auto_login=True,
            machine_name=machine_name,
        )

    def resolve_backend(
        self,
        *,
        num_qubits: int,
        prefer_hardware: Optional[Sequence[str] | str] = None,
        rank_weights: Optional[Dict[str, float]] = None,
    ) -> ResolvedBackend:
        del rank_weights  # cqlib currently does not expose queue-based ranking in this layer.

        machine_name = self._machine_name or "tianyan176"
        if isinstance(prefer_hardware, str) and prefer_hardware.strip():
            machine_name = prefer_hardware.strip()
        elif isinstance(prefer_hardware, Sequence) and len(prefer_hardware) > 0:
            machine_name = str(prefer_hardware[0]).strip() or machine_name

        bundle = build_cqlib_backend_bundle(
            platform_obj=self._platform_obj,
            machine_name=machine_name,
            num_qubits=num_qubits,
        )
        return ResolvedBackend(
            provider=self.provider,
            hardware_name=machine_name,
            backend=bundle.backend,
            target_qubits=bundle.target_qubits,
            metadata={
                "platform_obj": self._platform_obj,
                "platform_name": self._platform_name,
                "machine_name": machine_name,
            },
        )
