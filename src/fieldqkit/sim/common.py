"""Shared simulator helpers for parameter resolution and operator materialization."""

from __future__ import annotations

from contextlib import suppress
from functools import lru_cache
from typing import Dict, Sequence

import torch

from ..circuit import QuantumCircuit
from .matrix import gate_matrix_dict


def auto_sim_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve simulation device: explicit > least-utilized CUDA > CPU.

    Args:
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        Resolved ``torch.device``.
    """
    if device is not None:
        return torch.device(device)
    cuda_device = _least_used_cuda_device()
    if cuda_device is not None:
        return cuda_device
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _pynvml_module():
    """Return an initialized ``pynvml`` module, or ``None`` if unavailable."""
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    with suppress(Exception):
        pynvml.nvmlInit()
        return pynvml
    return None


def _cuda_utilization_percent(device_index: int) -> int | None:
    """Return current GPU compute utilization (0-100) via NVML, or ``None``."""
    pynvml = _pynvml_module()
    if pynvml is None:
        return None
    with suppress(Exception):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    return None


def _cuda_free_memory_bytes(device_index: int) -> int | None:
    """Return free CUDA memory for a device, or ``None`` if unavailable."""
    with suppress(Exception):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        return int(free_bytes)
    return None


def _least_used_cuda_device() -> torch.device | None:
    """Pick the CUDA device with the lowest GPU utilization, if CUDA is available.

    Ranks by NVML compute utilization (lower is better); ties (and the case where
    NVML is unavailable) are broken by largest free memory.
    """
    if not torch.cuda.is_available():
        return None

    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return None

    best_device_index = 0
    best_key: tuple[int, int] | None = None
    for device_index in range(device_count):
        util = _cuda_utilization_percent(device_index)
        free_bytes = _cuda_free_memory_bytes(device_index)
        util_key = 101 if util is None else util  # push unknowns to the back
        free_key = -1 if free_bytes is None else free_bytes
        key = (util_key, -free_key)
        if best_key is None or key < best_key:
            best_key = key
            best_device_index = device_index
    return torch.device(f"cuda:{best_device_index}")


def resolve_param(
    qc: QuantumCircuit,
    param,
    param_values: Dict[str, object] | None = None,
):
    """Resolve scalar/symbolic parameter value for simulation.

    Args:
        qc (*QuantumCircuit*): Quantum circuit containing parameter bindings.
        param: Scalar value (``int``/``float``) or symbolic parameter name (``str``).
        param_values (*Dict[str, object] | None*): Optional external parameter value mapping. Defaults to ``None``.

    Returns:
        ``float`` resolved numeric value.

    Raises:
        TypeError: f'unsupported parameter type: {type(param)}'
        ValueError: f'missing parameter value for {name}'
    """
    if isinstance(param, (float, int)):
        return float(param)
    if isinstance(param, str):
        if param_values is not None and param in param_values:
            return param_values[param]
        if param in qc.params_value:
            value = qc.params_value[param]
            if isinstance(value, (float, int)):
                return float(value)

        def _symbol_resolver(name: str):
            """Look up a symbolic parameter name and return its numeric value.

            Checks ``pi``, then *param_values*, then ``qc.params_value`` in
            that order.

            Args:
                name (*str*): Symbolic parameter name to resolve.

            Returns:
                Numeric value for the symbol.

            Raises:
                ValueError: If *name* cannot be found in any value source.
            """
            if name == "pi":
                return float(torch.pi)
            if param_values is not None and name in param_values:
                return param_values[name]
            if name in qc.params_value and isinstance(qc.params_value[name], (float, int)):
                return float(qc.params_value[name])
            raise ValueError(f"missing parameter value for {name}")

        return qc._eval_param_expression(param, symbol_resolver=_symbol_resolver)
    raise TypeError(f"unsupported parameter type: {type(param)}")


def materialize_gate_matrix(
    gate: str,
    params,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """Return gate matrix tensor for fixed/parameterized gates.

    Args:
        gate (*str*): Gate name (e.g. ``'rx'``, ``'cz'``).
        params: Numeric parameter values for parameterized gates.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        ``torch.Tensor`` unitary matrix.
    """
    mat_or_fn = gate_matrix_dict[gate]
    if callable(mat_or_fn):
        return mat_or_fn(*params, dtype=dtype, device=device)
    return mat_or_fn.to(device=device, dtype=dtype)


def single_pauli(op: str, *, dtype, device):
    """Return a single-qubit Pauli matrix tensor.

    Args:
        op (*str*): Pauli operator: ``'X'``, ``'Y'``, or ``'Z'``.
        dtype: Torch data type.
        device: Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        ``torch.Tensor`` of shape ``(2, 2)``.

    Raises:
        ValueError: f'unsupported Pauli: {op}'
    """
    if op == "X":
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    if op == "Y":
        return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
    if op == "Z":
        return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    raise ValueError(f"unsupported Pauli: {op}")


def build_param_values_from_tensor(*, params, param_names: Sequence[str]) -> Dict[str, object]:
    """Convert a parameter tensor/vector into the symbolic name->value map.

    Args:
        params: Parameter values.
        param_names (*Sequence[str]*): Names of variational parameters.

    Returns:
        ``Dict[str, object]`` mapping parameter names to their values.

    Raises:
        ValueError: f'params length must be {expected}'
    """
    expected = len(param_names)
    if params.numel() != expected:
        raise ValueError(f"params length must be {expected}")
    flat_params = params.reshape(-1)
    return {name: flat_params[i] for i, name in enumerate(param_names)}
