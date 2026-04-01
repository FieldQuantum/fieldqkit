"""Shared simulator helpers for parameter resolution and operator materialization."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from ..circuit import QuantumCircuit
from .matrix import gate_matrix_dict


def auto_sim_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve simulation device: explicit > CUDA > CPU.

    Args:
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``torch.device`` result.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_param(
    qc: QuantumCircuit,
    param,
    param_values: Dict[str, object] | None = None,
):
    """Resolve scalar/symbolic parameter value for simulation.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        param: Param.
        param_values (*Dict[str, object] | None*): Param values (``Dict[str, object] | None``). Defaults to ``None``.

    Returns:
        Result.

    Raises:
        TypeError: f'unsupported parameter type: {type(param)}
        ValueError: f'missing parameter value for {name}
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
        gate (*str*): Gate specification or name.
        params: Parameter values.
        dtype (*torch.dtype*): Torch data type.
        device (*torch.device*): Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        Result.
    """
    mat_or_fn = gate_matrix_dict[gate]
    if callable(mat_or_fn):
        return mat_or_fn(*params, dtype=dtype, device=device)
    return mat_or_fn.to(device=device, dtype=dtype)


def single_pauli(op: str, *, dtype, device):
    """Return a single-qubit Pauli matrix tensor.

    Args:
        op (*str*): Op (``str``).
        dtype: Torch data type.
        device: Torch device (``'cpu'`` or ``'cuda'``).

    Returns:
        Result.

    Raises:
        ValueError: f'unsupported Pauli: {op}
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
        Result dictionary.

    Raises:
        ValueError: f'params length must be {expected}
    """
    expected = len(param_names)
    if params.numel() != expected:
        raise ValueError(f"params length must be {expected}")
    flat_params = params.reshape(-1)
    return {name: flat_params[i] for i, name in enumerate(param_names)}
