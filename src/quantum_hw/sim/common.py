"""Shared simulator helpers for parameter resolution and operator materialization."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from ..circuit import QuantumCircuit
from .matrix import gate_matrix_dict


def resolve_param(
    qc: QuantumCircuit,
    param,
    param_values: Dict[str, object] | None = None,
):
    """Resolve scalar/symbolic parameter value for simulation."""
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
    """Return gate matrix tensor for fixed/parameterized gates."""
    mat_or_fn = gate_matrix_dict[gate]
    if callable(mat_or_fn):
        return mat_or_fn(*params, dtype=dtype, device=device)
    return mat_or_fn.to(device=device, dtype=dtype)


def single_pauli(op: str, *, dtype, device):
    """Return a single-qubit Pauli matrix tensor."""
    if op == "X":
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    if op == "Y":
        return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
    if op == "Z":
        return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    raise ValueError(f"unsupported Pauli: {op}")


def build_param_values_from_tensor(*, params, param_names: Sequence[str]) -> Dict[str, object]:
    """Convert a parameter tensor/vector into the symbolic name->value map."""
    expected = len(param_names)
    if params.numel() != expected:
        raise ValueError(f"params length must be {expected}")
    flat_params = params.reshape(-1)
    return {name: flat_params[i] for i, name in enumerate(param_names)}
