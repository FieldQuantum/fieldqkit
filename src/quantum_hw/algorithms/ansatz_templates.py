"""Shared ansatz construction helpers used by VQE and compression workflows."""

from __future__ import annotations

from typing import Sequence

from ..circuit import QuantumCircuit


def build_hardware_efficient_ansatz(
    num_qubits: int,
    params: Sequence[float],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    expected = 2 * num_qubits * (layers + 1)
    if len(params) != expected:
        raise ValueError(f"params length must be {expected} (2 * num_qubits * (layers + 1))")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.rx(float(params[idx]), q)
            idx += 1
        for q in range(num_qubits):
            qc.ry(float(params[idx]), q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    for q in range(num_qubits):
        qc.rx(float(params[idx]), q)
        idx += 1
    for q in range(num_qubits):
        qc.ry(float(params[idx]), q)
        idx += 1
    return qc


def build_hardware_efficient_ansatz_symbolic(
    num_qubits: int,
    param_names: Sequence[str],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    expected = 2 * num_qubits * (layers + 1)
    if len(param_names) != expected:
        raise ValueError(f"param_names length must be {expected} (2 * num_qubits * (layers + 1))")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.rx(param_names[idx], q)
            idx += 1
        for q in range(num_qubits):
            qc.ry(param_names[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    for q in range(num_qubits):
        qc.rx(param_names[idx], q)
        idx += 1
    for q in range(num_qubits):
        qc.ry(param_names[idx], q)
        idx += 1
    return qc


def build_ucc_num_params(num_qubits: int, layers: int) -> int:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if layers <= 0:
        raise ValueError("layers must be positive")
    return layers * (num_qubits + max(num_qubits - 1, 0))


def build_ucc_ansatz(
    num_qubits: int,
    params: Sequence[float],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    expected = build_ucc_num_params(num_qubits, layers)
    if len(params) != expected:
        raise ValueError(f"params length must be {expected} for ucc ansatz")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.ry(float(params[idx]), q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            qc.ry(float(params[idx]), q + 1)
            qc.cx(q, q + 1)
            idx += 1
    return qc


def build_ucc_ansatz_symbolic(
    num_qubits: int,
    param_names: Sequence[str],
    *,
    layers: int = 1,
) -> QuantumCircuit:
    expected = build_ucc_num_params(num_qubits, layers)
    if len(param_names) != expected:
        raise ValueError(f"param_names length must be {expected} for ucc ansatz")

    qc = QuantumCircuit(num_qubits)
    idx = 0
    for _ in range(layers):
        for q in range(num_qubits):
            qc.ry(param_names[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            qc.ry(param_names[idx], q + 1)
            qc.cx(q, q + 1)
            idx += 1
    return qc
