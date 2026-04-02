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
    """Construct an Ry-Rx layered ansatz with CZ entanglers.

    Each layer applies Rx and Ry rotations on every qubit followed by a
    nearest-neighbour CZ chain.  A final Rx+Ry rotation block is appended.

    Args:
        num_qubits (*int*): Number of qubits.
        params (*Sequence[float]*): Rotation angles consumed left-to-right.
        layers (*int*): Number of ansatz layers. Defaults to ``1``.

    Returns:
        Constructed ``QuantumCircuit``.

    Raises:
        ValueError: f'params length must be {expected} (2 * num_qubits * (layers + 1))'
    """
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
    """Symbolic variant of :func:`build_hardware_efficient_ansatz`.

    Returns a circuit whose rotation gates carry symbolic parameter names
    instead of concrete float values, suitable for template-based
    parameter-shift evaluation.

    Args:
        num_qubits (*int*): Number of qubits.
        param_names (*Sequence[str]*): Symbolic names bound to rotation gates.
        layers (*int*): Number of ansatz layers. Defaults to ``1``.

    Returns:
        Constructed ``QuantumCircuit``.

    Raises:
        ValueError: f'param_names length must be {expected} (2 * num_qubits * (layers + 1))'
    """
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
    """Return the number of variational parameters for a UCC ansatz.

    Args:
        num_qubits (*int*): Number of qubits.
        layers (*int*): Number of ansatz layers.

    Returns:
        Total parameter count.

    Raises:
        ValueError: num_qubits must be positive
        ValueError: layers must be positive
    """
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
    """Construct a Unitary Coupled-Cluster (UCC) inspired ansatz.

    Each layer applies single-qubit Ry rotations followed by CNOT-Ry-CNOT
    entangling blocks on neighbouring qubit pairs.

    Args:
        num_qubits (*int*): Number of qubits.
        params (*Sequence[float]*): Rotation angles consumed left-to-right.
        layers (*int*): Number of ansatz layers. Defaults to ``1``.

    Returns:
        Constructed ``QuantumCircuit``.

    Raises:
        ValueError: f'params length must be {expected} for ucc ansatz'
    """
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
    """Symbolic variant of :func:`build_ucc_ansatz`.

    Returns a UCC circuit with symbolic parameter names instead of concrete
    float values.

    Args:
        num_qubits (*int*): Number of qubits.
        param_names (*Sequence[str]*): Symbolic names bound to rotation gates.
        layers (*int*): Number of ansatz layers. Defaults to ``1``.

    Returns:
        Constructed ``QuantumCircuit``.

    Raises:
        ValueError: f'param_names length must be {expected} for ucc ansatz'
    """
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
