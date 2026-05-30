"""Quantum data-encoding circuits for QML.

Provides encoding strategies that map classical feature vectors to quantum
states.  Each function has two variants:

- **Concrete** (e.g. ``angle_encoding_circuit``): gates carry numeric values.
- **Symbolic** (e.g. ``angle_encoding_circuit_symbolic``): gates carry string
  parameter names like ``"x_0"``, ``"x_1"``, … so the circuit can be
  transpiled once and reused across different feature vectors.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..circuit import QuantumCircuit


# ---------------------------------------------------------------------------
# Angle encoding
# ---------------------------------------------------------------------------

def angle_encoding_circuit(
    features: Sequence[float],
    num_qubits: int,
    *,
    gate: str = "ry",
) -> QuantumCircuit:
    """Encode features as single-qubit rotation angles.

    Applies ``gate(feature_i)`` to qubit *i*.  If ``len(features) < num_qubits``
    the remaining qubits are left in |0⟩.

    Args:
        features: Classical feature values (one per qubit, at most *num_qubits*).
        num_qubits: Total number of qubits in the circuit.
        gate: Rotation gate name — ``"rx"``, ``"ry"``, or ``"rz"``.

    Returns:
        A ``QuantumCircuit`` with the encoding gates applied.
    """
    qc = QuantumCircuit(num_qubits)
    for i, val in enumerate(features[:num_qubits]):
        getattr(qc, gate)(float(val), i)
    return qc


def angle_encoding_circuit_symbolic(
    num_qubits: int,
    num_features: int,
    *,
    gate: str = "ry",
    prefix: str = "x",
) -> Tuple[QuantumCircuit, List[str]]:
    """Build a symbolic angle-encoding circuit.

    Returns a circuit with symbolic parameters ``x_0, x_1, …`` that can be
    composed with a trainable ansatz and transpiled once.

    Args:
        num_qubits: Total number of qubits.
        num_features: Number of features to encode (≤ *num_qubits*).
        gate: Rotation gate — ``"rx"``, ``"ry"``, or ``"rz"``.
        prefix: Parameter name prefix (default ``"x"``).

    Returns:
        ``(circuit, encoding_param_names)`` where *encoding_param_names*
        lists the symbolic parameter names in order.
    """
    n = min(num_features, num_qubits)
    param_names = [f"{prefix}_{i}" for i in range(n)]
    qc = QuantumCircuit(num_qubits)
    for i in range(n):
        getattr(qc, gate)(param_names[i], i)
    return qc, param_names


# ---------------------------------------------------------------------------
# IQP encoding
# ---------------------------------------------------------------------------

def iqp_encoding_circuit(
    features: Sequence[float],
    num_qubits: int,
    *,
    reps: int = 1,
) -> QuantumCircuit:
    """IQP (Instantaneous Quantum Polynomial) encoding.

    Structure per repetition:
      H on all qubits → RZ(x_i) on each qubit → RZZ(x_i * x_j) on adjacent pairs.

    Args:
        features: Classical feature values (at most *num_qubits*).
        num_qubits: Total number of qubits.
        reps: Number of repetitions of the encoding block.

    Returns:
        Encoded ``QuantumCircuit``.
    """
    qc = QuantumCircuit(num_qubits)
    n = min(len(features), num_qubits)
    for _ in range(reps):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(n):
            qc.rz(float(features[i]), i)
        for i in range(n - 1):
            angle = float(features[i]) * float(features[i + 1])
            qc.cx(i, i + 1)
            qc.rz(angle, i + 1)
            qc.cx(i, i + 1)
    # Final Hadamard layer: convert phase differences into amplitude differences
    for i in range(num_qubits):
        qc.h(i)
    return qc


def iqp_encoding_circuit_symbolic(
    num_qubits: int,
    num_features: int,
    *,
    reps: int = 1,
    prefix: str = "x",
) -> Tuple[QuantumCircuit, List[str]]:
    """Build a symbolic IQP-encoding circuit.

    Uses symbolic parameters ``x_0, x_1, …`` for single-qubit RZ gates.
    The ZZ-interaction gates use symbolic product expressions ``x_i*x_j``.

    Args:
        num_qubits: Total number of qubits.
        num_features: Number of features to encode (≤ *num_qubits*).
        reps: Number of repetitions of the encoding block.
        prefix: Parameter name prefix (default ``"x"``).

    Returns:
        ``(circuit, encoding_param_names)`` — the circuit and the list of
        base parameter names (the ZZ product expressions are derived from
        these automatically).
    """
    n = min(num_features, num_qubits)
    param_names = [f"{prefix}_{i}" for i in range(n)]
    qc = QuantumCircuit(num_qubits)
    for _ in range(reps):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(n):
            qc.rz(param_names[i], i)
        for i in range(n - 1):
            product_expr = f"{param_names[i]}*{param_names[i + 1]}"
            qc.cx(i, i + 1)
            qc.rz(product_expr, i + 1)
            qc.cx(i, i + 1)
    # Final Hadamard layer: convert phase differences into amplitude differences
    for i in range(num_qubits):
        qc.h(i)
    return qc, param_names


