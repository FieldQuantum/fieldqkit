from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _parse_pauli_string(pauli: str, num_qubits: int | None = None) -> List[Tuple[int, str]]:
    """Parse Pauli strings in either compact or indexed form."""
    pauli = pauli.strip()
    if not pauli:
        raise ValueError("pauli string is empty")

    tokens = pauli.split()
    if len(tokens) == 1 and tokens[0].isalpha():
        if num_qubits is None:
            num_qubits = len(tokens[0])
        if len(tokens[0]) != num_qubits:
            raise ValueError("pauli length mismatch with num_qubits")
        return [(i, p.upper()) for i, p in enumerate(tokens[0]) if p.upper() != "I"]

    parsed = []
    for tok in tokens:
        op = tok[0].upper()
        idx = int(tok[1:])
        if op not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"unsupported Pauli: {op}")
        if op != "I":
            parsed.append((idx, op))
    if num_qubits is not None:
        for idx, _ in parsed:
            if idx < 0 or idx >= num_qubits:
                raise ValueError("pauli index out of range")
    return parsed


def pauli_support(pauli: str, num_qubits: int | None = None) -> List[int]:
    """Return the sorted support indices of non-identity Pauli terms."""
    terms = _parse_pauli_string(pauli, num_qubits=num_qubits)
    return sorted({idx for idx, _ in terms})


def append_pauli_measurement(qc, pauli: str) -> None:
    """Append basis rotations and final measurements for a Pauli string."""
    num_qubits = qc.num_qubits if hasattr(qc, "num_qubits") else None
    terms = _parse_pauli_string(pauli, num_qubits=num_qubits)
    for idx, op in terms:
        if op == "X":
            qc.h(idx)
        elif op == "Y":
            # Use Sdg then H to rotate Y basis to Z.
            if hasattr(qc, "sdg"):
                qc.sdg(idx)
            else:
                qc.s(idx)
                qc.s(idx)
                qc.s(idx)
            qc.h(idx)
        elif op == "Z":
            pass
    qc.barrier()
    qc.measure_all()


def pauli_basis_pattern(pauli: str, num_qubits: int) -> List[str]:
    """Return basis pattern (I/X/Y/Z) per qubit for a Pauli string."""
    pattern = ["I"] * num_qubits
    terms = _parse_pauli_string(pauli, num_qubits=num_qubits)
    for idx, op in terms:
        pattern[idx] = op
    return pattern


def append_measurement_basis(qc, basis_pattern: Sequence[str]) -> None:
    """Apply basis rotations for a full pattern and append measurements."""
    for idx, op in enumerate(basis_pattern):
        if op == "X":
            qc.h(idx)
        elif op == "Y":
            # Use Sdg then H to rotate Y basis to Z.
            if hasattr(qc, "sdg"):
                qc.sdg(idx)
            else:
                qc.s(idx)
                qc.s(idx)
                qc.s(idx)
            qc.h(idx)
        elif op == "Z" or op == "I":
            pass
        else:
            raise ValueError(f"unsupported basis op: {op}")
    qc.barrier()
    qc.measure_all()


def _compatible_with_basis(pattern: Sequence[str], basis: Sequence[str]) -> bool:
    """Check whether two basis patterns are compatible for grouping."""
    for p, b in zip(pattern, basis):
        if p != "I" and b != "I" and p != b:
            return False
    return True


def _merge_basis(pattern: Sequence[str], basis: Sequence[str]) -> List[str]:
    """Merge two compatible basis patterns into a single pattern."""
    merged = list(basis)
    for i, p in enumerate(pattern):
        if merged[i] == "I" and p != "I":
            merged[i] = p
    return merged


def group_observables(observables: Sequence[str], num_qubits: int) -> List[Dict[str, object]]:
    """Group observables that can share a single measurement basis."""
    groups: List[Dict[str, object]] = []
    for obs in observables:
        pattern = pauli_basis_pattern(obs, num_qubits=num_qubits)
        placed = False
        for group in groups:
            basis = group["basis"]
            if _compatible_with_basis(pattern, basis):
                group["basis"] = _merge_basis(pattern, basis)
                group["observables"].append(obs)
                placed = True
                break
        if not placed:
            groups.append({"basis": pattern, "observables": [obs]})
    return groups


def pauli_expectation(samples: np.ndarray, pauli: str) -> float:
    """Compute expectation value from measurement samples in a Pauli basis."""
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    num_qubits = samples.shape[1]
    terms = _parse_pauli_string(pauli, num_qubits=num_qubits)

    if not terms:
        return 1.0

    eigenvalues = np.ones(samples.shape[0], dtype=float)
    for idx, op in terms:
        if op in {"Z", "X", "Y"}:
            eigenvalues *= 1.0 - 2.0 * samples[:, idx]
        else:
            raise ValueError(f"unsupported Pauli: {op}")
    return float(eigenvalues.mean())
