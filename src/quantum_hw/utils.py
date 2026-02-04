from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np


QASM_QUBIT_PATTERN = re.compile(r"q\[(\d+)\]")


def extract_qubits_from_openqasm2(qasm: str) -> List[int]:
    """Extract qubit indices referenced by an OpenQASM2 string."""
    qubits = set()
    for line in qasm.splitlines():
        stripped = line.strip()
        if stripped.startswith("qreg "):
            continue
        for m in QASM_QUBIT_PATTERN.finditer(line):
            qubits.add(int(m.group(1)))
    return sorted(qubits)


def counts_dict_to_vector(result: Dict[str, int], num_qubits: int, reverse_bits: bool = False) -> np.ndarray:
    """Convert a counts dict to a dense vector ordered by basis index."""
    bases = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]
    counts = np.zeros(len(bases), dtype=int)
    for i, base in enumerate(bases):
        key = base[::-1] if reverse_bits else base
        counts[i] = int(result.get(key, 0))
    return counts


def get_probabilities(result: Dict[str, int], num_qubits: int) -> np.ndarray:
    """Normalize counts into a probability vector (hardware bit order handled)."""
    counts = counts_dict_to_vector(result, num_qubits, reverse_bits=True)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def get_samples(result: Dict[str, int], num_qubits: int) -> np.ndarray:
    """Expand counts into a sample array aligned with get_probabilities bit order."""
    samples = []
    for key, count in result.items():
        # Reverse bit order to align with probability vector convention.
        bits = [int(b) for b in key[::-1]]
        samples.extend([bits] * count)
    return np.array(samples, dtype=int)

