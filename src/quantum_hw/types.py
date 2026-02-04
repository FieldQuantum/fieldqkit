from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


Counts = Dict[str, int]


@dataclass
class RunResult:
    task_ids: Optional[List[str]] = None
    samples: Optional[List[List[int]] | List[List[List[int]]]] = None
    probabilities: Optional[List[float] | List[List[float]]] = None
    probabilities_raw: Optional[List[float] | List[List[float]]] = None
    observable_values: Optional[float | Dict[str, float]] = None
    observable_values_raw: Optional[float | Dict[str, float]] = None


@dataclass
class CalibrationResult:
    target_qubits: List[int]
    per_qubit_confusion: Dict[int, List[List[float]]]
