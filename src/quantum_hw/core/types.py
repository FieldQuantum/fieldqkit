"""Dataclasses for public result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


Counts = Dict[str, int]


@dataclass
class RunResult:
	task_ids: Optional[List[str]]
	samples: List[List[List[int]]]
	samples_zne: Optional[List[List[List[int]]]]
	probabilities: List[List[float]]
	probabilities_raw: List[List[float]]
	observable_values: Dict[str, float]
	observable_values_raw: Dict[str, float]


@dataclass
class CalibrationResult:
	target_qubits: List[int]
	per_qubit_confusion: Dict[int, List[List[float]]]


@dataclass
class ShadowResult:
	task_ids: Optional[List[str]] = None
	samples: Optional[List[List[int]]] = None
	basis_patterns: Optional[List[List[str]]] = None
	observables: Optional[List[str]] = None
	observable_estimates: Optional[Dict[str, float]] = None
	observable_estimates_raw: Optional[Dict[str, float]] = None
	observable_stderr: Optional[Dict[str, float]] = None
	observable_stderr_raw: Optional[Dict[str, float]] = None
	num_samples: Optional[int] = None


@dataclass
class VQEResult:
	best_energy: float
	best_params: List[float]
	energy_history: List[float]
	params_history: Optional[List[List[float]]] = None
	grad_history: Optional[List[List[float]]] = None
	last_expectations: Optional[Dict[str, float]] = None
	clifford_fitting: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class QAOAResult:
	best_cost: float
	best_params: List[float]
	cost_history: List[float]
	params_history: Optional[List[List[float]]] = None
	grad_history: Optional[List[List[float]]] = None
	last_expectations: Optional[Dict[str, float]] = None
