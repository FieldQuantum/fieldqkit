"""Plotting helpers for probabilities and observables."""

from __future__ import annotations

import matplotlib.pyplot as plt


def _flatten_probabilities(probabilities):
	"""Extract the first probability vector from grouped run output.

	Args:
		probabilities: Probabilities (possibly nested list from grouped runs).

	Returns:
		Flat probability list or ``None``.
	"""
	if probabilities is None:
		return None
	if isinstance(probabilities, list) and probabilities and isinstance(probabilities[0], list):
		return probabilities[0]
	return probabilities


def _select_key_basis(raw_probs, mit_probs, num_qubits: int, max_states: int = 16):
	"""Select representative basis states for visualization.

	Args:
		raw_probs: Raw probability distribution (list or array).
		mit_probs: Mitigated probability distribution (list or array).
		num_qubits (*int*): Number of qubits.
		max_states (*int*): Maximum number of basis states to select for display. Defaults to ``16``.

	Returns:
		List of selected basis state indices for plotting.
	"""
	total = len(raw_probs)
	if total <= max_states:
		return list(range(total))
	scores = [max(r, m) for r, m in zip(raw_probs, mit_probs)]
	ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)
	selected = set(ranked[: max_states - 2])
	selected.add(0)
	selected.add(total - 1)
	return sorted(selected)


def _as_float(value, default: float = 0.0) -> float:
	"""Convert scalar-like value to float with a safe default.

	Args:
		value: Value to convert to float.
		default (*float*): Fallback if conversion fails. Defaults to ``0.0``.

	Returns:
		``float`` value, or *default* on failure.
	"""
	if value is None:
		return default
	if isinstance(value, (list, tuple)):
		if len(value) == 0:
			return default
		return _as_float(value[0], default=default)
	return float(value)


def _ordered_observable_keys(raw, mitigated, observables=None):
	"""Resolve observable keys in a stable plotting order.

	Args:
		raw: Raw observable expectation values (scalar, dict, or ``None``).
		mitigated: Mitigated observable expectation values (scalar, dict, or ``None``).
		observables: Observable operators for key ordering. Defaults to ``None``.

	Returns:
		Ordered list of observable key strings.
	"""
	raw_keys = set((raw or {}).keys()) if isinstance(raw, dict) else set()
	mit_keys = set((mitigated or {}).keys()) if isinstance(mitigated, dict) else set()
	all_keys = raw_keys | mit_keys
	if not all_keys:
		return []
	if observables is None:
		return sorted(all_keys)
	if isinstance(observables, str):
		preferred = [observables]
	else:
		preferred = list(observables)
	ordered = []
	seen = set()
	for key in preferred:
		if key in all_keys and key not in seen:
			ordered.append(key)
			seen.add(key)
	for key in sorted(all_keys):
		if key not in seen:
			ordered.append(key)
	return ordered


def plot_probabilities_compare(raw, mitigated, num_qubits: int, max_labels: int = 16) -> None:
	"""Plot raw vs mitigated probabilities for selected basis states.

	Args:
		raw: Raw probability distribution (dict or array).
		mitigated: Mitigated probability distribution (dict or array).
		num_qubits (*int*): Number of qubits.
		max_labels (*int*): Maximum number of basis-state labels on the x-axis. Defaults to ``16``.
	"""
	raw_probs = _flatten_probabilities(raw)
	mit_probs = _flatten_probabilities(mitigated)
	if raw_probs is None and mit_probs is None:
		return
	if raw_probs is None:
		raw_probs = [0.0] * len(mit_probs)
	if mit_probs is None:
		mit_probs = [0.0] * len(raw_probs)
	indices = _select_key_basis(raw_probs, mit_probs, num_qubits, max_states=max_labels)
	x = list(range(len(raw_probs)))
	width = 0.4
	plt.figure(figsize=(12, 4))
	plt.bar([i - width / 2 for i in x], raw_probs, width=width, color="#9E9E9E", label="Raw")
	plt.bar([i + width / 2 for i in x], mit_probs, width=width, color="#4C78A8", label="Mitigated")
	labels = [format(i, f"0{num_qubits}b") for i in indices]
	plt.xticks(indices, labels, rotation=90)
	plt.xlabel("Basis state")
	plt.ylabel("Probability")
	plt.title("Readout mitigation comparison")
	plt.tick_params(right=True, top=True)
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_observables_compare(raw, mitigated, observables=None) -> None:
	"""Plot comparison of observable expectations (scalar or dict).

	Args:
		raw: Raw observable expectation values (scalar or dict).
		mitigated: Mitigated observable expectation values (scalar or dict).
		observables: Observable operators for key ordering. Defaults to ``None``.
	"""
	if raw is None and mitigated is None:
		return
	if isinstance(mitigated, dict) or isinstance(raw, dict):
		keys = _ordered_observable_keys(raw, mitigated, observables=observables)
		if not keys:
			return
		raw_vals = [(raw or {}).get(k, 0.0) for k in keys]
		mit_vals = [(mitigated or {}).get(k, 0.0) for k in keys]
		x = list(range(len(keys)))
		width = 0.4
		plt.figure(figsize=(max(6, len(keys) * 0.9), 3))
		plt.bar([i - width / 2 for i in x], raw_vals, width=width, color="#9E9E9E", label="Raw")
		plt.bar([i + width / 2 for i in x], mit_vals, width=width, color="#4C78A8", label="Mitigated")
		plt.xticks(x, keys, rotation=45, ha="right")
		plt.ylabel("Expectation")
		plt.title("Observable comparison")
		plt.tick_params(right=True, top=True)
		plt.legend()
		plt.tight_layout()
		plt.show()
	else:
		raw_val = _as_float(raw)
		mitigated_val = _as_float(mitigated)
		plt.figure(figsize=(4, 3))
		plt.bar([0, 1], [raw_val, mitigated_val], color=["#9E9E9E", "#4C78A8"])
		plt.xticks([0, 1], ["Raw", "Mitigated"])
		if isinstance(observables, str):
			plt.ylabel(observables)
		else:
			plt.ylabel("Expectation")
		plt.tick_params(right=True, top=True)
		plt.tight_layout()
		plt.show()
