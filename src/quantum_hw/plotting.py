from __future__ import annotations

import matplotlib.pyplot as plt


def _flatten_probabilities(probabilities):
    if probabilities is None:
        return None
    if isinstance(probabilities, list) and probabilities and isinstance(probabilities[0], list):
        return probabilities[0]
    return probabilities


def _select_key_basis(raw_probs, mit_probs, num_qubits: int, max_states: int = 16):
    total = len(raw_probs)
    if total <= max_states:
        return list(range(total))
    scores = [max(r, m) for r, m in zip(raw_probs, mit_probs)]
    ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)
    selected = set(ranked[: max_states - 2])
    selected.add(0)
    selected.add(total - 1)
    return sorted(selected)


def plot_probabilities_compare(raw, mitigated, num_qubits: int, max_labels: int = 16) -> None:
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


def plot_observables_compare(raw, mitigated) -> None:
    if raw is None and mitigated is None:
        return
    if isinstance(mitigated, dict) or isinstance(raw, dict):
        keys = sorted(set((raw or {}).keys()) | set((mitigated or {}).keys()))
        raw_vals = [(raw or {}).get(k, 0.0) for k in keys]
        mit_vals = [(mitigated or {}).get(k, 0.0) for k in keys]
        x = list(range(len(keys)))
        width = 0.4
        plt.figure(figsize=(10, 3))
        plt.bar([i - width / 2 for i in x], raw_vals, width=width, color="#9E9E9E", label="Raw")
        plt.bar([i + width / 2 for i in x], mit_vals, width=width, color="#4C78A8", label="Mitigated")
        plt.xticks(x, keys, rotation=45, ha="right")
        plt.ylabel("Expectation")
        plt.title("Observable comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(4, 3))
        plt.bar([0, 1], [raw or 0.0, mitigated or 0.0], color=["#9E9E9E", "#4C78A8"])
        plt.xticks([0, 1], ["Raw", "Mitigated"])
        plt.ylabel("Expectation")
        plt.title("Observable comparison")
        plt.tight_layout()
        plt.show()

