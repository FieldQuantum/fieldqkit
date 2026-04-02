"""Quantum Machine Learning training routines.

Provides supervised classification via ``run_pqc_classifier``: train a PQC to
map encoded quantum states to class labels.

``run_pqc_classifier`` supports both **autograd** (simulator, torch) and
**parameter-shift** (hardware backend) gradient methods, mirroring the VQE
dual-path design.

The encoding and ansatz circuits are composed symbolically and transpiled
**once** — dramatically reducing the compilation overhead for parameter-shift
on hardware.
"""

from __future__ import annotations

import logging
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

import numpy as np

from ..circuit import QuantumCircuit
from ..core.types import QBMResult, QMLResult
from .ansatz_templates import (
    build_hardware_efficient_ansatz_symbolic,
)
from .optimizer_utils import (
    adam_update,
    evaluate_energy_with_backend,
    instantiate_transpiled_template,
)
from .qml_encoding import (
    angle_encoding_circuit_symbolic,
    iqp_encoding_circuit_symbolic,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_ansatz_symbolic(num_qubits: int, layers: int):
    """Build a hardware-efficient ansatz and return (circuit, param_names).

    Args:
        num_qubits (*int*): Number of qubits.
        layers (*int*): Number of ansatz layers.

    Returns:
        Tuple of ``(circuit, param_names)`` where *circuit* is the
        symbolic ``QuantumCircuit`` and *param_names* is a list of
        parameter name strings.
    """
    num_params = 2 * num_qubits * (layers + 1)
    param_names = [f"θ_{i}" for i in range(num_params)]
    qc = build_hardware_efficient_ansatz_symbolic(num_qubits, param_names, layers=layers)
    return qc, param_names


def _compose_circuits(front: QuantumCircuit, back: QuantumCircuit) -> QuantumCircuit:
    """Concatenate two circuits on the same register.

    Args:
        front (*QuantumCircuit*): Circuit applied first.
        back (*QuantumCircuit*): Circuit appended after *front*.

    Returns:
        Constructed ``QuantumCircuit``.
    """
    qc = front.deepcopy()
    for gate in back.gates:
        qc.gates.append(gate)
    # Merge qubit lists so transpilation sees the full register.
    merged = sorted(set(qc.qubits) | set(back.qubits))
    qc.qubits = merged
    return qc


# ---------------------------------------------------------------------------
# 1. Supervised PQC classifier
# ---------------------------------------------------------------------------

def _z_pauli_string(q: int, num_qubits: int) -> str:
    """Return a Pauli string with Z on qubit *q* and I elsewhere.

    Args:
        q (*int*): Target qubit index.
        num_qubits (*int*): Number of qubits.

    Returns:
        Pauli string with ``Z`` on qubit *q* and ``I`` elsewhere.
    """
    return "I" * q + "Z" + "I" * (num_qubits - q - 1)


def _classifier_loss_and_dl_dz(
    z_values: Sequence[float],
    label: int,
    num_classes: int,
) -> Tuple[float, np.ndarray]:
    """Compute classification loss and analytical dL/d⟨Z⟩.

    Args:
        z_values (*Sequence[float]*): Per-qubit ⟨Z⟩ expectation values.
        label (*int*): True class label.
        num_classes (*int*): Number of classes (2 for binary cross-entropy, >2 for softmax).

    Returns:
        ``(loss, dl_dz)`` where *dl_dz* has one entry per measurement qubit.
    """
    eps = 1e-10
    if num_classes == 2:
        z = z_values[0]
        p = np.clip((1.0 - z) / 2.0, eps, 1.0 - eps)
        y = float(label)
        loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        dl_dp = -(y / p - (1.0 - y) / (1.0 - p))
        dl_dz = dl_dp * (-0.5)
        return loss, np.array([dl_dz])
    else:
        logits = np.asarray(z_values, dtype=np.float64)
        logits_shifted = logits - logits.max()
        exp_l = np.exp(logits_shifted)
        softmax = exp_l / exp_l.sum()
        loss = -np.log(softmax[label] + eps)
        grad = softmax.copy()
        grad[label] -= 1.0
        return loss, grad


def _batch_loss_and_grads(
    z_values_list: Sequence[Sequence[float]],
    labels: Sequence[int],
    num_classes: int,
) -> Tuple[float, List[np.ndarray]]:
    """Average loss and per-sample dL/d⟨Z⟩ over a batch.

    Args:
        z_values_list (*Sequence[Sequence[float]]*): Per-sample ⟨Z⟩ values.
        labels (*Sequence[int]*): Target labels.
        num_classes (*int*): Number of classes.

    Returns:
        Tuple of ``(avg_loss, dl_dz_list)`` where *dl_dz_list* has one
        gradient array per sample.
    """
    total = 0.0
    dl_dz_list: List[np.ndarray] = []
    for z_vals, lab in zip(z_values_list, labels):
        l, dl = _classifier_loss_and_dl_dz(z_vals, lab, num_classes)
        total += l
        dl_dz_list.append(dl)
    return total / len(labels), dl_dz_list


def _predictions_from_z(
    z_values_list: Sequence[Sequence[float]],
    num_classes: int,
) -> List[int]:
    """Convert ⟨Z⟩ values to class predictions.

    Args:
        z_values_list (*Sequence[Sequence[float]]*): Per-sample ⟨Z⟩ expectation values.
        num_classes (*int*): Number of classes (2 for binary, >2 for multi-class).

    Returns:
        List of predicted class labels, one per sample.
    """
    preds: List[int] = []
    for z_vals in z_values_list:
        if num_classes == 2:
            preds.append(0 if z_vals[0] > 0 else 1)
        else:
            preds.append(int(np.argmax(z_vals)))
    return preds


def _get_z_autograd(build_state, expectation_pauli, template, param_names,
                    features_list, theta, z_observables, num_qubits):
    """Get ⟨Z⟩ values for all samples via autograd simulator (no grad).

    Args:
        build_state: Callable that builds a statevector from a circuit.
        expectation_pauli: Callable for Pauli expectation values.
        template: Symbolic ``QuantumCircuit`` template.
        param_names: Names of variational parameters.
        features_list: List of feature vectors, one per sample.
        theta: Current variational parameter values.
        z_observables: Pauli-Z strings to measure.
        num_qubits: Number of qubits.

    Returns:
        ``List[List[float]]`` of shape ``(n_samples, n_observables)``.
    """
    import torch
    results: List[List[float]] = []
    params_t = torch.tensor(theta, dtype=torch.float64)
    with torch.no_grad():
        for feat in features_list:
            feat_t = torch.tensor(feat, dtype=torch.float64)
            all_p = torch.cat([feat_t, params_t])
            state = build_state(template, params=all_p, param_names=param_names)
            z_vals = [
                float(expectation_pauli(state, obs, num_qubits=num_qubits).real)
                for obs in z_observables
            ]
            results.append(z_vals)
    return results


def _get_z_backend(client, template, param_names, features_list, theta,
                   z_observables, z_hamiltonian, backend_kwargs, name_prefix):
    """Get ⟨Z⟩ values for all samples via hardware backend.

    Args:
        client: ``QuantumHardwareClient`` instance.
        template: Transpiled symbolic ``QuantumCircuit`` template.
        param_names: Names of variational parameters.
        features_list: List of feature vectors, one per sample.
        theta: Current variational parameter values.
        z_observables: Pauli-Z strings to measure.
        z_hamiltonian: Hamiltonian terms for backend observable evaluation.
        backend_kwargs: Extra keyword arguments forwarded to the backend.
        name_prefix: Task name prefix for hardware submissions.

    Returns:
        ``List[List[float]]`` of shape ``(n_samples, n_observables)``.
    """
    results: List[List[float]] = []
    for si, feat in enumerate(features_list):
        all_vals = np.concatenate([feat, theta])
        qc_bound = instantiate_transpiled_template(template, param_names, all_vals)
        _, exps = evaluate_energy_with_backend(
            client, qc_bound, name=f"{name_prefix}_s{si}",
            hamiltonian=z_hamiltonian, **backend_kwargs,
        )
        results.append([exps.get(obs, 0.0) for obs in z_observables])
    return results


def run_pqc_classifier(
    num_qubits: int,
    train_data: Sequence[Tuple[Sequence[float], int]],
    *,
    test_data: Optional[Sequence[Tuple[Sequence[float], int]]] = None,
    encoding: Union[str, Callable] = "angle",
    encoding_kwargs: Optional[dict] = None,
    num_classes: int = 2,
    measurement_qubits: Optional[Sequence[int]] = None,
    layers: int = 2,
    max_iters: int = 100,
    learning_rate: float = 0.01,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    gradient_method: str = "autograd",
    # --- hardware / parameter-shift params (ignored for autograd) ---
    client=None,
    backend=None,
    chip_name: str = "",
    shots: int = 4096,
    shift: float = np.pi / 2,
    zne: bool = False,
    readout_mitigation: bool = False,
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
) -> QMLResult:
    """Train a parameterized quantum classifier.

    Training samples are ``(features, label)`` pairs.  The encoding circuit is
    built symbolically so the full template (encoding + ansatz) is transpiled
    **once** and reused across all samples.

    Args:
        num_qubits: Number of qubits.
        train_data: List of ``(features, label)`` pairs.
        test_data: Optional list of ``(features, label)`` pairs for validation.
            When provided, the best model is selected by test loss instead of
            training loss, and test accuracy is reported.
        encoding: Encoding strategy — ``"angle"`` / ``"iqp"``, or a callable
            ``(num_qubits, num_features) -> (QuantumCircuit, param_names)``.
        encoding_kwargs: Extra keyword arguments for encoding
            (for example ``{"gate": "rx"}``).
        num_classes: Number of classes (default 2).
        measurement_qubits: Indices of qubits to measure.
        layers: Number of ansatz layers.
        max_iters: Training iterations (full-batch).
        learning_rate: Adam learning rate.
        seed: Optional random seed.
        callback: ``(iter, loss)`` callback.
        gradient_method: ``"autograd"`` or ``"parameter-shift"``.
        client: ``QuantumHardwareClient`` instance (parameter-shift only).
        backend: Target backend (parameter-shift only).
        chip_name: Target chip identifier (parameter-shift only).
        shots: Measurement shots (parameter-shift only).
        shift: Parameter-shift magnitude (parameter-shift only).
        zne: Enable zero-noise extrapolation (parameter-shift only).
        readout_mitigation: Enable readout mitigation (parameter-shift only).
        target_qubits: Physical qubit mapping (parameter-shift only).
        qasm_version: OpenQASM serialisation version (parameter-shift only).
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates
            to ``U`` during transpilation.

    Returns:
        ``QMLResult`` with loss history and accuracy.

    Raises:
        ValueError: If *gradient_method* is invalid or parameter-shift mode lacks *client*/*backend*.
    """
    method = gradient_method.lower()
    if method not in ("autograd", "parameter-shift"):
        raise ValueError(f"gradient_method must be 'autograd' or 'parameter-shift', got {method!r}")
    if method == "parameter-shift" and (client is None or backend is None):
        raise ValueError("parameter-shift requires client and backend")

    if method == "autograd":
        import torch
        from ..sim.statevector import (
            build_state_from_symbolic as _build_state,
            expectation_pauli as _expectation_pauli,
        )
        if seed is not None:
            torch.manual_seed(int(seed))

    if seed is not None:
        np.random.seed(int(seed))

    if measurement_qubits is None:
        if num_classes == 2:
            measurement_qubits = [0]
        else:
            measurement_qubits = list(range(min(num_classes, num_qubits)))

    # ---- Build symbolic encoding ----
    features_list = [np.asarray(d, dtype=np.float64) for d, _ in train_data]
    num_features = len(features_list[0])
    labels = [int(lab) for _, lab in train_data]
    enc_kwargs = dict(encoding_kwargs or {})

    # Test data (optional)
    has_test = test_data is not None and len(test_data) > 0
    if has_test:
        test_features_list = [np.asarray(d, dtype=np.float64) for d, _ in test_data]
        test_labels = [int(lab) for _, lab in test_data]
        n_test = len(test_data)
    else:
        test_features_list = []
        test_labels = []
        n_test = 0

    if callable(encoding) and not isinstance(encoding, str):
        enc_qc, enc_param_names = encoding(num_qubits, num_features, **enc_kwargs)
    elif encoding == "angle":
        enc_qc, enc_param_names = angle_encoding_circuit_symbolic(
            num_qubits, num_features, **enc_kwargs,
        )
    elif encoding == "iqp":
        enc_qc, enc_param_names = iqp_encoding_circuit_symbolic(
            num_qubits, num_features, **enc_kwargs,
        )
    else:
        raise ValueError(f"Unknown encoding: {encoding!r}")

    ansatz_qc, ansatz_param_names = _build_ansatz_symbolic(num_qubits, layers)
    num_ansatz_params = len(ansatz_param_names)
    params = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=num_ansatz_params)

    all_param_names = list(enc_param_names) + list(ansatz_param_names)

    # Pauli strings for measurement qubits
    z_observables = [_z_pauli_string(q, num_qubits) for q in measurement_qubits]
    z_hamiltonian = [(1.0, obs) for obs in z_observables]
    n_samples = len(train_data)

    # ---- Compose symbolic template and transpile ONCE ----
    full_symbolic_template = _compose_circuits(enc_qc, ansatz_qc)

    if method == "parameter-shift":
        transpiled_template = client._transpile_with_backend(
            full_symbolic_template, backend, target_qubits=target_qubits,
            use_dd=False, use_gate_compressor=False,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        target_qubits_in_use = client._ordered_target_qubits_from_layout(
            compiled_qc=transpiled_template,
            original_qc=full_symbolic_template,
            num_qubits=num_qubits,
        )
        logger.info("transpiled ONCE (unified template), target_qubits=%s", target_qubits_in_use)
    else:
        transpiled_template = full_symbolic_template
        target_qubits_in_use = None

    loss_history: List[float] = []
    test_loss_history: List[float] = []
    params_history: List[List[float]] = []
    best_loss = float("inf")
    best_params = params.copy()

    # Adam state
    m = np.zeros(num_ansatz_params, dtype=np.float64)
    v = np.zeros(num_ansatz_params, dtype=np.float64)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    lr = learning_rate

    # Shared backend kwargs (parameter-shift only, avoids repeating 6+ times)
    if method == "parameter-shift":
        backend_kwargs = dict(
            num_qubits=num_qubits, backend=backend, chip_name=chip_name,
            shots=shots, zne=zne, readout_mitigation=readout_mitigation,
            target_qubits=target_qubits_in_use, qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )

    logger.info(
        "start: %dq, %d ansatz params, %d encoding params, %d samples, %d iters, gradient=%s",
        num_qubits, num_ansatz_params, len(enc_param_names), n_samples, max_iters, method,
    )

    for it in range(max_iters):
        # ---- autograd path ----
        if method == "autograd":
            params_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
            total_loss = torch.zeros((), dtype=torch.float64)

            for si in range(n_samples):
                feat_t = torch.tensor(features_list[si], dtype=torch.float64)
                all_params_t = torch.cat([feat_t, params_t])
                state = _build_state(
                    full_symbolic_template,
                    params=all_params_t,
                    param_names=all_param_names,
                )

                label = labels[si]
                if num_classes == 2:
                    exp_val = _expectation_pauli(state, z_observables[0], num_qubits=num_qubits).real
                    prob_1 = (1.0 - exp_val) / 2.0
                    target = torch.tensor(float(label), dtype=torch.float64)
                    total_loss = total_loss - (
                        target * torch.log(prob_1 + 1e-10)
                        + (1.0 - target) * torch.log(1.0 - prob_1 + 1e-10)
                    )
                else:
                    logits = []
                    for obs in z_observables:
                        exp_val = _expectation_pauli(state, obs, num_qubits=num_qubits).real
                        logits.append(exp_val)
                    logits_t = torch.stack(logits)
                    log_probs = torch.log_softmax(logits_t, dim=0)
                    total_loss = total_loss - log_probs[label]

            avg_loss = total_loss / n_samples
            avg_loss.backward()
            loss_val = float(avg_loss.detach().cpu().item())
            grads = params_t.grad.detach().cpu().numpy().astype(float, copy=True)

        # ---- parameter-shift path ----
        else:
            # Forward: get ⟨Z_q⟩ for each sample
            sample_z = _get_z_backend(
                client, transpiled_template, all_param_names,
                features_list, params, z_observables, z_hamiltonian,
                backend_kwargs, name_prefix=f"qml_iter{it}",
            )

            # Loss + analytical dL/d⟨Z⟩ per sample
            loss_val, dl_dz_list = _batch_loss_and_grads(
                sample_z, labels, num_classes,
            )

            # Gradient via parameter-shift + chain rule (only over ansatz params)
            grads = np.zeros(num_ansatz_params, dtype=np.float64)
            for i in range(num_ansatz_params):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += shift
                params_minus[i] -= shift

                z_plus = _get_z_backend(
                    client, transpiled_template, all_param_names,
                    features_list, params_plus, z_observables, z_hamiltonian,
                    backend_kwargs, name_prefix=f"qml_iter{it}_g{i}p",
                )
                z_minus = _get_z_backend(
                    client, transpiled_template, all_param_names,
                    features_list, params_minus, z_observables, z_hamiltonian,
                    backend_kwargs, name_prefix=f"qml_iter{it}_g{i}m",
                )

                grad_i = 0.0
                for si in range(n_samples):
                    for qi in range(len(z_observables)):
                        dz = 0.5 * (z_plus[si][qi] - z_minus[si][qi])
                        grad_i += dl_dz_list[si][qi] * dz
                grads[i] = grad_i / n_samples

        # ---- Adam update (shared) ----
        params, m, v = adam_update(
            params, grads, m, v, it + 1,
            lr=lr, beta1=beta1, beta2=beta2, eps=adam_eps,
        )

        loss_history.append(loss_val)
        params_history.append(params.tolist())

        # ---- Test loss evaluation ----
        test_loss_val = None
        if has_test:
            if method == "autograd":
                test_z = _get_z_autograd(
                    _build_state, _expectation_pauli, full_symbolic_template,
                    all_param_names, test_features_list, params,
                    z_observables, num_qubits,
                )
            else:
                test_z = _get_z_backend(
                    client, transpiled_template, all_param_names,
                    test_features_list, params, z_observables, z_hamiltonian,
                    backend_kwargs, name_prefix=f"qml_iter{it}_test",
                )
            test_loss_val, _ = _batch_loss_and_grads(
                test_z, test_labels, num_classes,
            )
            test_loss_history.append(test_loss_val)

        # ---- Best model selection ----
        selection_loss = test_loss_val if test_loss_val is not None else loss_val
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_params = params.copy()

        if it % max(1, max_iters // 10) == 0:
            msg = f"[qml-classifier] iter {it} loss={loss_val:.6f}"
            if test_loss_val is not None:
                msg += f" test_loss={test_loss_val:.6f}"
            logger.info("%s", msg)
    if method == "autograd":
        train_z = _get_z_autograd(
            _build_state, _expectation_pauli, full_symbolic_template,
            all_param_names, features_list, best_params,
            z_observables, num_qubits,
        )
    else:
        train_z = _get_z_backend(
            client, transpiled_template, all_param_names,
            features_list, best_params, z_observables, z_hamiltonian,
            backend_kwargs, name_prefix="qml_eval",
        )
    train_preds = _predictions_from_z(train_z, num_classes)
    accuracy = sum(p == l for p, l in zip(train_preds, labels)) / n_samples

    # ---- Test accuracy evaluation ----
    test_accuracy = None
    if has_test:
        if method == "autograd":
            test_z_final = _get_z_autograd(
                _build_state, _expectation_pauli, full_symbolic_template,
                all_param_names, test_features_list, best_params,
                z_observables, num_qubits,
            )
        else:
            test_z_final = _get_z_backend(
                client, transpiled_template, all_param_names,
                test_features_list, best_params, z_observables, z_hamiltonian,
                backend_kwargs, name_prefix="qml_test_eval",
            )
        test_preds = _predictions_from_z(test_z_final, num_classes)
        test_accuracy = sum(p == l for p, l in zip(test_preds, test_labels)) / n_test

    msg = f"[qml-classifier] done. best_loss={best_loss:.6f} train_accuracy={accuracy:.4f}"
    if test_accuracy is not None:
        msg += f" test_accuracy={test_accuracy:.4f}"
    logger.info("%s", msg)

    return QMLResult(
        task="supervised",
        best_loss=best_loss,
        best_params=best_params.tolist(),
        loss_history=loss_history,
        params_history=params_history,
        accuracy=accuracy,
        test_loss_history=test_loss_history if test_loss_history else None,
        test_accuracy=test_accuracy,
    )


# ---------------------------------------------------------------------------
# 2. Unsupervised QNN — learn probability distributions from samples
# ---------------------------------------------------------------------------

def _mmd_rbf(samples_p: np.ndarray, samples_q: np.ndarray, sigma: float) -> float:
    """Compute MMD² with RBF kernel between two sets of binary vectors.

    Uses Hamming distance as the distance metric for the Gaussian kernel.

    Args:
        samples_p (*np.ndarray*): Binary sample array of shape ``(N, d)``.
        samples_q (*np.ndarray*): Binary sample array of shape ``(M, d)``.
        sigma (*float*): RBF kernel bandwidth.

    Returns:
        MMD² value as a float.
    """
    def _gram_mean(a: np.ndarray, b: np.ndarray) -> float:
        """Compute the mean RBF kernel value between two sample sets.

        Args:
            a (*np.ndarray*): First sample array of shape ``(Na, d)``.
            b (*np.ndarray*): Second sample array of shape ``(Nb, d)``.

        Returns:
            Mean RBF kernel value between *a* and *b*.
        """
        # Hamming distances → RBF kernel
        diff = a[:, None, :] ^ b[None, :, :]  # (Na, Nb, d)
        dist_sq = diff.sum(axis=-1).astype(np.float64)  # (Na, Nb)
        return float(np.exp(-dist_sq / (2.0 * sigma ** 2)).mean())

    return _gram_mean(samples_p, samples_p) - 2 * _gram_mean(samples_p, samples_q) + _gram_mean(samples_q, samples_q)


def _deduplicate_samples(samples: np.ndarray):
    """Deduplicate sample rows and compute normalized weights.

    Args:
        samples (*np.ndarray*): Binary measurement samples of shape ``(N, n_qubits)``.

    Returns:
        Tuple of ``(unique_samples, weights)`` where *unique_samples* has
        shape ``(K, n_qubits)`` and *weights* is a float64 array of length K
        summing to 1.
    """
    # Use structured view for np.unique on rows
    n = samples.shape[0]
    uniq, counts = np.unique(samples, axis=0, return_counts=True)
    weights = counts.astype(np.float64) / n
    return uniq, weights


def _simulate_samples(
    qc: QuantumCircuit,
    shots: int,
    param_values: dict,
    seed: int | None,
) -> np.ndarray:
    """Simulate and return measurement samples in big-endian bit order.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        shots (*int*): Number of measurement shots.
        param_values (*dict*): Parameter name → value mapping.
        seed (*int | None*): Random seed for reproducibility.

    Returns:
        Integer array of shape ``(shots, n_qubits)`` with entries 0/1.
    """
    from ..sim.statevector import simulate_counts as _sim_counts_sv
    from ..core.utils import get_samples as _get_samples
    counts = _sim_counts_sv(qc, shots, param_values=param_values, seed=seed)
    num_qubits = int(qc.nqubits)
    return _get_samples(counts, num_qubits)


def run_qnn_unsupervised(
    num_qubits: int,
    train_samples: np.ndarray,
    *,
    test_samples: Optional[np.ndarray] = None,
    ansatz: str = "hardware_efficient",
    layers: int = 2,
    max_iters: int = 100,
    learning_rate: float = 0.01,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    gradient_method: str = "autograd",
    # --- parameter-shift / hardware params ---
    client=None,
    backend=None,
    chip_name: str = "",
    shots: int = 4096,
    shift: float = np.pi / 2,
    zne: bool = False,
    readout_mitigation: bool = False,
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
    # --- MMD params (parameter-shift only) ---
    mmd_sigma: float = 1.0,
    # --- generation ---
    gen_shots: int = 1024,
) -> QBMResult:
    """Train a QNN to learn the probability distribution behind given samples.

    **autograd** path (simulator): Negative log-likelihood loss.
        Directly computes ``P(b|θ)`` via ``sample_probabilities`` and minimises
        ``-1/N Σ log P(bᵢ|θ)``.

    **parameter-shift** path (hardware): Maximum Mean Discrepancy (MMD) loss.
        Samples from the circuit on hardware and minimises MMD² between the
        training distribution and the generated distribution using an RBF kernel.

    Args:
        num_qubits: Number of qubits.
        train_samples: ``(N, num_qubits)`` integer array with entries 0/1,
            big-endian (column *i* = qubit *i*).
        test_samples: Optional ``(M, num_qubits)`` array for monitoring.
        ansatz: Ansatz type (only ``"hardware_efficient"`` supported).
        layers: Number of ansatz layers.
        max_iters: Training iterations.
        learning_rate: Adam learning rate.
        seed: Optional random seed.
        callback: ``(iter, loss)`` callback.
        gradient_method: ``"autograd"`` or ``"parameter-shift"``.
        client: ``QuantumHardwareClient`` instance (parameter-shift only).
        backend: Target backend (parameter-shift only).
        chip_name: Target chip identifier (parameter-shift only).
        shots: Measurement shots (parameter-shift only).
        shift: Parameter-shift magnitude (parameter-shift only).
        zne: Enable zero-noise extrapolation (parameter-shift only).
        readout_mitigation: Enable readout mitigation (parameter-shift only).
        target_qubits: Physical qubit mapping (parameter-shift only).
        qasm_version: OpenQASM serialisation version (parameter-shift only).
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates
            to ``U`` during transpilation.
        mmd_sigma: RBF kernel bandwidth for MMD (parameter-shift only).
        gen_shots: Number of shots for sample generation at the end.

    Returns:
        ``QBMResult`` with loss history and generated samples
        (``generated_samples`` is a ``List[List[int]]``).

    Raises:
        ValueError: If *gradient_method* is invalid or parameter-shift mode lacks *client*/*backend*.
    """
    method = gradient_method.lower()
    if method not in ("autograd", "parameter-shift"):
        raise ValueError(f"gradient_method must be 'autograd' or 'parameter-shift', got {method!r}")
    if method == "parameter-shift" and (client is None or backend is None):
        raise ValueError("parameter-shift requires client and backend")

    train_samples = np.asarray(train_samples, dtype=np.int64)
    n_train = train_samples.shape[0]

    if method == "autograd":
        import torch
        from ..sim.statevector import (
            build_state_from_symbolic as _build_state,
            sample_probabilities as _sample_probs_sv,
        )

    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(int(seed))
        if method == "autograd":
            torch.manual_seed(int(seed))

    # ---- Build ansatz (no encoding — just parameterized circuit) ----
    ansatz_qc, ansatz_param_names = _build_ansatz_symbolic(num_qubits, layers)
    num_params = len(ansatz_param_names)
    params = rng.uniform(-np.pi, np.pi, size=num_params)

    # Deduplicate training samples for NLL efficiency
    unique_train, train_weights = _deduplicate_samples(train_samples)
    n_unique_train = unique_train.shape[0]

    has_test = test_samples is not None and len(test_samples) > 0
    if has_test:
        test_samples = np.asarray(test_samples, dtype=np.int64)
        unique_test, test_weights = _deduplicate_samples(test_samples)

    # Transpile once for parameter-shift
    if method == "parameter-shift":
        transpiled_template = client._transpile_with_backend(
            ansatz_qc, backend, target_qubits=target_qubits,
            use_dd=False, use_gate_compressor=False,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        target_qubits_in_use = client._ordered_target_qubits_from_layout(
            compiled_qc=transpiled_template,
            original_qc=ansatz_qc,
            num_qubits=num_qubits,
        )
        backend_kwargs = dict(
            num_qubits=num_qubits, backend=backend, chip_name=chip_name,
            shots=shots, zne=zne, readout_mitigation=readout_mitigation,
            target_qubits=target_qubits_in_use, qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        logger.info("transpiled ONCE, target_qubits=%s", target_qubits_in_use)

        def _run_and_get_samples(qc_bound, name, n_shots=None):
            """Submit bound circuit and return (N, n_qubits) sample array.

            Args:
                qc_bound: Bound quantum circuit with all parameters resolved.
                name: Task name for the backend submission.
                n_shots: Override shot count. Defaults to ``None`` (use ``backend_kwargs``).

            Returns:
                ``np.ndarray`` of shape ``(N, n_qubits)`` with ``int64`` measurement samples.
            """
            kw = backend_kwargs if n_shots is None else {**backend_kwargs, "shots": n_shots}
            res = client._run_with_backend(
                qc_bound, name, observables=[], transpile=False, **kw,
            )
            return np.asarray(res.samples[0], dtype=np.int64)
    else:
        transpiled_template = ansatz_qc

    loss_history: List[float] = []
    test_loss_history: List[float] = []
    params_history: List[List[float]] = []
    best_loss = float("inf")
    best_params = params.copy()

    # Adam state
    m = np.zeros(num_params, dtype=np.float64)
    v = np.zeros(num_params, dtype=np.float64)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    lr = learning_rate

    logger.info(
        "start: %dq, %d params, %d train samples (%d unique), %d iters, gradient=%s, loss=%s",
        num_qubits, num_params, n_train, n_unique_train, max_iters, method,
        "NLL" if method == "autograd" else "MMD",
    )

    for it in range(max_iters):
        # ---- autograd: NLL loss ----
        if method == "autograd":
            params_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
            state = _build_state(
                ansatz_qc, params=params_t, param_names=ansatz_param_names,
            )
            probs = _sample_probs_sv(state, unique_train)
            weights_t = torch.tensor(train_weights, dtype=torch.float64, device=probs.device)
            nll = -(weights_t * torch.log(probs + 1e-10)).sum()
            nll.backward()
            loss_val = float(nll.detach().cpu().item())
            grads = params_t.grad.detach().cpu().numpy().astype(float, copy=True)

        # ---- parameter-shift: MMD loss ----
        else:
            # Forward: sample from current params
            qc_fwd = instantiate_transpiled_template(
                transpiled_template, ansatz_param_names, params,
            )
            gen_array = _run_and_get_samples(qc_fwd, f"qnn_iter{it}")
            loss_val = _mmd_rbf(train_samples, gen_array, sigma=mmd_sigma)

            # Gradient via parameter-shift on MMD
            grads = np.zeros(num_params, dtype=np.float64)
            for i in range(num_params):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += shift
                params_minus[i] -= shift

                qc_p = instantiate_transpiled_template(
                    transpiled_template, ansatz_param_names, params_plus,
                )
                qc_m = instantiate_transpiled_template(
                    transpiled_template, ansatz_param_names, params_minus,
                )
                gen_p = _run_and_get_samples(qc_p, f"qnn_iter{it}_p{i}")
                gen_m = _run_and_get_samples(qc_m, f"qnn_iter{it}_m{i}")
                mmd_p = _mmd_rbf(train_samples, gen_p, sigma=mmd_sigma)
                mmd_m = _mmd_rbf(train_samples, gen_m, sigma=mmd_sigma)
                grads[i] = (mmd_p - mmd_m) / (2.0 * shift)

        # ---- Adam update ----
        params, m, v = adam_update(
            params, grads, m, v, it + 1,
            lr=lr, beta1=beta1, beta2=beta2, eps=adam_eps,
        )

        loss_history.append(loss_val)
        params_history.append(params.tolist())

        # ---- Test loss ----
        test_loss_val = None
        if has_test:
            if method == "autograd":
                with torch.no_grad():
                    params_eval = torch.tensor(params, dtype=torch.float64)
                    state_eval = _build_state(
                        ansatz_qc, params=params_eval, param_names=ansatz_param_names,
                    )
                    probs_test = _sample_probs_sv(state_eval, unique_test)
                    test_weights_t = torch.tensor(test_weights, dtype=torch.float64, device=probs_test.device)
                    test_loss_val = float(-(test_weights_t * torch.log(probs_test + 1e-10)).sum().cpu().item())
            else:
                qc_test = instantiate_transpiled_template(
                    transpiled_template, ansatz_param_names, params,
                )
                gen_test = _run_and_get_samples(qc_test, f"qnn_iter{it}_test")
                test_loss_val = _mmd_rbf(test_samples, gen_test, sigma=mmd_sigma)
            test_loss_history.append(test_loss_val)

        # ---- Best model selection ----
        selection_loss = test_loss_val if test_loss_val is not None else loss_val
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_params = params.copy()

        if it % max(1, max_iters // 10) == 0:
            msg = f"[qnn-unsupervised] iter {it} loss={loss_val:.6f}"
            if test_loss_val is not None:
                msg += f" test_loss={test_loss_val:.6f}"
            logger.info("%s", msg)
        if callback is not None:
            callback(it, loss_val)

    # ---- Generate samples with best params ----
    if method == "autograd":
        param_values = {name: float(val) for name, val in zip(ansatz_param_names, best_params)}
        generated = _simulate_samples(ansatz_qc, gen_shots, param_values, seed)
    else:
        qc_gen = instantiate_transpiled_template(
            transpiled_template, ansatz_param_names, best_params,
        )
        generated = _run_and_get_samples(qc_gen, "qnn_gen", n_shots=gen_shots)

    generated_list = generated.tolist()
    logger.info("done. best_loss=%.6f, generated %d samples", best_loss, len(generated_list))

    return QBMResult(
        best_loss=best_loss,
        best_params=best_params.tolist(),
        loss_history=loss_history,
        test_loss_history=test_loss_history if test_loss_history else None,
        params_history=params_history,
        generated_samples=generated_list,
    )
