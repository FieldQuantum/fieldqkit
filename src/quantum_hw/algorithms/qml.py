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

from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ..circuit import QuantumCircuit
from ..core.types import QMLResult
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
    """Build a hardware-efficient ansatz and return (circuit, param_names)."""
    num_params = 2 * num_qubits * (layers + 1)
    param_names = [f"θ_{i}" for i in range(num_params)]
    qc = build_hardware_efficient_ansatz_symbolic(num_qubits, param_names, layers=layers)
    return qc, param_names


def _compose_circuits(front: QuantumCircuit, back: QuantumCircuit) -> QuantumCircuit:
    """Concatenate two circuits on the same register."""
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
    """Return a Pauli string with Z on qubit *q* and I elsewhere."""
    return "I" * q + "Z" + "I" * (num_qubits - q - 1)


def _classifier_loss_and_dl_dz(
    z_values: Sequence[float],
    label: int,
    num_classes: int,
) -> Tuple[float, np.ndarray]:
    """Compute classification loss and analytical dL/d⟨Z⟩.

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


def run_pqc_classifier(
    num_qubits: int,
    train_data: Sequence[Tuple[Sequence[float], int]],
    *,
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
) -> QMLResult:
    """Train a parameterized quantum classifier.

    Training samples are ``(features, label)`` pairs.  The encoding circuit is
    built symbolically so the full template (encoding + ansatz) is transpiled
    **once** and reused across all samples.

    Args:
        num_qubits: Number of qubits.
        train_data: List of ``(features, label)`` pairs.
        encoding: Encoding strategy — ``"angle"`` / ``"iqp"``, or a callable
            ``(num_qubits, num_features) -> (QuantumCircuit, param_names)``.
        encoding_kwargs: Extra keyword arguments forwarded to the encoding
            builder (e.g. ``{"gate": "rx"}``).
        num_classes: Number of classes (default 2).
        measurement_qubits: Indices of qubits to measure.
        layers: Number of ansatz layers.
        max_iters: Training iterations (full-batch).
        learning_rate: Adam learning rate.
        seed: Optional random seed.
        callback: ``(iter, loss)`` callback.
        gradient_method: ``"autograd"`` or ``"parameter-shift"``.
        client / backend / chip_name / shots / shift / zne /
        readout_mitigation / target_qubits / qasm_version:
            Hardware parameters (parameter-shift path only).

    Returns:
        ``QMLResult`` with loss history and accuracy.
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
        )
        target_qubits_in_use = client._ordered_target_qubits_from_layout(
            compiled_qc=transpiled_template,
            original_qc=full_symbolic_template,
            num_qubits=num_qubits,
        )
        print(f"[qml-classifier] transpiled ONCE (unified template), "
              f"target_qubits={target_qubits_in_use}")
    else:
        transpiled_template = full_symbolic_template
        target_qubits_in_use = None

    loss_history: List[float] = []
    params_history: List[List[float]] = []
    best_loss = float("inf")
    best_params = params.copy()

    # Adam state
    m = np.zeros(num_ansatz_params, dtype=np.float64)
    v = np.zeros(num_ansatz_params, dtype=np.float64)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    lr = learning_rate

    print(f"[qml-classifier] start: {num_qubits}q, {num_ansatz_params} ansatz params, "
          f"{len(enc_param_names)} encoding params, "
          f"{n_samples} samples, {max_iters} iters, gradient={method}")

    def _make_all_values(sample_idx: int, theta: np.ndarray) -> np.ndarray:
        """Concatenate encoding feature values + ansatz params."""
        return np.concatenate([features_list[sample_idx], theta])

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
            sample_z: List[List[float]] = []
            for si in range(n_samples):
                all_vals = _make_all_values(si, params)
                qc_bound = instantiate_transpiled_template(
                    transpiled_template, all_param_names, all_vals,
                )
                _, exps = evaluate_energy_with_backend(
                    client, qc_bound,
                    name=f"qml_iter{it}_s{si}",
                    num_qubits=num_qubits, backend=backend, chip_name=chip_name,
                    shots=shots, hamiltonian=z_hamiltonian,
                    zne=zne, readout_mitigation=readout_mitigation,
                    target_qubits=target_qubits_in_use,
                    qasm_version=qasm_version,
                )
                sample_z.append([exps.get(obs, 0.0) for obs in z_observables])

            # Loss + analytical dL/d⟨Z⟩ per sample
            total_loss_val = 0.0
            dl_dz_list: List[np.ndarray] = []
            for si in range(n_samples):
                l, dl = _classifier_loss_and_dl_dz(sample_z[si], labels[si], num_classes)
                total_loss_val += l
                dl_dz_list.append(dl)
            loss_val = total_loss_val / n_samples

            # Gradient via parameter-shift + chain rule (only over ansatz params)
            grads = np.zeros(num_ansatz_params, dtype=np.float64)
            for i in range(num_ansatz_params):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += shift
                params_minus[i] -= shift

                grad_i = 0.0
                for si in range(n_samples):
                    all_vals_p = _make_all_values(si, params_plus)
                    all_vals_m = _make_all_values(si, params_minus)
                    qc_p = instantiate_transpiled_template(
                        transpiled_template, all_param_names, all_vals_p,
                    )
                    qc_m = instantiate_transpiled_template(
                        transpiled_template, all_param_names, all_vals_m,
                    )
                    _, exps_p = evaluate_energy_with_backend(
                        client, qc_p,
                        name=f"qml_iter{it}_g{i}p_s{si}",
                        num_qubits=num_qubits, backend=backend, chip_name=chip_name,
                        shots=shots, hamiltonian=z_hamiltonian,
                        zne=zne, readout_mitigation=readout_mitigation,
                        target_qubits=target_qubits_in_use,
                        qasm_version=qasm_version,
                    )
                    _, exps_m = evaluate_energy_with_backend(
                        client, qc_m,
                        name=f"qml_iter{it}_g{i}m_s{si}",
                        num_qubits=num_qubits, backend=backend, chip_name=chip_name,
                        shots=shots, hamiltonian=z_hamiltonian,
                        zne=zne, readout_mitigation=readout_mitigation,
                        target_qubits=target_qubits_in_use,
                        qasm_version=qasm_version,
                    )
                    for qi, obs in enumerate(z_observables):
                        dz = 0.5 * (exps_p.get(obs, 0.0) - exps_m.get(obs, 0.0))
                        grad_i += dl_dz_list[si][qi] * dz
                grads[i] = grad_i / n_samples

        # ---- Adam update (shared) ----
        params, m, v = adam_update(
            params, grads, m, v, it + 1,
            lr=lr, beta1=beta1, beta2=beta2, eps=adam_eps,
        )

        loss_history.append(loss_val)
        params_history.append(params.tolist())
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params.copy()

        if it % max(1, max_iters // 10) == 0:
            print(f"[qml-classifier] iter {it} loss={loss_val:.6f}")
        if callback is not None:
            callback(it, loss_val)

    # ---- Accuracy evaluation ----
    correct = 0
    if method == "autograd":
        params_t = torch.tensor(best_params, dtype=torch.float64)
        for si in range(n_samples):
            with torch.no_grad():
                feat_t = torch.tensor(features_list[si], dtype=torch.float64)
                all_params_t = torch.cat([feat_t, params_t])
                state = _build_state(
                    full_symbolic_template,
                    params=all_params_t,
                    param_names=all_param_names,
                )
                if num_classes == 2:
                    exp_val = _expectation_pauli(state, z_observables[0], num_qubits=num_qubits).real
                    pred = 0 if float(exp_val) > 0 else 1
                else:
                    logits = [
                        float(_expectation_pauli(state, obs, num_qubits=num_qubits).real)
                        for obs in z_observables
                    ]
                    pred = int(np.argmax(logits))
                if pred == labels[si]:
                    correct += 1
    else:
        for si in range(n_samples):
            all_vals = _make_all_values(si, best_params)
            qc_bound = instantiate_transpiled_template(
                transpiled_template, all_param_names, all_vals,
            )
            _, exps = evaluate_energy_with_backend(
                client, qc_bound,
                name=f"qml_eval_s{si}",
                num_qubits=num_qubits, backend=backend, chip_name=chip_name,
                shots=shots, hamiltonian=z_hamiltonian,
                zne=zne, readout_mitigation=readout_mitigation,
                target_qubits=target_qubits_in_use,
                qasm_version=qasm_version,
            )
            if num_classes == 2:
                pred = 0 if exps.get(z_observables[0], 0.0) > 0 else 1
            else:
                logits = [exps.get(obs, 0.0) for obs in z_observables]
                pred = int(np.argmax(logits))
            if pred == labels[si]:
                correct += 1

    accuracy = correct / n_samples
    print(f"[qml-classifier] done. best_loss={best_loss:.6f} accuracy={accuracy:.4f}")

    return QMLResult(
        task="supervised",
        best_loss=best_loss,
        best_params=best_params.tolist(),
        loss_history=loss_history,
        params_history=params_history,
        accuracy=accuracy,
    )
