"""Shared optimization utilities for VQE, QAOA, and other variational algorithms.

Provides:
- Hamiltonian types and observable evaluation
- Parameter-shift gradient computation on hardware backends
- Clifford fitting (affine noise correction)
- Adam optimizer step
- Parameterized circuit instantiation helpers
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from ..api.backend import Backend
from ..circuit import QuantumCircuit
from ..compile.optimize import GateCompressor

Hamiltonian = List[Tuple[float, str]]
CliffordFitMap = Dict[str, Tuple[float, float]]


# ---------------------------------------------------------------------------
# Observable & energy helpers
# ---------------------------------------------------------------------------

def normalize_observable_values(values):
    """Flatten list-of-dict or single-element list into ``Dict[str, float]``.

    Args:
        values: Raw observable values returned by a backend.  May be a list of
            dicts, a single-element list wrapping a scalar, or already a dict.

    Returns:
        A merged ``{observable: value}`` dictionary, a bare scalar if
        *values* was a single-element non-dict list, or *values* unchanged.
    """
    if isinstance(values, list) and values:
        if isinstance(values[0], dict):
            merged: Dict[str, float] = {}
            for item in values:
                merged.update(item)
            return merged
        if len(values) == 1:
            return values[0]
    return values


def ensure_observable_map(observables: Sequence[str], values) -> Dict[str, float]:
    """Guarantee a ``{observable: value}`` dict from various result shapes.

    Args:
        observables: Ordered Pauli-string labels expected from the backend.
        values: Raw backend result (dict, list of dicts, or scalar).

    Returns:
        A ``Dict[str, float]`` mapping each observable to its expectation.

    Raises:
        RuntimeError: If *values* shape cannot be reconciled with *observables*.
    """
    if not observables:
        return {}
    values = normalize_observable_values(values)
    if isinstance(values, dict):
        return {k: float(v) for k, v in values.items()}
    if len(observables) == 1:
        return {observables[0]: float(values)}
    raise RuntimeError("observable_values shape mismatch")


def energy_from_expectations(hamiltonian: Hamiltonian, expectations: Dict[str, float]) -> float:
    """Compute energy expectation ⟨H⟩ = Σ coeff_i · ⟨O_i⟩.

    Args:
        hamiltonian: List of ``(coefficient, pauli_string)`` terms.
        expectations: Per-observable expectation values.

    Returns:
        Scalar energy as a float.
    """
    return float(sum(coeff * expectations.get(obs, 0.0) for coeff, obs in hamiltonian))


# ---------------------------------------------------------------------------
# Template instantiation
# ---------------------------------------------------------------------------

def instantiate_transpiled_template(
    transpiled_template: QuantumCircuit,
    param_names: Sequence[str],
    params: np.ndarray,
) -> QuantumCircuit:
    """Clone a transpiled template and bind numeric parameter values.

    Args:
        transpiled_template: A pre-transpiled circuit with symbolic parameters.
        param_names: Ordered list of parameter symbol names.
        params: Numeric values for each parameter.

    Returns:
        A deep-copied circuit with all symbols resolved to concrete values.
    """
    qc = transpiled_template.deepcopy()
    values = {name: float(params[i]) for i, name in enumerate(param_names)}
    qc.apply_value(values, deep=True)
    return qc


# ---------------------------------------------------------------------------
# Energy evaluation on hardware
# ---------------------------------------------------------------------------

def evaluate_energy_with_backend(
    client,
    qc: QuantumCircuit,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    shots: int,
    hamiltonian: Hamiltonian,
    zne: bool,
    readout_mitigation: bool,
    clifford_fit_map: Optional[CliffordFitMap] = None,
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Run a bound circuit on a backend and compute the Hamiltonian energy.

    Args:
        client: ``QuantumHardwareClient`` instance.
        qc: Bound (all parameters resolved) quantum circuit.
        name: Task name sent to the backend.
        num_qubits: Number of logical qubits.
        backend: ``Backend`` handle (Simulator or hardware).
        chip_name: Chip identifier string.
        shots: Number of measurement shots.
        hamiltonian: Cost Hamiltonian terms ``(coeff, pauli_str)``.
        zne: Enable zero-noise extrapolation.
        readout_mitigation: Enable readout error mitigation.
        clifford_fit_map: Optional per-observable affine correction map.
        target_qubits: Physical qubit mapping, if any.
        qasm_version: OpenQASM version for circuit serialisation.
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates
            to ``U`` during transpilation.

    Returns:
        ``(energy, expectations)`` where *energy* is ⟨H⟩ and *expectations*
        maps each observable string to its measured expectation value.
    """
    observables = [term[1] for term in hamiltonian]

    result = client._run_with_backend(
        qc,
        name,
        num_qubits,
        backend=backend,
        chip_name=chip_name,
        shots=shots,
        zne=zne,
        readout_mitigation=readout_mitigation,
        observables=observables,
        return_probabilities=False,
        print_true=False,
        transpile=False,
        target_qubits=target_qubits,
        qasm_version=qasm_version,
        convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
    )
    expectations_raw = ensure_observable_map(observables, result.observable_values)
    expectations = apply_clifford_fit(expectations_raw, clifford_fit_map)
    energy = energy_from_expectations(hamiltonian, expectations)
    return energy, expectations


# ---------------------------------------------------------------------------
# Clifford fitting
# ---------------------------------------------------------------------------

# 24 single-qubit Clifford elements in U3(theta, phi, lambda) parameterization.
_CLIFFORD_U3_PARAMS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (np.pi, 0.0, np.pi),
    (np.pi, 0.0, 0.0),
    (np.pi, np.pi / 2.0, np.pi / 2.0),
    (np.pi, np.pi / 2.0, -np.pi / 2.0),
    (np.pi, -np.pi / 2.0, np.pi / 2.0),
    (np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0),
    (np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0),
    (np.pi / 2.0, np.pi / 2.0, np.pi / 2.0),
    (np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0),
    (np.pi / 2.0, 0.0, 0.0),
    (np.pi / 2.0, np.pi, 0.0),
    (np.pi / 2.0, 0.0, np.pi),
    (np.pi / 2.0, np.pi, np.pi),
    (-np.pi / 2.0, 0.0, 0.0),
    (-np.pi / 2.0, np.pi, 0.0),
    (-np.pi / 2.0, 0.0, np.pi),
    (-np.pi / 2.0, np.pi, np.pi),
    (0.0, np.pi / 2.0, np.pi / 2.0),
    (0.0, -np.pi / 2.0, -np.pi / 2.0),
    (0.0, np.pi / 2.0, -np.pi / 2.0),
    (0.0, -np.pi / 2.0, np.pi / 2.0),
    (0.0, np.pi, 0.0),
    (0.0, 0.0, np.pi),
)


def build_single_qubit_rotation_gate_list(transpiled_template: QuantumCircuit) -> List[Tuple[int, str]]:
    """Collect ``(gate_index, gate_name)`` for every single-qubit rotation gate.

    Used as the set of randomization sites for Clifford fitting.  Returns
    every ``u``/``rx``/``ry``/``rz`` occurrence, regardless of whether
    its parameters are concrete or still symbolic.

    Args:
        transpiled_template: A transpiled circuit.

    Returns:
        List of ``(gate_index, gate_name)`` tuples.
    """
    one_qubit_param_gates = {"u", "rx", "ry", "rz"}
    out: List[Tuple[int, str]] = []
    for gate_index, gate_info in enumerate(transpiled_template.gates):
        gate = str(gate_info[0]).lower()
        if gate in one_qubit_param_gates:
            out.append((gate_index, gate))
    return out


def randomize_single_qubit_gates_to_clifford(
    transpiled_template: QuantumCircuit,
    rng: np.random.Generator,
    single_qubit_gates: Sequence[Tuple[int, str]],
    num_non_clifford_gates: int = 0,
) -> Tuple[QuantumCircuit, Tuple[tuple, ...]]:
    """Replace parameterized 1-qubit gates with random Clifford (or Haar) unitaries.

    Used in Clifford fitting to generate calibration circuits whose ideal
    expectation values can be computed efficiently.

    Args:
        transpiled_template: Template circuit to randomize.
        rng: NumPy random generator.
        single_qubit_gates: Output of :func:`build_single_qubit_rotation_gate_list`.
        num_non_clifford_gates: Number of gates to replace with Haar-random
            unitaries instead of Cliffords (for mixed calibration).

    Returns:
        ``(circuit, signature)`` — the modified circuit and a hashable
        tuple identifying the chosen gate parameters.
    """
    qc = transpiled_template.deepcopy()
    signature: List[tuple] = []
    if not single_qubit_gates:
        raise ValueError("clifford fitting requires at least one single-qubit rotation gate")

    n_total = len(single_qubit_gates)
    n_non_clifford = int(max(0, min(int(num_non_clifford_gates), n_total)))
    non_clifford_indices = set()
    if n_non_clifford > 0:
        chosen = rng.choice(np.arange(n_total), size=n_non_clifford, replace=False)
        non_clifford_indices = {int(i) for i in np.asarray(chosen).tolist()}

    new_gates = list(qc.gates)
    for local_idx, (gate_index, gate) in enumerate(single_qubit_gates):
        gate_info = new_gates[gate_index]
        qubit = int(gate_info[-1])

        if local_idx in non_clifford_indices:
            u = float(rng.uniform(0.0, 1.0))
            v = float(rng.uniform(0.0, 1.0))
            w = float(rng.uniform(0.0, 1.0))
            theta = float(2.0 * np.arcsin(np.sqrt(u)))
            phi = float(2.0 * np.pi * v)
            lam = float(2.0 * np.pi * w)
            kind = "u_haar"
        else:
            theta, phi, lam = _CLIFFORD_U3_PARAMS[int(rng.integers(0, len(_CLIFFORD_U3_PARAMS)))]
            theta = float(theta)
            phi = float(phi)
            lam = float(lam)
            kind = "u_clifford"

        new_gates[gate_index] = ("u", theta, phi, lam, qubit)
        signature.append((gate_index, gate, kind, theta, phi, lam, qubit))

    qc.gates = new_gates
    return qc, tuple(signature)


def sample_unique_randomized_clifford_circuits(
    transpiled_template: QuantumCircuit,
    *,
    rng: np.random.Generator,
    num_samples: int,
    single_qubit_gates: Sequence[Tuple[int, str]],
    num_non_clifford_gates: int = 0,
) -> List[QuantumCircuit]:
    """Sample approximately unique Clifford-randomized calibration circuits.

    Repeatedly calls :func:`randomize_single_qubit_gates_to_clifford` and
    deduplicates by gate-parameter signature.

    Args:
        transpiled_template: Template circuit.
        rng: NumPy random generator.
        num_samples: Desired number of unique calibration circuits.
        single_qubit_gates: Parameterized gate indices from
            :func:`build_single_qubit_rotation_gate_list`.
        num_non_clifford_gates: Haar-random gates per circuit.

    Returns:
        List of unique randomized circuits (may be fewer than *num_samples*
        if the gate space is small).
    """
    target = max(int(num_samples), 0)
    if target == 0:
        return []
    sampled: List[QuantumCircuit] = []
    seen = set()
    attempts = 0
    max_attempts = max(target * 20, 100)
    while len(sampled) < target and attempts < max_attempts:
        attempts += 1
        qc, signature = randomize_single_qubit_gates_to_clifford(
            transpiled_template,
            rng,
            single_qubit_gates,
            num_non_clifford_gates=int(num_non_clifford_gates),
        )
        if signature in seen:
            continue
        seen.add(signature)
        sampled.append(qc)
    return sampled


def fit_linear_clifford_map(noisy_values: Sequence[float], ideal_values: Sequence[float]) -> Tuple[float, float]:
    """Fit affine correction ``ideal ≈ a · noisy + b`` from calibration data.

    Args:
        noisy_values: Expectation values measured on hardware.
        ideal_values: Corresponding expectation values from noise-free
            simulation.

    Returns:
        ``(a, b)`` coefficients.  Falls back to ``(1.0, mean_shift)``
        when the data variance is too small for a reliable fit.
    """
    x = np.asarray(noisy_values, dtype=float)
    y = np.asarray(ideal_values, dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 1.0, 0.0
    if float(np.std(x)) <= 0.05 or float(np.std(y)) <= 0.05:
        a, b = 1.0, float(np.mean(y) - np.mean(x))
    else:
        try:
            a, b = np.polyfit(x, y, 1)
            if not (np.isfinite(a) and np.isfinite(b)):
                a, b = 1.0, float(np.mean(y) - np.mean(x))
        except (np.linalg.LinAlgError, ValueError):
            a, b = 1.0, float(np.mean(y) - np.mean(x))
    return float(a), float(b)


def apply_clifford_fit(expectations: Dict[str, float], fit_map: Optional[CliffordFitMap]) -> Dict[str, float]:
    """Apply per-observable affine correction to raw expectations.

    Args:
        expectations: Raw ``{observable: value}`` from the backend.
        fit_map: Mapping ``{observable: (a, b)}`` from
            :func:`fit_linear_clifford_map`.  If ``None``, returns
            *expectations* unchanged.

    Returns:
        Corrected expectations, clipped to ``[-1, 1]``.
    """
    if not fit_map:
        return expectations
    corrected = dict(expectations)
    for obs, (a, b) in fit_map.items():
        if obs in corrected:
            corrected[obs] = float(np.clip(a * corrected[obs] + b, -1.0, 1.0))
    return corrected


def _ideal_expectations_clifford_aware(
    client,
    ideal_qc: QuantumCircuit,
    *,
    observables: Sequence[str],
    num_qubits: int,
    target_qubits: Optional[Sequence[int]],
) -> Optional[Dict[str, float]]:
    """Compute ideal expectations via the scalable Heisenberg-picture path.

    Args:
        client: ``QuantumHardwareClient`` instance.
        ideal_qc: Bound (parameter-free) calibration circuit on physical qubits.
        observables: Pauli-string labels to evaluate.
        num_qubits: Number of logical qubits (length of each Pauli string).
        target_qubits: Logical→physical qubit mapping.  ``None`` means the
            circuit already uses dense logical indices.

    Returns:
        ``{observable: expectation}`` dict, or ``None`` if the scalable
        path is not applicable (unsupported gate or branch-count
        explosion); callers should then fall back to statevector.
    """
    from ..sim.clifford import CliffordError, simulate_clifford_expectations
    from ..sim.clifford_t import simulate_clifford_t_expectations

    qc_sim, _mapping = client._compact_for_sim(ideal_qc, target_qubits=target_qubits)
    try:
        return simulate_clifford_expectations(qc_sim, observables, num_qubits=int(num_qubits))
    except CliffordError:
        pass
    try:
        return simulate_clifford_t_expectations(qc_sim, observables, num_qubits=int(num_qubits))
    except (CliffordError, RuntimeError):
        return None


def build_clifford_fit_map(
    client,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    observables: Sequence[str],
    shots: int,
    zne: bool,
    readout_mitigation: bool,
    transpiled_template: QuantumCircuit,
    num_samples: int,
    num_non_clifford_gates: int,
    seed: Optional[int],
    target_qubits: Optional[Sequence[int]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
) -> CliffordFitMap:
    """Build per-observable affine correction map via Clifford calibration.

    Generates *num_samples* Clifford-randomized circuits, runs each on
    both the noisy *backend* and a noise-free simulator, and fits an
    affine model per observable.  Every single-qubit rotation in
    *transpiled_template* is treated as a randomization site (regardless
    of whether its parameters are symbolic or concrete).

    Args:
        client: ``QuantumHardwareClient`` instance.
        name: Task-name prefix for submitted jobs.
        num_qubits: Number of logical qubits.
        backend: Target hardware backend.
        chip_name: Hardware chip identifier.
        observables: Pauli-string labels to calibrate.
        shots: Shots per calibration circuit.
        zne: Enable ZNE during calibration (should match the main run).
        readout_mitigation: Enable readout mitigation during calibration
            (should match the main run).
        transpiled_template: Pre-compiled circuit template whose
            single-qubit rotations will be randomized.
        num_samples: Number of calibration circuits to generate.
        num_non_clifford_gates: Per-circuit count of single-qubit gates
            replaced with Haar-random unitaries instead of random Cliffords.
        seed: Optional RNG seed for reproducibility.
        target_qubits: Physical qubit mapping.
        qasm_version: OpenQASM version.
        convert_single_qubit_gate_to_u: Whether to convert single-qubit
            gates to ``U`` during downstream transpilation passes.

    Returns:
        ``CliffordFitMap`` — ``{observable: (a, b)}`` affine coefficients.
        Returns ``{}`` when there is nothing to do (``num_samples <= 0``
        or empty *observables*), and an identity-like map when
        *transpiled_template* has no rotation sites to randomize.
    """
    if num_samples <= 0 or not observables:
        return {}
    observables = list(dict.fromkeys(observables))
    single_qubit_gates = build_single_qubit_rotation_gate_list(transpiled_template)
    if not single_qubit_gates:
        return {obs: (1.0, 0.0) for obs in observables}

    fake_hamiltonian: Hamiltonian = [(1.0, obs) for obs in observables]
    fit_inputs_noisy: Dict[str, List[float]] = {obs: [] for obs in observables}
    fit_inputs_ideal: Dict[str, List[float]] = {obs: [] for obs in observables}

    rng = np.random.default_rng(seed)
    sim_backend = Backend("Simulator")

    sampled_clifford_circuits = sample_unique_randomized_clifford_circuits(
        transpiled_template,
        rng=rng,
        num_samples=num_samples,
        single_qubit_gates=single_qubit_gates,
        num_non_clifford_gates=int(num_non_clifford_gates),
    )
    for si, clifford_qc in enumerate(sampled_clifford_circuits):
        noisy_qc = clifford_qc.deepcopy()
        ideal_qc = clifford_qc.deepcopy()

        _, noisy_expectations = evaluate_energy_with_backend(
            client,
            noisy_qc,
            name=f"{name}_clifford_noisy_{si}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=fake_hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )

        # Prefer the scalable Heisenberg-picture simulator for the ideal
        # branch.  It is exact (no shot noise) and scales as O(g · n) for
        # pure-Clifford circuits, O(4^k · g · n) when k non-Clifford
        # rotations are present (k = num_non_clifford_gates).  Falls back
        # to the dense statevector path on unsupported gates.
        ideal_expectations = _ideal_expectations_clifford_aware(
            client,
            ideal_qc,
            observables=observables,
            num_qubits=num_qubits,
            target_qubits=target_qubits,
        )
        if ideal_expectations is None:
            _, ideal_expectations = evaluate_energy_with_backend(
                client,
                ideal_qc,
                name=f"{name}_clifford_ideal_{si}",
                num_qubits=num_qubits,
                backend=sim_backend,
                chip_name="Simulator",
                shots=shots * 10,
                hamiltonian=fake_hamiltonian,
                zne=False,
                readout_mitigation=False,
                target_qubits=target_qubits,
                qasm_version=qasm_version,
            )

        for obs in observables:
            fit_inputs_noisy[obs].append(float(noisy_expectations.get(obs, 0.0)))
            fit_inputs_ideal[obs].append(float(ideal_expectations.get(obs, 0.0)))

    fit_map: CliffordFitMap = {}
    for obs in observables:
        fit_map[obs] = fit_linear_clifford_map(fit_inputs_noisy[obs], fit_inputs_ideal[obs])
    return fit_map


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def parameter_shift_gradient(
    client,
    params: np.ndarray,
    *,
    name: str,
    num_qubits: int,
    backend: Backend,
    chip_name: str,
    shots: int,
    hamiltonian: Hamiltonian,
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    param_template: Optional[QuantumCircuit] = None,
    param_names: Optional[Sequence[str]] = None,
    clifford_fit_map: Optional[CliffordFitMap] = None,
    target_qubits: Optional[Sequence[int]] = None,
    circuit_transform: Optional[Callable[[QuantumCircuit, Optional[int]], QuantumCircuit]] = None,
    qasm_version: str = "2.0",
    convert_single_qubit_gate_to_u: bool = True,
) -> np.ndarray:
    """Compute gradients via the parameter-shift rule on hardware.

    For each parameter *i*, evaluates the energy at ``params[i] ± shift``
    and returns ``grad[i] = 0.5 * (E⁺ - E⁻)``.

    Args:
        client: ``QuantumHardwareClient`` instance.
        params: Current parameter vector.
        name: Task-name prefix.
        num_qubits: Number of logical qubits.
        backend: Target backend.
        chip_name: Chip identifier.
        shots: Shots per evaluation.
        hamiltonian: Cost Hamiltonian.
        shift: Parameter shift (typically ``π/2``).
        zne: Enable ZNE.
        readout_mitigation: Enable readout mitigation.
        param_template: Pre-transpiled symbolic circuit template.
        param_names: Ordered parameter names.
        clifford_fit_map: Optional affine correction map.
        target_qubits: Physical qubit mapping.
        circuit_transform: Optional post-instantiation circuit transform
            (e.g. compression), called as ``transform(qc, param_index)``.
        qasm_version: OpenQASM version.
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates to U during transpilation.

    Returns:
        Gradient vector with the same shape as *params*.

    Raises:
        ValueError: If *param_template* or *param_names* is ``None``.
    """
    if param_template is None or param_names is None:
        raise ValueError("parameter_shift_gradient requires param_template and param_names")

    _gate_compressor = GateCompressor(convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u)
    grads = np.zeros_like(params, dtype=float)
    for i in range(params.size):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        qc_plus = instantiate_transpiled_template(param_template, param_names, params_plus)
        qc_minus = instantiate_transpiled_template(param_template, param_names, params_minus)
        if circuit_transform is not None:
            qc_plus = circuit_transform(qc_plus, i)
            qc_minus = circuit_transform(qc_minus, i)
        qc_plus = _gate_compressor.run(qc_plus)
        qc_minus = _gate_compressor.run(qc_minus)
        e_plus, _ = evaluate_energy_with_backend(
            client,
            qc_plus,
            name=f"{name}_grad_p{i}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            clifford_fit_map=clifford_fit_map,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        e_minus, _ = evaluate_energy_with_backend(
            client,
            qc_minus,
            name=f"{name}_grad_m{i}",
            num_qubits=num_qubits,
            backend=backend,
            chip_name=chip_name,
            shots=shots,
            hamiltonian=hamiltonian,
            zne=zne,
            readout_mitigation=readout_mitigation,
            clifford_fit_map=clifford_fit_map,
            target_qubits=target_qubits,
            qasm_version=qasm_version,
            convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
        )
        grads[i] = 0.5 * (e_plus - e_minus)
    return grads


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

def adam_update(
    params: np.ndarray,
    grads: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one step of the Adam optimizer.

    Args:
        params: Current parameter vector.
        grads: Gradient vector.
        m: First-moment estimate (updated in-place semantics via return).
        v: Second-moment estimate.
        t: 1-based iteration counter (used for bias correction).
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Small constant for numerical stability.

    Returns:
        ``(updated_params, m, v)`` — new parameters and updated moments.
    """
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * (grads ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v


# ---------------------------------------------------------------------------
# Generic variational optimization loop
# ---------------------------------------------------------------------------

def run_variational_loop(
    client,
    *,
    tag: str,
    name: str,
    num_qubits: int,
    param_names: List[str],
    symbolic_qc: QuantumCircuit,
    hamiltonian: Hamiltonian,
    params: np.ndarray,
    backend: Backend,
    chip_name: str,
    shots: int,
    max_iters: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    shift: float,
    zne: bool,
    readout_mitigation: bool,
    gradient_method: str,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    transpiled_template: Optional[QuantumCircuit] = None,
    gradient_param_template: Optional[QuantumCircuit] = None,
    target_qubits: Optional[Sequence[int]] = None,
    clifford_fit_map: Optional[CliffordFitMap] = None,
    circuit_transform: Optional[Callable] = None,
    qasm_version: str = "2.0",
    extra_info: str = "",
    convert_single_qubit_gate_to_u: bool = True,
    device: "torch.device | str | None" = None,
) -> dict:
    """Core variational optimization loop shared by VQE, QAOA, etc.

    Runs *max_iters* iterations of Adam optimization with either
    ``autograd`` (torch, Simulator-only) or ``parameter-shift`` (hardware)
    gradient computation.

    Args:
        client: ``QuantumHardwareClient`` instance.
        tag: Log prefix (e.g. ``"vqe"`` or ``"qaoa"``).
        name: Task-name prefix for submitted jobs.
        num_qubits: Number of logical qubits.
        param_names: Ordered parameter symbol names.
        symbolic_qc: Symbolic ansatz circuit (used by autograd path).
        hamiltonian: Cost Hamiltonian terms.
        params: Initial parameter vector.
        backend: Target backend.
        chip_name: Target chip identifier.
        shots: Shots per energy/gradient evaluation.
        max_iters: Maximum optimization iterations.
        learning_rate: Adam learning rate.
        beta1: Adam first-moment decay.
        beta2: Adam second-moment decay.
        eps: Adam numerical-stability epsilon.
        shift: Parameter-shift magnitude.
        zne: Enable zero-noise extrapolation.
        readout_mitigation: Enable readout mitigation.
        gradient_method: ``"autograd"`` or ``"parameter-shift"``
            (must be pre-validated by caller).
        seed: Optional torch manual seed (autograd path only).
        callback: Per-iteration callback ``(iter, cost, params)``.
        transpiled_template: Pre-transpiled circuit template
            (parameter-shift path).
        gradient_param_template: Template for gradient computation;
            defaults to *transpiled_template* if ``None``.
        target_qubits: Physical qubit mapping.
        clifford_fit_map: Optional affine correction map.
        circuit_transform: Optional post-instantiation circuit transform
            (e.g. compression); called as ``transform(qc, param_index)``
            where *param_index* is ``None`` for the base energy evaluation
            or an ``int`` for parameter-shift evaluations.
        qasm_version: OpenQASM version.
        extra_info: Additional info for the start log message.
        convert_single_qubit_gate_to_u: Whether to convert single-qubit gates
            to ``U`` during transpilation.
    Returns:
        Dict with ``best_cost``, ``best_params``, ``cost_history``,
        ``params_history``, ``grad_history``, ``last_expectations``.
    """
    method = gradient_method
    if gradient_param_template is None:
        gradient_param_template = transpiled_template

    if method == "autograd" and str(chip_name).lower() == "simulator":
        import torch
        from ..sim import energy_and_expectations as _energy_and_expectations
        if seed is not None:
            torch.manual_seed(int(seed))

    _cloud_qasm_template: Optional[str] = None
    _cloud_platform = None
    _cloud_hamiltonian: Optional[List[Dict]] = None
    if method == "autograd" and str(chip_name).lower() == "fieldquantum_sim":
        _cloud_qasm_template = symbolic_qc.to_openqasm2(symbolic=True)
        _cloud_platform = client._active_resolved_backend.metadata["platform_obj"]
        _cloud_hamiltonian = [
            {"coeff": float(c), "pauli": str(p)} for c, p in hamiltonian
        ]
        logger.info(
            "[%s] cloud autograd via FieldQuantum server @ %s",
            tag, _cloud_platform.base_url,
        )

    cost_history: List[float] = []
    params_history: List[List[float]] = []
    grad_history: List[List[float]] = []
    best_cost = float("inf")
    best_params = params.copy()
    last_expectations: Dict[str, float] = {}
    m = np.zeros_like(params, dtype=float)
    v = np.zeros_like(params, dtype=float)

    _gate_compressor = GateCompressor(convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u)

    info = f"params={len(param_names)} iters={max_iters} shots={shots} gradient={method}"
    if extra_info:
        info = f"{extra_info} {info}"
    logger.info("[%s] start optimization: %s", tag, info)

    for it in range(max_iters):
        if method == "autograd" and str(chip_name).lower() == "fieldquantum_sim":
            # Single HTTP call: server runs sampling + parameter-shift internally
            # and returns energy, per-Pauli expectations, and gradients.
            result = _cloud_platform.run_expectation(
                qasm=_cloud_qasm_template,
                param_names=list(param_names),
                param_values=params.tolist(),
                hamiltonian=_cloud_hamiltonian,
            )
            cost = float(result["energy"])
            expectations = {k: float(v) for k, v in result.get("expectations", {}).items()}
            grads = np.array(result["gradients"], dtype=float)
        elif method == "autograd" and str(chip_name).lower() == "simulator":
            params_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
            cost_t, expectations = _energy_and_expectations(
                symbolic_qc,
                params=params_t,
                param_names=param_names,
                hamiltonian=hamiltonian,
                device=device,
            )
            cost_t.backward()
            cost = float(cost_t.detach().cpu().item())
            grads = params_t.grad.detach().cpu().numpy().astype(float, copy=True)
        else:
            if gradient_param_template is None:
                raise RuntimeError(f"[{tag}] gradient_param_template not prepared")
            qc = instantiate_transpiled_template(
                gradient_param_template, param_names, params,
            )
            if circuit_transform is not None:
                qc = circuit_transform(qc, None)
            qc = _gate_compressor.run(qc)
            cost, expectations = evaluate_energy_with_backend(
                client,
                qc,
                name=f"{name}_iter{it}",
                num_qubits=num_qubits,
                backend=backend,
                chip_name=chip_name,
                shots=shots,
                hamiltonian=hamiltonian,
                zne=zne,
                readout_mitigation=readout_mitigation,
                clifford_fit_map=clifford_fit_map,
                target_qubits=target_qubits,
                qasm_version=qasm_version,
                convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
            )
            grads = parameter_shift_gradient(
                client,
                params,
                name=f"{name}_iter{it}",
                num_qubits=num_qubits,
                backend=backend,
                chip_name=chip_name,
                shots=shots,
                hamiltonian=hamiltonian,
                shift=shift,
                zne=zne,
                readout_mitigation=readout_mitigation,
                param_template=gradient_param_template,
                param_names=param_names,
                clifford_fit_map=clifford_fit_map,
                target_qubits=target_qubits,
                circuit_transform=circuit_transform,
                qasm_version=qasm_version,
                convert_single_qubit_gate_to_u=convert_single_qubit_gate_to_u,
            )

        grad_norm = float(np.linalg.norm(grads))
        logger.info("[%s] iter %d cost=%.6f grad_norm=%.6f", tag, it, cost, grad_norm)

        params, m, v = adam_update(
            params, grads, m, v, it + 1,
            lr=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
        )

        cost_history.append(float(cost))
        params_history.append(params.tolist())
        grad_history.append(grads.tolist())
        last_expectations = expectations
        if cost < best_cost:
            best_cost = float(cost)
            best_params = params.copy()
            logger.info("[%s] iter %d new best=%.6f", tag, it, best_cost)

        if callback is not None:
            callback(it, float(cost), params)

    return {
        "best_cost": best_cost,
        "best_params": best_params.tolist(),
        "cost_history": cost_history,
        "params_history": params_history,
        "grad_history": grad_history,
        "last_expectations": last_expectations,
    }

