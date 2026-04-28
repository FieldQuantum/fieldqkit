"""Simulator interface and threshold-based dispatch.

This module is the single place that decides whether to use statevector or MPS.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from ..circuit import QuantumCircuit

from .mps import energy_and_expectations as _energy_and_expectations_mps
from .mps import expectation_pauli as _expectation_pauli_mps
from .mps import sample_probabilities as _sample_probabilities_mps
from .mps import simulate_counts as _simulate_counts_mps
from .mps import simulate_mps as _simulate_mps
from .statevector import energy_and_expectations as _energy_and_expectations_statevector
from .statevector import simulate_statevector as _simulate_statevector
from .statevector import expectation_pauli as _expectation_pauli_statevector
from .statevector import sample_probabilities as _sample_probabilities_statevector
from .statevector import simulate_counts as _simulate_counts_statevector


MPS_THRESHOLD_QUBITS: int = 16
_UNSET = object()  # sentinel for dynamic default resolution


def get_sim_config() -> dict:
    """Return the current simulator configuration.

    Returns:
        ``dict`` with keys ``'mps_threshold_qubits'`` and ``'max_bond_dim'``.
    """
    from . import mps as _mps_mod

    return {
        "mps_threshold_qubits": MPS_THRESHOLD_QUBITS,
        "max_bond_dim": _mps_mod.MAX_BOND_DIM,
    }


def set_sim_config(
    *,
    mps_threshold_qubits: Optional[int] = None,
    max_bond_dim: Optional[int] = _UNSET,  # type: ignore[assignment]
) -> None:
    """Update simulator hyper-parameters at runtime.

    Args:
        mps_threshold_qubits (*int | None*): Qubit count above which MPS is
            used instead of statevector.  ``None`` (default) leaves the value
            unchanged.
        max_bond_dim (*int | None*): Maximum MPS bond dimension.  Pass ``None``
            to disable truncation.  Omitting the argument (internal sentinel)
            leaves the value unchanged.

    Example::

        from quantum_hw.sim import set_sim_config
        set_sim_config(mps_threshold_qubits=20, max_bond_dim=512)
    """
    global MPS_THRESHOLD_QUBITS
    from . import mps as _mps_mod

    if mps_threshold_qubits is not None:
        if not isinstance(mps_threshold_qubits, int) or mps_threshold_qubits < 1:
            raise ValueError("mps_threshold_qubits must be a positive integer")
        MPS_THRESHOLD_QUBITS = mps_threshold_qubits

    if max_bond_dim is not _UNSET:
        if max_bond_dim is not None:
            if not isinstance(max_bond_dim, int) or max_bond_dim < 1:
                raise ValueError("max_bond_dim must be a positive integer or None")
        _mps_mod.MAX_BOND_DIM = max_bond_dim


def _extract_measurements(qc: QuantumCircuit):
    """Extract qubit-to-cbit mapping from measure gates in a circuit.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.

    Returns:
        Tuple of ``(measured_qubits, measured_cbits, num_cbits)`` or ``None``
        if the circuit contains no measurement gates.
    """
    measured_qubits: list[int] = []
    measured_cbits: list[int] = []
    for gate in getattr(qc, "gates", []):
        if gate[0] == "measure":
            measured_qubits.extend(gate[1])
            measured_cbits.extend(gate[2])
    if not measured_qubits:
        return None
    num_cbits = max(measured_cbits) + 1
    return measured_qubits, measured_cbits, num_cbits


def _project_counts_to_cbits(
    counts: Dict[str, int],
    measured_qubits: Sequence[int],
    measured_cbits: Sequence[int],
    num_cbits: int,
) -> Dict[str, int]:
    """Project full-qubit simulator counts to the classical-bit subspace.

    Each qubit in *measured_qubits* is mapped to the corresponding classical
    bit in *measured_cbits*.  Unmeasured qubits are marginalized out.

    Args:
        counts (*Dict[str, int]*): Full-qubit counts from the simulator.
        measured_qubits (*Sequence[int]*): Qubit indices that are measured.
        measured_cbits (*Sequence[int]*): Classical bit index each qubit maps to.
        num_cbits (*int*): Width of the output bitstring (``max(cbits) + 1``).

    Returns:
        Counts dictionary with ``num_cbits``-wide bitstrings.
    """
    projected: Dict[str, int] = {}
    for bitstring, count in counts.items():
        cbits = [0] * num_cbits
        for q, c in zip(measured_qubits, measured_cbits):
            cbits[c] = int(bitstring[q])
        key = "".join(str(b) for b in cbits)
        projected[key] = projected.get(key, 0) + count
    return projected


def simulate_counts(
    qc: QuantumCircuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    param_values: Dict[str, object] | None = None,
    max_bond_dim: int | None | object = _UNSET,
    device: torch.device | str | None = None,
) -> Dict[str, int]:
    """Simulate counts with threshold-based backend selection.

    Args:
        qc (*QuantumCircuit*): Quantum circuit.
        shots (*int*): Number of measurement shots.
        seed (*Optional[int]*): Random seed for reproducibility. Defaults to ``None``.
        param_values (*Dict[str, object] | None*): Parameter name to value mapping. Defaults to ``None``.
        max_bond_dim (*int | None*): Maximum MPS bond dimension (MPS backend only). ``None`` means no truncation. Defaults to current ``mps.MAX_BOND_DIM``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``Dict[str, int]`` mapping bitstrings to their occurrence counts.
        When the circuit contains explicit measure gates with a qubit-to-cbit
        mapping, the returned bitstrings are projected to the classical-bit
        subspace (width = ``max(cbit) + 1``).
    """
    from . import mps as _mps_mod
    if max_bond_dim is _UNSET:
        max_bond_dim = _mps_mod.MAX_BOND_DIM

    nqubits = int(getattr(qc, "nqubits", 0) or 0)
    if nqubits > MPS_THRESHOLD_QUBITS:
        raw = _simulate_counts_mps(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
            max_bond_dim=max_bond_dim,
            device=device,
        )
    else:
        raw = _simulate_counts_statevector(
            qc,
            shots,
            seed=seed,
            param_values=param_values,
            device=device,
        )

    meas = _extract_measurements(qc)
    if meas is not None:
        return _project_counts_to_cbits(raw, *meas)
    return raw


def expectation_pauli(
    state,
    pauli: str,
    *,
    num_qubits: int,
):
    """Return <psi|P|psi> using threshold-based backend selection.

    Args:
        state: Flat statevector tensor (≤ threshold qubits) or MPS tensor list (> threshold qubits).
        pauli (*str*): Pauli string (e.g. ``'XZI'``).
        num_qubits (*int*): Number of qubits.

    Returns:
        Expectation value as a scalar.
    """
    if int(num_qubits) > MPS_THRESHOLD_QUBITS:
        return _expectation_pauli_mps(state, pauli, num_qubits=num_qubits)
    return _expectation_pauli_statevector(state, pauli, num_qubits=num_qubits)


def sample_probabilities(
    state,
    samples,
    *,
    num_qubits: int,
):
    """Return probabilities for sample vectors via threshold-based dispatch.

    Args:
        state: Statevector or MPS.
        samples: ``(N, n_qubits)`` integer tensor/array with entries 0/1.
        num_qubits (*int*): Number of qubits, used to select backend.

    Returns:
        Probability tensor for the given samples.
    """
    if int(num_qubits) > MPS_THRESHOLD_QUBITS:
        return _sample_probabilities_mps(state, samples)
    return _sample_probabilities_statevector(state, samples)


def energy_and_expectations(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names,
    hamiltonian,
    max_bond_dim: int | None | object = _UNSET,
    device: torch.device | str | None = None,
):
    """Evaluate Hamiltonian energy via threshold-based backend selection.

    Args:
        symbolic_qc (*QuantumCircuit*): Symbolic (unbound) quantum circuit.
        params (*torch.Tensor*): 1-D tensor of variational parameter values.
        param_names (*List[str]*): Names of variational parameters, matching ``params`` element-wise.
        hamiltonian (*List[Tuple[float, str]]*): Target Hamiltonian as coefficient–Pauli-string pairs.
        max_bond_dim (*int | None*): Maximum MPS bond dimension (MPS backend only). ``None`` means no truncation. Defaults to current ``mps.MAX_BOND_DIM``.
        device (*torch.device | str | None*): Torch device (``'cpu'`` or ``'cuda'``). Defaults to ``None``.

    Returns:
        ``(energy, expectations)`` tuple from the selected backend.
    """
    from . import mps as _mps_mod
    if max_bond_dim is _UNSET:
        max_bond_dim = _mps_mod.MAX_BOND_DIM

    nqubits = int(getattr(symbolic_qc, "nqubits", 0) or 0)
    if nqubits > MPS_THRESHOLD_QUBITS:
        return _energy_and_expectations_mps(
            symbolic_qc,
            params=params,
            param_names=param_names,
            hamiltonian=hamiltonian,
            max_bond_dim=max_bond_dim,
            device=device,
        )
    return _energy_and_expectations_statevector(
        symbolic_qc,
        params=params,
        param_names=param_names,
        hamiltonian=hamiltonian,
        device=device,
    )


def build_state_from_symbolic(
    symbolic_qc: QuantumCircuit,
    *,
    params,
    param_names: Sequence[str],
    max_bond_dim: int | None | object = _UNSET,
    device: torch.device | str | None = None,
):
    """Build a simulator state from a symbolic circuit and a differentiable param tensor.

    Dispatches to statevector (flat tensor) or MPS (list of tensors) based on
    qubit count vs ``MPS_THRESHOLD_QUBITS``.  The returned state object can be
    passed directly to the interface-layer :func:`expectation_pauli` and
    :func:`sample_probabilities`, which perform the same dispatch.

    Args:
        symbolic_qc (*QuantumCircuit*): Symbolic (unbound) quantum circuit.
        params (*torch.Tensor*): 1-D differentiable parameter tensor.
        param_names (*Sequence[str]*): Names corresponding to *params* elements.
        max_bond_dim (*int | None*): MPS bond dimension cap (MPS backend only).
            ``None`` means no truncation.  Defaults to current ``mps.MAX_BOND_DIM``.
        device (*torch.device | str | None*): Torch device. Defaults to ``None``.

    Returns:
        Flat complex statevector tensor of length ``2**n`` (statevector backend)
        or list of MPS site tensors (MPS backend).
    """
    from .common import build_param_values_from_tensor
    from . import mps as _mps_mod

    if max_bond_dim is _UNSET:
        max_bond_dim = _mps_mod.MAX_BOND_DIM

    nqubits = int(getattr(symbolic_qc, "nqubits", 0) or 0)
    param_values = build_param_values_from_tensor(params=params, param_names=param_names)
    if nqubits > MPS_THRESHOLD_QUBITS:
        return _simulate_mps(
            symbolic_qc,
            param_values=param_values,
            max_bond_dim=max_bond_dim,
            device=device,
        )
    return _simulate_statevector(symbolic_qc, param_values=param_values, device=device)
