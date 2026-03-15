from itertools import product

from quantum_hw.algorithms.vqe import _should_use_shadow_measurement


def _all_pauli_strings(num_qubits: int):
    out = []
    for ops in product("IXYZ", repeat=num_qubits):
        if all(op == "I" for op in ops):
            continue
        out.append("".join(ops))
    return out


def test_shadow_auto_switch_uses_only_min_groups_threshold():
    observables = _all_pauli_strings(4)

    use_shadow = _should_use_shadow_measurement(
        measurement_mode="auto",
        observables=observables,
        num_qubits=4,
        shadow_min_groups=32,
    )
    assert use_shadow is True


def test_shadow_auto_switch_respects_min_groups_cutoff():
    observables = _all_pauli_strings(4)

    use_shadow = _should_use_shadow_measurement(
        measurement_mode="auto",
        observables=observables,
        num_qubits=4,
        shadow_min_groups=96,
    )
    assert use_shadow is False
