"""Reproducibility guarantees of the transpilation pipeline.

Three properties are covered:

1. With default settings, transpilation is **deterministic** — the same circuit
   compiles to the same gate sequence every time.
2. When randomness is explicitly enabled (``routing_initial_mapping='random'``,
   ``routing_random_choice=True`` or ``routing_n_trials > 1``), passing ``seed``
   makes the result reproducible again.
3. The passes own **private** generators: they never read from, nor advance, the
   global ``random`` / ``numpy.random`` state.
"""

import random

import networkx as nx
import numpy as np
import pytest

from fieldqkit.api.backend import Backend
from fieldqkit.api.client import QuantumHardwareClient
from fieldqkit.circuit import QuantumCircuit
from fieldqkit.compile.layout import Layout
from fieldqkit.compile.routing import SabreRouting
from fieldqkit.compile.transpiler import Transpiler


def _line_graph(n):
    g = nx.Graph()
    for i in range(n - 1):
        g.add_edge(i, i + 1, fidelity=1.0)
    g.graph["normal_order"] = list(range(n))
    return g


def _ghz(n, style="star"):
    """``style='star'`` fans out from qubit 0 (needs SWAPs on a line);
    ``style='chain'`` is the nearest-neighbour cascade (needs none)."""
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i) if style == "star" else qc.cx(i - 1, i)
    qc.measure(list(range(n)), list(range(n)))
    return qc


def _gates(qc):
    return [tuple(str(x) for x in g) for g in qc.gates]


def _n_swaps(qc):
    return sum(1 for g in qc.gates if g[0] == "swap")


# --------------------------------------------------------------------------
# 1. default settings are deterministic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["star", "chain"])
def test_transpiler_defaults_are_deterministic(style):
    """Repeated transpiles of one circuit must be gate-for-gate identical."""
    runs = [_gates(Transpiler(chip_backend=None).run(_ghz(8, style))) for _ in range(8)]
    assert all(r == runs[0] for r in runs), "default transpilation is not reproducible"


def test_sabre_defaults_are_deterministic():
    g = _line_graph(7)
    qc = _ghz(7, "star")
    runs = [_gates(SabreRouting(g, iterations=5).run(qc)) for _ in range(8)]
    assert all(r == runs[0] for r in runs)


def test_default_mapping_beats_random_on_nearest_neighbour_circuit():
    """A cascaded GHZ needs zero SWAPs; the default must actually find that."""
    routed = Transpiler(chip_backend=None).run(_ghz(8, "chain"))
    assert _n_swaps(routed) == 0


def test_transpiler_default_argument_values():
    """Guard the defaults themselves so a future edit cannot silently flip them."""
    import inspect

    params = inspect.signature(Transpiler.run).parameters
    assert params["routing_initial_mapping"].default == "trivial"
    assert params["routing_random_choice"].default is False
    assert params["routing_n_trials"].default == 1
    assert params["seed"].default is None


# --------------------------------------------------------------------------
# 2. explicit randomness + seed is reproducible
# --------------------------------------------------------------------------


_RANDOM_KW = dict(
    routing_initial_mapping="random",
    routing_random_choice=True,
    routing_n_trials=4,
)


def test_seed_makes_random_routing_reproducible():
    runs = [
        _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=123, **_RANDOM_KW))
        for _ in range(6)
    ]
    assert all(r == runs[0] for r in runs)


def test_different_seeds_may_differ_but_each_is_stable():
    a1 = _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=1, **_RANDOM_KW))
    a2 = _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=1, **_RANDOM_KW))
    b1 = _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=2, **_RANDOM_KW))
    b2 = _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=2, **_RANDOM_KW))
    assert a1 == a2 and b1 == b2


def test_sabre_seed_reproducible():
    g = _line_graph(7)
    qc = _ghz(7, "star")
    kw = dict(iterations=5, n_trials=4, do_random_choice=True, initial_mapping="random")
    runs = [_gates(SabreRouting(g, seed=42, **kw).run(qc)) for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_unseeded_random_routing_still_varies():
    """Opting into randomness without a seed stays random (that is the point)."""
    seen = {
        tuple(_gates(Transpiler(chip_backend=None).run(_ghz(10, "star"), **_RANDOM_KW)))
        for _ in range(12)
    }
    assert len(seen) > 1


# --------------------------------------------------------------------------
# 3. global RNG state is neither consumed nor depended upon
# --------------------------------------------------------------------------


def test_routing_does_not_consume_global_random_state():
    random.seed(999)
    expected = [random.random() for _ in range(4)]

    random.seed(999)
    Transpiler(chip_backend=None).run(_ghz(9, "star"), **_RANDOM_KW)
    actual = [random.random() for _ in range(4)]

    assert actual == expected, "transpilation advanced the global `random` state"


def test_layout_does_not_consume_global_numpy_state():
    """Circuits wider than Layout.algorithm_switch_threshold take the BFS path."""
    np.random.seed(555)
    expected = np.random.random(4).tolist()

    np.random.seed(555)
    Transpiler(Backend("Simulator")).run(_ghz(14, "chain"), use_dd=False)
    actual = np.random.random(4).tolist()

    assert actual == expected, "layout selection advanced the global NumPy state"


def test_routing_ignores_global_seed():
    """Seeding the global RNG must not change compiled output any more."""
    random.seed(1)
    a = _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=5, **_RANDOM_KW))
    random.seed(2)
    b = _gates(Transpiler(chip_backend=None).run(_ghz(8, "star"), seed=5, **_RANDOM_KW))
    assert a == b


def test_layout_seed_is_reproducible():
    bk = Backend("Simulator")
    a = Layout(bk, seed=3).select_much_qubits_from_backend(12)
    b = Layout(bk, seed=3).select_much_qubits_from_backend(12)
    assert a == b


# --------------------------------------------------------------------------
# 4. client / algorithm layer forward the routing knobs
# --------------------------------------------------------------------------


def _bare_client():
    """A client instance without network setup — _transpile_with_backend is pure."""
    return QuantumHardwareClient.__new__(QuantumHardwareClient)


def test_client_transpile_exposes_routing_params():
    import inspect

    params = inspect.signature(QuantumHardwareClient._transpile_with_backend).parameters
    for name in ("routing_initial_mapping", "routing_random_choice", "seed", "niter"):
        assert name in params, f"{name} is not reachable through the client API"
    assert params["routing_initial_mapping"].default == "trivial"
    assert params["routing_random_choice"].default is False
    # escape hatch for anything else Transpiler.run grows later
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_client_transpile_is_deterministic_by_default():
    client, bk = _bare_client(), Backend("Simulator")
    runs = [
        _gates(client._transpile_with_backend(_ghz(6, "chain"), bk, use_dd=False))
        for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs)


def test_client_transpile_forwards_seed_and_mapping():
    client, bk = _bare_client(), Backend("Simulator")
    kw = dict(use_dd=False, routing_initial_mapping="random",
              routing_random_choice=True, routing_n_trials=3)
    a = _gates(client._transpile_with_backend(_ghz(6, "star"), bk, seed=11, **kw))
    b = _gates(client._transpile_with_backend(_ghz(6, "star"), bk, seed=11, **kw))
    assert a == b


def test_client_transpile_forwards_explicit_mapping_list():
    client, bk = _bare_client(), Backend("Simulator")
    target = [2, 3, 4, 5]
    out = client._transpile_with_backend(
        _ghz(4, "chain"), bk, target_qubits=target,
        routing_initial_mapping=target, use_dd=False,
    )
    assert set(out.qubits) == set(target)


def test_client_transpile_kwargs_escape_hatch():
    """Unknown-to-the-client kwargs reach Transpiler.run untouched."""
    client, bk = _bare_client(), Backend("Simulator")
    out = client._transpile_with_backend(
        _ghz(5, "chain"), bk, use_dd=False, use_gate_compressor=False,
        use_translate_to_basis=False,
    )
    assert any(g[0] == "cx" for g in out.gates)


@pytest.mark.parametrize(
    "module_name, func_name",
    [
        ("fieldqkit.algorithms.vqe", "run_vqe_with_backend"),
        ("fieldqkit.algorithms.qaoa", "run_qaoa_with_backend"),
        ("fieldqkit.algorithms.qml", "run_pqc_classifier"),
        ("fieldqkit.algorithms.qml", "run_qnn_unsupervised"),
        ("fieldqkit.algorithms.qml", "run_qnn_conditional"),
        ("fieldqkit.algorithms.circuit_compression", "build_compression_transform"),
    ],
)
def test_algorithm_entrypoints_accept_transpile_options(module_name, func_name):
    import importlib
    import inspect

    fn = getattr(importlib.import_module(module_name), func_name)
    params = inspect.signature(fn).parameters
    assert "transpile_options" in params
    assert params["transpile_options"].default is None
