"""Tests for the QAOA module: Hamiltonian builders, ansatz, and autograd optimization."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantum_hw.algorithms.qaoa import (
    build_maxcut_hamiltonian,
    build_custom_cost_hamiltonian,
    build_qaoa_ansatz_symbolic,
    run_qaoa_with_backend,
    QAOARunner,
)
from quantum_hw.api.backend import Backend
from quantum_hw.core.types import QAOAResult


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------


class TestBuildMaxcutHamiltonian:
    def test_triangle_graph(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        h = build_maxcut_hamiltonian(edges, num_qubits=3)
        assert len(h) == 3
        for coeff, pauli in h:
            assert coeff == -0.5
            assert len(pauli) == 3
            assert pauli.count("Z") == 2

    def test_single_edge(self):
        h = build_maxcut_hamiltonian([(0, 1)], num_qubits=2)
        assert len(h) == 1
        assert h[0] == (-0.5, "ZZ")

    def test_edge_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            build_maxcut_hamiltonian([(0, 3)], num_qubits=3)

    def test_self_loop_raises(self):
        with pytest.raises(ValueError, match="self-loop"):
            build_maxcut_hamiltonian([(1, 1)], num_qubits=2)

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_maxcut_hamiltonian([], num_qubits=0)


class TestBuildCustomCostHamiltonian:
    def test_valid_terms(self):
        terms = [(0.5, "ZZI"), (-0.3, "IZZ")]
        h = build_custom_cost_hamiltonian(terms, num_qubits=3)
        assert len(h) == 2
        assert h[0] == (0.5, "ZZI")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            build_custom_cost_hamiltonian([(1.0, "ZZ")], num_qubits=3)

    def test_invalid_char_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            build_custom_cost_hamiltonian([(1.0, "ZA")], num_qubits=2)


# ---------------------------------------------------------------------------
# Ansatz builder
# ---------------------------------------------------------------------------


class TestBuildQaoaAnsatzSymbolic:
    def test_basic_structure(self):
        edges = [(0, 1), (1, 2)]
        names, qc = build_qaoa_ansatz_symbolic(3, edges, p=2)
        assert names == ["gamma_0", "beta_0", "gamma_1", "beta_1"]
        # Should have: 3 H + 2*(2 RZZ + 3 RX) = 3 + 2*5 = 13 gates
        assert len(qc.gates) == 13

    def test_p_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_qaoa_ansatz_symbolic(2, [(0, 1)], p=0)

    def test_single_layer(self):
        names, qc = build_qaoa_ansatz_symbolic(2, [(0, 1)], p=1)
        assert len(names) == 2
        # 2 H + 1 RZZ + 2 RX = 5
        assert len(qc.gates) == 5


# ---------------------------------------------------------------------------
# Autograd optimization (Simulator)
# ---------------------------------------------------------------------------


class TestQAOAAutograd:
    def test_runs_on_simulator(self):
        edges = [(0, 1)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=2)

        result = run_qaoa_with_backend(
            object(),
            name="test_qaoa_autograd",
            num_qubits=2,
            backend=Backend("Simulator"),
            chip_name="Simulator",
            hamiltonian=hamiltonian,
            edges=edges,
            p=1,
            shots=512,
            max_iters=5,
            learning_rate=0.2,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
            shift=np.pi / 2,
            zne=False,
            readout_mitigation=False,
            seed=42,
            gradient_method="autograd",
        )

        assert isinstance(result, QAOAResult)
        assert len(result.cost_history) == 5
        assert np.isfinite(result.best_cost)
        assert result.best_cost <= min(result.cost_history)
        assert len(result.best_params) == 2

    def test_rejects_non_simulator(self):
        edges = [(0, 1)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=2)

        with pytest.raises(ValueError, match="only supported on Simulator"):
            run_qaoa_with_backend(
                object(),
                name="test_qaoa_bad_chip",
                num_qubits=2,
                backend=Backend("Simulator"),
                chip_name="Baihua",
                hamiltonian=hamiltonian,
                edges=edges,
                p=1,
                shots=512,
                max_iters=1,
                learning_rate=0.2,
                beta1=0.9,
                beta2=0.98,
                eps=1e-8,
                shift=np.pi / 2,
                zne=False,
                readout_mitigation=False,
                gradient_method="autograd",
            )

    def test_triangle_maxcut_cost_decreases(self):
        """For a triangle graph the cost should decrease over iterations."""
        edges = [(0, 1), (1, 2), (0, 2)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=3)

        result = run_qaoa_with_backend(
            object(),
            name="test_qaoa_triangle",
            num_qubits=3,
            backend=Backend("Simulator"),
            chip_name="Simulator",
            hamiltonian=hamiltonian,
            edges=edges,
            p=2,
            shots=512,
            max_iters=10,
            learning_rate=0.15,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
            shift=np.pi / 2,
            zne=False,
            readout_mitigation=False,
            seed=7,
            gradient_method="autograd",
        )

        assert result.best_cost < result.cost_history[0]

    def test_custom_init_params(self):
        edges = [(0, 1)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=2)

        result = run_qaoa_with_backend(
            object(),
            name="test_qaoa_init",
            num_qubits=2,
            backend=Backend("Simulator"),
            chip_name="Simulator",
            hamiltonian=hamiltonian,
            edges=edges,
            p=1,
            shots=512,
            max_iters=3,
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.98,
            eps=1e-8,
            shift=np.pi / 2,
            zne=False,
            readout_mitigation=False,
            init_params=[0.5, 1.0],
            gradient_method="autograd",
        )

        assert len(result.cost_history) == 3

    def test_wrong_init_params_length_raises(self):
        edges = [(0, 1)]
        hamiltonian = build_maxcut_hamiltonian(edges, num_qubits=2)

        with pytest.raises(ValueError, match="init_params length"):
            run_qaoa_with_backend(
                object(),
                name="test_qaoa_bad_init",
                num_qubits=2,
                backend=Backend("Simulator"),
                chip_name="Simulator",
                hamiltonian=hamiltonian,
                edges=edges,
                p=1,
                shots=512,
                max_iters=1,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.98,
                eps=1e-8,
                shift=np.pi / 2,
                zne=False,
                readout_mitigation=False,
                init_params=[0.5, 1.0, 2.0],  # p=1 expects 2 params
                gradient_method="autograd",
            )


# ---------------------------------------------------------------------------
# Optimizer utils (shared helpers)
# ---------------------------------------------------------------------------


class TestOptimizerUtils:
    def test_energy_from_expectations(self):
        from quantum_hw.algorithms.optimizer_utils import energy_from_expectations
        h = [(-0.5, "ZZ"), (0.3, "ZI")]
        exp = {"ZZ": 0.8, "ZI": -0.5}
        e = energy_from_expectations(h, exp)
        assert abs(e - (-0.5 * 0.8 + 0.3 * (-0.5))) < 1e-12

    def test_adam_update_moves_params(self):
        from quantum_hw.algorithms.optimizer_utils import adam_update
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, -0.2])
        m = np.zeros(2)
        v = np.zeros(2)
        new_params, _, _ = adam_update(params, grads, m, v, 1,
                                       lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        assert not np.allclose(new_params, params)

    def test_normalize_observable_values_dict_list(self):
        from quantum_hw.algorithms.optimizer_utils import normalize_observable_values
        result = normalize_observable_values([{"ZZ": 0.5}, {"ZI": -0.3}])
        assert result == {"ZZ": 0.5, "ZI": -0.3}

    def test_normalize_observable_values_single(self):
        from quantum_hw.algorithms.optimizer_utils import normalize_observable_values
        result = normalize_observable_values([0.42])
        assert result == 0.42

    def test_fit_linear_clifford_map_identity(self):
        from quantum_hw.algorithms.optimizer_utils import fit_linear_clifford_map
        # Perfect data: ideal == noisy => a~1, b~0
        noisy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ideal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        a, b = fit_linear_clifford_map(noisy, ideal)
        assert abs(a - 1.0) < 0.01
        assert abs(b) < 0.01
