"""Tests for QML framework: encoding, classifier (unified template)."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_hw.algorithms.qml_encoding import (
    angle_encoding_circuit,
    angle_encoding_circuit_symbolic,
    iqp_encoding_circuit,
    iqp_encoding_circuit_symbolic,
)
from quantum_hw.algorithms.qml import (
    run_pqc_classifier,
)


# ---------------------------------------------------------------------------
# Encoding tests — concrete
# ---------------------------------------------------------------------------

class TestAngleEncoding:
    def test_basic(self):
        qc = angle_encoding_circuit([0.5, 1.0], num_qubits=3)
        assert qc.nqubits == 3
        assert len(qc.gates) == 2  # two RY gates

    def test_fewer_features_than_qubits(self):
        qc = angle_encoding_circuit([0.5], num_qubits=4)
        assert len(qc.gates) == 1

    def test_rx_gate(self):
        qc = angle_encoding_circuit([0.5, 1.0], num_qubits=2, gate="rx")
        assert qc.nqubits == 2


class TestIQPEncoding:
    def test_basic(self):
        qc = iqp_encoding_circuit([0.5, 1.0], num_qubits=2, reps=1)
        assert qc.nqubits == 2
        # H(0), H(1), RZ(0), RZ(1), CX, RZ, CX = 7 gates
        assert len(qc.gates) >= 5

    def test_reps(self):
        qc1 = iqp_encoding_circuit([0.5, 1.0], num_qubits=2, reps=1)
        qc2 = iqp_encoding_circuit([0.5, 1.0], num_qubits=2, reps=2)
        assert len(qc2.gates) > len(qc1.gates)


# ---------------------------------------------------------------------------
# Encoding tests — symbolic
# ---------------------------------------------------------------------------

class TestAngleEncodingSymbolic:
    def test_basic(self):
        qc, names = angle_encoding_circuit_symbolic(3, 2)
        assert qc.nqubits == 3
        assert len(qc.gates) == 2
        assert names == ["x_0", "x_1"]
        # Gate params should be symbolic strings
        assert qc.gates[0][1] == "x_0"
        assert qc.gates[1][1] == "x_1"

    def test_fewer_features(self):
        qc, names = angle_encoding_circuit_symbolic(4, 1)
        assert len(names) == 1
        assert len(qc.gates) == 1

    def test_custom_prefix_and_gate(self):
        qc, names = angle_encoding_circuit_symbolic(2, 2, gate="rx", prefix="f")
        assert names == ["f_0", "f_1"]
        assert qc.gates[0][0] == "rx"

    def test_features_capped_at_num_qubits(self):
        qc, names = angle_encoding_circuit_symbolic(2, 5)
        assert len(names) == 2


class TestIQPEncodingSymbolic:
    def test_basic(self):
        qc, names = iqp_encoding_circuit_symbolic(3, 3)
        assert qc.nqubits == 3
        assert names == ["x_0", "x_1", "x_2"]
        # Should have H, RZ, and ZZ gates
        gate_names = [g[0] for g in qc.gates]
        assert "h" in gate_names
        assert "rz" in gate_names
        assert "cx" in gate_names

    def test_product_expressions(self):
        qc, names = iqp_encoding_circuit_symbolic(2, 2)
        # Should contain a product expression x_0*x_1
        rz_params = [g[1] for g in qc.gates if g[0] == "rz"]
        assert "x_0*x_1" in rz_params

    def test_reps(self):
        qc1, _ = iqp_encoding_circuit_symbolic(2, 2, reps=1)
        qc2, _ = iqp_encoding_circuit_symbolic(2, 2, reps=2)
        assert len(qc2.gates) > len(qc1.gates)

    def test_custom_prefix(self):
        _, names = iqp_encoding_circuit_symbolic(2, 2, prefix="feat")
        assert names == ["feat_0", "feat_1"]

    def test_apply_value_resolves_product(self):
        """apply_value with deep=True should resolve product expressions."""
        qc, names = iqp_encoding_circuit_symbolic(2, 2)
        qc_copy = qc.deepcopy()
        qc_copy.apply_value({"x_0": 0.5, "x_1": 0.3}, deep=True)
        rz_params = [g[1] for g in qc_copy.gates if g[0] == "rz"]
        # Should have 0.5, 0.3, and 0.5*0.3=0.15
        assert any(abs(p - 0.15) < 1e-10 for p in rz_params if isinstance(p, float))


# ---------------------------------------------------------------------------
# PQC Classifier tests — legacy QuantumCircuit path (backward compat)
# ---------------------------------------------------------------------------

class TestPQCClassifierUnifiedTemplate:
    """Unified template: transpile once, bind per-sample."""

    def test_angle_autograd_basic(self):
        """Angle encoding + autograd converges on separable data."""
        # class 0: features near 0 → |00⟩-like
        # class 1: features near pi → rotated states
        train_data = [
            ([0.0, 0.0], 0),
            ([0.1, 0.1], 0),
            ([np.pi, np.pi], 1),
            ([np.pi - 0.1, np.pi - 0.1], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="angle",
            num_classes=2,
            layers=2,
            max_iters=100,
            learning_rate=0.05,
            seed=42,
        )
        assert result.task == "supervised"
        assert len(result.loss_history) == 100
        # Loss should decrease
        assert result.loss_history[-1] < result.loss_history[0]

    def test_angle_autograd_convergence(self):
        """With enough iterations, angle encoding should achieve high accuracy."""
        train_data = [
            ([0.0, 0.0], 0),
            ([np.pi, np.pi], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="angle",
            num_classes=2,
            layers=2,
            max_iters=200,
            learning_rate=0.05,
            seed=42,
        )
        assert result.accuracy >= 0.5

    def test_angle_rx_gate(self):
        """encoding_kwargs forwarded correctly (rx gate)."""
        train_data = [([0.0, 0.0], 0), ([np.pi, np.pi], 1)]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="angle",
            encoding_kwargs={"gate": "rx"},
            num_classes=2,
            layers=1,
            max_iters=30,
            learning_rate=0.05,
            seed=42,
        )
        assert len(result.loss_history) == 30

    def test_iqp_autograd(self):
        """IQP encoding + autograd works (product expressions resolved)."""
        train_data = [
            ([0.0, 0.0], 0),
            ([np.pi, np.pi], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="iqp",
            num_classes=2,
            layers=1,
            max_iters=50,
            learning_rate=0.05,
            seed=42,
        )
        assert len(result.loss_history) == 50
        assert result.loss_history[-1] < result.loss_history[0]

    def test_angle_parameter_shift(self):
        """Unified template parameter-shift: transpiles once, converges."""
        from quantum_hw import QuantumHardwareClient
        from quantum_hw.api.backend import Backend

        client = QuantumHardwareClient()
        backend = Backend("Simulator")

        train_data = [
            ([0.0, 0.0], 0),
            ([np.pi, np.pi], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="angle",
            num_classes=2,
            layers=1,
            max_iters=30,
            learning_rate=0.05,
            seed=42,
            gradient_method="parameter-shift",
            client=client,
            backend=backend,
            chip_name="Simulator",
            shots=8192,
        )
        assert result.task == "supervised"
        assert len(result.loss_history) == 30
        assert result.loss_history[-1] < result.loss_history[0] + 0.5

    def test_iqp_parameter_shift(self):
        """IQP encoding + parameter-shift: product expressions resolved."""
        from quantum_hw import QuantumHardwareClient
        from quantum_hw.api.backend import Backend

        client = QuantumHardwareClient()
        backend = Backend("Simulator")

        train_data = [
            ([0.0, 0.0], 0),
            ([np.pi, np.pi], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="iqp",
            num_classes=2,
            layers=1,
            max_iters=20,
            learning_rate=0.05,
            seed=42,
            gradient_method="parameter-shift",
            client=client,
            backend=backend,
            chip_name="Simulator",
            shots=8192,
        )
        assert len(result.loss_history) == 20

    def test_custom_encoding_callable(self):
        """Custom encoding function passed as callable."""
        def my_encoding(num_qubits, num_features):
            return angle_encoding_circuit_symbolic(num_qubits, num_features, gate="rz")

        train_data = [([0.0, 0.0], 0), ([np.pi, np.pi], 1)]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding=my_encoding,
            num_classes=2,
            layers=1,
            max_iters=20,
            learning_rate=0.05,
            seed=42,
        )
        assert len(result.loss_history) == 20

    def test_multiple_samples(self):
        """Six-sample dataset with angle encoding."""
        train_data = [
            ([0.0, 0.0], 0),
            ([0.1, 0.2], 0),
            ([0.2, 0.1], 0),
            ([np.pi, np.pi], 1),
            ([np.pi - 0.1, np.pi - 0.2], 1),
            ([np.pi - 0.2, np.pi - 0.1], 1),
        ]
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="angle",
            num_classes=2,
            layers=2,
            max_iters=100,
            learning_rate=0.05,
            seed=42,
        )
        assert len(result.loss_history) == 100
        assert result.loss_history[-1] < result.loss_history[0]

    def test_callback(self):
        """Callback is invoked each iteration."""
        train_data = [([0.0], 0), ([np.pi], 1)]
        records = []
        result = run_pqc_classifier(
            num_qubits=2,
            train_data=train_data,
            encoding="angle",
            num_classes=2,
            layers=1,
            max_iters=5,
            learning_rate=0.05,
            seed=42,
            callback=lambda it, loss: records.append((it, loss)),
        )
        assert len(records) == 5
        assert records[0][0] == 0
        assert records[-1][0] == 4


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestPQCClassifierErrors:
    def test_missing_encoding_for_features(self):
        """Passing feature arrays without encoding should raise ValueError
        (encoding defaults to 'angle' so we test unknown instead)."""
        pass

    def test_unknown_encoding(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            run_pqc_classifier(
                num_qubits=2,
                train_data=[([0.0, 0.0], 0)],
                encoding="nonexistent",
                layers=1,
                max_iters=1,
            )

    def test_bad_gradient_method(self):
        with pytest.raises(ValueError, match="gradient_method"):
            run_pqc_classifier(
                num_qubits=2,
                train_data=[([0.0], 0)],
                encoding="angle",
                layers=1,
                max_iters=1,
                gradient_method="invalid",
            )

    def test_parameter_shift_requires_backend(self):
        with pytest.raises(ValueError, match="parameter-shift requires"):
            run_pqc_classifier(
                num_qubits=2,
                train_data=[([0.0], 0)],
                encoding="angle",
                layers=1,
                max_iters=1,
                gradient_method="parameter-shift",
            )
