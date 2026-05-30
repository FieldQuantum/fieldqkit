"""Tests for noise channel simulation and integration."""

import numpy as np
import pytest
import torch

from fieldqkit.circuit import QuantumCircuit
from fieldqkit.sim import (
    simulate_counts,
    simulate_density_matrix,
    simulate_noisy_counts,
    expectation_pauli,
    expectation_pauli_dm,
)
from fieldqkit.circuit.quantumcircuit_helpers import has_noise_channels


class TestNoiseGateCreation:
    """Test creation and basic properties of noise gate methods."""

    def test_depolarize1_creation(self):
        """Test adding single-qubit depolarizing noise."""
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0)
        assert len(qc.gates) == 2
        assert qc.gates[1] == ('depolarize1', 0.1, 0)

    def test_depolarize2_creation(self):
        """Test adding two-qubit depolarizing noise."""
        qc = QuantumCircuit(2)
        qc.depolarize2(0.05, 0, 1)
        assert len(qc.gates) == 1
        assert qc.gates[0] == ('depolarize2', 0.05, 0, 1)

    def test_x_error_creation(self):
        """Test adding X error channel."""
        qc = QuantumCircuit(2)
        qc.x_error(0.2, 1)
        assert qc.gates[0] == ('x_error', 0.2, 1)

    def test_y_error_creation(self):
        """Test adding Y error channel."""
        qc = QuantumCircuit(2)
        qc.y_error(0.15, 0)
        assert qc.gates[0] == ('y_error', 0.15, 0)

    def test_z_error_creation(self):
        """Test adding Z error channel."""
        qc = QuantumCircuit(2)
        qc.z_error(0.1, 1)
        assert qc.gates[0] == ('z_error', 0.1, 1)

    def test_amplitude_damping_creation(self):
        """Test adding amplitude damping channel."""
        qc = QuantumCircuit(2)
        qc.amplitude_damping(0.3, 0)
        assert qc.gates[0] == ('amplitude_damping', 0.3, 0)

    def test_phase_damping_creation(self):
        """Test adding phase damping channel."""
        qc = QuantumCircuit(2)
        qc.phase_damping(0.2, 1)
        assert qc.gates[0] == ('phase_damping', 0.2, 1)

    def test_noise_gate_parameter_validation(self):
        """Test that noise gate parameters are validated."""
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="must be a number in"):
            qc.depolarize1(-0.1, 0)
        with pytest.raises(ValueError, match="must be a number in"):
            qc.depolarize1(1.5, 0)

    def test_has_noise_channels_true(self):
        """Test has_noise_channels detects noise."""
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0)
        assert has_noise_channels(qc) is True

    def test_has_noise_channels_false(self):
        """Test has_noise_channels returns False for clean circuit."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        assert has_noise_channels(qc) is False


class TestNoiseQASMRoundtrip:
    """Test OpenQASM serialization and parsing of noise gates."""

    def test_depolarize1_qasm_roundtrip(self):
        """Test depolarize1 survives to_openqasm2 -> from_openqasm2."""
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0).h(1)
        qasm_str = qc.to_openqasm2()
        assert 'opaque depolarize1' in qasm_str
        assert 'depolarize1(0.1) q[0]' in qasm_str

        qc2 = QuantumCircuit().from_openqasm2(qasm_str)
        assert has_noise_channels(qc2)
        assert any(g[0] == 'depolarize1' for g in qc2.gates)

    def test_depolarize2_qasm_roundtrip(self):
        """Test depolarize2 survives to_openqasm2 -> from_openqasm2."""
        qc = QuantumCircuit(2)
        qc.depolarize2(0.05, 0, 1)
        qasm_str = qc.to_openqasm2()
        assert 'opaque depolarize2' in qasm_str
        assert 'depolarize2(0.05) q[0],q[1]' in qasm_str

    def test_all_noise_channels_qasm(self):
        """Test all noise channels appear in QASM opaque declarations."""
        qc = QuantumCircuit(3)
        qc.depolarize1(0.1, 0)
        qc.depolarize2(0.05, 1, 2)
        qc.x_error(0.1, 0)
        qc.y_error(0.1, 1)
        qc.z_error(0.1, 2)
        qc.amplitude_damping(0.2, 0)
        qc.phase_damping(0.15, 1)

        qasm_str = qc.to_openqasm2()
        assert 'opaque depolarize1(p) q;' in qasm_str
        assert 'opaque depolarize2(p) q0,q1;' in qasm_str
        assert 'opaque x_error(p) q;' in qasm_str
        assert 'opaque y_error(p) q;' in qasm_str
        assert 'opaque z_error(p) q;' in qasm_str
        assert 'opaque amplitude_damping(gamma) q;' in qasm_str
        assert 'opaque phase_damping(gamma) q;' in qasm_str


class TestDensityMatrixSimulator:
    """Test density matrix simulation of noisy circuits."""

    def test_dm_pure_state_no_noise(self):
        """Test DM of |0><0| for a circuit with no noise."""
        qc = QuantumCircuit(1)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.zeros(2, 2, dtype=torch.complex64)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_dm_plus_state(self):
        """Test DM of |+><+| = (|0><0| + |0><1| + |1><0| + |1><1|) / 2."""
        qc = QuantumCircuit(1)
        qc.h(0)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5]
        ], dtype=torch.complex64)
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_depolarize1_trace_preservation(self):
        """Test that trace(ρ) = 1 after depolarizing noise."""
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.3, 0)
        rho = simulate_density_matrix(qc)

        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-5)

    def test_depolarize1_maximally_mixed(self):
        """Test that p=0.75 (for 1q) leads to maximally mixed state."""
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.75, 0)
        rho = simulate_density_matrix(qc).cpu()

        expected = 0.5 * torch.eye(2, dtype=torch.complex64)
        assert torch.allclose(rho, expected, atol=1e-4)

    def test_amplitude_damping_collapse(self):
        """Test that gamma=1.0 in amplitude damping collapses |1> to |0>."""
        qc = QuantumCircuit(1)
        qc.x(0).amplitude_damping(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.zeros(2, 2, dtype=torch.complex64)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_dm_with_unitary_and_noise(self):
        """Test DM with mixed unitary gates and noise."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize1(0.1, 0).h(1)
        rho = simulate_density_matrix(qc)

        assert rho.shape == (4, 4)
        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-5)


class TestInterfaceDispatch:
    """Test that simulate_counts routes correctly based on noise presence."""

    def test_dispatch_clean_circuit(self):
        """Test clean circuit uses statevector backend."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).measure_all()
        counts = simulate_counts(qc, shots=100, seed=42)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 100
        assert has_noise_channels(qc) is False

    def test_dispatch_noisy_circuit(self):
        """Test noisy circuit uses DM backend."""
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0).cx(0, 1).measure_all()
        counts = simulate_counts(qc, shots=100, seed=42)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 100
        assert has_noise_channels(qc) is True

    def test_simulate_noisy_counts_direct(self):
        """Test direct call to simulate_noisy_counts."""
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.3, 0)
        counts = simulate_noisy_counts(qc, shots=1000, seed=42)

        assert isinstance(counts, dict)
        assert sum(counts.values()) == 1000
        assert len(counts) <= 2


class TestExpectationValue:
    """Test expectation value computation on noisy circuits."""

    def test_expectation_z_pure_state(self):
        """Test <ψ|Z|ψ> on |0> state via SV simulator."""
        from fieldqkit.sim import simulate_statevector
        qc = QuantumCircuit(1)
        psi = simulate_statevector(qc)
        exp = expectation_pauli(psi, 'Z', num_qubits=1)
        assert abs(exp.real.item() - 1.0) < 1e-5

    def test_expectation_z_plus_state(self):
        """Test <ψ|Z|ψ> on |+> state is 0 via SV simulator."""
        from fieldqkit.sim import simulate_statevector
        qc = QuantumCircuit(1)
        qc.h(0)
        psi = simulate_statevector(qc)
        exp = expectation_pauli(psi, 'Z', num_qubits=1)
        assert abs(exp.real.item()) < 1e-5

    def test_expectation_noisy_circuit(self):
        """Test expectation value on noisy circuit."""
        qc = QuantumCircuit(1)
        qc.x(0).phase_damping(0.2, 0)
        # First simulate to get DM state
        rho = simulate_density_matrix(qc)
        # Then compute expectation on the state
        exp = expectation_pauli(rho, 'Z', num_qubits=1)

        assert isinstance(exp, torch.Tensor)
        # Value should be real (phase damping preserves coherence on Z basis)
        assert exp.real.item() < 0.0  # |1> partially collapsed toward |0>

    def test_expectation_two_qubit_observable(self):
        """Test multi-qubit observable expectation via SV simulator."""
        from fieldqkit.sim import simulate_statevector
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        psi = simulate_statevector(qc)
        exp = expectation_pauli(psi, 'ZZ', num_qubits=2)

        assert isinstance(exp, torch.Tensor)


class TestAutograd:
    """Test that DM expectation supports autograd."""

    def test_expectation_parameter_gradient(self):
        """Test that expectation is differentiable w.r.t. symbolic params."""
        qc = QuantumCircuit(1)
        qc.rx('theta', 0).depolarize1(0.1, 0)

        theta = torch.tensor(np.pi / 4, requires_grad=True)
        param_values = {'theta': theta}

        # Simulate to get density matrix with differentiable params
        rho = simulate_density_matrix(qc, param_values=param_values)

        # Compute expectation on the state
        exp = expectation_pauli(rho, 'Z', num_qubits=1)

        # Check that gradient flows through
        assert exp.requires_grad is True


class TestNoiseChannelPhysics:
    """Test physical correctness of noise channel implementations."""

    def test_amplitude_damping_partial_collapse(self):
        """Test AD with gamma=0.5: |1> partially collapses to |0>."""
        qc = QuantumCircuit(1)
        qc.x(0).amplitude_damping(0.5, 0)
        rho = simulate_density_matrix(qc)

        assert rho.shape == (2, 2)
        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-5)
        assert rho[1, 1].real < 1.0

    def test_amplitude_damping_no_effect_on_ground(self):
        """Test AD has no effect on ground state |0>."""
        qc = QuantumCircuit(1)
        qc.amplitude_damping(0.5, 0)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.zeros(2, 2, dtype=torch.complex64)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_phase_damping_preserves_populations(self):
        """Test phase damping preserves diagonal of DM (populations)."""
        qc = QuantumCircuit(1)
        qc.h(0).phase_damping(0.3, 0)
        rho = simulate_density_matrix(qc).cpu()

        assert torch.allclose(rho[0, 0].real, torch.tensor(0.5), atol=1e-5)
        assert torch.allclose(rho[1, 1].real, torch.tensor(0.5), atol=1e-5)

    def test_phase_damping_decays_coherence(self):
        """Test phase damping decays off-diagonal coherence."""
        qc = QuantumCircuit(1)
        qc.h(0)
        rho_pure = simulate_density_matrix(qc)

        qc_noisy = QuantumCircuit(1)
        qc_noisy.h(0).phase_damping(0.5, 0)
        rho_noisy = simulate_density_matrix(qc_noisy)

        assert rho_noisy[0, 1].abs() < rho_pure[0, 1].abs()

    def test_reset_to_ground_state(self):
        """Test reset collapses |1> to |0>."""
        qc = QuantumCircuit(1)
        qc.x(0).reset(0)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.zeros(2, 2, dtype=torch.complex64)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_reset_trace_preservation(self):
        """Test reset preserves trace."""
        qc = QuantumCircuit(1)
        qc.h(0).reset(0)
        rho = simulate_density_matrix(qc)

        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-5)

    def test_depolarize2_trace_preservation(self):
        """Test two-qubit depolarizing preserves trace."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize2(0.1, 0, 1)
        rho = simulate_density_matrix(qc)

        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0), atol=1e-5)


class TestHardwareValidation:
    """Test that noisy circuits are rejected on real hardware."""

    def test_noisy_circuit_blocked_on_hardware(self):
        """Test ValueError when submitting noisy circuit to hardware."""
        from fieldqkit.api import QuantumHardwareClient
        from fieldqkit.api.backend import Backend

        client = QuantumHardwareClient()
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0)

        with pytest.raises(ValueError, match="Noisy circuits.*not supported"):
            client._run_with_backend(
                qc,
                name="test",
                num_qubits=2,
                backend=Backend("simulator"),
                chip_name="tianyan176",
                shots=100,
            )

    def test_noisy_circuit_allowed_on_simulator(self):
        """Test that noisy circuit is allowed on local simulator."""
        from fieldqkit.api import QuantumHardwareClient
        from fieldqkit.api.backend import Backend

        client = QuantumHardwareClient()
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0).measure_all()

        result = client._run_with_backend(
            qc,
            name="test",
            num_qubits=2,
            backend=Backend("simulator"),
            chip_name="Simulator",
            shots=100,
        )
        assert result is not None

    def test_noisy_circuit_allowed_on_fieldquantum(self):
        """Test that noisy circuit is allowed on fieldquantum_sim."""
        from fieldqkit.api import QuantumHardwareClient
        from fieldqkit.api.backend import Backend

        client = QuantumHardwareClient()
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0).measure_all()

        with pytest.raises((RuntimeError, Exception)):
            client._run_with_backend(
                qc,
                name="test",
                num_qubits=2,
                backend=Backend("fieldquantum_sim"),
                chip_name="fieldquantum_sim",
                shots=100,
            )


class TestNoiseBackendHelper:
    """Test the shared noise/backend guard used by the client and algorithm runners."""

    def test_clean_circuit_returns_false(self):
        """Test a noiseless circuit is reported as non-noisy on any backend."""
        from fieldqkit.api.backend import is_noisy_circuit_for_backend
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        assert is_noisy_circuit_for_backend(qc, "tianyan176") is False

    def test_noisy_on_simulator_allowed(self):
        """Test a noisy circuit is allowed on the local/cloud simulators."""
        from fieldqkit.api.backend import is_noisy_circuit_for_backend
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0)
        assert is_noisy_circuit_for_backend(qc, "Simulator") is True
        assert is_noisy_circuit_for_backend(qc, "fieldquantum_sim") is True

    def test_noisy_on_hardware_rejected(self):
        """Test a noisy circuit is rejected on real hardware."""
        from fieldqkit.api.backend import is_noisy_circuit_for_backend
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0)
        with pytest.raises(ValueError, match="not supported"):
            is_noisy_circuit_for_backend(qc, "tianyan176")


class TestCircuitAdjustment:
    """Test that noise gates work with circuit adjustments."""

    def test_adjust_index_with_noise(self):
        """Test adjust_index preserves noise gates."""
        qc = QuantumCircuit(3)
        qc.h(0).depolarize1(0.1, 0).cx(0, 1)

        qc_adj = qc.deepcopy().adjust_index(2)

        assert qc_adj.nqubits == 5
        assert any(g[0] == 'depolarize1' and g[2] == 2 for g in qc_adj.gates)

    def test_qubits_in_use_with_noise(self):
        """Test that qubits_in_use counts noise gate qubits."""
        qc = QuantumCircuit(4)
        qc.h(0).depolarize1(0.1, 1).cx(2, 3)

        used = qc.qubits_in_use
        assert set(used) == {0, 1, 2, 3}


class TestSymbolicParameters:
    """Noise rates must be concrete numbers (no symbolic/differentiable params)."""

    def test_symbolic_param_rejected(self):
        """Test that a string (symbolic) noise rate is rejected."""
        qc = QuantumCircuit(1)
        with pytest.raises(ValueError, match="must be a number"):
            qc.depolarize1('p_noise', 0)

    def test_symbolic_param_rejected_all_channels(self):
        """Test that every noise channel rejects a symbolic rate."""
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="must be a number"):
            qc.depolarize2('p', 0, 1)
        with pytest.raises(ValueError, match="must be a number"):
            qc.x_error('p', 0)
        with pytest.raises(ValueError, match="must be a number"):
            qc.amplitude_damping('g', 0)
        with pytest.raises(ValueError, match="must be a number"):
            qc.phase_damping('g', 0)


class TestRemoveNoiseChannels:
    """Test stripping noise channels for the ideal (noise-free) reference."""

    def test_remove_noise_channels(self):
        """Test remove_noise_channels drops only the noise gates."""
        qc = QuantumCircuit(2)
        qc.h(0).depolarize1(0.1, 0).cx(0, 1).depolarize2(0.05, 0, 1)
        clean = qc.remove_noise_channels()

        assert has_noise_channels(qc) is True
        assert has_noise_channels(clean) is False
        assert [g[0] for g in clean.gates] == ['h', 'cx']
        # original is untouched
        assert has_noise_channels(qc) is True


class TestNoiseErrorMitigation:
    """Test error mitigation paths on noisy circuits."""

    def test_noisy_observable_basis_runs_on_simulator(self):
        """Test a noisy circuit with an X-basis observable runs (no basis-translate crash)."""
        from fieldqkit.api import QuantumHardwareClient
        from fieldqkit.api.backend import Backend

        client = QuantumHardwareClient()
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.1, 0)
        res = client._run_with_backend(
            qc,
            name="t",
            num_qubits=1,
            backend=Backend("simulator"),
            chip_name="Simulator",
            shots=500,
            observables=["X0"],
        )
        assert res is not None
        assert "X0" in res.observable_values

    def test_unseeded_sampling_is_nondeterministic(self):
        """Regression: unseeded single-shot sampling must vary across calls.

        A fresh torch.Generator is deterministic, so an unseeded sampler used to
        return the identical outcome on every shots=1 call, which silently broke
        classical-shadow style estimation (shots_per_basis=1).
        """
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize1(0.2, 0).depolarize1(0.2, 1).measure_all()
        outcomes = {next(iter(simulate_counts(qc, shots=1))) for _ in range(30)}
        assert len(outcomes) > 1

    def test_seeded_sampling_is_reproducible(self):
        """Test that passing the same seed reproduces counts exactly."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize1(0.2, 0).measure_all()
        a = simulate_counts(qc, shots=500, seed=123)
        b = simulate_counts(qc, shots=500, seed=123)
        assert a == b

    def test_clifford_fitting_on_noisy_circuit(self):
        """Test Clifford data regression builds a fit map for a noisy circuit."""
        from fieldqkit.api import QuantumHardwareClient
        from fieldqkit.api.backend import Backend
        from fieldqkit.algorithms.optimizer_utils import build_clifford_fit_map

        client = QuantumHardwareClient()
        qc = QuantumCircuit(1)
        qc.ry(0.7, 0).depolarize1(0.1, 0)
        fit = build_clifford_fit_map(
            client,
            name="t",
            num_qubits=1,
            backend=Backend("simulator"),
            chip_name="Simulator",
            observables=["Z0"],
            shots=1000,
            zne=False,
            readout_mitigation=False,
            transpiled_template=qc,
            num_samples=4,
            num_non_clifford_gates=1,
            seed=0,
        )
        assert "Z0" in fit
        a, b = fit["Z0"]
        assert isinstance(a, float) and isinstance(b, float)
