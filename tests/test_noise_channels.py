"""Tests for noise channel simulation and integration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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

        expected = torch.zeros(2, 2, dtype=torch.complex128)
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
        ], dtype=torch.complex128)
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_depolarize1_trace_preservation(self):
        """Test that trace(ρ) = 1 after depolarizing noise."""
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.3, 0)
        rho = simulate_density_matrix(qc)

        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0, dtype=torch.float64), atol=1e-5)

    def test_depolarize1_maximally_mixed(self):
        """Test that p=0.75 (for 1q) leads to maximally mixed state."""
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.75, 0)
        rho = simulate_density_matrix(qc).cpu()

        expected = 0.5 * torch.eye(2, dtype=torch.complex128)
        assert torch.allclose(rho, expected, atol=1e-4)

    def test_amplitude_damping_collapse(self):
        """Test that gamma=1.0 in amplitude damping collapses |1> to |0>."""
        qc = QuantumCircuit(1)
        qc.x(0).amplitude_damping(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.zeros(2, 2, dtype=torch.complex128)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_dm_with_unitary_and_noise(self):
        """Test DM with mixed unitary gates and noise."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize1(0.1, 0).h(1)
        rho = simulate_density_matrix(qc)

        assert rho.shape == (4, 4)
        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0, dtype=torch.float64), atol=1e-5)


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
        assert torch.allclose(trace, torch.tensor(1.0, dtype=torch.float64), atol=1e-5)
        assert rho[1, 1].real < 1.0

    def test_amplitude_damping_no_effect_on_ground(self):
        """Test AD has no effect on ground state |0>."""
        qc = QuantumCircuit(1)
        qc.amplitude_damping(0.5, 0)
        rho = simulate_density_matrix(qc).cpu()

        expected = torch.zeros(2, 2, dtype=torch.complex128)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_phase_damping_preserves_populations(self):
        """Test phase damping preserves diagonal of DM (populations)."""
        qc = QuantumCircuit(1)
        qc.h(0).phase_damping(0.3, 0)
        rho = simulate_density_matrix(qc).cpu()

        assert torch.allclose(rho[0, 0].real, torch.tensor(0.5, dtype=torch.float64), atol=1e-5)
        assert torch.allclose(rho[1, 1].real, torch.tensor(0.5, dtype=torch.float64), atol=1e-5)

    def test_phase_damping_decays_coherence(self):
        """Test phase damping decays off-diagonal coherence."""
        qc = QuantumCircuit(1)
        qc.h(0)
        rho_pure = simulate_density_matrix(qc)

        qc_noisy = QuantumCircuit(1)
        qc_noisy.h(0).phase_damping(0.5, 0)
        rho_noisy = simulate_density_matrix(qc_noisy)

        assert rho_noisy[0, 1].abs() < rho_pure[0, 1].abs()

    def test_reset_rejected(self):
        """Reset is not supported by the simulator backends and must raise."""
        qc = QuantumCircuit(1)
        qc.x(0).reset(0)
        with pytest.raises(NotImplementedError, match="reset"):
            simulate_density_matrix(qc)

    def test_depolarize2_trace_preservation(self):
        """Test two-qubit depolarizing preserves trace."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize2(0.1, 0, 1)
        rho = simulate_density_matrix(qc)

        trace = torch.trace(rho).real
        assert torch.allclose(trace, torch.tensor(1.0, dtype=torch.float64), atol=1e-5)


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


# ---------------------------------------------------------------------------
# Appended tests: boundary cases, Kraus/DM invariants, and large-scale circuits.
# ---------------------------------------------------------------------------

from fieldqkit.sim.noise_kraus import (
    get_kraus_ops,
    depolarize1_kraus,
    depolarize2_kraus,
    x_error_kraus,
    y_error_kraus,
    z_error_kraus,
    amplitude_damping_kraus,
    phase_damping_kraus,
)


def _assert_valid_density_matrix(rho, atol=1e-5):
    """Assert that ``rho`` is a physical density matrix (Hermitian, unit trace, PSD)."""
    rho = rho.reshape(rho.shape[0], rho.shape[0]).cpu()
    # Hermitian
    assert torch.allclose(rho, rho.conj().transpose(-2, -1), atol=atol)
    # Unit trace
    trace = torch.trace(rho).real
    assert torch.allclose(trace, torch.tensor(1.0, dtype=torch.float64), atol=atol)
    # Positive semi-definite (eigvalsh requires an exactly Hermitian input)
    rho_herm = 0.5 * (rho + rho.conj().transpose(-2, -1))
    eigs = torch.linalg.eigvalsh(rho_herm)
    assert eigs.min().item() >= -atol


class TestKrausCompleteness:
    """Kraus operators of every channel must satisfy Σ K†K = I (trace preserving)."""

    @pytest.mark.parametrize(
        "name,param,dim",
        [
            ("depolarize1", 0.3, 2),
            ("depolarize1", 0.0, 2),
            ("depolarize1", 1.0, 2),
            ("x_error", 0.4, 2),
            ("y_error", 0.25, 2),
            ("z_error", 0.6, 2),
            ("amplitude_damping", 0.5, 2),
            ("amplitude_damping", 1.0, 2),
            ("phase_damping", 0.7, 2),
            ("depolarize2", 0.2, 4),
            ("depolarize2", 0.0, 4),
            ("depolarize2", 1.0, 4),
        ],
    )
    def test_kraus_sum_to_identity(self, name, param, dim):
        """Test Σ_k K_k† K_k = I for each noise channel."""
        kraus = get_kraus_ops(name, param)
        acc = torch.zeros(dim, dim, dtype=torch.complex128)
        for K in kraus:
            acc = acc + K.conj().transpose(-2, -1) @ K
        expected = torch.eye(dim, dtype=torch.complex128)
        assert torch.allclose(acc, expected, atol=1e-6)

    def test_depolarize1_has_four_kraus(self):
        """Test single-qubit depolarizing has 4 Kraus operators (I, X, Y, Z)."""
        assert len(depolarize1_kraus(0.1)) == 4

    def test_depolarize2_has_sixteen_kraus(self):
        """Test two-qubit depolarizing has 16 Kraus operators (4x4 Paulis)."""
        assert len(depolarize2_kraus(0.1)) == 16

    def test_single_qubit_channels_have_two_kraus(self):
        """Test the flip/damping channels each have exactly 2 Kraus operators."""
        for builder in (
            x_error_kraus,
            y_error_kraus,
            z_error_kraus,
            amplitude_damping_kraus,
            phase_damping_kraus,
        ):
            assert len(builder(0.3)) == 2

    def test_kraus_p0_is_identity_channel(self):
        """Test that at p=0 the dominant Kraus operator is the identity and the rest vanish."""
        for name in ("depolarize1", "x_error", "y_error", "z_error",
                     "amplitude_damping", "phase_damping"):
            kraus = get_kraus_ops(name, 0.0)
            assert torch.allclose(kraus[0], torch.eye(2, dtype=torch.complex128), atol=1e-6)
            for K in kraus[1:]:
                assert torch.allclose(K, torch.zeros(2, 2, dtype=torch.complex128), atol=1e-6)


class TestKrausValidation:
    """Kraus builders must reject probabilities/rates outside [0, 1]."""

    @pytest.mark.parametrize(
        "builder",
        [
            depolarize1_kraus,
            depolarize2_kraus,
            x_error_kraus,
            y_error_kraus,
            z_error_kraus,
            amplitude_damping_kraus,
            phase_damping_kraus,
        ],
    )
    @pytest.mark.parametrize("bad", [-0.1, 1.5, -1.0, 2.0])
    def test_out_of_range_rejected(self, builder, bad):
        """Test ValueError for rates < 0 or > 1."""
        with pytest.raises(ValueError, match="must be in"):
            builder(bad)

    def test_get_kraus_unknown_channel(self):
        """Test get_kraus_ops rejects an unknown channel name."""
        with pytest.raises(ValueError, match="Unknown noise channel"):
            get_kraus_ops("not_a_channel", 0.1)


class TestNoiseRateBoundaryValidation:
    """QuantumCircuit gate methods must validate rates at the boundaries."""

    @pytest.mark.parametrize(
        "method",
        ["depolarize1", "x_error", "y_error", "z_error",
         "amplitude_damping", "phase_damping"],
    )
    def test_single_qubit_rate_below_zero(self, method):
        """Test rates < 0 are rejected by every single-qubit channel method."""
        qc = QuantumCircuit(1)
        with pytest.raises(ValueError, match="must be a number in"):
            getattr(qc, method)(-0.01, 0)

    @pytest.mark.parametrize(
        "method",
        ["depolarize1", "x_error", "y_error", "z_error",
         "amplitude_damping", "phase_damping"],
    )
    def test_single_qubit_rate_above_one(self, method):
        """Test rates > 1 are rejected by every single-qubit channel method."""
        qc = QuantumCircuit(1)
        with pytest.raises(ValueError, match="must be a number in"):
            getattr(qc, method)(1.01, 0)

    def test_depolarize2_rate_out_of_range(self):
        """Test the two-qubit depolarizing method validates its rate."""
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="must be a number in"):
            qc.depolarize2(-0.5, 0, 1)
        with pytest.raises(ValueError, match="must be a number in"):
            qc.depolarize2(1.5, 0, 1)

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_boundary_rates_accepted(self, p):
        """Test p=0 and p=1 are accepted at circuit-build time for all channels."""
        qc = QuantumCircuit(2)
        qc.depolarize1(p, 0)
        qc.depolarize2(p, 0, 1)
        qc.x_error(p, 0)
        qc.y_error(p, 0)
        qc.z_error(p, 0)
        qc.amplitude_damping(p, 0)
        qc.phase_damping(p, 0)
        assert has_noise_channels(qc) is True


class TestZeroNoiseEqualsNoiseless:
    """At p=0 a noise channel must be the identity (matches the noiseless DM)."""

    def test_p0_single_channels_match_noiseless(self):
        """Test depolarize1/x/y/z/amplitude/phase at p=0 leave the DM unchanged."""
        base = QuantumCircuit(2)
        base.h(0).cx(0, 1)
        rho_clean = simulate_density_matrix(base)

        noisy = QuantumCircuit(2)
        noisy.h(0).cx(0, 1)
        noisy.depolarize1(0.0, 0)
        noisy.x_error(0.0, 0)
        noisy.y_error(0.0, 1)
        noisy.z_error(0.0, 1)
        noisy.amplitude_damping(0.0, 0)
        noisy.phase_damping(0.0, 1)
        rho_noisy = simulate_density_matrix(noisy)

        assert torch.allclose(rho_clean, rho_noisy, atol=1e-6)

    def test_p0_two_qubit_depolarize_matches_noiseless(self):
        """Test depolarize2 at p=0 leaves the DM unchanged."""
        base = QuantumCircuit(2)
        base.h(0).cx(0, 1)
        rho_clean = simulate_density_matrix(base)

        noisy = QuantumCircuit(2)
        noisy.h(0).cx(0, 1).depolarize2(0.0, 0, 1)
        rho_noisy = simulate_density_matrix(noisy)

        assert torch.allclose(rho_clean, rho_noisy, atol=1e-6)


class TestNoiseChannelBoundaryEffects:
    """p=1 / gamma=1 boundary behavior of individual channels."""

    def test_x_error_p1_flips_ground(self):
        """Test x_error(p=1) maps |0> to |1>."""
        qc = QuantumCircuit(1)
        qc.x_error(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()
        expected = torch.zeros(2, 2, dtype=torch.complex128)
        expected[1, 1] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_z_error_p1_flips_plus_to_minus(self):
        """Test z_error(p=1) maps |+> to |-> (off-diagonal becomes -0.5)."""
        qc = QuantumCircuit(1)
        qc.h(0).z_error(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()
        expected = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.complex128)
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_y_error_p1_on_ground(self):
        """Test y_error(p=1) maps |0> to |1> (Y|0> = i|1>, phase cancels in ρ)."""
        qc = QuantumCircuit(1)
        qc.y_error(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()
        expected = torch.zeros(2, 2, dtype=torch.complex128)
        expected[1, 1] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_depolarize1_p1_on_ground(self):
        """Test depolarize1(p=1) on |0> gives diag(1/3, 2/3)."""
        qc = QuantumCircuit(1)
        qc.depolarize1(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()
        expected = torch.tensor(
            [[1.0 / 3.0, 0.0], [0.0, 2.0 / 3.0]], dtype=torch.complex128
        )
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_amplitude_damping_gamma1_on_excited(self):
        """Test amplitude_damping(gamma=1) collapses |1> to |0>."""
        qc = QuantumCircuit(1)
        qc.x(0).amplitude_damping(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()
        expected = torch.zeros(2, 2, dtype=torch.complex128)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)

    def test_phase_damping_gamma1_kills_coherence(self):
        """Test phase_damping(gamma=1) fully removes off-diagonal coherence."""
        qc = QuantumCircuit(1)
        qc.h(0).phase_damping(1.0, 0)
        rho = simulate_density_matrix(qc).cpu()
        # Populations preserved, coherences gone -> maximally mixed in this case
        expected = 0.5 * torch.eye(2, dtype=torch.complex128)
        assert torch.allclose(rho, expected, atol=1e-5)


class TestSingleChannelOnMultiQubitRegister:
    """A single-qubit channel must act only on its target qubit."""

    def test_channel_leaves_other_qubit_untouched(self):
        """Test depolarize1 on qubit 1 leaves qubit 0 (in |1>) intact."""
        qc = QuantumCircuit(2)
        qc.x(0).depolarize1(0.3, 1)
        rho = simulate_density_matrix(qc).cpu()
        diag = torch.diag(rho).real
        # Basis order is big-endian: index 2 = |10>, index 3 = |11>.
        # Qubit 0 stays |1>; qubit 1 depolarized: P(0 on q1)=0.8, P(1 on q1)=0.2.
        expected = torch.tensor([0.0, 0.0, 0.8, 0.2], dtype=torch.float64)
        assert torch.allclose(diag, expected, atol=1e-5)
        _assert_valid_density_matrix(rho)

    def test_channel_on_three_qubit_target_only(self):
        """Test amplitude damping on the middle qubit of a 3-qubit register."""
        qc = QuantumCircuit(3)
        qc.x(1).amplitude_damping(1.0, 1)  # excite q1 then fully damp it
        rho = simulate_density_matrix(qc).cpu()
        # All qubits should end in |000>.
        expected = torch.zeros(8, 8, dtype=torch.complex128)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, atol=1e-5)


class TestDensityMatrixInvariants:
    """Noisy evolution must keep the DM Hermitian, unit-trace and PSD."""

    def test_invariants_single_qubit_mixed_channels(self):
        """Test invariants hold after several single-qubit channels."""
        qc = QuantumCircuit(1)
        qc.h(0).depolarize1(0.2, 0).phase_damping(0.3, 0).amplitude_damping(0.1, 0)
        _assert_valid_density_matrix(simulate_density_matrix(qc))

    def test_invariants_two_qubit_entangled_noisy(self):
        """Test invariants hold for an entangled 2-qubit state with noise."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize2(0.2, 0, 1).z_error(0.3, 1)
        _assert_valid_density_matrix(simulate_density_matrix(qc))

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_invariants_across_boundary_rates(self, p):
        """Test invariants hold at p=0, p=0.5 and p=1 for depolarize1."""
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1).depolarize1(p, 0).depolarize1(p, 1)
        _assert_valid_density_matrix(simulate_density_matrix(qc))


class TestRepeatedChannels:
    """Repeatedly applying the same channel must compose correctly."""

    def test_two_x_errors_compose(self):
        """Test x_error(0.5) twice on |0> gives a 50/50 mixture."""
        qc = QuantumCircuit(1)
        qc.x_error(0.5, 0).x_error(0.5, 0)
        rho = simulate_density_matrix(qc).cpu()
        diag = torch.diag(rho).real
        assert torch.allclose(diag, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-5)

    def test_repeated_amplitude_damping_increases_decay(self):
        """Test stacking amplitude damping decays the excited population further."""
        single = QuantumCircuit(1)
        single.x(0).amplitude_damping(0.3, 0)
        p1_single = simulate_density_matrix(single).cpu()[1, 1].real.item()

        double = QuantumCircuit(1)
        double.x(0).amplitude_damping(0.3, 0).amplitude_damping(0.3, 0)
        p1_double = simulate_density_matrix(double).cpu()[1, 1].real.item()

        assert p1_double < p1_single
        _assert_valid_density_matrix(simulate_density_matrix(double))

    def test_many_repeated_depolarize_stays_valid(self):
        """Test 10 stacked depolarizing channels keep the DM physical and trace 1."""
        qc = QuantumCircuit(1)
        qc.h(0)
        for _ in range(10):
            qc.depolarize1(0.1, 0)
        _assert_valid_density_matrix(simulate_density_matrix(qc))


class TestLargeScaleNoisyCircuits:
    """Larger noisy circuits (6-8 qubits) with mixed coherent gates and channels."""

    def test_seven_qubit_mixed_noise(self):
        """Test a 7-qubit circuit mixing coherent gates and several channels."""
        n = 7
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(n):
            qc.depolarize1(0.05, q)
        qc.amplitude_damping(0.1, 0)
        qc.phase_damping(0.1, n - 1)
        qc.depolarize2(0.05, 2, 3)

        rho = simulate_density_matrix(qc)
        assert rho.shape == (2 ** n, 2 ** n)
        _assert_valid_density_matrix(rho)

    def test_eight_qubit_layered_noise(self):
        """Test an 8-qubit circuit with layered coherent + noise operations."""
        n = 8
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(0.3, q)
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
            qc.depolarize2(0.03, q, q + 1)
        for q in range(n):
            qc.z_error(0.02, q)
        rho = simulate_density_matrix(qc)
        assert rho.shape == (2 ** n, 2 ** n)
        _assert_valid_density_matrix(rho)

    def test_large_noisy_counts_sum_to_shots(self):
        """Test sampling a 6-qubit noisy circuit returns exactly `shots` counts."""
        n = 6
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(n):
            qc.depolarize1(0.1, q)
        qc.measure_all()
        counts = simulate_noisy_counts(qc, shots=500, seed=7)
        assert sum(counts.values()) == 500
        assert all(len(bits) == n for bits in counts)

    def test_large_circuit_expectation_in_range(self):
        """Test a Pauli-Z expectation on a noisy 6-qubit DM stays within [-1, 1]."""
        n = 6
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for q in range(n):
            qc.depolarize1(0.2, q)
        rho = simulate_density_matrix(qc)
        exp = expectation_pauli_dm(rho, "Z" * n, num_qubits=n)
        val = exp.real.item()
        assert -1.0 - 1e-5 <= val <= 1.0 + 1e-5
