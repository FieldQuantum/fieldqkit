r"""This module contains the Transpiler class, which is designed to convert quantum circuits
into formats that are more suitable for execution on hardware backends.

SPDX-License-Identifier: MIT
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

from __future__ import annotations

import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

from ..api.backend import Backend

from ..circuit import QuantumCircuit
from .decompose import ThreeQubitGateDecompose
from .layout import Layout
from .routing import SabreRouting
from .translate import TranslateToBasisGates
from .optimize import GateCompressor
from .schedule import DynamicalDecoupling
from .dag import split_qubits


class Transpiler:
    r"""The transpilation process involves converting the operations
    in the circuit to those supported by the device and swapping
    qubits (via swap gates) within the circuit to overcome limited
    qubit connectivity.
    """

    def __init__(self, chip_backend: Backend | None = None, *, convert_single_qubit_gate_to_u: bool | None = None):
        """Initialize the transpiler with optional backend and gate conversion settings.

        Args:
            chip_backend (*Backend | None*): Hardware backend providing topology and fidelity data. Defaults to ``None``.
            convert_single_qubit_gate_to_u (*bool | None*): Whether to convert single-qubit gates to U gates. Defaults to ``None``.
        """
        self.chip_backend = chip_backend
        self._convert_single_qubit_gate_to_u_override = convert_single_qubit_gate_to_u

    def run(
        self,
        qc: QuantumCircuit,
        target_qubits: list | None = None,
        niter: int = 5,
        use_dd: bool = True,
        use_three_qubit_decompose: bool = True,
        use_sabre_routing: bool = True,
        use_translate_to_basis: bool = True,
        use_gate_compressor: bool = True,
        routing_initial_mapping: str = "random",
        routing_random_choice: bool = True,
        noise_aware: bool | None = None,
        routing_n_trials: int = 1,
    ):
        """Execute the full transpilation pipeline on a quantum circuit.

        Args:
            qc (*QuantumCircuit*): Quantum circuit.
            target_qubits (*list | None*): Physical qubit indices for layout selection. Defaults to ``None``.
            niter (*int*): Number of SABRE routing iterations (forward+reverse passes). Defaults to ``5``.
            use_dd (*bool*): Whether to insert dynamical decoupling sequences into idle windows. Defaults to ``True``.
            use_three_qubit_decompose (*bool*): Whether to decompose three-qubit gates into one- and two-qubit gates. Defaults to ``True``.
            use_sabre_routing (*bool*): Whether to run SABRE routing for qubit connectivity mapping. Defaults to ``True``.
            use_translate_to_basis (*bool*): Whether to translate gates to the backend's native basis gate set. Defaults to ``True``.
            use_gate_compressor (*bool*): Whether to compress adjacent compatible gates. Defaults to ``True``.
            routing_initial_mapping (*str*): Initial qubit mapping strategy: ``'trivial'``, ``'random'``, or an explicit list. Defaults to ``'random'``.
            routing_random_choice (*bool*): Whether to randomly select among equally-scored SWAP candidates. Defaults to ``True``.
            noise_aware (*bool | None*): Whether to use noise-aware strategies. Defaults to ``None``.
            routing_n_trials (*int*): Number of independent SABRE trials to run, keeping the best result. Defaults to ``1``.

        Returns:
            *QuantumCircuit*: The transpiled quantum circuit ready for hardware execution.

        Raises:
            TypeError: If *qc* is not a ``QuantumCircuit``.
            ValueError: If quantum circuit can be divided along the qubits, random initial mapping is restricted.
        """
        if isinstance(qc, QuantumCircuit):
            pass
        else:
            raise TypeError("Expected a QuantumCircuit, but got a {}.".format(type(qc)))

        if target_qubits is None:
            target_qubits = []

        if self.chip_backend is None:
            logger.warning("No chip specified, defaulting to a linearly connected layout for simulation.")
            import networkx as nx

            # Build a simple line topology based on logical qubit order.
            subgraph = nx.Graph()
            qubits = list(sorted(qc.qubits))
            subgraph.add_edges_from([(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)])
            subgraph.graph["normal_order"] = qc.qubits
            self.two_qubit_gate_basis = "cx"
            self.convert_single_qubit_gate_to_u = False
        else:
            # Use the backend's basis and topology-aware layout selection.
            self.two_qubit_gate_basis = self.chip_backend.two_qubit_gate_basis
            self.convert_single_qubit_gate_to_u = True if self._convert_single_qubit_gate_to_u_override is None else self._convert_single_qubit_gate_to_u_override
            subgraph = Layout(self.chip_backend).select_layout(
                qc,
                target_qubits=target_qubits,
                use_chip_priority=True,
                select_criteria={
                    "key": "fidelity_var",
                    "topology": "linear",
                },
            )
        if use_sabre_routing and routing_initial_mapping == "random":
            if len(split_qubits(qc)) > 1:
                raise ValueError(
                    "If quantum circuit can be divided along the qubits, random initial mapping is restricted."
                )

        # Default noise_aware: True when using a real backend, False otherwise.
        # For the built-in Simulator all fidelities are 1.0 so noise-aware
        # routing provides no benefit (and would degenerate).
        if noise_aware is None:
            if self.chip_backend is None:
                noise_aware = False
            elif getattr(self.chip_backend, 'chip_name', '') in ('Simulator',):
                noise_aware = False
            else:
                noise_aware = True

        passes = []
        if use_three_qubit_decompose:
            passes.append(ThreeQubitGateDecompose())
        if use_sabre_routing:
            passes.append(
                SabreRouting(
                    subgraph,
                    initial_mapping=routing_initial_mapping,
                    do_random_choice=routing_random_choice,
                    iterations=niter,
                    noise_aware=noise_aware,
                    n_trials=routing_n_trials,
                )
            )
        if use_translate_to_basis:
            passes.append(
                TranslateToBasisGates(
                    convert_single_qubit_gate_to_u=self.convert_single_qubit_gate_to_u,
                    two_qubit_gate_basis=self.two_qubit_gate_basis,
                )
            )
        if use_gate_compressor:
            passes.append(GateCompressor())
        if use_dd:
            try:
                # Append dynamical decoupling using backend gate lengths.
                t1g = self.chip_backend.chip_info["global_info"]["one_qubit_gate_length"]
                t2g = self.chip_backend.chip_info["global_info"]["two_qubit_gate_length"]
                passes.append(DynamicalDecoupling(t1g, t2g))
            except Exception:
                pass
        for pass_obj in passes:
            prev_qc = qc
            qc = pass_obj.run(qc)
            # Keep routing layout metadata available to API-layer measurement mapping.
            for attr in ("logical_to_physical", "physical_to_logical"):
                if hasattr(prev_qc, attr):
                    setattr(qc, attr, deepcopy(getattr(prev_qc, attr)))
        return qc
