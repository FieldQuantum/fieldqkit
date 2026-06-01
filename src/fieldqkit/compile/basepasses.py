"""This module contains the TranspilerPass class, an abstract base class for defining
transpiler passes that transform or optimize quantum circuits.

SPDX-License-Identifier: Apache-2.0
Original source: quarkstudio, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

from abc import ABC, abstractmethod

from ..circuit import QuantumCircuit


class TranspilerPass(ABC):
    """Abstract base class for defining transpiler passes that transform or optimize quantum circuits."""

    def __init__(self):
        """Initialize the transpiler pass.
        """
        pass

    @abstractmethod
    def run(self, qc: QuantumCircuit):
        """Execute the transpiler pass on a quantum circuit.

        Args:
            qc (QuantumCircuit): The quantum circuit to be processed by this transpiler pass.
        """
        pass
