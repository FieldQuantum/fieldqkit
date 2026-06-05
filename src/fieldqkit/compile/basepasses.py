"""This module contains the TranspilerPass class, an abstract base class for defining
transpiler passes that transform or optimize quantum circuits.

SPDX-License-Identifier: Apache-2.0 AND MIT
Modified from quarkcircuit by FieldQuantum; this file has been altered from the
original and is redistributed as part of fieldqkit under the Apache License,
Version 2.0. The original work is licensed under the MIT License:

Copyright (c) 2024 XX Xiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
