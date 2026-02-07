# Copyright (c) 2024 XX Xiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

r""" 
This module contains the QuantumCircuit class, which offers an intuitive interface for designing, visualizing, 
and converting quantum circuits in various formats such as OpenQASM 2.0 and 3.0.
"""

import copy
from typing import Iterable, Optional
import numpy as np
from .quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    three_qubit_gates_available,
    functional_gates_available,
    convert_gate_info_to_dag_info,
    add_gates_to_lines,
    )
from .qasm2 import parse_openqasm2_to_gates
from .qasm3 import parse_openqasm3_to_gates
from .render import draw_circuit, draw_circuit_simply
from .utils import u3_decompose, zyz_decompose, kak_decompose
from .matrix import h_mat

def generate_ghz_state(nqubits: int) -> 'QuantumCircuit':
    r"""
    Produce a GHZ state on n qubits.

    Args:
        nqubits (int): The number of qubits. Must be >= 2.

    Returns:
        QuantumCircuit: A quantum circuit representing the GHZ state.
    """
    cir =  QuantumCircuit(nqubits)
    cir.h(0)
    for i in range(1,nqubits):
        cir.cx(0,i)
    return cir

class QuantumCircuit:
    r"""
    A class used to build and manipulate a quantum circuit.

    This class allows you to create quantum circuits with a specified number of quantum and classical bits. 
    The circuit can be customized using various quantum gates, and additional features (such as simulation support, 
    circuit summary, and more) will be added in future versions.
    
    Attributes:
        nqubits (int or None): Number of quantum bits in the circuit.
        ncbits (int or None): Number of classical bits in the circuit.
    """
    def __init__(self, *args):
        r"""
        Initialize a QuantumCircuit object.

        The constructor supports three different initialization modes:
        1. `QuantumCircuit()`: Creates a circuit with `nqubits` and `ncbits` both set to `None`.
        2. `QuantumCircuit(nqubits)`: Creates a circuit with the specified number of quantum bits (`nqubits`), 
        and classical bits (`ncbits`) set to the same value as `nqubits`.
        3. `QuantumCircuit(nqubits, ncbits)`: Creates a circuit with the specified number of quantum bits (`nqubits`) 
        and classical bits (`ncbits`).

        Args:
            *args: Variable length argument list used to specify the number of qubits and classical bits.

        Raises:
            ValueError: If more than two arguments are provided, or if the arguments are not in one of the specified valid forms.
        """
        if len(args) == 0:
            self.nqubits = 0
            self.ncbits = 0
        elif len(args) == 1:
            self.nqubits = args[0]
            self.ncbits = self.nqubits
        elif len(args) == 2:
            self.nqubits = args[0]
            self.ncbits = args[1]
        else:
            raise ValueError("Support only QuantumCircuit(), QuantumCircuit(nqubits) or QuantumCircuit(nqubits,ncbits).")
        
        self.qubits = []
        self.gates = []
        self.params_value = {}

    def deepcopy(self) -> 'QuantumCircuit':
        new_qc = QuantumCircuit(self.nqubits,self.ncbits)
        new_qc.qubits = copy.deepcopy(self.qubits)
        new_qc.params_value = copy.deepcopy(self.params_value)
        new_qc.gates = copy.deepcopy(self.gates)
        return new_qc

    def adjust_index(self, num: int, *, cbit_offset: Optional[int] = None):
        if cbit_offset is None:
            cbit_offset = num
        gates = []
        for gate_info in self.gates:
            gate = gate_info[0]
            if gate in one_qubit_gates_available.keys():
                qubit = gate_info[-1] + num
                gates.append((gate,qubit))
            elif gate in two_qubit_gates_available.keys():
                qubit1 = gate_info[1] + num
                qubit2 = gate_info[2] + num
                gates.append((gate,qubit1,qubit2))
            elif gate in one_qubit_parameter_gates_available.keys():
                qubit = gate_info[-1] + num
                gates.append((gate,*gate_info[1:-1],qubit))
            elif gate in ['reset']:
                qubit = gate_info[-1] + num
                gates.append((gate,qubit))
            elif gate in ['barrier']:
                qubits = [idx + num for idx in gate_info[1]]
                gates.append((gate,tuple(qubits)))
            elif gate in ['measure']:
                qubits = [idx + num for idx in gate_info[1]]
                cbits = [idx + cbit_offset for idx in gate_info[-1]]
                gates.append((gate,qubits,cbits))
        self.gates = gates   
        self.nqubits = self.nqubits + num
        self.ncbits = self.ncbits + cbit_offset
        self.qubits = [idx + num for idx in self.qubits] 

    @property
    def cbits(self):
        cbits = []
        for gate_info in self.gates:
            if gate_info[0] == 'measure':
                for cbit in gate_info[2]:
                    cbits.append(cbit)
                    
        return sorted(set(cbits))

    def _add_qubits(self,*args):
        # Deduplicate and sort qubits.
        temp_set = set(self.qubits).union(args)
        self.qubits = sorted(temp_set)
        return self

    def _resolve_param(self, param):
        if isinstance(param, (float, int)):
            return float(param)
        if isinstance(param, str):
            if param not in self.params_value:
                raise ValueError(f"please apply value for parameter {param}")
            value = self.params_value[param]
            if isinstance(value, (float, int)):
                return float(value)
            raise ValueError(f"please apply value for parameter {value}")
        raise TypeError(f"Wrong param type! {param}")

    def _resolve_param_list(self, params):
        return [self._resolve_param(param) for param in params]

    def from_openqasm2(self,openqasm2_str: str) -> 'QuantumCircuit':
        r"""
        Initializes the QuantumCircuit object based on the given OpenQASM 2.0 string.

        Args:
            openqasm2_str (str): A string representing a quantum circuit in OpenQASM 2.0 format.
        """
        assert('OPENQASM 2.0' in openqasm2_str)
        new_gates,qubit_used,cbit_used = parse_openqasm2_to_gates(openqasm2_str)
        self.nqubits = max(qubit_used) + 1 if qubit_used else 0
        self.ncbits = max(cbit_used) + 1 if cbit_used else 0
        self.qubits = list(qubit_used) #[i for i in range(self.nqubits)]
        self.gates = new_gates
        return self
    
    def from_openqasm3(self, openqasm3_str: str) -> 'QuantumCircuit':
        r"""
        Initializes the QuantumCircuit object based on the given OpenQASM 3 string.

        Args:
            openqasm3_str (str): A string representing a quantum circuit in OpenQASM 3 format.
        """
        assert('OPENQASM 3.0' in openqasm3_str)
        new_gates, qubit_used, cbit_used = parse_openqasm3_to_gates(openqasm3_str)
        self.nqubits = max(qubit_used) + 1 if qubit_used else 0
        self.ncbits = max(cbit_used) + 1 if cbit_used else 0
        self.qubits = list(qubit_used)
        self.gates = new_gates
        return self
    
    def id(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a Identity gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('id', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def x(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a X gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('x', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def y(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a Y gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('y', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def z(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a Z gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('z', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def s(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a S gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('s', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def sdg(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a S dagger gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('sdg', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def sx(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a Sqrt(X) gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('sx', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")
        
    def sxdg(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a Sqrt(X) dagger gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('sxdg', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def t(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a T gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('t', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def tdg(self, qubit: int) -> 'QuantumCircuit':
        r"""Add a T dagger gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('tdg', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")
               
    def h(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a H gate.

        Args:
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('h', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")

    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        r"""
        Add a SWAP gate.

        Args:
            qubit1 (int): The first qubit to apply the gate to.
            qubit2 (int): The second qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(qubit1,qubit2) < self.nqubits:
            if qubit1 != qubit2:
                self.gates.append(('swap', qubit1,qubit2))
                self._add_qubits(qubit1,qubit2)
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")
        
    def iswap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        r"""
        Add a ISWAP gate.

        Args:
            qubit1 (int): The first qubit to apply the gate to.
            qubit2 (int): The second qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(qubit1, qubit2) < self.nqubits:
            if qubit1 != qubit2:
                self.gates.append(('iswap', qubit1,qubit2))
                self._add_qubits(qubit1,qubit2)
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")
        
    def ecr(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        r"""
        Add a ECR gate.

        Args:
            qubit1 (int): The first qubit to apply the gate to.
            qubit2 (int): The second qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(qubit1, qubit2) < self.nqubits:
            if qubit1 != qubit2:
                self.gates.append(('ecr', qubit1,qubit2))
                self._add_qubits(qubit1,qubit2)
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")
        
    def cx(self, control_qubit: int, target_qubit: int) -> 'QuantumCircuit':
        r"""
        Add a CX gate.

        Args:
            control_qubit (int): The qubit used as control.
            target_qubit (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(control_qubit,target_qubit) < self.nqubits:
            if control_qubit != target_qubit:
                self.gates.append(('cx', control_qubit,target_qubit))
                self._add_qubits(control_qubit,target_qubit)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit and target_qubit are both {control_qubit}")
        else:
            raise ValueError("Qubit index out of range")
        
    def cnot(self, control_qubit: int, target_qubit: int) -> 'QuantumCircuit':
        r"""
        Add a CNOT gate.

        Args:
            control_qubit (int): The qubit used as control.
            target_qubit (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(control_qubit,target_qubit) < self.nqubits:
            if control_qubit != target_qubit:
                self.cx(control_qubit, target_qubit)
                self._add_qubits(control_qubit,target_qubit)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit and target_qubit are both {control_qubit}")
        else:
            raise ValueError("Qubit index out of range")
                
    def cy(self, control_qubit: int, target_qubit: int) -> 'QuantumCircuit':
        r"""
        Add a CY gate.

        Args:
            control_qubit (int): The qubit used as control.
            target_qubit (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(control_qubit,target_qubit) < self.nqubits:
            if control_qubit != target_qubit:
                self.gates.append(('cy', control_qubit,target_qubit))
                self._add_qubits(control_qubit,target_qubit)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit and target_qubit are both {control_qubit}")
        else:
            raise ValueError("Qubit index out of range")
        
    def cz(self, control_qubit: int, target_qubit: int) -> 'QuantumCircuit':
        r"""
        Add a CZ gate.

        Args:
            control_qubit (int): The qubit used as control.
            target_qubit (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(control_qubit,target_qubit) < self.nqubits:
            if control_qubit != target_qubit:
                self.gates.append(('cz', control_qubit, target_qubit))
                self._add_qubits(control_qubit,target_qubit)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit and target_qubit are both {control_qubit}")
        else:
            raise ValueError("Qubit index out of range")

    def ccz(self,control_qubit1:int,control_qubit2:int,target_qubit:int) -> 'QuantumCircuit':
        """Add CCZ gate.

        Args:
            control_qubit1 (int): The qubit used as the first control.
            control_qubit2 (int): The qubit used as the second control.
            target_qubit (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        qubits0 = [control_qubit1,control_qubit2,target_qubit]
        if max(qubits0) < self.nqubits:
            if len(set(qubits0)) == 3:
                self.gates.append(('ccz',control_qubit1,control_qubit2,target_qubit))
                self._add_qubits(*qubits0)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit1 {control_qubit1} control_qubit2 {control_qubit2} target_qubit {target_qubit}")
        else:
            raise ValueError("Qubit index out of range")
        
    def ccx(self,control_qubit1:int,control_qubit2:int,target_qubit:int) -> 'QuantumCircuit':
        """Add CCX gate.

        Args:
            control_qubit1 (int): The qubit used as the first control.
            control_qubit2 (int): The qubit used as the second control.
            target_qubit (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        qubits0 = [control_qubit1,control_qubit2,target_qubit]
        if max(qubits0) < self.nqubits:
            if len(set(qubits0)) == 3:
                self.gates.append(('ccx',control_qubit1,control_qubit2,target_qubit))
                self._add_qubits(*qubits0)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit1 {control_qubit1} control_qubit2 {control_qubit2} target_qubit {target_qubit}")
        else:
            raise ValueError("Qubit index out of range")
        
    def cswap(self,control_qubit:int,target_qubit1:int,target_qubit2:int) -> 'QuantumCircuit':
        """Add CSWAP gate.

        Args:
            control_qubit (int): The qubit used as control.
            target_qubit1 (int): The qubit targeted by the gate.
            target_qubit2 (int): The qubit targeted by the gate.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        qubits0 = [control_qubit,target_qubit1,target_qubit2]
        if max(qubits0) < self.nqubits:
            if len(set(qubits0)) == 3:
                self.gates.append(('cswap',control_qubit,target_qubit1,target_qubit2))
                self._add_qubits(*qubits0)
            else:
                raise ValueError(f"Qubit index conflict: control_qubit1 {control_qubit} control_qubit2 {target_qubit1} target_qubit {target_qubit2}")
        else:
            raise ValueError("Qubit index out of range")
        
    def p(self, theta: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a Phase gate.

        Args:
            theta (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('p', theta, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
        else:
            raise ValueError("Qubit index out of range")
        
    def r(self, theta: float, phi:float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a R gate.

        $$
        R(\theta,\phi) = e^{-i\frac{\theta}{2}(\cos{\phi x}+\sin{\phi y})} = \begin{bmatrix}
         \cos(\frac{\theta}{2})             & -i e^{-i\phi}\sin(\frac{\theta}{2}) \\
         -i e^{i\phi}\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})      
        \end{bmatrix}
        $$

        Args:
            theta (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('r', theta, phi, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
            if isinstance(phi,str):
                self.params_value[phi] = phi
        else:
            raise ValueError("Qubit index out of range")
        
    def u(self, theta: float, phi: float, lamda: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a U3 gate.

        The U3 gate is a single-qubit gate with the following matrix representation:

        $$
        U3(\theta, \phi, \lambda) = \begin{bmatrix}
            \cos(\theta/2) & -e^{i\lambda} \sin(\theta/2) \\
            e^{i\phi} \sin(\theta/2) & e^{i(\phi + \lambda)} \cos(\theta/2)
            \end{bmatrix}
        $$

        Args:
            theta (float): The rotation angle of the gate.
            phi (float): The rotation angle of the gate.
            lamda (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('u', theta, phi, lamda, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
            if isinstance(phi,str):
                self.params_value[phi] = phi
            if isinstance(lamda,str):
                self.params_value[lamda] = lamda
        else:
            raise ValueError("Qubit index out of range")

    def u3(self, theta: float, phi: float, lamda: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a U3 gate.

        The U3 gate is a single-qubit gate with the following matrix representation:

        $$
        U3(\theta, \phi, \lambda) = \begin{bmatrix}
            \cos(\theta/2) & -e^{i\lambda} \sin(\theta/2) \\
            e^{i\phi} \sin(\theta/2) & e^{i(\phi + \lambda)} \cos(\theta/2)
            \end{bmatrix}
        $$

        Args:
            theta (float): The rotation angle of the gate.
            phi (float): The rotation angle of the gate.
            lamda (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.u(theta, phi, lamda, qubit)
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
            if isinstance(phi,str):
                self.params_value[phi] = phi
            if isinstance(lamda,str):
                self.params_value[lamda] = lamda
        else:
            raise ValueError("Qubit index out of range")   

    def rx(self, theta: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a RX gate.

        Args:
            theta (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('rx', theta, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
        else:
            raise ValueError("Qubit index out of range")
        
    def ry(self, theta: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a RY gate.

        Args:
            theta (float: The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('ry', theta, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
        else:
            raise ValueError("Qubit index out of range")
        
    def rz(self, theta: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a RZ gate.

        Args:
            theta (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('rz', theta, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
        else:
            raise ValueError("Qubit index out of range")
        
    def rxx(self, theta: float, qubit1: int, qubit2:int) -> 'QuantumCircuit':
        r"""
        Add a RXX gate.

        $$
        Rxx(\theta) = e^{-i\frac{\theta}{2}X\otimes X} = 
        \begin{bmatrix}
         \cos(\frac{\theta}{2})  & 0 & 0 & -i\sin(\frac{\theta}{2}) \\
         0 & \cos(\frac{\theta}{2}) & -i\sin(\frac{\theta}{2}) & 0 \\
         0 & -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) & 0 \\
         -i\sin(\frac{\theta}{2}) & 0 & 0 & \cos(\frac{\theta}{2})
        \end{bmatrix}.
        $$

        Args:
            theta (float): The rotation angle of the gate.
            qubit1 (int): The qubit to apply the gate to.
            qubit2 (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(qubit1, qubit2) < self.nqubits:
            if qubit1 != qubit2:
                self.gates.append(('rxx', theta, qubit1, qubit2))
                self._add_qubits(qubit1,qubit2)
                if isinstance(theta,str):
                    self.params_value[theta] = theta
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")
        
    def ryy(self, theta: float, qubit1: int, qubit2:int) -> 'QuantumCircuit':
        r"""
        Add a RYY gate.

        $$
        Ryy(\theta) = e^{-i\frac{\theta}{2}Y\otimes Y} = 
        \begin{bmatrix}
         \cos(\frac{\theta}{2})  & 0 & 0 & i\sin(\frac{\theta}{2}) \\
         0 & \cos(\frac{\theta}{2}) & -i\sin(\frac{\theta}{2}) & 0 \\
         0 & -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) & 0 \\
         i\sin(\frac{\theta}{2}) & 0 & 0 & \cos(\frac{\theta}{2})
        \end{bmatrix}.
        $$

        Args:
            theta (float): The rotation angle of the gate.
            qubit1 (int): The qubit to apply the gate to.
            qubit2 (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(qubit1, qubit2) < self.nqubits:
            if qubit1 != qubit2:
                self.gates.append(('ryy', theta, qubit1, qubit2))
                self._add_qubits(qubit1, qubit2)
                if isinstance(theta, str):
                    self.params_value[theta] = theta
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")
        
    def rzz(self, theta: float, qubit1: int, qubit2:int) -> 'QuantumCircuit':
        r"""
        Add a RZZ gate.

        $$
        Rzz(\theta) = e^{-i\frac{\theta}{2}Z\otimes Z} = 
        \begin{bmatrix}
         e^{-i\frac{\theta}{2}}  & 0 & 0 & 0 \\
         0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
         0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
         0 & 0 & 0 & e^{-i\frac{\theta}{2}}
        \end{bmatrix}.
        $$

        Args:
            theta (float): The rotation angle of the gate.
            qubit1 (int): The qubit to apply the gate to.
            qubit2 (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(qubit1, qubit2) < self.nqubits:
            if qubit1 != qubit2:
                self.gates.append(('rzz', theta, qubit1, qubit2))
                self._add_qubits(qubit1,qubit2)
                if isinstance(theta,str):
                    self.params_value[theta] = theta
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")

    def cp(self, theta: float, control_qubit: int, target_qubit:int) -> 'QuantumCircuit':
        r"""
        Add a Cphase gate.

        $$
        Rzz(\theta) = I \otimes |0\rangle\langle 0| + P \otimes |1\rangle\langle 1| = 
        \begin{bmatrix}
         1  & 0 & 0 & 0 \\
         0 & 1 & 0 & 0 \\
         0 & 0 & 1 & 0 \\
         0 & 0 & 0 & e^{i\theta}
        \end{bmatrix}.
        $$

        Args:
            theta (float): The rotation angle of the gate.
            control_qubit (int): The qubit to apply the gate to.
            target_qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if max(control_qubit, target_qubit) < self.nqubits:
            if control_qubit != target_qubit:
                self.gates.append(('cp', theta, control_qubit, target_qubit))
                self._add_qubits(control_qubit, target_qubit)
                if isinstance(theta,str):
                    self.params_value[theta] = theta
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {control_qubit}")
        else:
            raise ValueError("Qubit index out of range")
               
    def mapping_to_others(self,mapping:dict) -> 'QuantumCircuit':
        """Map current qubit indices to new indices.
    
        Args:
            mapping (dict): A dictionary specifying the mapping from current qubit indices to target indices.
    
        Returns:
            dict: A dictionary with updated qubit index mapping.
        """
        assert(len(self.qubits) == len(mapping))
        new = []
        for gate_info in self.gates:
            gate = gate_info[0]
            if gate in one_qubit_gates_available.keys():
                new.append((gate,mapping[gate_info[1]]))
            elif gate in two_qubit_gates_available.keys():
                new.append((gate,*[mapping[q] for q in gate_info[1:]]))
            elif gate in one_qubit_parameter_gates_available.keys():
                new.append((gate,*gate_info[1:-1],mapping[gate_info[-1]]))
            elif gate in two_qubit_parameter_gates_available.keys():
                new.append((gate,gate_info[1],*[mapping[q] for q in gate_info[2:]]))
            elif gate in three_qubit_gates_available.keys():
                new.append((gate,*[mapping[q] for q in gate_info[1:]]))
            elif gate in functional_gates_available.keys():
                if gate == 'measure':
                    qubitlst = [mapping[q] for q in gate_info[1]]
                    cbitlst = gate_info[2]
                    new.append((gate,qubitlst,cbitlst))
                elif gate == 'barrier':
                    qubitlst = [mapping[q] for q in gate_info[1] if q in mapping] 
                    new.append((gate,tuple(qubitlst)))
                elif gate == 'delay':
                    qubitlst = [mapping[q] for q in gate_info[-1]]
                    new.append((gate,gate_info[1],tuple(qubitlst)))
                elif gate == 'reset':
                    qubit0 = mapping[gate_info[1]]
                    new.append((gate,qubit0))
        self.nqubits = max(mapping.values())+1
        self.qubits = list(sorted(mapping.values()))
        self.gates = new
        return self

    def shallow_apply_value(self,params_dic):
        for k,v in params_dic.items():
            self.params_value[k] = v

    def deep_apply_value(self,params_dic):
        for k,v in params_dic.items():
            self.params_value[k] = v

        gates = []
        for gate_info in self.gates:
            gate = gate_info[0]
            if gate in one_qubit_parameter_gates_available.keys():
                params = list(gate_info[1:-1])
                qubit = gate_info[-1]
                for idx, param in enumerate(params):
                    if param in params_dic.keys():
                        params[idx] = params_dic[param]
                gate_info = (gate,*params,qubit)
                gates.append(gate_info)
            elif gate in two_qubit_parameter_gates_available.keys():
                params = list(gate_info[1:-2])
                qubits = gate_info[-2:]
                for idx, param in enumerate(params):
                    if param in params_dic.keys():
                        params[idx] = params_dic[param]
                gate_info = (gate,*params,*qubits)
                gates.append(gate_info)
            else:
                gates.append(gate_info)
        self.gates = gates

    def u3_for_unitary(self, unitary: np.ndarray, qubit: int):
        r"""
        Decomposes a 2x2 unitary matrix into a U3 gate and applies it to a specified qubit.

        Args:
            unitary (np.ndarray): A 2x2 unitary matrix.
            qubit (int): The qubit to apply the gate to.
        """
        assert(unitary.shape == (2,2))
        assert(qubit < self.nqubits)
        theta,phi,lamda,phase = u3_decompose(unitary)
        self.gates.append(('u', theta, phi, lamda, qubit))
        self._add_qubits(qubit)

    def zyz_for_unitary(self, unitary: np.ndarray, qubit:int) -> 'QuantumCircuit':
        r"""
        Decomposes a 2x2 unitary matrix into Rz-Ry-Rz gate sequence and applies it to a specified qubit.

        Args:
            unitary (np.ndarray): A 2x2 unitary matrix.
            qubit (int): The qubit to apply the gate sequence to.
        """
        assert(unitary.shape == (2,2))
        assert(qubit < self.nqubits)
        theta, phi, lamda, alpha = zyz_decompose(unitary)
        self.gates.append(('rz', lamda, qubit))
        self.gates.append(('ry', theta, qubit))
        self.gates.append(('rz', phi, qubit))
        self._add_qubits(qubit)

    def kak_for_unitary(self, unitary: np.ndarray, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        r"""
        Decomposes a 4 x 4 unitary matrix into a sequence of CZ and U3 gates using KAK decomposition and applies them to the specified qubits.

        Args:
            unitary (np.ndarray): A 4 x 4 unitary matrix.
            qubit1 (int): The first qubit to apply the gates to.
            qubit2 (int): The second qubit to apply the gates to.
        """
        assert(unitary.shape == (4,4))
        assert(qubit1 != qubit2)
        rots1, rots2 = kak_decompose(unitary)
        self.u3_for_unitary(rots1[0], qubit1)
        self.u3_for_unitary(h_mat @ rots2[0], qubit2)
        self.gates.append(('cz', qubit1, qubit2))
        self.u3_for_unitary(rots1[1], qubit1)
        self.u3_for_unitary(h_mat @ rots2[1] @ h_mat, qubit2)
        self.gates.append(('cz', qubit1, qubit2))
        self.u3_for_unitary(rots1[2], qubit1)
        self.u3_for_unitary(h_mat @ rots2[2] @ h_mat, qubit2)
        self.gates.append(('cz', qubit1, qubit2))        
        self.u3_for_unitary(rots1[3], qubit1)
        self.u3_for_unitary(rots2[3] @ h_mat, qubit2)
        self._add_qubits(qubit1,qubit2)

    def reset(self, qubit: int) -> 'QuantumCircuit':
        r"""
        Add reset to qubit.

        Args:
            qubit (int): The qubit to apply the instruction to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('reset', qubit))
            self._add_qubits(qubit)
        else:
            raise ValueError("Qubit index out of range")
        
    def delay(self,duration:int|float, *qubits:tuple[int],unit='s') -> 'QuantumCircuit':
        r"""
        Adds delay to qubits, the unit is s.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        # convert 'ns' 'ms' 'us' to 's
        if unit == 'ns':
            duration = duration * 1e-9
        elif unit == 'us':
            duration = duration * 1e-6
        elif unit =='ms':
            duration = duration * 1e-3

        if not qubits: # it will add barrier for all qubits
            self.gates.append(('delay', duration, tuple(self.qubits)))
        else:
            if max(qubits) < self.nqubits:
                self.gates.append(('delay', duration, qubits))
                self._add_qubits(*qubits)
            else:
                raise ValueError("Qubit index out of range")
        
    def barrier(self,*qubits: tuple[int]) -> 'QuantumCircuit':
        r"""
        Adds barrier to qubits.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if not qubits: # it will add barrier for all qubits
            self.gates.append(('barrier', tuple(self.qubits)))
        else:
            if max(qubits) < self.nqubits:
                if len(set(qubits)) == len(qubits):
                    self.gates.append(('barrier', qubits))
                else:
                    raise(ValueError(f'Qubit index conflict. {qubits}'))
            else:
                raise ValueError("Qubit index out of range")
            
    def remove_barrier(self) -> 'QuantumCircuit':
        r"""
        Remove all barrier gates from the quantum circuit.

        Returns:
            QuantumCircuit: The updated quantum circuit with all barrier gates removed.
        """
        new = []
        for gate_info in self.gates:
            gate  = gate_info[0]
            if gate != 'barrier':
                new.append(gate_info)
        self.gates = new
        return self
    
    def remove_gate(self,gate_name:str):
        r"""
        Remove specified gates from the circuit.

        Returns:
            QuantumCircuit: The updated quantum circuit with specified gates removed.
        
        """
        new = []
        for gate_info in self.gates:
            gate  = gate_info[0]
            if gate != gate_name:
                new.append(gate_info)
        self.gates = new
        return self
    
    def count_gate(self,gate_name:str) -> int:
        r"""Count target gates in this QuantumCircuit.

        Returns:
            int: The number of gates.
        """
        num = 0
        for gate_info in self.gates:
            gate  = gate_info[0]
            if gate == gate_name:
                num += 1
            else:
                continue
        return num
    
    def measure(self,qubitlst: int | Iterable[int], cbitlst: int | Iterable[int]) -> 'QuantumCircuit':
        r"""Adds measurement to qubits.

        Args:
            qubitlst (int | list): Qubit(s) to measure.
            cbitlst (int | list): Classical bit(s) to place the measure results in.
        """
        if isinstance(qubitlst,Iterable):
            qubitlst = list(qubitlst)
            cbitlst = list(cbitlst)
            if (len(set(qubitlst)) == len(qubitlst) and 
                len(set(cbitlst)) == len(cbitlst) and 
                len(qubitlst) == len(cbitlst)):
                self.gates.append(('measure', qubitlst,cbitlst))
                self._add_qubits(*qubitlst)
            else:
                raise(ValueError(f'Qubit or Cbits index conflict. {qubitlst} {cbitlst}'))
        elif isinstance(qubitlst,int):
            if qubitlst < self.nqubits:
                self.gates.append(('measure', [qubitlst], [cbitlst]))
                self._add_qubits(qubitlst)
            else:
                raise ValueError("Qubit index out of range")
        else:
            raise(ValueError(''))

    def measure_all(self) -> 'QuantumCircuit':
        r"""
        Adds measurement to all qubits.
        """
        qubitlst = [i for i in self.qubits]
        cbitlst = [i for i in range(len(qubitlst))]
        self.gates.append(('measure', qubitlst,cbitlst))

    @property
    def to_latex(self) -> str:
        print('If you need this feature, please contact the developer.')    

    @property
    def to_openqasm2(self) -> str:
        r"""
        Export the quantum circuit to an OpenQASM 2 program in a string.

        Returns:
            str: An OpenQASM 2 string representing the circuit.
        """
        return self._to_openqasm(version="2.0")

    @property
    def to_openqasm3(self) -> str:
        r"""
        Export the quantum circuit to an OpenQASM 3 program in a string.

        Returns:
            str: An OpenQASM 3 string representing the circuit.
        """
        return self._to_openqasm(version="3.0")

    def _to_openqasm(self, version: str) -> str:
        lines = self._openqasm_header(version)
        for gate_info in self.gates:
            lines.extend(self._openqasm_gate_lines(gate_info, version))
        return "\n".join(lines)

    def _openqasm_header(self, version: str) -> list[str]:
        gates0 = [gate[0] for gate in self.gates]
        lines = []
        if version == "2.0":
            lines.append("OPENQASM 2.0;")
            lines.append("include \"qelib1.inc\";")
            if 'delay' in gates0:
                lines.append("opaque delay(param0) q0;")
            if 'r' in gates0:
                lines.append("gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - param1) q0; }")
            lines.append(f"qreg q[{self.nqubits}];")
            lines.append(f"creg c[{self.ncbits}];")
        elif version == "3.0":
            lines.append("OPENQASM 3.0;")
            lines.append("include \"stdgates.inc\";")
            if 'delay' in gates0:
                lines.append("defcalgrammar \"openpulse\";")
            if 'r' in gates0:
                lines.append("gate r(theta,phi) q { u(theta,phi - pi/2,pi/2 - phi) q; }")
            lines.append(f"qubit[{self.nqubits}] q;")
            lines.append(f"bit[{self.ncbits}] c;")
        else:
            raise ValueError(f"Unsupported OpenQASM version: {version}")
        return lines

    def _openqasm_gate_lines(self, gate_info, version: str) -> list[str]:
        gate = gate_info[0]
        if gate in one_qubit_gates_available.keys():
            return [f"{gate} q[{gate_info[1]}];"]
        if gate in two_qubit_gates_available.keys():
            return [f"{gate} q[{gate_info[1]}],q[{gate_info[2]}];"]
        if gate in three_qubit_gates_available.keys():
            return [f"{gate} q[{gate_info[1]}],q[{gate_info[2]}],q[{gate_info[3]}];"]
        if gate in two_qubit_parameter_gates_available.keys():
            theta = self._resolve_param(gate_info[1])
            return [f"{gate}({theta}) q[{gate_info[2]}],q[{gate_info[3]}];"]
        if gate in one_qubit_parameter_gates_available.keys():
            if gate == 'u':
                theta, phi, lamda = self._resolve_param_list(gate_info[1:4])
                return [f"{gate}({theta},{phi},{lamda}) q[{gate_info[-1]}];"]
            if gate == 'r':
                theta, phi = self._resolve_param_list(gate_info[1:3])
                return [f"{gate}({theta},{phi}) q[{gate_info[-1]}];"]
            param_value = self._resolve_param(gate_info[1])
            return [f"{gate}({param_value}) q[{gate_info[2]}];"]
        if gate in ['reset']:
            return [f"{gate} q[{gate_info[1]}];"]
        if gate in ['delay']:
            lines = []
            for qubit in gate_info[2]:
                if version == "2.0":
                    lines.append(f"{gate}({gate_info[1]}) q[{qubit}];")
                else:
                    lines.append(f"{gate}[{gate_info[1]}] q[{qubit}];")
            return lines
        if gate in ['barrier']:
            line = f"{gate} q[{gate_info[1][0]}]"
            for idx in gate_info[1][1:]:
                line += f",q[{idx}]"
            return [line + ";"]
        if gate in ['measure']:
            lines = []
            for idx in range(len(gate_info[1])):
                if version == "2.0":
                    lines.append(f"measure q[{gate_info[1][idx]}] -> c[{gate_info[2][idx]}];")
                else:
                    lines.append(f"c[{gate_info[2][idx]}] = measure q[{gate_info[1][idx]}];")
            return lines
        raise ValueError(
            f"Sorry, Quark could not find the corresponding OpenQASM {version} syntax for now. Please contact the developer for assistance.{gate}"
        )

    @property
    def depth(self) -> int:
        r"""Count QuantumCircuit depth.

        Returns:
            int: QuantumCircuit depth.
        """
        import networkx as nx

        new = []
        for gate_info in self.gates:
            gate  = gate_info[0]
            if gate != 'barrier':
                new.append(gate_info) 

        node_list,edge_list = convert_gate_info_to_dag_info(self.nqubits,self.qubits,new,show_qubits=False)
        dag = nx.DiGraph()
        dag.add_nodes_from(node_list)
        dag.add_edges_from(edge_list)
        dag_nodes_layered = list(nx.topological_generations(dag))
        return len(dag_nodes_layered)
    
    @property
    def ncz(self) -> int:
        r"""Count all two-qubit gates in this QuantumCircuit.

        Returns:
            int: The number of two-qubit gates.
        """
        ncz = 0
        for gate_info in self.gates:
            gate  = gate_info[0]
            if gate in two_qubit_gates_available.keys():
                ncz += 1
            elif gate in two_qubit_parameter_gates_available.keys():
                ncz += 1
            else:
                continue
        return ncz
    
    @property
    def qubits_in_use(self) -> list[int]:
        r"""Get the list of qubits that have gates applied to them.

        Returns:
            list[int]: A list of qubit indices that are used in the circuit.
        """
        used_qubits = set()
        for gate_info in self.gates:
            gate = gate_info[0]
            if gate in one_qubit_gates_available.keys():
                used_qubits.add(gate_info[1])
            elif gate in two_qubit_gates_available.keys():
                used_qubits.add(gate_info[1])
                used_qubits.add(gate_info[2])
            elif gate in three_qubit_gates_available.keys():
                used_qubits.add(gate_info[1])
                used_qubits.add(gate_info[2])
                used_qubits.add(gate_info[3])
            elif gate in one_qubit_parameter_gates_available.keys():
                used_qubits.add(gate_info[-1])
            elif gate in two_qubit_parameter_gates_available.keys():
                used_qubits.add(gate_info[-2])
                used_qubits.add(gate_info[-1])
            elif gate in functional_gates_available.keys():
                if gate == 'measure':
                    for q in gate_info[1]:
                        used_qubits.add(q)
                elif gate == 'barrier':
                    for q in gate_info[1]:
                        used_qubits.add(q)
                elif gate == 'delay':
                    for q in gate_info[-1]:
                        used_qubits.add(q)
                elif gate == 'reset':
                    used_qubits.add(gate_info[1])
        return sorted(list(used_qubits))
    
    def draw(self, width: int = 4) -> None:
        r"""
        Draw the quantum circuit.

        Args:
            width (int, optional): The width between gates. Defaults to 4.
        """
        lines1,lines_use = add_gates_to_lines(self.nqubits,self.ncbits,self.gates,self.params_value, width = width)
        draw_circuit(lines1)

    def draw_simply(self, width: int = 4) -> None:
        r"""
        Draw a simplified quantum circuit diagram.
        
        This method visualizes the quantum circuit by displaying only the qubits that have gates applied to them,
        omitting any qubits without active gates. The result is a cleaner, more concise circuit diagram.

        Args:
            width (int, optional): The width between gates. Defaults to 4.
        """
        lines1,lines_use = add_gates_to_lines(self.nqubits,self.ncbits,self.gates,self.params_value, width=width)
        draw_circuit_simply(lines1, lines_use, self.nqubits)
