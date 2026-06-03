"""This module contains the QuantumCircuit class, which offers an intuitive interface for designing,
visualizing, and converting quantum circuits in various formats such as OpenQASM 2.0.

SPDX-License-Identifier: Apache-2.0
Original source: quarkcircuit, Copyright (c) YL Feng.
See THIRD_PARTY_NOTICES for full license text.
"""

from __future__ import annotations

import copy
import ast
import re
from typing import Iterable, Optional
import numpy as np
from .quantumcircuit_helpers import (
    one_qubit_gates_available,
    two_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    three_qubit_gates_available,
    functional_gates_available,
    noise_channel_gates_available,
    single_qubit_noise_channel_gates_available,
    two_qubit_noise_channel_gates_available,
    convert_gate_info_to_dag_info,
    add_gates_to_lines,
    )
from .qasm2 import parse_openqasm2_to_gates
from .render import draw_circuit, draw_circuit_simply
from .utils import u3_decompose, zyz_decompose, kak_decompose
from .matrix import h_mat

class QuantumCircuit:
    r"""
    A class used to build and manipulate a quantum circuit.

    This class allows you to create quantum circuits with a specified number of quantum and classical bits. 
    The circuit can be customized using various quantum gates, and additional features (such as simulation support, 
    circuit summary, and more) will be added in future versions.

    Note:
        All gate-appending methods (e.g. ``h``, ``x``, ``cx``, ``u``, ``rx``,
        ``barrier``, ``delay``, ``measure``, ``measure_all``, …) mutate the
        circuit in place **and** return ``self``, so calls can be chained:
        ``qc.h(0).cx(0, 1).measure_all()``.

    Attributes:
        nqubits (int or None): Number of quantum bits in the circuit.
        ncbits (int or None): Number of classical bits in the circuit.
    """
    def __init__(self, *args):
        r"""
        Initialize a QuantumCircuit object.

        The constructor supports three different initialization modes:
        1. `QuantumCircuit()`: Creates a circuit with `nqubits` and `ncbits` both set to ``0``.
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
        """Create an independent copy of the circuit including qubits, gates, and parameters.

        Returns:
            Deep copy of this ``QuantumCircuit``.
        """
        new_qc = QuantumCircuit(self.nqubits,self.ncbits)
        new_qc.qubits = copy.deepcopy(self.qubits)
        new_qc.params_value = copy.deepcopy(self.params_value)
        new_qc.gates = copy.deepcopy(self.gates)
        return new_qc

    def adjust_index(self, num: int, *, cbit_offset: Optional[int] = None) -> 'QuantumCircuit':
        """Shift all qubit and classical-bit indices in the circuit.

        Commonly used for circuit concatenation and sub-circuit packing.
        Each qubit index is offset by *num*; each classical-bit index is
        offset by *cbit_offset* (defaults to *num* when ``None``).

        Args:
            num (*int*): Offset added to every qubit index.
            cbit_offset (*Optional[int]*): Offset added to every cbit index. Defaults to ``None`` (same as *num*).

        Returns:
            This ``QuantumCircuit`` instance (mutated in-place).

        Raises:
            ValueError: Unsupported gate type in adjust_index: {gate}
        """
        if cbit_offset is None:
            cbit_offset = num
        gates = []
        for gate_info in self.gates:
            gate = gate_info[0]
            if gate in one_qubit_gates_available:
                qubit = gate_info[-1] + num
                gates.append((gate,qubit))
            elif gate in two_qubit_gates_available:
                qubit1 = gate_info[1] + num
                qubit2 = gate_info[2] + num
                gates.append((gate,qubit1,qubit2))
            elif gate in three_qubit_gates_available:
                qubit1 = gate_info[1] + num
                qubit2 = gate_info[2] + num
                qubit3 = gate_info[3] + num
                gates.append((gate, qubit1, qubit2, qubit3))
            elif gate in one_qubit_parameter_gates_available:
                qubit = gate_info[-1] + num
                gates.append((gate,*gate_info[1:-1],qubit))
            elif gate in two_qubit_parameter_gates_available:
                qubit1 = gate_info[-2] + num
                qubit2 = gate_info[-1] + num
                gates.append((gate, *gate_info[1:-2], qubit1, qubit2))
            elif gate in single_qubit_noise_channel_gates_available:
                qubit = gate_info[-1] + num
                gates.append((gate, gate_info[1], qubit))
            elif gate in two_qubit_noise_channel_gates_available:
                qubit1 = gate_info[2] + num
                qubit2 = gate_info[3] + num
                gates.append((gate, gate_info[1], qubit1, qubit2))
            elif gate in ['reset']:
                qubit = gate_info[-1] + num
                gates.append((gate,qubit))
            elif gate in ['delay']:
                qubits = [idx + num for idx in gate_info[2]]
                gates.append((gate, gate_info[1], tuple(qubits)))
            elif gate in ['barrier']:
                qubits = [idx + num for idx in gate_info[1]]
                gates.append((gate,tuple(qubits)))
            elif gate in ['measure']:
                qubits = [idx + num for idx in gate_info[1]]
                cbits = [idx + cbit_offset for idx in gate_info[-1]]
                gates.append((gate,qubits,cbits))
            else:
                raise ValueError(f"Unsupported gate type in adjust_index: {gate}")
        self.gates = gates   
        self.nqubits = self.nqubits + num
        self.ncbits = self.ncbits + cbit_offset
        self.qubits = [idx + num for idx in self.qubits] 
        return self

    @property
    def cbits(self):
        """Sorted list of unique classical bit indices used in measurement gates.

        Returns:
            List of classical bit indices used in the circuit.
        """
        cbits = []
        for gate_info in self.gates:
            if gate_info[0] == 'measure':
                for cbit in gate_info[2]:
                    cbits.append(cbit)
                    
        return sorted(set(cbits))

    def _add_qubits(self,*args):
        """Merge new qubit indices into the circuit's sorted qubit set.

        Args:
            *args: Qubit indices to add.
        """
        # Deduplicate and sort qubits.
        temp_set = set(self.qubits).union(args)
        self.qubits = sorted(temp_set)
        return self

    def _resolve_param(self, param):
        """Resolve a numeric or symbolic parameter to a float value.

        Args:
            param: Numeric value (``int``/``float``) or symbolic name (``str``).

        Returns:
            ``float`` resolved value.

        Raises:
            TypeError: Wrong param type! {param}
        """
        if isinstance(param, (float, int)):
            return float(param)
        if isinstance(param, str):
            if param in self.params_value:
                value = self.params_value[param]
                if isinstance(value, (float, int)):
                    return float(value)
            # Support symbolic parameter expressions like "-theta".
            return self._eval_param_expression(param)
        raise TypeError(f"Wrong param type! {param}")

    def _resolve_param_list(self, params):
        """Resolve a list of parameters to float values.

        Args:
            params: Sequence of numeric or symbolic parameter values.

        Returns:
            List of resolved ``float`` values.
        """
        return [self._resolve_param(param) for param in params]

    def _fmt_param(self, p, symbolic: bool) -> str:
        """Format a gate parameter as a string for OpenQASM output.

        Args:
            p: Parameter value — numeric or symbolic string.
            symbolic: When ``True`` and *p* is a string, return it verbatim
                without attempting numeric resolution.

        Returns:
            String representation of the parameter.
        """
        if symbolic and isinstance(p, str):
            return p
        return str(self._resolve_param(p))

    def _eval_param_expression(self, expr: str, *, symbol_resolver=None):
        """Safely evaluate a parameter expression.

        Args:
            expr (*str*): Parameter expression string (e.g. ``"theta_0"`` or ``"2*pi + alpha"``).
            symbol_resolver: Optional callable mapping symbol names to numeric
                values.  When omitted, symbols are resolved from ``self.params_value``.

        Returns:
            Evaluated numeric result.

        Raises:
            ValueError: unsupported parameter expression: {expr}
        """
        expr = str(expr).strip().replace('π', 'pi').replace('np.pi', 'pi')

        if symbol_resolver is None:
            def symbol_resolver(name: str):
                """Resolve a symbolic parameter name to its numeric value.

                Args:
                    name (*str*): Symbolic parameter name.

                Returns:
                    ``float`` value of the parameter.

                Raises:
                    ValueError: please apply value for parameter {name}
                """
                if name == "pi":
                    return float(np.pi)
                if name not in self.params_value:
                    raise ValueError(f"please apply value for parameter {name}")
                value = self.params_value[name]
                if isinstance(value, (int, float)):
                    return float(value)
                raise ValueError(f"please apply value for parameter {name}")

        def _eval(node):
            """Recursively evaluate an AST node to a numeric value.

            Args:
                node: AST node from ``ast.parse``.

            Returns:
                ``float`` evaluated result.

            Raises:
                ValueError: unsupported parameter expression: {expr}
            """
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("unsupported constant in parameter expression")
            if isinstance(node, ast.Name):
                return symbol_resolver(node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = _eval(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
            ):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                return left ** right
            raise ValueError(f"unsupported parameter expression: {expr}")

        try:
            tree = ast.parse(expr, mode="eval")
        except Exception as exc:
            raise ValueError(f"unsupported parameter expression: {expr}") from exc
        result = _eval(tree)
        if isinstance(result, (int, float)):
            return float(result)
        return result

    def _parse_pauli_string(self, pauli: str, *, num_qubits: Optional[int] = None):
        """Parse a Pauli string in compact or indexed format.

        Compact format examples: "XIZY", "ZZII".
Indexed format examples: "X1 Y2 Z3 Z4".

        Args:
            pauli (*str*): Pauli string, e.g. ``'XIZY'`` (compact) or ``'X1 Y2 Z3'`` (indexed).
            num_qubits (*Optional[int]*): Number of qubits. Defaults to ``None``.

        Returns:
            List of ``(qubit_index, pauli_op)`` tuples, omitting identity entries.

        Raises:
            TypeError: pauli must be a string
            ValueError: pauli string is empty
        """
        if not isinstance(pauli, str):
            raise TypeError("pauli must be a string")
        pauli = pauli.strip()
        if not pauli:
            raise ValueError("pauli string is empty")

        tokens = pauli.split()
        if len(tokens) == 1 and tokens[0].isalpha():
            compact = tokens[0].upper()
            if any(ch not in {"I", "X", "Y", "Z"} for ch in compact):
                raise ValueError("unsupported Pauli in compact string")
            if num_qubits is not None and len(compact) != num_qubits:
                raise ValueError("pauli length mismatch with num_qubits")
            return [(idx, op) for idx, op in enumerate(compact) if op != "I"]

        parsed = []
        used_indices = set()
        for tok in tokens:
            if not re.match(r"^[IXYZixyz]\d+$", tok):
                raise ValueError(f"invalid indexed Pauli token: {tok}")
            op = tok[0].upper()
            idx = int(tok[1:])
            if op != "I":
                if idx in used_indices:
                    raise ValueError("duplicate pauli index")
                used_indices.add(idx)
                parsed.append((idx, op))

        if num_qubits is not None:
            for idx, _ in parsed:
                if idx < 0 or idx >= num_qubits:
                    raise ValueError("pauli index out of range")
        return parsed

    def from_openqasm2(self,openqasm2_str: str) -> 'QuantumCircuit':
        r"""
        Initializes the QuantumCircuit object based on the given OpenQASM 2.0 string.

        Args:
            openqasm2_str (str): A string representing a quantum circuit in OpenQASM 2.0 format.

        Returns:
            This ``QuantumCircuit`` instance, populated from the parsed program.
        """
        if 'OPENQASM 2.0' not in openqasm2_str:
            raise ValueError("Input is not a valid OpenQASM 2.0 program")
        new_gates,qubit_used,cbit_used = parse_openqasm2_to_gates(openqasm2_str)
        self.nqubits = max(qubit_used) + 1 if qubit_used else 0
        self.ncbits = max(cbit_used) + 1 if cbit_used else 0
        self.qubits = sorted(qubit_used)
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
            return self
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
                return self
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
                return self
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
                return self
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
                return self
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
                return self
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
                return self
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
                return self
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
                return self
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
                return self
            else:
                raise ValueError(f"Qubit index conflict: control_qubit1 {control_qubit1} control_qubit2 {control_qubit2} target_qubit {target_qubit}")
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
            theta (float): Polar rotation angle.
            phi (float): Azimuthal phase angle.
            lamda (float): Diagonal phase angle.
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
            return self
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
            return self
        else:
            raise ValueError("Qubit index out of range")
        
    def ry(self, theta: float, qubit: int) -> 'QuantumCircuit':
        r"""
        Add a RY gate.

        Args:
            theta (float): The rotation angle of the gate.
            qubit (int): The qubit to apply the gate to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if qubit < self.nqubits:
            self.gates.append(('ry', theta, qubit))
            self._add_qubits(qubit)
            if isinstance(theta,str):
                self.params_value[theta] = theta
            return self
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
            return self
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
                return self
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
                return self
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
                return self
            else:
                raise ValueError(f"Qubit index conflict: qubit1 and qubit2 are both {qubit1}")
        else:
            raise ValueError("Qubit index out of range")

    def pauli_evolution(self, theta: float | str, pauli: str) -> 'QuantumCircuit':
        """Append ``exp(-i * theta/2 * P)`` for a Pauli string ``P``.

        The Pauli string supports compact format (e.g. ``"IXYZ"``) and
        indexed format (e.g. ``"X1 Y2 Z3 Z4"``).

        Args:
            theta (*float | str*): Rotation angle in radians.
            pauli (*str*): Pauli string, e.g. ``'XIZY'`` (compact) or ``'X1 Y2 Z3'`` (indexed).
        """
        terms = self._parse_pauli_string(pauli, num_qubits=self.nqubits)
        if not terms:
            # P = I, the unitary is a global phase and is skipped in circuit form.
            return self

        support = sorted(terms, key=lambda t: t[0])
        qubits = [q for q, _ in support]

        # Basis change: U_dag * P * U = Z...Z.
        for q, op in support:
            if op == "X":
                self.h(q)
            elif op == "Y":
                self.sdg(q)
                self.h(q)

        target = qubits[-1]
        if len(qubits) == 1:
            rz_theta = float(theta) if isinstance(theta, (int, float)) else theta
            self.rz(rz_theta, target)
        else:
            for i in range(len(qubits) - 1):
                self.cx(qubits[i], qubits[i+1])
            rz_theta = float(theta) if isinstance(theta, (int, float)) else theta
            self.rz(rz_theta, target)
            for i in range(len(qubits) - 2, -1, -1):
                self.cx(qubits[i], qubits[i+1])

        # Undo basis change: U_dag.
        for q, op in reversed(support):
            if op == "X":
                self.h(q)
            elif op == "Y":
                self.h(q)
                self.s(q)

        self._add_qubits(*qubits)
        if isinstance(theta, str):
            self.params_value[theta] = theta
        return self
               
    def mapping_to_others(self,mapping:dict) -> 'QuantumCircuit':
        """Map current qubit indices to new indices.
    
        Args:
            mapping (dict): A dictionary specifying the mapping from current qubit indices to target indices.
    
        Returns:
            ``QuantumCircuit``: This circuit with remapped qubit indices.
        """
        if len(self.qubits) != len(mapping):
            raise ValueError("Mapping size must match number of used qubits")
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
            elif gate in single_qubit_noise_channel_gates_available:
                new.append((gate, gate_info[1], mapping[gate_info[2]]))
            elif gate in two_qubit_noise_channel_gates_available:
                new.append((gate, gate_info[1], mapping[gate_info[2]], mapping[gate_info[3]]))
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

    @staticmethod
    def _resolve_expr(param, params_dic):
        """Resolve a symbolic parameter expression when all symbols are bound.

        Args:
            param: Parameter value (returned as-is if numeric) or symbolic expression string.
            params_dic: Dictionary mapping parameter names to numeric values.

        Returns:
            Resolved numeric value, or the original string if unresolvable.
        """
        if not isinstance(param, str):
            return param
        if param in params_dic:
            return params_dic[param]

        temp_qc = QuantumCircuit(0)
        temp_qc.params_value.update(params_dic)
        try:
            return temp_qc._eval_param_expression(param)
        except ValueError:
            return param

    def apply_value(self, params_dic: dict, *, deep: bool = False) -> 'QuantumCircuit':
        """Apply parameter values to the circuit.

        Args:
            params_dic (dict): Mapping from parameter name to numeric value.
            deep (bool, optional): If True, materialize values into gate tuples.
                If False, only update ``params_value`` table. Defaults to False.

        Returns:
            QuantumCircuit: This circuit instance.
        """
        for k,v in params_dic.items():
            self.params_value[k] = v

        if not deep:
            return self

        _resolve = self._resolve_expr
        gates = []
        for gate_info in self.gates:
            gate = gate_info[0]
            if gate in one_qubit_parameter_gates_available.keys():
                params = list(gate_info[1:-1])
                qubit = gate_info[-1]
                for idx, param in enumerate(params):
                    params[idx] = _resolve(param, params_dic)
                gate_info = (gate,*params,qubit)
                gates.append(gate_info)
            elif gate in two_qubit_parameter_gates_available.keys():
                params = list(gate_info[1:-2])
                qubits = gate_info[-2:]
                for idx, param in enumerate(params):
                    params[idx] = _resolve(param, params_dic)
                gate_info = (gate,*params,*qubits)
                gates.append(gate_info)
            else:
                gates.append(gate_info)
        self.gates = gates
        return self

    def u3_for_unitary(self, unitary: np.ndarray, qubit: int):
        r"""
        Decomposes a 2x2 unitary matrix into a U3 gate and applies it to a specified qubit.

        Args:
            unitary (np.ndarray): A 2x2 unitary matrix.
            qubit (int): The qubit to apply the gate to.

        Returns:
            This ``QuantumCircuit`` instance.
        """
        if unitary.shape != (2, 2):
            raise ValueError("unitary must be a 2x2 matrix")
        if qubit >= self.nqubits:
            raise ValueError("Qubit index out of range")
        theta,phi,lamda,phase = u3_decompose(unitary)
        self.gates.append(('u', theta, phi, lamda, qubit))
        self._add_qubits(qubit)
        return self

    def zyz_for_unitary(self, unitary: np.ndarray, qubit:int) -> 'QuantumCircuit':
        r"""
        Decomposes a 2x2 unitary matrix into Rz-Ry-Rz gate sequence and applies it to a specified qubit.

        Args:
            unitary (np.ndarray): A 2x2 unitary matrix.
            qubit (int): The qubit to apply the gate sequence to.

        Returns:
            This ``QuantumCircuit`` instance.
        """
        if unitary.shape != (2, 2):
            raise ValueError("unitary must be a 2x2 matrix")
        if qubit >= self.nqubits:
            raise ValueError("Qubit index out of range")
        theta, phi, lamda, alpha = zyz_decompose(unitary)
        self.gates.append(('rz', lamda, qubit))
        self.gates.append(('ry', theta, qubit))
        self.gates.append(('rz', phi, qubit))
        self._add_qubits(qubit)
        return self

    def kak_for_unitary(self, unitary: np.ndarray, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        r"""
        Decomposes a 4 x 4 unitary matrix into a sequence of CZ and U3 gates using KAK decomposition and applies them to the specified qubits.

        Args:
            unitary (np.ndarray): A 4 x 4 unitary matrix.
            qubit1 (int): The first qubit to apply the gates to.
            qubit2 (int): The second qubit to apply the gates to.

        Returns:
            This ``QuantumCircuit`` instance.
        """
        if unitary.shape != (4, 4):
            raise ValueError("unitary must be a 4x4 matrix")
        if qubit1 == qubit2:
            raise ValueError("qubit1 and qubit2 must be different")
        rots1, rots2 = kak_decompose(unitary)
        self.u3_for_unitary(rots1[0], qubit2)
        self.u3_for_unitary(h_mat @ rots2[0], qubit1)
        self.gates.append(('cz', qubit1, qubit2))
        self.u3_for_unitary(rots1[1], qubit2)
        self.u3_for_unitary(h_mat @ rots2[1] @ h_mat, qubit1)
        self.gates.append(('cz', qubit1, qubit2))
        self.u3_for_unitary(rots1[2], qubit2)
        self.u3_for_unitary(h_mat @ rots2[2] @ h_mat, qubit1)
        self.gates.append(('cz', qubit1, qubit2))
        self.u3_for_unitary(rots1[3], qubit2)
        self.u3_for_unitary(rots2[3] @ h_mat, qubit1)
        self._add_qubits(qubit1,qubit2)
        return self

    def depolarize1(self, p: float, qubit: int) -> 'QuantumCircuit':
        r"""Add a single-qubit depolarizing noise channel.

        Args:
            p (float): Error probability (0 ≤ p ≤ 1).
            qubit (int): The qubit to apply the channel to.

        Raises:
            ValueError: If qubit out of circuit range or p out of range.
        """
        if isinstance(p, str) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Depolarizing probability must be a number in [0, 1], got {p!r}")
        if qubit < self.nqubits:
            self.gates.append(('depolarize1', p, qubit))
            self._add_qubits(qubit)
            return self
        else:
            raise ValueError("Qubit index out of range")

    def depolarize2(self, p: float, qubit0: int, qubit1: int) -> 'QuantumCircuit':
        r"""Add a two-qubit depolarizing noise channel.

        Args:
            p (float): Error probability (0 ≤ p ≤ 1).
            qubit0 (int): The first qubit.
            qubit1 (int): The second qubit.

        Raises:
            ValueError: If qubits out of circuit range or p out of range.
        """
        if isinstance(p, str) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Depolarizing probability must be a number in [0, 1], got {p!r}")
        if max(qubit0, qubit1) < self.nqubits:
            if qubit0 != qubit1:
                self.gates.append(('depolarize2', p, qubit0, qubit1))
                self._add_qubits(qubit0, qubit1)
                return self
            else:
                raise ValueError(f"Qubit index conflict: qubit0 and qubit1 are both {qubit0}")
        else:
            raise ValueError("Qubit index out of range")

    def x_error(self, p: float, qubit: int) -> 'QuantumCircuit':
        r"""Add a single-qubit bit-flip (X) error channel.

        Args:
            p (float): Error probability (0 ≤ p ≤ 1).
            qubit (int): The qubit to apply the channel to.

        Raises:
            ValueError: If qubit out of circuit range or p out of range.
        """
        if isinstance(p, str) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Error probability must be a number in [0, 1], got {p!r}")
        if qubit < self.nqubits:
            self.gates.append(('x_error', p, qubit))
            self._add_qubits(qubit)
            return self
        else:
            raise ValueError("Qubit index out of range")

    def y_error(self, p: float, qubit: int) -> 'QuantumCircuit':
        r"""Add a single-qubit Y error channel.

        Args:
            p (float): Error probability (0 ≤ p ≤ 1).
            qubit (int): The qubit to apply the channel to.

        Raises:
            ValueError: If qubit out of circuit range or p out of range.
        """
        if isinstance(p, str) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Error probability must be a number in [0, 1], got {p!r}")
        if qubit < self.nqubits:
            self.gates.append(('y_error', p, qubit))
            self._add_qubits(qubit)
            return self
        else:
            raise ValueError("Qubit index out of range")

    def z_error(self, p: float, qubit: int) -> 'QuantumCircuit':
        r"""Add a single-qubit phase-flip (Z) error channel.

        Args:
            p (float): Error probability (0 ≤ p ≤ 1).
            qubit (int): The qubit to apply the channel to.

        Raises:
            ValueError: If qubit out of circuit range or p out of range.
        """
        if isinstance(p, str) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Error probability must be a number in [0, 1], got {p!r}")
        if qubit < self.nqubits:
            self.gates.append(('z_error', p, qubit))
            self._add_qubits(qubit)
            return self
        else:
            raise ValueError("Qubit index out of range")

    def amplitude_damping(self, gamma: float, qubit: int) -> 'QuantumCircuit':
        r"""Add an amplitude damping channel (energy dissipation).

        Args:
            gamma (float): Damping parameter (0 ≤ γ ≤ 1).
            qubit (int): The qubit to apply the channel to.

        Raises:
            ValueError: If qubit out of circuit range or gamma out of range.
        """
        if isinstance(gamma, str) or not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Damping parameter must be a number in [0, 1], got {gamma!r}")
        if qubit < self.nqubits:
            self.gates.append(('amplitude_damping', gamma, qubit))
            self._add_qubits(qubit)
            return self
        else:
            raise ValueError("Qubit index out of range")

    def phase_damping(self, gamma: float, qubit: int) -> 'QuantumCircuit':
        r"""Add a phase damping channel (dephasing).

        Args:
            gamma (float): Dephasing parameter (0 ≤ γ ≤ 1).
            qubit (int): The qubit to apply the channel to.

        Raises:
            ValueError: If qubit out of circuit range or gamma out of range.
        """
        if isinstance(gamma, str) or not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Dephasing parameter must be a number in [0, 1], got {gamma!r}")
        if qubit < self.nqubits:
            self.gates.append(('phase_damping', gamma, qubit))
            self._add_qubits(qubit)
            return self
        else:
            raise ValueError("Qubit index out of range")

    def remove_noise_channels(self) -> 'QuantumCircuit':
        r"""Return a copy of this circuit with all noise channel gates removed.

        Useful for obtaining the ideal (noise-free) version of a noisy circuit,
        e.g. as the reference branch in Clifford data regression.

        Returns:
            A new ``QuantumCircuit`` containing only the non-noise gates.
        """
        new = self.deepcopy()
        new.gates = [g for g in new.gates if g[0] not in noise_channel_gates_available]
        return new

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
            return self
        else:
            raise ValueError("Qubit index out of range")
        
    def delay(self,duration:int|float, *qubits:tuple[int],unit='s') -> 'QuantumCircuit':
        """Adds delay to qubits.

        Args:
            duration (*int | float*): Delay duration (in the unit given by *unit*).
            *qubits (*tuple[int]*): Qubit indices to delay. If omitted, delays all qubits.
            unit: Time unit ``'s'``, ``'ms'``, ``'us'``, or ``'ns'``. Defaults to ``'s'``.

        Raises:
            ValueError: Qubit index out of range
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
            return self
        else:
            if max(qubits) < self.nqubits:
                self.gates.append(('delay', duration, qubits))
                self._add_qubits(*qubits)
                return self
            else:
                raise ValueError("Qubit index out of range")
        
    def barrier(self,*qubits: tuple[int]) -> 'QuantumCircuit':
        """Adds barrier to qubits.

        Args:
            *qubits (tuple[int]): Qubits to add barrier to.

        Raises:
            ValueError: If qubit out of circuit range.
        """
        if not qubits: # it will add barrier for all qubits
            self.gates.append(('barrier', tuple(self.qubits)))
            return self
        else:
            if max(qubits) < self.nqubits:
                if len(set(qubits)) == len(qubits):
                    self.gates.append(('barrier', qubits))
                    return self
                else:
                    raise(ValueError(f'Qubit index conflict. {qubits}'))
            else:
                raise ValueError("Qubit index out of range")
            
    def remove_barrier(self) -> 'QuantumCircuit':
        """Remove all barrier gates from the quantum circuit.

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
        """Remove specified gates from the circuit.

        Args:
            gate_name (str): Name of the quantum gate.

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
        """Count target gates in this QuantumCircuit.

        Args:
            gate_name (str): Name of the quantum gate.

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
                return self
            else:
                raise(ValueError(f'Qubit or Cbits index conflict. {qubitlst} {cbitlst}'))
        elif isinstance(qubitlst,int):
            if qubitlst < self.nqubits:
                self.gates.append(('measure', [qubitlst], [cbitlst]))
                self._add_qubits(qubitlst)
                return self
            else:
                raise ValueError("Qubit index out of range")
        else:
            raise(ValueError(''))

    def measure_all(self) -> 'QuantumCircuit':
        """Adds measurement to all qubits.
        """
        qubitlst = [i for i in self.qubits]
        cbitlst = [i for i in range(len(qubitlst))]
        self.gates.append(('measure', qubitlst,cbitlst))
        return self

    @property
    def to_latex(self) -> str:
        """Export the quantum circuit to a LaTeX string.

        Raises:
            NotImplementedError: to_latex is not implemented yet.
        """
        raise NotImplementedError("to_latex is not implemented yet")

    def to_openqasm2(self, symbolic: bool = False) -> str:
        """Export the quantum circuit to an OpenQASM 2 program in a string.

        Args:
            symbolic: When ``True``, unbound string parameters are emitted
                verbatim instead of being resolved numerically.  Use this to
                produce a QASM *template* for later parameter substitution.

        Returns:
            str: An OpenQASM 2 string representing the circuit.
        """
        return self._to_openqasm(version="2.0", symbolic=symbolic)

    def _to_openqasm(self, version: str = "2.0", symbolic: bool = False) -> str:
        """Serialize the circuit to an OpenQASM 2.0 string.

        Args:
            version (*str*): Must be ``"2.0"``.

        Returns:
            Complete OpenQASM 2.0 program string.
        """
        lines = self._openqasm_header(version)
        for gate_info in self.gates:
            lines.extend(self._openqasm_gate_lines(gate_info, symbolic=symbolic))
        return "\n".join(lines)

    def _openqasm_header(self, version: str = "2.0") -> list[str]:
        """Generate OpenQASM 2.0 header lines (version, includes, register declarations).

        Args:
            version (*str*): Must be ``"2.0"``.

        Returns:
            List of header line strings.

        Raises:
            ValueError: If *version* is not ``"2.0"``.
        """
        if version != "2.0":
            raise ValueError(f"Unsupported OpenQASM version: {version}")
        gates0 = [gate[0] for gate in self.gates]
        lines = [
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
        ]
        if 'delay' in gates0:
            lines.append("opaque delay(param0) q0;")
        for gate_name in single_qubit_noise_channel_gates_available:
            if gate_name in gates0:
                arg = "gamma" if "damping" in gate_name else "p"
                lines.append(f"opaque {gate_name}({arg}) q;")
        for gate_name in two_qubit_noise_channel_gates_available:
            if gate_name in gates0:
                arg = "gamma" if "damping" in gate_name else "p"
                lines.append(f"opaque {gate_name}({arg}) q0,q1;")
        lines.append(f"qreg q[{self.nqubits}];")
        lines.append(f"creg c[{self.ncbits}];")
        return lines

    def _openqasm_gate_lines(self, gate_info, symbolic: bool = False) -> list[str]:
        """Convert a single gate tuple to OpenQASM 2.0 instruction lines.

        Args:
            gate_info: Gate tuple from ``self.gates``.

        Returns:
            List of gate instruction line strings.

        Raises:
            ValueError: If the gate is not supported in OpenQASM 2.0.
        """
        gate = gate_info[0]
        if gate in one_qubit_gates_available.keys():
            return [f"{gate} q[{gate_info[1]}];"]
        if gate in two_qubit_gates_available.keys():
            return [f"{gate} q[{gate_info[1]}],q[{gate_info[2]}];"]
        if gate in three_qubit_gates_available.keys():
            return [f"{gate} q[{gate_info[1]}],q[{gate_info[2]}],q[{gate_info[3]}];"]
        if gate in two_qubit_parameter_gates_available.keys():
            theta = self._fmt_param(gate_info[1], symbolic)
            return [f"{gate}({theta}) q[{gate_info[2]}],q[{gate_info[3]}];"]
        if gate in one_qubit_parameter_gates_available.keys():
            if gate == 'u':
                theta = self._fmt_param(gate_info[1], symbolic)
                phi = self._fmt_param(gate_info[2], symbolic)
                lamda = self._fmt_param(gate_info[3], symbolic)
                return [f"{gate}({theta},{phi},{lamda}) q[{gate_info[-1]}];"]
            param_value = self._fmt_param(gate_info[1], symbolic)
            return [f"{gate}({param_value}) q[{gate_info[2]}];"]
        if gate in ['reset']:
            return [f"{gate} q[{gate_info[1]}];"]
        if gate in ['delay']:
            return [f"{gate}({gate_info[1]}) q[{qubit}];" for qubit in gate_info[2]]
        if gate in ['barrier']:
            line = f"{gate} q[{gate_info[1][0]}]"
            for idx in gate_info[1][1:]:
                line += f",q[{idx}]"
            return [line + ";"]
        if gate in ['measure']:
            return [
                f"measure q[{gate_info[1][idx]}] -> c[{gate_info[2][idx]}];"
                for idx in range(len(gate_info[1]))
            ]
        if gate in single_qubit_noise_channel_gates_available:
            param_value = self._fmt_param(gate_info[1], symbolic)
            return [f"{gate}({param_value}) q[{gate_info[2]}];"]
        if gate in two_qubit_noise_channel_gates_available:
            param_value = self._fmt_param(gate_info[1], symbolic)
            return [f"{gate}({param_value}) q[{gate_info[2]}],q[{gate_info[3]}];"]
        raise ValueError(f"Unsupported gate for OpenQASM 2.0: {gate}")

    @property
    def depth(self) -> int:
        """Count QuantumCircuit depth.

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
        """Count all two-qubit gates in this QuantumCircuit.

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
        """Get the list of qubits that have gates applied to them.

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
            elif gate in single_qubit_noise_channel_gates_available:
                used_qubits.add(gate_info[2])
            elif gate in two_qubit_noise_channel_gates_available:
                used_qubits.add(gate_info[2])
                used_qubits.add(gate_info[3])
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
