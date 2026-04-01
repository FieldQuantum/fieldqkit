"""QASM to QCIS converter modified from cqlib"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial, singledispatch, update_wrapper
import inspect
import math
import operator
from pathlib import Path
from typing import List, Optional, Union

from openqasm3 import parse
from openqasm3.ast import (
    BinaryExpression,
    BooleanLiteral,
    ClassicalDeclaration,
    DurationLiteral,
    FloatLiteral,
    Identifier,
    ImaginaryLiteral,
    Include,
    IndexedIdentifier,
    IntegerLiteral,
    QuantumBarrier,
    QuantumGate,
    QuantumGateDefinition,
    QuantumMeasurementStatement,
    QuantumPhase,
    QuantumReset,
    QubitDeclaration,
    UnaryExpression,
)

# ---------------------------------------------------------------------------
# Instruction dataclass
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """Basic data class for storing QASM AST data and outputting QCIS."""

    name: str
    qubit_index: List[int]
    arguments: Optional[List[Union[int, float]]] = None

    def __str__(self):
        """Format this instruction as a QCIS-compatible string.

        Returns:
            str: The QCIS instruction string.
        """
        instr_str = self.name.upper() + " "
        for i in self.qubit_index:
            instr_str += f"Q{i} "
        if self.arguments:
            for i in self.arguments:
                if isinstance(i, float) and self.name.lower() == "rz":
                    # GuoDun rejects RZ at exactly ±π; clamp to open interval
                    import math
                    if abs(abs(i) - math.pi) < 1e-12:
                        i = math.copysign(math.pi - 1e-10, i)
                instr_str += f"{i} "
        return instr_str.rstrip()

# ---------------------------------------------------------------------------
# AST expression evaluator
# ---------------------------------------------------------------------------

_unary_operator = {
    "-": partial(operator.sub, 0),
}

_binary_operator_map = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "**": operator.pow,
}


def _traversal_binary_tree(node, var_map):
    """Recursively evaluate an OpenQASM AST expression tree and return the numeric result.

    Args:
        node: An OpenQASM AST expression node.
        var_map: A mapping from variable names to their numeric values.

    Returns:
        The evaluated numeric result.

    Raises:
        NotImplementedError: f'Binary operator {node.op.name} not implemented.
        TypeError: f'Invalid input type {type(node)} found when traversing b...
    """
    if isinstance(node, (IntegerLiteral, FloatLiteral, ImaginaryLiteral, BooleanLiteral, DurationLiteral)):
        return node.value
    elif isinstance(node, Identifier):
        return var_map[node.name]
    elif isinstance(node, BinaryExpression):
        if node.op.name not in _binary_operator_map:
            raise NotImplementedError(f"Binary operator {node.op.name} not implemented.")
        return _binary_operator_map[node.op.name](
            _traversal_binary_tree(node.lhs, var_map),
            _traversal_binary_tree(node.rhs, var_map),
        )
    elif isinstance(node, UnaryExpression):
        if node.op.name not in _unary_operator:
            raise NotImplementedError(f"Unary operator {node.op.name} not implemented.")
        return _unary_operator[node.op.name](_traversal_binary_tree(node.expression, var_map))
    else:
        raise TypeError(f"Invalid input type {type(node)} found when traversing binary tree.")

# ---------------------------------------------------------------------------
# Native QCIS decomposition rules
# ---------------------------------------------------------------------------

class NativeQcisRules:
    """Decompose QASM gates into QCIS native gates (X2P, Y2P, RZ, X2M, Y2M, CZ)."""

    pi = round(math.pi, 6)
    i_duration = 60

    @staticmethod
    def x(inp: Instruction):
        """Decompose Pauli-X gate into two X2P half-rotations.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("x2p", inp.qubit_index), Instruction("x2p", inp.qubit_index)]

    @staticmethod
    def y(inp: Instruction):
        """Decompose Pauli-Y gate into two Y2P half-rotations.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("y2p", inp.qubit_index), Instruction("y2p", inp.qubit_index)]

    @staticmethod
    def z(inp: Instruction):
        """Decompose Pauli-Z gate into an RZ(π) rotation.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("rz", inp.qubit_index, [NativeQcisRules.pi])]

    @staticmethod
    def h(inp: Instruction):
        """Decompose Hadamard gate into Y2M followed by RZ(π).

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("y2m", inp.qubit_index), Instruction("rz", inp.qubit_index, [NativeQcisRules.pi])]

    @staticmethod
    def sx(inp: Instruction):
        """Map SX gate to a single X2P native gate.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("x2p", inp.qubit_index)]

    @staticmethod
    def sxdg(inp: Instruction):
        """Map SXDG gate to a single X2M native gate.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("x2m", inp.qubit_index)]

    @staticmethod
    def s(inp: Instruction):
        """Decompose S gate into RZ(π/2).

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("rz", inp.qubit_index, [NativeQcisRules.pi / 2])]

    @staticmethod
    def sdg(inp: Instruction):
        """Decompose S-dagger gate into RZ(-π/2).

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("rz", inp.qubit_index, [-NativeQcisRules.pi / 2])]

    @staticmethod
    def t(inp: Instruction):
        """Decompose T gate into RZ(π/4).

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("rz", inp.qubit_index, [NativeQcisRules.pi / 4])]

    @staticmethod
    def tdg(inp: Instruction):
        """Decompose T-dagger gate into RZ(-π/4).

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("rz", inp.qubit_index, [-NativeQcisRules.pi / 4])]

    @staticmethod
    def rx(inp: Instruction):
        """Decompose RX(θ) gate into Y2M, RZ(θ), Y2P sequence.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("y2m", inp.qubit_index),
            Instruction("rz", inp.qubit_index, inp.arguments),
            Instruction("y2p", inp.qubit_index),
        ]

    @staticmethod
    def ry(inp: Instruction):
        """Decompose RY(θ) gate into X2P, RZ(θ), X2M sequence.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("x2p", inp.qubit_index),
            Instruction("rz", inp.qubit_index, inp.arguments),
            Instruction("x2m", inp.qubit_index),
        ]

    @staticmethod
    def rz(inp: Instruction):
        """Pass through RZ gate as a native QCIS instruction.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [inp]

    @staticmethod
    def u(inp: Instruction):
        """Decompose U(θ,φ,λ) gate into five native instructions: RZ, X2P, RZ, X2M, RZ.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("rz", inp.qubit_index, [inp.arguments[1]]),
            Instruction("x2p", inp.qubit_index),
            Instruction("rz", inp.qubit_index, [inp.arguments[0]]),
            Instruction("x2m", inp.qubit_index),
            Instruction("rz", inp.qubit_index, [inp.arguments[2]]),
        ]

    @staticmethod
    def u1(inp: Instruction):
        """Decompose U1(λ) phase gate into RZ(λ).

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("rz", inp.qubit_index, inp.arguments)]

    @staticmethod
    def u2(inp: Instruction):
        """Decompose U2(φ,λ) gate into RZ(φ), Y2P, RZ(λ) sequence.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("rz", inp.qubit_index, [inp.arguments[0]]),
            Instruction("y2p", inp.qubit_index),
            Instruction("rz", inp.qubit_index, [inp.arguments[1]]),
        ]

    @staticmethod
    def u3(inp: Instruction):
        """Decompose U3(θ,φ,λ) gate into RZ(λ), X2P, RZ(θ), X2M, RZ(φ) sequence.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("rz", inp.qubit_index, [inp.arguments[2]]),
            Instruction("x2p", inp.qubit_index),
            Instruction("rz", inp.qubit_index, [inp.arguments[0]]),
            Instruction("x2m", inp.qubit_index),
            Instruction("rz", inp.qubit_index, [inp.arguments[1]]),
        ]

    @staticmethod
    def id(inp: Instruction):
        """Convert identity gate to a QCIS idle (I) instruction.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("i", inp.qubit_index, [NativeQcisRules.i_duration])]

    @staticmethod
    def delay(inp: Instruction):
        """Convert delay instruction to a QCIS idle (I) instruction with optional custom duration.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        duration = inp.arguments[0] if inp.arguments else NativeQcisRules.i_duration
        return [Instruction("i", inp.qubit_index, [duration])]

    @staticmethod
    def cx(inp: Instruction):
        """Decompose CNOT (CX) gate into Y2M, CZ, Y2P on the target qubit.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("y2m", [inp.qubit_index[1]]),
            Instruction("cz", inp.qubit_index),
            Instruction("y2p", [inp.qubit_index[1]]),
        ]

    @staticmethod
    def cz(inp: Instruction):
        """Pass through CZ gate as a native QCIS instruction.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [Instruction("cz", inp.qubit_index)]

    @staticmethod
    def cy(inp: Instruction):
        """Decompose controlled-Y (CY) gate into X2P, CZ, X2M on the target qubit.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        return [
            Instruction("x2p", [inp.qubit_index[1]]),
            Instruction("cz", inp.qubit_index),
            Instruction("x2m", [inp.qubit_index[1]]),
        ]

    @staticmethod
    def ch(inp: Instruction):
        """Decompose controlled-Hadamard (CH) gate using S, H, T, CX, Tdg, H, Sdg decomposition.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        res = []
        res.extend(NativeQcisRules.s(Instruction("s", [inp.qubit_index[1]])))
        res.extend(NativeQcisRules.h(Instruction("h", [inp.qubit_index[1]])))
        res.extend(NativeQcisRules.t(Instruction("t", [inp.qubit_index[1]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", inp.qubit_index)))
        res.extend(NativeQcisRules.tdg(Instruction("tdg", [inp.qubit_index[1]])))
        res.extend(NativeQcisRules.h(Instruction("h", [inp.qubit_index[1]])))
        res.extend(NativeQcisRules.sdg(Instruction("sdg", [inp.qubit_index[1]])))
        return res

    @staticmethod
    def swap(inp: Instruction):
        """Decompose SWAP gate into three CX (CNOT) gates.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        rev = list(inp.qubit_index)
        rev.reverse()
        res = []
        res.extend(NativeQcisRules.cx(Instruction("cx", inp.qubit_index)))
        res.extend(NativeQcisRules.cx(Instruction("cx", rev)))
        res.extend(NativeQcisRules.cx(Instruction("cx", inp.qubit_index)))
        return res

    @staticmethod
    def crz(inp: Instruction):
        """Decompose controlled-RZ (CRZ) gate into RZ, CX, RZ, CX sequence.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        res = [Instruction("rz", [inp.qubit_index[1]], [i / 2 for i in inp.arguments])]
        res.extend(NativeQcisRules.cx(Instruction("cx", inp.qubit_index)))
        res.append(Instruction("rz", [inp.qubit_index[1]], [-i / 2 for i in inp.arguments]))
        res.extend(NativeQcisRules.cx(Instruction("cx", inp.qubit_index)))
        return res

    @staticmethod
    def cp(inp: Instruction):
        """Decompose controlled-phase (CP) gate using RZ on control qubit and CRZ decomposition.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        res = [Instruction("rz", [inp.qubit_index[0]], [i / 2 for i in inp.arguments])]
        res.extend(NativeQcisRules.crz(Instruction("crz", inp.qubit_index, inp.arguments)))
        return res

    @staticmethod
    def ccx(inp: Instruction):
        """Decompose Toffoli (CCX) gate into the standard H, CX, T, Tdg sequence.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        q = inp.qubit_index
        res = []
        res.extend(NativeQcisRules.h(Instruction("h", [q[2]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[1], q[2]])))
        res.extend(NativeQcisRules.tdg(Instruction("tdg", [q[2]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[0], q[2]])))
        res.extend(NativeQcisRules.t(Instruction("t", [q[2]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[1], q[2]])))
        res.extend(NativeQcisRules.tdg(Instruction("tdg", [q[2]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[0], q[2]])))
        res.extend(NativeQcisRules.t(Instruction("t", [q[1]])))
        res.extend(NativeQcisRules.t(Instruction("t", [q[2]])))
        res.extend(NativeQcisRules.h(Instruction("h", [q[2]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[0], q[1]])))
        res.extend(NativeQcisRules.t(Instruction("t", [q[0]])))
        res.extend(NativeQcisRules.tdg(Instruction("tdg", [q[1]])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[0], q[1]])))
        return res

    @staticmethod
    def cu3(inp: Instruction):
        """Decompose controlled-U3(θ,φ,λ) gate using RZ, CX, and U sub-decompositions.

        Args:
            inp (Instruction): The gate instruction to decompose.

        Returns:
            list[Instruction]: Native QCIS instructions.
        """
        q = inp.qubit_index
        a = inp.arguments
        res = [Instruction("rz", [q[1]], [(a[1] - a[2]) / 2])]
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[0], q[1]])))
        res.extend(NativeQcisRules.u(Instruction("u", [q[1]], [-a[0] / 2, 0, -(a[1] + a[2]) / 2])))
        res.extend(NativeQcisRules.cx(Instruction("cx", [q[0], q[1]])))
        res.extend(NativeQcisRules.u(Instruction("u", [q[1]], [a[0] / 2, a[2], 0])))
        return res

# ---------------------------------------------------------------------------
# QasmToQcis converter
# ---------------------------------------------------------------------------

_include_file_path_map = {
    'qelib1.inc': Path(__file__).parent / "include/qelib1.inc",
}


def _duration_literal_to_seconds(duration_literal: DurationLiteral) -> float:
    """Convert an OpenQASM DurationLiteral AST node to a float value in seconds.

    Args:
        duration_literal (DurationLiteral): The duration literal node from the AST.

    Returns:
        float: The duration in seconds.
    """
    value = float(duration_literal.value)
    unit = getattr(duration_literal, "unit", "s")
    unit_name = unit.name if hasattr(unit, "name") else str(unit)
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}.get(unit_name, 1.0)
    return value * scale


def _meth_dispatch(func):
    """Decorator that adapts singledispatch to work with instance methods, dispatching on the second argument.

    Args:
        func: The function to wrap with singledispatch.

    Returns:
        A wrapper function that dispatches based on the type of the second argument.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        """Dispatch to the registered implementation based on the second argument's type."""
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


class QasmToQcis:
    """Convert OpenQASM to QCIS instruction string.

    Uses QCIS native gates by default: X2P, Y2P, RZ, X2M, Y2M, CZ.
    """

    def __init__(self, rule=None):
        """Initialize the QASM-to-QCIS converter.

        Args:
            rule: A class providing static decomposition methods. Defaults to NativeQcisRules.
        """
        if rule is None:
            rule = NativeQcisRules
        self.instruct_convert_rule_dict = dict(inspect.getmembers(rule, inspect.isfunction))
        self.qcis_str = ""
        self.qubit_map = {}
        self.var_map = {"pi": round(math.pi, 6), "π": round(math.pi, 6)}

    @_meth_dispatch
    def _parse_argument(self, argument: object, var_map=None):
        """Parse an OpenQASM gate argument into a numeric value.

        Args:
            argument (object): An OpenQASM AST argument node.
            var_map: Variable name to value mapping. Defaults to None.

        Raises:
            NotImplementedError: f'Invalid input type {type(argument)} found when parsing ...
        """
        raise NotImplementedError(f"Invalid input type {type(argument)} found when parsing argument {argument}.")

    @_parse_argument.register(IntegerLiteral)
    @_parse_argument.register(FloatLiteral)
    @_parse_argument.register(ImaginaryLiteral)
    @_parse_argument.register(BooleanLiteral)
    def _(self, argument, var_map=None):
        """.

        Args:
            argument: Argument.
            var_map: Var map. Defaults to ``None``.

        Returns:
            Result.
        """
        return argument.value

    @_parse_argument.register(DurationLiteral)
    def _(self, argument, var_map=None):
        """.

        Args:
            argument: Argument.
            var_map: Var map. Defaults to ``None``.

        Returns:
            Result.
        """
        return _duration_literal_to_seconds(argument)

    @_parse_argument.register(UnaryExpression)
    @_parse_argument.register(Identifier)
    @_parse_argument.register(BinaryExpression)
    def _(self, argument, var_map=None):
        """.

        Args:
            argument: Argument.
            var_map: Var map. Defaults to ``None``.

        Returns:
            Result.
        """
        if var_map is None:
            var_map = self.var_map
        return _traversal_binary_tree(argument, var_map)

    @_meth_dispatch
    def _parse_qubit(self, qubit, qubit_map=None):
        """Parse an OpenQASM qubit reference into an integer index.

        Args:
            qubit: An OpenQASM AST qubit node.
            qubit_map: Gate-local qubit name to index mapping. Defaults to None.

        Raises:
            NotImplementedError: f'Invalid input type {type(qubit)} found when parse argum...
        """
        raise NotImplementedError(f"Invalid input type {type(qubit)} found when parse argument {qubit}.")

    @_parse_qubit.register(Identifier)
    def _(self, qubit, qubit_map=None):
        """.

        Args:
            qubit: Target qubit index.
            qubit_map: Qubit map. Defaults to ``None``.

        Returns:
            Result.

        Raises:
            KeyError: f'qubit map not defined when parsing qubit {qubit} with t...
        """
        if qubit_map is None:
            raise KeyError(f"qubit map not defined when parsing qubit {qubit} with type {type(qubit)}.")
        return qubit_map[qubit.name]

    @_parse_qubit.register(IndexedIdentifier)
    def _(self, qubit, qubit_map=None):
        """.

        Args:
            qubit: Target qubit index.
            qubit_map: Qubit map. Defaults to ``None``.

        Returns:
            Result.
        """
        return self.qubit_map[(qubit.name.name, qubit.indices[0][0].value)]

    @_meth_dispatch
    def _parse_ast_statement(self, statement: object, var_map=None, qubit_map=None):
        """Parse an OpenQASM AST statement into a list of QCIS instructions.

        Args:
            statement (object): An OpenQASM AST statement node.
            var_map: Variable name to value mapping. Defaults to None.
            qubit_map: Gate-local qubit name to index mapping. Defaults to None.

        Returns:
            list[Instruction]: The corresponding QCIS instructions.

        Raises:
            NotImplementedError: f'Invalid input type {type(statement)} found when parsing...
        """
        statement_type = statement.__class__.__name__
        if statement_type in {"CalibrationGrammarDeclaration"}:
            return ""
        if statement_type in {"DelayInstruction", "QuantumDelay"}:
            qubits = getattr(statement, "qubits", None)
            if qubits is None:
                qubit = getattr(statement, "qubit", None)
                qubits = [qubit] if qubit is not None else []
            qubit_index = [self._parse_qubit(qubit, qubit_map=qubit_map) for qubit in qubits]
            duration_node = getattr(statement, "duration", None)
            args = [self._parse_argument(duration_node, var_map=var_map)] if duration_node is not None else None
            gate_instruction = Instruction("delay", qubit_index, args)
            return self.instruct_convert_rule_dict["delay"](gate_instruction)
        raise NotImplementedError(f"Invalid input type {type(statement)} found when parsing statement {statement}.")

    @_parse_ast_statement.register(Include)
    def _(self, statement, var_map=None, qubit_map=None):
        """.

        Args:
            statement: Statement.
            var_map: Var map. Defaults to ``None``.
            qubit_map: Qubit map. Defaults to ``None``.

        Returns:
            Result.

        Raises:
            FileNotFoundError: f'Include file {include_file_name} not found.
        """
        include_file_name = statement.filename
        if include_file_name == "stdgates.inc":
            return ""
        if not Path(include_file_name).exists():
            if include_file_name in _include_file_path_map:
                include_file_path = _include_file_path_map[include_file_name]
            else:
                raise FileNotFoundError(f"Include file {include_file_name} not found.")
        else:
            include_file_path = Path(include_file_name)
        with open(include_file_path, "r") as f:
            self.convert_to_qcis(f.read())
        return ""

    @_parse_ast_statement.register(ClassicalDeclaration)
    @_parse_ast_statement.register(QuantumPhase)
    def _(self, statement, var_map=None, qubit_map=None):
        """.

        Args:
            statement: Statement.
            var_map: Var map. Defaults to ``None``.
            qubit_map: Qubit map. Defaults to ``None``.

        Returns:
            Result.
        """
        return ""

    @_parse_ast_statement.register(QubitDeclaration)
    def _(self, statement):
        """.

        Args:
            statement: Statement.

        Returns:
            Result.
        """
        name = statement.qubit.name
        size = statement.size.value
        start_count = len(self.qubit_map)
        for i in range(size):
            self.qubit_map[(name, i)] = i + start_count
        return ""

    @_parse_ast_statement.register(QuantumGate)
    def _(self, statement, var_map=None, qubit_map=None):
        """.

        Args:
            statement: Statement.
            var_map: Var map. Defaults to ``None``.
            qubit_map: Qubit map. Defaults to ``None``.

        Returns:
            Result.

        Raises:
            NotImplementedError: f'Qasm Modifier {modifiers} is not supported in QCIS order.
        """
        if statement.modifiers:
            modifiers = [i.modifier.name for i in statement.modifiers]
            raise NotImplementedError(f"Qasm Modifier {modifiers} is not supported in QCIS order.")
        gate_name = statement.name.name.lower()
        qubit_index = [self._parse_qubit(qubit, qubit_map=qubit_map) for qubit in statement.qubits]
        args = [self._parse_argument(i, var_map=var_map) for i in statement.arguments] if statement.arguments else None
        gate_instruction = Instruction(gate_name, qubit_index, args)
        if gate_name in self.instruct_convert_rule_dict:
            return self.instruct_convert_rule_dict[gate_name](gate_instruction)
        else:
            raise NotImplementedError(f"QASM Gate {gate_name} is not supported.")

    @_parse_ast_statement.register(QuantumMeasurementStatement)
    def _(self, statement):
        """.

        Args:
            statement: Statement.

        Returns:
            Result.
        """
        qubit = statement.measure.qubit
        qubit_index = [self._parse_qubit(qubit)]
        return [Instruction("m", qubit_index)]

    @_parse_ast_statement.register(QuantumReset)
    def _(self, statement):
        """.

        Args:
            statement: Statement.

        Returns:
            Result.
        """
        qubit = statement.qubits
        qubit_index = [self._parse_qubit(qubit)]
        return [Instruction("rst", qubit_index)]

    @_parse_ast_statement.register(QuantumBarrier)
    def _(self, statement):
        """.

        Args:
            statement: Statement.

        Returns:
            Result.
        """
        qubit_index = [self._parse_qubit(i) for i in statement.qubits]
        return [Instruction("b", qubit_index)]

    @_parse_ast_statement.register(QuantumGateDefinition)
    def _(self, statement):
        """.

        Args:
            statement: Statement.

        Returns:
            Result.
        """
        gate_name = statement.name.name.lower()
        if gate_name in self.instruct_convert_rule_dict:
            return ""

        def temp_gate(input_instruction: Instruction):
            """Apply a user-defined gate by expanding its body statements.

            Args:
                input_instruction (Instruction): The gate instruction with bound arguments and qubits.

            Returns:
                list[Instruction]: The expanded native QCIS instructions.
            """
            res = []
            arg_dict = dict(
                [(arg.name, input_instruction.arguments[i]) for i, arg in enumerate(statement.arguments)]
            )
            for var in self.var_map:
                if var not in arg_dict:
                    arg_dict[var] = self.var_map[var]
            qubit_name_dict = dict(
                [(qubit.name, input_instruction.qubit_index[i]) for i, qubit in enumerate(statement.qubits)]
            )
            for gate_statement in statement.body:
                instruct_temp = self._parse_ast_statement(gate_statement, var_map=arg_dict, qubit_map=qubit_name_dict)
                res.extend(instruct_temp)
            return res

        self.instruct_convert_rule_dict[gate_name] = temp_gate
        return ""

    def convert_to_qcis(self, qasm: str):
        """Convert an OpenQASM program string into a QCIS instruction string.

        Args:
            qasm (str): The OpenQASM program source.

        Returns:
            str: The converted QCIS instruction string.
        """
        qasm_ast = parse(qasm)
        for statement in qasm_ast.statements:
            for instruct_temp in self._parse_ast_statement(statement):
                self.qcis_str += str(instruct_temp)
                self.qcis_str += "\n"
        return self.qcis_str.rstrip("\n")
