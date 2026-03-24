# This code is part of cqlib.
#
# Copyright (C) 2024 China Telecom Quantum Group, QuantumCTek Co., Ltd.,
# Center for Excellence in Quantum Information and Quantum Physics.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from functools import singledispatch, update_wrapper
import inspect
import math
from pathlib import Path
from typing import Union
from openqasm3 import parse
from openqasm3.ast import (
    QuantumGate,
    QuantumReset,
    QuantumMeasurementStatement,
    QuantumBarrier,
    QubitDeclaration,
    ClassicalDeclaration,
    Include,
    IntegerLiteral,
    UnaryExpression,
    FloatLiteral,
    Identifier,
    ImaginaryLiteral,
    BooleanLiteral,
    DurationLiteral,
    BinaryExpression,
    QuantumGateDefinition,
    IndexedIdentifier,
    QuantumPhase,
)
from .data import Instruction
from .utils import _traversal_binary_tree
from .rules import NativeQcisRules

include_file_path_map = {
    'qelib1.inc': Path(__file__).parent / "include/qelib1.inc"
}


def _duration_literal_to_seconds(duration_literal: DurationLiteral) -> float:
    value = float(duration_literal.value)
    unit = getattr(duration_literal, "unit", "s")
    unit_name = unit.name if hasattr(unit, "name") else str(unit)
    scale = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
    }.get(unit_name, 1.0)
    return value * scale


def _meth_dispatch(func):
    """
    Implement singledispatch for instance method within class.

    Args:
        func: input instance func.

    Returns:
        func:wrapped func.

    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


class QasmToQcis:
    """
    Class to transform QASM to QCIS order.

    Only use QCIS native gate by default rules.

    QCIS native gates include: X2P, Y2P, RZ, X2M, Y2M, CZ
    """

    def __init__(self, rule=None):
        if rule is None:
            rule = NativeQcisRules
        self.instruct_convert_rule_dict = dict(
            inspect.getmembers(rule, inspect.isfunction)
        )
        self.qcis_str = ""
        self.qubit_map = {}
        self.var_map = {"pi": round(math.pi, 6), "π": round(math.pi, 6)}

    @_meth_dispatch
    def _parse_argument(self, argument: object, var_map=None):
        raise NotImplementedError(
            f"Invalid input type {type(argument)} found when parsing argument {argument}."
        )

    @_parse_argument.register(IntegerLiteral)
    @_parse_argument.register(FloatLiteral)
    @_parse_argument.register(ImaginaryLiteral)
    @_parse_argument.register(BooleanLiteral)
    def _(self, argument, var_map=None):
        return argument.value

    @_parse_argument.register(DurationLiteral)
    def _(self, argument, var_map=None):
        return _duration_literal_to_seconds(argument)

    @_parse_argument.register(UnaryExpression)
    @_parse_argument.register(Identifier)
    @_parse_argument.register(BinaryExpression)
    def _(self, argument, var_map=None):
        if var_map is None:
            var_map = self.var_map
        return _traversal_binary_tree(argument, var_map)

    @_meth_dispatch
    def _parse_qubit(self, qubit, qubit_map=None):
        raise NotImplementedError(
            f"Invalid input type {type(qubit)} found when parse argument {qubit}."
        )

    @_parse_qubit.register(Identifier)
    def _(self, qubit, qubit_map=None):
        if qubit_map is None:
            raise KeyError(
                f"qubit map not defined when parsing qubit {qubit} with type {type(qubit)}."
            )
        return qubit_map[qubit.name]

    @_parse_qubit.register(IndexedIdentifier)
    def _(self, qubit, qubit_map=None):
        return self.qubit_map[(qubit.name.name, qubit.indices[0][0].value)]

    @_meth_dispatch
    def _parse_ast_statement(self, statement: object, var_map=None, qubit_map=None):
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
            if duration_node is None:
                args = None
            else:
                args = [self._parse_argument(duration_node, var_map=var_map)]
            gate_instruction = Instruction("delay", qubit_index, args)
            return self.instruct_convert_rule_dict["delay"](gate_instruction)
        raise NotImplementedError(
            f"Invalid input type {type(statement)} found when parsing statement {statement}."
        )

    @_parse_ast_statement.register(Include)
    def _(self, statement, var_map=None, qubit_map=None):
        include_file_name = statement.filename
        if include_file_name == "stdgates.inc":
            # OpenQASM3 standard gates are language-level built-ins for parsing.
            return ""
        if not Path(include_file_name).exists():
            if include_file_name in include_file_path_map:
                include_file_path = include_file_path_map[include_file_name]
            else:
                raise FileNotFoundError(
                    f"Include file {include_file_name} not found."
                )
        else:
            include_file_path = Path(include_file_name)
        with open(include_file_path, "r") as f:
            include_qasm_str = f.read()
            self.convert_to_qcis(include_qasm_str)
        return ""

    @_parse_ast_statement.register(ClassicalDeclaration)
    @_parse_ast_statement.register(QuantumPhase)
    def _(self, statement, var_map=None, qubit_map=None):
        return ""

    @_parse_ast_statement.register(QubitDeclaration)
    def _(self, statement):
        name = statement.qubit.name
        size = statement.size.value
        start_count = len(self.qubit_map)
        for i in range(size):
            self.qubit_map[(name, i)] = i + start_count
        return ""

    @_parse_ast_statement.register(QuantumGate)
    def _(self, statement, var_map=None, qubit_map=None):
        # check if gate include modifiers. If so, throw an error:
        if statement.modifiers:
            modifiers = [i.modifier.name for i in statement.modifiers]
            raise NotImplementedError(
                f"Qasm Modifier {modifiers} is not supported in QCIS order."
            )
        gate_name = statement.name.name.lower()
        qubit_index = [
            self._parse_qubit(qubit, qubit_map=qubit_map) for qubit in statement.qubits
        ]
        if statement.arguments:
            args = [
                self._parse_argument(i, var_map=var_map) for i in statement.arguments
            ]
        else:
            args = None
        gate_instruction = Instruction(gate_name, qubit_index, args)
        if gate_name in self.instruct_convert_rule_dict:
            return self.instruct_convert_rule_dict[gate_name](gate_instruction)
        else:
            raise NotImplementedError(f"QASM Gate {gate_name} is not supported.")

    @_parse_ast_statement.register(QuantumMeasurementStatement)
    def _(self, statement):
        qubit = statement.measure.qubit
        qubit_index = [self._parse_qubit(qubit)]
        return [Instruction("m", qubit_index)]

    @_parse_ast_statement.register(QuantumReset)
    def _(self, statement):
        qubit = statement.qubits
        qubit_index = [self._parse_qubit(qubit)]
        return [Instruction("rst", qubit_index)]

    @_parse_ast_statement.register(QuantumBarrier)
    def _(self, statement):
        qubit_index = [self._parse_qubit(i) for i in statement.qubits]
        return [Instruction("b", qubit_index)]

    @_parse_ast_statement.register(QuantumGateDefinition)
    def _(self, statement):
        gate_name = statement.name.name.lower()
        # only include gate which is not defined in rules.
        if gate_name in self.instruct_convert_rule_dict:
            return ""

        # construct name:loc_idx mapping inside gate definition
        # define new gate func. New gate can be decomposed of other native gate or gate defined before the statement.
        def temp_gate(input_instruction: Instruction):
            res = []
            # construct mapping from local var name defined in gate definition to input instruction value.
            arg_dict = dict(
                [
                    (arg.name, input_instruction.arguments[i])
                    for i, arg in enumerate(statement.arguments)
                ]
            )
            # add global var name-value pairs
            for var in self.var_map:
                if var not in arg_dict:
                    arg_dict[var] = self.var_map[var]
            # construct mapping from local qubit name defined in gate definition to global qubit index.
            qubit_name_dict = dict(
                [
                    (qubit.name, input_instruction.qubit_index[i])
                    for i, qubit in enumerate(statement.qubits)
                ]
            )
            for gate_statement in statement.body:
                instruct_temp = self._parse_ast_statement(
                    gate_statement, var_map=arg_dict, qubit_map=qubit_name_dict
                )
                res.extend(instruct_temp)
            return res

        self.instruct_convert_rule_dict[gate_name] = temp_gate
        return ""

    def convert_to_qcis(self, qasm: str):
        """
        Convert QASM string into QCIS string.

        Args:
            qasm (str): input qasm string.

        Returns:
            str: converted qcis string.

        """
        qasm_ast = parse(qasm)
        for statement in qasm_ast.statements:
            for instruct_temp in self._parse_ast_statement(statement):
                self.qcis_str += str(instruct_temp)
                self.qcis_str += "\n"

        return self.qcis_str.rstrip("\n")

    def convert_to_qcis_from_file(self, qasm_file: Union[str, Path]):
        """
        Convert QASM order from file into QCIS string.

        Args:
            qasm_file (str | Path): input qasm file path.

        Returns:
            str: converted qcis string.
        """
        with open(qasm_file, "r") as f:
            qasm_str = f.read()
            return self.convert_to_qcis(qasm_str)
