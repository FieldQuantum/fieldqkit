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

from functools import partial
import operator
from openqasm3.ast import (
    IntegerLiteral,
    UnaryExpression,
    FloatLiteral,
    Identifier,
    ImaginaryLiteral,
    BooleanLiteral,
    DurationLiteral,
    BinaryExpression,
)

unary_operator = {
    # undefined unary operator ~ !
    "-": partial(operator.sub, 0)
}
binary_operator_map = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    # '&&': operator.and_,
    "&": operator.and_,
    # '||': operator.or_,
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
    if isinstance(
        node,
        (
            IntegerLiteral,
            FloatLiteral,
            ImaginaryLiteral,
            BooleanLiteral,
            DurationLiteral,
        ),
    ):
        return node.value
    elif isinstance(node, Identifier):
        return var_map[node.name]
    elif isinstance(node, BinaryExpression):
        if node.op.name not in binary_operator_map:
            raise NotImplementedError(
                f"Binary operator {node.op.name} not implemented."
            )
        return binary_operator_map[node.op.name](
            _traversal_binary_tree(node.lhs, var_map),
            _traversal_binary_tree(node.rhs, var_map),
        )
    elif isinstance(node, UnaryExpression):
        if node.op.name not in unary_operator:
            raise NotImplementedError(f"Unary operator {node.op.name} not implemented.")
        return unary_operator[node.op.name](
            _traversal_binary_tree(node.expression, var_map)
        )
    else:
        raise TypeError(
            f"Invalid input type {type(node)} found when traversing binary tree."
        )
