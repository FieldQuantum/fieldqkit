"""Resolve gate angles and their parameter Jacobians without a simulator.

Parameter-shift identities apply to individual gate angles.  A named parameter
may occur in several angles, or inside an expression, so its gradient requires
the chain rule.  Forward-mode scalars let the circuit's existing safe expression
parser compute these small classical Jacobians using only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np

from ..circuit import QuantumCircuit


@dataclass(frozen=True)
class _AngleValue:
    """A real expression value and its derivative with respect to parameters."""

    value: float
    derivative: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.value) or not np.all(np.isfinite(self.derivative)):
            raise ValueError("gate angle or its derivative is not finite")

    def _coerce(self, other):
        if isinstance(other, _AngleValue):
            return other
        if isinstance(other, Real):
            return _AngleValue(float(other), np.zeros_like(self.derivative))
        raise TypeError("gate angle expressions must be real-valued")

    def __add__(self, other):
        other = self._coerce(other)
        return _AngleValue(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return _AngleValue(-self.value, -self.derivative)

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        return _AngleValue(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if other.value == 0.0:
            raise ValueError("division by zero in a gate angle expression")
        value = self.value / other.value
        return _AngleValue(value, (self.derivative - value * other.derivative) / other.value)

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, other):
        other = self._coerce(other)
        if not np.any(other.derivative):
            # Constant exponents need no logarithm.  In particular, x**2 has
            # derivative zero at x=0, and negative bases allow integer powers.
            exponent = other.value
            if exponent == 0.0:
                return self._coerce(1.0)
            if exponent == 1.0:
                return self
            if self.value < 0.0 and not exponent.is_integer():
                raise ValueError("negative gate angle base requires an integer exponent")
            value = self.value ** exponent
            if not np.any(self.derivative):
                return self._coerce(value)
            factor = exponent * self.value ** (exponent - 1.0)
            return _AngleValue(value, factor * self.derivative)

        if self.value <= 0.0:
            raise ValueError("a variable exponent requires a positive gate angle base")
        value = self.value ** other.value
        return _AngleValue(
            value,
            value * (
                other.derivative * np.log(self.value)
                + other.value * self.derivative / self.value
            ),
        )

    def __rpow__(self, other):
        return self._coerce(other) ** self


def parameter_shift_terms(
    template: QuantumCircuit,
    param_names: Sequence[str],
    params: np.ndarray,
) -> list[tuple[int, int, float, np.ndarray]]:
    """Return differentiable gate angles for occurrence-wise parameter shift.

    Each result is ``(gate_index, angle_index, value, derivative)``.  The angle
    index addresses the circuit's gate tuple (its name occupies index zero),
    and ``derivative`` follows the order of ``param_names``.  Constant angles
    and angles whose derivatives are all zero at this point are omitted.

    Supports RX/RY/RZ, RXX/RYY/RZZ, and all three U angles, including the same
    arithmetic expressions supported by ``QuantumCircuit``.  Numeric entries
    in ``template.params_value`` supply fixed parameter values; the explicit
    trainable parameter vector takes precedence.  The template is not changed.

    Raises:
        ValueError: For an invalid parameter vector or name mapping, unresolved
            symbols, non-real/non-finite angles, or undefined derivatives.
    """
    names = list(param_names)
    values = np.asarray(params, dtype=float)
    if values.ndim != 1 or len(names) != values.size:
        raise ValueError("params must be a one-dimensional vector matching param_names")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("parameter names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("param_names must be unique")
    if not np.all(np.isfinite(values)):
        raise ValueError("params must contain only finite values")

    derivatives = np.eye(values.size, dtype=float)
    symbols = {
        name: _AngleValue(float(value), derivatives[index])
        for index, (name, value) in enumerate(zip(names, values))
    }
    numeric_bindings = dict(zip(names, values.tolist()))

    def constant(value):
        if not isinstance(value, Real):
            raise ValueError("gate angle expressions must resolve to real numbers")
        return _AngleValue(float(value), np.zeros(values.size, dtype=float))

    def resolve_trainable_symbol(name):
        if name == "pi":
            return float(np.pi)
        if name in symbols:
            return symbols[name]
        raise ValueError(f"missing parameter value for {name}")

    def resolve_symbol(name):
        if name == "pi" or name in symbols:
            return resolve_trainable_symbol(name)
        value = template.params_value.get(name)
        if isinstance(value, Real):
            return constant(value)
        raise ValueError(f"missing parameter value for {name}")

    angle_slots = {
        "rx": (1,), "ry": (1,), "rz": (1,),
        "rxx": (1,), "ryy": (1,), "rzz": (1,),
        "u": (1, 2, 3),
    }
    terms = []
    for gate_index, gate in enumerate(template.gates):
        for angle_index in angle_slots.get(gate[0], ()):
            expression = gate[angle_index]
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    if isinstance(expression, str):
                        if expression in symbols:
                            resolved = symbols[expression]
                        else:
                            # Select the same path as deep binding before
                            # differentiating.  An unresolved expression may
                            # be overridden by a fixed full-expression entry,
                            # even if a subexpression has no finite derivative.
                            bound = template._resolve_expr(expression, numeric_bindings)
                            if not isinstance(bound, str):
                                resolved = template._eval_param_expression(
                                    expression, symbol_resolver=resolve_trainable_symbol,
                                )
                            else:
                                # Unresolved strings survive deep binding and
                                # then use the template's normal lookup order.
                                bound_value = template.params_value.get(expression)
                                if isinstance(bound_value, Real):
                                    resolved = constant(bound_value)
                                else:
                                    resolved = template._eval_param_expression(
                                        expression, symbol_resolver=resolve_symbol,
                                    )
                    else:
                        resolved = constant(expression)
                    if not isinstance(resolved, _AngleValue):
                        resolved = constant(resolved)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
                raise ValueError(
                    f"cannot differentiate angle {angle_index} of gate {gate_index} "
                    f"({gate[0]}): {exc}"
                ) from exc
            if np.any(resolved.derivative):
                terms.append((gate_index, angle_index, resolved.value, resolved.derivative.copy()))
    return terms
