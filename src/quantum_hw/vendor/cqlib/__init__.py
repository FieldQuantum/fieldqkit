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

"""Vendored subset of cqlib used by quantum_hw API adapters."""

from .exceptions import CqlibError, CqlibInputParaError, CqlibRequestError

__all__ = [
    "CqlibError",
    "CqlibRequestError",
    "CqlibInputParaError",
]
