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

"""Exceptions for errors raised by cqlib."""


class CqlibError(Exception):
    """Base class for errors raised by Cqlib."""

    def __init__(self, message):
        """Set the error message."""
        super().__init__(message)
        self.message = message

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class VisualizationError(CqlibError):
    """
    A custom exception class for representing errors in visualization processes.

    """

    def __init__(self, message):
        """
        Initializes a VisualizationError exception instance.

        Args:
            message: str - The error message describing the issue in detail.
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """
        Returns a string representation of the exception.

        Returns:
            str - The error message as a string.
        """
        return self.message


class CqlibRequestError(CqlibError):
    """Class for request errors raised by Cqlib."""

    def __init__(self, message, status_code=None):
        """Initialize the exception with a message and optional status code."""
        super().__init__(message)
        self.status_code = status_code
        if status_code is not None:
            self.message = f"Request failed with status code {status_code}: {message}"
        else:
            self.message = message


class CqlibInputParaError(CqlibError):
    """Class for input errors raised by Cqlib."""


class QASMParserError(CqlibError):
    """Exception raised for errors in the OpenQASM parser within Cqlib."""

    def __init__(self, message="Error occurred during OpenQASM parsing."):
        """Set the error message specific to QASM parsing."""
        super().__init__(message)
