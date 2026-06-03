"""QCIS instruction primitives and direct QuantumCircuit-to-QCIS conversion."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Instruction dataclass
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """QCIS instruction primitive."""

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
                    if abs(abs(i) - math.pi) < 1e-12:
                        i = math.copysign(math.pi - 1e-10, i)
                instr_str += f"{i} "
        return instr_str.rstrip()

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
        """Decompose U(θ,φ,λ) gate into five native instructions: RZ(λ), X2P, RZ(θ), X2M, RZ(φ).

        ``inp.arguments`` is ordered ``[θ, φ, λ]`` (matching ``QuantumCircuit.u`` and
        ``u_mat``). The native sequence reconstructs ``U(θ,φ,λ)`` up to a global phase.

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


# ---------------------------------------------------------------------------
# Direct QuantumCircuit → QCIS conversion
# ---------------------------------------------------------------------------


def circuit_to_qcis(qc) -> str:
    """Convert a QuantumCircuit directly to a QCIS instruction string.

    Translates gate tuples from ``qc.gates`` using :class:`NativeQcisRules`,
    bypassing the QASM text round-trip.  The circuit should already be
    transpiled (basis gates only) before calling this function.

    Args:
        qc: :class:`~fieldqkit.circuit.QuantumCircuit` to convert.

    Returns:
        str: QCIS instruction string.

    Raises:
        NotImplementedError: If a gate has no :class:`NativeQcisRules` decomposition.
    """
    from ..circuit.quantumcircuit_helpers import (
        one_qubit_gates_available,
        one_qubit_parameter_gates_available,
        two_qubit_gates_available,
        three_qubit_gates_available,
        two_qubit_parameter_gates_available,
    )

    rules = dict(inspect.getmembers(NativeQcisRules, inspect.isfunction))
    lines: list[str] = []

    for gate_info in qc.gates:
        gate = gate_info[0]
        instrs: list[Instruction] = []

        if gate in one_qubit_gates_available:
            instr = Instruction(gate, [gate_info[1]])
            if gate not in rules:
                raise NotImplementedError(f"Gate '{gate}' has no NativeQcisRules decomposition.")
            instrs = rules[gate](instr)

        elif gate in one_qubit_parameter_gates_available:
            # gate_info layout: (gate, *params, qubit)
            qubit = gate_info[-1]
            params = list(gate_info[1:-1])
            instr = Instruction(gate, [qubit], params)
            if gate not in rules:
                raise NotImplementedError(f"Gate '{gate}' has no NativeQcisRules decomposition.")
            instrs = rules[gate](instr)

        elif gate in two_qubit_gates_available:
            instr = Instruction(gate, [gate_info[1], gate_info[2]])
            if gate not in rules:
                raise NotImplementedError(f"Gate '{gate}' has no NativeQcisRules decomposition.")
            instrs = rules[gate](instr)

        elif gate in three_qubit_gates_available:
            instr = Instruction(gate, [gate_info[1], gate_info[2], gate_info[3]])
            if gate not in rules:
                raise NotImplementedError(f"Gate '{gate}' has no NativeQcisRules decomposition.")
            instrs = rules[gate](instr)

        elif gate in two_qubit_parameter_gates_available:
            # gate_info layout: (gate, theta, qubit1, qubit2)
            instr = Instruction(gate, [gate_info[2], gate_info[3]], [gate_info[1]])
            if gate not in rules:
                raise NotImplementedError(f"Gate '{gate}' has no NativeQcisRules decomposition.")
            instrs = rules[gate](instr)

        elif gate == "measure":
            # gate_info: ('measure', [qubits], [cbits])
            for qubit in gate_info[1]:
                instrs.append(Instruction("m", [qubit]))

        elif gate == "reset":
            # gate_info: ('reset', qubit)
            instrs = [Instruction("rst", [gate_info[1]])]

        elif gate == "barrier":
            # gate_info: ('barrier', qubits_tuple)
            instrs = [Instruction("b", list(gate_info[1]))]

        elif gate == "delay":
            # gate_info: ('delay', duration_seconds, qubits_tuple)
            # QuantumCircuit stores duration in seconds; QCIS I expects nanoseconds.
            duration_ns = round(gate_info[1] * 1e9)
            for qubit in gate_info[2]:
                instrs.append(Instruction("i", [qubit], [duration_ns]))

        else:
            raise NotImplementedError(f"Gate '{gate}' is not supported in circuit_to_qcis.")

        for instr in instrs:
            lines.append(str(instr))

    return "\n".join(lines)
