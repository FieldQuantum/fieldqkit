"""OpenQASM 2.0 Parser"""

import re
import numpy as np

from .quantumcircuit_helpers import (
    parse_expression,
    one_qubit_gates_available,
    two_qubit_gates_available,
    three_qubit_gates_available,
    one_qubit_parameter_gates_available,
    two_qubit_parameter_gates_available,
    functional_gates_available,
)

__all__ = [
    "parse_expression",
    "parse_openqasm2_regs",
    "parse_openqasm2_custom_gates",
    "parse_openqasm2_to_gates",
]


def _record_qubits(qubit_used: list, *qubits: int) -> None:
    """Record qubits.

    Args:
        qubit_used (*list*): Qubit used (``list``).
        *qubits (*int*): *qubits (``int``).
    """
    qubit_used.extend(qubits)


def parse_openqasm2_regs(openqasm2_str: str):
    """Parse openqasm2 regs.

    Args:
        openqasm2_str (*str*): Openqasm2 str (``str``).

    Returns:
        Parsed result.
    """
    qreg_pattern = r"qreg\s+(\w+)\[(\d+)\];"
    creg_pattern = r"creg\s+(\w+)\[(\d+)\];"
    qregs = []
    cregs = []
    for name, size in re.findall(qreg_pattern, openqasm2_str):
        qregs.append((name, int(size)))
    for name, size in re.findall(creg_pattern, openqasm2_str):
        cregs.append((name, int(size)))
    new_lines = []
    for line in openqasm2_str.splitlines():
        if re.search(qreg_pattern, line) or re.search(creg_pattern, line):
            continue
        new_lines.append(line)

    new_qasm = "\n".join(new_lines)
    return qregs, cregs, new_qasm


def parse_openqasm2_custom_gates(openqasm2_str: str):
    """Parse custom gate definitions and remove their blocks, keeping calls intact.

    Args:
        openqasm2_str (*str*): Openqasm2 str (``str``).

    Returns:
        Parsed result.

    Raises:
        ValueError: f'parse error {name} !
    """
    def parse_instruction(line: str):
        """Parse instruction.

        Args:
            line (*str*): Line (``str``).

        Returns:
            Parsed result.

        Raises:
            ValueError: f'parse error {name} !
        """
        line = line.strip().rstrip(";")
        pattern = r"^(\w+)\s*(?:\(([^)]*)\))?\s*(.*)$"
        m = re.match(pattern, line)
        if not m:
            return None
        name = m.group(1)
        if name in one_qubit_gates_available:
            pass
        elif name in two_qubit_gates_available:
            pass
        elif name in three_qubit_gates_available:
            pass
        elif name in one_qubit_parameter_gates_available:
            pass
        elif name in two_qubit_parameter_gates_available:
            pass
        elif name in functional_gates_available:
            pass
        else:
            raise (ValueError(f"parse error {name} !"))

        params = [p.strip() for p in m.group(2).split(",")] if m.group(2) else []
        qargs = [q.strip() for q in re.split(r"[, ]+", m.group(3).strip()) if q.strip()]
        return (name, *params, *qargs)

    pattern = r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s+([a-zA-Z0-9_,\s]+)\s*\{([^}]*)\}"
    gate_pattern = re.compile(pattern, re.DOTALL)
    gates = {}
    for match in gate_pattern.finditer(openqasm2_str):
        name = match.group(1).strip()
        params_str = [p.strip() for p in match.group(2).split(",")] if match.group(2) else []
        params = [str(parse_expression(i)) if "pi" in i or "π" in i else i for i in params_str]
        qargs = [q.strip() for q in match.group(3).split(",") if q.strip()]
        body_raw = match.group(4).strip()

        body = []
        for stmt in body_raw.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            parsed = parse_instruction(stmt)
            if parsed:
                body.append(parsed)

        gates[name] = {"params_and_qregs": params + qargs, "definition": body}
    openqasm2_str = re.sub(pattern, "", openqasm2_str, flags=re.DOTALL)
    return gates, openqasm2_str


def generate_reg_map(regs):
    """Generate reg map.

    Args:
        regs: Regs.

    Returns:
        Result.
    """
    num = sum(v for _, v in regs)
    all_reg = [i for i in range(num)]

    reg_map = {}
    idx = 0
    for reg, num in regs:
        reg_map[reg] = dict(zip(range(num), all_reg[idx : idx + num]))
        idx += num
    return reg_map


def sparse_gate_params_qregs(line):
    """Parse a QASM gate line into gate name, parameter string, and qubit register string.

    Args:
        line: A single QASM instruction line.

    Returns:
        Tuple of ``(gate, params_str, qregs_str)``, or ``(None, None, None)``
        if the line cannot be parsed.
    """
    pattern_measure = re.compile(
        r"""
        ^\s*
        measure
        \s+
        (?P<qreg>\w+(?:\[\d+\])?)
        \s*
        ->
        \s*
        (?P<creg>\w+(?:\[\d+\])?)
        \s*
        ;
        \s*$
    """,
        re.VERBOSE,
    )
    m = pattern_measure.match(line)
    if m:
        return "measure", None, f"{m.group('qreg')},{m.group('creg')}"

    pattern = re.compile(
        r"""
        ^\s*
        (?P<gate>\w+)
        (?:\((?P<params>[^\)]*)\))?
        \s+
        (?P<qregs>[\w\[\],\s]+)
        \s*;
        $
    """,
        re.VERBOSE,
    )
    m = pattern.match(line)
    if m is None:
        return None, None, None

    gate = m.group("gate")
    params_str = m.group("params")
    qregs_str = m.group("qregs")

    return gate, params_str, qregs_str


def get_positions_list(gate, qregs_str, qreg_map, creg_map):
    """Get positions list.

    Args:
        gate: Gate specification or name.
        qregs_str: Qregs str.
        qreg_map: Qreg map.
        creg_map: Creg map.

    Returns:
        Retrieved data.

    Raises:
        ValueError: measurement gate: position parse error
    """
    try:
        qregs = qregs_str.split(",")
    except Exception:
        qregs = []
    cc = []
    if qregs != []:
        for qq in qregs:
            qs = re.findall(r"(\w+)\[(\d+)\]", qq)
            if qs != []:
                cc += qs
            else:
                cc.append(qq)
    if gate == "measure":
        if len(cc) != 2:
            raise ValueError("measurement gate: position parse error")
        q_position = []
        if isinstance(cc[0], tuple):
            q_position.append(qreg_map[cc[0][0]][int(cc[0][1])])
        if isinstance(cc[0], str):
            if cc[0] in qreg_map:
                q_position += list(qreg_map[cc[0]].values())
        c_position = []
        if isinstance(cc[1], tuple):
            c_position.append(creg_map[cc[1][0]][int(cc[1][1])])
        if isinstance(cc[1], str):
            if cc[1] in creg_map:
                c_position += list(creg_map[cc[1]].values())
        positions = [q_position, c_position]
    else:
        positions = []
        for c in cc:
            if isinstance(c, tuple):
                positions.append([qreg_map[c[0]][int(c[1])]])
            if isinstance(c, str):
                if c in qreg_map:
                    positions.append(list(qreg_map[c].values()))
    return positions


def parse_openqasm2_to_gates(openqasm2_str):
    """Parse gate information from an input OpenQASM 2.0 string, and update gates, supporting multiple registers.

    Args:
        openqasm2_str: Openqasm2 str.

    Returns:
        Parsed result.

    Raises:
        ValueError: f'{gate} takes 2 quantum arguments, but got {len(position...
    """
    qregs_used, cregs_used, openqasm2_str = parse_openqasm2_regs(openqasm2_str)
    qreg_map = generate_reg_map(qregs_used)
    creg_map = generate_reg_map(cregs_used)

    custom_gates, openqasm2_str = parse_openqasm2_custom_gates(openqasm2_str)

    new = []
    qubit_used = []
    cbit_used = []
    clean_qasm = openqasm2_str.strip()
    for line in clean_qasm.splitlines():
        if line == "":
            continue
        elif set(line) == {"\t"}:
            continue
        line_clear = line.split("//")[0].strip()
        gate, params_str, qregs_str = sparse_gate_params_qregs(line_clear)
        if params_str is not None:
            params = [parse_expression(p) for p in params_str.split(",")]
        else:
            params = []
        positions = get_positions_list(gate, qregs_str, qreg_map, creg_map)
        if gate in one_qubit_gates_available:
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate, q))
                _record_qubits(qubit_used, q)
        elif gate in two_qubit_gates_available:
            if len(positions) != 2:
                raise ValueError(f"{gate} takes 2 quantum arguments, but got {len(positions)}.")
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f"{gate} takes 2 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append((gate, positions[0][idx], positions[1][idx]))
                _record_qubits(qubit_used, positions[0][idx], positions[1][idx])
        elif gate in three_qubit_gates_available:
            if len(positions) != 3:
                raise ValueError(f"{gate} takes 3 quantum arguments, but got {len(positions)}.")
            if len(positions[0]) != len(positions[1]) or len(positions[0]) != len(positions[2]):
                raise ValueError(f"{gate} takes 3 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append((gate, positions[0][idx], positions[1][idx], positions[2][idx]))
                _record_qubits(qubit_used, positions[0][idx], positions[1][idx], positions[2][idx])
        elif gate in one_qubit_parameter_gates_available:
            qubits = [p for pp in positions for p in pp]
            if gate == "u" or gate == "u3":
                for q in qubits:
                    new.append(("u", params[0], params[1], params[2], q))
                    _record_qubits(qubit_used, q)
            elif gate == "r":
                for q in qubits:
                    new.append((gate, params[0], params[1], q))
                    _record_qubits(qubit_used, q)
            else:
                for q in qubits:
                    new.append((gate, *params, q))
                    _record_qubits(qubit_used, q)
        elif gate in ["u1", "u2"]:
            qubits = [p for pp in positions for p in pp]
            if gate == "u1":
                for q in qubits:
                    new.append(("u", 0, 0, params[0], q))
                    _record_qubits(qubit_used, q)
            elif gate == "u2":
                for q in qubits:
                    new.append(("u", np.pi / 2, params[0], params[1], q))
                    _record_qubits(qubit_used, q)
        elif gate in two_qubit_parameter_gates_available:
            if len(positions) != 2:
                raise ValueError(f"{gate} takes 2 quantum arguments, but got {len(positions)}.")
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f"{gate} takes 2 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append((gate, *params, positions[0][idx], positions[1][idx]))
                _record_qubits(qubit_used, positions[0][idx], positions[1][idx])
        elif gate in ["cu1"]:
            if len(positions) != 2:
                raise ValueError(f"{gate} takes 2 quantum arguments, but got {len(positions)}.")
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f"{gate} takes 2 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append(("cp", *params, positions[0][idx], positions[1][idx]))
                _record_qubits(qubit_used, positions[0][idx], positions[1][idx])
        elif gate in ["delay"]:
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate, *params, (q,)))
                _record_qubits(qubit_used, q)
        elif gate in ["reset"]:
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate, q))
                _record_qubits(qubit_used, q)
        elif gate in ["barrier"]:
            qubits = [p for pp in positions for p in pp]
            new.append((gate, tuple(qubits)))
        elif gate in ["measure"]:
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f"{gate} takes 2 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append((gate, [positions[0][idx]], [positions[1][idx]]))
                _record_qubits(qubit_used, positions[0][idx])
                cbit_used.append(positions[1][idx])
        elif gate in ["OPENQASM", "include", "opaque", "gate", "qreg", "creg", "//"]:
            continue
        elif gate in custom_gates:
            positions_lengths = [len(position) for position in positions]
            if len(set(positions_lengths)) > 1:
                raise ValueError(f"custom gate {gate} sparse failer!")
            for idx in range(len(positions[0])):
                qubits = [position[idx] for position in positions]
                params_qreg_dic = dict(zip(custom_gates[gate]["params_and_qregs"], params + qubits))
                new0 = []
                for gate0_info in custom_gates[gate]["definition"]:
                    cc = []
                    for key in gate0_info[1:]:
                        if key in params_qreg_dic:
                            cc.append(params_qreg_dic[key])
                        else:
                            if key.isdigit():
                                cc.append(int(key))
                            else:
                                cc.append(parse_expression(key))
                    new0.append(tuple([gate0_info[0]] + cc))
                new += new0
                _record_qubits(qubit_used, *qubits)
        elif gate is None:
            pass
        else:
            raise (
                ValueError(
                    f"Sorry, an unrecognized OpenQASM 2.0 syntax {gate} was detected. Please contact the developer for assistance."
                )
            )

    if cbit_used == []:
        cbit_used = [i for i in range(len(set(qubit_used)))]
    return new, set(qubit_used), set(cbit_used)
