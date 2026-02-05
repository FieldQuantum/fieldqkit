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
    "parse_openqasm2_to_gates_dump",
]


def _record_qubits(qubit_used: list, *qubits: int) -> None:
    qubit_used.extend(qubits)


def parse_openqasm2_regs(openqasm2_str: str):
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
    """Parse custom gate definitions and remove their blocks, keeping calls intact."""
    def parse_instruction(line: str):
        line = line.strip().rstrip(";")
        pattern = r"^(\w+)\s*(?:\(([^)]*)\))?\s*(.*)$"
        m = re.match(pattern, line)
        if not m:
            return None
        name = m.group(1)
        if name in one_qubit_gates_available.keys():
            pass
        elif name in two_qubit_gates_available.keys():
            pass
        elif name in three_qubit_gates_available.keys():
            pass
        elif name in one_qubit_parameter_gates_available.keys():
            pass
        elif name in two_qubit_parameter_gates_available.keys():
            pass
        elif name in functional_gates_available.keys():
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
    print("Custom gate detected")
    for k, v in gates.items():
        print(k, v)
    return gates, openqasm2_str


def generate_reg_map(regs, type: str = ""):
    num = sum([v for _, v in regs])
    all_reg = [i for i in range(num)]

    reg_map = {}
    idx = 0
    for reg, num in regs:
        reg_map[reg] = dict(zip(range(num), all_reg[idx : idx + num]))
        idx += num
    if len(reg_map) > 1:
        for k, v in reg_map.items():
            print(f"{type} reg name {k}, mapping:{v}")
    return reg_map


def sparse_gate_params_qregs(line):
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
            if cc[0] in qreg_map.keys():
                q_position += list(qreg_map[cc[0]].values())
        c_position = []
        if isinstance(cc[1], tuple):
            c_position.append(creg_map[cc[1][0]][int(cc[1][1])])
        if isinstance(cc[1], str):
            if cc[1] in creg_map.keys():
                c_position += list(creg_map[cc[1]].values())
        positions = [q_position, c_position]
    else:
        positions = []
        for c in cc:
            if isinstance(c, tuple):
                positions.append([qreg_map[c[0]][int(c[1])]])
            if isinstance(c, str):
                if c in qreg_map.keys():
                    positions.append(list(qreg_map[c].values()))
    return positions


def parse_openqasm2_to_gates(openqasm2_str) -> None:
    r"""
    Parse gate information from an input OpenQASM 2.0 string, and update gates, supporting multiple registers.
    """
    qregs_used, cregs_used, openqasm2_str = parse_openqasm2_regs(openqasm2_str)
    if len(qregs_used) > 1 or len(cregs_used) > 1:
        print("Multiple registers detected. For subsequent compilation, the program will merge them. The mapping is as follows:")
    qreg_map = generate_reg_map(qregs_used, "Qubit")
    creg_map = generate_reg_map(cregs_used, "Cbit")

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
        if gate in one_qubit_gates_available.keys():
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate, q))
                _record_qubits(qubit_used, q)
        elif gate in two_qubit_gates_available.keys():
            if len(positions) != 2:
                raise ValueError(f"{gate} takes 2 quantum arguments, but got {len(positions)}.")
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f"{gate} takes 2 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append((gate, positions[0][idx], positions[1][idx]))
                _record_qubits(qubit_used, positions[0][idx], positions[1][idx])
        elif gate in three_qubit_gates_available.keys():
            if len(positions) != 3:
                raise ValueError(f"{gate} takes 3 quantum arguments, but got {len(positions)}.")
            if len(positions[0]) != len(positions[1]) or len(positions[0]) != len(positions[2]):
                raise ValueError(f"{gate} takes 3 different quantum arguments length.")
            for idx in range(len(positions[0])):
                new.append((gate, positions[0][idx], positions[1][idx], positions[2][idx]))
                _record_qubits(qubit_used, positions[0][idx], positions[1][idx], positions[2][idx])
        elif gate in one_qubit_parameter_gates_available.keys():
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
        elif gate in two_qubit_parameter_gates_available.keys():
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
        elif gate in custom_gates.keys():
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
                        if key in params_qreg_dic.keys():
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
                    f"Sorry, an unrecognized OpenQASM 2.0 syntax {gate} was detected by quarkcircuit. Please contact the developer for assistance."
                )
            )

    if cbit_used == []:
        cbit_used = [i for i in range(len(set(qubit_used)))]
    return new, set(qubit_used), set(cbit_used)


def parse_openqasm2_to_gates_dump(openqasm2_str) -> None:
    r"""
    Parse gate information from an input OpenQASM 2.0 string, and update gates
    """
    qregs, cregs, _ = parse_openqasm2_regs(openqasm2_str)
    if len(qregs) > 1 or len(cregs) > 1:
        raise (ValueError("Sorry, currently only one quantum or classical register definition is supported"))

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
        gate = line.split()[0].split("(")[0]
        position = [int(num) for num in re.findall(r"\[(\d+)\]", line)]
        if gate in one_qubit_gates_available.keys():
            new.append((gate, position[0]))
            qubit_used.append(position[0])
        elif gate in two_qubit_gates_available.keys():
            new.append((gate, position[0], position[1]))
            qubit_used.append(position[0])
            qubit_used.append(position[1])
        elif gate in three_qubit_gates_available.keys():
            new.append((gate, position[0], position[1], position[2]))
            qubit_used.append(position[0])
            qubit_used.append(position[1])
            qubit_used.append(position[2])
        elif gate in one_qubit_parameter_gates_available.keys():
            if gate == "u" or gate == "u3":
                params_str = re.search(r"\(([^)]+)\)", line).group(1).split(",")
                params = [parse_expression(i) for i in params_str]
                new.append(("u", params[0], params[1], params[2], position[-1]))
                qubit_used.append(position[-1])
            elif gate == "r":
                params_str = re.search(r"\(([^)]+)\)", line).group(1).split(",")
                params = [parse_expression(i) for i in params_str]
                new.append((gate, params[0], params[1], position[-1]))
                qubit_used.append(position[-1])
            else:
                param_str = re.search(r"\(([^)]+)\)", line).group(1)
                param = parse_expression(param_str)
                new.append((gate, param, position[-1]))
                qubit_used.append(position[-1])
        elif gate in ["u1", "u2"]:
            if gate == "u1":
                params_str = re.search(r"\(([^)]+)\)", line).group(1).split(",")
                params = [parse_expression(i) for i in params_str]
                new.append(("u", 0, 0, params[0], position[-1]))
                qubit_used.append(position[-1])
            elif gate == "u2":
                params_str = re.search(r"\(([^)]+)\)", line).group(1).split(",")
                params = [parse_expression(i) for i in params_str]
                new.append(("u", np.pi / 2, params[0], params[1], position[-1]))
                qubit_used.append(position[-1])
        elif gate in two_qubit_parameter_gates_available.keys():
            param_str = re.search(r"\(([^)]+)\)", line).group(1)
            param = parse_expression(param_str)
            new.append((gate, param, position[-2], position[-1]))
            qubit_used.append(position[-2])
            qubit_used.append(position[-1])
        elif gate in ["cu1"]:
            param_str = re.search(r"\(([^)]+)\)", line).group(1)
            param = parse_expression(param_str)
            new.append(("cp", param, position[-2], position[-1]))
            qubit_used.append(position[-2])
            qubit_used.append(position[-1])
        elif gate in ["delay"]:
            param = float(re.search(r"\(([^)]+)\)", line).group(1))
            new.append((gate, param, (position[-1],)))
            qubit_used.append(position[-1])
        elif gate in ["reset"]:
            new.append((gate, position[0]))
            qubit_used.append(position[0])
        elif gate in ["barrier"]:
            if position == []:
                line0 = line.strip().rstrip(";")
                pattern = r"barrier\s+(.+?)"
                match = re.match(pattern, line0)
                q_name = match.groups()[0]
                if q_name == qregs[0][0]:
                    new.append((gate, tuple([i for i in range(qregs[0][1])])))
                else:
                    raise (ValueError(f"Sorry, an unrecognized OpenQASM 2.0 syntax {line} was detected by quarkcircuit."))
            else:
                new.append((gate, tuple(position)))
        elif gate in ["measure"]:
            if position == []:
                line0 = line.strip().rstrip(";")
                pattern = r"measure\s+(.+?)\s*->\s*(.+)"
                match = re.match(pattern, line0)
                left, right = match.groups()
                q_name = left.strip()
                c_name = right.strip()
                if q_name == qregs[0][0] and c_name == cregs[0][0] and qregs[0][1] == cregs[0][1]:
                    for q in range(qregs[0][1]):
                        new.append(("measure", [q], [q]))
                else:
                    raise (
                        ValueError(
                            f"check qreg name {qregs[0][0]} and parse q_name {q_name} is consistent, \
                    \ncheck creg name {cregs[0][0]} and parse c_name {c_name} is consistent, \
                    \ncheck qregs size {qregs[0][1]} and cregs size {cregs[0][1]} is equal."
                        )
                    )
            else:
                new.append((gate, [position[0]], [position[1]]))
                qubit_used.append(position[0])
                cbit_used.append(position[1])
        elif gate in ["OPENQASM", "include", "opaque", "gate", "qreg", "creg", "//"]:
            continue
        elif gate in custom_gates.keys():
            try:
                params_str = re.search(r"\(([^)]+)\)", line).group(1).split(",")
                params = [parse_expression(i) for i in params_str]
            except Exception:
                params = []
            params_qreg_dic = dict(zip(custom_gates[gate]["params_and_qregs"], params + position))
            new0 = []
            for gate0_info in custom_gates[gate]["definition"]:
                cc = []
                for key in gate0_info[1:]:
                    if key in params_qreg_dic.keys():
                        cc.append(params_qreg_dic[key])
                    else:
                        if key.isdigit():
                            cc.append(int(key))
                        else:
                            cc.append(parse_expression(key))
                new0.append(tuple([gate0_info[0]] + cc))
            new += new0
        else:
            raise (
                ValueError(
                    f"Sorry, an unrecognized OpenQASM 2.0 syntax {gate} was detected by quarkcircuit. Please contact the developer for assistance."
                )
            )

    if cbit_used == []:
        cbit_used = [i for i in range(len(set(qubit_used)))]
    return new, set(qubit_used), set(cbit_used)
