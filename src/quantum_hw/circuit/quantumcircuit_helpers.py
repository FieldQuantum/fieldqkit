# Copyright (c) 2024 XX Xiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

r""" 
The module contains some tools for quantumcircuit.
"""
import numpy as np
import copy,re

one_qubit_gates_available = {
    'id':'I', 'x':'X', 'y':'Y', 'z':'Z',
    's':'S', 'sdg':'Sdg','t':'T', 'tdg':'Tdg',
    'h':'H', 'sx':'√X','sxdg':'√Xdg',
    }
two_qubit_gates_available = {
    'cx':'●X', 'cnot':'●X', 'cy':'●Y', 'cz':'●Z', 'swap':'XX', 'iswap':'✶✶',
    } 
three_qubit_gates_available = {'ccz':'●●●','ccx':'●●X','cswap':'●XX'} 
one_qubit_parameter_gates_available = {'rx':'Rx', 'ry':'Ry', 'rz':'Rz', 'p':'P', 'u':'U', 'u3':'U', 'r':'R'}
two_qubit_parameter_gates_available = {'rxx':'Rxx', 'ryy':'Ryy', 'rzz':'Rzz','cp':'●P'} # CPhase
functional_gates_available = {'barrier':'░', 'measure':'M', 'reset':'|0>','delay':'Delay'}

def convert_gate_info_to_dag_info(nqubits:int,qubits:list,gates:list,show_qubits:bool=True) -> tuple[list,list]:
    #print('check',nqubits,qubits)
    qubit_dic = [None for _ in range(nqubits)]
    node_list = []
    edge_list = []

    if show_qubits:
        for qubit in qubits:
            node_info = ('q'+str(qubit),{'qubits':[qubit]})
            node_list.append(node_info)
            qubit_dic[qubit] = node_info[0]

    # seperate_measure_instruction
    gates0 = []
    for gate_info in gates:
       gate = gate_info[0]
       if gate == 'measure':
           for idx,qubit in enumerate(gate_info[1]):
               gates0.append((gate,[qubit],[gate_info[2][idx]]))
       else:
           gates0.append(gate_info)   

    for idx,gate_info in enumerate(gates0):
        # node 
        gate = gate_info[0]
        if gate in one_qubit_gates_available.keys():
            qubits = [gate_info[1]]
            node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits})
        elif gate in two_qubit_gates_available.keys():
            qubits = list(gate_info[1:])
            node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits})
        elif gate in three_qubit_gates_available.keys():
            qubits = list(gate_info[1:])
            node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits})
        elif gate in one_qubit_parameter_gates_available.keys():
            if gate == 'u': # three params
                qubits = [gate_info[-1]]
                params = list(gate_info[1:4])
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits, 'params':params})
            elif gate == 'r':
                qubits = [gate_info[-1]]
                params = list(gate_info[1:3])
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits, 'params':params})                
            else: # one params
                qubits = [gate_info[-1]]
                params = [gate_info[1]]
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits, 'params':params})      
        elif gate in two_qubit_parameter_gates_available.keys():
            qubits = list(gate_info[2:])
            params = [gate_info[1]]
            node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits, 'params':params})  
        elif gate in functional_gates_available.keys():
            if gate == 'measure':
                qubits = [gate_info[1][0]]
                cbits = [gate_info[2][0]]
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits, 'cbits':cbits})
            elif gate == 'barrier':
                qubits = [*gate_info[1]]
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits})
            elif gate == 'delay':
                qubits = [*gate_info[2]]
                duration = gate_info[1]
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits,'duration':duration})
            elif gate == 'reset':
                qubits = [gate_info[1]]
                node_info = (gate+'_'+str(idx)+'_'+str(qubits),{'qubits':qubits})    
        node_list.append(node_info)
        
        # edge
        if gate in two_qubit_gates_available.keys() or gate in two_qubit_parameter_gates_available.keys():
            if qubit_dic[qubits[0]] == qubit_dic[qubits[1]]:
                if qubit_dic[qubits[0]] is not None:
                    edge_info = (qubit_dic[qubits[0]],node_info[0],{"qubit":list(sorted(qubits))})
                    edge_list.append(edge_info)
            else:
                for qubit in qubits:
                    if qubit_dic[qubit] is not None:
                        edge_info = (qubit_dic[qubit],node_info[0],{"qubit" : [qubit]})
                        edge_list.append(edge_info)  
        elif gate in three_qubit_gates_available.keys():
            # 三个比特对应的node相同，两个比特对应的node相同（包括三种情况），三个比特对应的node不同
            pre_node_list = [qubit_dic[qubits[0]],qubit_dic[qubits[1]],qubit_dic[qubits[2]]]
            if len(set(pre_node_list)) == 3:
                for qubit in qubits:
                    if qubit_dic[qubit] is not None:
                        edge_info = (qubit_dic[qubit],node_info[0],{"qubit" : [qubit]})
                        edge_list.append(edge_info)
            elif len(set(pre_node_list)) == 2:
                if qubit_dic[qubits[0]] == qubit_dic[qubits[1]]:
                    if qubit_dic[qubits[0]] is not None:
                        edge_info = (qubit_dic[qubits[0]],node_info[0],{"qubit":list(sorted([qubits[0],qubits[1]]))})
                        edge_list.append(edge_info)
                    if qubit_dic[qubits[2]] is not None:
                        edge_info = (qubit_dic[qubits[2]],node_info[0],{"qubit":[qubits[2]]})
                        edge_list.append(edge_info)
                elif qubit_dic[qubits[0]] == qubit_dic[qubits[2]]:
                    if qubit_dic[qubits[0]] is not None:
                        edge_info = (qubit_dic[qubits[0]],node_info[0],{"qubit":list(sorted([qubits[0],qubits[2]]))})
                        edge_list.append(edge_info)
                    if qubit_dic[qubits[1]] is not None:
                        edge_info = (qubit_dic[qubits[1]],node_info[0],{"qubit":[qubits[1]]})
                        edge_list.append(edge_info)
                elif qubit_dic[qubits[1]] == qubit_dic[qubits[2]]:
                    if qubit_dic[qubits[1]] is not None:
                        edge_info = (qubit_dic[qubits[1]],node_info[0],{"qubit":list(sorted([qubits[1],qubits[2]]))})
                        edge_list.append(edge_info)
                    if qubit_dic[qubits[0]] is not None:
                        edge_info = (qubit_dic[qubits[0]],node_info[0],{"qubit":[qubits[0]]})
                        edge_list.append(edge_info)   
            elif len(set(pre_node_list)) == 1:
                if qubit_dic[qubits[0]] is not None:
                    edge_info = (qubit_dic[qubits[0]],node_info[0],{"qubit":list(sorted(qubits))})
                    edge_list.append(edge_info) 
        elif gate in ['barrier','delay']:
            temp = [[],[]]
            for qubit in qubits:
                if qubit_dic[qubit] is not None:
                    edge_info_0 = (qubit_dic[qubit],node_info[0])
                    edge_info_1 = [qubit]
                    if edge_info_0 in temp[0]:
                        idx = temp[0].index(edge_info_0)
                        temp[1][idx] += edge_info_1
                    else:
                        temp[0].append(edge_info_0)
                        temp[1].append(edge_info_1)
            for idx, edge in enumerate(temp[0]):
                edge_info = (edge[0],edge[1],{"qubit":list(sorted(temp[1][idx]))})
                edge_list.append(edge_info)
        else:
           #print(gate_info,qubits[0])
            assert(len(qubits) == 1)
            if qubit_dic[qubits[0]] is not None:
                edge_info = (qubit_dic[qubits[0]],node_info[0],{"qubit" : [qubits[0]]})
                edge_list.append(edge_info)
                
        for qubit in qubits:
            qubit_dic[qubit] = node_info[0]
    return np.array(node_list), np.array(edge_list)

def is_multiple_of_pi(n, tolerance: float = 1e-9) -> str:
    r"""
    Determines if a given number is approximately a multiple of π (pi) within a given tolerance.

    Args:
        n (float): The number to be checked.
        tolerance (float, optional): The allowable difference between the number and a multiple of π. Defaults to 1e-9.

    Returns:
        str: A string representation of the result. If the number is close to a multiple of π, 
             it returns a string in the form of "kπ" where k is a rounded multiplier (e.g., "2π" for 2 x π).
             If n is approximately 0, it returns "0.0".
             Otherwise, it returns a string representation of the number rounded to three decimal places.
    """
    result = n / np.pi
    aprox = round(result,2)
    if abs(result - aprox) < tolerance:
        if np.allclose(aprox, 0.0):
            return str(0.0)
        else:
            expression = f'{aprox}π'
            return expression
    else:
        return str(round(n,3))
    
#def parse_expression(expr):
#    return float(eval(expr, {"pi": np.pi, "np": np,'π':np.pi}))

def parse_expression(expr: str) -> float:
    expr = expr.strip().replace('π','pi') 
    pattern = r'^([+-]?\d*\.?\d*)?\s*pi(?:\s*/\s*([+-]?\d*\.?\d+))?$'
    m = re.match(pattern, expr)
    if m:
        coef_str, denom_str = m.groups()
        coef = float(coef_str) if coef_str not in (None, '', '+', '-') else (1.0 if coef_str in (None, '', '+') else -1.0)
        denom = float(denom_str) if denom_str else 1.0
        return coef * np.pi / denom
    else:
        return float(eval(expr,{'np':np,'pi':np.pi,'π':np.pi}))

def parse_openqasm2_regs(openqasm2_str:str):
    qreg_pattern = r"qreg\s+(\w+)\[(\d+)\];"
    creg_pattern = r"creg\s+(\w+)\[(\d+)\];"
    qregs = []
    cregs = []
    for name, size in re.findall(qreg_pattern, openqasm2_str):
        qregs.append((name,int(size)))
    for name, size in re.findall(creg_pattern, openqasm2_str):
        cregs.append((name,int(size)))
    # 删除所有声明 qreg/creg 的行
    new_lines = []
    for line in openqasm2_str.splitlines():
        if re.search(qreg_pattern, line) or re.search(creg_pattern, line):
            continue  
        new_lines.append(line)

    new_qasm = "\n".join(new_lines)
    return qregs,cregs,new_qasm

def parse_openqasm2_custom_gates(openqasm2_str:str):
    def parse_instruction(line: str):
        line = line.strip().rstrip(';')
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
        #elif name in ['u1','u2']:
        #    pass
        elif name in two_qubit_parameter_gates_available.keys():
            pass
        elif name in functional_gates_available.keys():
            pass
        else:
            raise(ValueError(f'parse error {name} !'))

        params = [p.strip() for p in m.group(2).split(",")] if m.group(2) else []
        qargs = [q.strip() for q in re.split(r"[, ]+", m.group(3).strip()) if q.strip()]
        return (name,*params,*qargs)
    
    pattern = r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s+([a-zA-Z0-9_,\s]+)\s*\{([^}]*)\}"
    gate_pattern = re.compile(pattern, re.DOTALL)
    gates = {}
    for match in gate_pattern.finditer(openqasm2_str):
        name = match.group(1).strip()
        params_str = [p.strip() for p in match.group(2).split(",")] if match.group(2) else []
        params = [str(parse_expression(i)) if 'pi' in i or 'π' in i else i for i in params_str]
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

        gates[name] = {
            "params_and_qregs": params+qargs,
            "definition": body
        }
    openqasm2_str = re.sub(pattern, "", openqasm2_str, flags=re.DOTALL) # remove custom gate str 
    print('Custom gate detected')
    for k,v in gates.items():
        print(k, v)
    return gates, openqasm2_str

def generate_reg_map(regs,type=''):
    num = sum([v for _,v in regs])
    all_reg = [i for i in range(num)]
    
    reg_map = {}
    idx = 0
    for reg,num in regs:
        reg_map[reg] = dict(zip(range(num),all_reg[idx:idx+num]))
        idx += num
    if len(reg_map)>1:
        for k,v in reg_map.items():
            print(f'{type} reg name {k}, mapping:{v}')
    return reg_map

def sparse_gate_params_qregs(line):
    pattern_measure = re.compile(r'''
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
    ''', re.VERBOSE)
    m = pattern_measure.match(line)
    if m:
        return 'measure', None, f"{m.group('qreg')},{m.group('creg')}"

    pattern = re.compile(r'''
        ^\s*
        (?P<gate>\w+)                 
        (?:\((?P<params>[^\)]*)\))?   
        \s+
        (?P<qregs>[\w\[\],\s]+)      
        \s*;
        $
    ''', re.VERBOSE)
    m = pattern.match(line)
    if m is None:
        return None, None, None

    gate = m.group("gate")
    params_str = m.group("params")
    qregs_str = m.group("qregs")   

    return gate, params_str, qregs_str

def get_positions_list(gate,qregs_str,qreg_map,creg_map):
    try:
        qregs = qregs_str.split(',')
    except:
        qregs = []
    cc = []
    if qregs != []:
        for qq in qregs:
            qs = re.findall(r'(\w+)\[(\d+)\]', qq)
            if qs != []:
                cc+=qs
            else:
                cc.append(qq)
    if gate == 'measure':
        if len(cc) != 2:
            raise ValueError('measurement gate: position parse error')
        q_position = []
        if isinstance(cc[0],tuple):
            q_position.append(qreg_map[cc[0][0]][int(cc[0][1])])
        if isinstance(cc[0],str):
            if cc[0] in qreg_map.keys():
                q_position += list(qreg_map[cc[0]].values())          
        c_position = []
        if isinstance(cc[1],tuple):
            c_position.append(creg_map[cc[1][0]][int(cc[1][1])])
        if isinstance(cc[1],str):
            if cc[1] in creg_map.keys():
                c_position += list(creg_map[cc[1]].values())
        positions = [q_position, c_position]
    else:
        positions = []
        for c in cc:
            if isinstance(c,tuple):
                positions.append([qreg_map[c[0]][int(c[1])]])
            if isinstance(c,str):
                if c in qreg_map.keys():
                    positions.append(list(qreg_map[c].values()))
    return positions

def parse_openqasm2_to_gates(openqasm2_str) -> None:
    r"""
    Parse gate information from an input OpenQASM 2.0 string, and update gates, supporting multiple registers.
    """
    qregs_used,cregs_used,openqasm2_str = parse_openqasm2_regs(openqasm2_str)
    if len(qregs_used) >1 or len(cregs_used)>1:
        print('Multiple registers detected. For subsequent compilation, the program will merge them. The mapping is as follows:')
    qreg_map = generate_reg_map(qregs_used,'Qubit')
    creg_map = generate_reg_map(cregs_used,'Cbit')

    custom_gates,openqasm2_str = parse_openqasm2_custom_gates(openqasm2_str)

    new = []
    qubit_used = []
    cbit_used = []
    clean_qasm = openqasm2_str.strip() 
    for line in clean_qasm.splitlines():
        if line == '':
            continue
        elif set(line) == {'\t'}: # check tab
            continue
        line_clear = line.split('//')[0].strip()
        gate,params_str,qregs_str = sparse_gate_params_qregs(line_clear)
        #print(gate,params_str,qregs_str)
        if params_str is not None:
            params = [parse_expression(p) for p in params_str.split(',')]
        else:
            params = []
        positions = get_positions_list(gate,qregs_str,qreg_map,creg_map)
        if gate in one_qubit_gates_available.keys():
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate,q))
                qubit_used.append(q)
        elif gate in two_qubit_gates_available.keys():
            if len(positions) != 2:
                raise ValueError(f'{gate} takes 2 quantum arguments, but got {len(positions)}.')
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f'{gate} takes 2 different quantum arguments length.')
            for idx in range(len(positions[0])):
                new.append((gate,positions[0][idx],positions[1][idx]))
                qubit_used.append(positions[0][idx])
                qubit_used.append(positions[1][idx])
        elif gate in three_qubit_gates_available.keys():
            if len(positions) != 3:
                raise ValueError(f'{gate} takes 3 quantum arguments, but got {len(positions)}.')
            if len(positions[0]) != len(positions[1]) or len(positions[0]) != len(positions[2]):
                raise ValueError(f'{gate} takes 3 different quantum arguments length.')
            for idx in range(len(positions[0])):
                new.append((gate,positions[0][idx],positions[1][idx],positions[2][idx]))
                qubit_used.append(positions[0][idx])
                qubit_used.append(positions[1][idx])
                qubit_used.append(positions[2][idx])        
        elif gate in one_qubit_parameter_gates_available.keys():
            qubits = [p for pp in positions for p in pp]
            if gate == 'u' or gate == 'u3':
                for q in qubits:
                    new.append(('u', params[0], params[1], params[2], q))
                    qubit_used.append(q)
            elif gate == 'r':
                for q in qubits:
                    new.append((gate, params[0], params[1], q))
                    qubit_used.append(q)
            else:
                for q in qubits:
                    new.append((gate, *params, q))
                    qubit_used.append(q)
        elif gate in ['u1','u2']:
            qubits = [p for pp in positions for p in pp]
            if gate == 'u1':
                for q in qubits:
                    new.append(('u', 0, 0, params[0], q))
                    qubit_used.append(q)
            elif gate == 'u2':
                for q in qubits:
                    new.append(('u', np.pi/2, params[0], params[1], q))
                    qubit_used.append(q)
        elif gate in two_qubit_parameter_gates_available.keys():
            if len(positions) != 2:
                raise ValueError(f'{gate} takes 2 quantum arguments, but got {len(positions)}.')
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f'{gate} takes 2 different quantum arguments length.') 
            for idx in range(len(positions[0])):
                new.append((gate, *params, positions[0][idx], positions[1][idx]))
                qubit_used.append(positions[0][idx])
                qubit_used.append(positions[1][idx])
        elif gate in ['cu1']:
            if len(positions) != 2:
                raise ValueError(f'{gate} takes 2 quantum arguments, but got {len(positions)}.')
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f'{gate} takes 2 different quantum arguments length.') 
            for idx in range(len(positions[0])):
                new.append(('cp', *params, positions[0][idx], positions[1][idx]))
                qubit_used.append(positions[0][idx])
                qubit_used.append(positions[1][idx])
        elif gate in ['delay']:
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate,*params,(q,)))
                qubit_used.append(q)
        elif gate in ['reset']:
            qubits = [p for pp in positions for p in pp]
            for q in qubits:
                new.append((gate,q))
                qubit_used.append(q)
        elif gate in ['barrier']:
            qubits = [p for pp in positions for p in pp]
            new.append((gate, tuple(qubits)))
            #qubit_used += list(qubits)
        elif gate in ['measure']:
            if len(positions[0]) != len(positions[1]):
                raise ValueError(f'{gate} takes 2 different quantum arguments length.') 
            for idx in range(len(positions[0])):
                new.append((gate, [positions[0][idx]], [positions[1][idx]])) 
                qubit_used.append(positions[0][idx])
                cbit_used.append(positions[1][idx])
        elif gate in ['OPENQASM','include','opaque','gate','qreg','creg','//']:
            continue
        elif gate in custom_gates.keys():
            positions_lengths = [len(position) for position in positions]
            if len(set(positions_lengths)) > 1:
                raise ValueError(f'custom gate {gate} sparse failer!' )
            for idx in range(len(positions[0])):
                qubits = [position[idx] for position in positions]
                params_qreg_dic = dict(zip(custom_gates[gate]['params_and_qregs'],params+qubits))
                new0 = []
                for gate0_info in custom_gates[gate]['definition']:
                    cc = []
                    for key in gate0_info[1:]:
                        if key in params_qreg_dic.keys():
                            cc.append(params_qreg_dic[key])
                        else:
                            if key.isdigit():
                                cc.append(int(key))
                            else:
                                cc.append(parse_expression(key))
                    new0.append(tuple([gate0_info[0]]+cc))
                new += new0
        elif gate is None:
            pass
        else:
            raise(ValueError(f"Sorry, an unrecognized OpenQASM 2.0 syntax {gate} was detected by quarkcircuit. Please contact the developer for assistance."))
    
    if cbit_used == []:
        cbit_used = [i for i in range(len(set(qubit_used)))]
    return new,set(qubit_used),set(cbit_used)


def parse_openqasm2_to_gates_dump(openqasm2_str) -> None:
    r"""
    Parse gate information from an input OpenQASM 2.0 string, and update gates
    """
    qregs,cregs,_ = parse_openqasm2_regs(openqasm2_str)
    if len(qregs) >1 or len(cregs)>1:
        raise(ValueError(f"Sorry, currently only one quantum or classical register definition is supported"))
    
    custom_gates,openqasm2_str = parse_openqasm2_custom_gates(openqasm2_str)
    new = []
    qubit_used = []
    cbit_used = []
    clean_qasm = openqasm2_str.strip() 
    for line in clean_qasm.splitlines():
        if line == '':
            continue
        elif set(line) == {'\t'}: # check tab
            continue
        gate = line.split()[0].split('(')[0]
        position = [int(num) for num in re.findall(r"\[(\d+)\]", line)] 
        if gate in one_qubit_gates_available.keys():
            new.append((gate,position[0]))
            qubit_used.append(position[0])
        elif gate in two_qubit_gates_available.keys():
            new.append((gate,position[0],position[1]))
            qubit_used.append(position[0])
            qubit_used.append(position[1])
        elif gate in three_qubit_gates_available.keys():
            new.append((gate,position[0],position[1],position[2]))
            qubit_used.append(position[0])
            qubit_used.append(position[1])
            qubit_used.append(position[2])            
        elif gate in one_qubit_parameter_gates_available.keys():
            if gate == 'u' or gate == 'u3':
                params_str = re.search(r'\(([^)]+)\)', line).group(1).split(',')
                params = [parse_expression(i) for i in params_str]
                new.append(('u', params[0], params[1], params[2], position[-1]))
                qubit_used.append(position[-1])
            elif gate == 'r':
                params_str = re.search(r'\(([^)]+)\)', line).group(1).split(',')
                params = [parse_expression(i) for i in params_str]
                new.append((gate, params[0], params[1], position[-1]))
                qubit_used.append(position[-1])
            else:
                param_str = re.search(r'\(([^)]+)\)', line).group(1)
                param = parse_expression(param_str)
                new.append((gate, param, position[-1]))
                qubit_used.append(position[-1])   
        elif gate in ['u1','u2']:
            if gate == 'u1':
                params_str = re.search(r'\(([^)]+)\)', line).group(1).split(',')
                params = [parse_expression(i) for i in params_str]
                new.append(('u', 0, 0, params[0], position[-1]))
                qubit_used.append(position[-1])
            elif gate == 'u2':
                params_str = re.search(r'\(([^)]+)\)', line).group(1).split(',')
                params = [parse_expression(i) for i in params_str]
                new.append(('u', np.pi/2, params[0], params[1], position[-1]))
                qubit_used.append(position[-1])
        elif gate in two_qubit_parameter_gates_available.keys():
            param_str = re.search(r'\(([^)]+)\)', line).group(1)
            param = parse_expression(param_str)
            new.append((gate, param, position[-2], position[-1]))
            qubit_used.append(position[-2])
            qubit_used.append(position[-1])
        elif gate in ['cu1']:
            param_str = re.search(r'\(([^)]+)\)', line).group(1)
            param = parse_expression(param_str)
            new.append(('cp', param, position[-2], position[-1]))
            qubit_used.append(position[-2])
            qubit_used.append(position[-1])
        elif gate in ['delay']:
            param = float(re.search(r'\(([^)]+)\)', line).group(1))
            new.append((gate,param,(position[-1],)))
            qubit_used.append(position[-1])
        elif gate in ['reset']:
            new.append((gate,position[0]))
            qubit_used.append(position[0])
        elif gate in ['barrier']:
            if position == []:
                line0 = line.strip().rstrip(';')
                pattern = r"barrier\s+(.+?)"
                match = re.match(pattern, line0)
                q_name = match.groups()[0]
                if q_name == qregs[0][0]:
                    new.append((gate, tuple([i for i in range(qregs[0][1])]))) 
                else:
                    raise(ValueError(f"Sorry, an unrecognized OpenQASM 2.0 syntax {line} was detected by quarkcircuit."))
            else:
                new.append((gate, tuple(position)))
            #qubit_used += list(position)
        elif gate in ['measure']:
            if position == []:
                line0 = line.strip().rstrip(';')
                pattern = r"measure\s+(.+?)\s*->\s*(.+)"
                match = re.match(pattern, line0)
                left, right = match.groups()
                q_name = left.strip()
                c_name = right.strip()
                if q_name == qregs[0][0] and c_name == cregs[0][0] and qregs[0][1]==cregs[0][1]:
                    for q in range(qregs[0][1]):
                        new.append(('measure', [q], [q])) 
                else:
                    raise(ValueError(f"check qreg name {qregs[0][0]} and parse q_name {q_name} is consistent, \
                    \ncheck creg name {cregs[0][0]} and parse c_name {c_name} is consistent, \
                    \ncheck qregs size {qregs[0][1]} and cregs size {cregs[0][1]} is equal."))
            else:
                new.append((gate, [position[0]], [position[1]])) 
                qubit_used.append(position[0])
                cbit_used.append(position[1])
        elif gate in ['OPENQASM','include','opaque','gate','qreg','creg','//']:
            continue
        elif gate in custom_gates.keys():
            try:
                params_str = re.search(r'\(([^)]+)\)', line).group(1).split(',')
                params = [parse_expression(i) for i in params_str]
            except:
                params = []
            params_qreg_dic = dict(zip(custom_gates[gate]['params_and_qregs'],params+position))
            new0 = []
            for gate0_info in custom_gates[gate]['definition']:
                cc = []
                for key in gate0_info[1:]:
                    if key in params_qreg_dic.keys():
                        cc.append(params_qreg_dic[key])
                    else:
                        if key.isdigit():
                            cc.append(int(key))
                        else:
                            cc.append(parse_expression(key))
                new0.append(tuple([gate0_info[0]]+cc))
            new += new0
        else:
            raise(ValueError(f"Sorry, an unrecognized OpenQASM 2.0 syntax {gate} was detected by quarkcircuit. Please contact the developer for assistance."))
    
    if cbit_used == []:
        cbit_used = [i for i in range(len(set(qubit_used)))]
    return new,set(qubit_used),set(cbit_used)


def parse_qlisp_to_gates(qlisp: list) -> tuple[list, list, list]:
    r"""
    Parse gate information from an input qlisp list.

     Args:
        qlisp (list): qlisp

     Returns:
        tuple[list, list, list]: A tuple containing:
            An gate information list.
            An qubit information list.
            An cbit information list.
    """
    new = []
    qubit_used = []
    cbit_used = []
    for gate_info in qlisp:
        gate = gate_info[0]
        if gate in ['X', 'Y', 'Z', 'S', 'T', 'H']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append((gate.lower(), qubit0))
            qubit_used.append(qubit0)
        elif gate in ['Z/2']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('s', qubit0))
            qubit_used.append(qubit0)
        elif gate in ['I']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('id', qubit0))
            qubit_used.append(qubit0)
        elif gate in ['-S','-T']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append((gate[1].lower() + 'dg', qubit0))
            qubit_used.append(qubit0)
        elif gate in ['-Z/2']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('sdg', qubit0))
            qubit_used.append(qubit0)
        elif gate in ['X/2']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('sx', qubit0))
            qubit_used.append(qubit0)
        elif gate in ['-X/2']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('sxdg', qubit0))
            qubit_used.append(qubit0)
        elif gate[0] in ['u3','U']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('u', gate[1], gate[2], gate[3], qubit0))
            qubit_used.append(qubit0)
        elif gate[0] in ['u1']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('p', gate[1], qubit0))  
            qubit_used.append(qubit0)      
        elif gate[0] in ['u2']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('u',np.pi/2, gate[1], gate[2],qubit0))  
            qubit_used.append(qubit0)
        elif gate[0] in ['R']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append(('r',np.pi/2, gate[1], qubit0))
            qubit_used.append(qubit0)
        #elif gate[0] in ['rfUnitary']:
        #    qubit0 = int(gate_info[1].split('Q')[1])
        #    new.append(('r', gate[1], gate[2], qubit0))
        elif gate[0] in ['CU']:
            qubit1 = int(gate_info[1][0].split('Q')[1])
            qubit2 = int(gate_info[1][1].split('Q')[1])
            new.append((gate[0].lower(),gate[1],qubit1,qubit2))
            qubit_used.append(qubit1)
            qubit_used.append(qubit2)
        elif gate in ['Cnot']:
            qubit1 = int(gate_info[1][0].split('Q')[1])
            qubit2 = int(gate_info[1][1].split('Q')[1])
            new.append(('cx', qubit1, qubit2))
            qubit_used.append(qubit1)
            qubit_used.append(qubit2)
        elif gate in ['CX','CY', 'CZ', 'SWAP']:
            qubit1 = int(gate_info[1][0].split('Q')[1])
            qubit2 = int(gate_info[1][1].split('Q')[1])
            new.append((gate.lower(), qubit1, qubit2))
            qubit_used.append(qubit1)
            qubit_used.append(qubit2)
        elif gate in ['CCZ','CCX','CSWAP']:
            qubit1 = int(gate_info[1][0].split('Q')[1])
            qubit2 = int(gate_info[1][1].split('Q')[1])
            qubit3 = int(gate_info[1][2].split('Q')[1])
            new.append((gate.lower(), qubit1, qubit2, qubit3))
            qubit_used.append(qubit1)
            qubit_used.append(qubit2) 
            qubit_used.append(qubit3)            
        elif gate in ['iSWAP']:
            qubit1 = int(gate_info[1][0].split('Q')[1])
            qubit2 = int(gate_info[1][1].split('Q')[1])
            new.append(('iswap', qubit1, qubit2))
            qubit_used.append(qubit1)
            qubit_used.append(qubit2)
        elif gate[0] in ['Rx', 'Ry', 'Rz', 'P']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append((gate[0].lower(), gate[1], qubit0))
            qubit_used.append(qubit0)
        elif gate in ['Reset']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append((gate.lower(), qubit0))
            qubit_used.append(qubit0)
        elif gate in ['Barrier']:
            qubitn = [int(istr.split('Q')[1]) for istr in gate_info[1]]
            new.append((gate.lower(), tuple(qubitn)))
            #qubit_used += qubitn
        elif gate[0] in ['Delay']:
            qubit0 = int(gate_info[1].split('Q')[1])
            new.append((gate[0].lower(), gate[1],(qubit0,)))
            qubit_used += [qubit0]
        elif gate[0] in ['Measure']:
            qubit0 = int(gate_info[1].split('Q')[1])
            cbit0 = gate[1]
            new.append((gate[0].lower(), [qubit0] ,[cbit0]))
            qubit_used += [qubit0]
            cbit_used += [cbit0]
        else:
            raise(ValueError(f'Sorry, an unrecognized qlisp syntax was detected by quarkcircuit. Please contact the developer for assistance. {gate[0]}'))
    
    return new, set(qubit_used), set(cbit_used)

def initialize_lines(nqubits:int,ncbits:int,gates:list) -> tuple[list, list]:
    r"""
    Initialize a blank circuit.

    Returns:
        tuple[list,list]: A tuple containing:
            - A list of fake gates element.
            - A list of fake gates element list.
    """
    nlines = 2 * nqubits + 1 + len(str(ncbits))
    gates_element = list('─ ' * nqubits) + ['═'] + [' '] * len(str(ncbits))
    gates_initial = copy.deepcopy(gates_element)
    qubits_expression = 'q'
    for i in range(nlines):
        if i in range(0, 2 * nqubits, 2):
            qi = i // 2
            if len(str(qi)) == 1:
                qn = qubits_expression + f'[{qi:<1}]  '
            elif len(str(qi)) == 2:
                qn = qubits_expression + f'[{qi:<2}] '
            elif len(str(qi)) == 3:
                qn = qubits_expression + f'[{qi:<3}]'
            gates_initial[i] = qn
        elif i in [2 * nqubits]:
            if len(str(ncbits)) == 1:
                c = f'c:  {ncbits}/'
            elif len(str(ncbits)) == 2:
                c = f'c: {ncbits}/'
            elif len(str(ncbits)) == 3:
                c = f'c:{ncbits}/'
            gates_initial[i] = c
        else:
            gates_initial[i] = ' ' * 6   
    n = len(gates) + nqubits
    gates_layerd = [gates_initial] + [copy.deepcopy(gates_element) for _ in range(n)]
    return gates_element,gates_layerd

def generate_gates_layerd(nqubits:int,ncbits:int,gates:list,params_value:dict) -> list:
    r"""Assign gates to their respective layers loosely.

    Returns:
        list: A list of gates element list.
    """
    lines_use = []
    # according plot layer distributed gates
    gates_element,gates_layerd = initialize_lines(nqubits,ncbits,gates)
    for gate_info in gates:
        gate = gate_info[0]
        if gate in one_qubit_gates_available.keys():
            pos0 = gate_info[1]
            for idx in range(len(gates_layerd)-1,-1,-1):
                if gates_layerd[idx][2*pos0] != '─':
                    gates_layerd[idx+1][2*pos0] = one_qubit_gates_available[gate]
                    lines_use.append(2 * pos0)
                    lines_use.append(2 * pos0 + 1)
                    break
        elif gate in two_qubit_gates_available.keys():
            pos0 = min(gate_info[1],gate_info[2])
            pos1 = max(gate_info[1],gate_info[2])
            for idx in range(len(gates_layerd)-1,-1,-1):
                if gates_layerd[idx][2*pos0:2*pos1+1] != list('─ ')*(pos1-pos0)+['─']:
                    gates_layerd[idx+1][2*gate_info[1]] = two_qubit_gates_available[gate][0]
                    gates_layerd[idx+1][2*gate_info[2]] = two_qubit_gates_available[gate][-1]
                    lines_use.append(2*pos0)
                    lines_use.append(2*pos0+1)
                    lines_use.append(2*pos1)
                    lines_use.append(2*pos1+1)
                    for i in range(2*pos0+1,2*pos1):
                        gates_layerd[idx+1][i] = '│'
                    break
        elif gate in three_qubit_gates_available.keys():
            sorted_qubits = sorted([gate_info[1],gate_info[2],gate_info[3]])
            pos0 = sorted_qubits[0]
            pos1 = sorted_qubits[1]
            pos2 = sorted_qubits[2]
            for idx in range(len(gates_layerd)-1,-1,-1):
                if gates_layerd[idx][2*pos0:2*pos2+1] != list('─ ')*(pos2-pos0)+['─']:
                    gates_layerd[idx+1][2*gate_info[1]] = three_qubit_gates_available[gate][0]
                    gates_layerd[idx+1][2*gate_info[2]] = three_qubit_gates_available[gate][1]
                    gates_layerd[idx+1][2*gate_info[3]] = three_qubit_gates_available[gate][2]
                    lines_use.append(2*pos0)
                    lines_use.append(2*pos0+1)
                    lines_use.append(2*pos1)
                    lines_use.append(2*pos1+1)
                    lines_use.append(2*pos2)
                    lines_use.append(2*pos2+1)
                    for i in range(2*pos0+1,2*pos1):
                        gates_layerd[idx+1][i] = '│'
                    for i in range(2*pos1+1,2*pos2):
                        gates_layerd[idx+1][i] = '│'
                    break
        elif gate in two_qubit_parameter_gates_available.keys():
            if gate in ['cp']:
                pos0 = min(gate_info[2],gate_info[3])
                pos1 = max(gate_info[2],gate_info[3])
                if isinstance(gate_info[1],(float,int)):
                    theta0_str = is_multiple_of_pi(gate_info[1])
                elif isinstance(gate_info[1],str):
                    param = params_value[gate_info[1]]
                    if isinstance(param,(float,int)):
                        theta0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        theta0_str = param
                gate_express = two_qubit_parameter_gates_available[gate][1]+f'({theta0_str})'
                if len(gate_express) % 2 == 0:
                    gate_express = two_qubit_parameter_gates_available[gate][1]+f'({theta0_str})─'
                for idx in range(len(gates_layerd)-1,-1,-1):
                    if gates_layerd[idx][2*pos0:2*pos1+1] != list('─ ')*(pos1-pos0)+['─']:
                        gates_layerd[idx+1][2*gate_info[2]] = (len(gate_express)//2)*'─' + two_qubit_parameter_gates_available[gate][0] + (len(gate_express)//2)*'─'
                        gates_layerd[idx+1][2*gate_info[3]] = gate_express
                        lines_use.append(2*pos0)
                        lines_use.append(2*pos0+1)
                        lines_use.append(2*pos1)
                        lines_use.append(2*pos1+1)
                        for i in range(2*pos0+1,2*pos1):
                            if i % 2 == 0:
                                gates_layerd[idx+1][i] = (len(gate_express)//2)*'─' + '│' + (len(gate_express)//2)*'─'
                            else:
                                gates_layerd[idx+1][i] = (len(gate_express)//2)*' ' + '│' + (len(gate_express)//2)*' '

                        break

            else:
                pos0 = min(gate_info[2],gate_info[3])
                pos1 = max(gate_info[2],gate_info[3])
                if isinstance(gate_info[1],(float,int)):
                    theta0_str = is_multiple_of_pi(gate_info[1])
                elif isinstance(gate_info[1],str):
                    param = params_value[gate_info[1]]
                    if isinstance(param,(float,int)):
                        theta0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        theta0_str = param
                gate_express = two_qubit_parameter_gates_available[gate]+f'({theta0_str})'
                if len(gate_express)%2 == 0:
                    gate_express += ' '
                for idx in range(len(gates_layerd)-1,-1,-1):
                    if gates_layerd[idx][2*pos0:2*pos1+1] != list('─ ')*(pos1-pos0)+['─']:
                        dif0 = (len(gate_express) - 1)//2
                        if pos0 == gate_info[2]: 
                            gates_layerd[idx+1][2*pos0] = '┌' + '─'*dif0 +'0'+'─'*dif0 + '┐'
                            gates_layerd[idx+1][2*pos1] = '└' + '─'*dif0 +'1'+'─'*dif0 + '┘'
                            lines_use.append(2*pos0)
                            lines_use.append(2*pos0 + 1)
                            lines_use.append(2*pos1)
                            lines_use.append(2*pos1 + 1)
                        elif pos0 == gate_info[3]:
                            gates_layerd[idx+1][2*pos0] = '┌' + '─'*dif0 +'1'+'─'*dif0 + '┐'
                            gates_layerd[idx+1][2*pos1] = '└' + '─'*dif0 +'0'+'─'*dif0 + '┘'
                            lines_use.append(2*pos0)
                            lines_use.append(2*pos0 + 1)
                            lines_use.append(2*pos1)
                            lines_use.append(2*pos1 + 1)
                        for i in range(2*pos0+1,2*pos1):
                            gates_layerd[idx+1][i] = '│' + ' '*len(gate_express) + '│'
                        gates_layerd[idx+1][2*pos0 + (pos1-pos0)] = '│' + gate_express + '│'
                        break
        elif gate in one_qubit_parameter_gates_available.keys():
            if gate == 'u':
                if isinstance(gate_info[1],(float,int)):
                    theta0_str = is_multiple_of_pi(gate_info[1])
                elif isinstance(gate_info[1],str):
                    param = params_value[gate_info[1]]
                    if isinstance(param,(float,int)):
                        theta0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        theta0_str = param
                if isinstance(gate_info[2],(float,int)):
                    phi0_str = is_multiple_of_pi(gate_info[2])
                elif isinstance(gate_info[2],str):
                    param = params_value[gate_info[2]]
                    if isinstance(param,(float,int)):
                        phi0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        phi0_str = param
                if isinstance(gate_info[3],(float,int)):
                    lamda0_str = is_multiple_of_pi(gate_info[3])
                elif isinstance(gate_info[3],str):
                    param = params_value[gate_info[3]]
                    if isinstance(param,(float,int)):
                        lamda0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        lamda0_str = param
                pos0 = gate_info[-1]
                for idx in range(len(gates_layerd)-1,-1,-1):
                    if gates_layerd[idx][2*pos0] != '─':
                        params_str = '(' + theta0_str + ',' + phi0_str + ',' + lamda0_str + ')'
                        gates_layerd[idx+1][2*pos0] = one_qubit_parameter_gates_available[gate] + params_str
                        lines_use.append(2*pos0)
                        lines_use.append(2*pos0 + 1)
                        break      
            elif gate == 'r':      
                if isinstance(gate_info[1],(float,int)):
                    theta0_str = is_multiple_of_pi(gate_info[1])
                elif isinstance(gate_info[1],str):
                    param = params_value[gate_info[1]]
                    if isinstance(param,(float,int)):
                        theta0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        theta0_str = param
                if isinstance(gate_info[2],(float,int)):
                    phi0_str = is_multiple_of_pi(gate_info[2])
                elif isinstance(gate_info[2],str):
                    param = params_value[gate_info[2]]
                    if isinstance(param,(float,int)):
                        phi0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        phi0_str = param       
                pos0 = gate_info[-1]
                for idx in range(len(gates_layerd)-1,-1,-1):
                    if gates_layerd[idx][2*pos0] != '─':
                        params_str = '(' + theta0_str + ',' + phi0_str + ')'
                        gates_layerd[idx+1][2*pos0] = one_qubit_parameter_gates_available[gate] + params_str
                        lines_use.append(2*pos0)
                        lines_use.append(2*pos0 + 1)
                        break    
            else:
                if isinstance(gate_info[1],(float,int)):
                    theta0_str = is_multiple_of_pi(gate_info[1])
                elif isinstance(gate_info[1],str):
                    param = params_value[gate_info[1]]
                    if isinstance(param,(float,int)):
                        theta0_str = is_multiple_of_pi(param)
                    elif isinstance(param,str):
                        theta0_str = param
                #theta0_str = is_multiple_of_pi(gate_info[1])
                pos0 = gate_info[2]
                for idx in range(len(gates_layerd)-1,-1,-1):
                    if gates_layerd[idx][2*pos0] != '─':
                        gates_layerd[idx+1][2*pos0] = one_qubit_parameter_gates_available[gate]+'('+theta0_str+')'
                        lines_use.append(2*pos0)
                        lines_use.append(2*pos0 + 1)
                        break  
        elif gate in ['reset']:
            pos0 = gate_info[1]
            for idx in range(len(gates_layerd)-1,-1,-1):
                if gates_layerd[idx][2*pos0] != '─':
                    gates_layerd[idx+1][2*pos0] = functional_gates_available[gate]
                    lines_use.append(2 * pos0)
                    lines_use.append(2 * pos0 + 1)
                    break
        elif gate in ['barrier']:
            poslst0 = gate_info[1]
            poslst = []
            for j in poslst0:
                if j + 1 in poslst0:
                    poslst.append(2*j)
                    poslst.append(2*j+1)
                else:
                    poslst.append(2*j)
            for idx in range(len(gates_layerd)-1,-1,-1):
                e_ = [gates_layerd[idx][2*i] for i in poslst0]
                if all(e == '─' for e in e_) is False:
                    for i in poslst:
                        gates_layerd[idx+1][i] = functional_gates_available[gate]
                    break
        elif gate in ['delay']:
            poslst0 = gate_info[-1]
            poslst = []
            for j in poslst0:
                if j + 1 in poslst0:
                    poslst.append(2*j)
                else:
                    poslst.append(2*j)
            for idx in range(len(gates_layerd)-1,-1,-1):
                e_ = [gates_layerd[idx][2*i] for i in poslst0]
                if all(e == '─' for e in e_) is False:
                    for i in poslst:
                        gates_layerd[idx+1][i] = functional_gates_available[gate]+f'({gate_info[1]:.1e}s)'
                    break
        elif gate in ['measure']:
            for j in range(len(gate_info[1])):
                pos0 = gate_info[1][j]
                pos1 = gate_info[2][j]
                for idx in range(len(gates_layerd)-1,-1,-1):
                    if gates_layerd[idx][2*pos0:] != gates_element[2*pos0:]:
                        gates_layerd[idx+1][2*pos0] = functional_gates_available[gate]
                        lines_use.append(2*pos0)
                        lines_use.append(2*pos0 + 1)
                        for i in range(2*pos0+1,2*nqubits,1):
                            gates_layerd[idx+1][i] = '│'
                        for i in range(2*nqubits+1, 2*nqubits+1+len(str(pos1))):
                            gates_layerd[idx+1][i] = str(pos1)[i-2*nqubits-1]
                        break
    for idx in range(len(gates_layerd)-1,-1,-1):
        if gates_layerd[idx] != gates_element:
            cut = idx + 1
            break
    return gates_layerd[:cut],lines_use
        
def format_gates_layerd(nqubits:int,ncbits:int,gates:list,params_value:dict) -> list:
    r"""Unify the width of each layer's gate strings

     Returns:
        list: A new list of gates element list.
    """
    gates_layerd,lines_use = generate_gates_layerd(nqubits,ncbits,gates,params_value)
    gates_layerd_format = [gates_layerd[0]]
    for lst in gates_layerd[1:]:
        max_length = max(len(item) for item in lst)
        if max_length == 1:
            gates_layerd_format.append(lst)
        else:
            if max_length % 2 == 0:
                max_length += 1
            dif0 = max_length // 2
            for idx in range(len(lst)):
                if len(lst[idx]) == 1:
                    if idx < 2 * nqubits:
                        if idx % 2 == 0:
                            lst[idx] = '─' * dif0 + lst[idx] + '─' * dif0
                        else:
                            lst[idx] = ' ' * dif0 + lst[idx] + ' ' * dif0
                    elif idx == 2 * nqubits:
                        lst[idx] = '═' * dif0 + lst[idx] + '═' * dif0
                    else:
                        lst[idx] = ' ' * dif0 + lst[idx] + ' ' * dif0
                else:
                    dif1 = max_length - len(lst[idx])
                    if idx%2 == 0:
                        lst[idx] = lst[idx] + '─' * dif1
                    else:
                        lst[idx] = lst[idx] + ' ' * dif1
            gates_layerd_format.append(lst)
    return gates_layerd_format,lines_use
    
def add_gates_to_lines(nqubits:int,ncbits:int,gates:list,params_value:dict, width: int = 4) -> list:
    r"""Add gates to lines.
    Args:
        width (int, optional): The width between gates. Defaults to 4.
    Returns:
        list: A list of lines.
    """
    gates_layerd_format,lines_use = format_gates_layerd(nqubits,ncbits,gates,params_value)
    nl = len(gates_layerd_format[0])
    lines1 = [str() for _ in range(nl)]
    for i in range(nl):
        for j in range(len(gates_layerd_format)):
            if i < 2 * nqubits:
                if i % 2 == 0:
                    lines1[i] += gates_layerd_format[j][i] + '─' * width
                else:
                    lines1[i] += gates_layerd_format[j][i] + ' ' * width
            elif i == 2 * nqubits:
                lines1[i] += gates_layerd_format[j][i] + '═' * width
            elif i > 2 * nqubits:
                lines1[i] += gates_layerd_format[j][i] + ' ' * width
    return lines1,lines_use