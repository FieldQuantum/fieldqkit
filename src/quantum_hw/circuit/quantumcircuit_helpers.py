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

r"""Helper utilities for circuit parsing, DAG conversion, and rendering."""
import numpy as np
import copy,re

one_qubit_gates_available = {
    'id':'I', 'x':'X', 'y':'Y', 'z':'Z',
    's':'S', 'sdg':'Sdg','t':'T', 'tdg':'Tdg',
    'h':'H', 'sx':'√X','sxdg':'√Xdg',
    }
two_qubit_gates_available = {
    'cx':'●X', 'cnot':'●X', 'cy':'●Y', 'cz':'●Z', 'swap':'XX', 'iswap':'✶✶', 'ecr':'╬╬'
    } 
three_qubit_gates_available = {'ccz':'●●●','ccx':'●●X','cswap':'●XX'} 
one_qubit_parameter_gates_available = {'rx':'Rx', 'ry':'Ry', 'rz':'Rz', 'p':'P', 'u':'U', 'u3':'U', 'r':'R'}
two_qubit_parameter_gates_available = {'rxx':'Rxx', 'ryy':'Ryy', 'rzz':'Rzz','cp':'●P'} # CPhase
functional_gates_available = {'barrier':'░', 'measure':'M', 'reset':'|0>','delay':'Delay'}


def _record_qubits(qubit_used: list, *qubits: int) -> None:
    qubit_used.extend(qubits)



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

    # separate_measure_instruction
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
            # Three-qubit cases: all same node; two-share-one; or all different.
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

def _safe_eval_expression(expr: str) -> float:
    """Safely evaluate a numeric expression that may contain pi."""
    import ast

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return float(np.pi)
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise ValueError("Unsupported expression.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression.") from exc
    return float(_eval(tree))


def parse_expression(expr: str) -> float:
    expr = expr.strip().replace('π', 'pi')
    expr = expr.replace('np.pi', 'pi')
    pattern = r'^([+-]?\d*\.?\d*)?\s*pi(?:\s*/\s*([+-]?\d*\.?\d+))?$'
    m = re.match(pattern, expr)
    if m:
        coef_str, denom_str = m.groups()
        coef = float(coef_str) if coef_str not in (None, '', '+', '-') else (1.0 if coef_str in (None, '', '+') else -1.0)
        denom = float(denom_str) if denom_str else 1.0
        return coef * np.pi / denom
    return _safe_eval_expression(expr)










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
