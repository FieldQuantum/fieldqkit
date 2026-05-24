"""
OpenQASM 3.0 Parser (using openqasm3 AST).
"""

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

__all__ = ["parse_openqasm3_to_gates"]


def _record_qubits(qubit_used: list, *qubits: int) -> None:
	"""Append qubit indices to the tracking list for register inference.

	Args:
		qubit_used (*list*): Accumulator list of qubit indices encountered so far.
		*qubits (*int*): Qubit indices to record.
	"""
	qubit_used.extend(qubits)


def parse_openqasm3_to_gates(openqasm3_str: str) -> tuple[list, set, set]:
	"""Parse gate information from an OpenQASM 3.0 program string.

	Args:
		openqasm3_str (*str*): An OpenQASM 3.0 program string.

	Returns:
		Tuple of ``(gates, qubit_used, cbit_used)``.

	Raises:
		ValueError: Unsupported OpenQASM3 expression.
		ImportError: openqasm3 is required for OpenQASM 3.0 parsing.
	"""
	try:
		import openqasm3
		from openqasm3 import printer as qasm3_printer
	except Exception as exc:
		raise ImportError("openqasm3 is required for OpenQASM 3.0 parsing.") from exc

	parse_fn = getattr(openqasm3, "parse", None)
	if parse_fn is None:
		try:
			from openqasm3 import parser as qasm3_parser
			parse_fn = qasm3_parser.parse
		except Exception as exc:
			raise ImportError("openqasm3 parser is unavailable.") from exc

	program = parse_fn(openqasm3_str)
	statements = getattr(program, "statements", None) or getattr(program, "body", [])

	qreg_map = {}
	creg_map = {}
	next_q = 0
	next_c = 0
	custom_gates = {}

	new = []
	qubit_used = []
	cbit_used = []

	def _name_of(node):
		# openqasm3 AST field names vary across versions; keep compatibility.
		"""Extract the name string from an AST node, handling version differences.

		Args:
			node: OpenQASM 3 AST node.

		Returns:
			``str`` name, or ``None`` if the node is ``None`` or has no recognisable name attribute.
		"""
		if node is None:
			return None
		name = getattr(node, "name", None)
		if isinstance(name, str):
			return name
		if hasattr(name, "name"):
			return name.name
		if hasattr(node, "identifier") and hasattr(node.identifier, "name"):
			return node.identifier.name
		if hasattr(node, "identifier") and isinstance(getattr(node.identifier, "id", None), str):
			return node.identifier.id
		if isinstance(getattr(node, "id", None), str):
			return node.id
		if isinstance(getattr(node, "symbol", None), str):
			return node.symbol
		if isinstance(getattr(node, "string", None), str):
			return node.string
		return getattr(node, "id", None)

	def _register_symbol(name, size_value: int, is_qubit: bool):
		"""Register a qubit or classical register symbol.

		Args:
			name: Register name.
			size_value (*int*): Number of bits/qubits in the register.
			is_qubit (*bool*): ``True`` for qubit register, ``False`` for classical.
		"""
		nonlocal next_q, next_c
		if name is None:
			return
		if is_qubit:
			qreg_map[name] = {i: next_q + i for i in range(size_value)}
			next_q += size_value
		else:
			creg_map[name] = {i: next_c + i for i in range(size_value)}
			next_c += size_value

	def _extract_decl_name_and_size(decl):
		"""Extract the name, type name, and size from a declaration AST node.

		Args:
			decl: Declaration AST node.

		Returns:
			``(name, dtype_name, size_value)`` tuple.
		"""
		name = (
			_name_of(decl)
			or _name_of(getattr(decl, "identifier", None))
			or _name_of(getattr(decl, "qubit", None))
			or _name_of(getattr(decl, "bit", None))
		)
		dtype = getattr(decl, "type", None)
		dtype_name = dtype.__class__.__name__ if dtype is not None else ""
		size = getattr(decl, "size", None) or getattr(dtype, "size", None) or getattr(dtype, "designator", None)
		size_value = int(_expr_to_float(size)) if size is not None else 1
		return name, dtype_name, size_value

	def _ensure_register(name, idx, is_qubit: bool):
		"""Lazily create a register if it does not already exist.

		Args:
			name: Register name.
			idx: Maximum index seen (used to infer register size).
			is_qubit (*bool*): ``True`` for qubit register, ``False`` for classical.
		"""
		nonlocal next_q, next_c
		if name is None:
			return
		if is_qubit and name in qreg_map:
			return
		if (not is_qubit) and name in creg_map:
			return
		size_value = int(idx) + 1 if idx is not None else 1
		if is_qubit:
			qreg_map[name] = {i: next_q + i for i in range(size_value)}
			next_q += size_value
		else:
			creg_map[name] = {i: next_c + i for i in range(size_value)}
			next_c += size_value

	def _expr_to_float(expr):
		"""Recursively evaluate an OpenQASM 3 expression AST node to a ``float``.

		Args:
			expr: Expression AST node.

		Returns:
			``float`` value, or ``None`` if *expr* is ``None``.

		Raises:
			ValueError: Unsupported OpenQASM3 expression.
		"""
		if expr is None:
			return None
		try:
			expr_str = qasm3_printer.dumps(expr).strip()
			return parse_expression(expr_str)
		except Exception:
			pass
		node_type = expr.__class__.__name__
		if node_type in {"IntegerLiteral", "FloatLiteral", "ImaginaryLiteral"}:
			return float(getattr(expr, "value"))
		if node_type == "Identifier":
			if getattr(expr, "name", "") == "pi":
				return float(np.pi)
		if node_type == "UnaryExpression":
			op = getattr(expr, "op", None)
			value = _expr_to_float(getattr(expr, "expression", None))
			if op in {"+", "Plus"}:
				return value
			if op in {"-", "Minus"}:
				return -value
		if node_type in {"BinaryExpression", "BinaryOperation"}:
			left = _expr_to_float(getattr(expr, "lhs", None) or getattr(expr, "left", None))
			right = _expr_to_float(getattr(expr, "rhs", None) or getattr(expr, "right", None))
			op = getattr(expr, "op", None)
			if op in {"+", "Add"}:
				return left + right
			if op in {"-", "Sub"}:
				return left - right
			if op in {"*", "Mul"}:
				return left * right
			if op in {"/", "Div"}:
				return left / right
			if op in {"**", "Pow"}:
				return left**right
		raise ValueError("Unsupported OpenQASM3 expression.")

	def _duration_to_seconds(duration):
		"""Convert a duration literal AST node to seconds.

		Args:
			duration: Duration AST node.

		Returns:
			``float`` duration in seconds, or ``None`` if *duration* is ``None``.
		"""
		if duration is None:
			return None
		node_type = duration.__class__.__name__
		if node_type == "DurationLiteral":
			value = float(getattr(duration, "value"))
			unit = getattr(duration, "unit", "s")
			unit = unit.name if hasattr(unit, "name") else unit
			scale = {
				"s": 1.0,
				"ms": 1e-3,
				"us": 1e-6,
				"ns": 1e-9,
				"ps": 1e-12,
			}.get(unit, 1.0)
			return value * scale
		return _expr_to_float(duration)

	def _extract_index(index_expr):
		"""Extract an integer index from an index expression AST node.

		Args:
			index_expr: Index expression node.

		Returns:
			``int`` index, or ``None`` if *index_expr* is ``None``.
		"""
		if index_expr is None:
			return None
		if isinstance(index_expr, (int, float)):
			return int(index_expr)
		if isinstance(index_expr, (list, tuple)) and len(index_expr) == 1:
			return _extract_index(index_expr[0])
		value = getattr(index_expr, "value", None)
		if isinstance(value, (int, float)):
			return int(value)
		if hasattr(index_expr, "index") and not callable(getattr(index_expr, "index")):
			return int(_expr_to_float(index_expr.index))
		return int(_expr_to_float(index_expr))

	def _qubit_index(qref):
		"""Resolve a qubit reference AST node to its dense qubit index.

		Args:
			qref: Qubit reference (``Identifier``, ``IndexedIdentifier``, or ``int``).

		Returns:
			``int`` qubit index.

		Raises:
			ValueError: f'Unsupported qubit reference: {name}'
		"""
		if isinstance(qref, (int, np.integer)):
			return int(qref)
		name = _name_of(qref)
		indices = getattr(qref, "indices", None)
		if indices:
			idx = _extract_index(indices[0])
			if name not in qreg_map:
				_ensure_register(name, idx, True)
			return qreg_map[name][int(idx)]
		if name in qreg_map and len(qreg_map[name]) == 1:
			return list(qreg_map[name].values())[0]
		if name not in qreg_map:
			_ensure_register(name, None, True)
			if name in qreg_map and len(qreg_map[name]) == 1:
				return list(qreg_map[name].values())[0]
		raise ValueError(f"Unsupported qubit reference: {name}")

	def _bit_index(bref):
		"""Resolve a classical bit reference AST node to its dense bit index.

		Args:
			bref: Bit reference (``Identifier``, ``IndexedIdentifier``, or ``int``).

		Returns:
			``int`` classical bit index.

		Raises:
			ValueError: f'Unsupported bit reference: {name}'
		"""
		if isinstance(bref, (int, np.integer)):
			return int(bref)
		name = _name_of(bref)
		indices = getattr(bref, "indices", None)
		if indices:
			idx = _extract_index(indices[0])
			if name not in creg_map:
				_ensure_register(name, idx, False)
			return creg_map[name][int(idx)]
		if name in creg_map and len(creg_map[name]) == 1:
			return list(creg_map[name].values())[0]
		if name not in creg_map:
			_ensure_register(name, None, False)
			if name in creg_map and len(creg_map[name]) == 1:
				return list(creg_map[name].values())[0]
		raise ValueError(f"Unsupported bit reference: {name}")

	def _handle_gate_call(gate_name, args, qargs):
		"""Dispatch a gate call to the appropriate handler and append to *new*.

		Args:
			gate_name: Name of the quantum gate.
			args: Gate parameter arguments.
			qargs: Target qubit arguments.

		Raises:
			ValueError: f'{gate} takes 2 qubits, got {len(qubits)}'
		"""
		gate = gate_name.lower()
		qubits = [_qubit_index(q) for q in _as_qargs(qargs)]
		# --- Compatibility aliases: map to canonical internal gate names ---
		if gate == 'cnot':
			gate = 'cx'
		elif gate == 'u3':
			gate = 'u'
		if gate in one_qubit_gates_available.keys():
			for q in qubits:
				new.append((gate, q))
				_record_qubits(qubit_used, q)
		elif gate in two_qubit_gates_available.keys():
			if len(qubits) != 2:
				raise ValueError(f"{gate} takes 2 qubits, got {len(qubits)}")
			new.append((gate, qubits[0], qubits[1]))
			_record_qubits(qubit_used, *qubits)
		elif gate in three_qubit_gates_available.keys():
			if len(qubits) != 3:
				raise ValueError(f"{gate} takes 3 qubits, got {len(qubits)}")
			new.append((gate, qubits[0], qubits[1], qubits[2]))
			_record_qubits(qubit_used, *qubits)
		elif gate in one_qubit_parameter_gates_available.keys():
			params = [_expr_to_float(arg) for arg in args]
			for q in qubits:
				if gate == "u":
					new.append(("u", params[0], params[1], params[2], q))
				else:
					new.append((gate, params[0], q))
				_record_qubits(qubit_used, q)
		elif gate in ['u1', 'p']:
			# u1(λ)/p(λ) → u(0, 0, λ)
			params = [_expr_to_float(arg) for arg in args]
			for q in qubits:
				new.append(("u", 0.0, 0.0, params[0], q))
				_record_qubits(qubit_used, q)
		elif gate == 'u2':
			# u2(φ,λ) → u(π/2, φ, λ)
			params = [_expr_to_float(arg) for arg in args]
			for q in qubits:
				new.append(("u", np.pi / 2, params[0], params[1], q))
				_record_qubits(qubit_used, q)
		elif gate == 'r':
			# r(θ, φ) → u(θ, φ - π/2, π/2 - φ)
			params = [_expr_to_float(arg) for arg in args]
			for q in qubits:
				phi = params[1]
				new.append(("u", params[0], phi - np.pi / 2, np.pi / 2 - phi, q))
				_record_qubits(qubit_used, q)
		elif gate in ['cu1', 'cp']:
			# cu1(λ)/cp(λ) = controlled-phase: decompose into cx + rz
			params = [_expr_to_float(arg) for arg in args]
			if len(qubits) != 2:
				raise ValueError(f"{gate} takes 2 qubits, got {len(qubits)}")
			q0, q1 = qubits[0], qubits[1]
			lam = params[0]
			new.append(("rz", lam / 2, q0))
			new.append(("cx", q0, q1))
			new.append(("rz", -lam / 2, q1))
			new.append(("cx", q0, q1))
			new.append(("rz", lam / 2, q1))
			_record_qubits(qubit_used, q0, q1)
		elif gate in ['cswap', 'ccnot']:
			# cswap → CX+CCX+CX decomposition (Fredkin); ccnot → ccx
			if len(qubits) != 3:
				raise ValueError(f"{gate} takes 3 qubits, got {len(qubits)}")
			c, t1, t2 = qubits[0], qubits[1], qubits[2]
			if gate == 'cswap':
				# CSWAP(c, t1, t2) = CX(t2,t1) · CCX(c,t1,t2) · CX(t2,t1)
				new.append(('cx', t2, t1))
				new.append(('ccx', c, t1, t2))
				new.append(('cx', t2, t1))
			else:
				new.append(('ccx', c, t1, t2))
			_record_qubits(qubit_used, c, t1, t2)
		elif gate in two_qubit_parameter_gates_available.keys():
			params = [_expr_to_float(arg) for arg in args]
			if len(qubits) != 2:
				raise ValueError(f"{gate} takes 2 qubits, got {len(qubits)}")
			new.append((gate, params[0], qubits[0], qubits[1]))
			_record_qubits(qubit_used, *qubits)
		elif gate in custom_gates:
			_expand_custom_gate(gate, args, qubits)
		else:
			raise ValueError(f"Unsupported OpenQASM3 gate {gate}")

	def _expand_custom_gate(gate_name, args, qubits):
		"""Expand a custom gate definition into its constituent gate calls.

		Args:
			gate_name: Name of the custom gate.
			args: Parameter arguments to substitute.
			qubits: Target qubit indices.
		"""
		definition = custom_gates[gate_name]
		param_names = definition["params"]
		qarg_names = definition["qargs"]
		param_map = dict(zip(param_names, [_expr_to_float(a) for a in args]))
		qarg_map = dict(zip(qarg_names, qubits))
		for stmt in definition["body"]:
			if stmt[0] == "gate":
				gate, stmt_args, stmt_qargs = stmt[1]
				remapped_args = [param_map.get(a, a) for a in stmt_args]
				remapped_qargs = [qarg_map[a] if isinstance(a, str) else a for a in _as_qargs(stmt_qargs)]
				_handle_gate_call(gate, remapped_args, remapped_qargs)

	def _as_qargs(qargs):
		"""Normalise qubit arguments to a flat list.

		Args:
			qargs: Single qubit reference or list of qubit references.

		Returns:
			List of qubit argument nodes.
		"""
		if qargs is None:
			return []
		if isinstance(qargs, (list, tuple)):
			return list(qargs)
		return [qargs]

	for stmt in statements:
		stype = stmt.__class__.__name__
		if stype in {"DeclarationStatement"}:
			decl = getattr(stmt, "declaration", None)
			if decl is None:
				continue
			if hasattr(decl, "qubits") and decl.qubits:
				for q in decl.qubits:
					_register_symbol(_name_of(q), 1, True)
				continue
			if hasattr(decl, "bits") and decl.bits:
				for b in decl.bits:
					_register_symbol(_name_of(b), 1, False)
				continue
			name, dtype_name, size_value = _extract_decl_name_and_size(decl)
			if dtype_name in {"QubitType"}:
				_register_symbol(name, size_value, True)
			elif dtype_name in {"BitType", "ClassicalType"}:
				_register_symbol(name, size_value, False)
			continue
		if stype in {"QubitDeclaration", "QubitDeclarationStatement"}:
			name, _, size_value = _extract_decl_name_and_size(stmt)
			_register_symbol(name, size_value, True)
		elif stype in {"ClassicalDeclaration", "BitDeclaration", "ClassicalBitDeclaration"}:
			name, _, size_value = _extract_decl_name_and_size(stmt)
			_register_symbol(name, size_value, False)
		elif stype in {"QuantumGateDefinition", "GateDefinition"}:
			gate_name = _name_of(stmt)
			params = [p.name for p in getattr(stmt, "parameters", [])]
			qargs = [q.name for q in getattr(stmt, "qubits", [])]
			body = []
			for b in getattr(stmt, "body", []):
				if b.__class__.__name__ in {"QuantumGate", "GateCall"}:
					body.append(("gate", (_name_of(b), getattr(b, "arguments", []), getattr(b, "qubits", []))))
			custom_gates[gate_name] = {"params": params, "qargs": qargs, "body": body}
		elif stype in {"QuantumGate", "GateCall"}:
			_handle_gate_call(_name_of(stmt), getattr(stmt, "arguments", []), getattr(stmt, "qubits", []))
		elif stype in {"QuantumBarrier", "Barrier"}:
			qubits = [_qubit_index(q) for q in _as_qargs(getattr(stmt, "qubits", []))]
			new.append(("barrier", tuple(qubits)))
		elif stype in {"QuantumReset", "Reset"}:
			qubits = [_qubit_index(q) for q in _as_qargs(getattr(stmt, "qubits", []))]
			for q in qubits:
				new.append(("reset", q))
				_record_qubits(qubit_used, q)
		elif stype in {"DelayInstruction", "QuantumDelay"}:
			qubits = [_qubit_index(q) for q in _as_qargs(getattr(stmt, "qubits", []))]
			duration = _duration_to_seconds(getattr(stmt, "duration", None))
			for q in qubits:
				new.append(("delay", duration, (q,)))
				_record_qubits(qubit_used, q)
		elif stype in {"ClassicalAssignment", "AssignmentStatement"}:
			target = (
				getattr(stmt, "lvalue", None)
				or getattr(stmt, "target", None)
				or getattr(stmt, "lhs", None)
			)
			value = (
				getattr(stmt, "rvalue", None)
				or getattr(stmt, "value", None)
				or getattr(stmt, "rhs", None)
				or getattr(stmt, "expression", None)
			)
			if value is not None and value.__class__.__name__ in {"QuantumMeasurement", "Measurement"}:
				qubit = _qubit_index(getattr(value, "qubit", None) or getattr(value, "argument", None))
				if target is not None:
					cbit = _bit_index(target)
					new.append(("measure", [qubit], [cbit]))
					_record_qubits(qubit_used, qubit)
					cbit_used.append(cbit)
		elif stype in {"QuantumMeasurement", "Measurement"}:
			qubit = _qubit_index(getattr(stmt, "qubit", None) or getattr(stmt, "argument", None))
			target = getattr(stmt, "target", None)
			if target is not None:
				cbit = _bit_index(target)
				new.append(("measure", [qubit], [cbit]))
				_record_qubits(qubit_used, qubit)
				cbit_used.append(cbit)
		elif stype in {"MeasurementStatement", "QuantumMeasurementStatement"}:
			measurement = getattr(stmt, "measurement", None) or getattr(stmt, "measure", None) or stmt
			qubit = _qubit_index(getattr(measurement, "qubit", None) or getattr(measurement, "argument", None))
			target = (
				getattr(stmt, "target", None)
				or getattr(stmt, "cbit", None)
				or getattr(measurement, "target", None)
			)
			if target is not None:
				cbit = _bit_index(target)
				new.append(("measure", [qubit], [cbit]))
				_record_qubits(qubit_used, qubit)
				cbit_used.append(cbit)
		else:
			continue

	if cbit_used == []:
		cbit_used = [i for i in range(len(set(qubit_used)))]
	return new, set(qubit_used), set(cbit_used)
