from __future__ import annotations

from copy import copy


def apply_zne_cz_tripling(qct):
	"""Apply CZ gate tripling for simple ZNE scaling."""
	qct_new = copy(qct)
	gate_list = qct_new.gates
	gate_list_new = []
	for gate in gate_list:
		gate_list_new.append(gate)
		if gate[0] == "cz":
			gate_list_new.append(gate)
			gate_list_new.append(gate)
	qct_new.gates = gate_list_new
	return qct_new


def zne_linear_extrapolate(probs_1, probs_3):
	"""Linear extrapolation from scale 1 and 3 probabilities."""
	return (3 * probs_1 - probs_3) / 2.0
