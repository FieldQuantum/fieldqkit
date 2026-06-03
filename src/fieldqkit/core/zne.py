"""Zero-noise extrapolation helpers."""

from __future__ import annotations


def apply_zne_cz_tripling(qct):
	"""Apply CZ gate tripling for simple ZNE noise scaling (scale factor 3).

	Args:
		qct (*QuantumCircuit*): Circuit to apply CZ tripling to.

	Returns:
		A copy of the circuit with each CZ gate tripled.
	"""
	qct_new = qct.deepcopy()
	gate_list = qct_new.gates
	gate_list_new = []
	for gate in gate_list:
		gate_list_new.append(gate)
		if gate[0] == "cz":
			# Insert two extra CZ gates to scale noise (1x -> 3x).
			gate_list_new.append(gate)
			gate_list_new.append(gate)
	qct_new.gates = gate_list_new
	return qct_new


def zne_linear_extrapolate(probs_1, probs_3):
	"""Richardson linear extrapolation from noise scale 1 and 3 to zero noise.

	Works for both probability vectors and scalar expectations.
	Output probabilities are *not* renormalized.

	Args:
		probs_1 (*np.ndarray | float*): Probabilities or expectation values at noise scale factor 1.
		probs_3 (*np.ndarray | float*): Probabilities or expectation values at noise scale factor 3.

	Returns:
		Linearly extrapolated zero-noise estimate.
	"""
	# Assume linear dependence on noise scaling factor: extrapolate to zero noise.
	return (3 * probs_1 - probs_3) / 2.0
