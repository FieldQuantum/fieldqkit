"""FieldQuantum local cloud simulator server.

Exposes the local quantum simulator (statevector / MPS) via HTTP, making it
usable as a drop-in "cloud" backend through the FieldQuantum provider.

Usage::

    # Start with defaults (localhost:8765)
    python -m quantum_hw.api.fieldquantum_server

    # Custom port / host
    python -m quantum_hw.api.fieldquantum_server --port 9000 --host 0.0.0.0

REST API
--------
``GET /health``
    Liveness probe. Returns ``{"status": "ok"}``.

``POST /run``
    Run a circuit. Request body is JSON:

    *Sample mode* — returns measurement counts::

        {
            "mode": "sample",
            "qasm": "<OpenQASM 2.0 string with measurements>",
            "shots": 1024
        }

    Response::

        {"counts": {"0000": 512, "0011": 302, ...}}

    *Expectation mode* — returns energy + per-Pauli expectations + gradients::

        {
            "mode": "expectation",
            "qasm": "<OpenQASM 2.0 template, params as symbols e.g. theta_0>",
            "param_names": ["theta_0", "theta_1"],
            "param_values": [0.31, 1.72],
            "hamiltonian": [
                {"coeff": -1.0, "pauli": "Z0 Z1"},
                {"coeff": -0.5, "pauli": "X0"}
            ],
            "shots": 8192
        }

    Response::

        {
            "energy": -1.23,
            "expectations": {"Z0 Z1": -0.80, "X0": 0.31},
            "gradients": [0.12, -0.34]
        }
"""

from __future__ import annotations

import json
import logging
import math
import re
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory task store  {task_id: {"status": "finished"|"error", "result": {...}, "error": str}}
# ---------------------------------------------------------------------------
_task_store: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substitute_params(
    qasm: str, param_names: List[str], param_values: List[float]
) -> str:
    """Replace symbolic parameter names with numeric values in *qasm*.

    Uses whole-word matching so ``theta_0`` does not accidentally replace
    a substring of another identifier.

    Args:
        qasm: OpenQASM string possibly containing symbolic parameter names.
        param_names: Ordered list of symbolic names.
        param_values: Numeric values to substitute.

    Returns:
        QASM string with all symbolic names replaced by their float values.
    """
    for name, val in zip(param_names, param_values):
        qasm = re.sub(
            r"\b" + re.escape(name) + r"\b",
            str(float(val)),
            qasm,
        )
    return qasm


def _num_qubits_from_qasm(qasm: str) -> int:
    """Extract the total qubit count declared in *qasm* via ``qreg``.

    Falls back to 1 when no ``qreg`` declaration is found.

    Args:
        qasm: OpenQASM 2.0 source string.

    Returns:
        Total number of qubits (sum over all ``qreg`` declarations).
    """
    total = 0
    for m in re.finditer(r"qreg\s+\w+\[(\d+)\]", qasm):
        total += int(m.group(1))
    return max(total, 1)


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

def _handle_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a *sample* mode request using the local simulator.

    Args:
        data: Parsed JSON request body. Must contain ``"qasm"`` and optionally
              ``"shots"``, ``"param_names"``, ``"param_values"``.

    Returns:
        ``{"counts": {bitstring: int, ...}}``
    """
    # Import inside function to allow the server module to be imported without
    # immediately pulling in PyTorch / the full package.
    from ..circuit import QuantumCircuit  # noqa: PLC0415
    from ..sim import simulate_counts  # noqa: PLC0415

    qasm: str = data["qasm"]
    shots: int = int(data.get("shots", 1024))
    param_names: List[str] = data.get("param_names", [])
    param_values: List[float] = data.get("param_values", [])

    if param_names:
        qasm = _substitute_params(qasm, param_names, param_values)

    qc = QuantumCircuit().from_openqasm2(qasm)
    counts = simulate_counts(qc, shots)
    return {"counts": counts}


def _handle_expectation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an *expectation* mode request.

    Computes Pauli expectation values via basis-rotated sampling and returns
    parameter-shift gradients (2N+1 circuit evaluations for N parameters).

    Args:
        data: Parsed JSON request body. Must contain ``"qasm"``,
              ``"param_names"``, ``"param_values"``, and ``"hamiltonian"``.

    Returns:
        ``{"energy": float, "expectations": {pauli: float, ...}, "gradients": [float, ...]}``
    """
    from ..circuit import QuantumCircuit  # noqa: PLC0415
    from ..sim import simulate_counts  # noqa: PLC0415
    from ..core.observables import (  # noqa: PLC0415
        group_observables,
        pauli_expectation,
        append_measurement_basis,
    )
    from ..core.utils import get_samples  # noqa: PLC0415

    qasm_template: str = data["qasm"]
    param_names: List[str] = data.get("param_names", [])
    param_values: List[float] = list(data.get("param_values", []))
    hamiltonian: List[Dict[str, Any]] = data.get("hamiltonian", [])
    shots: int = int(data.get("shots", 8192))

    # Determine num_qubits from the QASM header (substitute any params first)
    qasm_concrete = _substitute_params(qasm_template, param_names, param_values)
    num_qubits = _num_qubits_from_qasm(qasm_concrete)

    # Separate identity term (constant energy offset)
    identity_energy = sum(
        float(term["coeff"])
        for term in hamiltonian
        if str(term.get("pauli", "")).strip().upper() == "I"
    )
    pauli_terms = [
        (float(term["coeff"]), str(term["pauli"]))
        for term in hamiltonian
        if str(term.get("pauli", "")).strip().upper() != "I"
    ]

    # Unique observable strings needed for measurement grouping
    unique_obs = list({p for _, p in pauli_terms})

    def _eval_energy(values: List[float]) -> float:
        """Evaluate energy at *values* by sampling each measurement group."""
        qasm = _substitute_params(qasm_template, param_names, values)
        qc_base = QuantumCircuit().from_openqasm2(qasm)

        if not unique_obs:
            return identity_energy

        groups = group_observables(unique_obs, num_qubits=num_qubits)
        obs_vals: Dict[str, float] = {}

        for group in groups:
            basis_pattern = group["basis"]
            qct = qc_base.deepcopy()
            # Remove any pre-existing measurements — the ansatz should not have them.
            qct.remove_gate("measure")
            target_qubits = list(range(num_qubits))
            if basis_pattern is not None:
                append_measurement_basis(qct, basis_pattern, target_qubits=target_qubits)
            else:
                qct.measure(target_qubits, list(range(num_qubits)))

            counts = simulate_counts(qct, shots)
            sample_key = next(iter(counts), "")
            num_bits = len(sample_key) if sample_key else num_qubits
            samples = get_samples(counts, num_bits)
            for obs in group["observables"]:
                obs_vals[obs] = float(pauli_expectation(samples, obs))

        energy = identity_energy + sum(c * obs_vals.get(p, 0.0) for c, p in pauli_terms)
        return energy

    # Forward pass at the requested parameter values
    energy = _eval_energy(param_values)

    # Collect per-Pauli expectations at the current parameter values
    # (reuse evaluations from the same grouped runs)
    qasm_cur = _substitute_params(qasm_template, param_names, param_values)
    qc_cur = QuantumCircuit().from_openqasm2(qasm_cur)
    expectations: Dict[str, float] = {"I": 1.0}  # Identity always 1

    if unique_obs:
        groups_cur = group_observables(unique_obs, num_qubits=num_qubits)
        for group in groups_cur:
            basis_pattern = group["basis"]
            qct = qc_cur.deepcopy()
            qct.remove_gate("measure")
            target_qubits = list(range(num_qubits))
            if basis_pattern is not None:
                append_measurement_basis(qct, basis_pattern, target_qubits=target_qubits)
            else:
                qct.measure(target_qubits, list(range(num_qubits)))
            counts = simulate_counts(qct, shots)
            sample_key = next(iter(counts), "")
            num_bits = len(sample_key) if sample_key else num_qubits
            samples_cur = get_samples(counts, num_bits)
            for obs in group["observables"]:
                expectations[obs] = float(pauli_expectation(samples_cur, obs))

    # Parameter-shift gradients: grad_i = (E(θ + π/2 e_i) − E(θ − π/2 e_i)) / 2
    gradients: List[float] = []
    for i in range(len(param_names)):
        vp = list(param_values)
        vp[i] += math.pi / 2.0
        vm = list(param_values)
        vm[i] -= math.pi / 2.0
        gradients.append(float((_eval_energy(vp) - _eval_energy(vm)) / 2.0))

    return {
        "energy": float(energy),
        "expectations": expectations,
        "gradients": gradients,
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _SimulatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler dispatching POST /run to the local simulator."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        # /task/<task_id>/status  or  /task/<task_id>/result
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "task":
            task_id, endpoint = parts[1], parts[2]
            entry = _task_store.get(task_id)
            if entry is None:
                self._send_json({"error": f"task not found: {task_id}"}, status=404)
                return
            if endpoint == "status":
                self._send_json({"task_id": task_id, "status": entry["status"]})
            elif endpoint == "result":
                if entry["status"] == "error":
                    self._send_json({"error": entry.get("error", "unknown error")}, status=500)
                else:
                    self._send_json(entry["result"])
            else:
                self.send_error(404, "Not found")
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            logger.error("Failed to parse request body: %s", exc)
            self._send_json({"error": f"Bad request: {exc}"}, status=400)
            return

        task_id = str(uuid.uuid4())
        mode = data.get("mode", "sample")
        try:
            if mode == "sample":
                result = _handle_sample(data)
            elif mode == "expectation":
                result = _handle_expectation(data)
            else:
                self._send_json({"error": f"Unknown mode: {mode!r}"}, status=400)
                return
        except Exception as exc:
            logger.error("Error handling %r request:\n%s", mode, traceback.format_exc())
            _task_store[task_id] = {"status": "error", "error": str(exc)}
            self._send_json({"error": str(exc)}, status=500)
            return

        _task_store[task_id] = {"status": "finished", "result": result}
        self._send_json({"task_id": task_id})

    def _send_json(self, obj: Any, status: int = 200) -> None:
        response = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format, *args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(port: int = 8765, host: str = "localhost") -> None:
    """Start the FieldQuantum simulator HTTP server (blocking).

    Args:
        port: TCP port to listen on. Defaults to ``8765``.
        host: Hostname or IP to bind to. Defaults to ``"localhost"``.
    """
    server = HTTPServer((host, port), _SimulatorHandler)
    logger.info(
        "FieldQuantum simulator server listening on http://%s:%d", host, port
    )
    print(f"FieldQuantum simulator server running on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def main() -> None:
    """CLI entry point for ``python -m quantum_hw.api.fieldquantum_server``."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="FieldQuantum local cloud simulator server"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="TCP port (default: 8765)"
    )
    parser.add_argument(
        "--host", default="localhost", help="Bind host (default: localhost)"
    )
    args = parser.parse_args()
    serve(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
