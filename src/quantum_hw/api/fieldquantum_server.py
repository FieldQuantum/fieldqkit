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

    *Expectation mode* — returns energy + per-Pauli expectations + gradients via
    automatic differentiation (exact statevector/MPS, no shots)::

        {
            "mode": "expectation",
            "qasm": "<OpenQASM 2.0 template, params as symbols e.g. theta_0>",
            "param_names": ["theta_0", "theta_1"],
            "param_values": [0.31, 1.72],
            "hamiltonian": [
                {"coeff": -1.0, "pauli": "Z0 Z1"},
                {"coeff": -0.5, "pauli": "X0"}
            ]
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
import re
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

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
    """Execute an *expectation* mode request via automatic differentiation.

    Uses exact statevector/MPS simulation with PyTorch autograd — no shots
    required.  Gradients are computed in a single backward pass.

    Args:
        data: Parsed JSON request body. Must contain ``"qasm"``,
              ``"param_names"``, ``"param_values"``, and ``"hamiltonian"``.

    Returns:
        ``{"energy": float, "expectations": {pauli: float, ...}, "gradients": [float, ...]}``
    """
    import torch  # noqa: PLC0415
    from ..circuit import QuantumCircuit  # noqa: PLC0415
    from ..sim import energy_and_expectations  # noqa: PLC0415

    qasm_template: str = data["qasm"]
    param_names: List[str] = data.get("param_names", [])
    param_values: List[float] = list(data.get("param_values", []))
    hamiltonian_raw: List[Dict[str, Any]] = data.get("hamiltonian", [])

    hamiltonian = [
        (float(term["coeff"]), str(term["pauli"]))
        for term in hamiltonian_raw
    ]

    symbolic_qc = QuantumCircuit().from_openqasm2(qasm_template)

    params = torch.tensor(param_values, dtype=torch.float64, requires_grad=bool(param_names))

    energy_tensor, expectations = energy_and_expectations(
        symbolic_qc,
        params=params,
        param_names=param_names,
        hamiltonian=hamiltonian,
    )

    gradients: List[float] = []
    if param_names:
        (grads,) = torch.autograd.grad(energy_tensor, params)
        gradients = grads.tolist()

    return {
        "energy": float(energy_tensor.detach().cpu().item()),
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
