"""Manual test for FieldQuantum ``expectation`` mode.

Run directly (not through pytest) once the cloud server's expectation path
is online::

    python test_fieldquantum_expectation.py

The script:
  1. Submits a small 2-qubit parametrised circuit + 2-term Hamiltonian.
  2. Polls until the task finishes.
  3. Prints the **raw** server response body (so you can see the actual
     wire shape, which is not pinned in the FieldQuantum docs).
  4. Prints the decoded payload returned by
     ``FieldQuantumPlatform.fetch_task_result`` and checks the expected
     keys (``energy`` / ``expectations`` / ``gradients``) and the
     gradients-length invariant.

Requires ``FIELDQUANTUM_API_TOKEN`` to be configured (env var or
``.quantum_hw.yaml``).
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")

from quantum_hw.api.quantum_platform.fieldquantum import FieldQuantumPlatform  # noqa: E402


QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry(theta_0) q[0];
ry(theta_1) q[1];
cz q[0],q[1];
ry(theta_2) q[0];
"""

PARAM_NAMES = ["theta_0", "theta_1", "theta_2"]
PARAM_VALUES = [0.31, 1.72, 0.85]
HAMILTONIAN = [
    {"coeff": -1.0, "pauli": "Z0 Z1"},
    {"coeff": -0.5, "pauli": "X0"},
]


def main() -> int:
    p = FieldQuantumPlatform()
    print(f"[submit] POST {p.base_url}/task/run  (mode=expectation)")
    task_id = p.submit_job({
        "mode": "expectation",
        "qasm": QASM,
        "param_names": PARAM_NAMES,
        "param_values": PARAM_VALUES,
        "hamiltonian": HAMILTONIAN,
    })
    print(f"[submit] task_id = {task_id}")

    # ---- poll status ----------------------------------------------------
    deadline = time.monotonic() + 600.0
    while True:
        status = p.query_task_status(task_id)
        print(f"[poll ] status = {status}")
        if status in ("finished", "failed", "error"):
            break
        if time.monotonic() > deadline:
            print("[poll ] timed out")
            return 2
        time.sleep(3.0)

    # ---- raw response (for inspecting the actual wire format) ----------
    raw = p._session.get(f"{p.base_url}/task/result/{task_id}", timeout=60)
    print(f"\n[raw  ] HTTP {raw.status_code}")
    print(f"[raw  ] body = {raw.text[:4000]}")

    # ---- decoded payload via the production code path -----------------
    print("\n[parse] FieldQuantumPlatform.fetch_task_result(...) =>")
    try:
        result = p.fetch_task_result(task_id)
    except Exception as exc:
        print(f"  RAISED: {exc}")
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- shape assertions ---------------------------------------------
    print("\n[check] expected keys: energy / expectations / gradients")
    missing = [k for k in ("energy", "expectations", "gradients") if k not in result]
    if missing:
        print(f"  MISSING keys: {missing}")
        return 4
    grads = result["gradients"]
    if not isinstance(grads, list) or len(grads) != len(PARAM_NAMES):
        print(
            f"  gradients length mismatch: got {len(grads) if hasattr(grads,'__len__') else '?'}, "
            f"expected {len(PARAM_NAMES)} (one per param_name)"
        )
        return 5
    exps = result["expectations"]
    if not isinstance(exps, dict):
        print(f"  expectations should be a dict, got {type(exps).__name__}")
        return 6
    for term in ("Z0 Z1", "X0"):
        if term not in exps:
            print(f"  expectations missing term {term!r}: got keys {list(exps)}")
            return 7

    print("\n[ok   ] all shape checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
