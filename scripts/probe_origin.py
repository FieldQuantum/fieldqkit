"""Probe Origin Quantum cloud (本源量子) via pyqpanda3.qcloud.

Usage:
    conda activate quantum
    python scripts/probe_origin.py [--shots N] [--chip CHIP_NAME]

Reads `origin.api_token` from .quantum_hw.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


def _load_token() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / ".quantum_hw.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    token = cfg.get("credentials", {}).get("origin", {}).get("api_token")
    if not token:
        raise SystemExit("origin api_token not found in .quantum_hw.yaml")
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Origin cloud platform")
    parser.add_argument("--chip", default=None, help="optional backend name to inspect/run")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--no-run", action="store_true", help="only list backends + chip info; do not submit")
    parser.add_argument("--url", default="https://qcloud.originqc.com.cn")
    args = parser.parse_args()

    from pyqpanda3.qcloud import QCloudService, JobStatus
    from pyqpanda3.intermediate_compiler import convert_qasm_string_to_qprog

    token = _load_token()
    print(f"[info] using URL = {args.url}")
    service = QCloudService(api_key=token, url=args.url)

    print("[step] backends() ->")
    backends = service.backends()
    print(json.dumps(backends, indent=2, ensure_ascii=False, default=str))

    if not backends:
        print("[warn] no backends returned")
        return 1

    chip_name = args.chip or next(iter(backends))
    print(f"\n[step] backend({chip_name!r}).chip_info() ->")

    backend = service.backend(chip_name)
    chip = backend.chip_info()
    print({
        "chip_id": chip.chip_id(),
        "qubits_num": chip.qubits_num(),
        "available_qubits_count": len(chip.available_qubits()),
        "basic_gates": chip.get_basic_gates(),
        "topology_edge_count": len(chip.get_chip_topology()),
        "single_gate_timing": chip.get_single_gate_timing(),
        "double_gate_timing": chip.get_double_gate_timing(),
        "high_frequency_qubits": chip.high_frequency_qubits()[:8],
    })

    sq = chip.single_qubit_info()
    if sq:
        s0 = sq[0]
        print("[step] single_qubit_info[0] ->", {
            "qubit_id": s0.get_qubit_id(),
            "frequency": s0.get_frequency(),
            "readout_fidelity": s0.get_readout_fidelity(),
            "single_gate_fidelity": s0.get_single_gate_fidelity(),
            "T1": s0.get_t1(),
            "T2": s0.get_t2(),
        })

    dq = chip.double_qubits_info()
    if dq:
        d0 = dq[0]
        print("[step] double_qubits_info[0] ->", {
            "qubits": d0.get_qubits(),
            "fidelity": d0.get_fidelity(),
        })

    if args.no_run:
        return 0

    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        "qreg q[2];\ncreg c[2];\n"
        "h q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    print("\n[step] submitting tiny Bell pair, shots=%d ..." % args.shots)
    prog = convert_qasm_string_to_qprog(qasm)
    from pyqpanda3.qcloud import QCloudOptions
    opts = QCloudOptions()
    opts.set_amend(True)
    opts.set_mapping(True)
    opts.set_optimization(True)
    opts.set_is_prob_counts(True)
    job = backend.run(prog, args.shots, opts)
    print(f"[info] job_id = {job.job_id()}")

    print("[step] polling status ...")
    deadline = time.time() + 600
    while time.time() < deadline:
        st = job.status()
        print(f"  status = {st.name}")
        if st in (JobStatus.FINISHED, JobStatus.FAILED):
            break
        time.sleep(5)

    res = job.result()
    print("[step] result ->")
    print({
        "job_id": res.job_id(),
        "status": res.job_status().name,
        "error_message": res.error_message(),
        "counts": res.get_counts(),
        "timing_info": res.timing_info(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
