"""Large-scale GHZ benchmark on OriginQ WK_C180 via run_auto.

Mirrors the structure of ``examples/demo_full.ipynb`` but targets the
OriginQ WK_C180 superconducting chip and exercises both ZNE and
readout-error mitigation.

Usage:
    conda activate quantum
    python scripts/demo_origin_wk_c180.py [--nq 8] [--shots 4096]
        [--no-mitigation] [--chip WK_C180]
"""

from __future__ import annotations

import argparse
import time
import traceback
from typing import Optional, Sequence

from quantum_hw import QuantumHardwareClient


def _section(title: str) -> None:
    print("\n" + "=" * 18, title, "=" * 18, flush=True)


def _ghz_observables(num_qubits: int) -> list[str]:
    # ZIIIZ-style nearest-neighbour ZZ correlations + global Z parity
    obs = []
    for i in range(num_qubits - 1):
        s = ["I"] * num_qubits
        s[i] = "Z"
        s[i + 1] = "Z"
        obs.append("".join(s))
    obs.append("Z" * num_qubits)
    return obs


def _run_case(
    client: QuantumHardwareClient,
    *,
    name: str,
    num_qubits: int,
    chip: str,
    shots: int,
    zne: bool,
    readout_mitigation: bool,
    observables: Sequence[str],
) -> Optional[object]:
    _section(f"{name}  (zne={zne}, readout={readout_mitigation}, n={num_qubits})")
    circuit = client.build_circuit("ghz", num_qubits=num_qubits, measure=False)
    print(f"[info] ghz circuit gate count: {len(circuit.gates)}", flush=True)

    t0 = time.time()
    try:
        result = client.run_auto(
            circuit=circuit,
            name=name,
            num_qubits=num_qubits,
            shots=shots,
            observables=list(observables),
            return_probabilities=True,
            provider="origin",
            prefer_chips=[chip],
            print_true=True,
            zne=zne,
            readout_mitigation=readout_mitigation,
            max_wait_time=3600,
            sleep_time=10,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[ERROR] case {name!r} failed after {elapsed:.1f}s: {exc}")
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    print(f"[done] case {name!r} took {elapsed:.1f}s")
    print(f"[result] task_ids = {result.task_ids}")
    if getattr(result, "observable_values_raw", None):
        print(f"[result] observable_values_raw = {result.observable_values_raw}")
    print(f"[result] observable_values     = {result.observable_values}")
    if getattr(result, "probabilities", None):
        probs = result.probabilities
        if isinstance(probs, dict):
            items = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
            print(f"[result] top-5 probabilities  = {items}")
        elif isinstance(probs, list) and probs:
            # First element is typically the (mitigated) distribution dict.
            first = probs[0] if isinstance(probs[0], dict) else None
            if first:
                items = sorted(first.items(), key=lambda kv: -kv[1])[:5]
                print(f"[result] top-5 probabilities[0] = {items}")
            else:
                print(f"[result] probabilities (list len={len(probs)})")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq", type=int, default=4, help="number of qubits in GHZ state")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--chip", default="WK_C180")
    parser.add_argument("--no-mitigation", action="store_true",
                        help="skip the ZNE + readout mitigation case")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the no-mitigation baseline case")
    args = parser.parse_args()

    client = QuantumHardwareClient()
    obs = _ghz_observables(args.nq)
    print(f"[info] target chip = {args.chip}")
    print(f"[info] observables = {obs}")

    summary: dict[str, object] = {}
    if not args.no_baseline:
        r1 = _run_case(
            client,
            name=f"wkc180_ghz{args.nq}_baseline",
            num_qubits=args.nq,
            chip=args.chip,
            shots=args.shots,
            zne=False,
            readout_mitigation=False,
            observables=obs,
        )
        summary["baseline"] = r1.observable_values if r1 else None

    if not args.no_mitigation:
        r2 = _run_case(
            client,
            name=f"wkc180_ghz{args.nq}_mitigated",
            num_qubits=args.nq,
            chip=args.chip,
            shots=args.shots,
            zne=True,
            readout_mitigation=True,
            observables=obs,
        )
        summary["mitigated_raw"] = (
            r2.observable_values_raw if r2 else None
        )
        summary["mitigated"] = r2.observable_values if r2 else None

    _section("summary")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
