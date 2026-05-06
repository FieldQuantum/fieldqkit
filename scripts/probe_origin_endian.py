"""Determine OriginQ bitstring endianness using an asymmetric circuit.

Circuit (3 qubits):
    X q[0]   # only qubit 0 flipped
    measure q[i] -> c[i]

Conventions (this package: q[0] = leftmost char of bitstring):
    big-endian (package convention):  X q[0] -> dominant "100"
    little-endian (IBM/Qiskit style):  X q[0] -> dominant "001"
"""
from pyqpanda3.qcloud import QCloudService, QCloudOptions
from pyqpanda3.intermediate_compiler import convert_qasm_string_to_qprog
import yaml, time, sys

CHIP = sys.argv[1] if len(sys.argv) > 1 else "PQPUMESH8"
SHOTS = 256
tok = yaml.safe_load(open('.quantum_hw.yaml', 'r', encoding='utf8'))['credentials']['origin']['api_token']

qasm = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    'qreg q[3];\ncreg c[3];\n'
    'x q[0];\n'           # only flip qubit 0
    'measure q[0]->c[0];\n'
    'measure q[1]->c[1];\n'
    'measure q[2]->c[2];\n'
)
print(f"=== chip={CHIP}  qasm circuit: X on q[0] only ===", flush=True)

svc = QCloudService(api_key=tok)
b = svc.backend(CHIP)
prog = convert_qasm_string_to_qprog(qasm)
opts = QCloudOptions()
opts.set_amend(False); opts.set_mapping(False); opts.set_optimization(False)
job = b.run(prog, SHOTS, opts)
for _ in range(120):
    s = job.status()
    if s.name in ('FINISHED', 'FAILED'):
        print(f"final status: {s.name}", flush=True)
        break
    print(f"status: {s.name}", flush=True)
    time.sleep(3)
res = job.result()
counts = dict(res.get_counts())
print(f"raw counts: {counts}", flush=True)
total = sum(counts.values())
print(f"total = {total} (shots requested {SHOTS})", flush=True)
if not counts:
    sys.exit(1)
top = max(counts.items(), key=lambda kv: kv[1])
print(f"dominant bitstring: {top[0]!r}  count={top[1]}", flush=True)
bs = top[0]
if bs == "100":
    print(">>> ENDIAN (raw cloud): big-endian  -> matches package convention, no reversal needed")
elif bs == "001":
    print(">>> ENDIAN (raw cloud): little-endian (IBM/Qiskit style, q[0] on right)")
    print("    Adapter must reverse bitstrings to align with package big-endian convention.")
else:
    print(f">>> Unexpected dominant bitstring {bs!r}; inspect raw counts above")
