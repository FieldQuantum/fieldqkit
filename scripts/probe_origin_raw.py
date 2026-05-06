"""Minimal probe to inspect raw counts, topology and fidelities from OriginQ."""
from pyqpanda3.qcloud import QCloudService, QCloudOptions
from pyqpanda3.intermediate_compiler import convert_qasm_string_to_qprog
import yaml, time

tok = yaml.safe_load(open('.quantum_hw.yaml', 'r', encoding='utf8'))['credentials']['origin']['api_token']
svc = QCloudService(api_key=tok)
b = svc.backend('WK_C180')
chip = b.chip_info()
print('=== topology edge count:', len(chip.get_chip_topology()), flush=True)
print('=== first 8 topology edges:', chip.get_chip_topology()[:8], flush=True)
print('=== qubits_num:', chip.qubits_num(), flush=True)
print('=== available_qubits count:', len(chip.available_qubits()), flush=True)
print('=== available_qubits[:20]:', chip.available_qubits()[:20], flush=True)

print('=== first 5 single-qubit fidelities:', flush=True)
for s in chip.single_qubit_info()[:5]:
    print('  Q', s.get_qubit_id(),
          'sg_fid=', s.get_single_gate_fidelity(),
          'rdo_fid=', s.get_readout_fidelity(),
          'T1=', s.get_t1(),
          'T2=', s.get_t2(),
          'freq=', s.get_frequency(), flush=True)

# distribution of fidelities
sg = [x.get_single_gate_fidelity() for x in chip.single_qubit_info()]
ro = [x.get_readout_fidelity() for x in chip.single_qubit_info()]
dq = [x.get_fidelity() for x in chip.double_qubits_info()]
print(f"=== single-gate fidelity: count={len(sg)} min={min(sg):.4f} median={sorted(sg)[len(sg)//2]:.4f} max={max(sg):.4f}", flush=True)
print(f"=== readout fidelity:     count={len(ro)} min={min(ro):.4f} median={sorted(ro)[len(ro)//2]:.4f} max={max(ro):.4f}", flush=True)
print(f"=== two-qubit fidelity:   count={len(dq)} min={min(dq):.4f} median={sorted(dq)[len(dq)//2]:.4f} max={max(dq):.4f}", flush=True)

print('=== first 8 two-qubit edges with fidelity:', flush=True)
for d in chip.double_qubits_info()[:8]:
    print('  pair', d.get_qubits(), 'fid=', d.get_fidelity(), flush=True)

qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        'qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n'
        'measure q[0]->c[0];\nmeasure q[1]->c[1];\n')
prog = convert_qasm_string_to_qprog(qasm)
opts = QCloudOptions(); opts.set_amend(False); opts.set_mapping(False); opts.set_optimization(False)
SHOTS = 256
print(f"\n=== submitting Bell pair, shots={SHOTS} ...", flush=True)
job = b.run(prog, SHOTS, opts)
for _ in range(120):
    s = job.status()
    print('status:', s.name, flush=True)
    if s.name in ('FINISHED', 'FAILED'):
        break
    time.sleep(3)
res = job.result()
counts = res.get_counts()
total = sum(counts.values())
print('=== raw counts dtype:', type(counts).__name__, ' sum =', total, ' shots requested =', SHOTS, flush=True)
print('=== counts:', dict(counts), flush=True)
print('=== are values integer? ', all(isinstance(v, int) for v in counts.values()), flush=True)
print('=== sum == shots?         ', total == SHOTS, flush=True)
try:
    probs = res.get_probs()
    print('=== get_probs() also available:', dict(probs), flush=True)
except Exception as exc:
    print('=== get_probs error:', exc, flush=True)
try:
    raw = res.origin_data()
    print('=== origin_data() len:', len(raw) if raw else 0, ' first 200 chars:', (raw or '')[:200], flush=True)
except Exception as exc:
    print('=== origin_data error:', exc, flush=True)
