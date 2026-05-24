"""
Generate 20 large-scale circuit simulation samples (up to 20 qubits).
No analytical validation -- purely for stress testing and data collection.
"""

from __future__ import annotations

import json
import math
import os
import sys

PI = math.pi


def _qreg(n):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\n'

def _qreg_creg(n):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\ncreg c[{n}];\n'

def _h_layer(n):
    return "\n".join(f"h q[{i}];" for i in range(n))

def _rx_layer(n, params):
    return "\n".join(f"rx({params[i % len(params)]}) q[{i}];" for i in range(n))

def _ry_layer(n, params):
    return "\n".join(f"ry({params[i % len(params)]}) q[{i}];" for i in range(n))

def _rz_layer(n, params):
    return "\n".join(f"rz({params[i % len(params)]}) q[{i}];" for i in range(n))

def _cx_chain(n):
    return "\n".join(f"cx q[{i}],q[{i+1}];" for i in range(n - 1))

def _cx_even(n):
    return "\n".join(f"cx q[{i}],q[{i+1}];" for i in range(0, n - 1, 2))

def _cx_odd(n):
    return "\n".join(f"cx q[{i}],q[{i+1}];" for i in range(1, n - 1, 2))

def _cx_ring(n):
    return _cx_chain(n) + f"\ncx q[{n-1}],q[0];"

def _ising_layer(n, param):
    lines = []
    for i in range(n - 1):
        lines += [f"cx q[{i}],q[{i+1}];", f"rz({param}) q[{i+1}];", f"cx q[{i}],q[{i+1}];"]
    return "\n".join(lines)

def _zz_chain(n, coeff=-1.0):
    return [{"coeff": coeff, "pauli": f"Z{i} Z{i+1}"} for i in range(n - 1)]

def _x_field(n, coeff=-0.5):
    return [{"coeff": coeff, "pauli": f"X{i}"} for i in range(n)]


large_cases = [
    # ── Expectation cases (L11-L20) ───────────────────────────────────────────

    # L11: 8q VQE Ising, 4 params with negation + product
    {
        "desc": "L11 | 8q VQE Ising, 4 params (-theta, theta*theta)",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(8)
                + _ry_layer(8, ["theta_0", "theta_1", "-theta_2", "theta_3"]) + "\n"
                + _cx_chain(8) + "\n"
                + _rz_layer(8, ["theta_0*theta_1", "theta_2", "-theta_3", "theta_1"]) + "\n"
                + _cx_chain(8) + "\n"
                + _ry_layer(8, ["theta_2+theta_3", "theta_0", "-theta_1", "theta_2*theta_3"])
            ),
            "param_names": ["theta_0", "theta_1", "theta_2", "theta_3"],
            "param_values": [0.4, 0.7, 1.1, 0.3],
            "hamiltonian": _zz_chain(8) + _x_field(8, -0.5),
        },
    },

    # L12: 10q QAOA p=2, gamma/beta params
    {
        "desc": "L12 | 10q QAOA p=2, 4 symbolic params",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(10)
                + _h_layer(10) + "\n"
                + _ising_layer(10, "gamma_0") + "\n"
                + _rx_layer(10, ["beta_0"]) + "\n"
                + _ising_layer(10, "gamma_1") + "\n"
                + _rx_layer(10, ["beta_1"])
            ),
            "param_names": ["gamma_0", "beta_0", "gamma_1", "beta_1"],
            "param_values": [0.5, 0.8, 0.3, 1.1],
            "hamiltonian": _zz_chain(10),
        },
    },

    # L13: 12q hardware-efficient ansatz, product params
    {
        "desc": "L13 | 12q HEA, 6 params with theta_i*theta_j",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(12)
                + "\n".join(f"ry(theta_{i%6}) q[{i}];" for i in range(12)) + "\n"
                + _cx_even(12) + "\n"
                + "\n".join(
                    f"rz(theta_{i%6}*theta_{(i+1)%6}) q[{i}];" for i in range(12)
                ) + "\n"
                + _cx_odd(12) + "\n"
                + "\n".join(f"ry(-theta_{i%6}) q[{i}];" for i in range(12))
            ),
            "param_names": [f"theta_{i}" for i in range(6)],
            "param_values": [0.3, 0.6, 0.9, 0.4, 0.7, 1.0],
            "hamiltonian": _zz_chain(12) + _x_field(12, -0.3),
        },
    },

    # L14: 10q Heisenberg-like (XX + YY + ZZ), 3 params with sum/product
    {
        "desc": "L14 | 10q Heisenberg (XX+YY+ZZ), 3 params",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(10)
                + _ry_layer(10, ["theta_0", "theta_1", "-theta_2"]) + "\n"
                + _cx_chain(10) + "\n"
                + _rz_layer(10, ["theta_1*theta_2", "-theta_0"]) + "\n"
                + _cx_chain(10) + "\n"
                + _rx_layer(10, ["theta_0+theta_2", "theta_1"])
            ),
            "param_names": ["theta_0", "theta_1", "theta_2"],
            "param_values": [0.6, 1.0, 0.4],
            "hamiltonian": (
                [{"coeff": -1.0, "pauli": f"X{i} X{i+1}"} for i in range(9)]
                + [{"coeff": -1.0, "pauli": f"Y{i} Y{i+1}"} for i in range(9)]
                + [{"coeff": -0.5, "pauli": f"Z{i} Z{i+1}"} for i in range(9)]
            ),
        },
    },

    # L15: 14q 4-layer ansatz, 5 params, mixed expressions
    {
        "desc": "L15 | 14q 4-layer ansatz, 5 params mixed",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(14)
                + "\n".join(f"ry(theta_{i%5}) q[{i}];" for i in range(14)) + "\n"
                + _cx_even(14) + "\n"
                + "\n".join(f"rz(theta_{i%5}*theta_{(i+1)%5}) q[{i}];" for i in range(14)) + "\n"
                + _cx_odd(14) + "\n"
                + "\n".join(f"ry(theta_{i%5}+theta_{(i+2)%5}) q[{i}];" for i in range(14)) + "\n"
                + _cx_even(14) + "\n"
                + "\n".join(f"rz(-theta_{i%5}) q[{i}];" for i in range(14)) + "\n"
                + _cx_odd(14) + "\n"
                + "\n".join(f"rx(2*theta_{i%5}) q[{i}];" for i in range(14))
            ),
            "param_names": [f"theta_{i}" for i in range(5)],
            "param_values": [0.4, 0.8, 1.2, 0.6, 1.0],
            "hamiltonian": _zz_chain(14) + _x_field(14, -0.5),
        },
    },

    # L16: 12q ring QAOA p=3, 6 params
    {
        "desc": "L16 | 12q ring QAOA p=3, 6 params",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(12)
                + _h_layer(12) + "\n"
                + _ising_layer(12, "gamma_0") + "\n"
                + f"cx q[11],q[0];\nrz(gamma_0) q[0];\ncx q[11],q[0];\n"
                + _rx_layer(12, ["beta_0"]) + "\n"
                + _ising_layer(12, "gamma_1") + "\n"
                + f"cx q[11],q[0];\nrz(gamma_1) q[0];\ncx q[11],q[0];\n"
                + _rx_layer(12, ["beta_1"]) + "\n"
                + _ising_layer(12, "gamma_2") + "\n"
                + f"cx q[11],q[0];\nrz(gamma_2) q[0];\ncx q[11],q[0];\n"
                + _rx_layer(12, ["beta_2"])
            ),
            "param_names": ["gamma_0", "beta_0", "gamma_1", "beta_1", "gamma_2", "beta_2"],
            "param_values": [0.5, 0.8, 0.4, 1.0, 0.3, 0.9],
            "hamiltonian": _zz_chain(12) + [{"coeff": -1.0, "pauli": "Z11 Z0"}],
        },
    },

    # L17: 8q deep circuit, 8 params, all expression types
    {
        "desc": "L17 | 8q deep 6-layer, 8 params, all expression types",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(8)
                + "\n".join(f"ry(theta_{i}) q[{i}];" for i in range(8)) + "\n"
                + _cx_chain(8) + "\n"
                + "\n".join(f"rz(theta_{i}*theta_{(i+1)%8}) q[{i}];" for i in range(8)) + "\n"
                + _cx_chain(8) + "\n"
                + "\n".join(f"rx(theta_{i}+theta_{(i+2)%8}) q[{i}];" for i in range(8)) + "\n"
                + _cx_chain(8) + "\n"
                + "\n".join(f"ry(-theta_{i}) q[{i}];" for i in range(8)) + "\n"
                + _cx_chain(8) + "\n"
                + "\n".join(f"rz(2*theta_{i}) q[{i}];" for i in range(8)) + "\n"
                + _cx_chain(8) + "\n"
                + "\n".join(f"rx(theta_{i}-theta_{(i+3)%8}) q[{i}];" for i in range(8))
            ),
            "param_names": [f"theta_{i}" for i in range(8)],
            "param_values": [0.3, 0.6, 0.9, 1.2, 0.5, 0.8, 1.1, 0.4],
            "hamiltonian": (
                _zz_chain(8)
                + _x_field(8, -0.3)
                + [{"coeff": -0.5, "pauli": f"Y{i} Y{i+1}"} for i in range(7)]
            ),
        },
    },

    # L18: 16q VQE, 4 params, sum/difference expressions
    {
        "desc": "L18 | 16q VQE, 4 params, theta_i+/-theta_j",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(16)
                + "\n".join(f"ry(theta_{i%4}+theta_{(i+1)%4}) q[{i}];" for i in range(16)) + "\n"
                + _cx_chain(16) + "\n"
                + "\n".join(f"rz(theta_{i%4}-theta_{(i+2)%4}) q[{i}];" for i in range(16)) + "\n"
                + _cx_odd(16) + "\n"
                + "\n".join(f"rx(2*theta_{i%4}) q[{i}];" for i in range(16))
            ),
            "param_names": [f"theta_{i}" for i in range(4)],
            "param_values": [0.5, 0.8, 1.2, 0.3],
            "hamiltonian": _zz_chain(16) + _x_field(16, -0.4),
        },
    },

    # L19: 20q VQE shallow brickwork, 4 params with product
    {
        "desc": "L19 | 20q VQE shallow brickwork, 4 params",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(20)
                + "\n".join(f"ry(theta_{i%4}) q[{i}];" for i in range(20)) + "\n"
                + _cx_even(20) + "\n"
                + "\n".join(f"rz(theta_{i%4}*theta_{(i+1)%4}) q[{i}];" for i in range(20)) + "\n"
                + _cx_odd(20) + "\n"
                + "\n".join(f"rx(-theta_{i%4}) q[{i}];" for i in range(20))
            ),
            "param_names": [f"theta_{i}" for i in range(4)],
            "param_values": [0.5, 0.9, 1.3, 0.7],
            "hamiltonian": _zz_chain(20) + _x_field(20, -0.5),
        },
    },

    # L20: 16q 5-layer VQE, 6 params, full expression variety
    {
        "desc": "L20 | 16q 5-layer VQE, 6 params, all expression types",
        "input": {
            "mode": "expectation",
            "qasm": (
                _qreg(16)
                + "\n".join(f"ry(theta_{i%6}+theta_{(i+1)%6}) q[{i}];" for i in range(16)) + "\n"
                + _cx_even(16) + "\n"
                + "\n".join(f"rz(theta_{i%6}*theta_{(i+2)%6}) q[{i}];" for i in range(16)) + "\n"
                + _cx_odd(16) + "\n"
                + "\n".join(f"rx(-theta_{i%6}) q[{i}];" for i in range(16)) + "\n"
                + _cx_even(16) + "\n"
                + "\n".join(f"ry(2*theta_{i%6}) q[{i}];" for i in range(16)) + "\n"
                + _cx_odd(16) + "\n"
                + "\n".join(f"rz(theta_{i%6}-theta_{(i+3)%6}) q[{i}];" for i in range(16))
            ),
            "param_names": [f"theta_{i}" for i in range(6)],
            "param_values": [0.3, 0.7, 1.1, 0.5, 0.9, 1.3],
            "hamiltonian": (
                _zz_chain(16)
                + _x_field(16, -0.4)
                + [{"coeff": -0.3, "pauli": f"Z{i} Z{i+2}"} for i in range(14)]
            ),
        },
    },
]


def main() -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from quantum_hw.api.fieldquantum_server import _handle_sample, _handle_expectation

    output_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(output_dir, exist_ok=True)

    def _save(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    n_ok = n_fail = 0
    for i, case in enumerate(large_cases):
        tag = f"L{i+1:02d}"
        mode = case["input"]["mode"]
        print(f"[{tag}] {case['desc']} ...", end=" ", flush=True)
        try:
            result = _handle_expectation(case["input"])
            e = result.get("energy")
            g = result.get("gradients", [])
            print(f"energy={e:.4f}, n_grads={len(g)}")
            index = i + 11
            n_ok += 1
        except Exception as exc:
            import traceback
            print(f"ERROR: {exc}")
            traceback.print_exc()
            result = {"error": str(exc)}
            n_fail += 1

        _save(os.path.join(output_dir, f"{mode}_{index:02d}_input.json"), case["input"])
        _save(os.path.join(output_dir, f"{mode}_{index:02d}_output.json"), result)

    print(f"\nDone: {n_ok}/20 succeeded, {n_fail} failed.")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
