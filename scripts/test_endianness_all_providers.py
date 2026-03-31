"""
四家硬件后端比特序（endianness）一致性验证脚本
================================================================

测试方法：
  构造一条 2-qubit 线路，只对 q[0] 施加 X 门，其余保持 |0⟩。
  通过 run_auto 提交到四家后端，观察返回 samples 中的比特位对应关系
  同时用 q[1] 做 X 门的对照实验来双重确认。

四家后端：Tencent / Quafu / TianYan / GuoDun
================================================================
"""

import json
import sys
import traceback
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_hw.api.client import QuantumHardwareClient
from quantum_hw.circuit import QuantumCircuit

SHOTS = 1024

PROVIDERS = {
    "tencent": {"prefer_chips": ["tianji_s2"]},
    "quafu": {"prefer_chips": ["Baihua"]},
    "tianyan": {"prefer_chips": ["tianyan176"]},
    "guodun": {"prefer_chips": ["gd_qc1"]},
}

# 用 QASM 字符串传入 run_auto，避免 QuantumCircuit qubit 计数问题
QASM_X_ON_Q0 = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[0];
z q[1];
"""

QASM_X_ON_Q1 = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
z q[0];
x q[1];
"""


def _analyze_samples(samples: List[List[int]], label: str) -> str:
    """从 samples 分析比特序。"""
    arr = np.array(samples)
    if arr.size == 0:
        return "unknown (no samples)"
    nshots = arr.shape[0]
    # 统计每列中 1 的比例
    col_means = arr.mean(axis=0)
    dominant_col = int(np.argmax(col_means))
    pct = col_means[dominant_col] * 100

    if label == "X_on_q0":
        if dominant_col == 0:
            return f"samples[:,0]~=1 ({pct:.1f}%)  =>  samples[i][0] = q[0]"
        else:
            return f"samples[:,{dominant_col}]~=1 ({pct:.1f}%)  =>  samples[i][{dominant_col}] = q[0]"
    elif label == "X_on_q1":
        if dominant_col == 1:
            return f"samples[:,1]~=1 ({pct:.1f}%)  =>  samples[i][1] = q[1]"
        else:
            return f"samples[:,{dominant_col}]~=1 ({pct:.1f}%)  =>  samples[i][{dominant_col}] = q[1]"
    return f"dominant col={dominant_col} ({pct:.1f}%)"


def _run_provider(provider: str, config: dict) -> Optional[Dict[str, Any]]:
    """用 run_auto 向指定后端提交两条线路。"""
    client = QuantumHardwareClient()
    results = {}

    for label, qasm in [("X_on_q0", QASM_X_ON_Q0), ("X_on_q1", QASM_X_ON_Q1)]:
        print(f"  提交线路: {label} ...")
        try:
            res = client.run_auto(
                circuit=qasm,
                name=f"endianness_{label}_{time.strftime('%Y%m%d_%H%M%S')}",
                num_qubits=2,
                provider=provider,
                shots=SHOTS,
                prefer_chips=config.get("prefer_chips"),
                return_probabilities=True,
                print_true=True,
            )
            # samples shape: [[samples_group0]]  (one group, no observables)
            samples = res.samples[0] if res.samples else []
            probabilities = res.probabilities[0] if res.probabilities else []
            results[label] = {
                "samples": samples,
                "probabilities": probabilities,
                "task_ids": res.task_ids,
            }
            arr = np.array(samples)
            col_means = arr.mean(axis=0).tolist() if arr.size else []
            print(f"    task_ids = {res.task_ids}")
            print(f"    samples shape = {arr.shape}")
            print(f"    column means (每列中 1 的比例) = {[f'{m:.3f}' for m in col_means]}")
            print(f"    probabilities |00>,|01>,|10>,|11> = {[f'{p:.3f}' for p in probabilities[:4]]}")
        except Exception as e:
            print(f"    [错误] {e}")
            traceback.print_exc()

    return results if results else None


def print_summary(all_results: dict):
    """打印汇总。"""
    print("\n\n" + "=" * 70)
    print("汇总: run_auto 返回的 samples 比特序分析")
    print("=" * 70)

    print(f"\n{'后端':<12} {'X_on_q0 分析':<55} {'X_on_q1 分析':<55}")
    print("-" * 122)

    consistency = {}

    for provider in ["tencent", "quafu", "tianyan", "guodun"]:
        if provider in all_results:
            r = all_results[provider]
            q0 = _analyze_samples(r["X_on_q0"]["samples"], "X_on_q0") if "X_on_q0" in r else "未测试"
            q1 = _analyze_samples(r["X_on_q1"]["samples"], "X_on_q1") if "X_on_q1" in r else "未测试"

            # 判断 samples[i][0] 对应哪个 qubit
            if "X_on_q0" in r:
                arr = np.array(r["X_on_q0"]["samples"])
                if arr.size:
                    dominant_col = int(np.argmax(arr.mean(axis=0)))
                    consistency[provider] = f"samples[i][0] = q[{dominant_col}]"
        else:
            q0 = "未测试 (提交失败或跳过)"
            q1 = "未测试 (提交失败或跳过)"

        print(f"{provider:<12} {q0:<55} {q1:<55}")

    # 一致性
    if len(consistency) >= 2:
        print(f"\n---")
        for p, mapping in consistency.items():
            print(f"  {p}: {mapping}")

        mappings = set(consistency.values())
        if len(mappings) == 1:
            print(f"\n[OK] 所有后端的 samples 比特排列一致: {mappings.pop()}")
        else:
            print(f"\n[WARN] 后端 samples 比特排列不一致!")


def main():
    print("=" * 70)
    print("四家后端比特序验证 (via run_auto)")
    print("=" * 70)

    all_results = {}
    for provider, config in PROVIDERS.items():
        print(f"\n{'='*50}")
        print(f"  {provider.upper()}")
        print(f"{'='*50}")
        try:
            result = _run_provider(provider, config)
            if result:
                all_results[provider] = result
        except Exception:
            print(f"  [错误]")
            traceback.print_exc()

    print_summary(all_results)

    # 保存结果
    output_dir = Path(__file__).parent / "cloud_response_samples"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "endianness_all_providers.json"

    # 转换 numpy 为 list
    save_data = {}
    for p, r in all_results.items():
        save_data[p] = {}
        for label, data in r.items():
            save_data[p][label] = {
                "task_ids": data.get("task_ids"),
                "probabilities": data.get("probabilities"),
                "sample_col_means": np.array(data["samples"]).mean(axis=0).tolist() if data["samples"] else [],
            }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
