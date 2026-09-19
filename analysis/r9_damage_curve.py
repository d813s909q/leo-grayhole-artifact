#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-9 (DA-C5): damage-vs-δ_a 曲线数据提取
============================================
在现有攻击套件框架内 (TCP Manila-Dalian 场景, 无注入拥塞):
  - δ_a  = 攻击显现后 node-12 聚合的每槽额外丢包率 (pkts/slot)
  - damage = 端到端吞吐损失 % = 1 − bytes_attack / bytes_baseline
  - 检测 = 部署配置 CUSUM (λ0=0.01, δ=1.0, ARL0*=1e3, h=3.62) 是否报警+延迟

档位: CONSTANT p ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}
      低档 (<=0.05) 3 seeds (新), 原档 s0..s9 (已有)

输出: analysis/comparison/r9_damage_curve.csv / .txt
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import run_cusum, choose_h_chain, aggregate_nodes

RUNROOT = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/",
                       "temp/runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "comparison")
MAL = 12
LAMBDA0, DELTA, TARGET_ARL = 0.01, 1.0, 1000.0

SPECS = []   # (label, p, run_name)
for p, tag in [(0.005, "d0p005"), (0.01, "d0p01"), (0.02, "d0p02"),
               (0.05, "d0p05")]:
    for s in range(3):
        SPECS.append((f"const_p{p}_s{s}", p, f"gh_pl_const_{tag}_s{s}"))
for p, tag in [(0.1, "0p1"), (0.2, "0p2"), (0.3, "0p3"), (0.4, "0p4"),
               (0.5, "0p5")]:
    SPECS.append((f"const_p{p}_s0", p, f"gh_pl_const_{tag}"))
    for s in range(1, 10):
        SPECS.append((f"const_p{p}_s{s}", p, f"gh_pl_const_{tag}_s{s}"))


def load_loss(name):
    p = os.path.join(RUNROOT, name, "logs_ns3", "isl_packet_loss.csv")
    d = pd.read_csv(p, header=None,
                    names=["src", "dst", "t_start", "t_end", "rx", "dropped"])
    d["slot"] = (d["t_start"] / 1e9).astype(int)
    return d


def load_bytes(name):
    p = os.path.join(RUNROOT, name, "logs_ns3", "tcp_flows.csv")
    df = pd.read_csv(p, header=None)
    return float(df.iloc[0, 7])       # bytes received


def main():
    h = choose_h_chain(LAMBDA0, DELTA, TARGET_ARL)
    print(f"deployed detector: lambda0={LAMBDA0} delta={DELTA} "
          f"h={h:.2f} ARL0*={TARGET_ARL:.0f}")

    base_bytes = load_bytes("gh_pl_baseline")
    print(f"baseline bytes = {base_bytes:.0f}")

    rows = []
    for label, p, name in SPECS:
        try:
            loss = load_loss(name)
            bts = load_bytes(name)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"[skip] {name}")
            continue
        # node 聚合 dropped (接收视角 src)
        agg = loss.groupby(["src", "slot"])["dropped"].sum().reset_index()
        piv = agg.pivot(index="slot", columns="src", values="dropped").fillna(0)
        if MAL not in piv:
            continue
        d12 = piv[MAL].values
        onset = int(np.argmax(d12 > 0))
        tail = d12[onset:]
        delta_a = float(tail.mean()) if len(tail) else 0.0
        total_dropped = int(loss["dropped"].sum())
        # 检测 (部署配置)
        _, alarm = run_cusum(d12, LAMBDA0, DELTA, h)
        # 诚实节点误报
        fp = 0
        for c in piv.columns:
            if c == MAL:
                continue
            _, a = run_cusum(piv[c].values, LAMBDA0, DELTA, h)
            if a is not None:
                fp += 1
        rows.append(dict(label=label, p=p, run=name,
                         delta_a=round(delta_a, 4),
                         throughput_loss_pct=round(
                             100 * (1 - bts / base_bytes), 2),
                         bytes=int(bts),
                         total_dropped=total_dropped,
                         onset=onset, detected=int(alarm is not None),
                         alarm_slot=alarm,
                         delay=(alarm - onset) if alarm is not None else None,
                         honest_fp=fp))
        r = rows[-1]
        print(f"[{label}] delta_a={r['delta_a']:.3f} "
              f"loss={r['throughput_loss_pct']:.1f}% "
              f"drop={total_dropped} det={r['detected']} "
              f"delay={r['delay']} fp={fp}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "r9_damage_curve.csv"), index=False)

    # 汇总 (按 p)
    g = df.groupby("p").agg(
        n=("label", "count"),
        delta_a_mean=("delta_a", "mean"),
        delta_a_min=("delta_a", "min"), delta_a_max=("delta_a", "max"),
        loss_mean=("throughput_loss_pct", "mean"),
        loss_min=("throughput_loss_pct", "min"),
        loss_max=("throughput_loss_pct", "max"),
        det_rate=("detected", "mean"),
        delay_med=("delay", "median"),
        fp_tot=("honest_fp", "sum")).reset_index()
    print("\n=== summary by p ===")
    print(g.to_string(index=False))
    g.to_csv(os.path.join(OUT, "r9_damage_summary.csv"), index=False)

    with open(os.path.join(OUT, "r9_damage_curve.txt"), "w") as f:
        f.write("R-9: damage vs delta_a (CONSTANT sweep, TCP scenario)\n")
        f.write(f"detector: lambda0={LAMBDA0} delta={DELTA} h={h:.2f} "
                f"ARL0*={TARGET_ARL:.0f}\n")
        f.write(f"baseline bytes = {base_bytes:.0f}\n")
        f.write("delta_a = post-onset mean extra drop rate at node 12 "
                "(pkts/slot); damage = throughput loss %\n\n")
        f.write(g.to_string(index=False))
        f.write("\n\nper-run detail:\n")
        f.write(df.to_string(index=False))
    print("\nR-9 DONE")


if __name__ == "__main__":
    main()
