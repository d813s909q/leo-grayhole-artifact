#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 (M2.2) 多种子检测报告
========================
对 70 个多种子残差 (10 seeds x 7 schedules) 跑与 Part A 完全一致的 CUSUM
检测 (lambda0=0.01, delta=1.0, h=choose_h_chain(0.01,1.0,1000)), 输出:
  - 每种 schedule 的检出率 (TPR) 及其 Wilson 95% 置信区间
  - 报警时隙 / 检测延迟 分布 (均值·中位·最小·最大)
  - 攻击丢包量 均值和标准差
  - 每种子明细 (multiseed_detail.csv) + 汇总 (multiseed_summary.csv)

用法 (Ubuntu-20.04):
    python3 multiseed_report.py
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import (load_residuals, aggregate_nodes, detect_nodes,
                            score_detection, choose_h_chain)

BASE = GH_OUT
MS_DIR = os.path.join(BASE, "multiseed")
OUT = os.path.join(BASE, "comparison")
MALICIOUS = {12}
DELTA = 1.0
LAMBDA0 = 0.01
TARGET_ARL = 1000.0          # 与 Part A 完全一致 (clean 场景)
N_SEEDS = 10
Z95 = 1.96

SPECS = [
    ("const_0p1", "CONSTANT", 0.1),
    ("const_0p2", "CONSTANT", 0.2),
    ("const_0p3", "CONSTANT", 0.3),
    ("const_0p4", "CONSTANT", 0.4),
    ("const_0p5", "CONSTANT", 0.5),
    ("onoff_1p0_20s", "ON_OFF", 1.0),
    ("scan_1p0_20s", "SCAN", 1.0),
]


def wilson_ci(k, n, z=Z95):
    """Wilson score 区间 (95%)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_one(spec, s, h):
    """对单个种子残差跑检测, 返回 detail dict (或 None 若残差缺失)."""
    path = os.path.join(MS_DIR, f"residuals_{spec}_s{s}.csv")
    if not os.path.exists(path):
        return None
    df = load_residuals(path)
    series = aggregate_nodes(df)
    results = detect_nodes(series, LAMBDA0, DELTA, h)
    score = score_detection(results, MALICIOUS)
    detected = 12 in score["true_detected"]
    alarm = results[12]["alarm"] if 12 in results else None
    node12 = series.get(12, np.zeros(1))
    onset = int(np.argmax(node12 > 0)) if (node12 > 0).any() else None
    delay = (alarm - onset) if (alarm is not None and onset is not None) else None
    return dict(spec=spec, seed=s, detected=detected, alarm_slot=alarm,
                attack_onset=onset, delay_slots=delay,
                total_dropped=int(df["dropped"].sum()),
                peak_S=float(results[12]["peak_S"]) if 12 in results else None)


def main():
    os.makedirs(OUT, exist_ok=True)
    h = choose_h_chain(LAMBDA0, DELTA, TARGET_ARL)

    detail_rows = []
    summary_rows = []
    print(f"{'schedule':>15s} {'mode':9s} {'rate':>5s} | "
          f"{'det':>7s} {'TPR[lo,hi]':>18s} | "
          f"{'delay mean/med':>14s} | {'dropped mean±sd':>16s}")
    print("-" * 100)
    for spec, mode, dr in SPECS:
        rows = []
        for s in range(N_SEEDS):
            r = run_one(spec, s, h)
            if r is not None:
                r.update(mode=mode, drop_rate=dr)
                detail_rows.append(r)
                rows.append(r)
        if not rows:
            print(f"{spec:>15s} {mode:9s} {dr:>5.2f} | NO DATA")
            continue
        n_avail = len(rows)
        n_det = sum(1 for r in rows if r["detected"])
        tpr = n_det / n_avail
        lo, hi = wilson_ci(n_det, n_avail)
        delays = [r["delay_slots"] for r in rows if r["delay_slots"] is not None]
        drops = np.array([r["total_dropped"] for r in rows], dtype=float)
        dl_mean = float(np.mean(delays)) if delays else float("nan")
        dl_med = float(np.median(delays)) if delays else float("nan")
        if n_avail < N_SEEDS:
            print(f"  [WARN] {spec}: only {n_avail}/{N_SEEDS} seeds available")
        summary_rows.append(dict(schedule=spec, mode=mode, drop_rate=dr,
                                 n_seeds=n_avail, n_detected=n_det,
                                 tpr=tpr, tpr_lo=lo, tpr_hi=hi,
                                 delay_mean=dl_mean, delay_median=dl_med,
                                 delay_min=(int(min(delays)) if delays else None),
                                 delay_max=(int(max(delays)) if delays else None),
                                 dropped_mean=float(drops.mean()),
                                 dropped_std=float(drops.std(ddof=1) if len(drops) > 1 else 0.0)))
        print(f"{spec:>15s} {mode:9s} {dr:>5.2f} | "
              f"{n_det:>3d}/{n_avail} {tpr:>8.2f}[{lo:.2f},{hi:.2f}] | "
              f"{dl_mean:>6.1f}/{dl_med:>6.1f} | "
              f"{drops.mean():>7.1f}±{drops.std(ddof=1) if len(drops) > 1 else 0.0:>6.1f}")

    pd.DataFrame(detail_rows).to_csv(os.path.join(OUT, "multiseed_detail.csv"),
                                     index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "multiseed_summary.csv"),
                                      index=False)
    with open(os.path.join(OUT, "multiseed_summary.txt"), "w") as f:
        f.write(f"P1 (M2.2) multi-seed detection (n={N_SEEDS} seeds/schedule, "
                f"lambda0={LAMBDA0}, delta={DELTA}, ARL0={TARGET_ARL:.0f}, "
                f"h={h:.2f})\n\n")
        f.write(pd.DataFrame(summary_rows).to_csv(index=False))
    print(f"\nDONE. detail -> {OUT}/multiseed_detail.csv, "
          f"summary -> {OUT}/multiseed_summary.csv")


if __name__ == "__main__":
    main()