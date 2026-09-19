#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2.4 (#11) tie-break 稳健性: 攻击者换 node 11 (与 12 并列 path persistence 61.5%)
=============================================================================
对 gh_pl_const11 (CONSTANT 0.5, 同种子) 检验: 检出/归因/延迟/误报是否与 node 12
结论一致。检测协议与 Part A 完全一致 (lambda0=0.01, delta=1.0,
h=choose_h_chain(0.01,1.0,1000))。附带 naive/trust 基线 (同 Part B 标定)。

用法 (Ubuntu-20.04): python3 tiebreak_check.py
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import (load_residuals, aggregate_nodes, detect_nodes,
                            score_detection, choose_h_chain)
from baselines import aggregate_node_field, naive_threshold, trust_detect

BASE = GH_OUT
MAL_NODE = 11
LAMBDA0 = 0.01
DELTA = 1.0
TARGET_ARL = 1000.0


def main():
    h = choose_h_chain(LAMBDA0, DELTA, TARGET_ARL)
    df = load_residuals(os.path.join(BASE, "attack_runs", "residuals_const11.csv"))
    clean = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))

    series = aggregate_nodes(df)
    results = detect_nodes(series, LAMBDA0, DELTA, h)
    score = score_detection(results, {MAL_NODE})

    detected = MAL_NODE in score["true_detected"]
    alarm = results[MAL_NODE]["alarm"] if MAL_NODE in results else None
    n11 = series.get(MAL_NODE, np.zeros(1))
    onset = int(np.argmax(n11 > 0)) if (n11 > 0).any() else None
    delay = (alarm - onset) if (alarm is not None and onset is not None) else None
    fp_nodes = sorted(score["false_positives"])

    nodes = sorted(aggregate_node_field(df, "dropped"))
    honest = [n for n in nodes if n != MAL_NODE]
    drop = aggregate_node_field(df, "dropped")
    rx = aggregate_node_field(df, "rx")
    drop_c = aggregate_node_field(clean, "dropped")
    tau = max(float(drop_c[n].sum()) for n in honest) + 0.5
    nv = naive_threshold(drop, tau)
    tv = trust_detect(drop, rx, 10, 0.99, 50)

    total_dropped = int(df["dropped"].sum())
    print(f"node-11 CONSTANT 0.5 run: total dropped = {total_dropped}")
    print(f"h = {h:.2f}  |  CUSUM: detected={detected}  alarm_slot={alarm}  "
          f"onset={onset}  delay={delay}")
    print(f"  false alarms (honest nodes): {fp_nodes if fp_nodes else 'NONE'}")
    print(f"  peak_S(node11) = "
          f"{results[MAL_NODE]['peak_S'] if MAL_NODE in results else 'n/a'}")
    print(f"naive: flagged node11={nv.get(MAL_NODE)}  honest_flagged="
          f"{sorted(n for n in honest if nv.get(n))}")
    print(f"trust: flagged node11={tv.get(MAL_NODE)}  honest_flagged="
          f"{sorted(n for n in honest if tv.get(n))}")
    print(f"rx total node11 = {int(rx[MAL_NODE].sum())}, "
          f"dropped total node11 = {int(drop[MAL_NODE].sum())}")


if __name__ == "__main__":
    main()
