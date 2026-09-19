#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m1.4: 攻击损失聚集窗口敏感性 (3s/5s/10s)

对 CONSTANT 0.5 攻击 run 的节点-12 残差, 统计攻击损失落在最近路由事件
(fstate 变化, handover_events.csv) ±w 秒内的比例, w ∈ {3,5,10,15}。
主场景 ISL 全程在阈值内 (无断链), 事件集合即 fstate 变化全集。

用法 (Ubuntu-20.04): python3 window_sensitivity.py
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import os

import numpy as np
import pandas as pd

BASE = GH_OUT
ATT = os.path.join(BASE, "residuals_const12.csv")
EVENTS = os.path.join(BASE, "handover_events.csv")
NODE = 12
WINDOWS = [3, 5, 10, 15]


def main():
    df = pd.read_csv(ATT)
    ev = pd.read_csv(EVENTS)
    ev_s = ev["t_ns"].values.astype(float) / 1e9

    sub = df[df["src"] == NODE].groupby("slot")["residual"].sum()
    slots = sub.index.values.astype(float)
    wts = np.maximum(sub.values, 0.0)  # 攻击残差 >= 0

    total = wts.sum()
    dist = np.array([np.min(np.abs(ev_s - t)) for t in slots])
    print(f"attack residual packets (node {NODE}) = {total:.0f} over "
          f"{int((wts > 0).sum())} slots; {len(ev_s)} routing events "
          f"(t = {sorted(set(ev_s.round(1)))})")
    for w in WINDOWS:
        frac = wts[dist <= w].sum() / total
        print(f"  within ±{w:2d} s of nearest event: {frac*100:.1f}% "
              f"({wts[dist <= w].sum():.0f}/{total:.0f})")


if __name__ == "__main__":
    main()
