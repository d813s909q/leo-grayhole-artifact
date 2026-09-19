#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.2 数据聚合: 卫星12 观测/期望/残差 逐时隙 + 诚实节点对照 → JSON"""
import json
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))                     # .../03-manuscript/figures
_ROOT = os.path.dirname(os.path.dirname(_HERE))                         # 项目根 (跨平台)
SRC = os.path.join(_ROOT, "02-experiment", "analysis", "residuals_const12.csv")
OUT = os.path.join(_HERE, "fig2_data.json")

df = pd.read_csv(SRC)
nslots = int(df["slot"].max()) + 1
slots = list(range(nslots))

def agg(node, col):
    s = pd.Series(0.0, index=slots)
    sub = df[df["src"] == node].groupby("slot")[col].sum()
    s.loc[sub.index] = sub.values
    return [float(v) for v in s.values]

data = {
    "slots": slots,
    "n12": {
        "observed": agg(12, "dropped"),
        "expected": agg(12, "b_total"),
        "residual": agg(12, "residual"),
        "handover": agg(12, "handover_events"),
    },
    "honest4": {"observed": agg(4, "dropped"), "residual": agg(4, "residual")},
}
with open(OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print("written", OUT, "nslots=", nslots)
print("n12 observed sum:", sum(data["n12"]["observed"]),
      "residual sum:", sum(data["n12"]["residual"]),
      "honest4 sum:", sum(data["honest4"]["observed"]))
