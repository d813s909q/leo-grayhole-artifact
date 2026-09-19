#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-13c: 验证 §V-B 'floor is conservative' 声明 (静态 h=3.62, λ0=0.01 反解)
对 h=3.62 扫描 ARL0(λ): 若 min_λ ARL0 < 2373, 则低速率真实残差会超预算, 声明需修正。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import mc_chain_arl

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "comparison")

H = 3.62          # 部署阈值 (λ0=0.01, δ=1.0, ARL0*=1e3 反解, 链 ARL0=2373)
DELTA = 1.0

rows = []
for lam in np.logspace(-3, 2.3, 34):
    arl, _ = mc_chain_arl(round(lam, 5), DELTA, H, 0.05)
    rows.append((round(lam, 5), round(arl, 1)))
df = pd.DataFrame(rows, columns=["lambda", "arl0_at_h3.62"])
i = int(df["arl0_at_h3.62"].idxmin())
print(df.iloc[max(0, i - 3):i + 4].to_string(index=False))
print(f"\nmin ARL0 = {df['arl0_at_h3.62'].min():.0f} at lambda = "
      f"{df.loc[i, 'lambda']}  (deployed chain: 2373)")
df.to_csv(os.path.join(OUT, "r13c_floor_check.csv"), index=False)
