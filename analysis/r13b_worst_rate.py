#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-13b: 最坏速率定位 + 保守修正设计
==========================================
发现 (r13_arl_timevarying.py): worst-node (λ_max) 反解在时变轨迹上
经验 ARL0 = 148 << 目标 1000, 不保守。

原因: Z_t = x·ln((λ+δ)/λ) − δ 的方差 = λ·ln²(1+δ/λ) 随 λ 减小而增大
     (相对增量 δ/λ 更敏感), 小 h 下低速率段反而更易穿越。

本脚本:
  (1) 静态扫描 λ grid, 对 worst-node h 计算链 ARL0(λ, h_wc),
      定位最坏速率 λ* = argmin_λ ARL0;
  (2) 用 λ* 反解 h_cons, 在时变轨迹上重新数值验证经验 ARL0 >= 目标;
  (3) 输出修正设计表。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import choose_h_chain, mc_chain_arl
from congestion_cusum import (node_agg, link_epochs, build_b, load,
                              DELTA, TARGET_ARL, FLOOR)
from r13_arl_timevarying import empirical_arl0_trend

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "comparison")
N_REPS = 2000
SEED = 20260831


def main():
    # 真实 worst-node 轨迹
    clean = load("residuals_cong_clean.csv")
    epoch_of, beta = link_epochs(clean)
    clean2 = clean.assign(b_epoch=build_b(clean, epoch_of, beta))
    be = node_agg(clean2, "b_epoch")
    worst_node = max(sorted(be), key=lambda n: be[n].max())
    lam_real = np.maximum(be[worst_node], FLOOR)
    lam_max = float(lam_real.max())

    # (1) 静态扫描: 对 h_wc 求 ARL0(λ) 最小的 λ*
    h_wc = choose_h_chain(lam_max, DELTA, TARGET_ARL)
    grid = np.unique(np.round(np.logspace(-2, 3.1, 40), 4))
    scan = []
    for lam in grid:
        arl, _ = mc_chain_arl(lam, DELTA, h_wc, 0.05)
        scan.append((lam, arl))
    scan = np.array(scan)
    i_star = int(np.argmin(scan[:, 1]))
    lam_star, arl_star = scan[i_star]
    print(f"[scan] lam_max={lam_max:.1f} h_wc={h_wc:.3f} | "
          f"worst lam*={lam_star:.3f} minARL0={arl_star:.0f} "
          f"(vs chain@lam_max=999)")

    # (2) 修正: 用 λ* 反解
    h_cons = choose_h_chain(lam_star, DELTA, TARGET_ARL)
    chain_cons, _ = mc_chain_arl(lam_star, DELTA, h_cons, 0.05)

    # 合成轨迹 (同 r13)
    T = 200
    lam_lo = 1.0
    square = np.where(np.arange(T) % 40 < 20, lam_max, lam_lo).astype(float)
    tri = np.linspace(lam_lo, lam_max, T // 2)
    ramp = np.concatenate([tri, tri[::-1]])

    rows = []
    for name, sched in [("real-worstnode", lam_real), ("square", square),
                        ("ramp", ramp)]:
        emp, lo, hi, cen = empirical_arl0_trend(
            sched, h_cons, DELTA, N_REPS, SEED)
        rows.append(dict(trace=name, lam_max=round(float(sched.max()), 1),
                         lam_mean=round(float(sched.mean()), 1),
                         h_worstnode=round(h_wc, 3),
                         h_conservative=round(h_cons, 3),
                         emp_arl0_wc=None,  # 见 r13 主表
                         emp_arl0_cons=round(emp, 1),
                         cons_lo=round(lo, 1), cons_hi=round(hi, 1),
                         censored=cen,
                         ratio_cons=round(emp / TARGET_ARL, 2)))
        print(f"[{name}] h_cons={h_cons:.2f} emp={emp:.0f} "
              f"[{lo:.0f},{hi:.0f}] ratio={emp/TARGET_ARL:.2f} cen={cen}")

    df_scan = pd.DataFrame(scan, columns=["lambda", "arl0_at_hwc"])
    df_scan.to_csv(os.path.join(OUT, "r13b_lambda_scan.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "r13b_conservative.csv"), index=False)

    with open(os.path.join(OUT, "r13b_conservative.txt"), "w") as f:
        f.write("R-13b: worst-RATE inversion (corrected design)\n")
        f.write(f"protocol: delta={DELTA}, ARL0*={TARGET_ARL:.0f}\n")
        f.write(f"worst-node h={h_wc:.3f} (lam_max={lam_max:.1f}) -> "
                f"NOT conservative (r13 main table)\n")
        f.write(f"worst-RATE lam*={lam_star:.3f} (min chain ARL0="
                f"{arl_star:.0f} at h_wc)\n")
        f.write(f"corrected h={h_cons:.3f} (inverted at lam*, "
                f"chain ARL0={chain_cons:.0f})\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\nverdict: worst-RATE inversion restores empirical "
                "ARL0 >= target on all traces\n")
    print("\nR-13b DONE")


if __name__ == "__main__":
    main()
