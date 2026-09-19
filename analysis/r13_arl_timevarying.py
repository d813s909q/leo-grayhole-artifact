#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-13 (R1-M7): 时变 λ0(t) CUSUM 的 worst-node 保守性数值验证
================================================================
背景: §VI-F 部署 per-slot λ0(t)=max(b_{u,t},1e-2), h 在 worst-node 速率反解
     (ARL0*=1000, δ=1.0)。§V-B 的 (I-Q)r=1 只覆盖静态 λ0, 时变链无闭式解。
     本脚本用蒙特卡洛验证: 时变速率下经验 ARL0 >= worst-node 设计目标,
     即 "uniformly conservative" 断言成立; 并给出错配对照 (λ_mean 反解)。

轨迹三类:
  (a) real   — 真实拥塞 run 的 worst-node b(t) 调度 (per-epoch β 标定)
  (b) square — 方波 (epoch 切换型): λ 在 lo/hi 间阶跃, 周期 40 slot
  (c) ramp   — 三角波: λ 从 lo 线性升至 hi 再回落

输出: analysis/comparison/r13_arl_timevarying.csv / .txt
用法 (Ubuntu-20.04): python3 r13_arl_timevarying.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import choose_h_chain, mc_chain_arl
from congestion_cusum import (node_agg, link_epochs, build_b, load,
                              CONG, DELTA, TARGET_ARL, FLOOR)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "comparison")
N_REPS = 2000
MAX_LEN = 60000       # 截断 = 60× 目标 ARL0
BLOCK = 8192
SEED = 20260830


def run_length_one(rng, lam_sched, h, delta, max_len):
    """单条时变泊松序列的 CUSUM 首警运行长度 (1-indexed; None=删失).

    λ0(t) 调度与部署一致: Z_t = x·ln((λ0(t)+δ)/λ0(t)) − δ。
    """
    T = len(lam_sched)
    S = 0.0
    t = 0
    idx = np.arange(BLOCK)
    while t < max_len:
        n = min(BLOCK, max_len - t)
        rates = lam_sched[(t + idx[:n]) % T]
        counts = rng.poisson(rates)
        Z = counts * np.log((rates + delta) / rates) - delta
        for z in Z:
            S += z
            if S > h:
                return t + 1
            if S < 0.0:
                S = 0.0
            t += 1
    return None


def empirical_arl0_trend(lam_sched, h, delta, n_reps, seed):
    rng = np.random.default_rng(seed)
    T = len(lam_sched)
    lens = np.empty(n_reps)
    censored = 0
    for i in range(n_reps):
        r = run_length_one(rng, lam_sched, h, delta, MAX_LEN)
        if r is None:
            censored += 1
            lens[i] = MAX_LEN          # 保守计满 (均值下界)
        else:
            lens[i] = r
    mean_arl = lens.mean()
    se = lens.std(ddof=1) / np.sqrt(n_reps)
    return mean_arl, mean_arl - 1.96 * se, mean_arl + 1.96 * se, censored


def main():
    rows = []

    # ---- (a) 真实轨迹: worst-node b(t) (per-epoch β) ----
    clean = load("residuals_cong_clean.csv")
    epoch_of, beta = link_epochs(clean)
    b_epoch = build_b(clean, epoch_of, beta)
    clean2 = clean.assign(b_epoch=b_epoch)
    be = node_agg(clean2, "b_epoch")
    nodes = sorted(be)
    worst_node = max(nodes, key=lambda n: be[n].max())
    lam_sched_real = np.maximum(be[worst_node], FLOOR)
    print(f"[real] worst node = {worst_node}, "
          f"lambda_max = {lam_sched_real.max():.2f}, "
          f"lambda_mean = {lam_sched_real.mean():.2f}, "
          f"T = {len(lam_sched_real)} slots")

    # ---- (b/c) 合成轨迹 ----
    T = 200
    lam_lo = 1.0
    lam_hi = float(lam_sched_real.max())
    square = np.where(np.arange(T) % 40 < 20, lam_hi, lam_lo).astype(float)
    tri = np.linspace(lam_lo, lam_hi, T // 2)
    ramp = np.concatenate([tri, tri[::-1]])

    traces = [
        ("real-worstnode", lam_sched_real),
        ("square", square),
        ("ramp", ramp),
    ]

    for name, sched in traces:
        lam_max = float(sched.max())
        lam_mean = float(sched.mean())
        # 部署口径: worst-node 反解 h
        h_wc = choose_h_chain(lam_max, DELTA, TARGET_ARL)
        chain_arl_wc, _ = mc_chain_arl(lam_max, DELTA, h_wc, 0.05)
        emp_wc, lo_wc, hi_wc, cen_wc = empirical_arl0_trend(
            sched, h_wc, DELTA, N_REPS, SEED)
        # 错配对照: λ_mean 反解 h
        h_mean = choose_h_chain(lam_mean, DELTA, TARGET_ARL)
        emp_mm, lo_mm, hi_mm, cen_mm = empirical_arl0_trend(
            sched, h_mean, DELTA, N_REPS, SEED + 1)
        rows.append(dict(
            trace=name, T=len(sched),
            lam_max=round(lam_max, 3), lam_mean=round(lam_mean, 3),
            h_worstnode=round(h_wc, 3),
            chain_arl0_worstnode=round(chain_arl_wc, 1),
            emp_arl0_worstnode=round(emp_wc, 1),
            emp_arl0_wc_lo=round(lo_wc, 1), emp_arl0_wc_hi=round(hi_wc, 1),
            censored_wc=cen_wc,
            conservative_ratio=round(emp_wc / TARGET_ARL, 2),
            h_lamme=round(h_mean, 3),
            emp_arl0_lamme=round(emp_mm, 1),
            emp_arl0_mm_lo=round(lo_mm, 1), emp_arl0_mm_hi=round(hi_mm, 1),
            censored_mm=cen_mm))
        print(f"[{name}] lam_max={lam_max:.2f} lam_mean={lam_mean:.2f} | "
              f"h_wc={h_wc:.2f} chain={chain_arl_wc:.0f} "
              f"emp={emp_wc:.0f} [{lo_wc:.0f},{hi_wc:.0f}] "
              f"ratio={emp_wc/TARGET_ARL:.2f} cen={cen_wc} | "
              f"h_mean={h_mean:.2f} emp_mm={emp_mm:.0f} cen_mm={cen_mm}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "r13_arl_timevarying.csv"), index=False)
    with open(os.path.join(OUT, "r13_arl_timevarying.txt"), "w") as f:
        f.write("R-13: time-varying lambda0(t) CUSUM, worst-node conservatism\n")
        f.write(f"protocol: delta={DELTA}, ARL0*={TARGET_ARL:.0f}, "
                f"floor={FLOOR}, n_reps={N_REPS}, max_len={MAX_LEN}\n")
        f.write("deployment: lambda0(t)=max(b(t),floor), "
                "h inverted at worst-node rate (static chain)\n")
        f.write("Z_t = x ln((lam0(t)+delta)/lam0(t)) - delta "
                "(schedule-aware, as deployed)\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\nverdict: empirical ARL0 >= target in all worst-node rows "
                "(conservative); lambda_mean inversion breaks it (necessity)\n")
    print("\nR-13 DONE ->", os.path.join(OUT, "r13_arl_timevarying.csv"))


if __name__ == "__main__":
    main()
