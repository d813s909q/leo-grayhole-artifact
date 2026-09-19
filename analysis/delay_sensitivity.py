#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 (M1.2) 小信号检测延迟敏感性
==============================
回应 mock review M1.2: 「zero-slot latency 有定义性成分 — 强攻击一步清阈值并不
意外; 缺小信号检测延迟数据 (均值/95 分位) 与 ARL₁ 理论值的对照」。

实验 A (合成 Poisson, 理论对照):
    λ0 (自然拥塞) × δ_a (真实攻击增量) 扫描, CUSUM 设计 δ=1.0 (与部署一致),
    h 由 ARL₀*=1e5 反解。攻击自 t=0 起全程存在 (与 ARL₁ 定义严格对齐),
    蒙特卡洛测延迟分布 (均值/中位/P95/200 时隙检出率), 与马尔可夫链
    ARL₁ (失配速率解, actual_delta=δ_a) 对照。

实验 B (真实攻击波形, 二项稀疏化):
    对 4 个真实 schedule 的节点-12 攻击残差序列做二项稀疏 (保留率 α,
    即等价更低速率的 CONSTANT 攻击), 叠加 Poisson(λ_cong) 自然拥塞,
    测延迟分布。把「zero-delay」定位为强攻击特例。

用法: python3 delay_sensitivity.py [--smoke]
      GH_SHOW=1 → 纯交互窗口 (不落盘, 保存用工具栏 Save); 无 GH_SHOW → 无头出工作图
输出: comparison/delay_theory_mc.csv, delay_real.csv,
      delay_sensitivity.png (仅无头模式)
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import (load_residuals, aggregate_nodes, choose_h_chain,
                            mc_chain_arl, run_cusum)

_HERE = os.path.dirname(os.path.abspath(__file__))                    # .../02-experiment/src
BASE = os.path.join(os.path.dirname(_HERE), "analysis")               # 跨平台 (WSL/Windows)
ATTACK_DIR = os.path.join(BASE, "attack_runs")
OUT = os.path.join(BASE, "comparison")

DESIGN_DELTA = 1.0        # 部署 CUSUM 的设计增量 (与论文一致)
TARGET_ARL0 = 1e5
GRID = 0.05               # 链解网格 (与 choose_h_chain 默认一致)

LAMBDAS = [0.05, 0.1, 0.5, 1.0, 2.0]        # λ0 / λ_cong 扫描
DELTAS_A = [0.05, 0.1, 0.2, 0.5, 1.0]       # 真实攻击增量扫描
N_TRIALS = 200
HORIZON = 1000            # 实验 A 时隙上限 (超出记为截尾)


# ----------------------------------------------------------------------
# 实验 A: 合成 Poisson — 延迟分布 vs ARL₁ 理论
# ----------------------------------------------------------------------
def mc_delay_h1(lambda0, delta_a, h, n_trials, horizon, seed):
    """H1 全程攻击 (自 t=0, S_0=0), 返回每个 trial 的报警时隙 (None=截尾)."""
    rng = np.random.default_rng(seed)
    logr = math.log((lambda0 + DESIGN_DELTA) / lambda0)
    drift = DESIGN_DELTA
    delays = []
    for _ in range(n_trials):
        x = rng.poisson(lambda0 + delta_a, size=horizon)
        S = 0.0
        alarm = None
        for t in range(horizon):
            S = max(0.0, S + x[t] * logr - drift)
            if S > h:
                alarm = t
                break
        delays.append(alarm)
    return delays


def experiment_a(smoke=False):
    n_trials = 20 if smoke else N_TRIALS
    horizon = 200 if smoke else HORIZON
    rows = []
    for lam in LAMBDAS:
        h = choose_h_chain(lam, DESIGN_DELTA, TARGET_ARL0, grid_step=GRID)
        for da in DELTAS_A:
            arl1, _ = mc_chain_arl(lam, DESIGN_DELTA, h, grid_step=GRID,
                                   x_max=80, attacking=True, actual_delta=da)
            p200_theory = (1.0 - math.exp(-200.0 / arl1)
                           if math.isfinite(arl1) else 0.0)
            delays = mc_delay_h1(lam, da, h, n_trials, horizon,
                                 seed=5000 + int(lam * 100) * 10 + int(da * 100))
            hit = [d for d in delays if d is not None]
            det200 = sum(1 for d in hit if d < 200) / n_trials
            det_h = len(hit) / n_trials
            d_mean = float(np.mean(hit)) if hit else None
            d_med = float(np.median(hit)) if hit else None
            d_p95 = float(np.percentile(hit, 95)) if hit else None
            rows.append(dict(
                lambda0=lam, delta_a=da, h=round(h, 2),
                arl1_chain=(round(arl1, 1) if math.isfinite(arl1) else None),
                p_det200_theory=round(p200_theory, 3),
                det200_mc=round(det200, 3), det_horizon_mc=round(det_h, 3),
                delay_mean=(round(d_mean, 1) if d_mean is not None else None),
                delay_median=(round(d_med, 1) if d_med is not None else None),
                delay_p95=(round(d_p95, 1) if d_p95 is not None else None),
                n_alarms=len(hit), n_trials=n_trials))
            r = rows[-1]
            print(f"λ0={lam:>5.2f} δ_a={da:>4.2f} h={h:>5.2f} | "
                  f"ARL₁={r['arl1_chain']!s:>8s} P200_th={r['p_det200_theory']:.3f} "
                  f"P200_mc={det200:.3f} | "
                  f"delay mean/med/p95 = {r['delay_mean']!s:>7s}/"
                  f"{r['delay_median']!s:>7s}/{r['delay_p95']!s:>7s} "
                  f"(n_alarm={len(hit)})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "delay_theory_mc.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# 实验 B: 真实攻击波形 × 二项稀疏 × 自然拥塞
# ----------------------------------------------------------------------
SCHEDULES = [
    ("const_0p1", os.path.join(ATTACK_DIR, "residuals_const_0p1.csv")),
    ("const_0p5", os.path.join(BASE, "residuals_const12.csv")),
    ("onoff_1p0_20s", os.path.join(ATTACK_DIR, "residuals_onoff_1p0_20s.csv")),
    ("scan_1p0_20s", os.path.join(ATTACK_DIR, "residuals_scan_1p0_20s.csv")),
]
ALPHAS = [0.1, 0.25, 0.5, 1.0]             # 攻击保留率 (等价低速率攻击)
LAM_CONG_B = [0.1, 0.5, 1.0, 2.0]


def experiment_b(smoke=False):
    n_trials = 20 if smoke else N_TRIALS
    rows = []
    for sched, path in SCHEDULES:
        df_res = load_residuals(path)
        series = aggregate_nodes(df_res)
        attack = np.rint(series.get(12, np.zeros(1))).astype(int)
        nslots = len(attack)
        onset = int(np.argmax(attack > 0)) if (attack > 0).any() else None
        for lam in LAM_CONG_B:
            h = choose_h_chain(lam, DESIGN_DELTA, TARGET_ARL0, grid_step=GRID)
            for alpha in ALPHAS:
                rng = np.random.default_rng(
                    9000 + int(lam * 100) * 10 + int(alpha * 100))
                delays = []
                for _ in range(n_trials):
                    a = rng.binomial(attack, alpha) if alpha < 1.0 else attack
                    x = rng.poisson(lam, size=nslots) + a
                    _, alarm = run_cusum(x.astype(float), lam,
                                         DESIGN_DELTA, h)
                    if alarm is not None:
                        delays.append(alarm - onset)
                hit = [d for d in delays if d >= 0]
                det = len(hit) / n_trials
                rate = (alpha * float(attack.sum())
                        / max(1, nslots - onset)) if onset is not None else 0.0
                rows.append(dict(
                    schedule=sched, alpha=alpha, lam_cong=lam, h=round(h, 2),
                    attack_onset=onset, mean_attack_rate=round(rate, 3),
                    det_rate=round(det, 3), n_trials=n_trials,
                    delay_mean=(round(float(np.mean(hit)), 1) if hit else None),
                    delay_median=(round(float(np.median(hit)), 1) if hit else None),
                    delay_p95=(round(float(np.percentile(hit, 95)), 1) if hit else None),
                    n_alarms=len(hit)))
                r = rows[-1]
                print(f"{sched:>13s} α={alpha:>4.2f} λ={lam:>4.2f} h={h:>5.2f} | "
                      f"r̄_a={r['mean_attack_rate']:>5.3f} det={det:.3f} | "
                      f"delay mean/med/p95 = {r['delay_mean']!s:>7s}/"
                      f"{r['delay_median']!s:>7s}/{r['delay_p95']!s:>7s}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "delay_real.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# 工作图 (figure_studio 后续做 IEEE 终稿版)
# ----------------------------------------------------------------------
def make_figure(df_a):
    try:
        import matplotlib
        interactive = bool(os.environ.get("GH_SHOW")) and (
            os.name == "nt" or bool(os.environ.get("DISPLAY")))
        matplotlib.use("TkAgg" if interactive else "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 不可用, 跳过 PNG)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) λ0=1.0: 延迟 vs δ_a (MC mean + P95 带 + 理论 ARL₁ 曲线)
    ax = axes[0]
    sub = df_a[df_a["lambda0"] == 1.0]
    das, dm, dp, th = [], [], [], []
    for da in sorted(sub["delta_a"].unique()):
        r = sub[sub["delta_a"] == da].iloc[0]
        if r["delay_mean"] is not None and r["arl1_chain"] is not None:
            das.append(da); dm.append(r["delay_mean"]); dp.append(r["delay_p95"])
            th.append(r["arl1_chain"])
    ax.plot(das, dm, "o-", label="MC mean delay")
    ax.fill_between(das, dm, dp, alpha=0.2, label="MC P95")
    ax.plot(das, th, "s--", label="Theory ARL$_1$ (chain)")
    ax.set_xlabel("actual attack increment $\\delta_a$ (pkts/slot)")
    ax.set_ylabel("detection delay (slots)")
    ax.set_title("(a) $\\lambda_0$=1.0, design $\\delta$=1.0")
    ax.legend(loc="best", framealpha=0.92).set_draggable(True); ax.grid(alpha=0.3)

    # (b) δ_a=0.5: 延迟 vs λ0
    ax = axes[1]
    sub = df_a[df_a["delta_a"] == 0.5]
    ls, dm, dp, th = [], [], [], []
    for lam in sorted(sub["lambda0"].unique()):
        r = sub[sub["lambda0"] == lam].iloc[0]
        if r["delay_mean"] is not None and r["arl1_chain"] is not None:
            ls.append(lam); dm.append(r["delay_mean"]); dp.append(r["delay_p95"])
            th.append(r["arl1_chain"])
    ax.plot(ls, dm, "o-", label="MC mean delay")
    ax.fill_between(ls, dm, dp, alpha=0.2, label="MC P95")
    ax.plot(ls, th, "s--", label="Theory ARL$_1$ (chain)")
    ax.set_xlabel("natural congestion rate $\\lambda_0$ (pkts/slot)")
    ax.set_ylabel("detection delay (slots)")
    ax.set_title("(b) $\\delta_a$=0.5, design $\\delta$=1.0")
    ax.legend(loc="best", framealpha=0.92).set_draggable(True); ax.grid(alpha=0.3)

    fig.tight_layout()
    if os.environ.get("GH_SHOW"):   # 交互: 关窗后自动 600dpi 导出论文图 Figure_5
        print("[interactive] 图例可拖曳/工具栏缩放; 关闭窗口即自动导出 Figure_5.png @600dpi")
        plt.show()
        _out = os.path.normpath(os.path.join(
            _HERE, "..", "..", "03-manuscript", "latex", "figures", "Figure_5.png"))
        fig.savefig(_out, dpi=600, bbox_inches="tight")
        print("[exported] Figure_5 @600dpi ->", _out)
        return
    fig.savefig(os.path.join(OUT, "delay_sensitivity.png"), dpi=150,
                bbox_inches="tight")
    print("figure ->", os.path.join(OUT, "delay_sensitivity.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="快速自检 (少 trial)")
    ap.add_argument("--cached", action="store_true",
                    help="直接读已算好的 CSV 出图 (默认: CSV 存在即自动走缓存)")
    ap.add_argument("--recompute", action="store_true",
                    help="强制重算 A/B 实验 (忽略缓存)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    csv_a = os.path.join(OUT, "delay_theory_mc.csv")
    csv_b = os.path.join(OUT, "delay_real.csv")
    use_cache = (not args.smoke and not args.recompute
                 and os.path.exists(csv_a) and os.path.exists(csv_b))

    if use_cache:
        df_a = pd.read_csv(csv_a)
        print(f"[cached] 跳过重算, 直接出图: {csv_a} ({len(df_a)} rows)")
    else:
        print("=== 实验 A: 合成 Poisson 延迟 vs ARL₁ 理论 (设计 δ=1.0) ===")
        df_a = experiment_a(args.smoke)
        print("\n=== 实验 B: 真实波形二项稀疏 × 自然拥塞 ===")
        experiment_b(args.smoke)
    if not args.smoke:
        make_figure(df_a)
    print("\nDONE.")


if __name__ == "__main__":
    main()