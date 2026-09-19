#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 (M1.4) 最小可检测 δ 曲线
============================
回应 mock review M1.4: 「全文只用 δ=1.0; 真实攻击者会选择 δ 尽量小。最小可
检测 δ (给定 ARL₀*=1e5、可接受 ARL₁≤T) 应有一条曲线或至少一个表格」。

对每个 λ0 与延迟预算 T ∈ {10,25,50,100,200}, 求满足 ARL₁(δ_a)=T 的最小真实
攻击增量 δ_min:

  - 部署曲线 (deployed): CUSUM 设计 δ=1.0 (论文部署配置), h=choose_h_chain
    (λ0,1.0,1e5); ARL₁ 用失配链解 (actual_delta=δ_a)。这是实际部署检测器
    的可检测性边界。
  - 匹配曲线 (matched): 每个候选 δ_a 重新设计 CUSUM (h 按该 δ_a 反解),
    经典 CUSUM 最优性下的下界 (信息论式参照)。

同时给出零漂移隐身边界 δ* = δ_design/ln((λ0+δ_design)/λ0) − λ0
(低于 δ* 时 CUSUM 统计量期望增量为负, 期望上不可检测, 与 05 文档结论衔接),
并在若干点做蒙特卡洛交叉验证 (论文风格: 链解 vs MC 偏差 %)。

用法: python3 delta_curve.py [--smoke]
      GH_SHOW=1 → 纯交互窗口 (不落盘, 保存用工具栏 Save); 无 GH_SHOW → 无头出工作图
输出: comparison/delta_min_deployed.csv, delta_min_matched.csv,
      delta_min_mc_check.csv, delta_min.png (仅无头模式)
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import choose_h_chain, mc_chain_arl

_HERE = os.path.dirname(os.path.abspath(__file__))                    # .../02-experiment/src
BASE = os.path.join(os.path.dirname(_HERE), "analysis")               # 跨平台 (WSL/Windows)
OUT = os.path.join(BASE, "comparison")

DESIGN_DELTA = 1.0
TARGET_ARL0 = 1e5
GRID = 0.05                 # 链解网格 (与论文其他链解一致)
LAMBDAS = [0.05, 0.1, 0.5, 1.0, 2.0]
BUDGETS = [10, 25, 50, 100, 200]


def delta_star(lam, design=DESIGN_DELTA):
    """零漂移边界: E[Z | rate λ0+δ_a] = 0 ⇒ δ_a = design/logr − λ0."""
    logr = math.log((lam + design) / lam)
    return design / logr - lam


def bisect(f, lo, hi, iters=30):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:      # f = ARL₁(δ) − T, 随 δ 单调下降
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def deployed_delta_min(lam, T):
    """部署设计 (δ=1.0): 解 ARL₁_mismatch(δ_a) = T.

    注: 二分下界须低于零漂移边界 δ* — 在 δ* 处反射随机游走的期望穿越时间
    ≈ h²/σ² (仅几十时隙), 故 δ_min(T) 在中等 T 下可以落在 δ* 之下
    (δ* 以下 ARL₁ 才随负漂移指数增长)。δ_a→0 时 ARL₁→ARL₀*=1e5 > T,
    f 在小区间为正, 单调下降保证二分安全。"""
    h = choose_h_chain(lam, DESIGN_DELTA, TARGET_ARL0, grid_step=GRID,
                       max_iters=30)
    dstar = delta_star(lam)

    def f(da):
        arl, _ = mc_chain_arl(lam, DESIGN_DELTA, h, grid_step=GRID, x_max=80,
                              attacking=True, actual_delta=da)
        return arl - T
    da_min = bisect(f, 1e-3, 30.0)
    return da_min, h, dstar


def matched_delta_min(lam, T):
    """匹配设计: 每个候选 δ 重解 h(λ0,δ,1e5), ARL₁(δ)=T."""
    def f(da):
        h = choose_h_chain(lam, da, TARGET_ARL0, grid_step=GRID, max_iters=25)
        arl, _ = mc_chain_arl(lam, da, h, grid_step=GRID, x_max=80,
                              attacking=True)
        return arl - T
    return bisect(f, 0.02, 10.0)


def mc_arl1(lam, delta_design, delta_actual, h, n_reps=2000,
            max_len=20000, seed=7):
    """失配 MC 验证: 过程率 λ0+δ_actual, 增量按设计 δ_design."""
    rng = np.random.default_rng(seed)
    logr = math.log((lam + delta_design) / lam)
    drift = delta_design
    run_lengths = []
    for _ in range(n_reps):
        x = rng.poisson(lam + delta_actual, size=max_len)
        S = 0.0
        rl = max_len
        for t in range(max_len):
            S = max(0.0, S + x[t] * logr - drift)
            if S > h:
                rl = t
                break
        run_lengths.append(rl)
    return float(np.mean(run_lengths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--recompute", action="store_true",
                    help="强制重算 (默认: CSV 缓存存在即直接出图)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    csv_dep = os.path.join(OUT, "delta_min_deployed.csv")
    csv_mat = os.path.join(OUT, "delta_min_matched.csv")
    if not args.smoke and not args.recompute and \
            os.path.exists(csv_dep) and os.path.exists(csv_mat):
        df_dep = pd.read_csv(csv_dep)
        df_mat = pd.read_csv(csv_mat)
        print(f"[cached] 跳过重算, 直接出图: {csv_dep} / {csv_mat}")
        make_figure(df_dep, df_mat)
        print("DONE.")
        return

    budgets = [100] if args.smoke else BUDGETS
    lambdas = [0.5, 1.0] if args.smoke else LAMBDAS

    rows_dep, rows_mat = [], []
    print("=== 部署设计 (δ=1.0, ARL₀*=1e5): δ_min(T) ===")
    print(f"{'λ0':>6s} {'δ*':>6s} {'h':>6s} | " + " ".join(f"T={t:<5d}" for t in budgets))
    for lam in lambdas:
        dstar = delta_star(lam)
        h = choose_h_chain(lam, DESIGN_DELTA, TARGET_ARL0, grid_step=GRID)
        line = f"{lam:>6.2f} {dstar:>6.3f} {h:>6.2f} | "
        for T in budgets:
            da, _, _ = deployed_delta_min(lam, T)
            line += f"{da:<6.3f} "
            rows_dep.append(dict(lambda0=lam, T_budget=T, delta_min=round(da, 4),
                                 h_deployed=round(h, 2), delta_star=round(dstar, 3)))
        print(line)

    print("\n=== 匹配设计 (每个 δ 重新设计, ARL₀*=1e5): δ_min(T) ===")
    print(f"{'λ0':>6s} | " + " ".join(f"T={t:<5d}" for t in budgets))
    for lam in lambdas:
        line = f"{lam:>6.2f} | "
        for T in budgets:
            da = matched_delta_min(lam, T)
            line += f"{da:<6.3f} "
            rows_mat.append(dict(lambda0=lam, T_budget=T, delta_min=round(da, 4)))
        print(line)

    df_dep = pd.DataFrame(rows_dep)
    df_mat = pd.DataFrame(rows_mat)
    df_dep.to_csv(os.path.join(OUT, "delta_min_deployed.csv"), index=False)
    df_mat.to_csv(os.path.join(OUT, "delta_min_matched.csv"), index=False)

    # ---- MC 交叉验证 (部署设计, 论文风格偏差%) ----
    checks = [(0.5, 100), (0.5, 25), (1.0, 100), (2.0, 200)]
    rows_mc = []
    print("\n=== MC 交叉验证 (部署设计) ===")
    print(f"{'λ0':>6s} {'T':>5s} {'δ_min':>7s} {'ARL₁_chain':>11s} {'ARL₁_MC':>9s} {'偏差%':>7s}")
    for lam, T in checks:
        da, h, _ = deployed_delta_min(lam, T)
        arl_ch, _ = mc_chain_arl(lam, DESIGN_DELTA, h, grid_step=GRID, x_max=80,
                                 attacking=True, actual_delta=da)
        arl_mc = mc_arl1(lam, DESIGN_DELTA, da, h,
                         n_reps=200 if args.smoke else 2000,
                         max_len=4000 if args.smoke else 20000, seed=7)
        dev = 100.0 * (arl_mc - arl_ch) / arl_ch
        rows_mc.append(dict(lambda0=lam, T_budget=T, delta_min=round(da, 4),
                            arl1_chain=round(arl_ch, 1), arl1_mc=round(arl_mc, 1),
                            dev_pct=round(dev, 1)))
        print(f"{lam:>6.2f} {T:>5d} {da:>7.3f} {arl_ch:>11.1f} {arl_mc:>9.1f} {dev:>6.1f}%")
    pd.DataFrame(rows_mc).to_csv(os.path.join(OUT, "delta_min_mc_check.csv"),
                                 index=False)

    # ---- 工作图 ----
    if not args.smoke:
        make_figure(df_dep, df_mat)

    print("\nDONE.")


def make_figure(df_dep, df_mat):
    """fig6: δ_min(T) 曲线 (部署 vs 匹配设计); GH_SHOW=1 时交互, 关窗即保存"""
    try:
        import matplotlib
        interactive = bool(os.environ.get("GH_SHOW")) and (
            os.name == "nt" or bool(os.environ.get("DISPLAY")))
        matplotlib.use("TkAgg" if interactive else "Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        markers = {"0.05": "o", "0.1": "s", "0.5": "^", "1.0": "D", "2.0": "v"}
        for lam in LAMBDAS:
            sd = df_dep[df_dep["lambda0"] == lam].sort_values("T_budget")
            sm = df_mat[df_mat["lambda0"] == lam].sort_values("T_budget")
            mk = markers.get(str(lam), "o")
            ax.plot(sd["T_budget"], sd["delta_min"], mk + "-", color="C%d" % LAMBDAS.index(lam),
                    label=f"$\\lambda_0$={lam} deployed")
            ax.plot(sm["T_budget"], sm["delta_min"], mk + "--", color="C%d" % LAMBDAS.index(lam),
                    alpha=0.5, label=f"$\\lambda_0$={lam} matched")
            ax.axhline(delta_star(lam), color="C%d" % LAMBDAS.index(lam),
                       ls=":", alpha=0.35)
        ax.set_xlabel("delay budget T (slots, ARL$_1$ $\\leq$ T)")
        ax.set_ylabel("minimum detectable $\\delta_a$ (pkts/slot)")
        ax.set_xscale("log")
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.92).set_draggable(True)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if interactive:            # 交互: 关窗后自动 600dpi 导出论文图 Figure_6
            print("[interactive] 图例可拖曳; 关闭窗口即自动导出 Figure_6.png @600dpi")
            plt.show()
            _out = os.path.normpath(os.path.join(
                _HERE, "..", "..", "03-manuscript", "latex", "figures", "Figure_6.png"))
            fig.savefig(_out, dpi=600, bbox_inches="tight")
            print("[exported] Figure_6 @600dpi ->", _out)
            return
        fig.savefig(os.path.join(OUT, "delta_min.png"), dpi=150,
                    bbox_inches="tight")
        print("figure ->", os.path.join(OUT, "delta_min.png"))
    except ImportError:
        print("(matplotlib 不可用, 跳过 PNG)")


if __name__ == "__main__":
    main()