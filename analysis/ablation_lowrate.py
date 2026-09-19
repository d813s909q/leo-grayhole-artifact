#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4-P0 后处理实验（无需新仿真）
================================
P0-③ 消融（M3.3）: 证明「鲁棒性来自期望模型，而非决策规则的复杂度」——
    把朴素阈值 / 信任规则直接喂给【残差序列】(d - λ_cong) 后，FPR 是否大幅下降，
    并对照它们跑在【原始观测】上的已知高 FPR。附带「残差 + 方差标定」版本，
    揭示期望模型的两层信息（均值 λ_cong + 方差 √(n·λ_cong)）各贡献多少。

P0-④ 低速率逃逸攻击下界（M3.1 / M1.4）: 攻击者以低于设计增量 δ=1.0 的速率
    δ_small 丢包时，检测概率 / 检测延迟随 δ_small 的变化，并给出理论临界速率
    δ* = 1/ln(1+1/λ0) - λ0（CUSUM 漂移=0 的不可检测边界），量化 graceful
    degradation 而非 cliff。

用法: python3 ablation_lowrate.py
输出: analysis/comparison/congestion_ablation.csv
      analysis/comparison/lowrate_attack.csv
      analysis/comparison/ablation_summary.txt
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
from cusum_detector import (load_residuals, run_cusum, choose_h_chain)
from baselines import (aggregate_node_field, naive_threshold)

BASE = GH_OUT
OUT = os.path.join(BASE, "comparison")
MALICIOUS = {12}
DELTA = 1.0          # CUSUM 设计攻击增量
TARGET_ARL = 1e5
N_TRIALS = 40        # 消融注入次数（P0 提升统计力：40 > 原 20）
LOW_N_REPS = 500     # 低速率攻击蒙特卡洛重复
LOW_NSLOTS = 200     # 观测窗长度（与主场景一致）


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def tpr_fpr(verdict, malicious):
    fp = tp = n_neg = 0
    for n, v in verdict.items():
        if n in malicious:
            tp += int(v)
        else:
            n_neg += 1
            fp += int(v)
    return (tp / len(malicious)), (fp / n_neg if n_neg else 0.0)


def trust_rule(drop_series, rx_series, window, tau, min_attempts, subtract=0.0):
    """信任规则: 窗口转发比 rx/(rx+drop*) < tau 即报警。

    drop* = max(0, dropped_t - subtract) —— subtract=λ_cong 时即「残差化」：
    把期望自然丢包从信誉分母中扣除，残留仅攻击丢包。"""
    verdict = {}
    for n in drop_series:
        d = drop_series[n]
        r = rx_series.get(n, np.zeros_like(d))
        alarmed = False
        dstar = np.maximum(0.0, d - subtract)
        for t in range(len(d) - window + 1):
            rxw = r[t:t + window].sum()
            dropw = dstar[t:t + window].sum()
            total = rxw + dropw
            if total >= min_attempts and rxw / total < tau:
                alarmed = True
                break
        verdict[n] = alarmed
    return verdict


# ----------------------------------------------------------------------
# P0-③ 消融
# ----------------------------------------------------------------------
def ablation():
    section("P0-③ 消融：期望模型 vs 决策规则（朴素阈值 / 信任 跑在残差 vs 观测）")
    base_df = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))
    att_df = load_residuals(os.path.join(BASE, "residuals_const12.csv"))

    drop_base = aggregate_node_field(base_df, "dropped")
    rx_base = aggregate_node_field(base_df, "rx")
    drop_att = aggregate_node_field(att_df, "dropped")

    nodes = sorted(drop_base)
    nslots = next(iter(drop_base.values())).shape[0]
    attack = drop_att.get(12, np.zeros(nslots))

    # 干净数据标定
    honest = [n for n in nodes if n not in MALICIOUS]
    tau_naive = max(drop_base[n].sum() for n in honest) + 0.5   # 干净总丢包=0 → 0.5
    trust_window = 10
    trust_min = 50
    tau_trust = 0.99

    lam_sweep = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    recs = []
    for lam in lam_sweep:
        lam_eff = lam if lam > 0 else 0.01
        h = choose_h_chain(lam_eff, DELTA, TARGET_ARL)
        # 方差标定：残差(去均值后)每节点总和的 std = sqrt(nslots*lam)
        tau_res_std = max(0.5, 3.0 * math.sqrt(nslots * lam))
        acc = {k: [] for k in [
            "ours_tpr", "ours_fpr",
            "naive_raw_tpr", "naive_raw_fpr",
            "naive_res_tpr", "naive_res_fpr",
            "naive_res_std_tpr", "naive_res_std_fpr",
            "trust_raw_tpr", "trust_raw_fpr",
            "trust_res_tpr", "trust_res_fpr"]}
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(5000 + int(lam * 100) + trial)
            X = {n: drop_base[n] + rng.poisson(lam, size=nslots) for n in nodes}
            X[12] = X[12] + attack            # 攻击 + 拥塞叠加

            # 我方 CUSUM（λ0=lam 期望模型）
            ov = {}
            for n in nodes:
                _, a = run_cusum(X[n], lam_eff, DELTA, h)
                ov[n] = a is not None
            acc["ours_tpr"].append(tpr_fpr(ov, MALICIOUS)[0])
            acc["ours_fpr"].append(tpr_fpr(ov, MALICIOUS)[1])

            # 朴素阈值：原始观测
            acc["naive_raw_tpr"].append(tpr_fpr(naive_threshold(X, tau_naive), MALICIOUS)[0])
            acc["naive_raw_fpr"].append(tpr_fpr(naive_threshold(X, tau_naive), MALICIOUS)[1])

            # 朴素阈值：残差序列 R = d - lam
            R = {n: X[n] - lam for n in nodes}
            acc["naive_res_tpr"].append(tpr_fpr(naive_threshold(R, tau_naive), MALICIOUS)[0])
            acc["naive_res_fpr"].append(tpr_fpr(naive_threshold(R, tau_naive), MALICIOUS)[1])

            # 朴素阈值：残差 + 方差标定
            acc["naive_res_std_tpr"].append(tpr_fpr(naive_threshold(R, tau_res_std), MALICIOUS)[0])
            acc["naive_res_std_fpr"].append(tpr_fpr(naive_threshold(R, tau_res_std), MALICIOUS)[1])

            # 信任：原始观测
            tv_raw = trust_rule(X, rx_base, trust_window, tau_trust, trust_min)
            acc["trust_raw_tpr"].append(tpr_fpr(tv_raw, MALICIOUS)[0])
            acc["trust_raw_fpr"].append(tpr_fpr(tv_raw, MALICIOUS)[1])

            # 信任：残差化（subtract=lam）
            tv_res = trust_rule(X, rx_base, trust_window, tau_trust, trust_min, subtract=lam)
            acc["trust_res_tpr"].append(tpr_fpr(tv_res, MALICIOUS)[0])
            acc["trust_res_fpr"].append(tpr_fpr(tv_res, MALICIOUS)[1])

        rec = dict(lam_cong=lam, h=round(h, 2), tau_res_std=round(tau_res_std, 3))
        for k, v in acc.items():
            rec[k] = round(float(np.mean(v)), 4)
        recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "congestion_ablation.csv"), index=False)

    cols = ["lam_cong", "ours_fpr", "naive_raw_fpr", "naive_res_fpr",
            "naive_res_std_fpr", "trust_raw_fpr", "trust_res_fpr"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n（TPR：所有方法、所有 λ 均 =1.0；差异完全在 FPR 列）")
    return df


# ----------------------------------------------------------------------
# P0-④ 低速率逃逸攻击下界
# ----------------------------------------------------------------------
def lowrate():
    section("P0-④ 低速率逃逸攻击下界（检测概率 vs 攻击率 δ_small）")
    # 背景（自然/参考率）取几个有代表性的 λ0
    lam0_sweep = [0.01, 0.1, 0.5, 1.0, 2.0]
    # 攻击额外丢包率（低于设计增量 δ=1.0 的区域为重点）
    d_sweep = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    rows = []
    for lam0 in lam0_sweep:
        h = choose_h_chain(lam0, DELTA, TARGET_ARL)
        # 理论临界速率：漂移 E[Z_t]=0 的 δ*（低于则 S_t 期望不增长 → 不可检测）
        dstar = 1.0 / math.log(1.0 + 1.0 / lam0) - lam0
        for ds in d_sweep:
            rng = np.random.default_rng(int(round((lam0 * 1000 + ds * 100))))
            detected = 0
            delays = []
            for _ in range(LOW_N_REPS):
                x = rng.poisson(lam0 + ds, size=LOW_NSLOTS)
                _, alm = run_cusum(x, lam0, DELTA, h)
                if alm is not None:
                    detected += 1
                    delays.append(alm)
            p_det = detected / LOW_N_REPS
            delay = float(np.mean(delays)) if delays else float("nan")
            # 期望额外丢包量（观测窗内），量化攻击造成的损害上界
            exp_drop = ds * LOW_NSLOTS
            rows.append(dict(lam0=lam0, delta_small=ds, dstar=round(dstar, 4),
                             h=round(h, 2), p_detect=round(p_det, 4),
                             mean_delay=round(delay, 2) if np.isfinite(delay) else None,
                             exp_extra_drop=exp_drop))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lowrate_attack.csv"), index=False)

    print(f"{'λ0':>5s} {'δ*=理论边界':>10s} {'h':>6s}  | 检测概率 p_detect per δ_small")
    for lam0 in lam0_sweep:
        sub = df[df["lam0"] == lam0]
        dstar = sub["dstar"].iloc[0]
        h = sub["h"].iloc[0]
        probstr = "  ".join(f"{r['delta_small']}:{r['p_detect']:.2f}"
                            for _, r in sub.iterrows())
        print(f"{lam0:>5.2f} {dstar:>10.4f} {h:>6.2f}  |  {probstr}")
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    a = ablation()
    b = lowrate()
    with open(os.path.join(OUT, "ablation_summary.txt"), "w") as f:
        f.write("=== P0-③ Ablation (naive/trust on residual vs raw) ===\n")
        f.write(a.to_csv(index=False))
        f.write("\n=== P0-④ Low-rate attack (detection prob vs delta_small) ===\n")
        f.write(b.to_csv(index=False))
    print("\nDONE. outputs in", OUT)


if __name__ == "__main__":
    main()