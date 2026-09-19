#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.5 攻击注入 + 基线对比 评估脚本
================================

Part A 攻击注入扫描 (真实仿真): 对 CONSTANT 0.1–0.5 / ON_OFF / SCAN 各 run 的残差
        跑 2.4 CUSUM, 输出「模式 → 是否检出 / 报警时隙 / 检测延迟 / 攻击丢包量」。
Part B 自然拥塞混淆对比: 在观测上注入加性 Poisson(λ_cong) 自然拥塞丢包, 对比
        我方(残差/基线感知 CUSUM) vs 朴素阈值 / 信任类(BiTrust) / CLIF(马氏距离),
        输出 λ_cong 扫描下各方法的 TPR / FPR(自然拥塞混淆率)。

用法: python3 run_comparison.py
依赖: cusum_detector.py, baselines.py (同目录), 残差 CSV 位于 analysis/ 与 analysis/attack_runs/
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import (load_residuals, aggregate_nodes,
                            detect_nodes, score_detection,
                            mc_chain_arl, choose_h_chain)
from baselines import (aggregate_node_field, naive_threshold, trust_detect,
                       fit_gaussian, mahalanobis, threshold_from_quantile)

BASE = GH_OUT
ATTACK_DIR = os.path.join(BASE, "attack_runs")
OUT = os.path.join(BASE, "comparison")
MALICIOUS = {12}

DELTA = 1.0          # 设计攻击增量
TARGET_ARL = 1e5     # Part B 目标 ARL₀
N_TRIALS = 200       # Part B 蒙特卡洛注入次数 (M1.3: ≥100 以支撑 FPR 置信区间)
Z95 = 1.96           # Wilson 95% 置信区间 z 值


def section(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ----------------------------------------------------------------------
# Part A: 攻击注入扫描 (真实仿真残差)
# ----------------------------------------------------------------------
def part_a():
    section("Part A: 攻击注入扫描 (CUSUM 检测 vs 攻击模式/丢包率)")
    # (name, residual csv, mode, drop_rate)
    specs = [
        ("const_0p1", os.path.join(ATTACK_DIR, "residuals_const_0p1.csv"), "CONSTANT", 0.1),
        ("const_0p2", os.path.join(ATTACK_DIR, "residuals_const_0p2.csv"), "CONSTANT", 0.2),
        ("const_0p3", os.path.join(ATTACK_DIR, "residuals_const_0p3.csv"), "CONSTANT", 0.3),
        ("const_0p4", os.path.join(ATTACK_DIR, "residuals_const_0p4.csv"), "CONSTANT", 0.4),
        ("const_0p5", os.path.join(BASE, "residuals_const12.csv"), "CONSTANT", 0.5),
        ("onoff_1p0_20s", os.path.join(ATTACK_DIR, "residuals_onoff_1p0_20s.csv"), "ON_OFF", 1.0),
        ("scan_1p0_20s", os.path.join(ATTACK_DIR, "residuals_scan_1p0_20s.csv"), "SCAN", 1.0),
    ]
    h = choose_h_chain(0.01, DELTA, 1000.0)  # 纯净场景阈值 (同 2.4)
    rows = []
    missing = []
    for name, path, mode, dr in specs:
        if not os.path.exists(path):
            missing.append(path)
            rows.append(dict(name=name, mode=mode, drop_rate=dr, detected=None,
                             alarm_slot=None, total_dropped=None, reason="missing residual"))
            continue
        df = load_residuals(path)
        series = aggregate_nodes(df)          # 残差按节点聚合
        results = detect_nodes(series, 0.01, DELTA, h)
        score = score_detection(results, MALICIOUS)
        total_dropped = int(df["dropped"].sum())
        detected = 12 in score["true_detected"]
        alarm = results[12]["alarm"] if 12 in results else None
        # 攻击时序起点: 首个 residual>0 的时隙 (攻击实际显现时刻)
        node12 = series.get(12, np.zeros(1))
        onset = int(np.argmax(node12 > 0)) if (node12 > 0).any() else None
        delay = (alarm - onset) if (alarm is not None and onset is not None) else None
        rows.append(dict(name=name, mode=mode, drop_rate=dr, detected=detected,
                         alarm_slot=alarm, total_dropped=total_dropped,
                         attack_onset=onset, delay_slots=delay))
    df_out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df_out.to_csv(os.path.join(OUT, "attack_sweep.csv"), index=False)
    print(f"{'name':16s} {'mode':9s} {'rate':>5s} {'dropped':>8s} "
          f"{'detected':>9s} {'alarm':>6s} {'onset':>6s} {'delay':>6s}")
    for r in rows:
        det = "YES" if r["detected"] else ("MISSING" if r["detected"] is None else "NO")
        print(f"{r['name']:16s} {r['mode']:9s} {r['drop_rate']:>5.1f} "
              f"{str(r['total_dropped']):>8s} {det:>9s} "
              f"{str(r['alarm_slot']):>6s} {str(r['attack_onset']):>6s} "
              f"{str(r['delay_slots']):>6s}")
    if missing:
        print("\n⚠️ 缺失残差文件 (需先运行 2.5 仿真 + baseline_model 提取):")
        for m in missing:
            print("   ", m)
    return df_out


# ----------------------------------------------------------------------
# Part B: 自然拥塞混淆对比
# ----------------------------------------------------------------------
def _features_from_arrays(drop, rx, util, ho, nodes):
    X = []
    for n in nodes:
        X.append([float(drop[n].mean()), float(rx[n].mean()),
                  float(util[n].mean()), float(ho[n].mean()),
                  float(drop[n].max())])
    return np.array(X, dtype=float)


def _verdict_to_counts(verdict_dict, malicious):
    """把 per-node bool 判决转成 (tp, fp) 原始计数 (分母由节点集合决定)."""
    tp = sum(1 for n in malicious if n in verdict_dict and verdict_dict[n])
    fp = sum(1 for n, v in verdict_dict.items() if n not in malicious and v)
    return tp, fp


def wilson_ci(k, n, z=Z95):
    """Wilson score 区间 (95%). 返回 (lo, hi); n=0 时返回 (0, 0)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rate_and_ci(k, n):
    """比例 + Wilson 95% 置信区间. 返回 (rate, lo, hi)."""
    lo, hi = wilson_ci(k, n)
    return (k / n) if n else 0.0, lo, hi


def _run_all_detectors(X, rx, util, ho, nodes, lam, delta, h, tau_naive,
                       tau_trust, trust_window, trust_min_attempts,
                       mu, cov_inv, thresh_clif, malicious):
    """在给定观测 X 上跑 4 个检测器, 返回 {detector: (tp, fp)} 原始计数."""
    from cusum_detector import run_cusum
    alarms = {}
    for n in nodes:
        _, a = run_cusum(X[n], lam, delta, h)
        alarms[n] = a is not None
    return {
        "ours": _verdict_to_counts(alarms, malicious),
        "naive": _verdict_to_counts(naive_threshold(X, tau_naive), malicious),
        "trust": _verdict_to_counts(
            trust_detect(X, rx, trust_window, tau_trust, trust_min_attempts),
            malicious),
        "clif": _verdict_to_counts(
            dict(zip(nodes, mahalanobis(_features_from_arrays(X, rx, util, ho, nodes),
                                        mu, cov_inv) > thresh_clif)),
            malicious),
    }


def part_b():
    section("Part B: 自然拥塞混淆对比 (Poisson(λ_cong) 加性注入)")
    base_df = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))
    att_df = load_residuals(os.path.join(BASE, "residuals_const12.csv"))

    # 节点级原始字段
    drop_base = aggregate_node_field(base_df, "dropped")
    rx_base = aggregate_node_field(base_df, "rx")
    util_base = aggregate_node_field(base_df, "util")
    ho_base = aggregate_node_field(base_df, "handover_events")
    drop_att = aggregate_node_field(att_df, "dropped")

    nodes = sorted(drop_base)
    honest = [n for n in nodes if n not in MALICIOUS]
    nslots = next(iter(drop_base.values())).shape[0]
    attack = drop_att.get(12, np.zeros(nslots))  # 攻击信号 (node 12 真实丢包)

    # ---- 干净数据标定各基线阈值 ----
    clean_totals = {n: float(drop_base[n].sum()) for n in honest}
    tau_naive = max(clean_totals.values()) + 0.5       # 干净总丢包=0 → 阈值 0.5

    trust_window = 10
    trust_min_attempts = 50   # 滑窗内总尝试不足 50 不评估 (冷启动/无流量节点)
    # 干净时 trust=1, 阈值取 0.99 (低于即报警)
    tau_trust = 0.99

    F_clean = _features_from_arrays(drop_base, rx_base, util_base, ho_base, nodes)
    mu, cov_inv = fit_gaussian(F_clean)
    dist_clean = mahalanobis(F_clean, mu, cov_inv)
    thresh_clif = threshold_from_quantile(dist_clean, 0.99)

    # ---- λ_cong 扫描 ----
    lam_sweep = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    dets = ["ours", "naive", "trust", "clif"]
    n_pos = len(MALICIOUS)                    # 恶意节点数 (=1)
    n_neg = len(honest)                       # 诚实节点数 (=16)
    n_pos_tot = n_pos * N_TRIALS              # TPR 的观测总次数
    n_neg_tot = n_neg * N_TRIALS              # FPR 的观测总次数
    recs = []
    for lam in lam_sweep:
        lam_eff = lam if lam > 0 else 0.01          # CUSUM 需 λ0>0
        h = choose_h_chain(lam_eff, DELTA, TARGET_ARL)
        tp = {d: 0 for d in dets}
        fp = {d: 0 for d in dets}
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(1000 + int(lam * 100) + trial)
            X = {}
            for n in nodes:
                X[n] = drop_base[n] + rng.poisson(lam, size=nslots)
            X[12] = X[12] + attack                     # 攻击 + 拥塞叠加
            counts = _run_all_detectors(
                X, rx_base, util_base, ho_base, nodes, lam_eff, DELTA, h,
                tau_naive, tau_trust, trust_window, trust_min_attempts,
                mu, cov_inv, thresh_clif, MALICIOUS)
            for d in dets:
                tp[d] += counts[d][0]
                fp[d] += counts[d][1]
        rec = dict(lam_cong=lam, arl0_target=TARGET_ARL, h=round(h, 2),
                   n_trials=N_TRIALS, n_pos_tot=n_pos_tot, n_neg_tot=n_neg_tot)
        for d in dets:
            tpr, tpr_lo, tpr_hi = rate_and_ci(tp[d], n_pos_tot)
            fpr, fpr_lo, fpr_hi = rate_and_ci(fp[d], n_neg_tot)
            rec[f"{d}_tpr"] = tpr
            rec[f"{d}_tpr_lo"] = tpr_lo
            rec[f"{d}_tpr_hi"] = tpr_hi
            rec[f"{d}_fpr"] = fpr
            rec[f"{d}_fpr_lo"] = fpr_lo
            rec[f"{d}_fpr_hi"] = fpr_hi
        recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "congestion_confusion.csv"), index=False)

    print(f"\nTPR/FPR 报告为 均值[Wilson95% lo,hi]；FPR 以 % 显示；"
          f"每格 TPR n={n_pos_tot}, FPR n={n_neg_tot} 次观测")
    print(f"{'λ_cong':>7s} | {'h':>6s} | " + " | ".join(
        f"{d[:5]:>5s} TPR[lo,hi]   FPR%[lo,hi]" for d in dets))
    print("-" * 150)
    for r in recs:
        line = f"{r['lam_cong']:>7.2f} | {r['h']:>6.2f} | "
        for d in dets:
            line += (f"{r[f'{d}_tpr']:.2f}[{r[f'{d}_tpr_lo']:.2f},"
                     f"{r[f'{d}_tpr_hi']:.2f}] "
                     f"{100*r[f'{d}_fpr']:5.2f}[{100*r[f'{d}_fpr_lo']:4.2f},"
                     f"{100*r[f'{d}_fpr_hi']:4.2f}] | ")
        print(line.rstrip())
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    a = part_a()
    b = part_b()
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("=== Part A: attack sweep (CUSUM) ===\n")
        f.write(a.to_csv(index=False))
        f.write("\n=== Part B: congestion confusion ===\n")
        f.write(b.to_csv(index=False))
    print("\nDONE. outputs in", OUT)


if __name__ == "__main__":
    main()