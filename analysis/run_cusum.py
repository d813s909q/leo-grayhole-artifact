#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.4 CUSUM 验证评估脚本
======================

三个验证维度 (对应 RQ2/RQ3 与 C2 的「可证明 FAR/ARL」):
  1) 理论 ARL 曲线 : 蒙特卡洛 vs 马尔可夫链数值解, 展示 h→ARL₀/ARL₁ 关系
  2) 纯净数据归因  : 残差恒 0 的 baseline + const12, 用极小 lambda0 验证
                    检测器把攻击定位到卫星 12 (RQ3)
  3) 噪声鲁棒性    : 注入 Poisson 自然丢包, 验证「自然拥塞混淆」下
                    FAR 受控 (ARL₀) 且攻击仍被检出 (RQ4 核心难点)

输出到 <out>/ 目录:
  arl_curve.csv, detection_clean.json, detection_noisy.json, summary.txt
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import (load_residuals, aggregate_nodes,
                            run_cusum, mc_arl, mc_chain_arl, choose_h_chain,
                            detect_nodes, score_detection)

import numpy as np
import pandas as pd

BASE = GH_OUT
OUT = os.path.join(BASE, "cusum")


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ----------------------------------------------------------------------
# 1) 理论 ARL 曲线
# ----------------------------------------------------------------------
def arl_curve(lambda0, delta, hs, out_dir, mc_h_max=8.0):
    """理论 ARL 曲线: 马尔可夫链数值解为主(精确), 蒙特卡洛仅在 h<=mc_h_max
    交叉验证(大 h 蒙特卡洛会触及运行长度上限被截尾, 数值解仍精确)."""
    rows = []
    for h in hs:
        a0ch, _ = mc_chain_arl(lambda0, delta, h)
        a1ch, _ = mc_chain_arl(lambda0, delta, h, attacking=True)
        a0mc = a1mc = cen0 = cen1 = None
        if h <= mc_h_max:
            a0mc, cen0 = mc_arl(lambda0, delta, h, n_reps=2000, seed=7,
                                max_len=50000)
            a1mc, cen1 = mc_arl(lambda0, delta, h, n_reps=2000, seed=7,
                                attacking=True, max_len=50000)
        rows.append(dict(h=round(h, 2),
                         arl0_mc=(round(a0mc, 1) if a0mc is not None else ""),
                         arl0_chain=round(a0ch, 1),
                         arl1_mc=(round(a1mc, 2) if a1mc is not None else ""),
                         arl1_chain=round(a1ch, 2),
                         censored0=(int(cen0) if cen0 is not None else ""),
                         censored1=(int(cen1) if cen1 is not None else "")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "arl_curve.csv"), index=False)
    print(f"{'h':>6} {'ARL0_MC':>11} {'ARL0_chain':>11} "
          f"{'ARL1_MC':>8} {'ARL1_chain':>11}")
    for r in rows:
        print(f"{r['h']:>6.1f} {str(r['arl0_mc']):>11} "
              f"{r['arl0_chain']:>11.1f} {str(r['arl1_mc']):>8} "
              f"{r['arl1_chain']:>11.2f}")
    return df


# ----------------------------------------------------------------------
# 2) 纯净 + 3) 噪声 检测
# ----------------------------------------------------------------------
def detect_on_data(df, lambda0, delta, h, malicious, inject_lambda0=None,
                   seed=42):
    """聚合节点残差, 可选注入自然丢包, 跑 CUSUM, 返回 (results, node_series)."""
    series = aggregate_nodes(df)
    if inject_lambda0 is not None:
        rng = np.random.default_rng(seed)
        for node in series:
            series[node] = series[node] + rng.poisson(
                inject_lambda0, size=series[node].shape).astype(float)
    results = detect_nodes(series, lambda0, delta, h)
    score = score_detection(results, malicious)
    return results, series, score


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []

    # ---- 参数 ----
    LAMBDA0_CLEAN = 0.01   # 纯净场景: 自然丢包率几乎为 0 (退化但可跑)
    LAMBDA0_NOISE = 0.5    # 噪声场景: 自然丢包 0.5 个/秒/节点
    DELTA = 1.0            # 设计假设的攻击增量 (每时隙)
    TARGET_ARL_CLEAN = 1000.0      # 纯净场景目标 ARL₀
    TARGET_ARL_NOISE = 100000.0    # 噪声场景目标 ARL₀ (保证 200 时隙窗口内
                                   # 每节点误报率 1-exp(-200/1e5)≈0.2% < 1%)
    MALICIOUS = {12}

    # ---- 1) ARL 曲线 ----
    section("1) 理论 ARL 曲线 (lambda0=0.5, delta=1.0)")
    arl_df = arl_curve(LAMBDA0_NOISE, DELTA, [3.0, 5.0, 8.0, 12.0, 18.0], OUT)

    # 反解阈值
    h_clean = choose_h_chain(LAMBDA0_CLEAN, DELTA, TARGET_ARL_CLEAN)
    h_noise = choose_h_chain(LAMBDA0_NOISE, DELTA, TARGET_ARL_NOISE)
    print(f"\n反解阈值: h(λ0={LAMBDA0_CLEAN})={h_clean:.2f} "
          f"@ARL₀≈{TARGET_ARL_CLEAN:.0f}  |  "
          f"h(λ0={LAMBDA0_NOISE})={h_noise:.2f} @ARL₀≈{TARGET_ARL_NOISE:.0f}")
    summary.append(f"threshold_clean={h_clean:.3f}")
    summary.append(f"threshold_noise={h_noise:.3f}")

    # ---- 读数据 ----
    base_df = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))
    att_df = load_residuals(os.path.join(BASE, "residuals_const12.csv"))

    # ---- 2) 纯净归因 ----
    section("2) 纯净数据归因 (baseline vs const12, λ0=0.01)")
    res_base, s_base, sc_base = detect_on_data(
        base_df, LAMBDA0_CLEAN, DELTA, h_clean, MALICIOUS)
    res_att, s_att, sc_att = detect_on_data(
        att_df, LAMBDA0_CLEAN, DELTA, h_clean, MALICIOUS)
    print(f"baseline: 报警={sc_base['tp']+sc_base['fp']} 误报节点={sc_base['false_positives']}")
    print(f"const12 : 检出恶意节点={sc_att['true_detected']}  "
          f"误报节点={sc_att['false_positives']}")
    for node in sorted(sc_att["true_detected"]):
        a = res_att[node]["alarm"]
        print(f"  节点 {node}: 报警时隙={a} (peak_S={res_att[node]['peak_S']:.1f})")
    with open(os.path.join(OUT, "detection_clean.json"), "w") as f:
        json.dump(dict(baseline=sc_base, attack=sc_att,
                       alarms={str(n): r["alarm"] for n, r in res_att.items()}),
                  f, indent=2)
    summary.append(f"clean: tpr={sc_att['tpr']:.2f} fpr={sc_att['fpr']:.3f} "
                   f"detected={sc_att['true_detected']}")

    # ---- 3) 噪声鲁棒 ----
    section("3) 噪声鲁棒性 (注入 λ0=0.5 自然丢包, δ=1.0, h 反解)")
    # 多次注入求平均 FAR / 检测, 更稳健
    n_trials = 20
    n_slots = None
    fp_rates, tp_rates, delays = [], [], []
    for trial in range(n_trials):
        _, _, scb = detect_on_data(base_df, LAMBDA0_NOISE, DELTA, h_noise,
                                   MALICIOUS, inject_lambda0=LAMBDA0_NOISE,
                                   seed=100 + trial)
        resa, _, sca = detect_on_data(att_df, LAMBDA0_NOISE, DELTA, h_noise,
                                      MALICIOUS, inject_lambda0=LAMBDA0_NOISE,
                                      seed=100 + trial)
        fp_rates.append(scb["fpr"])
        tp_rates.append(sca["tpr"])
        delays.append(resa[12]["alarm"] if 12 in sca["true_detected"] else None)

    n_slots = len(next(iter(aggregate_nodes(base_df).values())))
    fp_rates = np.array(fp_rates)
    tp_rates = np.array(tp_rates)
    delays = [d for d in delays if d is not None]
    print(f"跨 {n_trials} 次注入 (观测窗口 {n_slots} 时隙):")
    print(f"  每节点窗口误报率 均值={fp_rates.mean():.4f}  最大={fp_rates.max():.4f}")
    print(f"  TPR(检出率)       均值={tp_rates.mean():.3f}")
    print(f"  检测延迟(时隙)    均值={np.mean(delays):.1f}  中位={np.median(delays):.1f} "
          f"(n={len(delays)})")
    # FAR 理论对照
    arl0_noise, _ = mc_chain_arl(LAMBDA0_NOISE, DELTA, h_noise)
    far_theory = 1.0 / arl0_noise                       # 每节点·时隙误报率
    fpr_window_theory = 1.0 - math.exp(-n_slots / arl0_noise)  # 每节点窗口误报率
    print(f"  理论对照: ARL₀={arl0_noise:.0f}  FAR(每节点·时隙)=1/ARL₀={far_theory:.5f}")
    print(f"            每节点窗口误报率(理论)={fpr_window_theory:.4f}  vs  实测均值={fp_rates.mean():.4f}")
    with open(os.path.join(OUT, "detection_noisy.json"), "w") as f:
        json.dump(dict(n_trials=n_trials, n_slots=n_slots,
                       fpr_window_mean=float(fp_rates.mean()),
                       fpr_window_max=float(fp_rates.max()),
                       fpr_window_theory=float(fpr_window_theory),
                       tpr_mean=float(tp_rates.mean()),
                       delay_mean=float(np.mean(delays)) if delays else None,
                       delay_median=float(np.median(delays)) if delays else None,
                       theory_arl0=float(arl0_noise),
                       theory_far_per_slot=float(far_theory)), f, indent=2)
    summary.append(f"noisy: fpr_window_mean={fp_rates.mean():.4f} "
                   f"(theory {fpr_window_theory:.4f}) tpr_mean={tp_rates.mean():.3f} "
                   f"delay={np.mean(delays):.1f}s  ARL0={arl0_noise:.0f}")

    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\nDONE. outputs in", OUT)


if __name__ == "__main__":
    main()