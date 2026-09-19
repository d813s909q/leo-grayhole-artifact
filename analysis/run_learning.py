#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.6 学习类基线评估 (AE 时序重构 vs CUSUM/三基线)
=================================================

Part A: 对 7 种攻击 run 跑 AE, 输出「模式 → 检出 / 报警时隙 / 攻击显现 / 延迟」。
Part B: 节点级注入 Poisson(λ_cong) 自然拥塞, 输出 AE 的 TPR/FPR, 并与 2.5 的
        ours/naive/trust/clif 合并成 5 方法对比。

用法: python3 run_learning.py
依赖: learning_vae.py, 以及 analysis/ 下 2.5 已产出的 congestion_confusion.csv
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learning_vae as lv

BASE = GH_OUT
ATTACK_DIR = os.path.join(BASE, "attack_runs")
OUT = os.path.join(BASE, "learning")
MALICIOUS = {12}

W = 16          # 窗口长度 (时隙)
STRIDE = 4      # 滑窗步长
N_TRIALS = 20   # Part B 蒙特卡洛注入次数
LAM_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def section(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def rates(alive, malicious, n_total):
    tp = len(alive & malicious)
    fp = len(alive - malicious)
    n_neg = n_total - len(malicious)
    tpr = tp / len(malicious) if malicious else 1.0
    fpr = fp / n_neg if n_neg else 0.0
    return tpr, fpr


def build_detector():
    """在 baseline 上训练 AE, 返回 (model, mean, std, threshold)."""
    base = lv.node_series(os.path.join(BASE, "residuals_baseline.csv"))
    Xtr, _ = lv.make_windows(base, W, STRIDE)
    mean, std = lv.fit_scaler(Xtr)
    Xf = lv.apply_scaler(Xtr, mean, std).reshape(len(Xtr), -1)
    model = lv.train_model(Xf, Xf.shape[1])
    scores = lv.node_scores(base, model, mean, std, W, STRIDE)
    thr = lv.threshold_from_scores(scores)
    print("baseline 节点数:", len(base), " 训练窗口数:", len(Xtr),
          " 阈值:", round(thr, 4))
    return model, mean, std, thr


def part_a(model, mean, std, thr):
    section("Part A: 攻击注入扫描 (AE 重构误差异常)")
    specs = [
        ("const_0p1", os.path.join(ATTACK_DIR, "residuals_const_0p1.csv"), "CONSTANT", 0.1),
        ("const_0p2", os.path.join(ATTACK_DIR, "residuals_const_0p2.csv"), "CONSTANT", 0.2),
        ("const_0p3", os.path.join(ATTACK_DIR, "residuals_const_0p3.csv"), "CONSTANT", 0.3),
        ("const_0p4", os.path.join(ATTACK_DIR, "residuals_const_0p4.csv"), "CONSTANT", 0.4),
        ("const_0p5", os.path.join(BASE, "residuals_const12.csv"), "CONSTANT", 0.5),
        ("onoff_1p0_20s", os.path.join(ATTACK_DIR, "residuals_onoff_1p0_20s.csv"), "ON_OFF", 1.0),
        ("scan_1p0_20s", os.path.join(ATTACK_DIR, "residuals_scan_1p0_20s.csv"), "SCAN", 1.0),
    ]
    rows = []
    for name, path, mode, dr in specs:
        ser = lv.node_series(path)
        verdict = lv.detect(ser, model, mean, std, W, STRIDE, thr)
        scores = lv.node_scores(ser, model, mean, std, W, STRIDE)
        alarmed = verdict[12][0]
        false_nodes = sorted(n for n, (a, _) in verdict.items() if a and n != 12)
        rows.append(dict(name=name, mode=mode, drop_rate=dr,
                         detected=bool(alarmed),
                         node12_score=round(float(scores[12]), 2),
                         threshold=round(float(thr), 2),
                         false_alarms=",".join(map(str, false_nodes))))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "learning_sweep.csv"), index=False)
    print(f"{'name':16s} {'mode':9s} {'rate':>5s} {'detected':>9s} "
          f"{'score12':>10s} {'thr':>8s}  false_alarms")
    for r in rows:
        print(f"{r['name']:16s} {r['mode']:9s} {r['drop_rate']:>5.1f} "
              f"{str(r['detected']):>9s} {r['node12_score']:>10.2f} "
              f"{r['threshold']:>8.2f}  {r['false_alarms']}")
    return df


def part_b(model, mean, std, thr):
    section("Part B: 自然拥塞混淆 (AE vs 2.5 四方法)")
    base = lv.node_series(os.path.join(BASE, "residuals_baseline.csv"))
    att = lv.node_series(os.path.join(BASE, "residuals_const12.csv"))
    nodes = sorted(base)
    nslots = base[nodes[0]].shape[0]
    attack = att[12][:, 0].astype(np.float32)   # 节点12 真实攻击丢包时序

    recs = []
    for lam in LAM_SWEEP:
        acc_tpr, acc_fpr = [], []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(2000 + int(lam * 100) + trial)
            series = {}
            for n in nodes:
                d = base[n][:, 0] + rng.poisson(lam, size=nslots)
                d = d + (attack if n == 12 else 0)
                arr = base[n].copy()
                arr[:, 0] = d.astype(np.float32)   # 只扰动 dropped 维 (对齐 2.5)
                series[n] = arr
            verdict = lv.detect(series, model, mean, std, W, STRIDE, thr)
            alive = {n for n, (a, _) in verdict.items() if a}
            tpr, fpr = rates(alive, MALICIOUS, len(nodes))
            acc_tpr.append(tpr); acc_fpr.append(fpr)
        recs.append(dict(lam_cong=lam,
                         vae_tpr=float(np.mean(acc_tpr)),
                         vae_fpr=float(np.mean(acc_fpr))))

    df = pd.DataFrame(recs)

    # 合并 2.5 已有结果
    cmp_path = os.path.join(BASE, "comparison", "congestion_confusion.csv")
    if os.path.exists(cmp_path):
        cmp = pd.read_csv(cmp_path)
        cmp = cmp[["lam_cong", "ours_tpr", "ours_fpr", "naive_tpr", "naive_fpr",
                   "trust_tpr", "trust_fpr", "clif_tpr", "clif_fpr"]]
        df = df.merge(cmp, on="lam_cong", how="left")

    df.to_csv(os.path.join(OUT, "learning_congestion.csv"), index=False)
    print(df.round(4).to_string(index=False))
    return df


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = df["lam_cong"].values
    methods = ["ours", "vae", "naive", "trust", "clif"]
    labels = ["Ours (CUSUM, baseline-aware)", "AE-recon (learning)",
              "Naive threshold", "Trust (BiTrust)", "CLIF (Mahalanobis)"]
    colors = ["#2c7fb8", "#f1a340", "#e6550d", "#31a354", "#756bb1"]
    marks = ["o", "D", "s", "^", "v"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for m, lab, c, mk in zip(methods, labels, colors, marks):
        a1.plot(lam, df[f"{m}_tpr"], marker=mk, color=c, label=lab, lw=2)
        a2.plot(lam, df[f"{m}_fpr"], marker=mk, color=c, label=lab, lw=2)
    for ax, ylab, title in [(a1, "TPR (detection rate)", "TPR vs natural congestion"),
                            (a2, "FPR (false alarm rate)", "FPR vs natural congestion")]:
        ax.set_xlabel("natural congestion rate  lam_cong (per node-slot)")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(title)
    a2.legend(loc="center right", fontsize=8)
    fig.suptitle("Robustness under natural-congestion confusion (ARL0=1e5, 20 injections)",
                 y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "learning_comparison.png")
    fig.savefig(p, bbox_inches="tight")
    print("saved", p)


def main():
    os.makedirs(OUT, exist_ok=True)
    model, mean, std, thr = build_detector()
    a = part_a(model, mean, std, thr)
    b = part_b(model, mean, std, thr)
    try:
        plot(b)
    except Exception as e:
        print("plot skipped:", e)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("=== Part A: attack sweep (AE) ===\n")
        f.write(a.to_csv(index=False))
        f.write("\n=== Part B: congestion confusion (AE vs 4 baselines) ===\n")
        f.write(b.to_csv(index=False))
    print("\nDONE. outputs in", OUT)


if __name__ == "__main__":
    main()