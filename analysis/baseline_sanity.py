#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3.2 (#9): 基线重实现 sanity 对齐
==================================
回应 mock review M3.2: "CLIF/BiTrust/RPLAD3 均为 -style 重实现, 应 (a) 在 VI-A
明确重实现参数, (b) 与原文报告数字 sanity 对齐 (如 CLIF grayhole recall
96.9/95.9), 否则 Part B 的 FPR=100% 可能被质疑为实现劣化."

Part S1 无拥塞原始条件 (与原文评估条件同构 — 仿真器默认无自然拥塞):
    阈值标定协议与 run_comparison.py Part B 完全一致 (全部在干净 run 上标定),
    作用对象换为 70 个多种子攻击 run (7 schedules x 10 seeds, 真实仿真残差).
    → 4 检测器 TPR/FPR + Wilson 95% CI, 与 CLIF 原文数字对齐.

Part S2 真实拥塞场景 (仿真器队列溢出, 非合成 Poisson 注入):
    同一套重实现作用于 analysis/congestion/ 的 clean / attacked 原始观测:
      - clif (clean-fit) : 在无拥塞干净 run 上拟合 (与 S1/Part B 同协议)
      - clif (cong-fit)  : 在拥塞 clean run 上重新拟合 (训练分布匹配部署分布)
    → 若 clean-fit 在真实拥塞下 FPR 塌缩而 cong-fit 恢复, 则塌缩源于
      分布偏移 (CLIF 原文自认的 grayhole-congestion 重叠), 而非实现缺陷.

用法 (Ubuntu-20.04): python3 baseline_sanity.py
输出: analysis/comparison/baseline_sanity_s1.csv / _s2.csv / baseline_sanity.txt
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
from cusum_detector import (load_residuals, aggregate_nodes, detect_nodes,
                            score_detection, choose_h_chain)
from baselines import (aggregate_node_field, naive_threshold, trust_detect,
                       fit_gaussian, mahalanobis, threshold_from_quantile)

BASE = GH_OUT
MS_DIR = os.path.join(BASE, "multiseed")
CONG_DIR = os.path.join(BASE, "congestion")
OUT = os.path.join(BASE, "comparison")
MALICIOUS = {12}

# ---- 与 Part B / Part A 完全一致的检测协议 ----
LAMBDA0 = 0.01
DELTA = 1.0
TARGET_ARL = 1000.0        # Part A 纯净场景阈值
TRUST_WINDOW = 10
TRUST_MIN_ATTEMPTS = 50
TAU_TRUST = 0.99
CLIF_Q = 0.99
Z95 = 1.96

SPECS = [
    ("const_0p1", "CONSTANT", 0.1),
    ("const_0p2", "CONSTANT", 0.2),
    ("const_0p3", "CONSTANT", 0.3),
    ("const_0p4", "CONSTANT", 0.4),
    ("const_0p5", "CONSTANT", 0.5),
    ("onoff_1p0_20s", "ON_OFF", 1.0),
    ("scan_1p0_20s", "SCAN", 1.0),
]
N_SEEDS = 10


def wilson_ci(k, n, z=Z95):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def node_features(df, nodes, use_util=None):
    """逐节点跨层特征向量 (与 run_comparison._features_from_arrays 同构).

    use_util=None 时按 df 是否含 util 列自适应; 显式传 False 用于拥塞残差
    (该数据源无 util 列), 保证 fit/apply 特征集一致."""
    if use_util is None:
        use_util = "util" in df.columns
    drop = aggregate_node_field(df, "dropped")
    rx = aggregate_node_field(df, "rx")
    ho = aggregate_node_field(df, "handover_events")
    util = aggregate_node_field(df, "util") if use_util else None
    X = []
    for n in nodes:
        row = [float(drop[n].mean()), float(rx[n].mean())]
        if util is not None:
            row.append(float(util[n].mean()))
        row += [float(ho[n].mean()), float(drop[n].max())]
        X.append(row)
    return np.array(X, dtype=float)


def verdicts_all(df, h, tau_naive, mu, cov_inv, thresh_clif, use_util=None):
    """在一份残差 CSV 的原始观测上跑 4 个检测器, 返回 {det: {node: bool}}."""
    nodes = sorted(aggregate_node_field(df, "dropped"))
    drop = aggregate_node_field(df, "dropped")
    rx = aggregate_node_field(df, "rx")

    out = {}
    out["ours"] = {n: False for n in nodes}
    res_series = aggregate_nodes(df)
    res = detect_nodes(res_series, LAMBDA0, DELTA, h)
    for n, r in res.items():
        out["ours"][n] = r["alarm"] is not None

    out["naive"] = naive_threshold(drop, tau_naive)
    out["trust"] = trust_detect(drop, rx, TRUST_WINDOW, TAU_TRUST,
                                TRUST_MIN_ATTEMPTS)
    X = node_features(df, nodes, use_util=use_util)
    dist = mahalanobis(X, mu, cov_inv)
    out["clif"] = dict(zip(nodes, dist > thresh_clif))
    return out


def counts(verdict, malicious=MALICIOUS):
    tp = sum(1 for n in malicious if verdict.get(n, False))
    fp = sum(1 for n, v in verdict.items() if n not in malicious and v)
    return tp, fp


def counts_all_honest(verdict):
    """无攻击场景 (如 congested_clean): 所有节点皆诚实, 任何报警都是误报."""
    return 0, sum(1 for v in verdict.values() if v)


def fmt_rate(k, n):
    lo, hi = wilson_ci(k, n)
    return f"{k / n if n else 0.0:.3f}[{lo:.3f},{hi:.3f}]"


# ----------------------------------------------------------------------
# Part S1: 无拥塞原始条件, 70 多种子攻击 run
# ----------------------------------------------------------------------
def part_s1(h):
    clean = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))
    nodes = sorted(aggregate_node_field(clean, "dropped"))
    honest = [n for n in nodes if n not in MALICIOUS]

    # 干净标定 (与 Part B 完全一致)
    drop_c = aggregate_node_field(clean, "dropped")
    tau_naive = max(float(drop_c[n].sum()) for n in honest) + 0.5
    Xc = node_features(clean, nodes)
    mu, cov_inv = fit_gaussian(Xc)
    thresh_clif = threshold_from_quantile(mahalanobis(Xc, mu, cov_inv), CLIF_Q)

    dets = ["ours", "naive", "trust", "clif"]
    detail, summary = [], []
    for spec, mode, dr in SPECS:
        agg = {d: [0, 0] for d in dets}          # (tp, fp) 累计
        n_runs = 0
        for s in range(N_SEEDS):
            path = os.path.join(MS_DIR, f"residuals_{spec}_s{s}.csv")
            if not os.path.exists(path):
                continue
            df = load_residuals(path)
            v = verdicts_all(df, h, tau_naive, mu, cov_inv, thresh_clif)
            n_runs += 1
            for d in dets:
                tp, fp = counts(v[d])
                agg[d][0] += tp
                agg[d][1] += fp
                detail.append(dict(schedule=spec, mode=mode, drop_rate=dr,
                                   seed=s, detector=d, tp=tp, fp=fp))
        row = dict(schedule=spec, mode=mode, drop_rate=dr, n_runs=n_runs)
        for d in dets:
            row[f"{d}_tp"], row[f"{d}_fp"] = agg[d]
        summary.append(row)

    # 汇总: 全部 70 run + CONSTANT 家族 (50 个 RNG 独立实现, 见 08 报告 §3.1)
    def agg_rows(rows, label):
        out = dict(schedule=label, n_runs=sum(r["n_runs"] for r in rows))
        for d in dets:
            tp = sum(r[f"{d}_tp"] for r in rows)
            fp = sum(r[f"{d}_fp"] for r in rows)
            out[f"{d}_tp"], out[f"{d}_fp"] = tp, fp
        return out

    const_rows = [r for r in summary if r["mode"] == "CONSTANT"]
    summary.append(agg_rows(summary, "ALL (7x10)"))
    summary.append(agg_rows(const_rows, "CONSTANT family (5x10)"))

    print(f"\nS1 标定: tau_naive={tau_naive}, trust(w={TRUST_WINDOW}, "
          f"tau={TAU_TRUST}, min_att={TRUST_MIN_ATTEMPTS}), "
          f"CLIF q{CLIF_Q:.2f} thresh={thresh_clif:.3f}, h={h:.2f}")
    print(f"\n{'schedule':>24s} {'runs':>4s} | " + " | ".join(
        f"{d:>21s}" for d in dets))
    print("-" * 118)
    for r in summary:
        line = f"{r['schedule']:>24s} {r['n_runs']:>4d} | "
        n_pos = r["n_runs"] * len(MALICIOUS)
        n_neg = r["n_runs"] * len(honest)
        for d in dets:
            line += (f"TPR {fmt_rate(r[f'{d}_tp'], n_pos):>20s} "
                     f"FPR {fmt_rate(r[f'{d}_fp'], n_neg):>20s} | ")
        print(line)
    print(f"\n(FPR 分母 = runs x {len(honest)} 诚实节点; TPR 分母 = runs)")
    return pd.DataFrame(summary), pd.DataFrame(detail)


# ----------------------------------------------------------------------
# Part S2: 真实拥塞场景 (仿真器队列溢出)
# ----------------------------------------------------------------------
def part_s2(h):
    clean_nc = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))
    cong_clean = load_residuals(os.path.join(CONG_DIR, "residuals_cong_clean.csv"))
    cong_att = load_residuals(os.path.join(CONG_DIR, "residuals_cong_att12.csv"))

    nodes = sorted(aggregate_node_field(clean_nc, "dropped"))
    honest = [n for n in nodes if n not in MALICIOUS]

    drop_c = aggregate_node_field(clean_nc, "dropped")
    rx_c = aggregate_node_field(clean_nc, "rx")
    tau_naive = max(float(drop_c[n].sum()) for n in honest) + 0.5

    # clif clean-fit: 无拥塞干净 run 拟合 (4 特征, 拥塞数据无 util 列)
    Xc = node_features(clean_nc, nodes, use_util=False)
    mu, cov_inv = fit_gaussian(Xc)
    thresh_clean = threshold_from_quantile(mahalanobis(Xc, mu, cov_inv), CLIF_Q)

    # clif cong-fit: 拥塞 clean run 拟合 (训练分布 = 部署分布)
    Xcc = node_features(cong_clean, nodes, use_util=False)
    mu2, cov_inv2 = fit_gaussian(Xcc)
    thresh_cong = threshold_from_quantile(mahalanobis(Xcc, mu2, cov_inv2), CLIF_Q)

    def run_row(label, df, has_attack):
        drop = aggregate_node_field(df, "dropped")
        rx = aggregate_node_field(df, "rx")
        X = node_features(df, nodes, use_util=False)
        cnt = counts if has_attack else counts_all_honest
        row = dict(scenario=label, n_honest=(len(honest) if has_attack
                                             else len(nodes)))
        v = {
            "naive": naive_threshold(drop, tau_naive),
            "trust": trust_detect(drop, rx, TRUST_WINDOW, TAU_TRUST,
                                  TRUST_MIN_ATTEMPTS),
            "clif_cleanfit": dict(zip(nodes, mahalanobis(X, mu, cov_inv)
                                      > thresh_clean)),
            "clif_congfit": dict(zip(nodes, mahalanobis(X, mu2, cov_inv2)
                                     > thresh_cong)),
        }
        for d, vd in v.items():
            row[d] = cnt(vd)
            row[f"{d}_nodes"] = ",".join(str(n) for n, f in vd.items() if f)
        row["node12_dropped"] = int(drop[12].sum())
        row["honest_max_dropped"] = max(int(drop[n].sum()) for n in honest)
        return row

    rows = [run_row("congested_clean (no attack)", cong_clean, False),
            run_row("congested_attack (node12)", cong_att, True)]
    dets2 = ["naive", "trust", "clif_cleanfit", "clif_congfit"]
    df = pd.DataFrame([{
        "scenario": r["scenario"], "n_honest": r["n_honest"],
        "node12_dropped": r["node12_dropped"],
        "honest_max_dropped": r["honest_max_dropped"],
        **{f"{d}_tp": r[d][0] for d in dets2},
        **{f"{d}_fp": r[d][1] for d in dets2},
        **{f"{d}_nodes": r[f"{d}_nodes"] for d in dets2},
    } for r in rows])

    print("\nS2 真实拥塞 (congested_clean 无攻击者: 全 17 节点皆诚实, 任何报警都是"
          " 误报; congested_attack: TPR 分母=1, FPR 分母=16)")
    print(f"{'scenario':>28s} | " + " | ".join(
        f"{d:>26s}" for d in dets2))
    print("-" * 140)
    for r in rows:
        n_h = r["n_honest"]
        line = f"{r['scenario']:>28s} | "
        for d in dets2:
            tp, fp = r[d]
            tpr_s = f"TPR {tp}/1 " if tp or "attack" in r["scenario"] else "  --    "
            line += f"{tpr_s} FPR {fmt_rate(fp, n_h):>14s} | "
        print(line)
    print("\n被标记节点 (clean-fit 三基线在无攻击拥塞 run 即指控 node 12 ="
          " 归因失败演示):")
    for r in rows:
        print(f"  {r['scenario']:>28s}: node12 dropped={r['node12_dropped']}, "
              f"honest max dropped={r['honest_max_dropped']}")
        for d in dets2:
            print(f"    {d:>13s}: [{r[f'{d}_nodes']}]")
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    h = choose_h_chain(LAMBDA0, DELTA, TARGET_ARL)
    s1, detail = part_s1(h)
    s2 = part_s2(h)
    s1.to_csv(os.path.join(OUT, "baseline_sanity_s1.csv"), index=False)
    detail.to_csv(os.path.join(OUT, "baseline_sanity_detail.csv"), index=False)
    s2.to_csv(os.path.join(OUT, "baseline_sanity_s2.csv"), index=False)
    with open(os.path.join(OUT, "baseline_sanity.txt"), "w") as f:
        f.write("M3.2 baseline reimplementation sanity check\n")
        f.write(f"protocol: lambda0={LAMBDA0}, delta={DELTA}, ARL0={TARGET_ARL:.0f},"
                f" h={h:.2f}, trust(w={TRUST_WINDOW},tau={TAU_TRUST},"
                f"min_att={TRUST_MIN_ATTEMPTS}), CLIF q={CLIF_Q}\n\n")
        f.write("=== S1: congestion-free, 70 multiseed attack runs ===\n")
        f.write(s1.to_csv(index=False))
        f.write("\n=== S2: real simulator congestion ===\n")
        f.write(s2.to_csv(index=False))
    print(f"\nDONE. outputs in {OUT}")


if __name__ == "__main__":
    main()
