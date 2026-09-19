#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审稿修改 R2/R7: 拥塞混淆研究的去 oracle 与同等待遇 (like-for-like) 重跑.

回应 (editorial R2 / R0-W2 / R1-W3 / R2-W2 / DA C-2 M-4, R1-W5):
  A. 去oracle λ0: 防御者不再被告知 λ_cong —
     A1 win-est : λ̂ 由同速率的「先前无攻击窗口」(50 时隙, 诚实节点合并) 估计;
     A2 xfer    : λ̂ 在 λ=0.1 处标定, 迁移应用到所有速率级 (非平稳/速率漂移);
     A3 factor  : λ0 = f·λ_cong, f∈{0.5,0.75,1.25,2} 误估敏感性扫描.
  B. 同等待遇重拟合 (refit): 朴素阈值/信任/马氏 按每级「拥塞但无攻击」数据
     重新标定阈值后再评估 (与 ours 获得的逐级重反演对等).
  C. AE 大样本: 干净训练 AE 以 n=200 注入 (3,200 诚实节点观测) 计分,
     并给出 AE-per-level-refit (逐级在拥塞无攻击窗口上重训练) 对照.
  D. ours-oracle 一并按修正吸收判据重跑 (h=修正后逐级反演), 作为同源基准.

协议与 run_comparison.part_b 严格一致 (加性 Poisson(λ_cong) 后处理注入,
node12 叠加真实攻击序列); RNG: numpy default_rng (PCG64), 种子表见 SEED_*.
输出: analysis/comparison/congestion_confusion_v2.csv (+ 控制台摘要)
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import choose_h_chain                      # noqa: E402
from baselines import (aggregate_node_field, naive_threshold,  # noqa: E402
                       trust_detect, fit_gaussian, mahalanobis,
                       threshold_from_quantile)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "analysis"))
OUT = os.path.join(BASE, "comparison")
MAL = 12
DELTA = 1.0
TARGET_ARL = 1e5
FLOOR = 0.01
N_TRIALS = 200
LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
FACTORS = [0.5, 0.75, 1.25, 2.0]
EST_WIN = 50                     # 先前无攻击窗口长度 (时隙)
N_REFIT = 20                     # 每级重拟合用无攻击实现数
Z95 = 1.96
W_AE, STRIDE_AE = 16, 4
# RNG 种子表 (与旧 run_comparison 的 1000+... 区分; 全程 default_rng=PCG64)
SEED_TEST = 7_700_000            # 测试注入
SEED_REFIT = 8_800_000           # 重拟合无攻击实现
SEED_PRIOR = 9_900_000           # λ̂ 先验窗口
SEED_AE = 42                     # AE 训练

try:
    import learning_vae as lv
    HAS_AE = True
except Exception as e:                                   # torch 不可用时降级
    print("[warn] learning_vae unavailable -> AE columns skipped:", e)
    HAS_AE = False


def wilson(k, n, z=Z95):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return max(0., c - hw), min(1., c + hw)


def cusum_batch(X, lam0, delta, h):
    """向量化 CUSUM (与 cusum_detector.run_cusum 逐条等价).
    X: (n_trials, n_nodes, n_slots) → alarmed: (n_trials, n_nodes) bool."""
    logr = math.log((lam0 + delta) / lam0)
    S = np.zeros(X.shape[:2])
    alarmed = np.zeros(X.shape[:2], dtype=bool)
    for t in range(X.shape[2]):
        S = np.maximum(0.0, S + X[:, :, t] * logr - delta)
        alarmed |= S > h
    return alarmed


def features(drop, rx, util, ho, nodes):
    return np.array([[float(drop[n].mean()), float(rx[n].mean()),
                      float(util[n].mean()), float(ho[n].mean()),
                      float(drop[n].max())] for n in nodes])


def ratios_all(drop, rx, nodes, window=10, min_att=50):
    """所有 (节点, 窗口) 的转发比列表 (信任类重拟合用)."""
    out = []
    for n in nodes:
        d, r = np.asarray(drop[n], float), np.asarray(rx[n], float)
        att = r + d
        for s in range(0, len(d) - window + 1):
            a = att[s:s + window].sum()
            if a >= min_att:
                out.append(r[s:s + window].sum() / a)
    return out


def main():
    base_df = pd.read_csv(os.path.join(BASE, "residuals_baseline.csv"))
    att_df = pd.read_csv(os.path.join(BASE, "residuals_const12.csv"))
    drop_b = aggregate_node_field(base_df, "dropped")
    rx_b = aggregate_node_field(base_df, "rx")
    ut_b = aggregate_node_field(base_df, "util")
    ho_b = aggregate_node_field(base_df, "handover_events")
    drop_a = aggregate_node_field(att_df, "dropped")
    nodes = sorted(drop_b)
    honest = [n for n in nodes if n != MAL]
    nslots = len(drop_b[nodes[0]])
    attack = drop_a.get(MAL, np.zeros(nslots))
    B = np.array([drop_b[n] for n in nodes])                 # (n_nodes, slots)
    RX = np.array([rx_b[n] for n in nodes], float)
    UT = np.array([ut_b[n] for n in nodes], float)
    HO = np.array([ho_b[n] for n in nodes], float)
    i12 = nodes.index(MAL)

    # ---- 干净标定 (clean-fit 基线, 与原协议一致) ----
    tau_naive_c = max(float(drop_b[n].sum()) for n in honest) + 0.5
    F_c = features(drop_b, rx_b, ut_b, ho_b, nodes)
    mu_c, covinv_c = fit_gaussian(F_c)
    thr_clif_c = threshold_from_quantile(mahalanobis(F_c, mu_c, covinv_c), 0.99)

    # ---- AE: 干净训练 (n=200 注入计分) ----
    ae_model = ae_mean = ae_std = ae_thr = None
    if HAS_AE:
        ser_clean = lv.node_series(os.path.join(BASE, "residuals_baseline.csv"))
        Xtr, _ = lv.make_windows(ser_clean, W_AE, STRIDE_AE)
        ae_mean, ae_std = lv.fit_scaler(Xtr)
        Xf = lv.apply_scaler(Xtr, ae_mean, ae_std).reshape(len(Xtr), -1)
        lv.set_seed(SEED_AE)
        ae_model = lv.train_model(Xf, Xf.shape[1])
        sc = lv.node_scores(ser_clean, ae_model, ae_mean, ae_std, W_AE, STRIDE_AE)
        ae_thr = lv.threshold_from_scores(sc)
        print(f"[AE] clean-trained, threshold={ae_thr:.2f}")

    rows = []

    def add(level, variant, tp, fp, npos, nneg, note=""):
        tpr = tp / npos if npos else float("nan")
        fpr = fp / nneg if nneg else float("nan")
        tl, th = wilson(tp, npos)
        fl, fh = wilson(fp, nneg)
        rows.append(dict(lam_cong=level, variant=variant,
                         tpr=round(tpr, 4), tpr_lo=round(tl, 4),
                         tpr_hi=round(th, 4),
                         fpr=round(fpr, 4), fpr_lo=round(fl, 4),
                         fpr_hi=round(fh, 4),
                         n_pos=npos, n_neg=nneg, note=note))

    for lam in LEVELS:
        lam_eff = lam if lam > 0 else FLOOR
        rng = np.random.default_rng(SEED_TEST + int(lam * 1000))
        inj = rng.poisson(lam, size=(N_TRIALS, len(nodes), nslots)).astype(float)
        X = B[None, :, :] + inj
        X[:, i12, :] += attack[None, :]
        npos, nneg = N_TRIALS, N_TRIALS * len(honest)
        idx_h = [nodes.index(n) for n in honest]

        # ---- D. ours-oracle (修正判据逐级 h) ----
        h_o = choose_h_chain(lam_eff, DELTA, TARGET_ARL)
        al = cusum_batch(X, lam_eff, DELTA, h_o)
        add(lam, "ours-oracle", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"h={h_o:.3f}")

        # ---- A1. win-est: 先前无攻击窗口估计 λ̂ ----
        rng_p = np.random.default_rng(SEED_PRIOR + int(lam * 1000))
        prior = rng_p.poisson(lam, size=(len(honest), EST_WIN))
        lam_hat = max(FLOOR, float(prior.sum()) / (len(honest) * EST_WIN))
        h_w = choose_h_chain(lam_hat, DELTA, TARGET_ARL)
        al = cusum_batch(X, lam_hat, DELTA, h_w)
        add(lam, "ours-win-est", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"lam_hat={lam_hat:.4f}, h={h_w:.3f}")

        # ---- A2. xfer: λ̂ 在 λ=0.1 标定, 迁移到本级 ----
        rng_x = np.random.default_rng(SEED_PRIOR + 100)
        prior_x = rng_x.poisson(0.1, size=(len(honest), EST_WIN))
        lam_x = max(FLOOR, float(prior_x.sum()) / (len(honest) * EST_WIN))
        h_x = choose_h_chain(lam_x, DELTA, TARGET_ARL)
        al = cusum_batch(X, lam_x, DELTA, h_x)
        add(lam, "ours-xfer(0.1)", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"lam_hat={lam_x:.4f} (fixed), h={h_x:.3f}")

        # ---- A3. 误估因子扫描 ----
        if lam > 0:
            for f in FACTORS:
                lf = max(FLOOR, f * lam_eff)
                h_f = choose_h_chain(lf, DELTA, TARGET_ARL)
                al = cusum_batch(X, lf, DELTA, h_f)
                add(lam, f"ours-factor-{f}", int(al[:, i12].sum()),
                    int(al[:, idx_h].sum()), npos, nneg,
                    note=f"lam0={lf:.3f} (f={f}), h={h_f:.3f}")

        # ---- clean-fit 基线 (原协议) ----
        tp_n = fp_n = tp_t = fp_t = tp_c = fp_c = 0
        for k in range(N_TRIALS):
            ser = {n: X[k, j] for j, n in enumerate(nodes)}
            rx_s = {n: RX[j] for j, n in enumerate(nodes)}
            vn = naive_threshold(ser, tau_naive_c)
            vt = trust_detect(ser, rx_s, 10, 0.99, 50)
            F = features(ser, rx_s, {n: UT[j] for j, n in enumerate(nodes)},
                         {n: HO[j] for j, n in enumerate(nodes)}, nodes)
            vc = dict(zip(nodes, mahalanobis(F, mu_c, covinv_c) > thr_clif_c))
            for verdict, cnt in [(vn, "n"), (vt, "t"), (vc, "c")]:
                tp = int(bool(verdict.get(MAL, False)))
                fp = sum(1 for n in honest if verdict.get(n, False))
                if cnt == "n":
                    tp_n += tp; fp_n += fp
                elif cnt == "t":
                    tp_t += tp; fp_t += fp
                else:
                    tp_c += tp; fp_c += fp
        add(lam, "naive-cleanfit", tp_n, fp_n, npos, nneg)
        add(lam, "trust-cleanfit", tp_t, fp_t, npos, nneg)
        add(lam, "clif-cleanfit", tp_c, fp_c, npos, nneg)

        # ---- B. refit (同等待遇): 每级在拥塞无攻击实现上重标定 ----
        rng_r = np.random.default_rng(SEED_REFIT + int(lam * 1000))
        ref = rng_r.poisson(lam, size=(N_REFIT, len(nodes), nslots)).astype(float)
        ref = ref + B[None, :, :]
        # naive
        tau_n = max(float(ref[:, j, :].sum(axis=1).max())
                    for j in idx_h) + 0.5
        # trust: 重拟合实现的窗口转发比 1% 分位
        rats = []
        for k in range(N_REFIT):
            ser = {n: ref[k, j] for j, n in enumerate(nodes)}
            rx_s = {n: RX[j] for j, n in enumerate(nodes)}
            rats += ratios_all(ser, rx_s, honest)
        tau_t = float(np.percentile(rats, 1)) if rats else 0.99
        # clif: 逐 (节点, 实现) 特征行 (20×17 行) — 协方差在实现分布内估计,
        # 跨节点汇总会使注入设定下协方差退化 (各诚实节点同分布) 而虚高 FPR
        Fr = np.vstack([features({n: ref[k, j] for j, n in enumerate(nodes)},
                                 rx_s0 := {n: RX[j] for j, n in enumerate(nodes)},
                                 {n: UT[j] for j, n in enumerate(nodes)},
                                 {n: HO[j] for j, n in enumerate(nodes)},
                                 nodes) for k in range(N_REFIT)])
        mu_r, covinv_r = fit_gaussian(Fr)
        thr_r = threshold_from_quantile(mahalanobis(Fr, mu_r, covinv_r), 0.99)

        tp_n = fp_n = tp_t = fp_t = tp_c = fp_c = 0
        for k in range(N_TRIALS):
            ser = {n: X[k, j] for j, n in enumerate(nodes)}
            rx_s = {n: RX[j] for j, n in enumerate(nodes)}
            vn = naive_threshold(ser, tau_n)
            vt = trust_detect(ser, rx_s, 10, tau_t, 50)
            F = features(ser, rx_s, {n: UT[j] for j, n in enumerate(nodes)},
                         {n: HO[j] for j, n in enumerate(nodes)}, nodes)
            vc = dict(zip(nodes, mahalanobis(F, mu_r, covinv_r) > thr_r))
            for verdict, cnt in [(vn, "n"), (vt, "t"), (vc, "c")]:
                tp = int(bool(verdict.get(MAL, False)))
                fp = sum(1 for n in honest if verdict.get(n, False))
                if cnt == "n":
                    tp_n += tp; fp_n += fp
                elif cnt == "t":
                    tp_t += tp; fp_t += fp
                else:
                    tp_c += tp; fp_c += fp
        add(lam, "naive-refit", tp_n, fp_n, npos, nneg, note=f"tau={tau_n:.1f}")
        add(lam, "trust-refit", tp_t, fp_t, npos, nneg, note=f"tau={tau_t:.4f}")
        add(lam, "clif-refit", tp_c, fp_c, npos, nneg)

        # ---- C. AE: 干净训练 @ n=200; 逐级 refit ----
        if HAS_AE:
            tp = fp = 0
            for k in range(N_TRIALS):
                ser = {n: np.column_stack([X[k, j], RX[j], UT[j], HO[j]])
                       .astype(np.float32) for j, n in enumerate(nodes)}
                verd = lv.detect(ser, ae_model, ae_mean, ae_std,
                                 W_AE, STRIDE_AE, ae_thr)
                tp += int(bool(verd.get(MAL, (False,))[0]))
                fp += sum(1 for n in honest if verd.get(n, (False,))[0])
            add(lam, "ae-cleanfit-n200", tp, fp, npos, nneg)

            # AE 逐级重训练 (拥塞无攻击窗口; 用 N_AE_REFIT 个独立实现,
            # 阈值取这些实现自身分数的最大值 — 避免单实现均值导致阈值过紧)
            N_AE_REFIT = 5
            lv.set_seed(SEED_AE + 1 + int(lam * 1000))
            ws, sers = [], []
            for k in range(N_AE_REFIT):
                ser_r = {n: np.column_stack(
                    [ref[k, j], RX[j], UT[j], HO[j]]).astype(np.float32)
                    for j, n in enumerate(nodes)}
                Xw, _ = lv.make_windows(ser_r, W_AE, STRIDE_AE)
                ws.append(Xw)
                sers.append(ser_r)
            Xtr_r = np.concatenate(ws, axis=0)
            m_r, s_r = lv.fit_scaler(Xtr_r)
            Xf_r = lv.apply_scaler(Xtr_r, m_r, s_r).reshape(len(Xtr_r), -1)
            model_r = lv.train_model(Xf_r, Xf_r.shape[1])
            sc_r = np.concatenate(
                [np.array(list(lv.node_scores(s, model_r, m_r, s_r,
                                              W_AE, STRIDE_AE).values()))
                 for s in sers])
            thr_r_ae = float(np.max(sc_r))
            tp = fp = 0
            for k in range(N_TRIALS):
                ser = {n: np.column_stack([X[k, j], RX[j], UT[j], HO[j]])
                       .astype(np.float32) for j, n in enumerate(nodes)}
                verd = lv.detect(ser, model_r, m_r, s_r, W_AE, STRIDE_AE, thr_r_ae)
                tp += int(bool(verd.get(MAL, (False,))[0]))
                fp += sum(1 for n in honest if verd.get(n, (False,))[0])
            add(lam, "ae-refit-n200", tp, fp, npos, nneg,
                note=f"thr={thr_r_ae:.2f}")
        print(f"[done] lam={lam}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "congestion_confusion_v2.csv"), index=False)
    with open(os.path.join(OUT, "congestion_confusion_v2.txt"), "w") as f:
        f.write("R2/R7 de-oracle + like-for-like congestion study "
                f"(N_TRIALS={N_TRIALS}, RNG=numpy default_rng PCG64, "
                f"seeds test={SEED_TEST}+ refit={SEED_REFIT}+ "
                f"prior={SEED_PRIOR}+)\n")
        f.write(f"corrected absorption convention; delta={DELTA}; "
                f"ARL0*={TARGET_ARL:.0e}; est_win={EST_WIN} slots\n\n")
        f.write(df.to_csv(index=False))
    print("\nFPR% (tpr) summary by variant:")
    for v in df["variant"].unique():
        sub = df[df["variant"] == v].sort_values("lam_cong")
        line = "  ".join(f"{r.lam_cong}:{100*r.fpr:.2f}"
                         for r in sub.itertuples())
        print(f"{v:22s} {line}")
    print("\nDONE ->", os.path.join(OUT, "congestion_confusion_v2.csv"))


if __name__ == "__main__":
    main()
