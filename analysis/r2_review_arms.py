#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-2 (评审二轮): 编辑综合报告要求的新对照实验臂.

对应 review synthesis 共识#2 / DA-C3 / R3-C1:
  P1. 突发(非泊松)拥塞混淆 — MMPP(2 态马尔可夫调制)注入, 均值与 λ_cong 匹配,
      测试混淆过程偏离 CUSUM 名义模型(泊松)时的行为, 即论文自认未覆盖的方向
      (过散诚实残差). 含 varmatch 变体 (λ0 取方差匹配率) 与 GRU 学习型期望.
  P2. 残差输入基线 — 与 v2 完全同一测试注入流 (SEED_TEST+int(lam*1000)):
      resid+naive-threshold (oracle/est), resid-input AE (clean/refit).
      量化「期望模型 vs 决策规则」的贡献分解.
  P3. 学习型期望模型 — GRU 逐时隙期望丢包预测器 (LSTM-NDT 家族), 正部残差 →
      同款链反演 CUSUM; clean 训练 / 每级 refit 训练两种口径, 泊松与突发两协议.

协议与 congestion_deoracle.py 严格一致 (加性后处理注入, 攻击序列叠加 node12,
Wilson 95% CI, RNG=numpy default_rng PCG64). 输出: comparison/r2_review_arms.csv
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
import learning_vae as lv                                      # noqa: E402

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "analysis"))
OUT = os.path.join(BASE, "comparison")
MAL = 12
DELTA = 1.0
TARGET_ARL = 1e5
FLOOR = 0.01
N_TRIALS = 200
LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
N_REFIT = 20
EST_WIN = 50
Z95 = 1.96
W_AE, STRIDE_AE = 16, 4
# 与 v2 相同的流 (P2 同实现对比); 突发协议用独立新流
SEED_TEST = 7_700_000
SEED_REFIT = 8_800_000
SEED_PRIOR = 9_900_000
SEED_AE = 42
SEED_BTEST = 7_600_000
SEED_BREFIT = 8_600_000
SEED_BPRIOR = 9_600_000
# MMPP 参数: 平稳 ON 概率 pi=0.2, 平均 ON 段长 10 时隙 → 均值 = lam
MMPP_PI = 0.2
MMPP_MEAN_ON = 10.0


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
    out = []
    for n in nodes:
        d, r = np.asarray(drop[n], float), np.asarray(rx[n], float)
        att = r + d
        for s in range(0, len(d) - window + 1):
            a = att[s:s + window].sum()
            if a >= min_att:
                out.append(r[s:s + window].sum() / a)
    return out


def mmpp(rng, lam, size_nodes, n_slots, pi=MMPP_PI, mean_on=MMPP_MEAN_ON):
    """2 态马尔可夫调制泊松: 平稳均值 = lam, 时间相关 + 过散."""
    b = 1.0 / mean_on
    a = b * pi / (1 - pi)
    r_on = lam / pi if lam > 0 else 0.0
    on = rng.random(size_nodes) < pi
    out = np.zeros((size_nodes, n_slots))
    for t in range(n_slots):
        out[:, t] = rng.poisson(np.where(on, r_on, 0.0))
        u = rng.random(size_nodes)
        on = np.where(on, u > b, u < a)
    return out


# ----------------------------------------------------------------------
# GRU 学习型期望模型 (LSTM-NDT 家族的逐时隙预测器)
# ----------------------------------------------------------------------
class GRUPred(nn.Module):
    def __init__(self, nf=4, hidden=32):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1, :]).squeeze(-1)


def gru_train(ref, RX, UT, HO, nodes, seed, epochs=200, lr=1e-3, W=16):
    """ref: (n_refit, n_nodes, slots) 无攻击实现 → 监督样本 (窗口→下一时隙 drop)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs, ys = [], []
    for k in range(ref.shape[0]):
        for j in range(len(nodes)):
            d = ref[k, j]
            for t in range(W, ref.shape[2]):
                Xs.append(np.column_stack([
                    d[t - W:t], RX[j, t - W:t], UT[j, t - W:t], HO[j, t - W:t]]))
                ys.append(d[t])
    Xa = np.asarray(Xs, dtype=np.float32)
    ya = np.asarray(ys, dtype=np.float32)
    mu, sd = Xa.reshape(-1, 4).mean(0), Xa.reshape(-1, 4).std(0)
    sd[sd < 1e-6] = 1.0
    model = GRUPred().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor((Xa - mu) / sd)
    yt = torch.tensor(ya)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, 256):
            idx = perm[s:s + 256]
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 50 == 0:
            print(f"    [gru] ep{ep+1} mse={tot/n:.4f}")
    model.eval()
    return model, mu, sd, dev


def gru_residual(model, mu, sd, dev, series_j, RXj, UTj, HOj, W=16):
    """逐时隙期望预测 → 正部残差序列 (t<W 用训练均值预测, 即恒定 λ̂)."""
    d = np.asarray(series_j, float)
    T = len(d)
    res = np.zeros(T)
    if T > W:
        wins = np.stack([
            np.column_stack([d[t - W:t], RXj[t - W:t], UTj[t - W:t],
                             HOj[t - W:t]]) for t in range(W, T)])
        Xn = (wins - mu) / sd
        with torch.no_grad():
            pred = model(torch.tensor(Xn, dtype=torch.float32).to(dev))
            pred = pred.cpu().numpy()
        res[W:] = d[W:] - np.maximum(0.0, pred)
    mu0 = float(mu[0])
    res[:W] = d[:W] - max(0.0, mu0)
    return res


def gru_residual_batch(model, mu, sd, dev, X, RX, UT, HO, W=16):
    """X: (n_trials, n_nodes, slots) → 正部残差同形数组 (窗口批量前向)."""
    n_tr, n_n, T = X.shape
    out = np.zeros_like(X)
    mu0 = float(mu[0])
    # 收集全部窗口 (trial, node, t)
    idx, wins = [], []
    for k in range(n_tr):
        for j in range(n_n):
            d = X[k, j]
            for t in range(W, T):
                idx.append((k, j, t))
                wins.append(np.column_stack([d[t - W:t], RX[j, t - W:t],
                                             UT[j, t - W:t], HO[j, t - W:t]]))
    if wins:
        Xn = (np.asarray(wins, dtype=np.float32) - mu) / sd
        with torch.no_grad():
            pred = []
            for s in range(0, len(Xn), 8192):
                xb = torch.tensor(Xn[s:s + 8192]).to(dev)
                pred.append(model(xb).cpu().numpy())
            pred = np.concatenate(pred)
        for (k, j, t), p in zip(idx, pred):
            out[k, j, t] = X[k, j, t] - max(0.0, p)
    out[:, :, :W] = X[:, :, :W] - max(0.0, mu0)
    return out


# ----------------------------------------------------------------------
# AE (残差输入): 与 v2 相同的 lv 工具, 仅把通道 0 换为残差
# ----------------------------------------------------------------------
def resid_series_map(X_k, lam_hat, RX, UT, HO, nodes):
    return {n: np.column_stack([X_k[j] - lam_hat, RX[j], UT[j], HO[j]])
            .astype(np.float32) for j, n in enumerate(nodes)}


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
    B = np.array([drop_b[n] for n in nodes])
    RX = np.array([rx_b[n] for n in nodes], float)
    UT = np.array([ut_b[n] for n in nodes], float)
    HO = np.array([ho_b[n] for n in nodes], float)
    i12 = nodes.index(MAL)
    idx_h = [nodes.index(n) for n in honest]

    # 干净标定 (与 v2 相同)
    tau_naive_c = max(float(drop_b[n].sum()) for n in honest) + 0.5
    F_c = features(drop_b, rx_b, ut_b, ho_b, nodes)
    mu_c, covinv_c = fit_gaussian(F_c)
    thr_clif_c = threshold_from_quantile(mahalanobis(F_c, mu_c, covinv_c), 0.99)

    rows = []

    def add(level, proto, variant, tp, fp, npos, nneg, note=""):
        tl, th = wilson(tp, npos)
        fl, fh = wilson(fp, nneg)
        rows.append(dict(lam_cong=level, proto=proto, variant=variant,
                         tpr=round(tp / npos, 4) if npos else float("nan"),
                         fpr=round(fp / nneg, 4) if nneg else float("nan"),
                         fpr_lo=round(fl, 4), fpr_hi=round(fh, 4),
                         tpr_lo=round(tl, 4), tpr_hi=round(th, 4),
                         n_pos=npos, n_neg=nneg, note=note))

    def make_X(rng_gen, lam, seed, n_trials, n_refit=0):
        rng = np.random.default_rng(seed)
        if rng_gen == "poisson":
            inj = rng.poisson(lam, size=(n_trials, len(nodes), nslots))
        else:
            inj = np.stack([mmpp(rng, lam, len(nodes), nslots)
                            for _ in range(n_trials)])
        X = B[None, :, :] + inj.astype(float)
        return X

    # =================================================================
    for lam in LEVELS:
        lam_eff = lam if lam > 0 else FLOOR
        npos, nneg = N_TRIALS, N_TRIALS * len(honest)

        # ---------------- P2: 泊松协议, 与 v2 同测试流 ----------------
        X = make_X("poisson", lam, SEED_TEST + int(lam * 1000), N_TRIALS)
        X[:, i12, :] += attack[None, :]

        # λ̂ (win-est, 与 v2 同流同值)
        rng_p = np.random.default_rng(SEED_PRIOR + int(lam * 1000))
        prior = rng_p.poisson(lam, size=(len(honest), EST_WIN))
        lam_hat = max(FLOOR, float(prior.sum()) / (len(honest) * EST_WIN))

        # refit 无攻击实现 (与 v2 同流)
        rng_r = np.random.default_rng(SEED_REFIT + int(lam * 1000))
        ref = (rng_r.poisson(lam, size=(N_REFIT, len(nodes), nslots))
               .astype(float) + B[None, :, :])

        # (a) resid + naive threshold, oracle / est
        for tag, lh in [("oracle", lam_eff), ("est", lam_hat)]:
            R_ref = ref[:, idx_h, :].sum(axis=2) - lh * nslots
            tau_r = float(R_ref.max()) + 0.5
            R_test = X.sum(axis=2) - lh * nslots
            tp = int((R_test[:, i12] > tau_r).sum())
            fp = int((R_test[:, idx_h] > tau_r).sum())
            add(lam, "poisson", f"resid-thr-{tag}", tp, fp, npos, nneg,
                note=f"tau={tau_r:.1f}, lam0={lh:.3f}")

        # (b) resid-input AE: clean 训练 + 每级 refit 训练
        ser_clean = {n: np.column_stack([drop_b[n] - FLOOR, rx_b[n], ut_b[n],
                                         ho_b[n]]).astype(np.float32)
                     for n in nodes}
        Xtr, _ = lv.make_windows(ser_clean, W_AE, STRIDE_AE)
        m_c, s_c = lv.fit_scaler(Xtr)
        Xf = lv.apply_scaler(Xtr, m_c, s_c).reshape(len(Xtr), -1)
        lv.set_seed(SEED_AE)
        mdl = lv.train_model(Xf, Xf.shape[1])
        sc = np.array(list(lv.node_scores(ser_clean, mdl, m_c, s_c,
                                          W_AE, STRIDE_AE).values()))
        thr_c = float(np.max(sc)) + 1e-6
        tp = fp = 0
        for k in range(N_TRIALS):
            ser = resid_series_map(X[k], lam_hat, RX, UT, HO, nodes)
            verd = lv.detect(ser, mdl, m_c, s_c, W_AE, STRIDE_AE, thr_c)
            tp += int(bool(verd.get(MAL, (False,))[0]))
            fp += sum(1 for n in honest if verd.get(n, (False,))[0])
        add(lam, "poisson", "resid-ae-clean", tp, fp, npos, nneg,
            note=f"thr={thr_c:.1f}")

        N_AR = 5
        lv.set_seed(SEED_AE + 50 + int(lam * 1000))
        ws, sers = [], []
        for k in range(N_AR):
            ser_r = resid_series_map(ref[k], lam_hat, RX, UT, HO, nodes)
            Xw, _ = lv.make_windows(ser_r, W_AE, STRIDE_AE)
            ws.append(Xw)
            sers.append(ser_r)
        Xtr_r = np.concatenate(ws, axis=0)
        m_r, s_r = lv.fit_scaler(Xtr_r)
        Xf_r = lv.apply_scaler(Xtr_r, m_r, s_r).reshape(len(Xtr_r), -1)
        mdl_r = lv.train_model(Xf_r, Xf_r.shape[1])
        sc_r = np.concatenate(
            [np.array(list(lv.node_scores(s, mdl_r, m_r, s_r,
                                          W_AE, STRIDE_AE).values()))
             for s in sers])
        thr_rr = float(np.max(sc_r)) + 1e-6
        tp = fp = 0
        for k in range(N_TRIALS):
            ser = resid_series_map(X[k], lam_hat, RX, UT, HO, nodes)
            verd = lv.detect(ser, mdl_r, m_r, s_r, W_AE, STRIDE_AE, thr_rr)
            tp += int(bool(verd.get(MAL, (False,))[0]))
            fp += sum(1 for n in honest if verd.get(n, (False,))[0])
        add(lam, "poisson", "resid-ae-refit", tp, fp, npos, nneg,
            note=f"thr={thr_rr:.1f}")

        # (c) GRU 学习型期望 (泊松协议): refit 训练 + est λ0 (正部残差)
        gru, gmu, gsd, gdev = gru_train(ref, RX, UT, HO, nodes,
                                        seed=SEED_AE + 200 + int(lam * 1000))
        Rg_ref = np.maximum(
            gru_residual_batch(gru, gmu, gsd, gdev, ref, RX, UT, HO), 0.0)
        lam0_g = max(FLOOR, float(Rg_ref[:, idx_h, :].mean()))
        h_g = choose_h_chain(lam0_g, DELTA, TARGET_ARL)
        Rg_test = np.maximum(
            gru_residual_batch(gru, gmu, gsd, gdev, X, RX, UT, HO), 0.0)
        al = cusum_batch(Rg_test, lam0_g, DELTA, h_g)
        add(lam, "poisson", "gru-est-refit", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"lam0={lam0_g:.4f}, h={h_g:.3f}")
        print(f"[done] poisson lam={lam}")

        # ---------------- P1: 突发 (MMPP) 协议, 均值匹配 ----------------
        Xb = make_X("mmpp", lam, SEED_BTEST + int(lam * 1000), N_TRIALS)
        Xb[:, i12, :] += attack[None, :]
        rng_br = np.random.default_rng(SEED_BREFIT + int(lam * 1000))
        refb = np.stack([mmpp(rng_br, lam, len(nodes), nslots)
                         for _ in range(N_REFIT)]).astype(float) + B[None, :, :]

        # 先验窗口估计 (突发流)
        rng_bp = np.random.default_rng(SEED_BPRIOR + int(lam * 1000))
        prior_b = mmpp(rng_bp, lam, len(honest), EST_WIN)
        lam_hat_b = max(FLOOR, float(prior_b.sum()) / (len(honest) * EST_WIN))

        # (a) ours-oracle: λ0=λ (泊松设计 vs 突发真实)
        h_o = choose_h_chain(lam_eff, DELTA, TARGET_ARL)
        al = cusum_batch(Xb, lam_eff, DELTA, h_o)
        add(lam, "mmpp", "ours-oracle", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg, note=f"h={h_o:.3f}")

        # (b) ours-est: 突发先验窗口 λ̂
        h_w = choose_h_chain(lam_hat_b, DELTA, TARGET_ARL)
        al = cusum_batch(Xb, lam_hat_b, DELTA, h_w)
        add(lam, "mmpp", "ours-est", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"lam_hat={lam_hat_b:.4f}, h={h_w:.3f}")

        # (c) ours-est-varmatch: λ0 = 方差/均值 (过散矩匹配, 逐级 refit 估方差)
        vm = refb[:, idx_h, :].var(axis=(0, 2)).mean()
        lam_vm = max(FLOOR, float(vm) / lam_eff) if lam > 0 else FLOOR
        h_v = choose_h_chain(lam_vm, DELTA, TARGET_ARL)
        al = cusum_batch(Xb, lam_vm, DELTA, h_v)
        add(lam, "mmpp", "ours-est-varmatch", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"lam0={lam_vm:.3f}, h={h_v:.3f}")

        # (d) clean-fit 基线 (naive/trust/clif)
        tp_n = fp_n = tp_t = fp_t = tp_c = fp_c = 0
        for k in range(N_TRIALS):
            ser = {n: Xb[k, j] for j, n in enumerate(nodes)}
            rx_s = {n: RX[j] for j, n in enumerate(nodes)}
            vn = naive_threshold(ser, tau_naive_c)
            vt = trust_detect(ser, rx_s, 10, 0.99, 50)
            F = features(ser, rx_s, {n: UT[j] for j, n in enumerate(nodes)},
                         {n: HO[j] for j, n in enumerate(nodes)}, nodes)
            vc = dict(zip(nodes, mahalanobis(F, mu_c, covinv_c) > thr_clif_c))
            tp_n += int(bool(vn.get(MAL, False)))
            fp_n += sum(1 for n in honest if vn.get(n, False))
            tp_t += int(bool(vt.get(MAL, False)))
            fp_t += sum(1 for n in honest if vt.get(n, False))
            tp_c += int(bool(vc.get(MAL, False)))
            fp_c += sum(1 for n in honest if vc.get(n, False))
        add(lam, "mmpp", "naive-cleanfit", tp_n, fp_n, npos, nneg)
        add(lam, "mmpp", "trust-cleanfit", tp_t, fp_t, npos, nneg)
        add(lam, "mmpp", "clif-cleanfit", tp_c, fp_c, npos, nneg)

        # (e) GRU 学习型期望 (突发协议, 突发 refit 训练)
        gru_b, gmu_b, gsd_b, gdev_b = gru_train(
            refb, RX, UT, HO, nodes, seed=SEED_AE + 400 + int(lam * 1000))
        Rg_refb = np.maximum(
            gru_residual_batch(gru_b, gmu_b, gsd_b, gdev_b, refb, RX, UT, HO),
            0.0)
        lam0_gb = max(FLOOR, float(Rg_refb[:, idx_h, :].mean()))
        h_gb = choose_h_chain(lam0_gb, DELTA, TARGET_ARL)
        Rg_testb = np.maximum(
            gru_residual_batch(gru_b, gmu_b, gsd_b, gdev_b, Xb, RX, UT, HO),
            0.0)
        al = cusum_batch(Rg_testb, lam0_gb, DELTA, h_gb)
        add(lam, "mmpp", "gru-est-refit", int(al[:, i12].sum()),
            int(al[:, idx_h].sum()), npos, nneg,
            note=f"lam0={lam0_gb:.4f}, h={h_gb:.3f}")
        print(f"[done] mmpp lam={lam}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "r2_review_arms.csv"), index=False)
    with open(os.path.join(OUT, "r2_review_arms.txt"), "w") as f:
        f.write("R-2 review arms: residual-input baselines (poisson, same "
                "stream as v2) + MMPP bursty confusion (matched mean, "
                f"pi={MMPP_PI}, mean_ON={MMPP_MEAN_ON}) + GRU learned "
                "expectation\n")
        f.write(f"N_TRIALS={N_TRIALS}, ARL0*={TARGET_ARL:.0e}, "
                f"seeds test={SEED_TEST}/{SEED_BTEST}+ refit="
                f"{SEED_REFIT}/{SEED_BREFIT}+ prior={SEED_PRIOR}/"
                f"{SEED_BPRIOR}+\n\n")
        f.write(df.to_csv(index=False))
    print("\nFPR% by (proto, variant):")
    for (p, v), sub in df.groupby(["proto", "variant"]):
        sub = sub.sort_values("lam_cong")
        line = "  ".join(f"{r.lam_cong}:{100*r.fpr:.2f}"
                         for r in sub.itertuples())
        print(f"{p:8s} {v:20s} {line}")
    print("\nDONE ->", os.path.join(OUT, "r2_review_arms.csv"))


if __name__ == "__main__":
    main()
