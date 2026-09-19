#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.4-P0-① 混淆公平性论证的实证支撑（chicken-and-egg 污染实验）
================================================================

审稿人 M1.1 的公平性质疑：我方在拥塞混淆中把 λ0 设为 λ_cong（"上帝视角"），
而基线只用 clean 数据标定。要正面回应，需证明两件事（呼应消融实验结果）：

  (a) 若"公平地"把正确的 λ_cong 喂给朴素阈值（oracle 重标定），它也能接近 CUSUM
      → 差距不在规则复杂度，而在「获取 λ_cong 的能力」（消融已证：FPR 0.16–0.63%）。
  (b) 基线在真实部署中无法安全获得 λ_cong：它只能从【观测丢包】在线估计自然率，
      而观测丢包在攻击存在时被攻击丢包污染 → 在线估计 λ̂ 系统性高估 → 重标定的
      阈值被抬高 → 低强度攻击漏检。这就是 chicken-and-egg。

本脚本实证 (b)：对比「oracle 重标定」（防御者知道真实 λ_cong，等价于我方离线
期望基线经由利用率这一不受污染通道提供 λ_cong）与「online 重标定」（防御者从
全网观测丢包均值估计 λ̂，被攻击污染）两种策略下朴素阈值的检出率。

关键物理依据：我方 λ_cong 来自期望基线的拥塞分量 b_cong = rx·max(0,k·(u−θ))，
由利用率 u（队列到达率）驱动；灰洞攻击只丢包、不改到达率，故 u 在攻击期间不
被污染，k、θ 可由攻击前窗口离线拟合后继续外推。基线没有这条物理前向通道。

用法: python3 fairness_ablation.py
输出: analysis/comparison/fairness_pollution.csv
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

BASE = GH_OUT
OUT = os.path.join(BASE, "comparison")
NSLOTS = 200
N_TRIALS = 500
N_HONEST = 16          # 诚实节点数（1 个恶意节点，共 17）
THETA = 1.0 / 17.0     # 攻击者比例上界（单节点）


def section(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def naive_verdict(series_map, tau, malicious):
    """朴素阈值 sum > tau，返回 (tpr, fpr) 与攻击节点是否命中."""
    tp = fp = n_neg = 0
    for n, s in series_map.items():
        hit = bool(s.sum() > tau)
        if n in malicious:
            tp += int(hit)
        else:
            n_neg += 1
            fp += int(hit)
    return (tp / len(malicious)), (fp / n_neg if n_neg else 0.0)


def run():
    section("P0-① 混淆公平性：oracle vs online 重标定（朴素阈值）")
    lam_sweep = [0.1, 0.5, 1.0]                     # 自然拥塞率
    d_sweep = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]      # 攻击额外丢包率

    rows = []
    for lam in lam_sweep:
        tau_oracle = lam * NSLOTS + 3.0 * math.sqrt(NSLOTS * lam)
        for ds in d_sweep:
            rng = np.random.default_rng(7000 + int(lam * 100) + int(ds * 100))
            tpr_oracle = fpr_oracle = 0.0
            tpr_online = fpr_online = 0.0
            bias_sum = 0.0
            for _ in range(N_TRIALS):
                # 17 节点：16 诚实 @ λ, 1 攻击 @ λ+δ
                honest = rng.poisson(lam, size=(N_HONEST, NSLOTS))
                attack = rng.poisson(lam + ds, size=NSLOTS)

                # --- oracle 重标定（正确 λ_cong） ---
                # 阈值覆盖自然波动；攻击节点 sum≈λ·n+δ·n，诚实≈λ·n
                tau_o = tau_oracle
                hit_atk = bool(attack.sum() > tau_o)
                fp_o = int((honest.sum(axis=1) > tau_o).sum()) / N_HONEST
                tpr_oracle += int(hit_atk)
                fpr_oracle += fp_o

                # --- online 重标定（从观测丢包估计 λ̂，被攻击污染） ---
                # 全网均值：λ̂ = (Σ honest sum + attack sum) / (17·n)，期望 = λ + δ/17
                lam_hat = (honest.sum() + attack.sum()) / ((N_HONEST + 1) * NSLOTS)
                bias_sum += lam_hat - lam
                tau_hat = lam_hat * NSLOTS + 3.0 * math.sqrt(NSLOTS * lam_hat)
                hit_atk_hat = bool(attack.sum() > tau_hat)
                fp_hat = int((honest.sum(axis=1) > tau_hat).sum()) / N_HONEST
                tpr_online += int(hit_atk_hat)
                fpr_online += fp_hat

            rows.append(dict(
                lam_cong=lam,
                delta_small=ds,
                theta=THETA,
                tau_oracle=round(tau_oracle, 2),
                bias_lam_hat=round(bias_sum / N_TRIALS, 4),
                tpr_oracle=round(tpr_oracle / N_TRIALS, 4),
                fpr_oracle=round(fpr_oracle / N_TRIALS, 4),
                tpr_online=round(tpr_online / N_TRIALS, 4),
                fpr_online=round(fpr_online / N_TRIALS, 4),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fairness_pollution.csv"), index=False)

    for lam in lam_sweep:
        sub = df[df["lam_cong"] == lam]
        print(f"\nλ_cong={lam}  (oracle τ={sub['tau_oracle'].iloc[0]:.1f})")
        print(f"{'δ_small':>8s} {'λ̂偏差':>8s} {'TPR_oracle':>11s} {'FPR_oracle':>10s} "
              f"{'TPR_online':>11s} {'FPR_online':>10s} {'ΔTPR':>7s}")
        for _, r in sub.iterrows():
            d = r["tpr_oracle"] - r["tpr_online"]
            print(f"{r['delta_small']:>8.2f} {r['bias_lam_hat']:>8.4f} "
                  f"{r['tpr_oracle']:>11.2f} {r['fpr_oracle']:>10.3f} "
                  f"{r['tpr_online']:>11.2f} {r['fpr_online']:>10.3f} {d:>7.2f}")
    return df


def theta_sweep():
    """θ 扫描：污染(λ̂偏差)与 online 漏检随攻击者比例 θ 增长，oracle 不受影响。

    攻击者控制 θ·17 个节点，每个以 λ+δ 丢包。全网均值 λ̂ 被污染 θ·δ。
    oracle 重标定 = 我方离线期望基线（利用率通道）的等价，与 θ 无关。"""
    section("P0-① θ 扫描：chicken-and-egg 严重程度随攻击者比例增长")
    lam = 0.5
    d_sweep = [0.1, 0.2, 0.3, 0.5]
    theta_sweep = [1 / 17, 0.1, 0.25, 0.4]
    tau_oracle = lam * NSLOTS + 3.0 * math.sqrt(NSLOTS * lam)

    rows = []
    for theta in theta_sweep:
        n_att = max(1, int(round(theta * 17)))
        for ds in d_sweep:
            rng = np.random.default_rng(9000 + int(theta * 100) + int(ds * 100))
            tpr_oracle = fpr_oracle = 0.0
            tpr_online = fpr_online = 0.0
            bias_sum = 0.0
            for _ in range(N_TRIALS):
                n_h = 17 - n_att
                honest = rng.poisson(lam, size=(n_h, NSLOTS))
                attackers = rng.poisson(lam + ds, size=(n_att, NSLOTS))
                allsum = honest.sum() + attackers.sum()

                # oracle
                tpr_oracle += int((attackers.sum(axis=1) > tau_oracle).all())
                fpr_oracle += int((honest.sum(axis=1) > tau_oracle).sum()) / n_h

                # online（全网均值，被 攻击θ 污染）
                lam_hat = allsum / (17 * NSLOTS)
                bias_sum += lam_hat - lam
                tau_hat = lam_hat * NSLOTS + 3.0 * math.sqrt(NSLOTS * lam_hat)
                tpr_online += int((attackers.sum(axis=1) > tau_hat).all())
                fpr_online += int((honest.sum(axis=1) > tau_hat).sum()) / n_h

            rows.append(dict(
                theta=round(theta, 3), n_att=n_att, delta_small=ds,
                bias_lam_hat=round(bias_sum / N_TRIALS, 4),
                tpr_oracle=round(tpr_oracle / N_TRIALS, 4),
                fpr_oracle=round(fpr_oracle / N_TRIALS, 4),
                tpr_online=round(tpr_online / N_TRIALS, 4),
                fpr_online=round(fpr_online / N_TRIALS, 4),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fairness_theta_sweep.csv"), index=False)

    print(f"\nλ_cong={lam}  oracle τ={tau_oracle:.1f}（oracle 与 θ 无关）")
    for theta in theta_sweep:
        sub = df[df["theta"] == round(theta, 3)]
        print(f"\nθ={theta:.3f} (攻击节点数={sub['n_att'].iloc[0]})")
        print(f"{'δ_small':>8s} {'λ̂偏差':>8s} {'TPR_oracle':>11s} {'TPR_online':>11s} "
              f"{'ΔTPR':>7s} {'FPR_oracle':>10s} {'FPR_online':>10s}")
        for _, r in sub.iterrows():
            d = r["tpr_oracle"] - r["tpr_online"]
            print(f"{r['delta_small']:>8.2f} {r['bias_lam_hat']:>8.4f} "
                  f"{r['tpr_oracle']:>11.2f} {r['tpr_online']:>11.2f} "
                  f"{d:>7.2f} {r['fpr_oracle']:>10.3f} {r['fpr_online']:>10.3f}")
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d1 = run()
    d2 = theta_sweep()
    print("\nDONE. outputs:", os.path.join(OUT, "fairness_pollution.csv"),
          os.path.join(OUT, "fairness_theta_sweep.csv"))