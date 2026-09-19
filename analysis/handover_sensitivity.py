#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2.3 (#10): λ_h > 0 切换窗口敏感性注入
=======================================
回应 mock review M2.3: "评估平台的预计算路由使 λ_h=0 是平台便利性质而非框架验证;
若路由协议有收敛期 (OSPF/分布式), λ_h>0 时残差如何保持纯净? 可做小型敏感性注入
(切换窗口人为注入服从标定分布的瞬态丢失, 验证残差仍≈0)."

设计 (纯后处理, 复用 clean run 观测):
  1. 注入: 每条链路每时隙按 handover_events 计数注入 Poisson(λ_h·events) 瞬态
     丢包 (模拟分布式路由收敛期丢失), 种子 A = 观测实现。
  2. 标定: λ̂_h 从另一实现 (种子 B) 的前 100 时隙 (short attack-free window,
     与论文 IV-B 的标定协议一致) 用 MLE 拟合: λ̂_h = ΣX / Σevents。
  3. 期望模型: b_handover = λ̂_h · events (逐链路逐时隙); 残差 r = d − b。
  4. 检验三问:
     (a) 纯净性: 残差均值是否回到 0 (期望模型吸收注入);
     (b) 误报: 静态 λ₀=0.01 残差 CUSUM (主场景配置) vs 切换感知 λ₀(t) 调度
         (λ₀(t) = Σ_l λ̂_h·events, 与 V-B 拥塞重标定同一标定原则);
     (c) 共存: λ_h 注入 + 灰洞攻击叠加时检出/归因是否保持。

h 反解在 λ₀_max (最坏时隙) 上进行 → ARL₀ 界对全部时隙一致保守成立。

用法 (Ubuntu-20.04): python3 handover_sensitivity.py
输出: analysis/comparison/handover_sensitivity.csv / .txt
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
from cusum_detector import load_residuals, run_cusum, choose_h_chain

BASE = GH_OUT
OUT = os.path.join(BASE, "comparison")
MALICIOUS = 12
DELTA = 1.0
LAMBDA_FLOOR = 0.01      # 非窗口时隙的标定误差下限 (与主场景 λ0=0.01 一致)
TARGET_ARL = 1000.0      # 与 Part A/纯净场景一致
N_SEEDS = 50             # 每个 λ_h 的注入实现数
CAL_SLOTS = 100          # 标定窗口 (前 100 时隙, "short attack-free window")
LAMH_SWEEP = [0.5, 1.0, 2.0, 5.0]
Z95 = 1.96


def wilson_ci(k, n, z=Z95):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_grid():
    """clean run 的 (link, slot) 网格: events 计数 + dropped + 节点聚合容器."""
    df = load_residuals(os.path.join(BASE, "residuals_baseline.csv"))
    att = load_residuals(os.path.join(BASE, "residuals_const12.csv"))
    keys, events, clean_drop = [], [], []
    attack = {}
    slots = set()
    for r in df.itertuples(index=False):
        k = (int(r.src), int(r.dst), int(r.slot))
        keys.append(k)
        events.append(int(r.handover_events))
        clean_drop.append(int(r.dropped))
        slots.add(int(r.slot))
    for r in att.itertuples(index=False):
        attack[(int(r.src), int(r.dst), int(r.slot))] = int(r.dropped)
    nslots = max(slots) + 1
    nodes = sorted({k[0] for k in keys})
    # 攻击丢包时序 = 攻击 run dropped − clean dropped (逐链路逐时隙)
    attack_vec = np.array([attack.get(k, 0) - d for k, d in zip(keys, clean_drop)],
                          dtype=float)
    ev = np.array(events, dtype=float)
    cd = np.array(clean_drop, dtype=float)
    return keys, ev, cd, attack_vec, nslots, nodes


def node_aggregate(keys, values, nslots, nodes):
    """(link,slot) 向量 → dict[node] -> np.array[slot]."""
    out = {n: np.zeros(nslots) for n in nodes}
    for (src, _dst, slot), v in zip(keys, values):
        out[src][slot] += v
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    keys, ev, cd, attack_vec, nslots, nodes = load_grid()
    honest = [n for n in nodes if n != MALICIOUS]
    total_events = ev.sum()
    ev_agg = node_aggregate(keys, ev, nslots, nodes)
    max_ev_agg = max(a.max() for a in ev_agg.values())
    print(f"网格: {len(keys)} link-slot, {len(nodes)} 节点, {nslots} 时隙, "
          f"总事件 {total_events:.0f}, 节点聚合最大单时隙事件 {max_ev_agg:.0f}")
    print(f"clean dropped 总量 {cd.sum():.0f}, 攻击丢包总量 {attack_vec.sum():.0f}")

    rows = []
    for lam_h in LAMH_SWEEP:
        # 每实现: 观测种子 A, 标定种子 B (独立实现的前 100 时隙)
        stat = dict(lam_h=lam_h, n_seeds=N_SEEDS)
        lam_hat_list, resid_mean_list, resid_max_list = [], [], []
        fp_static = tp_static = fp_aware = tp_aware = 0
        delays, alarm_slots = [], []
        attack_only_aware = 0     # 攻击共存时是否只有 node 12 报警 (归因纯净)
        inj_totals = []
        for s in range(N_SEEDS):
            rng_obs = np.random.default_rng(90000 + s)
            rng_cal = np.random.default_rng(50000 + s)
            X_obs = rng_obs.poisson(lam_h * ev)
            X_cal = rng_cal.poisson(lam_h * ev)
            inj_totals.append(X_obs.sum())
            # 标定: 前 100 时隙的 MLE
            cal_mask = np.array([k[2] < CAL_SLOTS for k in keys])
            denom = ev[cal_mask].sum()
            lam_hat = (X_cal[cal_mask].sum() / denom) if denom > 0 else 0.0
            lam_hat = max(lam_hat, 0.01)
            lam_hat_list.append(lam_hat)

            # 观测与期望 (逐链路): d = clean + X; b = λ̂_h·events
            b_vec = lam_hat * ev
            d_vec = cd + X_obs
            r_vec = d_vec - b_vec
            R_agg = node_aggregate(keys, r_vec, nslots, nodes)
            d_agg = node_aggregate(keys, d_vec, nslots, nodes)
            b_agg = {n: ev_agg[n] * lam_hat for n in nodes}
            resid_mean_list.append(np.mean([abs(R_agg[n]).mean()
                                            for n in honest]))
            resid_max_list.append(max(abs(R_agg[n]).max() for n in honest))

            # (b) 无攻击: 静态 λ₀ vs 切换感知 λ₀(t)
            h_static = choose_h_chain(LAMBDA_FLOOR, DELTA, TARGET_ARL)
            lam_max = max(b_agg[n].max() for n in nodes) + LAMBDA_FLOOR
            h_aware = choose_h_chain(lam_max, DELTA, TARGET_ARL)
            for n in nodes:
                _, a_s = run_cusum(R_agg[n], LAMBDA_FLOOR, DELTA, h_static)
                if a_s is not None and n != MALICIOUS:
                    fp_static += 1
                sched = (lambda t, b=b_agg[n]:
                         max(b[t], LAMBDA_FLOOR) if t < len(b) else LAMBDA_FLOOR)
                _, a_w = run_cusum(d_agg[n], LAMBDA_FLOOR, DELTA, h_aware,
                                   lambda0_schedule=sched)
                if a_w is not None and n != MALICIOUS:
                    fp_aware += 1

            # (c) 攻击共存: 叠加 node 12 攻击丢包
            d_att = cd + X_obs + attack_vec
            R_att = d_att - b_vec
            R_att_agg = node_aggregate(keys, R_att, nslots, nodes)
            d_att_agg = node_aggregate(keys, d_att, nslots, nodes)
            alarmed_aware = {}
            for n in nodes:
                _, a_s = run_cusum(R_att_agg[n], LAMBDA_FLOOR, DELTA, h_static)
                if n == MALICIOUS and a_s is not None:
                    tp_static += 1
                sched = (lambda t, b=b_agg[n]:
                         max(b[t], LAMBDA_FLOOR) if t < len(b) else LAMBDA_FLOOR)
                _, a_w = run_cusum(d_att_agg[n], LAMBDA_FLOOR, DELTA, h_aware,
                                   lambda0_schedule=sched)
                alarmed_aware[n] = a_w
            if alarmed_aware.get(MALICIOUS) is not None:
                tp_aware += 1
                alarm_slots.append(alarmed_aware[MALICIOUS])
                if not any(v is not None for k, v in alarmed_aware.items()
                           if k != MALICIOUS):
                    attack_only_aware += 1
            # 攻击显现起点 (首个攻击残差>0 的时隙)
            att_series = node_aggregate(keys, attack_vec, nslots, nodes)[MALICIOUS]
            onset = int(np.argmax(att_series > 0)) if (att_series > 0).any() else None
            if alarmed_aware.get(MALICIOUS) is not None and onset is not None:
                delays.append(alarmed_aware[MALICIOUS] - onset)

        n_neg = N_SEEDS * len(honest)
        stat.update(
            lam_hat_mean=float(np.mean(lam_hat_list)),
            lam_hat_std=float(np.std(lam_hat_list)),
            injected_mean=float(np.mean(inj_totals)),
            resid_absmean_honest=float(np.mean(resid_mean_list)),
            resid_absmax_honest=float(np.max(resid_max_list)),
            fpr_static=fp_static / n_neg,
            fpr_static_ci=wilson_ci(fp_static, n_neg),
            fpr_aware=fp_aware / n_neg,
            fpr_aware_ci=wilson_ci(fp_aware, n_neg),
            tpr_static=tp_static / N_SEEDS,
            tpr_aware=tp_aware / N_SEEDS,
            alarm_slot_med=float(np.median(alarm_slots)) if alarm_slots else None,
            delay_med=float(np.median(delays)) if delays else None,
            delay_max=int(np.max(delays)) if delays else None,
            attribution_pure=attack_only_aware / N_SEEDS,
        )
        rows.append(stat)
        fs_lo, fs_hi = stat["fpr_static_ci"]
        fa_lo, fa_hi = stat["fpr_aware_ci"]
        print(f"\nλ_h={lam_h}: λ̂={stat['lam_hat_mean']:.3f}±{stat['lam_hat_std']:.3f}  "
              f"注入 {stat['injected_mean']:.0f} 包  "
              f"诚实残差 |r| mean {stat['resid_absmean_honest']:.2f} / "
              f"max {stat['resid_absmax_honest']:.0f}")
        print(f"  无攻击 FPR: 静态λ₀ {100*stat['fpr_static']:.1f}% "
              f"[{100*fs_lo:.1f},{100*fs_hi:.1f}]  vs  切换感知 "
              f"{100*stat['fpr_aware']:.1f}% [{100*fa_lo:.1f},{100*fa_hi:.1f}]")
        print(f"  攻击共存 TPR: 静态 {stat['tpr_static']:.2f}  切换感知 "
              f"{stat['tpr_aware']:.2f}  报警时隙 med "
              f"{stat['alarm_slot_med']}  延迟 med {stat['delay_med']} "
              f"max {stat['delay_max']}  归因纯净 {stat['attribution_pure']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "handover_sensitivity.csv"), index=False)
    with open(os.path.join(OUT, "handover_sensitivity.txt"), "w") as f:
        f.write(f"M2.3 (#10) handover-window sensitivity injection\n"
                f"protocol: Poisson(lam_h*events) per link-slot, lambda_hat MLE "
                f"from first {CAL_SLOTS} slots of independent realization, "
                f"h inverted at lambda0_max (uniformly conservative), "
                f"ARL0*={TARGET_ARL:.0f}, delta={DELTA}, n_seeds={N_SEEDS}\n\n")
        f.write(df.to_csv(index=False))
    print(f"\nDONE. -> {OUT}/handover_sensitivity.csv")


if __name__ == "__main__":
    main()
