#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-8 (R1-M3): 真实拥塞关键消融的多种子重复 (≥10 seeds) + TPR/FPR 区间
=======================================================================
消融 (与 §VI-F 同口径, 检测器不变只换输入):
  blind : congestion-blind CUSUM — 静态 λ0=0.01, h=choose_h_chain(0.01,δ,1e3)
  aware : baseline-aware  CUSUM — per-epoch β 标定 b(t), λ0(t)=max(b(t),0.01),
          h 在该种子 worst-node 速率反解 (论文 Table II 口径)

每个种子 s0..s10:
  - β 标定只用该种子的 clean run (无跨种子泄露; rx 不被灰洞污染)
  - att run = 同种子 clean 配置 + 灰洞卫星 12 (CONSTANT 0.5)

统计: TPR (node12 报警), FPR (诚实 node-run 报警率, clean+att),
      纯归因率 (仅 node12 报警的 run), 检测延迟, Wilson 95% 区间。

输出: analysis/comparison/r8_multiseed_congestion.csv / .txt
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusum_detector import run_cusum, choose_h_chain
from congestion_cusum import (node_agg, link_epochs, build_b,
                              DELTA, TARGET_ARL, LAMBDA_STATIC, FLOOR,
                              CP_THRESH)

RUNROOT = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/",
                       "temp/runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "comparison")
MAL = 12
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # s0 = 原始 run


def run_name(kind, k):
    """kind in {clean, att}; k=0 → 原 run, k>=1 → _s{k}."""
    if k == 0:
        return f"gh_cong_{kind}12" if kind == "att" else "gh_cong_clean"
    return f"gh_cong_{kind}12_s{k}" if kind == "att" else f"gh_cong_clean_s{k}"


def load_run(name):
    p = os.path.join(RUNROOT, name, "logs_ns3", "isl_packet_loss.csv")
    d = pd.read_csv(p, header=None,
                    names=["src", "dst", "t_start", "t_end", "rx", "dropped"])
    d["slot"] = (d["t_start"] / 1e9).astype(int)
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_variant(d_agg, b_agg, h, use_schedule):
    alarmed = {}
    for n, series in sorted(d_agg.items()):
        if use_schedule:
            sched = lambda t, b=b_agg[n]: (max(b[t], FLOOR)
                                           if t < len(b) else FLOOR)
            _, a = run_cusum(series, FLOOR, DELTA, h, lambda0_schedule=sched)
        else:
            _, a = run_cusum(series, LAMBDA_STATIC, DELTA, h)
        if a is not None:
            alarmed[n] = a
    return alarmed


def main():
    h_blind = choose_h_chain(LAMBDA_STATIC, DELTA, TARGET_ARL)
    print(f"h_blind = {h_blind:.2f} (lambda0={LAMBDA_STATIC}, "
          f"ARL0*={TARGET_ARL:.0f})")

    rows = []
    for k in SEEDS:
        cname, aname = run_name("clean", k), run_name("att", k)
        try:
            clean, att = load_run(cname), load_run(aname)
        except FileNotFoundError as e:
            print(f"[s{k}] SKIP ({e})")
            continue

        # per-epoch β 标定 (只用该种子 clean run)
        epoch_of, beta = link_epochs(clean, CP_THRESH)
        b_c = build_b(clean, epoch_of, beta)
        b_a = build_b(att, epoch_of, beta)
        clean2 = clean.assign(b_epoch=b_c)
        att2 = att.assign(b_epoch=b_a)

        d_c, d_a = node_agg(clean, "dropped"), node_agg(att, "dropped")
        be_c, be_a = node_agg(clean2, "b_epoch"), node_agg(att2, "b_epoch")
        nodes = sorted(d_c)
        honest = [n for n in nodes if n != MAL]

        lam_max = max(be_c[n].max() for n in nodes)
        h_aware = choose_h_chain(lam_max, DELTA, TARGET_ARL)

        # 攻击显现 slot (att 相对 clean 的额外丢包)
        onset = int(np.argmax(d_a[MAL] - d_c[MAL] > 0)) if MAL in d_a else None

        rec = dict(seed=k, lam_max=round(lam_max, 1),
                   h_aware=round(h_aware, 2), onset=onset)
        for variant, use_sched, bmap_c, bmap_a, h in [
                ("blind", False, None, None, h_blind),
                ("aware", True, be_c, be_a, h_aware)]:
            al_c = run_variant(d_c, bmap_c, h, use_sched)
            al_a = run_variant(d_a, bmap_a, h, use_sched)
            fp_clean = [n for n in al_c if n != MAL]
            fp_att = [n for n in al_a if n != MAL]
            tp = MAL in al_a
            rec[f"{variant}_clean_alarmed"] = ",".join(map(str, sorted(al_c)))
            rec[f"{variant}_att_alarmed"] = ",".join(map(str, sorted(al_a)))
            rec[f"{variant}_tp"] = int(tp)
            rec[f"{variant}_fp_clean"] = len(fp_clean)
            rec[f"{variant}_fp_att"] = len(fp_att)
            rec[f"{variant}_pure"] = int(tp and not fp_clean and not fp_att)
            rec[f"{variant}_delay"] = (al_a[MAL] - onset) if tp and onset \
                else None
        rows.append(rec)
        print(f"[s{k}] lam_max={lam_max:.0f} h_aware={h_aware:.2f} | "
              f"blind: tp={rec['blind_tp']} fp={rec['blind_fp_clean']}"
              f"+{rec['blind_fp_att']} | "
              f"aware: tp={rec['aware_tp']} fp={rec['aware_fp_clean']}"
              f"+{rec['aware_fp_att']} delay={rec['aware_delay']}")

    df = pd.DataFrame(rows)
    n_att = int(df["aware_tp"].size)
    n_honest_runs = n_att * 2 * 16          # seeds × (clean+att) × 16 nodes

    summary = {}
    for v in ("blind", "aware"):
        tp = int(df[f"{v}_tp"].sum())
        fp = int(df[f"{v}_fp_clean"].sum() + df[f"{v}_fp_att"].sum())
        pure = int(df[f"{v}_pure"].sum())
        delays = df[f"{v}_delay"].dropna()
        tpr_lo, tpr_hi = wilson(tp, n_att)
        fpr_lo, fpr_hi = wilson(fp, n_honest_runs)
        pur_lo, pur_hi = wilson(pure, n_att)
        summary[v] = dict(
            n_seeds=n_att, tp=tp, fp_node_runs=fp,
            tpr=tp / n_att, tpr_lo=tpr_lo, tpr_hi=tpr_hi,
            fpr=fp / n_honest_runs, fpr_lo=fpr_lo, fpr_hi=fpr_hi,
            pure=pure / n_att, pure_lo=pur_lo, pure_hi=pur_hi,
            delay_med=float(delays.median()) if len(delays) else None,
            delay_min=float(delays.min()) if len(delays) else None,
            delay_max=float(delays.max()) if len(delays) else None)
        s = summary[v]
        print(f"\n[{v}] TPR={s['tpr']:.2f} "
              f"[{s['tpr_lo']:.2f},{s['tpr_hi']:.2f}] "
              f"FPR={s['fpr']*100:.3f}% "
              f"[{s['fpr_lo']*100:.3f},{s['fpr_hi']*100:.3f}] "
              f"pure={s['pure']:.2f} delay_med={s['delay_med']}")

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "r8_multiseed_congestion.csv"), index=False)
    with open(os.path.join(OUT, "r8_multiseed_congestion.txt"), "w") as f:
        f.write("R-8: multi-seed repetition of the real-congestion ablation\n")
        f.write(f"protocol: {n_att} seeds (s0=original), CONSTANT-0.5 on "
                f"sat12, delta={DELTA}, ARL0*={TARGET_ARL:.0f}\n")
        f.write(f"beta calibration: per-(link,epoch) on each seed's own "
                f"clean run (CP thresh={CP_THRESH})\n")
        f.write(f"h_blind={h_blind:.2f} (static 0.01); "
                f"h_aware per-seed at worst-node rate\n\n")
        for v, s in summary.items():
            f.write(f"[{v}] TPR {s['tpr']:.3f} "
                    f"[{s['tpr_lo']:.3f},{s['tpr_hi']:.3f}] | "
                    f"FPR {s['fpr']*100:.3f}% "
                    f"[{s['fpr_lo']*100:.3f},{s['fpr_hi']*100:.3f}] | "
                    f"pure-attribution {s['pure']:.3f} | "
                    f"delay med/min/max = {s['delay_med']}/"
                    f"{s['delay_min']}/{s['delay_max']}\n")
        f.write("\nper-seed detail:\n")
        f.write(df.to_string(index=False))
    print("\nR-8 DONE ->", os.path.join(OUT, "r8_multiseed_congestion.csv"))


if __name__ == "__main__":
    main()
