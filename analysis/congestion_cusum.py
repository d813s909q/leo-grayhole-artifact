#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2.1/P0-② 补充: 真实拥塞下 CUSUM 的实证行为 (static vs 重标定)
===============================================================
07 报告验证了残差纯净性 (聚合层面); 本脚本实证检测器在真实拥塞数据上的行为:

  (a) static — 无拥塞感知的主场景配置 (λ0=0.01, h 反解 ARL0*=1000, 输入 =
      拥塞无感知残差 = 观测丢包, 因无拥塞标定期望丢包 ≈ 0):
      预期 clean/attack 均在 4 个承载拥塞丢包的节点报警 (0,4,11,12) —
      与 naive/trust/CLIF-cleanfit 同集合, 攻击归因退化为 4 选 1。

  (b) fullrun-β 重标定 — 逐链路常数 β_l (全程平均) 标定 b, λ0(t)=b(t) 调度:
      诚实瓶颈节点安静, 但节点 12 的汇聚链路在路由 epoch 间角色切换
      (瓶颈 epoch d≈834/slot, 下游 epoch d=0), 全程平均 β=0.386 留下
      ±513/-321 的换相残差 (总量恰为 0) → clean run 单节点误报。

  (c) per-epoch-β 重标定 — β̂_{l,e} 按 (链路, 路由 epoch) 分段标定 (epoch
      边界 = clean run 上 d/rx 变点, 部署中即离线路由更新时刻; rx 不被
      灰洞污染, 标定可跨 run 外推): 诚实残差逐时隙归零, clean 0 误报;
      attack 仅节点 12 报警 (残差 = 纯攻击 +50027)。

用法 (Ubuntu-20.04): python3 congestion_cusum.py
输出: analysis/congestion/congestion_cusum.csv / .txt
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
from cusum_detector import run_cusum, choose_h_chain

BASE = GH_OUT
CONG = os.path.join(BASE, "congestion")
MAL = 12
DELTA = 1.0
TARGET_ARL = 1000.0
LAMBDA_STATIC = 0.01
FLOOR = 0.01
CP_THRESH = 50.0


def load(name):
    return pd.read_csv(os.path.join(CONG, name))


def node_agg(df, field):
    """逐链路逐时隙字段 → dict[node] -> np.array[slot] (同节点多链路求和)."""
    piv = df.groupby(["src", "slot"])[field].sum().reset_index()
    nslots = int(df["slot"].max()) + 1
    out = {}
    for n, g in piv.groupby("src"):
        arr = np.zeros(nslots)
        arr[g["slot"].values.astype(int)] = g[field].values
        out[int(n)] = arr
    return out


def link_epochs(clean_df, cp_thresh=CP_THRESH):
    """clean run 上逐链路变点分段 → epoch 标签 + 每 (链路, epoch) 的 β̂ = Σd/Σrx."""
    epoch_of = {}
    beta = {}
    for (s, d), g in clean_df.groupby(["src", "dst"]):
        g = g.sort_values("slot")
        slots = g["slot"].values.astype(int)
        dd = g["dropped"].values.astype(float)
        rx = g["rx"].values.astype(float)
        ids = np.zeros(len(slots), dtype=int)
        eid = 0
        for i in range(1, len(slots)):
            if (abs(dd[i] - dd[i - 1]) > cp_thresh
                    or abs(rx[i] - rx[i - 1]) > cp_thresh):
                eid += 1
            ids[i] = eid
        epoch_of[(s, d)] = dict(zip(slots.tolist(), ids.tolist()))
        for e in np.unique(ids):
            m = ids == e
            sr, sd = rx[m].sum(), dd[m].sum()
            beta[(s, d, int(e))] = (sd / sr) if sr > 0 else 0.0
    return epoch_of, beta


def build_b(df, epoch_of, beta):
    """per-epoch β̂ × 本 run 自己的 rx → 逐链路逐时隙 b (返回与 df 同序的 Series)."""
    vals = []
    for r in df.itertuples(index=False):
        key = (int(r.src), int(r.dst))
        e = epoch_of.get(key, {}).get(int(r.slot), 0)
        vals.append(float(r.rx) * beta.get((key[0], key[1], e), 0.0))
    return pd.Series(vals, index=df.index)


def run_variant(name, d_agg, b_agg, h, use_schedule):
    """跑一个 CUSUM 变体: use_schedule=True → λ0(t)=max(b(t),FLOOR); False → λ0=0.01 静态."""
    alarmed, slots = [], {}
    for n, series in sorted(d_agg.items()):
        if use_schedule:
            sched = (lambda t, b=b_agg[n]:
                     max(b[t], FLOOR) if t < len(b) else FLOOR)
            _, a = run_cusum(series, FLOOR, DELTA, h, lambda0_schedule=sched)
        else:
            _, a = run_cusum(series, LAMBDA_STATIC, DELTA, h)
        if a is not None:
            alarmed.append(n)
            slots[n] = a
    return alarmed, slots


def main():
    clean = load("residuals_cong_clean.csv")
    att = load("residuals_cong_att12.csv")

    d_c, d_a = node_agg(clean, "dropped"), node_agg(att, "dropped")
    bf_c, bf_a = node_agg(clean, "b_total"), node_agg(att, "b_total")
    nodes = sorted(d_c)
    honest = [n for n in nodes if n != MAL]
    nslots = len(next(iter(d_c.values())))
    print(f"节点 {len(nodes)} 个, {nslots} 时隙; "
          f"承载拥塞丢包的节点: {[n for n in nodes if d_c[n].sum() > 0]}")

    # ---- (a) static: 拥塞无感知残差 = 观测丢包 ----
    h_static = choose_h_chain(LAMBDA_STATIC, DELTA, TARGET_ARL)
    st_clean, stc = run_variant("static", d_c, None, h_static, False)
    st_att, sta = run_variant("static", d_a, None, h_static, False)
    print(f"\n(a) static (λ0={LAMBDA_STATIC}, h={h_static:.2f}, ARL0*={TARGET_ARL:.0f})")
    print(f"    clean : alarmed={st_clean} slots={ {n: stc[n] for n in st_clean} }")
    print(f"    attack: alarmed={st_att} slots={ {n: sta[n] for n in st_att} }")

    # ---- (b) fullrun-β 重标定: λ0(t) = b_fullrun(t) ----
    lam_max_f = max(bf_c[n].max() for n in nodes)
    h_full = choose_h_chain(lam_max_f, DELTA, TARGET_ARL)
    fr_clean, frc = run_variant("fullrun", d_c, bf_c, h_full, True)
    fr_att, fra = run_variant("fullrun", d_a, bf_a, h_full, True)
    print(f"\n(b) fullrun-β 重标定 (λ0(t)=b(t), λ0_max={lam_max_f:.0f}, h={h_full:.2f})")
    print(f"    clean : alarmed={fr_clean} slots={ {n: frc[n] for n in fr_clean} }")
    print(f"    attack: alarmed={fr_att} slots={ {n: fra[n] for n in fr_att} }")

    # ---- (c) per-epoch-β 重标定 ----
    epoch_of, beta = link_epochs(clean)
    n_epochs = sum(len(set(v.values())) for v in epoch_of.values())
    b_c = build_b(clean, epoch_of, beta)
    b_a = build_b(att, epoch_of, beta)
    clean2 = clean.assign(b_epoch=b_c, r_epoch=clean["dropped"] - b_c)
    att2 = att.assign(b_epoch=b_a, r_epoch=att["dropped"] - b_a)
    be_c, be_a = node_agg(clean2, "b_epoch"), node_agg(att2, "b_epoch")
    re_c, re_a = node_agg(clean2, "r_epoch"), node_agg(att2, "r_epoch")
    lam_max_e = max(be_c[n].max() for n in nodes)
    h_epoch = choose_h_chain(lam_max_e, DELTA, TARGET_ARL)
    pe_clean, pec = run_variant("per-epoch", d_c, be_c, h_epoch, True)
    pe_att, pea = run_variant("per-epoch", d_a, be_a, h_epoch, True)
    print(f"\n(c) per-epoch-β 重标定 ({n_epochs} 个 (链路,epoch) 段, "
          f"λ0_max={lam_max_e:.0f}, h={h_epoch:.2f})")
    print(f"    clean : alarmed={pe_clean} slots={ {n: pec[n] for n in pe_clean} }")
    print(f"    attack: alarmed={pe_att} slots={ {n: pea[n] for n in pe_att} }")

    # ---- 残差纯净性 (per-epoch) ----
    print("\n=== per-epoch 残差纯净性 ===")
    print(f"{'node':>4} {'clean_r_sum':>12} {'clean_max|r|':>12} "
          f"{'att_r_sum':>10} {'att_max|r|':>11}")
    res_rows = []
    for n in nodes:
        rc, ra = re_c[n], re_a[n]
        res_rows.append(dict(node=n, clean_r_sum=rc.sum(),
                             clean_absmax=abs(rc).max(), clean_std=rc.std(),
                             att_r_sum=ra.sum(), att_absmax=abs(ra).max(),
                             att_std=ra.std()))
        print(f"{n:4d} {rc.sum():12.1f} {abs(rc).max():12.1f} "
              f"{ra.sum():10.1f} {abs(ra).max():11.1f}")
    hon_absmax_clean = max(abs(re_c[n]).max() for n in honest)
    hon_absmax_att = max(abs(re_a[n]).max() for n in honest)
    att_sum = re_a[MAL].sum()
    onset = int(np.argmax(d_a[MAL] - d_c[MAL] > 0))
    delay = (pea.get(MAL) - onset) if MAL in pea else None
    print(f"\n诚实节点 (clean) 最大逐时隙 |r| = {hon_absmax_clean:.1f}")
    print(f"诚实节点 (attack) 最大逐时隙 |r| = {hon_absmax_att:.1f}")
    print(f"节点 12 (attack) 残差总和 = {att_sum:.0f} (纯攻击, ≈ drop_rate×rx)")
    print(f"攻击显现时隙 = {onset}, per-epoch 报警时隙 = {pea.get(MAL)}, "
          f"延迟 = {delay} 时隙")

    # ---- 汇总输出 ----
    def row(variant, h, clean_a, att_a, att_slot):
        fp_c = [n for n in clean_a if n != MAL]
        fp_a = [n for n in att_a if n != MAL]
        tp = MAL in att_a
        return dict(variant=variant, h=round(h, 2),
                    clean_alarmed=",".join(map(str, clean_a)),
                    clean_fp=len(fp_c), attack_alarmed=",".join(map(str, att_a)),
                    attack_tp=int(tp), attack_fp=len(fp_a),
                    node12_alarm_slot=att_slot,
                    delay=(att_slot - onset) if att_slot is not None else None)

    rows = [
        row("static(λ0=0.01)", h_static, st_clean, st_att, sta.get(MAL)),
        row("fullrun-β λ0(t)", h_full, fr_clean, fr_att, fra.get(MAL)),
        row("per-epoch-β λ0(t)", h_epoch, pe_clean, pe_att, pea.get(MAL)),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CONG, "congestion_cusum.csv"), index=False)
    pd.DataFrame(res_rows).to_csv(
        os.path.join(CONG, "congestion_cusum_residual.csv"), index=False)

    with open(os.path.join(CONG, "congestion_cusum.txt"), "w") as f:
        f.write("M2.1/P0-2 supplement: CUSUM behavior under real simulator congestion\n")
        f.write(f"protocol: delta={DELTA}, ARL0*={TARGET_ARL:.0f}, "
                f"static lam0={LAMBDA_STATIC}, epoch CP thresh={CP_THRESH:.0f}\n")
        f.write(f"attack onset slot = {onset}; honest per-epoch |r| max "
                f"(clean) = {hon_absmax_clean:.1f}; node12 attack residual "
                f"sum = {att_sum:.0f}\n\n")
        f.write(df.to_csv(index=False))
        f.write("\nper-epoch residual stats per node:\n")
        f.write(pd.DataFrame(res_rows).to_csv(index=False))
    print("\nDONE. ->", os.path.join(CONG, "congestion_cusum.csv"))


if __name__ == "__main__":
    main()
