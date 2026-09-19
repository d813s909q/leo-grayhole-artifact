#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-② 真实拥塞场景: 拥塞标定 + 残差验证
===========================================

与 4.1 共用「攻击前窗口离线拟合」流程, 但把论文的全局 b_cong 泛化为
**逐链路标定**的拥塞期望 (发现见下):

  b_cong(l,t) = rx(l,t) * beta_l,   beta_l = dropped_clean(l) / rx_clean(l)

为什么不是全局 k*(u-theta):
  Hypatia 的 isl_utilization.csv 记录的是「发送器忙碌占比」(served fraction),
  拥塞时瓶颈链路与下游链路都饱和到 u=1.0, 但只有瓶颈链路会溢出丢包。
  因此 served fraction 无法区分「瓶颈链路」与「下游链路」, 全局 k*(u-theta)
  会把下游链路也预测成丢包、造成巨大负残差。

  beta_l 的物理含义: beta_l ≈ max(0, u_arr(l) - 1), 即该链路「到达利用率
  超载比例」(arrival 超过 capacity 的部分), 由攻击前干净 run 逐链路标定。
  它是链路在路由拓扑中的位置属性 (瓶颈/汇聚/下游), 与攻击者无关。

无污染通道的保证 (关键):
  灰洞攻击是挂在接收设备上的 ReceiveErrorModel, 在 Receive() 里
  `m_rx_packets_current++` **之后**才做丢包判决。因此 rx (物理到达计数)
  不被灰洞污染 —— 攻击只会增加 dropped, 不会改变 rx。这使 beta_l 一经
  干净 run 标定, 即可在攻击期间用实时 rx 外推自然拥塞期望, 无需判决
  「哪个丢包是攻击」, 从而打破 chicken-and-egg。

数据源:
  run/logs_ns3/isl_packet_loss.csv  (接收视角: src=接收设备, dst=对端)
  run/logs_ns3/isl_utilization.csv  (发送视角, 仅作诊断)

用法:
  python3 congestion_residual.py --gen-data <dir> \
      --clean-run <dir> --attack-run <dir> --target 18 --out <dir>
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_model import (load_tles, load_isls, load_description,
                            contact_graph, mark_contact, handover_events,
                            interval_key)

FSTATE_STEP_NS = 100_000_000


# ----------------------------------------------------------------------
# 拥塞参数拟合: 逐链路 beta_l (攻击前窗口 = clean run 全程)
# ----------------------------------------------------------------------
def fit_congestion(clean_dir):
    """从无攻击拥塞 run 标定逐链路 beta_l = dropped/rx.

    返回 dict{beta: {(src,dst): float}, link_rows: [...], total_dropped, ...}
    """
    d = pd.read_csv(os.path.join(clean_dir, "logs_ns3", "isl_packet_loss.csv"),
                    header=None, names=["src", "dst", "t_start", "t_end", "rx", "dropped"])
    beta = {}
    agg = defaultdict(lambda: [0, 0])  # (src,dst) -> [rx, dropped]
    for r in d.itertuples(index=False):
        src, dst = int(r.src), int(r.dst)
        rx = int(r.rx)
        dr = int(r.dropped)
        agg[(src, dst)][0] += rx
        agg[(src, dst)][1] += dr

    link_rows = []
    for (src, dst), (rx, dr) in sorted(agg.items()):
        b = dr / rx if rx > 0 else 0.0
        beta[(src, dst)] = b
        link_rows.append({"src": int(src), "dst": int(dst), "rx": int(rx),
                          "dropped": int(dr), "beta": round(b, 6),
                          "offered": int(rx + dr),
                          "loss_rate": round(dr / (rx + dr), 6) if rx + dr else 0.0})

    return {
        "beta": beta,
        "link_rows": link_rows,
        "total_dropped": int(d["dropped"].sum()),
        "total_rx": int(d["rx"].sum()),
    }


# ----------------------------------------------------------------------
# 残差构建
# ----------------------------------------------------------------------
def build_residual(run_dir, contact, hevents, cong, interval_ns,
                   handover_loss_pkts=0.0, handover_window_s=1.0):
    d = pd.read_csv(os.path.join(run_dir, "logs_ns3", "isl_packet_loss.csv"),
                    header=None, names=["src", "dst", "t_start", "t_end", "rx", "dropped"])

    # 接触图双向查表
    up_lookup = {}
    for _, r in contact.iterrows():
        k1, k2 = (r.link_src, r.link_dst), (r.link_dst, r.link_src)
        v = bool(r.link_up)
        up_lookup.setdefault(k1, {})[interval_key(r.t_start, interval_ns)] = v
        up_lookup.setdefault(k2, {})[interval_key(r.t_start, interval_ns)] = v

    # 切换事件 → 窗口标记
    handover_marks = {}
    win_ns = int(handover_window_s * 1e9)
    for _, e in hevents.iterrows():
        node, nh_old, nh_new = int(e.node), int(e.next_hop_old), int(e.next_hop_new)
        for nb in {nh_old, nh_new}:
            for lag in range(0, win_ns, interval_ns):
                slot = interval_key(int(e.t_ns) + lag, interval_ns)
                handover_marks[(node, nb, slot)] = handover_marks.get((node, nb, slot), 0) + 1
                handover_marks[(nb, node, slot)] = handover_marks.get((nb, node, slot), 0) + 1

    beta = cong["beta"]
    rows = []
    for _, r in d.iterrows():
        src, dst, ts = int(r.src), int(r.dst), interval_key(r.t_start, interval_ns)
        rx, dropped = int(r.rx), int(r.dropped)
        link_up = up_lookup.get((src, dst), {}).get(ts, True)
        h = handover_marks.get((src, dst, ts), 0)
        b_handover = h * handover_loss_pkts
        b_cong = rx * beta.get((src, dst), 0.0)
        b_total = b_handover + b_cong
        rows.append({
            "src": src, "dst": dst, "slot": ts,
            "rx": rx, "dropped": dropped,
            "link_up": link_up, "handover_events": h,
            "b_congestion": round(b_cong, 4),
            "b_total": round(b_total, 4),
            "residual": round(dropped - b_total, 4),
        })
    return pd.DataFrame(rows)


def summarize(name, df):
    """打印 + 返回逐节点残差汇总 (均值/总和/标准差)。"""
    total_drop = float(df["dropped"].sum())
    total_b = float(df["b_total"].sum())
    total_r = float(df["residual"].sum())
    node_drop = df.groupby("src")["dropped"].sum()
    node_b = df.groupby("src")["b_total"].sum()
    node_res_sum = df.groupby("src")["residual"].sum()
    node_res_std = df.groupby("src")["residual"].std().fillna(0.0)

    print(f"\n=== '{name}' ===")
    print(f"  total dropped   = {total_drop:.1f}")
    print(f"  total b (pred)  = {total_b:.1f}")
    print(f"  total residual  = {total_r:.1f}")
    print(f"  {'node':>4s} {'dropped':>10s} {'b_pred':>10s} {'residual':>10s} {'res_std':>9s}")
    for n in sorted(node_res_sum.index):
        print(f"  {n:4d} {node_drop.get(n,0):10.1f} {node_b.get(n,0):10.1f} "
              f"{node_res_sum.get(n,0):10.1f} {node_res_std.get(n,0):9.1f}")
    return {"total_dropped": total_drop, "total_b": total_b, "total_r": total_r,
            "node_residual": dict(node_res_sum), "node_resid_std": dict(node_res_std)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-data", required=True)
    ap.add_argument("--clean-run", required=True)
    ap.add_argument("--attack-run", required=True)
    ap.add_argument("--target", type=int, default=18)
    ap.add_argument("--t-end-ns", type=int, default=200_000_000_000)
    ap.add_argument("--interval-ns", type=int, default=1_000_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[1/4] ephemeris + contact graph ...")
    sats = load_tles(args.gen_data)
    edges = load_isls(args.gen_data)
    desc = load_description(args.gen_data)
    max_isl_km = desc["max_isl_length_m"] / 1000.0
    contact = mark_contact(contact_graph(sats, edges, args.t_end_ns, args.interval_ns),
                           max_isl_km)

    print("[2/4] handover events ...")
    hevents = handover_events(args.gen_data, args.target, args.t_end_ns)

    print("[3/4] fit per-link congestion ratio beta_l on clean run ...")
    cong = fit_congestion(args.clean_run)
    df_beta = pd.DataFrame(cong["link_rows"]).sort_values("dropped", ascending=False)
    df_beta.to_csv(os.path.join(args.out, "congestion_beta.csv"), index=False)
    print(f"      total_dropped={cong['total_dropped']}  total_rx={cong['total_rx']}")
    print("      有流量的链路标定 (dropped>0 的):")
    for _, r in df_beta[df_beta["dropped"] > 0].iterrows():
        print(f"        ({r['src']},{r['dst']}) rx={r['rx']} dropped={r['dropped']} "
              f"beta={r['beta']:.4f} loss_rate={r['loss_rate']:.4f}")

    print("[4/4] build residuals (clean + attack) ...")
    clean_df = build_residual(args.clean_run, contact, hevents, cong, args.interval_ns)
    att_df = build_residual(args.attack_run, contact, hevents, cong, args.interval_ns)
    clean_df.to_csv(os.path.join(args.out, "residuals_cong_clean.csv"), index=False)
    att_df.to_csv(os.path.join(args.out, "residuals_cong_att12.csv"), index=False)

    clean_sum = summarize("gh_cong_clean (标定窗口)", clean_df)
    att_sum = summarize("gh_cong_att12 (灰洞 12)", att_df)

    print("\n================ VERIFICATION ================")
    # 诚实节点 (clean 全程无攻击): 残差应为 0 (标称, 仅有逐时隙自然波动)
    clean_res = clean_sum["node_residual"]
    honest_clean_max = max(abs(v) for v in clean_res.values())
    print(f"clean   : 最大 |诚实节点残差| = {honest_clean_max:.1f} (应为 ~0)")
    # 攻击节点 12 残差 = 纯攻击信号 (= 灰洞多丢的包)
    att_res = att_sum["node_residual"]
    n12 = att_res.get(12, 0.0)
    honest_att = {n: abs(v) for n, v in att_res.items() if n != 12}
    print(f"attack  : node12 残差 = {n12:.1f}  (纯攻击, 应为 ~+50000)")
    honest_max = max(honest_att.values(), default=0.0)
    print(f"attack  : 最大 |诚实节点残差| = {honest_max:.1f}")
    honest_std = max(att_sum["node_resid_std"].get(n, 0.0) for n in honest_att)
    print(f"attack  : 诚实节点逐时隙波动 (max res_std) = {honest_std:.1f}")
    if abs(n12) > 0:
        snr_vs_std = abs(n12) / honest_std if honest_std > 1e-9 else float("inf")
        print(f"attack  : 攻击残差 / 诚实波动 = {snr_vs_std:.0f}x")
    print("\nDONE. outputs in", args.out)


if __name__ == "__main__":
    main()