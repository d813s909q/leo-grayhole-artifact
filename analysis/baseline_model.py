#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.3 星历→期望基线建模 (Ephemeris-Grounded Expected Baseline)
============================================================

论文方法链条:
    观测丢包 d_{l,t}  (isl_packet_loss.csv, 逐链路 l 逐时隙 t)
    期望基线 b_{l,t}  = b_contact + b_handover + b_congestion   (本模块输出)
    攻击残余 r_{l,t}  = d_{l,t} - b_{l,t}                        (→ 2.4 CUSUM 输入)

三个期望分量的物理来源:
  1) b_contact    : 星历(SGP4/TLE)×ISL 拓扑 → 时变链路距离 → 接触图(链路存在性)。
                    链路断开时隙: 期望 rx=0, 丢包未定义(路由已绕开), 标记 link_down。
                    链路存在时隙: ISL 激光链路无误码模型 → 期望丢包 0。
  2) b_handover   : fstate(100ms) 的 next_hop 变化事件 → 切换时隙标记。
                    本平台路由由 fstate 预置(ns-3 无收敛期), 切换损失标定为
                    handover_loss_pkts/事件 (默认 0, 可由有损切换场景标定)。
  3) b_congestion : isl_utilization u_{l,t} 驱动的队列溢出期望:
                    b_cong = rx_{l,t} * max(0, k*(u_{l,t} - theta))
                    参数 (k, theta) 由无攻击 run 标定:
                      - 无攻击 dropped=0 → 无拥塞丢包证据 → k=0 (保守下界)
                      - theta = 观测到的最大利用率 (记录, 供后续拥塞场景重标定)

数据源:
  gen_data/  : tles.txt, isls.txt, description.txt, dynamic_state_100ms_for_200s/fstate_*.txt
  run/logs_ns3/: isl_packet_loss.csv, isl_utilization.csv

用法:
  python3 baseline_model.py --gen-data <dir> --run name=<run_dir> [--run name2=<dir2> ...]
                            --target 18 --out <out_dir>
"""

import argparse
import math
import os
import glob
from collections import defaultdict

import pandas as pd
from sgp4.api import Satrec, jday

FSTATE_STEP_NS = 100_000_000  # 100ms


# ----------------------------------------------------------------------
# 星历: TLE → SGP4 卫星位置
# ----------------------------------------------------------------------
def load_tles(gen_dir):
    """tles.txt: 首行 header, 之后 (name, l1, l2) 三行一组. 返回按卫星 id 排序的 Satrec 列表."""
    path = os.path.join(gen_dir, "tles.txt")
    lines = [l.rstrip() for l in open(path) if l.strip()]
    body = lines[1:]
    assert len(body) % 3 == 0, "tles.txt body is not a multiple of 3 lines"
    sats = []
    for i in range(0, len(body), 3):
        sats.append(Satrec.twoline2rv(body[i + 1], body[i + 2]))
    return sats


def load_isls(gen_dir):
    """isls.txt: 每行 'a b' 一条 ISL 边. 返回 [(a,b), ...]"""
    edges = []
    for line in open(os.path.join(gen_dir, "isls.txt")):
        a, b = map(int, line.split())
        edges.append((a, b))
    return edges


def load_description(gen_dir):
    d = {}
    for line in open(os.path.join(gen_dir, "description.txt")):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


# ----------------------------------------------------------------------
# 分量 1: 接触图 (星历驱动)
# ----------------------------------------------------------------------
def contact_graph(sats, edges, t_end_ns, interval_ns):
    """SGP4 逐 100ms 算 ISL 两端距离 → 按 interval 聚合 → 链路存在性.

    返回 DataFrame[link_src, link_dst, t_start, t_end, dist_mean_km, dist_max_km, link_up]
    """
    jd0, fr0 = jday(2000, 1, 1, 12, 0, 0)  # TLE epoch (00001.00000000)

    # 每条边在每个 100ms 步的距离 (km)
    n_steps = t_end_ns // FSTATE_STEP_NS
    cache = {}  # sat_id -> list[(x,y,z)] per step
    def positions(sid):
        if sid not in cache:
            sat = sats[sid]
            pts = []
            for s in range(n_steps):
                # 纳秒 → 秒 (÷1e9) → 天 (÷86400), 加到 TLE epoch 的 fr 上
                e, r, v = sat.sgp4(jd0, fr0 + s * FSTATE_STEP_NS / 1e9 / 86400.0)
                assert e == 0, f"sgp4 error {e} at step {s}"
                pts.append(r)
            cache[sid] = pts
        return cache[sid]

    rows = []
    per_interval = interval_ns // FSTATE_STEP_NS
    for (a, b) in edges:
        pa, pb = positions(a), positions(b)
        dists = [math.dist(pa[s], pb[s]) for s in range(n_steps)]
        for k in range(t_end_ns // interval_ns):
            seg = dists[k * per_interval:(k + 1) * per_interval]
            rows.append({
                "link_src": a, "link_dst": b,
                "t_start": k * interval_ns, "t_end": (k + 1) * interval_ns,
                "dist_mean_km": sum(seg) / len(seg),
                "dist_max_km": max(seg),
            })
    df = pd.DataFrame(rows)
    return df


def mark_contact(df, max_isl_len_km):
    df["link_up"] = df["dist_max_km"] <= max_isl_len_km
    return df


# ----------------------------------------------------------------------
# 分量 2: 切换事件 (fstate 驱动)
# ----------------------------------------------------------------------
def handover_events(gen_dir, target, t_end_ns):
    """扫描 fstate_<t>.txt, 提取每个 node 到 target 的 next_hop 变化事件.

    Hypatia fstate 增量机制: fstate_0 是全量初始路由; 之后非空文件只含
    变化的条目 (增量覆盖); 空文件 = 该时刻路由无变化.
    因此必须维护持久状态字典, 空文件跳过 (保持状态).

    返回 DataFrame[t_ns, node, target, next_hop_old, next_hop_new]
    """
    events = []
    state_dir = os.path.join(gen_dir, "dynamic_state_100ms_for_200s")
    files = sorted(glob.glob(os.path.join(state_dir, "fstate_*.txt")),
                   key=lambda p: int(os.path.basename(p)[7:-4]))
    state = {}  # node -> next_hop (对固定 target)
    n_updates = 0
    for fp in files:
        t_ns = int(os.path.basename(fp)[7:-4])
        if t_ns > t_end_ns:
            break
        lines = [l.strip() for l in open(fp) if l.strip()]
        if not lines:
            continue  # 空文件: 无路由变化, 状态保持
        n_updates += 1
        for line in lines:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            node, tgt, nh = int(parts[0]), int(parts[1]), int(parts[2])
            if tgt != target:
                continue
            if node in state and state[node] != nh:
                events.append({
                    "t_ns": t_ns, "node": node, "target": target,
                    "next_hop_old": state[node], "next_hop_new": nh,
                })
            state[node] = nh
    print(f"      [fstate] non-empty route updates={n_updates}, "
          f"target={target} handover events={len(events)}")
    return pd.DataFrame(events)


# ----------------------------------------------------------------------
# 分量 3: 拥塞 (利用率驱动, 无攻击 run 标定)
# ----------------------------------------------------------------------
def fit_congestion(baseline_run_dir):
    """从无攻击 run 标定 (k, theta).

    无攻击 dropped 全 0 → 无拥塞丢包证据 → k=0;
    theta = 观测最大利用率 (记录为拥塞阈值下界).
    返回 dict{k, theta, max_observed_dropped}
    """
    util_fp = os.path.join(baseline_run_dir, "logs_ns3", "isl_utilization.csv")
    loss_fp = os.path.join(baseline_run_dir, "logs_ns3", "isl_packet_loss.csv")
    u = pd.read_csv(util_fp, header=None,
                    names=["src", "dst", "t_start", "t_end", "util"])
    d = pd.read_csv(loss_fp, header=None,
                    names=["src", "dst", "t_start", "t_end", "rx", "dropped"])
    theta = float(u["util"].max()) if len(u) else 0.0
    max_dr = int(d["dropped"].max()) if len(d) else 0
    k = 0.0 if max_dr == 0 else None  # k=None 表示需要重标定 (有自然丢包的场景)
    return {"k": k, "theta": theta, "max_observed_dropped": max_dr}


# ----------------------------------------------------------------------
# 基线合成 + 残差
# ----------------------------------------------------------------------
def utilization_recv(util_df):
    """isl_utilization 是发送视角 (src 发往 dst); isl_packet_loss 是接收视角
    (src=接收设备所在卫星, dst=对端). 反转方向对齐为接收视角."""
    v = util_df.copy()
    tmp = v["src"].copy()
    v["src"] = v["dst"]
    v["dst"] = tmp
    return v


def interval_key(t_start, interval_ns):
    return t_start // interval_ns


def build_baseline(run_dir, contact, hevents, cong, interval_ns,
                   handover_loss_pkts=0.0, handover_window_s=1.0):
    """对齐观测粒度, 合成 b_{l,t} 并计算残差 r = d - b."""
    loss_fp = os.path.join(run_dir, "logs_ns3", "isl_packet_loss.csv")
    util_fp = os.path.join(run_dir, "logs_ns3", "isl_utilization.csv")
    d = pd.read_csv(loss_fp, header=None,
                    names=["src", "dst", "t_start", "t_end", "rx", "dropped"])
    u = utilization_recv(pd.read_csv(util_fp, header=None,
                                     names=["src", "dst", "t_start", "t_end", "util"]))

    # 接收视角链路键: (src=接收设备所在卫星, dst=对端). 接触图是无向边 → 双向查
    up_lookup = {}
    for _, r in contact.iterrows():
        k1, k2 = (r.link_src, r.link_dst), (r.link_dst, r.link_src)
        v = bool(r.link_up)
        up_lookup.setdefault(k1, {})[interval_key(r.t_start, interval_ns)] = v
        up_lookup.setdefault(k2, {})[interval_key(r.t_start, interval_ns)] = v

    # 切换事件 → (接收链路, 时隙) 窗口标记: node 的 next_hop 变化涉及链路 node<->old/new
    handover_marks = defaultdict(int)
    win_ns = int(handover_window_s * 1e9)
    for _, e in hevents.iterrows():
        node, nh_old, nh_new = int(e.node), int(e.next_hop_old), int(e.next_hop_new)
        for nb in {nh_old, nh_new}:
            for lag in range(0, win_ns, interval_ns):
                slot = interval_key(int(e.t_ns) + lag, interval_ns)
                handover_marks[(node, nb, slot)] += 1
                handover_marks[(nb, node, slot)] += 1

    # 利用率查表: (src,dst,slot) -> util (接收视角)
    util_lookup = {}
    for _, r in u.iterrows():
        util_lookup[(int(r.src), int(r.dst),
                     interval_key(r.t_start, interval_ns))] = float(r.util)

    k, theta = cong["k"], cong["theta"]
    rows = []
    for _, r in d.iterrows():
        src, dst, ts = int(r.src), int(r.dst), interval_key(r.t_start, interval_ns)
        rx, dropped = int(r.rx), int(r.dropped)
        link_up = up_lookup.get((src, dst), {}).get(ts, True)

        # 分量 1: 接触
        b_contact = 0 if link_up else 0  # 断链时隙丢包未定义, 由 link_down 列体现

        # 分量 2: 切换
        h = handover_marks.get((src, dst, ts), 0)
        b_handover = h * handover_loss_pkts

        # 分量 3: 拥塞 (k=None → 未标定, 置 0 并标记)
        util = util_lookup.get((src, dst, ts), 0.0)
        if k is None:
            b_cong = 0.0
        else:
            b_cong = rx * max(0.0, k * (util - theta))

        b_total = b_contact + b_handover + b_cong
        rows.append({
            "src": src, "dst": dst, "slot": ts,
            "t_start": r.t_start, "t_end": r.t_end,
            "rx": rx, "dropped": dropped,
            "link_up": link_up,
            "handover_events": h,
            "util": util,
            "b_contact": b_contact, "b_handover": b_handover,
            "b_congestion": round(b_cong, 4),
            "b_total": round(b_total, 4),
            "residual": round(dropped - b_total, 4),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-data", required=True, help="satgenpy gen_data dir")
    ap.add_argument("--run", action="append", required=True,
                    help="name=<run_dir> (可多次; 第一个视为无攻击基线 run)")
    ap.add_argument("--target", type=int, default=18, help="fstate target id (默认 18)")
    ap.add_argument("--t-end-ns", type=int, default=200_000_000_000)
    ap.add_argument("--interval-ns", type=int, default=1_000_000_000,
                    help="时隙粒度, 需与 isl_packet_loss interval 一致")
    ap.add_argument("--handover-loss-pkts", type=float, default=0.0,
                    help="每次切换事件的期望丢包 (本平台默认 0, 可标定)")
    ap.add_argument("--handover-window-s", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 星历 → 接触图
    print("[1/4] loading ephemeris ...")
    sats = load_tles(args.gen_data)
    edges = load_isls(args.gen_data)
    desc = load_description(args.gen_data)
    max_isl_km = desc["max_isl_length_m"] / 1000.0
    print(f"      sats={len(sats)}  isls={len(edges)}  max_isl_len={max_isl_km:.1f} km")

    print("[2/4] building contact graph (SGP4) ...")
    contact = mark_contact(contact_graph(sats, edges, args.t_end_ns, args.interval_ns),
                           max_isl_km)
    contact.to_csv(os.path.join(args.out, "contact_graph.csv"), index=False)
    n_down = int((~contact["link_up"]).sum())
    print(f"      interval-x-link rows={len(contact)}  link-down intervals={n_down}")

    # fstate → 切换事件
    print("[3/4] scanning fstate handover events (target=%d) ..." % args.target)
    hevents = handover_events(args.gen_data, args.target, args.t_end_ns)
    hevents.to_csv(os.path.join(args.out, "handover_events.csv"), index=False)
    print(f"      handover events={len(hevents)}")

    # 无攻击 run 标定拥塞参数
    runs = dict(x.split("=", 1) for x in args.run)
    names = list(runs.keys())
    base_name = names[0]
    print(f"[4/4] fitting congestion on baseline run '{base_name}' ...")
    cong = fit_congestion(runs[base_name])
    print(f"      k={cong['k']}  theta={cong['theta']:.4f}  "
          f"max_observed_dropped={cong['max_observed_dropped']}")

    pd.DataFrame([{"param": "k", "value": cong["k"]},
                  {"param": "theta", "value": cong["theta"]},
                  {"param": "max_observed_dropped", "value": cong["max_observed_dropped"]},
                  {"param": "handover_loss_pkts", "value": args.handover_loss_pkts}]).to_csv(
        os.path.join(args.out, "congestion_params.csv"), index=False)

    for name in names:
        res = build_baseline(runs[name], contact, hevents, cong,
                             args.interval_ns, args.handover_loss_pkts,
                             args.handover_window_s)
        fp = os.path.join(args.out, f"residuals_{name}.csv")
        res.to_csv(fp, index=False)
        pos = res[res["residual"] > 0]
        agg = pos.groupby(["src", "dst"])["residual"].sum().sort_values(ascending=False)
        print(f"\n=== run '{name}' ===")
        print(f"      total observed dropped = {int(res['dropped'].sum())}")
        print(f"      total baseline b       = {res['b_total'].sum():.1f}")
        print(f"      total residual r       = {res['residual'].sum():.1f}")
        print(f"      links with r>0: {len(agg)}")
        for (s, t), v in agg.items():
            print(f"        ({s:2d} <- {t:2d})  residual={v:.1f}")
    print("\nDONE. outputs in", args.out)


if __name__ == "__main__":
    main()