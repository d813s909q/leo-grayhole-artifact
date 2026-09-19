#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 isl_utilization.csv 计算每颗卫星的:
   1) 承载量 (util*时长 求和)          —— 旧的静态选择指标
   2) 路径持久性 (在转发路径上的时间占比) —— 新的动态接触图+持久性指标

isl_utilization.csv: <src>,<dst>,<start_ns>,<end_ns>,<util>
含义: ISL 卫星(src)->卫星(dst), 在 [start,end) 内利用率为 util。
      util>0 表示该时段该有向链路有转发流量。
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/temp/runs/kuiper_630_isls_sat_one_17_to_18_with_TcpNewReno_at_10_Mbps/logs_ns3/isl_utilization.csv")

T_END = 200_000_000_000  # 200s, 由场景决定

# 每节点: 承载量 (util*时长); 占用区间列表
load = defaultdict(float)
intervals = defaultdict(list)

with open(PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = line.split(",")
        if len(p) < 5:
            continue
        src, dst = int(p[0]), int(p[1])
        start, end = int(p[2]), int(p[3])
        util = float(p[4]) if len(p) > 4 else 0.0
        if start >= end:
            continue
        if util <= 0.0:
            # util=0 的整段记录不参与 (无流量)
            continue
        dur = end - start
        nodes = {src, dst}
        for n in nodes:
            load[n] += util * dur
            intervals[n].append((start, end))

def union_len(ivs):
    if not ivs:
        return 0.0
    ivs = sorted(ivs)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total

print(f"sim end = {T_END/1e9:.0f}s")
print(f"\n=== 节点排名: 承载量(旧指标) vs 路径持久性(新指标) ===")
print(f"{'sat':>4} {'load(util*s)':>13} {'on_path_s':>10} {'persistence':>11}")
rows = []
for n in sorted(load):
    onpath = union_len(intervals[n])
    pers = onpath / T_END
    rows.append((n, load[n], onpath, pers))

rows.sort(key=lambda r: -r[3])
print("--- 按路径持久性降序 ---")
for n, ld, onpath, pers in rows:
    print(f"{n:>4} {ld:>13.2f} {onpath/1e9:>10.2f} {pers*100:>10.1f}%")

print("\n--- 按承载量(旧指标)降序 ---")
for n, ld, onpath, pers in sorted(rows, key=lambda r: -r[1]):
    print(f"{n:>4} {ld:>13.2f} {onpath/1e9:>10.2f} {pers*100:>10.1f}%")