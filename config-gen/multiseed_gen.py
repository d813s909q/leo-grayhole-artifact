#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 (M2.2) 多种子 run 生成器
===========================
对 7 个攻击 schedule × 10 个独立随机种子, 复制源 run 目录的
config_ns3.properties + schedule.csv, 并改写 `simulation_seed`,
生成 70 个 run 目录供顺序仿真。

种子约定: simulation_seed = 123456789 + s (s=0..9); s=0 即原文种子,
    可与已有单种子结果精确对账 (验证管线保真)。

用法 (Ubuntu-20.04 root):
    python3 multiseed_gen.py
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import os
import shutil

RUNS = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/temp/runs")
BASE_SEED = 123456789
N_SEEDS = 10

# (spec 名称, 源 run 目录名)  —— 源目录已含各攻击模式的 grayhole_* 参数
SPECS = [
    ("const_0p1", "gh_pl_const_0p1"),
    ("const_0p2", "gh_pl_const_0p2"),
    ("const_0p3", "gh_pl_const_0p3"),
    ("const_0p4", "gh_pl_const_0p4"),
    ("const_0p5", "gh_pl_const12"),
    ("onoff_1p0_20s", "gh_pl_onoff_1p0_20s"),
    ("scan_1p0_20s", "gh_pl_scan_1p0_20s"),
]


def rewrite_seed(config_path, seed):
    with open(config_path) as f:
        lines = f.readlines()
    out = []
    seen = False
    for ln in lines:
        if ln.strip().startswith("simulation_seed"):
            ln = f"simulation_seed={seed}\n"
            seen = True
        out.append(ln)
    if not seen:
        out.append(f"simulation_seed={seed}\n")
    with open(config_path, "w") as f:
        f.writelines(out)


def main():
    created = skipped = 0
    for spec, src_name in SPECS:
        src = os.path.join(RUNS, src_name)
        if not os.path.isdir(src):
            print(f"[ERROR] 源 run 目录缺失: {src}")
            continue
        for s in range(N_SEEDS):
            name = f"gh_pl_{spec}_s{s}"
            dst = os.path.join(RUNS, name)
            seed = BASE_SEED + s
            if os.path.isdir(dst):
                skipped += 1
                continue
            os.makedirs(os.path.join(dst, "logs_ns3"), exist_ok=True)
            shutil.copy(os.path.join(src, "config_ns3.properties"),
                        os.path.join(dst, "config_ns3.properties"))
            shutil.copy(os.path.join(src, "schedule.csv"),
                        os.path.join(dst, "schedule.csv"))
            rewrite_seed(os.path.join(dst, "config_ns3.properties"), seed)
            created += 1
            print(f"created {name}  seed={seed}  (from {src_name})")
    print(f"\n共创建 {created} 个 run, 跳过已存在 {skipped} 个")


if __name__ == "__main__":
    main()