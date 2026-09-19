#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.5 攻击注入 run 生成器
=======================
从每个模板 run (gh_pl_const12) 复制 config_ns3.properties + schedule.csv,
改写 grayhole_* 参数, 生成:

  CONSTANT 扫描:   0.1 / 0.2 / 0.3 / 0.4   (0.5 复用已有 gh_pl_const12)
  ON_OFF 对抗:     1.0 @ on=off=20s
  SCAN   (等价):   1.0 @ on=off=20s

全部带 enable_isl_packet_loss_tracking=true (isl_packet_loss_tracking_interval_ns=1e9)。

用法 (Ubuntu-20.04 root 下):
    python3 generate_attack_runs.py
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import os
import shutil

RUNS = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/temp/runs")
TEMPLATE = os.path.join(RUNS, "gh_pl_const12")

# (name, mode, drop_rate, start_ns, on_ns, off_ns)
SPECS = [
    ("gh_pl_const_0p1", "CONSTANT", 0.1, 0, 1000000000, 1000000000),
    ("gh_pl_const_0p2", "CONSTANT", 0.2, 0, 1000000000, 1000000000),
    ("gh_pl_const_0p3", "CONSTANT", 0.3, 0, 1000000000, 1000000000),
    ("gh_pl_const_0p4", "CONSTANT", 0.4, 0, 1000000000, 1000000000),
    ("gh_pl_onoff_1p0_20s", "ON_OFF", 1.0, 0, 20000000000, 20000000000),
    ("gh_pl_scan_1p0_20s", "SCAN", 1.0, 0, 20000000000, 20000000000),
]


def rewrite_grayhole(config_path, mode, drop_rate, start_ns, on_ns, off_ns):
    with open(config_path) as f:
        lines = f.readlines()

    vals = {
        "grayhole_mode": mode,
        "grayhole_drop_rate": drop_rate,
        "grayhole_start_ns": start_ns,
        "grayhole_on_ns": on_ns,
        "grayhole_off_ns": off_ns,
    }
    seen = set()
    out = []
    for ln in lines:
        stripped = ln.strip()
        key = stripped.split("=", 1)[0] if "=" in stripped else ""
        if key in vals:
            ln = f"{key}={vals[key]}\n"
            seen.add(key)
        out.append(ln)
    # 模板里没有的键(如 on_ns/off_ns)直接追加, 避免回退到 ns-3 默认值
    for key in vals:
        if key not in seen:
            out.append(f"{key}={vals[key]}\n")
    with open(config_path, "w") as f:
        f.writelines(out)


def main():
    created = []
    for name, mode, dr, st, on, off in SPECS:
        d = os.path.join(RUNS, name)
        if os.path.exists(d):
            print(f"skip (exists): {name}")
            continue
        os.makedirs(os.path.join(d, "logs_ns3"), exist_ok=True)
        shutil.copy(os.path.join(TEMPLATE, "config_ns3.properties"),
                    os.path.join(d, "config_ns3.properties"))
        shutil.copy(os.path.join(TEMPLATE, "schedule.csv"),
                    os.path.join(d, "schedule.csv"))
        rewrite_grayhole(os.path.join(d, "config_ns3.properties"),
                         mode, dr, st, on, off)
        created.append(name)
        print(f"created: {name}  (mode={mode}, drop={dr}, on={on}ns, off={off}ns)")
    print(f"\n共创建 {len(created)} 个 run: {created}")


if __name__ == "__main__":
    main()