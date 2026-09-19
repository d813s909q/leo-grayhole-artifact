#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-8 + R-9 run 生成器（2026-08-30 二轮评审补充实验）

R-8: 真实拥塞多种子 —— 10 个新种子 × (clean + att12) 共 20 run
     (原 seed=123456789 的 gh_cong_clean/att12 作为 s0 保留)
R-9: damage-vs-δ_a 低速率档 —— CONSTANT p ∈ {0.005, 0.01, 0.02, 0.05} × 3 seeds

全部由既有模板派生, 不改动原始 run。
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import os
import shutil

RUNS = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/temp/runs")

# ---------------- R-8: 真实拥塞多种子 ----------------
# 沿用 gen_congestion_runs.py 的配置模板, 仅改 simulation_seed
NET = "reduced_kuiper_630_algorithm_free_one_only_over_isls"
DYN = "dynamic_state_100ms_for_200s"
QUEUE_PKT = 100
FROM, TO = 17, 18
UDP_RATE = 20.0
UDP_START = 0
UDP_DURATION = 200000000000
GRAYHOLE_RATE = 0.5
BASE_SEED = 123456789

CONG_TMPL = '''simulation_end_time_ns=200000000000
simulation_seed={seed}

satellite_network_dir="../../gen_data/{net}"
satellite_network_routes_dir="../../gen_data/{net}/{dyn}"
dynamic_state_update_interval_ns=100000000

isl_data_rate_megabit_per_s=10.0
gsl_data_rate_megabit_per_s=100.0
isl_max_queue_size_pkts={queue}
gsl_max_queue_size_pkts=100

enable_isl_utilization_tracking=true
isl_utilization_tracking_interval_ns=1000000000

tcp_socket_type=TcpNewReno

enable_udp_burst_scheduler=true
udp_burst_schedule_filename="udp_burst_schedule.csv"

enable_isl_packet_loss_tracking=true
isl_packet_loss_tracking_interval_ns=1000000000
{grayhole}
'''

GH_BLOCK = (f"grayhole_satellites=12\n"
            f"grayhole_drop_rate={GRAYHOLE_RATE}\n"
            f"grayhole_mode=CONSTANT\n"
            f"grayhole_start_ns=0\n")


def make_cong_run(name, seed, grayhole):
    d = os.path.join(RUNS, name)
    os.makedirs(os.path.join(d, "logs_ns3"), exist_ok=True)
    cfg = CONG_TMPL.format(seed=seed, net=NET, dyn=DYN, queue=QUEUE_PKT,
                           grayhole=grayhole)
    with open(os.path.join(d, "config_ns3.properties"), "w") as f:
        f.write(cfg)
    with open(os.path.join(d, "udp_burst_schedule.csv"), "w") as f:
        f.write(f"0,{FROM},{TO},{UDP_RATE},{UDP_START},{UDP_DURATION},,\n")
    sched = os.path.join(d, "schedule.csv")
    if os.path.exists(sched):
        os.remove(sched)


# ---------------- R-9: 低速率攻击档 ----------------
# 沿用 generate_attack_runs.py 的模板复制方式 (TCP schedule 场景)
TEMPLATE = os.path.join(RUNS, "gh_pl_const_0p1")


def make_attack_run(name, drop_rate, seed):
    d = os.path.join(RUNS, name)
    os.makedirs(os.path.join(d, "logs_ns3"), exist_ok=True)
    for fn in ("config_ns3.properties", "schedule.csv"):
        src = os.path.join(TEMPLATE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, fn))
    # 改写 grayhole + seed
    cfg_path = os.path.join(d, "config_ns3.properties")
    vals = {
        "grayhole_satellites": 12,
        "grayhole_mode": "CONSTANT",
        "grayhole_drop_rate": drop_rate,
        "grayhole_start_ns": 0,
        "grayhole_on_ns": 1000000000,
        "grayhole_off_ns": 1000000000,
        "simulation_seed": seed,
    }
    seen, out = set(), []
    with open(cfg_path) as f:
        for ln in f:
            s = ln.strip()
            key = s.split("=", 1)[0] if "=" in s else ""
            if key in vals:
                ln = f"{key}={vals[key]}\n"
                seen.add(key)
            out.append(ln)
    for k, v in vals.items():
        if k not in seen:
            out.append(f"{k}={v}\n")
    with open(cfg_path, "w") as f:
        f.writelines(out)


def main():
    # R-8: 10 种子
    for k in range(1, 11):
        seed = BASE_SEED + k
        make_cong_run(f"gh_cong_clean_s{k}", seed, "")
        make_cong_run(f"gh_cong_att12_s{k}", seed, GH_BLOCK)
    print("R-8: 20 congestion runs configured (s1..s10)")

    # R-9: 低速率档 × 3 seeds
    for p in (0.005, 0.01, 0.02, 0.05):
        tag = str(p).replace("0.", "0p").replace("p0", "p0")
        for s in range(3):
            seed = BASE_SEED + 100 + s
            make_attack_run(f"gh_pl_const_d{tag}_s{s}", p, seed)
    print("R-9: 12 low-rate attack runs configured")
    print("DONE")


if __name__ == "__main__":
    main()
