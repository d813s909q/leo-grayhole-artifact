#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-② 生成真实拥塞仿真 run: UDP 突发背景流压满瓶颈 ISL (GSL 100Mbps, ISL 10Mbps 瓶颈).

关键: TCP 有闭环拥塞控制, 会自发退避, 队列 util=1.0 也几乎从不溢出 (丢包=0).
      改用 UDP 突发流 (无退避, 固定目标速率 > 瓶颈), 确定性压满 ISL 队列产生自然拥塞丢包。

产出两个 run:
  gh_cong_clean  -> 无灰洞, 1 条 UDP 突发流 17->18 (20Mbps > 10Mbps 瓶颈), 产生自然拥塞丢包 (标定 k,theta)
  gh_cong_att12  -> 同上 + 灰洞卫星 12 (CONSTANT 0.5), 验证攻击残余纯净
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import os

RUNROOT = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/temp/runs")
QUEUE_PKT = 100      # ISL 队列 (包), 默认值; UDP 超卖下确定性溢出
FROM, TO = 17, 18
UDP_RATE = 20.0      # Mbps, > 10Mbps 瓶颈 → 持续溢出
UDP_START = 0
UDP_DURATION = 200000000000   # 200s (全程)
GRAYHOLE_RATE = 0.5

NET = "reduced_kuiper_630_algorithm_free_one_only_over_isls"
DYN = "dynamic_state_100ms_for_200s"

CONFIG_TMPL = '''simulation_end_time_ns=200000000000
simulation_seed=123456789

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


def make_run(name, grayhole_cfg):
    d = os.path.join(RUNROOT, name)
    os.makedirs(os.path.join(d, "logs_ns3"), exist_ok=True)
    # config
    cfg = CONFIG_TMPL.format(net=NET, dyn=DYN, queue=QUEUE_PKT, grayhole=grayhole_cfg)
    with open(os.path.join(d, "config_ns3.properties"), "w") as f:
        f.write(cfg)
    # UDP burst schedule: id,from,to,rate_mbps,start_ns,duration_ns,additional,metadata
    with open(os.path.join(d, "udp_burst_schedule.csv"), "w") as f:
        f.write(f"0,{FROM},{TO},{UDP_RATE},{UDP_START},{UDP_DURATION},,\n")
    # 移除旧的 TCP schedule (若有), 避免误读
    sched = os.path.join(d, "schedule.csv")
    if os.path.exists(sched):
        os.remove(sched)
    print(f"created {name}: UDP burst {FROM}->{TO} @ {UDP_RATE}Mbps, "
          f"queue={QUEUE_PKT}, grayhole={grayhole_cfg!r}")


grayhole_block = (f"grayhole_satellites=12\n"
                  f"grayhole_drop_rate={GRAYHOLE_RATE}\n"
                  f"grayhole_mode=CONSTANT\n"
                  f"grayhole_start_ns=0\n")

make_run("gh_cong_clean", "")
make_run("gh_cong_att12", grayhole_block)
print("DONE")