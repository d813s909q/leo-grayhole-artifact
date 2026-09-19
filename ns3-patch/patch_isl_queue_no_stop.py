#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-② 修复: 让 ISL DropTail 队列物理溢出 (而非被 BQL "stop" 机制静默丢弃).

根因: point-to-point-laser-helper.cc 在 Install 里对每个 ISL 设备做
  ndqi->GetTxQueue(0)->ConnectQueueTraces(queue)
它连上 Enqueue 回调 (PacketEnqueued) -> 队列接近满时 Stop() -> 上层
TrafficControlLayer::Send 看到 IsStopped 就 *静默丢包*, 永远不会走到
DropTail 的 Enqueue 失败分支 (那里才是我们的 CountCongestionDrop 计数点)。

修复: 移除两处 ConnectQueueTraces, 保留 NetDeviceQueueInterface 聚合
(qdisc 安装/卸载仍需要它), 使 ISL 队列退化为普通 DropTail -> 真实溢出。
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import sys

HELPER = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/helper/point-to-point-laser-helper.cc")

with open(HELPER) as f:
    src = f.read()

pairs = [
    ("  ndqiA->GetTxQueue (0)->ConnectQueueTraces (queueA);\n",
     "  // P0-2: disable BQL stop so ISL DropTail physically overflows (congestion drops)\n"
     "  // ndqiA->GetTxQueue (0)->ConnectQueueTraces (queueA);\n"),
    ("  ndqiB->GetTxQueue (0)->ConnectQueueTraces (queueB);\n",
     "  // ndqiB->GetTxQueue (0)->ConnectQueueTraces (queueB);\n"),
]

changed = 0
for old, new in pairs:
    if old in src:
        src = src.replace(old, new, 1)
        changed += 1
    else:
        print(f"WARN: anchor not found: {old!r}")

if changed == 0:
    print("ERROR: nothing patched")
    sys.exit(1)

with open(HELPER, "w") as f:
    f.write(src)

print(f"patched {changed} ConnectQueueTraces calls in {HELPER}")