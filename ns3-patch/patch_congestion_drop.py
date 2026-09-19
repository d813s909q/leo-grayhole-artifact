#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-② 补丁: 在 ISL 激光设备中统计「出队溢出」导致的自然拥塞丢包.

语义: isl_packet_loss.csv 采用「接收视角」(src=接收卫星/丢包发生地, dst=对端).
灰洞丢包发生在恶意卫星的 Receive() (error model), 已计入 src=恶意卫星.
拥塞丢包发生在发送方卫星的出队队列 Send() 的 Enqueue 溢出 —— 需按接收视角
记到「对端(接收方)设备」的 dropped 计数上, 与利用率(翻转后同为接收视角)对齐.
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import sys

H = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/model/point-to-point-laser-net-device.h")
C = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/model/point-to-point-laser-net-device.cc")


def patch(path, replacements):
    s = open(path, encoding="utf-8").read()
    for old, new in replacements:
        if old not in s:
            print(f"FATAL: pattern not found in {path}:\n  {old[:120]}")
            sys.exit(1)
        if s.count(old) > 1:
            print(f"WARN: pattern ambiguous ({s.count(old)}x) in {path}: {old[:80]}")
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)
    print(f"patched {path}")


# 1) header: 声明计数方法
patch(H, [
    (
        "    void EnableDropTracking(int64_t interval_ns);\n    const std::vector<std::pair<uint64_t, uint64_t>>& FinalizeDropTracking(void);\n",
        "    void EnableDropTracking(int64_t interval_ns);\n"
        "    void CountCongestionDrop(void);\n"
        "    const std::vector<std::pair<uint64_t, uint64_t>>& FinalizeDropTracking(void);\n",
    ),
])

# 2) cc: Send() 溢出分支 → 通知对端设备计数
patch(C, [
    (
        "  // Enqueue may fail (overflow)\n\n  m_macTxDropTrace (packet);\n  return false;\n",
        "  // Enqueue may fail (overflow)\n\n  m_macTxDropTrace (packet);\n\n"
        "  // Natural congestion (egress queue overflow): attribute in receive view to the\n"
        "  // peer device (the intended receiver), so isl_packet_loss.csv's dropped column\n"
        "  // aligns with other link-level fields (rx/util flip to receive view).\n"
        "  if (m_drop_tracking_enabled)\n"
        "    {\n"
        "      for (std::size_t i = 0; i < m_channel->GetNDevices(); ++i)\n"
        "        {\n"
        "          Ptr<PointToPointLaserNetDevice> peer = m_channel->GetPointToPointLaserDevice(i);\n"
        "          if (peer && peer.get() != this)\n"
        "            {\n"
        "              peer->CountCongestionDrop ();\n"
        "              break;\n"
        "            }\n"
        "        }\n"
        "    }\n  return false;\n",
    ),
])

# 3) cc: 定义 CountCongestionDrop (紧跟 AdvanceDropInterval 之后)
patch(C, [
    (
        "void\nPointToPointLaserNetDevice::AdvanceDropInterval(void) {\n    int64_t now_ns = Simulator::Now().GetNanoSeconds();\n",
        "void\nPointToPointLaserNetDevice::CountCongestionDrop(void) {\n"
        "    if (!m_drop_tracking_enabled) return;\n"
        "    AdvanceDropInterval();\n"
        "    m_dropped_packets_current++;\n}\n\n"
        "void\nPointToPointLaserNetDevice::AdvanceDropInterval(void) {\n    int64_t now_ns = Simulator::Now().GetNanoSeconds();\n",
    ),
])

print("ALL PATCHES APPLIED")