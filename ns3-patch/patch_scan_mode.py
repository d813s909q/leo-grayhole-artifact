#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch the Linux-side ns-3 sources to implement spatial SCAN rotation
(grayhole rotates its attack across ISL links) and keep ON_OFF unchanged.
Run as root on Ubuntu-20.04."""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))

import io

H = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/model/grayhole-error-model.h")
CC = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/model/grayhole-error-model.cc")
MAIN = os.path.join(GH_HYP, "ns3-sat-sim/simulator/scratch/main_satnet/main_satnet.cc")


def patch(path, replacements):
    with io.open(path, "r") as f:
        s = f.read()
    for old, new in replacements:
        if old not in s:
            raise SystemExit(f"[FAIL] pattern not found in {path}:\n{old}")
        s = s.replace(old, new, 1)
    with io.open(path, "w") as f:
        f.write(s)
    print(f"[OK] patched {path}")


# ---------------------------------------------------------------- header --
patch(H, [
    ("""  void AddLocalAddress (Ipv4Address addr);
""",
     """  void AddLocalAddress (Ipv4Address addr);

  void SetScanIndex (uint32_t idx);
  void SetScanLinkCount (uint32_t count);
"""),
    ("""  Ptr<UniformRandomVariable> m_random;
};
""",
     """  Ptr<UniformRandomVariable> m_random;
  uint32_t m_scanIndex;
  uint32_t m_scanLinkCount;
};
"""),
])

# ------------------------------------------------------------ source file --
patch(CC, [
    # add the uinteger header for UintegerValue accessors
    ("""#include "ns3/boolean.h"
""",
     """#include "ns3/boolean.h"
#include "ns3/uinteger.h"
"""),
    # attributes
    ("""    .AddAttribute ("ExemptLocalDestination",
""",
     """    .AddAttribute ("ScanIndex",
                   "Ordinal of this device among the node's ISL links (SCAN mode).",
                   UintegerValue (0),
                   MakeUintegerAccessor (&GrayholeErrorModel::m_scanIndex),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("ScanLinkCount",
                   "Total number of ISL links the SCAN attack rotates across.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&GrayholeErrorModel::m_scanLinkCount),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("ExemptLocalDestination",
"""),
    # constructor init
    ("""    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_random (CreateObject<UniformRandomVariable> ())
""",
     """    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_random (CreateObject<UniformRandomVariable> ()),
    m_scanIndex (0),
    m_scanLinkCount (1)
"""),
    # setters
    ("""void
GrayholeErrorModel::AddLocalAddress (Ipv4Address addr)
{
  m_localAddresses.insert (addr);
}
""",
     """void
GrayholeErrorModel::AddLocalAddress (Ipv4Address addr)
{
  m_localAddresses.insert (addr);
}

void
GrayholeErrorModel::SetScanIndex (uint32_t idx)
{
  m_scanIndex = idx;
}

void
GrayholeErrorModel::SetScanLinkCount (uint32_t count)
{
  m_scanLinkCount = count;
}
"""),
    # SCAN logic
    ("""    case ON_OFF:
    case SCAN:
      {
        int64_t period = (m_onDuration + m_offDuration).GetNanoSeconds ();
        if (period <= 0)
          {
            return true;
          }
        int64_t elapsed = (now - m_startTime).GetNanoSeconds ();
        int64_t phase = elapsed % period;
        return phase < m_onDuration.GetNanoSeconds ();
      }
""",
     """    case ON_OFF:
      {
        int64_t period = (m_onDuration + m_offDuration).GetNanoSeconds ();
        if (period <= 0)
          {
            return true;
          }
        int64_t elapsed = (now - m_startTime).GetNanoSeconds ();
        int64_t phase = elapsed % period;
        return phase < m_onDuration.GetNanoSeconds ();
      }

    case SCAN:
      {
        // Rotate the attack across the node's ISL links: link i is attacked
        // during the i-th dwell window of each full rotation cycle.
        uint32_t n = m_scanLinkCount > 0 ? m_scanLinkCount : 1;
        if (m_scanIndex >= n)
          {
            return false;
          }
        int64_t dwell = m_onDuration.GetNanoSeconds ();
        if (dwell <= 0)
          {
            return true;
          }
        int64_t elapsed = (now - m_startTime).GetNanoSeconds ();
        uint32_t active = static_cast<uint32_t>((elapsed / dwell) % n);
        return active == m_scanIndex;
      }
"""),
])

# ------------------------------------------------------- main_satnet.cc ----
patch(MAIN, [
    ("""            for (uint32_t i = 0; i < node->GetNDevices(); i++) {
                Ptr<NetDevice> dev = node->GetDevice(i);
                Ptr<GrayholeErrorModel> em = CreateObject<GrayholeErrorModel>();
                em->SetAttribute("DropRate", DoubleValue(grayhole_drop_rate));
                em->SetAttribute("StartTime", TimeValue(grayhole_start));
                em->SetMode(GrayholeErrorModel::ParseMode(grayhole_mode));
                em->SetOnDuration(grayhole_on);
                em->SetOffDuration(grayhole_off);
""",
     """            // SCAN mode rotates across the node's ISL (laser) links only.
            uint32_t n_devices = node->GetNDevices();
            std::vector<uint32_t> laser_idx;
            for (uint32_t i = 0; i < n_devices; i++) {
                if (DynamicCast<PointToPointLaserNetDevice>(node->GetDevice(i)) != 0) {
                    laser_idx.push_back(i);
                }
            }
            uint32_t n_lasers = laser_idx.size();
            for (uint32_t i = 0; i < n_devices; i++) {
                Ptr<NetDevice> dev = node->GetDevice(i);
                Ptr<GrayholeErrorModel> em = CreateObject<GrayholeErrorModel>();
                em->SetAttribute("DropRate", DoubleValue(grayhole_drop_rate));
                em->SetAttribute("StartTime", TimeValue(grayhole_start));
                em->SetMode(GrayholeErrorModel::ParseMode(grayhole_mode));
                em->SetOnDuration(grayhole_on);
                em->SetOffDuration(grayhole_off);
                em->SetScanLinkCount(n_lasers > 0 ? n_lasers : 1);
                em->SetScanIndex(n_lasers);  // default: never scanned
                if (DynamicCast<PointToPointLaserNetDevice>(dev) != 0) {
                    uint32_t ord = 0;
                    while (ord < laser_idx.size() && laser_idx[ord] != i) ord++;
                    em->SetScanIndex(ord);
                }
"""),
    ("""                      << ", terminal-exempt local addrs=" << n_local_addrs << ")" << std::endl;
""",
     """                      << ", terminal-exempt local addrs=" << n_local_addrs
                      << ", isl-links=" << n_lasers << ")" << std::endl;
"""),
])

print("\nAll patches applied successfully.")