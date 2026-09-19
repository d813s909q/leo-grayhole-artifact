#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 GrayholeErrorModel 与 main_satnet 做『仅丢转发包、不丢终点包』语义的精确插入（容忍行尾空格）。

语义：恶意卫星只丢弃『经它转发』的数据包；目的地址等于本节点自身接口地址的
『终点流量』不丢（经典选择性转发/灰洞定义）。
"""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import sys, shutil, re

BASE = os.path.join(GH_HYP, "ns3-sat-sim/simulator")
H    = BASE + "/contrib/satellite-network/model/grayhole-error-model.h"
CC   = BASE + "/contrib/satellite-network/model/grayhole-error-model.cc"
MAIN = BASE + "/scratch/main_satnet/main_satnet.cc"


def _pattern(old):
    parts = []
    for ln in old.split("\n"):
        e = re.escape(ln.rstrip())
        parts.append((e + r"[ ]*") if e else "")
    return "\n".join(parts)


def fuzzy_edit(path, pairs):
    shutil.copy2(path, path + ".bac2")
    with open(path) as f:
        s = f.read()
    for i, (old, new) in enumerate(pairs):
        pat = _pattern(old)
        cnt = len(re.findall(pat, s))
        if cnt != 1:
            print("FAIL %s edit#%d: matches=%d\n---anchor---\n%s\n---" % (path, i, cnt, old[:300]))
            sys.exit(1)
        s, n = re.subn(pat, lambda m: new, s, count=1)
        if n != 1:
            print("FAIL %s edit#%d: subn=%d" % (path, i, n))
            sys.exit(1)
    with open(path, "w") as f:
        f.write(s)
    print("OK   %s  (%d edits)" % (path, len(pairs)))


# ---------------- grayhole-error-model.h ----------------
fuzzy_edit(H, [
    ('''#include "ns3/error-model.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"''',
     '''#include "ns3/error-model.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"
#include "ns3/ipv4-address.h"

#include <set>'''),
    ('''  void SetOnDuration (Time onDuration);
  void SetOffDuration (Time offDuration);''',
     '''  void SetOnDuration (Time onDuration);
  void SetOffDuration (Time offDuration);

  void AddLocalAddress (Ipv4Address addr);'''),
    ('''  Mode m_mode;
  double m_dropRate;''',
     '''  Mode m_mode;
  double m_dropRate;
  bool m_exemptLocal;
  std::set<Ipv4Address> m_localAddresses;'''),
])

# ---------------- grayhole-error-model.cc ----------------
fuzzy_edit(CC, [
    ('''#include "ns3/string.h"
#include "ns3/abort.h"''',
     '''#include "ns3/string.h"
#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/ipv4-header.h"
#include "ns3/ppp-header.h"'''),
    ('''    .AddAttribute ("OffDuration",
                   "Length of the benign window (ON_OFF/SCAN mode).",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&GrayholeErrorModel::m_offDuration),
                   MakeTimeChecker ())
    ;''',
     '''    .AddAttribute ("OffDuration",
                   "Length of the benign window (ON_OFF/SCAN mode).",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&GrayholeErrorModel::m_offDuration),
                   MakeTimeChecker ())
    .AddAttribute ("ExemptLocalDestination",
                   "Do not drop packets destined to this node (dropped packets are forwarded only).",
                   BooleanValue (true),
                   MakeBooleanAccessor (&GrayholeErrorModel::m_exemptLocal),
                   MakeBooleanChecker ())
    ;'''),
    ('''    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_random (CreateObject<UniformRandomVariable> ())''',
     '''    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_exemptLocal (true),
    m_random (CreateObject<UniformRandomVariable> ())'''),
    ('''void
GrayholeErrorModel::SetOffDuration (Time offDuration)
{
  m_offDuration = offDuration;
}''',
     '''void
GrayholeErrorModel::SetOffDuration (Time offDuration)
{
  m_offDuration = offDuration;
}

void
GrayholeErrorModel::AddLocalAddress (Ipv4Address addr)
{
  m_localAddresses.insert (addr);
}'''),
    ('''bool
GrayholeErrorModel::DoCorrupt (Ptr<Packet> p)
{
  if (!IsAttackActive ())
    {
      return false;
    }
  return m_random->GetValue () < m_dropRate;
}''',
     '''bool
GrayholeErrorModel::DoCorrupt (Ptr<Packet> p)
{
  if (!IsAttackActive ())
    {
      return false;
    }

  if (m_exemptLocal)
    {
      // Strip the point-to-point header, then inspect the IPv4 destination.
      Ptr<Packet> copy = p->Copy ();
      PppHeader ppp;
      if (copy->GetSize () < ppp.GetSerializedSize ())
        {
          return false;  // too small to carry IP; do not drop control frames
        }
      copy->RemoveHeader (ppp);
      Ipv4Header ipHeader;
      if (copy->PeekHeader (ipHeader))
        {
          if (m_localAddresses.count (ipHeader.GetDestination ()) > 0)
            {
              return false;  // destined to this node: terminal traffic, not forwarding
            }
        }
    }

  return m_random->GetValue () < m_dropRate;
}'''),
])

# ---------------- main_satnet.cc ----------------
fuzzy_edit(MAIN, [
    ('#include "ns3/grayhole-error-model.h"',
     '#include "ns3/grayhole-error-model.h"\n#include "ns3/ipv4.h"'),
    ('''        for (int64_t sat_id : evil_ids) {
            Ptr<Node> node = topology->GetSatelliteNodes().Get(sat_id);
            for (uint32_t i = 0; i < node->GetNDevices(); i++) {''',
     '''        for (int64_t sat_id : evil_ids) {
            Ptr<Node> node = topology->GetSatelliteNodes().Get(sat_id);
            Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
            uint32_t n_local_addrs = 0;
            for (uint32_t ii = 0; ii < ipv4->GetNInterfaces(); ii++) {
                n_local_addrs += ipv4->GetNAddresses(ii);
            }
            for (uint32_t i = 0; i < node->GetNDevices(); i++) {'''),
    ('''                em->SetOnDuration(grayhole_on);
                em->SetOffDuration(grayhole_off);''',
     '''                em->SetOnDuration(grayhole_on);
                em->SetOffDuration(grayhole_off);

                // Register this node's own addresses so terminal (non-forwarded)
                // traffic is never dropped: the grayhole only attacks forwarded packets.
                for (uint32_t ii = 0; ii < ipv4->GetNInterfaces(); ii++) {
                    for (uint32_t aa = 0; aa < ipv4->GetNAddresses(ii); aa++) {
                        em->AddLocalAddress(ipv4->GetAddress(ii, aa).GetLocal());
                    }
                }'''),
    ('''            std::cout << "  > Injected grayhole on satellite " << sat_id
                      << " (drop_rate=" << grayhole_drop_rate
                      << ", mode=" << grayhole_mode << ")" << std::endl;''',
     '''            std::cout << "  > Injected grayhole on satellite " << sat_id
                      << " (drop_rate=" << grayhole_drop_rate
                      << ", mode=" << grayhole_mode
                      << ", terminal-exempt local addrs=" << n_local_addrs << ")" << std::endl;'''),
])

print("ALL_OK")