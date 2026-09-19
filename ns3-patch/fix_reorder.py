#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
p = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/model/grayhole-error-model.cc")
s = open(p).read()

old = """  : m_mode (CONSTANT),
    m_dropRate (0.0),
    m_startTime (Seconds (0)),
    m_endTime (Seconds (0)),
    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_exemptLocal (true),
    m_random (CreateObject<UniformRandomVariable> ())"""
new = """  : m_mode (CONSTANT),
    m_dropRate (0.0),
    m_exemptLocal (true),
    m_startTime (Seconds (0)),
    m_endTime (Seconds (0)),
    m_onDuration (Seconds (1)),
    m_offDuration (Seconds (1)),
    m_random (CreateObject<UniformRandomVariable> ())"""

assert s.count(old) == 1, "anchor not unique: %d" % s.count(old)
open(p, "w").write(s.replace(old, new, 1))
print("done")