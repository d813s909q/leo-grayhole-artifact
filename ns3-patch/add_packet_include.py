#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
p = os.path.join(GH_HYP, "ns3-sat-sim/simulator/contrib/satellite-network/model/grayhole-error-model.cc")
s = open(p).read()
old = '#include "ns3/abort.h"'
new = '#include "ns3/abort.h"\n#include "ns3/packet.h"'
assert s.count(old) == 1, "anchor not unique: %d" % s.count(old)
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("done")