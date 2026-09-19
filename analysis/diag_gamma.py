#!/usr/bin/env python3
"""m1.1: Markov-chain discretization (gamma) and Poisson-truncation convergence."""
# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import sys
sys.path.insert(0, _HERE)
from cusum_detector import mc_chain_arl

print("=== ARL0(lambda0=0.5, delta=1.0): gamma and x_max convergence ===")
for h in [3.0, 8.0, 18.0]:
    row = []
    for g in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        a, _ = mc_chain_arl(0.5, 1.0, h, grid_step=g)
        row.append(f"g={g:<6}:{a:.4g}")
    print(f"h={h:<5} " + "  ".join(row))
for h in [8.0, 18.0]:
    row = []
    for xm in [20, 40, 60, 80]:
        a, _ = mc_chain_arl(0.5, 1.0, h, grid_step=0.05, x_max=xm)
        row.append(f"x_max={xm:<3}:{a:.4g}")
    print(f"h={h:<5} " + "  ".join(row))

print("\n=== ARL1(lambda0=0.5, delta=1.0, attacking) ===")
for h in [3.0, 8.0, 18.0]:
    row = []
    for g in [0.1, 0.05, 0.025]:
        a, _ = mc_chain_arl(0.5, 1.0, h, grid_step=g, attacking=True)
        row.append(f"g={g:<6}:{a:.4g}")
    print(f"h={h:<5} " + "  ".join(row))
