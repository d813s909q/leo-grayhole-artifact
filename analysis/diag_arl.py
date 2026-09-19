# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import sys, math
sys.path.insert(0, _HERE)
from cusum_detector import mc_chain_arl, choose_h_chain

print("=== mc_chain_arl vs h for several lambda0 (delta=1.0) ===")
for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    vals = []
    for h in [3, 5, 8, 10, 12, 15, 20, 30, 50, 100]:
        a, n = mc_chain_arl(lam, 1.0, h)
        vals.append(f"{h}:{a:.0f}")
    print(f"lam={lam:5.2f}  " + "  ".join(vals))

print("\n=== choose_h_chain target=1e5 ===")
for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    h = choose_h_chain(lam, 1.0, 1e5)
    print(f"lam={lam:5.2f} -> h={h:.3f}")