# --- artifact path resolution (anonymized release) ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
GH_OUT = _os.environ.get("GH_OUT", _os.path.join(_HERE, "..", "outputs"))
GH_HYP = _os.environ.get("HYPATIA", _os.path.expanduser("~/hypatia"))
import csv, glob, os

base = os.path.join(GH_HYP, "integration_tests/test_manila_dalian_over_kuiper/temp/runs")

def summarize(name):
    p = os.path.join(base, name, "logs_ns3", "isl_packet_loss.csv")
    if not os.path.exists(p):
        return name, "NO LOSS FILE", -1, -1, -1
    total_rx = total_dropped = 0
    n_drop_rows = 0
    with open(p) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 6:
                continue
            rx = int(row[4]); dr = int(row[5])
            total_rx += rx; total_dropped += dr
            if dr > 0:
                n_drop_rows += 1
    return name, "ok", total_rx, total_dropped, n_drop_rows

for name in sorted(os.listdir(base)):
    if os.path.isdir(os.path.join(base, name)):
        n, s, rx, dr, drr = summarize(name)
        if s == "ok":
            print(f"{n:55s} rx={rx:9d} dropped={dr:6d} drop_rows={drr}")
        else:
            print(f"{n:55s} {s}")