#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# P1 (M2.2) 多种子顺序仿真: 10 seeds x 7 schedules = 70 runs (逐条跑避免 OOM)
set -e
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs

SPECS=(const_0p1 const_0p2 const_0p3 const_0p4 const_0p5 onoff_1p0_20s scan_1p0_20s)
N_SEEDS=10

done_count=0
skip_count=0
for spec in "${SPECS[@]}"; do
  for s in $(seq 0 $((N_SEEDS-1))); do
    name="gh_pl_${spec}_s${s}"
    if [ -f "$RUNROOT/$name/logs_ns3/isl_packet_loss.csv" ]; then
      echo ">>> SKIP $name (already has isl_packet_loss.csv)"
      skip_count=$((skip_count+1))
      continue
    fi
    echo "======================================================"
    echo ">>> RUN $name"
    echo "======================================================"
    cd "$SIM"
    ./waf --run="main_satnet --run_dir='$RUNROOT/$name'" 2>&1 | tee "$RUNROOT/$name/logs_ns3/console.txt"
    if [ ! -f "$RUNROOT/$name/logs_ns3/isl_packet_loss.csv" ]; then
      echo "!!! ERROR: no isl_packet_loss.csv for $name"
      exit 1
    fi
    echo ">>> DONE $name"
    done_count=$((done_count+1))
  done
done
echo "ALL MULTISEED RUNS COMPLETE (newly ran=$done_count, skipped=$skip_count)"