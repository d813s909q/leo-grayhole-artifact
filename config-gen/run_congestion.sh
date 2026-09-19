#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# P0-② 真实拥塞仿真: 顺序运行 clean + attacked 两组
set -e
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs

for name in gh_cong_clean gh_cong_att12; do
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
done
echo "ALL CONGESTION RUNS COMPLETE"