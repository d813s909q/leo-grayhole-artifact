#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 重新运行 ON_OFF / SCAN (修正 20s on/off + 空间轮转后)
set -e
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs

for name in gh_pl_onoff_1p0_20s gh_pl_scan_1p0_20s; do
  echo "======================================================"
  echo ">>> RERUN $name"
  echo "======================================================"
  cd "$SIM"
  ./waf --run="main_satnet --run_dir='$RUNROOT/$name'" 2>&1 | tee "$RUNROOT/$name/logs_ns3/console_rerun.txt"
  if [ ! -f "$RUNROOT/$name/logs_ns3/isl_packet_loss.csv" ]; then
    echo "!!! ERROR: no isl_packet_loss.csv for $name"
    exit 1
  fi
  echo ">>> DONE $name"
done
echo "ALL RERUN COMPLETE"