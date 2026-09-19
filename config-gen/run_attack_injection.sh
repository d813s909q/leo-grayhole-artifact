#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 2.5 攻击注入：顺序运行所有新生成的 run（内存 7.8GB 偏紧，逐条跑避免 OOM）
set -e
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs

RUNS=(gh_pl_const_0p1 gh_pl_const_0p2 gh_pl_const_0p3 gh_pl_const_0p4 gh_pl_onoff_1p0_20s gh_pl_scan_1p0_20s)

for name in "${RUNS[@]}"; do
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
echo "ALL RUNS COMPLETE"