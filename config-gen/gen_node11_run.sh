#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# M2.4 (#11) tie-break 稳健性: 攻击者换为 node 11 (与 12 并列 path persistence 61.5%)
# 复制 gh_pl_const12 模板, 改 grayhole_satellites=11, CONSTANT 0.5, 同种子
set -e
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs
SIM="$HYPATIA"/ns3-sat-sim/simulator
NAME=gh_pl_const11

if [ ! -d "$RUNROOT/$NAME" ]; then
  mkdir -p "$RUNROOT/$NAME/logs_ns3"
  cp "$RUNROOT/gh_pl_const12/config_ns3.properties" "$RUNROOT/$NAME/config_ns3.properties"
  cp "$RUNROOT/gh_pl_const12/schedule.csv" "$RUNROOT/$NAME/schedule.csv"
  sed -i 's/^grayhole_satellites=.*/grayhole_satellites=11/' "$RUNROOT/$NAME/config_ns3.properties"
fi
echo "=== config grayhole keys ==="
grep grayhole "$RUNROOT/$NAME/config_ns3.properties"

cd "$SIM"
./waf --run="main_satnet --run_dir='$RUNROOT/$NAME'" 2>&1 | tail -5
ls -la "$RUNROOT/$NAME/logs_ns3/isl_packet_loss.csv"
echo "NODE11 RUN COMPLETE"
