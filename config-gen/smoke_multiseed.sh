#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 冒烟测试: 跑 const_0p1 的两个种子 s0 / s1, 验证管线 + s0 复现原文结果
set -e
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs

for name in gh_pl_const_0p1_s0 gh_pl_const_0p1_s1; do
  echo "======================================================"
  echo ">>> RUN $name"
  echo "======================================================"
  cd "$SIM"
  ./waf --run="main_satnet --run_dir='$RUNROOT/$name'" 2>&1 | tee "$RUNROOT/$name/logs_ns3/console.txt"
  echo ">>> dropped col sum (col6):"
  awk -F, '{s+=$6} END {print s}' "$RUNROOT/$name/logs_ns3/isl_packet_loss.csv"
done

echo "======================================================"
echo ">>> 校验 s0 是否与原文 gh_pl_const_0p1 逐字节一致:"
if diff -q "$RUNROOT/gh_pl_const_0p1_s0/logs_ns3/isl_packet_loss.csv" \
            "$RUNROOT/gh_pl_const_0p1/logs_ns3/isl_packet_loss.csv"; then
  echo "IDENTICAL (s0 == original, expected)"
else
  echo "DIFFERENT (unexpected for same seed!)"
fi
echo ">>> 校验 s0 vs s1 是否不同 (不同种子应不同):"
if diff -q "$RUNROOT/gh_pl_const_0p1_s0/logs_ns3/isl_packet_loss.csv" \
            "$RUNROOT/gh_pl_const_0p1_s1/logs_ns3/isl_packet_loss.csv" >/dev/null; then
  echo "IDENTICAL (unexpected - seed may not affect drops!)"
else
  echo "DIFFERENT (expected: seed changes drop realization)"
fi
echo "SMOKE COMPLETE"