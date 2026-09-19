#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# P1 (M2.2) 多种子并行仿真: xargs -P 并发 (每 run 独立目录, 无文件冲突)
# 用法: JOBS=8 bash run_multiseed_parallel.sh   (JOBS 默认 8)
set -u
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUNROOT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs
SPECS=(const_0p1 const_0p2 const_0p3 const_0p4 const_0p5 onoff_1p0_20s scan_1p0_20s)
N_SEEDS=10
JOBS=${JOBS:-8}

run_one() {
  local name="$1"
  local d="$RUNROOT/$name"
  if [ -f "$d/logs_ns3/isl_packet_loss.csv" ]; then
    echo "SKIP  $name"
    return 0
  fi
  mkdir -p "$d/logs_ns3"
  ( cd "$SIM" && ./waf --run="main_satnet --run_dir='$d'" ) \
      > "$d/logs_ns3/console.txt" 2>&1
  if [ ! -f "$d/logs_ns3/isl_packet_loss.csv" ]; then
    echo "FAIL  $name (no isl_packet_loss.csv)"
    return 1
  fi
  echo "DONE  $name"
}
export -f run_one
export SIM RUNROOT

names=""
for spec in "${SPECS[@]}"; do
  for s in $(seq 0 $((N_SEEDS-1))); do
    names="$names gh_pl_${spec}_s${s}"
  done
done

echo ">>> parallel multi-seed: JOBS=$JOBS, started $(date '+%H:%M:%S')"
echo "$names" | tr ' ' '\n' | grep -v '^$' \
  | xargs -P "$JOBS" -I{} bash -c 'run_one {}'
rc=$?
n_done=$(ls "$RUNROOT"/gh_pl_*_s*/logs_ns3/isl_packet_loss.csv 2>/dev/null | wc -l)
echo ">>> COMPLETE: $n_done/70 runs have isl_packet_loss.csv (xargs rc=$rc), finished $(date '+%H:%M:%S')"
exit $rc