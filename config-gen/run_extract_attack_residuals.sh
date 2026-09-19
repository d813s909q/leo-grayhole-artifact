#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 2.5 为全部攻击 run 提取残差 (baseline_model.py)
set -e
SRC=<project>/02-experiment/src
GEN="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/gen_data/reduced_kuiper_630_algorithm_free_one_only_over_isls
RUNS="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs
OUT=outputs/attack_runs

mkdir -p "$OUT"

python3 $SRC/baseline_model.py \
  --gen-data $GEN \
  --run baseline=$RUNS/gh_pl_baseline \
  --run const_0p1=$RUNS/gh_pl_const_0p1 \
  --run const_0p2=$RUNS/gh_pl_const_0p2 \
  --run const_0p3=$RUNS/gh_pl_const_0p3 \
  --run const_0p4=$RUNS/gh_pl_const_0p4 \
  --run onoff_1p0_20s=$RUNS/gh_pl_onoff_1p0_20s \
  --run scan_1p0_20s=$RUNS/gh_pl_scan_1p0_20s \
  --target 18 \
  --interval-ns 1000000000 \
  --out $OUT

echo; echo "=== outputs ==="; ls -la $OUT