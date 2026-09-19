#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# P1 (M2.2) 提取 70 个多种子 run 的残差 (baseline_model.py 一次加载星历批量处理)
set -e
SRC=<project>/02-experiment/src
GEN="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/gen_data/reduced_kuiper_630_algorithm_free_one_only_over_isls
RUNS="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs
OUT=outputs/multiseed

mkdir -p "$OUT"

SPECS=(const_0p1 const_0p2 const_0p3 const_0p4 const_0p5 onoff_1p0_20s scan_1p0_20s)
N_SEEDS=10

ARGS=(--gen-data "$GEN" --run "baseline=$RUNS/gh_pl_baseline")
for spec in "${SPECS[@]}"; do
  for s in $(seq 0 $((N_SEEDS-1))); do
    name="gh_pl_${spec}_s${s}"
    ARGS+=(--run "${spec}_s${s}=$RUNS/$name")
  done
done

python3 "$SRC/baseline_model.py" "${ARGS[@]}" --target 18 --interval-ns 1000000000 --out "$OUT"

echo; echo "=== outputs ==="; ls -la "$OUT" | head -10