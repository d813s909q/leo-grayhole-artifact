#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 2.3 基线建模运行: baseline(无攻击) 标定 + const12(攻击) 残差验证
set -e
SRC=<project>/02-experiment/src
GEN="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/gen_data/reduced_kuiper_630_algorithm_free_one_only_over_isls
RUNS="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs
OUT="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/analysis/baseline_model

python3 $SRC/baseline_model.py \
  --gen-data $GEN \
  --run baseline=$RUNS/gh_pl_baseline \
  --run const12=$RUNS/gh_pl_const12 \
  --run semantics=$RUNS/gh_pl_semantics \
  --target 18 \
  --interval-ns 1000000000 \
  --out $OUT

echo; echo "=== outputs ==="; ls -la $OUT