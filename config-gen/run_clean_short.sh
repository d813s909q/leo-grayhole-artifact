#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 临时: 运行 clean run 并抓取溢出计数器
cd "$HYPATIA"/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='"$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs/gh_cong_clean'" 2>&1 | grep -E 'ISL-OVERFLOW-TOTAL|ISL data rate|ISL max queue|Read schedule|Finished simulation'