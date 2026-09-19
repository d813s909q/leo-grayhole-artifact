#!/bin/bash
# --- artifact path resolution (anonymized release) ---
HYPATIA="${HYPATIA:-$HOME/hypatia}"
# 终点豁免语义冒烟：CONSTANT@12，确认 local addrs 注册 + 转发包仍被丢（≈291）
set -e
BASE="$HYPATIA"/integration_tests/test_manila_dalian_over_kuiper/temp/runs
SRC=$BASE/kuiper_630_isls_sat_one_17_to_18_with_TcpNewReno_at_10_Mbps
SIM="$HYPATIA"/ns3-sat-sim/simulator
RUN=$BASE/gh_pl_semantics

rm -rf $RUN
mkdir -p $RUN
cp $SRC/config_ns3.properties $RUN/
cp $SRC/schedule.csv $RUN/
cat >> $RUN/config_ns3.properties << 'EOF'
enable_isl_packet_loss_tracking=true
isl_packet_loss_tracking_interval_ns=1000000000
grayhole_satellites=12
grayhole_drop_rate=0.5
grayhole_mode=CONSTANT
grayhole_start_ns=0
EOF

cd $SIM
echo "=== RUN START $(date) ==="
./waf --run="main_satnet --run_dir='$RUN'" > /tmp/semantics.log 2>&1
echo "=== RUN DONE exit=$? ==="
grep -iE "Injected|assert|fatal|terminate|segmentation" /tmp/semantics.log | head -5
echo "--- isl_packet_loss.csv totals ---"
awk -F, '{rx+=$5; dr+=$6} END{printf "total_rx=%d total_dropped=%d\n", rx, dr}' $RUN/logs_ns3/isl_packet_loss.csv
echo "=== throughput ==="
tail -1 $RUN/logs_ns3/tcp_flows.csv | awk -F, '{print "throughput_bytes="substr($8,1,20)}'