# Grayhole Detection in LEO Satellite Networks — Anonymized Artifact

This repository is the anonymized artifact accompanying the manuscript
**"Ephemeris-Grounded Packet-Loss Attribution for Grayhole Detection in
LEO Satellite Networks with a Provable False-Alarm Budget"** (anonymized
for review; available from the author upon acceptance).

It contains the complete grayhole injection extension, the run-configuration
generators, the residual/detection analysis pipeline, and every figure script
used in the paper.

## Layout

```
anon-repo/
├── ns3-patch/      # GrayholeErrorModel + simulator patches (deploy into Hypatia)
├── config-gen/     # run configuration generation & batch execution
├── analysis/       # expectation baseline, CUSUM detector, baselines, robustness
├── figures/        # paper figure scripts
└── outputs/        # all pipeline outputs land here (created by the scripts)
```

## Environment

- Ubuntu 20.04
- [Hypatia](https://github.com/snkassing/hypatia) with ns-3.31, checked out at
  `$HYPATIA` (default `~/hypatia`)
- Python >= 3.8 with `numpy`, `scipy`, `pandas`, `matplotlib`
- LaTeX (TeX Live) only if you recompile the manuscript (not included here)

Two environment variables control all paths:

| Variable   | Default                  | Meaning                              |
|------------|--------------------------|--------------------------------------|
| `HYPATIA`  | `$HOME/hypatia`          | Hypatia checkout root                |
| `GH_OUT`   | `<repo>/outputs`         | analysis output root                 |

## Reproduction

**Step 1 — deploy the ns-3 patch.** Copy `ns3-patch/grayhole-error-model.{cc,h}`
and `ns3-patch/main_satnet.cc` into
`$HYPATIA/ns3-sat-sim/simulator/contrib/satellite-network/{model,helper}` per
the paths inside the patch scripts, register them via `ns3-patch/wscript`, then
apply the simulator patches:

```bash
cd ns3-patch
python3 patch_isl_queue_no_stop.py   # queue-overflow loss no longer silently dropped
python3 patch_congestion_drop.py     # attribute overflow drops to ISL loss log
python3 patch_scan_mode.py           # link-scanning attack mode
python3 add_packet_include.py
python3 fix_reorder.py               # member-initialization-order fix
```

Rebuild ns-3 once afterwards.

**Step 2 — generate and run scenarios.** The paper uses 90 runs in the core
suite plus the review-experiment batches (R-8/R-9):

```bash
cd config-gen
python3 generate_attack_runs.py          # 7 attack schedules, single seed
python3 gen_congestion_runs.py           # gh_cong_clean / gh_cong_att12
bash    gen_node11_run.sh                # tie-break: attacker on satellite 11
python3 multiseed_gen.py                 # 7 schedules x 10 seeds
bash    run_multiseed_parallel.sh        # batch execution (or run_multiseed.sh)
python3 analyze_persistence.py           # subnet selection criterion (Sec. VI-A)
python3 r8r9_gen_runs.py                 # R-8 multi-seed congestion + R-9 low-rate sweep
```

**Step 3 — analysis and figures.** Each script writes into `$GH_OUT`:

```bash
cd analysis
python3 run_comparison.py          # attack sweep + congestion-confusion table
python3 multiseed_report.py        # TPR/FPR with Wilson CIs
python3 congestion_residual.py     # real-congestion residuals (Sec. VI-E)
python3 congestion_cusum.py        # static vs per-epoch-beta CUSUM
python3 handover_sensitivity.py    # lambda_h>0 injections (Sec. VI-F)
python3 tiebreak_check.py          # attacker-on-11 robustness (Sec. VI-F)
python3 baseline_sanity.py         # reimplementation sanity alignment (Sec. VI-A)
python3 delay_sensitivity.py       # small-signal delay curves (Fig. 5)
python3 delta_curve.py             # minimum detectable delta (Fig. 6)
python3 r8_multiseed_congestion.py # 11-seed real-congestion ablation (Sec. VI-E)
python3 r9_damage_curve.py         # damage-vs-rate sweep (Sec. VI-B)
python3 r13_arl_timevarying.py     # time-varying ARL check (Sec. VI-E)
python3 r13b_worst_rate.py         # worst-rate inversion lambda*=0.11, h 1.07->4.90
python3 r2_review_arms.py          # resid-thr / resid-AE / MMPP arms (Sec. VI-I)
cd ../figures
python3 figure_studio.py           # paper figures
```

## Paper-to-code map

| Paper item | Script |
|---|---|
| Figures 2-4, Tables I-III | `figures/figure_studio.py` |
| Fig. 5 (delay vs delta_a, lambda_0) | `analysis/delay_sensitivity.py` |
| Fig. 6 (delta_min curves) | `analysis/delta_curve.py` |
| Fig. 7 (congestion confusion) | `analysis/run_comparison.py` + `analysis/emit_cong_tables.py` |
| Multi-seed TPR/FPR CIs | `analysis/multiseed_report.py` |
| Congestion confusion table | `analysis/run_comparison.py` |
| Estimated-rate / misestimation arms | `analysis/congestion_deoracle.py` |
| Real queue-overflow congestion | `config-gen/gen_congestion_runs.py` + `analysis/congestion_{residual,cusum}.py` |
| 11-seed raw-vs-residual ablation (352 node-runs) | `config-gen/r8r9_gen_runs.py` + `analysis/r8_multiseed_congestion.py` |
| Damage-vs-rate sweep (46-69%, ~24x TCP amplification) | `analysis/r9_damage_curve.py` |
| Time-varying ARL / worst-rate inversion | `analysis/r13_arl_timevarying.py` + `r13b_worst_rate.py` |
| MMPP bursty congestion + resid-thr/resid-AE (Table V) | `analysis/r2_review_arms.py` |
| lambda_h > 0 handover injections | `analysis/handover_sensitivity.py` |
| Tie-break (attacker on 11) | `config-gen/gen_node11_run.sh` + `analysis/tiebreak_check.py` |
| Baseline reimplementation sanity | `analysis/baseline_sanity.py` |
| Subnet selection / persistence | `config-gen/analyze_persistence.py` |
| CUSUM threshold inversion (h = 3.62 at lambda_0 = 0.01, delta = 1.0, ARL0* = 1e3) | `analysis/cusum_detector.py` |

## Notes

- All scripts default `GH_OUT` to `<repo>/outputs`; the directory is created on
  demand and its contents are regenerable from the runs.
- The simulator patches are written against the Hypatia commit the paper used;
  context lines may need adjustment on newer revisions.
- Frozen analysis outputs backing every table/figure ship with the submission
  as a separate data archive; raw per-slot residual traces are regenerated by
  the run pipeline above.
- Author identity is withheld during review.
