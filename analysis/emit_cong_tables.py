#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 congestion_confusion_v2.csv 生成 LaTeX 表行 + 更新图7数据源 (旧格式)."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CMP = os.path.normpath(os.path.join(HERE, "..", "analysis", "comparison"))
LRN = os.path.normpath(os.path.join(HERE, "..", "analysis", "learning"))

df = pd.read_csv(os.path.join(CMP, "congestion_confusion_v2.csv"))


def cell(v, lo, hi, mode="hi"):
    if mode == "hi":
        return f"{100*v:.2f} [{100*hi:.2f}]"
    return f"{100*v:.2f} [{100*lo:.2f}, {100*hi:.2f}]"


def row(level, variants, mode="hi"):
    out = []
    for name in variants:
        r = df[(df.lam_cong == level) & (df.variant == name)].iloc[0]
        out.append(cell(r.fpr, r.fpr_lo, r.fpr_hi, mode))
    return " & ".join(out)


V_MAIN = ["ours-oracle", "ours-win-est", "ae-cleanfit-n200",
          "trust-cleanfit", "naive-cleanfit", "clif-cleanfit"]
V_REFIT = ["ours-win-est", "naive-refit", "trust-refit", "clif-refit",
           "ae-refit-n200"]

print("=== Table 4 (main) rows ===")
for lam in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    lamf = float(lam)
    print(f"{lam:.2f} & " + row(lamf, V_MAIN) + r" \\")

print("\n=== Table like-for-like rows ===")
for lam in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    lamf = float(lam)
    print(f"{lam:.2f} & " + row(lamf, V_REFIT) + r" \\")

# ---- 图 7 数据源 (旧 learning_congestion.csv 格式, 全部取自 v2 权威重跑) ----
piv = df.pivot_table(index="lam_cong", columns="variant", values=["tpr", "fpr"])
out = pd.DataFrame({"lam_cong": sorted(df.lam_cong.unique())})
out["vae_tpr"] = [piv[("tpr", "ae-cleanfit-n200")][l] for l in out.lam_cong]
out["vae_fpr"] = [piv[("fpr", "ae-cleanfit-n200")][l] for l in out.lam_cong]
for k, v in [("ours", "ours-oracle"), ("naive", "naive-cleanfit"),
             ("trust", "trust-cleanfit"), ("clif", "clif-cleanfit")]:
    out[f"{k}_tpr"] = [piv[("tpr", v)][l] for l in out.lam_cong]
    out[f"{k}_fpr"] = [piv[("fpr", v)][l] for l in out.lam_cong]
out.to_csv(os.path.join(LRN, "learning_congestion.csv"), index=False)
print("\n[updated] learning_congestion.csv (source: v2 authoritative rerun)")
