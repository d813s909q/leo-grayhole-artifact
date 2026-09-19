import pandas as pd

df = pd.read_csv("../analysis/comparison/r2_review_arms.csv")
print("min TPR overall:", df.tpr.min())
sub = df[df.proto == "mmpp"][["lam_cong", "variant", "tpr", "note"]]
print(sub[sub.variant.isin(["ours-est-varmatch", "ours-oracle",
                            "gru-est-refit"])].to_string(index=False))
print()
p = df[df.proto == "poisson"][["lam_cong", "variant", "tpr", "note"]]
print(p[p.variant.isin(["gru-est-refit", "resid-thr-oracle",
                        "resid-thr-est", "resid-ae-clean",
                        "resid-ae-refit"])].to_string(index=False))
