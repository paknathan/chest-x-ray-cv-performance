import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = pd.read_csv("ablation_macro_auc.csv")
style = {"ResNet-18": ("#2a6fdb", "o"), "MobileNetV3-Small": ("#1b9e77", "s"), "DeiT-Tiny": ("#d95f02", "^")}
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
for m, (c, mk) in style.items():
    d = r[r.model == m].sort_values("fraction")
    ax.errorbar(d.fraction, d.test_macro_auc, yerr=[d.test_macro_auc - d.ci_low, d.ci_high - d.test_macro_auc],
                color=c, marker=mk, ms=7, lw=2, capsize=4, label=m)
    last = d[d.fraction == 100].test_macro_auc.item()
    ax.annotate(f"{last:.3f}", (100, last), xytext=(6, 0), textcoords="offset points", va="center", color=c, fontsize=9)
ax.axhline(0.5, color="grey", ls="--", lw=1); ax.text(26, 0.505, "chance (0.5)", color="grey", fontsize=8)
ax.set_xticks([25, 50, 100]); ax.set_xticklabels(["25%\n(1,828)", "50%\n(3,632)", "100%\n(7,263)"])
ax.set_xlim(20, 108); ax.set_ylim(0.48, 0.77)
ax.set_xlabel("Training data used (images)"); ax.set_ylabel("Test macro AUC (14 findings)")
ax.set_title("Data-scarcity ablation: test AUC vs. training size", fontsize=11)
ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
fig.text(0.01, 0.005, "Error bars: 95% patient-level bootstrap CI (500 resamples). Same test set (5,757 images) for all runs.",
         fontsize=6.5, color="grey")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("ablation_auc_vs_fraction.png")
