"""Accuracy-vs-efficiency Pareto plot: full-data test macro AUC vs CPU latency and GMACs."""
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
eff = pd.read_csv("efficiency_benchmark.csv")
auc = pd.read_csv("ablation_macro_auc.csv").query("fraction == 100")
names = {"resnet18": "ResNet-18", "mobilenetv3_small_100": "MobileNetV3-Small", "deit_tiny_patch16_224": "DeiT-Tiny"}
eff["model_name"] = eff["model"].map(names)
d = eff.merge(auc, left_on="model_name", right_on="model", suffixes=("_id", ""))
style = {"ResNet-18": ("#2a6fdb", "o"), "MobileNetV3-Small": ("#1b9e77", "s"), "DeiT-Tiny": ("#d95f02", "^")}

def pareto(df, cost):
    """Models not beaten by another that is both cheaper (or equal) and more accurate."""
    keep = []
    for _, r in df.iterrows():
        dominated = ((df[cost] <= r[cost]) & (df.test_macro_auc >= r.test_macro_auc) &
                     ((df[cost] < r[cost]) | (df.test_macro_auc > r.test_macro_auc))).any()
        keep.append(not dominated)
    return df[keep].sort_values(cost)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), dpi=200, sharey=True)
panels = [("cpu_latency_ms", "CPU latency, batch 1 (ms, 4 threads)", False),
          ("gmacs", "Compute per image (GMACs, log scale)", True)]
rows = []
for ax, (cost, label, log) in zip(axes, panels):
    f = pareto(d, cost)
    ax.plot(f[cost], f.test_macro_auc, color="grey", ls="--", lw=1.2, zorder=1, label="Pareto frontier")
    for _, r in d.iterrows():
        c, mk = style[r.model_name]
        xerr = [[r.cpu_latency_std_ms], [r.cpu_latency_std_ms]] if cost == "cpu_latency_ms" else None
        ax.errorbar(r[cost], r.test_macro_auc, yerr=[[r.test_macro_auc - r.ci_low], [r.ci_high - r.test_macro_auc]],
                    xerr=xerr, color=c, marker=mk, ms=9, capsize=4, lw=1.5, zorder=3)
        tag = "" if r.model_name in set(f.model_name) else "  (dominated)"
        unit = "ms" if cost == "cpu_latency_ms" else "GMACs"
        ax.annotate(f"{r.model_name}{tag}\n{r[cost]:.3g} {unit}, AUC {r.test_macro_auc:.3f}", (r[cost], r.test_macro_auc),
                    xytext=(10, -30 if r.model_name != "ResNet-18" else 8), textcoords="offset points", fontsize=7.5, color=c)
    if log: ax.set_xscale("log"); ax.set_xlim(0.03, 4)
    else: ax.set_xlim(15, 80)
    ax.set_xlabel(label); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
axes[0].set_ylabel("Test macro AUC (14 findings, 100% data)")
axes[0].set_ylim(0.62, 0.74)
fig.suptitle("Accuracy vs. efficiency (upper-left is better)", fontsize=11)
fig.text(0.01, 0.005, "AUC error bars: 95% patient-level bootstrap CI. Latency error bars: SD over 5 trials. "
         "All models 224x224 input, float32.", fontsize=6.5, color="grey")
fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig("pareto_accuracy_vs_efficiency.png")

t = d[["model_name", "test_macro_auc", "ci_low", "ci_high", "params_m", "gmacs", "cpu_latency_ms",
       "gpu_latency_ms", "gpu_peak_allocated_mb"]].copy()
t["on_frontier_latency"] = t.model_name.isin(pareto(d, "cpu_latency_ms").model_name)
t["on_frontier_gmacs"] = t.model_name.isin(pareto(d, "gmacs").model_name)
t.sort_values("cpu_latency_ms").round(4).to_csv("pareto_table.csv", index=False)
print(t.sort_values("cpu_latency_ms").round(3).to_string(index=False))
