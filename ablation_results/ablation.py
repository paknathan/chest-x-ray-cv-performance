"""Data-scarcity ablation: test macro AUC vs training fraction, with patient-level bootstrap CIs."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = "/mnt/user-data/uploads/group_project/chest-x-ray-cv-performance"
L = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia",
     "Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"]
RUNS = {
 "ResNet-18":         {100: "resnet-18/resnet18_full/preds_resnet18_full.csv",
                        50: "resnet-18/resnet18_50/preds_resnet18_50.csv",
                        25: "resnet-18/resnet18_25/preds_resnet18_25.csv"},
 "MobileNetV3-Small": {100: "mobilenet_results/mobilenetv3s_100/preds_mobilenetv3_small_100_100.csv",
                        50: "mobilenet_results/mobilenetv3s_50/preds_mobilenetv3_small_100_50.csv",
                        25: "mobilenet_results/mobilenetv3s_25/preds_mobilenetv3_small_100_25.csv"},
 "DeiT-Tiny":         {100: "results_Ver1_DeiT-Tiny/runs/deit_tiny_full/preds_deit_tiny_patch16_224_full.csv",
                        50: "DeiT-Tiny_runs_50_25/deit_tiny_50/preds_deit_tiny_patch16_224_50.csv",
                        25: "DeiT-Tiny_runs_50_25/deit_tiny_25/preds_deit_tiny_patch16_224_25.csv"},
}
test = pd.read_csv(f"{B}/test_small.csv")
test["key"] = test["Image Index"].map(lambda s: str(s).split("/")[-1])
Y = test[L].to_numpy()
P = {}
for m, fr in RUNS.items():
    for f, path in fr.items():
        d = pd.read_csv(f"{B}/{path}")
        d["key"] = d["Image Index"].map(lambda s: str(s).split("/")[-1])
        d = d.set_index("key").loc[test["key"]]          # align rows to the test CSV
        assert len(d) == len(test) and not d[L].isna().any().any(), (m, f)
        P[(m, f)] = d[L].to_numpy()

def macro(y, p):
    return np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].sum() < len(y)])

# point estimates + per-class AUC
rows, per_class = [], {}
for (m, f), p in P.items():
    rows.append({"model": m, "fraction": f, "test_macro_auc": macro(Y, p)})
    per_class[f"{m} {f}%"] = [roc_auc_score(Y[:, i], p[:, i]) for i in range(len(L))]

# patient-level bootstrap (same resamples for every run, so differences are paired)
rng = np.random.default_rng(2026)
pats = test["Patient ID"].to_numpy(); up = np.unique(pats)
idx_by_pat = {u: np.flatnonzero(pats == u) for u in up}
NB = 500
boot = {k: [] for k in P}
for _ in range(NB):
    idx = np.concatenate([idx_by_pat[u] for u in rng.choice(up, len(up), replace=True)])
    for k, p in P.items():
        boot[k].append(macro(Y[idx], p[idx]))
boot = {k: np.array(v) for k, v in boot.items()}

res = pd.DataFrame(rows)
res["ci_low"]  = [np.percentile(boot[(r.model, r.fraction)], 2.5)  for r in res.itertuples()]
res["ci_high"] = [np.percentile(boot[(r.model, r.fraction)], 97.5) for r in res.itertuples()]
res = res.sort_values(["model", "fraction"], ascending=[True, False])
res.to_csv("ablation_macro_auc.csv", index=False)
print(res.round(4).to_string(index=False))

# drops relative to each model's full-data AUC, with paired CIs
drops = []
for m in RUNS:
    for f in (50, 25):
        d = boot[(m, 100)] - boot[(m, f)]
        pt = res.query("model==@m and fraction==100").test_macro_auc.item() - res.query("model==@m and fraction==@f").test_macro_auc.item()
        drops.append({"model": m, "comparison": f"100% -> {f}%", "auc_drop": pt,
                      "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
drops = pd.DataFrame(drops); drops.to_csv("ablation_drops.csv", index=False)
print("\n", drops.round(4).to_string(index=False))

# is DeiT's drop larger than each CNN's? (difference of drops, paired)
print()
for f in (50, 25):
    dd = boot[("DeiT-Tiny",100)] - boot[("DeiT-Tiny",f)]
    for cnn in ("ResNet-18", "MobileNetV3-Small"):
        dc = boot[(cnn,100)] - boot[(cnn,f)]
        diff = dd - dc
        print(f"DeiT drop minus {cnn} drop (100->{f}%): {np.mean(diff):+.4f}  95% CI [{np.percentile(diff,2.5):+.4f}, {np.percentile(diff,97.5):+.4f}]")

pc = pd.DataFrame(per_class, index=L).T
pc.insert(0, "positives_in_test", "")
pc = pd.concat([pd.DataFrame([Y.sum(0)], index=["positives_in_test"], columns=L).assign(positives_in_test=""), pc[L]])
pc.round(3).to_csv("ablation_per_class_auc.csv")
np.savez("boot.npz", **{f"{m}|{f}": v for (m, f), v in boot.items()})
