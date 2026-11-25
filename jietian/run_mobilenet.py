#!/usr/bin/env python3
"""Run the team's unmodified patched trainer and export both output conventions."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "mobilenetv3_small_100"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_labels(script):
    tree = ast.parse(script.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LABELS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Trainer does not define LABELS")


def read_split(path, labels):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if [c for c in columns if c in labels] != labels:
        raise ValueError(f"Label order differs from train.py: {path}")
    ids = [r["Image Index"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate images: {path}")
    positives = {label: 0 for label in labels}
    for row in rows:
        for label in labels:
            value = float(row[label])
            if value not in (0, 1):
                raise ValueError(f"Invalid label in {path}: {label}")
            positives[label] += int(value)
    degenerate = [l for l, n in positives.items() if n in (0, len(rows))]
    if degenerate:
        raise ValueError(f"Classes without both outcomes in {path}: {degenerate}")
    # NIH filenames encode patient identity; check against metadata as well.
    patients = {Path(r["Image Index"]).name.split("_")[0].lstrip("0") or "0" for r in rows}
    for row in rows:
        patient = Path(row["Image Index"]).name.split("_")[0].lstrip("0") or "0"
        if "Patient ID" in row and str(int(float(row["Patient ID"]))) != patient:
            raise ValueError(f"Patient metadata mismatch: {row['Image Index']}")
    return rows, patients, positives


def preflight(args, labels):
    splits = {}
    names = {"val": "val_small.csv", "test": "test_small.csv"}
    names.update({f: "train_small.csv" if f == "100" else f"train_{f}.csv"
                  for f in args.fractions})
    for key, filename in names.items():
        path = args.data_dir / filename
        rows, patients, positives = read_split(path, labels)
        splits[key] = {"path": path, "rows": rows, "patients": patients,
                       "positives": positives}
    for f in args.fractions:
        for other in ("val", "test"):
            if splits[f]["patients"] & splits[other]["patients"]:
                raise ValueError(f"Patient leakage: train {f} / {other}")
    if splits["val"]["patients"] & splits["test"]["patients"]:
        raise ValueError("Patient leakage: val / test")
    for small, large in (("25", "50"), ("50", "100"), ("25", "100")):
        if small in splits and large in splits:
            larger = {r["Image Index"]: r for r in splits[large]["rows"]}
            for row in splits[small]["rows"]:
                if row["Image Index"] not in larger or row != larger[row["Image Index"]]:
                    raise ValueError(f"Train {small} is not a matching subset of {large}")
    wanted = {Path(r["Image Index"]).name
              for s in splits.values() for r in s["rows"]}
    found = {}
    for path in args.image_root.rglob("*.png"):
        if path.name in wanted:
            if path.name in found:
                raise ValueError(f"Ambiguous duplicate image: {path.name}")
            found[path.name] = path
    missing = wanted - found.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing: {sorted(missing)[:5]}")
    report = {key: {"csv": str(s["path"]), "sha256": sha256(s["path"]),
                    "images": len(s["rows"]), "patients": len(s["patients"]),
                    "positives": s["positives"]} for key, s in splits.items()}
    print(json.dumps(report, indent=2), flush=True)
    return splits, report


def export_run(out, fraction, labels, splits):
    run = f"mobilenetv3s_{fraction}"
    old = f"{MODEL}_{fraction}"
    summary = json.loads((out / "run_summary.json").read_text())
    if not math.isfinite(summary["test_macro_auc"]):
        raise ValueError("Nonfinite test AUC")
    for split, source in (("val", out / f"preds_{old}_val.csv"),
                          ("test", out / f"preds_{old}.csv")):
        with source.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["Image Index"] + labels:
                raise ValueError(f"Unexpected predictions columns: {source}")
            rows = list(reader)
        if [r["Image Index"] for r in rows] != [r["Image Index"] for r in splits[split]["rows"]]:
            raise ValueError(f"Prediction row order mismatch: {source}")
        if any(not 0 <= float(r[l]) <= 1 for r in rows for l in labels):
            raise ValueError(f"Invalid probabilities: {source}")
        shutil.copy2(source, out / f"preds_{run}_{split}.csv")
    shutil.copy2(out / f"{old}_best_weights.pt", out / f"{run}_best.pt")
    history = json.loads((out / "history.json").read_text())
    with (out / f"log_{run}.csv").open("w", newline="") as f:
        fields = ["epoch", "train_loss", "val_loss", "val_macro_auc", "epoch_time_sec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row["val_auc"] if k == "val_macro_auc" else row[k] for k in fields})
    keys = ["gpu_name", "total_train_time_sec", "peak_gpu_mem_mb", "best_epoch", "best_val_auc"]
    (out / f"info_{run}.json").write_text(json.dumps({k: summary[k] for k in keys}, indent=2))
    return {"run": run, **summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-script", type=Path, default=Path(__file__).resolve().parents[1] / "train.py")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--fractions", nargs="+", choices=["100", "50", "25"], default=["100", "50", "25"])
    p.add_argument("--settings", choices=["patched", "original"], default="patched",
                   help="patched: seed=2026/wd=0.0001; original Nov 23 message: seed=42/wd=0.01")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    labels = read_labels(args.train_script)
    splits, report = preflight(args, labels)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "preflight.json").write_text(json.dumps(report, indent=2))
    if args.preflight_only:
        print("Preflight PASSED; no training started.")
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Enable a GPU runtime before training.")
    seed, wd = (2026, 0.0001) if args.settings == "patched" else (42, 0.01)
    results = []
    for fraction in args.fractions:
        out = args.output_root / f"mobilenetv3s_{fraction}"
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite results: {out}. Use a new output root.")
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(args.train_script),
               "--train-csv", str(splits[fraction]["path"]),
               "--val-csv", str(splits["val"]["path"]),
               "--test-csv", str(splits["test"]["path"]),
               "--image-root", str(args.image_root), "--output-dir", str(out),
               "--model", MODEL, "--fraction", fraction, "--pretrained",
               "--epochs", "10", "--batch-size", "32", "--lr", "0.0001",
               "--weight-decay", str(wd), "--seed", str(seed), "--num-workers", "2"]
        (out / "config.json").write_text(json.dumps({
            "command": cmd, "settings": args.settings, "train_script_sha256": sha256(args.train_script),
            "splits": report, "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)}, indent=2))
        print("Starting", fraction, "on", torch.cuda.get_device_name(0), flush=True)
        with (out / "console.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"Training failed; inspect {out / 'console.log'}")
        results.append(export_run(out, fraction, labels, splits))
        (args.output_root / "all_run_summaries.json").write_text(json.dumps(results, indent=2))
        print(f"Completed {fraction}: test_macro_auc={results[-1]['test_macro_auc']}", flush=True)


if __name__ == "__main__":
    main()
