#!/usr/bin/env python3
"""Create deterministic, nested patient-level training subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def choose_prefix(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Choose a patient-disjoint subset whose image count is near fraction * N.

    The 25% and 50% subsets are nested because both are prefixes of the same
    deterministic patient permutation.
    """
    rng = np.random.default_rng(seed)
    patients = df["Patient ID"].drop_duplicates().to_numpy()
    rng.shuffle(patients)
    target = len(df) * fraction
    selected = []
    count = 0
    for patient in patients:
        selected.append(patient)
        count += int((df["Patient ID"] == patient).sum())
        if count >= target:
            break
    out = df[df["Patient ID"].isin(selected)].copy()
    # Preserve original row order so output is stable and easy to diff.
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    required = {"Image Index", "Patient ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subset25 = choose_prefix(df, 0.25, args.seed)
    subset50 = choose_prefix(df, 0.50, args.seed)
    ids25 = set(subset25["Patient ID"])
    ids50 = set(subset50["Patient ID"])
    if not ids25.issubset(ids50):
        raise AssertionError("25% subset is not nested in 50% subset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset25.to_csv(args.output_dir / "train_25.csv", index=False)
    subset50.to_csv(args.output_dir / "train_50.csv", index=False)

    print(f"full:  {len(df):5d} images, {df['Patient ID'].nunique():4d} patients")
    for name, subset in [("25%", subset25), ("50%", subset50)]:
        print(f"{name:>4}: {len(subset):5d} images, {subset['Patient ID'].nunique():4d} patients")


if __name__ == "__main__":
    main()
