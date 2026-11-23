#!/usr/bin/env python3
"""Shared multi-label chest X-ray training/evaluation script.

Example:
  python train.py --train-csv data/train_small.csv --val-csv data/val_small.csv \
      --test-csv data/test_small.csv --image-root /path/to/images \
      --model resnet18 --fraction full --epochs 10 --output-dir runs/resnet18_full
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageNameResolver:
    """Resolve paths by filename, never by machine-specific CSV prefixes."""

    def __init__(self, image_root: Path):
        self.image_root = image_root
        self.by_name = {}
        for path in image_root.rglob("*"):
            if path.is_file():
                self.by_name.setdefault(path.name, path)

    def resolve(self, name: str) -> Path:
        basename = Path(str(name)).name
        path = self.by_name.get(basename)
        if path is None:
            raise FileNotFoundError(f"Could not find {basename!r} below {self.image_root}")
        return path


class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, resolver: ImageNameResolver,
                 train: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.resolver = resolver
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(self.resolver.resolve(row["Image Index"])).convert("RGB")
        target = torch.tensor(row[LABELS].to_numpy(dtype=np.float32))
        return self.transform(image), target


def make_model(name: str, num_labels: int) -> nn.Module:
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision model: {name}")
    model = getattr(models, name)(weights=None)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_labels)
        elif isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
            classifier[-1] = nn.Linear(classifier[-1].in_features, num_labels)
        else:
            raise ValueError(f"Cannot replace classifier for {name}")
    else:
        raise ValueError(f"Unsupported model head for {name}")
    return model


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    positives = frame[LABELS].sum(axis=0).to_numpy(dtype=np.float64)
    negatives = len(frame) - positives
    # BCEWithLogitsLoss(pos_weight) weights positive errors by N-/N+.
    return torch.tensor(negatives / np.maximum(positives, 1.0), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_logits, all_targets = [], []
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * len(images)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    aucs = [roc_auc_score(targets[:, i], probs[:, i])
            for i in range(targets.shape[1]) if len(np.unique(targets[:, i])) == 2]
    return total_loss / len(loader.dataset), float(np.mean(aucs)) if aucs else float("nan"), probs


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, val_auc, args):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_auc": val_auc,
        "args": vars(args),
    }, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--fraction", default="full")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = (pd.read_csv(x) for x in (args.train_csv, args.val_csv, args.test_csv))
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        missing = set(LABELS + ["Image Index"]) - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")
    resolver = ImageNameResolver(args.image_root)
    # Fail before training if any filename is unavailable.
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for image_name in df["Image Index"]:
            resolver.resolve(image_name)
        print(f"resolved {name}: {len(df)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(ChestXrayDataset(train_df, resolver, train=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ChestXrayDataset(val_df, resolver), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ChestXrayDataset(test_df, resolver), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = make_model(args.model, len(LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights(train_df).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
        scheduler.step()
        val_loss, val_auc, _ = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": running / len(train_loader.dataset),
               "val_loss": val_loss, "val_auc": val_auc,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        save_checkpoint(args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                        model, optimizer, scheduler, epoch, val_auc, args)
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, scheduler,
                            epoch, val_auc, args)
        print(json.dumps(row))

    best = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"])
    _, test_auc, test_probs = evaluate(model, test_loader, device, criterion)
    pred = pd.DataFrame(test_probs, columns=LABELS)
    pred.insert(0, "Image Index", test_df["Image Index"].to_numpy())
    pred.to_csv(args.output_dir / f"preds_{args.model}_{args.fraction}.csv", index=False)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (args.output_dir / "run_summary.json").write_text(json.dumps({
        "best_val_auc": best_auc, "test_macro_auc": test_auc,
        "device": str(device), "num_labels": len(LABELS), "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
