from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from .config import default_cfg, Config
from dataclasses import asdict
from .data.random import RandomTileDataset
from .models.build import create_model
from .metrics import binary_iou
from .utils.seed import set_seed
from .utils.logging import get_console, log_metrics
from .utils.paths import ensure_dir


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    ds = RandomTileDataset(
        length=128, seed=cfg.runtime.seed, image_size=cfg.data.image_size
    )
    n_val = max(16, len(ds) // 5)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.runtime.seed)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    return train_loader, val_loader


def train(cfg: Config | None = None, out_dir: str | None = None) -> Path:
    cfg = cfg or default_cfg
    if out_dir is not None:
        cfg.runtime.out_dir = out_dir

    set_seed(cfg.runtime.seed)
    device = torch.device(cfg.runtime.device)
    console = get_console()

    train_loader, val_loader = build_dataloaders(cfg)

    model = create_model(cfg.model.name, cfg.data.num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    out_path = ensure_dir(cfg.runtime.out_dir)
    metrics_csv = out_path / "metrics.csv"
    best_ckpt = out_path / "best.pt"
    best_iou = -1.0

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_iou"])
        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            running_loss = 0.0
            total = 0
            for batch in train_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].float().to(device)
                optim.zero_grad()
                logits = model(imgs)
                if logits.shape[1] == 1:
                    logits_b = logits[:, 0]
                else:
                    logits_b = logits[:, 0]
                loss = criterion(logits_b, masks)
                loss.backward()
                optim.step()
                running_loss += float(loss.item()) * imgs.size(0)
                total += imgs.size(0)

            train_loss = running_loss / max(1, total)

            # Validation
            model.eval()
            with torch.no_grad():
                ious = []
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    logits = model(imgs)
                    iou = binary_iou(logits, masks)
                    ious.append(iou)
                val_iou = float(sum(ious) / max(1, len(ious)))

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_iou:.6f}"])
            f.flush()
            log_metrics(console, epoch, train_loss, val_iou)

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, best_ckpt)

    return best_ckpt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=default_cfg.runtime.out_dir)
    parser.add_argument("--epochs", type=int, default=default_cfg.train.epochs)
    args = parser.parse_args(argv)

    cfg = default_cfg
    cfg.train.epochs = args.epochs
    train(cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
