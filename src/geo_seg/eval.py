from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from .models.build import create_model
from .data.random import RandomTileDataset
from .metrics import binary_iou


def evaluate(ckpt_path: str, out_dir: str | None = None) -> float:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"]
    # Fallbacks
    num_classes = 1
    model_name = "unet"
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        model_name = (
            ckpt["cfg"].get("model", {}).get("name", model_name)
            if isinstance(ckpt["cfg"].get("model"), dict)
            else model_name
        )
        num_classes = (
            ckpt["cfg"].get("data", {}).get("num_classes", num_classes)
            if isinstance(ckpt["cfg"].get("data"), dict)
            else num_classes
        )

    model = create_model(model_name, num_classes)
    model.load_state_dict(state)
    model.eval()

    ds = RandomTileDataset(length=32, seed=42)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    with torch.no_grad():
        ious = []
        for batch in loader:
            imgs = batch["image"]
            masks = batch["mask"]
            logits = model(imgs)
            iou = binary_iou(logits, masks)
            ious.append(iou)
    mean_iou = float(sum(ious) / max(1, len(ious)))

    out_path = Path(out_dir) if out_dir else Path(ckpt_path).parent
    (out_path).mkdir(parents=True, exist_ok=True)
    with open(out_path / "eval.json", "w") as f:
        json.dump({"iou": mean_iou}, f)

    print(f"IoU: {mean_iou:.4f}")
    return mean_iou


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args(argv)
    evaluate(args.ckpt, args.out_dir)


if __name__ == "__main__":
    main()
