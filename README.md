# geo-seg

![CI](https://github.com/Evolvion2/geo-seg/actions/workflows/ci.yml/badge.svg)

One-liner: Tiny, CPU-friendly baseline for geospatial semantic segmentation.
Topics: geospatial, segmentation, pytorch, unet, deeplabv3

## Quickstart
- Setup: `make setup` (installs requirements and editable package)
- Lint: `make lint` (ruff)
- Format: `make format` (black; applies formatting)
- Test: `make test` (pytest -q)
- Train: `make run` (writes to `runs/debug1/`)
- Eval: `make eval` (reads `runs/debug1/best.pt`, writes `eval.json`)
- Demo: `make demo` (Streamlit UI)

## Data Contract
- Folder layout (Day 2-ready):
  - Images: `data/images/{*.png|*.jpg|*.tif}`
  - Masks: `data/masks/{*.png}` (matching basenames)
- Masks are single-channel, integer class IDs.
- Random dataset already conforms to `image_size` and `num_classes`.

## Config
- See `src/geo_seg/config.py` for defaults.
- Fields: `data.root, data.image_size, data.num_classes; train.batch_size, epochs, lr, num_workers; model.name; runtime.device, seed, out_dir`.

## Metrics
- Primary: binary IoU (0/0 → 1.0 by convention). Masks are single‑channel integer IDs; predictions are thresholded at 0.5.
- Template:

  | run | epoch | train_loss | val_iou |
  |-----|-------|------------|---------|
  | debug1 | 1..N | x.xxx | x.xxx |

## Docker
- Build: `docker build -t ghcr.io/<you>/geo-seg:latest .`
- Smoke: `docker run --rm -v $(pwd)/out:/out ghcr.io/<you>/geo-seg:latest` (writes `/out/best.pt`).

## License
- Apache-2.0. See `LICENSE`.
