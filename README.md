# geo-seg

![CI](https://github.com/Evolvion2/geo-seg/actions/workflows/ci.yml/badge.svg)

## Problem
- Minimal, CPU-friendly baseline for geospatial semantic segmentation with simple models.

## Data Contract
- Image/Mask folders (to be finalized Day 2):
- Images: `data/images/*.png`, Masks: `data/masks/*.png` (same basename).
- Masks are single-channel, integer class IDs.

## Config
- Single flat config today; see `geo_seg/config.py`.
- Key fields: `data.root, data.image_size, data.num_classes, train.batch_size, train.epochs, train.lr, train.num_workers, model.name, runtime.device, runtime.seed, runtime.out_dir`.

## Commands
- Setup: `make setup`
- Lint: `make lint`
- Test: `make test`
- Train: `make run`
- Eval: `make eval`
- Demo: `make demo`
- Clean: `make clean`

## Docker
- Build: `docker build -t ghcr.io/<you>/geo-seg:latest .`
- Run (smoke): `docker run --rm -v $(pwd)/out:/out ghcr.io/<you>/geo-seg:latest` (writes `/out/best.pt`).

## Metrics
- Primary: binary IoU (0/0 -> 1.0 convention).

## License
- Apache-2.0. See `LICENSE` for full text.
