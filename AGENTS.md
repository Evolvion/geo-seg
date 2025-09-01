# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/geo_seg/` (models, data, utils, training, eval, demo).
- Tests: `tests/` (pytest). Keep new tests close to the code they cover.
- Artifacts: `runs/` (checkpoints, metrics, eval.json). Temporary during dev.
- Config: `src/geo_seg/config.py` (dataclasses: data/train/model/runtime).

## Build, Test, and Development Commands
- Setup: `make setup` — install requirements and editable package.
- Lint: `make lint` — run `ruff` on `src` and `tests`.
- Format: `make format` — apply `black` to `src` and `tests`.
- Test: `make test` — run `pytest -q`.
- Train: `make run` — trains and writes to `runs/debug1/`.
- Eval: `make eval` — evaluates `runs/debug1/best.pt` → `eval.json`.
- Demo: `make demo` — Streamlit app (`geo_seg.app`).
- Clean: `make clean` — remove `runs/` and `__pycache__`.

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indentation, type hints encouraged.
- Use `black` (format) and `ruff` (lint). Ensure both pass before PRs.
- Naming: packages/modules `snake_case`, classes `CamelCase`, functions/vars `snake_case`.
- Keep functions small; prefer pure helpers in `geo_seg/utils/`.

## Testing Guidelines
- Framework: `pytest` (configured in `pytest.ini`, `pythonpath=src`).
- Run: `pytest -q` or targeted (e.g., `pytest tests/test_model_forward.py -q`).
- Add tests for new behavior; mimic patterns in `tests/`.
- Ensure determinism by using `set_seed` when applicable.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (≤72 chars), body explaining rationale when needed.
  Example: `train: log IoU and save best checkpoint`.
- PRs: include description, linked issues, commands used (e.g., `make test`), and before/after metrics or screenshots (demo).
- CI hygiene: run `make lint format test` locally before opening PR.

## Architecture & Tips
- Models via `geo_seg.models.build:create_model("unet"|"deeplabv3", num_classes)`.
- Data defaults to `RandomTileDataset` for CPU sanity checks.
- Training/Eval: `geo_seg.train` writes `metrics.csv` and `best.pt`; `geo_seg.eval` computes mean IoU.
- Configuration: adjust via `Config` or CLI flags (e.g., `python -m geo_seg.train --epochs 1 --out_dir runs/exp1`).
