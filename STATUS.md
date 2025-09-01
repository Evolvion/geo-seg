Status: Day-1 Review

- commit: b3e2997
- branch: main
- updated: 2025-09-01T13:32:22Z

Checklist (ground-truth)

- [x] README.md: title, purpose, quickstart, metrics (IoU 0/0=1.0, thresh 0.5)
- [x] LICENSE: Apache-2.0
- [x] .gitignore / .dockerignore: runs/, data/, venv, caches, weights, egg-info
- [x] .github/workflows: ci.yml, docker.yml present
- [x] CONTRIBUTING.md, SECURITY.md, CODEOWNERS, PR template
- [x] Package layout: src/geo_seg/** exists with data/models/utils
- [x] Data package: src/geo_seg/data/{__init__,base,random}.py
- [x] No datasets, weights, or build artifacts tracked
- [x] requirements.txt: minimal deps; pillow included

Notes

- CI includes informational Bandit step.
- Ruff and Black line-length aligned at 100.
- Hydra is not used and was removed from requirements.

