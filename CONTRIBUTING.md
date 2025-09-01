# Contributing to geo-seg

## Quickstart
- Python 3.11. `make setup && make test`
- Train smoke: `make run` → writes `runs/debug1/best.pt`
- Eval: `make eval`  Demo: `make demo`

## Branch/Commits
- Branch per change. Conventional Commits (feat|fix|chore|docs|test).

## CI
- PRs must pass: ruff, black --check, pytest.
- No datasets or model weights in git. Use `data/` and `runs/` locally; both are gitignored.

## Code style
- Black line length 100. Ruff enabled.
- Add tests for new modules. Mark slow tests with `@pytest.mark.slow`.

## Security
- No secrets in repo. Report issues via SECURITY.md.

