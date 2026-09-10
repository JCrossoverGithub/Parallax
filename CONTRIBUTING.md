# Contributing to Parallax

## Local setup

```bash
uv sync --locked --all-groups
```

Do not install project dependencies directly with `pip`. Add runtime dependencies with `uv add`
and development dependencies with `uv add --dev` so `pyproject.toml` and `uv.lock` remain in sync.

## Required checks

Before committing, run:

```bash
uv lock --check

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src tests
uv run --locked pytest \
  --cov=parallax \
  --cov-report=term-missing \
  --cov-fail-under=100

uv build --no-sources

cd web
npm ci
npm test -- --watch=false
npm run build
cd ..

bash -n scripts/live-soak-acceptance.sh

uv run python -m json.tool \
  docs/evidence/live-soak-2026-09-09.json \
  >/dev/null

git diff --check
```

## Git workflow

1. Start from a clean, synchronized `main` branch.
2. Create one descriptive feature branch.
3. Keep the change focused and add tests with the implementation.
4. Run the required checks.
5. Review `git diff` and `git diff --cached`.
6. Stage explicit paths rather than using `git add .`.
7. Use a concise imperative commit message.
8. Push the branch, open a pull request, and require CI before merge.
9. Return to `main`, fast-forward, delete the merged branch, and prune the remote.

## Data and model artifacts

Do not commit raw PCAP files, HDF5 datasets, generated features, experiment output, model weights,
credentials, tokens, or environment files. Small test fixtures must be intentionally selected,
redistribution-safe, and documented.

## Claims and documentation

Documentation must distinguish measured results from planned capabilities. Changes to data
partitioning, feature semantics, uncertainty calculation, trust boundaries, or major dependencies
require an architecture decision record.
