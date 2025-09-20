# Repository Guidelines

## Project Structure & Modules
- Source: `src/` (core logic in `src/core/`, calculators in `src/calculators/`, utilities in `src/utils/`, CLI in `src/scripts/`).
- Scripts: `scripts/` operational tools (distributed runner, monitoring, queries).
- Config & Examples: `config/` (e.g., `resources.yaml`), `examples/` sample workflows.
- Data & Outputs: `data/`, `results/`, `logs/` (do not write to repo root).
- Tests: `tests/` (pytest discovery enabled).

## Build, Test, and Development
- Environment
  - python -m venv .venv && source .venv/bin/activate
  - pip install -e .[dev]
- Run workflows
  - atlas-qe-workflow eos examples/gaas_eos_study/gaas_eos_study.yaml
  - python scripts/run_distributed_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml
- Tests & coverage
  - pytest -q
  - pytest --cov=src --cov-report=term-missing
- Lint/format/type-check
  - black .
  - flake8 src tests
  - mypy src

## Coding Style & Conventions
- Python 3.11+, Black formatting (line length 88), 4‑space indent.
- Type hints required for public functions (`mypy` strict-ish); prefer explicit return types.
- Naming: modules/functions `snake_case`, classes `CapWords`, constants `UPPER_CASE`.
- File placement: new executables in `scripts/`; reusable logic in `src/` packages; no generated files in project root.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Discovery: files `tests/test_*.py`, classes `Test*`, functions `test_*`.
- Add tests with each new module/bugfix; aim for meaningful coverage (HTML report via `--cov-report=html`).
- Prefer small, deterministic tests; use example configs under `examples/` when needed.

## Commit & PR Guidelines
- Commits: imperative subject (<= 72 chars), concise body with rationale when non-trivial.
- Scope changes narrowly; keep code, tests, and docs together.
- PRs: clear description, what/why, testing notes, and any config updates (e.g., `config/resources.yaml`). Link issues if applicable. Include sample command to reproduce.
- All checks pass locally: format, lint, type-check, tests.

## Configuration & Security Tips
- Update `config/resources.yaml` with local/remote paths for ATLAS/QE binaries and resource limits; do not hardcode secrets.
- Outputs should go under `results/` and `logs/`; avoid committing large artifacts. Add new patterns to `.gitignore` as needed.
