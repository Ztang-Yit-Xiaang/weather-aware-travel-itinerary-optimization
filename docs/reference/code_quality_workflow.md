# Code Quality Workflow

This repository keeps source, configs, docs, and tests under version control. Generated maps, CSV/JSON outputs, caches, and local quality reports should stay out of commits.

## Install Tools

```bash
pip install ruff pre-commit pytest coverage vulture
```

For editable package installs:

```bash
pip install -e .
```

## Formatting And Linting

```bash
ruff format .
ruff check . --fix
```

Ruff is configured in `pyproject.toml` and excludes generated folders such as `results/`.

## Pre-Commit

```bash
pre-commit install
pre-commit run --all-files
```

The pre-commit setup runs Ruff plus lightweight YAML/JSON, whitespace, end-of-file, and large-file checks.

## Tests And Coverage

```bash
coverage run -m pytest
coverage report -m
```

The pytest configuration uses `tests/` and adds `src/` to `PYTHONPATH`.

## Dead-Code Report

```bash
python scripts/find_dead_code.py
```

This writes `results/quality/vulture_report.txt`. Vulture findings are review prompts only; do not delete code automatically from this report.

## Dashboard Artifacts

The full modular dashboard supports two loading modes:

- Localhost or GitHub Pages: JSON and GeoJSON are loaded with `fetch()`.
- Direct `file://`: generated JavaScript fallback assets are injected as scripts and read from `window.DASHBOARD_*` globals.

Serve the full dashboard with:

```bash
python scripts/serve_dashboard.py
```

Then open the printed localhost URL. The lightweight share map is standalone and can be opened directly.

Validate dashboard output with:

```bash
python scripts/validate_dashboard_export.py
```

Generated dashboard folders, maps, CSV/JSON outputs, caches, and quality reports are ignored so regenerated artifacts do not clutter commits.
