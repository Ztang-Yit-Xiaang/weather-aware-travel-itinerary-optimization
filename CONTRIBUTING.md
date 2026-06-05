# Contributing

Thanks for helping improve the weather-aware itinerary planner. This project is a research prototype, so the most valuable contributions make the tool easier to run, easier to inspect, or more honest about its data and optimization limits.

## Development Setup

### Prerequisites

- Python 3.12+
- A virtual environment or Conda environment
- Optional: a valid Gurobi license for Gurobi-backed routes

### Install

```bash
pip install -e .
```

For notebook execution and dashboard validation, install the project dependencies from `pyproject.toml` and `requirements.txt` as needed.

## Common Commands

Run tests:

```bash
python -m pytest
```

Validate dashboard exports:

```bash
python scripts/validate_dashboard_export.py
```

Validate the nature-aware route pipeline:

```bash
python scripts/validate_nature_route_pipeline.py --strict
```

Serve the generated dashboard:

```bash
python scripts/serve_dashboard.py
```

Run the production notebook with the nature config:

```bash
TRIP_CONFIG_PATH="configs/nature_trip_config.yaml" \
python -m jupyter nbconvert \
  --to notebook \
  --execute notebook/production_system_blueprint.ipynb \
  --output production_system_blueprint_nature_executed.ipynb \
  --output-dir notebook \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3
```

## Code Style

- Use Ruff for formatting and linting.
- Keep reusable code in `src/itinerary_system/`.
- Keep tests in `tests/`.
- Prefer clear data contracts over hidden assumptions.
- Add comments only when they explain non-obvious modeling or validation choices.

Useful local checks:

```bash
ruff format .
ruff check .
python -m pytest
```

## Generated Artifact Policy

The project generates many large files: maps, route GeoJSON, CSV outputs, caches, executed notebooks, and validation reports. Most of these should stay out of commits unless the contribution is specifically about publishing a demo artifact or updating a report/proposal.

Before committing generated files, ask:

- Is this artifact small enough to review?
- Is it needed by the README, report, or dashboard demo?
- Can it be regenerated from code and config instead?
- Does it contain stale or misleading route data?

Screenshots for docs are okay when they make the project easier to understand.

## Agent-Assisted Workflow

AI coding agents are welcome here, but large changes should start with a short plan. A good plan should state:

- user-facing goal;
- files or subsystems affected;
- data artifacts that may change;
- tests and validation commands;
- known limitations or follow-up risks.

This keeps optimization, dashboard, and report changes from drifting apart.

## Pull Request Checklist

Before opening a PR or sharing a patch:

- Summarize what changed and why.
- List commands run, especially tests and dashboard validators.
- Include screenshots for dashboard, map, or README visual changes.
- Mention any generated artifacts intentionally included.
- Note whether Gurobi-dependent behavior was tested locally.
- Keep unrelated cleanup out of the change.

## Good First Contribution Areas

- Better data-source adapters and provenance fields.
- Clearer dashboard labels for saved optimized routes versus preview-only routes.
- Smaller, more reliable validation fixtures.
- Documentation that helps a new user open the dashboard quickly.
- Route explanation fields such as why selected, why skipped, and source confidence.

## Questions

Open an issue or discussion on the repository:

https://github.com/Ztang-Yit-Xiaang/weather-aware-travel-itinerary-optimization
