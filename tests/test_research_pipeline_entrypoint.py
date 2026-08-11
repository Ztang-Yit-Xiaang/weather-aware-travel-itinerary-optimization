from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_research_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_research_pipeline_script", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "input_mode": "frozen-artifacts",
        "artifact_dir": "results/outputs",
        "business_path": None,
        "hotels_csv": None,
        "cities": [],
        "primary_city": "Santa Barbara",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_frozen_artifact_mode_builds_executor(tmp_path):
    pd.DataFrame([{"method": "demo"}]).to_csv(tmp_path / "production_method_comparison.csv", index=False)
    pd.DataFrame([{"method": "demo", "stop_name": "A"}]).to_csv(
        tmp_path / "production_method_route_stops.csv", index=False
    )

    executor = MODULE.build_executor(args(artifact_dir=str(tmp_path)))

    assert callable(executor)


def test_raw_catalog_mode_requires_explicit_inputs():
    with pytest.raises(ValueError, match="raw-catalog mode requires"):
        MODULE.build_executor(args(input_mode="raw-catalog"))
