from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_research_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_research_pipeline_raw_catalog", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_raw_catalog_mode_loads_json_lines(monkeypatch, tmp_path):
    business_path = tmp_path / "business.json"
    business_path.write_text(json.dumps({"business_id": "poi_1", "name": "Place"}) + "\n", encoding="utf-8")
    hotels_path = tmp_path / "hotels.csv"
    pd.DataFrame([{"hotel_id": "hotel_1", "name": "Hotel"}]).to_csv(hotels_path, index=False)
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(MODULE, "build_production_generation_executor", fake_builder)
    args = argparse.Namespace(
        input_mode="raw-catalog",
        artifact_dir=None,
        business_path=str(business_path),
        hotels_csv=str(hotels_path),
        cities=["Santa Barbara"],
        primary_city="Santa Barbara",
    )

    MODULE.build_executor(args)

    assert captured["all_business_df"].iloc[0]["business_id"] == "poi_1"
    assert captured["hotels_df"].iloc[0]["hotel_id"] == "hotel_1"
    assert captured["city_names"] == ["Santa Barbara"]
