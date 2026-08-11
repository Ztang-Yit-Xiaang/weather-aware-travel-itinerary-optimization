from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebook" / "production_system_blueprint.ipynb"


def test_production_notebook_is_a_thin_pipeline_client():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8-sig"))
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )

    assert "scripts" in code and "run_research_pipeline.py" in code
    assert "--refresh-policy" in code and '"never"' in code
    assert len(code.splitlines()) <= 60
    for forbidden in (
        "requests",
        "gurobipy",
        "geodesic",
        "build_enriched_catalog",
        "solve_hierarchical_trip_with_gurobi",
        "fetch_osm_accommodations",
    ):
        assert forbidden not in code
