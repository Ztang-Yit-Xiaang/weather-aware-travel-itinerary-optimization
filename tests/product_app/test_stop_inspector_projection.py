from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.geography_v2 import _stop_features
from itinerary_system.product_app.models import ComponentHealthV1
from itinerary_system.product_app.runtime import ProductRuntime

ROOT = Path(__file__).resolve().parents[2]


def test_real_accepted_stop_retains_canonical_inspector_data_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(ProductRuntime, "_probe_map", lambda self:ComponentHealthV1("map", "ready", False, "maplibre_ready"))

    def client():
        return TestClient(create_product_app(repository_root=ROOT, state_root=tmp_path / "state",
            additional_allowed_authorities=("testserver",)))

    exchanges = []

    def post(api, path, body, headers=None):
        response = api.post(path, json=body, headers=headers)
        payload = response.json()
        exchanges.append({"path":path,"input":body,"status":response.status_code,
            "output":{key:value for key,value in payload.items() if key != "mutation_token"}})
        (tmp_path / "exact_inspector_api_io.json").write_text(json.dumps(exchanges, indent=2, sort_keys=True), encoding="utf-8")
        assert response.status_code == 200, payload
        return payload

    with client() as api:
        created = post(api, "/api/sessions", {})
        session = created["session"]
        path = f"/api/sessions/{session['session_id']}"
        headers = {"X-Session-Token":created["mutation_token"]}
        staged = post(api, path + "/draft/operations", {"expected_revision":session["revision"],
            "type":"replace_nearby","target":"bixby_creek_bridge_viewpoint",
            "parameters":{"candidate_id":"pfeiffer_big_sur_state_park"},"source":"typed_stop_editor"}, headers)
        preview = post(api, path + "/preview", {"expected_revision":staged["session"]["revision"]}, headers)
        proposal = preview["proposal"]
        canonical = next(row for row in proposal["runtime_execution"]["child_plan"]["selected_stops"] if row["stop_id"] == "griffith_observatory")
        accepted = post(api, path + "/accept", {"expected_revision":preview["session"]["revision"],
            "confirmation":"accept_evaluated_repair","idempotency_key":"inspector_accept"}, headers)
        assert accepted["session"]["accepted_plan_id"] == proposal["child_plan_id"]
    with client() as api:
        response = api.get(path, headers=headers)
        assert response.status_code == 200
        restored = response.json()
        exchanges.append({"path":path,"method":"GET","input":None,"status":response.status_code,"output":restored})
        (tmp_path / "exact_inspector_api_io.json").write_text(json.dumps(exchanges, indent=2, sort_keys=True), encoding="utf-8")
        geography = restored["workspace"]["geography"]
        plan = next(row for row in geography["plans"] if row["plan_id"] == proposal["child_plan_id"])
        properties = next(row["properties"] for row in plan["stops"]["features"] if row["properties"]["stop_id"] == "griffith_observatory")
        assert properties["sequence_index"] == 8 and properties["stop_order"] == 1
        assert properties["visit_duration_minutes"] == canonical["visit_duration_minutes"] == 45
        assert properties["itinerary_role"] is canonical["itinerary_role"] is None
        metadata = properties["inspector_metadata"]
        assert metadata["schema_version"] == "plan-stop-inspector-v1"
        for field in ("duration_constraint","weather_feasible","weather_risk","source_list","source_confidence"):
            assert metadata[field] == canonical.get(field)
        assert metadata["duration_constraint"] is None
        assert "place_categories" not in properties and "freshness" not in metadata
        assert plan["content_hash"] == proposal["child_plan_content_hash"]
        input_path = tmp_path / "actual_inspector_input.json"
        input_path.write_text(json.dumps({"geography":geography,"plan_id":plan["plan_id"]}), encoding="utf-8")
        completed = subprocess.run(["node", str(Path(__file__).with_name("stop_inspector_projection.test.mjs"))],
            env={**os.environ,"INSPECTOR_INPUT":str(input_path)}, capture_output=True, text=True, check=False)
        (tmp_path / "node_output.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        assert completed.returncode == 0, completed.stdout + completed.stderr
    (tmp_path / "exact_inspector_api_io.json").write_text(json.dumps(exchanges, indent=2, sort_keys=True), encoding="utf-8")


def test_inspector_projection_preserves_unknowns_and_does_not_invent_exact_constraint():
    stop = {"stop_id":"s","name":"Unknown metadata","day":1,"latitude":34,"longitude":-118}
    row = _stop_features("plan_a","hash","original",("s",),{"s":stop},{})[0]["properties"]
    assert row["visit_duration_minutes"] is None
    assert row["itinerary_role"] is None
    assert all(value is None for key,value in row["inspector_metadata"].items() if key != "schema_version")
    stop["visit_duration_minutes"] = 45
    row = _stop_features("plan_a","hash","original",("s",),{"s":stop},{})[0]["properties"]
    assert row["visit_duration_minutes"] == 45 and row["inspector_metadata"]["duration_constraint"] is None
    stop["duration_constraint"] = {"mode":"exact","preferred_minutes":45,"minimum_minutes":45,"maximum_minutes":45}
    original = deepcopy(stop)
    row = _stop_features("plan_a","hash","original",("s",),{"s":stop},{})[0]["properties"]
    assert row["inspector_metadata"]["duration_constraint"] == stop["duration_constraint"]
    row["inspector_metadata"]["duration_constraint"]["preferred_minutes"] = 60
    assert stop == original


def test_destination_information_is_projected_without_changing_or_scheduling_reference_places():
    stop = {"stop_id":"yosemite_national_park","name":"Yosemite National Park","day":3,
            "latitude":37.74,"longitude":-119.59,"description":"Recorded park description",
            "official_url":"https://www.nps.gov/yose/","place_categories":["nature"]}
    original = deepcopy(stop)
    rows = _stop_features("plan_a","hash","original",("yosemite_national_park",),{stop["stop_id"]:stop},{})
    assert len(rows) == 1 and stop == original
    properties = rows[0]["properties"]
    assert properties["description"] == stop["description"]
    assert properties["official_url"] == stop["official_url"]
    assert len(properties["destination_places"]) == 6
    assert all("guide" in place and "day" not in place for place in properties["destination_places"])
    properties["place_categories"].append("culture")
    assert stop == original


@pytest.mark.parametrize("value", [0, 1, -1, 0.0, 1.0, 0.25])
def test_inspector_numeric_metadata_has_explicit_finite_float_contract(value):
    stop = {"stop_id":"s","day":1,"latitude":34,"longitude":-118,
        "weather_risk":value,"source_confidence":value}
    original = deepcopy(stop)
    metadata = _stop_features("p","hash","original",("s",),{"s":stop},{})[0]["properties"]["inspector_metadata"]
    for field in ("weather_risk", "source_confidence"):
        assert type(metadata[field]) is float
        assert metadata[field] == float(value)
    assert stop == original


@pytest.mark.parametrize("value", [None, True, False, "0.2", [], {}, float("nan"),
    float("inf"), float("-inf"), 10 ** 400])
def test_inspector_numeric_metadata_preserves_invalid_or_missing_as_unknown(value):
    stop = {"stop_id":"s","day":1,"latitude":34,"longitude":-118,
        "weather_risk":value,"source_confidence":value}
    metadata = _stop_features("p","hash","original",("s",),{"s":stop},{})[0]["properties"]["inspector_metadata"]
    assert metadata["weather_risk"] is None
    assert metadata["source_confidence"] is None
