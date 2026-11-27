"""Adversarial evidence fixtures, not browser substitutes, for graphics recovery."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_vertical_slice as core  # noqa: E402
import verify_e2e1_webgl_recovery as verifier  # noqa: E402


def _time(seconds):
    return f"2026-09-08T12:00:{seconds:06.3f}+00:00"


def _fixture():
    props = {"plan_id": "child", "content_hash": "hash", "stop_id": "pfeiffer_big_sur_state_park"}
    stop = {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [-121, 36]}}
    leg = {
        "type": "Feature",
        "properties": {"plan_id": "child", "content_hash": "hash", "route_leg_id": "leg"},
        "geometry": {"type": "LineString", "coordinates": [[-122, 37], [-121, 36]]},
    }
    geo = {
        "plan_id": "child",
        "content_hash": "hash",
        "stops": {"type": "FeatureCollection", "features": [stop]},
        "validated_legs": {"type": "FeatureCollection", "features": [leg]},
    }
    digest = core._completed_race_geometry_hash(geo)
    rendered = {
        "renderedMapPlanId": "child",
        "renderedGeometryHash": digest,
        "renderedMarkerIds": json.dumps(["pfeiffer_big_sur_state_park"]),
    }
    stage = {
        **rendered,
        "activePlanId": "child",
        "activePlanContentHash": "hash",
        "previewState": "PREVIEW_RENDERED",
        "previewRequestId": "request",
    }
    dataset = {
        **rendered,
        "renderedPlanContentHash": "hash",
        "mapReady": "true",
        "mapReadyAt": _time(0),
        "mapTileCoverage": "1",
        "mapCoverageMethod": "loaded_tile_projected_union_area_v1",
    }
    sources = {
        name: {"data": {"features": [{**feature, "properties": {**feature["properties"], "display_role": "active"}}]}}
        for name, feature in (("v2-routes", leg), ("v2-stops", stop))
    }
    row = {
        "observed_at": _time(1),
        "context_lost": False,
        "map_dataset": dataset,
        "stage_dataset": stage,
        "status": {"hidden": True},
        "all_tiles_loaded": True,
        "map_loaded": True,
        "sources": sources,
        "layers": [{"id": "v2-route-active"}, {"id": "v2-stops-active"}],
        "rendered_features": [
            {"layer": layer, "properties": feature["properties"], "geometry": feature["geometry"]}
            for layer, feature in (("v2-route-active", leg), ("v2-stops-active", stop))
        ],
        "events": [],
    }
    loss = {
        "type": "webglcontextlost",
        "at": _time(2),
        "trusted": True,
        "context_lost": True,
        "map_dataset": {"mapReady": "false"},
        "stage_dataset": {"activePlanId": "child", "previewState": "MAP_RENDER_FAILED"},
    }
    restore = {
        "type": "webglcontextrestored",
        "at": _time(4),
        "trusted": True,
        "context_lost": False,
        "map_dataset": {"mapReady": "false"},
        "stage_dataset": {"activePlanId": "child", "previewState": "PREVIEW_DATA_READY"},
    }
    observations = [deepcopy(row) for _ in range(7)]
    for index, seconds in enumerate((1, 1.5, 2.1, 3.4, 4.1, 5, 5.5)):
        observations[index]["observed_at"] = _time(seconds)
    for index in (2, 3):
        observations[index].update(
            context_lost=True,
            map_dataset=deepcopy(loss["map_dataset"]),
            stage_dataset=deepcopy(loss["stage_dataset"]),
            sources={},
            layers=[],
            rendered_features=[],
            all_tiles_loaded=False,
            map_loaded=False,
            status={"hidden": False},
            events=[deepcopy(loss)],
        )
    for index in (4, 5, 6):
        observations[index]["events"] = [deepcopy(loss), deepcopy(restore)]
        observations[index]["map_dataset"]["mapReadyAt"] = _time(4.05)
    faults = [
        {
            "action": "real_WEBGL_lose_context." + action,
            "javascript": "window.__e2eWebglRecovery.extension." + action + "()",
            "at": _time(seconds),
            "api_response_mocking": False,
            "synthetic_event_dispatch": False,
        }
        for action, seconds in (("loseContext", 1.9), ("restoreContext", 3.9))
    ]
    return observations, faults, geo


def test_real_context_sequence_requires_withdrawal_then_new_canonical_ready():
    observations, faults, geo = _fixture()
    assert verifier.graphics_sequence(observations, faults, geo, "request") == observations[5]


@pytest.mark.parametrize(
    "fault",
    [
        "stale_map_id",
        "stale_ready_time",
        "stale_stage_ack",
        "hidden_failure",
        "fake_event",
        "early_restore_ack",
        "short_loss",
        "ready_before_restore",
        "missing_event",
        "synthetic_dispatch",
        "restored_other_source",
        "missing_drawn_stops",
        "wrong_request",
        "nan_coverage",
        "changed_during_capture",
        "foreign_active_plan",
    ],
)
def test_graphics_counterexamples_never_pass(fault):
    rows, faults, geo = _fixture()
    if fault == "stale_map_id":
        rows[2]["map_dataset"]["renderedMapPlanId"] = "child"
    elif fault == "stale_ready_time":
        rows[2]["map_dataset"]["mapReadyAt"] = _time(0)
    elif fault == "stale_stage_ack":
        rows[2]["stage_dataset"]["renderedGeometryHash"] = "old"
    elif fault == "hidden_failure":
        rows[3]["status"]["hidden"] = True
    elif fault == "fake_event":
        rows[6]["events"][0]["trusted"] = False
    elif fault == "early_restore_ack":
        rows[6]["events"][1]["map_dataset"]["mapReady"] = "true"
    elif fault == "short_loss":
        rows[3]["observed_at"] = _time(2.2)
    elif fault == "ready_before_restore":
        rows[5]["map_dataset"]["mapReadyAt"] = rows[6]["map_dataset"]["mapReadyAt"] = _time(0)
    elif fault == "missing_event":
        rows[6]["events"].pop()
    elif fault == "synthetic_dispatch":
        faults[0]["synthetic_event_dispatch"] = True
    elif fault == "restored_other_source":
        rows[5]["sources"]["v2-stops"]["data"]["features"][0]["geometry"]["coordinates"] = [0, 0]
    elif fault == "missing_drawn_stops":
        rows[5]["rendered_features"].pop()
    elif fault == "wrong_request":
        rows[5]["stage_dataset"]["previewRequestId"] = "other"
    elif fault == "nan_coverage":
        rows[5]["map_dataset"]["mapTileCoverage"] = "NaN"
    elif fault == "changed_during_capture":
        rows[6]["layers"].append({"id": "unexpected"})
    else:
        foreign = deepcopy(rows[5]["sources"]["v2-stops"]["data"]["features"][0])
        foreign["properties"]["plan_id"] = "parent"
        rows[5]["sources"]["v2-stops"]["data"]["features"].append(foreign)
    with pytest.raises(core.E2E1VerificationError):
        verifier.graphics_sequence(rows, faults, geo, "request")


def test_missing_graphics_evidence_rejects_stably(tmp_path):
    with pytest.raises(core.E2E1VerificationError, match="e2e1_webgl_evidence_malformed"):
        verifier.verify(tmp_path, "http://127.0.0.1:8127")
