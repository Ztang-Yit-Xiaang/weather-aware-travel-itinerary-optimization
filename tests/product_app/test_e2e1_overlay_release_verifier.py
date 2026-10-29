"""Counterexamples for native canvas release over a visible overlay."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_e2e1_overlay_release as verifier  # noqa: E402
import verify_e2e1_vertical_slice as core  # noqa: E402


def fixture():
    routes = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[-121, 36], [-122, 37]]},
                "properties": {"plan_id": "parent", "content_hash": "hash", "role": "original", "route_leg_id": "leg"},
            }
        ],
    }
    digest = core._completed_race_geometry_hash({"validated_legs": routes})
    parent = {
        "active_plan_id": "parent",
        "rendered_map_plan_id": "parent",
        "active_plan_content_hash": "hash",
        "rendered_geometry_hash": digest,
        "rendered_marker_ids": ["stop"],
    }
    rendered = deepcopy(routes)
    rendered["features"][0]["properties"]["display_role"] = "original"
    sources = {
        "v2-routes": {"data": rendered},
        "v2-stops": {
            "data": {
                "features": [
                    {
                        "properties": {
                            "stop_id": "stop",
                            "plan_id": "parent",
                            "content_hash": "hash",
                            "role": "original",
                            "display_role": "original",
                        }
                    }
                ]
            }
        },
    }
    ack = {
        "mapReady": "true",
        "mapTileCoverage": "1",
        "mapCoverageMethod": "loaded_tile_projected_union_area_v1",
        "renderedMapPlanId": "parent",
        "renderedPlanContentHash": "hash",
        "renderedGeometryHash": digest,
        "renderedMarkerIds": json.dumps(["stop"]),
        "mapReadyAt": "2026-09-08T14:00:00Z",
    }
    start, end = {"x": 10, "y": 20}, {"x": 50, "y": 20}
    inputs = [
        {
            "phase": phase,
            "type": kind,
            "trusted": True,
            "target_is_canvas": True,
            "hit_is_context": kind != "mousedown",
            "buttons": int(kind == "mousedown"),
            **(start if kind == "mousedown" else end),
        }
        for phase in ("capture", "bubble")
        for kind in ("mousedown", "pointerup", "mouseup")
    ]
    observation = {
        "same_map_instance": True,
        "instance": 0,
        "moving": False,
        "sources_before": sources,
        "sources_after": deepcopy(sources),
        "ack_before": ack,
        "ack_after": {**ack, "mapReadyAt": "2026-09-08T14:00:02Z"},
        "inputs": inputs,
        "map_events": [
            {
                "type": kind,
                "moving": index < 2,
                "map_ready": "false" if index < 2 else "true",
                "at": "2026-09-08T14:00:01Z",
            }
            for index, kind in enumerate(("movestart", "dragstart", "dragend", "moveend", "idle"))
        ],
    }
    gesture = {
        "action": "real_canvas_pan_release_over_context",
        "endpoint_is_context": True,
        "endpoint_html": "<dt>Distance</dt>",
        "start": start,
        "end": end,
        "observation": observation,
    }
    ready = {
        "screenshot_started_at": "2026-09-08T14:00:03Z",
        "rendered_plan_id": "parent",
        "rendered_geometry_hash": digest,
        "stable_readiness_before_and_after": True,
    }
    return gesture, parent, ready


def test_actual_release_requires_native_input_end_and_canonical_sources():
    assert verifier.verify(*fixture())["marker_overlay_release_proven"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "not_overlay",
        "too_short",
        "untrusted",
        "wrong_target",
        "wrong_endpoint",
        "missing_bubble",
        "missing_moveend",
        "never_not_ready",
        "stuck_moving",
        "wrong_source",
        "old_ready_time",
        "early_shot",
        "wrong_plan",
        "nan_coverage",
    ],
)
def test_plausible_release_cannot_mask_missing_real_gesture_or_stale_readiness(fault):
    gesture, parent, ready = fixture()
    observed = gesture["observation"]
    if fault == "not_overlay":
        gesture["endpoint_is_context"] = False
    elif fault == "too_short":
        gesture["end"] = gesture["start"]
    elif fault in {"untrusted", "wrong_target", "wrong_endpoint"}:
        observed["inputs"][2][
            {"untrusted": "trusted", "wrong_target": "target_is_canvas", "wrong_endpoint": "hit_is_context"}[fault]
        ] = False
    elif fault == "missing_bubble":
        observed["inputs"] = observed["inputs"][:3]
    elif fault == "missing_moveend":
        observed["map_events"].pop(3)
    elif fault == "never_not_ready":
        for event in observed["map_events"]:
            event["map_ready"] = "true"
    elif fault == "stuck_moving":
        observed["moving"] = True
    elif fault == "wrong_source":
        observed["sources_after"]["v2-stops"]["data"]["features"] = []
    elif fault == "old_ready_time":
        observed["ack_after"]["mapReadyAt"] = observed["ack_before"]["mapReadyAt"]
    elif fault == "early_shot":
        ready["screenshot_started_at"] = observed["ack_before"]["mapReadyAt"]
    elif fault == "wrong_plan":
        observed["ack_after"]["renderedMapPlanId"] = "other"
    else:
        observed["ack_after"]["mapTileCoverage"] = "nan"
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify(gesture, parent, ready)


@pytest.mark.parametrize(
    "fault", ["geometry", "plan_id", "content_hash", "role", "display_role", "missing_display_role", "empty"]
)
def test_renderer_decoration_does_not_hide_changed_canonical_data(fault):
    gesture, parent, ready = fixture()
    for sources in (gesture["observation"]["sources_before"], gesture["observation"]["sources_after"]):
        routes = sources["v2-routes"]["data"]
        if fault == "empty":
            routes["features"] = []
        elif fault == "geometry":
            routes["features"][0]["geometry"]["coordinates"][0][0] += 1
        elif fault == "missing_display_role":
            routes["features"][0]["properties"].pop("display_role")
        else:
            routes["features"][0]["properties"][fault] = "other"
    with pytest.raises(core.E2E1VerificationError):
        verifier.verify(gesture, parent, ready)
