"""Bind a real canvas pan released over a context panel to restored readiness.

This does not claim a draft-marker release over an overlay was exercised.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import verify_e2e1_vertical_slice as core


def require(condition, reason):
    core._require(condition, "e2e1_overlay_release_" + reason)


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timestamp_without_zone")
    return result


def _verify(gesture, parent, readiness):
    require(
        gesture["action"] == "real_canvas_pan_release_over_context"
        and gesture["endpoint_is_context"] is True
        and bool(gesture["endpoint_html"]),
        "not_overlay_release",
    )
    start, end, observed = gesture["start"], gesture["end"], gesture["observation"]
    require(
        all(
            type(point[key]) in (int, float) and math.isfinite(point[key])
            for point in (start, end)
            for key in ("x", "y")
        )
        and math.hypot(end["x"] - start["x"], end["y"] - start["y"]) > 8,
        "invalid_displacement",
    )
    require(
        observed["same_map_instance"] is True
        and observed["moving"] is False
        and type(observed["instance"]) is int
        and observed["instance"] >= 0,
        "map_still_moving_or_replaced",
    )
    sources = observed["sources_before"]
    routes = sources["v2-routes"]["data"]
    features = routes["features"]
    require(
        routes["type"] == "FeatureCollection"
        and isinstance(features, list)
        and features
        and all(
            row["properties"]["display_role"] == row["properties"]["role"] == "original"
            and row["properties"]["plan_id"] == parent["active_plan_id"]
            and row["properties"]["content_hash"] == parent["active_plan_content_hash"]
            for row in features
        ),
        "foreign_or_undecorated_route_source",
    )
    # collectFeatureCollection adds this verified display-only property. The
    # remaining fields must hash to the exact independently canonical route.
    canonical_routes = {
        **routes,
        "features": [
            {**row, "properties": {key: value for key, value in row["properties"].items() if key != "display_role"}}
            for row in features
        ],
    }
    require(
        sources == observed["sources_after"]
        and core._completed_race_geometry_hash({"validated_legs": canonical_routes}) == parent["rendered_geometry_hash"]
        and {row["properties"]["stop_id"] for row in sources["v2-stops"]["data"]["features"]}
        == set(parent["rendered_marker_ids"]),
        "canonical_sources_changed_or_unbound",
    )
    for ack in (observed["ack_before"], observed["ack_after"]):
        coverage = float(ack["mapTileCoverage"])
        require(
            ack["mapReady"] == "true"
            and math.isfinite(coverage)
            and 0.95 <= coverage <= 1
            and ack["mapCoverageMethod"] == "loaded_tile_projected_union_area_v1"
            and ack["renderedMapPlanId"] == parent["active_plan_id"] == parent["rendered_map_plan_id"]
            and ack["renderedPlanContentHash"] == parent["active_plan_content_hash"]
            and ack["renderedGeometryHash"] == parent["rendered_geometry_hash"]
            and json.loads(ack["renderedMarkerIds"]) == parent["rendered_marker_ids"],
            "readiness_or_plan_mismatch",
        )
    require(
        stamp(observed["ack_before"]["mapReadyAt"])
        < stamp(observed["ack_after"]["mapReadyAt"])
        <= stamp(readiness["screenshot_started_at"])
        and readiness["rendered_plan_id"] == parent["active_plan_id"]
        and readiness["rendered_geometry_hash"] == parent["rendered_geometry_hash"]
        and readiness["stable_readiness_before_and_after"] is True,
        "stale_readiness_or_early_screenshot",
    )
    for phase in ("capture", "bubble"):
        selected = [event for event in observed["inputs"] if event["phase"] == phase]
        require(
            [event["type"] for event in selected] == ["mousedown", "pointerup", "mouseup"], "incomplete_native_release"
        )
        for event, point, buttons in ((selected[0], start, 1), (selected[1], end, 0), (selected[2], end, 0)):
            require(
                event["trusted"] is True
                and event["target_is_canvas"] is True
                and event["buttons"] == buttons
                and event["hit_is_context"] is (buttons == 0)
                and all(
                    type(event[key]) in (int, float) and math.isfinite(event[key]) and abs(event[key] - point[key]) <= 1
                    for key in ("x", "y")
                ),
                "untrusted_or_wrong_endpoint",
            )
    events = observed["map_events"]
    names = [event["type"] for event in events]
    require(
        all(name in names for name in ("movestart", "dragstart", "dragend", "moveend", "idle")),
        "missing_map_gesture_end",
    )
    indices = [names.index(name) for name in ("movestart", "dragstart", "dragend", "moveend", "idle")]
    require(
        indices == sorted(indices)
        and len(set(indices)) == 5
        and [stamp(event["at"]) for event in events] == sorted(stamp(event["at"]) for event in events)
        and any(event["moving"] is True and event["map_ready"] == "false" for event in events)
        and events[indices[-1]]["moving"] is False,
        "readiness_not_withdrawn_or_gesture_order_invalid",
    )
    return {"verified": True, "real_canvas_overlay_release": True, "marker_overlay_release_proven": False}


def verify(gesture, parent, readiness):
    try:
        return _verify(gesture, parent, readiness)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        if isinstance(error, core.E2E1VerificationError):
            raise
        raise core.E2E1VerificationError("e2e1_overlay_release_evidence_malformed") from error
