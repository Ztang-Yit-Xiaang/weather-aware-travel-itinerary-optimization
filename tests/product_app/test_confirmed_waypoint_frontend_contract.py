from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP = STATIC / "js" / "app.js"
MAP = STATIC / "js" / "map.js"
CONTROLLER = STATIC / "js" / "map-controller-v2.js"
CSS = STATIC / "css" / "app.css"
NODE_REDUCER_TEST = Path(__file__).with_name("confirmed_waypoint_reducer.test.mjs")


def test_confirm_is_capability_gated_and_uses_exact_server_preview_identity() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'schema_version !== "map-edit-capabilities-v1"' in source
    assert 'capability?.enabled === true' in source
    assert 'capability.feedback_tier === "route_checked"' in source
    assert "capability.preview_executable === false" in source
    assert "capability.evaluated_repair === false" in source
    assert 'preview.validation_state === "route_checked"' in source
    assert 'preview.draft_append_allowed === true' in source
    assert 'preview.selected_access_point?.road_validated === true' in source
    assert '/map/snap-previews/${encodeURIComponent(preview.snap_preview_id)}/confirm' in source
    assert "warning_acknowledged: warningAcknowledged" in source


def test_confirm_body_never_reposts_browser_snapped_or_route_evidence() -> None:
    source = APP.read_text(encoding="utf-8")
    confirm = source[source.index("async function confirmMapWaypoint"):]
    confirm = confirm[: confirm.index("function refreshMapEditDialog")]
    request = confirm[confirm.index("const body = {"): confirm.index("try {")]

    assert "expected_revision: state.session.revision" in request
    assert "warning_acknowledged: warningAcknowledged" in request
    assert "name," in request
    assert "role:" in request
    assert "duration," in request
    assert "snapped_coordinate" not in request
    assert "selected_access_point" not in request
    assert "affected_route_legs" not in request
    assert "raw_coordinate" not in request


def test_duration_modes_encode_all_four_fields_truthfully() -> None:
    source = APP.read_text(encoding="utf-8")
    duration = source[source.index("function durationFromConfirmationForm"):]
    duration = duration[: duration.index("async function confirmMapWaypoint")]

    for mode in ("exact", "preferred", "minimum", "maximum", "range"):
        assert f'mode === "{mode}"' in duration
    for field in (
        "preferred_minutes",
        "minimum_minutes",
        "maximum_minutes",
    ):
        assert field in duration
    assert "minimum <= maximum" in duration
    assert 'value >= 15 && value <= 480' in duration
    assert 'const minimum = ["minimum", "range"].includes(mode);' in source
    assert 'const maximum = ["maximum", "range"].includes(mode);' in source


def test_ordered_reducer_restores_add_relocate_route_evidence_and_undo_view() -> None:
    source = APP.read_text(encoding="utf-8")
    reducer = MAP.read_text(encoding="utf-8")

    assert "export function reduceConfirmedMapDraftOperations" in reducer
    assert 'operation.source !== "confirmed_map_interaction"' in reducer
    assert "exactKeys(parameters, CONFIRMED_PARAMETER_KEYS)" in reducer
    assert 'operation.type === "relocate_custom_waypoint"' in reducer
    assert "!sameInsertion(parameters.insertion, previous.insertion)" in reducer
    assert "legs.length !== 2" in reducer
    assert "incoming.destination_id !== parameters.waypoint_id" in reducer
    assert "outgoing.origin_id !== parameters.waypoint_id" in reducer
    assert "pin.insertion?.route_leg_id || null" in source
    assert "selected_day: pin.targetWaypointId ? pin.day : state.session.selected_day" in source
    assert "if (state.mapInteraction.selectedWaypointId && !selectedDraftWaypoint())" in source


def test_node_adversarial_reducer_regression() -> None:
    completed = subprocess.run(
        ["node", str(NODE_REDUCER_TEST)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "adversarial cases passed" in completed.stdout


def test_route_checked_draft_is_not_presented_as_evaluated_or_previewable() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "Route-checked draft; evaluated repair is not enabled for arbitrary waypoints yet." in source
    assert "state.session.draft.length === 0 || hasRouteCheckedOnlyDraft()" in source
    assert "if (hasRouteCheckedOnlyDraft())" in source
    assert "evaluated repair is not enabled yet" in source


def test_controller_renders_only_validated_draft_routes_and_edit_only_custom_drag() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    adapter = MAP.read_text(encoding="utf-8")

    assert 'leg.road_validated !== true' in controller
    assert 'leg.validation_status !== "road_validated"' in controller
    assert 'throw new Error("draft_route_leg_invalid")' in controller
    assert 'id: "v2-draft-route"' in controller
    assert 'draggable: interactionMode === "edit" && waypoint.waypoint_id === selectedWaypointId' in controller
    assert 'if (interactionMode === "edit" && waypoint.waypoint_id === selectedWaypointId)' in controller
    assert 'marker.on("dragend"' in controller
    assert 'marker.on("dragstart"' in controller
    assert "if (wasDragged)" in controller
    assert "onCustomWaypointDrag" in controller
    assert "draftWaypoints" in adapter
    assert "draftRouteLegs" in adapter


def test_catalog_stops_remain_non_draggable_and_custom_waypoints_have_dom_equivalent() -> None:
    source = APP.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")

    stop_layers = controller[controller.index('map.addSource("v2-stops"'):]
    stop_layers = stop_layers[: stop_layers.index('map.addSource("v2-route-anchors"')]
    assert "Marker" not in stop_layers
    assert "draggable" not in stop_layers
    assert 'class="accessible-draft-waypoint-list"' in source
    assert "data-draft-waypoint=" in source
    assert "data-relocate-waypoint=" in source
    assert 'aria-label="Relocate ${escapeHtml(waypoint.name)}"' in source


def test_waypoint_controls_and_markers_meet_touch_target_contract() -> None:
    css = CSS.read_text(encoding="utf-8")

    assert '.draft-waypoint-marker { width: 44px; min-width: 44px; height: 44px; min-height: 44px;' in css
    assert ".accessible-draft-waypoint button { min-height: 44px;" in css
    assert '.map-context-card input[type="text"]' in css
    assert '.map-context-card input[type="number"]' in css


def test_escape_cancels_only_the_active_non_modal_map_preview() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'event.key === "Escape"' in source
    assert 'state.mapInteraction.selectionKind === "exploratory"' in source
    assert '!document.querySelector("dialog[open]")' in source
    assert "cancelExploratoryPin();" in source
