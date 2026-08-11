from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP = STATIC / "js" / "app.js"
MAP = STATIC / "js" / "map.js"
CSS = STATIC / "css" / "app.css"
INDEX = STATIC / "index.html"


def test_workspace_consumes_geography_v2_and_exposes_route_gaps() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'geography?.schema_version === "product-geography-v2"' in source
    assert '["ready", "ready_with_gaps"].includes(geography.status)' in source
    assert "plan?.validated_legs?.features" in source
    assert "plan?.gaps?.features" in source
    assert "Unvalidated route gap:" in source
    assert "No required route gaps are hidden." in source
    assert 'role="alert"' in source


def test_map_lifecycle_is_delegated_to_the_instance_owned_v2_controller() -> None:
    source = MAP.read_text(encoding="utf-8")

    assert 'from "./map-controller-v2.js?v=20260810-stability5"' in source
    assert "createGeographicMapController" in source
    assert "activeController?.dispose()" in source
    assert "activePlanId: planId" in source
    assert "selectedRouteLegId" in source
    assert "interactionMode" in source
    assert "exploratoryPin" in source
    assert "maplibregl" not in source
    assert "new window.maplibregl.Map" not in source


def test_select_and_edit_modes_have_keyboard_equivalent_controls() -> None:
    source = APP.read_text(encoding="utf-8")
    markup = INDEX.read_text(encoding="utf-8")

    assert 'class="map-mode-switch" role="group" aria-label="Map interaction mode"' in source
    assert 'data-map-mode="select"' in source
    assert 'data-map-mode="edit"' in source
    assert 'class="accessible-stop-list"' in source
    assert 'class="accessible-route-list"' in source
    assert 'data-geo-route-leg=' in source
    assert "function selectMapStop(selection)" in source
    assert "function selectRouteLeg(selection)" in source
    assert 'id="map-edit-button" type="button" aria-pressed="false"' in markup


def test_empty_map_click_is_exploratory_and_never_appends_a_draft() -> None:
    source = APP.read_text(encoding="utf-8")
    create_pin = source[source.index("function createExploratoryPin"):]
    create_pin = create_pin[: create_pin.index("function cancelExploratoryPin")]

    assert 'validation_status: "raw_coordinate_only"' in create_pin
    assert 'intent: "explore_only"' in create_pin
    assert 'selectionKind = "exploratory"' in create_pin
    assert "addDraft(" not in create_pin
    assert "append" not in create_pin.lower()
    assert "Exploratory only" in source
    assert "A raw point is not a road-valid stop." in source
    assert 'id="confirm-map-waypoint-form"' in source
    assert "roadCheckedPreviewCanConfirm(preview)" in source


def test_snap_preview_is_server_owned_non_persistent_and_confirmation_bounded() -> None:
    source = APP.read_text(encoding="utf-8")
    snap = source[source.index("async function requestExploratorySnapPreview"):]
    snap = snap[: snap.index("function refreshMapEditDialog")]

    assert "/map/snap-preview`" in snap
    assert "expected_revision: state.session.revision" in snap
    assert "longitude: pin.longitude" in snap
    assert "latitude: pin.latitude" in snap
    assert "selected_route_segment_id: pin.targetWaypointId" in snap
    assert "pin.insertion?.route_leg_id || null" in snap
    assert ": state.session.selected_segment_id" in snap
    assert 'travel_mode: "driving"' in snap
    assert 'preview?.schema_version !== "map-snap-preview-v1"' in snap
    assert "preview.persisted !== false" in snap
    assert "no draft was appended" in snap.lower()
    assert "addDraft(" not in snap


def test_context_inspector_never_claims_freehand_or_automatic_execution() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "This is a trip stop, not a movable catalog place." in source
    assert "it does not make the polyline freely draggable" in source
    assert "This point cannot become an executable draft." in source
    assert "Open typed stop edits" in source
    assert "Add supported route feedback" in source
    assert '"route_feedback",\n    { preference: "reduce_contextual_risk" },\n    "selected_route"' in source
    assert "no authoritative snapped coordinate was returned" in source.lower()


def test_interaction_cards_keep_touch_size_and_mobile_bounds() -> None:
    css = CSS.read_text(encoding="utf-8")

    assert ".map-mode-switch button" in css
    assert ".map-context-card" in css
    assert ".map-context-card button { min-height: 44px;" in css
    assert ".accessible-route-list button { width: 100%; min-height: 44px;" in css
    assert ".map-context-card { left: 12px; right: 12px;" in css
    assert ".route-gap-summary" in css
