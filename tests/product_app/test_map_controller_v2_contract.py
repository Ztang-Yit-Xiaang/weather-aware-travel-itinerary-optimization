from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = (
    ROOT
    / "src"
    / "itinerary_system"
    / "product_app"
    / "static"
    / "js"
    / "map-controller-v2.js"
)


def _source() -> str:
    return CONTROLLER.read_text(encoding="utf-8")


def test_controller_is_instance_owned_disposable_and_pane_scoped() -> None:
    source = _source()

    assert "export function createGeographicMapController" in source
    assert 'paneId = "map"' in source
    assert "let map = null;" in source
    assert "let generation = 0;" in source
    assert "map.remove();" in source
    assert "pane_id: paneId" in source
    assert "let activeMap" not in source
    assert "disposeGeographicMap" not in source


def test_controller_requires_actual_v2_geography_and_exact_active_plan() -> None:
    source = _source()

    assert 'const EXPECTED_GEOGRAPHY_SCHEMA = "product-geography-v2";' in source
    assert 'new Set(["ready", "ready_with_gaps"])' in source
    assert 'throw new Error("active_plan_required")' in source
    assert 'throw new Error("active_plan_not_found")' in source
    assert "geography.plans.find((plan) => plan?.plan_id === activePlanId)" in source
    assert "active.plan_id === original.plan_id ? [original] : [original, active]" in source
    assert "geography.plans.map(validatePlan)" not in source


def test_controller_uses_actual_plan_fields_and_only_validated_lines() -> None:
    source = _source()

    for field in ("route_path", "validated_legs", "gaps", "coverage"):
        assert field in source
    assert "plan.routes" not in source
    assert "plan.route_anchors" not in source
    assert "plan.route_gaps" not in source
    assert 'feature.properties.validation_status !== "road_validated"' in source
    assert "feature.properties.road_validated !== true" in source
    assert "feature.properties.fallback_used !== false" in source
    assert 'selected.plans, "validated_legs", originalPlanId, activePlanId' in source
    assert 'type: "line"' in source


def test_route_anchors_are_derived_from_route_path() -> None:
    source = _source()

    assert "function deriveRouteAnchors(" in source
    assert "plan.route_path.features" in source
    assert "feature.properties.route_anchor === true" in source
    assert "feature.geometry !== null" in source
    assert 'id: "v2-route-anchors"' in source


def test_null_geometry_gaps_have_text_and_markers_but_never_lines() -> None:
    source = _source()

    assert "function resolveGapPoint(plan, gap)" in source
    assert "properties.leg_index - 1" in source
    assert "function resolveGapMarkers(" in source
    assert "Unvalidated route gap:" in source
    assert "marker_available: coordinate !== null" in source
    assert 'id: "v2-route-gap-markers"' in source
    assert 'id: "v2-route-gap-labels"' in source
    assert "unresolvedGapCount" in source
    gap_layers = source[source.index('map.addSource("v2-route-gaps"'):]
    gap_layers = gap_layers[: gap_layers.index('map.addSource("v2-exploratory-pin"')]
    assert 'type: "line"' not in gap_layers
    assert gap_layers.count('type: "circle"') == 1
    assert gap_layers.count('type: "symbol"') == 1


def test_route_stop_and_gap_clicks_preserve_exact_identity() -> None:
    source = _source()

    assert "function exactRouteLegSelection(event, onRouteLegSelect)" in source
    assert "route_leg_id: properties.route_leg_id" in source
    assert "plan_id: properties.plan_id" in source
    assert "origin_id: properties.origin_id" in source
    assert "destination_id: properties.destination_id" in source
    assert "function exactStopSelection(event, onStopSelect)" in source
    assert "stop_id: properties.stop_id" in source
    assert "function exactGapSelection(event, onGapSelect)" in source
    assert 'nextMap.on("click", "v2-route-gap-markers"' in source


def test_empty_map_click_is_edit_only_and_pin_is_explicitly_provisional() -> None:
    source = _source()

    assert 'interactionMode = "select"' in source
    assert 'if (interactionMode !== "edit") return;' in source
    assert "onEmptyMapClick = () => {}" in source
    assert "queryRenderedFeatures(event.point" in source
    assert 'validation_status: "raw_coordinate_only"' in source
    assert 'id: "v2-exploratory-pin"' in source
    exploratory_section = source[source.index("function exploratoryPinFeature"):]
    exploratory_section = exploratory_section[: exploratory_section.index("function exactRouteLegSelection")]
    assert 'type: "line"' not in exploratory_section


def test_controller_reports_truthful_accessible_textual_fallback() -> None:
    source = _source()

    assert "onTextFallback = () => {}" in source
    assert "function textFallbackFor(plans, gaps)" in source
    assert "road-validated route." in source
    assert "textual_fallback: true" in source
    for field in (
        "visiblePlanCount",
        "stopCount",
        "routeAnchorCount",
        "routeLegCount",
        "roadValidatedLegCount",
        "gapCount",
        "unresolvedGapCount",
    ):
        assert field in source
    assert "feasible" not in source.lower()
    assert "eligible" not in source.lower()


def test_dual_pane_camera_sync_has_recursion_protection() -> None:
    source = _source()

    assert "export function synchronizeMapCameras" in source
    assert "let synchronizationInProgress = false;" in source
    assert "if (synchronizationInProgress) return;" in source
    assert "suppressNextCameraEvent" in source
    assert "const unchanged =" in source
    assert "applySynchronizedCamera" in source
    assert "subscribeCamera" in source
    assert "unsubscribeLeft();" in source
    assert "unsubscribeRight();" in source


def test_assets_remain_loopback_only_and_canvas_accessibility_is_delegated() -> None:
    source = _source()

    assert 'const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost"]);' in source
    assert "candidate.origin !== base.origin" in source
    assert 'throw new Error("map_asset_url_not_loopback")' in source
    assert 'canvas.setAttribute("aria-hidden", "true")' in source
    assert 'canvas.setAttribute("tabindex", "-1")' in source
    assert "caller owns the" in source.lower()
    assert "accessible-stop-list" not in source
    assert "https://" not in source


def test_failed_runtime_assets_are_removed_so_a_later_render_can_retry() -> None:
    source = _source()

    assert 'existing?.dataset.loadState === "failed"' in source
    assert source.count("existing.remove();") >= 3
    assert 'link.dataset.loadState = "failed"' in source
    assert "link.remove();" in source
    assert 'script.dataset.loadState = "failed"' in source
    assert "script.remove();" in source


def test_map_ready_waits_for_visible_idle_tile_coverage_and_observes_size() -> None:
    source = _source()

    assert "function waitForVisibleContainer(renderGeneration)" in source
    assert "new ResizeObserver" in source
    assert 'nextMap.once("idle"' in source
    assert "nextMap.areTilesLoaded()" in source
    assert "tileCoverageRatio < 0.95" in source
    assert 'container.dataset.mapReady = "true"' in source
    assert "nextMap.fitBounds(fittedBounds" in source
    assert 'window.addEventListener("resize", resize)' in source
    assert "resizeObserver?.disconnect();" in source
