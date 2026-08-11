from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP_JS = STATIC / "js" / "app.js"
MAP_JS = STATIC / "js" / "map.js"
MAP_CONTROLLER_JS = STATIC / "js" / "map-controller-v2.js"
COMPARE_JS = STATIC / "js" / "compare.js"
INDEX = STATIC / "index.html"
APP_CSS = STATIC / "css" / "app.css"


def test_app_delegates_to_the_focused_geographic_renderer() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert '} from "./map.js?v=20260810-stability5";' in source
    assert "renderGeographicMap" in source
    assert "workspace.geography" in source
    assert 'id="geographic-map"' in source
    assert 'aria-hidden="true"' in source
    assert 'role="application"' not in source
    assert 'class="accessible-stop-list"' in source
    assert "function textualRouteEvidence(plan, daySummary)" in source
    assert "Route continuity: after Day 3 at" in source
    assert "Booking impact:" in source
    assert "Eligibility:" in source


def test_hidden_visual_map_removes_maplibre_keyboard_targets() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert "function hideVisualCanvas(map)" in source
    assert 'canvas.setAttribute("aria-hidden", "true")' in source
    assert 'canvas.setAttribute("tabindex", "-1")' in source
    assert source.count("hideVisualCanvas(nextMap);") == 2


def test_renderer_accepts_only_validated_artifact_backed_geojson() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert 'const EXPECTED_GEOGRAPHY_SCHEMA = "product-geography-v2"' in source
    assert "geography?.schema_version !== EXPECTED_GEOGRAPHY_SCHEMA" in source
    assert "feature.properties.road_validated !== true" in source
    assert "feature.properties.fallback_used !== false" in source
    assert "feature.properties.plan_id !== plan.plan_id" in source
    assert "window.maplibregl.Map" in source
    assert "window.maplibregl.addProtocol" in source
    assert "window.pmtiles.Protocol" in source


def test_renderer_uses_distinct_route_styles_and_linked_attribution() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert 'id: "v2-route-original"' in source
    assert 'id: "v2-route-active"' in source
    assert 'id: "v2-route-selected"' in source
    assert 'line-dasharray' in source
    assert "attribution.url" in APP_JS.read_text(encoding="utf-8")
    assert 'class="map-attribution"' in APP_JS.read_text(encoding="utf-8")
    assert "selectedRouteLegId" in source


def test_route_trial_keeps_only_original_and_active_artifact_plans_visible() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert "active.plan_id === original.plan_id ? [original] : [original, active]" in source
    assert "geography.plans.flatMap" not in source
    assert 'id: "v2-route-original"' in source
    assert 'id: "v2-route-active"' in source
    assert '"line-cap": "round"' in source


def test_route_trial_uses_only_the_configured_loopback_osm_runtime() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert 'const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost"])' in source
    assert "candidate.origin !== base.origin" in source
    assert "localAssetUrl(config.style_url, base)" in source
    assert "https://" not in source


def test_route_trial_keeps_route_anchors_and_stops_as_distinct_evidence_layers() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert "function deriveRouteAnchors" in source
    assert 'id: "v2-route-anchors"' in source
    assert 'id: "v2-stops-active"' in source
    assert 'id: "v2-stops-original"' in source
    assert "exactStopSelection" in source


def test_renderer_rejects_non_loopback_assets_and_preserves_text_fallback() -> None:
    renderer = MAP_CONTROLLER_JS.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")

    assert 'const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost"])' in renderer
    assert "candidate.origin !== base.origin" in renderer
    assert 'throw new Error("map_asset_url_not_loopback")' in renderer
    assert 'canvas.setAttribute("tabindex", "-1")' in renderer
    assert "Text route description" in app
    assert "state.workspace.map_alternative" in app


def test_compare_selection_preserves_context_and_drives_alternative_evidence() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    compare = COMPARE_JS.read_text(encoding="utf-8")

    assert "onSelectAlternative: (planId) => selectContext({ selected_alternative_id: planId })" in source
    assert 'Object.hasOwn(patch, "selected_stop_id")' in source
    assert 'Object.hasOwn(patch, "selected_alternative_id")' in source
    assert "workspace?.alternative_evidence?.[selectedId]" in compare
    assert "evidence.plan_content_hash === option.plan_content_hash" in compare
    assert 'code: "compare_option_not_selected"' in compare
    assert "selectedAlternativeId: activeAlternativeId()" in source
    assert "Strict route time" in compare
    assert "Road-validated route" in compare


def test_w3_exposes_every_stop_and_enables_typed_draft_controls() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")

    assert "Original only; removed by" in app
    assert 'selectedAlternative ? "selected alternative" : "registered repair"' in app
    assert 'id="map-edit-button" type="button"' in index
    assert 'id="map-edit-button" type="button" disabled' not in index
    assert '<fieldset class="w3-controls">' in index
    for operation in ("keep", "move", "lock", "flexible", "feedback", "replace", "add"):
        assert f'id="draft-{operation}"' in index
    assert "state.workspace.draft_capabilities?.candidate_choices" in app
    assert "candidate.replaces_stop_id === selected.id" in app
    assert "refreshMapEditDialog()" in app
    assert 'if ($("#map-edit-dialog").open) refreshMapEditDialog();' in app
    assert 'operationEnabled("replace_nearby")' in app
    assert 'operationEnabled("add_candidate")' in app
    assert "The selected marker belongs to a repair preview" in app
    assert "Drafts edit the accepted plan" in index
    assert "pendingSelection = request" in app
    assert "if (pendingSelection) await pendingSelection" in app
    assert "selectedCandidate.replaces_stop_id !== selectedStopId" in app
    assert "selectedCandidateId = null" in app
    copilot = (STATIC / "js" / "copilot.js").read_text(encoding="utf-8")
    assert 'plan.role === "original"' in app
    assert "candidate.properties?.stop_id === state.session.selected_stop_id" in app
    assert 'alreadyAdded ? "Added to draft" : "Add to draft"' in copilot
    assert "await restore({ announce: false })" in copilot
    assert "The parent plan is unchanged" in app


def test_map_candidate_and_full_selection_context_are_synchronized_to_copilot() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    copilot = (STATIC / "js" / "copilot.js").read_text(encoding="utf-8")

    assert '$("#registered-candidate").addEventListener("change"' in app
    assert "selected_candidate_id: event.currentTarget.value || null" in app
    assert "Candidate:" in copilot
    assert "Segment:" in copilot
    assert "Alternative:" in copilot


def test_mobile_toolbar_keeps_visible_labels_and_baseline_is_non_mutating() -> None:
    css = APP_CSS.read_text(encoding="utf-8")
    compare = (STATIC / "js" / "compare.js").read_text(encoding="utf-8")

    assert ".toolbar-button span { display: none; }" not in css
    assert "html::-webkit-scrollbar { display: none; }" in css
    assert ".topbar { gap: 8px; }" in css
    assert ".textual-map-alternative[open] { bottom: 112px; max-height: min(42dvh, 300px); overflow: auto; }" in css
    assert 'status-pill">Current trip' in compare
    assert "Baseline map shown" in compare
    assert "accept-repair" not in compare


def test_control_layout_preserves_comfort_and_map_clearance_across_breakpoints() -> None:
    css = APP_CSS.read_text(encoding="utf-8")
    markup = INDEX.read_text(encoding="utf-8")

    assert 'role="toolbar" aria-label="Map actions"' in markup
    assert ".toolbar-button { min-width: 44px; min-height: 44px; flex: 0 0 auto;" in css
    assert ".main-workspace { min-width: 0; min-height: 0; overflow: hidden;" in css
    assert ".draft-bar { position: absolute; z-index: 6; left: 16px; right: 16px;" in css
    assert "#geographic-map.geographic-map { position: absolute; inset: 0; width: 100%; height: 100%;" in css
    assert ".map-context-card { left: 12px; right: 12px; top: auto; bottom: 112px;" in css
    assert ".map-legend { left: 10px; right: 10px; bottom: 108px;" in css
    assert ".action-stack .primary:disabled { opacity: 1;" in css
    assert ".itinerary-rail { display: none; }" in css
    assert ".repair-panel { position: static; width: auto; transform: none;" in css
    assert ".workspace-toolbar { justify-content: flex-start; scroll-snap-type: inline proximity; }" in css
    assert ".workspace-toolbar::-webkit-scrollbar { display: block; height: 4px; }" in css
    assert ".map-stage { min-height: max(430px, calc(100dvh - 250px));" in css
    assert ".textual-map-alternative[open] { bottom: 112px;" in css
    assert ".draft-bar #inline-preview { grid-column: 3; grid-row: 1; }" in css
    assert ".evidence-grid > .evidence-card:nth-child(3)" in css
    assert "max-height: calc(100dvh - 24px)" in css


def test_map_runtime_removes_failed_assets_and_reports_late_errors() -> None:
    source = MAP_CONTROLLER_JS.read_text(encoding="utf-8")

    assert 'existing.dataset.loadState === "loaded"' in source
    assert "existing.remove();" in source
    assert "script.remove();" in source
    assert 'nextMap.on("error"' in source
    assert 'code: "map_render_failed"' in source
    assert 'element.hidden = false;' in APP_JS.read_text(encoding="utf-8")


def test_selected_alternative_drives_its_own_markers_and_text_route() -> None:
    renderer = MAP_JS.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")

    assert "requestedPlanId" in renderer
    assert "activePlanId: planId" in renderer
    assert "const displayed = proposalEligible ? selectedAlternative : selectedAlternative || repaired || original;" in app
    assert "routeSummaryForPlan(selectedAlternative)" in app


def test_compare_and_evidence_preserve_nulls_and_selected_plan_ownership() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    compare = (STATIC / "js" / "compare.js").read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")

    assert 'preservation == null ? "Unavailable"' in app
    assert 'return "Unavailable";' in compare
    assert "Select one available repair above to preview its route" in compare
    assert "alternatives[0]" not in app
    assert '"Unevaluated reference"' in app
    assert "const methods = selectedEvidence?.methods || {};" in app
    assert "const sourceHashCount = Object.keys(selectedEvidence?.source_hashes || {}).length;" in app
    assert "routeSummaryForPlan(selectedPlan)" in app
    assert "evidence-bundle?plan_id=${encodeURIComponent(selectedAlternative.plan_id)}" in app
    assert "2 · Policy and modeled-feasibility checks" in app
    assert app.index("3 · Route and source evidence") < app.index("5 · Technical lineage")
    assert "W3 persists typed drafts" in index
    assert "acceptance remains disabled until the W5" in index


def test_w3_restores_session_without_putting_the_mutation_token_in_a_url() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'const SESSION_STORAGE_KEY = "itinerary-repair-copilot.session.v1"' in source
    assert "sessionStorage.setItem" in source
    assert "mutation_token: payload.mutation_token" in source
    assert "X-Session-Token" in source
    assert "?mutation_token=" not in source
    assert "restoreOrCreateSession" in source


def test_w3_renders_evaluated_preview_state_and_metrics_without_registered_fallback() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")
    css = APP_CSS.read_text(encoding="utf-8")

    assert 'const repair = evaluated || registered ? proposal.repair : state.workspace.repair;' in app
    assert 'registered ? "Registered fixture proposal"' in app
    assert 'evaluated ? "No eligible draft repair"' in app
    assert 'proposal?.reason || "Unavailable"' in app
    assert 'const r = evaluated ? proposal.repair : state.workspace.repair;' in app
    assert 'evaluated ? proposal.evaluation_status || proposal.eligibility' in app
    assert 'registered ? proposal.certificate_id || "Registered fixture"' in app
    assert "function activeProposalPlan()" in app
    assert "function activeGeography()" in app
    assert 'proposal?.schema_version === "draft-preview-v1"' in app
    assert 'state.session.proposal?.schema_version === "draft-preview-v1"' in app
    assert 'const displayed = proposalEligible ? selectedAlternative : selectedAlternative || repaired || original;' in app
    assert 'code: "evaluated_preview_geography_unavailable"' in app
    assert "proposalMatches ? activeProposalPlan()" in app
    assert "Preview evidence is session-local" in app
    assert 'aria-labelledby="map-edit-title"' in index
    assert 'aria-labelledby="info-dialog-title"' in index
    assert ".edit-actions button, .prototype-tools button, .dialog-card footer button { min-height: 44px;" in css
