from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
COMPARE = STATIC / "js" / "compare.js"
APP = STATIC / "js" / "app.js"
CSS = STATIC / "css" / "app.css"


def _resolve_with_node(selected: str | None, *, plan_hash: str = "hash-child", evidence_hash: str = "hash-child") -> dict[str, object]:
    workspace = {
        "alternatives": [
            {
                "plan_id": "child-plan",
                "plan_content_hash": "hash-child",
                "method_label": "Recommended repair",
                "status": "ineligible",
                "display_status": "Ineligible",
                "ranking_eligible": False,
                "weighted_edit_cost": None,
                "route_total_minutes": None,
            }
        ],
        "geography": {
            "plans": [
                {
                    "plan_id": "child-plan",
                    "content_hash": plan_hash,
                }
            ]
        },
        "alternative_evidence": {
            "child-plan": {
                "plan_id": "child-plan",
                "plan_content_hash": evidence_hash,
            }
        },
    }
    session = {"selected_alternative_id": selected}
    script = f"""
      const module = await import({json.dumps(COMPARE.as_uri() + '?contract-test')});
      const result = module.resolveCompareSelection(
        {json.dumps(workspace)},
        {json.dumps(session)},
      );
      process.stdout.write(JSON.stringify({{
        state: result.state,
        code: result.code,
        planId: result.option?.plan_id ?? null,
        evidenceExact: result.evidenceExact ?? false,
      }}));
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_exact_selected_option_resolves_even_when_ineligible_and_metrics_are_null() -> None:
    assert _resolve_with_node("child-plan") == {
        "state": "selected",
        "code": "compare_option_ready",
        "planId": "child-plan",
        "evidenceExact": True,
    }


def test_missing_selection_never_falls_back_to_the_first_alternative() -> None:
    assert _resolve_with_node(None) == {
        "state": "unavailable",
        "code": "compare_option_not_selected",
        "planId": None,
        "evidenceExact": False,
    }


def test_plan_hash_mismatch_fails_closed_before_map_or_evidence_resolution() -> None:
    assert _resolve_with_node("child-plan", plan_hash="forged-hash") == {
        "state": "unavailable",
        "code": "compare_option_hash_mismatch",
        "planId": "child-plan",
        "evidenceExact": False,
    }


def test_evidence_hash_mismatch_keeps_option_inspectable_but_disables_evidence() -> None:
    assert _resolve_with_node("child-plan", evidence_hash="stale-hash") == {
        "state": "selected",
        "code": "compare_option_evidence_unavailable",
        "planId": "child-plan",
        "evidenceExact": False,
    }


def test_compare_owns_two_map_controllers_and_mobile_toggle() -> None:
    source = COMPARE.read_text(encoding="utf-8")

    assert source.count("createGeographicMapController({") == 2
    assert 'paneId: "compare-baseline"' in source
    assert 'paneId: "compare-option"' in source
    assert "synchronizeMapCameras(baselineController, optionController)" in source
    assert 'window.matchMedia("(max-width: 820px)")' in source
    assert 'data-compare-mobile-view="baseline"' in source
    assert 'data-compare-mobile-view="option"' in source
    assert 'window.dispatchEvent(new Event("resize"))' in source
    assert "baselineController.dispose();" in source
    assert "optionController?.dispose();" in source


def test_baseline_map_renders_before_an_option_is_selected() -> None:
    source = COMPARE.read_text(encoding="utf-8")

    assert 'if (!baseline) {' in source
    assert 'if (!baseline || resolution.state !== "selected")' not in source
    baseline_render = source.index("baselineController.render({")
    option_gate = source.index('if (resolution.state === "selected") {', baseline_render)
    assert baseline_render < option_gate
    assert "optionController?.dispose();" in source


def test_compare_source_is_strict_utf8_without_replacement_text() -> None:
    source = COMPARE.read_bytes().decode("utf-8", errors="strict")

    assert "\ufffd" not in source
    assert "鈥?" not in source


def test_selection_stays_on_compare_and_evidence_is_explicit() -> None:
    compare_source = COMPARE.read_text(encoding="utf-8")
    app_source = APP.read_text(encoding="utf-8")
    active_render = app_source[app_source.index("function renderCompare()") :]
    active_render = active_render[: active_render.index("function comparisonTable()")]

    assert "onSelectAlternative(button.dataset.compareOption)" in compare_source
    assert "navigate(" not in compare_source
    assert "onSelectAlternative: (planId) => selectContext({ selected_alternative_id: planId })" in active_render
    assert "onOpenEvidence" in active_render
    assert "alternatives[0]" not in app_source
    assert "No first-option fallback was used." in app_source


def test_compare_preserves_unavailable_and_has_no_acceptance_action() -> None:
    source = COMPARE.read_text(encoding="utf-8")

    assert 'return "Unavailable";' in source
    assert "Ineligible options remain inspectable" in source
    assert "Select one available repair above to preview its route" in source
    assert "Choose option (not enabled)" in source
    assert "accept-repair" not in source
    assert "/accept" not in source


def test_each_map_pane_has_local_status_and_text_fallback() -> None:
    source = COMPARE.read_text(encoding="utf-8")

    assert 'data-compare-map-status' in source
    assert 'data-compare-text' in source
    assert "setPaneFallback(baselinePane, payload)" in source
    assert "setPaneFallback(optionPane, payload)" in source
    assert "setPaneStatus(baselinePane, payload)" in source
    assert "setPaneStatus(optionPane, payload)" in source
    assert "Use the text route evidence in this pane." in source


def test_compare_maps_wait_for_nonzero_layout_and_resize_after_ready() -> None:
    source = COMPARE.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert 'class="map-stage compare-map-stage"' in source
    assert 'class="compare-map-canvas"' in source
    assert ".map-stage.compare-map-stage { width: 100%; height: clamp(320px, 42vh, 520px); min-height: 320px; max-height: 520px; }" in css
    assert ".compare-map-canvas { position: absolute; inset: 0; width: 100%; height: 100%;" in css
    assert "const generation = compareGeneration;" in source
    assert "if (generation !== compareGeneration || !root.isConnected) return;" in source
    assert "activeCompare = startCompareMaps(root, workspace, mapConfig, baseline, resolution);" in source
    assert source.count('window.dispatchEvent(new Event("resize"))') >= 3


def test_camera_sync_starts_only_after_both_maps_finish_their_initial_fit() -> None:
    source = COMPARE.read_text(encoding="utf-8")
    lifecycle = source[source.index("function startCompareMaps(") :]
    lifecycle = lifecycle[: lifecycle.index("export function disposeCompareWorkspace")]

    ready_gate = lifecycle.index('if (readyPanes.size !== 2')
    sync_assignment = lifecycle.index(
        "stopCameraSync = synchronizeMapCameras(baselineController, optionController)",
    )
    after_option_render = lifecycle[lifecycle.index("optionController.render({") :]
    after_option_render = after_option_render[: after_option_render.index("return {")]

    assert "const readyPanes = new Set();" in lifecycle
    assert 'recordReady("baseline", payload);' in lifecycle
    assert 'recordReady("option", payload);' in lifecycle
    assert ready_gate < sync_assignment
    assert "synchronizeMapCameras(" not in after_option_render
    assert "stopCameraSync?.();" in lifecycle


def test_compare_is_user_oriented_and_keeps_technical_identity_in_evidence() -> None:
    source = COMPARE.read_text(encoding="utf-8")

    assert "The recommended eligible repair is selected automatically" in source
    assert "View advanced Evidence" in source
    assert "Technical plan, diff, hash, method, and certificate identities" in source
    assert "<code>${escapeHtml(option.plan_id)}" not in source
    assert "<code>${escapeHtml(plan.content_hash)}" not in source
