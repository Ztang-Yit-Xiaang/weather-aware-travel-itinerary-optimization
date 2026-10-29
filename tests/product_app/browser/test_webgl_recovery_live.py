"""A real evaluated child survives real WEBGL_lose_context loss and restoration.

Only the browser's real graphics extension injects the fault. No API, optimizer,
map event, tile state, or graphics method is mocked. No screenshot/video is
captured while the context is lost or before loaded-tile readiness.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytest.importorskip("playwright.sync_api")
from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
    _independent_geometry_hash,
    _map_ack,
    _preserve_browser_evidence,
    _write_json,
    expect,
    sync_playwright,
)
from waypoint_ui_fixture import OBSERVE_REAL_MAPS

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_WEBGL_RECOVERY_DISPOSABLE") == "1"
CONTROLLER = "src/itinerary_system/product_app/static/js/map-controller-v2.js"
SOURCE_PATHS = (CONTROLLER, "tests/product_app/browser/test_webgl_recovery_live.py",
    "tests/product_app/browser/test_e2e1_live_copilot_chat.py", "tests/product_app/browser/waypoint_ui_fixture.py")

# Observation only: getStyle/queryRenderedFeatures read the actual renderer.
OBSERVE_GRAPHICS = """() => {
  const {map,gl,events}=window.__e2eWebglRecovery;
  const container=map.getContainer(),lost=gl.isContextLost();
  const style=lost ? null : map.getStyle();
  return {observed_at:new Date().toISOString(),context_lost:lost,
    map_dataset:{...container.dataset},stage_dataset:{...document.querySelector('.map-stage').dataset},
    status:{text:document.querySelector('#map-render-status').textContent,
      hidden:document.querySelector('#map-render-status').hidden,
      dataset:{...document.querySelector('#map-render-status').dataset}},
    all_tiles_loaded:!lost && map.areTilesLoaded(),map_loaded:!lost && map.loaded(),
    sources:Object.fromEntries(Object.entries(style?.sources || {}).filter(([key])=>key.startsWith('v2-'))),
    layers:style?.layers.filter(layer=>layer.id.startsWith('v2-')) || [],
    rendered_features:lost ? [] : map.queryRenderedFeatures()
      .filter(feature=>feature.layer.id.startsWith('v2-'))
      .map(feature=>({layer:feature.layer.id,properties:feature.properties,geometry:feature.geometry})),
    events:[...events]};
}"""


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_rows():
    return [{"path":path,"sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
        "size_bytes":(ROOT/path).stat().st_size} for path in SOURCE_PATHS]


def _assert_withdrawn(observation, *, context_lost, preview_state):
    assert observation["context_lost"] is context_lost
    assert observation["map_dataset"]["mapReady"] == "false"
    assert not set(observation["map_dataset"]).intersection({"mapReadyAt","mapTileCoverage","mapCoverageMethod",
        "renderedMapPlanId","renderedPlanContentHash","renderedGeometryHash","renderedMarkerIds"})
    assert not set(observation["stage_dataset"]).intersection({"renderedMapPlanId","renderedGeometryHash","renderedMarkerIds"})
    assert observation["stage_dataset"]["previewState"] == preview_state


def _assert_canonical_renderer(observation, geography, geometry_hash):
    assert not observation["context_lost"] and observation["map_loaded"] and observation["all_tiles_loaded"]
    assert observation["map_dataset"]["mapReady"] == "true"
    assert .95 <= float(observation["map_dataset"]["mapTileCoverage"]) <= 1
    assert observation["map_dataset"]["mapCoverageMethod"] == "loaded_tile_projected_union_area_v1"
    expected_markers = sorted(feature["properties"]["stop_id"] for feature in geography["stops"]["features"])
    for dataset in (observation["map_dataset"], observation["stage_dataset"]):
        assert dataset["renderedMapPlanId"] == geography["plan_id"]
        assert dataset["renderedGeometryHash"] == geometry_hash
        assert json.loads(dataset["renderedMarkerIds"]) == expected_markers
    assert observation["map_dataset"]["renderedPlanContentHash"] == geography["content_hash"]
    assert observation["stage_dataset"]["activePlanId"] == geography["plan_id"]
    assert observation["stage_dataset"]["activePlanContentHash"] == geography["content_hash"]
    assert observation["stage_dataset"]["previewState"] == "PREVIEW_RENDERED"
    for source, field in (("v2-routes","validated_legs"),("v2-stops","stops")):
        actual = [feature for feature in observation["sources"][source]["data"]["features"]
            if feature["properties"]["plan_id"] == geography["plan_id"]]
        # Display-role is the only frontend decoration; canonical values remain exact.
        normalized = [{**feature,"properties":{key:value for key,value in feature["properties"].items()
            if key != "display_role"}} for feature in actual]
        assert normalized == geography[field]["features"], f"{source} differs from the backend child artifact"
        assert actual and all(feature["properties"]["display_role"] == "active" for feature in actual)
    active = [feature for feature in observation["rendered_features"]
        if feature["layer"] in {"v2-route-active","v2-stops-active"}]
    assert {feature["layer"] for feature in active} == {"v2-route-active","v2-stops-active"}
    assert all(feature["properties"]["plan_id"] == geography["plan_id"] for feature in active)
    assert all(feature["properties"]["content_hash"] == geography["content_hash"] for feature in active)
    visible_markers = {feature["properties"]["stop_id"] for feature in active if feature["layer"] == "v2-stops-active"}
    assert "pfeiffer_big_sur_state_park" in visible_markers
    assert "bixby_creek_bridge_viewpoint" not in visible_markers


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires dedicated deterministic webgl-recovery mode")
def test_real_webgl_loss_withdraws_ack_and_restores_exact_evaluated_child(tmp_path):
    evidence = Path(os.environ.get("PRODUCT_E2E_WEBGL_RECOVERY_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    http_index, checkpoints, fault_inputs, errors, console_errors, warnings, failures = [], [], [], [], [], [], []
    initial_sources = _source_rows()
    phase = "preflight"
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, **({"executable_path":str(CHROME)} if CHROME.is_file() else {}))
        context = browser.new_context(viewport={"width":1440,"height":900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        page = context.new_page()
        page.on("pageerror", lambda error:errors.append(str(error)))
        page.on("console", lambda message:console_errors.append(message.text) if message.type == "error" else None)
        page.on("console", lambda message:warnings.append(message.text) if message.type == "warning" else None)
        page.on("requestfailed", lambda request:failures.append({"url":request.url,"failure":request.failure}))

        def record(response):
            path = urlsplit(response.url).path
            if not path.startswith("/api/") and not path.endswith("/map-controller-v2.js"):
                return
            row = {"phase":phase, "observed_at":datetime.now(UTC).isoformat(), **_http_record(response)}
            # Persist just the non-secret build witness, never authorization/session headers.
            row["request_product_build_id"] = response.request.headers.get("x-product-build-id")
            name = f"http_{len(http_index)+1:03d}.json"
            _write_json(evidence/name, row)
            http_index.append({"file":name,"phase":phase,"method":row["method"],"status":response.status,"url":response.url})
        page.on("response", record)

        def checkpoint(label):
            value = page.evaluate(OBSERVE_GRAPHICS)
            artifact = f"{label}.json"
            _write_json(evidence/artifact, value)
            checkpoints.append({"label":label,"artifact":artifact,"observed_at":value["observed_at"]})
            return value

        def capture(label, geography):
            before = checkpoint(label+"_before_capture")
            _assert_canonical_renderer(before, geography, expected_hash)
            _capture_ready(page, evidence/f"{label}.png")
            after = checkpoint(label+"_after_capture")
            _assert_canonical_renderer(after, geography, expected_hash)
            for field in ("map_dataset","stage_dataset","sources","layers","rendered_features","events"):
                assert after[field] == before[field], f"{field} changed during screenshot capture"
            return after

        with _preserve_browser_evidence(context, page, evidence, {"page_errors":errors,"console_errors":console_errors,"request_failures":failures}):
            try:
                health_response = context.request.get(f"{BASE_URL}/api/health")
                health_text = health_response.text()
                _write_json(evidence/"preflight_health.json", {"method":"GET","url":health_response.url,
                    "response_status":health_response.status,"response_body_utf8":health_text,
                    "response_body_sha256":hashlib.sha256(health_text.encode()).hexdigest()})
                assert health_response.status == 200
                health = health_response.json()
                assert health["ready"] and health["components"]["map"]["status"] == "ready"
                assert health["components"]["routing"]["status"] == "ready"
                assert health["components"]["openai"]["code"] == "deterministic_adapter_selected"
                assert health["components"]["openai"]["status"] == "disabled"

                phase = "initial_parent"
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                parent = _map_ack(page)
                _capture_ready(page, evidence/"initial_parent.png")
                client_build = page.locator('meta[name="product-build-id"]').get_attribute("content")
                assert client_build == health["build_id"]
                phase = "stage_replacement"
                page.locator('.day-card[data-day="5"]').click()
                expect(page.locator("#map-context-title")).to_contain_text("Bixby Creek Bridge")
                page.locator("#map-edit-button").click()
                page.locator("#inspect-edit-stop").click()
                page.locator(".legacy-edit-controls summary").click()
                page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                with page.expect_response(lambda response:response.url.endswith("/draft/operations")) as staged:
                    page.locator("#draft-replace").click()
                assert staged.value.status == 200
                page.locator("#close-map-edit").click()
                assert _map_ack(page)["rendered_map_plan_id"] == parent["rendered_map_plan_id"]
                phase = "real_optimizer_preview"
                with page.expect_response(lambda response:response.url.endswith("/preview"), timeout=90_000) as previewed:
                    page.locator("#inline-preview").click()
                assert previewed.value.status == 200
                response = previewed.value.json()
                proposal = response["proposal"]
                execution = proposal["runtime_execution"]
                evaluation = execution["evaluation_certificate"]
                geography = proposal["geography_plan"]
                _write_json(evidence/"preview_response.json", proposal)
                _write_json(evidence/"plan_diff.json", execution["plan_diff"])
                _write_json(evidence/"evaluation_summary.json", evaluation)
                _write_json(evidence/"route_geometry.json", geography)
                assert proposal["state"] == "eligible" and proposal["acceptance_eligible"] is True
                assert execution["evaluation_executed"] is True and evaluation["comparison_eligibility"] == "eligible"
                assert evaluation["eligible"] is True and evaluation["failure_count"] == 0
                assert execution["evaluation_id"] == evaluation["evaluation_id"] == evaluation["certificate_id"] == proposal["certificate_id"]
                assert evaluation["route_validation"]["fallback_leg_count"] == evaluation["route_validation"]["missing_leg_count"] == 0
                assert execution["parent_plan_id"] == proposal["parent_plan_id"] == parent["active_plan_id"]
                assert execution["child_plan_id"] == execution["child_plan"]["plan_id"] == evaluation["plan_id"] == geography["plan_id"] == proposal["child_plan_id"]
                assert execution["child_plan_content_hash"] == evaluation["plan_content_hash"] == geography["content_hash"] == proposal["child_plan_content_hash"]
                assert execution["plan_diff"]["child_plan_id"] == proposal["child_plan_id"]
                assert execution["plan_diff"]["parent_plan_id"] == parent["active_plan_id"]
                assert response["session"]["accepted_plan_id"] == parent["active_plan_id"]
                assert [stop["stop_id"] for stop in execution["plan_diff"]["added_stops"]] == ["pfeiffer_big_sur_state_park"]
                assert [stop["stop_id"] for stop in execution["plan_diff"]["deleted_stops"]] == ["bixby_creek_bridge_viewpoint"]
                expect(page.locator(".map-stage")).to_have_attribute("data-preview-state","PREVIEW_RENDERED",timeout=40_000)
                child = _map_ack(page)
                expected_hash = _independent_geometry_hash(page, geography)
                expected_markers = sorted(feature["properties"]["stop_id"] for feature in geography["stops"]["features"])
                assert child["active_plan_id"] == child["rendered_map_plan_id"] == proposal["child_plan_id"]
                assert child["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                assert child["rendered_geometry_hash"] == expected_hash != parent["rendered_geometry_hash"]
                assert child["rendered_marker_ids"] == expected_markers

                installed = page.evaluate("""() => {
                  const map=[...window.__e2eObservedMaps].reverse().find(value=>value.getContainer().id==='geographic-map'&&value.getContainer().isConnected);
                  const gl=map.painter.context.gl,extension=gl.getExtension('WEBGL_lose_context'),events=[];
                  window.__e2eWebglRecovery={map,gl,extension,events};
                  for(const type of ['webglcontextlost','webglcontextrestored']) map.on(type,event=>{
                    events.push({type,at:new Date().toISOString(),trusted:event.originalEvent.isTrusted,
                      context_lost:gl.isContextLost(),map_dataset:{...map.getContainer().dataset},
                      stage_dataset:{...document.querySelector('.map-stage').dataset}});
                  });
                  return {extension_available:!!extension,context_lost:gl.isContextLost()};
                }""")
                assert installed == {"extension_available":True,"context_lost":False}, "A real working WEBGL_lose_context extension is required, not a skipped graphics gate"
                phase = "before_context_loss"
                before = capture("before_context_loss", geography)
                assert before["stage_dataset"]["previewRequestId"] == execution["request_id"]
                phase = "context_lost"
                fault_inputs.append({"action":"real_WEBGL_lose_context.loseContext", "javascript":"window.__e2eWebglRecovery.extension.loseContext()",
                    "at":page.evaluate("() => new Date().toISOString()"),"api_response_mocking":False,"synthetic_event_dispatch":False})
                page.evaluate("() => window.__e2eWebglRecovery.extension.loseContext()")
                page.wait_for_function("() => window.__e2eWebglRecovery.gl.isContextLost() && window.__e2eWebglRecovery.events.some(event=>event.type==='webglcontextlost')",timeout=10_000)
                lost = checkpoint("context_lost_immediate")
                _assert_withdrawn(lost, context_lost=True, preview_state="MAP_RENDER_FAILED")
                assert not lost["status"]["hidden"]
                assert lost["events"][-1]["trusted"] is True
                _assert_withdrawn(lost["events"][-1], context_lost=True, preview_state="MAP_RENDER_FAILED")
                assert lost["stage_dataset"]["activePlanId"] == child["active_plan_id"]
                page.wait_for_timeout(1200)
                _assert_withdrawn(checkpoint("context_lost_after_1200ms"), context_lost=True, preview_state="MAP_RENDER_FAILED")
                phase = "context_restoring"
                fault_inputs.append({"action":"real_WEBGL_lose_context.restoreContext", "javascript":"window.__e2eWebglRecovery.extension.restoreContext()",
                    "at":page.evaluate("() => new Date().toISOString()"),"api_response_mocking":False,"synthetic_event_dispatch":False})
                page.evaluate("() => window.__e2eWebglRecovery.extension.restoreContext()")
                page.wait_for_function("() => !window.__e2eWebglRecovery.gl.isContextLost() && window.__e2eWebglRecovery.events.some(event=>event.type==='webglcontextrestored')",timeout=20_000)
                restoration = checkpoint("context_restored_event")
                restored_event = next(event for event in restoration["events"] if event["type"] == "webglcontextrestored")
                assert restored_event["trusted"] is True
                _assert_withdrawn(restored_event, context_lost=False, preview_state="PREVIEW_DATA_READY")
                page.wait_for_function("""() => {
                  const {map,gl}=window.__e2eWebglRecovery;
                  return !gl.isContextLost() && map.loaded() && map.areTilesLoaded()
                    && map.getContainer().dataset.mapReady==='true'
                    && Number(map.getContainer().dataset.mapTileCoverage)>=.95;
                }""",timeout=40_000)
                phase = "after_context_restoration"
                after = capture("after_context_restoration", geography)
                assert _map_ack(page) == child
                assert after["map_dataset"]["mapReadyAt"] >= restored_event["at"]
                assert after["sources"] == before["sources"] and after["layers"] == before["layers"]
                assert sorted(map(_json_hash,after["rendered_features"])) == sorted(map(_json_hash,before["rendered_features"]))
                assert [event["type"] for event in after["events"]] == ["webglcontextlost","webglcontextrestored"]
                assert errors == console_errors == failures == []
                assert all(row["status"] < 400 for row in http_index)
                assert not any(row["url"].endswith(("/accept","/copilot/messages")) for row in http_index)
                assert _source_rows() == initial_sources, "Source changed during the browser journey"
                served = [row for row in http_index if urlsplit(row["url"]).path.endswith("/map-controller-v2.js")]
                assert len(served) == 1
                served_controller = json.loads((evidence/served[0]["file"]).read_text(encoding="utf-8"))
                assert served_controller["response_body_sha256"] == initial_sources[0]["sha256"]
                for row in http_index:
                    if row["method"] == "POST":
                        assert json.loads((evidence/row["file"]).read_text(encoding="utf-8"))["request_product_build_id"] == client_build
                _write_json(evidence/"source_identity.json", {"schema_version":"webgl-browser-source-identity-v1",
                    "health_build_id":health["build_id"],"html_build_id":client_build,"inputs_before":initial_sources,
                    "inputs_after":_source_rows(),"served_controller_http":served[0]["file"],"source_unchanged":True,
                    "full_source_scope":"Runner file_io_manifest.json binds the complete production/test/fixture source universe"})
                _write_json(evidence/"map_render_ack.json", child)
                _write_json(evidence/"llm_io.json", {"provider_call_count":0,"llm_input":None,"llm_output":None,
                    "reason":"No chat submitted; provider dispatch is additionally measured by the finalized application-server isolation receipt."})
                _write_json(evidence/"browser_assertions.json", {"schema_version":"webgl-recovery-browser-v1","assertions_passed":True,
                    "request_id":execution["request_id"],"parent_plan_id":parent["active_plan_id"],"child_plan_id":child["active_plan_id"],
                    "child_plan_content_hash":child["active_plan_content_hash"],"evaluation_id":execution["evaluation_id"],
                    "comparison_eligibility":evaluation["comparison_eligibility"],"geometry_hash":expected_hash,
                    "rendered_map_plan_id":child["rendered_map_plan_id"],"rendered_marker_ids":expected_markers,
                    "real_context_loss":True,"real_context_restoration":True,"trusted_graphics_events":True,
                    "ack_withdrawn_immediately":True,"ack_stays_withdrawn_1200ms":True,"restoration_waits_for_real_ready":True,
                    "sources_sha256":_json_hash(after["sources"]),"layers_sha256":_json_hash(after["layers"]),
                    "rendered_features_sha256":_json_hash(sorted(map(_json_hash,after["rendered_features"]))),
                    "renderer_matches_backend_child":True,"renderer_exact_before_after":True,"no_screenshot_while_lost":True,
                    "accept_called":False,"api_response_mocking":False,"graphics_implementation_mocking":False})
            finally:
                _write_json(evidence/"http_index.json", http_index)
                _write_json(evidence/"checkpoints.json", checkpoints)
                _write_json(evidence/"graphics_fault_inputs.json", fault_inputs)
                _write_json(evidence/"browser_errors.json", {"page_errors":errors,"console_errors":console_errors,
                    "console_warnings":warnings,"request_failures":failures})
        browser.close()
