"""Real UI remount gate with passive DOM observation, never readiness injection.

Run one ordinary fresh deterministic server. The actual UI creates and evaluates
the replacement preview, then Select/Edit remount that same canonical plan.
The observer only reads DOM state; it does not alter fetch, crypto, MapLibre,
timers, API replies, or readiness. No acceptance or provider call is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
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

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_REMOUNT_DISPOSABLE") == "1"
OBSERVE_REMOUNT = r"""(() => {
  const ids = new WeakMap(); let nextId = 0, active = null;
  const id = element => element ? (ids.has(element) ? ids.get(element) : (ids.set(element, ++nextId), nextId)) : null;
  const data = element => element ? {...element.dataset} : null;
  function sample(reason, detail = null) {
    if (!active) return;
    if (active.rows.length >= 5000) { active.overflow = true; return; }
    const map = document.querySelector('#geographic-map');
    const stage = document.querySelector('.map-stage');
    const accept = document.querySelector('#accept-repair');
    active.rows.push({reason, detail, sequence:active.rows.length+1,
      observed_at:new Date().toISOString(), performance_ms:performance.now(), path:location.pathname,
      previous_map_id:id(active.previous), previous_connected:active.previous?.isConnected ?? false,
      previous_map:data(active.previous), current_map_id:id(map), current_connected:map?.isConnected ?? false,
      current_map:data(map), stage:data(stage),
      accept:{present:!!accept, disabled:accept?.disabled ?? null, visible:!!accept?.getClientRects().length},
      status_code:document.querySelector('#map-render-status')?.dataset.mapStatusCode ?? null});
  }
  new MutationObserver(records => sample('mutation', {
    count:records.length, attributes:[...new Set(records.filter(row => row.type==='attributes').map(row => row.attributeName))],
    child_list:records.some(row => row.type==='childList'),
  })).observe(document, {subtree:true, childList:true, attributes:true, attributeFilter:[
    'disabled','hidden','data-preview-state','data-map-ready','data-map-ready-at','data-map-tile-coverage',
    'data-rendered-map-plan-id','data-rendered-plan-content-hash','data-rendered-geometry-hash','data-rendered-marker-ids',
  ]});
  document.addEventListener('click', event => {
    if (active) sample('trusted_control_event', {trusted:event.isTrusted, target_id:event.target?.closest('button')?.id ?? null});
  }, true);
  window.__samePlanRemountObserver = {
    arm(label) {
      if (active) throw new Error('remount_observer_already_armed');
      active = {label, previous:document.querySelector('#geographic-map'), rows:[], overflow:false};
      if (!active.previous) throw new Error('remount_observer_missing_parent_map');
      const watched = active;
      sample('armed');
      function frame() { if (active !== watched) return; sample('animation_frame'); requestAnimationFrame(frame); }
      requestAnimationFrame(frame);
    },
    stop() {
      sample('stopped');
      const result = active ? {label:active.label, rows:active.rows, overflow:active.overflow} : null;
      active = null;
      return result;
    },
    snapshot() { return active ? {label:active.label, rows:active.rows, overflow:active.overflow} : null; },
  };
})();"""


def _hash(record):
    unsigned = {key: value for key, value in record.items() if key != "content_hash"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires a fresh isolated same-plan-remount server")
def test_ui_remount_withdraws_acceptance_until_same_plan_is_ready(tmp_path: Path) -> None:
    evidence = Path(os.environ.get("PRODUCT_E2E_REMOUNT_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    http, requests, errors, failures, checkpoints = [], [], [], [], []
    phase = "initial"
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {}))
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REMOUNT)
        page = context.new_page()
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failures.append({"url": request.url, "failure": request.failure}))
        page.on("request", lambda request: requests.append({"method": request.method, "url": request.url}))

        def record(response):
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            row = {"phase": phase, **_http_record(response)}
            name = f"http_{len(http)+1:03d}.json"  # Allocate only after body reads finish.
            _write_json(evidence / name, row)
            http.append({"file": name, "phase": row["phase"], "method": row["method"], "url": row["url"], "status": row["response_status_code"]})
        page.on("response", record)

        def mutations():
            return [row for row in requests if row["method"] == "POST"
                and row["url"].endswith(("/draft/operations", "/preview", "/accept"))]

        with _preserve_browser_evidence(context, page, evidence, {"console_errors": errors, "network_failures": failures}):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                parent = _map_ack(page)
                _capture_ready(page, evidence / "parent_ready.png")
                page.locator('.day-card[data-day="5"]').click()
                page.locator("#map-edit-button").click()
                page.locator("#inspect-edit-stop").click()
                page.locator(".legacy-edit-controls summary").click()
                page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                phase = "stage_replacement"
                with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as staged:
                    page.locator("#draft-replace").click()
                assert staged.value.status == 200
                operation = staged.value.json()["operation"]
                assert {key: operation[key] for key in ("type", "target", "parameters", "source")} == {
                    "type": "replace_nearby", "target": "bixby_creek_bridge_viewpoint",
                    "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"}, "source": "map"}
                page.locator("#close-map-edit").click()
                assert _map_ack(page)["rendered_map_plan_id"] == parent["active_plan_id"]
                phase = "real_evaluated_preview"
                expect(page.locator("#inline-preview")).to_be_enabled(timeout=15_000)
                with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as previewed:
                    page.locator("#inline-preview").click()
                assert previewed.value.status == 200, previewed.value.text()
                payload = previewed.value.json()
                proposal = payload["proposal"]
                assert proposal["state"] == proposal["eligibility"] == "eligible"
                assert payload["session"]["accepted_plan_id"] == proposal["parent_plan_id"] == parent["active_plan_id"]
                execution = proposal["runtime_execution"]
                certificate, diff = execution["evaluation_certificate"], execution["plan_diff"]
                assert execution["evaluation_executed"] is True
                assert proposal["child_plan_id"] == certificate["plan_id"] == diff["child_plan_id"] == execution["child_plan"]["plan_id"]
                assert proposal["child_plan_content_hash"] == certificate["plan_content_hash"] == _hash(execution["child_plan"])
                assert proposal["certificate_id"] == certificate["certificate_id"] and certificate["failures"] == []
                assert _hash(certificate) == proposal["certificate_content_hash"]
                assert _hash(diff) == proposal["diff_content_hash"]
                expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED", timeout=45_000)
                expect(page.locator("#accept-repair")).to_be_enabled()
                ready = _map_ack(page)
                assert ready["rendered_map_plan_id"] == proposal["child_plan_id"]
                assert ready["rendered_geometry_hash"] == _independent_geometry_hash(page, proposal["geography_plan"])
                assert set(ready["rendered_marker_ids"]) == (set(parent["rendered_marker_ids"]) - {"bixby_creek_bridge_viewpoint"}) | {"pfeiffer_big_sur_state_park"}
                _capture_ready(page, evidence / "preview_ready_before_remount.png")
                _write_json(evidence / "preview_response.json", payload)
                _write_json(evidence / "evaluation_summary.json", certificate)
                _write_json(evidence / "plan_diff.json", diff)
                _write_json(evidence / "map_render_ack.json", ready)

                for selector, label in [("#map-select-button", "select_remount"), ("#map-edit-button", "edit_remount")]:
                    phase = label
                    mutation_count = len(mutations())
                    page.evaluate("label => window.__samePlanRemountObserver.arm(label)", label)
                    page.locator(selector).click()
                    expect(page.locator("#geographic-map")).to_have_attribute("data-map-ready", "true", timeout=45_000)
                    expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_RENDERED")
                    expect(page.locator("#accept-repair")).to_be_enabled()
                    after = _map_ack(page)
                    _capture_ready(page, evidence / f"{label}_ready.png")
                    observation = page.evaluate("window.__samePlanRemountObserver.stop()")
                    _write_json(evidence / f"{label}_observer.json", observation)
                    assert observation and not observation["overflow"]
                    rows = observation["rows"]
                    trusted = [row for row in rows if row["reason"] == "trusted_control_event"
                        and row["detail"]["target_id"] == selector.removeprefix("#")]
                    assert len(trusted) == 1 and trusted[0]["detail"]["trusted"] is True
                    replaced = [row for row in rows if row["current_map_id"] is not None
                        and row["current_map_id"] != row["previous_map_id"]]
                    pending = [row for row in replaced if row["current_map"].get("mapReady") != "true"]
                    assert pending, "The observer must witness real map replacement, not only its final ready state"
                    for row in pending:
                        assert row["previous_connected"] is False and row["current_connected"] is True
                        assert row["accept"] == {"present": True, "disabled": True, "visible": True}, row
                        assert row["stage"]["previewState"] == "PREVIEW_DATA_READY", row
                        assert row["stage"]["activePlanId"] == proposal["child_plan_id"], row
                        assert row["stage"]["activePlanContentHash"] == proposal["child_plan_content_hash"], row
                        assert row["current_map"].get("mapReady") == "false", row
                        for field in ("renderedMapPlanId", "renderedGeometryHash", "renderedMarkerIds"):
                            assert field not in row["stage"], row
                        for record in (row["previous_map"], row["current_map"]):
                            for field in ("mapReadyAt", "mapTileCoverage", "mapCoverageMethod", "renderedMapPlanId",
                                "renderedPlanContentHash", "renderedGeometryHash", "renderedMarkerIds"):
                                assert field not in record, row
                    final = rows[-1]
                    assert final["current_map_id"] != final["previous_map_id"]
                    assert final["current_map"]["mapReady"] == "true"
                    assert float(final["current_map"]["mapTileCoverage"]) >= 0.95
                    assert final["current_map"]["renderedMapPlanId"] == final["stage"]["renderedMapPlanId"] == proposal["child_plan_id"]
                    assert final["current_map"]["renderedPlanContentHash"] == proposal["child_plan_content_hash"]
                    assert final["current_map"]["renderedGeometryHash"] == ready["rendered_geometry_hash"]
                    assert final["accept"] == {"present": True, "disabled": False, "visible": True}
                    assert after == ready and len(mutations()) == mutation_count
                    checkpoints.append({"control": selector, "observation": f"{label}_observer.json",
                        "pending_samples": len(pending), "ready_ack": after, "no_mutation_requests": True, "assertions_passed": True})
                assert errors == [] and failures == []
                assert sum(row["url"].endswith("/accept") for row in mutations()) == 0
                assert sum(row["url"].endswith("/preview") for row in mutations()) == 1
                assert not any(row["url"].endswith("/copilot/messages") for row in requests)
                assert not any(row["status"] >= 400 for row in http)
                _write_json(evidence / "browser_assertions.json", {"schema_version": "same-plan-remount-browser-v1",
                    "parent": parent, "child_plan_id": proposal["child_plan_id"], "child_plan_content_hash": proposal["child_plan_content_hash"],
                    "request_id": proposal["compiled_request"]["source_request_id"], "evaluation_id": proposal["certificate_id"],
                    "geometry_hash": ready["rendered_geometry_hash"], "checkpoints": checkpoints,
                    "observer_is_read_only": True, "api_or_digest_or_readiness_injection": False,
                    "acceptance_performed": False, "assertions_passed": True})
                _write_json(evidence / "llm_io.json", {"provider_call_count": 0, "llm_input": None, "llm_output": None,
                    "reason": "Direct UI draft, real deterministic optimizer preview, and read-only remount observation; no chat call."})
            finally:
                if not page.is_closed():
                    _write_json(evidence / "observer_at_exit.json", page.evaluate("window.__samePlanRemountObserver?.snapshot() ?? null"))
                _write_json(evidence / "http_index.json", http)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
                _write_json(evidence / "completed_checkpoints.json", checkpoints)
        browser.close()
