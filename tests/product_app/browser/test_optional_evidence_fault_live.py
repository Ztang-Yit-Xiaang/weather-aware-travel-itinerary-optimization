"""Declared backend-projection fault; real browser, HTTP, accepted artifacts intact."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("playwright.sync_api")
from build_coherence_fixture import copy_product_inputs, generation_evidence_bindings, guarded_generation
from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
    _map_ack,
    _write_json,
    expect,
    sync_playwright,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
RELATIVE = "src/itinerary_system/product_app/service.py"
ANCHOR = '        view["geography"] = geography\n        if snapshot is not None:'
FAULT = '        view["geography"] = geography\n        # Disposable optional-evidence projection fault; accepted snapshot is untouched.\n        for option in view.get("alternatives", []):\n            if option.get("role") == "low_driving":\n                view.get("alternative_evidence", {}).pop(option["plan_id"], None)\n        if snapshot is not None:'
PROTECTED = ("plan_repository", "workspaces", "decisions", "transactions", "acceptance_snapshots")


def protected_state(root):
    return {
        path.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "utf8": path.read_text(encoding="utf-8"),
        }
        for name in PROTECTED
        for path in sorted((root / name).rglob("*"))
        if path.is_file()
    }


@pytest.mark.skipif(
    not BASE_URL or os.environ.get("PRODUCT_E2E_OPTIONAL_EVIDENCE_FAULT_DISPOSABLE") != "1",
    reason="Requires a fresh private-copy self-managed optional-evidence-fault server",
)
def test_optional_evidence_projection_fault_preserves_accepted_child_and_back_navigation(tmp_path):
    root = Path(__file__).resolve().parents[3]
    evidence = Path(os.environ.get("PRODUCT_E2E_OPTIONAL_EVIDENCE_FAULT_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    run_id = os.environ["PRODUCT_E2E_RUN_ID"]
    http, requests, errors, failures, checkpoints = [], [], [], [], {}
    request_rows = {}
    phase = "generation_a"
    saved_source = None
    with (
        tempfile.TemporaryDirectory(prefix="e", dir=root / "tmp", delete=False) as temporary,
        sync_playwright() as runtime,
    ):
        repository, state = Path(temporary) / "repo", Path(temporary) / "state"
        copy_product_inputs(repository, evidence)
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("""(() => {
          window.__optionalEvidenceInputs=[];
          document.addEventListener('click',event=>{
            const target=event.target.closest?.('[data-compare-option],#evidence-back,.nav-item');
            if(!target)return;
            window.__optionalEvidenceInputs.push({trusted:event.isTrusted,type:event.type,at:Date.now(),
              id:target.id,option:target.dataset.compareOption??null,route:target.dataset.route??null,
              path:location.pathname,label:target.textContent.trim(),visible:!!target.getClientRects().length,
              disabled:target.matches(':disabled')});
          },true);
        })();""")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)

        def now():
            return datetime.now(UTC).isoformat()

        def requested(request):
            row = {
                "sequence": len(requests) + 1,
                "phase": phase,
                "method": request.method,
                "url": request.url,
                "request_body_sha256": hashlib.sha256((request.post_data or "").encode()).hexdigest(),
                "started_at": now(),
                "response_at": None,
                "finished_at": None,
                "failed_at": None,
                "status": None,
                "failure": None,
                "http_file": None,
            }
            requests.append(row)
            request_rows[request] = row

        def failed(request):
            request_rows[request].update(failed_at=now(), failure=request.failure)
            failures.append(dict(request_rows[request]))

        page.on("request", requested)
        page.on("requestfailed", failed)
        page.on("requestfinished", lambda request: request_rows[request].update(finished_at=now()))

        def record(response):
            raw = request_rows[response.request]
            raw.update(response_at=now(), status=response.status)
            if "/api/" not in response.url:
                return
            value = {
                "phase": raw["phase"],
                "request_sequence": raw["sequence"],
                "observed_at": now(),
                **_http_record(response),
            }
            name = f"http_{len(http) + 1:03d}.json"
            raw["http_file"] = name
            _write_json(evidence / name, value)
            http.append(
                {
                    "file": name,
                    "phase": raw["phase"],
                    "method": value["method"],
                    "url": value["url"],
                    "status": response.status,
                }
            )

        page.on("response", record)

        def capture(name, generation):
            with patch.dict(os.environ, {"PRODUCT_E2E_RUN_ID": run_id + "_" + generation}):
                _capture_ready(page, evidence / generation / (name + ".png"))
            if name != "compare_after_back":
                checkpoints[name] = _map_ack(page)

        try:
            with guarded_generation(repository, state, evidence, BASE_URL, run_id, "generation_a"):
                page.goto(BASE_URL + "/app/itinerary", wait_until="domcontentloaded")
                capture("initial_parent", "generation_a")
                with page.expect_response(
                    lambda r: r.url.endswith("/selection") and r.request.post_data_json.get("selected_day") == 5
                ):
                    page.locator('.day-card[data-day="5"]').click()
                expect(page.locator("#map-context-title")).to_contain_text("Bixby Creek Bridge")
                page.locator("#map-edit-button").click()
                page.locator("#inspect-edit-stop").click()
                page.locator(".legacy-edit-controls summary").click()
                with page.expect_response(
                    lambda r: (
                        r.url.endswith("/selection")
                        and r.request.post_data_json.get("selected_candidate_id") == "pfeiffer_big_sur_state_park"
                    )
                ):
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                with page.expect_response(lambda r: r.url.endswith("/draft/operations")) as staged:
                    page.locator("#draft-replace").click()
                assert staged.value.status == 200
                page.locator("#close-map-edit").click()
                expect(page.locator("#inline-preview")).to_be_enabled()
                with page.expect_response(lambda r: r.url.endswith("/preview"), timeout=90_000) as preview:
                    page.locator("#inline-preview").click()
                assert preview.value.status == 200
                proposal = preview.value.json()["proposal"]
                assert proposal["state"] == "eligible"
                expect(page.locator(".map-stage")).to_have_attribute(
                    "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
                )
                capture("evaluated_child", "generation_a")
                expect(page.locator("#accept-repair")).to_be_enabled()
                with page.expect_response(lambda r: r.url.endswith("/accept"), timeout=60_000) as accepted:
                    page.locator("#accept-repair").click()
                    expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                    expect(page.locator("#acceptance-message")).to_have_text(
                        "Accept this independently evaluated repair as the durable trip plan?"
                    )
                    page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                        "button", name="Accept repair", exact=True
                    ).click()
                assert accepted.value.status == 200
                acceptance = accepted.value.json()
                expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PARENT_RENDERED")
                capture("accepted_child", "generation_a")
                session = acceptance["session"]
                session_url = BASE_URL + "/api/sessions/" + session["session_id"]
                with page.expect_response(
                    lambda r: r.url == session_url and r.request.method == "GET"
                ) as initial_reload:
                    page.reload(wait_until="domcontentloaded")
                before = initial_reload.value.json()
                capture("accepted_before_fault", "generation_a")
                optional = next(item for item in before["workspace"]["alternatives"] if item["role"] == "low_driving")
                assert optional["plan_id"] != session["accepted_plan_id"]
                assert optional["plan_id"] in before["workspace"]["alternative_evidence"]
                page.wait_for_load_state("networkidle")
            protected_before = protected_state(state)
            source = repository / RELATIVE
            saved_source = source.read_bytes()
            original = saved_source.decode()
            newline = "\r\n" if b"\r\n" in saved_source else "\n"
            anchor, replacement = (text.replace("\n", newline) for text in (ANCHOR, FAULT))
            assert original.count(anchor) == 1
            fault = original.replace(anchor, replacement).encode()
            source.write_bytes(fault)
            _write_json(
                evidence / "fault_patch.json",
                {
                    "schema_version": "optional-evidence-projection-fault-v1",
                    "path": RELATIVE,
                    "before_utf8": original,
                    "after_utf8": fault.decode(),
                    "before_sha256": hashlib.sha256(saved_source).hexdigest(),
                    "after_sha256": hashlib.sha256(fault).hexdigest(),
                    "target_role": "low_driving",
                    "target_plan_id": optional["plan_id"],
                    "production_source_modified": False,
                    "client_state_injection": False,
                    "fetch_mocking": False,
                    "scope": "Deliberate backend optional comparison evidence projection omission; canonical accepted evidence unchanged",
                },
            )
            phase = "generation_b_reload"
            with guarded_generation(repository, state, evidence, BASE_URL, run_id, "generation_b"):
                with page.expect_response(lambda r: r.url == session_url and r.request.method == "GET") as reloaded:
                    page.reload(wait_until="domcontentloaded")
                after = reloaded.value.json()
                assert after["session"] == before["session"]
                expected = dict(before["workspace"]["alternative_evidence"])
                del expected[optional["plan_id"]]
                assert after["workspace"]["alternative_evidence"] == expected
                assert after["workspace"]["accepted_plan_pointer"] == before["workspace"]["accepted_plan_pointer"]
                capture("accepted_during_fault", "generation_b")
                phase = "select_optional"
                page.locator('.nav-item[data-route="/app/compare"]').click()
                expect(page.locator("[data-compare-workspace]")).to_be_visible()
                with page.expect_response(
                    lambda r: (
                        r.url.endswith("/selection")
                        and r.request.post_data_json.get("selected_alternative_id") == optional["plan_id"]
                    )
                ) as selected:
                    page.locator(f'[data-compare-option="{optional["plan_id"]}"]').click()
                selected_payload = selected.value.json()
                expect(page.locator("#compare-evidence")).to_be_disabled()
                page.wait_for_function(
                    "() => {const maps=[...document.querySelectorAll('[data-compare-map-canvas]')];return maps.length===2&&maps.every(e=>e.dataset.mapReady==='true' && Number(e.dataset.mapTileCoverage)>=.95)}",
                    timeout=45_000,
                )
                phase = "evidence_unavailable"
                page.locator('.nav-item[data-route="/app/evidence"]').click()
                expect(page.locator("#workspace-content h2")).to_have_text("Evidence unavailable")
                expect(page.locator("#workspace-content code")).to_have_text("compare_option_evidence_unavailable")
                unavailable = {
                    "path": page.url,
                    "html": page.locator("#workspace-content").inner_html(),
                    "heading": page.locator("#workspace-content h2").inner_text(),
                    "code": page.locator("#workspace-content code").inner_text(),
                    "back_visible": page.locator("#evidence-back").is_visible(),
                    "back_disabled": page.locator("#evidence-back").is_disabled(),
                    "evidence_panel_count": page.locator("[data-evidence-plan-id]").count(),
                }
                phase = "evidence_back"
                page.locator("#evidence-back").click()
                expect(page).to_have_url(BASE_URL + "/app/compare")
                capture("compare_after_back", "generation_b")
                compare = page.locator("[data-compare-map-canvas]").evaluate_all(
                    "elements=>elements.map(e=>({...e.dataset}))"
                )
                expect(page.locator('[data-compare-map-pane="baseline"] h3')).to_have_text("Current trip route")
                expect(page.locator(".alternative-card").first.locator("h3")).to_have_text("Current accepted trip")
                compare_labels = {
                    "baseline_heading": page.locator('[data-compare-map-pane="baseline"] h3').inner_text(),
                    "baseline_card": page.locator(".alternative-card").first.locator("h3").inner_text(),
                    "baseline_plan_id": compare[0]["renderedMapPlanId"],
                    "original_plan_id": acceptance["result"]["pointer"]["original_plan_id"],
                }
                phase = "return_to_accepted"
                page.locator('.nav-item[data-route="/app/itinerary"]').click()
                capture("accepted_after_back", "generation_b")
                page.wait_for_load_state("networkidle")
                inputs = page.evaluate("window.__optionalEvidenceInputs")
            protected_after = protected_state(state)
            assert protected_before == protected_after and protected_before
            assert not errors and not failures
            assert all(item["status"] == 200 for item in http)
            source.write_bytes(saved_source)
            copied = json.loads((evidence / "private_fixture_inputs.json").read_text(encoding="utf-8"))
            _write_json(
                evidence / "private_restored_inputs.json",
                [
                    {
                        "path": row["path"],
                        "sha256": hashlib.sha256((repository / row["path"]).read_bytes()).hexdigest(),
                        "size_bytes": (repository / row["path"]).stat().st_size,
                    }
                    for row in copied["files"]
                ],
            )
            _write_json(evidence / "protected_state.json", {"before": protected_before, "after": protected_after})
            _write_json(
                evidence / "browser_assertions.json",
                {
                    "schema_version": "optional-evidence-fault-browser-v1",
                    "context": "accepted_child",
                    "coverage_category": "declared_backend_projection_failure_path",
                    "natural_interaction_claimed": False,
                    "assertions_passed": True,
                    "initial": before,
                    "faulted": after,
                    "acceptance": acceptance,
                    "proposal": proposal,
                    "selected": selected_payload,
                    "unavailable": unavailable,
                    "compare_after_back": compare,
                    "compare_labels": compare_labels,
                    "checkpoints": checkpoints,
                    "optional_plan_id": optional["plan_id"],
                    "private_source_restored_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
            )
            _write_json(evidence / "control_inputs.json", inputs)
            _write_json(evidence / "generation_bindings.json", generation_evidence_bindings(evidence, run_id))
            _write_json(
                evidence / "llm_io.json",
                {
                    "provider_call_count": 0,
                    "llm_input": None,
                    "llm_output": None,
                    "reason": "Direct-edit acceptance plus declared optional backend projection fault; no LLM invocation",
                },
            )
        finally:
            if saved_source is not None:
                (repository / RELATIVE).write_bytes(saved_source)
            _write_json(evidence / "http_index.json", http)
            _write_json(evidence / "browser_requests.json", requests)
            _write_json(evidence / "console_errors.json", errors)
            _write_json(evidence / "network_failures.json", failures)
            browser.close()
