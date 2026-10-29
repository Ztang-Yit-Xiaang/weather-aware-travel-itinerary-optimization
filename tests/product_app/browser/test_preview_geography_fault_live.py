"""Declared private-backend geography faults after real repair and evaluation."""
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
    _credential_safe_body,
    _http_record,
    _map_ack,
    _write_json,
    expect,
    sync_playwright,
)
from test_optional_evidence_fault_live import protected_state

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
RELATIVE = "src/itinerary_system/product_app/api.py"
ANCHOR = "            return await asyncio.to_thread(publish_result)"


def fault_source(before, kind):
    assert kind in {"missing", "foreign"} and before.count(ANCHOR) == 1
    replacement = '''            canonical = await asyncio.to_thread(publish_result)
            # Declared disposable response-only fault AFTER real publication.
            from pathlib import Path as FaultPath
            wire = json.loads(json.dumps(canonical, allow_nan=False))
            if canonical["proposal"].get("state") != "eligible":
                raise RuntimeError("geometry_fault_requires_real_eligible_child")
            geometry = FAULT_GEOMETRY
            wire["proposal"]["geography_plan"] = geometry
            wire["session"]["proposal"]["geography_plan"] = geometry
            FaultPath(__file__).resolve().parents[3].joinpath("geography_fault_boundary.json").write_text(
                json.dumps({"canonical": canonical, "wire": wire}, ensure_ascii=False, allow_nan=False, sort_keys=True),
                encoding="utf-8")
            return wire'''
    geometry = "None" if kind == "missing" else (
        'next(plan for plan in service.workspace_view_for_session(session)["geography"]["plans"] '
        'if plan["plan_id"] == canonical["proposal"]["parent_plan_id"])')
    return before.replace(ANCHOR, replacement.replace("FAULT_GEOMETRY", geometry))


def retain_backend_evidence(repository, state, session_id, evidence):
    """Retain available failure evidence too, without exporting authentication verifiers."""
    boundary_path = repository / "geography_fault_boundary.json"
    paths = [("backend_boundary_bytes.json", boundary_path)]
    if session_id is not None:
        paths.append(("persisted_session.json", state / "sessions" / (session_id + ".json")))
    for name, path in paths:
        if not path.is_file():
            continue
        original = path.read_bytes()
        raw = original.decode("utf-8")
        redactions = []
        if name == "persisted_session.json":
            projected = json.loads(raw)
            assert "token_verifier" in projected
            projected["token_verifier"] = "OMITTED_SECRET"
            raw = json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            redactions = ["/token_verifier"]
        safe, unexpected = _credential_safe_body(raw)
        assert unexpected == redactions and safe == raw, "Unexpected credential in backend state evidence"
        _write_json(evidence / name, {"utf8": safe, "sha256": hashlib.sha256(safe.encode()).hexdigest(),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "body_capture": "credential_redacted" if redactions else "exact_utf8",
            "credential_redactions": redactions, "plaintext_credentials_persisted": False})
        if name == "backend_boundary_bytes.json":
            _write_json(evidence / "backend_boundary.json", json.loads(safe))


@pytest.mark.skipif(not BASE_URL or os.environ.get("PRODUCT_E2E_PREVIEW_GEOGRAPHY_FAULT_DISPOSABLE") != "1",
                    reason="Requires disposable private-copy geography-fault lifecycles")
@pytest.mark.parametrize("kind", ["missing", "foreign"])
def test_real_eligible_child_with_bad_geography_is_blocked_without_parent_or_draft_loss(tmp_path, kind):
    root = Path(__file__).resolve().parents[3]
    evidence = Path(os.environ.get("PRODUCT_E2E_PREVIEW_GEOGRAPHY_FAULT_EVIDENCE_DIR", tmp_path)) / kind
    evidence.mkdir(parents=True, exist_ok=True)
    run_id = os.environ["PRODUCT_E2E_RUN_ID"] + "_" + kind
    http, requests, errors, failures, checkpoints = [], [], [], [], {}
    observed, request_rows = {}, {}
    phase = "generation_a"
    saved_source = None
    staged = None
    with tempfile.TemporaryDirectory(prefix="g", dir=root / "tmp", delete=False) as temporary, sync_playwright() as runtime:
        repository, state = Path(temporary) / "repo", Path(temporary) / "state"
        copy_product_inputs(repository, evidence)
        browser = runtime.chromium.launch(headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {}))
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("""window.__geometryInputs=[];
          document.addEventListener('click',event=>{
            const e=event.target.closest?.('button');if(!e)return;
            window.__geometryInputs.push({id:e.id,day:e.dataset.day??null,trusted:event.isTrusted,at:Date.now(),
              disabled:e.matches(':disabled'),visible:!!e.getClientRects().length});
          },true);""")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda event: errors.append(event.text) if event.type == "error" else None)

        def now():
            return datetime.now(UTC).isoformat()

        def requested(request):
            row = {"sequence": len(requests) + 1, "phase": phase, "method": request.method, "url": request.url,
                "request_body_sha256": hashlib.sha256((request.post_data or "").encode()).hexdigest(),
                "started_at": now(), "response_at": None, "finished_at": None, "failed_at": None,
                "status": None, "failure": None, "http_file": None}
            requests.append(row)
            request_rows[request] = row

        def failed(request):
            request_rows[request].update(failed_at=now(), failure=request.failure)
            failures.append(dict(request_rows[request]))

        def response_record(response):
            raw = request_rows[response.request]
            raw.update(response_at=now(), status=response.status)
            if "/api/" not in response.url:
                return
            record = {"phase": raw["phase"], "request_sequence": raw["sequence"], "observed_at": now(), **_http_record(response)}
            filename = f"http_{len(http) + 1:03d}.json"
            raw["http_file"] = filename
            _write_json(evidence / filename, record)
            http.append({"file": filename, "phase": raw["phase"], "method": record["method"], "url": record["url"], "status": response.status})
            if response.status == 200:
                payload = json.loads(record["response_body_utf8"])
                if isinstance(payload, dict):
                    observed.update({key: payload[key] for key in ("session", "workspace") if key in payload})

        page.on("request", requested)
        page.on("requestfinished", lambda request: request_rows[request].update(finished_at=now()))
        page.on("requestfailed", failed)
        page.on("response", response_record)

        def capture(label, generation):
            with patch.dict(os.environ, {"PRODUCT_E2E_RUN_ID": run_id + "_" + generation}):
                _capture_ready(page, evidence / generation / (label + ".png"))
            checkpoints[label] = _map_ack(page)

        try:
            with guarded_generation(repository, state, evidence, BASE_URL, run_id, "generation_a"):
                page.goto(BASE_URL + "/app/itinerary", wait_until="domcontentloaded")
                capture("initial_parent", "generation_a")
                initial = json.loads(json.dumps(observed))
                with page.expect_response(lambda r: r.url.endswith("/draft/impact-preview")):
                    page.locator('.day-card[data-day="5"]').click()
                expect(page.locator("#map-context-title")).to_contain_text("Bixby")
                page.locator("#map-edit-button").click()
                page.locator("#inspect-edit-stop").click()
                page.locator(".legacy-edit-controls summary").click()
                with page.expect_response(lambda r: r.url.endswith("/draft/impact-preview")):
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                with page.expect_response(lambda r: r.url.endswith("/draft/operations")) as staged_response:
                    page.locator("#draft-replace").click()
                assert staged_response.value.status == 200
                staged = staged_response.value.json()["session"]
                assert len(staged["draft"]) == 1 and staged["proposal"] is None
                page.locator("#close-map-edit").click()
                expect(page.locator("#inline-preview")).to_be_enabled()
                capture("staged_parent", "generation_a")
                _write_json(evidence / "staging_inputs.json", page.evaluate("window.__geometryInputs"))
                page.wait_for_load_state("networkidle")
            protected_before = protected_state(state)
            source = repository / RELATIVE
            saved_source = source.read_bytes()
            fault = fault_source(saved_source.decode(), kind).encode()
            source.write_bytes(fault)
            _write_json(evidence / "fault_patch.json", {"schema_version": "preview-geography-response-fault-v1", "kind": kind,
                "path": RELATIVE, "before_utf8": saved_source.decode(), "after_utf8": fault.decode(),
                "before_sha256": hashlib.sha256(saved_source).hexdigest(), "after_sha256": hashlib.sha256(fault).hexdigest(),
                "production_source_modified": False, "frontend_fetch_mocking": False, "optimizer_mocking": False,
                "evaluator_mocking": False, "canonical_artifacts_mutated": False})
            phase = "generation_b"
            with guarded_generation(repository, state, evidence, BASE_URL, run_id, "generation_b"):
                with page.expect_response(lambda r: r.url.endswith("/api/sessions/" + staged["session_id"]) and r.request.method == "GET") as restored:
                    page.reload(wait_until="domcontentloaded")
                restoration = restored.value.json()
                assert restoration["session"] == staged
                assert restoration["workspace"]["accepted_plan_pointer"] == initial["workspace"]["accepted_plan_pointer"]
                capture("restored_staged_parent", "generation_b")
                expect(page.locator("#inline-preview")).to_be_enabled()
                with page.expect_response(lambda r: r.url.endswith("/preview"), timeout=90_000) as preview:
                    page.locator("#inline-preview").click()
                assert preview.value.status == 200
                wire = preview.value.json()
                assert wire["proposal"]["state"] == "eligible"
                expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", "PREVIEW_FAILED", timeout=40_000)
                expect(page.locator("#toast-region")).to_contain_text("The evaluated repair response did not match")
                expect(page.locator("#accept-repair")).to_be_disabled()
                expect(page.locator(".draft-bar strong")).to_contain_text("1 draft")
                expect(page.locator("#inline-undo")).to_be_enabled()
                capture("blocked_parent", "generation_b")
                ui = page.evaluate("""()=>({at:new Date().toISOString(),map:{...document.querySelector('.map-stage').dataset},
                    metric:{...document.querySelector('#metric-strip').dataset},
                    draft_text:document.querySelector('.draft-bar strong').textContent,
                    accept_disabled:document.querySelector('#accept-repair').disabled,
                    undo_disabled:document.querySelector('#inline-undo').disabled,
                    toast:document.querySelector('#toast-region').textContent,
                    timeline:document.querySelector('#day-list').textContent})""")
                inputs = page.evaluate("window.__geometryInputs")
                page.wait_for_load_state("networkidle")
            boundary_path = repository / "geography_fault_boundary.json"
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
            assert boundary["wire"] == wire
            _write_json(evidence / "protected_state.json", {"before": protected_before, "after": protected_state(state)})
            source.write_bytes(saved_source)
            copied = json.loads((evidence / "private_fixture_inputs.json").read_text(encoding="utf-8"))
            _write_json(evidence / "private_restored_inputs.json", [{"path": row["path"],
                "sha256": hashlib.sha256((repository / row["path"]).read_bytes()).hexdigest(),
                "size_bytes": (repository / row["path"]).stat().st_size} for row in copied["files"]])
            _write_json(evidence / "browser_assertions.json", {"schema_version": "preview-geography-fault-browser-v1",
                "kind": kind, "coverage_category": "declared_backend_response_failure_path", "assertions_passed": True,
                "initial": initial, "staged": staged, "restoration": restoration, "wire": wire,
                "checkpoints": checkpoints, "ui": ui, "source_restored_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            _write_json(evidence / "control_inputs.json", inputs)
            _write_json(evidence / "generation_bindings.json", generation_evidence_bindings(evidence, run_id))
            _write_json(evidence / "llm_io.json", {"provider_call_count": 0, "llm_input": None, "llm_output": None,
                "reason": "Real deterministic repair and independent evaluation; only private response geography is faulted."})
        finally:
            retain_backend_evidence(repository, state, staged["session_id"] if staged else None, evidence)
            if saved_source is not None:
                (repository / RELATIVE).write_bytes(saved_source)
            _write_json(evidence / "http_index.json", http)
            _write_json(evidence / "browser_requests.json", requests)
            _write_json(evidence / "console_errors.json", errors)
            _write_json(evidence / "network_failures.json", failures)
            browser.close()
