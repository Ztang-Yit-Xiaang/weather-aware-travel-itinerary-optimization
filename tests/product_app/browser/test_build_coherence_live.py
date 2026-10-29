"""Actual loaded page -> server restart -> blocked write -> explicit safe reload.

This test owns both guarded server lifecycles. The runner must not launch another
server on its port. Only a private registry label changes; production source,
canonical trip artifacts, permissions and provider boundaries are not mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

pytest.importorskip("playwright.sync_api")
from build_coherence_fixture import (
    change_registry_label,
    copy_product_inputs,
    file_state,
    generation_evidence_bindings,
    guarded_generation,
)
from test_e2e1_live_copilot_chat import (
    CHROME,
    _capture_ready,
    _http_record,
    _map_ack,
    _preserve_browser_evidence,
    _write_json,
    expect,
    sync_playwright,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_BUILD_COHERENCE_DISPOSABLE") == "1"
SESSION_KEY = "itinerary-repair-copilot.session.v1"


def session_identity(page):
    return page.evaluate(
        """async key => {
      const raw = sessionStorage.getItem(key);
      const parsed = JSON.parse(raw || 'null');
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw || ''));
      return {session_id:parsed?.session_id, stored_credentials_sha256:[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('')};
    }""",
        SESSION_KEY,
    )


@pytest.mark.skipif(
    not BASE_URL or not DISPOSABLE, reason="Requires dedicated build-coherence mode with an unused disposable port"
)
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_cached_page_requires_reload_and_preserves_active_plan_across_real_restart(tmp_path, plan_context):
    evidence = Path(os.environ.get("PRODUCT_E2E_BUILD_COHERENCE_EVIDENCE_DIR", tmp_path)) / plan_context
    evidence.mkdir(parents=True, exist_ok=True)
    run_id = os.environ["PRODUCT_E2E_RUN_ID"] + "_" + plan_context
    http_index, requests, checkpoints, page_errors = [], [], {}, []
    console_errors, request_failures = [], []
    phase = "generation_a"
    private_base = Path(__file__).resolve().parents[3] / "tmp"
    private_base.mkdir(exist_ok=True)
    # Pytest redirects the default temp root and expands names to 32 hex digits.
    # Keep the explicit root/prefix short enough for the copied routing paths.
    with (
        tempfile.TemporaryDirectory(prefix="b", dir=private_base, delete=False) as temporary,
        sync_playwright() as runtime,
    ):
        private = Path(temporary)
        repository, durable_state = private / "repo", private / "state"
        copy_product_inputs(repository, evidence)
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("""document.addEventListener('click', event => {
          if (!event.target.closest?.('#reload-current-build')) return;
          sessionStorage.setItem('e2e.build.reload', JSON.stringify({
            selector:'#reload-current-build', trusted:event.isTrusted, clicked_at:new Date().toISOString(),
            title:document.querySelector('#loading-screen')?.innerText,
            client_build:document.querySelector('meta[name="product-build-id"]')?.content,
            path:location.pathname}));
        }, true);""")
        page = context.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda item: console_errors.append(item.text) if item.type == "error" else None)
        page.on("requestfailed", lambda item: request_failures.append({"url": item.url, "failure": item.failure}))
        page.on(
            "request",
            lambda request: requests.append(
                {
                    "phase": phase,
                    "method": request.method,
                    "url": request.url,
                    "build_header": request.headers.get("x-product-build-id"),
                }
            ),
        )

        def record(response):
            if "/api/" not in response.url:
                return
            row = {"phase": phase, **_http_record(response)}
            name = f"http_{len(http_index) + 1:03d}.json"
            _write_json(evidence / name, row)
            http_index.append(
                {"file": name, "phase": phase, "method": row["method"], "status": response.status, "url": response.url}
            )

        page.on("response", record)

        def capture(name):
            generation = "generation_a" if phase == "generation_a" else "generation_b"
            with patch.dict(os.environ, {"PRODUCT_E2E_RUN_ID": f"{run_id}_{generation}"}):
                _capture_ready(page, evidence / generation / f"{name}.png")
            checkpoints[name] = _map_ack(page)
            return checkpoints[name]

        def recovery(code, title):
            expect(page.locator("#loading-screen")).to_contain_text(title)
            expect(page.locator("#reload-current-build")).to_be_visible()
            expect(page.locator("#app-shell")).to_be_hidden()
            # No map is visible in a recovery screen: preserve its DOM, not a pre-ready screenshot.
            _write_json(
                evidence / f"{code}_visible_recovery.json",
                {
                    "code": code,
                    "text": page.locator("#loading-screen").inner_text(),
                    "html": page.locator("#loading-screen").inner_html(),
                    "stored_session": session_identity(page),
                    "captured_at": datetime.now(UTC).isoformat(),
                },
            )

        def open_duration():
            page.locator('.day-card[data-day="7"]').click()
            expect(page.locator("#map-context-title")).to_contain_text("Griffith")
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            page.locator("#typed-edit-surface summary").filter(has_text="Duration and time window").click()
            form = page.locator('[data-typed-operation="set_stop_duration"]')
            form.locator('[name="duration-preferred"]').fill("60")
            expect(form.locator('button[type="submit"]')).to_be_enabled()
            # Day selection also mounts a new map behind the editor. Do not
            # terminate or deliberately invalidate its server mid-request.
            page.wait_for_function(
                """() => document.querySelector('#geographic-map')?.dataset.mapReady === 'true'""", timeout=30_000
            )
            page.wait_for_load_state("networkidle", timeout=20_000)
            return form

        with _preserve_browser_evidence(
            context, page, evidence, {"page_errors": page_errors, "request_failures": request_failures}
        ):
            try:
                with guarded_generation(repository, durable_state, evidence, BASE_URL, run_id, "generation_a") as first:
                    page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                    parent = capture("initial_parent")
                    client_build = page.locator('meta[name="product-build-id"]').get_attribute("content")
                    assert client_build == first["build_id"]
                    identity = session_identity(page)
                    proposal, decision, accepted_ack = None, None, parent
                    if plan_context == "accepted_child":
                        page.locator('.day-card[data-day="5"]').click()
                        expect(page.locator("#map-context-title")).to_contain_text("Bixby Creek Bridge")
                        page.locator("#map-edit-button").click()
                        page.locator("#inspect-edit-stop").click()
                        page.locator(".legacy-edit-controls summary").click()
                        page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                        with page.expect_response(
                            lambda response: response.url.endswith("/draft/operations")
                        ) as staged:
                            page.locator("#draft-replace").click()
                        assert staged.value.status == 200
                        page.locator("#close-map-edit").click()
                        with page.expect_response(
                            lambda response: response.url.endswith("/preview"), timeout=90_000
                        ) as previewed:
                            page.locator("#inline-preview").click()
                        assert previewed.value.status == 200, previewed.value.text()
                        proposal = previewed.value.json()["proposal"]
                        assert proposal["state"] == "eligible"
                        assert proposal["parent_plan_id"] == parent["active_plan_id"]
                        expect(page.locator(".map-stage")).to_have_attribute(
                            "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
                        )
                        preview_ack = capture("evaluated_child")
                        assert preview_ack["active_plan_id"] == proposal["child_plan_id"]
                        assert preview_ack["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                        with page.expect_response(
                            lambda response: response.url.endswith("/accept"), timeout=90_000
                        ) as accepted:
                            page.locator("#accept-repair").click()
                            expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
                            expect(page.locator("#acceptance-message")).to_have_text(
                                "Accept this independently evaluated repair as the durable trip plan?"
                            )
                            page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
                                "button", name="Accept repair", exact=True
                            ).click()
                        assert accepted.value.status == 200, accepted.value.text()
                        decision = accepted.value.json()["result"]["decision"]
                        expect(page.locator(".map-stage")).to_have_attribute(
                            "data-preview-state", "PARENT_RENDERED", timeout=30_000
                        )
                        accepted_ack = capture("accepted_child")
                        assert accepted_ack["active_plan_id"] == proposal["child_plan_id"]
                        assert "pfeiffer_big_sur_state_park" in accepted_ack["rendered_marker_ids"]
                        assert "bixby_creek_bridge_viewpoint" not in accepted_ack["rendered_marker_ids"]
                    stale_form = open_duration()
                    assert session_identity(page) == identity
                change_registry_label(repository, evidence, "b")
                phase = "generation_b_old_page"
                with guarded_generation(
                    repository, durable_state, evidence, BASE_URL, run_id, "generation_b"
                ) as second:
                    assert second["build_id"] != client_build
                    assert page.locator('meta[name="product-build-id"]').get_attribute("content") == client_build
                    before = file_state(durable_state)
                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as blocked:
                        stale_form.locator('button[type="submit"]').click()
                    assert blocked.value.status == 409
                    assert blocked.value.json() == {"detail": "product_client_build_mismatch"}
                    assert blocked.value.request.headers["x-product-build-id"] == client_build
                    after = file_state(durable_state)
                    _write_json(evidence / "stale_write_state_hashes.json", {"before": before, "after": after})
                    assert after == before, "Blocked stale write must not alter durable state"
                    recovery("product_client_build_mismatch", "Reload to use the updated app")
                    assert session_identity(page) == identity
                    phase = "generation_b_reloaded"
                    with page.expect_response(
                        lambda response: (
                            response.url.endswith(f"/api/sessions/{identity['session_id']}")
                            and response.request.method == "GET"
                        )
                    ) as restored:
                        page.locator("#reload-current-build").click()
                    assert restored.value.status == 200
                    restored_payload = restored.value.json()
                    assert restored_payload["session"]["accepted_plan_id"] == accepted_ack["active_plan_id"]
                    assert restored_payload["session"]["draft"] == []
                    pointer = restored_payload["workspace"]["accepted_plan_pointer"]
                    assert pointer.get("decision_id") == (decision["decision_id"] if decision else None)
                    after_reload = capture("active_plan_after_update")
                    assert after_reload == accepted_ack
                    reload_input = json.loads(page.evaluate("sessionStorage.getItem('e2e.build.reload')"))
                    assert reload_input["trusted"] is True and reload_input["client_build"] == client_build
                    _write_json(evidence / "reload_control_input.json", reload_input)
                    assert session_identity(page) == identity
                    assert page.locator('meta[name="product-build-id"]').get_attribute("content") == second["build_id"]
                    current_form = open_duration()
                    phase = "generation_b_source_drift"
                    before = file_state(durable_state)
                    before_registry = change_registry_label(repository, evidence, "drift")
                    with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as drifted:
                        current_form.locator('button[type="submit"]').click()
                    assert drifted.value.status == 503
                    assert drifted.value.json() == {"detail": "product_server_build_changed"}
                    after = file_state(durable_state)
                    _write_json(evidence / "source_drift_write_state_hashes.json", {"before": before, "after": after})
                    assert after == before
                    recovery("product_server_build_changed", "Restart the local app, then reload")
                    assert session_identity(page) == identity
                    registry = repository / "configs/product_app_registry.json"
                    drift_registry = registry.read_bytes()
                    registry.write_bytes(before_registry)
                    _write_json(
                        evidence / "registry_restore.json",
                        {
                            "path": "configs/product_app_registry.json",
                            "process_restarted": False,
                            "before_utf8": drift_registry.decode(),
                            "after_utf8": registry.read_text(encoding="utf-8"),
                            "before_sha256": hashlib.sha256(drift_registry).hexdigest(),
                            "after_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                            "expected_generation_b_sha256": hashlib.sha256(before_registry).hexdigest(),
                        },
                    )
                    assert registry.read_bytes() == before_registry
                    try:
                        with urlopen(f"{BASE_URL}/api/health", timeout=5):
                            raise AssertionError("Restoring files must not unlatch the running server")
                    except HTTPError as failure:
                        assert failure.code == 503
                        body = failure.read().decode()
                        assert json.loads(body) == {"detail": "product_server_build_changed"}
                        _write_json(
                            evidence / "restored_source_latched_health.json",
                            {"status": failure.code, "body_utf8": body},
                        )
                assert page_errors == []
                assert request_failures == []
                assert all(
                    "server responded with a status of" in message
                    and any(f" {code} " in message for code in (409, 503))
                    for message in console_errors
                ), console_errors
                assert (
                    len([row for row in requests if row["method"] == "POST" and row["url"].endswith("/api/sessions")])
                    == 1
                )
                assert not any("/copilot/messages" in row["url"] for row in requests)
                errors = [row for row in http_index if row["status"] >= 400]
                assert [row["status"] for row in errors] == [409, 503]
                receipts = [
                    json.loads((evidence / name / "provider_isolation.json").read_text(encoding="utf-8"))
                    for name in ("generation_a", "generation_b")
                ]
                provider_calls = sum(row["sent_provider_calls"] for row in receipts)
                assert provider_calls == 0
                generation_bindings = generation_evidence_bindings(evidence, run_id)
                _write_json(evidence / "generation_bindings.json", generation_bindings)
                assert sum(row["provider_call_count"] for row in generation_bindings) == provider_calls
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": provider_calls,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "No chat submitted; both real server generations finalized their independent provider isolation receipts.",
                    },
                )
                _write_json(
                    evidence / "browser_assertions.json",
                    {
                        "schema_version": "build-coherence-browser-v1",
                        "assertions_passed": True,
                        "first_build_id": first["build_id"],
                        "second_build_id": second["build_id"],
                        "context": plan_context,
                        "accepted_plan_id": accepted_ack["active_plan_id"],
                        "accepted_plan_content_hash": accepted_ack["active_plan_content_hash"],
                        "certificate_id": proposal["certificate_id"] if proposal else None,
                        "decision_id": decision["decision_id"] if decision else None,
                        "session": identity,
                        "blocked_mutation_state_unchanged": True,
                        "explicit_reload_session_preserved": True,
                        "source_drift_latched": True,
                        "provider_call_count": provider_calls,
                        "provider_receipts": [
                            "generation_a/provider_isolation.json",
                            "generation_b/provider_isolation.json",
                        ],
                    },
                )
            finally:
                _write_json(evidence / "http_index.json", http_index)
                _write_json(evidence / "request_index.json", requests)
                _write_json(evidence / "map_checkpoints.json", checkpoints)
                _write_json(
                    evidence / "browser_errors.json",
                    {
                        "page_errors": page_errors,
                        "console_errors": console_errors,
                        "request_failures": request_failures,
                    },
                )
        browser.close()
