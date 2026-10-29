"""Real typed variants: evaluated preview, canonical rendering, reload, then Undo.

These two ordered cases use one fresh disposable server. They deliberately claim
evaluated-preview-only coverage, not acceptance of every variant. The second
case creates its accepted baseline with actual Replace/Preview/Accept controls.
Original-plan edits must remain ineligible while the weather disruption persists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
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
from waypoint_ui_fixture import (
    CURRENT_MAP,
    OBSERVE_REAL_MAPS,
    browser_network_record,
    classify_browser_network,
    map_cancellation_policy,
)

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
VARIANTS = [
    *(("role", value, "stearns_wharf", 6) for value in ("attraction", "activity", "meal", "rest_stop", "scenic_stop")),
    *(("time_window", value, "stearns_wharf", 6) for value in ("earliest_only", "latest_only", "both")),
    ("duration", "exact", "griffith_observatory", 7),
    ("order", "default", "sequoia_national_park", 4),
]


def _identity(ack):
    return {key: value for key, value in ack.items() if key != "preview_state"}


def _artifact_hash(record):
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[
        :16
    ]


@pytest.mark.skipif(
    not BASE_URL or os.environ.get("PRODUCT_E2E_EVALUATED_VARIANTS_DISPOSABLE") != "1",
    reason="Requires a fresh deterministic evaluated-variants server; no user-state attachment",
)
@pytest.mark.parametrize("plan_context", ["original", "accepted_child"])
def test_real_evaluated_variants_preview_reload_and_undo(tmp_path, plan_context):
    sys.path.insert(0, str(ROOT / "scripts"))
    from e2e1_interaction_coverage import required_variants

    evidence = Path(os.environ.get("PRODUCT_E2E_EVALUATED_VARIANTS_EVIDENCE_DIR", tmp_path)) / plan_context
    evidence.mkdir(parents=True, exist_ok=False)
    http, assertions, inventory, errors, failures, requests = [], [], [], [], [], []
    observed = {}
    control_actions = []
    phase = "initial"
    map_configuration = None
    map_configuration_http_file = None
    network_policy = None
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        from control_action_fixture import (
            OBSERVE_CONTROL_ACTIONS,
            after_action,
            before_action,
            fill_native_text,
            select_native_option,
        )

        context.add_init_script(OBSERVE_CONTROL_ACTIONS)
        context.add_init_script("""document.addEventListener('click', event => {
          const control=event.target.closest?.('button');
          if(control) (window.__evaluatedControlClicks ||= []).push({id:control.id,
            trusted:event.isTrusted,path:location.pathname,at:Date.now()});
        },true);""")
        page = context.new_page()
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failures.append(browser_network_record(request, phase)))
        page.on("request", lambda request: requests.append({"method": request.method, "url": request.url}))

        def record(response):
            nonlocal map_configuration, map_configuration_http_file
            if not urlsplit(response.url).path.startswith("/api/"):
                return
            captured_phase = phase
            row = {"phase": captured_phase, **_http_record(response)}
            # Body reads may re-enter this callback. Allocate only after reading.
            name = f"http_{len(http) + 1:03d}.json"
            _write_json(evidence / name, row)
            http.append(
                {
                    "file": name,
                    "phase": captured_phase,
                    "method": row["method"],
                    "url": row["url"],
                    "status": row["response_status_code"],
                }
            )
            if row["response_status_code"] == 200:
                payload = json.loads(row["response_body_utf8"])
                if isinstance(payload, dict):
                    if response.url == f"{BASE_URL}/api/map/config":
                        assert map_configuration is None or payload == map_configuration, (
                            "Map configuration changed during the journey"
                        )
                        map_configuration, map_configuration_http_file = payload, name
                    for key in ("workspace", "session"):
                        if key in payload:
                            observed[key] = payload[key]
                    if row["url"].endswith("/draft/impact-preview"):
                        observed["impact"] = payload

        page.on("response", record)

        def expand(details):
            if details.get_attribute("open") is None:
                details.locator(":scope > summary").click()

        def open_stop(day):
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            page.locator(f'.day-card[data-day="{day}"]').click()
            page.locator("#map-edit-button").click()
            page.locator("#inspect-edit-stop").click()
            expect(page.locator('[data-editor-kind="stop"]')).to_be_visible()

        def stage(kind, target, parameters, control):
            expect(control).to_be_enabled()
            input_receipt = before_action(page, control)
            with page.expect_response(lambda response: response.url.endswith("/draft/operations")) as completed:
                control.click()
            assert completed.value.status == 200, completed.value.text()
            request, payload = completed.value.request.post_data_json, completed.value.json()
            for _ in range(300):
                if observed.get("impact", {}).get("session_revision") == payload["session"]["revision"]:
                    break
                page.wait_for_timeout(50)
            assert observed.get("impact", {}).get("session_revision") == payload["session"]["revision"]
            expected = {"type": kind, "target": target, "parameters": parameters}
            assert {key: request[key] for key in expected} == expected
            assert request["source"] == "map"
            assert {key: payload["operation"][key] for key in expected} == expected
            assert payload["session"]["draft"] == [payload["operation"]]
            assert payload["session"]["accepted_plan_id"] == parent["active_plan_id"]
            expect(page.locator(".draft-bar strong")).to_have_text("1 draft change", timeout=15_000)
            control_actions.append(
                after_action(
                    page,
                    input_receipt,
                    context="original" if phase == "baseline_replace_accept" else plan_context,
                    phase=phase,
                    request=request,
                    response=payload,
                )
            )
            _write_json(evidence / "control_action_receipts.json", control_actions)
            return request, payload

        def alias_input(selector):
            control = page.locator(selector)
            expect(control).to_be_visible()
            expect(control).to_be_enabled(timeout=15_000)
            return {"selector": selector, "visible": True, "disabled": False, "path": urlsplit(page.url).path}

        def alias_result(value):
            value["raw_clicks"] = page.evaluate("window.__evaluatedControlClicks || []")
            value["event"] = value["raw_clicks"][-1]
            assert value["event"]["trusted"] is True and value["event"]["id"] == value["selector"][1:]
            return value

        def show_alias(selector):
            if selector.startswith("#context-"):
                page.locator('.nav-item[data-route="/app/repairs"]').click()
            elif selector.startswith("#draft-"):
                page.locator("#map-edit-button").click()
                page.locator("#inspect-edit-stop").click()
                expect(page.locator("#map-edit-dialog")).to_be_visible()
                expand(page.locator(".legacy-edit-controls"))

        def leave_alias():
            if page.locator("#map-edit-dialog").is_visible():
                page.locator("#close-map-edit").click()
            if urlsplit(page.url).path != "/app/itinerary":
                page.locator('.nav-item[data-route="/app/itinerary"]').click()

        def preview(selector="#inline-preview"):
            show_alias(selector)
            control = alias_input(selector)
            started_at = datetime.now(UTC).isoformat()
            with page.expect_response(lambda response: response.url.endswith("/preview"), timeout=90_000) as completed:
                page.locator(selector).click()
            control = alias_result(control)
            assert completed.value.status == 200, completed.value.text()
            leave_alias()
            return completed.value.json(), {
                "request_body": completed.value.request.post_data_json,
                "started_at": started_at,
                "received_at": datetime.now(UTC).isoformat(),
                "control": control,
            }

        def assert_surface_identity(ack, geography):
            expect(page.locator("#metric-strip")).to_have_attribute("data-plan-id", ack["active_plan_id"])
            expect(page.locator("#metric-strip")).to_have_attribute(
                "data-plan-content-hash", ack["active_plan_content_hash"]
            )
            details = page.locator(".textual-map-alternative")
            expand(details)
            text = page.locator(".text-route-summary")
            expect(text).to_be_visible()
            expect(text).to_have_attribute("data-plan-id", ack["active_plan_id"])
            expect(text).to_have_attribute("data-plan-content-hash", ack["active_plan_content_hash"])
            by_day = {}
            for feature in geography["stops"]["features"]:
                stop = feature["properties"]
                by_day.setdefault(int(stop["day"]), []).append(stop)
            timeline = []
            for day in range(1, 8):
                stops = sorted(by_day.get(day, []), key=lambda stop: stop["stop_order"])
                card = page.locator(f'.day-card[data-day="{day}"]')
                names = " · ".join(stop["name"] for stop in stops) or "No recorded stop"
                expect(card.locator(".day-copy strong")).to_have_text(names)
                expect(card).to_have_attribute("data-stop", stops[0]["stop_id"] if stops else "")
                timeline.append({"day": day, "stop_ids": [stop["stop_id"] for stop in stops], "visible_names": names})
            return {
                "metrics": page.locator("#metric-strip").evaluate("e => ({...e.dataset})"),
                "timeline": timeline,
                "timeline_text": page.locator("#day-list").inner_text(),
                "text_route": {"scope": text.evaluate("e => ({...e.dataset})"), "text": text.inner_text()},
            }

        with _preserve_browser_evidence(
            context, page, evidence, {"console_errors": errors, "request_failures": failures}
        ):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                parent = _map_ack(page)
                assert "bixby_creek_bridge_viewpoint" in parent["rendered_marker_ids"], (
                    "Requires unchanged fresh original baseline"
                )
                _capture_ready(page, evidence / "initial.png")
                if plan_context == "accepted_child":
                    phase = "baseline_replace_accept"
                    open_stop(5)
                    expand(page.locator(".legacy-edit-controls"))
                    page.locator("#registered-candidate").select_option("pfeiffer_big_sur_state_park")
                    stage(
                        "replace_nearby",
                        "bixby_creek_bridge_viewpoint",
                        {"candidate_id": "pfeiffer_big_sur_state_park"},
                        page.locator("#draft-replace"),
                    )
                    page.locator("#close-map-edit").click()
                    baseline, _ = preview()
                    proposal = baseline["proposal"]
                    assert proposal["state"] == "eligible"
                    expect(page.locator(".map-stage")).to_have_attribute(
                        "data-preview-state", "PREVIEW_RENDERED", timeout=40_000
                    )
                    rendered = _map_ack(page)
                    assert rendered["active_plan_id"] == rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
                    assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                    assert rendered["rendered_geometry_hash"] == _independent_geometry_hash(
                        page, proposal["geography_plan"]
                    )
                    assert "pfeiffer_big_sur_state_park" in rendered["rendered_marker_ids"]
                    assert "bixby_creek_bridge_viewpoint" not in rendered["rendered_marker_ids"]
                    _capture_ready(page, evidence / "baseline_preview.png")
                    expect(page.locator("#accept-repair")).to_be_enabled()
                    with page.expect_response(
                        lambda response: response.url.endswith("/accept"), timeout=60_000
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
                    acceptance = accepted.value.json()
                    assert acceptance["session"]["accepted_plan_id"] == proposal["child_plan_id"]
                    assert acceptance["session"]["proposal"] is None
                    assert acceptance["result"]["decision"]["selected_plan_id"] == proposal["child_plan_id"]
                    assert (
                        acceptance["result"]["decision"]["certificate_content_hash"]
                        == proposal["certificate_content_hash"]
                    )
                    _write_json(evidence / "baseline_preview_response.json", baseline)
                    _write_json(evidence / "baseline_acceptance_response.json", acceptance)
                    page.reload(wait_until="domcontentloaded")
                    expect(page.locator(".map-stage")).to_have_attribute(
                        "data-preview-state", "PARENT_RENDERED", timeout=40_000
                    )
                    parent = _map_ack(page)
                    assert _identity(parent) == _identity(rendered)
                _capture_ready(page, evidence / "baseline.png")
                workspace = observed["workspace"]
                parent_geography = next(
                    plan for plan in workspace["geography"]["plans"] if plan["plan_id"] == parent["active_plan_id"]
                )
                capabilities = {key: workspace[key] for key in ("draft_capabilities", "typed_edit_capabilities")}
                typed = capabilities["typed_edit_capabilities"]
                required = {
                    operation: required_variants(operation, capability, typed)
                    for operation, capability in typed["operations"].items()
                }
                covered_families = {"set_stop_" + row[0] for row in VARIANTS}
                expected_variants = {
                    (operation, row["id"])
                    for operation in covered_families
                    for row in required[operation]
                    if row["feedback_tier"] == "evaluated"
                }
                assert expected_variants == {("set_stop_" + kind, variant) for kind, variant, _, _ in VARIANTS}
                _write_json(
                    evidence / "capability_snapshot.json",
                    {
                        "schema_version": "runtime-capability-snapshot-v1",
                        "context": plan_context,
                        "session_id": observed["session"]["session_id"],
                        "plan_id": parent["active_plan_id"],
                        "plan_content_hash": parent["active_plan_content_hash"],
                        "capabilities": capabilities,
                        "required_variants": {"typed_edit_capabilities": required},
                    },
                )
                baseline_surface = assert_surface_identity(parent, parent_geography)
                acceptance_count = sum(row["method"] == "POST" and row["url"].endswith("/accept") for row in requests)
                latest_bound = None
                latest_bound_source = None

                for variant_index, (kind, variant, target, day) in enumerate(VARIANTS):
                    phase = f"{kind}_{variant}"
                    case = evidence / phase
                    case.mkdir()
                    open_stop(day)
                    operation_type = "set_stop_" + kind
                    form = page.locator(f'[data-typed-operation="{operation_type}"]')
                    expand(form.locator("xpath=ancestor::details[1]"))
                    parent_stop = next(
                        feature["properties"]
                        for feature in parent_geography["stops"]["features"]
                        if feature["properties"]["stop_id"] == target
                    )
                    if kind == "role":
                        assert parent_stop.get("itinerary_role") != variant, "Do not claim an unchanged role as an edit"
                        assert form.locator('[name="role"]').input_value() == (parent_stop.get("itinerary_role") or "")
                        select_native_option(form.locator('[name="role"]'), variant)
                        parameters = {"role": variant}
                    elif kind == "duration":
                        select_native_option(form.locator('[name="duration_mode"]'), "exact")
                        form.locator('[name="duration-preferred"]').fill("60")
                        parameters = {
                            "duration": {
                                "mode": "exact",
                                "preferred_minutes": 60,
                                "minimum_minutes": 60,
                                "maximum_minutes": 60,
                            }
                        }
                    elif kind == "time_window":
                        earliest = "10:00" if variant != "latest_only" else None
                        if variant != "earliest_only":
                            assert latest_bound is not None, (
                                "Derive a feasible latest bound from the real preceding evaluator trace"
                            )
                        latest = latest_bound if variant != "earliest_only" else None
                        fill_native_text(form.locator('[name="earliest_arrival"]'), earliest or "")
                        fill_native_text(form.locator('[name="latest_departure"]'), latest or "")
                        parameters = {"earliest_arrival": earliest, "latest_departure": latest}
                    else:
                        select_native_option(form.locator('[name="sequence_position"]'), "2")
                        parameters = {"day": day, "sequence_index": 1}
                    expect(form).to_be_visible()
                    expect(form.locator('button[type="submit"]')).to_be_enabled()
                    inventory.append(
                        {
                            "context": plan_context,
                            "operation": operation_type,
                            "variant": variant,
                            "selector": f'[data-typed-operation="{operation_type}"] button[type="submit"]',
                            "visible": True,
                            "disabled": False,
                            "form_text": form.inner_text(),
                            "fields": form.evaluate(
                                "e => [...e.elements].map(n => ({name:n.name,value:n.value,disabled:n.disabled}))"
                            ),
                        }
                    )
                    _write_json(
                        evidence / "runtime_control_inventory.json",
                        {
                            "schema_version": "runtime-control-inventory-v1",
                            "context": plan_context,
                            "observations": inventory,
                        },
                    )
                    operation_input, staged = stage(
                        operation_type, target, parameters, form.locator('button[type="submit"]')
                    )
                    _write_json(case / "operation_input.json", operation_input)
                    _write_json(
                        case / "edit_preconditions.json",
                        {
                            "parent_stop": parent_stop,
                            "latest_bound_source": latest_bound_source if kind == "time_window" and latest else None,
                        },
                    )
                    _write_json(case / "stage_response.json", staged)
                    impact = observed["impact"]
                    assert impact["session_revision"] == staged["session"]["revision"]
                    assert impact["parent_plan_id"] == parent["active_plan_id"]
                    assert impact["parent_plan_content_hash"] == parent["active_plan_content_hash"]
                    assert impact["is_certified"] is False
                    assert impact["summary"]["evaluated_executable_count"] == 1
                    assert impact["summary"]["draft_only_count"] == 0
                    assert impact["summary"]["can_run_evaluated_preview"] is True
                    assert impact["operations"][0]["operation_id"] == staged["operation"]["operation_id"]
                    _write_json(case / "draft_impact_response.json", impact)
                    page.locator("#close-map-edit").click()
                    staged_ack = _map_ack(page)
                    assert _identity(staged_ack) == _identity(parent)
                    _capture_ready(page, case / "staged_parent.png")
                    payload, timing = preview(
                        ("#inline-preview", "#draft-preview", "#context-preview")[variant_index % 3]
                    )
                    _write_json(case / "preview_response.json", payload)
                    _write_json(case / "preview_timing.json", timing)
                    proposal = payload["proposal"]
                    expected_outcome = "ineligible" if plan_context == "original" else "eligible"
                    assert proposal["state"] == expected_outcome, proposal.get("reason")
                    assert proposal["eligibility"] == expected_outcome
                    assert payload["session"]["accepted_plan_id"] == parent["active_plan_id"]
                    assert payload["session"]["draft"] == [staged["operation"]]
                    assert proposal["parent_plan_id"] == parent["active_plan_id"]
                    assert proposal["parent_plan_content_hash"] == parent["active_plan_content_hash"]
                    assert proposal["child_plan_id"] != parent["active_plan_id"]
                    assert proposal["child_plan_content_hash"] != parent["active_plan_content_hash"]
                    assert proposal["compiled_request"]["operations"] == [
                        {key: value for key, value in staged["operation"].items() if key != "created_at"}
                    ]
                    certificate = (
                        proposal.get("evaluation_certificate")
                        or proposal.get("certificate_schedule_evidence", {}).get("certificate_record")
                        or proposal.get("certificate_role_evidence", {}).get("certificate_record")
                    )
                    assert certificate is not None
                    assert certificate["plan_id"] == proposal["child_plan_id"]
                    assert certificate["plan_content_hash"] == proposal["child_plan_content_hash"]
                    assert certificate["comparison_eligibility"] == expected_outcome
                    assert certificate["certificate_id"] == proposal["certificate_id"]
                    assert (
                        _artifact_hash(certificate)
                        == certificate["content_hash"]
                        == proposal["certificate_content_hash"]
                    )
                    diff = proposal["plan_diff"]
                    assert _artifact_hash(diff) == proposal["diff_content_hash"]
                    assert diff["diff_id"] == proposal["diff_id"]
                    assert diff["parent_plan_id"] == parent["active_plan_id"]
                    assert diff["child_plan_id"] == proposal["child_plan_id"]
                    geography = proposal["geography_plan"]
                    assert geography["plan_id"] == proposal["child_plan_id"]
                    assert geography["content_hash"] == proposal["child_plan_content_hash"]
                    assert geography["coverage"]["complete"] is True and geography["coverage"]["gap_count"] == 0
                    route = certificate["route_validation"]
                    assert route["required_leg_count"] > 0
                    assert route["road_validated_leg_count"] == route["required_leg_count"]
                    assert route["fallback_leg_count"] == route["missing_leg_count"] == 0
                    changed = next(
                        feature["properties"]
                        for feature in geography["stops"]["features"]
                        if feature["properties"]["stop_id"] == target
                    )
                    if kind == "role":
                        assert changed["itinerary_role"] == variant
                        changes = diff["role_changes"]
                        assert len(changes) == 1 and changes[0]["stop_id"] == target
                        assert changes[0]["from_role"] == parent_stop.get("itinerary_role")
                        assert changes[0]["from_role"] != changes[0]["to_role"]
                        assert changes[0]["to_role"] == variant
                        assert proposal["role_impact"]["semantic_scope"]["place_identity_unchanged"] is True
                        assert proposal["role_impact"]["semantic_scope"]["semantic_fit_claim"] == "unavailable"
                    elif kind == "duration":
                        assert changed["visit_duration_minutes"] == 60
                        assert len(diff["duration_changes"]) == 1
                        assert diff["duration_changes"][0]["stop_id"] == target
                        assert diff["duration_changes"][0]["to_minutes"] == 60
                    elif kind == "time_window":
                        changes = diff["time_window_changes"]
                        assert len(changes) == 1 and changes[0]["stop_id"] == target
                        assert changes[0]["to_earliest_arrival"] == parameters["earliest_arrival"]
                        assert changes[0]["to_latest_departure"] == parameters["latest_departure"]
                        assert (changes[0]["from_earliest_arrival"], changes[0]["from_latest_departure"]) != (
                            earliest,
                            latest,
                        )
                        trace = next(
                            row
                            for row in proposal["schedule_impact"]["child"]["stop_traces"]
                            if row["stop_id"] == target
                        )
                        if earliest:
                            assert trace["service_start_minute"] >= 600
                        if latest:
                            latest_minute = int(latest[:2]) * 60 + int(latest[3:])
                            assert trace["departure_minute"] <= latest_minute
                            assert trace["latest_departure_status"] == "satisfied"
                        if variant == "earliest_only":
                            # Thirty minutes beyond the next half-hour boundary is
                            # feasible for this exact observed schedule, not a fixed
                            # guessed departure time or a change to place opening hours.
                            bound_minute = math.ceil(trace["departure_minute"] / 30) * 30 + 30
                            assert 0 < bound_minute < 1440
                            latest_bound = f"{bound_minute // 60:02d}:{bound_minute % 60:02d}"
                            latest_bound_source = {
                                "case": phase,
                                "child_plan_id": proposal["child_plan_id"],
                                "evaluation_id": proposal["certificate_id"],
                                "trace": trace,
                                "chosen_latest_departure": latest_bound,
                            }
                    else:
                        assert changed["stop_order"] == 2
                    _write_json(case / "evaluation_summary.json", certificate)
                    _write_json(case / "plan_diff.json", diff)
                    state = "PREVIEW_INELIGIBLE" if plan_context == "original" else "PREVIEW_RENDERED"
                    expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", state, timeout=40_000)
                    rendered = _map_ack(page)
                    expected_geometry_hash = _independent_geometry_hash(page, geography)
                    if plan_context == "original":
                        assert [finding["code"] for finding in certificate["failures"]] == ["weather_infeasible"], (
                            "The registered weather issue must be the blocker, not an unrelated failure hidden as expected ineligibility"
                        )
                        assert proposal["acceptance_eligible"] is False
                        assert _identity(rendered) == _identity(parent)
                        expect(page.locator("#accept-repair")).to_be_disabled()
                        expect(page.locator("#repair-panel")).to_contain_text("No eligible draft repair")
                        expect(page.locator("#repair-panel")).to_contain_text("The accepted itinerary remains active")
                        assert (
                            page.locator("#metric-strip").get_attribute("data-evaluation-id")
                            == baseline_surface["metrics"]["evaluationId"]
                        )
                    else:
                        assert certificate["failures"] == []
                        assert (
                            rendered["active_plan_id"] == rendered["rendered_map_plan_id"] == proposal["child_plan_id"]
                        )
                        assert rendered["active_plan_content_hash"] == proposal["child_plan_content_hash"]
                        assert rendered["rendered_geometry_hash"] == expected_geometry_hash
                        assert set(rendered["rendered_marker_ids"]) == set(parent["rendered_marker_ids"])
                        expect(page.locator("#metric-strip")).to_have_attribute(
                            "data-evaluation-id", proposal["certificate_id"]
                        )
                        expect(page.locator("#accept-repair")).to_be_enabled()
                    displayed_geography = parent_geography if plan_context == "original" else geography
                    surfaces = assert_surface_identity(rendered, displayed_geography)
                    advanced = page.locator("#repair-panel .evaluated-edit-evidence")
                    expand(advanced)
                    expect(advanced).to_contain_text(proposal["child_plan_id"])
                    expect(advanced).to_contain_text(proposal["certificate_id"])
                    expect(advanced).to_contain_text(proposal["diff_id"])
                    _write_json(
                        case / "visible_surfaces.json", {**surfaces, "advanced_evidence": advanced.inner_text()}
                    )
                    _capture_ready(page, case / "preview_result.png")
                    if plan_context == "accepted_child":
                        page.locator('[data-route="/app/compare"]').first.click()
                        compare = page.locator('[data-compare-map-pane="option"] [data-compare-map-canvas]')
                        expect(compare).to_have_attribute(
                            "data-rendered-map-plan-id", proposal["child_plan_id"], timeout=30_000
                        )
                        _capture_ready(page, case / "compare.png")
                        page.locator("#compare-evidence").click()
                        view = page.locator("[data-evidence-plan-id]")
                        expect(view).to_have_attribute("data-evidence-plan-id", proposal["child_plan_id"])
                        expect(view).to_have_attribute("data-plan-content-hash", proposal["child_plan_content_hash"])
                        expect(view).to_have_attribute("data-evaluation-id", proposal["certificate_id"])
                        _write_json(
                            case / "evidence_panel.json",
                            {"scope": view.evaluate("e => ({...e.dataset})"), "text": view.inner_text()},
                        )
                        page.locator('[data-route="/app/itinerary"]').first.click()
                    page.reload(wait_until="domcontentloaded")
                    expect(page.locator(".map-stage")).to_have_attribute("data-preview-state", state, timeout=40_000)
                    reloaded = _map_ack(page)
                    assert reloaded == rendered
                    assert observed["session"]["accepted_plan_id"] == parent["active_plan_id"]
                    assert observed["session"]["proposal"]["child_plan_id"] == proposal["child_plan_id"]
                    assert observed["session"]["draft"] == [staged["operation"]]
                    assert_surface_identity(reloaded, displayed_geography)
                    _capture_ready(page, case / "after_reload.png")
                    # Once a child preview is rendered, preview markers are
                    # deliberately not editable in place. Undo remains available
                    # from the workspace and Repairs surfaces; the collapsed
                    # typed-editor shortcut is only reachable while the accepted
                    # parent is the active editing context.
                    undo_selector = ("#inline-undo", "#context-undo")[variant_index % 2]
                    show_alias(undo_selector)
                    undo_control = alias_input(undo_selector)
                    with page.expect_response(lambda response: response.url.endswith("/draft/undo")) as undone:
                        page.locator(undo_selector).click()
                    undo_control = alias_result(undo_control)
                    leave_alias()
                    assert undone.value.status == 200, undone.value.text()
                    restored = undone.value.json()
                    assert restored["undone"] == staged["operation"]
                    assert restored["session"]["draft"] == [] and restored["session"]["proposal"] is None
                    assert restored["session"]["accepted_plan_id"] == parent["active_plan_id"]
                    expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
                    assert _identity(_map_ack(page)) == _identity(parent)
                    _capture_ready(page, case / "undo_restored_parent.png")
                    assert (
                        sum(row["method"] == "POST" and row["url"].endswith("/accept") for row in requests)
                        == acceptance_count
                    )
                    result = {
                        "schema_version": "evaluated-variant-browser-v1",
                        "context": plan_context,
                        "operation": operation_type,
                        "variant": variant,
                        "feedback_tier": "evaluated",
                        "scope": "evaluated_preview_only_then_undo",
                        "variant_acceptance_claimed": False,
                        "operation_input": operation_input,
                        "operation_id": staged["operation"]["operation_id"],
                        "request_id": proposal["compiled_request"]["source_request_id"],
                        "parent_plan_id": parent["active_plan_id"],
                        "parent_plan_content_hash": parent["active_plan_content_hash"],
                        "child_plan_id": proposal["child_plan_id"],
                        "child_plan_content_hash": proposal["child_plan_content_hash"],
                        "evaluation_id": proposal["certificate_id"],
                        "comparison_eligibility": expected_outcome,
                        "failure_codes": [finding["code"] for finding in certificate["failures"]],
                        "child_geometry_hash": expected_geometry_hash,
                        "parent": parent,
                        "staged": staged_ack,
                        "rendered": rendered,
                        "reloaded": reloaded,
                        "restored": _map_ack(page),
                        "assertions_passed": True,
                        "completed_at": datetime.now(UTC).isoformat(),
                    }
                    _write_json(case / "undo_response.json", restored)
                    _write_json(
                        case / "action_aliases.json",
                        {
                            "schema_version": "evaluated-action-aliases-v1",
                            "context": plan_context,
                            "preview": timing["control"],
                            "undo": undo_control,
                            "request_id": proposal["compiled_request"]["source_request_id"],
                            "operation_id": staged["operation"]["operation_id"],
                            "assertions_passed": True,
                        },
                    )
                    _write_json(case / "trace.json", result)
                    _write_json(case / "map_render_ack.json", rendered)
                    assertions.append(result)
                    _write_json(evidence / "variant_assertions.json", assertions)
                assert len(assertions) == len(VARIANTS)
                phase = "final_network_observation"
                asset_sources = page.evaluate(
                    """() => {
                  const map = """
                    + CURRENT_MAP
                    + """;
                  const style = map.getStyle();
                  return {container_id:map.getContainer().id,map_dataset:{...map.getContainer().dataset},
                    sprite:style.sprite??null,
                    vector_sources:Object.fromEntries(Object.entries(style.sources)
                      .filter(([,source])=>source.type==='vector'))};
                }"""
                )
                assert asset_sources["container_id"] == "geographic-map"
                assert asset_sources["map_dataset"]["mapReady"] == "true"
                assert float(asset_sources["map_dataset"]["mapTileCoverage"]) >= 0.95
                assert asset_sources["map_dataset"]["renderedMapPlanId"] == parent["active_plan_id"]
                assert asset_sources["map_dataset"]["renderedPlanContentHash"] == parent["active_plan_content_hash"]
                assert asset_sources["map_dataset"]["renderedGeometryHash"] == parent["rendered_geometry_hash"]
                network_policy = map_cancellation_policy(
                    map_configuration, {"sources": asset_sources["vector_sources"], "sprite": asset_sources["sprite"]}
                )
                _write_json(
                    evidence / "map_asset_sources.json",
                    {
                        "schema_version": "real-map-asset-sources-v1",
                        "scope": "Read-only public getStyle() vector sources and actual ready container dataset; no map-disposal cause inferred",
                        "phase": phase,
                        "context": plan_context,
                        "map_configuration_http_file": map_configuration_http_file,
                        **asset_sources,
                    },
                )
                classification = classify_browser_network(failures, network_policy)
                assert errors == [] and classification["blocking_failure_indexes"] == []
                assert not any(row["method"] == "POST" and "/copilot/messages" in row["url"] for row in requests)
                assert len({row["file"] for row in http}) == len(http)
                _write_json(
                    evidence / "browser_assertions.json",
                    {
                        "schema_version": "evaluated-variants-browser-v1",
                        "context": plan_context,
                        "scope": "evaluated_preview_only_then_undo",
                        "variant_acceptance_claimed": False,
                        "baseline_acceptance_through_real_ui": plan_context == "accepted_child",
                        "variants": assertions,
                        "assertions_passed": True,
                        "console_errors": errors,
                        "network_failures": failures,
                        "network_classification": classification,
                    },
                )
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "Actual typed controls and canonical optimizer/evaluator; no chat or provider call.",
                    },
                )
            finally:
                _write_json(evidence / "http_index.json", http)
                _write_json(evidence / "console_errors.json", errors)
                _write_json(evidence / "network_failures.json", failures)
                if network_policy is not None:
                    _write_json(
                        evidence / "network_classification.json",
                        {
                            "map_configuration_http_file": map_configuration_http_file,
                            "rendered_map_asset_sources_file": "map_asset_sources.json",
                            **classify_browser_network(failures, network_policy),
                        },
                    )
        browser.close()
