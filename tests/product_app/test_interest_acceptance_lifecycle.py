"""Opt-in integration against a freshly built, real-road coherent demo package."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from owned_product_fixture import ROOT, RUN_ID, build_owned_product_registry

from itinerary_system.product_app.api import create_product_app


def test_required_interest_points_follow_preview_acceptance_and_reload(tmp_path):
    source = os.environ.get("INTEREST_COHERENT_PACKAGE")
    if not source:
        pytest.skip("Set INTEREST_COHERENT_PACKAGE to an independently evaluated coherent package")
    # The normal config pins an additive route delta to the unavailable old
    # package. This independent real-road fixture uses its own verified matrix.
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "src", repository / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", repository / "configs", ignore=shutil.ignore_patterns("product_route_delta.json"))
    registry = build_owned_product_registry(
        repository,
        source_package=Path(source),
        existence_target="stearns_wharf",
        repository_root=repository,
    )
    state_root = tmp_path / "state"
    evidence = Path(os.environ.get("INTEREST_EVIDENCE_ROOT", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    exchanges, authentication = [], {}

    def client():
        return TestClient(
            create_product_app(
                repository_root=repository,
                registry_path=registry,
                state_root=state_root,
                additional_allowed_authorities=("testserver",),
            )
        )

    def call(api, method, path, body=None, params=None):
        response = api.request(method, path, json=body, params=params, headers=authentication)
        assert response.status_code == 200, response.text
        result = response.json()
        exchanges.append(
            {
                "method": method,
                "path": path,
                "input": body,
                "query": params,
                "status": response.status_code,
                "output": {
                    key: value
                    for key, value in result.items()
                    if key
                    in {"session", "proposal", "required_points", "accepted_plan_id", "accepted_plan_content_hash"}
                },
            }
        )
        return result

    def explore(api, path, payload):
        plan = next(
            plan
            for plan in payload["workspace"]["geography"]["plans"]
            if plan["plan_id"] == payload["session"]["accepted_plan_id"]
        )
        feature = next(
            leg for leg in plan["validated_legs"]["features"] if leg["properties"]["origin_id"] == "stearns_wharf"
        )
        properties = feature["properties"]
        result = call(
            api,
            "GET",
            path + "/map/interest-regions",
            params={
                "expected_revision": payload["session"]["revision"],
                "day": properties["day"],
                "route_leg_id": properties["route_leg_id"],
            },
        )
        assert result["accepted_plan_id"] == plan["plan_id"]
        assert result["accepted_plan_content_hash"] == plan["content_hash"]
        assert len(result["required_points"]) == 1
        required = result["required_points"][0]
        assert required["place_id"] == "stearns_wharf"
        assert required["constraint_ids"] == ["owned_interest_existence"]
        assert required["conflict_status"] == "satisfied"
        assert required["coordinate"] is not None
        assert required["interest_probability_mean"] != 1
        return result

    with client() as api:
        created = call(api, "POST", "/api/sessions", {"run_id": RUN_ID})
        authentication["X-Session-Token"] = created["mutation_token"]
        path = f"/api/sessions/{created['session']['session_id']}"
        before = explore(api, path, created)
        staged = call(
            api,
            "POST",
            path + "/draft/operations",
            {
                "expected_revision": created["session"]["revision"],
                "source": "typed_stop_editor",
                "type": "replace_nearby",
                "target": "bixby_creek_bridge_viewpoint",
                "parameters": {"candidate_id": "pfeiffer_big_sur_state_park"},
            },
        )
        preview = call(api, "POST", path + "/preview", {"expected_revision": staged["session"]["revision"]})
        assert preview["proposal"]["state"] == "eligible"
        assert preview["session"]["accepted_plan_id"] == before["accepted_plan_id"]
        child = preview["proposal"]["runtime_execution"]["child_plan"]
        assert any(rule["constraint_id"] == "owned_interest_existence" for rule in child["owned_constraints"])
        call(
            api,
            "POST",
            path + "/accept",
            {
                "expected_revision": preview["session"]["revision"],
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "interest-required-acceptance-0001",
            },
        )
        accepted = call(api, "GET", path)
        after = explore(api, path, accepted)
        assert after["accepted_plan_id"] == child["plan_id"] != before["accepted_plan_id"]
        assert after["required_points"] == before["required_points"]
    with client() as api:
        reloaded = call(api, "GET", path)
        assert explore(api, path, reloaded)["required_points"] == after["required_points"]
    (evidence / "accepted_required_lifecycle.json").write_text(
        json.dumps(
            {
                "evidence_mode": "synthetic_confirmed_input_constraints_real_route_preview_acceptance",
                "response_injection": False,
                "passed": True,
                "source_package": str(Path(source).resolve()),
                "review_registry": str(registry),
                "repository_root": str(repository),
                "state_root": str(state_root),
                "exchanges": exchanges,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_accepted_required_point_is_rendered_and_selectable_without_response_injection():
    base = os.environ.get("INTEREST_REQUIRED_LIVE_URL")
    if not base:
        pytest.skip("Set INTEREST_REQUIRED_LIVE_URL to the isolated acceptance review server")
    from playwright.sync_api import expect, sync_playwright

    sys.path.insert(0, str(ROOT / "tests/product_app/browser"))
    from waypoint_ui_fixture import CURRENT_MAP, OBSERVE_REAL_MAPS, point_pixels, wait_map

    evidence = Path(os.environ["INTEREST_EVIDENCE_ROOT"])
    errors, created, predictions = [], {}, []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.add_init_script(OBSERVE_REAL_MAPS)

        def record(response):
            if response.url.endswith("/api/sessions") and response.request.method == "POST":
                created.update(response.json())
            if "/map/interest-regions?" in response.url:
                assert response.status == 200, response.text()
                predictions.append(response.json())

        page.on("response", record)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base + "/app/map", wait_until="domcontentloaded")
        page.locator("[data-day]").first.wait_for(timeout=30000)
        accepted = next(
            plan
            for plan in created["workspace"]["geography"]["plans"]
            if plan["plan_id"] == created["session"]["accepted_plan_id"]
        )
        route = next(
            leg["properties"]
            for leg in accepted["validated_legs"]["features"]
            if leg["properties"]["origin_id"] == "stearns_wharf"
        )

        def open_region():
            page.locator(f'[data-day="{route["day"]}"]').first.click()
            page.locator("#map-edit-button").click()
            page.get_by_text("Text route description", exact=True).click()
            page.locator(f'[data-geo-route-leg="{route["route_leg_id"]}"]').click()
            page.locator("#explore-interest-regions").click()
            page.locator("#interest-place-selector").wait_for(timeout=30000)
            wait_map(page)

        open_region()
        assert len(predictions[-1]["required_points"]) == 1
        required = predictions[-1]["required_points"][0]
        assert required["constraint_ids"] == ["owned_interest_existence"]
        assert predictions[-1]["accepted_plan_content_hash"] == accepted["content_hash"]
        drawn = page.evaluate(
            "() => {const map=" + CURRENT_MAP + ";return {"
            "color:map.getPaintProperty('interest-required','circle-color'),"
            "rendered:map.queryRenderedFeatures({layers:['interest-required']}).length,"
            "source:map.getStyle().sources['interest-required'].data};}"
        )
        assert drawn["color"] == "#082f49" and drawn["rendered"] >= 1
        page.locator("#geographic-map").scroll_into_view_if_needed()
        pixel = point_pixels(page, required["coordinate"])
        page.mouse.click(pixel["x"], pixel["y"])
        expect(page.locator(".interest-place-detail h3")).to_have_text("Must-go: Stearns Wharf")
        wait_map(page)
        page.screenshot(path=str(evidence / "accepted_required_actual_map.png"), full_page=True)
        page.locator(".interest-region-panel").scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / "accepted_required_actual_details.png"), full_page=True)
        page.reload()
        page.locator("[data-day]").first.wait_for(timeout=30000)
        open_region()
        assert predictions[-1]["required_points"] == [required]
        assert not errors, errors
        (evidence / "accepted_required_browser.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "response_injection": False,
                    "evidence_mode": "actual_browser_of_independent_input_constraint_fixture_after_real_acceptance",
                    "accepted_plan_id": predictions[-1]["accepted_plan_id"],
                    "accepted_plan_content_hash": predictions[-1]["accepted_plan_content_hash"],
                    "required_point": required,
                    "drawn": drawn,
                    "reload_preserves_required_point": True,
                    "page_errors": errors,
                    "optional_point_count": len(predictions[-1]["poi_predictions"]),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        browser.close()
