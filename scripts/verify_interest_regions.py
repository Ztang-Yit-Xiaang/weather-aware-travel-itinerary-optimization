"""Observe the real local product; required-point rendering uses a labeled fixture."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from itinerary_system.product_app.interest_regions import required_interest_points  # noqa: E402

OUTPUT = ROOT / "results/interest_regions_20261020"
BASE = "http://127.0.0.1:8127"
LEG = "leg_stearns_wharf_surf_n_wear_s_beach_house"


def main():
    module = ast.parse((ROOT / "tests/product_app/browser/waypoint_ui_fixture.py").read_text())
    helpers = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign) and node.targets[0].id in {"OBSERVE_REAL_MAPS", "CURRENT_MAP"}
    }
    current_map = helpers["CURRENT_MAP"]
    receipt = {"schema_version": "interest-regions-browser-review-v1", "base_url": BASE, "checks": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel="chrome")
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        context.add_init_script(helpers["OBSERVE_REAL_MAPS"])
        page = context.new_page()
        responses, credentials, errors, writes = [], {}, [], []

        def record_response(response):
            if response.url.endswith("/api/sessions") and response.request.method == "POST":
                credentials.update(response.json())
            if "/map/interest-regions?" in response.url:
                assert response.status == 200, response.text()
                responses.append(response.json())

        page.on("response", record_response)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                writes.append(request.url.split(BASE)[-1])
                if request.method != "GET" and "/api/" in request.url
                else None
            ),
        )

        def map_state():
            return page.evaluate(
                "() => { const map = " + current_map + "; return {"
                "ready:document.querySelector('#geographic-map').dataset.mapReady,"
                "coverage:Number(document.querySelector('#geographic-map').dataset.mapTileCoverage),"
                "cameraBounds:map.getBounds().toArray(),"
                "layers:map.getStyle().layers.filter(l=>l.id.startsWith('interest')).map(l=>l.id),"
                "renderedPois:map.getLayer('interest-pois') ? map.queryRenderedFeatures({layers:['interest-pois']}).length : 0,"
                "boundary:map.getLayer('interest-region-boundary') ? map.getFilter('interest-region-boundary') : null,"
                "requiredColor:map.getLayer('interest-required') ? map.getPaintProperty('interest-required','circle-color') : null,"
                "requiredFeatures:map.getSource('interest-required') ? map.getStyle().sources['interest-required'].data.features : []}; }"
            )

        def wait_map(layer=True):
            page.wait_for_function(
                "() => {const map=" + current_map + ";return map && "
                "document.querySelector('#geographic-map')?.dataset.mapReady==='true' && "
                + ("!!map.getLayer('interest-pois')" if layer else "!map.getLayer('interest-pois')")
                + ";}",
                timeout=30000,
            )

        def open_region():
            page.locator("#map-edit-button").click()
            page.get_by_text("Text route description", exact=True).click()
            page.locator(f'[data-geo-route-leg="{LEG}"]').click()
            # Deliberately click immediately, before selection persistence can finish.
            page.locator("#explore-interest-regions").click()
            page.locator("#interest-place-selector").wait_for(timeout=30000)
            wait_map()

        def session():
            sid = credentials["session"]["session_id"]
            response = context.request.get(
                f"{BASE}/api/sessions/{sid}", headers={"X-Session-Token": credentials["mutation_token"]}
            )
            assert response.status == 200
            return response.json()

        def screenshot(name):
            page.screenshot(path=str(OUTPUT / f"{name}.png"), full_page=True)

        page.goto(BASE + "/app", wait_until="domcontentloaded")
        page.locator('[data-day="4"]').first.wait_for()
        page.locator('[data-route="/app/map"]').first.click()
        page.locator('[data-day="4"]').first.click()
        open_region()
        before = session()
        assert len(responses[-1]["poi_predictions"]) == 95
        assert map_state()["renderedPois"] > 10
        assert map_state()["coverage"] >= 0.95
        receipt["checks"].append({"name": "real_complete_catalog_and_immediate_click", "map": map_state()})
        screenshot("interest_desktop_map")
        page.locator("#geographic-map").scroll_into_view_if_needed()
        click_point = page.evaluate(
            "() => {const map=" + current_map + "; const rect=map.getContainer().getBoundingClientRect();"
            "const occupied=['v2-stops-active','v2-stops-original','v2-route-active','v2-route-original'].filter(id=>map.getLayer(id));"
            "for(const feature of map.queryRenderedFeatures({layers:['interest-pois']})){"
            "const point=map.project(feature.geometry.coordinates),x=rect.x+point.x,y=rect.y+point.y;"
            "if(document.elementFromPoint(x,y)===map.getCanvas() && !map.queryRenderedFeatures(point,{layers:occupied}).length)"
            "return {x,y,placeId:feature.properties.place_id};} return null;}"
        )
        assert click_point is not None
        page.mouse.click(click_point["x"], click_point["y"])
        expect(page.locator("#interest-place-selector")).to_have_value(click_point["placeId"])
        wait_map()
        receipt["checks"].append({"name": "real_pointer_selects_the_same_poi_in_the_list"})
        count = len(responses)
        page.locator("#interest-mass-level").select_option("0.95")
        wait_map()
        assert map_state()["boundary"][-1] == 0.95 and len(responses) == count
        receipt["checks"].append({"name": "mass_switch_is_local", "new_probability_requests": 0})
        probabilities = {}
        for preference in ("1", "-1"):
            with page.expect_response(lambda response: "/map/interest-regions?" in response.url):
                page.locator("#interest-preference").select_option(preference)
            page.locator("#interest-place-selector").wait_for()
            wait_map()
            probabilities[preference] = [
                point["interest_probability_mean"] for point in responses[-1]["poi_predictions"]
            ]
        delta = max(abs(a - b) for a, b in zip(probabilities["1"], probabilities["-1"], strict=True))
        assert delta > 0.2
        receipt["checks"].append(
            {"name": "stated_preference_changes_probability", "maximum_probability_difference": delta}
        )
        selected = responses[-1]["poi_predictions"][0]
        page.locator("#interest-place-selector").select_option(selected["place_id"])
        expect(page.locator(".interest-place-detail")).to_contain_text(selected["name"])
        expect(page.locator(".interest-place-detail")).to_contain_text("90% model uncertainty")
        page.locator(".interest-region-panel").scroll_into_view_if_needed()
        screenshot("interest_desktop_details")
        page.locator("#explore-interest-place").click()
        expect(page.locator("#exploratory-meaning")).to_be_visible()
        after = session()
        assert before["session"]["revision"] == after["session"]["revision"]
        assert before["session"]["accepted_plan_id"] == after["session"]["accepted_plan_id"]
        assert before["session"]["draft"] == after["session"]["draft"] == []
        receipt["checks"].append(
            {"name": "point_details_and_existing_check_entry_without_draft", "accepted_plan_unchanged": True}
        )
        page.locator('[data-day="5"]').first.click()
        expect(page.locator(".interest-region-panel")).to_have_count(0)
        wait_map(False)
        page.reload()
        page.locator('[data-day="4"]').first.wait_for()
        wait_map(False)
        receipt["checks"].append({"name": "day_switch_and_reload_remove_old_layer"})

        # The source run has no must-go ownership. Exercise canonical mapping and
        # the real renderer without changing frozen plans or claiming live user data.
        geography = after["workspace"]["geography"]
        parent_id = after["session"]["accepted_plan_id"]
        owned = [
            {
                "constraint_id": "render_fixture_" + target,
                "origin": "user",
                "strength": "hard",
                "scope": "stop",
                "target_id": target,
                "relation": "must_keep",
                "value": True,
                "confirmed": True,
                "relaxation_policy": "explicit_only",
                "evidence_refs": [],
                "schema_version": "owned-constraint-v1",
            }
            for target in ("stearns_wharf", selected["place_id"], "fixture_missing_stop")
        ]
        fixture_catalog = SimpleNamespace(
            get_place=lambda target: (
                SimpleNamespace(
                    name=selected["name"], display_coordinate=SimpleNamespace(**selected["display_coordinate"])
                )
                if target == selected["place_id"]
                else None
            )
        )
        required = required_interest_points(
            {"plan_id": parent_id, "owned_constraints": owned}, geography, 4, fixture_catalog
        )
        for point in required:
            point["interest_probability_mean"] = 0.2

        def required_fixture(route):
            response = route.fetch()
            payload = response.json()
            payload["required_points"] = required
            route.fulfill(response=response, json=payload)

        page.route("**/map/interest-regions?*", required_fixture)
        page.locator('[data-day="4"]').first.click()
        open_region()
        assert map_state()["requiredColor"] == "#082f49", json.dumps(map_state())
        assert len(map_state()["requiredFeatures"]) == 2
        assert {point["properties"]["conflict_status"] for point in map_state()["requiredFeatures"]} == {
            "satisfied",
            "required_stop_missing",
        }
        page.get_by_text("Confirmed must-go points (3)", exact=True).click()
        expect(page.locator(".interest-region-panel")).to_contain_text("Required Stop Missing")
        receipt["checks"].append(
            {
                "name": "canonical_required_render_fixture",
                "evidence_mode": "synthetic_constraints_only",
                "missing_coordinate_not_invented": True,
                "color": map_state()["requiredColor"],
                "model_probability": 0.2,
            }
        )
        screenshot("interest_required_fixture_map")
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator(".interest-region-panel").scroll_into_view_if_needed()
        wait_map()
        dimensions = page.locator(".interest-region-panel").evaluate(
            "element => ({width:element.clientWidth,scrollWidth:element.scrollWidth})"
        )
        assert dimensions["scrollWidth"] <= dimensions["width"]
        screenshot("interest_mobile_details")
        page.locator("#close-interest-regions").click()
        wait_map(False)
        receipt["checks"].append({"name": "mobile_no_panel_overflow_and_hide_removes_layers", **dimensions})
        assert not errors, errors
        receipt.update(
            {
                "passed": True,
                "page_errors": errors,
                "observed_probability_responses": len(responses),
                "real_user_feedback_count": 0,
                "write_endpoints": sorted(set(writes)),
            }
        )
        browser.close()
    (OUTPUT / "browser_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
