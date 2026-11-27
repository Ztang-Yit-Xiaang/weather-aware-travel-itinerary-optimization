"""Four-size real UI journeys; no API/optimizer responses mocked, no chat sent."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from waypoint_ui_fixture import CURRENT_MAP, OBSERVE_REAL_MAPS

pytest.importorskip("playwright.sync_api")
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

VISIBILITY = r"""selectors => {
  const rect = r => ({left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height});
  return {at:new Date().toISOString(),viewport:{width:innerWidth,height:innerHeight},
    scroll_owners:[document.scrollingElement,...document.querySelectorAll('.workspace-content,.textual-map-alternative,.map-context-card,.metric-strip')]
      .filter(Boolean).map(e=>({tag:e.tagName,class_name:e.className,scroll_top:e.scrollTop,scroll_left:e.scrollLeft,
        client_height:e.clientHeight,scroll_height:e.scrollHeight,rect:rect(e.getBoundingClientRect())})),
    elements:selectors.map(selector=>{
      const e=document.querySelector(selector); if(!e)return {selector,present:false};
      const r=e.getBoundingClientRect(); let clip={left:Math.max(0,r.left),top:Math.max(0,r.top),
        right:Math.min(innerWidth,r.right),bottom:Math.min(innerHeight,r.bottom)};
      const ancestors=[];
      for(let parent=e.parentElement;parent;parent=parent.parentElement){
        const style=getComputedStyle(parent),p=parent.getBoundingClientRect();
        if(style.overflowX!=='visible'){clip.left=Math.max(clip.left,p.left+parent.clientLeft);clip.right=Math.min(clip.right,p.left+parent.clientLeft+parent.clientWidth);}
        if(style.overflowY!=='visible'){clip.top=Math.max(clip.top,p.top+parent.clientTop);clip.bottom=Math.min(clip.bottom,p.top+parent.clientTop+parent.clientHeight);}
        ancestors.push({tag:parent.tagName,class_name:parent.className,overflow_x:style.overflowX,overflow_y:style.overflowY,
          scroll_top:parent.scrollTop,scroll_left:parent.scrollLeft,rect:rect(p),
          client_left:parent.clientLeft,client_top:parent.clientTop,client_width:parent.clientWidth,client_height:parent.clientHeight,
          display:style.display,visibility:style.visibility,opacity:Number(style.opacity)});
      }
      const area=Math.max(0,clip.right-clip.left)*Math.max(0,clip.bottom-clip.top);
      const points=[[.5,.5],[.1,.1],[.9,.1],[.1,.9],[.9,.9]].map(([dx,dy])=>{
        const x=r.left+r.width*dx,y=r.top+r.height*dy,hit=document.elementFromPoint(x,y);
        return {x,y,hit:hit?{tag:hit.tagName,id:hit.id,class_name:hit.className}:null,
          unoccluded:!!hit&&(hit===e||e.contains(hit)||hit.contains(e))};
      });
      const style=getComputedStyle(e);
      return {selector,present:true,text:e.textContent,disabled:e.disabled??null,rect:rect(r),clip,
        display:style.display,visibility:style.visibility,opacity:Number(style.opacity),
        visible_ratio:r.width*r.height>0?area/(r.width*r.height):0,points,ancestors};
    })};
}"""

METRIC_SCROLL_STATE = r"""() => {
  const e=document.querySelector('#metric-strip'),style=getComputedStyle(e),r=e.getBoundingClientRect();
  return {at:new Date().toISOString(),scope:{...e.dataset},scroll_top:e.scrollTop,scroll_left:e.scrollLeft,
    client_height:e.clientHeight,client_width:e.clientWidth,scroll_height:e.scrollHeight,scroll_width:e.scrollWidth,
    overflow_x:style.overflowX,overflow_y:style.overflowY,
    rect:{left:r.left,top:r.top,width:r.width,height:r.height},
    metrics:[...e.querySelectorAll('.metric')].map(m=>({scope:{...m.dataset},
      label:m.querySelector('span').textContent,value:m.querySelector('strong').textContent,note:m.querySelector('small').textContent})),
    page_scroll:{top:document.scrollingElement.scrollTop,left:document.scrollingElement.scrollLeft},
    workspace_scroll:{top:document.querySelector('.workspace-content').scrollTop,left:document.querySelector('.workspace-content').scrollLeft}};
}"""


def metric_footer_reachability(page, root, initial):
    """Native wheel reachability, with explicit setup/restoration and no screenshot."""
    record = {
        "schema_version": "workspace-metric-scroll-v1",
        "before": page.evaluate(METRIC_SCROLL_STATE),
        "setup": "locator_scroll_into_view_if_needed",
        "restoration": "native_reverse_wheel_then_Control+Home",
        "map_before": initial,
        "commands": [],
    }
    footer = page.locator("#metric-strip")
    page.evaluate("""() => {
      window.__metricScrollEvents=[];
      const e=document.querySelector('#metric-strip');
      for(const type of ['wheel','scroll'])e.addEventListener(type,event=>window.__metricScrollEvents.push({
        type,at:new Date().toISOString(),trusted:event.isTrusted,target_tag:event.target.tagName,
        delta_x:event.deltaX??null,delta_y:event.deltaY??null,scroll_top:e.scrollTop,scroll_left:e.scrollLeft}),{passive:true});
      document.addEventListener('keydown',event=>{if(event.key==='Home')window.__metricScrollEvents.push({
        type:'keydown',at:new Date().toISOString(),trusted:event.isTrusted,key:event.key,ctrl_key:event.ctrlKey});});
    }""")
    try:
        assert record["before"]["page_scroll"] == record["before"]["workspace_scroll"] == {"top": 0, "left": 0}
        footer.scroll_into_view_if_needed()
        footer.hover()
        record["onscreen"] = page.evaluate(METRIC_SCROLL_STATE)
        view = record["onscreen"]
        delta_x = view["scroll_width"] if view["scroll_width"] > view["client_width"] else 0
        delta_y = view["scroll_height"] if view["scroll_height"] > view["client_height"] else 0
        record["overflow_axes"] = {"x": delta_x > 0, "y": delta_y > 0}
        if delta_x or delta_y:
            record["commands"].append(
                {"action": "wheel_to_end", "delta_x": delta_x, "delta_y": delta_y, "at": datetime.now(UTC).isoformat()}
            )
            page.mouse.wheel(delta_x, delta_y)
        page.wait_for_function("""() => {const e=document.querySelector('#metric-strip');
          return Math.abs(e.scrollLeft-(e.scrollWidth-e.clientWidth))<=1 && Math.abs(e.scrollTop-(e.scrollHeight-e.clientHeight))<=1;}""")
        record["end"] = page.evaluate(METRIC_SCROLL_STATE)
        record["detail_visibility"] = page.evaluate(VISIBILITY, ["#metric-strip .metric:last-child small"])
    finally:
        view = page.evaluate(METRIC_SCROLL_STATE)
        if view["scroll_left"] or view["scroll_top"]:
            footer.hover()
            record["commands"].append(
                {
                    "action": "wheel_restore",
                    "delta_x": -view["scroll_width"] if view["scroll_left"] else 0,
                    "delta_y": -view["scroll_height"] if view["scroll_top"] else 0,
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            command = record["commands"][-1]
            page.mouse.wheel(command["delta_x"], command["delta_y"])
        page.wait_for_function(
            """() => {const e=document.querySelector('#metric-strip');return e.scrollLeft===0&&e.scrollTop===0;}"""
        )
        record["commands"].append(
            {"action": "keyboard_home", "key": "Control+Home", "at": datetime.now(UTC).isoformat()}
        )
        page.keyboard.press("Control+Home")
        page.wait_for_function(
            "() => document.scrollingElement.scrollTop===0 && document.scrollingElement.scrollLeft===0"
        )
        record["restored"] = page.evaluate(METRIC_SCROLL_STATE)
        record["events"] = page.evaluate("window.__metricScrollEvents")
        _write_json(root / "metric_footer_scroll.json", record)
    record["map_after"] = _map_ack(page)
    record["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(root / "metric_footer_scroll.json", record)
    assert record["map_after"] == initial
    for key in ("scope", "metrics", "scroll_top", "scroll_left", "page_scroll", "workspace_scroll"):
        assert record["restored"][key] == record["before"][key], key
    detail = record["detail_visibility"]["elements"][0]
    assert detail["visible_ratio"] >= 0.98 and all(point["unoccluded"] for point in detail["points"])


@pytest.mark.skipif(not BASE_URL, reason="Requires isolated real application server")
@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 800), (1024, 768), (390, 844)])
def test_workspace_navigation_focus_and_map_identity(tmp_path: Path, width: int, height: int) -> None:
    root = Path(os.environ.get("PRODUCT_E2E_STABILITY_EVIDENCE_DIR", tmp_path)) / f"{width}x{height}"
    root.mkdir(parents=True, exist_ok=True)
    mobile = width == 390
    console: list[str] = []
    warnings: list[str] = []
    failures: list[dict] = []
    http: list[dict] = []
    checkpoints: list[dict] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": width, "height": height}, is_mobile=mobile, has_touch=mobile)
        context.add_init_script(OBSERVE_REAL_MAPS)
        page = context.new_page()
        page.on("pageerror", lambda error: console.append(str(error)))
        page.on("console", lambda message: console.append(message.text) if message.type == "error" else None)
        page.on("console", lambda message: warnings.append(message.text) if message.type == "warning" else None)
        page.on(
            "requestfailed",
            lambda request: failures.append(
                {
                    "url": request.url,
                    "failure": request.failure,
                    "classification": "browser_cancelled"
                    if request.failure == "net::ERR_ABORTED"
                    else "network_failure",
                }
            ),
        )

        def response_record(response):
            if urlsplit(response.url).path.startswith("/api/"):
                row = _http_record(response)
                row["observed_at"] = datetime.now(UTC).isoformat()
                filename = f"http_{len(http) + 1:03d}.json"
                _write_json(root / filename, row)
                http.append(
                    {
                        "file": filename,
                        "method": row["method"],
                        "url": row["url"],
                        "status": row["response_status_code"],
                    }
                )

        page.on("response", response_record)

        def activate(locator):
            locator.tap() if mobile else locator.click()

        def navigate(route):
            nav = ".mobile-nav" if mobile else ".navigation-rail"
            activate(page.locator(f'{nav} [data-route="/app/{route}"]'))
            expect(page).to_have_url(f"{BASE_URL}/app/{route}")

        def capture(label):
            _capture_ready(page, root / f"{label}.png")
            value = page.evaluate("""() => ({
              route:location.pathname, captured_at:new Date().toISOString(),
              document_width:document.documentElement.scrollWidth, viewport_width:innerWidth,
              metric_scope:{...document.querySelector('#metric-strip').dataset},
              text_scope:{...document.querySelector('.text-route-summary')?.dataset},
              evidence_scope:{...document.querySelector('[data-evidence-plan-id]')?.dataset},
              selected_day:document.querySelector('.day-card[aria-pressed=true]')?.dataset.day,
              selected_stop:document.querySelector('[data-geo-stop][aria-pressed=true]')?.dataset.geoStop,
              focus_id:document.activeElement?.id,
              dock:{open:!document.querySelector('#copilot-dock').hidden,
                width:document.querySelector('#copilot-dock').getBoundingClientRect().width,
                left:document.querySelector('#copilot-dock').getBoundingClientRect().left,
                workspace_right:document.querySelector('#main-workspace').getBoundingClientRect().right,
                repair_panel_display:getComputedStyle(document.querySelector('#repair-panel')).display,
                composer_value:document.querySelector('#copilot-input').value},
              scroll_owners:[...document.querySelectorAll('.workspace-content,.itinerary-rail,.repair-panel,.conversation,.textual-map-alternative,.metric-strip')]
                .map(e => ({class_name:e.className,overflow_y:getComputedStyle(e).overflowY,overflow_x:getComputedStyle(e).overflowX,
                  client_height:e.clientHeight,scroll_height:e.scrollHeight,client_width:e.clientWidth,scroll_width:e.scrollWidth})),
            })""")
            assert value["document_width"] <= value["viewport_width"] + 1
            ready = json.loads((root / f"{label}.readiness.json").read_text(encoding="utf-8"))
            if not mobile and label.startswith("copilot_"):
                position = page.evaluate(
                    """() => {
                  const map = """
                    + CURRENT_MAP
                    + """;
                  const container = document.querySelector('#geographic-map');
                  const stopId = document.querySelector('[data-geo-stop][aria-pressed=true]')?.dataset.geoStop;
                  const feature = map.getStyle().sources['v2-stops'].data.features.find(item =>
                    item.properties.stop_id === stopId && item.properties.plan_id === container.dataset.renderedMapPlanId);
                  if (!feature) return {stop_id:stopId, feature_found:false};
                  const point = map.project(feature.geometry.coordinates), rect = container.getBoundingClientRect();
                  const probes = [[0,0],[-10,0],[10,0],[0,-10],[0,10]].map(([dx,dy]) => {
                    const x=point.x+dx,y=point.y+dy,hit=document.elementFromPoint(rect.left+x,rect.top+y);
                    return {x,y,unoccluded:hit===map.getCanvas(),
                      hit:hit?{tag:hit.tagName,id:hit.id,class_name:hit.className}:null};
                  });
                  return {stop_id:stopId, feature_found:true, plan_id:feature.properties.plan_id,
                    coordinates:feature.geometry.coordinates, x:point.x, y:point.y, width:rect.width, height:rect.height,
                    visibility_probes:probes,
                    rendered_stop_ids:map.queryRenderedFeatures(point,{layers:['v2-stops-selected-day']})
                      .map(item=>item.properties.stop_id), observed_at:new Date().toISOString()};
                }"""
                )
                _write_json(root / f"{label}.selected_marker.json", position)
                assert position["feature_found"], position
                assert 12 <= position["x"] <= position["width"] - 12, position
                assert 12 <= position["y"] <= position["height"] - 12, position
                assert position["stop_id"] in position["rendered_stop_ids"], position
                assert all(probe["unoccluded"] for probe in position["visibility_probes"]), position
            checkpoints.append({"label": label, **value, "readiness": ready})
            if label in {"initial_trip", "text_route_day5", "after_module_switching"} or (
                not mobile and label.startswith("copilot_")
            ):
                selectors = ["#inline-undo", "#inline-preview"]
                if label.startswith("copilot_"):
                    selectors.append("#map-context-title")
                    if label != "copilot_closed":
                        selectors += ["#copilot-submit", "#close-copilot"]
                if label == "text_route_day5":
                    selectors += [
                        ".textual-map-alternative > summary",
                        ".text-route-summary > h3:first-child",
                        ".text-route-summary dd:first-of-type",
                    ]
                visible = page.evaluate(VISIBILITY, selectors)
                _write_json(root / f"{label}.visibility.json", visible)
                # Record the long general description separately; it may need an intentional internal scroll.
                if label == "text_route_day5":
                    _write_json(
                        root / "text_route_description.visibility.json",
                        page.evaluate(VISIBILITY, [".textual-map-alternative > p"]),
                    )
                assert all(
                    row.get("present")
                    and row["visible_ratio"] >= 0.98
                    and all(point["unoccluded"] for point in row["points"])
                    for row in visible["elements"]
                ), visible
            if not mobile and value["dock"]["open"]:
                assert value["dock"]["workspace_right"] <= value["dock"]["left"] + 1, value["dock"]
                assert value["dock"]["repair_panel_display"] == "none", value["dock"]
                expected_width = (
                    int(label.removeprefix("copilot_width_")) if label.startswith("copilot_width_") else 440
                )
                assert abs(value["dock"]["width"] - expected_width) <= 1, value["dock"]
            return ready

        with _preserve_browser_evidence(context, page, root, {"console_errors": console, "request_failures": failures}):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                initial = _map_ack(page)
                initial_ready = capture("initial_trip")
                metric_footer_reachability(page, root, initial)
                expect(page.locator("#map-render-status")).to_be_hidden()
                if not mobile:
                    assert 500 <= initial_ready["container"]["height"] <= 760
                assert page.locator(".map-mode-switch").count() == 0
                expect(page.locator(".flow")).to_be_hidden()
                if mobile:
                    rectangles = page.locator(".draft-bar strong, #inline-undo, #inline-preview").evaluate_all(
                        "elements => elements.map(e => {const r=e.getBoundingClientRect(); return {x:r.x,y:r.y,right:r.right,bottom:r.bottom};})"
                    )
                    for index, left in enumerate(rectangles):
                        for right in rectangles[index + 1 :]:
                            assert (
                                left["right"] <= right["x"]
                                or right["right"] <= left["x"]
                                or left["bottom"] <= right["y"]
                                or right["bottom"] <= left["y"]
                            )
                    expect(page.locator(".map-context-card.guidance")).to_be_hidden()
                for reload_index in range(3):
                    page.reload(wait_until="domcontentloaded")
                    assert _map_ack(page)["rendered_map_plan_id"] == initial["rendered_map_plan_id"]
                    capture("after_reload" if reload_index == 0 else f"after_reload_{reload_index + 1}")

                activate(page.locator(".textual-map-alternative > summary"))
                expect(page.locator(".text-route-summary")).to_be_visible()
                _write_json(
                    root / "text_route_before_selection.visibility.json",
                    page.evaluate(
                        VISIBILITY, [".textual-map-alternative > summary", ".text-route-summary > h3:first-child"]
                    ),
                )
                with page.expect_response(
                    lambda response: (
                        response.url.endswith("/draft/impact-preview") and response.request.method == "POST"
                    )
                ) as impact:
                    with page.expect_response(
                        lambda response: (
                            response.url.endswith("/selection")
                            and response.request.method == "POST"
                            and response.request.post_data_json.get("selected_stop_id")
                            == "bixby_creek_bridge_viewpoint"
                        )
                    ) as selected:
                        activate(page.locator('[data-geo-stop="bixby_creek_bridge_viewpoint"]'))
                _write_json(root / "text_route_selection_http.json", _http_record(selected.value))
                _write_json(root / "text_route_impact_http.json", _http_record(impact.value))
                assert selected.value.status == impact.value.status == 200
                selection = selected.value.json()["session"]
                assert (
                    selection["selected_stop_id"] == "bixby_creek_bridge_viewpoint" and selection["selected_day"] == 5
                )
                assert impact.value.json()["session_revision"] == selection["revision"]
                expect(page.locator('[data-geo-stop="bixby_creek_bridge_viewpoint"]')).to_have_attribute(
                    "aria-pressed", "true"
                )
                _map_ack(page)
                expect(page.locator(".textual-map-alternative")).to_have_attribute("open", "")
                expect(page.locator(".text-route-summary")).to_contain_text("Bixby Creek Bridge")
                expect(page.locator(".text-route-summary")).to_have_attribute(
                    "data-plan-id", initial["rendered_map_plan_id"]
                )
                capture("text_route_day5")
                activate(page.locator(".textual-map-alternative > summary"))
                activate(page.locator("#map-edit-button"))
                _map_ack(page)
                assert page.locator(".textual-map-alternative").get_attribute("open") is None
                activate(page.locator("#map-select-button"))
                _map_ack(page)
                assert page.locator(".textual-map-alternative").get_attribute("open") is None

                navigate("compare")
                expect(page.locator("[data-compare-workspace]")).to_be_visible()
                expect(page.locator("#compare-evidence")).to_be_enabled()
                compare_ready = capture("compare_recommendation")
                option = next(row for row in compare_ready["visible_maps"] if row["pane"] == "option")
                assert option["plan_id"] != initial["rendered_map_plan_id"]
                activate(page.locator("#compare-evidence"))
                expect(page.locator("[data-evidence-plan-id]")).to_have_attribute(
                    "data-evidence-plan-id", option["plan_id"]
                )
                # Evidence has no map: save a DOM/state trace, not a false map-ready screenshot.
                _write_json(
                    root / "evidence_scope.json",
                    {
                        "plan_id": option["plan_id"],
                        "content_hash": option["plan_content_hash"],
                        "visible_text": page.locator("#workspace-content").inner_text(),
                    },
                )
                navigate("itinerary")
                capture("after_module_switching")

                opener = page.locator("#open-copilot")
                activate(opener)
                expect(page.locator("#copilot-input")).to_be_focused()
                page.locator("#copilot-input").fill("Unsaved UI test text; do not send.")
                if mobile:
                    expect(page.locator("#copilot-dock")).to_have_attribute("aria-modal", "true")
                    page.locator("#copilot-submit").focus()
                    page.keyboard.press("Tab")
                    assert page.evaluate("document.querySelector('#copilot-dock').contains(document.activeElement)")
                    page.set_viewport_size({"width": 390, "height": 480})
                    page.locator("#copilot-input").focus()
                    expect(page.locator("#copilot-input")).to_be_in_viewport()
                    expect(page.locator("#copilot-submit")).to_be_in_viewport()
                    capture("reduced_keyboard_viewport")
                    page.set_viewport_size({"width": width, "height": height})
                capture("copilot_open")
                if not mobile:
                    resizer = page.locator("#dock-resizer")
                    resizer.focus()
                    resize_commands = []
                    for target_width, key, count in (
                        (560, "ArrowLeft", 6),
                        (320, "ArrowRight", 12),
                        (440, "ArrowLeft", 6),
                    ):
                        for _ in range(count):
                            resizer.press(key)
                        expect(resizer).to_have_attribute("aria-valuenow", str(target_width))
                        resize_commands.append(
                            {
                                "action": "keyboard_resize",
                                "key": key,
                                "count": count,
                                "target_width": target_width,
                                "at": datetime.now(UTC).isoformat(),
                            }
                        )
                        capture(f"copilot_width_{target_width}")
                    if width == 1440:
                        for resized_width in (1024, 1440):
                            page.set_viewport_size({"width": resized_width, "height": height})
                            resize_commands.append(
                                {
                                    "action": "window_resize",
                                    "width": resized_width,
                                    "height": height,
                                    "at": datetime.now(UTC).isoformat(),
                                }
                            )
                            capture(f"copilot_resized_{resized_width}")
                    _write_json(root / "dock_resize_commands.json", resize_commands)
                    expect(page.locator("#copilot-input")).to_have_value("Unsaved UI test text; do not send.")
                page.keyboard.press("Escape")
                expect(page.locator("#copilot-dock")).to_be_hidden()
                expect(opener).to_be_focused()
                capture("copilot_closed")

                if width == 1440:
                    for resized_width in (1024, 1440):
                        page.set_viewport_size({"width": resized_width, "height": height})
                        capture(f"resized_{resized_width}")
                assert console == []
                assert not [row for row in failures if row["classification"] == "network_failure"]
                assert not [row for row in http if row["status"] >= 400]
                assert not [row for row in http if row["method"] == "POST" and "/copilot/messages" in row["url"]]
                _write_json(
                    root / "trace.json",
                    {
                        "schema_version": "workspace-stability-browser-v1",
                        "viewport": {"width": width, "height": height},
                        "touch_enabled": mobile,
                        "keyboard_viewport_simulation": mobile,
                        "os_virtual_keyboard_tested": False,
                        "initial": initial,
                        "checkpoints": checkpoints,
                        "assertions_passed": True,
                        "completed_at": datetime.now(UTC).isoformat(),
                    },
                )
                _write_json(
                    root / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "Chatbox focus and unsent draft only; no chat submitted. Guard receipt measures provider isolation.",
                    },
                )
            finally:
                _write_json(root / "browser_http_index.json", http)
                _write_json(root / "console_errors.json", console)
                _write_json(root / "console_warnings.json", warnings)
                _write_json(root / "network_events.json", failures)
                _write_json(root / "checkpoints.json", checkpoints)
        browser.close()
