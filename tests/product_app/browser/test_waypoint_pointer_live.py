"""Actual canvas click, road check, custom stop, marker drag, confirm, and Undo.

No setup API calls or provider calls. Public getters observe the real MapLibre
renderer; all mutations use the website's visible controls and real local API.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

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
from waypoint_ui_fixture import (
    CURRENT_MAP,
    OBSERVE_REAL_MAPS,
    assert_overlay,
    browser_network_record,
    classify_browser_network,
    focus_with_pointer,
    map_cancellation_policy,
    nearby_road_coordinate,
    overlay_observation,
    point_pixels,
    wait_map,
)

BASE_URL = os.environ.get("PRODUCT_E2E_BASE_URL")
DISPOSABLE = os.environ.get("PRODUCT_E2E_WAYPOINT_POINTER_DISPOSABLE") == "1"
DAY_FIVE_LEG = "leg_comfort_inn_sequoia_kings_canyon_bixby_creek_bridge_viewpoint"

# Read-only aiming: client coordinates must hit visible canvas, not merely lie
# inside a map rectangle covered by the text-route and contextual overlays.
CONTEXT_RELEASE_GEOMETRY = """map => {
  const canvas=map.getCanvas(),panelElement=document.querySelector('.map-context-card');
  const fields=r=>({left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height});
  const container=fields(map.getContainer().getBoundingClientRect());
  const canvasRect=fields(canvas.getBoundingClientRect()),panel=fields(panelElement.getBoundingClientRect());
  const viewport={left:0,top:0,right:document.documentElement.clientWidth,bottom:innerHeight};
  const intersect=(a,b)=>{const left=Math.max(a.left,b.left),top=Math.max(a.top,b.top),
    right=Math.min(a.right,b.right),bottom=Math.min(a.bottom,b.bottom);
    return {left,top,right,bottom,width:Math.max(0,right-left),height:Math.max(0,bottom-top)};};
  const visibleCanvas=intersect(intersect(container,canvasRect),viewport),visiblePanel=intersect(panel,viewport);
  const sample=point=>{const hit=document.elementFromPoint(point.x,point.y);return {...point,
    hit_is_canvas:hit===canvas,hit_is_context:hit?.closest('.map-context-card')===panelElement,
    hit_tag:hit?.tagName || null,hit_id:hit?.id || null,hit_html:hit?.outerHTML?.slice(0,300) || null};};
  const legacy=[];
  for(const fy of [.45,.6,.3,.7]) for(const fx of [.45,.3,.6,.7]) legacy.push(sample({
    x:container.left+container.width*fx,y:container.top+container.height*fy}));
  const starts=[],ends=[];
  if(visibleCanvas.width>8 && visibleCanvas.height>8)
    for(const fy of [.45,.6,.3,.7,.15,.85]) for(const fx of [.45,.3,.6,.7,.15,.85]) starts.push(sample({
      x:visibleCanvas.left+4+(visibleCanvas.width-8)*fx,y:visibleCanvas.top+4+(visibleCanvas.height-8)*fy}));
  if(visiblePanel.width>8 && visiblePanel.height>8)
    for(const fy of [.5,.25,.75]) for(const fx of [.5,.25,.75]) ends.push(sample({
      x:visiblePanel.left+4+(visiblePanel.width-8)*fx,y:visiblePanel.top+4+(visiblePanel.height-8)*fy}));
  let start=null,end=null,endpoint=null;
  for(const candidate of starts) {
    if(!candidate.hit_is_canvas) continue;
    const destinations=ends.filter(point=>point.hit_is_context && Math.hypot(point.x-candidate.x,point.y-candidate.y)>8)
      .sort((a,b)=>Math.hypot(a.x-candidate.x,a.y-candidate.y)-Math.hypot(b.x-candidate.x,b.y-candidate.y));
    if(destinations.length) {start={x:candidate.x,y:candidate.y};endpoint=destinations[0];end={x:endpoint.x,y:endpoint.y};break;}
  }
  return {schema_version:'pointer-visible-frame-v1',observed_at:new Date().toISOString(),
    viewport,scroll:{x:scrollX,y:scrollY},container,canvas:canvasRect,panel,visible_canvas:visibleCanvas,visible_panel:visiblePanel,
    native_text_route_open:document.querySelector('.textual-map-alternative')?.open === true,
    legacy_full_container_samples:legacy,visible_canvas_samples:starts,visible_panel_samples:ends,
    start,end,endpoint_html:endpoint?.hit_html || null,endpoint_is_context:endpoint?.hit_is_context === true};
}"""


def assert_native_marker_release(events, destination, before):
    """Bind physical release to one trusted event across the real panel remount."""
    assert before["hit_is_context"] is True and type(before["hit_panel_id"]) is int
    assert before["hit_panel_id"] > 0
    assert before["current_panel_id"] == before["hit_panel_id"]
    for key in ("x", "y"):
        assert math.isfinite(before[key]) and abs(before[key] - destination[key]) <= 1
    assert events and all(
        event["trusted"] is True
        and event["phase"] in ("capture", "bubble")
        and event["type"] in ("gotpointercapture", "mouseup", "lostpointercapture")
        and (event["type"] == "lostpointercapture" or event["target_is_original_marker"] is True)
        for event in events
    )
    releases = {}
    for phase in ("capture", "bubble"):
        selected = [event for event in events if event["phase"] == phase]
        names = [event["type"] for event in selected]
        assert names.count("gotpointercapture") == names.count("mouseup") == 1
        assert names.index("gotpointercapture") < names.index("mouseup")
        release = selected[names.index("mouseup")]
        assert type(release["event_id"]) is int and release["event_id"] > 0
        assert math.isfinite(release["event_time_stamp"]) and release["event_time_stamp"] > 0
        for key in ("x", "y"):
            assert math.isfinite(release[key]) and abs(release[key] - destination[key]) <= 1
        releases[phase] = release
    capture, bubble = releases["capture"], releases["bubble"]
    assert capture["event_id"] == bubble["event_id"]
    assert capture["event_time_stamp"] == bubble["event_time_stamp"]
    assert events.index(capture) < events.index(bubble)
    assert capture["hit_is_context"] is True
    assert capture["hit_panel_id"] == capture["captured_panel_id"] == before["hit_panel_id"]
    assert capture["current_panel_id"] == before["hit_panel_id"]
    assert capture["captured_panel_connected"] is True
    # Drag-end synchronously replaces the panel before this same mouseup bubbles.
    assert bubble["captured_panel_id"] == before["hit_panel_id"]
    assert bubble["captured_panel_connected"] is False
    assert type(bubble["current_panel_id"]) is int and bubble["current_panel_id"] > 0
    assert bubble["current_panel_id"] != before["hit_panel_id"]
    assert bubble["current_context_title"] == "Unconfirmed relocation"


def release_pan_over_context(page):
    """A real user can release above an overlay; do not avoid that regression."""
    wait_map(page)
    before = page.evaluate("() => (" + CONTEXT_RELEASE_GEOMETRY + ")(" + CURRENT_MAP + ")")
    page.locator("#geographic-map").scroll_into_view_if_needed()
    wait_map(page)
    gesture = page.evaluate(
        """() => {
      const map = """
        + CURRENT_MAP
        + """;
      const frame = ("""
        + CONTEXT_RELEASE_GEOMETRY
        + """)(map),canvas=map.getCanvas();
      const {start,end}=frame;
      if(!start || !end) return {start,end,frame_after:frame,endpoint_is_context:false};
      const hit=document.elementFromPoint(end.x,end.y);
      const trace={instance:window.__e2eObservedMaps.indexOf(map),inputs:[],map_events:[],
        sources_before:map.getStyle().sources,sprite_before:map.getStyle().sprite??null,
        ack_before:{...map.getContainer().dataset}};
      const listeners=[];
      for(const capture of [true,false]) for(const type of ['mousedown','mouseup','pointerup']) {
        const listener=event=>{
          const under=document.elementFromPoint(event.clientX,event.clientY);
          trace.inputs.push({type,phase:capture?'capture':'bubble',trusted:event.isTrusted,
            x:event.clientX,y:event.clientY,buttons:event.buttons,
            target_is_canvas:event.target===canvas,hit_is_context:!!under?.closest('.map-context-card'),
            target:event.target.outerHTML?.slice(0,600),hit:under?.outerHTML.slice(0,600)});
        };
        document.addEventListener(type,listener,capture);
        listeners.push(()=>document.removeEventListener(type,listener,capture));
      }
      for(const type of ['movestart','dragstart','dragend','moveend','idle']) {
        const listener=()=>trace.map_events.push({type,moving:map.isMoving(),
          map_ready:map.getContainer().dataset.mapReady,at:new Date().toISOString()});
        map.on(type,listener);listeners.push(()=>map.off(type,listener));
      }
      window.__e2eContextRelease={map,trace,stop:()=>listeners.forEach(remove=>remove())};
      return {start,end,frame_after:frame,endpoint_html:hit?.outerHTML.slice(0,600),endpoint_is_context:!!hit?.closest('.map-context-card')};
    }"""
    )
    gesture["framing"] = {
        "action": "locator_scroll_into_view_if_needed",
        "selector": "#geographic-map",
        "before": before,
        "after": gesture.pop("frame_after"),
        "map_camera_modified": False,
        "text_route_toggled": False,
    }
    # Retain actual rectangles/hits even when no valid native gesture exists.
    page.evaluate("value=>window.__e2eLastContextRelease=value", gesture)
    assert gesture["start"] and gesture["endpoint_is_context"], (
        "No visible canvas-to-context gesture; inspect retained framing rectangles and element hits"
    )
    assert (
        (gesture["start"]["x"] - gesture["end"]["x"]) ** 2 + (gesture["start"]["y"] - gesture["end"]["y"]) ** 2
    ) ** 0.5 > 8
    try:
        assert page.evaluate(
            "point=>document.elementFromPoint(point.x,point.y) === document.querySelector('#geographic-map canvas')",
            gesture["start"],
        )
        page.mouse.move(**gesture["start"])
        page.mouse.down()
        try:
            page.mouse.move(**gesture["end"], steps=20)
            assert page.evaluate(
                """point=>!!document.elementFromPoint(point.x,point.y)
              ?.closest('.map-context-card')""",
                gesture["end"],
            )
            page.wait_for_timeout(200)
        finally:
            page.mouse.up()
        page.wait_for_function(
            """() => {
          const map = """
            + CURRENT_MAP
            + """;
          return !map.isMoving() && !map.isEasing() && map.getContainer().dataset.mapReady==='true'
            && Number(map.getContainer().dataset.mapTileCoverage)>=.95;
        }""",
            timeout=30_000,
        )
    finally:
        observed = page.evaluate(
            """() => {
          const {map,trace,stop}=window.__e2eContextRelease;stop();
          const current = """
            + CURRENT_MAP
            + """;
          return {...trace,same_map_instance:current===map,moving:map.isMoving(),
            sources_after:map.getStyle().sources,sprite_after:map.getStyle().sprite??null,
            ack_after:{...map.getContainer().dataset}};
        }"""
        )
        gesture["observation"] = observed
        # The caller persists this object even if the structural assertion fails.
        page.evaluate("value=>window.__e2eLastContextRelease=value", gesture)
    assert observed["same_map_instance"] and not observed["moving"]
    assert observed["sources_before"] == observed["sources_after"]
    assert observed["sprite_before"] == observed["sprite_after"]
    assert observed["ack_after"]["mapReady"] == "true"
    assert float(observed["ack_after"]["mapTileCoverage"]) >= 0.95
    for phase in ("capture", "bubble"):
        released = [event for event in observed["inputs"] if event["type"] == "mouseup" and event["phase"] == phase]
        assert len(released) == 1 and released[0]["trusted"]
        assert released[0]["target_is_canvas"] and released[0]["hit_is_context"]
        assert abs(released[0]["x"] - gesture["end"]["x"]) <= 1 and abs(released[0]["y"] - gesture["end"]["y"]) <= 1
    events = [event["type"] for event in observed["map_events"]]
    assert all(event in events for event in ("dragstart", "dragend", "moveend", "idle"))
    assert any(event["moving"] and event["map_ready"] == "false" for event in observed["map_events"])
    return {"action": "real_canvas_pan_release_over_context", **gesture}


def aim_marker_release_over_context(page, origin, target):
    """Position real road geometry under the panel using only actual pan/zoom."""
    aiming = []
    for _ in range(14):
        wait_map(page)
        placement = page.evaluate(
            """coordinates => {
          const map = """
            + CURRENT_MAP
            + """;
          const rect=map.getContainer().getBoundingClientRect(),canvas=map.getCanvas();
          const origin=map.project(coordinates.origin),target=map.project(coordinates.target);
          const from={x:rect.x+origin.x,y:rect.y+origin.y},to={x:rect.x+target.x,y:rect.y+target.y};
          const marker=document.querySelector('.draft-waypoint-marker');
          const startVisible=document.elementFromPoint(from.x,from.y)===marker;
          const targetCovered=!!document.elementFromPoint(to.x,to.y)?.closest('.map-context-card');
          const delta={x:to.x-from.x,y:to.y-from.y};
          if(startVisible && targetCovered) return {ready:true,from,to};
          if(Math.hypot(delta.x,delta.y)<70) return {zoom:true,from,to,start_visible:startVisible};
          const panel=document.querySelector('.map-context-card').getBoundingClientRect();
          const candidates=[];
          for(const fy of [0,.25,.5,.75,1]) for(const fx of [0,.25,.5,.75,1]) {
            const end={x:panel.left+4+(panel.width-8)*fx,y:panel.top+4+(panel.height-8)*fy};
            const start={x:end.x-delta.x,y:end.y-delta.y};
            const hit=document.elementFromPoint(start.x,start.y);
            if(start.x>=rect.left+25 && start.x<=rect.right-25
                && start.y>=rect.top+25 && start.y<=rect.bottom-25
                && (hit===canvas || hit===marker)
                && document.elementFromPoint(end.x,end.y)?.closest('.map-context-card')) {
              candidates.push({x:start.x-from.x,y:start.y-from.y});
            }
          }
          candidates.sort((a,b)=>Math.hypot(a.x,a.y)-Math.hypot(b.x,b.y));
          const move=candidates[0];
          if(!move) return {unavailable:true,from,to,delta};
          const scale=Math.min(1,rect.width*.18/Math.max(1,Math.abs(move.x)),rect.height*.18/Math.max(1,Math.abs(move.y)));
          for(const fraction of [1,.5,.25]) for(const fy of [.6,.45,.3,.7,.2,.8]) for(const fx of [.3,.45,.2,.6,.7]) {
            const start={x:rect.x+rect.width*fx,y:rect.y+rect.height*fy};
            const end={x:start.x+move.x*scale*fraction,y:start.y+move.y*scale*fraction};
            if(Array.from({length:16},(_,index)=>index/15).every(part=>
                document.elementFromPoint(start.x+(end.x-start.x)*part,start.y+(end.y-start.y)*part)===canvas)) {
              return {pan:{start,end},from,to,delta};
            }
          }
          return {unavailable:true,from,to,delta};
        }""",
            {"origin": origin, "target": target},
        )
        aiming.append(placement)
        page.evaluate("observations=>window.__e2eMarkerAiming=observations", aiming)
        if placement.get("ready"):
            return placement["to"], aiming
        assert not placement.get("unavailable"), "No real unobstructed pan can align the road target under the panel"
        if placement.get("zoom"):
            assert placement["start_visible"], "The actual marker must remain accessible during real zoom aiming"
            old_zoom = point_pixels(page, origin)["zoom"]
            page.mouse.move(**placement["from"])
            page.keyboard.down("Control")
            try:
                page.mouse.wheel(0, -240)
            finally:
                page.keyboard.up("Control")
            page.wait_for_function(
                """old => {
              const map = """
                + CURRENT_MAP
                + """;
              return map.getZoom()>old+.1 && !map.isMoving();
            }""",
                arg=old_zoom,
                timeout=20_000,
            )
        else:
            pan = placement["pan"]
            page.mouse.move(**pan["start"])
            page.mouse.down()
            try:
                page.mouse.move(**pan["end"], steps=15)
                page.wait_for_timeout(200)
            finally:
                page.mouse.up()
    raise AssertionError("Actual pointer aiming did not place the road target beneath the visible context panel")


@pytest.mark.skipif(not BASE_URL or not DISPOSABLE, reason="Requires dedicated deterministic waypoint-pointer mode")
def test_map_pointer_add_relocate_and_two_undos_preserve_accepted_plan(tmp_path):
    evidence = Path(os.environ.get("PRODUCT_E2E_WAYPOINT_POINTER_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    http_index, checkpoints, inputs, page_errors, console_errors, failures = [], {}, [], [], [], []
    map_configuration = None
    map_configuration_http_file = None
    network_policy = None
    phase = "initial"
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, **({"executable_path": str(CHROME)} if CHROME.is_file() else {})
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(OBSERVE_REAL_MAPS)
        page = context.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda item: console_errors.append(item.text) if item.type == "error" else None)
        page.on("requestfailed", lambda request: failures.append(browser_network_record(request, phase)))

        def record(response):
            nonlocal map_configuration, map_configuration_http_file
            if response.status >= 400:
                failures.append(browser_network_record(response.request, phase, http_status=response.status))
            if "/api/" not in response.url:
                return
            row = {"phase": phase, **_http_record(response)}
            name = f"http_{len(http_index) + 1:03d}.json"
            _write_json(evidence / name, row)
            http_index.append(
                {"file": name, "phase": phase, "method": row["method"], "status": response.status, "url": response.url}
            )
            if response.url == f"{BASE_URL}/api/map/config" and response.status == 200:
                current = json.loads(row["response_body_utf8"])
                assert map_configuration is None or current == map_configuration, (
                    "Map configuration changed during the journey"
                )
                map_configuration, map_configuration_http_file = current, name

        page.on("response", record)

        def capture(name, operation=None):
            before = overlay_observation(page, operation)
            observation = {
                "schema_version": "waypoint-overlay-capture-v1",
                "checkpoint": name,
                "expected_operation": operation,
                "before": before,
                "after": None,
                "assertions_passed": False,
            }
            _write_json(evidence / f"{name}.overlay_observation.json", observation)
            assert_overlay(before, operation)
            _capture_ready(page, evidence / f"{name}.png")
            ack, overlay = _map_ack(page), overlay_observation(page, operation)
            observation.update(after=overlay, map_ack=ack)
            _write_json(evidence / f"{name}.overlay_observation.json", observation)
            assert_overlay(overlay, operation)
            assert before == overlay, (
                "The actual draft overlay must stay unchanged throughout map-ready screenshot capture"
            )
            assert overlay["plan_id"] == ack["active_plan_id"] == ack["rendered_map_plan_id"]
            assert overlay["plan_content_hash"] == ack["active_plan_content_hash"]
            assert overlay["geometry_hash"] == ack["rendered_geometry_hash"]
            assert overlay["accepted_marker_ids"] == ack["rendered_marker_ids"]
            checkpoints[name] = {
                "map_ack": ack,
                "overlay": overlay,
                "draft_status": page.locator(".draft-bar").inner_text(),
                "visible_context": page.locator(".map-context-card").inner_text(),
            }
            observation["assertions_passed"] = True
            _write_json(evidence / f"{name}.overlay_observation.json", observation)
            return ack

        def confirm(button_label, operation_type):
            with page.expect_response(
                lambda response: "/map/snap-previews/" in response.url and response.url.endswith("/confirm")
            ) as response:
                page.get_by_role("button", name=button_label, exact=True).click()
            assert response.value.status == 200, response.value.text()
            payload = response.value.json()
            assert payload["feedback_tier"] == "route_checked" and payload["evaluated_repair"] is False
            assert payload["operation"]["type"] == operation_type
            expect(page.locator("#inline-preview")).to_be_disabled()
            expect(page.locator(".draft-bar")).to_contain_text("evaluated repair is not enabled")
            return payload

        with _preserve_browser_evidence(
            context, page, evidence, {"page_errors": page_errors, "request_failures": failures}
        ):
            try:
                page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
                parent = capture("initial_parent")
                network_policy = map_cancellation_policy(
                    map_configuration, page.evaluate("() => (" + CURRENT_MAP + ").getStyle()")
                )
                phase = "select_route"
                page.locator('.day-card[data-day="5"]').click()
                page.locator(".textual-map-alternative summary").click()
                with page.expect_response(lambda response: response.url.endswith("/selection")) as selected:
                    page.locator(f'[data-geo-route-leg="{DAY_FIVE_LEG}"]').click()
                assert selected.value.status == 200
                expect(page.locator(".map-context-card")).to_contain_text("Road-validated route leg")
                page.locator("#map-edit-button").click()
                text_route = page.locator(".textual-map-alternative")
                assert text_route.evaluate("element=>element.open") is False
                text_route.locator("summary").click()
                assert text_route.evaluate("element=>element.open") is True
                phase = "canvas_overlay_release"
                inputs.append(release_pan_over_context(page))
                assert capture("canvas_overlay_release") == parent
                phase = "raw_map_click"
                # Keep the open text route for the overlay-release regression
                # above, then return to canvas editing through its real control.
                assert text_route.evaluate("element=>element.open") is True
                framing = {
                    "schema_version": "native-canvas-focus-transition-v1",
                    "before": page.evaluate("() => (" + CONTEXT_RELEASE_GEOMETRY + ")(" + CURRENT_MAP + ")"),
                    "map_ack_before": _map_ack(page),
                    "action": "native_text_route_summary_click",
                    "selector": ".textual-map-alternative summary",
                }
                page.evaluate("""() => {
                  document.querySelector('.textual-map-alternative summary').addEventListener('click',event=>{
                    window.__e2eTextRouteClose={trusted:event.isTrusted,at:new Date().toISOString(),
                      target_is_summary:!!event.target.closest('.textual-map-alternative summary')};
                  },{once:true});
                }""")
                text_route.locator("summary").click()
                wait_map(page)
                framing["after"] = page.evaluate("() => (" + CONTEXT_RELEASE_GEOMETRY + ")(" + CURRENT_MAP + ")")
                framing["map_ack_after"] = _map_ack(page)
                framing["click"] = page.evaluate("window.__e2eTextRouteClose")
                _write_json(evidence / "canvas_focus_transition.json", framing)
                assert framing["click"]["trusted"] is True and framing["click"]["target_is_summary"] is True
                assert framing["after"]["native_text_route_open"] is False
                assert any(sample["hit_is_canvas"] for sample in framing["after"]["legacy_full_container_samples"])
                assert _map_ack(page) == parent
                coordinate = [-121.94165, 36.51710]
                pixels = focus_with_pointer(page, coordinate)
                inputs.append({"action": "real_canvas_click", "target_coordinate": coordinate, "pixels": pixels})
                page.evaluate("""() => {
                  document.querySelector('#geographic-map canvas').addEventListener('click',event=>{
                    window.__e2eRawCanvasClick={trusted:event.isTrusted,at:new Date().toISOString(),
                      target_is_canvas:event.target===document.querySelector('#geographic-map canvas'),
                      x:event.clientX,y:event.clientY};
                  },{capture:true,once:true});
                }""")
                page.mouse.click(pixels["x"], pixels["y"])
                inputs[-1]["native_click"] = page.evaluate("window.__e2eRawCanvasClick")
                expect(page.locator("#map-context-title")).to_have_text("Unconfirmed map point")
                expect(page.locator(".draft-bar strong")).to_have_text("0 draft changes")
                page.locator("#exploratory-meaning").select_option("possible_stop")
                phase = "road_check_add"
                with page.expect_response(
                    lambda response: response.url.endswith("/map/snap-preview"), timeout=60_000
                ) as snap:
                    page.locator("#check-road-access").click()
                assert snap.value.status == 200, snap.value.text()
                snap_payload = snap.value.json()
                assert snap_payload["validation_state"] == "route_checked"
                assert snap_payload["persisted"] is False and snap_payload["operation_intent"] == "add_custom_waypoint"
                page.locator("#waypoint-name").fill("Pointer-tested Point Lobos draft stop")
                page.locator("#waypoint-role").select_option("scenic_stop")
                page.locator("#waypoint-duration-preferred").fill("45")
                if snap_payload["confirmation_required"]:
                    page.locator("#snap-warning-ack").check()
                phase = "confirm_add"
                added = confirm("Add custom stop", "add_custom_waypoint")
                operation = added["operation"]
                waypoint_id = operation["target"]
                assert len(added["session"]["draft"]) == 1
                assert added["session"]["accepted_plan_id"] == parent["active_plan_id"]
                assert capture("added_draft", operation) == parent

                phase = "select_waypoint_and_drag"
                marker = page.locator(".draft-waypoint-marker")
                point = operation["parameters"]["snapped_coordinate"]
                focus_with_pointer(page, [point["longitude"], point["latitude"]])
                marker.click()
                page.locator("#relocate-custom-waypoint").click()
                focus_with_pointer(page, [point["longitude"], point["latitude"]])
                wait_map(page)
                target = nearby_road_coordinate(operation)
                destination, aiming = aim_marker_release_over_context(
                    page,
                    [point["longitude"], point["latitude"]],
                    target,
                )
                marker_bounds = marker.bounding_box()
                assert marker_bounds is not None
                start = {
                    "x": marker_bounds["x"] + marker_bounds["width"] / 2,
                    "y": marker_bounds["y"] + marker_bounds["height"] / 2,
                }
                inputs.append(
                    {
                        "action": "real_marker_pointer_drag",
                        "waypoint_id": waypoint_id,
                        "from_pixels": start,
                        "target_coordinate_from_actual_road_geometry": target,
                        "to_pixels": destination,
                        "real_pointer_aiming": aiming,
                    }
                )
                page.evaluate("""() => {
                  const marker=document.querySelector('.draft-waypoint-marker'),events=[],listeners=[];
                  const eventIds=new WeakMap(),panelIds=new WeakMap();
                  let nextEventId=0,nextPanelId=0,capturedPanel=null;
                  const panelId=panel=>{
                    if(!panel)return null;
                    if(!panelIds.has(panel))panelIds.set(panel,++nextPanelId);
                    return panelIds.get(panel);
                  };
                  const observePoint=point=>{
                    const hit=Number.isFinite(point.x) ? document.elementFromPoint(point.x,point.y) : null;
                    const panel=hit?.closest('.map-context-card'),current=document.querySelector('.map-context-card');
                    return {x:point.x,y:point.y,hit_is_context:!!panel,hit_panel_id:panelId(panel),
                      captured_panel_id:panelId(capturedPanel),captured_panel_connected:capturedPanel?.isConnected ?? null,
                      current_panel_id:panelId(current),current_context_title:current?.querySelector('#map-context-title')?.textContent,
                      hit:hit?.outerHTML.slice(0,600)};
                  };
                  for(const capture of [true,false]) for(const type of ['gotpointercapture','mouseup','lostpointercapture']) {
                    const listener=event=>{
                      if(!eventIds.has(event))eventIds.set(event,++nextEventId);
                      if(type==='mouseup' && capture)capturedPanel=document.elementFromPoint(event.clientX,event.clientY)
                        ?.closest('.map-context-card') ?? null;
                      events.push({type,phase:capture?'capture':'bubble',trusted:event.isTrusted,
                        event_id:eventIds.get(event),event_time_stamp:event.timeStamp,
                        target_is_original_marker:event.target===marker,target:event.target.outerHTML?.slice(0,600),
                        ...observePoint({x:event.clientX,y:event.clientY})});
                    };
                    document.addEventListener(type,listener,capture);
                    listeners.push(()=>document.removeEventListener(type,listener,capture));
                  }
                  window.__e2eMarkerRelease={events,observePoint,stop:()=>listeners.forEach(remove=>remove())};
                }""")
                with page.expect_response(
                    lambda response: response.url.endswith("/map/snap-preview"), timeout=60_000
                ) as moved_snap:
                    page.mouse.move(start["x"], start["y"])
                    page.mouse.down()
                    try:
                        page.mouse.move(destination["x"], destination["y"], steps=20)
                        before_release = page.evaluate(
                            "point=>window.__e2eMarkerRelease.observePoint(point)", destination
                        )
                        inputs[-1]["release_hit_before_mouseup"] = before_release
                        assert before_release["hit_is_context"] is True
                    finally:
                        page.mouse.up()
                assert moved_snap.value.status == 200, moved_snap.value.text()
                release_events = page.evaluate(
                    "() => {window.__e2eMarkerRelease.stop();return window.__e2eMarkerRelease.events}"
                )
                inputs[-1]["native_capture_release_events"] = release_events
                assert_native_marker_release(release_events, destination, before_release)
                gesture = page.evaluate("() => window.__e2eMarkerGestures.at(-1)")
                assert gesture[0]["type"] == "mousedown" and gesture[-1]["type"] == "mouseup"
                assert len(gesture) >= 3 and all(event["trusted"] for event in gesture)
                assert any(event["type"] == "mousemove" and event["buttons"] == 1 for event in gesture)
                assert abs(gesture[0]["x"] - start["x"]) <= 1 and abs(gesture[0]["y"] - start["y"]) <= 1
                assert abs(gesture[-1]["x"] - destination["x"]) <= 1 and abs(gesture[-1]["y"] - destination["y"]) <= 1
                assert (
                    (gesture[-1]["x"] - gesture[0]["x"]) ** 2 + (gesture[-1]["y"] - gesture[0]["y"]) ** 2
                ) ** 0.5 > 8
                inputs[-1]["trusted_dom_mouse_events"] = gesture
                moved_preview = moved_snap.value.json()
                assert moved_preview["operation_intent"] == "relocate_custom_waypoint"
                assert moved_preview["validation_state"] == "route_checked" and moved_preview["persisted"] is False
                expect(page.locator("#map-context-title")).to_have_text("Unconfirmed relocation")
                expect(page.locator(".draft-bar strong")).to_have_text("1 draft change")
                if moved_preview["confirmation_required"]:
                    page.locator("#snap-warning-ack").check()
                phase = "confirm_relocation"
                relocated = confirm("Confirm relocation", "relocate_custom_waypoint")
                moved = relocated["operation"]
                assert moved["target"] == waypoint_id
                assert len(relocated["session"]["draft"]) == 2
                for field in ("name", "role", "duration", "day", "insertion"):
                    assert moved["parameters"][field] == operation["parameters"][field]
                assert moved["parameters"]["snapped_coordinate"] != operation["parameters"]["snapped_coordinate"]
                moved_point = moved["parameters"]["snapped_coordinate"]
                focus_with_pointer(page, [moved_point["longitude"], moved_point["latitude"]])
                assert capture("relocated_draft", moved) == parent

                for undo_index, (label, expected_undone, remaining) in enumerate(
                    (("undo_relocation", moved, operation), ("undo_add", operation, None))
                ):
                    phase = label
                    expected_revision = relocated["session"]["revision"] + undo_index + 1
                    # Undo's HTTP body is not its UI completion. The accepted
                    # parent identity is unchanged while the old draft can remain
                    # drawn until this exact new revision's impact check resolves.
                    with page.expect_response(
                        lambda response, expected_revision=expected_revision: (
                            response.url.endswith("/draft/impact-preview")
                            and response.request.post_data_json["expected_revision"] == expected_revision
                        ),
                        timeout=30_000,
                    ) as impact:
                        with page.expect_response(lambda response: response.url.endswith("/draft/undo")) as undone:
                            page.locator("#inline-undo").click()
                    assert undone.value.status == 200, undone.value.text()
                    payload = undone.value.json()
                    assert payload["undone"]["operation_id"] == expected_undone["operation_id"]
                    assert payload["session"]["draft"] == ([remaining] if remaining else [])
                    assert payload["session"]["revision"] == expected_revision
                    _write_json(evidence / f"{label}.impact_http_io.json", _http_record(impact.value))
                    assert impact.value.status == 200, impact.value.text()
                    assert impact.value.json()["session_revision"] == expected_revision
                    expect(page.locator(".draft-bar strong")).to_have_text(
                        "1 draft change" if remaining else "0 draft changes", timeout=30_000
                    )
                    assert capture(label, remaining) == parent
                phase = "reload_after_undo"
                page.reload(wait_until="domcontentloaded")
                assert capture("parent_after_reload") == parent
                assert page_errors == console_errors == []
                assert classify_browser_network(failures, network_policy)["blocking_failure_indexes"] == []
                assert all(row["status"] < 400 for row in http_index)
                assert not any(row["url"].endswith(("/preview", "/accept", "/copilot/messages")) for row in http_index)
                _write_json(
                    evidence / "llm_io.json",
                    {
                        "provider_call_count": 0,
                        "llm_input": None,
                        "llm_output": None,
                        "reason": "No chat submitted; provider dispatch is additionally measured by the finalized application-server isolation receipt.",
                    },
                )
                _write_json(
                    evidence / "browser_assertions.json",
                    {
                        "schema_version": "waypoint-pointer-browser-v1",
                        "assertions_passed": True,
                        "accepted_plan_id": parent["active_plan_id"],
                        "accepted_plan_content_hash": parent["active_plan_content_hash"],
                        "accepted_geometry_hash": parent["rendered_geometry_hash"],
                        "waypoint_id": waypoint_id,
                        "added_operation_id": operation["operation_id"],
                        "relocated_operation_id": moved["operation_id"],
                        "real_canvas_click": True,
                        "real_marker_pointer_drag": True,
                        "real_canvas_overlay_release": True,
                        "real_marker_overlay_release": True,
                        "setup_api_mutations": False,
                        "accepted_plan_unchanged": True,
                        "evaluated_repair_claimed": False,
                        "final_draft_count": 0,
                        "renderer_observation": "Read-only public getters on real MapLibre instances; no map or API response mocks",
                    },
                )
            finally:
                if phase == "select_waypoint_and_drag":
                    _write_json(
                        evidence / "marker_release_failure.json",
                        page.evaluate("""() => ({
                      aiming:window.__e2eMarkerAiming || [],events:window.__e2eMarkerRelease?.events || [],
                      marker_gestures:window.__e2eMarkerGestures || []})"""),
                    )
                if phase == "canvas_overlay_release":
                    diagnostic = page.evaluate("() => window.__e2eLastContextRelease || null")
                    if diagnostic:
                        _write_json(evidence / "overlay_release_failure.json", diagnostic)
                _write_json(evidence / "http_index.json", http_index)
                _write_json(evidence / "map_checkpoints.json", checkpoints)
                _write_json(evidence / "pointer_inputs.json", inputs)
                _write_json(
                    evidence / "browser_errors.json",
                    {"page_errors": page_errors, "console_errors": console_errors, "request_failures": failures},
                )
                if network_policy is not None:
                    _write_json(
                        evidence / "network_classification.json",
                        {
                            "map_configuration_http_file": map_configuration_http_file,
                            **classify_browser_network(failures, network_policy),
                        },
                    )
        context.close()
        browser.close()
