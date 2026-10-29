"""Read-only observation of the actual MapLibre instances used by the page.

The constructor wrapper returns the original production Map, unmodified. Tests
use public getters to locate genuine canvas pixels and inspect drawn data; every
edit is still a browser click, drag, or form submission. No API, geometry, map
event, application state, or router result is mocked or supplied by this helper.
"""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from urllib.parse import urlsplit


def browser_network_record(request, phase, *, http_status=None):
    """Keep the raw observed failure; a cancelled fetch is not an HTTP status."""
    return {"event":"http_error" if http_status is not None else "request_failed",
        "url":request.url,"method":request.method,"resource_type":request.resource_type,
        "phase":phase,"observed_at":datetime.now(UTC).isoformat(),"range":request.headers.get("range"),
        "failure":request.failure,"http_status":http_status}


def map_cancellation_policy(map_config, rendered_style):
    """Pin URLs to the actual API configuration and real renderer's vector sources."""
    if not isinstance(map_config, dict) or not isinstance(rendered_style, dict):
        raise ValueError("map_cancellation_source_invalid")
    if (map_config.get("schema_version") != "product-map-configuration-v2"
            or map_config.get("provider") != "maplibre_pmtiles" or map_config.get("status") != "ready"):
        raise ValueError("map_cancellation_configuration_not_ready")
    base = map_config.get("base_url")
    if not isinstance(base, str):
        raise ValueError("map_cancellation_origin_invalid")
    origin = urlsplit(base)
    if (origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost", "::1"}
            or origin.port is None or origin.username is not None or origin.password is not None
            or origin.path or origin.query or origin.fragment):
        raise ValueError("map_cancellation_origin_invalid")

    def exact_asset(url, directory, suffix):
        if not isinstance(url, str):
            raise ValueError("map_cancellation_asset_invalid")
        parsed = urlsplit(url)
        if (parsed.scheme != origin.scheme or parsed.netloc != origin.netloc or parsed.query or parsed.fragment
                or not parsed.path.startswith(directory) or not parsed.path.endswith(suffix)
                or "%" in parsed.path or ".." in parsed.path or "\\" in parsed.path):
            raise ValueError("map_cancellation_asset_invalid")
        return url

    style_url = exact_asset(map_config.get("style_url"), "/styles/", ".json")
    sources = rendered_style.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("map_cancellation_sources_missing")
    vectors, archives = {}, set()
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise ValueError("map_cancellation_source_invalid")
        if source.get("type") != "vector":
            continue
        url = source.get("url")
        if not isinstance(name, str) or not name or not isinstance(url, str) or not url.startswith("pmtiles://"):
            raise ValueError("map_cancellation_vector_source_invalid")
        archives.add(exact_asset(url.removeprefix("pmtiles://"), "/data/", ".pmtiles"))
        vectors[name] = {"type":"vector","url":url}
    if not archives:
        raise ValueError("map_cancellation_archive_missing")
    sprite_base = rendered_style.get("sprite")
    sprite_assets = set()
    if sprite_base is not None:
        if not isinstance(sprite_base, str):
            raise ValueError("map_cancellation_sprite_invalid")
        parsed = urlsplit(sprite_base)
        if (parsed.scheme != origin.scheme or parsed.netloc != origin.netloc or parsed.query or parsed.fragment
                or not parsed.path.startswith("/sprites/") or parsed.path.endswith((".json", ".png"))
                or "%" in parsed.path or ".." in parsed.path or "\\" in parsed.path):
            raise ValueError("map_cancellation_sprite_invalid")
        sprite_assets.update(sprite_base + suffix for suffix in (".json", ".png", "@2x.json", "@2x.png"))
    return {"schema_version":"local-map-cancellation-policy-v1",
        "source":"actual_api_map_configuration_and_real_map_public_style_getter",
        "map_configuration":{key:map_config[key] for key in ("schema_version","provider","status","base_url","style_url")},
        "rendered_vector_sources":vectors,"rendered_sprite_base":sprite_base,
        "allowlisted_urls":sorted({style_url,*archives,*sprite_assets}),
        "scope":"Browser-reported cancellation only; no map-disposal cause is inferred. Readiness and identity gates remain mandatory."}


def classify_browser_network(records, policy):
    """Partition raw observations without discarding or rewriting any event."""
    if not isinstance(records, list) or not isinstance(policy, dict):
        raise ValueError("browser_network_observation_invalid")
    expected = map_cancellation_policy(policy.get("map_configuration"),
        {"sources":policy.get("rendered_vector_sources"),"sprite":policy.get("rendered_sprite_base")})
    if policy != expected:
        raise ValueError("browser_network_policy_mismatch")
    classifications, cancelled, blocking = [], [], []
    for index, row in enumerate(records):
        valid = isinstance(row, dict)
        if valid:
            valid = (set(row) == {"event","url","method","resource_type","phase","observed_at","range","failure","http_status"}
                and isinstance(row["event"],str) and row["event"] in {"http_error","request_failed"}
                and all(isinstance(row[key], str) and row[key] for key in ("url","method","resource_type","phase","observed_at"))
                and (row["range"] is None or isinstance(row["range"], str)
                    and re.fullmatch(r"bytes=(?:[0-9]+-[0-9]*|-[0-9]+)",row["range"]) is not None))
        if valid:
            try:
                valid = datetime.fromisoformat(row["observed_at"]).utcoffset() is not None
            except ValueError:
                valid = False
        if valid:
            valid = (row["event"] == "http_error" and type(row["http_status"]) is int
                and 400 <= row["http_status"] <= 599 and row["failure"] is None
                or row["event"] == "request_failed" and row["http_status"] is None
                and isinstance(row["failure"], str) and bool(row["failure"]))
        if not valid:
            classification = "malformed_network_record"
        elif row["event"] == "http_error":
            classification = "http_error"
        elif row["failure"] == "net::ERR_ABORTED":
            classification = ("browser_cancelled_local_map_asset" if row["method"] == "GET"
                and row["resource_type"] in {"fetch","xhr"} and row["url"] in policy["allowlisted_urls"]
                else "browser_cancelled_other_request")
        else:
            classification = "network_failure"
        nonblocking = classification == "browser_cancelled_local_map_asset"
        (cancelled if nonblocking else blocking).append(index)
        classifications.append({"raw_record_index":index,"classification":classification,"blocking":not nonblocking})
    return {"schema_version":"browser-network-classification-v1","policy":policy,
        "raw_record_count":len(records),"classifications":classifications,
        "browser_cancelled_local_map_asset_indexes":cancelled,"blocking_failure_indexes":blocking}

OBSERVE_REAL_MAPS = """(() => {
  const maps = [];
  Object.defineProperty(window, '__e2eObservedMaps', {value:maps});
  const gestures = [];
  Object.defineProperty(window, '__e2eMarkerGestures', {value:gestures});
  let gesture = null;
  for (const type of ['mousedown','mousemove','mouseup']) document.addEventListener(type, event => {
    if (type === 'mousedown' && event.target.closest?.('.draft-waypoint-marker')) {
      gesture = []; gestures.push(gesture);
    }
    if (gesture) gesture.push({type,trusted:event.isTrusted,x:event.clientX,y:event.clientY,buttons:event.buttons});
    if (type === 'mouseup') gesture = null;
  }, true);
  let library;
  Object.defineProperty(window, 'maplibregl', {configurable:true,
    get() { return library; },
    set(value) {
      library = value;
      const OriginalMap = value.Map;
      value.Map = new Proxy(OriginalMap, {
        construct(target, argumentsList, newTarget) {
          const map = Reflect.construct(target, argumentsList, newTarget);
          maps.push(map);
          return map;
        }
      });
    }
  });
})();"""

CURRENT_MAP = """[...window.__e2eObservedMaps].reverse().find(map =>
  map.getContainer().id === 'geographic-map' && map.getContainer().isConnected)"""


def wait_map(page):
    page.wait_for_function("""() => {
      const element = document.querySelector('#geographic-map');
      return element?.dataset.mapReady === 'true' && Number(element.dataset.mapTileCoverage) >= .95;
    }""", timeout=30_000)


def point_pixels(page, coordinate):
    return page.evaluate("""coordinate => {
      const map = """ + CURRENT_MAP + """;
      const point = map.project(coordinate), bounds = map.getContainer().getBoundingClientRect();
      return {x:bounds.x+point.x,y:bounds.y+point.y,zoom:map.getZoom(),
        canvas:{x:bounds.x,y:bounds.y,width:bounds.width,height:bounds.height}};
    }""", coordinate)


def focus_with_pointer(page, coordinate):
    """Pan/zoom with real pointer input, using public map projection only to aim."""
    wait_map(page)
    for _ in range(16):
        observation = point_pixels(page, coordinate)
        visible = page.evaluate("""point => {
          const element=document.querySelector('#geographic-map'),rect=element.getBoundingClientRect();
          const hit=document.elementFromPoint(point.x,point.y),margin=Math.min(48,rect.width*.1,rect.height*.1);
          return point.x>=rect.left+margin && point.x<=rect.right-margin
            && point.y>=rect.top+margin && point.y<=rect.bottom-margin
            && (hit===element.querySelector('canvas') || (element.contains(hit) && !!hit?.closest('.draft-waypoint-marker')));
        }""", observation)
        clear_scan = page.evaluate("""() => {
          const element = document.querySelector('#geographic-map');
          const rect = element.getBoundingClientRect(), canvas = element.querySelector('canvas');
          const isSurface = hit => hit === canvas || hit === canvas?.parentElement
            || hit?.classList?.contains('maplibregl-canvas-container');
          const probes = [];
          for (const fy of [.5,.6,.4,.7,.3,.8,.2]) for (const fx of [.5,.6,.4,.7,.3,.8,.2]) {
            const x = rect.x+rect.width*fx, y = rect.y+rect.height*fy;
            const hit = document.elementFromPoint(x,y);
            if (isSurface(hit)) return {point:{x,y},probes};
            probes.push({x,y,tag:hit?.tagName || null,id:hit?.id || null,
              classes:[...hit?.classList || []].slice(0,6)});
          }
          return {point:null,probes};
        }""")
        clear = clear_scan["point"]
        assert clear, f"A real, unobstructed map canvas point must be available: {clear_scan['probes']}"
        # A visible exact projection is already a valid pointer target. Avoid
        # repeatedly centering it to subpixel precision before each wheel step.
        goal = clear
        canvas = observation["canvas"]
        dx, dy = goal["x"] - observation["x"], goal["y"] - observation["y"]
        if not visible:
            scale = min(1, canvas["width"] * .2 / max(abs(dx), 1), canvas["height"] * .2 / max(abs(dy), 1))
            pan = page.evaluate("""delta => {
              const element=document.querySelector('#geographic-map'), rect=element.getBoundingClientRect();
              const canvas=element.querySelector('canvas');
              const isSurface = hit => hit === canvas || hit === canvas?.parentElement
                || hit?.classList?.contains('maplibregl-canvas-container');
              for (const fraction of [1,.5,.25]) for (const fy of [.45,.6,.3,.7,.2,.8]) for (const fx of [.3,.45,.6,.2,.7,.8]) {
                const start={x:rect.x+rect.width*fx,y:rect.y+rect.height*fy};
                const end={x:start.x+delta.x*fraction,y:start.y+delta.y*fraction};
                if (Array.from({length:16},(_,index)=>index/15).every(part=>
                  isSurface(document.elementFromPoint(start.x+(end.x-start.x)*part,start.y+(end.y-start.y)*part)))) return {start,end};
              }
              return null;
            }""", {"x":dx*scale,"y":dy*scale})
            assert pan, "The complete real pan path, including mouse release, must stay on unobstructed canvas"
            page.mouse.move(pan["start"]["x"], pan["start"]["y"])
            page.mouse.down()
            try:
                page.mouse.move(pan["end"]["x"], pan["end"]["y"], steps=15)
                assert page.evaluate("""point => {
                  const canvas=document.querySelector('#geographic-map canvas'),hit=document.elementFromPoint(point.x,point.y);
                  return hit===canvas || hit===canvas?.parentElement || hit?.classList?.contains('maplibregl-canvas-container');
                }""", pan["end"]), "The pan release became covered by another UI surface"
                # A deliberate pause ends the real gesture without kinetic
                # overshoot, so repeated projection-based aiming converges.
                page.wait_for_timeout(200)
            finally:
                page.mouse.up()
            wait_map(page)
            continue
        if observation["zoom"] >= 16:
            return observation
        page.mouse.move(observation["x"], observation["y"])
        page.keyboard.down("Control")
        try:
            page.mouse.wheel(0, -600)
        finally:
            page.keyboard.up("Control")
        page.wait_for_function("""previous => {
          const map = """ + CURRENT_MAP + """;
          return map.getZoom() > previous + .1 && !map.isMoving();
        }""", arg=observation["zoom"], timeout=20_000)
        wait_map(page)
    raise AssertionError("Could not aim at a road coordinate using actual map pan and zoom")


def nearby_road_coordinate(operation):
    """Choose an actual returned road coordinate, not invented routing geometry."""
    point = operation["parameters"]["snapped_coordinate"]
    origin = [point["longitude"], point["latitude"]]
    candidates = []
    for leg in operation["parameters"]["affected_route_legs"]:
        for coordinate in leg["geometry"]["coordinates"]:
            distance = math.hypot((coordinate[0]-origin[0])*math.cos(math.radians(origin[1])), coordinate[1]-origin[1])*111_320
            if 30 <= distance <= 130:
                candidates.append((distance, coordinate))
    assert candidates, "Actual router geometry must supply a nearby relocation target"
    return min(candidates, key=lambda row: abs(row[0]-65))[1]


def overlay_observation(page, operation=None):
    wait_map(page)
    coordinate = operation["parameters"]["snapped_coordinate"] if operation else None
    return page.evaluate("""coordinate => {
      const map = """ + CURRENT_MAP + """;
      const element = map.getContainer(), bounds = element.getBoundingClientRect();
      const source = map.getStyle().sources['v2-draft-routes'];
      const routes = source?.data || {type:'FeatureCollection',features:[]};
      const markers = [...element.querySelectorAll('.draft-waypoint-marker')].map(marker => {
        const rect = marker.getBoundingClientRect();
        const center = {x:rect.x+rect.width/2-bounds.x,y:rect.y+rect.height/2-bounds.y};
        const projected = coordinate ? map.project([coordinate.longitude,coordinate.latitude]) : null;
        return {label:marker.getAttribute('aria-label'),text:marker.textContent,
          selected:marker.classList.contains('is-selected'),
          center_in_map:center,expected_coordinate:coordinate,
          distance_from_expected_pixels:projected ? Math.hypot(center.x-projected.x,center.y-projected.y) : null,
          visible:rect.width>0 && rect.height>0 && rect.right>bounds.left && rect.left<bounds.right
            && rect.bottom>bounds.top && rect.top<bounds.bottom};
      });
      const rendered = map.getLayer('v2-draft-route')
        ? map.queryRenderedFeatures({layers:['v2-draft-route']}).map(row => row.properties.route_leg_id) : [];
      return {scope:'read_only_real_map_observation_not_production_overlay_ack',
        plan_id:element.dataset.renderedMapPlanId,plan_content_hash:element.dataset.renderedPlanContentHash,
        geometry_hash:element.dataset.renderedGeometryHash,
        accepted_marker_ids:JSON.parse(element.dataset.renderedMarkerIds || '[]'),
        draft_routes:routes,draft_route_source_loaded:source ? map.isSourceLoaded('v2-draft-routes') : true,
        rendered_draft_route_leg_ids:[...new Set(rendered)].sort(),draft_markers:markers};
    }""", coordinate)


def assert_overlay(observed, operation):
    if operation is None:
        assert observed["draft_routes"]["features"] == []
        assert observed["draft_markers"] == []
        assert observed["rendered_draft_route_leg_ids"] == []
        return
    legs = operation["parameters"]["affected_route_legs"]
    features = observed["draft_routes"]["features"]
    assert observed["draft_route_source_loaded"]
    assert len(features) == len(legs) == 2
    assert {feature["properties"]["route_leg_id"] for feature in features} == {leg["route_leg_id"] for leg in legs}
    for feature in features:
        leg = next(row for row in legs if row["route_leg_id"] == feature["properties"]["route_leg_id"])
        assert feature["geometry"] == leg["geometry"]
        assert feature["properties"]["road_validated"] is True
        assert feature["properties"]["display_role"] == "route_checked_draft"
        assert leg["route_leg_id"] == "preview_leg_" + leg["query_hash"][:16]
    assert set(observed["rendered_draft_route_leg_ids"]) == {leg["route_leg_id"] for leg in legs}, "Both actual affected route legs must be drawn in the fitted viewport"
    assert len(observed["draft_markers"]) == 1 and observed["draft_markers"][0]["visible"]
    assert operation["parameters"]["name"] in observed["draft_markers"][0]["label"]
    assert observed["draft_markers"][0]["expected_coordinate"] == operation["parameters"]["snapped_coordinate"]
    assert observed["draft_markers"][0]["distance_from_expected_pixels"] <= 1.5
