from __future__ import annotations

import json
import statistics
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, expect
from PIL import Image

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

BASE_URL = "http://127.0.0.1:8127/app/itinerary"
OUTPUT = Path(r"results\stability_pass_8127")

@dataclass
class Capture:
    label: str
    width: int
    height: int
    path: Path
    state: dict[str, Any]
    issues: list[str] = field(default_factory=list)


def analyze_map_visibility(path: Path, page, selector: str) -> dict[str, float | int | str | bool]:
    # Read map container screenshot and compute simple visual coverage heuristics.
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        pixels = list(rgb.getdata())
        total = w * h
        if total == 0:
            return {"coverage_ratio": 0.0, "gray_ratio": 1.0, "width": w, "height": h, "size_bytes": path.stat().st_size}
        gray = 0
        near_gray = 0
        for r, g, b in pixels:
            if abs(r - g) <= 16 and abs(g - b) <= 16 and abs(r - b) <= 16:
                near_gray += 1
            if 200 <= r <= 250 and 200 <= g <= 250 and 200 <= b <= 250:
                gray += 1
        # rough heuristic: not loaded / blank is high gray
        coverage = 1.0 - (near_gray / total)
        gray_ratio = gray / total

    rect = page.eval_on_selector(
        selector,
        "el => {\n          const m = el.getBoundingClientRect();\n          let maxCanvasArea = 0;\n          let totalCanvasArea = 0;\n          const canv = Array.from(el.querySelectorAll('canvas'));\n          const containerArea = Math.max(1, Math.round(m.width * m.height));\n          canv.forEach(c => {\n            const r = c.getBoundingClientRect();\n            const ix0 = Math.max(m.left, r.left);\n            const iy0 = Math.max(m.top, r.top);\n            const ix1 = Math.min(m.right, r.right);\n            const iy1 = Math.min(m.bottom, r.bottom);\n            const iArea = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);\n            totalCanvasArea += Math.round(iArea);\n            const cArea = Math.max(0, Math.round(r.width * r.height));\n            if (cArea > maxCanvasArea) maxCanvasArea = cArea;\n          });\n          return {\n            mapWidth: Math.round(m.width),\n            mapHeight: Math.round(m.height),\n            hasCanvas: canv.length > 0,\n            canvasCount: canv.length,\n            maxCanvasArea: maxCanvasArea,\n            coverageArea: totalCanvasArea,\n            containerArea: containerArea,\n            containerY: Math.round(m.top),\n            canvasTopMin: (el.querySelector('canvas') ? Math.round(el.querySelector('canvas').getBoundingClientRect().top) : 0),\n            canvasTopOffset: (el.querySelector('canvas') ? Math.round(Math.max(0, el.querySelector('canvas').getBoundingClientRect().top - m.top)) : 0),\n          };\n        }",
    )
    rect["coverage_ratio_area"] = (rect["coverageArea"] / rect["containerArea"]) if rect.get("containerArea") else 0.0
    rect["has_canvas"] = bool(rect.get("hasCanvas"))
    rect["coverage_ratio_color"] = coverage
    rect["coverage_gray_ratio"] = gray_ratio
    rect["width"] = w
    rect["height"] = h
    rect["size_bytes"] = path.stat().st_size
    return rect


def wait_for_map_ready(page, timeout_ms=45000):
    # Block until map container exists, canvas is laid out, and map status is ready or warning.
    page.wait_for_selector("#geographic-map", timeout=timeout_ms)
    page.wait_for_function(
        """
        () => {
          const map = document.querySelector('#geographic-map');
          if (!map) return false;
          const status = document.querySelector('#map-render-status');
          const statusText = (status?.textContent || '').trim();
          const statusVisible = status && !status.hidden && getComputedStyle(status).display !== 'none' && status.offsetHeight > 0;
          const mapRect = map.getBoundingClientRect();
          const canvases = map.querySelectorAll('canvas');
          if (canvases.length === 0) return false;
          let covered = 0;
          let mapArea = Math.max(1, mapRect.width * mapRect.height);
          for (const c of canvases) {
            const r = c.getBoundingClientRect();
            const ix0 = Math.max(mapRect.left, r.left);
            const iy0 = Math.max(mapRect.top, r.top);
            const ix1 = Math.min(mapRect.right, r.right);
            const iy1 = Math.min(mapRect.bottom, r.bottom);
            const iArea = Math.max(0, (ix1 - ix0) * (iy1 - iy0));
            covered = Math.max(covered, iArea);
          }
          const areaCoverage = covered / mapArea;
          const readyText = !statusVisible
            || statusText.includes('Only the accepted-plan route is available in this pane.')
            || statusText.includes('required route connection');
          const notFailed = !statusText.includes('could not be verified');
          return status !== null && areaCoverage >= 0.90 && readyText && notFailed;
        }
        """,
        timeout=timeout_ms,
    )


def screenshot_and_log(page, label: str, selector: str, width: int, height: int, out_dir: Path, report: dict[str, Any]):
    ts = datetime.now(timezone.utc).isoformat()
    out_file = out_dir / f"{label}.png"
    if selector:
        page.locator(selector).scroll_into_view_if_needed()
        page.locator(selector).screenshot(path=str(out_file), animations='disabled')
        state = analyze_map_visibility(out_file, page, selector)
    else:
        page.screenshot(path=str(out_file), full_page=True)
        state = {"full_page": True, "size_bytes": out_file.stat().st_size}
    state = dict(state)
    state["screenshot_timestamp_utc"] = ts
    state["viewport"] = {"width": width, "height": height}
    state["url"] = page.url
    state["title"] = page.title()
    return Capture(label, width, height, out_file, state=state)


def safe_text(page, selector):
    try:
        locator = page.locator(selector)
        if locator.count() == 0:
            return None
        return locator.inner_text()
    except Exception:
        return None


def collect_timeline_state(page):
    cards = page.locator(".day-card")
    day_cards = cards.all_inner_texts() if cards.count() else []
    stop_count = page.locator("[data-geo-stop]").count()
    route_button_count = page.locator("[data-geo-route-leg]").count()
    map_mode = safe_text(page, ".map-mode-switch [aria-pressed='true']")
    selected_day = safe_text(page, ".toolbar-day-badge")
    return {
        "timeline_days": [t.strip() for t in day_cards],
        "visible_stop_button_count": stop_count,
        "visible_route_button_count": route_button_count,
        "selected_day_marker": selected_day,
        "map_mode_active": map_mode,
    }


def collect_navigation_state(page):
    active_nav = [
        el.inner_text().strip()
        for el in page.locator("button.nav-item.is-active").all()
    ]
    return {"active_nav": active_nav}


def run_case(page, base_label, width, height, actions):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#app-shell")).to_be_visible(timeout=15000)
    page.wait_for_selector("#loading-screen", state="hidden", timeout=15000)

    # wait before any screenshot
    page.wait_for_timeout(300)

    # module-specific actions
    actions(page)

    # capture basic runtime signals
    console_errors = actions.console_errors if hasattr(actions, "console_errors") else []
    request_failures = actions.request_failures if hasattr(actions, "request_failures") else []

    # attempt final map-ready wait before screenshot if geographic map exists in current route
    route = page.url
    if "/app/itinerary" in route:
        wait_for_map_ready(page)
        capture = screenshot_and_log(page, base_label, "#geographic-map", width, height, OUTPUT, {})
    elif "/app/compare" in route:
        # wait for both compare map canvases where visible
        page.wait_for_function(
            """
            () => {
              const statuses = document.querySelectorAll('[data-compare-map-status]');
              if (statuses.length === 0) return false;
              const visible = Array.from(statuses).filter(s => getComputedStyle(s).display !== 'none');
              return visible.length >= 1;
            }
            """,
            timeout=45000,
        )
        capture = screenshot_and_log(page, base_label, ".route-workspace", width, height, OUTPUT, {})
    else:
        capture = screenshot_and_log(page, base_label, "#main-workspace", width, height, OUTPUT, {})

    # map container and selected selection diagnostics for relevant pages
    diag = {}
    if "/app/itinerary" in route:
        diag = page.evaluate(
            """() => {
              const map = document.querySelector('#geographic-map');
              const status = document.querySelector('#map-render-status');
              return {
                mapStatusText: (status?.textContent || '').trim(),
                mapStatusHidden: status ? status.hidden : null,
                mapWidth: map ? Math.round(map.getBoundingClientRect().width) : 0,
                mapHeight: map ? Math.round(map.getBoundingClientRect().height) : 0,
                mapTop: map ? Math.round(map.getBoundingClientRect().top) : 0,
                mapBottom: map ? Math.round(map.getBoundingClientRect().bottom) : 0,
                viewportHeight: Math.round(window.innerHeight),
                selectedDayId: window.state?.selected_day || null,
                selectedSegmentId: window.state?.selected_segment_id || null,
                selectedStopId: window.state?.selected_stop_id || null,
                selectedAlternativeId: window.state?.selected_alternative_id || null,
              };
            }"
        )
    cap = capture
    cap.state.update({
        "route": route,
        "timeline": collect_timeline_state(page) if "/app/itinerary" in route else {},
        "navigation": collect_navigation_state(page),
        "diagnostics": diag,
        "console_errors": console_errors,
        "network_failures": request_failures,
    })

    # evaluate coverage rule in code
    if "/app/itinerary" in route:
        vis = cap.state.get("coverage_ratio_area", 0)
        cap.issues = []
        if vis < 0.95:
            cap.issues.append(f"map area coverage {vis:.3f} below 95%")
    return cap


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exe = next((str(p) for p in CHROME_CANDIDATES if p.is_file()), None)
    if not exe:
        raise RuntimeError("No Chrome/Edge executable found")

    report: dict[str, Any] = {
        "base_url": BASE_URL,
        "starts": datetime.now(timezone.utc).isoformat(),
        "captures": [],
        "failures": [],
    }

    with sync_playwright() as p:
        with p.chromium.launch(headless=True, executable_path=exe) as browser:
            page = browser.new_page(viewport={"width": 1280, "height": 720})

            console_errors: list[str] = []
            network_failures: list[str] = []

            def on_console(msg):
                if msg.type == "error":
                    loc = msg.location or {}
                    console_errors.append(f"{msg.type.upper()}: {msg.text} [{loc.get('url')}:{loc.get('lineNumber')}]".strip())

            def on_pageerror(err):
                network_failures.append(f"pageerror: {err}")

            def on_req_failed(req):
                network_failures.append(f"{req.url} -> {req.failure.error_text if req.failure else 'failed'}")

            page.on("console", on_console)
            page.on("pageerror", on_pageerror)
            page.on("requestfailed", on_req_failed)

            # attach for closures
            action = lambda x: None  # noqa: E731
            action.console_errors = console_errors
            action.request_failures = network_failures

            # Case 1: initial trip 1440x900
            cap = run_case(page, "trip_initial_1440x900", 1440, 900, action)
            report["captures"].append(_serialize_capture(cap))

            # Case 2: reload 1440x900
            def on_reload(pg):
                pg.reload(wait_until="domcontentloaded")
                pg.wait_for_selector("#app-shell", timeout=20000)
                pg.wait_for_selector("#loading-screen", state="hidden", timeout=15000)

            cap = run_case(page, "trip_after_reload_1440x900", 1440, 900, on_reload)
            report["captures"].append(_serialize_capture(cap))

            # Case 3: map/text route panel + module switching at 1280x800
            def on_text_route_and_switch(pg):
                # open text route description
                summary = pg.locator(".textual-map-alternative summary")
                if summary.count():
                    summary.first.click()
                    pg.wait_for_timeout(250)
                # switch to Compare and back to Trip
                pg.get_by_role("button", name="CPCompare").click()
                pg.wait_for_url("**/app/compare", timeout=20000)
                pg.wait_for_selector("main", timeout=10000)
                pg.wait_for_timeout(800)
                # return to trip
                pg.get_by_role("button", name="ITItinerary").click()
                pg.wait_for_url("**/app/itinerary", timeout=20000)

            cap = run_case(page, "text_route_panel_and_switch_1280x800", 1280, 800, on_text_route_and_switch)
            report["captures"].append(_serialize_capture(cap))

            # Case 4: compare selected recommendation at 1024x768
            def on_compare(pg):
                pg.get_by_role("button", name="CPCompare").click()
                pg.wait_for_url("**/app/compare", timeout=20000)
                # pick selected option if any
                chosen = pg.locator('[data-compare-option][aria-pressed="true"]')
                if chosen.count() > 0:
                    chosen.first.click()
                else:
                    # prefer recommended label
                    rec = pg.locator(".alternative-card.recommended [data-compare-option]")
                    if rec.count() > 0:
                        rec.first.click()
                pg.wait_for_timeout(800)

            cap = run_case(page, "compare_selected_recommendation_1024x768", 1024, 768, on_compare)
            report["captures"].append(_serialize_capture(cap))

            # Case 5: module switches + resize stress at 1280 then 1440
            def on_resize(pg):
                pg.get_by_role("button", name="RPRepairs").click()
                pg.wait_for_url("**/app/repairs", timeout=20000)
                pg.wait_for_timeout(400)
                pg.set_viewport_size({"width": 1024, "height": 768})
                pg.wait_for_timeout(500)
                pg.set_viewport_size({"width": 1440, "height": 900})
                pg.wait_for_timeout(600)
                pg.get_by_role("button", name="ITItinerary").click()
                pg.wait_for_url("**/app/itinerary", timeout=20000)

            cap = run_case(page, "module_switch_resize_1440x900", 1440, 900, on_resize)
            report["captures"].append(_serialize_capture(cap))

            # Case 6: mobile 390x844 with open copilot
            def on_mobile(pg):
                # ensure mobile view menu state
                try:
                    pg.set_viewport_size({"width": 390, "height": 844})
                    if pg.locator("#mobile-copilot").count():
                        pg.locator("#mobile-copilot").click()
                        pg.wait_for_timeout(250)
                    if pg.locator("button[data-map-mode='edit']").count():
                        pg.locator("button[data-map-mode='edit']").click()
                except Exception:
                    pass

            cap = run_case(page, "mobile_390x844", 390, 844, on_mobile)
            report["captures"].append(_serialize_capture(cap))

            # collect one additional map container dimension audit by waiting window resize sequence
            # (map gap stress check)
            if True:
                page.set_viewport_size({"width": 1440, "height": 900})
                page.wait_for_timeout(1000)
                d1 = page.evaluate("""() => {
                    const map = document.querySelector('#geographic-map');
                    if (!map) return null;
                    const r = map.getBoundingClientRect();
                    return {width: Math.round(r.width), height: Math.round(r.height)};
                }""")
                page.set_viewport_size({"width": 1024, "height": 768})
                page.wait_for_timeout(1000)
                d2 = page.evaluate("""() => {
                    const map = document.querySelector('#geographic-map');
                    if (!map) return null;
                    const r = map.getBoundingClientRect();
                    return {width: Math.round(r.width), height: Math.round(r.height)};
                }""")
                page.set_viewport_size({"width": 1440, "height": 900})
                page.wait_for_timeout(1000)
                d3 = page.evaluate("""() => {
                    const map = document.querySelector('#geographic-map');
                    if (!map) return null;
                    const r = map.getBoundingClientRect();
                    return {width: Math.round(r.width), height: Math.round(r.height)};
                }""")
                report["resize_sequence_map_dims"] = [d1, d2, d3]

            # Evaluate selection coherence checks
            page.goto(f"{BASE_URL}/../itinerary", wait_until="domcontentloaded")
            page.wait_for_selector("#app-shell", timeout=15000)
            page.wait_for_selector("#loading-screen", state="hidden", timeout=15000)
            # pick day 4 if available
            if page.locator('.day-card[data-day="4"]').count():
                page.locator('.day-card[data-day="4"]').click()
                page.wait_for_timeout(400)
            if page.locator('[data-nav="/app/map"]').count():
                page.locator('[data-nav="/app/map"]').click()
                page.wait_for_timeout(500)
                page.locator('[data-nav="/app/itinerary"]').click()
                page.wait_for_timeout(400)

            report["final_state"] = {
                "selected_state": page.evaluate("() => ({\n      selected_day: window.state?.selected_day || null,\n      selected_stop_id: window.state?.selected_stop_id || null,\n      selected_segment_id: window.state?.selected_segment_id || null,\n      selected_candidate_id: window.state?.selected_candidate_id || null,\n      selected_alternative_id: window.state?.selected_alternative_id || null,\n    })"),
                "route_text_example": safe_text(page, "details.textual-map-alternative p")[:500] if safe_text(page, "details.textual-map-alternative p") else None,
            }

            report["ends"] = datetime.now(timezone.utc).isoformat()

    out_json = OUTPUT / "stability_pass_report.json"
    out_json.write_text(json.dumps(_prepare_output(report), indent=2), encoding="utf-8")


def _serialize_capture(cap: Capture) -> dict[str, Any]:
    return {
        "label": cap.label,
        "width": cap.width,
        "height": cap.height,
        "screenshot": str(cap.path),
        "state": cap.state,
        "issues": cap.issues,
    }


def _prepare_output(report: dict[str, Any]) -> dict[str, Any]:
    return report


if __name__ == "__main__":
    main()
