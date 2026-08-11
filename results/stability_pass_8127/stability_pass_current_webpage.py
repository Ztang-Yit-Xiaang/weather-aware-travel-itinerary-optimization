from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
BASE_URL = "http://127.0.0.1:8127"
OUT_DIR = Path("results") / "stability_pass_8127"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_browser() -> str | None:
    for path in CHROME_PATHS:
        p = Path(path)
        if p.is_file():
            return str(p)
    return None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coverage_from_pixels(img_path: Path) -> float:
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        total = w * h
        if total == 0:
            return 0.0
        raw = list(rgb.getdata())
        non_gray = 0
        for r, g, b in raw:
            if abs(r - g) > 16 or abs(g - b) > 16 or abs(r - b) > 16:
                non_gray += 1
        return non_gray / total


def _map_diagnostics(page, selector: str) -> dict[str, Any]:
    return page.evaluate(
        """(sel) => {
            const map = document.querySelector(sel);
            if (!map) return {present:false};
            const statusEl = document.querySelector('#map-render-status');
            const statusText = (statusEl?.textContent || '').trim();
            const rect = map.getBoundingClientRect();
            const canvases = Array.from(map.querySelectorAll('canvas'));
            let covered = 0;
            let totalCanvasArea = 0;
            let visibleCanvases = 0;
            for (const canvas of canvases) {
                if (!canvas || !canvas.offsetWidth || !canvas.offsetHeight) continue;
                visibleCanvases += 1;
                const cr = canvas.getBoundingClientRect();
                const ix0 = Math.max(rect.left, cr.left);
                const iy0 = Math.max(rect.top, cr.top);
                const ix1 = Math.min(rect.right, cr.right);
                const iy1 = Math.min(rect.bottom, cr.bottom);
                const a = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
                if (a > covered) covered = a;
                totalCanvasArea += Math.max(0, Math.round(cr.width * cr.height));
            }
            const mapArea = Math.max(1, Math.round(rect.width * rect.height));
            return {
                present: true,
                statusText,
                statusHidden: statusEl ? statusEl.hidden : true,
                mapWidth: Math.round(rect.width),
                mapHeight: Math.round(rect.height),
                mapTop: Math.round(rect.top),
                mapLeft: Math.round(rect.left),
                mapBottom: Math.round(rect.bottom),
                canvasCount: canvases.length,
                visibleCanvases,
                mapArea,
                coveredArea: Math.round(covered),
                totalCanvasArea,
                areaCoverageRatio: mapArea ? covered / mapArea : 0,
            };
        }""",
        selector,
    )


def _selection_state(page) -> dict[str, Any]:
    return page.evaluate(
        """() => {
            const selectedDay = document.querySelector('.day-card[aria-pressed="true"]');
            const selectedStop = document.querySelector('[data-geo-stop][aria-pressed="true"]');
            const selectedLeg = document.querySelector('[data-geo-route-leg][aria-pressed="true"]');
            const selectedCandidate = document.querySelector('[data-geo-candidate][aria-pressed="true"], [data-poi-candidate][aria-pressed="true"]');
            const selectedAlternative = document.querySelector('[data-compare-option][aria-pressed="true"]') ||
                document.querySelector('.alternative-card[data-recommended="true"] [data-compare-option]');
            return {
                selected_day_id: selectedDay ? selectedDay.getAttribute('data-day') : null,
                selected_stop_id: selectedStop ? selectedStop.getAttribute('data-geo-stop') : null,
                selected_segment_id: selectedLeg ? selectedLeg.getAttribute('data-geo-route-leg') : null,
                selected_candidate_id: selectedCandidate ? (selectedCandidate.getAttribute('data-geo-candidate') || selectedCandidate.getAttribute('data-poi-candidate') || null) : null,
                selected_alternative_id: selectedAlternative ? (selectedAlternative.getAttribute('data-compare-option') || selectedAlternative.getAttribute('data-alternative-id') || null) : null,
                current_route: location.pathname,
                route_text_open: !!document.querySelector('details.textual-map-alternative[open]'),
                toolbar_mode: (document.querySelector('.map-mode-switch [aria-pressed="true"]') || {}).textContent,
                viewport: {w: window.innerWidth, h: window.innerHeight},
                documentHeight: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight),
            };
        }""",
    )


def _active_nav(page) -> list[str]:
    return page.eval_on_selector_all(
        "button[data-nav].is-active",
        "nodes => nodes.map((n) => n.textContent.trim())",
    )


def _click_nav_button(page, text_fragment: str) -> bool:
    """Best-effort click for stable nav labels across builds."""
    page.wait_for_timeout(100)
    if re.fullmatch(r"[A-Za-z]+", text_fragment):
        lowered = text_fragment.lower()
        patterns = {
            "itinerary": [r"itinerary", r"itin"],
            "map": [r"^mp", r"map"],
            "repairs": [r"repairs", r"rp"],
            "compare": [r"compare", r"cp"],
            "evidence": [r"evidence", r"ev"],
        }
        pats = patterns.get(lowered, [re.escape(lowered)])
        for pattern in pats:
            btn = page.locator("button", has_text=re.compile(pattern, re.I))
            if btn.count():
                btn.first.click()
                return True
        btn = page.locator(f'button[data-nav="{text_fragment}"]')
        if btn.count():
            btn.first.click()
            return True
    return False


def _wait_for_map_ready(page, timeout_ms: int = 45000, selector: str = "#geographic-map") -> tuple[bool, str, float]:
    deadline = time.time() + timeout_ms / 1000
    last_area = 0.0
    last_status = "waiting"
    while time.time() < deadline:
        try:
            page.wait_for_selector(selector, state="visible", timeout=2000)
        except Exception:
            time.sleep(0.2)
            continue

        diag = _map_diagnostics(page, selector)
        status = (diag.get("statusText") or "").lower()
        if "could not be verified" in status:
            return False, f"map_failed:{status}", diag.get("areaCoverageRatio", 0.0)
        if not status:
            # If status message not yet propagated, still permit initial geometry-only ready check.
            last_status = "no_status"
        else:
            last_status = status

        area = float(diag.get("areaCoverageRatio", 0.0))
        last_area = area
        if area >= 0.95 and ("ready" in status or "coverage" in status or "verified" in status):
            return True, status, area
        if diag.get("present") and diag.get("canvasCount", 0) > 0 and area > 0:
            # if canvas exists and status text still booting, continue
            pass
        time.sleep(0.25)
    return False, f"timeout_{last_status}_{last_area:.3f}", last_area


def _wait_for_compare_ready(page, timeout_ms: int = 45000) -> tuple[bool, str, list[float]]:
    deadline = time.time() + timeout_ms / 1000
    ratios: list[float] = []
    last = "waiting"
    while time.time() < deadline:
        stage_stats = page.evaluate(
            """() => {
                const scope = document.querySelectorAll('.compare-map-stage');
                const out = [];
                scope.forEach((pane, idx) => {
                    const status = pane.querySelector('[data-compare-map-status]');
                    const map = pane.querySelector('.geographic-map, .compare-map-canvas, [id="geographic-map"]') || pane;
                    const r = map.getBoundingClientRect();
                    const can = map.querySelectorAll('canvas');
                    let covered = 0;
                    can.forEach(c=>{
                        const cr = c.getBoundingClientRect();
                        const a = Math.max(0, Math.min(r.right, cr.right)-Math.max(r.left, cr.left)) *
                                  Math.max(0, Math.min(r.bottom, cr.bottom)-Math.max(r.top, cr.top));
                        covered = Math.max(covered, a);
                    });
                    out.push({
                        idx,
                        status: status ? (status.textContent || '').trim() : '',
                        visible: getComputedStyle(pane).display !== 'none',
                        mapWidth: Math.round(r.width),
                        mapHeight: Math.round(r.height),
                        ratio: r.width > 0 && r.height > 0 ? (covered / (r.width*r.height)) : 0,
                        canvasCount: can.length,
                    });
                });
                return out;
            }""",
        )
        if not stage_stats:
            time.sleep(0.2)
            continue
        vis = [s for s in stage_stats if s.get("visible")]
        if not vis:
            time.sleep(0.2)
            continue
        ratio_values = [float(s.get("ratio", 0.0)) for s in vis]
        status_text = " | ".join(s.get("status", "") for s in vis).lower()
        ratios = ratio_values
        if any(("could not be verified" in (s.get("status", "").lower()) for s in vis)):
            return False, f"compare_failed:{status_text}", ratio_values
        if all(v >= 0.95 for v in ratio_values) and all("loading" not in (s.get("status", "").lower()) for s in vis):
            return True, status_text, ratio_values
        last = status_text or "pending"
        time.sleep(0.25)
    return False, f"compare_timeout:{last}", ratios


def _capture_state_image(page, shot_path: Path, selector: str | None = None, full_page: bool = False) -> dict[str, Any]:
    if selector:
        target = page.locator(selector)
        target.wait_for(state="visible", timeout=10000)
        target.scroll_into_view_if_needed()
        target.screenshot(path=str(shot_path), animations="disabled")
    else:
        page.screenshot(path=str(shot_path), full_page=full_page, animations="disabled")
    state = {
        "screenshot_path": str(shot_path),
        "screenshot_bytes": shot_path.stat().st_size,
        "screenshot_timestamp_utc": _now_iso(),
        "screenshot_dimensions": {"w": 0, "h": 0},
    }
    try:
        with Image.open(shot_path) as im:
            w, h = im.size
            state["screenshot_dimensions"] = {"w": w, "h": h}
    except Exception:
        pass
    state["coverage_ratio"] = _coverage_from_pixels(shot_path)
    return state


def _run_scenario(page, name: str, width: int, height: int, action: Callable[[Any, dict[str, Any]], None], wait_compare: bool = False) -> dict[str, Any]:
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
    page.wait_for_selector("#app-shell", timeout=15000)
    page.wait_for_selector("#loading-screen", state="hidden", timeout=15000)

    logs: dict[str, Any] = {"label": name, "viewport": {"w": width, "h": height}, "steps": []}
    action(page, logs)
    page.wait_for_timeout(250)

    if wait_compare:
        ready, reason, ratios = _wait_for_compare_ready(page)
        shot_target = None
        logs["coverage_ratio"] = min(ratios) if ratios else 0.0
    else:
        ready, reason, ratio = _wait_for_map_ready(page)
        shot_target = "#geographic-map"
        logs["coverage_ratio"] = ratio

    logs["ready"] = bool(ready)
    logs["ready_reason"] = reason
    logs["issues"] = []
    if not ready or logs["coverage_ratio"] < 0.95:
        logs["issues"].append(f"map_coverage<{logs['coverage_ratio']:.3f} or not_ready:{reason}")

    shot_path = OUT_DIR / f"{name}_map.png"
    shot_meta = _capture_state_image(page, shot_path, selector=shot_target)
    if shot_target:
        shot_meta.update(_map_diagnostics(page, shot_target))

    logs["shot"] = shot_meta
    logs["page_shot"] = _capture_state_image(page, OUT_DIR / f"{name}_page.png", selector=None, full_page=False)

    # state snapshots
    logs["selected_state"] = _selection_state(page)
    logs["active_nav"] = _active_nav(page)
    logs["console_errors"] = page._console_errors.copy()
    logs["network_failures"] = page._network_failures.copy()
    logs["route"] = page.url
    logs["screenshot_after_ready"] = bool(logs["ready"])
    return logs


def _scenario_initial(pg, logs):
    return


def _scenario_reload(pg, logs):
    pg.reload(wait_until="domcontentloaded")
    pg.wait_for_selector("#app-shell", timeout=15000)
    pg.wait_for_selector("#loading-screen", state="hidden", timeout=15000)


def _scenario_compare_cycle(pg, logs):
    # trip -> compare -> trip
    if not _click_nav_button(pg, "compare"):
        if not _click_nav_button(pg, "CP"):
            logs["issues"].append("compare_nav_missing")
    pg.wait_for_timeout(400)
    pg.wait_for_url("**/app/compare", timeout=12000)
    pg.wait_for_timeout(700)
    if not _click_nav_button(pg, "itinerary"):
        if not _click_nav_button(pg, "IT"):
            logs["issues"].append("itinerary_nav_missing")
    pg.wait_for_url("**/app/itinerary", timeout=12000)


def _scenario_copilot(pg, logs):
    opener = pg.locator("#open-copilot")
    if opener.count() > 0:
        opener.click()
        pg.wait_for_selector("#copilot-dock", state="visible", timeout=12000)
        pg.wait_for_timeout(300)
        pg.locator("#close-copilot").click()
        pg.wait_for_selector("#copilot-dock", state="hidden", timeout=12000)
    else:
        logs["steps"].append("copilot_button_missing")


def _scenario_resize(pg, logs):
    if "/app/itinerary" not in pg.url:
        if not _click_nav_button(pg, "itinerary"):
            if not _click_nav_button(pg, "IT"):
                logs["issues"].append("itinerary_nav_missing")
        pg.wait_for_url("**/app/itinerary", timeout=12000)
    d1 = _map_diagnostics(pg, "#geographic-map")
    pg.set_viewport_size({"width": 1024, "height": 768})
    pg.wait_for_timeout(700)
    d2 = _map_diagnostics(pg, "#geographic-map")
    pg.set_viewport_size({"width": 1440, "height": 900})
    pg.wait_for_timeout(700)
    d3 = _map_diagnostics(pg, "#geographic-map")
    logs["steps"].append({"map_dimensions": [d1, d2, d3]})


def _scenario_compare_1024(pg, logs):
    if "/app/compare" not in pg.url:
        if not _click_nav_button(pg, "compare"):
            if not _click_nav_button(pg, "CP"):
                logs["issues"].append("compare_nav_missing")
    pg.wait_for_url("**/app/compare", timeout=12000)
    rec = pg.locator('.alternative-card[data-recommended="true"] [data-compare-option]')
    if rec.count() > 0:
        rec.first.click()
    pg.wait_for_timeout(500)


def _scenario_text_route(pg, logs):
    if "/app/itinerary" not in pg.url:
        if not _click_nav_button(pg, "itinerary"):
            if not _click_nav_button(pg, "IT"):
                logs["issues"].append("itinerary_nav_missing")
        pg.wait_for_url("**/app/itinerary", timeout=12000)
    summary = pg.locator(".textual-map-alternative summary")
    if summary.count() > 0:
        summary.first.click()
        pg.wait_for_timeout(400)


def _scenario_mobile(pg, logs):
    if "/app/itinerary" not in pg.url:
        if not _click_nav_button(pg, "itinerary"):
            if not _click_nav_button(pg, "IT"):
                logs["issues"].append("itinerary_nav_missing")
        pg.wait_for_url("**/app/itinerary", timeout=12000)
    mobile_copilot = pg.locator("#mobile-copilot")
    if mobile_copilot.count():
        mobile_copilot.click()
        if pg.locator("#copilot-dock").count():
            pg.wait_for_selector("#copilot-dock", state="visible", timeout=8000)
            pg.locator("#close-copilot").click()
            pg.wait_for_selector("#copilot-dock", state="hidden", timeout=8000)
    sel_btn = pg.locator("button[data-map-mode='select']")
    if sel_btn.count():
        sel_btn.first.click()


def _main() -> None:
    report: dict[str, Any] = {
        "base_url": BASE_URL,
        "started_utc": _now_iso(),
        "checks": [],
        "console_errors": [],
        "network_failures": [],
        "acceptance_criteria": {},
    }

    exe = _find_browser()
    if exe is None:
        raise RuntimeError("No Chrome/Edge browser found")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=exe)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page._console_errors = []
        page._network_failures = []

        def on_console(msg):
            if msg.type == "error":
                loc = msg.location or {}
                page._console_errors.append(f"console:{msg.text}::{loc.get('url', '')}:{loc.get('lineNumber', '')}")

        def on_request_failed(req):
            page._network_failures.append(f"requestfailed:{req.url}:{req.failure.error_text if req.failure else 'unknown'}")

        def on_response(resp):
            if resp.status >= 500 and ("/api/" in resp.url or "maplibre" in resp.url):
                page._network_failures.append(f"response:{resp.status}:{resp.url}")

        page.on("console", on_console)
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)

        c1 = _run_scenario(page, "trip_initial_1440x900", 1440, 900, _scenario_initial)
        c2 = _run_scenario(page, "trip_after_reload_1440x900", 1440, 900, _scenario_reload)
        c3 = _run_scenario(page, "trip_compare_trip_1440x900", 1440, 900, _scenario_compare_cycle)
        c4 = _run_scenario(page, "copilot_toggle_1440x900", 1440, 900, _scenario_copilot)
        c5 = _run_scenario(page, "resize_1440x1024_1440_900", 1440, 900, _scenario_resize)
        c6 = _run_scenario(page, "compare_recommended_1024x768", 1024, 768, _scenario_compare_1024, wait_compare=True)
        c7 = _run_scenario(page, "text_route_panel_1280x800", 1280, 800, _scenario_text_route)
        c8 = _run_scenario(page, "mobile_390x844", 390, 844, _scenario_mobile)

        report["checks"].extend([c1, c2, c3, c4, c5, c6, c7, c8])
        report["console_errors"] = page._console_errors
        report["network_failures"] = page._network_failures

        # acceptance criteria
        report["acceptance_criteria"]["fresh_load_complete_tiles"] = bool(c1["ready"] and c1["coverage_ratio"] >= 0.95)
        report["acceptance_criteria"]["reload_complete_tiles"] = bool(c2["ready"] and c2["coverage_ratio"] >= 0.95)
        report["acceptance_criteria"]["trip_compare_trip_no_gap"] = bool(c3["ready"] and c3["coverage_ratio"] >= 0.95)
        report["acceptance_criteria"]["copilot_open_close_no_gap"] = bool(c4["ready"] and c4["coverage_ratio"] >= 0.95)
        report["acceptance_criteria"]["resize_no_gap"] = bool(c5["ready"] and c5["coverage_ratio"] >= 0.95)
        report["acceptance_criteria"]["compare_1024_768"] = bool(c6["ready"])
        report["acceptance_criteria"]["text_route_visible"] = bool(c7["ready"])
        report["acceptance_criteria"]["mobile_view_captured"] = bool(c8["ready"])
        report["acceptance_criteria"]["coverage_ge_95_all_relevant"] = all(
            chk.get("coverage_ratio", 0.0) >= 0.95 for chk in [c1, c2, c3, c4, c5, c7, c8]
        )
        fresh_map = c1.get("shot", {}).get("mapHeight", 0)
        report["acceptance_criteria"]["map_height_560_760_oriented"] = 560 <= fresh_map <= 760

        report["acceptance_criteria"]["page_viewport_match"] = (
            c1["page_shot"]["screenshot_dimensions"]["w"] >= 1440
            and c1["page_shot"]["screenshot_dimensions"]["h"] >= 900
        )

        # extra geometry/logging requirements
        report["acceptance_criteria"]["no_screenshot_before_ready"] = all(
            chk.get("ready") and chk.get("screenshot_after_ready", False) for chk in report["checks"]
        )

        browser.close()

    report["ended_utc"] = _now_iso()
    out_json = OUT_DIR / "stability_pass_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # create poster using screenshots only
    image_paths = [
        OUT_DIR / "trip_initial_1440x900_page.png",
        OUT_DIR / "trip_after_reload_1440x900_page.png",
        OUT_DIR / "trip_compare_trip_1440x900_page.png",
        OUT_DIR / "copilot_toggle_1440x900_page.png",
        OUT_DIR / "resize_1440x1024_1440_900_page.png",
        OUT_DIR / "compare_recommended_1024x768_page.png",
        OUT_DIR / "text_route_panel_1280x800_page.png",
        OUT_DIR / "mobile_390x844_page.png",
    ]
    image_paths = [p for p in image_paths if p.exists()]

    if image_paths:
        with Image.open(image_paths[0]) as im0:
            base_w, base_h = im0.size
        thumb_w = 360
        thumb_h = int(base_h * (thumb_w / base_w))
        cols = 4
        rows = (len(image_paths) + cols - 1) // cols
        canvas_w = cols * (thumb_w + 180) + 40
        canvas_h = rows * (thumb_h + 60) + 100
        poster = Image.new("RGB", (canvas_w, canvas_h), (246, 246, 246))
        drawer = ImageDraw.Draw(poster)
        try:
            header_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            header_font = None
            label_font = None
        drawer.text((20, 20), "Current webpage stability evidence", fill=(16, 16, 16), font=header_font)
        drawer.text(
            (20, 52),
            "Source: live captures only, no synthetic images",
            fill=(64, 64, 64),
            font=label_font,
        )

        for i, src in enumerate(image_paths):
            x = 20 + (i % cols) * (thumb_w + 180)
            y = 90 + (i // cols) * (thumb_h + 60)
            with Image.open(src) as im:
                im = im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                poster.paste(im, (x, y))
            drawer.text((x, 0 + 0 + y + thumb_h + 4), src.stem, fill=(0, 0, 0), font=label_font)

        poster_path = OUT_DIR / "poster_current_webpage.png"
        poster.save(poster_path)
        report["poster_path"] = str(poster_path)


if __name__ == "__main__":
    _main()
