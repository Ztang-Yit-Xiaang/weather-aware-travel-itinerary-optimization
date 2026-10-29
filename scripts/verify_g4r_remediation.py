from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat
from playwright.sync_api import Page, sync_playwright

BASE_URL = "http://127.0.0.1:8127"
OUT_DIR = Path("results/product_app/g4r_remediation_20260811")
VIEWPORTS = ((1440, 900), (1280, 800), (1024, 768), (390, 844))
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_state(page: Page, selector: str = "#geographic-map") -> dict[str, Any]:
    return page.eval_on_selector(
        selector,
        """node => {
          const rect = node.getBoundingClientRect();
          return {
            ready: node.dataset.mapReady === 'true',
            ready_at: node.dataset.mapReadyAt || null,
            tile_coverage: Number(node.dataset.mapTileCoverage || '0'),
            width: Math.round(rect.width), height: Math.round(rect.height),
            top: Math.round(rect.top), bottom: Math.round(rect.bottom),
            canvas_count: node.querySelectorAll('canvas').length,
          };
        }""",
    )


def wait_for_map(page: Page, selector: str = "#geographic-map") -> dict[str, Any]:
    page.locator(f'{selector}[data-map-ready="true"]').wait_for(state="visible", timeout=45_000)
    state = map_state(page, selector)
    assert state["tile_coverage"] >= 0.95, state
    assert state["ready_at"], state
    assert state["canvas_count"] > 0, state
    return state


def wait_for_map_after(page: Page, previous_ready_at: str, selector: str = "#geographic-map") -> dict[str, Any]:
    page.wait_for_function(
        """([sel, previous]) => {
          const node = document.querySelector(sel);
          return node?.dataset.mapReady === 'true'
            && node.dataset.mapReadyAt
            && node.dataset.mapReadyAt !== previous;
        }""",
        arg=[selector, previous_ready_at],
        timeout=45_000,
    )
    state = map_state(page, selector)
    assert state["tile_coverage"] >= 0.95, state
    return state


def screenshot(page: Page, name: str, ready_at: str | None) -> dict[str, Any]:
    captured_at = utc_now()
    assert ready_at is None or datetime.fromisoformat(ready_at) <= datetime.fromisoformat(captured_at)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False, animations="disabled")
    with Image.open(path) as image:
        dimensions = image.size
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "dimensions": dimensions,
        "captured_at": captured_at,
        "map_ready_at": ready_at,
    }


def map_pixel_evidence(page: Page, name: str) -> dict[str, Any]:
    path = OUT_DIR / f"{name}_map.png"
    page.locator("#geographic-map").screenshot(path=str(path), animations="disabled")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        extrema = rgb.getextrema()
        grayish = sum(
            1 for red, green, blue in rgb.get_flattened_data() if max(red, green, blue) - min(red, green, blue) <= 8
        ) / max(1, rgb.width * rgb.height)
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "dimensions": rgb.size,
        "channel_stddev": [round(value, 3) for value in stat.stddev],
        "channel_extrema": extrema,
        "near_gray_fraction": round(grayish, 6),
    }


def page_state(page: Page) -> dict[str, Any]:
    return page.evaluate(
        """() => ({
          route: location.pathname,
          viewport: {width: innerWidth, height: innerHeight},
          document_height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
          nav_labels: [...document.querySelectorAll('.navigation-rail [data-route]')]
            .map(node => node.textContent.trim()),
          mobile_nav_visible: getComputedStyle(document.querySelector('.mobile-nav')).display !== 'none',
          selected_day: document.querySelector('.day-card[aria-pressed="true"]')?.dataset.day || null,
          selected_alternative: document.querySelector('[data-compare-option][aria-pressed="true"]')?.dataset.compareOption || null,
        })"""
    )


def open_app(page: Page, width: int, height: int) -> dict[str, Any]:
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"{BASE_URL}/app/itinerary", wait_until="domcontentloaded")
    page.locator("#loading-screen").wait_for(state="hidden", timeout=30_000)
    return wait_for_map(page)


def verify() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "g4r-remediation-live-evidence-v1",
        "started_at": utc_now(),
        "browser_surface": "project-installed Chromium fallback",
        "in_app_browser_limitation": "Desktop browser runtime setup failed before discovery.",
        "checks": [],
        "console_errors": [],
        "network_failures": [],
    }
    assert CHROME.is_file(), CHROME

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, executable_path=str(CHROME))
        context = browser.new_context()
        page = context.new_page()

        page.on(
            "console",
            lambda message: report["console_errors"].append(message.text) if message.type == "error" else None,
        )
        page.on(
            "requestfailed",
            lambda request: report["network_failures"].append({"url": request.url, "failure": request.failure}),
        )

        initial = open_app(page, 1440, 900)
        state = page_state(page)
        assert state["nav_labels"] == ["ITItinerary", "MPMap", "RPRepairs", "CPCompare", "EVEvidence"]
        assert 560 <= initial["height"] <= 760, initial
        report["checks"].append(
            {
                "name": "trip_initial_1440x900",
                "map": initial,
                "page": state,
                "pixels": map_pixel_evidence(page, "trip_initial_1440x900"),
                "screenshot": screenshot(page, "trip_initial_1440x900", initial["ready_at"]),
            }
        )

        page.reload(wait_until="domcontentloaded")
        page.locator("#loading-screen").wait_for(state="hidden", timeout=30_000)
        reloaded = wait_for_map(page)
        report["checks"].append(
            {
                "name": "trip_after_reload_1440x900",
                "map": reloaded,
                "pixels": map_pixel_evidence(page, "trip_after_reload_1440x900"),
                "screenshot": screenshot(page, "trip_after_reload_1440x900", reloaded["ready_at"]),
            }
        )

        page.locator('.navigation-rail [data-route="/app/compare"]').click()
        page.wait_for_url("**/app/compare")
        page.locator('[data-compare-map-pane="baseline"] [data-map-ready="true"]').wait_for(timeout=45_000)
        page.locator('[data-compare-map-pane="option"] [data-map-ready="true"]').wait_for(timeout=45_000)
        choose = page.locator("#compare-choose")
        assert choose.is_enabled()
        assert choose.inner_text() == "Choose option for Trip preview"
        assert page.locator('[data-compare-option][aria-pressed="true"]').count() == 1
        compare_maps = [
            map_state(page, '[data-compare-map-pane="baseline"] [data-compare-map-canvas]'),
            map_state(page, '[data-compare-map-pane="option"] [data-compare-map-canvas]'),
        ]
        assert all(item["tile_coverage"] >= 0.95 for item in compare_maps), compare_maps
        compare_shot = screenshot(
            page,
            "compare_recommended_1440x900",
            max(item["ready_at"] for item in compare_maps),
        )
        report["checks"].append(
            {
                "name": "compare_recommended_1440x900",
                "maps": compare_maps,
                "page": page_state(page),
                "screenshot": compare_shot,
            }
        )
        choose.click()
        page.wait_for_url("**/app/itinerary")
        returned = wait_for_map(page)
        report["checks"].append(
            {
                "name": "compare_choose_returns_trip",
                "map": returned,
                "screenshot": screenshot(page, "compare_choose_returns_trip", returned["ready_at"]),
            }
        )

        page.locator("#open-copilot").click()
        page.locator("#copilot-dock").wait_for(state="visible")
        chips = page.locator("#context-chips").inner_text()
        assert "plan_" not in chips, chips
        copilot_shot = screenshot(page, "copilot_open_1440x900", returned["ready_at"])
        page.locator("#close-copilot").click()
        page.locator("#copilot-dock").wait_for(state="hidden")
        copilot_closed = wait_for_map(page)
        report["checks"].append(
            {
                "name": "copilot_open_close",
                "context": chips,
                "map": copilot_closed,
                "open_screenshot": copilot_shot,
                "closed_screenshot": screenshot(page, "copilot_closed_1440x900", copilot_closed["ready_at"]),
            }
        )

        page.set_viewport_size({"width": 1280, "height": 800})
        text_map = wait_for_map_after(page, copilot_closed["ready_at"])
        details = page.locator("details.textual-map-alternative")
        details.locator("summary").click()
        route_text = details.inner_text()
        assert "Stop unavailable" not in route_text, route_text
        assert "Original route" in route_text and "Proposed route" in route_text
        report["checks"].append(
            {
                "name": "text_route_panel_1280x800",
                "map": text_map,
                "text": route_text,
                "screenshot": screenshot(page, "text_route_panel_1280x800", text_map["ready_at"]),
            }
        )

        page.set_viewport_size({"width": 1440, "height": 900})
        wait_for_map_after(page, text_map["ready_at"])
        page.locator('.navigation-rail [data-route="/app/evidence"]').click()
        page.wait_for_url("**/app/evidence")
        report["checks"].append(
            {"name": "evidence_open", "screenshot": screenshot(page, "evidence_open_1440x900", None)}
        )
        page.locator("#evidence-back").click()
        page.wait_for_url("**/app/itinerary")
        after_evidence = wait_for_map(page)
        report["checks"].append(
            {
                "name": "evidence_close",
                "map": after_evidence,
                "screenshot": screenshot(page, "evidence_closed_1440x900", after_evidence["ready_at"]),
            }
        )

        resize_states = []
        previous_ready_at = after_evidence["ready_at"]
        for width, height in ((1024, 768), (1440, 900)):
            page.set_viewport_size({"width": width, "height": height})
            resized = wait_for_map_after(page, previous_ready_at)
            resize_states.append(
                {
                    "map": resized,
                    "screenshot": screenshot(page, f"resize_{width}x{height}", resized["ready_at"]),
                }
            )
            previous_ready_at = resized["ready_at"]
        report["checks"].append({"name": "resize_1440_1024_1440", "states": resize_states})

        for width, height in VIEWPORTS[1:]:
            state_map = open_app(page, width, height)
            state_page = page_state(page)
            if width == 390:
                assert state_page["mobile_nav_visible"]
                assert state_map["height"] >= 500, state_map
            report["checks"].append(
                {
                    "name": f"trip_{width}x{height}",
                    "map": state_map,
                    "page": state_page,
                    "pixels": map_pixel_evidence(page, f"trip_{width}x{height}"),
                    "screenshot": screenshot(page, f"trip_{width}x{height}", state_map["ready_at"]),
                }
            )

        assert not report["console_errors"], report["console_errors"]
        assert not report["network_failures"], report["network_failures"]
        browser.close()

    report["ended_at"] = utc_now()
    report["verdict"] = "implementation_live_checks_passed_pending_independent_g4r_reaudit"
    (OUT_DIR / "live_evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = verify()
    print(json.dumps({"verdict": result["verdict"], "checks": len(result["checks"])}, indent=2))
