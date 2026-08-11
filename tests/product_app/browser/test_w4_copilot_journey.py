from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
Browser = playwright.Browser
Page = playwright.Page
expect = playwright.expect
sync_playwright = playwright.sync_playwright


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scripts" / "run_product_app.py"
CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"product launcher exited before readiness with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("product launcher did not expose /api/health within 15 seconds")


@pytest.fixture(scope="module")
def product_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    state_root = tmp_path_factory.mktemp("w4-browser-state")
    environment = os.environ.copy()
    environment.update(
        {
            "PRODUCT_APP_ORIGIN": base_url,
            "PRODUCT_COPILOT_ADAPTER": "deterministic",
            "PRODUCT_MAP_PROVIDER": "maplibre_pmtiles",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(LAUNCHER),
            "--port",
            str(port),
            "--state-root",
            str(state_root),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as runtime:
        executable = next((path for path in CHROME_CANDIDATES if path.is_file()), None)
        launch = {"headless": True}
        if executable is not None:
            launch["executable_path"] = str(executable)
        try:
            instance = runtime.chromium.launch(**launch)
        except Exception as exc:  # pragma: no cover - depends on local browser installation
            pytest.skip(f"No usable Chromium browser is installed: {exc}")
        yield instance
        instance.close()


def _open_app(browser: Browser, product_server: str, *, width: int, height: int = 900) -> tuple[Page, list[str]]:
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    errors: list[str] = []
    def capture_console(message) -> None:
        if message.type != "error":
            return
        location = message.location
        source = location.get("url") or "unknown-source"
        errors.append(f"console: {message.text} [{source}:{location.get('lineNumber', 0)}]")

    page.on("console", capture_console)
    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
    page.goto(f"{product_server}/app", wait_until="domcontentloaded")
    expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
    expect(page.locator("#loading-screen")).to_be_hidden()
    assert page.url.startswith(f"{product_server}/app")
    return page, errors


def _assert_no_horizontal_overflow(page: Page) -> None:
    dimensions = page.evaluate(
        """() => ({
          documentWidth: document.documentElement.scrollWidth,
          viewportWidth: document.documentElement.clientWidth,
          bodyWidth: document.body.scrollWidth,
        })"""
    )
    assert dimensions["documentWidth"] <= dimensions["viewportWidth"]
    assert dimensions["bodyWidth"] <= dimensions["viewportWidth"]


def _assert_composer_reachable(page: Page) -> None:
    textarea = page.locator("#copilot-input")
    send = page.locator("#copilot-form button[type='submit']")
    expect(textarea).to_be_visible()
    expect(send).to_be_visible()
    for control in (textarea, send):
        box = control.bounding_box()
        assert box is not None
        assert box["x"] >= 0
        assert box["y"] >= 0
        assert box["x"] + box["width"] <= page.viewport_size["width"]
        assert box["y"] + box["height"] <= page.viewport_size["height"]


def test_deterministic_copilot_persists_context_and_supports_deletion(
    browser: Browser,
    product_server: str,
    tmp_path: Path,
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        day_three = page.locator('.day-card[data-day="3"]')
        day_three.click()
        expect(day_three).to_have_attribute("aria-pressed", "true")

        opener = page.locator("#open-copilot")
        opener.focus()
        opener.press("Enter")
        expect(page.locator("#copilot-dock")).to_be_visible()
        expect(page.locator("#copilot-provider-label")).to_have_text("Deterministic demo")
        expect(page.locator("#copilot-disclosure")).to_contain_text("requests stay on this computer")
        expect(page.locator("#context-chips")).to_contain_text("Day 3")
        expect(page.locator("#context-chips")).to_contain_text("Griffith Observatory")
        expect(page.locator("#copilot-input")).to_be_focused()

        page.locator("#copilot-input").fill("Review a safer weather repair")
        page.locator("#copilot-form button[type='submit']").click()
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Proposal ready", timeout=15_000)
        expect(page.locator(".message-user")).to_have_count(1)
        expect(page.locator(".copilot-proposal-actions")).to_be_visible()

        page.locator("#close-copilot").click()
        expect(page.locator("#copilot-dock")).to_be_hidden()
        expect(opener).to_be_focused()
        opener.click()
        expect(page.locator(".message-user")).to_have_count(1)

        page.reload(wait_until="domcontentloaded")
        expect(page.locator("#app-shell")).to_be_visible(timeout=15_000)
        page.locator("#open-copilot").click()
        expect(page.locator(".message-user")).to_have_count(1)
        expect(page.locator("#context-chips")).to_contain_text("Day 3")
        expect(page.locator("#context-chips")).to_contain_text("Griffith Observatory")

        settings = page.locator("#transcript-settings-button")
        settings.click()
        expect(page.locator("#transcript-settings-dialog")).to_be_visible()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#delete-current-conversation").click()
        expect(page.locator("#transcript-settings-dialog")).to_be_hidden()
        expect(settings).to_be_focused()
        expect(page.locator(".message-user")).to_have_count(0)

        page.locator("#copilot-input").fill("Keep the original plan")
        page.locator("#copilot-form button[type='submit']").click()
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Proposal ready", timeout=15_000)
        expect(page.locator(".message-user")).to_have_count(1)
        settings.click()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#delete-all-conversations").click()
        expect(page.locator("#transcript-settings-dialog")).to_be_hidden()
        expect(settings).to_be_focused()
        expect(page.locator(".message-user")).to_have_count(0)

        assert errors == []
    except Exception:
        page.screenshot(path=str(tmp_path / "w4-desktop-failure.png"), full_page=True)
        raise
    finally:
        page.context.close()


@pytest.mark.parametrize("width", [1280, 430, 390, 360])
def test_copilot_is_reachable_without_overflow_at_required_widths(
    browser: Browser,
    product_server: str,
    tmp_path: Path,
    width: int,
) -> None:
    page, errors = _open_app(browser, product_server, width=width)
    try:
        opener = page.locator("#open-copilot")
        opener.click()
        expect(page.locator("#copilot-dock")).to_be_visible()
        _assert_no_horizontal_overflow(page)
        _assert_composer_reachable(page)

        page.locator("#close-copilot").click()
        expect(opener).to_be_focused()
        assert errors == []
    except Exception:
        page.screenshot(path=str(tmp_path / f"w4-{width}-failure.png"), full_page=True)
        raise
    finally:
        page.context.close()


def test_copilot_page_has_no_malformed_utf8(browser: Browser, product_server: str, tmp_path: Path) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        body = page.locator("body").inner_text()
        malformed_markers = ("�", "鈥", "鈰", "鈫", "脳", "鉁", "鈿")
        assert not [marker for marker in malformed_markers if marker in body]
        assert errors == []
    except Exception:
        page.screenshot(path=str(tmp_path / "w4-utf8-failure.png"), full_page=True)
        raise
    finally:
        page.context.close()


def test_provider_failure_remains_visible_after_conversation_restore(
    browser: Browser, product_server: str
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        page.route(
            "**/copilot/messages",
            lambda route: route.fulfill(
                status=504,
                content_type="application/json",
                body='{"detail":"openai_timeout"}',
            ),
        )
        page.locator("#open-copilot").click()
        page.locator("#copilot-input").fill("Review the repair")
        page.locator("#copilot-form button[type='submit']").click()

        expect(page.locator("#copilot-lifecycle")).to_contain_text("Failed")
        expect(page.locator("#copilot-lifecycle")).not_to_contain_text("Conversation restored")
        expect(page.locator("#copilot-input")).to_have_value("Review the repair")
        assert errors
        assert all("504" in error and "/copilot/messages" in error for error in errors)
    finally:
        page.context.close()


def test_reduced_height_mobile_copilot_traps_focus_and_closes_with_escape(
    browser: Browser, product_server: str
) -> None:
    page, errors = _open_app(browser, product_server, width=390, height=360)
    try:
        opener = page.locator("#open-copilot")
        opener.click()
        dock = page.locator("#copilot-dock")
        expect(dock).to_have_attribute("role", "dialog")
        expect(dock).to_have_attribute("aria-modal", "true")
        _assert_composer_reachable(page)

        send = page.locator("#copilot-form button[type='submit']")
        send.focus()
        send.press("Tab")
        expect(page.locator("#transcript-settings-button")).to_be_focused()

        page.keyboard.press("Escape")
        expect(dock).to_be_hidden()
        expect(opener).to_be_focused()
        assert errors == []
    finally:
        page.context.close()
