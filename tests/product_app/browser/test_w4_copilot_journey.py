from __future__ import annotations

import json
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
            raise AssertionError(f"product launcher exited before readiness with code {process.returncode}")
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
        expect(page.locator("#context-chips")).to_contain_text("Yosemite National Park")
        expect(page.locator("#copilot-input")).to_be_focused()

        page.locator("#copilot-input").fill("Replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park")
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
        expect(page.locator("#context-chips")).to_contain_text("Yosemite National Park")

        settings = page.locator("#transcript-settings-button")
        settings.click()
        expect(page.locator("#transcript-settings-dialog")).to_be_visible()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#delete-current-conversation").click()
        expect(page.locator("#transcript-settings-dialog")).to_be_hidden()
        expect(settings).to_be_focused()
        expect(page.locator(".message-user")).to_have_count(0)

        page.evaluate("""() => {
          const original = window.fetch.bind(window);
          window.__keepOriginalStream = null;
          window.fetch = async (...args) => {
            const response = await original(...args);
            if (response.url.endsWith('/copilot/messages')) {
              response.clone().text().then(raw => { window.__keepOriginalStream = raw; });
            }
            return response;
          };
        }""")
        page.locator("#copilot-input").fill("Keep the original plan")
        with page.expect_response(
            lambda response: response.url.endswith("/copilot/messages") and response.request.method == "POST"
        ) as kept:
            page.locator("#copilot-form button[type='submit']").click()
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Proposal ready", timeout=15_000)
        from test_e2e1_live_copilot_chat import _http_record, _sse_events

        page.wait_for_function("typeof window.__keepOriginalStream === 'string'")
        capture = _http_record(kept.value, response_body=page.evaluate("window.__keepOriginalStream"))
        (tmp_path / "keep_original_http_io.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")
        payload = _sse_events(capture["response_body_utf8"])[-1]["payload"]
        assert [row["type"] for row in payload["turn"]["interpretation"]["intents"]] == ["keep_original"]
        assert payload["session"]["draft"] == [] and payload["session"]["proposal"] is None
        expect(page.get_by_role("button", name="Add to draft", exact=True)).to_have_count(0)
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


def test_known_california_place_is_named_without_auto_action(
    browser: Browser,
    product_server: str,
    tmp_path: Path,
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        page.locator("#open-copilot").click()
        page.locator("#copilot-input").fill("what if I want to visit red wood")
        page.locator("#copilot-form button[type='submit']").click()

        expect(page.locator("#copilot-lifecycle")).to_contain_text("Clarification required", timeout=15_000)
        assistant = page.locator(".message-assistant").last
        expect(assistant).to_contain_text("Redwood National and State Parks")
        expect(assistant).to_contain_text("road access and route are not validated")
        expect(assistant).to_contain_text("Nothing was changed")
        expect(page.locator(".copilot-proposal-actions")).to_have_count(0)
        assert errors == []
    except Exception:
        page.screenshot(path=str(tmp_path / "w4-redwood-failure.png"), full_page=True)
        raise
    finally:
        page.context.close()


def test_user_can_stop_waiting_and_retry_same_message_identity(
    browser: Browser,
    product_server: str,
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        page.evaluate(
            """() => {
              window.__hciOriginalFetch = window.fetch.bind(window);
              window.__hciCapturedBodies = [];
              window.fetch = (input, init = {}) => {
                const url = typeof input === "string" ? input : input.url;
                if (url.endsWith("/copilot/messages")) {
                  window.__hciCapturedBodies.push(JSON.parse(init.body));
                  return new Promise((resolve, reject) => {
                    init.signal.addEventListener(
                      "abort",
                      () => reject(new DOMException("Aborted", "AbortError")),
                      { once: true },
                    );
                  });
                }
                return window.__hciOriginalFetch(input, init);
              };
            }"""
        )
        page.locator("#open-copilot").click()
        page.locator("#copilot-input").fill("Review a safer weather repair")
        page.locator("#copilot-submit").click()

        pending = page.locator('.message-user[data-delivery-state="sending"]')
        expect(pending).to_have_count(1)
        expect(page.locator("#copilot-submit")).to_have_attribute("aria-label", "Stop waiting for Copilot")
        page.locator("#copilot-submit").click()
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Stopped")
        expect(page.locator('.message-user[data-delivery-state="cancelled"]')).to_have_count(1)
        retry = page.get_by_role("button", name="Retry", exact=True)
        expect(retry).to_be_visible()
        first_id = page.evaluate("() => window.__hciCapturedBodies[0].client_message_id")

        page.evaluate("window.fetch = window.__hciOriginalFetch")
        with page.expect_response(
            lambda response: response.url.endswith("/copilot/messages") and response.request.method == "POST",
            timeout=30_000,
        ) as retried_info:
            retry.click()
        retried = retried_info.value
        assert retried.status == 200, retried.text()
        assert json.loads(retried.request.post_data or "{}")["client_message_id"] == first_id
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Proposal ready", timeout=15_000)
        expect(page.locator('.message-user[data-client-message-id="' + first_id + '"]')).to_have_count(1)
        assert not [error for error in errors if "/copilot/messages" in error or error.startswith("pageerror:")]
    finally:
        page.context.close()


def test_incomplete_stream_retry_reconciles_the_same_message_identity(
    browser: Browser,
    product_server: str,
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.__hciIncompleteStreamBodies = [];
              window.__hciIncompleteStreamInjected = false;
              window.fetch = (input, init = {}) => {
                const url = typeof input === "string" ? input : input.url;
                if (url.endsWith("/copilot/messages")) {
                  const body = JSON.parse(init.body);
                  window.__hciIncompleteStreamBodies.push(body);
                  if (!window.__hciIncompleteStreamInjected) {
                    window.__hciIncompleteStreamInjected = true;
                    const requestId = "copilot_request_incomplete_stream_test";
                    const events = [
                      {
                        schema_version: "copilot-stream-v1",
                        type: "message.accepted",
                        sequence: 1,
                        request_id: requestId,
                        client_message_id: body.client_message_id,
                      },
                      {
                        schema_version: "copilot-stream-v1",
                        type: "assistant.status",
                        sequence: 2,
                        request_id: requestId,
                        client_message_id: body.client_message_id,
                        state: "interpreting",
                        message: "Reading the current itinerary.",
                      },
                    ];
                    const raw = events.map(
                      (event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`,
                    ).join("");
                    return Promise.resolve(new Response(raw, {
                      status: 200,
                      headers: {
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Cache-Control": "no-store",
                      },
                    }));
                  }
                }
                return originalFetch(input, init);
              };
            }"""
        )
        page.locator("#open-copilot").click()
        page.locator("#copilot-input").fill("Review a safer weather repair")
        page.locator("#copilot-submit").click()

        expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", "cancelled", timeout=15_000)
        cancelled = page.locator('.message-assistant[data-delivery-state="cancelled"]').last
        expect(cancelled).to_contain_text("result was not confirmed")
        expect(cancelled).to_contain_text("same request identity")
        retry = cancelled.get_by_role("button", name="Retry")
        expect(retry).to_be_visible()
        expect(cancelled.get_by_role("button", name="Edit message")).to_be_visible()
        first_id = page.evaluate("() => window.__hciIncompleteStreamBodies[0].client_message_id")

        with page.expect_response(
            lambda response: response.url.endswith("/copilot/messages") and response.request.method == "POST",
            timeout=30_000,
        ) as retried_info:
            retry.click()
        retried = retried_info.value
        assert retried.status == 200, retried.text()
        bodies = page.evaluate("window.__hciIncompleteStreamBodies")
        assert len(bodies) == 2
        assert bodies[0]["client_message_id"] == bodies[1]["client_message_id"] == first_id
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Proposal ready", timeout=15_000)
        expect(page.locator(f'.message-user[data-client-message-id="{first_id}"]')).to_have_count(1)
        assert not [error for error in errors if error.startswith("pageerror:")]
    finally:
        page.context.close()


def test_truncated_terminal_recovers_persisted_success_without_false_failure(
    browser: Browser,
    product_server: str,
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    try:
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.__hciRecoveredSuccessBodies = [];
              window.fetch = async (input, init = {}) => {
                const url = typeof input === "string" ? input : input.url;
                const response = await originalFetch(input, init);
                if (!url.endsWith("/copilot/messages")) return response;
                window.__hciRecoveredSuccessBodies.push(JSON.parse(init.body));
                const raw = await response.text();
                const frames = raw.replaceAll("\\r\\n", "\\n").split("\\n\\n")
                  .filter((frame) => frame.trim());
                const truncated = `${frames.slice(0, 2).join("\\n\\n")}\\n\\n`;
                return new Response(truncated, {
                  status: response.status,
                  headers: {
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Cache-Control": "no-store",
                  },
                });
              };
            }"""
        )
        page.locator("#open-copilot").click()
        page.locator("#copilot-input").fill("Replace Bixby Creek Bridge Viewpoint with Pfeiffer Big Sur State Park")
        page.locator("#copilot-submit").click()

        expect(page.locator("#copilot-lifecycle")).to_have_attribute("data-state", "proposal_ready", timeout=15_000)
        expect(page.locator("#copilot-lifecycle")).to_contain_text("Proposal ready")
        expect(page.locator(".copilot-proposal-card")).to_be_visible()
        expect(page.locator('.message-assistant[data-delivery-state="completed"]')).to_have_count(1)
        expect(page.get_by_role("button", name="Retry", exact=True)).to_have_count(0)
        assert page.evaluate("window.__hciRecoveredSuccessBodies.length") == 1
        client_id = page.evaluate("window.__hciRecoveredSuccessBodies[0].client_message_id")
        expect(page.locator(f'.message-user[data-client-message-id="{client_id}"]')).to_have_count(1)
        assert not [error for error in errors if "/copilot/messages" in error or error.startswith("pageerror:")]
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


def test_unconfirmed_provider_failure_remains_retryable_without_claiming_failure(
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

        expect(page.locator("#copilot-lifecycle")).to_contain_text("Stopped")
        expect(page.locator("#copilot-lifecycle")).not_to_contain_text("Conversation restored")
        expect(page.locator("#copilot-input")).to_have_value("")
        unconfirmed = page.locator('.message-assistant[data-delivery-state="cancelled"]').last
        expect(unconfirmed).to_contain_text("result was not confirmed")
        expect(unconfirmed.get_by_role("button", name="Retry")).to_be_visible()
        expect(unconfirmed.get_by_role("button", name="Edit message")).to_be_visible()
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


def test_w5_keep_original_records_and_restores_durable_decision(
    browser: Browser,
    product_server: str,
    tmp_path: Path,
) -> None:
    page, errors = _open_app(browser, product_server, width=1280)
    expect(page.locator("#keep-original")).to_be_visible()
    expect(page.locator("#keep-original")).to_be_enabled()
    expect(page.locator("#accept-repair")).to_be_disabled()

    confirmation_messages: list[str] = []

    with page.expect_response(
        lambda response: response.url.endswith("/keep-original") and response.request.method == "POST"
    ) as response_info:
        page.locator("#keep-original").click()
        expect(page.get_by_role("dialog", name="Confirm trip decision", exact=True)).to_be_visible()
        confirmation_messages.append(page.locator("#acceptance-message").inner_text())
        page.get_by_role("dialog", name="Confirm trip decision", exact=True).get_by_role(
            "button", name="Keep original", exact=True
        ).click()
    response = response_info.value
    assert response.status == 200
    action = response.json()
    assert action["schema_version"] == "acceptance-action-v1"
    assert action["result"]["decision"]["decision_type"] == "keep_original"
    assert action["result"]["decision"]["session_revision_after"] == 1
    decision_id = action["result"]["decision"]["decision_id"]
    session_id = action["session"]["session_id"]
    assert action["result"]["pointer"]["decision_id"] == decision_id
    assert confirmation_messages == ["Keep the immutable original plan as the durable trip plan?"]
    expect(page.locator("#toast-region")).to_contain_text("Original plan kept. The durable decision was recorded.")

    with page.expect_response(
        lambda candidate: candidate.url.endswith(f"/api/sessions/{session_id}") and candidate.request.method == "GET"
    ) as restored_info:
        page.reload(wait_until="domcontentloaded")
    restored = restored_info.value.json()
    expect(page.locator("#loading-screen")).to_be_hidden()
    pointer = restored["workspace"]["accepted_plan_pointer"]
    assert pointer["decision_id"] == decision_id
    assert pointer["revision"] == 1
    assert restored["session"]["accepted_plan_id"] == pointer["accepted_plan_id"]

    page.screenshot(path=str(tmp_path / "w5_keep_original_after_reload.png"), full_page=True)
    assert errors == []
    page.context.close()
