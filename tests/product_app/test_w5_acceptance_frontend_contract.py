"""Static/executable contract checks for explicit W5 decision controls."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "itinerary_system" / "product_app" / "static" / "js" / "app.js"
INDEX = ROOT / "src" / "itinerary_system" / "product_app" / "static" / "index.html"


def test_w5_controls_are_confirmed_enabled_and_server_bound() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'id="accept-repair" ${canAccept ? "" : "disabled"}' in source
    assert '<button id="keep-original">Keep original</button>' in source
    assert "window.confirm(message)" not in source
    assert "dialog.showModal()" in source
    assert 'dialog.addEventListener("close"' in source
    assert 'confirmation = acceptRepair ? "accept_evaluated_repair" : "keep_original_plan"' in source
    assert "idempotency_key: `${endpoint}:${crypto.randomUUID()}`" in source
    assert 'submitAcceptance("accept_repair")' in source
    assert 'submitAcceptance("keep_original")' in source
    assert 'schema_version !== "acceptance-action-v1"' in source
    assert "decision.session_revision_after !== decision.session_revision_before + 1" in source
    assert "pointer.accepted_plan_id !== decision.selected_plan_id" in source
    assert "value.session.accepted_plan_id !== pointer.accepted_plan_id" in source
    assert "Acceptance and Keep original remain disabled until" not in source
    markup = INDEX.read_text(encoding="utf-8")
    assert 'id="acceptance-dialog" aria-labelledby="acceptance-title"' in markup
    assert '<button value="cancel" autofocus>Cancel</button>' in markup
    assert 'id="acceptance-confirm" class="primary" value="confirm"' in markup


def test_w5_confirmation_cancellation_and_stale_state_behavior() -> None:
    completed = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("map_preview_lifecycle_contract.test.mjs"))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_w5_app_cache_token_and_javascript_syntax() -> None:
    markup = INDEX.read_text(encoding="utf-8")
    # The server binds the page to the current source generation, not a hand-edited date token.
    assert "/static/js/app.js?v=__PRODUCT_BUILD_ID__" in markup
    api = (INDEX.parent.parent / "api.py").read_text(encoding="utf-8")
    assert 'b"__PRODUCT_BUILD_ID__", runtime.build_id.encode("ascii")' in api
    assert "security.validate_browser_build(request, runtime.build_id)" in api

    completed = subprocess.run(
        ["node", "--check", str(APP)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
