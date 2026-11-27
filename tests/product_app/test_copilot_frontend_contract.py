from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP_JS = STATIC / "js" / "app.js"
COPILOT_JS = STATIC / "js" / "copilot.js"
INDEX = STATIC / "index.html"
CSS = STATIC / "css" / "app.css"
EXECUTABLE_CONTRACT = ROOT / "tests" / "product_app" / "copilot_draftability_contract.test.mjs"
INTERACTION_CONTRACT = ROOT / "tests" / "product_app" / "copilot_interaction_state_contract.test.mjs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reference_links_are_clickable_without_html_interpretation() -> None:
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("copilot_reference_links.test.mjs"))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_copilot_is_a_focused_controller_restored_during_boot() -> None:
    app = _read(APP_JS)
    controller = _read(COPILOT_JS)

    assert 'from "./copilot.js?' in app
    assert "createCopilotController" in app
    assert "await copilot.restore()" in app
    assert "/conversation`" in controller
    assert "payload.conversation" in controller


def test_provider_labels_and_persistent_disclosures_are_truthful() -> None:
    markup = _read(INDEX)
    controller = _read(COPILOT_JS)

    assert 'id="copilot-provider-label"' in markup
    assert 'id="copilot-disclosure"' in markup
    assert 'provider === "openai" ? "OpenAI Copilot" : "Deterministic demo"' in controller
    assert (
        "visible trip context, your message, and a bounded recent conversation window are sent to OpenAI" in controller
    )
    assert "requests stay on this computer" in controller
    assert "expire after 30 days without a new turn" in controller
    assert "separately saved test/evidence reports are not deleted" in controller
    assert "Fixture Copilot" not in markup
    assert "Fixture Copilot" not in controller


def test_message_submission_is_idempotent_bounded_and_failure_safe() -> None:
    controller = _read(COPILOT_JS)
    app = _read(APP_JS)

    assert "const controller = new AbortController()" in controller
    assert "activeRequest.controller.abort()" in controller
    assert 'crypto.randomUUID().replaceAll("-", "")' in controller
    assert "client_message_id: clientMessageId ||" in controller
    assert "expected_revision: state.session.revision" in controller
    assert "ephemeralTurn" in controller
    assert 'Accept: "text/event-stream"' in app
    assert "event.client_message_id !== body.client_message_id" in app
    assert "event.sequence <= lastSequence" in app
    assert "setSubmitting(true)" in controller
    assert "setSubmitting(false)" in controller
    assert "await restore({ announce: false })" in controller
    assert "Provider processing or usage may already have occurred" in controller
    assert "Retry will reconcile this same request" in controller
    assert "const recoveredTurn = restored ?" in controller
    assert "recoveredTurn.interpretation?.state || recoveredTurn.state" in controller
    assert "setLifecycle(lifecycle, recoveredTurn.assistant_message)" in controller
    assert "Retry will reconcile this same request identity" in controller
    assert "const canonicalFailure = Boolean(error.code)" not in controller


def test_transcripts_are_rendered_with_text_content_only() -> None:
    controller = _read(COPILOT_JS)

    assert ".textContent = text" in controller
    assert ".textContent = label" in controller
    assert ".textContent = value" in controller
    assert "innerHTML" not in controller
    assert "insertAdjacentHTML" not in controller


def test_lifecycle_states_have_an_atomic_live_region() -> None:
    markup = _read(INDEX)
    controller = _read(COPILOT_JS)

    assert 'id="copilot-lifecycle"' in markup
    assert 'role="status"' in markup
    assert 'aria-live="polite"' in markup
    assert 'aria-atomic="true"' in markup
    for state in (
        "sending",
        "interpreting",
        "cancelling",
        "cancelled",
        "preview_running",
        "preview_rendering",
        "preview_rendered",
        "preview_ineligible",
        "clarification_required",
        "permission_required",
        "proposal_ready",
        "refused",
        "failed",
    ):
        assert state in controller


def test_proposal_actions_are_explicit_and_never_accept_a_plan() -> None:
    controller = _read(COPILOT_JS)

    assert 'interpretation?.state === "proposal_ready"' in controller
    assert 'window.confirm("Add this typed Copilot suggestion' not in controller
    assert 'window.confirm("Run the deterministic repair' not in controller
    assert 'button("Show on map"' in controller
    assert 'alreadyAdded ? "Added to draft" : "Add to draft"' in controller
    assert 'button("Cannot add to draft"' in controller
    assert "copilotDraftability(intent, getState().workspace)" in controller
    assert "repair-preview or discovery-only stop" in controller
    assert "This exact suggestion is staged" in controller
    assert "() => addIntentToDraft(intent)" in controller
    assert "if (previewActionActive || requestActive) return;" in controller
    assert "previewActionActive = true;" in controller
    assert '$$(".copilot-proposal-actions button").forEach' in controller
    assert "buttonControl.disabled = true;" in controller
    assert "previewActionActive = false;" in controller
    assert '"Preview this repair"' in controller
    assert "previewIntentFromCopilot" in controller
    assert "copilotOneClickEligibility" in controller
    assert "sameCopilotDraftOperation" in controller
    assert 'button("Compare"' in controller
    assert 'button("Review evidence"' in controller
    proposal_actions = controller[
        controller.index("function buildProposalActions") : controller.index("function validSet")
    ]
    assert "const actionTarget = intentDescription(intent)" in proposal_actions
    assert 'const actionContext = intent.type === "replan_trip" ? "trip plan" : actionTarget' in proposal_actions
    assert 'preview.setAttribute("aria-label", `Preview this repair: ${actionContext}`)' in proposal_actions
    assert 'add.setAttribute("aria-label"' in proposal_actions
    assert 'unavailable.setAttribute("aria-label", `Cannot add to draft: ${actionContext}`)' in proposal_actions
    assert "/accept" not in controller
    assert "/keep-original" not in controller


def test_copilot_draftability_guard_is_executable() -> None:
    result = subprocess.run(
        ["node", str(EXECUTABLE_CONTRACT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "copilot draftability and atomic combined eligibility contract passed" in result.stdout


def test_copilot_preview_rechecks_exact_draft_after_pending_selection() -> None:
    result = subprocess.run(
        ["node", str(ROOT / "tests/product_app/copilot_preview_boundary_contract.test.mjs")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Actual preview boundary rejects changed Copilot drafts after selection" in result.stdout


def test_capacity_block_executes_real_controller_without_unknown_retry() -> None:
    result = subprocess.run(
        ["node", str(ROOT / "tests/product_app/copilot_capacity_contract.test.mjs")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "copilot capacity controller contract passed" in result.stdout


def test_copilot_interaction_state_contract_is_executable() -> None:
    staged = subprocess.run(
        ["node", str(ROOT / "tests/product_app/copilot_staged_preview_contract.test.mjs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert staged.returncode == 0, staged.stdout + staged.stderr
    result = subprocess.run(
        ["node", str(INTERACTION_CONTRACT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "copilot interaction state contract passed" in result.stdout


def test_copilot_session_revision_guards_execute_in_normal_pytest() -> None:
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests/product_app/copilot_session_revision_contract.test.mjs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_copilot_conversation_generation_guards_execute_in_normal_pytest() -> None:
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests/product_app/copilot_conversation_generation_contract.test.mjs")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_map_highlights_are_allow_listed_before_selection() -> None:
    controller = _read(COPILOT_JS)

    assert "function validSet(field)" in controller
    assert "function firstValidated(highlights, field)" in controller
    assert "function acceptedDayForStop(stopId)" in controller
    assert ".find((value) => allowed.has(value))" in controller
    assert 'firstValidated(highlights, "day_ids") ?? acceptedDayForStop(stop)' in controller
    assert "if (!Object.keys(patch).length)" in controller
    assert "await selectContext(patch)" in controller
    assert "route geometry" not in controller.lower()


def test_transcript_deletion_requires_confirmation_and_restores_focus() -> None:
    markup = _read(INDEX)
    controller = _read(COPILOT_JS)

    for element_id in (
        "transcript-settings-dialog",
        "delete-current-conversation",
        "delete-all-conversations",
    ):
        assert f'id="{element_id}"' in markup
    assert "Delete this local conversation?" in controller
    assert "Delete every local Copilot conversation?" in controller
    assert 'confirmation: "delete_all_conversations"' in controller
    assert 'headers: { "X-Session-Id": state.session.session_id }' in controller
    assert "settingsOpener?.focus()" in controller
    assert "dockOpener?.focus()" in controller


def test_copilot_layout_remains_responsive_and_utf8_is_valid() -> None:
    markup = _read(INDEX)
    stylesheet = _read(CSS)

    assert ".copilot-dock" in stylesheet
    assert "minmax(0, 1fr)" in stylesheet
    assert "safe-area-inset-bottom" in stylesheet
    assert "@media (max-width: 820px)" in stylesheet
    assert "@media (max-width: 820px) and (max-height: 500px)" in stylesheet
    assert ".copilot-dock .icon-button { width: 44px; height: 44px; }" in stylesheet
    assert 'role="complementary"' in markup
    assert 'role="log"' in markup
    assert 'id="copilot-submit"' in markup
    assert 'aria-valuenow="440"' in markup
    assert 'event.key === "Escape"' in _read(COPILOT_JS)
    assert 'dock.setAttribute("aria-modal", "true")' in _read(COPILOT_JS)
    forbidden = ("\ufffd", "\u9225", "\u923b", "\u9241", "\u8113", "\u9451")
    for source in (markup, stylesheet, _read(APP_JS), _read(COPILOT_JS)):
        assert all(token not in source for token in forbidden)
