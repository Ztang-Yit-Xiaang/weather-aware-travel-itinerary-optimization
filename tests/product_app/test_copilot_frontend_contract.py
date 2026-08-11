from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "itinerary_system" / "product_app" / "static"
APP_JS = STATIC / "js" / "app.js"
COPILOT_JS = STATIC / "js" / "copilot.js"
INDEX = STATIC / "index.html"
CSS = STATIC / "css" / "app.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    assert "visible trip context, your message, and a bounded recent conversation window are sent to OpenAI" in controller
    assert "requests stay on this computer" in controller
    assert "stored locally for 30 days" in controller
    assert "Fixture Copilot" not in markup
    assert "Fixture Copilot" not in controller


def test_message_submission_is_idempotent_bounded_and_failure_safe() -> None:
    controller = _read(COPILOT_JS)

    assert "if (requestActive || !available()) return" in controller
    assert 'crypto.randomUUID().replaceAll("-", "")' in controller
    assert "client_message_id: clientMessageId" in controller
    assert "expected_revision: state.session.revision" in controller
    assert "input.value = message" in controller
    assert "setSubmitting(true)" in controller
    assert "setSubmitting(false)" in controller
    assert "await restore({ announce: false })" in controller
    assert controller.index("await restore({ announce: false })") < controller.index(
        'setLifecycle("failed", error.message)'
    )


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
    assert 'window.confirm("Add this typed Copilot suggestion' in controller
    assert 'window.confirm("Run the deterministic repair' in controller
    assert 'button("Show on map"' in controller
    assert 'alreadyAdded ? "Added to draft" : "Add to draft"' in controller
    assert "(event) => addIntentToDraft(editIntent, event.currentTarget)" in controller
    assert "control.disabled = true" in controller
    assert 'button("Preview repair"' in controller
    assert 'button("Compare"' in controller
    assert 'button("Review evidence"' in controller
    assert "/accept" not in controller
    assert "/keep-original" not in controller


def test_map_highlights_are_allow_listed_before_selection() -> None:
    controller = _read(COPILOT_JS)

    assert "function validSet(field)" in controller
    assert "function firstValidated(highlights, field)" in controller
    assert ".find((value) => allowed.has(value))" in controller
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
    assert '.copilot-dock .icon-button { width: 44px; height: 44px; }' in stylesheet
    assert 'role="complementary"' in markup
    assert 'aria-valuenow="440"' in markup
    assert 'event.key === "Escape"' in _read(COPILOT_JS)
    assert 'dock.setAttribute("aria-modal", "true")' in _read(COPILOT_JS)
    forbidden = ("\ufffd", "\u9225", "\u923b", "\u9241", "\u8113", "\u9451")
    for source in (markup, stylesheet, _read(APP_JS), _read(COPILOT_JS)):
        assert all(token not in source for token in forbidden)
