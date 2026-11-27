"""Cross-day moves in the coherent demo: chat intent -> repair -> independent evaluation.

Moves use the frozen move-leg road delta and the joint cross-day search in the
deterministic repair heuristic. Unreasonable moves are rejected with the day
that breaks the daily time limit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot import DeterministicCopilotAdapter, request_semantic_guard
from itinerary_system.product_app.registry import ProductRunRegistry
from itinerary_system.product_app.service import ProductService

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "product_app_registry.json"
REPLACE_BIXBY = {
    "type": "replace_nearby",
    "target": "bixby_creek_bridge_viewpoint",
    "parameters": {"candidate_id": "point_lobos_state_natural_reserve"},
}


def _preview(client: TestClient, operations: list[dict]) -> tuple[dict, dict, dict]:
    created = client.post("/api/sessions", json={}).json()
    session_id = created["session"]["session_id"]
    headers = {"X-Session-Token": created["mutation_token"]}
    staged = client.post(
        f"/api/sessions/{session_id}/draft/operations",
        headers=headers,
        json={
            "expected_revision": created["session"]["revision"],
            "operations": [{**row, "source": "typed_stop_editor", "evidence_refs": []} for row in operations],
        },
    )
    assert staged.status_code == 200, staged.text
    preview = client.post(
        f"/api/sessions/{session_id}/preview",
        headers=headers,
        json={"expected_revision": staged.json()["session"]["revision"]},
    )
    assert preview.status_code == 200, preview.text
    return preview.json()["proposal"], preview.json()["session"], headers


@pytest.fixture
def client(tmp_path):
    # One app per test: two loaded product apps at once can exhaust memory on
    # a small machine, and the restart test below opens its own apps.
    app = create_product_app(
        repository_root=ROOT,
        registry_path=REGISTRY,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_lock_and_move_are_enabled_but_flexible_is_not(client):
    created = client.post("/api/sessions", json={}).json()
    operations = created["workspace"]["draft_capabilities"]["operations"]
    assert operations["move_day"]["preview_executable"] is True
    assert operations["lock_stop"]["preview_executable"] is True
    assert operations["mark_flexible"]["preview_executable"] is False
    combinations = created["workspace"]["draft_capabilities"]["executable_combinations"]
    assert any({"move_day", "replace_nearby"} <= set(row["operations"]) for row in combinations)


def test_move_to_another_day_is_evaluated_and_empties_the_source_day(client):
    proposal, _, _ = _preview(
        client,
        [
            REPLACE_BIXBY,
            {"type": "move_day", "target": "griffith_observatory", "parameters": {"day": 6}},
        ],
    )
    assert proposal["state"] == "eligible", proposal.get("message")
    days = {}
    for feature in proposal["geography_plan"]["stops"]["features"]:
        days.setdefault(feature["properties"]["day"], set()).add(feature["properties"]["stop_id"])
    assert days[6] == {"griffith_observatory", "stearns_wharf"}
    assert 7 not in days
    # Days 1-6 hold 2+2+2+3+2+3 legs; day 7 keeps its drive from Santa Barbara
    # to the airport after Griffith leaves, so it is still checked.
    legs = proposal["route_validation"]
    assert legs["required_leg_count"] == legs["road_validated_leg_count"] == 15
    assert legs["missing_leg_count"] == 0


def test_move_that_leaves_a_middle_day_empty_is_evaluated(client):
    proposal, _, _ = _preview(
        client,
        [
            REPLACE_BIXBY,
            {"type": "move_day", "target": "stearns_wharf", "parameters": {"day": 7}},
        ],
    )
    assert proposal["state"] == "eligible", proposal.get("message")


def test_unreasonable_move_is_rejected_with_the_day_over_the_time_limit(client):
    proposal, _, _ = _preview(
        client,
        [
            REPLACE_BIXBY,
            {"type": "move_day", "target": "kings_canyon_national_park", "parameters": {"day": 3}},
        ],
    )
    assert proposal["state"] == "not_evaluated"
    assert proposal["reason"] == "no_feasible_local_candidate"
    assert "Day 3" in proposal["message"] and "12 hours" in proposal["message"]


def test_accepted_move_survives_restart(tmp_path):
    state = tmp_path / "state"

    def app():
        return create_product_app(
            repository_root=ROOT,
            registry_path=REGISTRY,
            state_root=state,
            additional_allowed_authorities=("testserver",),
        )

    with TestClient(app()) as first:
        proposal, session, headers = _preview(
            first,
            [
                REPLACE_BIXBY,
                {"type": "move_day", "target": "griffith_observatory", "parameters": {"day": 6}},
            ],
        )
        assert proposal["state"] == "eligible"
        accepted = first.post(
            f"/api/sessions/{session['session_id']}/accept",
            headers=headers,
            json={
                "expected_revision": session["revision"],
                "confirmation": "accept_evaluated_repair",
                "idempotency_key": "cross_day_move_accept",
            },
        )
        assert accepted.status_code == 200, accepted.text
    with TestClient(app()) as second:
        restored = second.get(f"/api/sessions/{session['session_id']}", headers=headers)
        assert restored.status_code == 200, restored.text
        assert restored.json()["session"]["accepted_plan_id"] == accepted.json()["session"]["accepted_plan_id"]


def test_chat_add_is_route_ready_with_the_move_leg_delta(tmp_path):
    service = ProductService(ProductRunRegistry(ROOT, REGISTRY), tmp_path, copilot=DeterministicCopilotAdapter())
    session, token, view = service.create_session(service.registry.default.run_id)
    assert view["draft_capabilities"]["operations"]["add_candidate"]["preview_executable"] is True
    import asyncio

    response = asyncio.run(
        service.interpret_copilot_message(
            session_id=session.session_id,
            mutation_token=token,
            expected_revision=0,
            client_message_id="client_message_00000000000000000000000000000008",
            message="Add Pfeiffer Big Sur State Park to day 5.",
        )
    )
    interpretation = response["turn"]["interpretation"]
    assert interpretation["state"] == "proposal_ready", interpretation["state"]
    assert [(i["type"], i["candidate_id"], i["destination_day"]) for i in interpretation["intents"]] == [
        ("add_candidate", "pfeiffer_big_sur_state_park", 5)
    ]


@pytest.mark.parametrize(
    ("message", "keeps"),
    [
        ("I want to visit Kings Canyon National Park on day 5.", False),  # benchmark m28n
        ("I want to visit Kings Canyon National Park on the fifth day.", False),
        ("I want to visit Kings Canyon National Park on day 4.", True),  # its accepted day
        ("I want to visit Kings Canyon National Park.", True),
    ],
)
def test_visit_request_naming_another_day_is_not_turned_into_keep(tmp_path, message, keeps):
    from itinerary_system.product_app.openai_copilot import _accepted_stop_inclusion

    service = ProductService(ProductRunRegistry(ROOT, REGISTRY), tmp_path, copilot=DeterministicCopilotAdapter())
    session, _, _ = service.create_session(service.registry.default.run_id)
    result = _accepted_stop_inclusion(service.copilot_context(session), message)
    if keeps:
        assert result is not None and [intent.type for intent in result.intents] == ["keep_stop"]
    else:
        assert result is None


def test_chat_combination_with_a_move_is_one_proposal(tmp_path):
    service = ProductService(ProductRunRegistry(ROOT, REGISTRY), tmp_path, copilot=DeterministicCopilotAdapter())
    session, _, _ = service.create_session(service.registry.default.run_id)
    context = service.copilot_context(session)
    guarded = request_semantic_guard(
        context,
        "Replace Bixby Creek Bridge Viewpoint with Point Lobos State Natural Reserve and move Griffith Observatory to day 6",
    )
    assert guarded is None or guarded.state == "proposal_ready", guarded
    if guarded is not None:
        assert {intent.type for intent in guarded.intents} == {"replace_nearby", "move_day"}
