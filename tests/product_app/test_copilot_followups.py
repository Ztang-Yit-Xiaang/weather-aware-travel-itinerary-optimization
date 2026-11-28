"""Follow-ups that hand a choice back, and replies that name the change.

Found by the 2026-11-28 live follow-up probe (.duck/probe/probe_followups.py):
"propose this change" repeated the question, "Point Lobos" after "swap that
bridge" asked for the stop again, and every proposal said only "I prepared a
typed itinerary suggestion".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from itinerary_system.product_app.api import create_product_app
from itinerary_system.product_app.copilot_copy import describe_proposal

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCT_COPILOT_ADAPTER", "deterministic")
    app = create_product_app(
        repository_root=ROOT,
        state_root=tmp_path / "state",
        additional_allowed_authorities=("testserver",),
    )
    with TestClient(app) as test_client:
        yield test_client


def _conversation(client: TestClient, messages: list[str]) -> list[dict]:
    created = client.post("/api/sessions", json={}).json()
    session_id = created["session"]["session_id"]
    headers = {"X-Session-Token": created["mutation_token"]}
    revision = created["session"]["revision"]
    turns = []
    for message in messages:
        response = client.post(
            f"/api/sessions/{session_id}/copilot/messages",
            headers=headers,
            json={
                "expected_revision": revision,
                "client_message_id": f"client_message_{uuid4().hex}",
                "message": message,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        revision = body["session"]["revision"]
        turns.append(body["turn"]["interpretation"])
    return turns


def _intents(turn: dict) -> list[tuple]:
    return [(row["type"], row.get("target_stop_id"), row.get("candidate_id")) for row in turn["intents"]]


@pytest.mark.parametrize("follow_up", ["propose this chage", "you choose", "yes do it", "whatever you think is best"])
def test_weather_request_then_handing_back_the_choice_proposes_one_replacement(client, follow_up):
    asked, proposed = _conversation(
        client, ["The coast looks stormy on day 5, can we do something safer?", follow_up]
    )
    assert asked["state"] == "clarification_required"
    assert "Bixby Creek Bridge Viewpoint" in asked["assistant_message"]
    assert proposed["state"] == "proposal_ready", proposed["assistant_message"]
    [(kind, target, candidate)] = _intents(proposed)
    assert (kind, target) == ("replace_nearby", "bixby_creek_bridge_viewpoint")
    assert candidate in {"point_lobos_state_natural_reserve", "pfeiffer_big_sur_state_park"}
    assert "You left the choice to me" in proposed["assistant_message"]
    assert "nothing is saved until you accept" in proposed["assistant_message"]


def test_something_nearby_is_a_delegated_choice(client):
    [proposed] = _conversation(client, ["Swap that bridge for something nearby."])
    assert proposed["state"] == "proposal_ready", proposed["assistant_message"]
    assert _intents(proposed)[0][:2] == ("replace_nearby", "bixby_creek_bridge_viewpoint")


def test_naming_only_the_replacement_names_the_stop_it_replaces(client):
    [proposed] = _conversation(client, ["Replace it with Point Lobos State Natural Reserve"])
    assert proposed["state"] == "proposal_ready", proposed["assistant_message"]
    assert _intents(proposed) == [
        ("replace_nearby", "bixby_creek_bridge_viewpoint", "point_lobos_state_natural_reserve")
    ]
    assert "replace Bixby Creek Bridge Viewpoint on Day 5 with Point Lobos" in proposed["assistant_message"]


def test_moving_a_stop_to_its_own_day_is_not_proposed(client):
    _, answered = _conversation(client, ["Move Stearns Wharf", "day 6"])
    assert answered["state"] != "proposal_ready"
    assert "already on Day 6" in answered["assistant_message"]


def test_proposal_copy_names_each_change():
    context = SimpleNamespace(
        known_places=(
            {"place_id": "a", "name": "Alpha"},
            {"place_id": "b", "name": "Beta"},
        ),
        capabilities={"stop_positions": ({"stop_id": "a", "day": 2, "sequence_index": 0},)},
    )

    def intent(**fields):
        return SimpleNamespace(
            **{
                "type": None,
                "target_stop_id": None,
                "candidate_id": None,
                "destination_day": None,
                "preference": None,
                "parameters": None,
                **fields,
            }
        )

    assert describe_proposal(context, (intent(type="remove_stop", target_stop_id="a"),)) == (
        "I prepared a change that will remove Alpha from Day 2 with no replacement. "
        "I will check the new route and schedule on the map; nothing is saved until you accept it."
    )
    both = describe_proposal(
        context,
        (
            intent(type="move_day", target_stop_id="a", destination_day=4),
            intent(type="add_candidate", candidate_id="b", destination_day=3),
        ),
    )
    assert "(1) move Alpha from Day 2 to Day 4; (2) add Beta on Day 3" in both
    assert describe_proposal(context, (intent(type="unknown_kind"),)) is None
